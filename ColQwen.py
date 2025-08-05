from byaldi import RAGMultiModalModel
import torch, os


# Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Completely disable CUDA
torch.cuda.is_available = lambda: False  # Override CUDA check
device = torch.device("cpu")

file_path = "./reassembled_pdfs/Declaration of academic integrity1.pdf"

VENV_BASE = r"C:\AI server\venv"
POPPLER_PATH = os.path.join(VENV_BASE, "poppler-24.08.0", "Library", "bin")
os.environ["PATH"] += os.pathsep + POPPLER_PATH

LOCAL_BASE_PATH = r"C:\AI server\models\colpaligemma-3b-pt-448-base"
INDEX_ROOT = r"C:\Users\sgdrig01\Desktop\indexed_data"

@app.post("/api/upsert_pdf")
async def upsert_pdf(
    file: UploadFile = File(...),
    category: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """Handle PDF upload with category-based storage and indexing."""
    try:
        # 1. Validate inputs
        if not category.replace('_', '').isalnum():  # Allows underscores
            raise HTTPException(
                status_code=400,
                detail="Category name can only contain alphanumeric characters and underscores"
            )

        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are accepted"
            )

        # 2. Read and store the PDF in chunks
        file_bytes = await file.read()
        result_storing = await store_pdf_chunks(
            category=category,
            file_bytes=file_bytes,
            filename=file.filename
        )

        if result_storing["status"] != "success":
            if result_storing["status"] == "duplicate":
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "duplicate",
                        "message": result_storing["message"],
                        "sha256": result_storing["sha256"]
                    }
                )
            raise HTTPException(
                status_code=400,
                detail=result_storing["message"]
            )

        # 3. Retrieve the stored PDF for indexing
        result_retrieving = await retrieve_pdf(
            category=category,
            filename=file.filename,
            save_to_disk=True  # Save to disk for indexing
        )

        if result_retrieving["status"] != "success":
            raise HTTPException(
                status_code=404,
                detail=result_retrieving["message"]
            )

        # 4. Initialize RAG model
        RAG = RAGMultiModalModel.from_pretrained(
            LOCAL_BASE_PATH,
            device=device,
            verbose=1,
            index_root=INDEX_ROOT
        )

        # 5. Prepare index directory structure
        file_name = os.path.splitext(file.filename)[0]
        category_index_dir = os.path.join(INDEX_ROOT, category)
        
        # Create category directory if it doesn't exist
        os.makedirs(category_index_dir, exist_ok=True)
        index_name = os.path.join(category, f"{file_name}_index")
        full_index_path = os.path.join(INDEX_ROOT, index_name)

        # 6. Create and store the index
        RAG.index(
            input_path=result_retrieving["file_path"],
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True
        )

        # 7. Verify index was created
        if not os.path.exists(full_index_path):
            raise HTTPException(
                status_code=500,
                detail="Index creation failed - no index files found"
            )

        # 8. Schedule cleanup of temporary PDF file
        if background_tasks and result_retrieving.get("file_path"):
            background_tasks.add_task(
                lambda p: os.remove(p) if os.path.exists(p) else None,
                result_retrieving["file_path"]
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "PDF stored and indexed successfully",
                "category": category,
                "filename": file.filename,
                "sha256": result_storing["sha256"],
                "index_location": full_index_path,
                "file_size": len(file_bytes),
                "chunks_stored": result_storing.get("chunks", 0)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )
async def retrieve_pdf(category: str, filename: str, save_to_disk: bool = False) -> dict:
    try:
        async with app.state.db_pool.acquire() as conn:
            # Get first chunk to verify existence
            first_chunk = await conn.fetchrow(
                f"""SELECT sha256_hash FROM {category} 
                    WHERE file_name = $1 AND chunk_number = 0
                    LIMIT 1""",
                filename
            )
            if not first_chunk:
                return {
                    "status": "error",
                    "message": f"File '{filename}' not found in '{category}'"
                }

            # Get all chunks
            chunks = await conn.fetch(
                f"""SELECT chunk_data FROM {category} 
                    WHERE file_name = $1
                    ORDER BY chunk_number""",
                filename
            )
            
            # Reassemble
            merged_pdf = b"".join([row["chunk_data"] for row in chunks])
            
            # Optional save
            file_path = None
            if save_to_disk:
                os.makedirs("reassembled_pdfs", exist_ok=True)
                file_path = os.path.join("reassembled_pdfs", filename)
                with open(file_path, "wb") as f:
                    f.write(merged_pdf)

            return {
                "status": "success",
                "message": "PDF retrieved",
                "sha256": first_chunk["sha256_hash"],
                "size": len(merged_pdf),
                "file_path": file_path
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Retrieval error: {str(e)}"
        }
   
async def store_pdf_chunks(category: str, file_bytes: bytes, filename: str) -> dict:
    """
    Stores PDF in category-specific table with:
    - Single table design (merged metadata and chunks)
    - Unique document tracking via (doc_id, chunk_number)
    - Prevents duplicate content while allowing filename reuse
    """
    sha256_hash = compute_sha256(file_bytes)
    doc_id = f"{sha256_hash}_{filename}"  # Unique document identifier
    
    try:
        async with app.state.db_pool.acquire() as conn:
            # Create table if not exists
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {category} (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT NOT NULL,          -- sha256_hash + filename
                    file_name TEXT NOT NULL,
                    sha256_hash TEXT NOT NULL,
                    chunk_number INT NOT NULL,
                    chunk_data BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(doc_id, chunk_number)    -- Prevents duplicate chunks
                )
            """)

            # Check for existing content (first chunk only)
            existing = await conn.fetchrow(
                f"""SELECT file_name FROM {category} 
                    WHERE doc_id = $1 AND chunk_number = 0 
                    LIMIT 1""",
                doc_id
            )
            if existing:
                return {
                    "status": "duplicate",
                    "message": f"PDF already exists as '{existing['file_name']}'",
                    "sha256": sha256_hash
                }

            # Store all chunks
            chunks = split_into_chunks(file_bytes)
            for i, chunk in enumerate(chunks):
                await conn.execute(
                    f"""
                    INSERT INTO {category} 
                    (doc_id, file_name, sha256_hash, chunk_number, chunk_data)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (doc_id, chunk_number) DO NOTHING
                    """,
                    doc_id, filename, sha256_hash, i, chunk
                )

            # Verify all chunks stored
            stored_count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {category} WHERE doc_id = $1",
                doc_id
            )
            
            if stored_count != len(chunks):
                await conn.execute(f"DELETE FROM {category} WHERE doc_id = $1", doc_id)
                return {
                    "status": "error",
                    "message": "Failed to store all chunks",
                    "sha256": sha256_hash
                }

            return {
                "status": "success",
                "message": f"Stored in '{category}' as {filename}",
                "sha256": sha256_hash,
                "chunks": stored_count
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Database error: {str(e)}",
            "sha256": sha256_hash
        }
