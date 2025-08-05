import glob
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from auth import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, get_current_user, get_current_role
from Split_Mod import  PDFProcessor
import hashlib
import psycopg2
import os
from datetime import datetime, timedelta
from typing import Optional
import asyncpg
import bcrypt
import torch
import base64
from PIL import Image, ImageEnhance
from io import BytesIO
import io
from byaldi import RAGMultiModalModel
import aiofiles
import aiohttp
from aiohttp import ClientTimeout
import ollama
import uvicorn
import asyncio
import traceback
from pydantic import BaseModel
import logging
import time
import aiofiles.os as aios
import shutil, uuid

app = FastAPI()
app.state.latest_result = None


logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Completely disable CUDA
torch.cuda.is_available = lambda: False  # Override CUDA check
device = torch.device("cpu")

VENV_BASE = r"C:\Users\sgdrig01\Desktop\AI App Internship project\Backup server\venv"
POPPLER_PATH = os.path.join(VENV_BASE, "poppler-24.08.0", "Library", "bin")
os.environ["PATH"] += os.pathsep + POPPLER_PATH

LOCAL_BASE_PATH = r"C:\Users\sgdrig01\Desktop\AI App Internship project\AI server\models\colpali-v1.3"
INDEX_ROOT = r"C:\Users\sgdrig01\Desktop\AI App Internship project\indexed_data"

# Use for testing multimodal without going through colpali
@app.post("/analyze-image/")
async def analyze_image(
    image: UploadFile = File(...),
    query: str = Form(...)
):
    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Only image files (JPEG, PNG) are allowed"
        )

    try:
        start_time = time.time()
        print("\nStarting image analysis...")

        # Read and encode the image file
        image_data = await image.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        # Call the existing function with the converted base64
        result = await get_multimodal_res(image_b64, query)

        elapsed = time.time() - start_time
        print(f"Image analysis completed in {elapsed:.2f} seconds")

        return result
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to process image: {str(e)}"
            }
        )
##########################################################################################

# Original enhancer: 
def enhance_image_b64(original_b64: str, debug: bool = False) -> str:
    """
    Enhance image quality (contrast + sharpness) while keeping base64 format.
    
    Args:
        original_b64: Base64 encoded image string
        
    Returns:
        Enhanced base64 image string (PNG format)
    """
    try:
        # Decode base64
        image_data = base64.b64decode(original_b64)
        img = Image.open(io.BytesIO(image_data))
        
        if debug:  # Show original
            img.show(title="ORIGINAL INPUT")

        # Convert to grayscale if needed (better for text/number clarity)
        if img.mode != 'L':
           img = img.convert('L')  # 'L' = 8-bit grayscale
        
        saturation = ImageEnhance.Color(img)
        contrast = ImageEnhance.Contrast(img)
        #brightness = ImageEnhance.Brightness(img)
        #sharpness = ImageEnhance.Sharpness(img)

        img = saturation.enhance(0.5)
        #img = contrast.enhance(0.9)  
        
        
        # Sharpen edges (helps distinguish similar numbers)
        # img = img.filter(ImageFilter.SHARPEN)
        
        if debug:  # Show enhanced
            img.show(title="ENHANCED OUTPUT (What model sees)")

        # Optional: Uncomment to binarize (black/white only)
        # img = img.point(lambda x: 0 if x < 180 else 255, '1')  # Adjust threshold
        
        # Re-encode to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")  # PNG preserves quality
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    except Exception as e:
        # Return original if enhancement fails
        print(f"Enhancement failed (using original): {str(e)}")
        return original_b64


# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database config
DB_CONFIG = {
    "dbname": "sew-postgres",
    "user": "postgres",
    "password": "1Q2wazsx1q2wazsx",
    "host": "localhost",
    "port": "5432"
}

def compute_sha256(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()

def validate_pdf(file_bytes: bytes) -> bool:
    """More flexible PDF validation that accepts most PDF files."""
    # Basic check for PDF header (first 8 bytes should contain "%PDF")
    if not file_bytes.startswith(b"%PDF-"):
        return False
    
    # Optional: Check for "%%EOF" somewhere in the last 1KB (many PDFs put it at end)
    # This is more flexible than requiring exact ending
    eof_pos = file_bytes.rfind(b"%%EOF")
    if eof_pos == -1 or eof_pos < len(file_bytes) - 1024:
        # If no EOF marker found, still accept but warn (many valid PDFs don't have it)
        print("Warning: PDF lacks standard EOF marker, but may still be valid")
        return True
    
    return True

def split_into_chunks(file_bytes: bytes, chunk_size=1_000_000) -> list:
    """Split PDF into 1MB chunks."""
    return [file_bytes[i:i + chunk_size] for i in range(0, len(file_bytes), chunk_size)]

async def check_duplicate(category: str, sha256_hash: str) -> bool:
    """Check if PDF exists in specified category table."""
    async with app.state.db_pool.acquire() as conn:
        result = await conn.fetchrow(
            f"SELECT 1 FROM {category} WHERE sha256_hash = $1 LIMIT 1",
            sha256_hash
        )
        return result is not None

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
 
# Need to fine-tune k's value for ColPali, change to use gpu & diff venv_base dir and May need change to ColQwen if more accurate when testing with bigger docs    
async def process_colpali(
    category: str,
    query: str,
    save_to_disk: bool = False,
    debug_intermediate: bool = False
) -> dict:
    try:
        index_path = r"C:\Users\sgdrig01\Desktop\AI App Internship project\indexed_data\Gear units\Assembly and operating instructions_index"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found at {index_path}")

        RAG = RAGMultiModalModel.from_index(index_path, device=device)

        results = RAG.search(query, k=1)
        if not results:
            raise ValueError("No results found for query")

        # Process image 
        image_bytes = base64.b64decode(results[0].base64)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert back to base64 for API response
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Save to disk if requested
        image_path = None
        if save_to_disk:
            os.makedirs("colpali_output", exist_ok=True)
            filename = f"{category}_result.png"
            image_path = os.path.join("colpali_output", filename)
            
            async with aiofiles.open(image_path, "wb") as f:
                await f.write(image_bytes)

        # Show image (sync operation)
        # image.show()
    
        return {
            "status": "success",
            "image_base64": image_base64,
            "query": query,
            "image_path": image_path,
            "message": "Processed successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ColPali processing failed: {str(e)}"
        )

async def get_multimodal_res(image_b64: str, query: str) -> dict:
    prompt = f"""<|im_start|>system
You are a form data extractor. Your response must contain ONLY these 2 lines:

Field: [exact field name from image]
Requirement: [visible text OR "Not specified"]

RULES:
1. NEVER add:
   - Prefixes like "Answer:"
   - Explanations
   - Unrequested analysis
2. If field exists but text is unclear: "Requirement: Not clearly visible"
3. Maximum 40 words total<|im_end|>
<|im_start|>user
Locate and extract: {query}<|im_end|>
<|im_start|>assistant
Field:  """

    if not image_b64 or not query:
        return {
            "status": "error",
            "message": "Missing required parameters: image_b64 or query."
        }

    model_name = "llama3.2-vision:11b-instruct-q8_0"
    base_url = "http://localhost:11434/api"
    
    # Increased timeouts
    model_check_timeout = ClientTimeout(total=60)   # 1 minute for model check
    model_pull_timeout = ClientTimeout(total=1800)  # 30 minutes for model pull (large model)
    generation_timeout = ClientTimeout(total=300)   # 10 minutes for generation

    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Check if model exists
            try:
                async with session.get(
                    f"{base_url}/tags",
                    timeout=model_check_timeout
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to fetch model tags: {await resp.text()}")

                    models_json = await resp.json()
                    model_names = [m.get("name") for m in models_json.get("models", [])]
                    
                    if model_name not in model_names:
                        print(f"Model not found. Pulling {model_name}...")
                        async with session.post(
                            f"{base_url}/pull",
                            json={"name": model_name},
                            timeout=model_pull_timeout
                        ) as pull_resp:
                            if pull_resp.status != 200:
                                raise Exception(f"Failed to pull model: {await pull_resp.text()}")
                            
                            # For large models, we need to wait for the download to complete
                            async for line in pull_resp.content:
                                if b'"status":"success"' in line:
                                    break
                        print("Model pulled successfully.")
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "message": "Timeout while checking/pulling the model. The model is large and may take longer to download."
                }
            except Exception as model_err:
                return {
                    "status": "error",
                    "message": f"Model check/pull failed: {str(model_err)}"
                }

            # Step 2: Generate response
            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "num_ctx": 512,
                    "temperature": 1.4,
                    "top_k": 50,
                    "repeat_penalty": 1.0,
                    "mirostat_mode": 2,
                    "mirostat_tau": 3.0,
                    "seed": 552
                }
            }

            try:
                async with session.post(
                    f"{base_url}/generate",
                    json=payload,
                    timeout=generation_timeout
                ) as resp:
                    if resp.status != 200:
                        error_detail = await resp.text()
                        return {
                            "status": "error",
                            "message": f"Model response failed: {error_detail}"
                        }

                    result = await resp.json()
                    answer = result.get("response", "").strip()
                    if not answer:
                        return {
                            "status": "error",
                            "message": "Empty response from the model."
                        }

                    return {
                        "status": "success",
                        "answer": answer,
                        "image_b64": image_b64,
                        "query_used": query
                    }
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "message": "The model took too long to respond. This might be due to the model size or system resources."
                }

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "traceback": tb
        }
                            

@app.post("/api/upsert_pdf")
async def upsert_pdf(
    file: UploadFile = File(...),
    category: str = Form(...),
    background_tasks: BackgroundTasks = None
    ):
    """Handle PDF upload with category-based storage and indexing."""
    try:
        # 1. Validate inputs
        if not category.replace('_', '').replace(' ', '').isalnum():
            raise HTTPException(
                status_code=400,
                detail="Category name can only contain alphanumeric characters, underscores, and spaces"
            )

        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are accepted"
            )

        # 2. Save the uploaded PDF to a temp file -----------------------------
        temp_dir = "temp_uploaded_pdfs"
        os.makedirs(temp_dir, exist_ok=True)

        safe_filename = Path(file.filename).name          # keep original name
        temp_path = os.path.join(temp_dir, safe_filename)

        with open(temp_path, "wb") as out_file:
            while chunk := await file.read(8192):          # 8 KB chunks
                out_file.write(chunk)

        # 3. Move it into a category‑specific permanent location -------------
        stored_dir = os.path.join("uploaded_pdfs", category)
        os.makedirs(stored_dir, exist_ok=True)

        stored_path = os.path.join(stored_dir, safe_filename)
        shutil.move(temp_path, stored_path)                # temp → permanent
      
        # 4. Initialize RAG model
        RAG = RAGMultiModalModel.from_pretrained(
            LOCAL_BASE_PATH,
            device=device,
            verbose=1,
            index_root=INDEX_ROOT
        )

        # 5. Prepare index directory structure
        file_name = os.path.splitext(safe_filename)[0]
        category_index_dir = os.path.join(INDEX_ROOT, category)
        
        # Create category directory if it doesn't exist
        os.makedirs(category_index_dir, exist_ok=True)
        index_name = os.path.join(category, f"{file_name}_index")
        full_index_path = os.path.join(INDEX_ROOT, index_name)
        
        # 6. Create and store the index
        print("\nIndexing started...")
        start_index = time.time()

        RAG.index(
            input_path=stored_path,
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True
        )
        end_index = time.time()
        index_duration = end_index - start_index
        print(f"Indexing completed in {index_duration / 60:.2f} minutes")

        # 7. Verify index was created
        if not os.path.exists(full_index_path):
            raise HTTPException(
                status_code=500,
                detail="Index creation failed - no index files found"
            )
        
        # 8. (Optional) clean up any lingering temp file ---------------------
        if background_tasks:
            background_tasks.add_task(
                lambda: os.path.exists(temp_path) and os.remove(temp_path)
            )

       
        # 9. Respond ---------------------------------------------------------
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "PDF indexed successfully",
                "category": category,
                "filename": file.filename,
                "stored_path": stored_path,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )




@app.post("/api/users")
async def handle_create_user(request: Request):
    data = await request.json()
    print("Received data:", data)
    try:
        login_id = data['login_id']
        password = data['password']
        role = data.get('role', 'user')
        print("Role being used:", role)  # Default to 'user' if not specified
        
        # Validate role
        if role not in ['admin', 'user']:
            raise HTTPException(status_code=400, detail="Invalid role specified")
            
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        async with app.state.db_pool.acquire() as conn:
            # Check if login_id already exists
            existing_user = await conn.fetchrow(
                "SELECT 1 FROM users WHERE login_id = $1;", 
                login_id
            )
            if existing_user:
                raise HTTPException(
                    status_code=400, 
                    detail="Login ID already exists"
                )
                
            # Insert new user
            await conn.execute("""
                INSERT INTO users (login_id, password_hash, role, created_at)
                VALUES ($1, $2, $3, $4);
            """, login_id, hashed.decode('utf-8'), role, datetime.now())
            
            return {
                "status": "success", 
                "message": "User created",
                "role": role
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login")
async def handle_login(request: Request):
    data = await request.json()
    try:
        login_id = data['login_id']
        password = data['password']
        async with app.state.db_pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT password_hash, role FROM users WHERE login_id = $1;
            """, login_id)
            
            if not result or not bcrypt.checkpw(
                password.encode('utf-8'), 
                result['password_hash'].encode('utf-8')
            ):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # Include role in the token data
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": login_id, "role": result['role']}, 
                expires_delta=access_token_expires
            )

            return {
                "access_token": access_token, 
                "token_type": "bearer",
                "role": result['role']
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
       
@app.get("/history")
async def get_history(current_user_login_id: str = Depends(get_current_user)):
    try:
        async with app.state.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT date, question, answer, out_image_b64  
                FROM chat_history 
                WHERE users = $1
                ORDER BY date DESC
                """,
                current_user_login_id
            )
            # Convert asyncpg.Record to dict
            result = [
            {
                "date": row["date"].isoformat(),
                "question": row["question"],
                "answer": row["answer"],
		        "image_b64": row["out_image_b64"]
            }
            for row in rows
        ]
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        print("Failed to fetch chat history:", e)
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")

@app.get("/api/role")
async def get_user_role(role: str = Depends(get_current_role)):
    return {"role": role}

@app.post("/api/create_pin")
async def handle_create_pin(request: Request):
    data = await request.json()
    try:
        pin = data['pin']
        hashed = bcrypt.hashpw(pin.encode('utf-8'), bcrypt.gensalt())
        async with app.state.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO upsert_auth (pin_hash, created_at)
                VALUES ($1, $2);
            """, hashed.decode('utf-8'), datetime.now())
            return {"status": "success", "message": "Pin created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pin_login")
async def handle_pin_login(request: Request):
    data = await request.json()
    try:
        pin = data['pin']
        
        async with app.state.db_pool.acquire() as conn:
            # Get the most recent PIN hash (assuming single PIN system)
            result = await conn.fetchrow("""
                SELECT pin_hash FROM upsert_auth
                ORDER BY created_at DESC
                LIMIT 1;
            """)
            
            if not result or not bcrypt.checkpw(pin.encode('utf-8'), result['pin_hash'].encode('utf-8')):
                raise HTTPException(status_code=401, detail="Invalid PIN")
                
            return {
                "status": "success", 
                "message": "PIN login successful"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Login failed: {str(e)}"
        )

@app.get("/api/categories")
async def get_categories():
    """Get all category names from the 'categories' table"""
    try:
        async with app.state.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT category FROM categories
                ORDER BY category ASC
            """)
            
            categories = [row['category'] for row in rows]

            #print(f"Current categories: {categories}")
            return {
                "status": "success",
                "categories": categories,
                "count": len(categories)
            }

    except Exception as e:
        print(f"Error fetching categories: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch categories: {str(e)}"
        )
 
class LatestResultResponse(BaseModel):
    status: str
    query: Optional[str] = None
    category: Optional[str] = None
    file_name: Optional[str] = None
    colpali_image: Optional[str] = None
    multimodal_response: Optional[str] = None
    pdf_metadata: Optional[dict] = None

class PDFQueryRequest(BaseModel):
    query: str
    category: str

@app.post("/api/process_pdf_query")
async def process_pdf_query(
    request: PDFQueryRequest,
    background_tasks: BackgroundTasks,
    current_user_login_id: str = Depends(get_current_user)
):
    temp_pdf_path = f"temp_{request.category}.pdf"
    temp_image_path = None

    try:
        start_time = time.time()
        print("\nStarting process query...")

        colpali_result = await process_colpali(
            category=request.category,
            query=request.query,
            save_to_disk=True
        )
        temp_image_path = colpali_result.get("image_path")
        image_b64 = colpali_result.get("image_base64")
        if not image_b64:
            raise HTTPException(status_code=500, detail="No image base64 returned from ColPali")


        multimodal_result = await get_multimodal_res(image_b64=image_b64, query=request.query)
        if multimodal_result.get("status") != "success":
            raise HTTPException(status_code=500, detail="Multimodal model failed: " + multimodal_result.get("message", "Unknown error"))

        answer_text = multimodal_result["answer"]

        elapsed = time.time() - start_time
        print(f"Image analysis completed in {elapsed:.2f} seconds")

        background_tasks.add_task(cleanup_files, temp_pdf_path, temp_image_path)

        result = {
            "status": "success",
            "query": request.query,
            "category": request.category,
            "colpali_image": image_b64,
            "multimodal_response": multimodal_result,
        }

        app.state.latest_result = result

        async with app.state.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_history 
                (date, question, answer, users, out_image_b64)
                VALUES (NOW(), $1, $2, $3, $4)
                """,
                request.query,
                answer_text,
                current_user_login_id,
                image_b64
            )

        return result

    except HTTPException:
        cleanup_files(temp_pdf_path, temp_image_path)
        raise

    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Unhandled exception in /api/process_pdf_query:\n", traceback_str)
        cleanup_files(temp_pdf_path, temp_image_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
   
@app.get("/api/get_latest_result", response_model=LatestResultResponse)
async def get_latest_result():
    try:
        if not hasattr(app.state, 'latest_result') or app.state.latest_result is None:
            return {
                "status": "error",
                "message": "No results available"
            }
        
        print(f"📦 Returning stored result: {app.state.latest_result}")  # Debug log
        return app.state.latest_result
        
    except Exception as e:
        print(f"❌ GET endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
def cleanup_files(pdf_path: str, image_path: str):
    try:
        """Helper function to clean up temporary files"""
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
    except Exception as e:
        print(f"Cleanup warning: {str(e)}")

@app.on_event("startup")
async def startup():
    app.state.db_pool = await asyncpg.create_pool(
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["dbname"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"]
    )

@app.on_event("shutdown")
async def shutdown():
    await app.state.db_pool.close()

# Main Function
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)