import os
import torch
import base64
import time
from PIL import Image
from io import BytesIO
import psycopg2
from byaldi import RAGMultiModalModel
from PyPDF2 import PdfReader

# === Poppler setup(for CPU-based PDF processing) ===
VENV_BASE = r"C:\Users\sgdrig01\Desktop\AI App Internship project\AI server\venv"
POPPLER_PATH = os.path.join(VENV_BASE, "poppler-24.08.0", "Library", "bin")
os.environ["PATH"] += os.pathsep + POPPLER_PATH
print(f"Poppler path: {POPPLER_PATH} ({'Exists' if os.path.exists(POPPLER_PATH) else 'Missing'})")

# === Force CPU usage ===
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

# === PostgreSQL connection ===
conn = psycopg2.connect(
    dbname="sew-postgres",
    user="postgres",
    password="1Q2wazsx1q2wazsx",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# === Load RAG model ===
print("\nLoading RAG model...")
start_time = time.time()
INDEX_ROOT = r"C:\Users\sgdrig01\Desktop\indexed_data"
LOCAL_BASE_PATH = r"C:\Users\sgdrig01\Desktop\AI App Internship project\AI server\models\colpali-v1.3"
RAG = RAGMultiModalModel.from_pretrained(
    LOCAL_BASE_PATH,
    device=device,
    verbose=1,
    index_root=INDEX_ROOT
)
print(f"Model loaded in {time.time() - start_time:.2f} seconds")

# === Index the PDF ===
file_path = "./reassembled_pdfs/scott.pdf"
print("\nIndexing started...")
start_index = time.time()

# Index the PDF
RAG = RAGMultiModalModel.from_index(
    index_path=r'C:\Users\sgdrig01\Desktop\indexed_data\Scott',
    device=device,
    verbose=1
)

end_index = time.time()
index_duration = end_index - start_index
print(f"Indexing completed in {index_duration / 60:.2f} minutes")

# === Pages/min performance logging ===
try:
    num_pages = len(PdfReader(file_path).pages)
    print(f"Indexed {num_pages} pages in {index_duration:.2f} seconds → {num_pages / (index_duration / 60):.2f} pages/minute")
except:
    print("Page count failed. Could not compute indexing speed.")

# === Query test ===
query = "If I have a reach of 403, what size should i get?"
print("\nRunning search...")
start_query = time.time()
results = RAG.search(query, k=5)
print(f"Query executed in {time.time() - start_query:.2f} seconds")

# === Decode and show result image ===
try:
    image_bytes = base64.b64decode(results[0].base64)
    image = Image.open(BytesIO(image_bytes))
    image.show()
except Exception as e:
    print(f"Failed to show result image: {e}")


#Code works with text, images very suspect (version of transformers package: pip install transformers==4.51.3)