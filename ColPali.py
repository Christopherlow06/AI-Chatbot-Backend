import os
import torch
import base64
from PIL import Image
from io import BytesIO
import psycopg2
from byaldi import RAGMultiModalModel

# Poppler setup (for PDF processing)

os.environ['VERIFY_SSL_CERTS'] = 'False'
VENV_BASE = r"C:\AI server\venv"
POPPLER_PATH = os.path.join(VENV_BASE, "poppler-24.08.0", "Library", "bin")
os.environ["PATH"] += os.pathsep + POPPLER_PATH
print(f"Poppler: {POPPLER_PATH} ({'Exists' if os.path.exists(POPPLER_PATH) else 'Missing'})")

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

# PostgreSQL connection
conn = psycopg2.connect(
    dbname="sew-postgres",
    user="postgres",
    password="1Q2wazsx1q2wazsx",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Document to index
file_path = "./reassembled_pdfs/Declaration of academic integrity1.pdf"
#file_path = "./reassembled_pdfs/scott.pdf"

# Load RAG model 
LOCAL_BASE_PATH = r"C:\AI server\models\colpaligemma-3b-pt-448-base"
RAG = RAGMultiModalModel.from_pretrained(
    LOCAL_BASE_PATH,
    device=device,
    verbose=1
)

# Index the PDF
RAG.index(
    input_path=file_path,
    index_name="ToBeSentToMultiModal",
    store_collection_with_index=True,
    overwrite=True
)

# Query test
query = "Under Declaration of academic integrity, what does polytechnic rule state?"
results = RAG.search(query, k=1)
print("Query successful.")

# Decode and show result image
image_bytes = base64.b64decode(results[0].base64)
image = Image.open(BytesIO(image_bytes))
image.show()

