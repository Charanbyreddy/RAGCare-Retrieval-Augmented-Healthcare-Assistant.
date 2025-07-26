# build_index.py

import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

UPLOAD_FOLDER = "uploads"
VECTOR_DIR = "vectorstore"
os.makedirs(VECTOR_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load and chunk all PDFs
def load_text_chunks():
    chunks = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith(".pdf"):
            path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(path)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                # Split into 500-character chunks
                for i in range(0, len(text), 500):
                    chunk = text[i:i+500]
                    if chunk.strip():
                        chunks.append(chunk)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return chunks

print("🔄 Loading and chunking PDF text...")
chunks = load_text_chunks()

print("📐 Embedding chunks...")
embeddings = model.encode(chunks, convert_to_numpy=True)

print("📦 Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

with open(os.path.join(VECTOR_DIR, "index.pkl"), "wb") as f:
    pickle.dump(index, f)

chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
with open(os.path.join(VECTOR_DIR, "chunk_map.pkl"), "wb") as f:
    pickle.dump(chunk_map, f)

print("✅ FAISS index and chunk map saved to vectorstore/")

