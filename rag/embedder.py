from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    return np.array(model.encode([text])[0], dtype='float32')
