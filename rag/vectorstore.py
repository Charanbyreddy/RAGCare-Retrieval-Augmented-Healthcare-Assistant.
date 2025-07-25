import pickle
import numpy as np

def load_vector_store():
    with open("vectorstore/index.pkl", "rb") as f:
        return pickle.load(f)

def search_similar_chunks(query_embedding, vector_store, top_k=3):
    scores = [(chunk, cosine_similarity(query_embedding, emb))
              for chunk, emb in vector_store]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
