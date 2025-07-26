import faiss
import pickle
import os

VECTOR_INDEX_PATH = "vectorstore/index.pkl"
CHUNK_MAP_PATH = "vectorstore/chunk_map.pkl"

def load_vector_index():
    if not os.path.exists(VECTOR_INDEX_PATH) or not os.path.exists(CHUNK_MAP_PATH):
        raise Exception("Vector index or chunk map file not found.")
    with open(VECTOR_INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    with open(CHUNK_MAP_PATH, "rb") as f:
        chunk_map = pickle.load(f)
    return index, chunk_map

def search_similar_chunks(index, chunk_map, embedded_query, k=3):
    scores, indices = index.search(embedded_query.reshape(1, -1), k)
    return [chunk_map[i] for i in indices[0]]
