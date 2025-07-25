import os
from rag.embedder import embed_text
from rag.vectorstore import load_vector_store, search_similar_chunks

def get_context_chunks(file_path):
    if not file_path or not os.path.exists(file_path):
        return ""

    try:
        vs = load_vector_store()
        with open(file_path, "r", encoding="utf-8") as f:
            query_text = f.read()
        embedded = embed_text(query_text)
        chunks = search_similar_chunks(embedded, vs, k=3)
        return "\n".join(chunks)
    except FileNotFoundError:
        return ""  # Avoid crashing if index.pkl is missing
    except Exception as e:
        return f"Error in RAG: {str(e)}"
