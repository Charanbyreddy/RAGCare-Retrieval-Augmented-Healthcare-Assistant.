from rag.embedder import embed_text
from rag.vectorstore import load_vector_index, search_similar_chunks
import openai
import requests

# Load FAISS index and chunk mapping
index, chunk_map = load_vector_index()

def get_contextual_answer(query, model, api_key):
    embedded_query = embed_text(query)
    top_chunks = search_similar_chunks(index, chunk_map, embedded_query, k=3)

    context = "\n".join(top_chunks)
    prompt = f"Use the below context to answer the question.\nContext:\n{context}\n\nQuestion: {query}"

    if model == "openai":
        client = openai.OpenAI(api_key=api_key)
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return chat_completion.choices[0].message.content.strip()

    elif model == "groq":
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
        )
        return res.json()["choices"][0]["message"]["content"].strip()

    else:
        raise Exception("Unknown model")
