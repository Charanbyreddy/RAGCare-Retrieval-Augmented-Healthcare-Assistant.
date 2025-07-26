# 🩺 RAGCare – Retrieval-Augmented Healthcare Assistant
**Empowering users to ask medical questions and receive LLM-powered answers grounded in real PDF documents.**

RAGCare is an intelligent chatbot built with **Flask**, combining **LLMs (OpenAI + Groq)** and **RAG (Retrieval-Augmented Generation)** to understand and respond to queries using **uploaded or preloaded medical documents**.


---

## 🧩 Features

✅ Upload or use default medical PDFs  
✅ RAG-based intelligent responses  
✅ Choose between **GPT-4 (OpenAI)** and **LLaMA 3 (Groq)**  
✅ Real-time **token usage and latency stats**  
✅ Toggle **dark/light UI**  
✅ **Export** conversation  
✅ Modern UI with chat bubbles, file preview, and scrollable history

---

## 🧠 How It Works

```
         User Query + [PDF or Preloaded Docs]
                     ↓
     Embed query using SentenceTransformer
                     ↓
   Search FAISS index for similar chunks (context)
                     ↓
      Append context → Prompt to LLM via API
                     ↓
     LLM generates answer → Display in UI
```

---

## 🗂️ Project Structure

```
📦 RAGCare/
├── app.py                   # Main Flask app
├── templates/
│   └── index.html           # Frontend UI
├── static/
│   └── style.css            # Custom styling
├── rag/
│   ├── embedder.py          # Sentence embedding
│   ├── vectorstore.py       # FAISS index utils
│   └── rag_utils.py         # Chunk retrieval
├── uploads/                 # PDF input folder
├── vectorstore/
│   └── index.pkl            # Precomputed vector store
├── .env                     # API keys
└── README.md                # You're here!
```

---

## ⚙️ Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/ragcare-health-assistant.git
cd ragcare-health-assistant
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # or use venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Add API keys

Create a `.env` file in the root:

```ini
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
```

### 4. Add documents

Place PDFs inside `uploads/`. These will be embedded and preloaded.

### 5. Run the app

```bash
python app.py
```

Go to: [http://localhost:5000](http://localhost:5000)

---

## 📄 Sample Questions You Can Ask

- “What are the symptoms of diabetes?”
- “Summarize the report I uploaded.”
- “Is the hemoglobin level in this PDF normal?”
- “What medications are used for high BP?”

---

## 🧪 Tech Stack & Tools

| Tool                | Purpose                      |
|---------------------|------------------------------|
| **Flask**           | Web framework                |
| **OpenAI API (GPT-4)** | LLM generation           |
| **Groq API (LLaMA 3)** | High-speed LLM alternative |
| **SentenceTransformer** | Text embeddings         |
| **FAISS**           | Vector similarity search     |
| **HTML/CSS + JS**   | Chat UI                      |
| **dotenv**          | API key management           |

---

## 🧾 `requirements.txt`

```txt
Flask==2.3.2
openai==1.3.6
python-dotenv==1.0.0
requests==2.31.0
sentence-transformers==2.2.2
```

---

## 🔮 Future Improvements

- [ ] Dockerize for 1-click deployment
- [ ] Integrate Pinecone/Chroma for scalable vector DB
- [ ] Add user login & persistent history
- [ ] Stream real-time typing output from LLM
- [ ] Deploy on Render or HuggingFace Spaces

---

## 📤 Export Chat

Export your conversation anytime by clicking the **“Export Chat”** button.

---

## 🏷️ License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for more information.

---

## 🙋‍♂️ Acknowledgments

- [OpenAI](https://openai.com/)
- [Groq](https://groq.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [FontAwesome](https://fontawesome.com/)
- [FAISS](https://github.com/facebookresearch/faiss)

---

## 👨‍💻 Author

**RAGCare** was built by Sri Charan Byreddy as a portfolio project exploring the power of **multi-model RAG systems** in healthcare document analysis.

---
