import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import openai
import requests
from rag.rag_utils import get_contextual_answer

load_dotenv()
app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get("question", "").strip()
    model = request.form.get("model", "openai").lower()

    if not question:
        return jsonify({"response": "Please enter a question."}), 400

    try:
        if model == "openai":
            if not OPENAI_API_KEY:
                raise Exception("OpenAI API key missing.")
            response = get_contextual_answer(question, model="openai", api_key=OPENAI_API_KEY)
        elif model == "groq":
            if not GROQ_API_KEY:
                raise Exception("Groq API key missing.")
            response = get_contextual_answer(question, model="groq", api_key=GROQ_API_KEY)
        else:
            return jsonify({"response": "Unknown model selected."}), 400

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
