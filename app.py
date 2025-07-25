# app.py
import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import openai
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Debug: Print current working directory and list files
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Debug: Print loaded API keys (first 10 chars for security)
print("OPENAI_API_KEY loaded:", "Yes" if OPENAI_API_KEY else "No")
if OPENAI_API_KEY:
    print("OPENAI_API_KEY (first 10 chars):", OPENAI_API_KEY[:10])

print("GROQ_API_KEY loaded:", "Yes" if GROQ_API_KEY else "No")
if GROQ_API_KEY:
    print("GROQ_API_KEY (first 10 chars):", GROQ_API_KEY[:10])

# Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Ask route
@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get("question", "").strip()
    model = request.form.get("model", "openai").strip().lower()
    
    logger.debug(f"Received question: {question}")
    logger.debug(f"Selected model: {model}")
    
    if not question:
        logger.warning("Empty question received")
        return jsonify({"response": "Please enter a question."}), 400
    
    try:
        if model == "openai":
            if not OPENAI_API_KEY:
                raise Exception("OpenAI API key not configured. Please check your .env file.")
            response = openai_chat(question)
        elif model == "groq":
            if not GROQ_API_KEY:
                raise Exception("Groq API key not configured. Please check your .env file.")
            response = groq_chat(question)
        else:
            logger.warning(f"Unknown model selected: {model}")
            return jsonify({"response": "Error: Unknown model selected."}), 400
            
        logger.debug(f"Response generated: {response}")
        return jsonify({"response": response})
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"response": f"Error: {str(e)}"}), 500

# OpenAI GPT-4 response (Fixed version)
def openai_chat(prompt):
    logger.debug("Calling OpenAI API")
    try:
        # Create client without problematic arguments
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Make API call
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using gpt-3.5-turbo as it's more accessible
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Provide accurate health information but always remind users to consult healthcare professionals for medical advice."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        # More detailed error message
        error_msg = str(e)
        if "module 'openai' has no attribute 'OpenAI'" in error_msg:
            return "OpenAI library version issue. Please run: pip install openai>=1.0.0"
        raise Exception(f"OpenAI API error: {error_msg}")

# Groq LLaMA 3 response
def groq_chat(prompt):
    logger.debug("Calling Groq API")
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful medical assistant. Provide accurate health information but always remind users to consult healthcare professionals for medical advice."},
                {"role": "user", "content": prompt}
            ],
            "model": "llama3-8b-8192",
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        logger.debug(f"Groq API response status: {res.status_code}")
        
        if res.status_code != 200:
            logger.error(f"Groq API error: {res.text}")
            raise Exception(f"Groq API error: {res.text}")
            
        response_data = res.json()
        return response_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        raise Exception(f"Groq API error: {str(e)}")

# Health check route
@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "openai_configured": bool(OPENAI_API_KEY),
        "groq_configured": bool(GROQ_API_KEY)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)