import os
import logging
import asyncio
from typing import List, Dict
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
import httpx

from utils import clean_response

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Environment vars
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ORG = os.getenv("GROQ_ORG")

if not GROQ_API_KEY or not GROQ_ORG:
    raise ValueError("GROQ_API_KEY or GROQ_ORG is not set")

# Groq API
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Available models
MODELS = {
    "deepseek-r1-distill-llama-70b": "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "qwen-qwq-32b": "qwen-qwq-32b",
    "qwen-2.5-coder-32b": "qwen-2.5-coder-32b"
}

# Pydantic Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "deepseek-r1-distill-llama-70b"
    temperature: float = 0.7
    max_tokens: int = 800
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    model: str

# Async Groq call
async def call_groq_api(payload: Dict) -> Dict:
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            "Groq-Organization": GROQ_ORG
        }
        try:
            response = await client.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30.0
            )
            logger.info(f"Groq API responded: {response.status_code}")
            if response.status_code != 200:
                raise Exception(f"Groq Error {response.status_code}: {response.text}")
            return response.json()
        except httpx.RequestError as e:
            raise Exception(f"Groq API connection error: {str(e)}")

@app.route("/")
def health_check():
    return jsonify({"status": "ok"})

@app.route("/models")
def get_models():
    return jsonify({"models": list(MODELS.keys())})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        chat_request = ChatRequest(**data)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"detail": e.errors()}), 400

    if chat_request.model not in MODELS:
        return jsonify({
            "detail": f"Invalid model. Available models: {', '.join(MODELS.keys())}"
        }), 400

    formatted_messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]

    payload = {
        "model": MODELS[chat_request.model],
        "messages": formatted_messages,
        "temperature": chat_request.temperature,
        "max_tokens": chat_request.max_tokens,
        "stream": chat_request.stream
    }

    async def process_chat():
        try:
            start = asyncio.get_event_loop().time()
            data = await call_groq_api(payload)
            end = asyncio.get_event_loop().time()

            assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not assistant_message:
                raise Exception("Empty response from Groq")

            response_time = (end - start) * 1000
            logger.info(f"Groq response time: {response_time:.2f}ms")

            cleaned = clean_response(assistant_message)
            response_obj = ChatResponse(response=cleaned, model=chat_request.model)
            return jsonify(response_obj.dict())
        except Exception as e:
            logger.exception("Error in Groq processing")
            return jsonify({"detail": str(e)}), 500

    return asyncio.run(process_chat())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)