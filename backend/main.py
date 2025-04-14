import os
import httpx
import asyncio
import logging
from typing import List, Dict
from dotenv import load_dotenv

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utils import clean_response

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("groq-api")

# Initialize FastAPI app
app = FastAPI(title="Groq Chatbot API")

# Get environment variables
GROQ_API_KEY = "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"
GROQ_ORG = "org_01jqcz3ymzf9qv6m0cf2fbga88"

if not GROQ_API_KEY or not GROQ_ORG:
    raise ValueError("GROQ_API_KEY or GROQ_ORG is not set in the environment")

# Groq API endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# CORS Configuration
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")  # Ej: http://localhost:3000,https://mydomain.com

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Available models
MODELS = {
    "deepseek-r1-distill-llama-70b": "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "qwen-qwq-32b": "qwen-qwq-32b",
    "qwen-2.5-coder-32b": "qwen-2.5-coder-32b"
}

# Pydantic models
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

# Groq API Call
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

            logger.info(f"Groq API responded with: {response.status_code}")
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Groq API Error: {response.text}"
                )

            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Groq API connection error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Model requested: {request.model}")

    if request.model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Available models: {', '.join(MODELS.keys())}"
        )

    formatted_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    payload = {
        "model": MODELS[request.model],
        "messages": formatted_messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": request.stream
    }

    try:
        start = asyncio.get_event_loop().time()
        data = await call_groq_api(payload)
        end = asyncio.get_event_loop().time()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            raise HTTPException(status_code=500, detail="No valid response from Groq API")

        logger.info(f"Groq response time: {(end - start) * 1000:.2f}ms")
        return ChatResponse(response=clean_response(content), model=request.model)

    except Exception as e:
        logger.exception("Error while handling chat request")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    return {"models": list(MODELS.keys())}

@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)