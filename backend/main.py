import os
import logging
from typing import List, Dict
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import httpx
import asyncio

from utils import clean_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

CORS(app, supports_credentials=True)

# Get Groq API key
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

# Groq API endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Available models
MODELS = {
    "deepseek-r1-distill-llama-70b": "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "qwen-qwq-32b": "qwen-qwq-32b",
    "qwen-2.5-coder-32b": "qwen-2.5-coder-32b",
}


# Helper function to make API calls to Groq
async def call_groq_api(payload: Dict) -> Dict:
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            "Groq-Organization": "org_01jqcz3ymzf9qv6m0cf2fbga88",
        }

        try:
            response = await client.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30.0,  # Extended timeout
            )

            # Log the full response
            logger.error(f"Groq API Response: {response.status_code} - {response.text}")

            if response.status_code != 200:
                error_message = (
                    f"Groq API Error: {response.status_code} - {response.text}"
                )
                return {"error": error_message}

            return response.json()
        except httpx.RequestError as e:
            return {"error": f"Error communicating with Groq API: {str(e)}"}


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    logger.info(f"Received request: {data}")

    # Validate model selection
    model = data.get("model", "deepseek-r1-distill-llama-70b")
    if model not in MODELS:
        return (
            jsonify({"error": f"Model must be one of: {', '.join(MODELS.keys())}"}),
            400,
        )

    # Format messages for the Groq API
    formatted_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in data.get("messages", [])
    ]

    # Prepare the payload for Groq
    payload = {
        "model": MODELS[model],
        "messages": formatted_messages,
        "temperature": data.get("temperature", 0.7),
        "max_tokens": data.get("max_tokens", 800),
        "stream": data.get("stream", False),
    }

    # Call the Groq API
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        start_time = loop.time()
        response_data = loop.run_until_complete(call_groq_api(payload))
        end_time = loop.time()

        if "error" in response_data:
            logger.error(f"Groq API Error: {response_data['error']}")
            return jsonify({"error": response_data["error"]}), 500

        # Extract response text
        assistant_message = (
            response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        )

        if not assistant_message:
            logger.error(
                f"Groq API response did not contain valid 'choices': {response_data}"
            )
            return jsonify({"error": "Invalid response from Groq API"}), 500

        # Log response time
        response_time = (end_time - start_time) * 1000  # in milliseconds
        logger.info(f"Groq response time: {response_time:.2f}ms")

        assistant_message = clean_response(assistant_message)
        return jsonify({"response": assistant_message, "model": model})

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error processing chat request: {str(e)}"}), 500


@app.route("/models", methods=["GET"])
def get_models():
    """Return the list of available models"""
    return jsonify({"models": list(MODELS.keys())})


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT") or 5000)
