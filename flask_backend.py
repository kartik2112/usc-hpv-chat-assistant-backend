# HPV Health Assistant - Flask Backend
# Securely proxies OpenAI API requests
# Install: pip install flask flask-cors openai python-dotenv

import os
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import io

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# CORS Configuration - Allow requests from your frontend
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",  # Local development
            "http://localhost:5000",  # Local development
            "https://kartik2112.github.io",  # Replace with your domain
            "*"  # For testing only - restrict in production
        ],
        "methods": ["POST", "GET", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize OpenAI client with API key from environment variable
# NEVER hardcode API keys in code
openai_api_key = os.getenv("OPENAI_API_KEY")
TTS_MODEL = 'tts-1'  # or 'tts-1-hd' for higher quality

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set!")

client = OpenAI(api_key=openai_api_key)

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "HPV Health Assistant Backend is running"
    })

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    """
    Proxy endpoint for OpenAI Chat Completions
    
    Expected JSON payload:
    {
        "messages": [
            {"role": "system", "content": "You are..."},
            {"role": "user", "content": "Hello"}
        ],
        "model": "gpt-3.5-turbo",
        "max_tokens": 800,
        "temperature": 0.7
    }
    """
    
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    try:
        # Get request data
        data = request.get_json()
        
        # Validate required fields
        if not data or "messages" not in data:
            return jsonify({
                "error": "Missing 'messages' field in request"
            }), 400
        
        messages = data.get("messages", [])
        model = data.get("model", "gpt-3.5-turbo")
        max_tokens = data.get("max_tokens", 800)
        temperature = data.get("temperature", 0.7)
        
        # Call OpenAI API (API key is securely stored on backend)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract response content
        assistant_message = response.choices[0].message.content
        
        return jsonify({
            "success": True,
            "message": assistant_message,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500
    
@app.route('/api/tts', methods=['POST', 'OPTIONS'])
def text_to_speech():
    """
    Text-to-Speech endpoint using OpenAI TTS API

    Request body:
    {
        "text": "Text to convert to speech",
        "language": "en" or "es",
        "voice": "alloy", "echo", "fable", "onyx", "nova", or "shimmer"
    }

    Returns:
        Audio MP3 blob with proper CORS headers
    """
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()

        # Validate required fields
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text in request'}), 400

        text = data.get('text', '').strip()
        language = data.get('language', 'en')  # 'en' or 'es'
        voice = data.get('voice', 'echo')  # Default voice

        # Validate text is not empty
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Limit text length to avoid API errors (max 4096 chars)
        if len(text) > 4096:
            text = text[:4096]

        # Validate voice option (OpenAI supports: alloy, echo, fable, onyx, nova, shimmer)
        valid_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        if voice not in valid_voices:
            voice = 'echo' if language == 'en' else 'onyx'

        print(f"[TTS] Generating: text='{text[:50]}...', language={language}, voice={voice}")

        # Call OpenAI TTS API
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=text,
            response_format='mp3'
        )

        # Get audio bytes
        audio_bytes = io.BytesIO(response.content)
        audio_bytes.seek(0)

        print(f"[TTS] Successfully generated audio ({len(response.content)} bytes)")

        # Create response with proper headers
        response_obj = make_response(send_file(
            audio_bytes,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name='speech.mp3'
        ))

        # Ensure CORS headers are set
        response_obj.headers['Access-Control-Allow-Origin'] = '*'
        response_obj.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response_obj.headers['Content-Type'] = 'audio/mpeg'
        response_obj.headers['Cache-Control'] = 'no-cache'

        return response_obj

    except Exception as e:
        print(f"[TTS] Error: {str(e)}")
        error_response = jsonify({'error': str(e)})
        error_response.headers['Access-Control-Allow-Origin'] = '*'
        return error_response, 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "HPV Health Assistant Backend"
    }), 200

if __name__ == "__main__":
    # Development mode - use debug=False in production
    app.run(
        host="0.0.0.0",  # Listen on all interfaces
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("FLASK_ENV") == "development"
    )
