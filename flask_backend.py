# HPV Health Assistant - Flask Backend
# Securely proxies OpenAI API requests
# Install: pip install flask flask-cors openai python-dotenv

import os
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

AUDIO_MODE = 'a2a'  # Change to 'tts' or 'a2a' to switch modes

logger.info(f"🎯 Backend Audio Mode: {AUDIO_MODE.upper()}")
logger.info(f"   TTS Mode = gpt-3.5-turbo + TTS API")
logger.info(f"   A2A Mode = gpt-4o-audio-preview")

# Configuration
OPENAI_TEXT_MODEL = 'gpt-3.5-turbo'
OPENAI_AUDIO_MODEL = 'gpt-4o-audio-preview'
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

    Expected JSON payload:
    {
        "messages": [
            {"role": "system", "content": "You are..."},
            {"role": "user", "content": "Hello"}
        ],
        "model": "gpt-5-mini",
        "max_tokens": 800
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
        model = data.get("model", "gpt-5-mini")
        max_tokens = data.get("max_tokens", 800)
        # temperature = data.get("temperature", 0.7)
        
        # Call OpenAI API (API key is securely stored on backend)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            # temperature=temperature
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
        voice = data.get('voice', 'nova')  # Default voice
        speed = data.get('speed', 0.7)  # Default speed

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
            response_format='mp3',
            speed=speed
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

# ============================================================================
# A2A ENDPOINT (only used when AUDIO_MODE == 'a2a')
# ============================================================================

@app.route('/api/audio-chat', methods=['POST', 'OPTIONS'])
def audio_chat():
    """Audio-to-audio chat endpoint using gpt-4o-audio-preview"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()

        if not data or 'audio' not in data:
            return jsonify({'error': 'Missing audio data'}), 400

        audio_b64 = data.get('audio')
        audio_format = data.get('audio_format', 'webm')
        language = data.get('language', 'en')
        chat_history = data.get('chat_history', [])

        logger.info(f"[A2A] Received audio (format={audio_format}, lang={language}, history_len={len(chat_history)})")

        # Determine voice based on language
        voice_option = 'echo' if language == 'en' else 'onyx'

        # Build messages for gpt-4o-audio-preview
        messages = [{'role': 'system', 'content': SYSTEM_MESSAGE}]

        # Add recent chat history (limit to last 10)
        if chat_history:
            messages.extend(chat_history[-10:])

        # Add the audio input message
        messages.append({
            'role': 'user',
            'content': [
                {
                    'type': 'input_audio',
                    'input_audio': {
                        'data': audio_b64,
                        'format': audio_format
                    }
                }
            ]
        })

        logger.info(f"[A2A] Calling gpt-4o-audio-preview with {len(messages)} messages")

        # Call OpenAI gpt-4o-audio-preview API
        response = client.chat.completions.create(
            model=OPENAI_AUDIO_MODEL,
            modalities=['text', 'audio'],
            audio={
                'voice': voice_option,
                'format': 'wav'
            },
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )

        # Extract response
        message = response.choices[0].message
        text_response = message.content if message.content else "I heard your message."

        # Get audio response
        audio_response_b64 = None
        if hasattr(message, 'audio') and message.audio:
            audio_response_b64 = message.audio.get('data')

        logger.info(f"[A2A] Response generated (text={len(text_response)} chars, has_audio={audio_response_b64 is not None})")

        return jsonify({
            'text_response': text_response,
            'audio_response': audio_response_b64,
            'transcript': message.audio.get('transcript') if hasattr(message, 'audio') else None
        }), 200

    except Exception as e:
        logger.error(f"[A2A] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# STATUS ENDPOINT - SHOWS CURRENT MODE AND CAPABILITIES
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check with mode information"""
    return jsonify({
        'status': 'ok',
        'audio_mode': AUDIO_MODE.upper(),
        'endpoints': {
            'text_chat': 'ready',
            'tts': 'ready' if AUDIO_MODE == 'tts' or True else 'available',
            'audio_chat': 'ready' if AUDIO_MODE == 'a2a' or True else 'available'
        },
        'models': {
            'text': OPENAI_TEXT_MODEL,
            'audio': OPENAI_AUDIO_MODEL,
            'tts': TTS_MODEL
        }
    }), 200


@app.route('/api/mode', methods=['GET'])
def get_mode():
    """Get current audio mode"""
    return jsonify({
        'current_mode': AUDIO_MODE.upper(),
        'available_modes': ['tts', 'a2a'],
        'mode_descriptions': {
            'tts': 'Text-to-Speech (Browser Speech Rec + TTS API)',
            'a2a': 'Audio-to-Audio (gpt-4o-audio-preview)'
        },
        'instructions': 'To switch modes, update AUDIO_MODE variable and redeploy'
    }), 200

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("Starting HPV Chat Assistant Backend (Dual Mode Support)")
    print("="*80)
    print(f"\n📊 Audio Mode: {AUDIO_MODE.upper()}")

    if AUDIO_MODE == 'tts':
        print("   ✓ Using: Browser Speech Recognition + OpenAI TTS")
        print("   ✓ Endpoint: /api/tts")
        print("   ✓ Model: gpt-3.5-turbo")
    else:
        print("   ✓ Using: OpenAI Audio-to-Audio (gpt-4o-audio-preview)")
        print("   ✓ Endpoint: /api/audio-chat")
        print("   ✓ Model: gpt-4o-audio-preview")

    print(f"\nAlways available:")
    print("   ✓ /api/chat (text chat)")
    print("   ✓ /api/tts (speech synthesis)")
    print("   ✓ /api/audio-chat (audio conversation)")
    print("   ✓ /health (status)")
    print("   ✓ /mode (mode information)")
    print("\n" + "="*80)

    app.run(debug=True, host='0.0.0.0', port=5000)