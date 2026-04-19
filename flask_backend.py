# HPV Health Assistant - Flask Backend
# Securely proxies OpenAI API requests
# Install: pip install flask flask-cors openai python-dotenv

import os
import uuid
import json
import threading
import hmac
import hashlib
import time
import secrets
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from functools import wraps
from flask import Flask, request, jsonify, send_file, make_response, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import io
import re
import logging
import bcrypt
from flask_apscheduler import APScheduler

from rag_pipeline import ask_rag_question, build_rag_agent, ask_rag_question_stream

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Deployment environment detection
#
# Render automatically injects RENDER=true into every service's environment.
# We use this to skip loading the presidio_analyzer + spaCy PHI detection
# engine on Render (512 MB RAM limit) and only enable it on sackend.isi.edu,
# which has sufficient memory.
# ---------------------------------------------------------------------------
IS_RENDER   = bool(os.getenv("RENDER"))          # True on onrender.com
PHI_ENABLED = not IS_RENDER                       # False on Render, True on sackend

# Timezone helper — converts UTC ISO strings (as stored in session files) to
# Pacific Time (America/Los_Angeles handles both PST/UTC-8 and PDT/UTC-7).
_PACIFIC = ZoneInfo('America/Los_Angeles')

def to_pst(iso_utc: str | None) -> str | None:
    """Convert a UTC ISO-8601 string to a Pacific-time ISO-8601 string.

    Accepts both naive strings (assumed UTC) and strings ending in 'Z' or
    '+00:00'. Returns None unchanged so callers don't need to guard."""
    if not iso_utc:
        return None
    try:
        # Normalise 'Z' suffix which fromisoformat() rejects on Python < 3.11
        normalised = iso_utc.replace('Z', '+00:00')
        dt_utc = datetime.fromisoformat(normalised)
        # If stored as naive (no tzinfo), treat as UTC
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        dt_pst = dt_utc.astimezone(_PACIFIC)
        return dt_pst.isoformat()
    except (ValueError, TypeError):
        return iso_utc  # return original string if parsing fails

class Config(object):
    SCHEDULER_API_ENABLED = True
app = Flask(__name__)
app.config.from_object(Config())
scheduler = APScheduler()

# ---------------------------------------------------------------------------
# CORS — allow all origins unconditionally.
#
# Why "*" and not a list of specific origins?
#
#   This API uses server-side secrets (OPENAI_API_KEY stored in Render's
#   environment). The browser never sends credentials (cookies, auth headers)
#   so there is no security risk in using a wildcard origin.
#
#   The browser's CORS check only protects against *other websites reading
#   your responses*. Since every response here is meant to be read by the
#   calling page (kartik2112.github.io), allowing all origins is correct.
#
#   Restricting to a list would only matter if the API issued auth tokens
#   tied to the browser session — it doesn't.
#
# How CORS works (short version):
#   1. Browser sends a preflight OPTIONS request asking "can I call you?".
#   2. Server replies with Access-Control-Allow-Origin: *.
#   3. Browser sees * → allows the real GET/POST request to proceed.
#   4. Without the header the browser blocks the response in JS — the server
#      still received and processed the request, the client just can't read it.
#
# The @app.after_request hook guarantees the header is present on *every*
# response Flask ever sends, including unhandled 500 errors that bypass the
# flask-cors middleware.
# ---------------------------------------------------------------------------
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

@app.after_request
def ensure_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# ---------------------------------------------------------------------------
# Server-side domain allowlist  (independent of CORS)
#
# CORS is a *browser* mechanism — the browser decides whether JS can read a
# response based on the Access-Control-Allow-Origin header.  It does NOT
# prevent the server from receiving and processing the request.
#
# This before_request hook adds a real server-side gate: if the Origin header
# sent by the browser is not in ALLOWED_ORIGINS, Flask returns 403 before any
# business logic runs.  The 403 still carries Access-Control-Allow-Origin: *
# (added by ensure_cors_headers above) so the browser CAN read the error
# body — the request is blocked by our code, not by the browser's CORS check.
#
# Requests without an Origin header (curl, Postman, server-to-server) are
# allowed through unconditionally, because only browsers attach Origin.
# If you also want to block non-browser callers, check Referer or add an
# API-key header instead.
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = {
    "https://kartik2112.github.io",
    "https://dipsurvey-ann.isi.edu",
    "https://sackend.isi.edu",
}

@app.before_request
def restrict_to_allowed_origins():
    origin = request.headers.get("Origin", "")
    # No Origin header → non-browser client (curl, Postman, etc.) → allow through
    if not origin:
        return
    if origin not in ALLOWED_ORIGINS:
        logger.warning(f"Rejected request from disallowed origin: {origin}")
        return jsonify({"error": "Origin not allowed"}), 403

# Initialize OpenAI client with API key from environment variable
# NEVER hardcode API keys in code
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configuration
OPENAI_TEXT_MODEL = "gpt-5-mini"
MAX_COMPLETION_TOKENS = 1200
OPENAI_AUDIO_MODEL = 'gpt-4o-audio-preview'
TTS_MODEL = 'tts-1'  # or 'tts-1-hd' for higher quality
AUDIO_MODE = 'a2a'  # Change to 'tts' or 'a2a' to switch modes

logger.info(f"🎯 Backend Audio Mode: {AUDIO_MODE.upper()}")
logger.info(f"   TTS Mode = {OPENAI_TEXT_MODEL} + TTS API")
logger.info(f"   A2A Mode = gpt-4o-audio-preview")


if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set!")

client = OpenAI(api_key=openai_api_key)
rag_agent, _rag_pipeline = build_rag_agent(openai_text_model=OPENAI_TEXT_MODEL, max_completion_tokens=MAX_COMPLETION_TOKENS)


def daily_task():
    """Your daily logic goes here (e.g., making an API call, cleaning up data)"""
    global rag_agent, _rag_pipeline
    print("Daily RAG Refresh task is running...")
    rag_agent, _rag_pipeline = build_rag_agent(openai_text_model=OPENAI_TEXT_MODEL, max_completion_tokens=MAX_COMPLETION_TOKENS)

# PHI detection engine — only loaded on sackend (not on Render).
# presidio_analyzer + spaCy's en_core_web_md model consumes ~300 MB of RAM,
# which exhausts the 512 MB limit on Render's free tier and causes OOM crashes.
if PHI_ENABLED:
    import spacy
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # Always load the English model.
    _phi_models = [{"lang_code": "en", "model_name": "en_core_web_md"}]
    _phi_supported_languages = ["en"]

    # Attempt to load the Spanish model.  If it is not installed the engine
    # degrades gracefully: Spanish text falls back to the English model (still
    # catches numeric / format-based PHI such as SSNs, phones, and e-mails).
    # To install the model run:  python -m spacy download es_core_news_md
    try:
        spacy.load("es_core_news_md")
        _phi_models.append({"lang_code": "es", "model_name": "es_core_news_md"})
        _phi_supported_languages.append("es")
        logger.info("Spanish spaCy model (es_core_news_md) loaded — bilingual PHI detection enabled")
    except OSError:
        logger.warning(
            "Spanish spaCy model (es_core_news_md) not found. "
            "Spanish PHI detection will fall back to the English model. "
            "Run:  python -m spacy download es_core_news_md"
        )

    _nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": _phi_models,
    }).create_engine()
    pii_analysis_service = AnalyzerEngine(
        nlp_engine=_nlp_engine,
        supported_languages=_phi_supported_languages,
    )
    logger.info(f"PHI detection engine loaded (languages: {_phi_supported_languages})")
else:
    pii_analysis_service = None
    _phi_supported_languages = []
    logger.info("PHI detection disabled (Render deployment — PHI_ENABLED=False, saving ~300 MB RAM)")

# ---------------------------------------------------------------------------
# Explicit HIPAA PHI entity allowlist
# ---------------------------------------------------------------------------
# Only these Presidio entity types are checked. Omitting broad types like
# LOCATION, ORGANIZATION, NRP, and URL prevents false positives on common
# medical terms (e.g. spaCy's NER misidentifying "HPV" as a PERSON name).
# Reference: 45 CFR §164.514(b) — HIPAA Safe Harbour 18 identifiers.
# ---------------------------------------------------------------------------
HIPAA_PHI_ENTITIES = [
    "PERSON",             # Patient / provider names
    "PHONE_NUMBER",       # Phone and fax numbers
    "EMAIL_ADDRESS",      # Email addresses
    "US_SSN",             # Social Security Numbers
    "CREDIT_CARD",        # Credit / debit card numbers
    "US_BANK_NUMBER",     # Bank account numbers
    "US_DRIVER_LICENSE",  # Driver's licence numbers
    "US_PASSPORT",        # Passport numbers
    "MEDICAL_LICENSE",    # Medical licence / DEA numbers
    "IP_ADDRESS",         # Device IP addresses
]

# Minimum confidence score for Presidio detections.
#
# Score reference (measured against en_core_web_md / es_core_news_md):
#   PERSON (spaCy NER)   → always exactly 0.85  — capped by spaCy's pipeline
#   EMAIL_ADDRESS        → always 1.00           — pure regex pattern
#   IP_ADDRESS           → 0.60                  — already below 0.85, never fires
#   MRN / NHC            → custom regex below    — bypasses Presidio scoring
#
# Raising the threshold above spaCy's NER ceiling (0.85) means PERSON
# entities are no longer reported by Presidio.  This eliminates the entire
# class of NER-based false positives — capitalised Spanish adjectives,
# medical acronyms, etc. — while keeping the genuinely high-precision
# detections: email addresses (1.00) and custom-regex MRNs.
#
# The stopword filter added below remains in place as defence-in-depth for
# any future model that may return a PERSON score above this threshold.
PHI_SCORE_THRESHOLD = 0.90

# ---------------------------------------------------------------------------
# Spanish PERSON-entity stopword filter
# ---------------------------------------------------------------------------
# The Spanish spaCy NER model (es_core_news_md) occasionally folds common
# function words into a PERSON span — e.g. tagging "es Importante" as a
# person name in "Porque es Importante una biopsia".  The two-token minimum
# check already eliminates single-word false positives, but multi-word
# function-word combinations slip through.
#
# This set contains Spanish articles, prepositions, conjunctions, pronouns,
# common verbs, and high-frequency adjectives that should NEVER appear as a
# token inside a genuine person name.  If any token in a PERSON span matches
# (case-insensitive) an entry here, the detection is discarded as a false
# positive.  Genuine names ("María García", "Dr. Martínez") contain none of
# these words and are not affected.
# ---------------------------------------------------------------------------
_SPANISH_NAME_STOPWORDS: frozenset[str] = frozenset({
    # articles
    "el", "la", "los", "las", "lo", "un", "una", "unos", "unas",
    # prepositions & contractions
    "a", "al", "ante", "bajo", "cabe", "con", "contra", "de", "del",
    "desde", "durante", "en", "entre", "hacia", "hasta", "para",
    "por", "sin", "sobre", "tras",
    # conjunctions
    "e", "ni", "o", "pero", "porque", "que", "si", "sino", "u", "y", "ya",
    # pronouns
    "él", "ella", "ellas", "ello", "ellos", "le", "les", "me", "mi",
    "mis", "nos", "os", "se", "su", "sus", "te", "ti", "tú", "usted",
    "ustedes", "vos", "yo",
    # high-frequency verbs (forms that appear in capitalised positions)
    "es", "está", "están", "fue", "hay", "ser", "son", "tiene", "tienen",
    # demonstratives & determiners
    "esa", "esas", "ese", "esos", "esta", "estas", "este", "esto", "estos",
    "aquel", "aquella", "aquellos", "aquellas",
    # common adverbs
    "aquí", "ahí", "allí", "así", "bien", "como", "cuando", "donde",
    "más", "menos", "muy", "no", "nunca", "siempre", "también", "tan",
    "todo", "ya",
    # common adjectives that are sometimes capitalised for emphasis
    "actual", "común", "comun", "especial", "general", "importante",
    "necesaria", "necesario", "normal", "nuevo", "nueva", "posible",
    "principal",
})


def detect_phi_backend(text, language="en"):
    """Detect HIPAA-relevant PHI in text using Presidio + spaCy (local only).

    Returns an empty list immediately when PHI_ENABLED is False (Render
    deployment), so no memory is consumed by the NLP model on that host.

    On sackend (PHI_ENABLED=True) it returns a list of detected entity type
    strings, or an empty list when no PHI is found. Only the entity types in
    HIPAA_PHI_ENTITIES are checked, and only detections above
    PHI_SCORE_THRESHOLD are reported — this prevents medical acronyms like
    'HPV' from triggering false positives.

    Args:
        text:     The user message to scan.
        language: BCP-47 language code sent by the client ("en" or "es").
                  Falls back to "en" if the requested language model is not
                  installed.
    """
    if not PHI_ENABLED:
        return []

    # Use the requested language only when its model was successfully loaded;
    # otherwise degrade gracefully to English.
    lang = language if language in _phi_supported_languages else "en"

    detections = set()

    results = pii_analysis_service.analyze(
        text=text,
        language=lang,
        entities=HIPAA_PHI_ENTITIES,
        score_threshold=PHI_SCORE_THRESHOLD,
    )
    for result in results:
        # For PERSON entities, require the matched span to contain at least two
        # whitespace-separated tokens (e.g. "Jane Doe", "Dr. García").
        # Single capitalised words — such as "Importante" in Spanish or medical
        # abbreviations — are routinely mis-tagged as person names by spaCy's NER
        # models in both English and Spanish.  Real patient / provider names in
        # conversational text virtually always include a first AND last name.
        if result.entity_type == "PERSON":
            matched_text = text[result.start:result.end].strip()
            tokens = matched_text.split()
            if len(tokens) < 2:
                continue
            # For Spanish text, discard PERSON spans that contain any token
            # matching a known Spanish function word / stopword.  The Spanish
            # NER model often wraps adjacent function words (e.g. "es",
            # "Importante") into a PERSON span, producing false positives on
            # ordinary medical questions.  Real patient names ("María García",
            # "Dr. Ramírez") never contain these words.
            if lang == "es" and any(t.lower() in _SPANISH_NAME_STOPWORDS for t in tokens):
                continue
        detections.add(result.entity_type)

    # Medical Record Numbers — custom regex not covered by Presidio's built-ins.
    # English: "MRN: 12345678" or "MR# 9876"
    # Spanish: "NHC: 12345678" (Número de Historia Clínica)
    if re.search(r'\b(?:MRN|mrn|MR#|mr#|NHC|nhc)[\s:]*\d{4,12}\b', text):
        detections.add('MEDICAL_RECORD_NUMBER')

    return list(detections)


# ============================================================================
# PHI GUARDRAIL SELF-TEST
# ============================================================================
# Runs automatically at server startup (module import time) whenever
# PHI_ENABLED=True.  Any failure aborts startup with a RuntimeError so a
# misconfigured or regressed PHI engine is never silently deployed.
#
# The same sentences live in test_phi_guardrails.py for use with pytest.
# Keep the two lists in sync when adding new edge-case sentences.
# ============================================================================

_PHI_SELFTEST_CLEAN_EN = [
    "What is HPV and how common is it?",
    "HPV is the most common sexually transmitted infection in the United States.",
    "Most people who are sexually active will get HPV at some point in their lives.",
    "There are more than 200 types of HPV, and about 40 affect the genital area.",
    "High-risk HPV types can cause cervical, anal, and throat cancers.",
    "Low-risk HPV types like 6 and 11 cause most genital warts.",
    "Many HPV infections clear on their own without causing health problems.",
    "HPV can be transmitted through vaginal, anal, or oral sex.",
    "You can have HPV without knowing it because it often causes no symptoms.",
    "HPV spreads through skin-to-skin contact, not through bodily fluids.",
    "The HPV vaccine is safe and highly effective.",
    "Gardasil 9 protects against nine types of HPV.",
    "The vaccine is recommended for boys and girls starting at age eleven or twelve.",
    "Adults up to age forty-five can still benefit from the HPV vaccine.",
    "It is best to get vaccinated before becoming sexually active.",
    "Three doses may be recommended for people who start the vaccine series after age fifteen.",
    "The HPV vaccine does not contain live virus and cannot cause an HPV infection.",
    "Side effects of the HPV vaccine are usually mild and include soreness at the injection site.",
    "Millions of people worldwide have received the HPV vaccine without serious side effects.",
    "Vaccination can reduce the risk of HPV-related cancers by up to ninety-nine percent.",
    "A Pap smear collects cells from the cervix to look for abnormal changes.",
    "Women between twenty-one and sixty-five should get regular cervical cancer screenings.",
    "An HPV test can detect high-risk HPV types even before cell changes occur.",
    "A colposcopy is a procedure used to examine the cervix more closely.",
    "What is the difference between a Pap test and an HPV test?",
    "Regular screening can detect precancerous changes early, when they are easiest to treat.",
    "Women aged thirty to sixty-five may have a co-test every five years.",
    "An abnormal Pap smear result does not necessarily mean you have cancer.",
    "How long does it take for HPV to develop into cervical cancer?",
    "It can take fifteen to twenty years for HPV to progress to cervical cancer.",
    "Why is a biopsy important for diagnosing cervical changes?",
    "A biopsy removes a small tissue sample to check for abnormal or cancerous cells.",
    "LEEP is a procedure that removes abnormal cervical tissue with an electric wire loop.",
    "Cryotherapy can be used to freeze and destroy abnormal cervical cells.",
    "Genital warts can be treated with topical medications or removed surgically.",
    "There is currently no cure for HPV itself, but most infections clear naturally.",
    "The immune system usually clears an HPV infection within one to two years.",
    "Persistent high-risk HPV infection is the main cause of cervical cancer.",
    "BRCA mutations are a separate risk factor for breast and ovarian cancer.",
    "HER2 is a protein that can affect how breast cancer cells grow.",
    "Using condoms consistently can reduce but not eliminate the risk of HPV transmission.",
    "Limiting the number of sexual partners lowers the risk of HPV exposure.",
    "Quitting smoking can reduce the risk of HPV-related cancers.",
    "A strong immune system helps the body clear HPV infections faster.",
    "Can HPV be transmitted through kissing or casual contact?",
    "HPV is not spread through toilet seats, doorknobs, or swimming pools.",
    "How effective is the HPV vaccine in preventing genital warts?",
    "What should I do if my HPV test comes back positive?",
    "Is HPV the same as HIV or herpes?",
    "Talk to your healthcare provider about the best screening schedule for you.",
]

_PHI_SELFTEST_CLEAN_ES = [
    "Porque es Importante una biopsia",
    "¿Por qué es importante realizarse una biopsia?",
    "El VPH es la infección de transmisión sexual más común en los Estados Unidos.",
    "La mayoría de las personas sexualmente activas contraerán el VPH en algún momento.",
    "Existen más de doscientos tipos de VPH, y alrededor de cuarenta afectan el área genital.",
    "Los tipos de VPH de alto riesgo pueden causar cáncer de cuello uterino, anal y de garganta.",
    "Los tipos de VPH de bajo riesgo como el seis y el once causan la mayoría de las verrugas genitales.",
    "Muchas infecciones por VPH desaparecen solas sin causar problemas de salud.",
    "El VPH se puede transmitir durante las relaciones sexuales vaginales, anales u orales.",
    "Puede tener VPH sin saberlo porque a menudo no causa síntomas.",
    "La vacuna contra el VPH es segura y muy eficaz.",
    "Gardasil 9 protege contra nueve tipos de VPH.",
    "La vacuna se recomienda para niños y niñas a partir de los once o doce años.",
    "Los adultos de hasta cuarenta y cinco años aún pueden beneficiarse de la vacuna contra el VPH.",
    "Es mejor vacunarse antes de iniciar la actividad sexual.",
    "La vacuna no contiene virus vivo y no puede causar una infección por VPH.",
    "Los efectos secundarios de la vacuna contra el VPH suelen ser leves.",
    "Millones de personas en todo el mundo han recibido la vacuna sin efectos secundarios graves.",
    "¿A qué edad se recomienda comenzar la serie de vacunas?",
    "La vacunación puede reducir el riesgo de cánceres relacionados con el VPH hasta en un noventa y nueve por ciento.",
    "Una prueba de Papanicolaou recolecta células del cuello uterino para detectar cambios anormales.",
    "Las mujeres entre veintiún y sesenta y cinco años deben realizarse exámenes de detección regulares.",
    "Una prueba del VPH puede detectar tipos de VPH de alto riesgo antes de que ocurran cambios celulares.",
    "¿Cuál es la diferencia entre una prueba de Papanicolaou y una prueba del VPH?",
    "Los exámenes regulares pueden detectar cambios precancerosos cuando son más fáciles de tratar.",
    "Un resultado anormal en la prueba de Papanicolaou no significa necesariamente que tenga cáncer.",
    "¿Cuánto tiempo tarda el VPH en convertirse en cáncer de cuello uterino?",
    "Pueden pasar entre quince y veinte años para que el VPH progrese a cáncer cervical.",
    "La colposcopía es un procedimiento que examina el cuello uterino con más detalle.",
    "¿Con qué frecuencia debo hacerme pruebas de detección del cáncer de cuello uterino?",
    "¿Por qué es necesaria una biopsia para diagnosticar cambios cervicales?",
    "Una biopsia extrae una pequeña muestra de tejido para detectar células anormales o cancerosas.",
    "El procedimiento LEEP elimina tejido cervical anormal con un aro de alambre eléctrico.",
    "La crioterapia puede usarse para congelar y destruir células cervicales anormales.",
    "Las verrugas genitales se pueden tratar con medicamentos tópicos o extirpar quirúrgicamente.",
    "Actualmente no existe una cura para el VPH en sí, pero la mayoría de las infecciones desaparecen naturalmente.",
    "El sistema inmunológico generalmente elimina una infección por VPH en uno o dos años.",
    "La infección persistente por VPH de alto riesgo es la principal causa del cáncer cervical.",
    "¿Qué sucede si la prueba del VPH resulta positiva?",
    "Es posible que su médico recomiende un seguimiento más frecuente si tiene VPH de alto riesgo.",
    "Usar condones de manera constante puede reducir, pero no eliminar, el riesgo de transmisión del VPH.",
    "Limitar el número de parejas sexuales reduce el riesgo de exposición al VPH.",
    "Dejar de fumar puede reducir el riesgo de cánceres relacionados con el VPH.",
    "Un sistema inmunológico fuerte ayuda al cuerpo a eliminar las infecciones por VPH más rápidamente.",
    "¿Se puede transmitir el VPH a través de los besos o el contacto casual?",
    "El VPH no se propaga a través de asientos de inodoro, manijas de puertas o piscinas.",
    "¿Es el VPH lo mismo que el VIH o el herpes?",
    "Consulte con su proveedor de atención médica sobre el mejor cronograma de detección para usted.",
    "¿Qué tan común es la infección por VPH entre los hombres?",
    "Los hombres también pueden transmitir el VPH aunque no tengan síntomas visibles.",
]

# High-confidence PHI that must still be detected (email = 1.00, MRN = custom regex).
_PHI_SELFTEST_PHI = [
    ("Please email me at jsmith@example.com with the results.", "en"),
    ("Send the report to maria.garcia@hospital.org as soon as possible.", "en"),
    ("Contact the clinic at info@healthcenter.org for an appointment.", "en"),
    ("My MRN: 00012345", "en"),
    ("Reference number MRN: 9876543", "en"),
    ("Patient reference NHC: 00056789", "es"),
    ("Mi número de historia clínica NHC: 12345678", "es"),
    ("Por favor envíame un correo a paciente@correo.com con los resultados.", "es"),
    ("Escríbeme a consulta@clinica.mx para más información.", "es"),
]


def _run_phi_selftest() -> None:
    """Validate the PHI detection engine against known-clean and known-PHI sentences.

    Called once at module import time whenever PHI_ENABLED is True.  Raises
    RuntimeError on the first failure so the server refuses to start with a
    misconfigured or regressed guardrail.
    """
    false_positives: list[tuple[str, str, list]] = []
    false_negatives: list[tuple[str, str]] = []

    for sentence in _PHI_SELFTEST_CLEAN_EN:
        hits = detect_phi_backend(sentence, language="en")
        if hits:
            false_positives.append(("en", sentence, hits))

    for sentence in _PHI_SELFTEST_CLEAN_ES:
        hits = detect_phi_backend(sentence, language="es")
        if hits:
            false_positives.append(("es", sentence, hits))

    for sentence, lang in _PHI_SELFTEST_PHI:
        if not detect_phi_backend(sentence, language=lang):
            false_negatives.append((lang, sentence))

    en_fp = sum(1 for l, *_ in false_positives if l == "en")
    es_fp = sum(1 for l, *_ in false_positives if l == "es")
    logger.info(
        f"PHI self-test: EN {50 - en_fp}/50 clean  |  "
        f"ES {50 - es_fp}/50 clean  |  "
        f"true-positives {len(_PHI_SELFTEST_PHI) - len(false_negatives)}/{len(_PHI_SELFTEST_PHI)}"
    )

    if false_positives or false_negatives:
        lines = ["PHI guardrail self-test FAILED — server startup aborted."]
        for lang, sentence, hits in false_positives:
            lines.append(f"  FALSE POSITIVE [{lang}]: '{sentence}' → {hits}")
        for lang, sentence in false_negatives:
            lines.append(f"  FALSE NEGATIVE [{lang}]: '{sentence}' (expected detection, got none)")
        raise RuntimeError("\n".join(lines))


if PHI_ENABLED:
    _run_phi_selftest()


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

SESSIONS_DIR = 'sessions'
SESSION_TIMEOUT_MINUTES = 2
os.makedirs(SESSIONS_DIR, exist_ok=True)

sessions = {}          # { session_id: { events, messages, last_activity, created_at } }
sessions_lock = threading.Lock()

# ── Sessions Dashboard Auth ───────────────────────────────────────────────────
# Password hash is stored as an environment variable — never in source code.
# To generate a hash for your chosen password run:
#   python3 -c "import bcrypt; print(bcrypt.hashpw(b'YOUR_PASSWORD', bcrypt.gensalt(rounds=12)).decode())"
# Then set SESSIONS_PASSWORD_HASH=<output> in Render's environment variables.
#
# SESSIONS_TOKEN_SECRET is used to sign dashboard access tokens.
# Set it to any long random string in Render's environment variables, e.g.:
#   python3 -c "import secrets; print(secrets.token_hex(32))"
# If not set, a random secret is generated at startup (tokens won't survive restarts).
SESSIONS_PASSWORD_HASH = os.environ.get('SESSIONS_PASSWORD_HASH', '')
SESSIONS_TOKEN_SECRET  = os.environ.get('SESSIONS_TOKEN_SECRET', secrets.token_hex(32))
SESSIONS_TOKEN_EXPIRY  = 7200   # 2 hours in seconds

# In-memory rate-limit store: { ip: (failure_count, lockout_until_unix) }
_auth_rate_limit: dict = {}
_AUTH_MAX_FAILURES = 5
_AUTH_LOCKOUT_SECONDS = 900   # 15 minutes


def _make_dashboard_token() -> str:
    """Return a signed, expiring token for dashboard access."""
    expiry = str(int(time.time()) + SESSIONS_TOKEN_EXPIRY)
    sig = hmac.new(SESSIONS_TOKEN_SECRET.encode(), expiry.encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{sig}"


def _validate_dashboard_token(token: str) -> bool:
    """Return True iff the token is unexpired and its HMAC is valid."""
    if not token:
        return False
    try:
        expiry_str, sig = token.rsplit('.', 1)
        expiry = int(expiry_str)
    except (ValueError, AttributeError):
        return False
    if expiry < int(time.time()):
        return False
    expected = hmac.new(SESSIONS_TOKEN_SECRET.encode(), expiry_str.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def require_dashboard_token(f):
    """Decorator: reject requests that don't carry a valid dashboard token.
    OPTIONS preflights are always passed through so CORS handshakes work."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method == 'OPTIONS':
            return f(*args, **kwargs)   # let the route handle the preflight
        auth_header = request.headers.get('Authorization', '')
        token = auth_header.removeprefix('Bearer ').strip()
        if not _validate_dashboard_token(token):
            return jsonify({'error': 'Unauthorized. Please authenticate.'}), 401
        return f(*args, **kwargs)
    return decorated


def generate_session_summary(messages):
    """Call the LLM to produce a patient-questions + doctor-action-items summary.

    PHI-flagged messages are stripped before sending to the LLM as a
    defence-in-depth measure (the frontend should not store them in the
    session at all, but we filter here too to be safe).
    """
    try:
        if not messages:
            return {"patient_questions": "No conversation recorded.", "action_items": ""}

        # Remove any message whose content contains PHI so it is never sent to the LLM.
        clean_messages = [
            m for m in messages
            if not detect_phi_backend(m.get("content", ""))
        ]

        if not clean_messages:
            return {"patient_questions": "No conversation recorded.", "action_items": ""}

        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in clean_messages
        )
        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a clinical documentation assistant. Given a patient–assistant "
                    "conversation about HPV, produce a concise JSON summary with exactly two fields:\n"
                    "1. 'patient_questions': A bullet-point list of the main questions and concerns "
                    "raised by the patient. (ONLY CAPTURE QUESTIONS ASKED BY THE USER IN THE JSON. DO NO HALLUCINATE. DO NOT ADD ANY ADDITIONAL QUESTIONS)\n"
                    "2. 'action_items': A bullet-point list of follow-up action items for the "
                    "healthcare provider to initiate the conversation in that direction. "
                    "These should only be with respect to questions that indicate misconceptions about "
                    "HPV the patient demonstrated with their questions. "
                    "Do not capture all possible actions that a provider should do. Focus on the few most important ones based on the conversation. "
                    "Please refrain from converting every question into an action item. Keep this extremely brief, concise and to the point.\n"
                    "Return ONLY valid JSON. Example:\n"
                    '{"patient_questions": "• Question 1\\n• Question 2", '
                    '"action_items": "• Action 1\\n• Action 2"}'
                )
            },
            {
                "role": "user",
                "content": f"Conversation:\n{conversation_text}"
            }
        ]
        response = client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            messages=summary_prompt,
            max_completion_tokens=5000,
            response_format={"type": "json_object"}   # forces raw JSON — no markdown fences
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            raise ValueError("LLM returned empty content")
        return json.loads(raw)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}\nResponse received: {response}")
        return {"patient_questions": "Summary unavailable.", "action_items": ""}


def _reconstruct_messages_from_events(events):
    """Build a conversation messages list from event records.

    Used as a fallback in save_session_to_disk when the messages array is
    empty — e.g. the session expired mid-conversation before any sync landed.
    Only conversation-type events are included; system/analytics events are
    skipped so the conversation tab stays in sync with what the events tab shows.
    """
    _ROLE_MAP = {
        'User Text Query':          'user',
        'User Voice Query':         'user',
        'Assistant Text Response':  'assistant',
        'Assistant Voice Response': 'assistant',
    }
    messages = []
    for evt in events:
        role = _ROLE_MAP.get(evt.get('type', ''))
        if role and evt.get('details', '').strip():
            messages.append({'role': role, 'content': evt['details']})
    return messages


def save_session_to_disk(session_id, session_data, summary):
    """Write session JSON and a human-readable TXT transcript to the sessions/ folder."""
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    ended_at  = datetime.utcnow()
    created_at = session_data["created_at"]

    # Use the synced messages array; fall back to reconstructing from events if
    # it is empty (can happen when a session expires mid-conversation before any
    # syncMessagesToSession() call reached the server).  This guarantees the
    # conversation tab and the events tab are never out of sync.
    raw_messages = session_data.get("messages", [])
    if not raw_messages:
        raw_messages = _reconstruct_messages_from_events(session_data.get("events", []))
        if raw_messages:
            logger.info(
                f"Session {session_id}: messages reconstructed from events "
                f"({len(raw_messages)} entries)"
            )

    # Strip PHI-flagged messages before writing to disk (defence-in-depth).
    clean_messages = [
        m for m in raw_messages
        if not detect_phi_backend(m.get("content", ""))
    ]

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_filename = os.path.join(SESSIONS_DIR, f"session_{session_id}_{timestamp}.json")
    payload = {
        "session_id": session_id,
        "created_at": created_at.isoformat(),
        "ended_at":   ended_at.isoformat(),
        "events":     session_data.get("events", []),
        "messages":   clean_messages,
        "summary":    summary
    }
    with open(json_filename, 'w') as f:
        json.dump(payload, f, indent=2)

    # ── TXT ───────────────────────────────────────────────────────────────────
    txt_filename = os.path.join(SESSIONS_DIR, f"session_{session_id}_{timestamp}.txt")

    duration_secs = int((ended_at - created_at).total_seconds())
    duration_str  = f"{duration_secs // 60}m {duration_secs % 60}s"

    lines = [
        "HPV Health Assistant — Session Transcript",
        "=" * 60,
        f"Session ID : {session_id}",
        f"Started    : {created_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"Ended      : {ended_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"Duration   : {duration_str}",
        "",
        "─" * 60,
        "CONVERSATION",
        "─" * 60,
    ]
    if clean_messages:
        for msg in clean_messages:
            role    = "Patient" if msg.get("role") == "user" else "Assistant"
            content = (msg.get("content") or "").strip()
            lines.append(f"\n[{role}]")
            lines.append(content)
    else:
        lines.append("(no messages recorded)")

    lines += [
        "",
        "─" * 60,
        "EVENT LOG",
        "─" * 60,
    ]
    events = session_data.get("events", [])
    if events:
        for evt in events:
            try:
                offset = int((datetime.fromisoformat(evt.get("timestamp", ended_at.isoformat()))
                              - created_at).total_seconds())
                offset_str = f"+{offset // 60}m{offset % 60}s" if offset >= 60 else f"+{offset}s"
            except Exception:
                offset_str = "?"
            evt_type = evt.get("type", "")
            lang     = evt.get("language", "")
            details  = evt.get("details", "")
            lines.append(f"{offset_str:<10} {evt_type:<30} {lang:<6} {details}")
    else:
        lines.append("(no events recorded)")

    lines += [
        "",
        "─" * 60,
        "SUMMARY",
        "─" * 60,
        "",
        "Patient Questions:",
        summary.get("patient_questions", "—"),
        "",
        "Provider Action Items:",
        summary.get("action_items", "—"),
        "",
    ]

    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    logger.info(f"Session {session_id} saved → {json_filename} + {txt_filename}")
    return json_filename


def auto_expire_sessions():
    """APScheduler job: expire sessions inactive for > SESSION_TIMEOUT_MINUTES."""
    cutoff = datetime.utcnow() - timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    with sessions_lock:
        expired_ids = [sid for sid, s in sessions.items() if s["last_activity"] < cutoff]
    for sid in expired_ids:
        with sessions_lock:
            session_data = sessions.pop(sid, None)
        if session_data:
            # Use synced messages; fall back to event reconstruction if the session
            # expired before any syncMessagesToSession() call completed.
            messages = session_data.get("messages", []) or \
                       _reconstruct_messages_from_events(session_data.get("events", []))
            summary = generate_session_summary(messages)
            save_session_to_disk(sid, session_data, summary)
            logger.info(f"Auto-expired session {sid}")


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
    Streaming proxy endpoint for RAG-powered chat completions.

    Accepts the same JSON payload as before:
        { "messages": [...], "language": "en"|"es", ... }

    Returns a Server-Sent Events (text/event-stream) response so the
    frontend can render tokens as they arrive.  Each SSE event carries a
    JSON payload with a "type" discriminator:

        {"type": "token",       "token": "<text chunk>"}
        {"type": "done",        "message": "<full text>", "rag_used": true,
                                "phi_warning": false, "phi_types": []}
        {"type": "phi_warning", "message": "<warning text>",
                                "phi_warning": true, "phi_types": [...]}
        {"type": "error",       "error": "<message>"}
    """

    # Handle CORS preflight
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "Missing 'messages' field in request"}), 400

    messages = data.get("messages", [])

    # Language sent by the client ("en" or "es"). Used to select the
    # correct spaCy model inside detect_phi_backend().
    language = data.get("language", "en")

    # ── PHI check (fast, synchronous — runs before the stream opens) ─────────
    phi_warning = False
    phi_types   = []
    if messages:
        last_user_msg = next(
            (m['content'] for m in reversed(messages) if m.get('role') == 'user'),
            None,
        )
        if last_user_msg:
            phi_types = detect_phi_backend(last_user_msg, language=language)
            if phi_types:
                phi_warning = True
                logger.warning(f"PHI detected in user message: {phi_types}")

    # ── SSE helpers ───────────────────────────────────────────────────────────
    _sse_headers = {
        "Cache-Control":    "no-cache",
        "X-Accel-Buffering": "no",   # disable nginx buffering when behind a proxy
    }

    if phi_warning:
        # Return the PHI warning as a single SSE event — no LLM call needed.
        def _phi_gen():
            event = json.dumps({
                "type":        "phi_warning",
                "message": (
                    "It looks like your message may contain personal information. "
                    "For your privacy, please remove any personal details (such as "
                    "names, phone numbers, addresses, or dates of birth) and try again."
                ),
                "phi_warning": True,
                "phi_types":   phi_types,
            })
            yield f"data: {event}\n\n"

        return Response(
            stream_with_context(_phi_gen()),
            mimetype="text/event-stream",
            headers=_sse_headers,
        )

    # ── Normal path: stream the RAG response token-by-token ──────────────────
    def _rag_gen():
        try:
            full_response = ""
            for token in ask_rag_question_stream(_rag_pipeline, messages):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            # Final event carries the complete assembled text so the frontend
            # can add it to chat history without re-concatenating.
            yield f"data: {json.dumps({'type': 'done', 'message': full_response, 'rag_used': True, 'phi_warning': False, 'phi_types': []})}\n\n"

        except Exception as exc:
            logger.error(f"/api/chat stream error: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return Response(
        stream_with_context(_rag_gen()),
        mimetype="text/event-stream",
        headers=_sse_headers,
    )
    
@app.route('/api/tts', methods=['POST', 'OPTIONS'])
def text_to_speech():
    """
    Streaming Text-to-Speech endpoint using OpenAI TTS API.

    Request body:
        { "text": "...", "language": "en"|"es", "voice": "nova", "speed": 0.8 }

    Returns an audio/mpeg stream so the browser can begin playback before the
    full audio has been generated (using MediaSource Extensions on the client).
    """
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text in request'}), 400

        text = data.get('text', '').strip()
        language = data.get('language', 'en')
        voice    = data.get('voice', 'nova')
        speed    = data.get('speed', 0.8)

        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        # OpenAI TTS maximum input length is 4 096 characters
        if len(text) > 4096:
            text = text[:4096]

        valid_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        if voice not in valid_voices:
            voice = 'nova'

        print(f"[TTS] Streaming: text='{text[:50]}...', lang={language}, voice={voice}, speed={speed}")

        def _audio_chunks():
            # with_streaming_response keeps the HTTP connection open and lets us
            # iterate the raw binary body in chunks as OpenAI produces them.
            with client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL,
                voice=voice,
                input=text,
                response_format='mp3',
                speed=speed,
            ) as tts_stream:
                for chunk in tts_stream.iter_bytes(chunk_size=4096):
                    yield chunk

        return Response(
            stream_with_context(_audio_chunks()),
            mimetype='audio/mpeg',
            headers={
                'Cache-Control':             'no-cache',
                'X-Accel-Buffering':         'no',
                'Access-Control-Allow-Origin': '*',
            },
        )

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
# SESSION MANAGEMENT ROUTES
# ============================================================================

@app.route('/api/session/start', methods=['POST', 'OPTIONS'])
def session_start():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    with sessions_lock:
        sessions[session_id] = {
            'events': [],
            'messages': [],
            'last_activity': now,
            'created_at': now
        }
    logger.info(f"Session started: {session_id}")
    return jsonify({'session_id': session_id}), 200


@app.route('/api/session/activity', methods=['POST', 'OPTIONS'])
def session_activity():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    data = request.get_json() or {}
    session_id = data.get('session_id')
    with sessions_lock:
        if session_id not in sessions:
            return jsonify({'error': 'session_expired'}), 404
        sessions[session_id]['last_activity'] = datetime.utcnow()
    return jsonify({'status': 'ok'}), 200


@app.route('/api/session/log', methods=['POST', 'OPTIONS'])
def session_log():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    data = request.get_json() or {}
    session_id = data.get('session_id')
    event = data.get('event')          # single event object (optional)
    messages = data.get('messages')    # full messages array (optional)
    with sessions_lock:
        if session_id not in sessions:
            return jsonify({'error': 'session_expired'}), 404
        sessions[session_id]['last_activity'] = datetime.utcnow()
        if event:
            sessions[session_id]['events'].append(event)
            # Re-sort after every insert so that fire-and-forget HTTP requests
            # that arrive out of order don't corrupt the event sequence.
            # Primary key: ISO-8601 timestamp (lexicographic == chronological).
            # Tiebreaker: client-assigned seq counter (monotonic within a session).
            sessions[session_id]['events'].sort(
                key=lambda e: (e.get('timestamp', ''), e.get('seq', 0))
            )
        if messages is not None:
            sessions[session_id]['messages'] = messages
    return jsonify({'status': 'ok'}), 200


@app.route('/api/session/end', methods=['POST', 'OPTIONS'])
def session_end():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    data = request.get_json() or {}
    session_id = data.get('session_id')
    # Allow caller to push a final messages snapshot
    final_messages = data.get('messages')
    with sessions_lock:
        session_data = sessions.pop(session_id, None)
    if not session_data:
        return jsonify({'error': 'session_not_found'}), 404
    if final_messages is not None:
        session_data['messages'] = final_messages

    # Use the final synced messages; fall back to event reconstruction so a
    # session is never incorrectly discarded when the messages array is empty
    # but the events tab has conversation content.
    effective_messages = session_data.get('messages', []) or \
                         _reconstruct_messages_from_events(session_data.get('events', []))

    # Skip persisting sessions that have no actual conversation content
    user_messages = [
        m for m in effective_messages
        if m.get('role') == 'user' and m.get('content', '').strip()
    ]
    if not user_messages:
        logger.info(f"Session {session_id} ended with no user messages — not saving.")
        return jsonify({'status': 'skipped', 'reason': 'no_messages'}), 200

    # Ensure session_data carries the effective messages so save_session_to_disk
    # and generate_session_summary both see the same complete conversation.
    session_data['messages'] = effective_messages
    summary = generate_session_summary(effective_messages)
    filename = save_session_to_disk(session_id, session_data, summary)
    logger.info(f"Session {session_id} ended and saved.")
    return jsonify({'status': 'saved', 'file': os.path.basename(filename), 'summary': summary}), 200


@app.route('/api/sessions/auth', methods=['POST', 'OPTIONS'])
def sessions_auth():
    """Validate the dashboard password and return a signed access token."""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if not SESSIONS_PASSWORD_HASH:
        return jsonify({'error': 'Sessions dashboard not configured — set SESSIONS_PASSWORD_HASH env var.'}), 503

    # ── IP-based rate limiting ────────────────────────────────────────────────
    ip = request.remote_addr or 'unknown'
    now = int(time.time())
    failures, lockout_until = _auth_rate_limit.get(ip, (0, 0))
    if lockout_until > now:
        remaining = lockout_until - now
        return jsonify({'error': f'Too many failed attempts. Try again in {remaining}s.'}), 429

    # ── Validate password ─────────────────────────────────────────────────────
    body = request.get_json(silent=True) or {}
    password = body.get('password', '')
    if not password:
        return jsonify({'error': 'Password required.'}), 400

    try:
        valid = bcrypt.checkpw(password.encode('utf-8'), SESSIONS_PASSWORD_HASH.encode('utf-8'))
    except Exception as exc:
        logger.error(f"bcrypt error: {exc}")
        return jsonify({'error': 'Server password configuration error.'}), 500

    if not valid:
        failures += 1
        lockout = (now + _AUTH_LOCKOUT_SECONDS) if failures >= _AUTH_MAX_FAILURES else 0
        _auth_rate_limit[ip] = (failures, lockout)
        remaining_attempts = max(0, _AUTH_MAX_FAILURES - failures)
        return jsonify({
            'error': 'Incorrect password.',
            'attempts_remaining': remaining_attempts
        }), 401

    # ── Success — clear rate limit, issue token ───────────────────────────────
    _auth_rate_limit.pop(ip, None)
    token = _make_dashboard_token()
    logger.info(f"Dashboard auth success from {ip}")
    return jsonify({'token': token, 'expires_in': SESSIONS_TOKEN_EXPIRY}), 200


@app.route('/api/sessions', methods=['GET', 'OPTIONS'])
@require_dashboard_token
def list_sessions():
    """Return metadata for all saved session files, newest first."""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        # Load all session JSON files — sort order is applied after reading
        # so we can sort by the actual created_at ISO timestamp stored inside
        # each file (sorting by filename would sort by UUID, not by time).
        files = [f for f in os.listdir(SESSIONS_DIR) if f.endswith('.json')]
        result = []
        for fname in files:
            fpath = os.path.join(SESSIONS_DIR, fname)
            with open(fpath, 'r') as fp:
                data = json.load(fp)
            # Backward-compat: fall back to event reconstruction for old files
            # that were saved with an empty messages array.
            messages = data.get('messages', []) or \
                       _reconstruct_messages_from_events(data.get('events', []))
            user_turns = sum(1 for m in messages if m.get('role') == 'user')
            created_at_pst = to_pst(data.get('created_at', ''))
            result.append({
                'filename': fname,
                'session_id': data.get('session_id', ''),
                'created_at': created_at_pst,
                'ended_at': to_pst(data.get('ended_at', '')),
                'message_count': len(messages),
                'user_turns': user_turns,
                'event_count': len(data.get('events', [])),
                'summary': data.get('summary', {})
            })
        # Sort newest-first by the PST created_at ISO string.
        # ISO-8601 strings with a consistent offset sort lexicographically
        # in chronological order, so this is safe.
        result.sort(
            key=lambda s: s['created_at'] or '',
            reverse=True
        )
        return jsonify({'sessions': result, 'total': len(result)}), 200
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions/<path:filename>', methods=['GET', 'OPTIONS'])
@require_dashboard_token
def get_session_detail(filename):
    """Return the full detail of a single saved session file."""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    safe_name = os.path.basename(filename)          # prevent path traversal
    if not safe_name.endswith('.json'):
        return jsonify({'error': 'Invalid filename'}), 400
    fpath = os.path.join(SESSIONS_DIR, safe_name)
    if not os.path.exists(fpath):
        return jsonify({'error': 'Session not found'}), 404
    try:
        with open(fpath, 'r') as fp:
            data = json.load(fp)

        # Backward-compat: session files saved before the message-sync fix
        # may have an empty (or missing) messages array even though the events
        # array contains full conversation content.  Reconstruct on the fly so
        # the conversation tab always matches the events tab for old files too.
        if not data.get('messages'):
            data['messages'] = _reconstruct_messages_from_events(
                data.get('events', [])
            )

        # Convert stored UTC timestamps to PST before returning
        data['created_at'] = to_pst(data.get('created_at'))
        data['ended_at']   = to_pst(data.get('ended_at'))
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error reading session {safe_name}: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# STARTUP
# ============================================================================

# Initialise scheduler at module level so it runs under both Flask dev server
# and gunicorn (single-worker) without needing if __name__ == '__main__'.
scheduler.init_app(app)
scheduler.add_job(id='daily_task_1am', func=daily_task, trigger='cron', hour=1, minute=0)
scheduler.add_job(id='session_cleanup', func=auto_expire_sessions,
                  trigger='interval', seconds=30)
scheduler.start()

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

    # When running with app.run(), set use_reloader=False to prevent jobs from running twice
    app.run(use_reloader=False) 