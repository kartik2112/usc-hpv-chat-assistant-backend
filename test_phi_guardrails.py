"""
test_phi_guardrails.py
======================
Regression tests for the backend PHI detection guardrail.

Each test asserts that a sentence containing NO real PHI is NOT flagged —
i.e. there are zero false positives.  The sentences cover typical HPV-health-
education questions in both English and Spanish, including edge cases that have
previously caused false positives (capitalised adjectives, Spanish function
words, medical acronyms, etc.).

Run with:
    pytest test_phi_guardrails.py -v

Prerequisites:
    • PHI_ENABLED must be True (the default on a local / sackend deployment).
    • Both en_core_web_md and es_core_news_md spaCy models must be installed.
      Install them with:
          python -m spacy download en_core_web_md
          python -m spacy download es_core_news_md
"""

import importlib
import os
import sys
import pytest

# ---------------------------------------------------------------------------
# Ensure PHI_ENABLED=True for the test run so the NLP stack is exercised.
# On a Render deployment (IS_RENDER=True) PHI is disabled; tests are skipped.
# ---------------------------------------------------------------------------
os.environ.pop("RENDER", None)   # force PHI_ENABLED=True

# Import the module after patching the environment so the top-level
# PHI_ENABLED check inside flask_backend sees the correct value.
import flask_backend as fb

# ---------------------------------------------------------------------------
# Skip everything when the NLP stack is genuinely unavailable
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not fb.PHI_ENABLED,
    reason="PHI detection disabled (Render environment or missing models).",
)


def phi(text: str, lang: str = "en") -> list:
    """Thin wrapper around detect_phi_backend for readability."""
    return fb.detect_phi_backend(text, language=lang)


# ===========================================================================
# ENGLISH — 50 sentences that must produce zero detections
# ===========================================================================
ENGLISH_CLEAN_SENTENCES = [
    # General HPV knowledge
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

    # Vaccination
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

    # Screening
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

    # Biopsy and treatment
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

    # Prevention and lifestyle
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


@pytest.mark.parametrize("sentence", ENGLISH_CLEAN_SENTENCES)
def test_english_no_false_positive(sentence):
    """No clean English sentence should trigger the PHI guardrail."""
    detections = phi(sentence, lang="en")
    assert detections == [], (
        f"False positive for English sentence:\n  '{sentence}'\n"
        f"  Detected: {detections}"
    )


# ===========================================================================
# SPANISH — 50 sentences that must produce zero detections
# ===========================================================================
SPANISH_CLEAN_SENTENCES = [
    # General HPV knowledge — previously failing sentences first
    "Porque es Importante una biopsia",
    "Que tan eficaz es la vacuna",
    "¿Por qué es importante realizarse una biopsia?",
    "El VPH es la infección de transmisión sexual más común en los Estados Unidos.",
    "La mayoría de las personas sexualmente activas contraerán el VPH en algún momento.",
    "Existen más de doscientos tipos de VPH, y alrededor de cuarenta afectan el área genital.",
    "Los tipos de VPH de alto riesgo pueden causar cáncer de cuello uterino, anal y de garganta.",
    "Los tipos de VPH de bajo riesgo como el seis y el once causan la mayoría de las verrugas genitales.",
    "Muchas infecciones por VPH desaparecen solas sin causar problemas de salud.",
    "El VPH se puede transmitir durante las relaciones sexuales vaginales, anales u orales.",
    "Puede tener VPH sin saberlo porque a menudo no causa síntomas.",

    # Vaccination
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

    # Screening
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

    # Biopsy and treatment
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

    # Prevention and lifestyle
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


@pytest.mark.parametrize("sentence", SPANISH_CLEAN_SENTENCES)
def test_spanish_no_false_positive(sentence):
    """No clean Spanish sentence should trigger the PHI guardrail."""
    detections = phi(sentence, lang="es")
    assert detections == [], (
        f"False positive for Spanish sentence:\n  '{sentence}'\n"
        f"  Detected: {detections}"
    )


# ===========================================================================
# SANITY — sentences that SHOULD be flagged (true positives).
# These confirm the guardrail still catches high-confidence PHI after the fix.
#
# Score reference (en_core_web_md / es_core_news_md):
#   EMAIL_ADDRESS  → 1.00  — pure regex, always caught at any reasonable threshold
#   PERSON (NER)   → 0.85  — spaCy's fixed ceiling, below PHI_SCORE_THRESHOLD (0.90);
#                            caught via dedicated low-threshold passes in detect_phi_backend
#   PHONE_NUMBER   → 0.75 base; context words boost to ≥ 0.90 via ContextAwareEnhancer
#   US_SSN         → 0.85 base; "SSN" context word boosts to 1.00
#   CREDIT_CARD    → 0.85 base (Luhn-valid); "credit card" context boosts to 1.00
#   MRN / NHC      → custom regex (bypasses Presidio scoring, always caught)
# ===========================================================================
PHI_SENTENCES = [
    # EMAIL_ADDRESS — pattern-based (score 1.00), caught in both languages
    ("Please email me at jsmith@example.com with the results.", "en"),
    ("Send the report to maria.garcia@hospital.org as soon as possible.", "en"),
    ("Contact the clinic at info@healthcenter.org for an appointment.", "en"),
    ("Por favor envíame un correo a paciente@correo.com con los resultados.", "es"),
    ("Escríbeme a consulta@clinica.mx para más información.", "es"),
    # MEDICAL_RECORD_NUMBER — custom regex, threshold-independent
    ("My MRN: 00012345", "en"),
    ("Reference number MRN: 9876543", "en"),
    ("Patient reference NHC: 00056789", "es"),
    ("Mi número de historia clínica NHC: 12345678", "es"),
    # PERSON (English) — dedicated low-threshold pass (0.85), two-token minimum
    ("My doctor Jane Smith will follow up with me next week.", "en"),
    ("Please schedule an appointment for Robert Johnson.", "en"),
    ("The referral is for patient Emily Davis.", "en"),
    # PHONE_NUMBER — context words boost score above 0.90
    ("My phone number is (213) 555-0147 if you need to reach me.", "en"),
    ("You can call me at 310-555-0182 to discuss the results.", "en"),
    # US_SSN — "SSN" context word boosts score to 1.00
    ("My SSN is 123-45-6789, please update my records.", "en"),
    # CREDIT_CARD — Luhn-valid test number (Visa 4111…) with "credit card" context
    ("My credit card number is 4111111111111111 for the copay.", "en"),
    # PERSON (Spanish) — low-threshold pass (0.85) + stopword filter
    ("El paciente Juan García asistirá a su cita el próximo lunes.", "es"),
    ("Por favor contacte a María López para más información sobre su diagnóstico.", "es"),
    ("La paciente Ana Martínez necesita renovar su receta médica esta semana.", "es"),
]


@pytest.mark.parametrize("sentence,lang", PHI_SENTENCES)
def test_true_positive(sentence, lang):
    """High-confidence PHI (email, MRN) must still be caught after threshold raise."""
    detections = phi(sentence, lang=lang)
    assert detections != [], (
        f"Expected PHI detection but got none for:\n  '{sentence}'"
    )


# ===========================================================================
# SESSION SUMMARY / SAVE — PHI filtering uses the session's language
# ===========================================================================

SPANISH_SESSION = [
    ("user", "¿Qué es el VPH?"),
    ("assistant", "El VPH (Virus del Papiloma Humano) es un grupo de virus relacionados, algunos de los cuales pueden causar verrugas o provocar ciertos cánceres."),
    ("user", "¿Cuáles son los síntomas de la infección por VPH?"),
    ("assistant", "La infección por VPH a menudo no tiene signos o síntomas. Las personas con infección por VPH generalmente no saben que la tienen. Algunos tipos de VPH pueden causar verrugas genitales. Cuando las infecciones por VPH persisten, pueden convertirse en cáncer."),
    ("user", "¿Qué tan común es el VPH?"),
    ("assistant", "La infección por VPH es la infección de transmisión sexual más común en los Estados Unidos. La mayoría de las personas que tienen relaciones sexuales contraerán una infección por VPH en algún momento de sus vidas."),
    ("user", "¿Cómo se transmite de una persona a otra?"),
    ("assistant", "Hay alrededor de 40 tipos de VPH que típicamente infectan los genitales. Se transmite a través del contacto piel con piel durante las relaciones sexuales vaginales, anales u orales. Puede contraer una infección por VPH incluso sin tener relaciones sexuales."),
    ("user", "¿Se pueden tratar las infecciones por VPH?"),
    ("assistant", "Las infecciones por VPH generalmente son lentas. En la mayoría de las personas, el sistema inmunológico elimina el VPH del cuerpo antes de que cause enfermedad. Las verrugas se pueden extirpar con medicamentos o cirugía."),
]

ENGLISH_SESSION = [
    ("user", "What is HPV?"),
    ("assistant", "HPV (Human Papillomavirus) is a group of related viruses, some of which can cause warts or lead to certain cancers."),
    ("user", "What are symptoms of HPV infection?"),
    ("assistant", "HPV infection often has no signs or symptoms. People with HPV infection usually do not know they have it. This is one reason why HPV spreads easily. Some types of HPV can cause genital warts, which are growths on the outside or inside of the vagina or penis. When HPV infections last longer, they can turn into cancer."),
    ("user", "How is it passed from one person to another?"),
    ("assistant", "There are about 40 types of HPV that typically infect the genitals. It is spread by skin-to-skin contact during vaginal, anal, or oral sex. You can get a HPV infection even if you do not have sexual intercourse."),
]


def _messages(turns):
    return [{"role": role, "content": content} for role, content in turns]


def _session_data(turns, lang):
    from datetime import datetime, timezone
    return {
        "variant": "general",
        "created_at": datetime.now(timezone.utc),
        "events": [
            {"seq": i, "type": "User Text Query" if role == "user" else "Assistant Text Response",
             "timestamp": datetime.now(timezone.utc).isoformat(), "language": lang, "details": content}
            for i, (role, content) in enumerate(turns, start=1)
        ],
        "messages": _messages(turns),
    }


@pytest.fixture
def captured_llm_input(monkeypatch):
    """Stub the OpenAI call and record the conversation text sent to it."""
    from types import SimpleNamespace
    sent = []

    def fake_create(**kwargs):
        sent.append(kwargs["messages"][-1]["content"])
        content = '{"patient_questions": "• q", "action_items": "• a"}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(fb.client.chat.completions, "create", fake_create)
    return sent


@pytest.fixture
def saved_payload(monkeypatch, tmp_path):
    """Redirect save_session_to_disk into tmp_path and return a loader for the JSON."""
    import json
    monkeypatch.setattr(fb, "_session_file_stem", lambda variant, sid: str(tmp_path / sid))
    return lambda sid: json.load(open(tmp_path / f"{sid}.json"))


def test_language_of_reads_session_events():
    assert fb._language_of(_session_data(SPANISH_SESSION, "es")) == "es"
    assert fb._language_of(_session_data(ENGLISH_SESSION, "en")) == "en"
    assert fb._language_of({"events": []}) == "en"


def test_merged_summary_uses_session_language(monkeypatch):
    seen = []
    monkeypatch.setattr(fb, "generate_session_summary",
                        lambda messages, variant, language="en": seen.append(language) or {})
    sessions = [("a", _session_data(SPANISH_SESSION, "es")),
                ("b", _session_data(SPANISH_SESSION, "es"))]
    fb._build_merged_session(sessions, fb.DEFAULT_VARIANT)
    assert seen == ["es"]


@pytest.mark.parametrize("turns,lang", [(SPANISH_SESSION, "es"), (ENGLISH_SESSION, "en")])
def test_summary_receives_every_turn(captured_llm_input, turns, lang):
    summary = fb.generate_session_summary(_messages(turns), language=lang)
    assert summary["action_items"] == "• a"
    for _role, content in turns:
        assert content in captured_llm_input[0]


@pytest.mark.parametrize("turns,lang", [(SPANISH_SESSION, "es"), (ENGLISH_SESSION, "en")])
def test_saved_transcript_keeps_every_turn(saved_payload, turns, lang):
    fb.save_session_to_disk(f"test-{lang}", _session_data(turns, lang), summary={})
    assert saved_payload(f"test-{lang}")["messages"] == _messages(turns)


def test_summary_still_drops_user_phi(captured_llm_input):
    phi_question = "Mi nombre es Juan García"
    turns = SPANISH_SESSION[:2] + [("user", phi_question)]
    fb.generate_session_summary(_messages(turns), language="es")
    assert phi_question not in captured_llm_input[0]
    assert SPANISH_SESSION[0][1] in captured_llm_input[0]


def test_saved_transcript_still_drops_user_phi(saved_payload):
    phi_question = "Mi nombre es Juan García"
    turns = SPANISH_SESSION[:2] + [("user", phi_question)]
    fb.save_session_to_disk("test-phi", _session_data(turns, "es"), summary={})
    assert saved_payload("test-phi")["messages"] == _messages(SPANISH_SESSION[:2])
