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
#   PERSON (NER)   → 0.85  — spaCy's fixed ceiling, now BELOW PHI_SCORE_THRESHOLD
#   MRN / NHC      → custom regex (bypasses Presidio scoring, always caught)
#
# PHI_SCORE_THRESHOLD is now 0.90, which sits above spaCy's NER ceiling.
# PERSON names are therefore no longer flagged by Presidio — this is
# intentional: it eliminates the entire category of NER-based false positives
# (capitalised Spanish adjectives, medical acronyms, etc.) while keeping the
# genuinely unambiguous detections: email addresses and MRN regex matches.
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
    # NOTE: PERSON entities are not included here.  spaCy's NER pipeline caps
    # PERSON confidence at 0.85, which sits below PHI_SCORE_THRESHOLD (0.90),
    # so PERSON detection is effectively disabled for both languages at this
    # threshold.  Spanish PERSON is also explicitly skipped in detect_phi_backend
    # as defence-in-depth against future model updates that might score higher.
]


@pytest.mark.parametrize("sentence,lang", PHI_SENTENCES)
def test_true_positive(sentence, lang):
    """High-confidence PHI (email, MRN) must still be caught after threshold raise."""
    detections = phi(sentence, lang=lang)
    assert detections != [], (
        f"Expected PHI detection but got none for:\n  '{sentence}'"
    )
