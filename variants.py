# Chat-assistant variants (audiences) — single source of truth.
#
# Every conversation belongs to exactly one variant. The variant decides:
#   • which folder the saved transcript goes to   (sessions/<key>/)
#   • the tag embedded in its filename + content   (session_<key>_<uuid>_<ts>.json)
#   • the audience-specific lines added to the LLM system prompt
#   • the extra context given to the summary model
#
# The patient UI picks a variant with ?variant=<key> (see variants.js in the
# frontend repo, which must list the same keys). The backend never trusts the
# key blindly: it is looked up here, so an unknown value can never be turned
# into a filesystem path.
#
# To add a variant: add a Variant below and register it in VARIANTS, then add
# the matching entry to variants.js in the frontend.

import os
from dataclasses import dataclass

SESSIONS_ROOT = 'sessions'


@dataclass(frozen=True)
class Variant:
    key: str                    # URL/API/filename tag — lowercase letters only
    label: str                  # Human-readable name (TXT transcripts, logs)
    audience_instructions: str  # Extra system-prompt bullet lines ('' for none)
    summary_context: str        # Extra sentence for the summary model ('' for none)

    @property
    def sessions_dir(self) -> str:
        # Resolved at call time so tests can point SESSIONS_ROOT at a temp dir.
        return os.path.join(SESSIONS_ROOT, self.key)


GENERAL = Variant(
    key='general',
    label='General HPV',
    audience_instructions=(
        "* ASSUME THE USER'S AGE IS ABOVE 26 when generating a response. Avoid mentioning "
        "details specific to age groups below 26.\n"
    ),
    summary_context='',
)

POSTPARTUM = Variant(
    key='postpartum',
    label='Post-partum HPV',
    audience_instructions=(
        "* The user has recently given birth (a post-partum patient). Where it helps, relate "
        "your answer to life after having a baby — for example, the HPV vaccine is safe after "
        "delivery and while breastfeeding, does not slow healing after a vaginal birth or "
        "C-section, and is not recommended during pregnancy.\n"
        "* Be warm and reassuring — new parents are often tired or anxious. Use short, "
        "simple sentences.\n"
        "* Do not assume the user's age, gender, or relationship status.\n"
    ),
    summary_context=(
        "The patient recently gave birth (post-partum). Frame action items for a "
        "post-partum visit."
    ),
)

VARIANTS = {v.key: v for v in (GENERAL, POSTPARTUM)}
DEFAULT_VARIANT = GENERAL


def parse_variant(raw):
    """Map a client-supplied key to a Variant.

    Missing/empty → DEFAULT_VARIANT (older clients never send one).
    Unknown value → None, so the caller can reject the request with a 400.
    """
    if raw is None or raw == '':
        return DEFAULT_VARIANT
    return VARIANTS.get(str(raw).strip().lower())
