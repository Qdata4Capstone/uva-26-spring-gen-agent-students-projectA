"""
Safety Service — crisis keyword detection and escalation logic.

This is a first-line filter that runs BEFORE the LLM.
If a user message matches crisis patterns the normal chat flow is bypassed
and pre-written crisis guidance is returned immediately.

Design philosophy
-----------------
- Err on the side of caution: prefer false positives over false negatives.
- Never diagnose; always direct to professional services.
- Keep crisis messaging calm, non-judgemental, and actionable.
"""

import re
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _coerce_user_profile(user_profile: Any) -> dict | None:
    """Accept dict or Pydantic model (e.g. UserProfile) from routers."""
    if user_profile is None:
        return None
    if isinstance(user_profile, dict):
        return user_profile
    if hasattr(user_profile, "model_dump"):
        return user_profile.model_dump()
    return None

# ---------------------------------------------------------------------------
# Crisis keyword patterns
# ---------------------------------------------------------------------------

# Tier-1: immediate danger — always escalate
_TIER1_PATTERNS = [
    r"\b(kill\s*(my)?self|suicide|suicidal)\b",
    r"\b(want\s+to\s+die|going\s+to\s+die|plan\s+to\s+die)\b",
    r"\b(end\s+my\s+life|ending\s+my\s+life)\b",
    r"\b(hurt\s*(my)?self|harm\s*(my)?self|self[- ]?harm)\b",
    r"\b(overdose|od\s+on)\b",
    r"\b(cut\s*(my)?self|cutting\s*(my)?self)\b",
    r"\bcris(is|es)\b.*\b(help|now|please)\b",
    r"\b(emergency|call\s+911|call\s+999)\b",
]

# Tier-2: significant distress — respond with extra care (still normal chat, but softer)
_TIER2_PATTERNS = [
    r"\b(hopeless|worthless|no\s+point|can('?t)?\s+go\s+on)\b",
    r"\b(depressed|depression|very\s+sad|extremely\s+sad)\b",
    r"\b(panic\s+attack|can('?t)?\s+breathe|heart\s+racing)\b",
    r"\b(abuse|abused|trauma|traumatized)\b",
]

_COMPILED_TIER1 = [re.compile(p, re.IGNORECASE) for p in _TIER1_PATTERNS]
_COMPILED_TIER2 = [re.compile(p, re.IGNORECASE) for p in _TIER2_PATTERNS]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@dataclass
class SafetyCheckResult:
    is_crisis: bool
    tier: int          # 0 = safe, 1 = immediate danger, 2 = elevated distress
    crisis_response: str | None


def check_safety(message: str, user_profile: Any | None = None) -> SafetyCheckResult:
    """
    Analyse a user message for crisis or distress signals.

    Returns a SafetyCheckResult.  If is_crisis is True the caller should
    return crisis_response immediately without calling the LLM.
    """
    profile = _coerce_user_profile(user_profile)
    for pattern in _COMPILED_TIER1:
        if pattern.search(message):
            logger.warning("Tier-1 crisis detected")
            return SafetyCheckResult(
                is_crisis=True,
                tier=1,
                crisis_response=_build_crisis_response(profile),
            )

    for pattern in _COMPILED_TIER2:
        if pattern.search(message):
            logger.info("Tier-2 distress signal detected — proceeding with extra care.")
            return SafetyCheckResult(is_crisis=False, tier=2, crisis_response=None)

    return SafetyCheckResult(is_crisis=False, tier=0, crisis_response=None)

def _build_crisis_response(user_profile: dict | None) -> str:
    name = user_profile.get("preferred_name") if user_profile else None
    is_uva = user_profile.get("is_uva_student") if user_profile else False
    greeting = f"{name}, " if name else ""

    base = f"""\
{greeting}I'm really glad you reached out, and I want you to know that what you're feeling matters deeply.
Right now, the most important thing is that you get immediate support from a trained crisis counsellor.
"""

    uva_block = ""
    if is_uva:
        uva_block = """
**If you are affiliated with UVA (University of Virginia), you can contact:**
- 🏫 **UVA Counseling & Psychological Services (CAPS)** — (434) 243-5150
- 🚨 **UVA Emergency / Campus Police** — (434) 924-7166
"""

    national_block = """
**Please reach out to one of these resources right away:**

- 📞 **988 Suicide & Crisis Lifeline** — call or text **988** (US)
- 💬 **Crisis Text Line** — text **HOME** to **741741**
- 🌍 **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/
- 🚨 **Emergency services** — call **911** or your local emergency number
"""

    closing = """\

    You don’t have to go through this alone. There are people who genuinely want to help and support you. ❤️

    *I'm an AI and not equipped to provide crisis care — please connect with one of the resources above.*
    """

    return base + uva_block + national_block + closing
