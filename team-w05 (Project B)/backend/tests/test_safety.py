"""
Unit tests for the safety/crisis detection service.
Run with:  pytest backend/tests/test_safety.py -v
"""

import pytest
from app.services.safety_service import check_safety


# ---------------------------------------------------------------------------
# Tier-1 crisis tests — must always escalate
# ---------------------------------------------------------------------------

CRISIS_MESSAGES = [
    "I want to kill myself",
    "I am going to end my life tonight",
    "I've been thinking about suicide",
    "I hurt myself earlier",
    "I'm planning to overdose",
    "I want to cut myself",
    "Please help me, I'm in crisis now",
]


@pytest.mark.parametrize("msg", CRISIS_MESSAGES)
def test_tier1_crisis_detected(msg):
    result = check_safety(msg)
    assert result.is_crisis is True
    assert result.tier == 1
    assert result.crisis_response is not None
    assert "988" in result.crisis_response or "741741" in result.crisis_response


# ---------------------------------------------------------------------------
# Tier-2 distress tests — not crisis, but elevated
# ---------------------------------------------------------------------------

DISTRESS_MESSAGES = [
    "I feel completely hopeless",
    "I've been very depressed lately",
    "I had a panic attack this morning",
]


@pytest.mark.parametrize("msg", DISTRESS_MESSAGES)
def test_tier2_distress_detected(msg):
    result = check_safety(msg)
    assert result.is_crisis is False
    assert result.tier == 2


# ---------------------------------------------------------------------------
# Safe messages — should not trigger escalation
# ---------------------------------------------------------------------------

SAFE_MESSAGES = [
    "I've been feeling a bit stressed at work",
    "Can you recommend meditation techniques?",
    "What is CBT?",
    "I'd like to improve my sleep habits",
    "Tell me about mindfulness",
]


@pytest.mark.parametrize("msg", SAFE_MESSAGES)
def test_safe_messages_pass(msg):
    result = check_safety(msg)
    assert result.is_crisis is False
    assert result.tier == 0
