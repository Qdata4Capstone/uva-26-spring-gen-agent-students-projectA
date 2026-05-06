"""
Heuristic tests aligned with tool routing (no network).

Legacy name kept for CI/scripts that still invoke test_pubmed.py.
"""

import pytest

from app.services.tool_route_rules import build_router_search_query, message_mentions_condition


@pytest.mark.parametrize(
    "msg,mentions_condition",
    [
        ("What therapies help with anxiety?", True),
        ("I've heard CBT is effective — is there research on that?", False),
        ("Are there medications for depression?", True),
        ("Tell me about mindfulness research", False),
        ("I just feel sad today", False),
        ("How are you?", False),
        ("What's the weather like?", False),
    ],
)
def test_message_mentions_condition(msg, mentions_condition):
    assert message_mentions_condition(msg) == mentions_condition


def test_build_router_search_query_strips_whitespace():
    msg = "I have been feeling very anxious and I want to know about therapy"
    query = build_router_search_query(msg)
    assert "anxious" in query
    assert "therapy" in query
