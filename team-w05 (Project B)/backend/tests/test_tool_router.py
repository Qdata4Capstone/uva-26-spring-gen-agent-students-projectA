"""Keyword routing for tool_router (no network)."""

from app.services.tool_route_rules import build_router_search_query, message_mentions_condition


def test_condition_keywords_detected():
    assert message_mentions_condition("I've been struggling with depression lately")
    assert message_mentions_condition("generalized anxiety and sleep")
    assert message_mentions_condition("PTSD after an accident")
    assert message_mentions_condition("bipolar disorder management")


def test_general_messages_no_condition_flag():
    assert not message_mentions_condition("What is the weather today?")
    assert not message_mentions_condition("How does photosynthesis work?")


def test_build_search_query_trims_and_caps():
    long = "word " * 200
    q = build_router_search_query(long)
    assert len(q) <= 400
    assert q.startswith("word")


def test_build_search_query_empty_fallback():
    assert build_router_search_query("   ") == "mental health"
