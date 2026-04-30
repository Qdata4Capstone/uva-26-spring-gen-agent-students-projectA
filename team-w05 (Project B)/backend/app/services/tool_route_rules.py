"""
Pure keyword / query helpers for tool routing (no MCP or HTTP imports).
"""

from __future__ import annotations

_RESEARCH_KEYWORDS: tuple[str, ...] = (
    "therapy", "therapies", "therapeutic",
    "treatment", "treatments", "treat",
    "medication", "medications", "medicine",
    "antidepressant", "ssri", "snri", "lithium",
    "cbt", "dbt", "emdr", "mindfulness",
    "clinical trial", "clinical study",
    "research", "study", "studies", "evidence",
    "how to help", "how to cope", "coping",
    "diagnosis", "diagnose", "symptom", "symptoms",
    "mental health",
)

_CONDITION_KEYWORDS: tuple[str, ...] = (
    "eating disorder",
    "seasonal affective",
    "panic disorder",
    "post traumatic",
    "post-traumatic",
    "postpartum depression",
    "bipolar disorder",
    "borderline personality",
    "mental health condition",
    "mental illness",
    "depression",
    "depressed",
    "anxiety",
    "anxious",
    "ptsd",
    "bipolar",
    "ocd",
    "schizophrenia",
    "adhd",
    "anorexia",
    "bulimia",
    "phobia",
    "borderline",
    "bpd",
    "dysthymia",
    "mania",
    "manic",
    "mdd",
    "gad",
    "psychosis",
    "psychotic",
    "schizoaffective",
)


def _normalize_message(message: str) -> str:
    return " ".join(message.lower().split())


def message_mentions_condition(message: str) -> bool:
    lower = _normalize_message(message)
    return any(kw in lower for kw in _CONDITION_KEYWORDS)


def message_needs_research(message: str) -> bool:
    lower = _normalize_message(message)
    return message_mentions_condition(message) or any(kw in lower for kw in _RESEARCH_KEYWORDS)


def build_router_search_query(message: str) -> str:
    """Compact query string for PubMed / ClinicalTrials."""
    text = " ".join(message.split())
    if len(text) > 400:
        text = text[:400]
    return text or "mental health"
