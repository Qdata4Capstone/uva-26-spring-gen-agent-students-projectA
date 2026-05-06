"""LangGraph shared state definition for the FinSynth workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state flowing through the FinSynth agent graph.

    Graph topology:
        Auditor → News Hound → Synthesizer → Fact Checker ─┬─ (density OK)  → END
                                                            └─ (density low) → Re-Synth → Post Fact Checker → END

    Fields updated by each node are merged automatically by LangGraph.
    ``thinking_log`` uses an *add* reducer so all nodes can append entries.
    """

    # Input
    ticker: str
    workflow_mode: str  # "normal" | "brief" | "extra"

    # Populated by Auditor (Node A)
    financial_data: dict[str, Any] | None
    auditor_analysis: str | None

    # Populated by News Hound (Node B)
    news_data: dict[str, Any] | None
    news_analysis: str | None

    # Populated by Synthesizer (Node C) — intermediate before fact-checking
    draft_report: str | None

    # Populated by Fact Checker (Node D)
    fact_check_result: dict[str, Any] | None
    citation_density: float | None
    resynthesis_needed: bool

    # Populated by Post Fact Checker (Node F) — only on the resynthesis path
    post_fact_check_result: dict[str, Any] | None
    post_citation_density: float | None

    # Final report — set by Fact Checker (density OK) or Post Fact Checker (density low)
    report: str | None

    # Accumulated thinking/progress entries (list-add reducer)
    thinking_log: Annotated[list[dict], operator.add]
