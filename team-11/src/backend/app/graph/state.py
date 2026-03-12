"""LangGraph shared state definition for the FinSynth workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state flowing through the Auditor → News Hound → Synthesizer graph.

    Fields updated by each node are merged automatically by LangGraph.
    ``thinking_log`` uses an *add* reducer so parallel nodes can both append.
    """

    # Input
    ticker: str

    # Populated by Auditor (Node A)
    financial_data: dict[str, Any] | None
    auditor_analysis: str | None

    # Populated by News Hound (Node B)
    news_data: dict[str, Any] | None
    news_analysis: str | None

    # Populated by Synthesizer (Node C)
    report: str | None

    # Accumulated thinking/progress entries (list-add reducer)
    thinking_log: Annotated[list[dict], operator.add]
