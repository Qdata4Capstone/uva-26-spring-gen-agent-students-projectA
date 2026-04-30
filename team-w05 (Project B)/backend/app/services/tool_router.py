"""
Tool routing — decides which external data sources to fetch for a user message.

Rules:
- Tier-1 crisis (via safety_service) → no tool calls; caller should return crisis copy.
- Message is medical/research-related → PubMed (+ ClinicalTrials if condition mentioned).
- Otherwise → no pre-fetch; Claude can still call tools during the loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.models.schemas import ClinicalTrialSummary, PubMedArticle
from app.services import clinical_trials
from app.services.mcp_client import mcp_client
from app.services.safety_service import check_safety
from app.services.tool_route_rules import build_router_search_query, message_mentions_condition, message_needs_research

logger = logging.getLogger(__name__)

# PubMed in the router uses MCP without Claude's retry layer — cap wait so the stream cannot hang forever.
_ROUTER_PUBMED_TIMEOUT_SEC = 90.0


@dataclass
class ToolRouteResult:
    """Outcome of routing + fetching for one user turn."""

    is_crisis: bool = False
    crisis_response: str | None = None
    distress_tier: int = 0
    pubmed_articles: list[PubMedArticle] = field(default_factory=list)
    clinical_trials: list[ClinicalTrialSummary] = field(default_factory=list)
    pubmed_query_used: str | None = None
    fetched_pubmed: bool = False
    fetched_clinical_trials: bool = False


async def route_user_message(message: str, user_profile: Any | None = None) -> ToolRouteResult:
    safety = check_safety(message, user_profile=user_profile)
    if safety.is_crisis:
        return ToolRouteResult(
            is_crisis=True,
            crisis_response=safety.crisis_response,
            distress_tier=safety.tier,
        )

    query = build_router_search_query(message)
    want_pubmed = message_needs_research(message)
    want_trials = message_mentions_condition(message)
    cap = settings.max_research_sources_per_turn

    result = ToolRouteResult(
        is_crisis=False,
        distress_tier=safety.tier,
        pubmed_query_used=query if want_pubmed else None,
    )

    if want_pubmed:
        try:
            mcp_result = await asyncio.wait_for(
                mcp_client.call_tool(
                    "pubmed_search",
                    {"query": query, "max_results": cap},
                ),
                timeout=_ROUTER_PUBMED_TIMEOUT_SEC,
            )
            raw = mcp_client.extract_text(mcp_result)
            for art in json.loads(raw):
                result.pubmed_articles.append(PubMedArticle(**art))
            result.fetched_pubmed = True
        except asyncio.TimeoutError:
            logger.warning(
                "Tool router PubMed fetch timed out after %.0fs (MCP hang or slow NCBI).",
                _ROUTER_PUBMED_TIMEOUT_SEC,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool router PubMed fetch failed: %s", exc)

    if want_trials:
        try:
            rows = await clinical_trials.search_clinical_trials(query, max_results=cap)
            trials = [ClinicalTrialSummary(**row) for row in rows]
            trials.sort(key=lambda t: (not t.recruiting, t.title.lower()))
            result.clinical_trials.extend(trials)
            result.fetched_clinical_trials = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool router clinical trials fetch failed: %s", exc)

    return result
