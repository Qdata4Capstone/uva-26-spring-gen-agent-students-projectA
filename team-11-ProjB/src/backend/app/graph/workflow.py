"""
LangGraph workflow builder and runner.

Graph topology:
    START → auditor → news_hound → synthesizer → fact_checker ─┬─ (density OK)  → END
                                                                └─ (density low) → resynth → post_fact_checker → END

The MCP server is launched as a stdio subprocess for the lifetime of each
analysis run.  All agent nodes share one MCP ClientSession.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..config import get_settings
from .nodes import (
    make_auditor_node,
    make_fact_checker_node,
    make_news_hound_node,
    make_post_fact_checker_node,
    make_resynth_node,
    make_synthesizer_node,
)
from .state import AgentState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────

def _route_after_fact_check(state: AgentState) -> str:
    """Conditional edge: route to resynth when citation density is too low."""
    if state.get("resynthesis_needed"):
        return "resynth"
    return END


def build_graph(mcp_session: ClientSession, llm: ChatOllama):
    """Construct and compile the LangGraph state machine."""
    logger.debug(
        "Building LangGraph state machine "
        "(auditor → news_hound → synthesizer → fact_checker → [resynth])"
    )

    auditor = make_auditor_node(mcp_session, llm)
    news_hound = make_news_hound_node(mcp_session, llm)
    synthesizer = make_synthesizer_node(llm)
    fact_checker = make_fact_checker_node()
    resynth = make_resynth_node(llm)
    post_fact_checker = make_post_fact_checker_node()

    builder = StateGraph(AgentState)
    builder.add_node("auditor", auditor)
    builder.add_node("news_hound", news_hound)
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("fact_checker", fact_checker)
    builder.add_node("resynth", resynth)
    builder.add_node("post_fact_checker", post_fact_checker)

    builder.add_edge(START, "auditor")
    builder.add_edge("auditor", "news_hound")
    builder.add_edge("news_hound", "synthesizer")
    builder.add_edge("synthesizer", "fact_checker")
    builder.add_conditional_edges("fact_checker", _route_after_fact_check)
    builder.add_edge("resynth", "post_fact_checker")
    builder.add_edge("post_fact_checker", END)

    graph = builder.compile()
    logger.debug("LangGraph state machine compiled successfully")
    return graph


# ─────────────────────────────────────────────────────────────────────
# Streaming runner
# ─────────────────────────────────────────────────────────────────────

async def run_analysis(ticker: str, workflow_mode: str = "normal") -> AsyncGenerator[dict[str, Any], None]:
    """
    Execute the full analysis pipeline for *ticker* and yield SSE-ready
    event dicts as each node completes.

    Yields dicts with shape::

        {"event": "thinking", "data": {"node": ..., "message": ..., "status": ...}}
        {"event": "report",   "data": {"content": "..."}}
        {"event": "error",    "data": {"message": "..."}}
        {"event": "done",     "data": {}}
    """
    ticker_upper = ticker.upper()
    logger.info("run_analysis started | ticker=%s", ticker_upper)
    t_pipeline_start = time.monotonic()

    settings = get_settings()

    # LLM instance
    logger.info(
        "Initialising LLM via Ollama | model=%s | base_url=%s | temperature=0.3",
        settings.ollama_model,
        settings.ollama_base_url,
    )
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
        client_kwargs={"timeout": settings.ollama_timeout_sec},
    )

    # ── Locate the MCP server script ─────────────────────────────
    mcp_server_path = str(Path(__file__).resolve().parent.parent / "mcp_server.py")
    logger.debug("MCP server script path: %s", mcp_server_path)

    # Forward API keys to the subprocess
    child_env = {**os.environ}
    child_env["BRAVE_SEARCH_API_KEY"] = settings.brave_search_api_key or ""
    child_env["FINANCIAL_API_KEY"] = settings.financial_api_key or ""

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[mcp_server_path],
        env=child_env,
    )

    logger.info(
        "Launching MCP subprocess | command=%s %s",
        sys.executable,
        mcp_server_path,
    )

    mode_labels = {
        "normal":      "Normal — full pipeline, fact-check & disclaimer, no re-synthesis",
        "brief":       "Brief — raw data direct to synthesizer, fact-check, no re-synthesis",
        "extra":       "Extra Revision — full pipeline, re-synthesis if density < 50%, post fact-check",
        "extra_force": "Extra Force — full pipeline, re-synthesis always, post fact-check",
    }
    yield {
        "event": "thinking",
        "data": {
            "node": "system",
            "message": f"Starting analysis for **{ticker_upper}** · {mode_labels.get(workflow_mode, workflow_mode)}",
            "status": "started",
        },
    }

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            logger.debug("stdio_client context established | ticker=%s", ticker_upper)
            async with ClientSession(read_stream, write_stream) as session:
                t_mcp = time.monotonic()
                await session.initialize()
                logger.info(
                    "MCP session initialised | ticker=%s | elapsed=%.3fs",
                    ticker_upper,
                    time.monotonic() - t_mcp,
                )

                # Verify tools are available
                tools_list = await session.list_tools()
                tool_names = [t.name for t in tools_list.tools]
                logger.info(
                    "MCP tools available | ticker=%s | tools=%s",
                    ticker_upper,
                    tool_names,
                )
                yield {
                    "event": "thinking",
                    "data": {
                        "node": "system",
                        "message": f"MCP server connected — tools available: {', '.join(tool_names)}",
                        "status": "progress",
                    },
                }

                # Build & run graph
                graph = build_graph(session, llm)

                initial_state: AgentState = {
                    "ticker": ticker_upper,
                    "workflow_mode": workflow_mode,
                    "financial_data": None,
                    "auditor_analysis": None,
                    "news_data": None,
                    "news_analysis": None,
                    "draft_report": None,
                    "report": None,
                    "fact_check_result": None,
                    "citation_density": None,
                    "resynthesis_needed": False,
                    "post_fact_check_result": None,
                    "post_citation_density": None,
                    "thinking_log": [],
                }

                logger.info("Starting graph execution | ticker=%s", ticker_upper)
                t_graph = time.monotonic()
                nodes_completed: list[str] = []

                # Stream node-by-node updates
                async for chunk in graph.astream(initial_state, stream_mode="updates"):
                    for node_name, state_update in chunk.items():
                        elapsed_node = time.monotonic() - t_graph
                        nodes_completed.append(node_name)
                        logger.info(
                            "Node completed | ticker=%s | node=%s | elapsed_since_graph_start=%.2fs",
                            ticker_upper,
                            node_name,
                            elapsed_node,
                        )

                        # Emit new thinking log entries
                        for log_entry in state_update.get("thinking_log", []):
                            logger.debug(
                                "Thinking log entry | ticker=%s | node=%s | status=%s | message=%s",
                                ticker_upper,
                                log_entry.get("node"),
                                log_entry.get("status"),
                                log_entry.get("message"),
                            )
                            yield {"event": "thinking", "data": log_entry}

                        # ── Save draft report to disk when synthesizer completes ──
                        if node_name == "synthesizer" and state_update.get("draft_report"):
                            draft_content = state_update["draft_report"]
                            try:
                                draft_dir = Path(__file__).resolve().parent.parent.parent / "draft_reports"
                                draft_dir.mkdir(exist_ok=True)
                                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                draft_path = draft_dir / f"{ticker_upper}_{timestamp}_draft.md"
                                draft_path.write_text(draft_content, encoding="utf-8")
                                logger.info(
                                    "Draft report saved | ticker=%s | path=%s",
                                    ticker_upper,
                                    draft_path,
                                )
                            except Exception as write_exc:
                                logger.warning(
                                    "Failed to save draft report | ticker=%s | error=%s",
                                    ticker_upper,
                                    str(write_exc),
                                )

                        # ── Emit & persist final report ───────────────────────────
                        # fact_checker  → final report on the density-OK path
                        # post_fact_checker → final report on the resynthesis path
                        # resynth       → intermediate; the post_fact_checker will
                        #                 overwrite it, so we emit but don't save yet
                        if state_update.get("report"):
                            report_content = state_update["report"]
                            logger.info(
                                "Report ready | ticker=%s | node=%s | report_chars=%d",
                                ticker_upper,
                                node_name,
                                len(report_content),
                            )

                            # Only persist the definitive final report
                            if node_name in ("fact_checker", "post_fact_checker"):
                                try:
                                    reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
                                    reports_dir.mkdir(exist_ok=True)
                                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                    report_path = reports_dir / f"{ticker_upper}_{timestamp}.md"
                                    report_path.write_text(report_content, encoding="utf-8")
                                    logger.info(
                                        "Report saved to disk | ticker=%s | path=%s",
                                        ticker_upper,
                                        report_path,
                                    )
                                except Exception as write_exc:
                                    logger.warning(
                                        "Failed to save report to disk | ticker=%s | error=%s",
                                        ticker_upper,
                                        str(write_exc),
                                    )

                            yield {"event": "report", "data": {"content": report_content}}

                logger.info(
                    "Graph execution complete | ticker=%s | nodes=%s | total_graph_elapsed=%.2fs",
                    ticker_upper,
                    nodes_completed,
                    time.monotonic() - t_graph,
                )

    except Exception as exc:
        logger.exception(
            "Analysis pipeline failed | ticker=%s | error=%s",
            ticker_upper,
            str(exc),
        )
        yield {"event": "error", "data": {"message": f"Analysis failed: {str(exc)}"}}

    total_elapsed = time.monotonic() - t_pipeline_start
    logger.info(
        "run_analysis finished | ticker=%s | total_elapsed=%.2fs",
        ticker_upper,
        total_elapsed,
    )
    yield {"event": "done", "data": {}}
