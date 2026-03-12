"""
LangGraph workflow builder and runner.

Graph topology:
    START --> auditor --> news_hound --> synthesizer --> END

The MCP server is launched as a stdio subprocess for the lifetime of each
analysis run.  The three agent nodes share one MCP ClientSession.
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

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..config import get_settings
from .nodes import make_auditor_node, make_news_hound_node, make_synthesizer_node
from .state import AgentState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────

def build_graph(mcp_session: ClientSession, llm: ChatGoogleGenerativeAI):
    """Construct and compile the LangGraph state machine."""
    logger.debug("Building LangGraph state machine (auditor → news_hound → synthesizer)")

    auditor = make_auditor_node(mcp_session, llm)
    news_hound = make_news_hound_node(mcp_session, llm)
    synthesizer = make_synthesizer_node(llm)

    builder = StateGraph(AgentState)
    builder.add_node("auditor", auditor)
    builder.add_node("news_hound", news_hound)
    builder.add_node("synthesizer", synthesizer)

    # Sequential: auditor → news_hound → synthesizer
    builder.add_edge(START, "auditor")
    builder.add_edge("auditor", "news_hound")
    builder.add_edge("news_hound", "synthesizer")
    builder.add_edge("synthesizer", END)

    graph = builder.compile()
    logger.debug("LangGraph state machine compiled successfully")
    return graph


# ─────────────────────────────────────────────────────────────────────
# Streaming runner
# ─────────────────────────────────────────────────────────────────────

async def run_analysis(ticker: str) -> AsyncGenerator[dict[str, Any], None]:
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

    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        logger.error("GEMINI_API_KEY is not configured — aborting analysis | ticker=%s", ticker_upper)
        yield {
            "event": "error",
            "data": {"message": "GEMINI_API_KEY is not configured. Please set it in backend/.env"},
        }
        yield {"event": "done", "data": {}}
        return

    # LLM instance
    logger.info("Initialising LLM | model=%s | temperature=0.3", settings.gemini_model)
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
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

    yield {
        "event": "thinking",
        "data": {
            "node": "system",
            "message": f"Starting analysis for **{ticker_upper}**…",
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
                    "financial_data": None,
                    "auditor_analysis": None,
                    "news_data": None,
                    "news_analysis": None,
                    "report": None,
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

                        # Emit report when ready
                        if state_update.get("report"):
                            report_content = state_update["report"]
                            report_len = len(report_content)
                            logger.info(
                                "Report ready | ticker=%s | report_chars=%d",
                                ticker_upper,
                                report_len,
                            )

                            # ── Persist report to disk ────────────────────
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
