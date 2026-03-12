"""
LangGraph node functions for the three-agent FinSynth workflow.

Node A -- The Auditor:     Fetches financials via MCP, analyses growth & margins with LLM.
Node B -- The News Hound:  Fetches news via MCP, performs sentiment analysis with LLM.
Node C -- The Synthesizer: Merges analyses from A & B into a structured Markdown report.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from mcp import ClientSession

from .state import AgentState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _log(node: str, message: str, status: str = "progress") -> dict:
    return {"node": node, "message": message, "status": status}


def _extract_text(result) -> str:
    """Pull plain text from an MCP CallToolResult."""
    parts: list[str] = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Factory functions (close over the MCP session & LLM)
# ─────────────────────────────────────────────────────────────────────

def make_auditor_node(mcp_session: ClientSession, llm: BaseChatModel):
    """Return the Auditor node function with MCP session + LLM baked in."""

    async def auditor(state: AgentState) -> dict[str, Any]:
        ticker = state["ticker"]
        logs: list[dict] = []
        logger.info("[auditor] Node started | ticker=%s", ticker)
        t_node = time.monotonic()

        # Step 1 -- Fetch financial data via MCP tool
        logs.append(_log("auditor", f"Fetching financial statements for **{ticker}** via MCP…", "started"))
        logger.debug("[auditor] Calling MCP tool 'get_financials' | ticker=%s", ticker)
        t_mcp = time.monotonic()

        mcp_result = await mcp_session.call_tool("get_financials", arguments={"ticker": ticker})
        raw_text = _extract_text(mcp_result)

        logger.debug(
            "[auditor] 'get_financials' returned | ticker=%s | response_chars=%d | elapsed=%.2fs",
            ticker,
            len(raw_text),
            time.monotonic() - t_mcp,
        )

        try:
            financial_data = json.loads(raw_text)
            logger.debug("[auditor] Parsed financial JSON successfully | ticker=%s", ticker)
        except json.JSONDecodeError as e:
            logger.warning(
                "[auditor] JSON parse failed — storing raw text | ticker=%s | error=%s",
                ticker,
                str(e),
            )
            financial_data = {"raw": raw_text}

        if "error" in financial_data:
            logger.error(
                "[auditor] MCP tool returned an error | ticker=%s | error=%s",
                ticker,
                financial_data["error"],
            )
            logs.append(_log("auditor", f"Warning — error fetching data: {financial_data['error']}", "error"))
            return {
                "financial_data": financial_data,
                "auditor_analysis": f"Error: {financial_data['error']}",
                "thinking_log": logs,
            }

        company = financial_data.get("company_name", ticker)
        years_available = [m.get("year") for m in financial_data.get("metrics", [])]
        logger.info(
            "[auditor] Financial data received | ticker=%s | company=%s | years=%s",
            ticker,
            company,
            years_available,
        )
        logs.append(_log("auditor", f"Received financial data for **{company}**"))

        # Step 2 -- LLM analysis
        logs.append(_log("auditor", "Analyzing financial statements with Gemini…"))
        logger.debug("[auditor] Sending financial data to LLM for analysis | ticker=%s", ticker)
        t_llm = time.monotonic()

        prompt = f"""You are a senior financial auditor. Analyze the following financial data for {ticker} and provide a detailed assessment.

FINANCIAL DATA:
{raw_text}

Provide your analysis covering:
1. **Revenue Trends**: YoY revenue growth trajectory and sustainability
2. **Profitability Analysis**: Gross, operating, and net margin trends
3. **Balance Sheet Health**: Leverage (debt-to-equity), liquidity (current ratio), asset quality
4. **Key Observations**: Notable strengths, weaknesses, or red flags

Be specific with numbers. Use the actual data provided. Format with clear headers."""

        response = await llm.ainvoke(prompt)
        analysis = response.content

        logger.info(
            "[auditor] LLM analysis complete | ticker=%s | analysis_chars=%d | llm_elapsed=%.2fs",
            ticker,
            len(analysis),
            time.monotonic() - t_llm,
        )
        logs.append(_log("auditor", "Financial analysis complete", "completed"))

        logger.info(
            "[auditor] Node finished | ticker=%s | total_elapsed=%.2fs",
            ticker,
            time.monotonic() - t_node,
        )
        return {
            "financial_data": financial_data,
            "auditor_analysis": analysis,
            "thinking_log": logs,
        }

    return auditor


def make_news_hound_node(mcp_session: ClientSession, llm: BaseChatModel):
    """Return the News Hound node function with MCP session + LLM baked in."""

    async def news_hound(state: AgentState) -> dict[str, Any]:
        ticker = state["ticker"]
        logs: list[dict] = []
        logger.info("[news_hound] Node started | ticker=%s", ticker)
        t_node = time.monotonic()

        # Step 1 -- Fetch news via MCP tool
        logs.append(_log("news_hound", f"Searching for recent news on **{ticker}** via MCP…", "started"))

        query = f"{ticker} stock news analysis"
        logger.debug(
            "[news_hound] Calling MCP tool 'search_news' | ticker=%s | query='%s'",
            ticker,
            query,
        )
        t_mcp = time.monotonic()

        mcp_result = await mcp_session.call_tool("search_news", arguments={"query": query})
        raw_text = _extract_text(mcp_result)

        logger.debug(
            "[news_hound] 'search_news' returned | ticker=%s | response_chars=%d | elapsed=%.2fs",
            ticker,
            len(raw_text),
            time.monotonic() - t_mcp,
        )

        try:
            news_data = json.loads(raw_text)
            logger.debug("[news_hound] Parsed news JSON successfully | ticker=%s", ticker)
        except json.JSONDecodeError as e:
            logger.warning(
                "[news_hound] JSON parse failed — storing raw text | ticker=%s | error=%s",
                ticker,
                str(e),
            )
            news_data = {"raw": raw_text}

        article_count = len(news_data.get("articles", []))
        source = news_data.get("source", "unknown")
        logger.info(
            "[news_hound] News data received | ticker=%s | articles=%d | source=%s",
            ticker,
            article_count,
            source,
        )
        logs.append(_log("news_hound", f"Found **{article_count}** articles (source: {source})"))

        # Step 2 -- LLM sentiment analysis
        logs.append(_log("news_hound", "Running sentiment analysis with Gemini…"))
        logger.debug("[news_hound] Sending news articles to LLM for sentiment analysis | ticker=%s", ticker)
        t_llm = time.monotonic()

        prompt = f"""You are a financial news analyst specializing in market sentiment. Analyze the following news articles about {ticker}.

NEWS ARTICLES:
{raw_text}

Provide your analysis covering:
1. **Overall Sentiment**: Bullish / Bearish / Neutral — with confidence level
2. **Key Themes & Developments**: Major news events or announcements
3. **Market Impact Assessment**: How these developments may affect the stock price
4. **Risk Factors**: Any mentioned or implied risks

Be concise but thorough. Back up your sentiment assessment with specific article references."""

        response = await llm.ainvoke(prompt)
        analysis = response.content

        logger.info(
            "[news_hound] LLM sentiment analysis complete | ticker=%s | analysis_chars=%d | llm_elapsed=%.2fs",
            ticker,
            len(analysis),
            time.monotonic() - t_llm,
        )
        logs.append(_log("news_hound", "Sentiment analysis complete", "completed"))

        logger.info(
            "[news_hound] Node finished | ticker=%s | total_elapsed=%.2fs",
            ticker,
            time.monotonic() - t_node,
        )
        return {
            "news_data": news_data,
            "news_analysis": analysis,
            "thinking_log": logs,
        }

    return news_hound


def make_synthesizer_node(llm: BaseChatModel):
    """Return the Synthesizer node function with LLM baked in."""

    async def synthesizer(state: AgentState) -> dict[str, Any]:
        ticker = state["ticker"]
        logs: list[dict] = []
        logger.info("[synthesizer] Node started | ticker=%s", ticker)
        t_node = time.monotonic()

        logs.append(_log("synthesizer", "Synthesizing final investment report…", "started"))

        auditor_analysis = state.get("auditor_analysis") or "No financial analysis available."
        news_analysis = state.get("news_analysis") or "No news analysis available."
        financial_data = state.get("financial_data") or {}

        company_name = financial_data.get("company_name", ticker)
        current_price = financial_data.get("current_price", "N/A")
        market_cap = financial_data.get("market_cap", "N/A")
        sector = financial_data.get("sector", "N/A")
        industry = financial_data.get("industry", "N/A")

        logger.debug(
            "[synthesizer] Company context | ticker=%s | company=%s | price=%s | market_cap=%s | sector=%s",
            ticker,
            company_name,
            current_price,
            market_cap,
            sector,
        )
        logger.debug(
            "[synthesizer] Inputs ready | ticker=%s | auditor_analysis_chars=%d | news_analysis_chars=%d",
            ticker,
            len(auditor_analysis),
            len(news_analysis),
        )
        logger.debug("[synthesizer] Sending synthesis prompt to LLM | ticker=%s", ticker)
        t_llm = time.monotonic()

        prompt = f"""You are a senior investment analyst at a top-tier firm. Synthesize the following analyses into a comprehensive, professional investment report for **{company_name} ({ticker})**.

COMPANY OVERVIEW:
- Current Price: ${current_price}
- Market Cap: {market_cap}
- Sector: {sector}
- Industry: {industry}

FINANCIAL ANALYSIS (from The Auditor):
{auditor_analysis}

NEWS & SENTIMENT ANALYSIS (from The News Hound):
{news_analysis}

Write the report in **Markdown format** with the following structure:

# Investment Report: {company_name} ({ticker})

## Executive Summary
A concise 3-4 sentence overview with the key takeaway.

## Company Snapshot
A quick table of key stats (price, market cap, P/E, sector).

## Financial Health
Detailed analysis of revenue, profitability, balance sheet.

## News & Market Sentiment
Summary of recent developments and overall market mood.

## Risk Assessment
Key risks organized by category (financial, market, operational).

## Investment Thesis
- **Bull Case**: Why the stock could outperform
- **Bear Case**: Why the stock could underperform
- **Base Case**: Most likely scenario

## Recommendation
Clear recommendation with rationale (Buy / Hold / Sell with conviction level).

---
*Report generated by FinSynth AI Agent*

Make it professional, data-driven, and actionable. Use specific numbers from the analyses."""

        response = await llm.ainvoke(prompt)
        report = response.content

        logger.info(
            "[synthesizer] LLM report generation complete | ticker=%s | report_chars=%d | llm_elapsed=%.2fs",
            ticker,
            len(report),
            time.monotonic() - t_llm,
        )
        logs.append(_log("synthesizer", "Investment report generated", "completed"))

        logger.info(
            "[synthesizer] Node finished | ticker=%s | total_elapsed=%.2fs",
            ticker,
            time.monotonic() - t_node,
        )
        return {
            "report": report,
            "thinking_log": logs,
        }

    return synthesizer
