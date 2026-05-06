"""
LangGraph node functions for the FinSynth agent workflow.

Node A -- The Auditor:      Fetches financials via MCP, analyses growth & margins with LLM.
Node B -- The News Hound:   Fetches news via MCP, performs sentiment analysis with LLM.
Node C -- The Synthesizer:  Merges analyses from A & B into a structured Markdown draft.
Node D -- The Fact Checker: Cross-references every numerical claim in the draft against
                            the raw financial_data ground truth. Computes citation density
                            and appends a calibrated disclaimer. Triggers re-synthesis when
                            density is too low.
Node E -- The Re-Synthesizer: Rewrites the report using only verified source figures when
                               the Fact Checker finds citation density below the threshold.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from mcp import ClientSession

from .state import AgentState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Shared helpers
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
# Hallucination-auditor helpers (Fact Checker / Re-Synthesizer)
# ─────────────────────────────────────────────────────────────────────

# Thresholds
_RESYNTHESIS_THRESHOLD: float = 0.50   # trigger rewrite below 50 % density
_MATCH_TOLERANCE: float = 0.05         # ±5 % relative tolerance for number matching

# Captures: optional $, a number (with optional comma-thousands), optional scale
# word/letter, optional % unit.  Years (1900–2100) and bare small integers are
# filtered out in post-processing rather than in the regex.
_CLAIM_RE = re.compile(
    r"\$?\s*"
    r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+)"          # the number
    r"(?:\s*(billion|million|trillion|thousand|[BMTK])\b)?"  # optional scale
    r"(?:\s*(%))?",                                        # optional % unit
    re.IGNORECASE,
)


def _build_fact_lookup(financial_data: dict) -> set[float]:
    """
    Flatten all numeric values from the raw financial_data dict into a set of
    canonical floats.  Large monetary values are added at their raw scale AND
    at the /1e6 (millions) and /1e9 (billions) scales so that report text using
    any of these representations will still find a match.
    """
    values: set[float] = set()

    def _ingest(v: Any) -> None:
        if v is None:
            return
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        if f == 0 or not (-1e16 < f < 1e16):
            return
        values.add(f)
        af = abs(f)
        if af >= 1e6:
            values.add(round(f / 1e6, 4))
        if af >= 1e9:
            values.add(round(f / 1e9, 4))
        if af >= 1e12:
            values.add(round(f / 1e12, 4))

    for key in [
        "current_price", "pe_ratio", "forward_pe",
        "fifty_two_week_high", "fifty_two_week_low", "market_cap",
    ]:
        _ingest(financial_data.get(key))

    # Dividend yield is stored as a decimal (e.g. 0.0053); reports use % (0.53)
    dy = financial_data.get("dividend_yield")
    if dy is not None:
        try:
            dy_f = float(dy)
            _ingest(dy_f)
            _ingest(dy_f * 100)
        except (TypeError, ValueError):
            pass

    for year_m in financial_data.get("metrics", []):
        for k, v in year_m.items():
            if k != "year":
                _ingest(v)

    return values


def _extract_claims(text: str) -> list[tuple[str, float]]:
    """
    Scan *text* for numerical claims and return (raw_match_string, normalised_value)
    pairs.  Filters out years, bare list-marker integers, and near-zero values.
    Deduplicates identical normalised values so repeated figures aren't
    double-counted.
    """
    seen: set[float] = set()
    results: list[tuple[str, float]] = []

    for m in _CLAIM_RE.finditer(text):
        num_str = m.group(1)
        if not num_str:
            continue
        try:
            base = float(num_str.replace(",", ""))
        except ValueError:
            continue

        scale = (m.group(2) or "").lower()
        has_pct = bool(m.group(3))

        # Skip years
        if 1900 <= base <= 2100 and not scale and not has_pct and base == int(base):
            continue

        # Skip bare small integers used as list markers or ordinals
        if base < 10 and not scale and not has_pct and "." not in num_str:
            continue

        # Normalise to canonical float
        mult: float = 1.0
        if scale in ("b", "billion"):
            mult = 1e9
        elif scale in ("m", "million"):
            mult = 1e6
        elif scale in ("t", "trillion"):
            mult = 1e12
        elif scale in ("k", "thousand"):
            mult = 1e3

        normalised = base * mult
        if abs(normalised) < 1e-6:
            continue

        # Deduplicate by a rounded key
        key = round(normalised, 2)
        if key in seen:
            continue
        seen.add(key)

        results.append((m.group(0).strip(), normalised))

    return results


def _is_verified(value: float, lookup: set[float]) -> bool:
    """Return True if *value* is within _MATCH_TOLERANCE of any entry in *lookup*."""
    for known in lookup:
        if known == 0:
            continue
        if abs(value - known) / abs(known) <= _MATCH_TOLERANCE:
            return True
    return False


def _run_fact_check(text: str, financial_data: dict) -> dict[str, Any]:
    """
    Core fact-checking algorithm, extracted so both the initial Fact Checker
    and the Post Fact Checker can share identical logic.

    Builds a lookup from *financial_data*, extracts all numerical claims from
    *text*, cross-references each claim, and returns a result dict containing
    citation_density, tier, per-claim lists, and a flag indicating whether the
    lookup was non-empty (used to gate resynthesis).
    """
    lookup = _build_fact_lookup(financial_data)
    claims = _extract_claims(text)
    total = len(claims)

    if total == 0:
        return {
            "citation_density": 1.0,
            "tier": "green",
            "total_claims": 0,
            "verified_claims": [],
            "unverified_claims": [],
            "lookup_populated": bool(lookup),
        }

    verified: list[str] = []
    unverified: list[str] = []
    for raw_text, norm_val in claims:
        if _is_verified(norm_val, lookup):
            verified.append(raw_text)
        else:
            unverified.append(raw_text)

    density = len(verified) / total
    tier, _ = _build_disclaimer(density, total, len(unverified))

    return {
        "citation_density": round(density, 4),
        "tier": tier,
        "total_claims": total,
        "verified_claims": verified,
        "unverified_claims": unverified,
        "lookup_populated": bool(lookup),
    }


def _build_disclaimer(density: float, total: int, unverified_count: int) -> tuple[str, str]:
    """
    Return (tier_label, disclaimer_markdown) calibrated to citation density.

    Tiers:
        green  — ≥ 80 %  verified
        amber  — 50–79 % verified
        red    — < 50 %  verified (also triggers resynthesis, so this is fallback)
    """
    verified_count = total - unverified_count
    if density >= 0.80:
        tier = "green"
        text = (
            "\n\n---\n"
            f"> **Data Integrity:** High citation density — "
            f"{verified_count}/{total} numerical claims verified against source data."
        )
    elif density >= 0.50:
        tier = "amber"
        text = (
            "\n\n---\n"
            f"> **⚠ Data Verification Notice:** {unverified_count} of {total} numerical "
            "claim(s) could not be directly verified against source financial data. "
            "Figures may reflect rounding, analyst estimates, or calculations not directly "
            "available in the underlying dataset. Exercise appropriate caution."
        )
    else:
        tier = "red"
        text = (
            "\n\n---\n"
            f"> **⛔ High Uncertainty Warning:** {unverified_count} of {total} numerical "
            f"claim(s) ({100 * (1 - density):.0f}% of total) could not be verified against "
            "source data. A significant portion of figures may be estimates, inferred, or "
            "inaccurate. This report should be treated as illustrative only."
        )
    return tier, text


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

        # Brief mode: skip the LLM analysis step entirely.
        # The raw financial_data will be fed directly to the Synthesizer.
        workflow_mode = state.get("workflow_mode", "normal")
        if workflow_mode == "brief":
            logs.append(_log(
                "auditor",
                "Brief mode — skipping LLM analysis, passing raw data to Synthesizer",
                "completed",
            ))
            logger.info(
                "[auditor] Brief mode — LLM skipped | ticker=%s | total_elapsed=%.2fs",
                ticker, time.monotonic() - t_node,
            )
            return {
                "financial_data": financial_data,
                "auditor_analysis": None,
                "thinking_log": logs,
            }

        # Step 2 -- LLM analysis (normal / extra modes)
        logs.append(_log("auditor", "Analyzing financial statements with Ollama…"))
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
        logs.append(_log("news_hound", "Running sentiment analysis with Ollama…"))
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

        workflow_mode = state.get("workflow_mode", "normal")

        if workflow_mode == "brief":
            # Brief mode: embed raw financial JSON directly — no LLM has touched the numbers yet.
            # This minimises the hallucination chain by grounding the synthesis in source data.
            fd_json = json.dumps(financial_data, indent=2, default=str)
            prompt = f"""You are a senior investment analyst. Write a comprehensive investment report for **{company_name} ({ticker})** using ONLY the raw financial data and news analysis provided below.

IMPORTANT: Every numerical figure you include MUST appear verbatim in the RAW FINANCIAL DATA. Do not round, estimate, extrapolate, or invent any numbers beyond what is explicitly provided.

COMPANY OVERVIEW:
- Current Price: ${current_price}
- Market Cap: {market_cap}
- Sector: {sector}
- Industry: {industry}

RAW FINANCIAL DATA (cite only these exact figures):
{fd_json}

NEWS & SENTIMENT ANALYSIS:
{news_analysis}

Write the report in **Markdown format** with the following structure:"""
        else:
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

Write the report in **Markdown format** with the following structure:"""

        prompt += f"""

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

Make it professional, data-driven, and actionable. Use specific numbers from the {"raw financial data" if workflow_mode == "brief" else "analyses"}."""

        response = await llm.ainvoke(prompt)
        report = response.content

        logger.info(
            "[synthesizer] LLM report generation complete | ticker=%s | report_chars=%d | llm_elapsed=%.2fs",
            ticker,
            len(report),
            time.monotonic() - t_llm,
        )
        logs.append(_log("synthesizer", "Draft report generated — sending to Fact Checker…", "completed"))

        logger.info(
            "[synthesizer] Node finished | ticker=%s | total_elapsed=%.2fs",
            ticker,
            time.monotonic() - t_node,
        )
        # Store as draft_report; the Fact Checker will promote it to `report`
        # (or hand off to the Re-Synthesizer if citation density is too low).
        return {
            "draft_report": report,
            "thinking_log": logs,
        }

    return synthesizer


# ─────────────────────────────────────────────────────────────────────
# Node D — Fact Checker  (no LLM — pure numerical cross-reference)
# ─────────────────────────────────────────────────────────────────────

def make_fact_checker_node():
    """
    Return the Fact Checker node.

    Extracts every numerical claim from the draft report and cross-references
    each against the raw financial_data ground truth (within ±5 % tolerance).
    Computes a citation density score and:
      - density ≥ 0.50 → appends a calibrated disclaimer and finalises the report
      - density  < 0.50 → sets resynthesis_needed=True so the graph routes to
                          the Re-Synthesizer instead
    """

    async def fact_checker(state: AgentState) -> dict[str, Any]:
        ticker = state["ticker"]
        logs: list[dict] = []
        t_node = time.monotonic()
        logger.info("[fact_checker] Node started | ticker=%s", ticker)

        logs.append(_log("fact_checker", "Auditing report against source financial data…", "started"))

        draft = state.get("draft_report") or ""
        financial_data = state.get("financial_data") or {}

        if not draft:
            logger.warning("[fact_checker] No draft report to check | ticker=%s", ticker)
            logs.append(_log("fact_checker", "No draft report available — skipping audit", "error"))
            return {
                "report": draft,
                "fact_check_result": {"error": "no draft report"},
                "citation_density": 1.0,
                "resynthesis_needed": False,
                "thinking_log": logs,
            }

        # Run shared fact-checking algorithm
        result = _run_fact_check(draft, financial_data)
        density = result["citation_density"]
        total = result["total_claims"]
        verified = result["verified_claims"]
        unverified = result["unverified_claims"]
        tier = result["tier"]

        logger.info(
            "[fact_checker] Audit complete | ticker=%s | total=%d | verified=%d | "
            "unverified=%d | density=%.2f",
            ticker, total, len(verified), len(unverified), density,
        )

        if total == 0:
            logger.info("[fact_checker] No numerical claims — density=1.0 | ticker=%s", ticker)
            logs.append(_log("fact_checker", "No numerical claims found — passing report through", "completed"))
            _, disclaimer = _build_disclaimer(1.0, 0, 0)
            return {
                "report": draft + disclaimer,
                "fact_check_result": result,
                "citation_density": 1.0,
                "resynthesis_needed": False,
                "thinking_log": logs,
            }

        # Resynthesis gating varies by workflow mode:
        #   normal — never resynth; fact check result is appended as a disclaimer and done
        #   brief  — never resynth; same as normal (speed + lower hallucination risk)
        #   extra  — resynth only when citation density drops below the threshold
        workflow_mode = state.get("workflow_mode", "normal")
        if workflow_mode == "extra":
            resynthesis_needed = density < _RESYNTHESIS_THRESHOLD and result.get("lookup_populated", False)
        elif workflow_mode == "extra_force":
            # Always resynthesize regardless of density — used for controlled experiments
            # to measure resynth impact even when the draft is already high-quality.
            resynthesis_needed = bool(result.get("lookup_populated", False))
        else:
            resynthesis_needed = False

        status = "progress" if resynthesis_needed else "completed"
        logs.append(_log(
            "fact_checker",
            f"Citation density: **{density:.0%}** — {len(verified)}/{total} claims verified "
            f"(tier: {tier})" + (" — triggering re-synthesis" if resynthesis_needed else ""),
            status,
        ))

        if resynthesis_needed:
            logger.info(
                "[fact_checker] Node finished (re-synthesis queued) | ticker=%s | elapsed=%.2fs",
                ticker, time.monotonic() - t_node,
            )
            # Do NOT set `report` — the Post Fact Checker will write the final report.
            return {
                "fact_check_result": result,
                "citation_density": density,
                "resynthesis_needed": True,
                "thinking_log": logs,
            }

        # Density acceptable — append disclaimer and promote draft to final report
        _, disclaimer_text = _build_disclaimer(density, total, len(unverified))
        logger.info(
            "[fact_checker] Node finished (report approved) | ticker=%s | elapsed=%.2fs",
            ticker, time.monotonic() - t_node,
        )
        return {
            "report": draft + disclaimer_text,
            "fact_check_result": result,
            "citation_density": density,
            "resynthesis_needed": False,
            "thinking_log": logs,
        }

    return fact_checker


# ─────────────────────────────────────────────────────────────────────
# Node E — Re-Synthesizer  (LLM rewrite grounded in verified data)
# ─────────────────────────────────────────────────────────────────────

def make_resynth_node(llm: BaseChatModel):
    """
    Return the Re-Synthesizer node.

    Called only when the Fact Checker finds citation density below the threshold.
    Rewrites the investment report using ONLY figures that exist in the verified
    financial_data ground truth, explicitly instructing the LLM not to invent
    or extrapolate any numbers absent from the source data.
    """

    async def resynth(state: AgentState) -> dict[str, Any]:
        ticker = state["ticker"]
        logs: list[dict] = []
        t_node = time.monotonic()
        logger.info("[resynth] Node started | ticker=%s", ticker)

        fact_check = state.get("fact_check_result") or {}
        density = state.get("citation_density") or 0.0
        unverified = fact_check.get("unverified_claims", [])
        total = fact_check.get("total_claims", 0)

        logs.append(_log(
            "resynth",
            f"Citation density too low ({density:.0%}, {len(unverified)}/{total} unverified) "
            "— rewriting report with verified facts only…",
            "started",
        ))

        financial_data = state.get("financial_data") or {}
        auditor_analysis = state.get("auditor_analysis") or "Not available."
        news_analysis = state.get("news_analysis") or "Not available."
        company_name = financial_data.get("company_name", ticker)
        fd_json = json.dumps(financial_data, indent=2, default=str)

        logs.append(_log("resynth", "Sending verified-facts rewrite prompt to Ollama…"))
        t_llm = time.monotonic()

        prompt = f"""You are a senior investment analyst performing a STRICT FACTUAL REWRITE of a financial report.

The original report for **{company_name} ({ticker})** had a citation density of only {density:.0%} — meaning {len(unverified)} of {total} numerical claims could not be verified against source financial data.

YOUR TASK: Rewrite the full investment report using ONLY figures that are explicitly present in the VERIFIED FINANCIAL DATA below.

═══════════════════════════════════════════════════
VERIFIED FINANCIAL DATA (Ground Truth — cite only these numbers):
{fd_json}
═══════════════════════════════════════════════════

AUDITOR ANALYSIS (qualitative context — do not copy its unverified numbers):
{auditor_analysis}

NEWS & SENTIMENT ANALYSIS (qualitative context only):
{news_analysis}

FLAGGED UNVERIFIED CLAIMS (DO NOT USE these figures — they could not be verified):
{json.dumps(unverified, indent=2)}

STRICT RULES:
1. Every numerical figure you cite MUST appear in the VERIFIED FINANCIAL DATA above
2. If a metric is not in the source data, use qualitative language (e.g., "revenue grew year-over-year" rather than inventing a percentage)
3. Preserve the full Markdown structure of the original report (all sections)
4. Add "[Data-Verified Rewrite]" at the end of the Executive Summary
5. Be transparent about data limitations — if certain figures are unavailable, state that explicitly

Write the complete revised investment report now in Markdown."""

        response = await llm.ainvoke(prompt)
        new_report = response.content

        logger.info(
            "[resynth] Rewrite complete | ticker=%s | report_chars=%d | llm_elapsed=%.2fs",
            ticker, len(new_report), time.monotonic() - t_llm,
        )

        logs.append(_log("resynth", "Rewrite complete — handing off to Post Fact Checker…", "completed"))
        logger.info(
            "[resynth] Node finished | ticker=%s | total_elapsed=%.2fs",
            ticker, time.monotonic() - t_node,
        )
        # Return the raw rewrite without any notice appended; the Post Fact Checker
        # will run a second verification pass and append the full audit trail table.
        return {
            "report": new_report,
            "thinking_log": logs,
        }

    return resynth


# ─────────────────────────────────────────────────────────────────────
# Node F — Post Fact Checker  (no LLM — re-runs verification on resynth output)
# ─────────────────────────────────────────────────────────────────────

def make_post_fact_checker_node():
    """
    Return the Post Fact Checker node.

    Re-runs the identical fact-checking algorithm on the Re-Synthesizer's output
    and appends a before/after audit trail table to the final report so readers
    can see the tangible improvement in citation density.
    """

    async def post_fact_checker(state: AgentState) -> dict[str, Any]:
        ticker = state["ticker"]
        logs: list[dict] = []
        t_node = time.monotonic()
        logger.info("[post_fact_checker] Node started | ticker=%s", ticker)

        logs.append(_log("post_fact_checker", "Re-verifying rewritten report against source data…", "started"))

        resynth_report = state.get("report") or ""
        financial_data = state.get("financial_data") or {}
        original_density = state.get("citation_density") or 0.0
        original_fc = state.get("fact_check_result") or {}

        if not resynth_report:
            logger.warning("[post_fact_checker] No report to re-verify | ticker=%s", ticker)
            logs.append(_log("post_fact_checker", "No report to re-verify", "error"))
            return {"post_fact_check_result": {}, "post_citation_density": None, "thinking_log": logs}

        result = _run_fact_check(resynth_report, financial_data)
        post_density = result["citation_density"]
        delta = post_density - original_density
        improvement_str = f"+{delta:.0%}" if delta >= 0 else f"{delta:.0%}"

        orig_total = original_fc.get("total_claims", 0)
        orig_unverified_count = len(original_fc.get("unverified_claims", []))
        orig_verified_count = orig_total - orig_unverified_count
        orig_tier = original_fc.get("tier", "red")

        post_total = result["total_claims"]
        post_unverified_count = len(result["unverified_claims"])
        post_verified_count = len(result["verified_claims"])
        post_tier = result["tier"]

        logger.info(
            "[post_fact_checker] Re-verification complete | ticker=%s | "
            "original_density=%.2f | post_density=%.2f | delta=%.2f",
            ticker, original_density, post_density, delta,
        )

        audit_table = (
            "\n\n---\n"
            "### Hallucination Audit Trail\n\n"
            "| Metric | Draft Report | Verified Rewrite |\n"
            "|:---|:---:|:---:|\n"
            f"| Citation Density | {original_density:.0%} | **{post_density:.0%}** |\n"
            f"| Verified Claims | {orig_verified_count} / {orig_total} "
            f"| **{post_verified_count} / {post_total}** |\n"
            f"| Unverified Claims | {orig_unverified_count} | **{post_unverified_count}** |\n"
            f"| Integrity Tier | {orig_tier} | **{post_tier}** |\n"
            f"| Density Improvement | — | **{improvement_str}** |\n\n"
            "*Automatically re-verified by FinSynth Hallucination Auditor against yfinance source data.*"
        )

        logs.append(_log(
            "post_fact_checker",
            f"Post-resynth density: **{post_density:.0%}** "
            f"(was {original_density:.0%} → improved by {improvement_str})",
            "completed",
        ))
        logger.info(
            "[post_fact_checker] Node finished | ticker=%s | total_elapsed=%.2fs",
            ticker, time.monotonic() - t_node,
        )
        return {
            "report": resynth_report + audit_table,
            "post_fact_check_result": result,
            "post_citation_density": post_density,
            "thinking_log": logs,
        }

    return post_fact_checker
