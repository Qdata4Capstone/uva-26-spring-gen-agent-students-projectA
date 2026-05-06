#!/usr/bin/env python3
"""
MCP Server exposing financial data tools for the FinSynth agents.

Tools:
  - get_financials(ticker) – income/balance sheet data with YoY growth & margins
  - search_news(query)     – recent news articles via Brave Search (or yfinance fallback)

Run directly:  python -m app.mcp_server   (stdio transport for LangGraph nodes)

NOTE: This process uses stdio transport, so stdout is reserved for the MCP
protocol.  All logging is directed to stderr only.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import httpx
import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP

# ── Logging (stderr only — stdout is MCP protocol) ───────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] mcp_server – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("MCP server process started (pid=%d)", os.getpid())

# ── MCP Server instance ──────────────────────────────────────────────
mcp = FastMCP("FinSynth Financial Tools")


# ── Helpers ───────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round((numerator / denominator) * 100, 2)


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return round(a / b, 2)


def _growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0:
        return None
    return round(((current - previous) / abs(previous)) * 100, 2)


# ── Tool: get_financials ─────────────────────────────────────────────

@mcp.tool()
async def get_financials(ticker: str) -> str:
    """
    Fetch financial data for a stock ticker including income statement,
    balance sheet, and computed YoY growth rates and margins.
    Returns a JSON string.
    """
    ticker_upper = ticker.upper()
    logger.info("[get_financials] Tool invoked | ticker=%s", ticker_upper)
    t_start = time.monotonic()

    try:
        logger.debug("[get_financials] Fetching yfinance Ticker object | ticker=%s", ticker_upper)
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        company_name = info.get("longName", ticker_upper)
        logger.info(
            "[get_financials] Ticker info retrieved | ticker=%s | company=%s | sector=%s",
            ticker_upper,
            company_name,
            info.get("sector"),
        )

        logger.debug("[get_financials] Fetching income statement | ticker=%s", ticker_upper)
        income_stmt = stock.income_stmt
        logger.debug("[get_financials] Fetching balance sheet | ticker=%s", ticker_upper)
        balance_sheet = stock.balance_sheet

        # ── Parse income statement ────────────────────────────────
        income_data: dict[str, dict] = {}
        if income_stmt is not None and not income_stmt.empty:
            cols = income_stmt.columns[:4]
            logger.debug(
                "[get_financials] Parsing income statement | ticker=%s | periods=%d | rows=%d",
                ticker_upper,
                len(cols),
                len(income_stmt.index),
            )
            for col in cols:
                year = str(col.year) if hasattr(col, "year") else str(col)
                income_data[year] = {
                    idx: _safe_float(income_stmt.loc[idx, col])
                    for idx in income_stmt.index
                }
        else:
            logger.warning("[get_financials] Income statement is empty | ticker=%s", ticker_upper)

        # ── Parse balance sheet ───────────────────────────────────
        balance_data: dict[str, dict] = {}
        if balance_sheet is not None and not balance_sheet.empty:
            cols = balance_sheet.columns[:4]
            logger.debug(
                "[get_financials] Parsing balance sheet | ticker=%s | periods=%d | rows=%d",
                ticker_upper,
                len(cols),
                len(balance_sheet.index),
            )
            for col in cols:
                year = str(col.year) if hasattr(col, "year") else str(col)
                balance_data[year] = {
                    idx: _safe_float(balance_sheet.loc[idx, col])
                    for idx in balance_sheet.index
                }
        else:
            logger.warning("[get_financials] Balance sheet is empty | ticker=%s", ticker_upper)

        # ── Compute metrics per year ──────────────────────────────
        years = sorted(
            set(list(income_data.keys()) + list(balance_data.keys())),
            reverse=True,
        )
        logger.debug(
            "[get_financials] Computing per-year metrics | ticker=%s | years=%s",
            ticker_upper,
            years,
        )
        metrics: list[dict] = []

        for i, year in enumerate(years):
            inc = income_data.get(year, {})
            bal = balance_data.get(year, {})

            revenue = inc.get("Total Revenue")
            net_income = inc.get("Net Income")
            gross_profit = inc.get("Gross Profit")
            operating_income = inc.get("Operating Income")
            total_assets = bal.get("Total Assets")
            total_liab = (
                bal.get("Total Liabilities Net Minority Interest")
                or bal.get("Total Liabilities")
            )
            equity = (
                bal.get("Stockholders Equity")
                or bal.get("Total Equity Gross Minority Interest")
            )
            current_assets = bal.get("Current Assets")
            current_liab = bal.get("Current Liabilities")

            prev_inc = income_data.get(years[i + 1], {}) if i + 1 < len(years) else {}

            year_metrics = {
                "year": year,
                "revenue": revenue,
                "revenue_growth_pct": _growth(revenue, prev_inc.get("Total Revenue")),
                "net_income": net_income,
                "net_income_growth_pct": _growth(net_income, prev_inc.get("Net Income")),
                "gross_profit": gross_profit,
                "operating_income": operating_income,
                "gross_margin_pct": _pct(gross_profit, revenue),
                "operating_margin_pct": _pct(operating_income, revenue),
                "net_margin_pct": _pct(net_income, revenue),
                "total_assets": total_assets,
                "total_liabilities": total_liab,
                "stockholders_equity": equity,
                "debt_to_equity": _ratio(total_liab, equity),
                "current_ratio": _ratio(current_assets, current_liab),
            }
            logger.debug(
                "[get_financials] Year metrics | ticker=%s | year=%s | revenue=%s | net_margin_pct=%s",
                ticker_upper,
                year,
                revenue,
                year_metrics["net_margin_pct"],
            )
            metrics.append(year_metrics)

        result = {
            "ticker": ticker_upper,
            "company_name": company_name,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "metrics": metrics,
        }

        elapsed = time.monotonic() - t_start
        logger.info(
            "[get_financials] Tool completed successfully | ticker=%s | years_computed=%d | elapsed=%.2fs",
            ticker_upper,
            len(metrics),
            elapsed,
        )
        return json.dumps(result, indent=2, default=str)

    except Exception as exc:
        elapsed = time.monotonic() - t_start
        logger.exception(
            "[get_financials] Tool raised an exception | ticker=%s | error=%s | elapsed=%.2fs",
            ticker_upper,
            str(exc),
            elapsed,
        )
        return json.dumps({"error": str(exc), "ticker": ticker})


# ── Tool: search_news ────────────────────────────────────────────────

@mcp.tool()
async def search_news(query: str) -> str:
    """
    Search for recent news articles about a stock or company.
    Uses Brave Search API when a key is available, otherwise falls back to yfinance news.
    Returns a JSON string.
    """
    logger.info("[search_news] Tool invoked | query='%s'", query)
    t_start = time.monotonic()

    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    brave_available = bool(api_key and api_key != "your_brave_search_api_key_here")
    logger.debug(
        "[search_news] Brave Search API key %s",
        "present — will attempt Brave first" if brave_available else "absent or placeholder — skipping to yfinance fallback",
    )

    # ── Try Brave Search first ────────────────────────────────────
    if brave_available:
        try:
            url = "https://api.search.brave.com/res/v1/news/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            }
            params = {"q": query, "count": 10, "freshness": "pw"}

            logger.debug("[search_news] Sending Brave Search request | url=%s | params=%s", url, params)
            t_brave = time.monotonic()

            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers, params=params, timeout=30.0)
                resp.raise_for_status()
                data = resp.json()

            articles = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("description", ""),
                    "published": r.get("age", ""),
                    "source": (r.get("meta_url") or {}).get("hostname", ""),
                }
                for r in data.get("results", [])
            ]
            logger.info(
                "[search_news] Brave Search succeeded | query='%s' | articles=%d | elapsed=%.2fs",
                query,
                len(articles),
                time.monotonic() - t_brave,
            )
            return json.dumps({"query": query, "articles": articles, "source": "brave"})

        except Exception as exc:
            logger.warning(
                "[search_news] Brave Search failed — falling back to yfinance | query='%s' | error=%s",
                query,
                str(exc),
            )

    # ── Fallback: yfinance news ───────────────────────────────────
    try:
        maybe_ticker = query.split()[0].upper()
        logger.debug(
            "[search_news] Fetching yfinance news | ticker=%s | query='%s'",
            maybe_ticker,
            query,
        )
        t_yf = time.monotonic()

        stock = yf.Ticker(maybe_ticker)
        raw_news = stock.news or []

        articles = []
        for item in raw_news[:10]:
            content = item.get("content", {}) if isinstance(item, dict) else {}
            articles.append({
                "title": content.get("title", item.get("title", "")),
                "url": content.get("canonicalUrl", {}).get("url", item.get("link", "")),
                "description": content.get("summary", item.get("publisher", "")),
                "published": content.get("pubDate", str(item.get("providerPublishTime", ""))),
                "source": content.get("provider", {}).get("displayName", item.get("publisher", "")),
            })

        logger.info(
            "[search_news] yfinance fallback succeeded | ticker=%s | articles=%d | elapsed=%.2fs",
            maybe_ticker,
            len(articles),
            time.monotonic() - t_yf,
        )
        return json.dumps({"query": query, "articles": articles, "source": "yfinance"})

    except Exception as exc:
        elapsed = time.monotonic() - t_start
        logger.exception(
            "[search_news] yfinance fallback also failed | query='%s' | error=%s | elapsed=%.2fs",
            query,
            str(exc),
            elapsed,
        )
        return json.dumps({"error": str(exc), "query": query})


# ── Entry point (stdio transport) ────────────────────────────────────
if __name__ == "__main__":
    mcp.run()
