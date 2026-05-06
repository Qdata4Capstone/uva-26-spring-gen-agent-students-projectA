#!/usr/bin/env python3
"""
Step 1 — Pre-fetch and cache financial + news data for all experiment tickers.

Run this ONCE before starting the experiment runner.  All 15 tickers are
fetched at the same point in time; every subsequent analysis run reads from
these frozen snapshots, so the yfinance ground-truth is identical across
brief / normal / extra modes for any given ticker.

Usage:
    cd backend
    python -m experiment.freeze_data

Output:
    experiment/frozen_data/{TICKER}_financials.json
    experiment/frozen_data/{TICKER}_news.json
    experiment/frozen_data/metadata.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make backend/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from experiment.tickers import TICKERS

FROZEN_DIR = Path(__file__).parent / "frozen_data"


# ── Financial data helpers (mirrors mcp_server.py logic) ─────────────

def _safe_float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _pct(n: float | None, d: float | None) -> float | None:
    if n is None or d is None or d == 0:
        return None
    return round((n / d) * 100, 2)


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return round(a / b, 2)


def _growth(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev is None or prev == 0:
        return None
    return round(((curr - prev) / abs(prev)) * 100, 2)


def fetch_financials(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info or {}

    income_stmt = stock.income_stmt
    balance_sheet = stock.balance_sheet

    income_data: dict[str, dict] = {}
    if income_stmt is not None and not income_stmt.empty:
        for col in income_stmt.columns[:4]:
            year = str(col.year) if hasattr(col, "year") else str(col)
            income_data[year] = {idx: _safe_float(income_stmt.loc[idx, col]) for idx in income_stmt.index}

    balance_data: dict[str, dict] = {}
    if balance_sheet is not None and not balance_sheet.empty:
        for col in balance_sheet.columns[:4]:
            year = str(col.year) if hasattr(col, "year") else str(col)
            balance_data[year] = {idx: _safe_float(balance_sheet.loc[idx, col]) for idx in balance_sheet.index}

    years = sorted(set(list(income_data.keys()) + list(balance_data.keys())), reverse=True)
    metrics: list[dict] = []

    for i, year in enumerate(years):
        inc = income_data.get(year, {})
        bal = balance_data.get(year, {})
        prev_inc = income_data.get(years[i + 1], {}) if i + 1 < len(years) else {}

        revenue = inc.get("Total Revenue")
        net_income = inc.get("Net Income")
        gross_profit = inc.get("Gross Profit")
        operating_income = inc.get("Operating Income")
        total_assets = bal.get("Total Assets")
        total_liab = bal.get("Total Liabilities Net Minority Interest") or bal.get("Total Liabilities")
        equity = bal.get("Stockholders Equity") or bal.get("Total Equity Gross Minority Interest")
        current_assets = bal.get("Current Assets")
        current_liab = bal.get("Current Liabilities")

        metrics.append({
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
        })

    return {
        "ticker": ticker.upper(),
        "company_name": info.get("longName", ticker.upper()),
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


async def fetch_news(ticker: str, brave_key: str | None) -> dict:
    query = f"{ticker} stock news analysis"

    if brave_key:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/news/search",
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": brave_key,
                    },
                    params={"q": query, "count": 10, "freshness": "pw"},
                    timeout=30.0,
                )
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
            return {"query": query, "articles": articles, "source": "brave"}
        except Exception as e:
            print(f"    Brave failed ({e}), falling back to yfinance news")

    # yfinance fallback
    stock = yf.Ticker(ticker)
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
    return {"query": query, "articles": articles, "source": "yfinance"}


async def main() -> None:
    FROZEN_DIR.mkdir(exist_ok=True)

    load_dotenv(Path(__file__).parent.parent / ".env")
    brave_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not brave_key or brave_key == "your_brave_search_api_key_here":
        brave_key = None

    print(f"News source: {'Brave Search' if brave_key else 'yfinance fallback'}")
    print(f"Freezing {len(TICKERS)} tickers → {FROZEN_DIR}\n")

    frozen_at = datetime.now(timezone.utc).isoformat()
    freeze_summary: list[dict] = []

    for t_info in TICKERS:
        ticker = t_info["ticker"]
        tier = t_info["tier"]
        print(f"  {ticker:6s} ({tier}) ...", end="", flush=True)
        t0 = time.monotonic()

        record: dict = {"ticker": ticker, "tier": tier, "financials_ok": False, "news_ok": False, "error": None}

        try:
            financials = fetch_financials(ticker)
            years_found = len(financials.get("metrics", []))
            lookup_size = sum(
                1 for k in ["current_price", "pe_ratio", "forward_pe", "market_cap",
                            "fifty_two_week_high", "fifty_two_week_low"]
                if financials.get(k) is not None
            )
            (FROZEN_DIR / f"{ticker}_financials.json").write_text(
                json.dumps(financials, indent=2, default=str), encoding="utf-8"
            )
            record["financials_ok"] = True
            record["years"] = years_found
            record["lookup_fields"] = lookup_size
        except Exception as e:
            record["error"] = f"financials: {e}"
            print(f" FINANCIALS ERROR: {e}")

        try:
            news = await fetch_news(ticker, brave_key)
            article_count = len(news.get("articles", []))
            (FROZEN_DIR / f"{ticker}_news.json").write_text(
                json.dumps(news, indent=2), encoding="utf-8"
            )
            record["news_ok"] = True
            record["articles"] = article_count
            record["news_source"] = news.get("source")
        except Exception as e:
            record["error"] = (record.get("error") or "") + f" news: {e}"

        elapsed = time.monotonic() - t0
        status = "ok" if record["financials_ok"] and record["news_ok"] else "partial"
        detail = f"{record.get('years', 0)}yr fin, {record.get('articles', 0)} articles"
        print(f" [{status}] {detail} ({elapsed:.1f}s)")

        freeze_summary.append(record)

    metadata = {
        "frozen_at": frozen_at,
        "tickers": [t["ticker"] for t in TICKERS],
        "summary": freeze_summary,
    }
    (FROZEN_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    ok_count = sum(1 for r in freeze_summary if r["financials_ok"] and r["news_ok"])
    print(f"\nFreeze complete: {ok_count}/{len(TICKERS)} tickers fully cached")
    print(f"Snapshot timestamp: {frozen_at}")


if __name__ == "__main__":
    asyncio.run(main())
