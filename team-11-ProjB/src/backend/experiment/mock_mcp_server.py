#!/usr/bin/env python3
"""
Mock MCP server for the experiment harness.

Serves pre-frozen financial and news data from disk instead of calling
yfinance or Brave Search live.  This guarantees that every (brief / normal /
extra) mode run on the same ticker sees identical ground-truth data, removing
data-drift as a confound.

Environment variable required:
    FINSYNTH_FROZEN_DIR — absolute path to the frozen_data/ directory

Launched automatically by run_experiment.py as a stdio subprocess.
Do not call this directly.
"""

import json
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    stream=sys.stderr,
    level=logging.WARNING,
    format="%(asctime)s [mock_mcp] %(message)s",
)
logger = logging.getLogger(__name__)

FROZEN_DIR = os.environ.get("FINSYNTH_FROZEN_DIR", "")
if not FROZEN_DIR:
    logger.error("FINSYNTH_FROZEN_DIR is not set — mock MCP server cannot start")
    sys.exit(1)

mcp = FastMCP("FinSynth Mock Tools")


@mcp.tool()
async def get_financials(ticker: str) -> str:
    """Return pre-frozen financial data for a ticker."""
    path = os.path.join(FROZEN_DIR, f"{ticker.upper()}_financials.json")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("No frozen financials for %s", ticker)
        return json.dumps({"error": f"No frozen data for {ticker}", "ticker": ticker})


@mcp.tool()
async def search_news(query: str) -> str:
    """Return pre-frozen news articles for the ticker extracted from the query."""
    ticker = query.split()[0].upper()
    path = os.path.join(FROZEN_DIR, f"{ticker}_news.json")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("No frozen news for %s", ticker)
        return json.dumps({
            "error": f"No frozen news for {ticker}",
            "query": query,
            "articles": [],
            "source": "mock_empty",
        })


if __name__ == "__main__":
    mcp.run()
