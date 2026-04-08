#!/usr/bin/env python3
"""
Quick test script: calls get_financials directly and pretty-prints the result.

Usage:
    python test_get_financials.py [TICKER]

Defaults to AAPL if no ticker is provided.
"""

import asyncio
import json
import sys

# Make sure the app package is importable from this directory
sys.path.insert(0, ".")

from app.mcp_server import get_financials


async def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching financials for: {ticker}\n{'─' * 50}")

    raw = await get_financials(ticker)
    data = json.loads(raw)

    if "error" in data:
        print(f"ERROR: {data['error']}")
        return

    print(f"Company     : {data['company_name']} ({data['ticker']})")
    print(f"Sector      : {data.get('sector')} / {data.get('industry')}")
    print(f"Price       : ${data.get('current_price')}")
    print(f"Market Cap  : ${data.get('market_cap'):,.0f}" if data.get("market_cap") else "Market Cap  : N/A")
    print(f"P/E (trail) : {data.get('pe_ratio')}")
    print(f"P/E (fwd)   : {data.get('forward_pe')}")
    print(f"52-wk range : ${data.get('fifty_two_week_low')} – ${data.get('fifty_two_week_high')}")

    print(f"\n{'─' * 50}")
    print(f"{'Year':<6} {'Revenue':>14} {'Rev Gr%':>8} {'Gross%':>8} {'Op%':>8} {'Net%':>8} {'D/E':>6} {'CR':>6}")
    print(f"{'─'*6} {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")

    for m in data.get("metrics", []):
        rev = m.get("revenue")
        rev_str = f"${rev/1e9:.1f}B" if rev else "N/A"
        rev_gr = f"{m.get('revenue_growth_pct'):+.1f}%" if m.get("revenue_growth_pct") is not None else "N/A"
        gm = f"{m.get('gross_margin_pct'):.1f}%" if m.get("gross_margin_pct") is not None else "N/A"
        om = f"{m.get('operating_margin_pct'):.1f}%" if m.get("operating_margin_pct") is not None else "N/A"
        nm = f"{m.get('net_margin_pct'):.1f}%" if m.get("net_margin_pct") is not None else "N/A"
        de = f"{m.get('debt_to_equity'):.2f}" if m.get("debt_to_equity") is not None else "N/A"
        cr = f"{m.get('current_ratio'):.2f}" if m.get("current_ratio") is not None else "N/A"
        print(f"{m['year']:<6} {rev_str:>14} {rev_gr:>8} {gm:>8} {om:>8} {nm:>8} {de:>6} {cr:>6}")

    print(f"\n{'─' * 50}")
    print("Full JSON output:\n")
    print(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
