"""
Experiment ticker universe — 15 tickers split across three data-richness tiers.

Tier criteria:
  large_cap  — S&P 500 mega-caps; yfinance returns full income stmt, balance sheet,
               PE, forward PE, dividend yield, 52-week range.  lookup_populated=True
               expected for virtually every run.
  mid_cap    — Moderate analyst coverage; some yfinance fields may be None (e.g.
               dividend_yield, forward_pe for unprofitable names).
  small_cap  — Sparse or incomplete yfinance data.  Several of these may produce
               lookup_populated=False, which is *intentional* — it lets us study how
               the fact-checker degrades and whether mode choice matters when the
               ground-truth lookup is thin.
"""

TICKERS: list[dict] = [
    # ── Large-cap (5) ────────────────────────────────────────────────
    {"ticker": "AAPL",  "tier": "large_cap", "name": "Apple"},
    {"ticker": "MSFT",  "tier": "large_cap", "name": "Microsoft"},
    {"ticker": "GOOGL", "tier": "large_cap", "name": "Alphabet"},
    {"ticker": "NVDA",  "tier": "large_cap", "name": "NVIDIA"},
    {"ticker": "META",  "tier": "large_cap", "name": "Meta Platforms"},

    # ── Mid-cap (5) ──────────────────────────────────────────────────
    {"ticker": "ROKU",  "tier": "mid_cap",   "name": "Roku"},
    {"ticker": "BILL",  "tier": "mid_cap",   "name": "BILL Holdings"},
    {"ticker": "DKNG",  "tier": "mid_cap",   "name": "DraftKings"},
    {"ticker": "CAR",   "tier": "mid_cap",   "name": "Avis Budget Group"},
    {"ticker": "LYFT",  "tier": "mid_cap",   "name": "Lyft"},

    # ── Small-cap (5) ────────────────────────────────────────────────
    {"ticker": "BIRD",  "tier": "small_cap", "name": "Allbirds"},
    {"ticker": "ACMR",  "tier": "small_cap", "name": "ACM Research"},
    {"ticker": "SOUN",  "tier": "small_cap", "name": "SoundHound AI"},
    {"ticker": "MVIS",  "tier": "small_cap", "name": "MicroVision"},
    {"ticker": "WKHS",  "tier": "small_cap", "name": "Workhorse Group"},
]

# Quick lookup by ticker string
TICKER_META: dict[str, dict] = {t["ticker"]: t for t in TICKERS}
