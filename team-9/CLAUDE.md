# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: p2p-trade-bot

Multi-agent prediction-market trading bot that exploits **longshot bias** in Kalshi NBA markets. Streams live trades via WebSocket, identifies mispriced contracts using historical calibration analysis, and routes them through a five-stage LLM agent pipeline before logging mock trades to SQLite.

**Team:** Jacob Huynh (sff8qc), Henry Chen (cqd3uk), Haoxuan Luo (ayr7tb)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY (required), KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH (for live streaming)
python mock_database_setup.py   # creates data/kalshi/markets/ and data/kalshi/trades/ parquet files
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude access (Haiku for Quant/Sentiment/Orchestrator, Sonnet for Critic) |
| `KALSHI_API_KEY_ID` | For live streaming | Kalshi API key UUID |
| `KALSHI_PRIVATE_KEY_PATH` | For live streaming | Absolute path to RSA private key `.pem` |
| `PAPER_STARTING_CASH` | No | Starting bankroll (default: `1000.0`) |
| `PAPER_MAX_CONTRACTS` | No | Max contracts per trade (default: `20`) |

Kalshi credentials are optional — when absent, market enrichment fields default to `"Unknown"` and the pipeline continues with mock data.

## Running

```bash
# Stream live Kalshi trades through the full agent pipeline
python -m src.pipeline.websocket_client

# Evaluate pending trades against final Kalshi results
python -m src.settle

# Report evaluated trades, bankroll, and market stats
python -m src.report_trades

# Query the live trade log directly
sqlite3 data/live_trades.db "SELECT ticker, action, yes_price, calibration_gap, status, pnl_usd FROM live_trades;"
```

## Tests

```bash
# Most tests — no API keys needed
pytest tests/ --ignore=tests/test_websocket.py -v

# WebSocket integration test — requires KALSHI + ANTHROPIC in .env
pytest tests/test_websocket.py -v

# Live pipeline test — requires ANTHROPIC_API_KEY + real parquet data
python tests/test_pipeline.py --live

# Single test file
pytest tests/test_bouncer.py -v
```

The test suite uses **real API calls** (ESPN, nba_api, DuckDB). LLM calls are mocked only in `test_pipeline.py` unit tests so they run without credentials.

## Agent Pipeline Architecture

```
Kalshi WebSocket → Router → Bouncer → QuantAgent + SentimentAgent (parallel)
                                              ↓
                                       LeadAnalyst (Orchestrator)
                                              ↓
                                        CriticAgent
                                              ↓
                                        TradeLogger (SQLite)
```

**Router** (`src/pipeline/router.py`) — classifies tickers: `KXNBAGAME-*` → full pipeline; `KXNBAWINS-*` / `KXNBASGPROP-*` → placeholder; other → dropped.

**Bouncer** (`src/pipeline/bouncer.py`) — longshot filter: YES ≤ 20¢ → `BET_NO`; YES ≥ 80¢ → `BET_YES`; 20–80¢ → dropped. Enriches trade packet via Kalshi REST.

**QuantAgent** (`src/agents/quant.py`) — all math computed in Python (DuckDB price-bucket queries + ESPN live context + nba_api team records); Claude Haiku writes one summary sentence. Verdicts: `EDGE_CONFIRMED` / `EDGE_WEAK` / `NO_EDGE` / `INSUFFICIENT_DATA`.

**SentimentAgent** (`src/agents/sentiment_agent.py`) — adds ESPN news context for GAME_WINNER contracts; Claude Haiku. Skips non-GAME_WINNER.

**LeadAnalyst** (`src/agents/orchestrator.py`) — runs Quant + Sentiment in `ThreadPoolExecutor`, computes Kelly fraction (capped at 15%), synthesizes narrative via Claude Haiku.

**CriticAgent** (`src/agents/critic.py`) — adversarial APPROVE/VETO using Claude Sonnet; hunts 7 failure modes (data contamination, liquidity, portfolio concentration, etc.).

**TradeLogger** (`src/execution/trade_logger.py`) — SQLite at `data/live_trades.db`; trades start as `PENDING_RESOLUTION`, become `EVALUATED` after `src/settle.py` runs.

## Data Layer

- `data/kalshi/markets/*.parquet` — finalized market metadata
- `data/kalshi/trades/*.parquet` — historical trade fills (mock by default; replace with real data from `jon-becker/prediction-market-analysis`)
- `data/live_trades.db` — SQLite live trade log (created automatically)
- `data/paper/` — paper trade book JSON + CSV equity curve

All DuckDB queries aggregate by **price bucket** across all finalized NBA markets — not by ticker — because live tickers won't appear in historical data.
