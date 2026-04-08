# p2p-trade-bot: Multi-Agent Kalshi NBA Prediction Market Trading Bot

**Team 9:** Jacob Huynh (sff8qc), Henry Chen (cqd3uk), Haoxuan Luo (ayr7tb)

---

## Introduction

Retail bettors in prediction markets systematically overprice NBA underdog contracts on Kalshi — a phenomenon called **longshot bias**. A team priced at 14¢ (implied 14% win probability) may only win 8% of the time historically, creating a 6 percentage-point calibration gap that a contrarian BET_NO strategy can exploit over many trades.

**p2p-trade-bot** automates this strategy end-to-end: it streams live Kalshi trades via WebSocket, filters for longshot opportunities, quantifies the historical edge using DuckDB, enriches with live ESPN context, and routes signals through a five-stage LLM agent pipeline before logging approved mock trades to SQLite.

All arithmetic (calibration gaps, Kelly fractions, verdicts) is computed in Python. The LLM agents evaluate signal quality and enforce risk management — they do not do math.

---

## Overall Function

```
Kalshi WebSocket → Router → Bouncer (longshot filter)
                                 ↓
                    QuantAgent ──┬── SentimentAgent   (parallel, Claude Haiku)
                                 ↓
                          LeadAnalyst / Orchestrator  (Kelly sizing, Claude Haiku)
                                 ↓
                           CriticAgent                (APPROVE / VETO, Claude Sonnet)
                                 ↓
                           TradeLogger → SQLite (live_trades.db)
```

**Longshot detection (Bouncer):**
- YES price ≤ 20¢ → `BET_NO` (YES underdog is overpriced; fade it)
- YES price ≥ 80¢ → `BET_YES` (NO underdog is overpriced; fade it)
- 20–80¢ → dropped (no systematic bias in this range)

**Resolution:** `python -m src.settle` polls Kalshi REST for final results and computes P&L on `PENDING_RESOLUTION` trades.

---

## Code Structure

```
team-9/
├── src/
│   ├── config.py                   # Env var config (PAPER_STARTING_CASH, etc.)
│   ├── settle.py                   # Kalshi REST-based trade resolution CLI
│   ├── report_trades.py            # CLI: bankroll + evaluated trade summary
│   │
│   ├── agents/
│   │   ├── orchestrator.py         # LeadAnalyst — parallel Quant+Sentiment synthesis,
│   │   │                           # Kelly fraction (capped at 15%), confidence scoring
│   │   ├── quant.py                # QuantAgent — calibration gap analysis (DuckDB + ESPN + nba_api)
│   │   ├── sentiment_agent.py      # SentimentAgent — ESPN news context (GAME_WINNER only)
│   │   ├── critic.py               # CriticAgent — adversarial APPROVE/VETO (7 failure modes)
│   │   └── researcher.py           # ResearchAgent — unused placeholder
│   │
│   ├── pipeline/
│   │   ├── router.py               # Ticker classifier: KXNBAGAME / KXNBAWINS / KXNBASGPROP
│   │   ├── bouncer.py              # Longshot filter + Kalshi REST enrichment
│   │   └── websocket_client.py     # Async Kalshi WebSocket (RSA-PSS auth, auto-reconnect)
│   │
│   ├── execution/
│   │   ├── trade_logger.py         # SQLite trade log (PENDING_RESOLUTION → EVALUATED)
│   │   └── trade_manager.py        # PaperTradeManager — position book + CSV equity curve
│   │
│   └── tools/
│       ├── kalshi_rest.py          # Kalshi REST API client (RSA-PSS auth)
│       ├── duckdb_tool.py          # Historical parquet queries (price-bucket aggregation)
│       ├── espn_tool.py            # ESPN scoreboard + news API wrapper
│       ├── nba_tool.py             # nba_api recent W/L records
│       └── news_tool.py            # News integration (placeholder)
│
├── mock_database_setup.py          # Generates realistic mock parquet data (with longshot bias)
├── requirements.txt
│
├── data/
│   ├── kalshi/
│   │   ├── markets/*.parquet       # Finalized market metadata
│   │   └── trades/*.parquet        # Historical trade fills
│   ├── live_trades.db              # SQLite live mock trade log
│   └── paper/                      # Paper trade book (book.json, trades.csv, equity.csv)
│
└── tests/
    ├── test_bouncer.py             # Bouncer filter unit tests (no API keys)
    ├── test_pipeline.py            # Full pipeline — LLM mocked for unit tests, --live flag for real
    ├── test_router.py              # Router classification + dispatch (no API keys)
    ├── test_espn_tool.py           # ESPN ticker parsing + live scoreboard (real public API)
    ├── test_nba_tool.py            # NBA ticker parsing + live nba_api
    ├── test_settle.py              # Settlement logic — mocked Kalshi REST (no keys needed)
    └── test_websocket.py           # WebSocket integration test (requires .env credentials)
```

---

## Installation

**Requirements:** Python 3.x

```bash
# 1. Clone and create virtual environment
cd team-9
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — see Configuration section below

# 4. Generate mock historical database
python mock_database_setup.py
# Creates data/kalshi/markets/ and data/kalshi/trades/ parquet files
```

### Configuration

Create `.env` in the project root:

```
# Required for LLM agents
ANTHROPIC_API_KEY=sk-ant-...

# Required for live WebSocket streaming + market enrichment
KALSHI_API_KEY_ID=your-api-key-id
KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_private_key.pem

# Optional — position sizing
PAPER_STARTING_CASH=1000.0
PAPER_MAX_CONTRACTS=20
```

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude Haiku (Quant/Sentiment/Orchestrator) + Sonnet (Critic) |
| `KALSHI_API_KEY_ID` | For live streaming | Kalshi API key UUID |
| `KALSHI_PRIVATE_KEY_PATH` | For live streaming | Absolute path to RSA private key `.pem` |
| `PAPER_STARTING_CASH` | No | Starting bankroll in dollars (default: `1000.0`) |
| `PAPER_MAX_CONTRACTS` | No | Max contracts per trade (default: `20`) |

> Kalshi credentials are optional for testing — when absent, market enrichment fields default to `"Unknown"` and the pipeline continues with mock data.

---

## How to Run

### Run the bot (live WebSocket streaming)

```bash
python -m src.pipeline.websocket_client
```

APPROVED signals are logged to `data/live_trades.db` as `PENDING_RESOLUTION`.

### Settle pending trades

Polls Kalshi REST for final results and computes P&L:

```bash
python -m src.settle
```

### View trade report

```bash
python -m src.report_trades
```

### Query the trade log directly

```bash
sqlite3 data/live_trades.db \
  "SELECT ticker, action, yes_price, calibration_gap, status, pnl_usd FROM live_trades;"
```

### Generate mock historical data

```bash
python mock_database_setup.py
```

Produces realistic parquet files with quadratic longshot bias in the 1–20¢ range, season-aware bias erosion for player props, and liquidity variation.

> To use real historical data: download parquet files from [jon-becker/prediction-market-analysis](https://github.com/jon-becker/prediction-market-analysis) and drop them into `data/kalshi/`.

---

## Running Tests

```bash
# Most tests — no API keys needed
pytest tests/ --ignore=tests/test_websocket.py -v

# Single test file
pytest tests/test_bouncer.py -v

# WebSocket integration test (requires KALSHI + ANTHROPIC in .env)
pytest tests/test_websocket.py -v

# Full pipeline with real Claude (requires ANTHROPIC_API_KEY + real parquet data)
python tests/test_pipeline.py --live
```

The test suite uses **real API calls** for ESPN, nba_api, and DuckDB. LLM calls are mocked in `test_pipeline.py` unit tests so they run without credentials.

---

## References

- **Video Demo:** https://youtu.be/l5mNaxWArlU
- **Kalshi API docs:** RSA-PSS authentication, WebSocket trade feed, REST market details
- **ESPN hidden API:** NBA scoreboard and news endpoints used by `espn_tool.py`
- **nba_api:** `nba_api` Python package for team W/L records
- **Real historical data:** [jon-becker/prediction-market-analysis](https://github.com/jon-becker/prediction-market-analysis)
