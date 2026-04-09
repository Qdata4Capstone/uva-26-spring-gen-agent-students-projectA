# Improvement Plan — p2p-trade-bot (team-9)

## Project Summary
A multi-agent trading bot that exploits longshot bias in Kalshi NBA prediction markets. It streams live trades via WebSocket, filters through a 6-stage agent pipeline (Router → Bouncer → QuantAgent + SentimentAgent → LeadAnalyst/Orchestrator → CriticAgent → TradeLogger), and persists mock trades to SQLite. All math is done in Python; LLMs are used only for qualitative synthesis.

---

## Strengths to Preserve
- Clean separation of concerns: each agent has a single responsibility
- Math in Python, LLM only for synthesis (not computation)
- Parallel Quant + Sentiment execution via `ThreadPoolExecutor`
- Tiered LLM usage: Haiku for speed, Sonnet for adversarial review
- Hard portfolio rules enforced before any LLM call
- Robust WebSocket with exponential backoff and RSA-PSS auth
- Good test coverage for existing components

---

## Priority 1 — Critical Gaps (Core Functionality)

### 1.1 Implement Real Trade Execution
**Problem:** `LiveTradeManager.execute()` immediately raises `NotImplementedError`. The bot cannot place real orders on Kalshi.

**Action:**
- Implement `execute()` using the Kalshi REST API's order placement endpoint (`POST /trade-api/v2/portfolio/orders`).
- Support market and limit order types.
- Add a `DRY_RUN` environment variable (default `true`) that logs the order without submitting it.
- Log all order attempts (accepted, rejected, filled) to SQLite alongside the trade record.

### 1.2 Implement TOTALS and PLAYER_PROP Strategies
**Problem:** Only `GAME_WINNER` has a full pipeline. TOTALS and PLAYER_PROP markets are placeholder stubs. This leaves ~66% of NBA Kalshi markets unprocessed.

**Action:**
- Design a separate calibration model for `TOTALS` (season win total) markets using historical over/under data.
- For `PLAYER_PROP` markets, implement a simpler signal: compare implied probability against historical player performance distributions.
- Route these through their own Bouncer filters with market-type-specific longshot thresholds.
- Ensure Critic hunts failure modes specific to each market type.

### 1.3 Implement the Missing Researcher Agent
**Problem:** The README mentions `researcher.py` in the architecture but the file does not exist.

**Action:**
- Clarify the intended role: deep fundamental analysis (team records, injuries, venue) before the Quant step.
- Implement a `ResearcherAgent` that fetches injury reports, recent team form, and home/away splits from public APIs (ESPN, NBA stats).
- Call it in parallel with the Quant agent inside the Orchestrator.

### 1.4 Add Real Historical Data Ingestion Pipeline
**Problem:** The mock database generator embeds artificial longshot bias. There is no script to ingest real Kalshi historical data.

**Action:**
- Write `scripts/ingest_historical.py` that downloads and normalizes real Kalshi trade data (e.g., from the jon-becker/prediction-market-analysis dataset referenced in the README).
- Validate that the ingested data reproduces expected longshot bias statistics before using it for edge calculations.
- Document the download and ingest steps in the README.

---

## Priority 2 — Robustness & Quality

### 2.1 Add Structured Logging (Replace `print()`)
**Problem:** All agent communication is via `print()`. There is no log level control, no structured output, and no way to pipe logs to an alerting system.

**Action:**
- Replace all `print()` calls with Python's `logging` module using a consistent format:
  ```
  %(asctime)s [%(name)s] %(levelname)s — %(message)s
  ```
- Write logs to a rotating file (`logs/bot_{date}.log`) in addition to stdout.
- Use `DEBUG` for raw WebSocket messages, `INFO` for trade decisions, `WARNING` for degraded API calls, `ERROR` for failures.

### 2.2 Validate Game Status Before Betting
**Problem:** The SentimentAgent searches only today + yesterday for ESPN game data. The bot may bet on games already in progress or already resolved.

**Action:**
- In the Bouncer, call the Kalshi REST API to check `market.status`. Reject any market that is not `open`.
- In the Orchestrator, cross-check the game start time from ESPN. If the game has started, gate with a higher confidence threshold or reject outright.

### 2.3 Add Bankroll Management with Drawdown Limits
**Problem:** `PAPER_STARTING_CASH` is fixed with no rebalancing. There are no stop-loss or profit-taking mechanisms. A losing streak depletes the bankroll without any circuit breaker.

**Action:**
- Implement a `BankrollManager` class that tracks current balance and drawdown from peak.
- Add configurable limits: `MAX_DRAWDOWN_PCT` (default 20%) and `DAILY_LOSS_LIMIT_USD`.
- Pause trading automatically when either limit is exceeded; resume after a configurable cooldown.

### 2.4 Add Performance Attribution Reporting
**Problem:** Settlement tracks P&L per trade but provides no breakdown by signal type, price bucket, confidence level, or game.

**Action:**
- Extend `report_trades.py` to compute:
  - Win rate and ROI by confidence level (HIGH / MEDIUM / LOW)
  - Win rate by price bucket (e.g., YES ≤ 10¢ vs 10–20¢)
  - Win rate by BET_YES vs BET_NO
  - Rolling 30-day P&L chart
- Output as a formatted terminal report and optionally as a CSV.

### 2.5 Centralize Magic Numbers as Config Constants
**Problem:** Threshold values (`20`, `80`, `1.5`, `0.75`, `15`, `200`, `100`) are repeated across files with no documentation of why they were chosen.

**Action:**
- Move all strategy parameters to `src/config.py` with descriptive names and inline comments:
  ```python
  LONGSHOT_THRESHOLD_LOW = 20   # YES ≤ 20¢: fade overpriced underdog
  LONGSHOT_THRESHOLD_HIGH = 80  # YES ≥ 80¢: fade overpriced favorite
  MIN_EDGE_PCT = 1.5            # Minimum calibration gap for EDGE_CONFIRMED
  MIN_SAMPLE_CONFIRMED = 200    # Min historical trades for EDGE_CONFIRMED
  KELLY_CAP = 0.15              # Maximum Kelly fraction
  ```

### 2.6 Expand Test Coverage
**Problem:** The Orchestrator's parallel execution and Kelly fraction calculation have no tests. Critic hard-rule enforcement is untested.

**Action:**
- Add unit tests for:
  - `orchestrator.py`: Kelly fraction edge cases (negative edge, divide by zero)
  - `critic.py`: hard rules (opposing positions, duplicate bets, zero order book depth)
  - `settle.py`: P&L calculation for wins and losses
  - `espn_tool.py`: ticker parsing with edge-case team abbreviations
- Add a full pipeline integration test using mock WebSocket messages and a mock Claude client.

---

## Priority 3 — Features & Code Quality

### 3.1 Refactor Ticker Parsing into a Shared Utility
**Problem:** Ticker parsing logic is duplicated between `espn_tool.py` and `nba_tool.py`.

**Action:**
- Create `src/tools/ticker_parser.py` with a single `parse_nba_ticker(ticker: str) -> dict` function.
- Import it in both tools; add validation that the parsed team abbreviations exist in a known NBA team list.

### 3.2 Replace String Literals with Enums
**Problem:** Verdict values (`EDGE_CONFIRMED`, `EDGE_WEAK`, `NO_EDGE`) and decision statuses (`READY`, `PASS`, `VETOED`, `APPROVED`) are string literals spread across multiple files.

**Action:**
- Define `VerdictType` and `DecisionStatus` enums in `src/models.py`.
- Replace all string comparisons with enum comparisons across the codebase.

### 3.3 Add Adaptive Kelly Scaling
**Problem:** The 15% Kelly cap is applied uniformly regardless of sample size or confidence level.

**Action:**
- Apply a confidence multiplier: `effective_kelly = kelly_fraction × confidence_multiplier`
  - HIGH confidence: multiplier = 1.0
  - MEDIUM confidence: multiplier = 0.5
  - LOW confidence: multiplier = 0.25
- Document the rationale in `config.py`.

### 3.4 Shorten and Modularize the Critic System Prompt
**Problem:** The Critic system prompt is 162 lines of repetitive instructions. This wastes tokens on every call.

**Action:**
- Extract the 7 failure modes into a concise numbered list (one line each).
- Move portfolio correlation rules to a separate template filled in programmatically with live portfolio data.
- Target: reduce from 162 lines to ~60 lines without losing coverage.

---

## Summary Roadmap

| Priority | Item | Effort |
|----------|------|--------|
| 1 | Implement real trade execution (with `DRY_RUN`) | High |
| 1 | Implement TOTALS + PLAYER_PROP strategies | High |
| 1 | Implement Researcher agent | Medium |
| 1 | Real historical data ingestion script | Medium |
| 2 | Replace `print()` with structured logging | Low |
| 2 | Validate game status before betting | Low |
| 2 | Bankroll management with drawdown limits | Medium |
| 2 | Performance attribution reporting | Medium |
| 2 | Centralize magic numbers in `config.py` | Low |
| 2 | Expand test coverage (Orchestrator, Critic, Settle) | Medium |
| 3 | Shared ticker parser utility | Low |
| 3 | Enums for verdict/decision strings | Low |
| 3 | Adaptive Kelly scaling by confidence | Low |
| 3 | Shorten Critic system prompt | Low |
