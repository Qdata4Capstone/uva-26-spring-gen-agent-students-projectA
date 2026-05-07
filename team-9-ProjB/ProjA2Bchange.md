# Project A to Project B Changes — team-9 (p2p-trade-bot)

## What Was Project A

Project A was a multi-agent prediction-market trading bot that exploits **longshot bias** in Kalshi NBA **game-winner** markets only. It streamed live Kalshi trades via WebSocket, filtered for longshot opportunities (YES price ≤ 20¢ or ≥ 80¢), and routed them through a five-stage LLM pipeline:

```
WebSocket → Router → Bouncer → QuantAgent + SentimentAgent (parallel)
                                         → LeadAnalyst/Orchestrator
                                         → CriticAgent
                                         → TradeLogger (SQLite)
```

The `PLAYER_PROP` and `TOTALS` market routes in the router were **stub placeholders** that printed a one-liner and returned `None`. There was no web UI; the only interface was the CLI and SQLite.

---

## 1. New Market Type: Player Props Pipeline

**Project A:** `PLAYER_PROP` (tickers `KXNBASGPROP-*`) was a stub — the router printed a single log line and returned `None`. No analysis was performed.

**Project B:** Player props are now a fully implemented second pipeline track. When the router detects a `KXNBAPTS-*` or `KXNBASGPROP-*` ticker, it:

1. Calls `kalshi_rest.get_market_details(ticker)` to fetch the market title and rules.
2. Applies `_parse_prop_from_market()` — a regex parser that extracts `player_name`, `prop_type` (PTS/REB/AST), and `prop_threshold` from the title. Supported patterns include both the live Kalshi format (`"Jaylon Tyson: 15+ points"`) and the legacy `"Will X score N+ points?"` form.
3. Applies the same longshot filter (YES ≤ 20¢ → `BET_NO`; YES ≥ 80¢ → `BET_YES`; mid-price dropped).
4. Builds a structured `trade_packet` containing `player_name`, `prop_type`, `prop_threshold`, and `contract_type = "PLAYER_PROP"`.
5. Routes the packet to the new `PropAgent + SentimentAgent (parallel) → LeadAnalyst.analyze_prop_signal() → CriticAgent` path.

The ticker classification in `router.py` was also extended to explicitly name and drop new market types (`KXNBA2D`, `KXNBA3D`, `KXNBASPREAD`, `KXNBATOTAL`, `KXNBA1HTOTAL`, `KXNBASERIES`) that were previously silently caught by the generic NBA fallback.

---

## 2. New Agent: PlayerPropAgent (`src/agents/prop_agent.py`)

**Project A:** No agent existed for player props.

**Project B:** `PlayerPropAgent` (`prop_agent.py`) is a stats-based edge analyzer for player prop markets. It does not use historical parquet calibration data (which does not exist for props). Instead, edge is computed entirely from recent `nba_api` game-log data:

- Fetches the player's last 10 games via `nba_player_stats_tool.get_player_recent_stats()`.
- Fetches season-level usage rate and true shooting % via `get_player_usage_rate()`.
- Fetches matchup history vs. the opponent via `get_player_matchup_history()` if an opponent abbreviation is available.
- Computes `hit_rate` (fraction of last N games where the player exceeded the prop line), rolling `recent_avg`, and `variance` in Python.
- Derives an `effective_win_rate` by inverting `hit_rate` for `BET_NO` trades.

Verdict thresholds differ from the game-winner calibration thresholds:

| Verdict | Condition |
|---|---|
| `EDGE_CONFIRMED` | `effective_win_rate > 65%` |
| `EDGE_WEAK` | `effective_win_rate 55–65%` |
| `NO_EDGE` | `effective_win_rate ≤ 55%` |
| `INSUFFICIENT_DATA` | fewer than 5 games sampled |

Kelly sizing is **capped at 5%** for props (vs. 15% for game winners) to reflect the higher per-game variance inherent in individual player stat lines.

The agent's output shape mirrors the `quant_summary` dict used by the game-winner path so that the Critic and TradeLogger can process both market types without structural changes.

---

## 3. New Tool: `nba_player_stats_tool.py`

**Project A:** The only NBA data tool was `nba_tool.py`, which fetched team-level W/L records for game-winner tickers. It was renamed `nba_team_tool.py` in Project B.

**Project B:** A new tool `nba_player_stats_tool.py` was added with four functions:

| Function | Purpose |
|---|---|
| `get_player_recent_stats(player_name, last_n=10)` | Last N game log: avg pts/reb/ast/min + per-game breakdown |
| `get_player_usage_rate(player_name)` | Season usage rate, true shooting %, pace context |
| `get_player_matchup_history(player_name, opponent_abbr, last_n=5)` | Avg pts/reb/ast in last N games vs. a specific opponent |
| `get_team_key_players(team_abbr, top_n=3)` | Top N scorers for a team: avg pts/reb/ast + last-5 scoring trend |

The first three functions are used by `PropAgent` for prop edge analysis. The fourth is used by the updated `GameQuantAgent` to enrich game-winner analysis with per-player scoring context.

---

## 4. Enhanced QuantAgent for Game Winners (`quant.py` → `game_quant_agent.py`)

**Project A:** `QuantAgent` (`quant.py`) fetched team W/L records from `nba_tool.py` and ESPN game context. It asked Claude for a **single-sentence** qualitative summary. It also fetched live order-book depth and included `orderbook_depth_at_price` in its output.

**Project B:** The agent was renamed `game_quant_agent.py` and enhanced:

- Now also fetches **key player stats** (top 3 scorers per team, last 10 games average and last 5 scoring trend) via `nba_player_stats_tool.get_team_key_players()`.
- Changed from a one-sentence to a **3-sentence structured summary**: Sentence 1 states the calibration gap and verdict; Sentence 2 notes team momentum and live game status and key player trends; Sentence 3 flags data quality concerns.
- Removed the live order-book depth fetch. The comment in the code explains this was intentional: snapshot order-book depth is unreliable immediately after a trade. `orderbook_depth_at_price` is no longer queried or returned.
- Outputs two additional fields: `home_key_players` and `away_key_players`.

---

## 5. Extended Orchestrator: Dual Pipeline Entry Points

**Project A:** `LeadAnalyst` had a single public method, `analyze_signal()`, which handled only game-winner trades.

**Project B:** `LeadAnalyst` was extended with a second entry point:

- `analyze_signal()` — unchanged game-winner path (imports `game_quant_agent.QuantAgent` instead of `quant.QuantAgent`).
- `analyze_prop_signal()` — new player-prop path: runs `PlayerPropAgent + SentimentAgent` in parallel, applies a Python-only gate (`PASS` if `verdict == INSUFFICIENT_DATA` or `verdict == NO_EDGE`), synthesizes prop stats + player news into a Critic-ready narrative via `_synthesize_prop()`, and maps the prop metrics into the `quant_summary` shape for Critic compatibility.

The prop gate threshold differs: `hit_rate ≤ 0.50` triggers a PASS (vs. `calibration_gap ≤ 0` for game winners).

---

## 6. Extended SentimentAgent: Player-Prop Enrichment

**Project A:** `SentimentAgent.enrich()` checked `contract_type == "GAME_WINNER"` and returned the packet unchanged for any other type (including `PLAYER_PROP`). Player prop trades received `sentiment_context = None`.

**Project B:** `SentimentAgent.enrich()` now has a second code path:

- If `contract_type == "PLAYER_PROP"`, it calls `_enrich_prop()`, which fetches the latest 30 ESPN NBA headlines, filters for articles mentioning the player by name (substring match on headline and description), and asks Claude Haiku for a 2–4 sentence player-specific summary covering injury/availability, usage changes, lineup context, and narrative.
- The game-winner path is unchanged.

---

## 7. Extended CriticAgent: Two New Failure Modes

**Project A:** The Critic hunted 7 failure modes (numbered 1–7 in the system prompt, plus a portfolio concentration check and a mandatory sentiment note).

**Project B:** Two new failure modes were added to the Critic's system prompt:

- **Failure Mode 9 — Player Prop Trades:** Explicit rules for `PLAYER_PROP` signals. Veto if `n_games_sampled < 5` with `EDGE_CONFIRMED` (overfitted on tiny sample); veto if `hit_rate > 0.90` (implausibly consistent); veto if variance is extremely high relative to the threshold gap. Kelly cap for props is 5% (not 15%); the Critic flags violations above that cap. Sentiment (injury news, usage changes, lineup) is weighted more heavily than in game-winner trades.

- The hard-rule check `_check_hard_rules()` in Project A included a **Rule D** for confirmed zero order-book depth (hard Python block before the LLM call). In Project B this rule was removed from the hard-block list — thin books can still fill via market orders, so it became a soft LLM concern rather than an automatic veto.

The Critic's model assignment (`claude-sonnet-4-6`) and structure are unchanged.

---

## 8. Extended TradeLogger Schema: `market_type`, `player_name`, `prop_threshold`

**Project A:** The SQLite `live_trades` table had no columns for player-specific data. All logged trades were implicitly game-winner trades.

**Project B:** Three columns were added to the schema:

| Column | Type | Purpose |
|---|---|---|
| `market_type` | TEXT | `'GAME_WINNER'` or `'PLAYER_PROP'` (default `'GAME_WINNER'`) |
| `player_name` | TEXT | Player name for `PLAYER_PROP` trades |
| `prop_threshold` | REAL | Numeric prop line for `PLAYER_PROP` trades |

Startup migrations are applied so that existing Project A databases are upgraded without data loss. The settle script (`src/settle.py`) works unchanged — it reads `result` directly from the Kalshi REST API for both market types once a market is finalized.

---

## 9. New Web UI: FastAPI + React Frontend

**Project A:** No web UI existed. The only interface was the CLI (`python -m src.pipeline.websocket_client`, `python -m src.settle`, `python -m src.report_trades`) and direct SQLite queries.

**Project B:** A complete visualization layer was added:

**Backend (`src/web/`):**
- `app.py` — FastAPI application (`uvicorn src.web.app:app --port 8000`). Runs the Kalshi WebSocket client in-process as a background task. Exposes REST endpoints:
  - `GET /api/health` — liveness + pipeline status
  - `GET /api/decisions` / `GET /api/decisions/{id}` — pipeline decision history (read from `data/decisions/` JSON files)
  - `GET /api/trades` — rows from `live_trades.db` with `?status=open|evaluated|all` filter
  - `GET /api/trades/summary` — aggregate P&L
  - `POST /api/settle` — stream settlement output back to the browser
  - `POST /api/mock` — replay a recent real decision or fall back to a hand-crafted fixture (no LLM call)
  - `WS /ws` — live pipeline event stream
- `decision_store.py` — reads/writes per-run decision JSON snapshots under `data/decisions/`.
- `events.py` — in-process event bus; agents emit events that are broadcast to WebSocket subscribers.
- `mock_fixtures.py` — hand-crafted game-winner and player-prop decision fixtures for UI development/testing.

**Frontend (`src/frontend/`):** React + TypeScript + Vite application (`npm run dev` on port 5173) with four pages:
- `/live` — three-column live view (incoming → processing → resolved) plus bouncer rejections; click any resolved decision to open the workflow visualization.
- `/decision/:id` — React Flow graph of Router → Bouncer → Quant/Prop ∥ Sentiment → Orchestrator → Critic → Logger; click any node for the full agent payload.
- `/trades` — filterable table of `live_trades.db` with expandable decision detail.
- `/settle` — run-settle button, summary stats, equity curve, and P&L-by-confidence chart.

New dependencies added to `requirements.txt`: `fastapi>=0.115.0`, `uvicorn>=0.30.0`.

---

## 10. New Test File: `test_mock_pipeline.py`

**Project A:** There was no test that exercised the full pipeline without API keys.

**Project B:** `tests/test_mock_pipeline.py` was added — a mock end-to-end demo that simulates 7 scenarios with all external calls mocked (Kalshi REST, nba_api, ESPN, Claude LLM). It produces the same formatted output as the live bot. Scenarios cover:

| # | Market | Outcome | What it demonstrates |
|---|---|---|---|
| 1 | GAME_WINNER | APPROVED | 7pp calibration gap, 418 samples, key player stats, Critic approves |
| 2 | GAME_WINNER | VETOED | `win_rate = 1.0` — Critic catches data contamination |
| 3 | GAME_WINNER | PASS | 55¢ mid-price — bouncer drops before pipeline runs |
| 4 | PLAYER_PROP | APPROVED | LeBron 25+ PTS, 80% hit rate, matchup history, Critic approves |
| 5 | PLAYER_PROP | VETOED | 95% hit rate — Critic flags as implausibly consistent |
| 6 | PLAYER_PROP | PASS | 40% hit rate — Python gate stops it before Critic is called |
| 7 | Logger | DB check | Logs one of each type; verifies `market_type`, `player_name`, `prop_threshold` columns |

A separate `tests/test_report_trades.py` was also added to test the reporting script.

---

## 11. Backtest Data Added

**Project A:** The `data/paper/` directory existed in the README but contained no committed files.

**Project B:** Three backtest result files are committed under `data/paper/`:
- `backtest_book.json` — portfolio state snapshot from a backtesting run
- `backtest_trades.csv` — log of simulated fills during the backtest
- `backtest_equity.csv` — equity curve across the backtest run

---

## 12. Repository Structure Changes

**Project A layout:**
```
team-9/
├── src/           (source at root)
├── tests/
├── data/
├── scripts/
└── requirements.txt
```

**Project B layout:** All source was moved one level deeper inside a new `src/` wrapper directory:
```
team-9-ProjB/
├── doc/
│   ├── README.md
│   └── final_presentation.pdf
├── IMPROVE.md
└── src/             (new top-level src wrapper)
    ├── src/         (application source)
    ├── tests/
    ├── data/
    ├── scripts/
    ├── frontend/    (new)
    └── requirements.txt
```

A `doc/` folder was also added containing the project's final presentation PDF and a high-level README.

---

## Summary Table: Project A vs. Project B

| Dimension | Project A | Project B |
|---|---|---|
| **Market types handled** | GAME_WINNER only | GAME_WINNER + PLAYER_PROP (full pipeline) |
| **Agents** | QuantAgent, SentimentAgent, LeadAnalyst, CriticAgent | + PlayerPropAgent; QuantAgent renamed to GameQuantAgent |
| **Quant agent summary** | 1 sentence | 3 structured sentences (edge / context / risk) |
| **Key player stats** | Not fetched | Top 3 scorers per team fetched and included in game-winner analysis |
| **Orderbook depth check** | Hard Python block (Rule D in Critic) | Removed — soft concern only; snapshot depth deemed unreliable |
| **Orchestrator entry points** | `analyze_signal()` only | `analyze_signal()` + `analyze_prop_signal()` |
| **Sentiment agent** | GAME_WINNER only; PLAYER_PROP was no-op | GAME_WINNER (unchanged) + player-specific ESPN news for PLAYER_PROP |
| **Critic failure modes** | 7 modes + portfolio concentration + sentiment note | + Failure Mode 9 (player prop data quality and Kelly cap) |
| **Prop router** | Stub — print and return None | Full parser: fetches market details, extracts player/prop/line via regex |
| **nba_api tools** | `nba_tool.py` (team W/L only) | `nba_team_tool.py` (team W/L) + `nba_player_stats_tool.py` (player stats) |
| **SQLite schema** | No player-prop columns | Added `market_type`, `player_name`, `prop_threshold` with migrations |
| **Kelly cap** | 15% for all trades | 15% for game winners; 5% for player props |
| **Web UI** | None (CLI only) | FastAPI backend + React/Vite frontend (4 pages + WebSocket event stream) |
| **Decision persistence** | None | `data/decisions/` JSON snapshots per pipeline run (powers workflow viz) |
| **Backtest data** | Empty `data/paper/` | Committed `backtest_book.json`, `backtest_trades.csv`, `backtest_equity.csv` |
| **Test coverage** | 6 test files; no no-key end-to-end test | + `test_mock_pipeline.py` (7 scenarios, no API keys) + `test_report_trades.py` |
| **Dependencies** | No web framework | Added `fastapi>=0.115.0`, `uvicorn>=0.30.0` |
| **Ticker classification** | `KXNBASGPROP-*` → PLAYER_PROP placeholder | `KXNBAPTS-*` and `KXNBASGPROP-*` → PLAYER_PROP full pipeline; explicit drop list for spread/total/series tickers |
| **Report script** | Not included in `src/` (mentioned in README) | `src/report_trades.py` included in the source tree |
| **Repo structure** | `team-9/src/` at root | `team-9-ProjB/src/src/` (one extra nesting level) + `doc/` folder |
