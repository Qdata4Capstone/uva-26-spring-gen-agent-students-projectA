# IMPROVE.md — team-9-ProjB (p2p-trade-bot — Player Props Extension)

## P1 — Critical / Security

- **SQL injection via f-string** (`bouncer.py:26`): `f"""...WHERE yes_price = {price}"""` builds a DuckDB query by string interpolation. Although the type is validated upstream, this pattern is fragile — use parameterized queries (`?` placeholders) instead.
- **Liquidity filter disabled for production** (`bouncer.py:81-84`): Minimum liquidity check is commented out with `# Disabled for testing`. The bot will accept zero-liquidity orders in production. Re-enable before any live trading scenario or add a feature flag.
- **Private key file not validated at startup** (`kalshi_rest.py:22`): `KALSHI_PRIVATE_KEY_PATH` is opened without checking existence or readability. A wrong path crashes the process mid-pipeline. Validate on startup and raise a clear `ConfigurationError`.

## P2 — Reliability

- **Blanket exception swallowing** (`bouncer.py:30-31`): `except Exception: return None` hides database corruption, network errors, and programming mistakes alike. Catch specific exceptions; log and re-raise unexpected ones.
- **Unimplemented market types silently return None** (`router.py:80-84`): `_handle_totals` and `_handle_props` are stubs returning placeholder results. A live KXNBAWINS trade enters the pipeline, gets a `None` result, and is silently dropped. Raise `NotImplementedError` or route to a dead-letter log until implemented.
- **No validation of API key / private key content**: `kalshi_rest.py` reads the key file but does not verify it is a valid PEM-encoded RSA key. A truncated or wrong-format file produces a cryptic `cryptography` library error at signing time.
- **Kelly sizing not bounded below zero** (`config.py`): If `PAPER_STARTING_CASH` is set to 0 or negative, Kelly calculations produce division-by-zero or negative bet sizes. Validate config values on import.

## P3 — Quality

- **No documentation of prop vs. game pipeline differences**: The new player-prop track (`KXNBAPTS`) shares most pipeline stages with game-winner, but the differences (prop parser, variance-adjusted Kelly) are not documented in `README.md` or inline. Future contributors will not know which stages are shared.
- **`PAPER_MAX_CONTRACTS` has no enforced upper bound**: Config accepts any integer; a misconfiguration could submit thousands of mock contracts in one tick. Add a sanity-check maximum (e.g., 100).
- **`kalshi_rest.py` returns `None` on any error**: Callers treat `None` as "no data available" but cannot distinguish a network timeout from a 401 auth failure from a 429 rate limit. Return a typed result or raise typed exceptions.
- **No integration test for prop parsing**: The prop parser that converts `KXNBAPTS` ticker strings into player/line/side is new in Project B and has no test coverage. A format change in Kalshi ticker naming would break it silently.

## P4 — Enhancements

- **Prop hit-rate history not persisted**: The calibration gap logic for game winners reads from a DuckDB parquet store, but player-prop hit rates are computed fresh each run. Persist prop history to DuckDB so the bot improves calibration over time.
- **CriticAgent VETO reason not logged separately**: The critic's VETO rationale is buried in the full agent trace. Write VETO reasons to a dedicated column in the SQLite trades log for post-hoc analysis.
- **No back-test mode for prop strategy**: The game-winner strategy has historical parquet data to back-test against. Add a `--backtest` flag that replays historical KXNBAPTS data through the prop pipeline.
- **Sentiment agent prompt not updated for props**: `SentimentAgent` was written for game-winner markets; the system prompt likely does not handle player-prop context well. Review and specialize the prompt for prop-market sentiment signals.
