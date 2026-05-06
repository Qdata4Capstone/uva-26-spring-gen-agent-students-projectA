# IMPROVE.md — team-11-ProjB (FinSynth — Extended)

## P1 — Critical / Security

- **yfinance schema changes silently corrupt output** (`mcp_server.py:130+`): Income statement and balance sheet parsing assumes a fixed column order from `yfinance`. If Yahoo Finance changes its schema, the MCP tool returns silently wrong financial data with no error. Add schema validation (check expected column names exist) before parsing.
- **No automated tests for the MCP server**: `test_get_financials.py` is a manual smoke test only. Zero test coverage for error cases (invalid ticker, missing data, network timeout). A regression in `get_financials` would not be caught before deployment.
- **CORS origins hardcoded to localhost** (`config.py:26`): Deployment behind a tunnel or on a remote server requires manually editing source. Move to `ALLOWED_ORIGINS` env var.

## P2 — Reliability

- **Ollama connectivity not checked on startup** (`config.py:20-21`): If the Ollama server is unreachable, the app starts successfully but every `/api/analyze` call hangs for up to 600 s before timing out. Add a startup probe and fail fast with a clear error.
- **SSE stream hangs on Ollama timeout** (`main.py:151-165`): `httpx` default timeout is 600 s. If Ollama is slow or down, the client receives no data for 10 minutes before an error. Set an explicit shorter timeout and stream a heartbeat or error event.
- **`run_analysis()` yields events without schema validation** (`main.py:107-128`): Malformed intermediate events (e.g., missing `type` key) are JSON-encoded and sent to the client, causing the frontend to silently drop or misrender them. Validate event shape before yielding.
- **yfinance data has no freshness check** (`mcp_server.py:99+`): `yf.Ticker().info` can return stale cached data. Add a timestamp check; warn the user if data is older than 24 h.

## P3 — Quality

- **Fact-checker number extraction unbounded** (`nodes.py:52-65`): The regex parses any integer including impossibly large values (e.g., `999999999999`). Add a reasonable upper bound (e.g., 10 trillion) to avoid matching arbitrary numbers in text.
- **Non-English financial scales ignored** (`nodes.py:84-92`): Scale map covers billion/million/trillion/thousand but not "lakh" or "crore" (common in Indian financial reporting). These are silently skipped, producing wrong totals for non-US companies.
- **Year values collide with financial data** (`nodes.py:68-116`): Fact lookup flattens all numbers including years (e.g., "fiscal year 2024" → `2024`), which can false-match against a revenue figure of "2024 million". Filter out 4-digit year-range numbers before comparison.
- **No retry logic on yfinance calls**: A single network blip causes the MCP tool to return an error immediately. Add 1–2 retries with backoff before surfacing the error.

## P4 — Enhancements

- **Auditor and News Hound run sequentially**: The LangGraph graph runs Node A → Node B serially. These nodes use independent data sources and could run in parallel, halving latency. Refactor to a parallel fan-out pattern in LangGraph.
- **No ticker symbol validation before yfinance call**: An invalid ticker (e.g., typo) causes yfinance to return empty data with no error. Add a pre-check (e.g., verify the ticker exists) and return a clear error to the user.
- **Report caching missing**: Identical ticker requests within a short window re-run the full pipeline. Add a short TTL cache (e.g., 5 min) keyed on ticker to avoid redundant LLM calls.
- **Frontend has no loading state timeout**: The SSE stream can run for a long time; the UI should show a timeout warning (e.g., after 60 s) if no complete event has arrived.
