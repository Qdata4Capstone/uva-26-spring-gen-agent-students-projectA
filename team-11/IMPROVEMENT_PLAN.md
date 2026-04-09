# Improvement Plan — FinSynth (team-11)

## Project Summary
FinSynth is a multi-agent investment analysis system. A Next.js frontend sends a ticker to a FastAPI backend, which runs a sequential LangGraph pipeline: Auditor (financial analysis) → News Hound (sentiment) → Synthesizer (report generation). Each node calls an MCP server (stdio subprocess) backed by yfinance and Brave Search. Results stream to the UI via Server-Sent Events.

---

## Strengths to Preserve
- Clean architecture: config → state → nodes → graph → main
- MCP subprocess with graceful Brave → yfinance fallback
- Real-time SSE streaming with live "thinking log"
- Comprehensive financial metrics (YoY growth, margins, ratios)
- Multi-level structured logging with timing instrumentation
- Professional Bloomberg-style UI with responsive layout

---

## Priority 1 — Critical Fixes (Correctness & Resilience)

### 1.1 Parallelize Independent Nodes
**Problem:** Auditor and News Hound run sequentially even though they are completely independent. This adds 2–5 unnecessary seconds to each analysis.

**Action:**
- Restructure the LangGraph graph: `START → [Auditor, News Hound] (parallel) → Synthesizer → END`.
- Use LangGraph's native parallel branching (fan-out/fan-in) to run both nodes concurrently.
- Verify that the `thinking_log` reducer correctly merges interleaved entries from both nodes.
- Expected latency improvement: ~40% reduction (6–13s → 4–8s).

### 1.2 Add Retry Logic with Timeout Handling
**Problem:** If an MCP tool call times out or the yfinance API is slow, the entire pipeline fails with a generic 500 error. LLM calls have no timeout.

**Action:**
- Wrap each MCP tool call in a retry loop with exponential backoff (max 3 attempts, starting at 1s).
- Add a timeout (e.g., 30s) to each `ainvoke()` LLM call using `asyncio.wait_for()`.
- If the final retry fails, emit a structured SSE error event with an actionable message (e.g., "Financial data unavailable — yfinance may be rate-limited. Try again in 60s.").

### 1.3 Add Input Validation for Ticker Symbol
**Problem:** The frontend accepts any string as a ticker. Invalid tickers (e.g., `"XYZ999"`) are not caught until the MCP server tries yfinance, producing a cryptic error.

**Action:**
- Add a pre-analysis validation step in `workflow.py`: call `yf.Ticker(ticker).info` and check that `info.get("symbol")` matches the input.
- If invalid, emit an SSE "error" event immediately with "Invalid ticker symbol" rather than starting the pipeline.
- Add a frontend hint in the search bar: "Enter a valid US stock ticker (e.g., AAPL, MSFT)".

### 1.4 Fix CORS for Production Deployment
**Problem:** CORS is hardcoded to `http://localhost:5173` in `main.py`. Deployment to any other host requires a code change.

**Action:**
- Read allowed origins from the `CORS_ORIGINS` environment variable (comma-separated list).
- Default to `http://localhost:5173` only in development mode.
- Add `CORS_ORIGINS` to `.env.example`.

### 1.5 Add Report Cleanup / Archival Strategy
**Problem:** Reports are saved to `/reports/{TICKER}_{TIMESTAMP}.md` indefinitely with no TTL or cleanup.

**Action:**
- Add a configurable `REPORT_RETENTION_DAYS` environment variable (default: 30).
- On startup, delete reports older than the retention period.
- Alternatively, store reports in a lightweight SQLite database with a `created_at` column for easy querying and cleanup.

---

## Priority 2 — Robustness & Quality

### 2.1 Add Response Validation for LLM Output
**Problem:** The Synthesizer's LLM output is used as-is without verifying it matches the expected Markdown structure. A malformed response is streamed directly to the frontend.

**Action:**
- After the Synthesizer LLM call, check that required sections exist (e.g., `## Executive Summary`, `## Recommendation`).
- If validation fails, retry the LLM call once with an explicit instruction to include the missing sections.
- If it still fails, emit a warning SSE event alongside the partial report.

### 2.2 Add Caching for Repeated Ticker Analysis
**Problem:** Submitting the same ticker twice within a session re-fetches all data and re-runs all LLM calls. This wastes API quota and time.

**Action:**
- Implement an in-memory LRU cache (using `functools.lru_cache` or `cachetools.TTLCache`) for MCP tool results, keyed by `(ticker, date)` with a 15-minute TTL.
- Cache the final report as well; serve it from cache if the ticker was analyzed in the last 15 minutes.

### 2.3 Fix yfinance Data Handling for Sparse Tickers
**Problem:** `mcp_server.py` takes the first 4 columns of financial statements without checking if fewer than 4 years of data exist. Small-cap or recently-listed tickers silently produce truncated or malformed JSON.

**Action:**
- Check the actual number of available columns before slicing; use all available columns up to 4.
- Add a data quality note to the JSON output: `"data_years_available": 2` so the Auditor LLM can caveat its analysis appropriately.

### 2.4 Add Rate Limiting on the API Endpoint
**Problem:** There is no throttling on the `/api/analyze` endpoint. A user can submit unlimited concurrent requests, exhausting Gemini API quotas.

**Action:**
- Add a `slowapi` rate limiter: max 5 requests per minute per IP.
- Return HTTP 429 with a `Retry-After` header when the limit is exceeded.
- Disable the Analyze button in the frontend while a request is in flight.

### 2.5 Build a Minimal Test Suite
**Problem:** The only test file is a manual integration script (`test_get_financials.py`). There are no automated tests.

**Action:**
- Write `pytest` tests for:
  - `mcp_server.py`: `get_financials()` with a known ticker (AAPL) — assert required keys in JSON
  - `nodes.py`: Auditor node with mocked MCP session and mocked LLM — assert `auditor_analysis` is populated
  - `main.py`: `POST /api/analyze` endpoint with mocked pipeline — assert SSE event sequence
- Use `pytest-asyncio` for async tests; mock Gemini with `unittest.mock.AsyncMock`.

---

## Priority 3 — Features & UX

### 3.1 Add Analysis History
**Problem:** Reports are displayed only in the current session. Navigating away loses all work.

**Action:**
- Store each completed analysis in SQLite (ticker, timestamp, report markdown, metadata).
- Add a "Recent Analyses" sidebar in the frontend listing the last 10 tickers analyzed.
- Allow clicking a history entry to reload the report without re-running the pipeline.

### 3.2 Add Report Export
**Problem:** Users can copy the report text but cannot export it as PDF or structured data.

**Action:**
- Add an "Export PDF" button in the frontend using a browser-side Markdown-to-PDF library (e.g., `jsPDF` + `marked`).
- Add a "Download JSON" button that returns the raw `financial_data` and `news_data` from the pipeline.

### 3.3 Expand Analysis Scope
**Problem:** The Auditor only analyzes income statement and balance sheet. The News Hound uses a generic query string. Missing: peer comparison, technical indicators, analyst ratings.

**Action (incremental):**
- Add a `get_analyst_ratings(ticker)` MCP tool using yfinance's `.recommendations` API.
- Add `get_peer_comparison(ticker)` that fetches the same metrics for 3 sector peers from yfinance.
- Pass both to the Synthesizer to enrich the Investment Thesis section.
- Improve the News Hound query: use `"{ticker} {company_name} earnings Q1 2026"` instead of the generic stock news query.

### 3.4 Make Model Configurable
**Problem:** `gemini-2.5-flash` is hardcoded in `config.py`. There is no way to switch models without editing source code.

**Action:**
- Add `GEMINI_MODEL` to `.env` / `config.py`; default to `gemini-2.5-flash`.
- Provide documentation on which model tiers are supported and their cost/latency trade-offs.

---

## Priority 4 — Documentation & Deployment

### 4.1 Add Monitoring / Observability
**Problem:** Logs go to stdout only. There are no metrics, traces, or alerting for production deployments.

**Action:**
- Add `prometheus-fastapi-instrumentator` to expose `/metrics` (request count, latency percentiles, error rates).
- Add OpenTelemetry tracing for the LangGraph node execution using `opentelemetry-sdk`.
- Document how to connect to a Grafana/Prometheus stack in the README.

### 4.2 Containerize the Application
**Problem:** There is no Dockerfile; the multi-service setup (backend + MCP server + frontend) is hard to reproduce.

**Action:**
- Write a `Dockerfile` for the FastAPI backend and a `Dockerfile` for the Next.js frontend.
- Write a `docker-compose.yml` that starts both services with one command.
- Add `.env.example` documenting all required and optional variables.

---

## Summary Roadmap

| Priority | Item | Effort |
|----------|------|--------|
| 1 | Parallelize Auditor + News Hound nodes | Medium |
| 1 | Add retry logic + timeout handling | Medium |
| 1 | Add ticker input validation | Low |
| 1 | Fix CORS for production | Low |
| 1 | Add report cleanup / archival | Low |
| 2 | Validate Synthesizer LLM output structure | Medium |
| 2 | Add caching (15-min TTL per ticker) | Medium |
| 2 | Fix yfinance sparse data handling | Low |
| 2 | Add rate limiting (5 req/min/IP) | Low |
| 2 | Build minimal test suite | Medium |
| 3 | Analysis history with SQLite | Medium |
| 3 | Report export (PDF + JSON) | Medium |
| 3 | Expand: analyst ratings + peer comparison | Medium |
| 3 | Make model configurable via `.env` | Low |
| 4 | Add Prometheus metrics + OpenTelemetry | Medium |
| 4 | Dockerfile + docker-compose | Medium |
