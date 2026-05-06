# FinSynth — Financial Synthesis AI Agent

A multi-agent AI system that analyzes stock financials and news to produce structured investment reports. Built with **LangGraph**, **Model Context Protocol (MCP)**, **FastAPI**, and **Next.js**.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Next.js Frontend                   │
│           (Bloomberg-style Dark Terminal UI)          │
│          Search Bar → Thinking Log + Report          │
└────────────────────────┬─────────────────────────────┘
                         │ SSE Stream (POST /api/analyze)
┌────────────────────────▼─────────────────────────────┐
│                  FastAPI Backend                      │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │            LangGraph State Machine              │ │
│  │                                                 │ │
│  │  [START] → Auditor → News Hound → Synthesizer  │ │
│  │             (Node A)   (Node B)     (Node C)    │ │
│  └──────┬──────────────────┬──────────────────────┘ │
│         │                  │                         │
│  ┌──────▼──────────────────▼──────────────────────┐ │
│  │         MCP Server (stdio transport)           │ │
│  │                                                │ │
│  │  Tools: get_financials(ticker)                 │ │
│  │         search_news(query)                     │ │
│  └──────┬──────────────────┬──────────────────────┘ │
└─────────┼──────────────────┼─────────────────────────┘
          │                  │
    ┌─────▼─────┐     ┌─────▼──────┐
    │  yfinance  │     │ Brave API  │
    │  (stocks)  │     │  (news)    │
    └───────────┘     └────────────┘
```

### Agent Nodes

| Node | Role | MCP Tool | Output |
|------|------|----------|--------|
| **A — The Auditor** | Financial analysis | `get_financials(ticker)` | YoY growth, margins, balance sheet health |
| **B — The News Hound** | Sentiment analysis | `search_news(query)` | News themes, sentiment score, risk factors |
| **C — The Synthesizer** | Report generation | — (LLM only) | Structured Markdown investment report |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com/) running locally, with the model named in `OLLAMA_MODEL` pulled (default tag: `gemma4`)

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env and optionally set OLLAMA_BASE_URL / OLLAMA_MODEL
# BRAVE_SEARCH_API_KEY is optional (falls back to yfinance news)

# Start the server
python run.py
# Autoreload is on by default; for a stable process (e.g. behind a tunnel): FINSYNTH_RELOAD=0 python run.py
```

The API will be available at `http://localhost:8000` (or the port in `PORT`).

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment (optional)
cp .env.local.template .env.local

# Start dev server
npm run dev
```

The UI will be available at `http://localhost:3000`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OLLAMA_BASE_URL` | No | Ollama base URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | No | Ollama model name (default: `gemma4`) |
| `OLLAMA_TIMEOUT_SEC` | No | Per-request HTTP timeout in seconds (default: `600`) |
| `CORS_ORIGINS` | No | JSON array of allowed browser origins, e.g. `["http://localhost:3000"]`. Required when the frontend is on another host or HTTPS. |
| `FINANCIAL_API_KEY` | No | Financial Modeling Prep key (future use) |
| `BRAVE_SEARCH_API_KEY` | No | Brave Search API key (falls back to yfinance news) |

## Running the Experiment

The experiment compares three workflow modes (`brief`, `normal`, `extra`) across 15 frozen tickers and measures latency, citation density, and re-synthesis behaviour.

### Prerequisites

The backend virtual environment must be active and dependencies installed (see [Backend Setup](#1-backend-setup)).

### Step 1 — Freeze data (run once)

Pre-fetches financial and news data for all tickers so every mode sees the same ground-truth snapshot:

```bash
cd backend
python -m experiment.freeze_data
```

Output: `experiment/frozen_data/{TICKER}_financials.json`, `{TICKER}_news.json`, and `metadata.json`.

### Step 2 — Run the experiment

```bash
cd backend

# Default: 1 trial, modes brief/normal/extra
python -m experiment.run_experiment

# Quick smoke test (1 trial, 1 mode)
python -m experiment.run_experiment --trials 1 --modes brief

# Full 3-trial run across all modes (writes to a single accumulating file)
python -m experiment.run_experiment --trials 3 --modes brief,normal,extra,extra_force \
    --output experiment/results/full_run.jsonl

# Preview the run plan without executing
python -m experiment.run_experiment --dry-run
```

Results are written to `experiment/results/experiment_<timestamp>.jsonl` (or the path given by `--output`). Interrupted runs can be resumed by passing the same `--output` path — completed entries are skipped automatically.

| Flag | Default | Description |
|------|---------|-------------|
| `--trials N` | `1` | Trials per (ticker, mode) pair |
| `--modes` | `brief,normal,extra` | Comma-separated modes to run (see table below) |
| `--output PATH` | timestamped file | JSONL results file; append-safe for resuming |
| `--dry-run` | — | Print the run plan without executing |

**Available modes:**

| Mode | LLM calls | Description |
|------|-----------|-------------|
| `brief` | 2 | Auditor + synthesizer only; skips news |
| `normal` | 3 | Full pipeline (auditor → news hound → synthesizer), no fact-check |
| `extra` | 3–4 | Full pipeline + fact-check; re-synthesizes only if citation density < 50% |
| `extra_force` | 4 | Full pipeline + fact-check; always re-synthesizes (useful for measuring re-synthesis impact unconditionally) |

### Step 3 — Analyze results

```bash
cd backend
python -m experiment.analyze experiment/results/experiment_<timestamp>.jsonl
```

Prints per-mode summaries (latency, citation density, word count, re-synthesis rate) and writes a flat CSV to the same directory for further analysis in Excel or pandas.

## Tech Stack

- **Backend**: Python, FastAPI, LangGraph, MCP, Pydantic
- **LLM**: Ollama (e.g. Gemma) via LangChain
- **Data**: yfinance (financials), Brave Search (news)
- **Frontend**: Next.js 16, Tailwind CSS, Shadcn UI, React Markdown
- **Streaming**: Server-Sent Events (SSE)
