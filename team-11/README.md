# FinSynth — Financial Synthesis AI Agent

**Team:** Grant Xiao (Team 11)

---

## Introduction

Retail investors often struggle to synthesize raw financial data with real-time news into actionable investment theses — a task that analysts at large firms spend hours on. **FinSynth** automates this workflow using a three-node multi-agent pipeline: one agent specializes in financial statement analysis, another in news sentiment, and a third synthesizes both into a structured Markdown investment report.

The system is built on **LangGraph** (state machine orchestration), **Model Context Protocol (MCP)** (tool access), **FastAPI** (streaming backend), and **Next.js** (Bloomberg-style terminal UI).

---

## Overall Function

Given a stock ticker, FinSynth:

1. **Auditor (Node A):** Fetches financial statements via `yfinance` and analyzes YoY revenue growth, margins, and balance sheet health
2. **News Hound (Node B):** Searches recent news via Brave Search API (or yfinance news fallback) and extracts sentiment themes and risk factors
3. **Synthesizer (Node C):** Combines both analyses into a structured Markdown investment report using Google Gemini

Each node's reasoning is streamed to the frontend as Server-Sent Events, producing a live "thinking log" alongside the final report. Tools are accessed through an MCP server (stdio transport) spawned as a subprocess by the FastAPI backend.

---

## Code Structure

```
team-11/
├── src/
│   ├── backend/                    # FastAPI + LangGraph backend
│   │   ├── app/                    # Application package
│   │   │   ├── __init__.py
│   │   │   ├── config.py           # Environment config (API keys, model names)
│   │   │   ├── graph/              # LangGraph state machine definition
│   │   │   │   └── ...             # Node definitions (auditor, news_hound, synthesizer)
│   │   │   ├── main.py             # FastAPI app + /api/analyze SSE endpoint
│   │   │   ├── mcp_server.py       # MCP server (stdio transport)
│   │   │   │                       # Exposes: get_financials(ticker), search_news(query)
│   │   │   └── schemas.py          # Pydantic request/response models
│   │   ├── run.py                  # Backend entry point (starts FastAPI server)
│   │   └── test_get_financials.py  # Manual integration test for financial data tool
│   └── frontend/                   # Next.js 16 frontend
│       ├── app/                    # Next.js app router (pages + layouts)
│       ├── components/             # React components (search bar, thinking log, report viewer)
│       ├── lib/                    # Utility functions (SSE client, formatting)
│       ├── public/                 # Static assets
│       ├── next.config.ts
│       └── package.json
└── README.md
```

**Data flow:** `Frontend → POST /api/analyze → FastAPI → LangGraph state machine → MCP server subprocess → yfinance / Brave Search → SSE stream back to frontend`

The MCP server is not run independently — it is spawned by FastAPI as a subprocess using stdio transport. LangGraph nodes call MCP tools synchronously and stream intermediate results back via the SSE endpoint.

---

## Installation

### Backend

**Requirements:** Python 3.11+

```bash
cd src/backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create `.env` in `src/backend/`:
```
GEMINI_API_KEY="your-gemini-key-here"
BRAVE_SEARCH_API_KEY="your-brave-key-here"   # optional
```

### Frontend

**Requirements:** Node.js 18+

```bash
cd src/frontend
npm install
```

---

## How to Run

### Backend

```bash
cd src/backend
source venv/bin/activate
python run.py
# API available at http://localhost:8000
```

### Frontend

```bash
cd src/frontend
npm run dev
# UI available at http://localhost:3000
```

Open `http://localhost:3000`, enter a stock ticker (e.g., `AAPL`, `NVDA`), and watch the agent think in real time.

### Manual tool test

```bash
cd src/backend
source venv/bin/activate
python test_get_financials.py
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | **Yes** | Google Gemini API key — used by all three LangGraph nodes |
| `BRAVE_SEARCH_API_KEY` | No | Brave Search API; falls back to yfinance news if absent |
| `FINANCIAL_API_KEY` | No | Financial Modeling Prep (reserved for future use) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, LangGraph, MCP (stdio), Pydantic |
| LLM | Google Gemini via LangChain |
| Financial data | yfinance |
| News | Brave Search API (fallback: yfinance news) |
| Frontend | Next.js 16, Tailwind CSS, Shadcn UI, React Markdown |
| Streaming | Server-Sent Events (SSE) |

---

## References

- **Video Demo:** https://youtu.be/W_KK-juW-bU
- [LangGraph](https://github.com/langchain-ai/langgraph) — state machine orchestration
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io) — tool access layer
- [yfinance](https://github.com/ranaroussi/yfinance) — financial data
- [Brave Search API](https://brave.com/search/api/) — news retrieval
