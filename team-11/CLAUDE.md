# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: FinSynth — Financial Synthesis AI Agent

A multi-agent AI system that analyzes stock financials and news to produce structured investment reports. The backend is a LangGraph state machine exposing tools via MCP; the frontend is a Bloomberg-style Next.js terminal UI that streams agent reasoning via SSE.

**Team:** Grant Xiao (Team 11)

## Setup & Running

### Backend (Python + FastAPI + LangGraph)

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Edit .env:
#   GEMINI_API_KEY="..."           (required)
#   BRAVE_SEARCH_API_KEY="..."     (optional — falls back to yfinance news)
python run.py   # API at http://localhost:8000
```

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev    # UI at http://localhost:3000
```

## Architecture

```
Next.js (port 3000)
    │  POST /api/analyze → SSE stream
FastAPI backend (port 8000)
    │
LangGraph state machine
    ├── Node A: The Auditor      → get_financials(ticker) via MCP
    ├── Node B: The News Hound   → search_news(query) via MCP
    └── Node C: The Synthesizer  → LLM only, no tools
    │
MCP Server (stdio transport)
    ├── get_financials  → yfinance
    └── search_news     → Brave Search API (or yfinance news fallback)
```

The three LangGraph nodes run sequentially: Auditor fetches financials (YoY growth, margins, balance sheet), News Hound fetches news (sentiment, risk factors), Synthesizer combines both into a structured Markdown investment report. Each node's reasoning is streamed to the frontend as SSE events.

The MCP server runs as a subprocess (stdio transport) — it is spawned by the FastAPI app, not run independently.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key (LLM for all three nodes) |
| `BRAVE_SEARCH_API_KEY` | No | Brave Search; falls back to yfinance news if absent |
| `FINANCIAL_API_KEY` | No | Financial Modeling Prep (reserved for future use) |

## Tech Stack

- **Backend:** Python, FastAPI, LangGraph, MCP (stdio), Pydantic
- **LLM:** Google Gemini via LangChain
- **Data:** yfinance (financials), Brave Search (news)
- **Frontend:** Next.js 16, Tailwind CSS, Shadcn UI, React Markdown
- **Streaming:** Server-Sent Events (SSE)
