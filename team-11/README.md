# FinSynth — Financial Synthesis AI Agent
Team 11: Grant Xiao

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
- A [Google Gemini API key](https://aistudio.google.com/apikey)

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Edit .env and add your GEMINI_API_KEY (required)
GEMINI_API_KEY="insert gemini key"
# BRAVE_SEARCH_API_KEY is optional (falls back to yfinance news)

# Start the server
python run.py
```

The API will be available at `http://localhost:8000`.

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

The UI will be available at `http://localhost:3000`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for LLM inference |
| `FINANCIAL_API_KEY` | No | Financial Modeling Prep key (future use) |
| `BRAVE_SEARCH_API_KEY` | No | Brave Search API key (falls back to yfinance news) |

## Tech Stack

- **Backend**: Python, FastAPI, LangGraph, MCP, Pydantic
- **LLM**: Google Gemini (via LangChain)
- **Data**: yfinance (financials), Brave Search (news)
- **Frontend**: Next.js 16, Tailwind CSS, Shadcn UI, React Markdown
- **Streaming**: Server-Sent Events (SSE)


Video Demo Link: https://youtu.be/W_KK-juW-bU 