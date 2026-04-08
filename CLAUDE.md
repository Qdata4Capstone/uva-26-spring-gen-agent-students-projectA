# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a course project collection for UVA Spring 2026 GenAI — each `team-XX/` subdirectory is an independent multi-agent AI project submitted by a student team. There is no shared build system or top-level test runner; each team manages its own dependencies, environment, and runtime.

## Team Projects Summary

| Directory | Project | Stack |
|-----------|---------|-------|
| `team-00/` | **Obscura** — Legal compliance agent for vision datasets (GDPR/CCPA de-identification) | Python 3.10+, PyTorch, FastAPI, Claude API |
| `team-07/` | **MedRAX** — Medical AI agent for radiology with ChestAgentBench | Python, pip, OpenAI API |
| `team-9/` | **p2p-trade-bot** — Multi-agent Kalshi NBA prediction-market trading bot | Python, DuckDB, SQLite, Claude Haiku/Sonnet |
| `team-10/` | **Agent Alignment Testbed** — Adaptive red-team evaluation with UCB1 bandit (MARSE) | Python 3.12, Streamlit, vLLM or OpenAI |
| `team-11/` | **FinSynth** — Financial synthesis agent with LangGraph + MCP | Python (FastAPI + LangGraph) + Next.js 18 |
| `team-6/` | **CardioRAG-CX** — Multimodal cardiac diagnostic agent (ECG + DICOM) | Python, Streamlit, vLLM, llama.cpp |
| `team-envcheck/` | **envcheck** — AI pre-flight diagnostic for environment compatibility | Python 3.12+, uv, Anthropic + Google Gemini |
| `team-w05/` | **Patient Education Agent** — Conversational medical-jargon explainer | Node.js 18+, React/Vite, Express, Claude API |

---

## Per-Team Dev Commands

### team-00 (Obscura)
```bash
cd team-00
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
# Web UI
cd src && uvicorn server:app --host 0.0.0.0 --port 8000
# CLI demo
cd src && python run_demo.py --input /path/to/dataset --target "CVPR 2026 submission"
# Evaluation
cd src && python evaluate.py --images /path/to/frames --output results.csv
```

### team-07 (MedRAX)
```bash
cd team-07/src
pip install -e .
# Set model_dir and OPENAI_API_KEY in .env, then:
python main.py
# Benchmarks (from team-07/src/experiments/)
python benchmark_gpt4o.py
python analyze_axes.py results/<logfile>.json ../benchmark/questions/ --model gpt4o
```

### team-9 (p2p-trade-bot)
```bash
cd team-9
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add ANTHROPIC_API_KEY, KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH
python mock_database_setup.py   # generate mock parquet data
# Run bot
python -m src.pipeline.websocket_client
# Settle pending trades
python -m src.settle
# Report
python -m src.report_trades
# Tests (no API keys needed for most)
pytest tests/ --ignore=tests/test_websocket.py -v
# WebSocket integration test (requires .env credentials)
pytest tests/test_websocket.py -v
# Live pipeline test (requires ANTHROPIC_API_KEY + real parquet)
python tests/test_pipeline.py --live
```

### team-10 (Agent Alignment Testbed)
```bash
cd team-10/src
python3.12 -m venv .venv && source .venv/bin/activate
pip install streamlit openai matplotlib vllm
# Create .env with OPENAI_API_KEY
# Edit config.py to set TARGET_LLM and RED_TEAM_LLM
# Interactive frontend
.venv/bin/streamlit run app.py --server.port 8501
# Red team frontend
.venv/bin/streamlit run streamlit_app.py --server.port 8503
# CLI red team experiment
.venv/bin/python run_experiment_cli.py medical 10 false
# Static baseline
.venv/bin/python run_baseline_cli.py
```

### team-11 (FinSynth)
```bash
# Backend
cd team-11/backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Edit .env: GEMINI_API_KEY required, BRAVE_SEARCH_API_KEY optional
python run.py   # API at http://localhost:8000

# Frontend
cd team-11/frontend
npm install
npm run dev    # UI at http://localhost:3000
```

### team-6 (CardioRAG-CX)
```bash
cd team-6/src/cardioagent_demo
pip install -r requirements.txt
# Start model servers (requires 2× A100 or similar):
# Terminal 1: Qwen3-VL on GPU 0,1 via llama.cpp server (port 8000)
# Terminal 2: LingShu-8B via vllm (port 8001)
streamlit run app.py --server.port 8501
# Or all-in-one:
chmod +x run.sh && ./run.sh
```
Works in fallback mode (CPU, ECGFounder only) when GPU model servers are not running.

### team-envcheck
```bash
cd team-envcheck
uv sync           # installs deps via uv
uv run python main.py
uv run pytest     # run tests
uv run ruff check # lint
```

### team-w05 (Patient Education Agent)
```bash
# Backend
cd team-w05/server && npm install
# Create .env with ANTHROPIC_API_KEY, NCBI_EMAIL, NCBI_API_KEY
npm run dev    # server at http://localhost:3001

# Frontend
cd team-w05/client && npm install
npm run dev    # client at http://localhost:5173
```

---

## Architecture Patterns Across Teams

**Agentic pipelines:** Most teams use a multi-stage pipeline with specialized agents (planner/orchestrator → specialist tools → critic/synthesizer). team-9 is the most explicit: Router → Bouncer → QuantAgent + SentimentAgent (parallel) → LeadAnalyst → CriticAgent → TradeLogger.

**LLM choices:** Claude (team-00, team-9, team-envcheck, team-w05), Gemini (team-11), OpenAI/vLLM (team-07, team-10), open-source via vLLM/llama.cpp (team-6, team-10).

**Frontend patterns:** FastAPI + Streamlit (team-00, team-6, team-10), FastAPI + Next.js with SSE streaming (team-11), Express + React/Vite (team-w05).

**Data patterns:** DuckDB over Parquet for historical analytics (team-9), SQLite for operational logs (team-9), WFDB/DICOM for medical signals (team-6, team-07).

**Environment variables:** All Python teams use `.env` files. Most teams require at minimum one LLM API key. team-9 additionally needs Kalshi RSA credentials. Never commit `.env` files.
