# UVA Spring 2026 — GenAI Project A Collection

This repository contains student team submissions for the UVA Spring 2026 Generative AI course (Project A). Each subfolder is an independent multi-agent AI project.

---

## Project Overview

| Folder | Project | Team | Overview | Agents |
|---|---|---|---|---|
| `team-00/` | **Obscura** | Wentao Zhou, Guangyi Xu, Jinwei Zhou | Legal compliance agent for vision datasets. Automates GDPR/CCPA de-identification for research datasets before submission to venues like CVPR or NeurIPS. | **Controller** (Claude) orchestrates tool selection; **Critic** runs adversarial re-identification checks to verify that no faces or license plates remain identifiable after blurring. |
| `team-07/` | **MedRAX** | Mengmeng Ma, Kathleen O'Donovan | Medical AI agent for radiology. Assists radiologists with chest X-ray analysis and introduces **ChestAgentBench**, a benchmark of 2,500+ complex medical queries across 8 categories. | A single modular **Radiology Agent** backed by OpenAI, with swappable tools: `ImageVisualizerTool`, `ChestXRayClassifierTool`, `ChestXRaySegmentationTool`, and others. Tools are selected at startup via a config list. |
| `team-9/` | **p2p-trade-bot** | Jacob Huynh, Henry Chen, Haoxuan Luo | Multi-agent prediction-market trading bot that exploits longshot bias in Kalshi NBA markets. Streams live trades via WebSocket, filters by calibration edge, and logs mock fills to SQLite. | **Router** (classifies tickers) → **Bouncer** (longshot filter) → **QuantAgent** + **SentimentAgent** (run in parallel; Claude Haiku) → **LeadAnalyst/Orchestrator** (Kelly sizing; Claude Haiku) → **CriticAgent** (adversarial APPROVE/VETO; Claude Sonnet) → **TradeLogger** |
| `team-10/` | **Agent Alignment Testbed** | Raffi Khondaker | Alignment evaluation framework with domain-specific target agents, an adaptive red team agent (MARSE) using a UCB1 bandit over 6 attack surfaces, and a static baseline evaluator (ABATE). No external agent frameworks. | **Target Agents** (medical, financial, customer service) each run a generative agent memory loop; **MARSE Red Team Agent** adaptively selects attack surfaces via UCB1 bandit; **ABATE** applies a fixed probe bank with an LLM judge. |
| `team-11/` | **FinSynth** | Grant Xiao | Financial synthesis agent that analyzes stock financials and news to produce structured investment reports. Backend is a LangGraph state machine with MCP tool access; frontend is a Bloomberg-style Next.js terminal UI with SSE streaming. | **Auditor** (Node A — fetches financials via `get_financials` MCP tool) → **News Hound** (Node B — fetches news via `search_news` MCP tool) → **Synthesizer** (Node C — LLM only, generates Markdown investment report). All nodes use Google Gemini. |
| `team-6/` | **CardioRAG-CX** | Chuankai Xu, Xinyue Xu, Youke Zhang | Multimodal cardiac diagnostic agent using open-source models. Integrates ECG signals (WFDB, EDF, CSV), DICOM medical imaging, and clinical notes into a unified diagnostic report. Designed for Rivanna HPC with 2–4× A100 GPUs. | **Planner** orchestrates tool calls; **ECGFounderTool** classifies ECG and generates waveforms (CPU); **LingShuTool** analyzes DICOM/MRI images via a vLLM server (GPU 2,3); **Qwen3-VL** (llama.cpp, GPU 0,1) synthesizes all findings into a final report. |
| `team-envcheck/` | **envcheck** | — | AI-powered pre-flight diagnostic tool that detects API breaking changes, dependency conflicts, and version mismatches before runtime. Uses both Claude and Google Gemini. | A single **Diagnostic Agent** that inspects the environment, queries both Anthropic and Google Gemini APIs for reasoning, and reports compatibility issues with suggested fixes. |
| `team-w05/` | **Patient Education Agent** | — | Conversational AI agent that translates medical jargon — conditions, procedures, medications — into plain language for patients. Backed by Claude with NCBI PubMed grounding for literature references. | A single **Patient Education Agent** (Claude) that handles multi-turn conversation, explains medical concepts in plain language, and optionally retrieves supporting literature via the NCBI E-utilities API. |

---

## Quick Start per Team

See each team's `README.md` and `CLAUDE.md` for detailed setup, environment variables, and run commands.

| Folder | Stack | Entry Point |
|---|---|---|
| `team-00/` | Python 3.10+, PyTorch, FastAPI, Claude API | `cd team-00/src && uvicorn server:app --port 8000` |
| `team-07/` | Python 3.8+, pip, OpenAI API | `cd team-07/src && python main.py` |
| `team-9/` | Python 3.x, DuckDB, SQLite, Claude API | `python -m src.pipeline.websocket_client` |
| `team-10/` | Python 3.12, Streamlit, vLLM or OpenAI | `cd team-10/src && .venv/bin/streamlit run app.py` |
| `team-11/` | Python (FastAPI + LangGraph) + Node.js 18 (Next.js) | `cd team-11/backend && python run.py` + `cd team-11/frontend && npm run dev` |
| `team-6/` | Python, Streamlit, vLLM, llama.cpp | `cd team-6/src/cardioagent_demo && streamlit run app.py` |
| `team-envcheck/` | Python 3.12+, uv, Claude + Gemini APIs | `cd team-envcheck && uv run python main.py` |
| `team-w05/` | Node.js 18+, Express, React/Vite, Claude API | `cd team-w05/server && npm run dev` + `cd team-w05/client && npm run dev` |
