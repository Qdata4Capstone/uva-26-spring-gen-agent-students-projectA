# UVA Spring 2026 — GenAI Project A & B Collection

This repository contains student team submissions for the UVA Spring 2026 Generative AI course. Each subfolder is an independent multi-agent AI project. `team-XX/` folders are Project A submissions; `team-XX-ProjB/` folders are Project B submissions.

---

## Project A Overview

| Folder | Project | Team | Overview | Agents |
|---|---|---|---|---|
| `team-00/` | **Obscura** | Wentao Zhou, Guangyi Xu, Jinwei Zhou | Legal compliance agent for vision datasets. Automates GDPR/CCPA de-identification for research datasets before submission to venues like CVPR or NeurIPS. | **Controller** (Claude) orchestrates tool selection; **Critic** runs adversarial re-identification checks to verify that no faces or license plates remain identifiable after blurring. |
| `team-07-ProjB/` (was `team-07/`) | **MedRAX** | Mengmeng Ma, Kathleen O'Donovan | Medical AI agent for radiology. Assists radiologists with chest X-ray analysis and introduces **ChestAgentBench**, a benchmark of 2,500+ complex medical queries across 8 categories. | A single modular **Radiology Agent** backed by OpenAI, with swappable tools: `ImageVisualizerTool`, `ChestXRayClassifierTool`, `ChestXRaySegmentationTool`, and others. Tools are selected at startup via a config list. |
| `team-9/` | **p2p-trade-bot** | Jacob Huynh, Henry Chen, Haoxuan Luo | Multi-agent prediction-market trading bot that exploits longshot bias in Kalshi NBA markets. Streams live trades via WebSocket, filters by calibration edge, and logs mock fills to SQLite. | **Router** (classifies tickers) → **Bouncer** (longshot filter) → **QuantAgent** + **SentimentAgent** (run in parallel; Claude Haiku) → **LeadAnalyst/Orchestrator** (Kelly sizing; Claude Haiku) → **CriticAgent** (adversarial APPROVE/VETO; Claude Sonnet) → **TradeLogger** |
| `team-10/` | **Agent Alignment Testbed** | Raffi Khondaker | Alignment evaluation framework with domain-specific target agents, an adaptive red team agent (MARSE) using a UCB1 bandit over 6 attack surfaces, and a static baseline evaluator (ABATE). No external agent frameworks. | **Target Agents** (medical, financial, customer service) each run a generative agent memory loop; **MARSE Red Team Agent** adaptively selects attack surfaces via UCB1 bandit; **ABATE** applies a fixed probe bank with an LLM judge. |
| `team-11/` | **FinSynth** | Grant Xiao | Financial synthesis agent that analyzes stock financials and news to produce structured investment reports. Backend is a LangGraph state machine with MCP tool access; frontend is a Bloomberg-style Next.js terminal UI with SSE streaming. | **Auditor** (Node A — fetches financials via `get_financials` MCP tool) → **News Hound** (Node B — fetches news via `search_news` MCP tool) → **Synthesizer** (Node C — LLM only, generates Markdown investment report). All nodes use Google Gemini. |
| `team-6/` | **CardioRAG-CX** | Chuankai Xu, Xinyue Xu, Youke Zhang | Multimodal cardiac diagnostic agent using open-source models. Integrates ECG signals (WFDB, EDF, CSV), DICOM medical imaging, and clinical notes into a unified diagnostic report. Designed for Rivanna HPC with 2–4× A100 GPUs. | **Planner** orchestrates tool calls; **ECGFounderTool** classifies ECG and generates waveforms (CPU); **LingShuTool** analyzes DICOM/MRI images via a vLLM server (GPU 2,3); **Qwen3-VL** (llama.cpp, GPU 0,1) synthesizes all findings into a final report. |
| `team-12-envcheck/` (was `team-envcheck/`) | **EnvCheck** | — | AI-powered pre-flight diagnostic tool that detects API breaking changes, dependency conflicts, and version mismatches before runtime. Uses both Claude and Google Gemini. | A single **Diagnostic Agent** that inspects the environment, queries both Anthropic and Google Gemini APIs for reasoning, and reports compatibility issues with suggested fixes. |
| `team-w05/` | **Patient Education Agent** | Tammy Ngo, Matt Juntima, Sebastian Pop | Conversational AI agent that translates medical jargon — conditions, procedures, medications — into plain language for patients. Backed by Claude with NCBI PubMed grounding for literature references. | A single **Patient Education Agent** (Claude) that handles multi-turn conversation, explains medical concepts in plain language, and optionally retrieves supporting literature via the NCBI E-utilities API. |

---

## Project B Overview

| Folder | Project | Team | Overview | Agents |
|---|---|---|---|---|
| `team-00-ProjB/` | **GPU Cluster Monitor** | Wentao Zhou, Guangyi Xu, Jinwei Zhou | Real-time GPU cluster monitoring dashboard that runs as a SLURM job. Tracks GPU/memory usage across multiple clusters, includes an LLM-powered chat agent for job queue analysis and GPU recommendations. | A single **Chat Agent** (OpenAI) with tools: web search for GPU/CUDA compatibility, `analyze_workspace` to scan a project directory and auto-recommend a GPU. |
| `team-07-ProjB/` | **Are Medical Agents Reliable Co-workers?** | Mengmeng Ma, Kathleen O'Donovan | Clinical-reasoning evaluation of multi-tool medical agents (MedRAX framework) over the Eurorad chest X-ray case bank. Measures tool-use recall, communication quality, reasoning coherence, and uncertainty calibration. | **eval_clinical** pipeline with 3-pass tool-grounded self-reflection; **Gradio** and **Chainlit** UI interfaces; evaluation runners across GPT-4o, Qwen3-VL, and baseline comparisons. |
| `team-6-ProjB/` | **CardioAgent (DeepRare-inspired)** | Chuankai Xu, Xinyue Xu, Youke Zhang | Multi-modal clinical decision-support agent for cardiology. Given ECG, chest X-ray, labs, and notes, produces a structured diagnosis with reasoning trace, retrieved supporting cases, and calibrated confidence. | **CardioAgentPlannerV4** (6-phase pipeline): **ECGTool** → **LingShu** (medical LLM) → **Qwen3-VL** (vision-language) → dual-pathway RAG retrieval (vector + keyword) → reflection loop → final synthesis. |
| `team-9-ProjB/` | **p2p-trade-bot (extended)** | Jacob Huynh, Henry Chen, Haoxuan Luo | Extended trading bot adding player prop markets (`KXNBAPTS`) alongside game-winner markets. Both market types run through the full multi-agent pipeline with prop-specific parsing and variance-adjusted Kelly sizing. | Same pipeline as Project A plus **PropAgent** for player prop parsing; both market tracks feed into **LeadAnalyst** → **CriticAgent** → **TradeLogger**. |
| `team-10/` | **Agent Alignment Testbed (extended)** | Raffi Khondaker | Extended alignment evaluation framework adding an ML-based violation detector, a unified CLI entrypoint, and a combined Streamlit UI. Supports MARSE adaptive red-team and ABATE static baseline across 4 target agent domains. | **Target Agents** (medical, weak_medical, financial, customer_service); **MARSE Red Team Agent** (adaptive UCB1 attacker); **ABATE** (fixed probe bank); **ML Violation Detector** (sentence-transformer + scikit-learn, trained via `train_marse_ml_judge.py`). |
| `team-11-ProjB/` | **FinSynth (extended)** | Grant Xiao | Extended financial synthesis agent with the same LangGraph + MCP architecture. | **Auditor** → **News Hound** → **Synthesizer** (same as Project A; see `team-11-ProjB/` for updates). |
| `team-w05 (Project B)/` | **Mental Health Bot** | Tammy Ngo, Matt Juntima, Sebastian Pop | Compassionate AI mental health support chatbot backed by peer-reviewed PubMed research. Built on MCP with LangGraph orchestration. Includes a crisis pre-filter that bypasses the LLM entirely for self-harm keywords. | **MCP server** (stdio) exposes PubMed search tools; **LangGraph state graph** orchestrates the Claude tool-use loop; **Claude** autonomously decides when to retrieve research. |

---

## Quick Start per Team

See each team's `README.md` and `CLAUDE.md` for detailed setup, environment variables, and run commands.

### Project A

| Folder | Stack | Entry Point |
|---|---|---|
| `team-00/` | Python 3.10+, PyTorch, FastAPI, Claude API | `cd team-00/src && uvicorn server:app --port 8000` |
| `team-07-ProjB/` | Python 3.8+, pip, OpenAI API | `cd team-07-ProjB/src && python main.py` |
| `team-9/` | Python 3.x, DuckDB, SQLite, Claude API | `cd team-9 && python -m src.pipeline.websocket_client` |
| `team-10/` | Python 3.12, Streamlit, vLLM or OpenAI | `cd team-10/src && .venv/bin/streamlit run app.py` |
| `team-11/` | Python (FastAPI + LangGraph) + Node.js 18 (Next.js) | `cd team-11/backend && python run.py` + `cd team-11/frontend && npm run dev` |
| `team-6/` | Python, Streamlit, vLLM, llama.cpp | `cd team-6/src/cardioagent_demo && streamlit run app.py` |
| `team-12-envcheck/` | Python 3.12+, uv, Claude + Gemini APIs | `cd team-12-envcheck && uv run python main.py` |
| `team-w05/` | Node.js 18+, Express, React/Vite, Claude API | `cd team-w05/server && npm run dev` + `cd team-w05/client && npm run dev` |

### Project B

| Folder | Stack | Entry Point |
|---|---|---|
| `team-00-ProjB/` | Python 3.10+, Flask, OpenAI API, SLURM | `cd team-00-ProjB/src && bash start.sh` |
| `team-07-ProjB/` | Python 3.8+, OpenAI API, Gradio, Chainlit | `cd team-07-ProjB/src && python main.py` |
| `team-6-ProjB/` | Python, vLLM, llama.cpp, Streamlit | `cd team-6-ProjB/src && streamlit run apps/app.py` |
| `team-9-ProjB/` | Python 3.x, DuckDB, SQLite, Claude API | `cd team-9-ProjB && python -m src.pipeline.websocket_client` |
| `team-10/` | Python 3.10+, Streamlit, OpenAI API | `cd team-10 && streamlit run src/app_redteam_combined.py --server.port 8503` |
| `team-11-ProjB/` | Python (FastAPI + LangGraph) + Node.js 18 | `cd team-11-ProjB/backend && python run.py` + `cd team-11-ProjB/frontend && npm run dev` |
| `team-w05 (Project B)/` | Node.js, Express, React, Claude API, LangGraph | `cd "team-w05 (Project B)/server" && npm run dev` |
