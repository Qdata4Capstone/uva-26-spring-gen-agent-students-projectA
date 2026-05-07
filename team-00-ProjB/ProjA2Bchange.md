# Project A to Project B Changes — Team 00

## What Project A Was

**Obscura** (Project A) is a legal-compliance agent for vision datasets. Given a dataset directory and a natural-language compliance target (e.g., "I want to submit to CVPR 2026"), the system audits the dataset against privacy regulations (GDPR, CCPA, conference open-science policies), applies face and license-plate de-identification using EgoBlur Gen2 or an OpenCV fallback, redacts PII from accompanying text files, verifies anonymization quality with an adversarial re-identification critic, and produces a structured compliance report.

**Stack:** Python 3.10+, PyTorch 2.2+, FastAPI, Claude API (Anthropic), OpenCV, EgoBlur Gen2 `.jit` weights.

---

## Project B: What It Is

**GPU Cluster Monitor** (Project B) is a real-time HPC monitoring dashboard that runs as a SLURM job. It collects GPU/CPU/memory availability across multiple SLURM clusters (local + remote via SSH), stores time-series snapshots in SQLite, and serves an interactive browser dashboard with a built-in chat agent powered by OpenAI GPT-4o.

This is a **complete application pivot**: the domain, technology stack, architecture, and use case are entirely different from Project A.

---

## 1. Change in Problem Domain and Purpose

| Aspect | Project A (Obscura) | Project B (GPU Cluster Monitor) |
|--------|---------------------|---------------------------------|
| Domain | Legal/privacy compliance for CV datasets | HPC infrastructure monitoring |
| Users | Computer vision researchers | Cluster users submitting GPU jobs |
| Core task | De-identify images, audit regulations | Monitor GPU availability, assist job submission |
| LLM role | Intent parsing, clip classification, compliance summarization | Interactive SLURM expert chat agent |
| LLM provider | Anthropic Claude (Haiku + Sonnet) | OpenAI GPT-4o |
| Input | Image/video dataset directory + natural-language compliance goal | Live SLURM cluster data + user chat messages |
| Output | De-identified dataset + compliance report JSON | Real-time web dashboard + chat responses + SLURM job submissions |

---

## 2. New Agent Architecture

**Project A** used an iterative closed-loop pipeline in `agent.py`:
```
LLM Controller (parse_intent) → EgoBlur/FaceBlur → EXIF Strip → Re-ID Critic → Report
```
The agent ran up to 3 self-correction retries when the re-ID critic flagged a clip as failing. All code was structured across `agent.py`, `controller.py`, `critic.py`, and the `tools/` directory.

**Project B** uses a **tool-augmented chat loop** inside a single Flask endpoint (`/api/chat` in `monitor.py`):
```
User message → collect live cluster state → GPT-4o (up to 5 iterations) → dispatch tool(s) → return response
```

The agent loop is driven by OpenAI function calling with up to five iterations before a forced text-only final call. Mutating tools (`submit_job`, `cancel_job`) never execute immediately — they emit a `pending_user_approval` token that is stored in `PENDING_ACTIONS` and only executed after the user clicks **Approve** in the dashboard UI.

Key new agent behaviors not present in Project A:
- **Two-stage confirmation flow** for destructive actions (`/api/agent/confirm`, `/api/agent/reject`).
- **Audit log** (`agent_actions.log`) for every tool proposal, execution, and rejection.
- **Live cluster context injection**: before every LLM call, the current state of both clusters is summarized as text and prepended to the user message.
- **Automatic job-context lookup**: any 5+ digit number in the user message triggers `scontrol show job` + `sprio` + `squeue` queries and appends the result as context.

---

## 3. New Tools

Project A had five internal tools in `src/agent/tools/`:
- `egoblur_tool.py` — EgoBlur Gen2 face/plate blurring
- `face_blur.py` — OpenCV fallback blur
- `pii_redactor.py` — regex PII redaction for text
- `knowledge.py` — compliance knowledge base (GDPR, CCPA, conference policies)
- `file_manager.py` — EXIF stripping, report generation

Project B defines six LLM-callable tools in the `TOOLS` list in `monitor.py` (lines 1679–1838):

| Tool | Description |
|------|-------------|
| `web_search` | DuckDuckGo search via `ddgs`; returns title/URL/snippet results |
| `query_history` | Queries SQLite history for avg/min/max free GPUs by cluster and GPU type |
| `submit_job` | Builds and proposes a sanitized `sbatch` script (requires user approval) |
| `cancel_job` | Proposes a `scancel` call for a specific job ID (requires user approval) |
| `analyze_workspace` | Walks a local filesystem path, extracts GPU-relevant signals (libs, model hints, parameter sizes, training configs) |
| `analyze_repository` | Fetches a public GitHub repo via the GitHub API and applies the same GPU-signal extraction |

None of these tools existed in Project A. The most novel are `analyze_workspace` and `analyze_repository`, which perform static analysis to recommend specific SLURM nodes and `--gres` flags. The `submit_job` tool includes a pre-flight `find_best_target()` call that validates and corrects partition names, GPU types, and node selection against live cluster data before constructing the sbatch script.

---

## 4. New UI

**Project A** used a plain HTML/CSS/JS single-page app in `src/static/index.html` served by FastAPI. It streamed the agent's reasoning steps via Server-Sent Events (SSE).

**Project B** replaced this with a much richer single-file HTML dashboard embedded as a Python string (`HTML_TEMPLATE`) directly in `monitor.py`. Key UI features added:

- **Dark terminal-style theme** using CSS custom properties (`--bg: #0f1117`, etc.), monospace font stack.
- **Tabbed interface** (Dashboard, History, Heatmap) with animated tab badges showing free GPU counts.
- **Real-time node table** with per-node rows showing GPU type, free/total GPUs, CPU load, memory usage bar, node state badge (IDLE/MIXED/ALLOCATED/RESERVED/DRAINING), and an inline running-jobs list.
- **Filter and sort controls** to filter nodes by cluster, GPU type, state, and free GPU count; column-header sorting.
- **GPU utilization bar charts** per node using Chart.js (loaded from CDN).
- **History tab**: time-series line chart of free GPUs over the last 1–24 hours.
- **Heatmap tab**: 7×24 day-of-week × hour-of-day grid showing average free GPUs.
- **Floating chat sidebar** (520 px wide) with a toggle button, message history, typing indicator, and tool-call cards with Approve/Reject buttons for pending SLURM actions.
- Auto-refresh every 5 seconds (`setInterval(fetchAll, 5000)`).

Project A had no such interactivity, charts, or real-time polling.

---

## 5. New Infrastructure

**Project A** ran as a standalone FastAPI process started manually with `uvicorn`.

**Project B** is designed to run **inside SLURM itself** as a submitted batch job, with the following infrastructure components new to Project B:

### 5.1 SLURM Job Management Scripts
- `job.sh` — SLURM batch script that starts the Flask app, writes `state.json`, and immediately submits the _next_ job (`sbatch --dependency=afterany:$SLURM_JOB_ID`) to form a self-renewing chain.
- `start.sh` — submits `job.sh`, polls SLURM until the job reaches RUNNING state, waits for the Cloudflare URL to appear in `state.json`, and prints access instructions.
- `stop.sh` — writes a `.stop_requested` sentinel, cancels all `lemon`-named jobs for the current user, cancels any pending next-job from `.next_job_id`, and force-kills the port if needed.
- `status.sh` — reads `state.json` and prints the current job state, local URL, and public URL.

None of these existed in Project A.

### 5.2 Cloudflare Tunnel
`run_cloudflare_tunnel_forever(port)` in `monitor.py` spawns `cloudflared tunnel --url http://127.0.0.1:<port>` in a daemon thread, parses the `*.trycloudflare.com` URL from stdout, writes it to `state.json` and `~/public_url`, and auto-restarts on crash with 10-second delays. Project A had no public-URL mechanism.

### 5.3 SQLite Historical Storage
`db_init()`, `db_insert_snapshot()`, and related functions store per-cluster and per-node snapshots every 60 seconds in `data/history.db` (two tables: `snapshots` and `node_snapshots`). Snapshots older than 30 days are pruned on each write. Three history API endpoints are exposed:
- `GET /api/history/<cluster_id>` — time-series data
- `GET /api/history/<cluster_id>/heatmap` — 7×24 heatmap grid

Project A had no persistent state.

### 5.4 Conda Environment
Project A used `pip` + `requirements.txt` (anthropic, fastapi, uvicorn, opencv, torch). Project B uses a conda `environment.yml` with only `flask`, `openai`, `ddgs`, and `requests` — a much lighter dependency footprint that works on headless HPC login nodes without GPU drivers.

### 5.5 Configuration File
Project A had no configuration file; all settings were environment variables or hardcoded paths. Project B introduces `config.json` (templated as `config.json.example`) for per-cluster SSH commands, conda path, port, SLURM partition, job name, and OpenAI model. The config is loaded at startup and drives the `CLUSTERS` dict used throughout.

### 5.6 State File
`state.json` is written by `job.sh` on startup and updated by `monitor.py` when the Cloudflare URL arrives. It contains `job_id`, `node`, `port`, `started`, `url`, and `public_url`. All shell scripts use this file to find the running dashboard.

---

## 6. New Evaluation

**Project A** had `src/evaluate.py`: a standalone CLI script that ran the de-identification pipeline on a set of image frames and reported face removal rate, re-identification rate, and SSIM per frame. It was entirely independent of the agent loop.

**Project B** has `src/evaluation.py`: a three-mode CLI evaluation suite for the dashboard itself:

| Mode | What it tests |
|------|---------------|
| `correctness` | Compares dashboard GPU/CPU counts against an independent `sinfo` query with configurable tolerance |
| `latency` | Measures `/api/data/<cluster>` p50/p95/max response times; target is p95 < 5 seconds |
| `reliability` | Samples the SLURM auto-renewal chain health and Cloudflare tunnel availability over time; supports an `--active-tunnel-test` flag that kills `cloudflared` and measures recovery time |

The correctness test uses a second, independent SLURM parser to avoid testing the parser against itself. The latency test also measures the raw `sinfo` floor to isolate dashboard overhead from network latency.

---

## 7. Data Model Changes

**Project A** data artifacts:
- `data/markdown_data/` — 5 sample PII documents for redaction testing
- `src/agent/knowledge/regulations.json` — structured GDPR, CCPA, and conference compliance rules
- Per-session compliance report JSON written to `output/`

**Project B** data artifacts:
- `data/history.db` — SQLite database with `snapshots` and `node_snapshots` tables
- `logs/agent_actions.log` — append-only JSONL audit log of every LLM tool call
- `logs/monitor_<jobid>.log` / `logs/monitor_<jobid>.err` — SLURM job stdout/stderr
- `state.json` — live job state (written by SLURM job, read by all shell scripts)
- `~/.next_job_id` — job ID of the pending renewal job

No image, video, or regulatory text data is used in Project B.

---

## 8. Known Issues Carried Forward (from IMPROVE.md)

The Project B IMPROVE.md documents several bugs not fixed in the current submission:

- **Hardcoded SSH username** in `evaluation.py:54`: `tsx4zn@login.hpc.virginia.edu` — not loaded from `config.json`.
- **Hardcoded job name `lemon`** in `start.sh:16`, `stop.sh:16`, `evaluation.py:343`, and `job.sh:2` — changing `slurm_job_name` in `config.json` has no effect.
- **Missing OpenAI API key validation** (`monitor.py:590-598`): client created even if key is empty; fails at first API call with 401.
- **XSS via innerHTML** (`monitor.py:2600`): SLURM node names injected without escaping.
- **Hardcoded port 8080 in stop.sh** (`stop.sh:35`): always kills port 8080 regardless of configured port.

---

## Summary Comparison Table

| Dimension | Project A (Obscura) | Project B (GPU Cluster Monitor) |
|-----------|---------------------|---------------------------------|
| **Problem domain** | Privacy/legal compliance for vision datasets | Real-time HPC GPU cluster monitoring |
| **LLM provider** | Anthropic Claude (Haiku + Sonnet) | OpenAI GPT-4o |
| **Agent pattern** | Iterative pipeline with self-correction loop | Tool-augmented chat loop with human-in-the-loop approval |
| **Agent entry point** | `src/agent/agent.py` (`LegalComplianceAgent.run()`) | `monitor.py` `api_chat()` Flask endpoint |
| **Tools** | 5 internal tools (egoblur, face_blur, pii_redactor, knowledge, file_manager) | 6 LLM-callable tools (web_search, query_history, submit_job, cancel_job, analyze_workspace, analyze_repository) |
| **Web framework** | FastAPI + Uvicorn | Flask (threaded) |
| **Frontend** | Separate `static/index.html` with SSE streaming | Single `HTML_TEMPLATE` string in `monitor.py` with Chart.js, tabbed UI, real-time polling |
| **Persistent storage** | None (reports written to output dir per session) | SQLite (`history.db`) with 30-day retention |
| **Deployment model** | `uvicorn server:app` — runs on any machine | SLURM batch job with self-renewing chain + Cloudflare public URL |
| **Infrastructure scripts** | None | `job.sh`, `start.sh`, `stop.sh`, `status.sh` |
| **Configuration** | Environment variables only | `config.json` (multi-cluster SSH, port, SLURM settings, model) |
| **Evaluation** | `evaluate.py`: face removal rate, re-ID rate, SSIM | `evaluation.py`: correctness (SLURM diff), latency (p95), reliability (renewal chain + tunnel recovery) |
| **Dependencies** | torch, fastapi, anthropic, opencv, scikit-image | flask, openai, ddgs, requests |
| **Lines of code (main file)** | ~378 (agent.py) + ~264 (controller.py) + ~253 (critic.py) | ~3113 (monitor.py, all-in-one) |
| **Multi-cluster support** | No | Yes (local CS + remote Rivanna/HPC via SSH) |
| **Human approval gate** | No | Yes (Approve/Reject buttons for submit_job and cancel_job) |
| **Audit logging** | No | Yes (`agent_actions.log` for all tool events) |
