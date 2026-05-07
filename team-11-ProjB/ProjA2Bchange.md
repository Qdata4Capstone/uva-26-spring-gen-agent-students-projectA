# Project A to Project B Changes — Team 11 (FinSynth)

## What Was Project A

**FinSynth** (Project A) is a multi-agent financial synthesis system. Given a stock ticker, a three-node LangGraph pipeline fetches financials and news via MCP tools and synthesizes them into a structured Markdown investment report. The backend is a FastAPI app backed by Google Gemini via LangChain; the frontend is a Bloomberg-style Next.js terminal UI that streams each agent's reasoning via SSE.

**Graph topology (Project A):**
```
START → Auditor (Node A) → News Hound (Node B) → Synthesizer (Node C) → END
```

The Synthesizer's output was the final report with no further verification. There was no mechanism to detect or correct LLM hallucinations — the report was emitted directly to the frontend.

---

## 1. New LLM Backend: Google Gemini → Ollama (local)

**Project A:** All three nodes used `ChatGoogleGenerativeAI` (Gemini) via `langchain-google-genai`. The required credential was `GEMINI_API_KEY` in `.env`. The model was configured via `settings.gemini_model`.

**Project B:** The LLM backend was replaced with a **local Ollama instance** (`ChatOllama` via `langchain-ollama`). Settings changed accordingly:

| Setting | Project A | Project B |
|---|---|---|
| LLM class | `ChatGoogleGenerativeAI` | `ChatOllama` |
| Config key | `GEMINI_API_KEY`, `gemini_model` | `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SEC` |
| Default model | Gemini (remote) | `gemma4` (local) |
| API key required | Yes (Gemini) | No (local Ollama) |

The Auditor's and News Hound's log messages were also updated accordingly (e.g., "Analyzing with Gemini…" → "Analyzing with Ollama…").

---

## 2. Extended Graph: Hallucination-Mitigation Pipeline

**Project A:** The workflow was a fixed 3-node linear chain. The Synthesizer emitted its output directly as `report`.

**Project B:** The workflow was extended to a **5-or-6-node graph with conditional routing**:

```
START → Auditor → News Hound → Synthesizer → Fact Checker ─┬─ (density ≥ 50%) → END
                                                             └─ (density < 50%) → Re-Synthesizer → Post Fact Checker → END
```

Three new nodes were added:

| Node | Project A | Project B |
|---|---|---|
| **Fact Checker** (Node D) | Not present | Pure-Python numerical cross-reference — no LLM call |
| **Re-Synthesizer** (Node E) | Not present | LLM rewrite grounded in verified source figures only |
| **Post Fact Checker** (Node F) | Not present | Identical fact-check re-run; appends before/after audit trail |

The Synthesizer's output is now stored as `draft_report` (not `report`). The Fact Checker promotes it to `report` (with a disclaimer) or hands off to the Re-Synthesizer if citation density is too low.

---

## 3. New Node: Fact Checker (Node D — no LLM)

**Project A:** No fact-checking existed. The Synthesizer's output was the final report.

**Project B:** The Fact Checker (`make_fact_checker_node`) performs a pure-Python audit:

1. Flattens all numeric values from `financial_data` (the raw yfinance ground truth) into a lookup set — values are indexed at raw scale, millions, and billions so the LLM's natural language representations (e.g., "$2.1 billion") still match.
2. Extracts every numerical claim from the draft report via `_CLAIM_RE` (a regex covering optional `$`, thousands-separated numbers, scale words B/M/T/K, and `%`). Years (1900–2100) and bare list-marker integers are filtered out.
3. Cross-references each claim against the lookup within **±5% relative tolerance** (`_MATCH_TOLERANCE = 0.05`).
4. Computes **citation density** = verified claims / total claims.
5. Applies a three-tier disclaimer based on density:
   - **green** (≥ 80%): high citation density notice
   - **amber** (50–79%): data verification warning
   - **red** (< 50%): high uncertainty warning
6. Routes to Re-Synthesizer if density < 50% (in `extra`/`extra_force` modes), otherwise appends the disclaimer and finalises the report.

The fact-checking logic is extracted into `_run_fact_check()` so the Post Fact Checker can share identical logic without duplication.

---

## 4. New Node: Re-Synthesizer (Node E)

**Project A:** Not present.

**Project B:** `make_resynth_node` is called only when the Fact Checker sets `resynthesis_needed = True`. It:

- Receives the list of unverified claims from the Fact Checker.
- Sends the LLM a strict rewrite prompt that provides the raw `financial_data` JSON as ground truth, explicitly lists the flagged unverified figures, and instructs the model to use only source-present numbers (using qualitative language for any absent metric).
- Appends `"[Data-Verified Rewrite]"` to the Executive Summary.
- Returns the rewritten report as `report` (to be further validated by the Post Fact Checker).

---

## 5. New Node: Post Fact Checker (Node F — no LLM)

**Project A:** Not present.

**Project B:** `make_post_fact_checker_node` re-runs the identical `_run_fact_check()` algorithm on the Re-Synthesizer's output and appends a **before/after Hallucination Audit Trail** table to the final report:

| Metric | Draft Report | Verified Rewrite |
|:---|:---:|:---:|
| Citation Density | *before* | **after** |
| Verified Claims | *n/N* | **n/N** |
| Unverified Claims | *count* | **count** |
| Integrity Tier | *tier* | **tier** |
| Density Improvement | — | **+X%** |

---

## 6. New Workflow Modes

**Project A:** `run_analysis(ticker)` accepted only a ticker. There was one fixed execution path.

**Project B:** `run_analysis(ticker, workflow_mode)` accepts a `workflow_mode` parameter. Four modes are defined:

| Mode | Auditor LLM | Synthesizer input | Resynthesis triggered |
|---|---|---|---|
| `normal` | Full LLM analysis | Auditor + News Hound analyses | Never |
| `brief` | Skipped — raw data passed directly | Raw financial JSON only | Never |
| `extra` | Full LLM analysis | Auditor + News Hound analyses | If density < 50% |
| `extra_force` | Full LLM analysis | Auditor + News Hound analyses | Always |

`extra_force` was added specifically for controlled experiments: it forces resynthesis even on high-quality drafts to measure the improvement delta. The `AnalyzeRequest` schema and `POST /api/analyze` endpoint were updated to accept and forward `workflow_mode`.

---

## 7. Extended AgentState

**Project A:** `AgentState` had 6 fields: `ticker`, `financial_data`, `auditor_analysis`, `news_data`, `news_analysis`, `report`.

**Project B:** `AgentState` was extended with 7 new fields:

| Field | Purpose |
|---|---|
| `workflow_mode` | Propagates the selected mode through all nodes |
| `draft_report` | Synthesizer output before fact-checking |
| `fact_check_result` | Per-claim audit dict from Fact Checker |
| `citation_density` | Numeric density score (0.0–1.0) |
| `resynthesis_needed` | Boolean gate for conditional routing |
| `post_fact_check_result` | Second audit dict from Post Fact Checker |
| `post_citation_density` | Density after re-synthesis |

---

## 8. New API Endpoint: `/api/chat`

**Project A:** The only endpoint was `POST /api/analyze` (SSE) and `GET /api/health`.

**Project B:** A second streaming endpoint was added:

- **`POST /api/chat`** — accepts a `ChatRequest` (the generated report + message history). Forwards the conversation to Ollama's `/api/chat` streaming API with a system prompt grounding the assistant in the report content. Emits SSE `token` events (incremental text chunks) and a final `done` event.

This enables the frontend's new chat panel to support follow-up questions grounded in the just-generated report.

---

## 9. New Frontend Component: Chat Panel

**Project A:** The frontend had four components: `dashboard.tsx`, `report-view.tsx`, `search-bar.tsx`, `thinking-log.tsx`. No chat UI existed.

**Project B:** A fifth component was added:

- **`chat-panel.tsx`** — a streaming chat sidebar that appears after a report is generated. Users can ask follow-up questions about the report. Backed by a new `lib/chat.ts` that wraps the `/api/chat` SSE endpoint. Responses are streamed token-by-token into an auto-scrolling message thread.

---

## 10. New: Experiment Framework

**Project A:** No experiment infrastructure existed.

**Project B:** A complete experiment suite was added under `src/backend/experiment/`:

| File | Purpose |
|---|---|
| `freeze_data.py` | Fetches and saves yfinance snapshots for 15 tickers to JSON files — ensures all modes see identical data during experiments |
| `mock_mcp_server.py` | Replacement MCP server that reads from frozen JSON snapshots instead of calling yfinance live |
| `run_experiment.py` | Runs brief/normal/extra on each of the 15 tickers for N trials; writes results to JSONL; runs are interleaved per ticker to distribute Ollama load variance |
| `analyze.py` | Reads the JSONL output and computes per-mode statistics: citation density distributions, resynthesis rates, latency, and improvement deltas |
| `tickers.py` | The 15 ticker symbols used across all experiment runs |

---

## 11. Draft Report Persistence

**Project A:** Only the final report was saved to disk under `backend/reports/`.

**Project B:** The Synthesizer's intermediate output is also saved before the Fact Checker processes it:

- **Draft reports:** saved to `backend/draft_reports/` as `<TICKER>_<timestamp>_draft.md`
- **Final reports:** saved only when the definitive node (`fact_checker` or `post_fact_checker`) emits the report — not for the Re-Synthesizer's intermediate output

One committed draft report (`GOOG_20260415T202620Z_draft.md`) is included in the repo as a reference example.

---

## 12. Repository Structure Changes

**Project A layout:**
```
team-11/
├── src/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── graph/
│   │   │   │   ├── nodes.py      (3 nodes: Auditor, News Hound, Synthesizer)
│   │   │   │   ├── state.py      (6-field AgentState)
│   │   │   │   └── workflow.py   (linear 3-node graph)
│   │   │   ├── config.py         (GEMINI_API_KEY, gemini_model)
│   │   │   ├── main.py           (/api/analyze, /api/health)
│   │   │   ├── mcp_server.py
│   │   │   └── schemas.py        (AnalyzeRequest only)
│   │   ├── requirements.txt      (root-level)
│   │   └── run.py
│   └── frontend/
│       └── components/           (4 components, no chat)
└── tests/
```

**Project B layout:**
```
team-11-ProjB/
├── src/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── graph/
│   │   │   │   ├── nodes.py      (6 nodes: +Fact Checker, Re-Synthesizer, Post Fact Checker)
│   │   │   │   ├── state.py      (13-field AgentState)
│   │   │   │   └── workflow.py   (conditional graph with 4 workflow modes)
│   │   │   ├── config.py         (OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SEC)
│   │   │   ├── main.py           (/api/analyze + /api/chat, /api/health)
│   │   │   ├── mcp_server.py
│   │   │   └── schemas.py        (AnalyzeRequest, ChatRequest, ChatMessage added)
│   │   ├── experiment/           (new)
│   │   │   ├── freeze_data.py
│   │   │   ├── mock_mcp_server.py
│   │   │   ├── run_experiment.py
│   │   │   ├── analyze.py
│   │   │   └── tickers.py
│   │   ├── draft_reports/        (new — Synthesizer intermediate outputs)
│   │   ├── requirements.txt      (moved inside backend/)
│   │   └── run.py
│   └── frontend/
│       ├── components/           (5 components: +chat-panel.tsx)
│       └── lib/
│           └── chat.ts           (new — /api/chat SSE client)
```

---

## Summary Table: Project A vs. Project B

| Dimension | Project A | Project B |
|---|---|---|
| **LLM provider** | Google Gemini (remote, `langchain-google-genai`) | Ollama (local, `langchain-ollama`) |
| **Default model** | Gemini (API key required) | `gemma4` (no API key needed) |
| **Graph nodes** | 3: Auditor, News Hound, Synthesizer | 6: + Fact Checker, Re-Synthesizer, Post Fact Checker |
| **Graph topology** | Linear (A → B → C → END) | Conditional (A → B → C → D ─┬─ END / └─ E → F → END) |
| **Workflow modes** | 1 (fixed) | 4: normal, brief, extra, extra_force |
| **Hallucination detection** | None | Citation density via pure-Python numerical cross-reference (±5% tolerance) |
| **Resynthesis** | None | Conditional (extra/extra_force modes) or forced (extra_force) |
| **Post-resynth audit** | None | Before/after density table appended to final report |
| **AgentState fields** | 6 | 13 |
| **Synthesizer output field** | `report` (final) | `draft_report` (intermediate) |
| **Disclaimer on report** | None | Tiered: green / amber / red based on citation density |
| **Chat endpoint** | None | `POST /api/chat` (Ollama-backed, token-streaming SSE) |
| **Chat UI** | None | `chat-panel.tsx` — follow-up Q&A grounded in the report |
| **Experiment framework** | None | `experiment/`: freeze_data, mock_mcp_server, run_experiment, analyze |
| **Draft report persistence** | None | Saved to `draft_reports/` before fact-checking |
| **Final report persistence** | Saved after Synthesizer | Saved only after Fact Checker or Post Fact Checker |
| **Requirements location** | `team-11/requirements.txt` (root) | `team-11-ProjB/src/backend/requirements.txt` |
