# EnvPilot: Project A → Project B Changes

> **Note:** The team submitted one combined folder (`team-12-envcheck-both`) rather than two separate folders. Project A refers to the original `team-envcheck/` at git commit `1df9809`; Project B is the current state of `team-12-envcheck-both`.

---

## 1. Core Architecture: From Static Scanner to Agentic Pipeline

**Project A** was a pure static-analysis tool — a 4-module pipeline run once and done:

```
Parser → Version Detector → Knowledge Base → Scanner
```

**Project B** added a full **LangGraph 8-node agent** on top, in a new `envcheck/agent/` package:

```
analysis → env_probe → kb_query ──┬── web_search → kb_update ──┐
                                  └── (skip if KB sufficient) ──┤
                                                                 ↓
                                                    plan → preflight ──┬── generation → END
                                                             ↑         └── (retry on failure) ──┘
```

| Node | Role |
|---|---|
| `analysis_node` | Parses the task, identifies packages and critical APIs |
| `env_probe_node` | Inspects the actual target environment for installed versions |
| `kb_query_node` | Queries the SQLite FTS knowledge base; sets a flag if there are gaps |
| `web_search_node` | DuckDuckGo search for migration guides (only triggered if KB has gaps) |
| `kb_update_node` | Writes new findings from web search back into the KB store |
| `plan_node` | LLM proposes specific APIs to use, aware of breaking changes |
| `preflight_node` | Runs proposed APIs in an **isolated subprocess** to verify they work before committing; loops back to `plan` on failure (up to a max retry count) |
| `generation_node` | Final code generation using only verified-safe APIs |

---

## 2. New Entrypoint: `envpilot.py`

A new 140-line CLI (`envpilot.py`) replaces the simple `main.py` demo as the primary way to run the full agentic workflow:

```bash
uv run python envpilot.py "Create a numpy script that computes trapezoid integration"
uv run python envpilot.py --env ./environments/case_numpy_2x "Create a pandas script using fillna"
```

Project A had only `main.py` as a scanner demo entry point.

---

## 3. Four New Infrastructure Modules

| Module | What It Does |
|---|---|
| `envcheck/mcp_server.py` | FastMCP server exposing 4 tools for IDE integration: env probe, KB query, KB update, preflight test |
| `envcheck/web_app.py` | FastAPI backend with SSE streaming for a browser-based scan/run UI |
| `envcheck/web_searcher.py` | DuckDuckGo-based web search for migration guides |
| `envcheck/knowledge_base_store.py` | SQLite FTS-backed searchable KB store (replaces in-memory dict lookup) |

---

## 4. Web UI

A new `envcheck/static/index.html` single-page app lets users submit code for scanning, search the KB, and stream full EnvPilot agent runs from a browser. Project A had no web interface.

Start it with:

```bash
uv run uvicorn envcheck.web_app:app --reload --port 8000
```

---

## 5. Benchmark / Evaluation Suite (`benchmark/` — new)

A full empirical evaluation framework was added:

| File | Purpose |
|---|---|
| `candidates.json` | 24 verified breaking-change cases from BigCodeBench, each with canonical failing code, corrected code, tests, and package pins |
| `run_eval.py` | Runs baseline (no EnvPilot) vs. EnvPilot side-by-side, measuring token counts, LLM call counts, and correctness |
| `verify_ground_truth.py` | Validates each candidate in an isolated `uv` environment |
| `build_candidates.py` | Builds the candidate pool from BigCodeBench |
| `runner_utils.py` | Shared utilities for isolated `uv` environment execution |

Run evaluation:

```bash
uv run python benchmark/verify_ground_truth.py --first 5
uv run python benchmark/run_eval.py --mode envpilot
```

---

## 6. LLM Comparison Demo (`demo_llm.py` — new)

A new `demo_llm.py` runs Claude vs. Gemini side-by-side on the same cases with mock mode for development and generates a summary report — complementing the original `demo.py`, which tested the static scanner only:

```bash
uv run python demo_llm.py --mock --case pandas_22
uv run python demo_llm.py --all --mock --report
uv run python demo_llm.py --provider gemini --case numpy_2x
```

---

## 7. Cursor IDE Integration

Two new files define a read-only diagnostic workflow usable directly from the Cursor IDE:

- `.cursor/commands/envcheck.diagnose.md` — 10-step diagnostic command spec
- `.cursor/rules/envcheck-project.mdc` — project-level rules for the IDE

---

## Summary

| Area | Project A | Project B |
|---|---|---|
| Core pipeline | Static 4-module scanner | Static scanner + 8-node LangGraph agent |
| Entry point | `main.py` (demo/eval) | `envpilot.py` (full agentic CLI) |
| Knowledge base | In-memory dict | SQLite FTS store (`knowledge_base_store.py`) |
| Web search | None | DuckDuckGo via `web_searcher.py` |
| Preflight verification | None | Isolated subprocess execution (`preflight_runner.py`) |
| Web UI | None | FastAPI + SSE single-page app |
| IDE integration | None | FastMCP server (4 tools) + Cursor command |
| Evaluation | 7 manual test cases | 24-case BigCodeBench benchmark suite with baseline comparison |
| LLM demo | Single-provider `demo.py` | Multi-provider Claude/Gemini comparison (`demo_llm.py`) |

**In short:** Project A was a smart static linter. Project B turned it into a full agentic code-generation assistant that proactively avoids breaking changes by querying a KB, searching the web for migration guides, planning safe APIs, verifying them via isolated execution before writing code, and exposing everything through an MCP server for IDE use.
