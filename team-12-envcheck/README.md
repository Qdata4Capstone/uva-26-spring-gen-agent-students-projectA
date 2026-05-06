# EnvPilot: Proactive Code Environment Diagnostic Tool

EnvPilot is a proactive code environment diagnostic tool built with Python. It uses AST static analysis and a version-aware breaking-change knowledge base to catch API compatibility issues before generated code is executed.

The latest version includes pre-execution API compatibility checking for LLM-generated code, breaking-change detection across major libraries, isolated per-case environment testing via `uv`, LLM-integrated workflow comparison for Claude and Gemini, automated Markdown and JSON scan reporting, a LangGraph-based agent workflow, a FastMCP server, a FastAPI web UI, benchmark/evaluation tooling, and a Cursor IDE diagnostic command (`/envcheck.diagnose`).

Team members: Yusen Wu, Tingfeng Lan, Yi Deng

## What EnvPilot Does

LLMs often generate Python code that is syntactically valid but incompatible with the exact package versions installed in a target environment. EnvPilot addresses this gap by checking code against known package-level breaking changes before runtime.

Core capabilities:

- Parses Python code with the built-in `ast` module without executing user code.
- Detects imports, aliases, attribute calls, method calls, and keyword-argument usage.
- Reads installed package versions from the active interpreter or an isolated virtual environment.
- Matches code against a curated breaking-change knowledge base.
- Reports all findings in one scan so LLM repair can happen in a single informed turn.
- Supports NumPy, SciPy, pandas, scikit-learn, NetworkX, Pydantic, and benchmark coverage for additional libraries such as seaborn, Pillow, matplotlib, and Flask.

## Latest Features

- **Agent workflow**: A LangGraph pipeline for analysis, environment probing, knowledge-base lookup, optional web search, deterministic preflight verification, planning, and final code generation.
- **Preflight API verification**: Proposed APIs are tested in the target environment before full code generation.
- **MCP integration**: `envpilot-mcp` exposes environment probing, knowledge-base query/update, web search, and preflight testing tools.
- **Web UI**: FastAPI backend with a single-page frontend for quick scans, knowledge-base search, and streamed EnvPilot runs.
- **Benchmark suite**: 24 verified compatibility cases with baseline vs. EnvPilot evaluation, token/call instrumentation, cached `uv` environments, and summary metrics.
- **Cursor command**: `.cursor/commands/envcheck.diagnose.md` defines a read-only diagnostic workflow for IDE use.

## Repository Structure

```text
team-12-envcheck/
├── envcheck/
│   ├── parser.py                    # AST parser for imports and API usage
│   ├── version_detector.py          # Installed package/version detection
│   ├── knowledge_base.py            # Curated breaking-change rules
│   ├── knowledge_base_store.py      # Searchable KB store used by agent/MCP/web
│   ├── scanner.py                   # Static compatibility scanner
│   ├── preflight_runner.py          # Isolated smoke-test execution
│   ├── mcp_server.py                # FastMCP tools
│   ├── web_app.py                   # FastAPI application
│   ├── static/index.html            # Web UI
│   └── agent/
│       ├── graph.py                 # LangGraph workflow topology
│       ├── nodes.py                 # Agent node implementations and metrics
│       ├── prompts.py               # LLM prompts
│       └── state.py                 # Agent state schema
├── benchmark/                       # Verified benchmark cases and eval runner
├── docs/                            # Demo walkthrough and command explanation
├── test_cases/                      # Original demo cases
├── envpilot.py                      # EnvPilot CLI
├── demo.py                          # Non-LLM workflow demo
├── demo_llm.py                      # Claude/Gemini workflow comparison demo
├── main.py                          # Scanner demo/eval entry point
├── pyproject.toml
└── uv.lock
```

## Installation

Requirements:

- Python 3.12+
- `uv`

```bash
cd team-12-envcheck
uv sync
```

LLM-backed modes require one of these environment variables:

```bash
export GEMINI_API_KEY="..."
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

## Usage

Run the original scanner/demo flow:

```bash
uv run python main.py
uv run python demo.py
```

Run the Claude/Gemini comparison demo:

```bash
uv run python demo_llm.py --mock --case pandas_22
uv run python demo_llm.py --all --mock --report
uv run python demo_llm.py --provider gemini --case numpy_2x
```

Run the EnvPilot agent:

```bash
uv run python envpilot.py "Create a numpy script that computes trapezoid integration"
uv run python envpilot.py --env ./environments/case_numpy_2x "Create a pandas script using fillna"
```

Start the web UI:

```bash
uv run uvicorn envcheck.web_app:app --reload --port 8000
```

Start the MCP server:

```bash
uv run envpilot-mcp
```

Run benchmark verification and evaluation:

```bash
uv run python benchmark/verify_ground_truth.py --first 5
uv run python benchmark/run_eval.py --case manual_011
uv run python benchmark/run_eval.py --mode envpilot
```

## Reports and Outputs

Generated artifacts are intentionally gitignored:

- `reports/`
- `benchmark/envs/`
- `benchmark/verification_report.json`
- `benchmark/eval_results.json`
- `benchmark/eval_summary.json`
- `benchmark/charts/`

The committed benchmark data lives in `benchmark/candidates.json` and contains verified bad-environment cases with canonical failing code, corrected code, tests, package pins, and metadata.

## Development

```bash
uv run pytest
uv run ruff check
uv run ruff format
```
