# IMPROVE.md — EnvPilot (team-12-envcheck-both)

## Project Overview

EnvPilot is an AI-powered pre-flight diagnostic system for detecting Python API breaking changes before runtime. It combines AST-based static analysis, a curated knowledge base of breaking changes, version detection, LLM reasoning (Claude/Gemini via LangGraph), a FastAPI web UI with SSE streaming, a Model Context Protocol server for IDE integration, and a BigCodeBench-based benchmark evaluation suite. The tool's core promise: flag incompatibilities *before* the "code → crash → debug" loop begins.

---

## What Changed Since Commit 1df9809

Compared to the original `team-envcheck/` at commit `1df9809`, `team-12-envcheck-both` adds a major architectural upgrade:

| Area | What Was Added |
|------|---------------|
| LangGraph agent | `envcheck/agent/` (entire new package: `graph.py` 131 lines, `nodes.py` 699 lines, `state.py`, `prompts.py`) — full 5-phase LLM workflow |
| Web UI | `envcheck/web_app.py` (436 lines) — FastAPI + SSE streaming single-page app |
| MCP server | `envcheck/mcp_server.py` (222 lines) — 4-tool FastMCP server for IDE integration |
| Web search | `envcheck/web_searcher.py` — DuckDuckGo-based search for migration guides |
| KB store | `envcheck/knowledge_base_store.py` — SQLite FTS searchable KB store |
| Preflight runner | `envcheck/preflight_runner.py` — isolated subprocess smoke-test execution |
| Static web UI | `envcheck/static/index.html` — browser-based scan interface |
| CLI entry point | `envpilot.py` (140 lines) — 5-phase LangGraph CLI runner |
| Benchmark suite | `benchmark/` (11 files) — BigCodeBench integration, ground-truth verification, eval runner |
| Dependencies | LangGraph, FastAPI, FastMCP, `google-genai`, `duckduckgo-search` all added |

The original 4-module architecture (Parser → Version Detector → KB → Scanner) is preserved and extended.

---

## Strengths to Preserve

- Clean 4-layer core: Parser → Version Detector → KB → Scanner — each independently testable
- AST-based parsing is robust to formatting; 4 distinct pattern types cover most common breaking-change patterns
- 5-phase LangGraph agent (analysis → env probe → KB query → web search → generation) is well-structured
- MCP server enables IDE-native integration without a separate daemon
- Isolated subprocess execution in `preflight_runner.py` correctly prevents arbitrary code from escaping
- Benchmark evaluation against real BigCodeBench cases provides reproducible metrics
- Metrics instrumentation in `nodes.py` (token counts, latency, LLM call counts) enables cost tracking
- Multi-provider LLM support (Claude, Gemini) with mock mode for development

---

## P1 — Critical Issues (Will Break Utility or CI Use)

### P1.1 Exit Code Always 0 — CI/CD Integration Broken
**Files:** `main.py`, `envpilot.py`  
Neither entry point exits with code 1 when findings are detected. Any CI/CD pipeline using EnvPilot as a gate will always pass regardless of results.  
**Fix:** In `main.py` and `envpilot.py`, after scanning, call `sys.exit(1)` when `len(scan_report.findings) > 0`. Add a `--no-fail` flag to suppress this for informational runs. Update the README with a GitHub Actions example:
```yaml
- run: uv run python -m envcheck --path src/ --fail-on-findings
```

### P1.2 Knowledge Base Still at ~20 Rules
**File:** `envcheck/knowledge_base.py` (351 lines — identical size to commit 1df9809's `team-envcheck/envcheck/knowledge_base.py`)  
The KB was not expanded as part of Project B. With ~20 rules covering only 6 libraries, EnvPilot misses the vast majority of real-world breaking changes. This is the primary gap limiting practical utility.  
**Fix:** Expand to at least 100 rules. Prioritized targets:
- **NumPy 2.0**: Complete the remaining removals (currently ~8 of ~50 documented removals are covered)
- **pandas 2.x**: `.applymap()` → `.map()`, `DataFrame.swapaxes()` removal, `append()` removal
- **scikit-learn 1.3/1.4**: `_validate_data` changes, `n_features_in_` deprecations
- **Matplotlib 3.8+**: `plt.show()` argument changes, `axes.set_aspect()` signature
- **Streamlit**: `st.cache` → `st.cache_data`/`st.cache_resource`, `st.experimental_*` removals
- **Python stdlib**: `distutils` (removed 3.12), `asyncio.coroutine` (removed 3.11), `collections.Mapping` (removed 3.10)

### P1.3 Zero Unit Tests
**File:** `tests/__init__.py` (empty)  
The 5-phase agent, scanner matchers, parser, version detector, and KB registration have no automated tests. The benchmark suite tests end-to-end behavior but cannot substitute for unit tests that isolate individual components.  
**Fix:** Add a `pytest` suite with:
- `test_parser.py`: test all 4 AST node types (import, attribute access, method call, method access) using inline code strings
- `test_version_detector.py`: test `compare_versions()` for pre-release, equal, invalid, and major/minor/patch cases
- `test_scanner.py`: test all 4 matcher functions against mock `InstalledPackages` and a known rule
- `test_knowledge_base.py`: test `register_breaking_change()` deduplication and `get_rules_for_library()`
- `test_regression.py`: for each of the 7 existing test cases in `test_cases/cases.py`, assert the scanner finds the expected finding in the broken code

Target ≥70% line coverage; run with `uv run pytest --cov`.

### P1.4 Web App Has No Authentication and Accepts Unlimited Input
**File:** `envcheck/web_app.py`, CORS configuration and `POST /api/scan`  
```python
allow_origins=["*"]  # open CORS
body.get("code", "")  # no size limit on submitted code
```
The web app is open to any origin and accepts arbitrarily large code blobs, making it trivially DoS-able when deployed.  
**Fix:**
- Add a `MAX_CODE_BYTES = 100_000` guard before scanning
- Restrict CORS to `["http://localhost:8000"]` by default; make it configurable via env var
- Add a simple bearer token check (`X-API-Key` header) when deployed in non-localhost environments

---

## P2 — Robustness Issues

### P2.1 Parser Does Not Track Simple Alias Assignments
**File:** `envcheck/parser.py`, `_SourceVisitor`  
The parser correctly handles `import numpy as np` → `np.trapz`, but not:
```python
lib = np
lib.trapz(x, y)   # false negative
```
This is a known limitation but common in dynamically-structured code.  
**Fix:** Extend `_SourceVisitor` to walk `Assign` nodes: when `targets=[Name(id='x')]` and `value=Name(id='np')`, add `'x' → 'np'` to the alias map. Limit to 1-hop aliases (no transitivity) to avoid complexity.

### P2.2 Version Detection Silent Failures
**File:** `envcheck/version_detector.py`, lines 64–87  
When `pip list` subprocess fails (conda env, frozen binary, permission error), the function returns an empty dict with no warning to the user. Scanning continues with "version unknown" for all packages, silently suppressing potentially valid findings.  
**Fix:**
- Log a visible warning when version detection fails: `"WARNING: version detection failed — running in version-unknown mode (all matching rules will be reported)"`
- Fall back to `importlib.metadata.packages_distributions()` as primary method; use `pip list` as secondary
- Fix the no-op normalization: `envcheck/version_detector.py:131` has `.replace("-", "-")` — change to `.replace("_", "-")`

### P2.3 Agent Nodes Swallow All Exceptions
**File:** `envcheck/agent/nodes.py`, lines 150–159, 212–217, 252–260  
Most nodes catch `Exception` broadly and log only a warning, then continue. This means a network failure during web search or a malformed LLM response looks identical to a successful skip.  
**Fix:** Distinguish between:
- **Recoverable errors** (network timeout, empty search results): log warning, continue with degraded state
- **Fatal errors** (invalid state schema, LLM API auth failure): raise `AgentError` to surface to the caller

### P2.4 Regex Extraction Fragile on Code Containing Triple Backticks
**File:** `envcheck/agent/nodes.py`, lines 389–406, `_extract_fenced_python`  
```python
re.search(r'```python\n(.*?)```', content, re.DOTALL)
```
This stops at the first ` ``` ` after the opening fence, even if the extracted code itself contains ` ``` ` (e.g., a docstring with a code example). This produces silently truncated output code.  
**Fix:** Use a more robust extraction: find the ```` ```python ```` marker, then find the matching closing ` ``` ` that appears on a line by itself.

### P2.5 Pre-release Version Handling
**File:** `envcheck/version_detector.py`, `compare_versions()`  
A version string like `2.0.0rc1` is not handled explicitly. Python's `packaging.version.parse()` treats it as `< 2.0.0`, which is correct — but the code adds raw pre-release strings in some paths without normalization, which can cause `InvalidVersion` exceptions.  
**Fix:** Normalize all version strings through `packaging.version.parse()` before comparison; catch `InvalidVersion` and log a warning rather than raising.

### P2.6 `_runs` Dict in Web App Is In-Memory Only
**File:** `envcheck/web_app.py`, line 48  
```python
_runs: dict[str, dict] = {}
```
All run history is lost on restart. In development this is acceptable, but the web UI implies persistence.  
**Fix:** For current scope, add a comment documenting the in-memory limitation. If persistence is desired, use an SQLite file (already used by `knowledge_base_store.py`) to persist run results.

---

## P3 — Code Quality & Maintainability

### P3.1 Hardcoded Model Names and Constants
**File:** `envcheck/agent/nodes.py`, `demo_llm.py`  
Model names (`claude-sonnet-4-20250514`, `gemini-2.5-flash`) are scattered across files. When models are updated, multiple files must be changed.  
**Fix:** Create `envcheck/config.py` with:
```python
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_PREFLIGHT_ATTEMPTS = 3
MAX_PLAN_ATTEMPTS = 2
```
Import constants from there throughout.

### P3.2 Replace `print()` with Structured Logging
**Files:** `main.py`, `demo.py`, `demo_llm.py`  
All output is via `print()`. No log level control; no way to distinguish info from debug output in batch runs.  
**Fix:** Use `logging.getLogger(__name__)` throughout; add `--verbose` flag to enable `DEBUG` output; add `--json-output` to emit structured JSON logs for CI integration.

### P3.3 Cursor IDE Command Partially Implemented
**File:** `.cursor/commands/envcheck.diagnose.md`  
The spec describes a 10-step diagnostic workflow. Only steps 4 (API scan) and 7 (runtime test) are implemented. Steps 1–3 are aspirational but documented as if complete, which is misleading to anyone trying to follow the spec.  
**Fix:** Add a status column to the spec: `[Implemented]`, `[Planned]`, `[Future]`. Implement steps 1–3 as a `python -m envcheck env-report` subcommand that detects the virtual environment, reads declared dependencies, and flags version drift against PyPI.

### P3.4 Mock Mode Results Are Not Representative
**File:** `demo_llm.py`, lines 99–174, `MOCK_PARTIAL_FIXES`  
The mock mode simulates partial fixes for multi-breaking-change cases. This makes mock-mode benchmark results look worse than real LLM behavior (by design), but the results file does not flag which runs used mock mode, making comparisons ambiguous.  
**Fix:** Add a `"mock": true` field to mock-mode output entries; exclude mock runs from aggregate metrics or label them separately.

---

## P4 — Features & Integration

### P4.1 Add Pre-commit Hook
The pre-commit hook was planned in the original `IMPROVEMENT_PLAN.md` and is the most practical path to adoption. It requires only the exit code fix (P1.1) to be complete.  
**Fix:** Add `.pre-commit-hooks.yaml`:
```yaml
- id: envpilot
  name: EnvPilot API Compatibility Check
  entry: python -m envcheck
  language: python
  types: [python]
  pass_filenames: true
```
Document in README. The hook becomes useful as soon as P1.1 and P1.2 are addressed.

### P4.2 Add `--fix` Output Mode
**Fix:** Add a `replacement` field to `BreakingChangeRule`. In `--fix` mode, print:
```
FOUND:   np.trapz(y, x)
REPLACE: np.trapezoid(y, x)   # NumPy 2.0+
```
For rules without a known replacement, print a link to the relevant migration guide.

### P4.3 Add GitHub Actions CI Workflow
No CI configuration exists. The project's own tests and linter are not run automatically.  
**Fix:** Add `.github/workflows/ci.yml`:
```yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run ruff check
      - run: uv run pytest --cov
      - run: uv run python -m envcheck --path envcheck/  # dogfood
```

### P4.4 Add a `RULE_SOURCES.md` Audit File
As the KB grows, knowing *where* each rule came from (release notes URL, GitHub issue, deprecation warning) is essential for auditing false positives.  
**Fix:** Create `envcheck/RULE_SOURCES.md` with a table: `library | rule_id | source_url | verified_on`. Update when adding rules.

---

## Summary Roadmap

| Priority | Item | File(s) | Effort |
|----------|------|---------|--------|
| P1 | Exit code 1 on findings + `--no-fail` flag | `main.py`, `envpilot.py` | Low |
| P1 | Expand KB to 100+ rules | `knowledge_base.py` | High |
| P1 | Write unit test suite (≥70% coverage) | `tests/` | High |
| P1 | Web app: input size limit + CORS + auth | `web_app.py` | Low |
| P2 | Parser alias assignment tracking | `parser.py` | Medium |
| P2 | Version detection fallback + warning | `version_detector.py` | Low |
| P2 | Fix no-op string normalization | `version_detector.py:131` | Low |
| P2 | Distinguish recoverable vs fatal agent errors | `agent/nodes.py` | Medium |
| P2 | Fix triple-backtick regex extraction | `agent/nodes.py:389` | Low |
| P2 | Handle pre-release version strings | `version_detector.py` | Low |
| P3 | Centralize model names/constants in `config.py` | `nodes.py`, `demo_llm.py` | Low |
| P3 | Replace `print()` with `logging` | `main.py`, `demo*.py` | Low |
| P3 | Mark Cursor command steps as Implemented/Planned | `.cursor/commands/` | Low |
| P3 | Flag mock-mode runs in output | `demo_llm.py` | Low |
| P4 | Add `.pre-commit-hooks.yaml` | — | Low |
| P4 | Add `--fix` mode with replacement suggestions | `scanner.py`, `knowledge_base.py` | Medium |
| P4 | Add GitHub Actions CI workflow | `.github/workflows/ci.yml` | Low |
| P4 | Add `RULE_SOURCES.md` audit file | `envcheck/` | Low |
