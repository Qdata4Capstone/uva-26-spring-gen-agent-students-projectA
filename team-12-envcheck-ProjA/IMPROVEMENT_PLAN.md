# Improvement Plan — envcheck (team-envcheck)

## Project Summary
EnvCheck is an AI pre-flight diagnostic tool that detects API breaking changes and version mismatches in Python code **without executing it**, using AST-based static analysis and a curated knowledge base of breaking changes. It demonstrates measurable ROI: 25% fewer LLM API calls, 31% fewer tokens, and 100% elimination of runtime crashes across 7 test cases.

---

## Strengths to Preserve
- Elegant 4-component architecture: Parser → Version Detector → Knowledge Base → Scanner
- AST-based parsing — robust to code formatting and comments
- 4 distinct matcher types (attribute access, imports, method calls, method absence)
- Comprehensive 7-test-case demonstration suite with real breaking changes
- Measured, reproducible ROI metrics
- Extensible knowledge base (add rules via `register_breaking_change()`)
- Multi-provider LLM support (Claude, Gemini, mock mode)

---

## Priority 1 — Critical Gaps (Core Functionality)

### 1.1 Expand the Knowledge Base (10× Current Coverage)
**Problem:** The knowledge base has only 20 rules covering 6 libraries. This is the primary limit on EnvCheck's usefulness.

**Action:**
- Target at least 100 additional rules covering:
  - **NumPy**: All 1.x → 2.0 removals (not just the 8 currently covered)
  - **pandas**: 2.1+ changes (`.map()` replacement for `.applymap()`, `FutureWarning` parameters)
  - **scikit-learn**: 1.3, 1.4 deprecations (e.g., `n_features_in_`, `_validate_data` changes)
  - **Matplotlib**: 3.8+ API changes (e.g., removed `plt.show()` arguments)
  - **Streamlit**: `st.cache` → `st.cache_data`/`st.cache_resource`, `st.experimental_*` removals
  - **Python stdlib**: `distutils` removal in 3.12, `asyncio.coroutine` removal in 3.11, `collections.abc` migration
  - **TensorFlow/Keras**: Major API changes between TF 1.x/2.x and Keras 2/3
- Add a `RULE_SOURCES.md` file documenting where each rule came from (release notes URL, GitHub issue) so the knowledge base stays auditable.

### 1.2 Add Pre-commit Hook Integration
**Problem:** EnvCheck is a standalone CLI only. Developers must manually invoke it; it will never become a habitual part of the workflow.

**Action:**
- Create `.pre-commit-hooks.yaml`:
  ```yaml
  - id: envcheck
    name: EnvCheck API Compatibility
    entry: python -m envcheck
    language: python
    types: [python]
  ```
- Return exit code 1 when findings are detected (currently the exit code is always 0).
- Add a `--fail-on-findings` flag (make it default for pre-commit hook use).
- Document the pre-commit setup in the README.

### 1.3 Build a Proper Test Suite
**Problem:** The `tests/` directory is nearly empty. The knowledge base, scanner, parser, and version detector have no automated tests.

**Action:**
- Write unit tests for:
  - `parser.py`: all 4 node types (import, attribute access, method call, method access) using synthetic code strings
  - `version_detector.py`: `compare_versions()` edge cases (pre-release, equal versions, invalid strings)
  - `scanner.py`: all 4 matchers against mock `InstalledPackages` and known rules
  - `knowledge_base.py`: rule registration, duplicate detection, invalid `removed_in` format
- Write regression tests: for each of the 7 existing test case's broken code snippets, assert that the scanner finds the expected finding.
- Target ≥ 70% line coverage; run with `pytest --cov`.

### 1.4 Return Non-Zero Exit Code on Findings
**Problem:** `envcheck` always exits 0 even when findings are detected. This makes CI/CD integration impossible — a CI step can't fail based on EnvCheck results.

**Action:**
- Exit with code 1 when `len(scan_report.findings) > 0`.
- Add a `--no-fail` flag to suppress this for informational use.
- Update the README with a CI/CD usage example (GitHub Actions):
  ```yaml
  - run: python -m envcheck --path src/ --fail-on-findings
  ```

---

## Priority 2 — Robustness & Quality

### 2.1 Add New Pattern Types to the Knowledge Base
**Problem:** The 4 existing pattern types (ATTRIBUTE, IMPORT, METHOD_CALL, METHOD_ACCESS) do not cover class/function renames, module reorganizations, or parameter reordering.

**Action:**
- Add a `RENAMED` pattern type: `(old_name, new_name, module, version)` — detects deprecated names and suggests the replacement.
- Add a `MODULE_MOVED` pattern type: `(old_module, new_module, symbol, version)` — covers cases like `scipy.integrate.cumtrapz` → `scipy.integrate.cumulative_trapezoid`.
- Add a `PARAMETER_REMOVED` pattern type: similar to `METHOD_CALL` but specifically for removed positional arguments.
- Update the scanner with corresponding matcher functions.

### 2.2 Improve Variable Alias Tracking in the Parser
**Problem:** The parser resolves `import numpy as np` → `np.trapz` correctly, but cannot track `x = np` then `x.trapz()`.

**Action:**
- Extend `_SourceVisitor` to track simple alias assignments: `Assign(targets=[Name(id='x')], value=Name(id='np'))`.
- Resolve chained aliases (at most 2 hops to avoid complexity).
- Document the known limitation: computed aliases (e.g., `alias = get_lib()`) remain unsupported.

### 2.3 Add Error Handling for Version Detection Edge Cases
**Problem:** Version detection assumes a standard pip/uv environment. Pre-release versions, conda environments, and frozen binaries may cause silent failures.

**Action:**
- When `pip list` fails, fall back to `importlib.metadata.packages_distributions()` as the primary method.
- Handle pre-release version strings (e.g., `2.0.0rc1`) explicitly in `compare_versions()`: treat them as `< 2.0.0` by default.
- If version detection completely fails, emit a warning and run in "version-unknown" mode (report potential findings regardless of installed version).

### 2.4 Add a GitHub Actions Workflow for CI
**Problem:** There is no CI configuration. The project's own tests and linting are not automatically run.

**Action:**
- Add `.github/workflows/ci.yml` that:
  1. Runs `uv sync` to install dependencies
  2. Runs `uv run ruff check` for linting
  3. Runs `uv run pytest --cov` for tests
  4. Runs the EnvCheck scanner on its own source code (dogfooding)
- Run on push to `main` and on PRs.

### 2.5 Replace Print Statements with Structured Logging
**Problem:** `demo_llm.py` and `main.py` use `print()` throughout. There is no log level control.

**Action:**
- Replace `print()` with `logging.info()` / `logging.debug()` / `logging.warning()`.
- Add a `--verbose` flag that enables `DEBUG` level output.
- Emit structured JSON logs when `--json-output` is passed (for CI/CD integration).

---

## Priority 3 — Features & Integration

### 3.1 Implement the Cursor IDE Command (Not Just the Spec)
**Problem:** `.cursor/commands/envcheck.diagnose.md` describes a 10-step diagnostic flow, but only steps 4 (API scan) and 7 (optional runtime test) are implemented. Steps 1–3 (environment discovery, dependency analysis, version reconciliation) are aspirational.

**Action:**
- Implement steps 1–3 as a new CLI subcommand: `python -m envcheck env-report`.
  - Step 1: Detect virtual environment (`.venv`, `venv`, `conda`, `uv` lockfile)
  - Step 2: Read `requirements.txt` or `pyproject.toml` for declared dependencies
  - Step 3: Compare declared vs. installed vs. latest on PyPI; flag version drift
- Update the Cursor command spec to reflect what is actually implemented vs. future work.

### 3.2 Add a `--fix` Mode with Auto-Suggestions
**Problem:** EnvCheck reports findings but does not suggest the replacement code. Users must look up the migration guide themselves.

**Action:**
- Add a `replacement` field to `BreakingChangeRule`: `replacement="np.trapezoid"` for the `np.trapz` removal.
- In the `--fix` mode output, print: `Found: np.trapz → Replace with: np.trapezoid (NumPy 2.0+)`.
- (Optional) Implement AST-based auto-fix for ATTRIBUTE and IMPORT rules using `libcst`.

### 3.3 Support More LLM Providers
**Problem:** Only Claude and Gemini are supported. OpenAI and local LLMs (Ollama) are excluded.

**Action:**
- Add an `openai` provider to `demo_llm.py` using the same `openai` Python SDK.
- Add an `ollama` provider using local HTTP API (`http://localhost:11434/api/chat`).
- Make the provider selectable via `--provider {claude,gemini,openai,ollama}` CLI flag.

### 3.4 Add a Knowledge Base Update Mechanism
**Problem:** The knowledge base is manually curated with no mechanism to update it when new library versions are released.

**Action:**
- Write a `scripts/fetch_breaking_changes.py` script that:
  - Queries GitHub releases API for target libraries (NumPy, pandas, scikit-learn, etc.)
  - Parses release notes for breaking change sections (heuristic: look for "deprecated", "removed", "breaking")
  - Outputs candidate rules in a structured format for human review
- This is a helper tool for maintainers, not automated rule ingestion (to preserve accuracy).

---

## Priority 4 — Documentation

### 4.1 Update `docs/envcheck-command-explanation.md` to Reflect Reality
**Problem:** The document describes 10 steps as if all implemented, but only 2 are. This is misleading.

**Action:**
- Clearly mark steps as "Implemented", "Planned", or "Future" with a status column.
- Add a "How to Add a New Rule" section with a step-by-step example.

### 4.2 Add a Troubleshooting Section to README
**Problem:** The README has no guidance on common failures (version detection not working, false positives, no findings for a known breaking change).

**Action:**
- Add a "Troubleshooting" section covering:
  - "No findings detected but I know the API changed" → how to add a custom rule
  - "False positive: my code doesn't actually use that pattern" → how to suppress
  - "Version detection shows wrong version" → how to manually specify versions

---

## Summary Roadmap

| Priority | Item | Effort |
|----------|------|--------|
| 1 | Expand knowledge base to 100+ rules | High |
| 1 | Add pre-commit hook integration | Low |
| 1 | Build unit test suite (≥70% coverage) | High |
| 1 | Return exit code 1 on findings | Low |
| 2 | Add RENAMED + MODULE_MOVED pattern types | Medium |
| 2 | Improve alias tracking in parser | Medium |
| 2 | Add error handling for version detection | Low |
| 2 | Add GitHub Actions CI workflow | Low |
| 2 | Replace print with structured logging | Low |
| 3 | Implement Cursor command steps 1–3 | Medium |
| 3 | Add `--fix` mode with replacement suggestions | Medium |
| 3 | Support OpenAI + Ollama providers | Medium |
| 3 | Knowledge base update helper script | Medium |
| 4 | Update documentation to reflect reality | Low |
| 4 | Add troubleshooting section to README | Low |
