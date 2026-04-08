# EnvCheck: AI-Powered Pre-Flight Environment Diagnostic

---

## Introduction

LLMs generate Python code based on training data that may be months or years old. As libraries evolve — functions get renamed, removed, or have their parameter signatures changed — the generated code "looks correct" but crashes at runtime. For example, asking an LLM to compute a trapezoidal integral with NumPy will very likely produce `np.trapz(y, x)`, a function **removed in NumPy 2.0**.

Existing tools (`pip check`, `mypy`, `ruff`, `pytest`) do not catch this class of error because they don't perform version-aware API compatibility analysis.

**EnvCheck** fills this gap: it performs **static analysis without executing code**, checking the actual installed library versions in a target environment against a curated knowledge base of known API breaking changes. It detects all compatibility issues in a single pass — before the code ever runs.

---

## Overall Function

EnvCheck runs a head-to-head comparison of two LLM-assisted coding workflows:

- **Scenario A (without EnvCheck):** LLM generates code → run → crash → send error to LLM → fix → run → crash again → ... Each crash only exposes one error, creating a multi-turn fix loop.
- **Scenario B (with EnvCheck):** LLM generates code → EnvCheck scans (no execution) → send all diagnostics to LLM → single precise fix → run successfully.

For a file with N compatibility issues, Scenario A needs at least N+1 LLM API calls. Scenario B always needs exactly 2 calls, regardless of N.

**Measured results across 7 test cases (Gemini 2.5-flash):**

| Metric | Without EnvCheck | With EnvCheck | Improvement |
|---|---|---|---|
| LLM API Calls | 16 | 12 | 25% fewer |
| Total Tokens | 8,382 | 5,758 | 31% reduction |
| Runtime Crashes | 10 | 0 | 100% eliminated |

**Knowledge base covers:** NumPy 2.0, SciPy 1.14, scikit-learn 1.2, pandas 2.0/2.2, NetworkX 3.0, Pydantic V1/V2 — 20+ rules.

---

## Code Structure

```
team-envcheck/
├── envcheck/                        # Core scanner engine (4-module pipeline)
│   ├── __init__.py
│   ├── parser.py                    # AST-based source code parser
│   │                                # Extracts: imports, alias map, attribute accesses,
│   │                                # method calls + kwargs — using Python's built-in ast module
│   ├── version_detector.py          # Reads exact installed package versions from a uv venv
│   │                                # Runs pip list --format=json in the target environment
│   ├── knowledge_base.py            # 20+ BreakingChangeRule entries
│   │                                # 4 pattern types: ATTRIBUTE, IMPORT, METHOD_CALL, METHOD_ACCESS
│   └── scanner.py                   # Matching engine — ties parser + versions + KB together
│                                    # Produces ScanReport with Finding list (file, line, fix)
├── test_cases/
│   └── cases.py                     # 7 test case definitions (prompt, broken code, fixed code)
│                                    # Cases: numpy_2x, scipy_114, sklearn_12, pandas_22,
│                                    #        networkx_3x, pandas_15, pydantic_v1
├── environments/                    # Per-case isolated uv virtual environments (gitignored)
├── reports/                         # Generated comparison reports (gitignored)
├── main.py                          # Test runner and --eval metrics mode
├── demo.py                          # Workflow comparison demo (no LLM API needed)
├── demo_llm.py                      # Full LLM demo (Claude or Gemini) — primary script
├── docs/
│   ├── demo-walkthrough.md          # Detailed presentation guide with step-by-step traces
│   └── envcheck-command-explanation.md  # Explanation of the CLI interface
├── pyproject.toml                   # Project metadata (requires Python ≥ 3.12)
└── uv.lock                          # Locked dependency versions
```

**Scanner pipeline:**
```
Source Code → parser.py → version_detector.py → knowledge_base.py → scanner.py → ScanReport
              (AST parse)  (read installed vers)  (breaking change KB)  (4 matchers)
```

---

## Installation

**Requirements:** Python 3.12+, `uv`

```bash
cd team-envcheck
uv sync    # installs anthropic>=0.83.0 and google-genai>=1.64.0
```

Set API keys (only needed for LLM demo modes):
```bash
export ANTHROPIC_API_KEY="sk-ant-..."    # for Claude provider
export GEMINI_API_KEY="AIza..."          # for Gemini provider
```

---

## How to Run

### Interactive LLM demo (primary script)

Runs a head-to-head comparison using a real LLM API:

```bash
# Single case — interactive (press Enter to advance each step)
uv run python demo_llm.py --provider gemini --case pandas_22

# Single case — auto-advance (2s delays, good for recorded demos)
uv run python demo_llm.py --provider gemini --case pandas_22 --auto

# All 7 cases — saves Markdown + JSON report
uv run python demo_llm.py --provider gemini --all --report

# Mock mode — no API key needed, simulates realistic multi-turn behavior
uv run python demo_llm.py --mock --case pandas_22
uv run python demo_llm.py --all --mock --report

# List available cases
uv run python demo_llm.py --list
```

**Provider options:**
```bash
# Claude (default)
uv run python demo_llm.py --provider claude --case numpy_2x

# Gemini
uv run python demo_llm.py --provider gemini --case numpy_2x

# Specific model
uv run python demo_llm.py --provider gemini --model gemini-2.5-pro --case numpy_2x
```

### Workflow demo (no LLM API needed)

```bash
uv run python demo.py
```

### Test runner and evaluation metrics

```bash
uv run python main.py
uv run python main.py --eval    # outputs metrics CSV
```

### Development

```bash
uv run pytest           # run tests
uv run ruff check       # lint
uv run ruff format      # format
```

### Available Test Cases

| Case ID | Library | Breaking Change |
|---|---|---|
| `numpy_2x` | NumPy ≥ 2.0 | `np.trapz`, `np.infty` removed |
| `scipy_114` | SciPy ≥ 1.14 | `cumtrapz`, `simps` removed |
| `sklearn_12` | scikit-learn ≥ 1.2 | `load_boston` removed |
| `pandas_22` | pandas ≥ 2.2 | `fillna(method=...)` kwarg + `.mad()` removed |
| `networkx_3x` | NetworkX ≥ 3.0 | `write_gpickle` / `read_gpickle` removed |
| `pandas_15` | pandas == 1.5.3 | `df.map()` didn't exist until 2.1 |
| `pydantic_v1` | Pydantic == 1.10 | `.model_dump()` is a V2 API |

---

## References

- **docs/demo-walkthrough.md:** Complete presentation guide with step-by-step traces of Scenario A and B for the `pandas_22` case
- **docs/envcheck-command-explanation.md:** CLI interface documentation
- Python `ast` module: used for AST-based parsing without code execution
- `packaging.version.Version`: used for correct semantic version comparison
