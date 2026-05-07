# EnvCheck Demo Walkthrough — Presentation Guide

This document is a comprehensive guide for presenting the EnvCheck demo. It explains **what the code does**, **why it does it**, and **what results to expect** at every stage. Use it as a script you can follow while running the live demo.

---

## Table of Contents

1. [The Problem Statement](#1-the-problem-statement)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Test Case Design — Why the Prompts Cause Failures](#3-test-case-design--why-the-prompts-cause-failures)
4. [The Scanner Engine — How Detection Works](#4-the-scanner-engine--how-detection-works)
5. [Demo Script Walkthrough (`demo_llm.py`)](#5-demo-script-walkthrough-demo_llmpy)
6. [Step-by-Step: What Happens When You Run a Case](#6-step-by-step-what-happens-when-you-run-a-case)
7. [Understanding the Results](#7-understanding-the-results)
8. [Commands Quick Reference](#8-commands-quick-reference)

---

## 1. The Problem Statement

### What Goes Wrong

LLMs generate Python code based on training data that may be months or years old. Python libraries evolve — functions get renamed, removed, or have their parameter signatures changed. The generated code "looks correct" but fails at runtime.

**Example:** If you ask an LLM to compute a trapezoidal integral with NumPy, it will very likely write `np.trapz(y, x)` — a function that was **removed in NumPy 2.0** and replaced by `np.trapezoid(y, x)`. The code compiles, passes syntax checks, but crashes when executed in any environment with NumPy ≥ 2.0.

### Why Existing Tools Don't Help

| Tool | What It Checks | Why It Misses This |
|------|---------------|-------------------|
| `pip check` | Transitive dependency conflicts | Doesn't look at API-level usage |
| `mypy` | Type annotations | Doesn't know which functions exist in which version |
| `ruff` / `pylint` | Code style, common errors | No version-aware API knowledge |
| `pytest` | Test assertions | Tells you tests fail, not *why* the API broke |

### What EnvCheck Does

EnvCheck performs **static analysis** (no code execution) against the **actual installed library versions** in a target environment. It detects API-level mismatches before the code ever runs.

---

## 2. System Architecture Overview

EnvCheck consists of four modules that form a pipeline:

```
Source Code ──→ AST Parser ──→ Version Detector ──→ Knowledge Base ──→ Diagnostic Report
                (parser.py)  (version_detector.py) (knowledge_base.py)   (scanner.py)
```

### Module 1: `envcheck/parser.py` — AST-Based Source Code Parser

**What it does:** Takes raw Python source code and extracts three types of structured information using Python's built-in `ast` (Abstract Syntax Tree) module.

**Why AST instead of regex:** Regular expressions would be fragile — they'd break on multi-line statements, comments, or non-standard formatting. The AST approach works on any syntactically valid Python code regardless of formatting, because it operates on the parsed tree structure, not raw text.

**What it extracts:**

| Data Type | Example Source | Extracted Structure |
|-----------|---------------|-------------------|
| **Imports** | `import pandas as pd` | `ImportInfo(module="pandas", alias="pd")` |
| **Imports (from)** | `from sklearn.datasets import load_boston` | `ImportInfo(module="sklearn.datasets", name="load_boston", is_from_import=True)` |
| **Attribute Access** | `np.trapz` | `AttributeAccess(object_name="np", attribute="trapz")` |
| **Method Calls** | `df.fillna(method="ffill")` | `MethodCall(object_name="df", method_name="fillna", keyword_args={"method": "ffill"})` |

**Key implementation detail — alias tracking:**  
The parser builds an `alias_map` dictionary (e.g., `{"np": "numpy", "pd": "pandas"}`). When the scanner later sees `np.trapz`, it resolves `np` → `numpy` via this map, so it knows to look up NumPy's breaking change rules.

**Key implementation detail — subscript handling:**  
For `df["A"].mad()`, the AST node for `df["A"]` is an `ast.Subscript`, not an `ast.Name`. The `_get_root_name()` function recursively traverses through `Subscript` nodes to find the root variable name (`df`), so the parser correctly identifies this as a method call on a pandas object.

### Module 2: `envcheck/version_detector.py` — Environment Version Reader

**What it does:** Reads the exact installed package versions from a **specific virtual environment** (not the system Python).

**Why this matters:** The whole point of EnvCheck is detecting mismatches between the *code's assumptions* and the *environment's reality*. We need to know that the target environment has `pandas==2.2.3`, not just "some version of pandas."

**How it works:**  
1. Takes a path to a `uv` virtual environment (e.g., `environments/case_pandas_22/`)
2. Runs `pip list --format=json` using that environment's Python binary
3. Parses the JSON output into a dictionary: `{"pandas": InstalledPackage(version="2.2.3"), ...}`

**Version comparison logic:**  
The `is_version_affected()` function checks: is the installed version ≥ the version where the API was removed? If yes, the API doesn't exist in this environment.

```python
# Example: pandas 2.2.3 installed, fillna(method=) removed in 2.2.0
is_version_affected("2.2.3", "2.2.0")  # → True (2.2.3 >= 2.2.0, API is gone)
```

Uses the `packaging.version.Version` class for correct semantic version comparison (so `1.14.0` > `1.2.0`, not `1.14.0` < `1.2.0` which naive string comparison would give).

### Module 3: `envcheck/knowledge_base.py` — Breaking Changes Registry

**What it does:** Stores a curated database of known API breaking changes, each described as a `BreakingChangeRule` with enough information to both *detect* the issue and *explain* the fix.

**Why a curated KB instead of automated extraction:** Automated approaches (e.g., diffing API surfaces between versions) would generate thousands of entries with many false positives. A curated KB trades coverage for precision — each rule has been verified and includes a human-readable fix suggestion.

**Each rule specifies:**

| Field | Purpose | Example |
|-------|---------|---------|
| `rule_id` | Unique identifier | `"pandas-fillna-method-removed"` |
| `library` | PyPI package name | `"pandas"` |
| `removed_in` | Version boundary | `"2.2.0"` |
| `pattern_type` | How to detect it | `PatternType.METHOD_CALL` |
| `symbol` | Function/method name | `"fillna"` |
| `method_kwargs` | Specific kwargs that trigger | `{"method": None}` |
| `old_api` / `new_api` | Before/after | `df.fillna(method="ffill")` → `df.ffill()` |

**Four pattern types:**

1. **ATTRIBUTE** — `module.function` removal (e.g., `np.trapz`, `nx.write_gpickle`)
2. **IMPORT** — `from module import name` removal (e.g., `from sklearn.datasets import load_boston`)
3. **METHOD_CALL** — A method still exists but a specific **parameter** was removed (e.g., `fillna(method=...)`)
4. **METHOD_ACCESS** — A method was removed entirely from an object (e.g., `.mad()`, `.model_dump()`)

**Current coverage:** 20+ rules across NumPy 2.0, SciPy 1.14, scikit-learn 1.2, pandas 2.0/2.2, NetworkX 3.0, and Pydantic V1/V2.

### Module 4: `envcheck/scanner.py` — The Matching Engine

**What it does:** Ties everything together. Takes the parser output + installed versions + knowledge base rules, and runs four matching passes to produce a diagnostic report.

**Why four separate matchers:** Each pattern type requires different matching logic:

- **`_match_attribute_rules`**: Resolves aliases via `alias_map`, then checks if `resolved_module.attribute` matches any rule. Example: `np.trapz` → resolve `np` to `numpy` → match against rule `numpy.trapz`.

- **`_match_import_rules`**: Checks `from X import Y` statements against rules. Example: `from sklearn.datasets import load_boston` → match `module=sklearn.datasets, symbol=load_boston`.

- **`_match_method_call_rules`**: Checks method calls AND their keyword arguments. Example: `df.fillna(method="ffill")` → match `symbol=fillna` AND `"method" in kwargs`. This is critical — `fillna()` itself still exists in pandas 2.2, only the `method=` parameter was removed.

- **`_match_method_access_rules`**: Checks if a method name is called on any object, with a heuristic check that the relevant library is imported. Example: `.mad()` called anywhere in a file that imports `pandas`.

**Output:** A `ScanReport` containing a list of `Finding` objects, each with: file path, line number, the matched code, the rule that triggered, the installed version, and severity.

---

## 3. Test Case Design — Why the Prompts Cause Failures

Each test case has a carefully crafted **prompt** designed to reliably trigger LLMs into generating outdated API calls. The prompts use **specific phrasing** that steers the LLM toward the old API.

### All 7 Cases

| # | Case ID | Library | What the Prompt Tricks the LLM Into | Why It Breaks |
|---|---------|---------|-------------------------------------|---------------|
| 1 | `numpy_2x` | NumPy ≥ 2.0 | Using `np.trapz()` and `np.infty` | Both removed in NumPy 2.0 → `AttributeError` |
| 2 | `scipy_114` | SciPy ≥ 1.14 | Importing `cumtrapz` and `simps` | Both removed in SciPy 1.14 → `ImportError` |
| 3 | `sklearn_12` | scikit-learn ≥ 1.2 | Loading `load_boston` dataset | Removed for ethical concerns → `ImportError` |
| 4 | `pandas_22` | pandas ≥ 2.2 | Using `fillna(method="ffill")` and `.mad()` | `method=` kwarg removed in 2.2, `.mad()` removed in 2.0 |
| 5 | `networkx_3x` | NetworkX ≥ 3.0 | Using `nx.write_gpickle()` / `read_gpickle()` | Removed for security concerns → `AttributeError` |
| 6 | `pandas_15` | pandas == 1.5.3 | Using `df.map()` (reverse-compat) | `.map()` didn't exist until pandas 2.1 → `AttributeError` |
| 7 | `pydantic_v1` | Pydantic == 1.10 | Using `user.model_dump()` | `.model_dump()` is a Pydantic V2 API → `AttributeError` |

### Prompt Engineering Techniques

The prompts deliberately use language that maps to the old API names:

- **numpy_2x:** *"numpy's built-in trapezoidal rule function"* → LLM outputs `np.trapz`, not `np.trapezoid`
- **numpy_2x:** *"numpy's explicit verbose infinity alias"* → LLM outputs `np.infty`, not `np.inf`
- **pandas_22:** *"explicitly passing the forward fill method argument into the fillna function"* → forces `fillna(method="ffill")` instead of `df.ffill()`
- **pandas_22:** *"pandas' built-in mad function"* → LLM calls `.mad()`, which was removed
- **sklearn_12:** *"the classic Boston Housing dataset"* → LLM uses `load_boston`, which was removed

### Multi-Issue Cases Are Key

Cases with **multiple breaking changes** (`numpy_2x`, `scipy_114`, `pandas_22`, `networkx_3x`) are the most impactful for the demo because:

1. When code crashes, the traceback **only shows the first error** (Python stops at the first exception)
2. The LLM fixes that one error, but the **second error is still hidden**
3. The code crashes again → another round trip to the LLM
4. This creates a **multi-turn fix loop** that wastes tokens and API calls

EnvCheck finds **all issues at once** via static analysis, enabling a single targeted fix.

---

## 4. The Scanner Engine — How Detection Works

### Complete Detection Flow for `pandas_22`

Let's trace exactly what happens when EnvCheck scans this broken code:

```python
import pandas as pd

df = pd.DataFrame({"A": [1.0, None, 3.0, None, 5.0]})
df = df.fillna(method="ffill")     # ← Problem 1
mad_value = df["A"].mad()          # ← Problem 2
print(f"MAD: {mad_value}")
```

**Step 1 — AST Parsing** produces:
```
imports:       [ImportInfo(module="pandas", alias="pd")]
alias_map:     {"pd": "pandas"}
method_calls:  [
    MethodCall(object="df", method="fillna", kwargs={"method": "ffill"}, line=4),
    MethodCall(object="df", method="mad", kwargs={}, line=5),
]
```

**Step 2 — Version Detection** reads from `environments/case_pandas_22/`:
```
packages: {"pandas": InstalledPackage(version="2.2.3"), ...}
```

**Step 3 — Rule Matching:**

*Matcher: `_match_method_call_rules`*
- Iterates over method_calls, finds `fillna` at line 4
- Checks rule `pandas-fillna-method-removed`: symbol matches (`fillna`), kwargs match (`"method"` is present)
- Checks version: `is_version_affected("2.2.3", "2.2.0")` → `True`
- **Finding generated:** Line 4, `fillna(method=ffill)`, fix: use `df.ffill()`

*Matcher: `_match_method_access_rules`*
- Iterates over method_calls, finds `mad` at line 5
- Checks rule `pandas-mad-removed`: symbol matches (`mad`)
- Checks if `pandas` is imported in this file → `True` (via `import pandas as pd`)
- Checks version: `is_version_affected("2.2.3", "2.0.0")` → `True`
- **Finding generated:** Line 5, `df.mad()`, fix: compute manually

**Step 4 — Report Output:**
```
EnvCheck Scan Report
============================================================
Files scanned: 1
Findings: 2 (2 errors, 0 warnings)
Scan time: 390ms

⛔ Line 4: df.fillna(method=ffill)
   fillna(method=...) keyword removed in pandas 2.2.
   Fix: df.fillna(method="ffill") → df.ffill()

⛔ Line 5: df.mad()
   .mad() was removed in pandas 2.0.
   Fix: df["col"].mad() → (df["col"] - df["col"].mean()).abs().mean()
```

All of this happens **without executing a single line of the user's code**.

---

## 5. Demo Script Walkthrough (`demo_llm.py`)

### What the Script Does

`demo_llm.py` runs a **head-to-head comparison** of two workflows using a real LLM API (Claude or Gemini):

**Scenario A — Without EnvCheck (reactive error-fix loop):**
```
Prompt → LLM generates code → Run → 💥 CRASH → Send error to LLM → Fix → Run → 💥 CRASH again → ...
```
Each crash only reveals one error. The LLM must fix them one at a time.

**Scenario B — With EnvCheck (proactive scan-then-fix):**
```
Prompt → LLM generates code → 🛡️ EnvCheck scan → Send diagnostics to LLM → One precise fix → ✅ Run successfully
```
EnvCheck finds all issues at once. The LLM fixes them all in one shot.

### Key Code Components

#### Data Tracking (`LLMCall`, `ScenarioResult`)

Every LLM API call is recorded as an `LLMCall` dataclass:
```python
@dataclass
class LLMCall:
    role: str               # "generate", "fix_from_error", "fix_from_envcheck"
    input_tokens: int       # Tokens sent to the LLM
    output_tokens: int      # Tokens received from the LLM
    latency_ms: float       # Round-trip time
```

Each scenario produces a `ScenarioResult` that aggregates all calls:
```python
@dataclass
class ScenarioResult:
    llm_calls: list[LLMCall]     # All API calls made
    execution_attempts: int       # How many times we tried to run the code
    runtime_crashes: int          # How many times it crashed
    envcheck_time_ms: float       # Time spent on EnvCheck scan (Scenario B only)
    envcheck_findings: int        # Issues found by scanner (Scenario B only)
```

This gives us precise metrics for comparison: total tokens, total calls, total crashes, latency, and estimated cost.

#### Environment Setup (`setup_environment`)

Each test case specifies an exact `pip install` command (e.g., `pip install "pandas>=2.2.0"`). The `setup_environment()` function:
1. Parses the environment string to extract package specs and optional `--python` version
2. Creates an isolated `uv` virtual environment in `environments/case_{id}/`
3. Installs the specified packages into that environment
4. Returns the environment path for code execution

This ensures each case runs against the **exact library version** that triggers the breaking change.

#### LLM API Abstraction (`call_llm`)

The `call_llm()` function dispatches to the correct provider:
```python
def call_llm(system_prompt, user_prompt, role_label):
    if MOCK_MODE:
        return _mock_call(...)      # Simulated response, no API key needed
    if PROVIDER == "claude":
        return _call_claude(...)    # Anthropic SDK
    elif PROVIDER == "gemini":
        return _call_gemini(...)    # Google GenAI SDK
```

Both real providers return actual token counts from the API response metadata.

#### Mock Mode and Partial Fixes

For offline demos or testing, `--mock` mode simulates LLM responses without API calls. The key design decision: **mock mode simulates realistic multi-turn behavior**.

The `MOCK_PARTIAL_FIXES` dictionary defines intermediate code states for multi-issue cases. For example, in `pandas_22`:
- Crash 1: `fillna(method="ffill")` fails → LLM fixes only this → code now has `df.ffill()` but still has `.mad()`
- Crash 2: `.mad()` fails → LLM fixes this → code is now fully correct

This makes the mock demo show **realistic token and call differences** between Scenario A and B.

### Scenario A — Detailed Code Flow

```python
def run_scenario_a(case, env_path):
    # Step A.1: Call LLM to generate code
    response = call_llm(system_prompt, case.problem, "generate")
    code = extract_code_from_response(response)

    # Step A.2+: Run → crash → fix loop
    while loop < MAX_FIX_LOOPS:
        result = run_in_env(env_path, code)

        if result.returncode == 0:
            break  # ✅ Success

        # 💥 Crashed — send error back to LLM
        fix_prompt = f"Code:\n{code}\n\nError:\n{error}\n\nPlease fix."
        response = call_llm(system_prompt, fix_prompt, "fix_from_error")
        code = extract_code_from_response(response)
```

**Why this accumulates tokens:** Each fix call includes the full code + the full error traceback in the prompt. As the code gets longer and errors get more complex, input tokens grow with each iteration.

### Scenario B — Detailed Code Flow

```python
def run_scenario_b(case, env_path):
    # Step B.1: Call LLM to generate code (same prompt as A)
    response = call_llm(system_prompt, case.problem, "generate")
    code = extract_code_from_response(response)

    # Step B.2: EnvCheck scan — NO code execution
    scan_report = scan_source(code, env_path=str(env_path))

    if scan_report.total_findings > 0:
        # Step B.3: Send code + EnvCheck diagnostics to LLM
        diagnostic = format_findings(scan_report)
        fix_prompt = f"Code:\n{code}\n\nIssues:\n{diagnostic}\n\nFix ALL issues."
        response = call_llm(system_prompt, fix_prompt, "fix_from_envcheck")
        code = extract_code_from_response(response)

    # Step B.4: Run — first and only execution
    result = run_in_env(env_path, code)
```

**Why this is more efficient:** The diagnostic sent to the LLM contains precise information — line numbers, exact old API → new API mappings. The LLM doesn't have to guess what went wrong from an error traceback; it receives a structured fix guide.

---

## 6. Step-by-Step: What Happens When You Run a Case

### Running the Interactive Demo

```bash
uv run python demo_llm.py --provider gemini --case pandas_22
```

Here is exactly what happens at each step:

### Step 1: Environment Setup
- Creates `environments/case_pandas_22/` with `uv venv`
- Installs `pandas>=2.2.0` into that isolated environment
- The environment now has `pandas 2.2.3` (latest in the 2.2.x line)

### Step 2: Scenario A Begins

**A.1 — LLM generates code:**
- System prompt: *"Write ONLY the Python code, wrapped in a \`\`\`python block."*
- User prompt: The test case's `problem` field (asks for `fillna` with method argument + `.mad()`)
- **Result:** LLM generates code with `df.fillna(method="ffill")` and `df["A"].mad()`
- **Tokens used:** ~130 input + ~200 output = ~330 total

**A.2 — First execution attempt:**
- Writes the generated code to a temp `.py` file
- Executes it using the environment's Python: `environments/case_pandas_22/bin/python script.py`
- **Result:** 💥 CRASH — `TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'`
- Python stopped at line 6. The `.mad()` issue on line 11 was never reached.

**A.3 — Send error to LLM (fix call #1):**
- Prompt includes: full code + full error traceback
- *"I ran this code and got this error. Please fix."*
- **Result:** LLM replaces `df.fillna(method="ffill")` with `df.ffill()`. But `.mad()` is still there — the LLM had no way to know about it since it never crashed on that line.
- **Tokens used:** ~360 input + ~220 output = ~580 total

**A.4 — Second execution attempt:**
- Runs the partially-fixed code
- **Result:** 💥 CRASH — `AttributeError: 'Series' object has no attribute 'mad'`
- Now a *different* error on a *different* line.

**A.5 — Send error to LLM (fix call #2):**
- Prompt includes: the updated code + the new error traceback
- **Result:** LLM replaces `.mad()` with `(df["A"] - df["A"].mean()).abs().mean()`
- **Tokens used:** ~470 input + ~270 output = ~740 total

**A.6 — Third execution attempt:**
- Runs the fully-fixed code
- **Result:** ✅ SUCCESS

**Scenario A total: 3 API calls, 2 crashes, ~1,650 tokens**

### Step 3: Scenario B Begins

**B.1 — LLM generates code (same prompt):**
- Identical prompt to A.1
- **Result:** LLM generates similar code with the same two issues
- **Tokens used:** ~130 input + ~200 output = ~330 total

**B.2 — EnvCheck scans the code:**
- `scan_source()` is called — **no code is executed**
- AST parser extracts: `fillna(method="ffill")` at line 6, `.mad()` at line 11
- Version detector reads: `pandas 2.2.3` from the environment
- Rule matcher finds **2 issues** in ~400ms:
  - Line 6: `fillna(method=...)` removed in 2.2.0 → use `df.ffill()`
  - Line 11: `.mad()` removed in 2.0.0 → compute manually
- **Both issues found in a single pass, before any execution**

**B.3 — Send diagnostics to LLM (fix call):**
- Prompt includes: full code + structured EnvCheck diagnostics:
  ```
  EnvCheck found the following API compatibility issues:
  - Line 6: `df.fillna(method=ffill)` — removed in pandas 2.2. Fix: use df.ffill()
  - Line 11: `df.mad()` — removed in pandas 2.0. Fix: compute manually
  ```
- **Result:** LLM fixes **both issues in one shot** because the diagnostics are precise and complete
- **Tokens used:** ~410 input + ~260 output = ~670 total

**B.4 — First and only execution:**
- Runs the fixed code
- **Result:** ✅ SUCCESS — first run, zero crashes

**Scenario B total: 2 API calls, 0 crashes, ~1,000 tokens**

### Step 4: Comparison Table

The demo then displays a side-by-side comparison table with metrics from both scenarios.

---

## 7. Understanding the Results

### Real Gemini API Results (7 cases, `gemini-2.5-flash`)

| Metric | Without EnvCheck (A) | With EnvCheck (B) | Improvement |
|--------|---------------------|-------------------|-------------|
| LLM API Calls | 16 | 12 | 4 fewer (25%) |
| Total Tokens | 8,382 | 5,758 | 2,624 saved (31%) |
| Runtime Crashes | 10 | 0 | 10 prevented (100%) |
| Est. API Cost | $0.0029 | $0.0022 | $0.0007 saved |

### Why Some Cases Show No Difference

Two cases (`networkx_3x` and `pandas_15`) may show Scenario A with **0 crashes** — meaning the LLM happened to generate correct code on the first try for those cases. When the LLM already generates compatible code, EnvCheck adds a scan step that doesn't save anything. This is expected and honest — EnvCheck's value scales with the **frequency and complexity of API mismatches**.

### Why `sklearn_12` Is the Most Dramatic Case

In our test run, `sklearn_12` showed **5 crashes in Scenario A** with **5 API calls**. This happened because:
1. Gemini tried to fix the `load_boston` removal but introduced new errors
2. Each fix attempt introduced a different problem (wrong import path, wrong function signature, etc.)
3. The reactive loop kept going until the max retry limit

Meanwhile, Scenario B: EnvCheck identified the issue in one scan, the LLM received a precise diagnostic, and fixed it correctly on the first try — **2 API calls, 0 crashes**.

### Key Insight: The Multi-Error Problem

The most important insight from the demo is the **multi-error problem**:

> When code has N API compatibility issues, the reactive loop (Scenario A) requires **at least N+1 API calls** because each crash only reveals one error. EnvCheck (Scenario B) always requires exactly **2 API calls** regardless of N, because static analysis finds all issues simultaneously.

This means EnvCheck's advantage **grows with code complexity** — the more compatibility issues in a single file, the more tokens and API calls it saves.

### Token Savings Breakdown

Where do the token savings come from?

1. **Fewer API calls** — Each eliminated call saves both input and output tokens
2. **No error tracebacks in prompts** — Scenario A includes full tracebacks (~200-500 tokens each) in fix prompts. Scenario B sends concise diagnostics (~50-100 tokens)
3. **No code accumulation** — In Scenario A, each fix prompt includes the full code + traceback, and the code may grow as the LLM adds comments/fixes. Scenario B's diagnostic is compact

---

## 8. Commands Quick Reference

### Interactive Demo (Best for Live Presentations)

```bash
# Pick the most impactful case for a live demo
uv run python demo_llm.py --provider gemini --case pandas_22

# This shows step-by-step with Enter-to-advance pauses:
# 1. Scenario A: generate → crash → fix → crash → fix → success
# 2. Scenario B: generate → scan → fix → success (first run!)
# 3. Side-by-side comparison table
```

### Auto-Advance Mode (For Recorded Demos)

```bash
# Same flow but advances automatically with 2s delays
uv run python demo_llm.py --provider gemini --case pandas_22 --auto
```

### All Cases + Report (For Data Collection)

```bash
# Run all 7 cases, print summary table, save Markdown + JSON report
uv run python demo_llm.py --provider gemini --all --report
# Reports saved to reports/ directory
```

### Mock Mode (Offline, No API Key Needed)

```bash
# Simulates LLM responses using predefined broken/fixed code
uv run python demo_llm.py --mock --case pandas_22
uv run python demo_llm.py --all --mock --report
```

### List Available Cases

```bash
uv run python demo_llm.py --list
```

### Provider Options

```bash
# Claude (default)
export ANTHROPIC_API_KEY="sk-ant-..."
uv run python demo_llm.py

# Gemini
export GEMINI_API_KEY="AIza..."
uv run python demo_llm.py --provider gemini

# Specific model
uv run python demo_llm.py --provider gemini --model gemini-2.5-pro
```

---

## Appendix: File Structure Summary

```
EnvCheck/
├── envcheck/                        # Core scanner engine
│   ├── __init__.py
│   ├── parser.py                    # AST-based source code parser
│   ├── version_detector.py          # Reads installed versions from uv envs
│   ├── knowledge_base.py            # 20+ breaking change rules
│   └── scanner.py                   # Matching engine, produces ScanReport
├── test_cases/
│   └── cases.py                     # 7 test case definitions (prompt + broken/fixed code)
├── environments/                    # Per-case isolated uv virtual environments (gitignored)
├── reports/                         # Generated reports (gitignored)
├── main.py                          # Test runner + --eval metrics mode
├── demo.py                          # Workflow comparison demo (no LLM API)
├── demo_llm.py                      # LLM API demo (Claude/Gemini) — primary presentation script
└── pyproject.toml                   # Project metadata and dependencies
```
