---
description: Perform a comprehensive diagnostic scan of the codebase to detect API compatibility issues, dependency conflicts, version mismatches, and runtime environment problems before execution.
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Goal

Proactively identify and report issues that would prevent the codebase from running correctly in the current environment. This includes API breaking changes, dependency version conflicts, deprecated usage patterns, missing dependencies, and Python version incompatibilities. The command produces a structured diagnostic report WITHOUT modifying any files.

## Operating Constraints

**STRICTLY READ-ONLY**: Do **not** modify any source files, configuration files, or dependency files. Output a structured diagnostic report only. Offer remediation suggestions that the user must explicitly approve before any changes.

**Scope Control**: If `$ARGUMENTS` specifies a subdirectory or file list, limit scanning to those paths. Otherwise, scan the entire project.

## Execution Steps

### 1. Environment Discovery

Gather current environment information by running these commands from the project root:

```bash
# Python version
python --version 2>&1 || python3 --version 2>&1

# Package manager detection & installed packages
pip list --format=json 2>/dev/null || pip3 list --format=json 2>/dev/null
# Or if uv is used:
uv pip list 2>/dev/null

# Check for virtual environment
echo "VIRTUAL_ENV=$VIRTUAL_ENV"
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"

# OS and platform info
uname -a 2>/dev/null || ver 2>/dev/null
```

Record:
- **Python version**: exact version (e.g., 3.12.1)
- **Package manager**: pip / uv / conda / poetry
- **Virtual environment**: active or not, path
- **Installed packages**: name → version mapping
- **OS/Platform**: Linux/macOS/Windows, architecture

### 2. Dependency Specification Analysis

Load and parse dependency specifications from the project:

**Detection priority order**:
1. `pyproject.toml` → `[project.dependencies]`, `[tool.uv.dev-dependencies]`, `[tool.poetry.dependencies]`
2. `requirements.txt` / `requirements-dev.txt`
3. `setup.py` / `setup.cfg`
4. `Pipfile` / `Pipfile.lock`
5. `conda.yaml` / `environment.yml`

For each declared dependency, extract:
- Package name
- Version constraint (e.g., `>=1.30.0`, `>=1.24,<2.0`)
- Whether it's a dev dependency

### 3. Dependency Conflict Detection

#### A. Declared vs Installed Version Mismatch

For each declared dependency:
- Check if it is installed in the current environment
- If installed, verify the installed version satisfies the declared constraint
- Flag: **MISSING** (not installed), **VERSION_MISMATCH** (installed but outside constraint), **OK** (satisfies constraint)

#### B. Transitive Dependency Conflicts

Run dependency resolution check:

```bash
# pip check for broken dependencies
pip check 2>&1

# Or with uv:
uv pip check 2>/dev/null
```

Parse output for:
- Packages with incompatible version requirements from different dependents
- Missing transitive dependencies
- Circular dependency warnings

#### C. Lock File Consistency

If lock file exists (`uv.lock`, `poetry.lock`, `Pipfile.lock`):
- Check if lock file is in sync with the dependency specification
- Flag if lock file is stale (dependency spec modified after lock file)

### 4. API Compatibility Scan

This is the **core analysis phase**. For each source file (`.py`) in the project:

#### A. Import Analysis

- Extract all import statements (`import X`, `from X import Y`)
- Map each import to the installed package and its version
- Cross-reference against known API changes for major version boundaries

#### B. Deprecated API Detection

For high-priority libraries detected in the project, check for known deprecated patterns:

**NumPy** (if installed):
- `np.bool`, `np.int`, `np.float`, `np.complex`, `np.object`, `np.str` → removed in NumPy 1.24+
- `np.mat` → deprecated
- `numpy.distutils` → removed in NumPy 2.0
- `np.AxisError` location changes in 2.0

**Pandas** (if installed):
- `DataFrame.append()` → removed in 2.0 (use `pd.concat`)
- `Series.swaplevel` argument changes
- `DataFrame.iteritems()` → renamed to `DataFrame.items()`
- Datetime-related deprecations

**Streamlit** (if installed):
- `st.cache` → deprecated, use `st.cache_data` or `st.cache_resource`
- `st.experimental_*` → many moved to stable API
- `st.beta_*` → removed
- Check `streamlit` API changes between minor versions

**Plotly** (if installed):
- `plotly.plotly` → deprecated in favor of `plotly.io`
- Template and theme API changes

**Scikit-learn** (if installed):
- `sklearn.utils.safe_indexing` → removed
- `sklearn.externals.joblib` → use `joblib` directly
- Parameter renaming across versions

**Polars** (if installed):
- Rapid API evolution between 0.x versions
- `lazy()` vs `scan_*` patterns
- Expression API changes

**General Python**:
- `asyncio.coroutine` decorator → removed in 3.11
- `collections.Mapping` etc. → moved to `collections.abc`
- `typing.Dict`, `typing.List` etc. → use `dict`, `list` in 3.9+
- `distutils` → removed in 3.12
- `imp` module → removed, use `importlib`
- `unittest.findTestCases` etc. changes

#### C. Static Import Verification

For each import statement found:
1. Attempt to verify the imported symbol exists in the installed version
2. Use the following strategy:
   - Run targeted Python one-liners to verify importability:
     ```bash
     python -c "from <module> import <symbol>" 2>&1
     ```
   - Only check imports that are flagged as potentially problematic (known API changes, major version boundaries)
   - Limit to max 30 verification checks to avoid excessive runtime

#### D. Type Hint Compatibility

If the project uses type hints:
- Check for `from __future__ import annotations` usage consistency
- Verify type hint syntax is compatible with the declared Python version
- Flag Python 3.10+ syntax (`X | Y` union types) if targeting older Python

### 5. Python Version Compatibility Check

- Compare declared `requires-python` with the active Python version
- Scan for syntax features that require specific Python versions:
  - Match statements (`match/case`) → Python 3.10+
  - Walrus operator (`:=`) → Python 3.8+
  - f-strings → Python 3.6+
  - `X | Y` type unions → Python 3.10+
  - `ExceptionGroup` → Python 3.11+
  - `tomllib` → Python 3.11+ (stdlib)
- Flag if active Python version doesn't meet `requires-python`

### 6. Configuration & Tooling Check

#### A. Linter/Formatter Configuration
- If `ruff` configured: verify ruff version compatibility with config format
- If `mypy` configured: check mypy config compatibility and plugin versions
- If `pytest` configured: check pytest version and plugin compatibility

#### B. Build System
- Verify build backend compatibility (`hatchling`, `setuptools`, `flit`, etc.)
- Check if `pyproject.toml` structure follows current PEP standards

### 7. Runtime Smoke Test (Optional)

If user requests deep check via `$ARGUMENTS` containing "deep" or "full":

```bash
# Attempt to import the main module
python -c "import src" 2>&1

# Run a minimal import chain test
python -c "
import sys
errors = []
# Try importing all project modules
import importlib, pathlib
for f in pathlib.Path('src').rglob('*.py'):
    module = str(f).replace('/', '.').replace('.py', '')
    try:
        importlib.import_module(module)
    except Exception as e:
        errors.append(f'{module}: {e}')
for e in errors:
    print(f'IMPORT_ERROR: {e}')
" 2>&1
```

Parse and categorize errors:
- **ImportError**: Missing dependency or wrong version
- **AttributeError**: API changed, symbol no longer exists
- **SyntaxError**: Python version incompatibility
- **Other**: Unexpected issues

### 8. Produce Diagnostic Report

Output a structured Markdown report with the following sections:

```markdown
## Environment Diagnostic Report

### Environment Summary

| Property | Value |
|----------|-------|
| Python Version | 3.12.x |
| Virtual Env | /path/to/venv (active) |
| Package Manager | uv |
| OS / Platform | Linux x86_64 |
| Scan Scope | Full project / [specified paths] |

### Dependency Status

| Package | Declared | Installed | Status | Notes |
|---------|----------|-----------|--------|-------|
| streamlit | >=1.30.0 | 1.32.0 | ✅ OK | |
| numpy | >=1.24,<2.0 | 1.26.4 | ✅ OK | |
| polars | >=0.20.0 | NOT FOUND | ❌ MISSING | Run: uv pip install polars |

### Conflict Detection

| ID | Severity | Type | Description | Location |
|----|----------|------|-------------|----------|
| C01 | CRITICAL | VERSION_MISMATCH | numpy 2.0 installed but <2.0 required | pyproject.toml:20 |
| C02 | HIGH | TRANSITIVE | Package A requires X>=2.0 but B requires X<2.0 | pip check output |

### API Compatibility Issues

| ID | Severity | File | Line(s) | Issue | Recommendation |
|----|----------|------|---------|-------|----------------|
| A01 | CRITICAL | src/data/loader.py | L45 | `np.float` removed in NumPy 1.24+ | Use `float` or `np.float64` |
| A02 | HIGH | src/app.py | L12 | `st.cache` deprecated since Streamlit 1.18 | Use `st.cache_data` |
| A03 | MEDIUM | src/utils/config.py | L3 | `from typing import Dict` unnecessary in 3.12 | Use `dict` directly |

### Python Version Compatibility

| Check | Status | Details |
|-------|--------|---------|
| requires-python (>=3.12) vs active (3.12.1) | ✅ PASS | |
| Syntax features | ✅ PASS | No incompatible syntax found |
| stdlib changes | ⚠️ WARN | Using `tomllib` - OK for 3.12, not backportable |

### Tooling & Configuration

| Tool | Config Status | Version Compat | Notes |
|------|--------------|----------------|-------|
| ruff | ✅ Valid | ✅ 0.1.0+ | |
| mypy | ⚠️ Stale config | ✅ 1.7.0+ | Consider updating mypy config |
| pytest | ✅ Valid | ✅ 7.4.0+ | |

### Metrics Summary

- Total files scanned: X
- Total imports analyzed: X
- Critical issues: X
- High issues: X
- Medium issues: X
- Low issues: X
- Overall Health: 🟢 HEALTHY / 🟡 WARNINGS / 🔴 BROKEN
```

### 9. Severity Classification

- **CRITICAL**: Will cause runtime crash or import failure. Must fix before running.
- **HIGH**: Deprecated API that may break in next upgrade or causes subtle bugs.
- **MEDIUM**: Style/compatibility warnings, unnecessary imports, future-proofing suggestions.
- **LOW**: Minor suggestions, Python version-specific optimizations.

### 10. Provide Remediation Plan

At the end of the report, output a prioritized action plan:

1. **Immediate fixes** (CRITICAL): Commands or code changes to resolve crashes
2. **Recommended fixes** (HIGH): Deprecation migrations to prevent future breakage
3. **Optional improvements** (MEDIUM/LOW): Modernization suggestions

Format each remediation as:
```
[Issue ID] → Fix: <specific action>
   Command: <exact command if applicable>
   Code change: <before → after pattern>
```

Ask: "Would you like me to apply the CRITICAL fixes automatically?" (Do NOT apply without approval.)

## Behavior Rules

- **NEVER modify files** (read-only diagnostic)
- **NEVER install or uninstall packages** (report only)
- **Limit runtime checks** to max 30 targeted verifications to avoid slow execution
- **Prioritize known breaking changes** for libraries detected in the project
- **Skip deep scan** unless user explicitly requests "deep" or "full" in arguments
- **Report zero issues gracefully**: If no issues found, output a clean health report with ✅ status
- **Be specific**: Always include file paths, line numbers, and exact API names in findings
- **No false positives**: Only flag issues with high confidence. Mark uncertain findings as "⚠️ POSSIBLE" with explanation
- If environment cannot be detected (e.g., no Python found), abort with clear error and setup instructions
