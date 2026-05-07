# EnvCheck: AI-Powered Code Environment Diagnostic Command

## 1. What is EnvCheck?

EnvCheck (`/envcheck.diagnose`) is a **Cursor IDE custom command** that leverages AI to perform comprehensive code environment diagnostics. It automatically detects whether your codebase can run correctly in the current environment — before you even press "Run".

Think of it as a **pre-flight check for your code**: just like pilots run system checks before takeoff, EnvCheck scans your project for compatibility issues, dependency conflicts, and API mismatches that would cause runtime failures.

---

## 2. The Problem It Solves

### Real-World Pain Points

Modern Python projects depend on dozens of external libraries, each with their own versioning and API evolution cycles. Common problems include:

| Problem | Example | Impact |
|---------|---------|--------|
| **API Breaking Changes** | `numpy.float` removed in NumPy 1.24+ | `AttributeError` at runtime |
| **Deprecated APIs** | `streamlit.cache` → `streamlit.cache_data` | Warning spam, eventual breakage |
| **Version Conflicts** | Package A needs `X>=2.0`, Package B needs `X<2.0` | Installation failure or subtle bugs |
| **Missing Dependencies** | Declared in `pyproject.toml` but not installed | `ModuleNotFoundError` at import |
| **Python Version Mismatch** | Code uses `match/case` but Python 3.9 is active | `SyntaxError` |
| **Stale Lock Files** | `uv.lock` out of sync with `pyproject.toml` | Non-reproducible builds |

### Why Existing Tools Are Not Enough

| Tool | What It Does | What It Misses |
|------|-------------|----------------|
| `pip check` | Transitive dependency conflicts | API-level compatibility, deprecated usage |
| `mypy` | Type errors | Runtime version mismatches, deprecated APIs |
| `ruff` / `pylint` | Code style & simple bugs | Dependency conflicts, environment issues |
| `pytest` | Test failures | Doesn't diagnose WHY imports fail |
| Manual checking | Everything... slowly | Doesn't scale, error-prone |

**EnvCheck fills the gap**: it combines dependency analysis, API compatibility scanning, and environment verification into a single, automated command.

---

## 3. How It Works — Architecture Overview

```
┌──────────────────────────────────────────────────┐
│              /envcheck.diagnose                   │
├──────────────────────────────────────────────────┤
│                                                  │
│  Phase 1: Environment Discovery                  │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐ │
│  │ Python Ver │ │ Pkg Manager│ │ Installed    │ │
│  │ Detection  │ │ Detection  │ │ Packages     │ │
│  └─────┬──────┘ └─────┬──────┘ └──────┬───────┘ │
│        └──────────┬───┘               │          │
│                   ▼                   ▼          │
│  Phase 2: Dependency Analysis                    │
│  ┌────────────────────────────────────────────┐  │
│  │ Declared vs Installed Version Matching     │  │
│  │ Transitive Conflict Detection (pip check)  │  │
│  │ Lock File Consistency Verification         │  │
│  └─────────────────┬──────────────────────────┘  │
│                    ▼                             │
│  Phase 3: API Compatibility Scan                 │
│  ┌────────────────────────────────────────────┐  │
│  │ Import Statement Extraction                │  │
│  │ Known Breaking Change Pattern Matching     │  │
│  │ Deprecated API Detection (per library)     │  │
│  │ Static Import Verification                 │  │
│  │ Type Hint Compatibility Check              │  │
│  └─────────────────┬──────────────────────────┘  │
│                    ▼                             │
│  Phase 4: Report Generation                      │
│  ┌────────────────────────────────────────────┐  │
│  │ Structured Diagnostic Report (Markdown)    │  │
│  │ Severity Classification (CRITICAL→LOW)     │  │
│  │ Prioritized Remediation Plan               │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 4. The 10-Step Execution Flow

### Step 1: Environment Discovery
- Detect Python version, virtual environment, package manager (pip/uv/conda/poetry)
- Collect all installed packages with exact versions
- Record OS and platform information

### Step 2: Dependency Specification Analysis
- Parse `pyproject.toml`, `requirements.txt`, or other dependency files
- Extract all declared dependencies with version constraints
- Distinguish between runtime and dev dependencies

### Step 3: Dependency Conflict Detection
- **Declared vs Installed**: Is every dependency installed? Does the version match?
- **Transitive Conflicts**: Do indirect dependencies create version impossible states?
- **Lock File Sync**: Is the lock file up-to-date with declarations?

### Step 4: API Compatibility Scan (Core)
- Extract all `import` and `from X import Y` statements from source files
- Match imports against a **knowledge base of known API breaking changes**
- Library-specific checks: NumPy, Pandas, Streamlit, Plotly, Scikit-learn, Polars, etc.
- Verify that imported symbols actually exist in the installed version

### Step 5: Python Version Compatibility
- Compare `requires-python` with the active interpreter
- Scan for syntax features that need specific Python versions
- Flag `match/case` (3.10+), walrus operator (3.8+), union types (3.10+), etc.

### Step 6: Configuration & Tooling Check
- Verify linter/formatter configs (ruff, mypy, pytest) are compatible
- Check build system configuration validity

### Step 7: Runtime Smoke Test (Optional)
- Only when user requests "deep" mode
- Attempt importing every project module
- Categorize failures: ImportError, AttributeError, SyntaxError

### Step 8: Produce Diagnostic Report
- Structured Markdown tables with findings
- Clear severity indicators: ✅ OK, ⚠️ WARN, ❌ CRITICAL

### Step 9: Severity Classification
- **CRITICAL**: Runtime crash inevitable
- **HIGH**: Deprecated API, will break on next upgrade
- **MEDIUM**: Compatibility warnings, future-proofing
- **LOW**: Minor optimization suggestions

### Step 10: Remediation Plan
- Prioritized fix list with exact commands and code change patterns
- Ask for approval before making any changes

---

## 5. Key Design Decisions

### Read-Only by Default
The command **never modifies files** unless explicitly approved by the user. This makes it safe to run at any time, even in production codebases.

### Knowledge-Based API Checking
Instead of just checking if imports succeed, EnvCheck maintains awareness of **known breaking changes** across major library versions. This catches issues like:
- `np.float` → removed in NumPy 1.24 (but the import might succeed if old version is installed!)
- `st.cache` → deprecated but still works (will break in future Streamlit versions)

### Severity-Driven Prioritization
Not all issues are equal. A missing dependency (CRITICAL) is more urgent than a deprecated type hint pattern (LOW). The report sorts by severity so developers fix what matters first.

### Scope Control
Users can scan the entire project or focus on specific directories/files via arguments. This scales from quick targeted checks to full project audits.

---

## 6. How It Compares to Existing SpecKit Commands

| Aspect | SpecKit Commands | EnvCheck |
|--------|-----------------|----------|
| **Focus** | Requirements, planning, implementation workflow | Runtime environment & code compatibility |
| **When to Use** | Before/during development | Before running / after dependency changes |
| **Modifies Files** | Yes (specs, plans, tasks) | No (read-only diagnostic) |
| **Output** | Artifacts (spec.md, plan.md, tasks.md) | Diagnostic report |
| **Target** | Feature development lifecycle | Code health & environment validation |

### Complementary Workflow

```
/speckit.specify → /speckit.plan → /speckit.tasks → /speckit.implement
                                                          ↓
                                              /envcheck.diagnose  ← Run before/after implementation
                                                          ↓
                                              Fix issues → Run project ✅
```

---

## 7. Example Output (TensorScope Project)

For a project like TensorScope (Python 3.12, Streamlit + Polars + NumPy + Plotly):

```
## Environment Diagnostic Report

### Environment Summary
| Property        | Value                    |
|-----------------|--------------------------|
| Python Version  | 3.12.1                   |
| Virtual Env     | .venv (active)           |
| Package Manager | uv                       |
| OS / Platform   | Linux x86_64             |

### Dependency Status
| Package          | Declared      | Installed | Status    |
|------------------|---------------|-----------|-----------|
| streamlit        | >=1.30.0      | 1.32.0    | ✅ OK     |
| numpy            | >=1.24,<2.0   | 1.26.4    | ✅ OK     |
| polars           | >=0.20.0      | 0.20.31   | ✅ OK     |
| streamlit-echarts| >=0.4.0       | NOT FOUND | ❌ MISSING |

### API Compatibility Issues
| ID  | Severity | File            | Issue                         |
|-----|----------|-----------------|-------------------------------|
| A01 | HIGH     | src/app.py:L12  | `st.cache` deprecated → use `st.cache_data` |
| A02 | MEDIUM   | src/utils.py:L3 | `from typing import Dict` unnecessary in 3.12 |

### Overall Health: 🟡 WARNINGS (1 missing dep, 1 deprecated API)
```

---

## 8. Implementation as a Cursor Custom Command

### File Location
```
.cursor/commands/envcheck.diagnose.md
```

### Command Format (Same as SpecKit)
```yaml
---
description: Perform comprehensive diagnostic scan...
---
```

### Invocation
In Cursor IDE, type in chat:
```
/envcheck.diagnose
```
Or with arguments for targeted scan:
```
/envcheck.diagnose src/data/
```
Or for deep analysis:
```
/envcheck.diagnose deep
```

### What Makes This a "Cursor Command"?
Cursor custom commands are **Markdown files** that contain structured instructions for the AI assistant. When invoked with `/command-name`, the AI reads the instructions and executes them step-by-step, using its tools (terminal commands, file reading, code analysis) to produce the desired output.

Key characteristics:
1. **YAML frontmatter**: `description` field for command discovery
2. **`$ARGUMENTS`**: User input passed to the command
3. **Structured execution steps**: Numbered phases the AI follows
4. **Tool integration**: AI uses terminal, file reading, and code analysis
5. **Formatted output**: Structured Markdown reports

---

## 9. Future Extensions

| Extension | Description |
|-----------|-------------|
| `envcheck.fix` | Automatically apply approved remediations |
| `envcheck.watch` | Monitor for new issues after dependency updates |
| `envcheck.migrate` | Guide major version migrations (e.g., NumPy 1.x → 2.0) |
| `envcheck.docker` | Verify Docker environment consistency |
| `envcheck.ci` | Generate CI pipeline health checks |

---

## 10. Summary

**EnvCheck** is a pre-flight diagnostic tool for Python projects that:

1. **Discovers** your environment (Python version, packages, virtual env)
2. **Analyzes** dependency declarations vs installed reality
3. **Scans** code for API compatibility issues with installed library versions
4. **Reports** findings with severity classification and exact locations
5. **Suggests** prioritized fixes without making any changes

It bridges the gap between "code looks correct" and "code actually runs" — catching the class of bugs that linters, type checkers, and tests individually miss.
