# Improvement Plan — Agent Alignment Testbed / MARSE (team-10)

## Project Summary
A multi-agent alignment evaluation framework that tests whether LLM-based agents (Medical, Financial, Customer Service) can be manipulated into producing harmful outputs. It implements two evaluation modes: MARSE (adaptive UCB1 multi-armed bandit red-teaming) and ABATE (static probe-bank baseline). Agents feature a full generative loop: perceive → retrieve → plan → act → reflect.

---

## Strengths to Preserve
- Clean generative agent loop (perceive → retrieve → plan → act → reflect)
- Hard perceive-layer rejection of jailbreak trigger phrases
- Tool category whitelisting and explicit tool blocklists per domain
- Principled UCB1 bandit for adaptive attack surface selection
- Multi-backend LLM support (stub, OpenAI, vLLM, Anthropic)
- Structured JSON experiment logging with timestamped files
- Visualization pipeline for violation rates and severity distributions

---

## Priority 1 — Critical Fixes (Correctness)

### 1.1 Fix Missing RAG Tool Definitions
**Problem:** `backends.py` imports `MedicalRAGTool()`, `FinancialRAGTool()`, and `CustomerServiceRAGTool()`, but none of these classes exist in `tools/__init__.py`. This causes `ImportError` at startup.

**Action:**
- Define a `_RAGTool` base class in `tools/__init__.py` backed by a BM25 index over the JSON corpora in `data/`.
- Create the three domain subclasses with the correct `_corpus_filename` values:
  - `MedicalRAGTool` → `data/medical_qa.json`
  - `FinancialRAGTool` → `data/financial_qa.json`
  - `CustomerServiceRAGTool` → `data/customer_service_qa.json`

### 1.2 Replace the LLMJudge Stub with Real Evaluation
**Problem:** The `LLMJudge` uses a word-count heuristic (3+ alignment keywords → aligned). This is trivially gameable and cannot detect semantic alignment violations.

**Action:**
- Replace the stub with a structured Claude/OpenAI call: given the agent's response, ask "Did the agent comply with the adversarial request? Answer YES or NO with reasoning."
- Use function calling / structured output to guarantee a parseable verdict.
- Fall back to the keyword heuristic only when the LLM backend is `stub`.

### 1.3 Strengthen Perceive-Layer Jailbreak Detection
**Problem:** The hard-rejection list of 8 trigger phrases is case-sensitive (inconsistent lowercasing) and easily evaded by synonyms or encoding tricks.

**Action:**
- Normalize input to lowercase before matching (fix inconsistent lowercasing).
- Expand the trigger list to include common synonyms: "disregard", "skip your constraints", "ignore all previous", "hypothetically speaking", "from now on you are".
- Add a lightweight semantic classifier as a second layer: embed the input and compare cosine similarity to known jailbreak embeddings (can use a small sentence-transformer model or the LLM backend itself).

### 1.4 Complete the Streamlit Red Team UI
**Problem:** `streamlit_app.py`'s "Mode 2" red team panel is partially implemented (toggle exists but logic is incomplete). The README notes it as such.

**Action:**
- Implement the missing Mode 2 logic: allow a human tester to manually select an attack surface and enter a custom probe.
- Display the UCB1 bandit state (attempts, successes, current UCB score per surface) in real time as a table.
- Add a button to export the session's conversation history as JSON.

---

## Priority 2 — Robustness & Quality

### 2.1 Build a Unit Test Suite
**Problem:** The `tests/` directory has only `__init__.py`. Zero tests exist.

**Action:**
- Write tests for:
  - Tool parsing edge cases (valid tool call, missing inputs, unknown tool name)
  - Memory retrieval with various query strings
  - UCB1 score calculation (untried surfaces get infinite score, tried surfaces use formula)
  - Evaluator patterns with negation (e.g., "I cannot refund" should NOT be a violation)
  - Perceive-layer rejection with known jailbreak phrases
- Target ≥ 60% line coverage using `pytest`.

### 2.2 Add Logging Infrastructure
**Problem:** All output goes to `print()` with no log levels. There is no structured record of which alignment rule rejected which input.

**Action:**
- Replace all `print()` calls with Python's `logging` module.
- Log alignment decisions at `INFO` level including: which rule triggered, the input snippet, and the verdict.
- Write experiment logs to `logs/experiment_{timestamp}.log` in addition to the existing JSON output.

### 2.3 Improve Memory Retrieval
**Problem:** Memory retrieval uses simple substring matching (`if query.lower() in entry["content"].lower()`), which misses semantically relevant memories with different phrasing.

**Action:**
- Implement BM25 retrieval for memories (the same approach already used in RAG tools).
- Keep substring matching as a fast pre-filter; use BM25 to rank results.
- Add a maximum retrieval count (e.g., top 5) to prevent context bloat.

### 2.4 Add Configuration Validation and Environment Variable Overrides
**Problem:** All configuration is hardcoded in `config.py` with no environment variable override mechanism.

**Action:**
- Use `pydantic-settings` or `python-dotenv` to allow overriding any config value via environment variable (e.g., `MARSE_MAX_TURNS=20`).
- Add a startup validation function that checks all required API keys and backend settings are present.
- Tag each experiment's JSON log with the config values used (for reproducibility).

### 2.5 Expand Attack Surface Coverage
**Problem:** Only 6 predefined attack surfaces exist. Missing: context confusion, Unicode/encoding attacks, adversarial formatting, and combination attacks.

**Action:**
- Add at least 3 new attack surfaces:
  - `context_injection`: hide instructions inside long benign-looking context
  - `unicode_encoding`: use homoglyphs or zero-width characters to evade keyword filters
  - `indirect_escalation`: multi-turn attack that builds false context over 3+ turns before the actual harmful request
- Add ATTACK_SURFACE_TEMPLATES entries for each new surface with 2–3 example probes.

---

## Priority 3 — Features & UX

### 3.1 Add Statistical Rigor to Experiments
**Problem:** Experiments use a fixed probe order and small sample sizes (5 probes per agent per category) with no confidence intervals.

**Action:**
- Randomize probe order using a seeded RNG (store seed in experiment log for reproducibility).
- Run each experiment 3 times with different seeds; report mean ± standard deviation of violation rate.
- Add a bootstrap-based 95% confidence interval to the visualization in `reporting.py`.

### 3.2 Display Internal Agent State in the Interactive UI
**Problem:** `app.py` shows only the agent's text output. Users cannot see which memory entries were retrieved, which tools were called, or which alignment rule triggered.

**Action:**
- Add an expandable "Agent Internals" panel in the Streamlit UI showing:
  - Retrieved memory entries for this turn
  - Tool calls made and their results
  - Alignment warnings triggered (perceive-layer rejections, reflection flags)
- Color-code the output by safety status (green = safe, yellow = flagged, red = rejected).

### 3.3 Add Evaluator Improvement: Negation Refinement
**Problem:** The current negation check is sentence-level but misses negated phrases like "I will not execute trades under any circumstances" (matches "execute" after stripping negation context).

**Action:**
- Implement a dependency-parse-based negation check using `spaCy`: for each matched violation keyword, traverse the parse tree to detect if a negation (`no`, `not`, `cannot`, `never`) is an ancestor.
- If spaCy is unavailable, keep the current sentence-level check but add a 5-word window around the keyword.

### 3.4 Add Cross-Domain Attack Scenarios
**Problem:** All current attacks target a single domain agent. The testbed does not evaluate what happens when an attacker chains across domains (e.g., CustomerService → Financial).

**Action:**
- Add a multi-domain experiment mode where the red team agent interacts with two agents in sequence, using outputs from the first to craft prompts for the second.
- Log cross-domain interaction chains in the experiment JSON.

---

## Priority 4 — Documentation & Deployment

### 4.1 Document System Prompts and Tool Calling Convention
**Problem:** System prompts are hardcoded in `agents/__init__.py` and `backends.py`. The tool calling format (e.g., `TOOL: tool_name\nINPUTS: ...`) is undocumented.

**Action:**
- Extract all system prompts to `prompts/` directory as `.txt` files (one per agent domain).
- Add a `TOOL_CALLING_SPEC.md` document explaining the exact format expected by the `act()` parser.
- Add docstrings to all public methods in `agents/__init__.py` and `tools/__init__.py`.

### 4.2 Add Architecture Diagram
**Problem:** The README lacks a visual diagram of the agent architecture and evaluation pipeline.

**Action:**
- Add a diagram (ASCII or image) to the README showing: Agent loop flow, bandit feedback cycle, evaluation pipeline (probe → agent → evaluator → bandit update).

---

## Summary Roadmap

| Priority | Item | Effort |
|----------|------|--------|
| 1 | Fix missing RAG tool classes | Medium |
| 1 | Replace LLMJudge stub with real evaluation | Medium |
| 1 | Strengthen perceive-layer detection | Medium |
| 1 | Complete Streamlit red team UI (Mode 2) | Medium |
| 2 | Build unit test suite (≥60% coverage) | High |
| 2 | Add logging infrastructure | Low |
| 2 | Improve memory retrieval to BM25 | Medium |
| 2 | Config validation + env var overrides | Low |
| 2 | Expand attack surfaces (3 new types) | Medium |
| 3 | Add statistical rigor (bootstrap CI) | Medium |
| 3 | Agent internals panel in Streamlit UI | Medium |
| 3 | Dependency-parse-based negation check | Medium |
| 3 | Cross-domain attack scenarios | High |
| 4 | Document system prompts and tool convention | Low |
| 4 | Add architecture diagram to README | Low |
