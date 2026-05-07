"""
EnvPilot Agent Nodes — Implementation of each phase in the state graph.

Each node function takes the current EnvPilotState and returns a partial
state update. The LLM is called via langchain-google-genai for analysis phases.
"""

import json
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from envcheck.agent.prompts import (
    ANALYSIS_PROMPT,
    ENV_PROBE_ANALYSIS_PROMPT,
    GENERATION_PROMPT,
    KB_ANALYSIS_PROMPT,
    KB_UPDATE_PROMPT,
    PLAN_PROMPT,
    PREFLIGHT_PROMPT,
    SYSTEM_PROMPT,
)
from envcheck.agent.state import EnvPilotState
from envcheck.knowledge_base_store import KnowledgeBaseStore
from envcheck.preflight_runner import run_preflight, to_dict as preflight_to_dict
from envcheck.version_detector import get_installed_packages
from envcheck.web_searcher import WebSearcher

logger = logging.getLogger("envpilot")

MAX_PREFLIGHT_ATTEMPTS = 3
MAX_PLAN_ATTEMPTS = 2  # plan ↔ preflight retry cap (each retry = 1 LLM call)

# Per-run instrumentation for benchmark eval.
# Reset at the start of each EnvPilot run, then read after invoke().
# `node_tokens` / `node_calls` are dicts keyed by node name (analysis,
# env_probe, kb_query, kb_update, plan, generation) so we can attribute
# tokens to each phase — useful for distinguishing "task work" from
# "KB-population work" (kb_update is the only node whose output is a side
# effect on the local KB rather than the current task's solution).
_metrics: dict = {
    "llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "web_search_calls": 0,
    "preflight_runs": 0,
    "kb_query_calls": 0,
    "node_tokens": {},   # node_name -> total tokens used by that node
    "node_calls": {},    # node_name -> LLM call count for that node
}


def reset_metrics() -> None:
    """Zero out all instrumentation counters. Call before each pipeline run."""
    for k in list(_metrics.keys()):
        if isinstance(_metrics[k], dict):
            _metrics[k] = {}
        else:
            _metrics[k] = 0


def get_metrics() -> dict:
    """Snapshot current counters (deep-copies the per-node dicts)."""
    snap = dict(_metrics)
    snap["node_tokens"] = dict(_metrics["node_tokens"])
    snap["node_calls"] = dict(_metrics["node_calls"])
    return snap


def _record_usage(response, node: str = "") -> None:
    """Update global + per-node counters from a langchain AIMessage response."""
    _metrics["llm_calls"] += 1
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        return
    in_t = int(usage.get("input_tokens") or 0)
    out_t = int(usage.get("output_tokens") or 0)
    tot_t = int(usage.get("total_tokens") or 0)
    _metrics["input_tokens"] += in_t
    _metrics["output_tokens"] += out_t
    _metrics["total_tokens"] += tot_t
    if node:
        _metrics["node_tokens"][node] = _metrics["node_tokens"].get(node, 0) + tot_t
        _metrics["node_calls"][node] = _metrics["node_calls"].get(node, 0) + 1


def _get_llm(model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model, temperature=0, max_tokens=4096)


def _parse_json_response(text: str) -> dict:
    """Extract JSON from an LLM response, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        start = 1
        end = len(lines) - 1
        if lines[-1].strip() == "```":
            cleaned = "\n".join(lines[start:end])
        else:
            cleaned = "\n".join(lines[start:])
    return json.loads(cleaned)


def _llm_call(prompt: str, node: str = "") -> dict:
    """Make an LLM call and parse the JSON response.

    `node` attributes the call's tokens to a specific phase in the per-node
    breakdown.
    """
    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    _record_usage(response, node)

    text = response.content
    if isinstance(text, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in text
        )
    return _parse_json_response(text)


# ============================================================================
# Phase 1: Analysis & Uncertainty Assessment
# ============================================================================

def analysis_node(state: EnvPilotState) -> dict[str, Any]:
    """Analyze the task, identify packages, and assign uncertainty score."""
    logger.info("[Phase 1] Analyzing task and assessing uncertainty...")

    prompt = ANALYSIS_PROMPT.format(task_description=state["task_description"])

    try:
        result = _llm_call(prompt, node="analysis")
        return {
            "identified_packages": result.get("identified_packages", []),
            "critical_apis": result.get("critical_apis", []),
            "uncertainty_score": result.get("uncertainty_score", 50),
            "phase": "analysis_complete",
            "messages": [HumanMessage(content=f"[Analysis] {json.dumps(result, indent=2)}")],
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            "identified_packages": [],
            "critical_apis": [],
            "uncertainty_score": 80,
            "phase": "analysis_error",
            "error": str(e),
            "messages": [HumanMessage(content=f"[Analysis Error] {e}")],
        }


# ============================================================================
# Phase 2: Environment Probing
# ============================================================================

def env_probe_node(state: EnvPilotState) -> dict[str, Any]:
    """Probe the local environment for installed package versions."""
    logger.info("[Phase 2] Probing environment...")

    env_path = state.get("env_path", "")
    packages = state.get("identified_packages", [])

    env_info: dict[str, Any] = {"packages": {}, "python": "", "probed": True}

    if env_path:
        from pathlib import Path
        installed = get_installed_packages(env_path)
        python_bin = Path(env_path) / "bin" / "python"
        if python_bin.exists():
            import subprocess
            try:
                ver = subprocess.run(
                    [str(python_bin), "--version"],
                    capture_output=True, text=True, timeout=5,
                )
                env_info["python"] = ver.stdout.strip()
            except Exception:
                env_info["python"] = "unknown"
    else:
        import sys
        from pathlib import Path
        installed = get_installed_packages(Path(sys.prefix))
        import platform
        env_info["python"] = platform.python_version()

    for pkg_name in packages:
        normalized = pkg_name.lower()
        if normalized in installed:
            env_info["packages"][pkg_name] = {
                "installed": True,
                "version": installed[normalized].version,
            }
        else:
            env_info["packages"][pkg_name] = {
                "installed": False,
                "version": None,
            }

    # LLM analysis of env results
    prompt = ENV_PROBE_ANALYSIS_PROMPT.format(env_info=json.dumps(env_info, indent=2))
    try:
        llm_result = _llm_call(prompt, node="env_probe")
        updated_uncertainty = llm_result.get(
            "updated_uncertainty_score", state.get("uncertainty_score", 50)
        )
    except Exception:
        updated_uncertainty = state.get("uncertainty_score", 50)

    return {
        "env_info": env_info,
        "uncertainty_score": updated_uncertainty,
        "phase": "env_probed",
        "messages": [HumanMessage(content=f"[Env Probe] {json.dumps(env_info, indent=2)}")],
    }


# ============================================================================
# Phase 3: Knowledge Alignment
# ============================================================================

def kb_query_node(state: EnvPilotState) -> dict[str, Any]:
    """Query the knowledge base for relevant breaking change rules."""
    logger.info("[Phase 3] Querying knowledge base...")
    _metrics["kb_query_calls"] += 1

    store = KnowledgeBaseStore()
    packages = state.get("identified_packages", [])
    all_results: list[dict] = []

    for pkg in packages:
        rules = store.query(library=pkg)
        all_results.extend(store.to_dict_list(rules))

    env_info = state.get("env_info", {})

    prompt = KB_ANALYSIS_PROMPT.format(
        kb_results=json.dumps(all_results, indent=2),
        env_info=json.dumps(env_info, indent=2),
    )

    kb_has_gaps = False
    try:
        llm_result = _llm_call(prompt, node="kb_query")
        kb_has_gaps = llm_result.get("kb_has_gaps", False)
        updated_uncertainty = llm_result.get(
            "updated_uncertainty_score", state.get("uncertainty_score", 50)
        )
    except Exception:
        kb_has_gaps = len(all_results) == 0
        updated_uncertainty = state.get("uncertainty_score", 50)

    store.close()

    return {
        "kb_results": all_results,
        "kb_has_gaps": kb_has_gaps,
        "uncertainty_score": updated_uncertainty,
        "phase": "kb_queried",
        "messages": [HumanMessage(
            content=f"[KB Query] Found {len(all_results)} rules. Gaps: {kb_has_gaps}"
        )],
    }


def web_search_node(state: EnvPilotState) -> dict[str, Any]:
    """Search the web for missing API documentation and breaking changes."""
    logger.info("[Phase 3b] Searching web for missing information...")
    _metrics["web_search_calls"] += 1

    searcher = WebSearcher()
    packages = state.get("identified_packages", [])
    all_results: list[dict] = []

    for pkg in packages:
        env_pkg_info = state.get("env_info", {}).get("packages", {}).get(pkg, {})
        version = env_pkg_info.get("version", "")
        query = f"breaking changes migration guide"
        if version:
            query = f"version {version} {query}"

        results = searcher.search_api_docs(pkg, query, max_results=3)
        all_results.extend(searcher.to_dict_list(results))

    return {
        "web_results": all_results,
        "phase": "web_searched",
        "messages": [HumanMessage(
            content=f"[Web Search] Found {len(all_results)} results"
        )],
    }


def kb_update_node(state: EnvPilotState) -> dict[str, Any]:
    """Extract and upsert new rules from web search results into the KB."""
    logger.info("[Phase 3c] Updating knowledge base with web findings...")

    web_results = state.get("web_results", [])
    if not web_results:
        return {
            "kb_updates": [],
            "phase": "kb_updated",
            "messages": [HumanMessage(content="[KB Update] No web results to process")],
        }

    prompt = KB_UPDATE_PROMPT.format(web_results=json.dumps(web_results, indent=2))

    kb_updates: list[dict] = []
    try:
        llm_result = _llm_call(prompt, node="kb_update")
        rules_to_upsert = llm_result.get("rules_to_upsert", [])

        if rules_to_upsert:
            from envcheck.knowledge_base import BreakingChangeRule, PatternType, Severity
            store = KnowledgeBaseStore()
            for rule_dict in rules_to_upsert:
                try:
                    br = BreakingChangeRule(
                        rule_id=rule_dict["rule_id"],
                        library=rule_dict["library"],
                        removed_in=rule_dict["removed_in"],
                        pattern_type=PatternType(rule_dict["pattern_type"]),
                        module_path=rule_dict["module_path"],
                        symbol=rule_dict["symbol"],
                        old_api=rule_dict["old_api"],
                        new_api=rule_dict["new_api"],
                        error_type=rule_dict["error_type"],
                        description=rule_dict["description"],
                        severity=Severity(rule_dict.get("severity", "error")),
                    )
                    store.upsert(br, source="web_search")
                    kb_updates.append(rule_dict)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid rule: {e}")
            store.close()
    except Exception as e:
        logger.warning(f"KB update LLM call failed: {e}")

    return {
        "kb_updates": kb_updates,
        "phase": "kb_updated",
        "messages": [HumanMessage(
            content=f"[KB Update] Upserted {len(kb_updates)} new rules"
        )],
    }


# ============================================================================
# Phase 4: Pre-flight Verification
# ============================================================================

def _llm_call_raw(prompt: str, node: str = "") -> str:
    """Call the LLM and return raw text (no JSON parse). Updates metrics."""
    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    _record_usage(response, node)

    text = response.content
    if isinstance(text, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in text
        )
    return text


def _extract_fenced_python(text: str) -> str:
    """Pull a Python code block out of a fenced response.

    Robust to:
      - properly closed fences:    ```python\n...\n```
      - missing closing fence:     ```python\n...   (LLM truncated)
      - leading/trailing prose
      - both ```python and bare ``` openers
    """
    import re
    text = text.strip()

    # Try closed fence first (greedy enough to grab full code, ?: non-capturing).
    m = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Open fence but no close — strip the opener and use everything after.
    m = re.search(r"```(?:python)?\s*\n(.*)$", text, re.DOTALL)
    if m:
        body = m.group(1)
        # Trim trailing stray backticks if any.
        body = re.sub(r"\n?```\s*$", "", body)
        return body.strip()

    # No fence at all — return raw (LLM ignored the format instruction).
    return text


def plan_node(state: EnvPilotState) -> dict[str, Any]:
    """Synthesize env_info + KB + web findings into a concrete API plan.

    Output: `proposed_apis` — dotted paths the agent will actually use in the
    final code. Preflight then verifies each exists in bad_env. If preflight
    fails, control returns here with the failure stdout fed back via
    `previous_attempt`, and the LLM picks different alternatives.
    """
    logger.info("[Phase 3d] Planning concrete API set...")
    attempts = state.get("plan_attempts", 0) + 1

    # If a prior plan attempt failed preflight, feed the failure back so the
    # LLM doesn't propose the same broken APIs again.
    prev_block = ""
    prev_apis = state.get("proposed_apis") or []
    prev_pf = state.get("preflight_result") or {}
    if attempts > 1 and prev_apis and not prev_pf.get("success"):
        stdout = (prev_pf.get("stdout") or "")[:1500]
        prev_block = (
            "PREVIOUS ATTEMPT FAILED.\n"
            f"You proposed: {prev_apis}\n"
            f"Preflight output:\n{stdout}\n"
            "Pick different APIs that ACTUALLY EXIST in the env."
        )

    prompt = PLAN_PROMPT.format(
        task_description=state["task_description"],
        env_info=json.dumps(state.get("env_info", {}), indent=2),
        critical_apis=json.dumps(state.get("critical_apis") or [], indent=2),
        kb_results=json.dumps(state.get("kb_results") or [], indent=2),
        web_results=json.dumps((state.get("web_results") or [])[:5], indent=2),
        previous_attempt=prev_block,
    )

    try:
        result = _llm_call(prompt, node="plan")
        proposed = result.get("proposed_apis") or []
        # Sanitize: must be list of strings with dots
        proposed = [a for a in proposed if isinstance(a, str) and "." in a]
        if not proposed:
            # Fallback: trust the original critical_apis from analysis
            proposed = state.get("critical_apis") or []
        return {
            "proposed_apis": proposed,
            "plan_attempts": attempts,
            "phase": "planned",
            "messages": [HumanMessage(
                content=f"[Plan attempt {attempts}] proposed_apis={proposed}"
            )],
        }
    except Exception as e:
        logger.warning(f"Plan LLM failed: {e}")
        return {
            "proposed_apis": state.get("critical_apis") or [],
            "plan_attempts": attempts,
            "phase": "plan_error",
            "error": str(e),
            "messages": [HumanMessage(content=f"[Plan Error] {e}")],
        }


# PyPI distribution name → Python import name (for cases where they differ).
# Used so preflight's `importlib.import_module("scikit-learn")` doesn't fail
# with a spurious ModuleNotFoundError when the analysis identifies packages
# by their pip name.
_PIP_TO_IMPORT_NAME = {
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "pillow": "PIL",
    "beautifulsoup4": "bs4",
    "pyyaml": "yaml",
    "opencv-python": "cv2",
    "python-dateutil": "dateutil",
    "msgpack-python": "msgpack",
    "google-auth": "google.auth",
    "google-cloud-storage": "google.cloud.storage",
}


def _to_import_name(pkg: str) -> str:
    """Map pip distribution name to Python import name (lower-case key lookup)."""
    return _PIP_TO_IMPORT_NAME.get(pkg.lower(), pkg)


def _build_deterministic_preflight(critical_apis: list[str],
                                    identified_packages: list[str]) -> str:
    """Build a deterministic preflight smoke test from analysis output.

    For each `identified_package`, try `importlib.import_module` (with
    pip-name→import-name normalization, e.g. scikit-learn → sklearn).
    For each dotted `critical_api` (e.g. 'seaborn.histplot',
    'pandas.DataFrame.append'), walk the dotted path with getattr —
    AttributeError or ImportError is recorded. Subprocess exit code =
    number of failures.

    No LLM call. Always produces valid Python. The stdout cleanly enumerates
    which APIs are missing, so the generation node can read it and avoid
    those symbols in the final code.
    """
    lines = [
        "import importlib",
        "import sys",
        "",
        "_failures = []",
        "_passed = []",
        "",
    ]

    for pkg in identified_packages:
        if not pkg or not isinstance(pkg, str):
            continue
        import_name = _to_import_name(pkg)
        lines += [
            f"try:",
            f"    importlib.import_module({import_name!r})",
            f"    _passed.append('import {pkg}')",
            f"except Exception as _e:",
            f"    _failures.append('import ' + {pkg!r} + ': ' + repr(_e))",
            "",
        ]

    # Helper: walk a dotted path "a.b.C.method" by finding the longest
    # importable prefix (handles submodules like sklearn.preprocessing,
    # matplotlib.pyplot, PIL.Image — which Python doesn't auto-load when
    # you `import sklearn` / `import matplotlib` / `import PIL`).
    lines += [
        "def _walk(api):",
        "    parts = api.split('.')",
        "    last_err = None",
        "    for i in range(len(parts), 0, -1):",
        "        try:",
        "            obj = importlib.import_module('.'.join(parts[:i]))",
        "            for attr in parts[i:]:",
        "                obj = getattr(obj, attr)",
        "            return obj",
        "        except (ImportError, ModuleNotFoundError) as _ie:",
        "            last_err = _ie",
        "            continue",
        "        except AttributeError as _ae:",
        "            last_err = _ae",
        "            break",
        "    raise last_err if last_err else ImportError(api)",
        "",
    ]

    for api in critical_apis:
        if not api or not isinstance(api, str) or "." not in api:
            continue
        lines.append(f"try:")
        lines.append(f"    _walk({api!r})")
        lines.append(f"    _passed.append({api!r})")
        lines.append("except Exception as _e:")
        lines.append(f"    _failures.append({api!r} + ': ' + repr(_e))")
        lines.append("")

    lines += [
        "for p in _passed:",
        "    print('PASS:', p)",
        "for f in _failures:",
        "    print('FAIL:', f)",
        "",
        "if _failures:",
        "    print(f'--- PREFLIGHT FAILED: {len(_failures)} broken API(s) ---')",
        "    sys.exit(1)",
        "else:",
        "    print(f'--- PREFLIGHT PASSED: {len(_passed)} check(s) ---')",
    ]
    return "\n".join(lines)


def preflight_node(state: EnvPilotState) -> dict[str, Any]:
    """Run a deterministic preflight smoke test built from analysis output.

    No LLM call: we already know the critical APIs from `analysis_node`. We
    just walk each dotted path with importlib + getattr and report which
    fail. The result (stdout enumerating PASS/FAIL per API) is fed to the
    generation node so it knows exactly which symbols to avoid.
    """
    logger.info("[Phase 4] Running deterministic preflight verification...")
    _metrics["preflight_runs"] += 1

    attempts = state.get("preflight_attempts", 0) + 1
    env_path = state.get("env_path", "")

    if not env_path:
        return {
            "preflight_result": {"success": True, "note": "No env_path, skipping preflight"},
            "preflight_attempts": attempts,
            "phase": "preflight_passed",
            "messages": [HumanMessage(content="[Preflight] Skipped (no env_path)")],
        }

    # Test the APIs the plan node proposed. Falls back to original
    # critical_apis if plan ran into trouble (still informative).
    apis_to_test = state.get("proposed_apis") or state.get("critical_apis") or []
    code = _build_deterministic_preflight(
        critical_apis=apis_to_test,
        identified_packages=state.get("identified_packages") or [],
    )

    result = run_preflight(code, env_path)
    result_dict = preflight_to_dict(result)

    phase = "preflight_passed" if result.success else "preflight_failed"

    return {
        "preflight_code": code,
        "preflight_result": result_dict,
        "preflight_attempts": attempts,
        "phase": phase,
        "messages": [HumanMessage(
            content=f"[Preflight] {'PASSED' if result.success else 'FAILED'}: "
                    f"{result_dict.get('stdout', '')[:500]}"
        )],
    }


# ============================================================================
# Phase 5: Final One-Pass Generation
# ============================================================================

def generation_node(state: EnvPilotState) -> dict[str, Any]:
    """Generate the final code with full environment context."""
    logger.info("[Phase 5] Generating final code...")

    prompt = GENERATION_PROMPT.format(
        proposed_apis=json.dumps(state.get("proposed_apis") or [], indent=2),
        env_info=json.dumps(state.get("env_info", {}), indent=2),
        kb_results=json.dumps(state.get("kb_results", []), indent=2),
        preflight_result=json.dumps(state.get("preflight_result", {}), indent=2),
        task_description=state["task_description"],
    )

    # Same fenced-markdown trick as preflight: avoid JSON-escaping Python code
    # (regex patterns / quotes / backslashes constantly break json.loads).
    try:
        text = _llm_call_raw(prompt, node="generation")
        final_code = _extract_fenced_python(text)
        # Try to find a NOTES: line for logging
        import re as _re
        m = _re.search(r"NOTES:\s*(.+)", text)
        notes = m.group(1).strip() if m else ""
        if not final_code:
            raise ValueError("Empty final_code from LLM")
    except Exception as e:
        return {
            "final_code": "",
            "phase": "generation_error",
            "error": str(e),
            "messages": [HumanMessage(content=f"[Generation Error] {e}")],
        }

    return {
        "final_code": final_code,
        "phase": "complete",
        "messages": [HumanMessage(
            content=f"[Generation Complete]\nNotes: {notes}\n\nCode:\n```python\n{final_code}\n```"
        )],
    }


# ============================================================================
# Routing functions for conditional edges
# ============================================================================

def route_after_kb_query(state: EnvPilotState) -> str:
    """If KB has gaps or LLM is uncertain, search web first; otherwise plan."""
    uncertainty = state.get("uncertainty_score", 50)
    kb_has_gaps = state.get("kb_has_gaps", False)

    if uncertainty > 20 or kb_has_gaps:
        return "web_search"
    return "plan"


def route_after_preflight(state: EnvPilotState) -> str:
    """If all proposed APIs verified, generate. Otherwise loop back to plan
    (with the failure stdout) to pick different alternatives, capped at
    MAX_PLAN_ATTEMPTS to prevent infinite loops."""
    result = state.get("preflight_result", {})
    plan_attempts = state.get("plan_attempts", 0)

    if result.get("success", False):
        return "generation"
    if plan_attempts >= MAX_PLAN_ATTEMPTS:
        logger.warning(
            f"Max plan attempts ({MAX_PLAN_ATTEMPTS}) reached, "
            "proceeding to generation with the partial info we have."
        )
        return "generation"
    return "plan"
