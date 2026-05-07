"""Run the benchmark eval: for each verified candidate, run both
  - baseline: a single LLM call given task + pip-freeze of bad_env
  - envpilot: the full LangGraph pipeline against bad_env
and score each generated solution by running it + test in bad_env.

Reports per-case rows and aggregate metrics matching the project goals:

  Effectiveness: first-pass success / final success / crash rate
  Efficiency:    latency / tokens / tool calls / overhead vs. repair savings

Usage:
  export GOOGLE_API_KEY=...
  uv run python benchmark/run_eval.py                  # all cases, both modes
  uv run python benchmark/run_eval.py --case manual_006
  uv run python benchmark/run_eval.py --first 5
  uv run python benchmark/run_eval.py --mode baseline  # baseline only
  uv run python benchmark/run_eval.py --mode envpilot  # envpilot only
  uv run python benchmark/run_eval.py --n 3            # repeat each case N times for stability

Output:
  benchmark/eval_results.json      — per-case rows
  benchmark/eval_summary.json      — aggregate metrics
  printed table on stdout
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Allow running as `python benchmark/run_eval.py`
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # for envcheck import

from runner_utils import (
    BENCHMARK_DIR,
    build_env,
    compose_test_script,
    load_candidates,
    run_in_env,
    venv_python,
)


RESULTS_PATH = BENCHMARK_DIR / "eval_results.json"
SUMMARY_PATH = BENCHMARK_DIR / "eval_summary.json"


@dataclass
class RunRecord:
    case_id: str
    task_id: str
    rule_label: str
    library_under_test: str
    kind: str
    mode: str  # "baseline" | "envpilot"
    repeat: int  # which repeat-index for stability sampling

    # Generation outcome
    final_code: str
    generation_error: str  # empty if generation succeeded
    duration_s: float

    # Score (running final_code + test in bad_env)
    test_passed: bool
    test_exit_code: int
    test_stderr_tail: str
    test_crashed: bool  # exception, not just assertion failure

    # Iterative baseline only: how many (code, run) cycles before pass/give-up.
    # 1 means iter 0 succeeded. EnvPilot leaves this at 1.
    iterations_used: int = 1

    # EnvPilot-only fields (0/empty for baseline)
    preflight_attempts: int = 0
    web_search_called: bool = False
    kb_updates_count: int = 0
    llm_calls: int = 0
    web_search_calls: int = 0
    preflight_runs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Per-node token breakdown (envpilot only; baseline uses 'baseline' bucket).
    # Keys: analysis / env_probe / kb_query / kb_update / plan / generation / baseline
    node_tokens: dict = field(default_factory=dict)
    node_calls: dict = field(default_factory=dict)

    # First-pass success: passed on the FIRST generation attempt.
    # baseline: iter 0 passes. envpilot: preflight_attempts <= 1 and passed.
    first_pass_success: bool = False


# --------- Baseline: reactive iterative LLM loop ----------

MAX_BASELINE_ITERS = 3  # Match EnvPilot's MAX_PREFLIGHT_ATTEMPTS for fair comparison

SYSTEM_MSG = "You are an expert Python developer."


def _build_baseline_prompt(instruct: str, entry_point: str,
                            code_prompt: str,
                            history: list[tuple[str, str]]) -> str:
    """Compose the user prompt for one baseline iteration.

    `code_prompt` is BCB's scaffold (imports + def line + module-level
    constants). It frames which libraries the LLM is required to use.
    Without it, the LLM may bypass the target library entirely (e.g. solve
    a "compute product" task with math.prod instead of numpy), which means
    the breaking change is never exercised.

    `history` is a list of (previous_code, traceback_tail) tuples from
    prior failed attempts. Empty on iter 0.
    """
    scaffold_block = (
        f"You must use exactly these imports and this function signature:\n\n"
        f"```python\n{code_prompt.rstrip()}\n```\n"
    )

    if not history:
        return (
            f"Write a complete Python function named `{entry_point}` that "
            f"does the following:\n\n"
            f"{instruct}\n\n"
            f"{scaffold_block}\n"
            "Output the complete Python code (imports + function definition "
            "with body). No markdown fences, no commentary, no explanation."
        )

    parts = [
        f"You're trying to write a Python function named `{entry_point}` "
        f"for this task:\n\n{instruct}\n\n"
        f"{scaffold_block}",
    ]
    for i, (prev_code, prev_err) in enumerate(history):
        parts.append(f"\n--- Attempt {i + 1} (failed) ---\n{prev_code}")
        parts.append(f"\nThis attempt failed with the following error when "
                     f"the test suite was run:\n\n{prev_err}")
    parts.append(
        "\nFix the code and output the complete corrected function "
        "(imports + def + body), still respecting the scaffold above. "
        "No markdown fences, no commentary."
    )
    return "\n".join(parts)


def _strip_markdown_fences(text: str) -> str:
    code = text.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


def _llm_invoke(prompt: str) -> tuple[str, str]:
    """Single Gemini call for baseline. Returns (text, error_str).

    Tags tokens under node='baseline' so per-node breakdown distinguishes
    baseline iterations from EnvPilot's graph nodes in the same metrics dict.
    """
    from envcheck.agent.nodes import _get_llm, _record_usage
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        response = _get_llm().invoke([
            SystemMessage(content=SYSTEM_MSG),
            HumanMessage(content=prompt),
        ])
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"

    _record_usage(response, node="baseline")

    text = response.content
    if isinstance(text, list):
        text = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in text
        )
    return text, ""


def run_baseline(case: dict, bad_env_path: Path,
                  max_iter: int = MAX_BASELINE_ITERS,
                  timeout_s: int = 90) -> tuple[str, str, dict, int, dict]:
    """Reactive baseline loop: generate code, run it + test in bad_env, on
    failure feed traceback back to LLM and retry. Stops on first pass or
    after max_iter failures.

    Returns (final_code, error_str, metrics, iterations_used, last_score)
    where last_score is the score_code() result of the last attempt:
      {"passed", "exit_code", "stderr_tail", "crashed"}
    """
    from envcheck.agent.nodes import get_metrics, reset_metrics
    reset_metrics()

    instruct = case["instruct_prompt"]
    entry = case.get("entry_point") or "task_func"
    code_prompt = case.get("code_prompt") or ""

    history: list[tuple[str, str]] = []
    final_code = ""
    last_score = {"passed": False, "exit_code": -1,
                  "stderr_tail": "<no run>", "crashed": True}
    iterations_used = 0

    for iter_idx in range(max_iter):
        iterations_used = iter_idx + 1
        prompt = _build_baseline_prompt(instruct, entry, code_prompt, history)
        text, err = _llm_invoke(prompt)
        if err:
            return "", err, get_metrics(), iterations_used, last_score

        final_code = _strip_markdown_fences(text)
        score = score_code(case, final_code, bad_env_path, timeout_s=timeout_s)
        last_score = {
            "passed": score[0], "exit_code": score[1],
            "stderr_tail": score[2], "crashed": score[3],
        }
        if score[0]:
            break  # passed
        # Retry: append (code, traceback) to history for next iter
        history.append((final_code, score[2]))

    return final_code, "", get_metrics(), iterations_used, last_score


# --------- EnvPilot: full LangGraph pipeline ----------

def run_envpilot(case: dict, bad_env_path: Path) -> tuple[str, str, dict, dict]:
    """Run EnvPilot. Returns (final_code, error_str, metrics, state_snapshot).

    The task_description we pass in is `instruct_prompt` augmented with the
    `entry_point` constraint — same minimal spec the baseline gets.
    """
    from envcheck.agent.graph import build_graph, get_default_initial_state
    from envcheck.agent.nodes import get_metrics, reset_metrics

    reset_metrics()
    entry = case.get("entry_point") or "task_func"
    code_prompt = (case.get("code_prompt") or "").rstrip()
    # Same minimal spec the baseline gets: instruct + entry_point + the BCB
    # scaffold (imports + signature + module constants). The scaffold pins
    # which libraries the LLM must use so it can't sidestep the breaking
    # change by switching to a different library.
    augmented_task = (
        f"{case['instruct_prompt']}\n\n"
        f"Define a Python function named `{entry}`. "
        f"You must use exactly these imports and this signature:\n\n"
        f"```python\n{code_prompt}\n```"
    )

    try:
        app = build_graph()
        state = app.invoke(get_default_initial_state(
            task_description=augmented_task,
            env_path=str(bad_env_path),
        ))
        final_code = _strip_markdown_fences(state.get("final_code", "") or "")
        snap = {
            "preflight_attempts": state.get("preflight_attempts", 0),
            "web_search_called": bool(state.get("web_results")),
            "kb_updates_count": len(state.get("kb_updates") or []),
            "phase_at_end": state.get("phase", ""),
        }
        return final_code, "", get_metrics(), snap
    except Exception as e:
        return "", f"{type(e).__name__}: {e}", get_metrics(), {}


# --------- Scoring ----------

def score_code(case: dict, code: str, bad_env_path: Path,
               timeout_s: int = 90) -> tuple[bool, int, str, bool]:
    """Run LLM-generated full code + test in bad_env.

    The LLM is expected to produce complete code (imports + def + body), so we
    do NOT prepend the case's `code_prompt` (BCB's scaffold). We just append
    the test and a unittest runner. Returns (passed, exit_code, stderr_tail,
    crashed); stderr_tail is the last ~2000 chars (used as feedback for the
    next baseline iter).
    """
    if not code.strip():
        return False, -1, "<no code generated>", True

    code = _strip_markdown_fences(code)
    if not code.endswith("\n"):
        code += "\n"

    script = (
        code
        + "\n"
        + case["test"]
        + "\n\n"
        + "if __name__ == '__main__':\n"
        + "    import unittest\n"
        + "    unittest.main(verbosity=2, exit=True)\n"
    )
    run = run_in_env(bad_env_path, script, timeout_s=timeout_s)
    # Crash = abnormal exit before/around tests (import error, syntax error,
    # uncaught exception). Plain assertion failure yields FAILED in stderr.
    crashed = run.timed_out or (
        "Traceback" in run.stderr and "FAILED" not in run.stderr
    )
    return run.passed, run.exit_code, run.stderr[-2000:], crashed


# --------- Orchestration ----------

def run_one(case: dict, mode: str, repeat: int = 0,
            timeout_s: int = 90) -> RunRecord:
    bad_env_path = build_env(
        case["bad_env_pip"],
        python_version=case.get("bad_python"),
    )

    base = dict(
        case_id=case["case_id"],
        task_id=case["task_id"],
        rule_label=case["rule_label"],
        library_under_test=case["library_under_test"],
        kind=case["kind"],
        mode=mode,
        repeat=repeat,
    )

    start = time.time()
    iterations_used = 1
    if mode == "baseline":
        code, err, metrics, iterations_used, last_score = run_baseline(
            case, bad_env_path, timeout_s=timeout_s,
        )
        # baseline already ran code+test internally for each iter
        passed = last_score["passed"]
        exit_code = last_score["exit_code"]
        stderr_tail = last_score["stderr_tail"]
        crashed = last_score["crashed"]
        snap: dict = {}
    elif mode == "envpilot":
        code, err, metrics, snap = run_envpilot(case, bad_env_path)
        if err:
            passed, exit_code, stderr_tail, crashed = False, -1, err, True
        else:
            passed, exit_code, stderr_tail, crashed = score_code(
                case, code, bad_env_path, timeout_s=timeout_s,
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    duration = time.time() - start

    if mode == "envpilot" and err:
        # generation itself errored; iterations_used not meaningful
        iterations_used = 0

    attempts = snap.get("preflight_attempts", 0)
    if mode == "baseline":
        first_pass = bool(passed) and iterations_used == 1
    else:  # envpilot
        first_pass = bool(passed) and attempts <= 1

    return RunRecord(
        **base,
        final_code=code,
        generation_error=err,
        duration_s=round(duration, 2),
        test_passed=passed,
        test_exit_code=exit_code,
        test_stderr_tail=stderr_tail,
        test_crashed=crashed,
        iterations_used=iterations_used,
        preflight_attempts=attempts,
        web_search_called=bool(snap.get("web_search_called")),
        kb_updates_count=int(snap.get("kb_updates_count", 0)),
        llm_calls=int(metrics.get("llm_calls", 0)),
        web_search_calls=int(metrics.get("web_search_calls", 0)),
        preflight_runs=int(metrics.get("preflight_runs", 0)),
        input_tokens=int(metrics.get("input_tokens", 0)),
        output_tokens=int(metrics.get("output_tokens", 0)),
        total_tokens=int(metrics.get("total_tokens", 0)),
        node_tokens=dict(metrics.get("node_tokens") or {}),
        node_calls=dict(metrics.get("node_calls") or {}),
        first_pass_success=first_pass,
    )


# --------- Aggregation ----------

def _safe_mean(xs: list[float]) -> float:
    return round(statistics.fmean(xs), 2) if xs else 0.0


def aggregate(records: list[RunRecord]) -> dict:
    by_mode: dict[str, list[RunRecord]] = {}
    for r in records:
        by_mode.setdefault(r.mode, []).append(r)

    summary: dict = {}
    for mode, rs in by_mode.items():
        n = len(rs)
        passed = [r for r in rs if r.test_passed]
        first_pass = [r for r in rs if r.first_pass_success]
        crashed = [r for r in rs if r.test_crashed]

        # Aggregate per-node tokens. We surface ALL node names that appear
        # across any record; missing keys count as 0.
        all_nodes = sorted({k for r in rs for k in r.node_tokens.keys()})
        node_token_means = {
            node: _safe_mean([r.node_tokens.get(node, 0) for r in rs])
            for node in all_nodes
        }
        node_call_means = {
            node: _safe_mean([r.node_calls.get(node, 0) for r in rs])
            for node in all_nodes
        }

        summary[mode] = {
            "n_runs": n,
            "n_unique_cases": len({r.case_id for r in rs}),

            # Effectiveness
            "first_pass_success_rate": round(len(first_pass) / n, 3) if n else 0,
            "final_success_rate":      round(len(passed) / n, 3) if n else 0,
            "crash_rate":              round(len(crashed) / n, 3) if n else 0,

            # Efficiency (means)
            "mean_duration_s":   _safe_mean([r.duration_s for r in rs]),
            "mean_total_tokens": _safe_mean([r.total_tokens for r in rs]),
            "mean_llm_calls":    _safe_mean([r.llm_calls for r in rs]),
            "mean_iterations":   _safe_mean([r.iterations_used for r in rs]),  # baseline retry rounds
            "mean_web_search":   _safe_mean([r.web_search_calls for r in rs]),
            "mean_preflight":    _safe_mean([r.preflight_runs for r in rs]),
            "mean_attempts":     _safe_mean([r.preflight_attempts for r in rs]),

            # Per-node token attribution. For envpilot, kb_update tokens are
            # "KB-population work" (benefits future runs); the rest are
            # "task work" for this case.
            "mean_node_tokens": node_token_means,
            "mean_node_calls":  node_call_means,
        }

    # Overhead vs. repair savings — only meaningful with both modes
    if "baseline" in by_mode and "envpilot" in by_mode:
        b = summary["baseline"]
        e = summary["envpilot"]
        token_overhead = e["mean_total_tokens"] - b["mean_total_tokens"]
        success_lift = e["final_success_rate"] - b["final_success_rate"]
        summary["delta"] = {
            "token_overhead":      round(token_overhead, 1),
            "tokens_per_extra_pass": (round(token_overhead / success_lift, 1)
                                       if success_lift > 0 else None),
            "success_rate_lift":   round(success_lift, 3),
            "duration_overhead_s": round(e["mean_duration_s"] - b["mean_duration_s"], 2),
        }

        # Per-case PAIRED comparison: pair baseline vs envpilot runs sharing
        # the same (case_id, repeat). This is the "tokens saved" view —
        # for the same problem, did EnvPilot's proactive diagnosis cost
        # fewer tokens than baseline's reactive retry loop?
        by_pair: dict[tuple[str, int], dict[str, RunRecord]] = {}
        for r in records:
            by_pair.setdefault((r.case_id, r.repeat), {})[r.mode] = r
        pairs = [
            (m["baseline"], m["envpilot"])
            for m in by_pair.values()
            if "baseline" in m and "envpilot" in m
        ]

        if pairs:
            token_diffs = [bl.total_tokens - ep.total_tokens for bl, ep in pairs]
            time_diffs = [bl.duration_s - ep.duration_s for bl, ep in pairs]
            n = len(pairs)

            both_passed = [(b, e) for b, e in pairs if b.test_passed and e.test_passed]
            base_failed_ep_passed = [(b, e) for b, e in pairs if not b.test_passed and e.test_passed]
            base_passed_ep_failed = [(b, e) for b, e in pairs if b.test_passed and not e.test_passed]
            both_failed = [(b, e) for b, e in pairs if not b.test_passed and not e.test_passed]

            summary["paired"] = {
                "n_pairs": n,

                # Token "savings" (positive = EnvPilot used fewer tokens than
                # baseline for the same case+repeat)
                "mean_tokens_saved_per_case": _safe_mean(token_diffs),
                "total_tokens_saved":         round(sum(token_diffs), 1),
                "n_envpilot_used_fewer_tokens": sum(1 for d in token_diffs if d > 0),
                "n_envpilot_used_more_tokens":  sum(1 for d in token_diffs if d < 0),

                # Latency
                "mean_seconds_saved_per_case": _safe_mean(time_diffs),

                # Outcome breakdown (the 2x2 of who-passed-what)
                "outcome_both_passed":            len(both_passed),
                "outcome_baseline_only_passed":   len(base_passed_ep_failed),
                "outcome_envpilot_only_passed":   len(base_failed_ep_passed),
                "outcome_both_failed":            len(both_failed),

                # Repair savings: when baseline failed entirely, those tokens
                # were "wasted" reactive retries. EnvPilot's tokens on the
                # same cases (whether it passed or not) is the alternative cost.
                "mean_baseline_tokens_when_failed": _safe_mean(
                    [b.total_tokens for b, _ in pairs if not b.test_passed]
                ),
                "mean_baseline_iters_when_failed": _safe_mean(
                    [b.iterations_used for b, _ in pairs if not b.test_passed]
                ),

                # Conditional savings: among "both passed" pairs, did
                # EnvPilot save tokens? (success is held constant)
                "tokens_saved_when_both_pass": _safe_mean(
                    [b.total_tokens - e.total_tokens for b, e in both_passed]
                ),
            }

    return summary


# --------- CLI ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="Run only one case_id")
    ap.add_argument("--first", type=int, help="Only first N cases")
    ap.add_argument("--mode", choices=["both", "baseline", "envpilot"],
                    default="both")
    ap.add_argument("--n", type=int, default=1,
                    help="Repeat each (case, mode) N times (LLM is non-deterministic)")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--only-verified", action="store_true", default=True,
                    help="Skip cases where verified=false (default true)")
    args = ap.parse_args()

    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set", file=sys.stderr)
        return 2

    cases = load_candidates()
    if args.only_verified:
        cases = [c for c in cases if c.get("verified")]
    if args.case:
        cases = [c for c in cases if c["case_id"] == args.case]
        if not cases:
            print(f"No verified case_id={args.case}", file=sys.stderr)
            return 1
    if args.first:
        cases = cases[: args.first]

    modes = ["baseline", "envpilot"] if args.mode == "both" else [args.mode]

    # Snapshot KB size before the run; envpilot's kb_update_node persists
    # rules into local SQLite, and we want to know how much the KB grew
    # over the whole eval (rather than wiping per-case, which would mask
    # the proactive learning EnvPilot does).
    def _kb_count() -> int:
        try:
            from envcheck.knowledge_base_store import KnowledgeBaseStore
            with KnowledgeBaseStore() as store:
                return store.count()
        except Exception as e:
            print(f"WARN: KB count failed: {e}", file=sys.stderr)
            return -1

    kb_before = _kb_count()
    print(f"KB rule count before run: {kb_before}")

    print(f"Running {len(cases)} cases × {len(modes)} modes × n={args.n} = "
          f"{len(cases) * len(modes) * args.n} runs\n")

    records: list[RunRecord] = []
    for case in cases:
        for mode in modes:
            for rep in range(args.n):
                tag = f"{case['case_id']}/{mode}/rep{rep}"
                print(f"--- {tag} ---")
                rec = run_one(case, mode, repeat=rep, timeout_s=args.timeout)
                records.append(rec)
                print(f"  passed={rec.test_passed} first_pass={rec.first_pass_success} "
                      f"iters={rec.iterations_used} duration={rec.duration_s}s "
                      f"tokens={rec.total_tokens} llm_calls={rec.llm_calls}")

    kb_after = _kb_count()
    print(f"\nKB rule count after run: {kb_after}  (grew by {kb_after - kb_before})")

    # Save
    RESULTS_PATH.write_text(json.dumps([asdict(r) for r in records], indent=2))
    summary = aggregate(records)
    summary["kb"] = {
        "size_before": kb_before,
        "size_after": kb_after,
        "growth": (kb_after - kb_before) if kb_before >= 0 and kb_after >= 0 else None,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nFull results: {RESULTS_PATH}")
    print(f"Summary:      {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
