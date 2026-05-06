"""Verify ground truth for each candidate:
  - Build bad_env (cached under benchmark/envs/)
  - Run canonical_solution + test in bad_env  → expect FAIL with error_type
  - Run correct_solution + test in bad_env    → expect PASS
  - Both required for `verified=True`. The first proves the breaking change
    is real; the second proves the case is *solvable* (an alternative API
    exists that yields the same behavior in bad_env).

Usage:
  uv run python benchmark/verify_ground_truth.py            # all cases
  uv run python benchmark/verify_ground_truth.py --case bcb_002
  uv run python benchmark/verify_ground_truth.py --rebuild  # force rebuild venvs
  uv run python benchmark/verify_ground_truth.py --first 5  # only first 5

Output:
  - Prints per-case verdict
  - Writes benchmark/verification_report.json with full results
  - Updates `verified` field in benchmark/candidates.json (if --update is passed)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Allow running as `python benchmark/verify_ground_truth.py` from repo root
sys.path.insert(0, str(Path(__file__).parent))

from runner_utils import (
    BENCHMARK_DIR,
    build_env,
    compose_test_script,
    load_candidates,
    run_in_env,
)


REPORT_PATH = BENCHMARK_DIR / "verification_report.json"
CANDIDATES_PATH = BENCHMARK_DIR / "candidates.json"


@dataclass
class CaseVerdict:
    case_id: str
    task_id: str
    rule_label: str
    canonical_exit_code: int
    canonical_timed_out: bool
    canonical_duration_s: float
    canonical_failed_as_expected: bool
    correct_exit_code: int
    correct_timed_out: bool
    correct_duration_s: float
    correct_passed_as_expected: bool
    error_type_matched: bool
    verified: bool
    canonical_stderr_tail: str
    correct_stderr_tail: str


def verify_one(case: dict, rebuild: bool = False, timeout_s: int = 90) -> CaseVerdict:
    bad_pip = case["bad_env_pip"]
    bad_python = case.get("bad_python")
    expected_err = case.get("error_type", "")

    print(f"  [build bad_env] py={bad_python or 'default'} {bad_pip}")
    bad_env = build_env(bad_pip, force_rebuild=rebuild, python_version=bad_python)

    canonical_script = compose_test_script(case, user_code=case["canonical_solution"])
    correct_script = compose_test_script(case, user_code=case["correct_solution"])

    print(f"  [run canonical in bad_env]  (expect FAIL)")
    canonical_run = run_in_env(bad_env, canonical_script, timeout_s=timeout_s)
    print(f"  [run correct   in bad_env]  (expect PASS)")
    correct_run = run_in_env(bad_env, correct_script, timeout_s=timeout_s)

    canonical_failed = canonical_run.exit_code != 0 or canonical_run.timed_out
    correct_passed = correct_run.passed

    err_matched = bool(expected_err) and (
        expected_err in canonical_run.stderr or expected_err in canonical_run.stdout
    )

    return CaseVerdict(
        case_id=case["case_id"],
        task_id=case["task_id"],
        rule_label=case["rule_label"],
        canonical_exit_code=canonical_run.exit_code,
        canonical_timed_out=canonical_run.timed_out,
        canonical_duration_s=round(canonical_run.duration_s, 2),
        canonical_failed_as_expected=canonical_failed,
        correct_exit_code=correct_run.exit_code,
        correct_timed_out=correct_run.timed_out,
        correct_duration_s=round(correct_run.duration_s, 2),
        correct_passed_as_expected=correct_passed,
        error_type_matched=err_matched,
        verified=canonical_failed and correct_passed,
        canonical_stderr_tail=canonical_run.stderr[-600:] if canonical_run.stderr else "",
        correct_stderr_tail=correct_run.stderr[-600:] if correct_run.stderr else "",
    )


def fmt_verdict(v: CaseVerdict) -> str:
    canon = "✓ FAIL (expected)" if v.canonical_failed_as_expected else "✗ PASS (unexpected)"
    correct = "✓ PASS (expected)" if v.correct_passed_as_expected else "✗ FAIL (unexpected)"
    err = " err_type✓" if v.error_type_matched else " err_type✗"
    overall = "✅ VERIFIED" if v.verified else "❌ INVALID"
    return (
        f"{overall} {v.case_id} ({v.task_id}, {v.rule_label})\n"
        f"   canonical: exit={v.canonical_exit_code:<4} {canon}{err}  ({v.canonical_duration_s}s)\n"
        f"   correct:   exit={v.correct_exit_code:<4} {correct}                     ({v.correct_duration_s}s)"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="Verify a single case_id (e.g. bcb_002)")
    ap.add_argument("--first", type=int, help="Only first N cases")
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild of venvs")
    ap.add_argument("--timeout", type=int, default=90, help="Per-run timeout (s)")
    ap.add_argument("--update", action="store_true",
                    help="Write verified=True/False back into candidates.json")
    ap.add_argument("--verbose-on-fail", action="store_true",
                    help="Print stderr tail when a case is INVALID")
    args = ap.parse_args()

    cases = load_candidates()
    if args.case:
        cases = [c for c in cases if c["case_id"] == args.case]
        if not cases:
            print(f"No case_id={args.case}", file=sys.stderr)
            return 1
    if args.first:
        cases = cases[: args.first]

    print(f"Verifying {len(cases)} cases...\n")

    verdicts: list[CaseVerdict] = []
    for i, case in enumerate(cases, 1):
        print(f"--- [{i}/{len(cases)}] {case['case_id']} ({case['task_id']}, "
              f"{case['rule_label']}) ---")
        try:
            v = verify_one(case, rebuild=args.rebuild, timeout_s=args.timeout)
        except Exception as e:
            print(f"  EXCEPTION during verify: {type(e).__name__}: {e}")
            v = CaseVerdict(
                case_id=case["case_id"], task_id=case["task_id"],
                rule_label=case["rule_label"],
                canonical_exit_code=-99, canonical_timed_out=False,
                canonical_duration_s=0.0, canonical_failed_as_expected=False,
                correct_exit_code=-99, correct_timed_out=False,
                correct_duration_s=0.0, correct_passed_as_expected=False,
                error_type_matched=False, verified=False,
                canonical_stderr_tail=f"BUILD/RUN ERROR: {e}",
                correct_stderr_tail="",
            )
        verdicts.append(v)
        print(fmt_verdict(v))
        if not v.verified and args.verbose_on_fail:
            print("   --- canonical stderr tail ---")
            print("   " + v.canonical_stderr_tail.replace("\n", "\n   "))
            print("   --- correct stderr tail ---")
            print("   " + v.correct_stderr_tail.replace("\n", "\n   "))
        print()

    # Summary
    n = len(verdicts)
    n_verified = sum(1 for v in verdicts if v.verified)
    n_err_matched = sum(1 for v in verdicts if v.error_type_matched)
    print("=" * 60)
    print(f"Verified: {n_verified}/{n}  (error_type matched in bad: {n_err_matched})")
    invalid = [v for v in verdicts if not v.verified]
    if invalid:
        print(f"\nInvalid cases ({len(invalid)}):")
        for v in invalid:
            reason = []
            if not v.canonical_failed_as_expected:
                reason.append(f"canonical passed (exit={v.canonical_exit_code})")
            if not v.correct_passed_as_expected:
                reason.append(f"correct failed (exit={v.correct_exit_code})")
            print(f"  {v.case_id} ({v.task_id}): {'; '.join(reason)}")

    # Write report
    REPORT_PATH.write_text(json.dumps([asdict(v) for v in verdicts], indent=2))
    print(f"\nReport written to {REPORT_PATH}")

    if args.update:
        all_cases = load_candidates()
        verified_ids = {v.case_id for v in verdicts if v.verified}
        for c in all_cases:
            if c["case_id"] in {v.case_id for v in verdicts}:
                c["verified"] = c["case_id"] in verified_ids
        CANDIDATES_PATH.write_text(json.dumps(all_cases, indent=2))
        print(f"Updated `verified` field in {CANDIDATES_PATH}")

    return 0 if n_verified == n else 1


if __name__ == "__main__":
    sys.exit(main())
