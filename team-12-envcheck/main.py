"""
EnvCheck Live Demo Runner

Orchestrates the full demo flow for each test case:
  1. Create isolated uv environment with the target library version
  2. Write the LLM-generated (broken) code to a temp file
  3. Run it → show the crash
  4. Run EnvCheck scanner → detect issues BEFORE runtime
  5. Show the fixed code
  6. Run fixed code → show success

Usage:
    python main.py                  # Run all cases
    python main.py numpy_2x        # Run a specific case
    python main.py --list           # List all available cases
    python main.py --setup-only     # Only create environments, don't run
    python main.py --eval           # Run evaluation with metrics
"""

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from test_cases.cases import ALL_CASES, TestCase
from envcheck.scanner import scan_source, ScanReport

# Project root
ROOT = Path(__file__).parent
ENV_DIR = ROOT / "environments"
GENERATED_DIR = ROOT / "test_cases" / "generated"


def ensure_dirs():
    """Create necessary directories."""
    ENV_DIR.mkdir(exist_ok=True)
    GENERATED_DIR.mkdir(exist_ok=True)


def banner(text: str, char: str = "="):
    """Print a formatted banner."""
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def step(num: int, text: str):
    """Print a step indicator."""
    print(f"\n  [{num}] {text}")
    print(f"  {'-' * 60}")


def setup_environment(case: TestCase) -> Path:
    """Create an isolated uv virtual environment for a test case."""
    import shlex

    env_path = ENV_DIR / f"case_{case.id}"
    python_bin = env_path / "bin" / "python"

    if python_bin.exists():
        print(f"    Environment already exists: {env_path}")
        return env_path

    print(f"    Creating environment: {env_path}")

    # Parse the environment string: remove "pip install " prefix, extract --python flag
    raw_env = case.environment.replace("pip install ", "")
    parts = shlex.split(raw_env)

    # Extract --python version if specified
    python_version = "3.12"
    install_pkgs = []
    i = 0
    while i < len(parts):
        if parts[i] == "--python" and i + 1 < len(parts):
            python_version = parts[i + 1]
            i += 2
        else:
            install_pkgs.append(parts[i])
            i += 1

    # Create venv
    result = subprocess.run(
        ["uv", "venv", str(env_path), "--python", python_version],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    ERROR creating venv: {result.stderr}")
        sys.exit(1)

    # Install packages
    result = subprocess.run(
        ["uv", "pip", "install", *install_pkgs],
        capture_output=True, text=True,
        env={**os.environ, "VIRTUAL_ENV": str(env_path)},
    )
    if result.returncode != 0:
        print(f"    ERROR installing packages: {result.stderr}")
        sys.exit(1)

    print(f"    Installed: {case.environment}")
    return env_path


def write_code_file(case: TestCase, code: str, suffix: str) -> Path:
    """Write code to a file in the generated directory."""
    filepath = GENERATED_DIR / f"{case.id}_{suffix}.py"
    filepath.write_text(code)
    return filepath


def run_in_env(env_path: Path, script_path: Path) -> subprocess.CompletedProcess:
    """Run a Python script in an isolated environment."""
    python_bin = env_path / "bin" / "python"
    return subprocess.run(
        [str(python_bin), str(script_path)],
        capture_output=True, text=True,
        timeout=30,
    )


def show_code(code: str, label: str):
    """Display code with a label."""
    print(f"\n    --- {label} ---")
    for i, line in enumerate(code.strip().split("\n"), 1):
        print(f"    {i:3d} | {line}")
    print()


def run_case(case: TestCase):
    """Execute the full demo flow for a single test case."""
    banner(f"CASE: {case.id} — {case.library}", "=")
    print(f"\n  Problem:\n{textwrap.indent(case.problem, '    ')}")

    # Step 1: Setup environment
    step(1, "Setting up isolated environment")
    print(f"    Command: {case.environment}")
    env_path = setup_environment(case)

    # Step 2: Show what LLM generates
    step(2, "LLM-generated code (expected to FAIL)")
    show_code(case.broken_code, "Broken Code")

    # Step 3: Run broken code → crash
    step(3, "Running broken code in target environment")
    broken_file = write_code_file(case, case.broken_code, "broken")
    result = run_in_env(env_path, broken_file)

    if result.returncode != 0:
        print(f"    ❌ CRASHED (exit code {result.returncode})")
        print(f"    Error output:")
        # Show last few lines of stderr (the actual error)
        error_lines = result.stderr.strip().split("\n")
        for line in error_lines[-5:]:
            print(f"      {line}")
        print(f"\n    Expected: {case.expected_error}")
    else:
        print(f"    ⚠️  Unexpectedly succeeded! Output:")
        print(textwrap.indent(result.stdout, "      "))

    # Step 4: Run EnvCheck scanner on the broken code
    step(4, "EnvCheck Scanner — detecting issues BEFORE runtime")
    scan_report = scan_source(
        case.broken_code,
        env_path=str(env_path),
        filepath=f"{case.id}_broken.py",
    )
    if scan_report.total_findings > 0:
        print(f"    🔍 EnvCheck found {scan_report.total_findings} issue(s) "
              f"in {scan_report.scan_time_ms:.0f}ms:")
        print()
        for finding in scan_report.findings:
            print(f"    ⛔ Line {finding.lineno}: {finding.matched_code}")
            print(f"       Rule: {finding.rule.rule_id}")
            print(f"       {finding.rule.description}")
            print(f"       Fix: {finding.rule.old_api} → {finding.rule.new_api}")
            print(f"       Installed: {finding.rule.library} {finding.installed_version} "
                  f"(changed in {finding.rule.removed_in})")
            print()
    else:
        print(f"    ⚠️  Scanner found 0 issues (expected {len(case.breaking_changes)})")
        print(f"    Scan time: {scan_report.scan_time_ms:.0f}ms")

    # Step 5: Show fixed code
    step(5, "Fixed code (correct modern API)")
    show_code(case.fixed_code, "Fixed Code")

    # Step 6: Run fixed code → success
    step(6, "Running fixed code in target environment")
    fixed_file = write_code_file(case, case.fixed_code, "fixed")
    result = run_in_env(env_path, fixed_file)

    if result.returncode == 0:
        print(f"    ✅ SUCCESS")
        print(f"    Output:")
        print(textwrap.indent(result.stdout.strip(), "      "))
    else:
        print(f"    ❌ Still failing: {result.stderr.strip()[-200:]}")

    print()


def run_evaluation(cases_to_run: list[TestCase]):
    """Run the full evaluation suite and produce a metrics summary.

    For each case:
      - Scan broken code → count true positives (should detect issues)
      - Scan fixed code → count false positives (should detect 0 issues)
      - Measure scan time
    """
    import time

    banner("EnvCheck Evaluation Suite", "▓")
    print(f"  Evaluating scanner on {len(cases_to_run)} test case(s)...\n")

    # Table header
    header = (f"  {'Case ID':<16} {'Library':<14} {'Expected':>8} {'Detected':>8} "
              f"{'FP':>4} {'Time(ms)':>9} {'Status':<8}")
    print(header)
    print(f"  {'-' * 75}")

    total_expected = 0
    total_detected = 0
    total_fp = 0
    total_time = 0.0
    case_results = []

    for case in cases_to_run:
        env_path = ENV_DIR / f"case_{case.id}"

        # Ensure environment exists
        if not (env_path / "bin" / "python").exists():
            print(f"  {case.id:<16} SKIP — environment not set up")
            continue

        # Scan broken code (true positives)
        broken_report = scan_source(
            case.broken_code, env_path=str(env_path),
            filepath=f"{case.id}_broken.py",
        )
        tp = broken_report.total_findings
        expected = len(case.breaking_changes)

        # Scan fixed code (false positives)
        fixed_report = scan_source(
            case.fixed_code, env_path=str(env_path),
            filepath=f"{case.id}_fixed.py",
        )
        fp = fixed_report.total_findings

        scan_time = broken_report.scan_time_ms + fixed_report.scan_time_ms

        # Determine status
        if tp >= expected and fp == 0:
            status = "✅ PASS"
        elif tp < expected:
            status = "❌ MISS"
        elif fp > 0:
            status = "⚠️  FP"
        else:
            status = "✅ PASS"

        print(f"  {case.id:<16} {case.library:<14} {expected:>8} {tp:>8} "
              f"{fp:>4} {scan_time:>8.0f} {status:<8}")

        total_expected += expected
        total_detected += tp
        total_fp += fp
        total_time += scan_time
        case_results.append({
            "id": case.id,
            "expected": expected,
            "detected": tp,
            "fp": fp,
            "time_ms": scan_time,
            "pass": tp >= expected and fp == 0,
        })

    # Summary
    print(f"  {'-' * 75}")
    detection_rate = (total_detected / total_expected * 100) if total_expected else 0
    pass_count = sum(1 for r in case_results if r["pass"])

    print(f"\n  📊 Evaluation Summary")
    print(f"  {'=' * 50}")
    print(f"  Cases evaluated:     {len(case_results)}")
    print(f"  Cases passed:        {pass_count}/{len(case_results)}")
    print(f"  Detection rate:      {total_detected}/{total_expected} "
          f"({detection_rate:.1f}%)")
    print(f"  False positives:     {total_fp}")
    print(f"  Total scan time:     {total_time:.0f}ms")
    print(f"  Avg scan time/case:  {total_time / len(case_results):.0f}ms" if case_results else "")
    print()

    if detection_rate >= 100 and total_fp == 0:
        print(f"  🎯 PERFECT SCORE — {detection_rate:.0f}% detection, 0 false positives!")
        if total_detected > total_expected:
            print(f"  ℹ️  Scanner detected {total_detected - total_expected} extra issue(s) "
                  f"beyond test case expectations (KB has more rules)")
    elif detection_rate >= 100:
        print(f"  ⚠️  {detection_rate:.0f}% detection but {total_fp} false positive(s) — needs tuning")
    else:
        print(f"  📈 Detection rate needs improvement: {detection_rate:.1f}%")

    return case_results


def main():
    parser = argparse.ArgumentParser(description="EnvCheck Live Demo Runner")
    parser.add_argument("case_id", nargs="?", help="Run a specific case by ID")
    parser.add_argument("--list", action="store_true", help="List all available cases")
    parser.add_argument("--setup-only", action="store_true", help="Only create environments")
    parser.add_argument("--eval", action="store_true", help="Run evaluation with metrics")
    args = parser.parse_args()

    ensure_dirs()

    if args.list:
        from test_cases.cases import print_case_summary
        print_case_summary()
        return

    cases_to_run = ALL_CASES
    if args.case_id:
        cases_to_run = [c for c in ALL_CASES if c.id == args.case_id]
        if not cases_to_run:
            print(f"Unknown case: {args.case_id}")
            print(f"Available: {', '.join(c.id for c in ALL_CASES)}")
            sys.exit(1)

    if args.setup_only:
        banner("Setting up all environments")
        for case in cases_to_run:
            print(f"\n  {case.id}:")
            setup_environment(case)
        print("\n  Done! All environments ready.")
        return

    if args.eval:
        # First ensure all environments exist
        for case in cases_to_run:
            env_path = ENV_DIR / f"case_{case.id}"
            if not (env_path / "bin" / "python").exists():
                print(f"  Setting up environment for {case.id}...")
                setup_environment(case)
        run_evaluation(cases_to_run)
        return

    # Full demo run
    banner("EnvCheck Live Demo", "▓")
    print(f"  Running {len(cases_to_run)} test case(s)...\n")

    results = {"pass": 0, "fail": 0}
    for case in cases_to_run:
        try:
            run_case(case)
            results["pass"] += 1
        except Exception as e:
            print(f"  ERROR running case {case.id}: {e}")
            results["fail"] += 1

    # Summary
    banner("Demo Summary")
    print(f"  Cases run: {results['pass'] + results['fail']}")
    print(f"  Successful demos: {results['pass']}")
    print(f"  Failed demos: {results['fail']}")
    total_changes = sum(len(c.breaking_changes) for c in cases_to_run)
    print(f"  Total breaking changes demonstrated: {total_changes}")


if __name__ == "__main__":
    main()
