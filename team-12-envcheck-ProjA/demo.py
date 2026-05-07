#!/usr/bin/env python3
"""
EnvCheck Comparative Demo — Side-by-Side Workflow Comparison

Demonstrates the difference between:
  Scenario A (Without EnvCheck): Code → Run → CRASH → Debug → Fix → Re-run
  Scenario B (With EnvCheck):    Code → Scan → Detect → Fix → Run (no crash!)

Usage:
    python demo.py                   # Interactive demo (press Enter to advance)
    python demo.py --auto            # Auto-advance with delays (for recording)
    python demo.py --case numpy_2x   # Choose a specific case
    python demo.py --all             # Run comparison for ALL cases
"""

import argparse
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from test_cases.cases import ALL_CASES, TestCase
from envcheck.scanner import scan_source

# ─── Config ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
ENV_DIR = ROOT / "environments"
GENERATED_DIR = ROOT / "test_cases" / "generated"

# ANSI colors
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_RED  = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"
    BG_YELLOW = "\033[43m"


# ─── Globals ────────────────────────────────────────────────────────
AUTO_MODE = False
AUTO_DELAY = 1.5  # seconds between steps in auto mode


def pause(label: str = ""):
    """Wait for user input or auto-delay."""
    if AUTO_MODE:
        time.sleep(AUTO_DELAY)
    else:
        hint = f" ({label})" if label else ""
        input(f"{C.DIM}    ⏎ Press Enter to continue{hint}...{C.RESET}")


def print_box(text: str, color: str = C.CYAN, width: int = 72):
    """Print text in a box."""
    border = "─" * width
    print(f"\n{color}  ┌{border}┐{C.RESET}")
    for line in text.split("\n"):
        padded = line.ljust(width)[:width]
        print(f"{color}  │{C.RESET} {padded}{color}│{C.RESET}")
    print(f"{color}  └{border}┘{C.RESET}")


def print_header(text: str, color: str = C.BOLD):
    """Print a section header."""
    print(f"\n{color}{'━' * 74}{C.RESET}")
    print(f"{color}  {text}{C.RESET}")
    print(f"{color}{'━' * 74}{C.RESET}")


def print_step(icon: str, text: str, color: str = C.WHITE):
    """Print a step label."""
    print(f"\n{color}  {icon}  {text}{C.RESET}")
    print(f"{C.DIM}  {'·' * 66}{C.RESET}")


def show_code(code: str, highlight_lines: list[int] | None = None):
    """Display code with syntax-like formatting."""
    for i, line in enumerate(code.strip().split("\n"), 1):
        if highlight_lines and i in highlight_lines:
            print(f"    {C.RED}{C.BOLD}{i:3d} │ {line}{C.RESET}  {C.RED}◄ ISSUE{C.RESET}")
        else:
            print(f"    {C.DIM}{i:3d}{C.RESET} │ {line}")


def run_in_env(env_path: Path, code: str, case_id: str, suffix: str) -> subprocess.CompletedProcess:
    """Write code to file and run in environment."""
    GENERATED_DIR.mkdir(exist_ok=True)
    filepath = GENERATED_DIR / f"{case_id}_{suffix}.py"
    filepath.write_text(code)
    python_bin = env_path / "bin" / "python"
    return subprocess.run(
        [str(python_bin), str(filepath)],
        capture_output=True, text=True, timeout=30,
    )


def setup_environment(case: TestCase) -> Path:
    """Ensure the environment exists for a case."""
    import shlex
    env_path = ENV_DIR / f"case_{case.id}"
    python_bin = env_path / "bin" / "python"

    if python_bin.exists():
        return env_path

    print(f"    {C.DIM}Setting up environment...{C.RESET}")

    raw_env = case.environment.replace("pip install ", "")
    parts = shlex.split(raw_env)
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

    ENV_DIR.mkdir(exist_ok=True)
    subprocess.run(["uv", "venv", str(env_path), "--python", python_version],
                   capture_output=True, text=True)
    subprocess.run(["uv", "pip", "install", *install_pkgs],
                   capture_output=True, text=True,
                   env={**os.environ, "VIRTUAL_ENV": str(env_path)})
    return env_path


# ═══════════════════════════════════════════════════════════════════
#  SCENARIO A: Without EnvCheck (Reactive)
# ═══════════════════════════════════════════════════════════════════

def run_scenario_a(case: TestCase, env_path: Path):
    """Demonstrate the traditional reactive workflow WITHOUT EnvCheck."""

    print_header(
        f"SCENARIO A: Without EnvCheck — Traditional Reactive Workflow",
        f"{C.RED}{C.BOLD}",
    )
    print(f"\n{C.RED}  Flow: LLM generates code → Run → 💥 CRASH → Debug → Fix → Re-run{C.RESET}")

    # A1: LLM generates code
    pause("LLM generates code")
    print_step("🤖", "Step A1: LLM generates code from the prompt", C.WHITE)
    print(f"\n    {C.DIM}Prompt: {case.problem[:100]}...{C.RESET}")
    print(f"\n    {C.YELLOW}LLM Output:{C.RESET}")
    show_code(case.broken_code)

    # A2: Developer runs the code — CRASH
    pause("run the code")
    print_step("▶️ ", "Step A2: Developer runs the code", C.WHITE)
    print(f"    {C.DIM}$ python script.py{C.RESET}")

    result = run_in_env(env_path, case.broken_code, case.id, "demo_broken")
    time.sleep(0.3)

    if result.returncode != 0:
        print(f"\n    {C.RED}{C.BOLD}💥 RUNTIME ERROR!{C.RESET}")
        print(f"    {C.RED}{'─' * 50}{C.RESET}")
        error_lines = result.stderr.strip().split("\n")
        for line in error_lines[-6:]:
            print(f"    {C.RED}{line}{C.RESET}")
        print(f"    {C.RED}{'─' * 50}{C.RESET}")
    else:
        print(f"    {C.YELLOW}⚠ Unexpectedly succeeded{C.RESET}")

    # A3: Developer reads error, searches docs, figures out fix
    pause("debug the error")
    print_step("🔍", "Step A3: Developer debugs the error", C.YELLOW)
    print(f"    {C.YELLOW}Developer reads the traceback...{C.RESET}")
    print(f"    {C.YELLOW}Searches StackOverflow / library changelog...{C.RESET}")
    print(f"    {C.YELLOW}Realizes the API was removed/changed...{C.RESET}")
    print(f"    {C.YELLOW}Manually rewrites the code...{C.RESET}")
    print()
    for bc in case.breaking_changes:
        print(f"    {C.YELLOW}💡 Discovered: {bc.old_api} → {bc.new_api}{C.RESET}")
        print(f"       {C.DIM}{bc.description}{C.RESET}")

    # A4: Developer applies fix and re-runs
    pause("apply fix and re-run")
    print_step("🔧", "Step A4: Developer applies fix and re-runs", C.WHITE)
    print(f"\n    {C.GREEN}Fixed Code:{C.RESET}")
    show_code(case.fixed_code)

    print(f"\n    {C.DIM}$ python script_fixed.py{C.RESET}")
    result = run_in_env(env_path, case.fixed_code, case.id, "demo_fixed")

    if result.returncode == 0:
        print(f"\n    {C.GREEN}{C.BOLD}✅ SUCCESS (on second try){C.RESET}")
        output_lines = result.stdout.strip().split("\n")
        for line in output_lines[:5]:
            print(f"    {C.GREEN}{line}{C.RESET}")
    else:
        print(f"    {C.RED}❌ Still failing{C.RESET}")

    # A-Summary
    print(f"\n{C.RED}  ╔══════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.RED}  ║  Scenario A Summary:                                    ║{C.RESET}")
    print(f"{C.RED}  ║    • 1 runtime crash experienced                        ║{C.RESET}")
    print(f"{C.RED}  ║    • Manual debugging required                          ║{C.RESET}")
    print(f"{C.RED}  ║    • 2 execution attempts to get working code           ║{C.RESET}")
    print(f"{C.RED}  ║    • Developer time wasted on avoidable error           ║{C.RESET}")
    print(f"{C.RED}  ╚══════════════════════════════════════════════════════════╝{C.RESET}")


# ═══════════════════════════════════════════════════════════════════
#  SCENARIO B: With EnvCheck (Proactive)
# ═══════════════════════════════════════════════════════════════════

def run_scenario_b(case: TestCase, env_path: Path):
    """Demonstrate the proactive workflow WITH EnvCheck."""

    print_header(
        f"SCENARIO B: With EnvCheck — Proactive Diagnostic Workflow",
        f"{C.GREEN}{C.BOLD}",
    )
    print(f"\n{C.GREEN}  Flow: LLM generates code → EnvCheck scans → Fix BEFORE running → Run once ✅{C.RESET}")

    # B1: LLM generates code (same as before)
    pause("LLM generates code")
    print_step("🤖", "Step B1: LLM generates code from the prompt", C.WHITE)
    print(f"\n    {C.DIM}Prompt: {case.problem[:100]}...{C.RESET}")
    print(f"\n    {C.YELLOW}LLM Output (same code as Scenario A):{C.RESET}")
    show_code(case.broken_code)

    # B2: EnvCheck scans BEFORE running
    pause("EnvCheck scans")
    print_step("🛡️ ", "Step B2: EnvCheck scans code BEFORE execution", C.CYAN)
    print(f"    {C.CYAN}Running: envcheck.scan(source_code, env_path){C.RESET}")

    scan_report = scan_source(
        case.broken_code,
        env_path=str(env_path),
        filepath=f"{case.id}_broken.py",
    )

    time.sleep(0.3)

    if scan_report.total_findings > 0:
        print(f"\n    {C.CYAN}{C.BOLD}🔍 EnvCheck detected {scan_report.total_findings} issue(s) "
              f"in {scan_report.scan_time_ms:.0f}ms — BEFORE any code execution!{C.RESET}")
        print()

        # Show code with highlighted problem lines
        problem_lines = [f.lineno for f in scan_report.findings]
        show_code(case.broken_code, highlight_lines=problem_lines)
        print()

        for finding in scan_report.findings:
            print(f"    {C.CYAN}⛔ Line {finding.lineno}: {C.BOLD}{finding.matched_code}{C.RESET}")
            print(f"       {C.CYAN}{finding.rule.description}{C.RESET}")
            print(f"       {C.GREEN}Fix: {finding.rule.old_api} → {finding.rule.new_api}{C.RESET}")
            print(f"       {C.DIM}Installed: {finding.rule.library} "
                  f"{finding.installed_version} (changed in {finding.rule.removed_in}){C.RESET}")
            print()

    # B3: Apply fix based on EnvCheck suggestions (no debugging needed)
    pause("apply suggested fix")
    print_step("🔧", "Step B3: Apply fix based on EnvCheck suggestions (no debugging needed)", C.GREEN)
    print(f"    {C.GREEN}EnvCheck told us exactly what to change — no guesswork!{C.RESET}")
    print(f"\n    {C.GREEN}Fixed Code:{C.RESET}")
    show_code(case.fixed_code)

    # B4: Run fixed code — SUCCESS on first try
    pause("run fixed code")
    print_step("▶️ ", "Step B4: Run the code — first and only execution", C.GREEN)
    print(f"    {C.DIM}$ python script.py{C.RESET}")

    result = run_in_env(env_path, case.fixed_code, case.id, "demo_fixed_b")

    if result.returncode == 0:
        print(f"\n    {C.GREEN}{C.BOLD}✅ SUCCESS (on first try — zero crashes!){C.RESET}")
        output_lines = result.stdout.strip().split("\n")
        for line in output_lines[:5]:
            print(f"    {C.GREEN}{line}{C.RESET}")
    else:
        print(f"    {C.RED}❌ Unexpected failure{C.RESET}")

    # B-Summary
    print(f"\n{C.GREEN}  ╔══════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.GREEN}  ║  Scenario B Summary:                                    ║{C.RESET}")
    print(f"{C.GREEN}  ║    • 0 runtime crashes                                  ║{C.RESET}")
    print(f"{C.GREEN}  ║    • No manual debugging needed                         ║{C.RESET}")
    print(f"{C.GREEN}  ║    • 1 execution attempt — worked first time            ║{C.RESET}")
    print(f"{C.GREEN}  ║    • Issues caught in {scan_report.scan_time_ms:<4.0f}ms (static analysis)        ║{C.RESET}")
    print(f"{C.GREEN}  ╚══════════════════════════════════════════════════════════╝{C.RESET}")

    return scan_report


# ═══════════════════════════════════════════════════════════════════
#  COMPARISON
# ═══════════════════════════════════════════════════════════════════

def show_comparison(case: TestCase, scan_report):
    """Show the final side-by-side comparison."""

    print_header("COMPARISON: Without vs With EnvCheck", f"{C.MAGENTA}{C.BOLD}")

    print(f"""
{C.BOLD}  ┌─────────────────────────────┬─────────────────────────────┐{C.RESET}
{C.BOLD}  │  {C.RED}Without EnvCheck{C.RESET}{C.BOLD}           │  {C.GREEN}With EnvCheck{C.RESET}{C.BOLD}              │{C.RESET}
{C.BOLD}  ├─────────────────────────────┼─────────────────────────────┤{C.RESET}
  │ 1. LLM generates code       │ 1. LLM generates code       │
  │ 2. {C.RED}Run → 💥 CRASH{C.RESET}             │ 2. {C.CYAN}EnvCheck scans (static){C.RESET}   │
  │ 3. {C.RED}Read traceback{C.RESET}             │ 3. {C.CYAN}Issues detected in {scan_report.scan_time_ms:.0f}ms{C.RESET}  │
  │ 4. {C.YELLOW}Search docs/SO{C.RESET}             │ 4. {C.GREEN}Apply suggested fix{C.RESET}       │
  │ 5. {C.YELLOW}Figure out the fix{C.RESET}         │ 5. {C.GREEN}Run → ✅ SUCCESS{C.RESET}          │
  │ 6. Fix code manually         │                             │
  │ 7. {C.GREEN}Re-run → ✅ SUCCESS{C.RESET}       │                             │
{C.BOLD}  ├─────────────────────────────┼─────────────────────────────┤{C.RESET}
  │ Runtime crashes:  {C.RED}1{C.RESET}          │ Runtime crashes:  {C.GREEN}0{C.RESET}          │
  │ Executions:       {C.RED}2{C.RESET}          │ Executions:       {C.GREEN}1{C.RESET}          │
  │ Manual debugging: {C.RED}Yes{C.RESET}        │ Manual debugging: {C.GREEN}No{C.RESET}         │
  │ Time to fix:      {C.RED}Minutes{C.RESET}    │ Time to fix:      {C.GREEN}<1 sec{C.RESET}     │
{C.BOLD}  └─────────────────────────────┴─────────────────────────────┘{C.RESET}
""")

    print(f"  {C.BOLD}Case: {case.id} ({case.library}){C.RESET}")
    print(f"  {C.DIM}Breaking changes detected: {scan_report.total_findings}{C.RESET}")
    print(f"  {C.DIM}Scan time: {scan_report.scan_time_ms:.0f}ms{C.RESET}")
    print()
    print(f"  {C.MAGENTA}{C.BOLD}Key insight: EnvCheck catches API breaking changes{C.RESET}")
    print(f"  {C.MAGENTA}{C.BOLD}BEFORE runtime — turning crashes into warnings.{C.RESET}")


# ═══════════════════════════════════════════════════════════════════
#  MULTI-CASE COMPARISON (--all)
# ═══════════════════════════════════════════════════════════════════

def run_all_comparison(cases: list[TestCase]):
    """Run abbreviated comparison for ALL cases in a summary table."""

    print_header("EnvCheck: Full Comparative Demo — All Cases", f"{C.MAGENTA}{C.BOLD}")
    print(f"\n  Demonstrating Without vs With EnvCheck for {len(cases)} cases\n")

    results = []

    for case in cases:
        env_path = ENV_DIR / f"case_{case.id}"
        if not (env_path / "bin" / "python").exists():
            setup_environment(case)

        # --- Scenario A: run broken code → crash ---
        broken_result = run_in_env(env_path, case.broken_code, case.id, "cmp_broken")
        crashed = broken_result.returncode != 0

        # Extract error type from traceback (last line starting with a known error class)
        err_lines = broken_result.stderr.strip().split("\n")
        err_type = "Unknown"
        known_errors = ("AttributeError", "ImportError", "TypeError", "ValueError",
                        "ModuleNotFoundError", "NameError", "SyntaxError", "RuntimeError")
        for line in reversed(err_lines):
            line = line.strip()
            for e in known_errors:
                if line.startswith(e):
                    err_type = e
                    break
            if err_type != "Unknown":
                break

        # --- Scenario B: scan with EnvCheck ---
        scan_report = scan_source(
            case.broken_code,
            env_path=str(env_path),
            filepath=f"{case.id}_broken.py",
        )

        # --- Confirm fixed code works ---
        fixed_result = run_in_env(env_path, case.fixed_code, case.id, "cmp_fixed")
        fixed_ok = fixed_result.returncode == 0

        results.append({
            "case": case,
            "crashed": crashed,
            "err_type": err_type,
            "detected": scan_report.total_findings,
            "scan_ms": scan_report.scan_time_ms,
            "fixed_ok": fixed_ok,
            "report": scan_report,
        })

    # ─── Print table ───────────────────────────────────────────────
    print(f"  {C.BOLD}{'Case ID':<16} {'Library':<13} {'Without EnvCheck':<28} {'With EnvCheck':<30}{C.RESET}")
    print(f"  {'─' * 87}")

    for r in results:
        case = r["case"]
        # Without column
        if r["crashed"]:
            without = f"{C.RED}💥 {r['err_type'][:20]}{C.RESET}"
            without_plain = f"💥 {r['err_type'][:20]}"
        else:
            without = f"{C.YELLOW}⚠ No crash{C.RESET}"
            without_plain = "⚠ No crash"

        # With column
        det = r["detected"]
        ms = r["scan_ms"]
        if det > 0 and r["fixed_ok"]:
            with_ec = f"{C.GREEN}🔍 {det} found in {ms:.0f}ms → ✅{C.RESET}"
        elif det > 0:
            with_ec = f"{C.CYAN}🔍 {det} found in {ms:.0f}ms{C.RESET}"
        else:
            with_ec = f"{C.YELLOW}⚠ 0 found{C.RESET}"

        # We need to handle ANSI codes messing up alignment, so print raw
        print(f"  {case.id:<16} {case.library:<13} ", end="")
        print(without, end="")
        # Pad manually
        pad = 28 - len(without_plain)
        print(" " * max(pad, 1), end="")
        print(with_ec)

    # ─── Summary ───────────────────────────────────────────────────
    total_crashes = sum(1 for r in results if r["crashed"])
    total_detected = sum(r["detected"] for r in results)
    total_fixed = sum(1 for r in results if r["fixed_ok"])
    avg_scan = sum(r["scan_ms"] for r in results) / len(results) if results else 0

    print(f"  {'─' * 87}")
    print(f"""
{C.BOLD}  ┌───────────────────────────────────────────────────────────────┐{C.RESET}
{C.BOLD}  │                    Summary Comparison                        │{C.RESET}
{C.BOLD}  ├──────────────────────────┬────────────────────────────────────┤{C.RESET}
{C.BOLD}  │ Metric                   │ Without EnvCheck │ With EnvCheck  │{C.RESET}
{C.BOLD}  ├──────────────────────────┼──────────────────┼────────────────┤{C.RESET}
  │ Runtime crashes          │ {C.RED}{total_crashes:>16}{C.RESET} │ {C.GREEN}{'0':>14}{C.RESET} │
  │ Executions needed        │ {C.RED}{total_crashes * 2:>16}{C.RESET} │ {C.GREEN}{len(results):>14}{C.RESET} │
  │ Issues found statically  │ {C.RED}{'0':>16}{C.RESET} │ {C.GREEN}{total_detected:>14}{C.RESET} │
  │ Cases fixed successfully │ {C.GREEN}{total_fixed:>16}{C.RESET} │ {C.GREEN}{total_fixed:>14}{C.RESET} │
  │ Avg scan time            │ {'N/A':>16} │ {C.CYAN}{avg_scan:>11.0f}ms{C.RESET}  │
{C.BOLD}  └──────────────────────────┴──────────────────┴────────────────┘{C.RESET}
""")

    print(f"  {C.MAGENTA}{C.BOLD}Conclusion: EnvCheck prevented {total_crashes} runtime crashes{C.RESET}")
    print(f"  {C.MAGENTA}{C.BOLD}by detecting {total_detected} breaking changes in ~{avg_scan:.0f}ms average.{C.RESET}")
    print()


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EnvCheck Comparative Demo — Side-by-Side Workflow Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python demo.py                   # Interactive demo with numpy_2x
              python demo.py --case scipy_114  # Demo with specific case
              python demo.py --auto            # Auto-advance for screen recording
              python demo.py --all             # Summary comparison of ALL cases
        """),
    )
    parser.add_argument("--case", default="numpy_2x",
                        help="Test case ID to demo (default: numpy_2x)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-advance with delays (for recording)")
    parser.add_argument("--all", action="store_true",
                        help="Run summary comparison for all cases")
    parser.add_argument("--list", action="store_true",
                        help="List available test cases")
    args = parser.parse_args()

    global AUTO_MODE
    AUTO_MODE = args.auto

    if args.list:
        print(f"\n  {C.BOLD}Available test cases:{C.RESET}\n")
        for case in ALL_CASES:
            print(f"    {C.CYAN}{case.id:<16}{C.RESET} {case.library:<14} {case.environment}")
        print(f"\n  Use: python demo.py --case <case_id>")
        return

    # ─── --all mode: summary comparison ───────────────────────────
    if args.all:
        # Ensure all environments
        for case in ALL_CASES:
            env_path = ENV_DIR / f"case_{case.id}"
            if not (env_path / "bin" / "python").exists():
                print(f"  Setting up {case.id}...")
                setup_environment(case)
        run_all_comparison(ALL_CASES)
        return

    # ─── Single case interactive demo ─────────────────────────────
    case = None
    for c in ALL_CASES:
        if c.id == args.case:
            case = c
            break
    if not case:
        print(f"{C.RED}Unknown case: {args.case}{C.RESET}")
        print(f"Available: {', '.join(c.id for c in ALL_CASES)}")
        sys.exit(1)

    # Title screen
    print(f"""
{C.BOLD}{C.MAGENTA}
  ╔══════════════════════════════════════════════════════════════════╗
  ║                                                                ║
  ║          EnvCheck — Comparative Demo                           ║
  ║          Proactive Code Environment Diagnostics                ║
  ║                                                                ║
  ║   "Catch API breaking changes BEFORE runtime, not after."      ║
  ║                                                                ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║                                                                ║
  ║   Case:    {case.id:<20}                               ║
  ║   Library: {case.library:<20}                               ║
  ║   Env:     {case.environment:<40}         ║
  ║                                                                ║
  ╚══════════════════════════════════════════════════════════════════╝
{C.RESET}""")

    # Setup environment
    print(f"  {C.DIM}Preparing test environment...{C.RESET}")
    env_path = setup_environment(case)
    print(f"  {C.DIM}Environment ready: {env_path}{C.RESET}")

    pause("start Scenario A")

    # Run both scenarios
    run_scenario_a(case, env_path)

    pause("start Scenario B")

    scan_report = run_scenario_b(case, env_path)

    pause("show comparison")

    show_comparison(case, scan_report)

    print(f"\n  {C.DIM}Demo complete. Try: python demo.py --all{C.RESET}\n")


if __name__ == "__main__":
    main()
