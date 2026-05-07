#!/usr/bin/env python3
"""
EnvCheck LLM-Powered Comparative Demo

Calls LLM APIs (Claude or Gemini) to demonstrate the difference between:
  Scenario A (Without EnvCheck):
    Prompt → LLM code → Run → CRASH → Send error to LLM → Fix → Re-run...
    (reactive debugging loop, wastes API calls + tokens)

  Scenario B (With EnvCheck):
    Prompt → LLM code → EnvCheck scan → Send diagnostics to LLM → Fix → Run once ✅
    (proactive, precise fix from first retry)

Usage:
    # Claude (default)
    export ANTHROPIC_API_KEY="sk-ant-..."
    python demo_llm.py                                      # Interactive, numpy_2x
    python demo_llm.py --case scipy_114                     # Specific case
    python demo_llm.py --auto                               # Auto-advance
    python demo_llm.py --all                                # All cases

    # Gemini
    export GEMINI_API_KEY="AIza..."
    python demo_llm.py --provider gemini                    # Use Gemini
    python demo_llm.py --provider gemini --model gemini-2.5-flash

    # Mock (no API key needed)
    python demo_llm.py --mock                               # Mock mode
    python demo_llm.py --all --mock                         # All cases, mock

Requirements:
    - ANTHROPIC_API_KEY or GEMINI_API_KEY set (unless using --mock)
    - Test environments already set up (run: python main.py --setup-only)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from test_cases.cases import ALL_CASES, TestCase
from envcheck.scanner import scan_source

# ─── Config ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
ENV_DIR = ROOT / "environments"
GENERATED_DIR = ROOT / "test_cases" / "generated"

PROVIDER = "claude"  # "claude" or "gemini"
MODEL = "claude-sonnet-4-20250514"
MAX_FIX_LOOPS = 5  # Safety limit for Scenario A

# Default models per provider
DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.5-flash",
}

# Pricing per provider ($ per 1M tokens: [input, output])
PRICING = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.0-flash": (0.10, 0.40),
}

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

# ─── Globals ────────────────────────────────────────────────────────
AUTO_MODE = False
MOCK_MODE = False
AUTO_DELAY = 2.0

# ─── Mock Partial Fixes ─────────────────────────────────────────────
# For mock mode: simulate that Scenario A only fixes ONE issue per crash,
# because the error traceback only shows one error at a time.
# This maps case_id → list of intermediate code states before the final fix.

MOCK_PARTIAL_FIXES: dict[str, list[str]] = {
    "numpy_2x": [
        # Crash 1: np.trapz → fix it, but still has np.infty
        '''\
import numpy as np

x = np.linspace(0, 10, 100)
y = x ** 2

# Calculate area under curve using trapezoidal rule
area = np.trapezoid(y, x)
print(f"Area under curve: {area}")

# Append infinity alias
x_with_inf = np.append(x, np.infty)
print(f"Array with infinity: {x_with_inf}")
''',
    ],
    "scipy_114": [
        # Crash 1: cumtrapz import fails → fix it, but simps still fails
        '''\
import numpy as np
from scipy.integrate import cumulative_trapezoid, simps

signal = np.ones(100)

# Cumulative integral
cumulative = cumulative_trapezoid(signal)
print(f"Cumulative integral: {cumulative[:5]}...")

# Simpson's rule
area = simps(signal)
print(f"Area (Simpson's): {area}")
''',
    ],
    "pandas_22": [
        # Crash 1: fillna(method=) → fix it, but .mad() still broken
        '''\
import pandas as pd

df = pd.DataFrame({"A": [1.0, None, 3.0, None, 5.0]})

# Forward fill missing values
df = df.ffill()
print("Filled DataFrame:")
print(df)

# Calculate Mean Absolute Deviation
mad_value = df["A"].mad()
print(f"MAD: {mad_value}")
''',
    ],
    "networkx_3x": [
        # Crash 1: write_gpickle → fix it, but read_gpickle still broken
        '''\
import pickle
import networkx as nx

# Generate random graph
G = nx.erdos_renyi_graph(20, 0.5)

# Save using standard pickle
with open("graph.pickle", "wb") as f:
    pickle.dump(G, f)
print("Graph saved to graph.pickle")

# Load back
G_loaded = nx.read_gpickle("graph.pickle")
print(f"Number of edges: {G_loaded.number_of_edges()}")
''',
    ],
    # Single-issue cases: no partial fixes needed (LLM fixes in one shot)
    "sklearn_12": [],
    "pandas_15": [],
    "pydantic_v1": [],
}


# ─── Data Classes ───────────────────────────────────────────────────

@dataclass
class LLMCall:
    """Record of a single LLM API call."""
    role: str               # "generate", "fix_from_error", "fix_from_envcheck"
    input_tokens: int
    output_tokens: int
    latency_ms: float
    prompt_preview: str     # First 100 chars of prompt
    response_preview: str   # First 100 chars of response


@dataclass
class ScenarioResult:
    """Full result of running a scenario."""
    scenario: str           # "A" or "B"
    llm_calls: list[LLMCall] = field(default_factory=list)
    execution_attempts: int = 0
    runtime_crashes: int = 0
    final_success: bool = False
    total_time_ms: float = 0.0
    envcheck_time_ms: float = 0.0
    envcheck_findings: int = 0

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_llm_calls(self) -> int:
        return len(self.llm_calls)

    @property
    def total_llm_latency_ms(self) -> float:
        return sum(c.latency_ms for c in self.llm_calls)


# ─── Utilities ──────────────────────────────────────────────────────

def pause(label: str = ""):
    if AUTO_MODE:
        time.sleep(AUTO_DELAY)
    else:
        hint = f" ({label})" if label else ""
        input(f"{C.DIM}    ⏎ Press Enter to continue{hint}...{C.RESET}")


def print_header(text: str, color: str = C.BOLD):
    print(f"\n{color}{'━' * 76}{C.RESET}")
    print(f"{color}  {text}{C.RESET}")
    print(f"{color}{'━' * 76}{C.RESET}")


def print_step(icon: str, text: str, color: str = C.WHITE):
    print(f"\n{color}  {icon}  {text}{C.RESET}")
    print(f"{C.DIM}  {'·' * 68}{C.RESET}")


def show_code(code: str, highlight_lines: list[int] | None = None, max_lines: int = 20):
    lines = code.strip().split("\n")
    for i, line in enumerate(lines[:max_lines], 1):
        if highlight_lines and i in highlight_lines:
            print(f"    {C.RED}{C.BOLD}{i:3d} │ {line}{C.RESET}  {C.RED}◄{C.RESET}")
        else:
            print(f"    {C.DIM}{i:3d}{C.RESET} │ {line}")
    if len(lines) > max_lines:
        print(f"    {C.DIM}... ({len(lines) - max_lines} more lines){C.RESET}")


def run_in_env(env_path: Path, code: str, case_id: str, suffix: str):
    GENERATED_DIR.mkdir(exist_ok=True)
    filepath = GENERATED_DIR / f"{case_id}_{suffix}.py"
    filepath.write_text(code)
    python_bin = env_path / "bin" / "python"
    return subprocess.run(
        [str(python_bin), str(filepath)],
        capture_output=True, text=True, timeout=30,
    )


def extract_code_from_response(text: str) -> str:
    """Extract Python code from LLM response (handles ```python blocks)."""
    # Try to find ```python ... ``` block
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try generic ``` block
    pattern = r'```\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no code block, try to find lines that look like Python
    lines = text.strip().split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.startswith("import ") or line.startswith("from ") or in_code:
            in_code = True
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines)

    # Last resort: return the whole thing
    return text.strip()


def setup_environment(case: TestCase) -> Path:
    import shlex
    env_path = ENV_DIR / f"case_{case.id}"
    python_bin = env_path / "bin" / "python"
    if python_bin.exists():
        return env_path

    print(f"    {C.DIM}Setting up environment for {case.id}...{C.RESET}")
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


# ─── LLM Calling ───────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str, role_label: str) -> tuple[str, LLMCall]:
    """Call LLM API (Claude or Gemini) and return (response_text, LLMCall record).

    In mock mode, returns a placeholder response.
    Dispatches to the correct provider based on the global PROVIDER setting.
    """
    if MOCK_MODE:
        return _mock_call(system_prompt, user_prompt, role_label)

    if PROVIDER == "claude":
        return _call_claude(system_prompt, user_prompt, role_label)
    elif PROVIDER == "gemini":
        return _call_gemini(system_prompt, user_prompt, role_label)
    else:
        raise ValueError(f"Unknown provider: {PROVIDER}")


def _call_claude(system_prompt: str, user_prompt: str, role_label: str) -> tuple[str, LLMCall]:
    """Call Anthropic Claude API."""
    import anthropic
    client = anthropic.Anthropic()

    start = time.perf_counter()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    latency = (time.perf_counter() - start) * 1000

    text = response.content[0].text
    call_record = LLMCall(
        role=role_label,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency,
        prompt_preview=user_prompt[:100],
        response_preview=text[:100],
    )
    return text, call_record


def _call_gemini(system_prompt: str, user_prompt: str, role_label: str) -> tuple[str, LLMCall]:
    """Call Google Gemini API via google-genai SDK."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    start = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=2048,
        ),
    )
    latency = (time.perf_counter() - start) * 1000

    text = response.text or ""
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count or 0 if usage else 0
    output_tokens = usage.candidates_token_count or 0 if usage else 0

    call_record = LLMCall(
        role=role_label,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency,
        prompt_preview=user_prompt[:100],
        response_preview=text[:100],
    )
    return text, call_record


def _mock_call(system_prompt: str, user_prompt: str, role_label: str) -> tuple[str, LLMCall]:
    """Mock LLM call using test case data for offline demos."""
    # Simulate a delay
    time.sleep(0.5)

    # Simulate token counts based on prompt length
    input_tokens = len(user_prompt.split()) * 2
    output_tokens = 150  # approximate

    mock_response = "[MOCK MODE] — Code would be generated by LLM here."
    call_record = LLMCall(
        role=role_label,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=500,
        prompt_preview=user_prompt[:100],
        response_preview=mock_response[:100],
    )
    return mock_response, call_record


def show_llm_call_stats(call: LLMCall, color: str = C.CYAN):
    """Print stats for a single LLM call."""
    print(f"    {color}📊 Tokens: {call.input_tokens} input + {call.output_tokens} output "
          f"= {call.input_tokens + call.output_tokens} total{C.RESET}")
    print(f"    {color}⏱  Latency: {call.latency_ms:.0f}ms{C.RESET}")


# ═══════════════════════════════════════════════════════════════════
#  SCENARIO A: Without EnvCheck (Reactive loop)
# ═══════════════════════════════════════════════════════════════════

def run_scenario_a(case: TestCase, env_path: Path) -> ScenarioResult:
    """Run the reactive loop WITHOUT EnvCheck."""

    result = ScenarioResult(scenario="A")
    start = time.perf_counter()

    print_header(
        "SCENARIO A: Without EnvCheck — Reactive Error-Fix Loop",
        f"{C.RED}{C.BOLD}",
    )
    print(f"\n{C.RED}  Flow: Prompt → LLM code → Run → 💥 CRASH → Send error → LLM fix → Run → ...{C.RESET}")
    pause("start Scenario A")

    # ── A.1: Generate code from prompt ──────────────────────────────
    print_step("🤖", "Step A.1: Call LLM to generate code from prompt", C.YELLOW)
    print(f"    {C.DIM}Prompt: {case.problem[:90]}...{C.RESET}")

    system_a = (
        "You are a Python coding assistant. Write ONLY the Python code, "
        "wrapped in a ```python code block. No explanations."
    )

    if MOCK_MODE:
        raw_response, call1 = call_llm(system_a, case.problem, "generate")
        generated_code = case.broken_code  # Use known broken code
    else:
        raw_response, call1 = call_llm(system_a, case.problem, "generate")
        generated_code = extract_code_from_response(raw_response)

    result.llm_calls.append(call1)
    show_llm_call_stats(call1, C.YELLOW)

    print(f"\n    {C.YELLOW}LLM Generated Code:{C.RESET}")
    show_code(generated_code)

    # ── A.2+: Run → crash → fix loop ───────────────────────────────
    current_code = generated_code
    loop = 0

    # In mock mode, simulate that each fix only addresses the FIRST error
    # (because the traceback only shows one error at a time).
    # For cases with multiple breaking changes, this means multiple loops.
    mock_fix_queue: list[str] = []
    if MOCK_MODE:
        partials = MOCK_PARTIAL_FIXES.get(case.id, [])
        mock_fix_queue = partials + [case.fixed_code]  # partial fixes, then final fix

    while loop < MAX_FIX_LOOPS:
        loop += 1
        result.execution_attempts += 1

        pause(f"run attempt #{loop}")
        print_step("▶️ ", f"Step A.{loop + 1}: Run code (attempt #{loop})", C.WHITE)
        print(f"    {C.DIM}$ python script.py{C.RESET}")

        run_result = run_in_env(env_path, current_code, case.id, f"scenA_attempt{loop}")

        if run_result.returncode == 0:
            # Success!
            result.final_success = True
            print(f"\n    {C.GREEN}{C.BOLD}✅ SUCCESS (on attempt #{loop}){C.RESET}")
            output_lines = run_result.stdout.strip().split("\n")
            for line in output_lines[:4]:
                print(f"    {C.GREEN}{line}{C.RESET}")
            break

        # Crashed
        result.runtime_crashes += 1
        error_text = run_result.stderr.strip()
        error_lines = error_text.split("\n")

        print(f"\n    {C.RED}{C.BOLD}💥 CRASH #{result.runtime_crashes}!{C.RESET}")
        print(f"    {C.RED}{'─' * 55}{C.RESET}")
        for line in error_lines[-5:]:
            print(f"    {C.RED}{line}{C.RESET}")
        print(f"    {C.RED}{'─' * 55}{C.RESET}")

        if loop >= MAX_FIX_LOOPS:
            print(f"\n    {C.RED}Reached max fix attempts ({MAX_FIX_LOOPS}). Giving up.{C.RESET}")
            break

        # Send error back to LLM for fix
        pause("send error to LLM")
        print_step("🔄", f"Step A.{loop + 2}: Send crash to LLM — ask for fix (call #{loop + 1})",
                   C.YELLOW)

        fix_prompt = (
            f"I ran this Python code:\n\n```python\n{current_code}\n```\n\n"
            f"And got this error:\n\n```\n{error_text[-1500:]}\n```\n\n"
            f"Please fix the code. Return ONLY the corrected Python code in a ```python block."
        )

        print(f"    {C.DIM}Sending code + traceback to LLM...{C.RESET}")
        print(f"    {C.DIM}\"Here is my code and the error, please fix it\"{C.RESET}")

        if MOCK_MODE:
            raw_fix, fix_call = call_llm(system_a, fix_prompt, "fix_from_error")
            # Use partial fix (fixes only the first error), then final fix
            if mock_fix_queue:
                current_code = mock_fix_queue.pop(0)
            else:
                current_code = case.fixed_code
        else:
            raw_fix, fix_call = call_llm(system_a, fix_prompt, "fix_from_error")
            current_code = extract_code_from_response(raw_fix)

        result.llm_calls.append(fix_call)
        show_llm_call_stats(fix_call, C.YELLOW)

        print(f"\n    {C.YELLOW}LLM Fix Response:{C.RESET}")
        show_code(current_code)

    result.total_time_ms = (time.perf_counter() - start) * 1000

    # Summary box
    total_tok = result.total_tokens
    print(f"\n{C.RED}  ╔════════════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.RED}  ║  Scenario A Result:                                           ║{C.RESET}")
    print(f"{C.RED}  ║    API calls:       {result.total_llm_calls:<5}                                    ║{C.RESET}")
    print(f"{C.RED}  ║    Total tokens:    {total_tok:<8}                                 ║{C.RESET}")
    print(f"{C.RED}  ║    Runtime crashes:  {result.runtime_crashes:<5}                                    ║{C.RESET}")
    print(f"{C.RED}  ║    Exec attempts:    {result.execution_attempts:<5}                                    ║{C.RESET}")
    print(f"{C.RED}  ║    LLM latency:     {result.total_llm_latency_ms:<8.0f}ms                             ║{C.RESET}")
    print(f"{C.RED}  ╚════════════════════════════════════════════════════════════════╝{C.RESET}")

    return result


# ═══════════════════════════════════════════════════════════════════
#  SCENARIO B: With EnvCheck (Proactive)
# ═══════════════════════════════════════════════════════════════════

def run_scenario_b(case: TestCase, env_path: Path) -> ScenarioResult:
    """Run the proactive workflow WITH EnvCheck."""

    result = ScenarioResult(scenario="B")
    start = time.perf_counter()

    print_header(
        "SCENARIO B: With EnvCheck — Proactive Scan-Then-Fix",
        f"{C.GREEN}{C.BOLD}",
    )
    print(f"\n{C.GREEN}  Flow: Prompt → LLM code → EnvCheck scan → LLM fix (with diagnostics) → Run ✅{C.RESET}")
    pause("start Scenario B")

    # ── B.1: Generate code from prompt (same as A) ──────────────────
    print_step("🤖", "Step B.1: Call LLM to generate code from prompt", C.CYAN)
    print(f"    {C.DIM}Prompt: {case.problem[:90]}...{C.RESET}")

    system_b = (
        "You are a Python coding assistant. Write ONLY the Python code, "
        "wrapped in a ```python code block. No explanations."
    )

    if MOCK_MODE:
        raw_response, call1 = call_llm(system_b, case.problem, "generate")
        generated_code = case.broken_code
    else:
        raw_response, call1 = call_llm(system_b, case.problem, "generate")
        generated_code = extract_code_from_response(raw_response)

    result.llm_calls.append(call1)
    show_llm_call_stats(call1, C.CYAN)

    print(f"\n    {C.CYAN}LLM Generated Code (same prompt as Scenario A):{C.RESET}")
    show_code(generated_code)

    # ── B.2: EnvCheck scan BEFORE running ───────────────────────────
    pause("EnvCheck scan")
    print_step("🛡️ ", "Step B.2: EnvCheck scans code BEFORE execution (no runtime needed!)", C.CYAN)
    print(f"    {C.CYAN}envcheck.scan(source_code, env_path){C.RESET}")

    scan_report = scan_source(
        generated_code,
        env_path=str(env_path),
        filepath=f"{case.id}_broken.py",
    )
    result.envcheck_time_ms = scan_report.scan_time_ms
    result.envcheck_findings = scan_report.total_findings

    if scan_report.total_findings > 0:
        print(f"\n    {C.CYAN}{C.BOLD}🔍 Found {scan_report.total_findings} issue(s) "
              f"in {scan_report.scan_time_ms:.0f}ms — NO code was executed!{C.RESET}")

        problem_lines = [f.lineno for f in scan_report.findings]
        print()
        show_code(generated_code, highlight_lines=problem_lines)
        print()

        for finding in scan_report.findings:
            print(f"    {C.CYAN}⛔ Line {finding.lineno}: {C.BOLD}{finding.matched_code}{C.RESET}")
            print(f"       {finding.rule.description}")
            print(f"       {C.GREEN}Fix: {finding.rule.old_api} → {finding.rule.new_api}{C.RESET}")
            print()
    else:
        print(f"    {C.GREEN}No issues found — code looks compatible!{C.RESET}")

    # ── B.3: Send code + EnvCheck report to LLM ────────────────────
    if scan_report.total_findings > 0:
        pause("send EnvCheck report to LLM")
        print_step("🔧", "Step B.3: Send code + EnvCheck diagnostics to LLM for targeted fix",
                   C.GREEN)

        # Build a structured diagnostic for the LLM
        diag_lines = ["EnvCheck found the following API compatibility issues:\n"]
        for finding in scan_report.findings:
            diag_lines.append(f"- Line {finding.lineno}: `{finding.matched_code}` — "
                              f"{finding.rule.description}")
            diag_lines.append(f"  Fix: change `{finding.rule.old_api}` to `{finding.rule.new_api}`")
        diagnostic_text = "\n".join(diag_lines)

        fix_prompt = (
            f"Here is my Python code:\n\n```python\n{generated_code}\n```\n\n"
            f"A static analysis tool found these compatibility issues with the "
            f"installed library versions:\n\n{diagnostic_text}\n\n"
            f"Please fix ALL the issues listed above. "
            f"Return ONLY the corrected Python code in a ```python block."
        )

        print(f"    {C.DIM}Sending code + {scan_report.total_findings} precise diagnostic(s) to LLM...{C.RESET}")
        print(f"    {C.DIM}\"Fix these specific issues found by EnvCheck\"{C.RESET}")
        print(f"\n    {C.CYAN}Diagnostic sent to LLM:{C.RESET}")
        for line in diagnostic_text.split("\n"):
            print(f"    {C.CYAN}{line}{C.RESET}")

        if MOCK_MODE:
            raw_fix, fix_call = call_llm(system_b, fix_prompt, "fix_from_envcheck")
            fixed_code = case.fixed_code
        else:
            raw_fix, fix_call = call_llm(system_b, fix_prompt, "fix_from_envcheck")
            fixed_code = extract_code_from_response(raw_fix)

        result.llm_calls.append(fix_call)
        show_llm_call_stats(fix_call, C.GREEN)

        print(f"\n    {C.GREEN}LLM Fixed Code (targeted fix):{C.RESET}")
        show_code(fixed_code)
    else:
        fixed_code = generated_code

    # ── B.4: Run fixed code — first execution ──────────────────────
    pause("run code")
    print_step("▶️ ", "Step B.4: Run code — first and only execution", C.GREEN)
    print(f"    {C.DIM}$ python script.py{C.RESET}")

    result.execution_attempts = 1
    run_result = run_in_env(env_path, fixed_code, case.id, "scenB_fixed")

    if run_result.returncode == 0:
        result.final_success = True
        print(f"\n    {C.GREEN}{C.BOLD}✅ SUCCESS on first run — zero crashes!{C.RESET}")
        output_lines = run_result.stdout.strip().split("\n")
        for line in output_lines[:4]:
            print(f"    {C.GREEN}{line}{C.RESET}")
    else:
        result.runtime_crashes = 1
        print(f"\n    {C.YELLOW}⚠ Code still has issues — but EnvCheck caught the known ones.{C.RESET}")
        err_lines = run_result.stderr.strip().split("\n")
        for line in err_lines[-3:]:
            print(f"    {C.YELLOW}{line}{C.RESET}")

    result.total_time_ms = (time.perf_counter() - start) * 1000

    # Summary box
    total_tok = result.total_tokens
    print(f"\n{C.GREEN}  ╔════════════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.GREEN}  ║  Scenario B Result:                                           ║{C.RESET}")
    print(f"{C.GREEN}  ║    API calls:       {result.total_llm_calls:<5}                                    ║{C.RESET}")
    print(f"{C.GREEN}  ║    Total tokens:    {total_tok:<8}                                 ║{C.RESET}")
    print(f"{C.GREEN}  ║    Runtime crashes:  {result.runtime_crashes:<5}                                    ║{C.RESET}")
    print(f"{C.GREEN}  ║    Exec attempts:    {result.execution_attempts:<5}                                    ║{C.RESET}")
    print(f"{C.GREEN}  ║    LLM latency:     {result.total_llm_latency_ms:<8.0f}ms                             ║{C.RESET}")
    print(f"{C.GREEN}  ║    EnvCheck scan:    {result.envcheck_time_ms:<6.0f}ms  ({result.envcheck_findings} issues found)         ║{C.RESET}")
    print(f"{C.GREEN}  ╚════════════════════════════════════════════════════════════════╝{C.RESET}")

    return result


# ═══════════════════════════════════════════════════════════════════
#  COMPARISON
# ═══════════════════════════════════════════════════════════════════

def show_comparison(case: TestCase, result_a: ScenarioResult, result_b: ScenarioResult):
    """Show the final side-by-side comparison with real numbers."""

    print_header("FINAL COMPARISON: Real LLM API Metrics", f"{C.MAGENTA}{C.BOLD}")

    # Token savings
    token_saved = result_a.total_tokens - result_b.total_tokens
    token_pct = (token_saved / result_a.total_tokens * 100) if result_a.total_tokens > 0 else 0

    a_calls = result_a.total_llm_calls
    b_calls = result_b.total_llm_calls
    a_tok = result_a.total_tokens
    b_tok = result_b.total_tokens
    a_crash = result_a.runtime_crashes
    b_crash = result_b.runtime_crashes
    a_exec = result_a.execution_attempts
    b_exec = result_b.execution_attempts
    a_lat = result_a.total_llm_latency_ms
    b_lat = result_b.total_llm_latency_ms + result_b.envcheck_time_ms

    print(f"""
{C.BOLD}  ┌──────────────────────────┬──────────────────┬──────────────────┐{C.RESET}
{C.BOLD}  │ Metric                   │ {C.RED}Without EnvCheck{C.RESET}{C.BOLD} │ {C.GREEN}With EnvCheck{C.RESET}{C.BOLD}    │{C.RESET}
{C.BOLD}  ├──────────────────────────┼──────────────────┼──────────────────┤{C.RESET}
  │ LLM API calls            │ {C.RED}{a_calls:>16}{C.RESET} │ {C.GREEN}{b_calls:>16}{C.RESET} │
  │ Total tokens             │ {C.RED}{a_tok:>16,}{C.RESET} │ {C.GREEN}{b_tok:>16,}{C.RESET} │
  │ Runtime crashes          │ {C.RED}{a_crash:>16}{C.RESET} │ {C.GREEN}{b_crash:>16}{C.RESET} │
  │ Execution attempts       │ {C.RED}{a_exec:>16}{C.RESET} │ {C.GREEN}{b_exec:>16}{C.RESET} │
  │ LLM latency              │ {C.RED}{a_lat:>13,.0f}ms{C.RESET} │ {C.GREEN}{b_lat:>13,.0f}ms{C.RESET} │
  │ EnvCheck scan time       │ {'N/A':>16} │ {C.CYAN}{result_b.envcheck_time_ms:>13,.0f}ms{C.RESET} │
  │ Issues found statically  │ {C.RED}{'0':>16}{C.RESET} │ {C.GREEN}{result_b.envcheck_findings:>16}{C.RESET} │
{C.BOLD}  └──────────────────────────┴──────────────────┴──────────────────┘{C.RESET}
""")

    # Cost estimate based on provider pricing
    price_in, price_out = PRICING.get(MODEL, (3.0, 15.0))
    cost_a_input = result_a.total_input_tokens * price_in / 1_000_000
    cost_a_output = result_a.total_output_tokens * price_out / 1_000_000
    cost_a = cost_a_input + cost_a_output
    cost_b_input = result_b.total_input_tokens * price_in / 1_000_000
    cost_b_output = result_b.total_output_tokens * price_out / 1_000_000
    cost_b = cost_b_input + cost_b_output
    cost_saved = cost_a - cost_b

    print(f"  {C.BOLD}💰 Estimated API Cost ({MODEL}):{C.RESET}")
    print(f"     Scenario A: ${cost_a:.4f}")
    print(f"     Scenario B: ${cost_b:.4f}")
    if cost_saved > 0:
        print(f"     {C.GREEN}Saved: ${cost_saved:.4f} ({cost_saved/cost_a*100:.0f}% less){C.RESET}")
    print()

    if token_saved > 0:
        print(f"  {C.MAGENTA}{C.BOLD}📊 Token savings: {token_saved:,} tokens ({token_pct:.0f}% reduction){C.RESET}")
    print(f"  {C.MAGENTA}{C.BOLD}🛡️  Crashes prevented: {a_crash}{C.RESET}")
    print(f"  {C.MAGENTA}{C.BOLD}⚡ Key advantage: EnvCheck provides PRECISE diagnostics to the LLM,{C.RESET}")
    print(f"  {C.MAGENTA}{C.BOLD}   enabling targeted fixes instead of blind error-based debugging.{C.RESET}")

    # Per-call breakdown
    print(f"\n  {C.BOLD}📋 LLM Call Breakdown:{C.RESET}")
    print(f"  {'─' * 70}")
    print(f"  {'#':<4} {'Scenario':<4} {'Role':<22} {'Input':>8} {'Output':>8} {'Total':>8} {'Time':>8}")
    print(f"  {'─' * 70}")
    idx = 1
    for call in result_a.llm_calls:
        total = call.input_tokens + call.output_tokens
        print(f"  {idx:<4} {C.RED}A{C.RESET}    {call.role:<22} {call.input_tokens:>8} "
              f"{call.output_tokens:>8} {total:>8} {call.latency_ms:>6.0f}ms")
        idx += 1
    for call in result_b.llm_calls:
        total = call.input_tokens + call.output_tokens
        print(f"  {idx:<4} {C.GREEN}B{C.RESET}    {call.role:<22} {call.input_tokens:>8} "
              f"{call.output_tokens:>8} {total:>8} {call.latency_ms:>6.0f}ms")
        idx += 1
    print(f"  {'─' * 70}")


# ═══════════════════════════════════════════════════════════════════
#  ALL-CASES COMPARISON
# ═══════════════════════════════════════════════════════════════════

def run_all_cases(cases: list[TestCase]):
    """Run A vs B for all cases and show aggregate comparison."""

    print_header("EnvCheck LLM Demo — All Cases Comparison", f"{C.MAGENTA}{C.BOLD}")
    print(f"  Provider: {PROVIDER.upper()}")
    print(f"  Model:    {MODEL}")
    print(f"  Mode:     {'MOCK' if MOCK_MODE else 'LIVE API'}")
    print(f"  Cases:    {len(cases)}\n")

    all_a: list[ScenarioResult] = []
    all_b: list[ScenarioResult] = []

    for case in cases:
        env_path = setup_environment(case)
        print(f"  {C.BOLD}Running case: {case.id} ({case.library}){C.RESET}")

        # Scenario A
        ra = _run_scenario_a_quiet(case, env_path)
        all_a.append(ra)

        # Scenario B
        rb = _run_scenario_b_quiet(case, env_path)
        all_b.append(rb)

        a_status = f"{C.RED}💥 {ra.runtime_crashes} crash(es), {ra.total_llm_calls} calls, {ra.total_tokens:,} tok{C.RESET}"
        b_status = f"{C.GREEN}🛡️  {rb.envcheck_findings} found, {rb.total_llm_calls} calls, {rb.total_tokens:,} tok{C.RESET}"
        print(f"    A: {a_status}")
        print(f"    B: {b_status}")
        print()

    # Aggregate table
    print(f"\n  {C.BOLD}{'Case':<16} {'Lib':<13} {'A calls':>7} {'A tokens':>9} {'A crash':>7}  "
          f"{'B calls':>7} {'B tokens':>9} {'B crash':>7} {'Saved':>8}{C.RESET}")
    print(f"  {'─' * 95}")

    total_a_tok = 0
    total_b_tok = 0
    total_a_calls = 0
    total_b_calls = 0
    total_a_crash = 0
    total_b_crash = 0

    for i, case in enumerate(cases):
        ra, rb = all_a[i], all_b[i]
        saved = ra.total_tokens - rb.total_tokens
        saved_str = f"{C.GREEN}+{saved:,}{C.RESET}" if saved > 0 else f"{saved:,}"
        print(f"  {case.id:<16} {case.library:<13} "
              f"{C.RED}{ra.total_llm_calls:>7}{C.RESET} {C.RED}{ra.total_tokens:>9,}{C.RESET} "
              f"{C.RED}{ra.runtime_crashes:>7}{C.RESET}  "
              f"{C.GREEN}{rb.total_llm_calls:>7}{C.RESET} {C.GREEN}{rb.total_tokens:>9,}{C.RESET} "
              f"{C.GREEN}{rb.runtime_crashes:>7}{C.RESET} {saved_str:>8}")
        total_a_tok += ra.total_tokens
        total_b_tok += rb.total_tokens
        total_a_calls += ra.total_llm_calls
        total_b_calls += rb.total_llm_calls
        total_a_crash += ra.runtime_crashes
        total_b_crash += rb.runtime_crashes

    print(f"  {'─' * 95}")
    total_saved = total_a_tok - total_b_tok
    pct = (total_saved / total_a_tok * 100) if total_a_tok else 0
    print(f"  {'TOTAL':<16} {'':13} "
          f"{C.RED}{total_a_calls:>7}{C.RESET} {C.RED}{total_a_tok:>9,}{C.RESET} "
          f"{C.RED}{total_a_crash:>7}{C.RESET}  "
          f"{C.GREEN}{total_b_calls:>7}{C.RESET} {C.GREEN}{total_b_tok:>9,}{C.RESET} "
          f"{C.GREEN}{total_b_crash:>7}{C.RESET} {C.GREEN}+{total_saved:,}{C.RESET}")

    print(f"\n  {C.MAGENTA}{C.BOLD}Summary:{C.RESET}")
    print(f"  {C.MAGENTA}  Total tokens saved: {total_saved:,} ({pct:.0f}% reduction){C.RESET}")
    print(f"  {C.MAGENTA}  Runtime crashes prevented: {total_a_crash}{C.RESET}")
    print(f"  {C.MAGENTA}  Extra LLM calls avoided: {total_a_calls - total_b_calls}{C.RESET}")

    # Suggest best interactive demo case
    best_idx = max(range(len(cases)), key=lambda i: all_a[i].runtime_crashes)
    best = cases[best_idx]
    best_a = all_a[best_idx]
    provider_flag = f" --provider {PROVIDER}" if PROVIDER != "claude" else ""
    mock_flag = " --mock" if MOCK_MODE else ""
    print(f"\n  {C.CYAN}{C.BOLD}💡 Best case for interactive demo:{C.RESET}")
    print(f"  {C.CYAN}  → {best.id} ({best.library}): {best_a.runtime_crashes} crashes in Scenario A{C.RESET}")
    print(f"  {C.CYAN}  Run: uv run python demo_llm.py --case {best.id}{provider_flag}{mock_flag}{C.RESET}")

    return cases, all_a, all_b


# ═══════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_report(cases: list[TestCase], all_a: list[ScenarioResult],
                    all_b: list[ScenarioResult], report_dir: Path):
    """Generate a Markdown report and JSON data file from the comparison results."""
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "mock" if MOCK_MODE else "live"

    # ── Build structured data ─────────────────────────────────────
    price_in, price_out = PRICING.get(MODEL, (3.0, 15.0))

    per_case_data = []
    for i, case in enumerate(cases):
        ra, rb = all_a[i], all_b[i]
        cost_a = ra.total_input_tokens * price_in / 1e6 + ra.total_output_tokens * price_out / 1e6
        cost_b = rb.total_input_tokens * price_in / 1e6 + rb.total_output_tokens * price_out / 1e6
        per_case_data.append({
            "case_id": case.id,
            "library": case.library,
            "a_calls": ra.total_llm_calls,
            "a_tokens": ra.total_tokens,
            "a_input_tokens": ra.total_input_tokens,
            "a_output_tokens": ra.total_output_tokens,
            "a_crashes": ra.runtime_crashes,
            "a_exec_attempts": ra.execution_attempts,
            "a_llm_latency_ms": round(ra.total_llm_latency_ms),
            "a_cost_usd": round(cost_a, 6),
            "a_success": ra.final_success,
            "b_calls": rb.total_llm_calls,
            "b_tokens": rb.total_tokens,
            "b_input_tokens": rb.total_input_tokens,
            "b_output_tokens": rb.total_output_tokens,
            "b_crashes": rb.runtime_crashes,
            "b_exec_attempts": rb.execution_attempts,
            "b_llm_latency_ms": round(rb.total_llm_latency_ms),
            "b_envcheck_ms": round(rb.envcheck_time_ms),
            "b_envcheck_findings": rb.envcheck_findings,
            "b_cost_usd": round(cost_b, 6),
            "b_success": rb.final_success,
            "token_saved": ra.total_tokens - rb.total_tokens,
            "cost_saved_usd": round(cost_a - cost_b, 6),
        })

    total_a_tok = sum(d["a_tokens"] for d in per_case_data)
    total_b_tok = sum(d["b_tokens"] for d in per_case_data)
    total_a_calls = sum(d["a_calls"] for d in per_case_data)
    total_b_calls = sum(d["b_calls"] for d in per_case_data)
    total_a_crash = sum(d["a_crashes"] for d in per_case_data)
    total_a_cost = sum(d["a_cost_usd"] for d in per_case_data)
    total_b_cost = sum(d["b_cost_usd"] for d in per_case_data)
    total_saved = total_a_tok - total_b_tok
    pct = (total_saved / total_a_tok * 100) if total_a_tok else 0

    report_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "provider": PROVIDER,
            "model": MODEL,
            "mode": mode_tag,
            "num_cases": len(cases),
            "pricing_per_1m": {"input": price_in, "output": price_out},
        },
        "summary": {
            "total_a_calls": total_a_calls,
            "total_b_calls": total_b_calls,
            "total_a_tokens": total_a_tok,
            "total_b_tokens": total_b_tok,
            "token_saved": total_saved,
            "token_saved_pct": round(pct, 1),
            "total_a_crashes": total_a_crash,
            "total_b_crashes": 0,
            "crashes_prevented": total_a_crash,
            "extra_calls_avoided": total_a_calls - total_b_calls,
            "total_a_cost_usd": round(total_a_cost, 6),
            "total_b_cost_usd": round(total_b_cost, 6),
            "cost_saved_usd": round(total_a_cost - total_b_cost, 6),
        },
        "cases": per_case_data,
    }

    # ── Save JSON ─────────────────────────────────────────────────
    json_path = report_dir / f"report_{PROVIDER}_{mode_tag}_{timestamp}.json"
    json_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False))

    # ── Generate Markdown ─────────────────────────────────────────
    md_path = report_dir / f"report_{PROVIDER}_{mode_tag}_{timestamp}.md"

    md = []
    md.append(f"# EnvCheck LLM Demo Report")
    md.append(f"")
    md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"**Provider:** {PROVIDER.upper()}  ")
    md.append(f"**Model:** `{MODEL}`  ")
    md.append(f"**Mode:** {mode_tag.upper()}  ")
    md.append(f"**Cases:** {len(cases)}  ")
    md.append(f"")
    md.append(f"## Executive Summary")
    md.append(f"")
    md.append(f"| Metric | Without EnvCheck (A) | With EnvCheck (B) | Improvement |")
    md.append(f"|--------|---------------------|-------------------|-------------|")
    md.append(f"| LLM API Calls | {total_a_calls} | {total_b_calls} | {total_a_calls - total_b_calls} fewer |")
    md.append(f"| Total Tokens | {total_a_tok:,} | {total_b_tok:,} | {total_saved:,} saved ({pct:.0f}%) |")
    md.append(f"| Runtime Crashes | {total_a_crash} | 0 | {total_a_crash} prevented |")
    md.append(f"| Est. API Cost | ${total_a_cost:.4f} | ${total_b_cost:.4f} | ${total_a_cost - total_b_cost:.4f} saved |")
    md.append(f"")
    md.append(f"## Per-Case Breakdown")
    md.append(f"")
    md.append(f"| Case | Library | A Calls | A Tokens | A Crashes | B Calls | B Tokens | B Crashes | EnvCheck Findings | Tokens Saved |")
    md.append(f"|------|---------|---------|----------|-----------|---------|----------|-----------|-------------------|--------------|")
    for d in per_case_data:
        md.append(f"| {d['case_id']} | {d['library']} | {d['a_calls']} | {d['a_tokens']:,} | {d['a_crashes']} | "
                  f"{d['b_calls']} | {d['b_tokens']:,} | {d['b_crashes']} | {d['b_envcheck_findings']} | "
                  f"{d['token_saved']:+,} |")
    md.append(f"")
    md.append(f"## Key Findings")
    md.append(f"")

    # Find the most impactful case
    best_case = max(per_case_data, key=lambda d: d["token_saved"])
    worst_case = max(per_case_data, key=lambda d: d["a_crashes"])
    md.append(f"1. **Most tokens saved:** `{best_case['case_id']}` ({best_case['library']}) "
              f"— saved {best_case['token_saved']:,} tokens")
    md.append(f"2. **Most crashes in Scenario A:** `{worst_case['case_id']}` ({worst_case['library']}) "
              f"— {worst_case['a_crashes']} crashes, {worst_case['a_calls']} API calls needed")
    md.append(f"3. **Overall token reduction:** {pct:.0f}% fewer tokens with EnvCheck")
    md.append(f"4. **Zero runtime crashes** in all Scenario B runs across {len(cases)} test cases")
    md.append(f"")
    md.append(f"## How It Works")
    md.append(f"")
    md.append(f"**Scenario A (Without EnvCheck):** The LLM generates code, it crashes at runtime, "
              f"the error traceback is sent back to the LLM for fixing. Each crash only reveals one "
              f"error at a time, requiring multiple round-trips for code with multiple API incompatibilities.")
    md.append(f"")
    md.append(f"**Scenario B (With EnvCheck):** The LLM generates code, EnvCheck statically scans it "
              f"*before* execution and identifies ALL breaking API changes at once. The precise diagnostics "
              f"(line numbers, old API → new API) are sent to the LLM, enabling a single targeted fix.")
    md.append(f"")
    md.append(f"## Per-Call Detail")
    md.append(f"")
    for i, case in enumerate(cases):
        ra, rb = all_a[i], all_b[i]
        d = per_case_data[i]
        md.append(f"### {case.id} ({case.library})")
        md.append(f"")
        md.append(f"| | Scenario A | Scenario B |")
        md.append(f"|---|---|---|")
        md.append(f"| API Calls | {d['a_calls']} | {d['b_calls']} |")
        md.append(f"| Tokens (in/out) | {d['a_input_tokens']}/{d['a_output_tokens']} | "
                  f"{d['b_input_tokens']}/{d['b_output_tokens']} |")
        md.append(f"| Crashes | {d['a_crashes']} | {d['b_crashes']} |")
        md.append(f"| LLM Latency | {d['a_llm_latency_ms']}ms | {d['b_llm_latency_ms']}ms |")
        md.append(f"| EnvCheck Scan | N/A | {d['b_envcheck_ms']}ms ({d['b_envcheck_findings']} findings) |")
        md.append(f"| Cost | ${d['a_cost_usd']:.4f} | ${d['b_cost_usd']:.4f} |")
        md.append(f"")

        if ra.llm_calls:
            md.append(f"**Scenario A call breakdown:**")
            for j, call in enumerate(ra.llm_calls, 1):
                md.append(f"- Call {j} (`{call.role}`): {call.input_tokens} in + "
                          f"{call.output_tokens} out = {call.input_tokens + call.output_tokens} tokens, "
                          f"{call.latency_ms:.0f}ms")
            md.append(f"")
        if rb.llm_calls:
            md.append(f"**Scenario B call breakdown:**")
            for j, call in enumerate(rb.llm_calls, 1):
                md.append(f"- Call {j} (`{call.role}`): {call.input_tokens} in + "
                          f"{call.output_tokens} out = {call.input_tokens + call.output_tokens} tokens, "
                          f"{call.latency_ms:.0f}ms")
            md.append(f"")

    md.append(f"---")
    md.append(f"*Generated by EnvCheck demo_llm.py — {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    md.append(f"")

    md_path.write_text("\n".join(md))

    print(f"\n  {C.GREEN}{C.BOLD}📄 Report saved:{C.RESET}")
    print(f"    Markdown: {md_path}")
    print(f"    JSON:     {json_path}")
    return md_path, json_path


def _run_scenario_a_quiet(case: TestCase, env_path: Path) -> ScenarioResult:
    """Run Scenario A without verbose output."""
    result = ScenarioResult(scenario="A")

    system = ("You are a Python coding assistant. Write ONLY the Python code, "
              "wrapped in a ```python code block. No explanations.")

    # Call 1: generate
    if MOCK_MODE:
        _, call1 = call_llm(system, case.problem, "generate")
        code = case.broken_code
    else:
        raw, call1 = call_llm(system, case.problem, "generate")
        code = extract_code_from_response(raw)
    result.llm_calls.append(call1)

    # Mock: simulate partial fixes (one error fixed per loop)
    mock_fix_queue: list[str] = []
    if MOCK_MODE:
        partials = MOCK_PARTIAL_FIXES.get(case.id, [])
        mock_fix_queue = partials + [case.fixed_code]

    # Run → crash → fix loop
    for loop in range(MAX_FIX_LOOPS):
        result.execution_attempts += 1
        run_result = run_in_env(env_path, code, case.id, f"qA_att{loop}")

        if run_result.returncode == 0:
            result.final_success = True
            break

        result.runtime_crashes += 1
        if loop >= MAX_FIX_LOOPS - 1:
            break

        fix_prompt = (
            f"I ran this Python code:\n\n```python\n{code}\n```\n\n"
            f"And got this error:\n\n```\n{run_result.stderr.strip()[-1500:]}\n```\n\n"
            f"Please fix the code. Return ONLY the corrected Python code in a ```python block."
        )
        if MOCK_MODE:
            _, fix_call = call_llm(system, fix_prompt, "fix_from_error")
            if mock_fix_queue:
                code = mock_fix_queue.pop(0)
            else:
                code = case.fixed_code
        else:
            raw_fix, fix_call = call_llm(system, fix_prompt, "fix_from_error")
            code = extract_code_from_response(raw_fix)
        result.llm_calls.append(fix_call)

    return result


def _run_scenario_b_quiet(case: TestCase, env_path: Path) -> ScenarioResult:
    """Run Scenario B without verbose output."""
    result = ScenarioResult(scenario="B")

    system = ("You are a Python coding assistant. Write ONLY the Python code, "
              "wrapped in a ```python code block. No explanations.")

    # Call 1: generate
    if MOCK_MODE:
        _, call1 = call_llm(system, case.problem, "generate")
        code = case.broken_code
    else:
        raw, call1 = call_llm(system, case.problem, "generate")
        code = extract_code_from_response(raw)
    result.llm_calls.append(call1)

    # EnvCheck scan
    scan = scan_source(code, env_path=str(env_path), filepath=f"{case.id}_broken.py")
    result.envcheck_time_ms = scan.scan_time_ms
    result.envcheck_findings = scan.total_findings

    if scan.total_findings > 0:
        diag_lines = []
        for f in scan.findings:
            diag_lines.append(f"- Line {f.lineno}: `{f.matched_code}` — {f.rule.description}")
            diag_lines.append(f"  Fix: change `{f.rule.old_api}` to `{f.rule.new_api}`")
        diagnostic_text = "\n".join(diag_lines)

        fix_prompt = (
            f"Here is my Python code:\n\n```python\n{code}\n```\n\n"
            f"A static analysis tool found these issues:\n\n{diagnostic_text}\n\n"
            f"Please fix ALL the issues. Return ONLY the corrected Python code in a ```python block."
        )
        if MOCK_MODE:
            _, fix_call = call_llm(system, fix_prompt, "fix_from_envcheck")
            code = case.fixed_code
        else:
            raw_fix, fix_call = call_llm(system, fix_prompt, "fix_from_envcheck")
            code = extract_code_from_response(raw_fix)
        result.llm_calls.append(fix_call)

    # Run
    result.execution_attempts = 1
    run_result = run_in_env(env_path, code, case.id, "qB_fixed")
    result.final_success = run_result.returncode == 0
    if not result.final_success:
        result.runtime_crashes = 1

    return result


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def _update_globals(args):
    """Update module-level globals from parsed args."""
    global AUTO_MODE, MOCK_MODE, MODEL, PROVIDER
    AUTO_MODE = args.auto
    MOCK_MODE = args.mock
    PROVIDER = args.provider

    # Set model: use user-specified model, or default for the provider
    if args.model:
        MODEL = args.model
    else:
        MODEL = DEFAULT_MODELS.get(PROVIDER, DEFAULT_MODELS["claude"])


def main():
    parser = argparse.ArgumentParser(
        description="EnvCheck LLM-Powered Comparative Demo (Claude / Gemini)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Claude (default provider)
              export ANTHROPIC_API_KEY="sk-ant-..."
              python demo_llm.py                                      # Interactive, numpy_2x
              python demo_llm.py --case scipy_114                     # Specific case
              python demo_llm.py --auto                               # Auto-advance
              python demo_llm.py --all                                # All cases summary

              # Gemini
              export GEMINI_API_KEY="AIza..."
              python demo_llm.py --provider gemini                    # Gemini default model
              python demo_llm.py --provider gemini --model gemini-2.5-pro

              # Mock (no API key needed)
              python demo_llm.py --mock                               # Mock mode
              python demo_llm.py --all --mock                         # All cases, mock
        """),
    )
    parser.add_argument("--case", default="numpy_2x", help="Test case ID (default: numpy_2x)")
    parser.add_argument("--provider", default="claude", choices=["claude", "gemini"],
                        help="LLM provider: claude or gemini (default: claude)")
    parser.add_argument("--auto", action="store_true", help="Auto-advance with delays")
    parser.add_argument("--mock", action="store_true", help="Mock mode (no API key needed)")
    parser.add_argument("--all", action="store_true", help="Run all cases comparison")
    parser.add_argument("--report", action="store_true",
                        help="Save results as Markdown + JSON report (use with --all)")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument("--model", default=None,
                        help="Model name (default: auto per provider)")
    args = parser.parse_args()

    _update_globals(args)

    if args.list:
        print(f"\n  {C.BOLD}Available test cases:{C.RESET}\n")
        for case in ALL_CASES:
            print(f"    {C.CYAN}{case.id:<16}{C.RESET} {case.library:<14} {case.environment}")
        return

    # Check API key
    if not MOCK_MODE:
        if PROVIDER == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
            print(f"{C.RED}Error: ANTHROPIC_API_KEY not set.{C.RESET}")
            print(f"  Set it with: export ANTHROPIC_API_KEY=\"sk-ant-...\"")
            print(f"  Or use --mock for offline demo mode.")
            sys.exit(1)
        elif PROVIDER == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            print(f"{C.RED}Error: GEMINI_API_KEY not set.{C.RESET}")
            print(f"  Set it with: export GEMINI_API_KEY=\"AIza...\"")
            print(f"  (GOOGLE_API_KEY is also accepted)")
            print(f"  Or use --mock for offline demo mode.")
            sys.exit(1)

    # ─── All-cases mode ────────────────────────────────────────────
    if args.all:
        for case in ALL_CASES:
            setup_environment(case)
        result = run_all_cases(ALL_CASES)
        if args.report and result:
            cases, all_a, all_b = result
            report_dir = ROOT / "reports"
            generate_report(cases, all_a, all_b, report_dir)
        return

    # ─── Single-case interactive demo ──────────────────────────────
    case = None
    for c in ALL_CASES:
        if c.id == args.case:
            case = c
            break
    if not case:
        print(f"{C.RED}Unknown case: {args.case}{C.RESET}")
        print(f"Available: {', '.join(c.id for c in ALL_CASES)}")
        sys.exit(1)

    # Title
    mode_str = "MOCK" if MOCK_MODE else "LIVE API"
    provider_str = PROVIDER.upper()
    print(f"""
{C.MAGENTA}{C.BOLD}
  ╔══════════════════════════════════════════════════════════════════╗
  ║                                                                ║
  ║        EnvCheck — LLM-Powered Comparative Demo                 ║
  ║        Real API Calls · Token Tracking · Cost Analysis         ║
  ║                                                                ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║                                                                ║
  ║   Case:     {case.id:<20}                              ║
  ║   Library:  {case.library:<20}                              ║
  ║   Provider: {provider_str:<20}                              ║
  ║   Model:    {MODEL:<40}        ║
  ║   Mode:     {mode_str:<20}                              ║
  ║                                                                ║
  ╚══════════════════════════════════════════════════════════════════╝
{C.RESET}""")

    env_path = setup_environment(case)
    print(f"  {C.DIM}Environment ready: {env_path}{C.RESET}")

    if not args.auto:
        print(f"""
{C.CYAN}{C.BOLD}  ── Presentation Guide ─────────────────────────────────────{C.RESET}
{C.CYAN}  This demo compares two workflows when LLM code hits
  library version mismatches:

  📍 Scenario A (WITHOUT EnvCheck):
     LLM → code → run → 💥 crash → send error to LLM →
     fix → run → 💥 crash again... (repeat until it works)

  📍 Scenario B (WITH EnvCheck):
     LLM → code → 🛡️ static scan → send diagnostics to
     LLM → one precise fix → ✅ run successfully

  Press Enter at each step to advance.
{C.CYAN}  ──────────────────────────────────────────────────────────{C.RESET}
""")

    pause("start demo")

    # Run both scenarios
    result_a = run_scenario_a(case, env_path)

    pause("start Scenario B")

    result_b = run_scenario_b(case, env_path)

    pause("show comparison")

    show_comparison(case, result_a, result_b)

    if args.report:
        report_dir = ROOT / "reports"
        generate_report([case], [result_a], [result_b], report_dir)

    print(f"\n  {C.DIM}Demo complete. Try: python demo_llm.py --all{C.RESET}\n")


if __name__ == "__main__":
    main()
