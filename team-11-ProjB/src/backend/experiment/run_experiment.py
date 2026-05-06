#!/usr/bin/env python3
"""
Step 2 — FinSynth Mode Comparison Experiment Runner.

Runs brief / normal / extra on each of the 15 frozen tickers for N trials
and writes every result to a JSONL file.  Latency, fact-check metrics, and
resynth behaviour are captured per run so analyze.py can compute comparisons.

Design notes:
  • Uses mock_mcp_server.py instead of the real MCP server so all three modes
    see the *same* yfinance snapshot for a given ticker (no data drift).
  • Runs are sequential — parallelism would corrupt latency measurements on a
    local Ollama instance.
  • Trials are interleaved per ticker (brief→normal→extra per ticker, repeat)
    to distribute any Ollama thermal/load variance evenly across modes.

Usage:
    cd backend
    python -m experiment.run_experiment                 # 3 trials, all modes
    python -m experiment.run_experiment --trials 1      # quick smoke test
    python -m experiment.run_experiment --modes brief,normal  # subset of modes
    python -m experiment.run_experiment --dry-run       # print plan only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_ollama import ChatOllama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import get_settings
from app.graph.workflow import build_graph
from app.graph.state import AgentState
from experiment.tickers import TICKERS, TICKER_META

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

EXPERIMENT_DIR = Path(__file__).parent
FROZEN_DIR = EXPERIMENT_DIR / "frozen_data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
MOCK_SERVER = str(EXPERIMENT_DIR / "mock_mcp_server.py")

MODES = ["brief", "normal", "extra", "extra_force"]
_EXPECTED_LLM_CALLS = {"brief": 2, "normal": 3, "extra": "3–4", "extra_force": 4}


# ── Single run ────────────────────────────────────────────────────────

async def run_single(
    ticker: str,
    tier: str,
    mode: str,
    trial: int,
    llm: ChatOllama,
) -> dict:
    """
    Execute one (ticker, mode, trial) using the frozen mock MCP server.
    Returns a structured result dict suitable for JSONL serialisation.
    """
    result: dict = {
        "ticker": ticker,
        "tier": tier,
        "mode": mode,
        "trial": trial,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Timing
        "total_latency_s": None,
        "node_latency_s": {},
        # LLM call budget
        "llm_calls_expected": _EXPECTED_LLM_CALLS[mode],
        # Fact-check (draft, pre-resynth)
        "fact_check": None,
        # Post fact-check (extra mode + resynth triggered)
        "post_fact_check": None,
        # Resynth behaviour
        "resynth_triggered": False,
        "density_improvement": None,
        # Output size
        "report_word_count": None,
        "draft_word_count": None,
        # Error
        "error": None,
    }

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MOCK_SERVER],
        env={**os.environ, "FINSYNTH_FROZEN_DIR": str(FROZEN_DIR)},
    )

    t_start = time.monotonic()
    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                graph = build_graph(session, llm)

                initial_state: AgentState = {
                    "ticker": ticker,
                    "workflow_mode": mode,
                    "financial_data": None,
                    "auditor_analysis": None,
                    "news_data": None,
                    "news_analysis": None,
                    "draft_report": None,
                    "report": None,
                    "fact_check_result": None,
                    "citation_density": None,
                    "resynthesis_needed": False,
                    "post_fact_check_result": None,
                    "post_citation_density": None,
                    "thinking_log": [],
                }

                # ── Stream graph, record per-node wall time ────────────
                node_latency: dict[str, float] = {}
                final: dict = {}
                prev_t = time.monotonic()

                async for chunk in graph.astream(initial_state, stream_mode="updates"):
                    t_now = time.monotonic()
                    for node_name, state_update in chunk.items():
                        node_latency[node_name] = round(t_now - prev_t, 3)
                        # Merge state (thinking_log is additive but we don't need it)
                        for k, v in state_update.items():
                            if k != "thinking_log":
                                final[k] = v
                    prev_t = t_now

                result["total_latency_s"] = round(time.monotonic() - t_start, 3)
                result["node_latency_s"] = node_latency

                # ── Extract fact-check metrics ─────────────────────────
                fc = final.get("fact_check_result") or {}
                if fc and "error" not in fc:
                    result["fact_check"] = {
                        "citation_density": fc.get("citation_density"),
                        "tier_label": fc.get("tier"),
                        "total_claims": fc.get("total_claims", 0),
                        "verified_count": len(fc.get("verified_claims", [])),
                        "unverified_count": len(fc.get("unverified_claims", [])),
                        "lookup_populated": fc.get("lookup_populated", False),
                    }

                # ── Post fact-check metrics (extra + resynth path) ─────
                pfc = final.get("post_fact_check_result") or {}
                if pfc:
                    result["post_fact_check"] = {
                        "citation_density": pfc.get("citation_density"),
                        "tier_label": pfc.get("tier"),
                        "total_claims": pfc.get("total_claims", 0),
                        "verified_count": len(pfc.get("verified_claims", [])),
                        "unverified_count": len(pfc.get("unverified_claims", [])),
                    }
                    orig = fc.get("citation_density", 0) or 0
                    post = pfc.get("citation_density", 0) or 0
                    result["density_improvement"] = round(post - orig, 4)

                result["resynth_triggered"] = bool(final.get("resynthesis_needed"))

                # ── Word counts ────────────────────────────────────────
                report = final.get("report") or ""
                draft = final.get("draft_report") or ""
                if report:
                    result["report_word_count"] = len(report.split())
                if draft:
                    result["draft_word_count"] = len(draft.split())

    except Exception as exc:
        result["total_latency_s"] = round(time.monotonic() - t_start, 3)
        result["error"] = str(exc)

    return result


# ── Experiment harness ────────────────────────────────────────────────

def _load_completed(path: Path) -> set[tuple[str, str, int]]:
    """Return the set of (ticker, mode, trial) tuples already recorded in path."""
    completed: set[tuple[str, str, int]] = set()
    if not path.exists():
        return completed
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                completed.add((r["ticker"], r["mode"], r["trial"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


async def main(trials: int, modes: list[str], output: Path | None, dry_run: bool) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # Validate frozen data
    metadata_path = FROZEN_DIR / "metadata.json"
    if not metadata_path.exists():
        print("ERROR: No frozen data found.  Run freeze_data.py first:\n"
              "  cd backend && python -m experiment.freeze_data")
        sys.exit(1)

    metadata = json.loads(metadata_path.read_text())
    missing = [t["ticker"] for t in TICKERS
               if not (FROZEN_DIR / f"{t['ticker']}_financials.json").exists()]
    if missing:
        print(f"WARNING: Missing frozen data for: {missing}")

    print(f"Frozen data snapshot: {metadata['frozen_at']}")

    # Load LLM once and reuse across all runs
    settings = get_settings()
    print(f"LLM: {settings.ollama_model} @ {settings.ollama_base_url}\n")

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
        client_kwargs={"timeout": settings.ollama_timeout_sec},
    )

    # Resolve output file — fixed path if supplied, otherwise timestamped
    if output is not None:
        results_path = output
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        results_path = RESULTS_DIR / f"experiment_{timestamp}.jsonl"

    # Load already-completed runs so we can skip them on resume
    completed = _load_completed(results_path)
    if completed:
        print(f"Resuming — {len(completed)} run(s) already in {results_path.name}, will skip them.")

    # Build run list: interleave modes within each ticker to spread Ollama
    # thermal/load variance evenly — all modes for ticker 1, then ticker 2…
    runs: list[tuple[str, str, str, int]] = []
    for trial in range(1, trials + 1):
        for t_info in TICKERS:
            for mode in modes:
                runs.append((t_info["ticker"], t_info["tier"], mode, trial))

    pending = [(ticker, tier, mode, trial) for ticker, tier, mode, trial in runs
               if (ticker, mode, trial) not in completed]
    total = len(runs)
    skip_count = total - len(pending)

    est_minutes = len(pending) * 2
    print(f"Plan: {len(TICKERS)} tickers × {len(modes)} modes × {trials} trial(s) = {total} runs")
    if skip_count:
        print(f"Skipping {skip_count} already-completed run(s) — {len(pending)} remaining")
    print(f"Modes: {modes}")
    print(f"Estimated runtime: ~{est_minutes}–{est_minutes*3} min (depends on LLM speed)")
    print(f"Results → {results_path}\n")

    if dry_run:
        print("DRY RUN — pending run plan:")
        for i, (ticker, tier, mode, trial) in enumerate(pending, 1):
            print(f"  [{i:3d}/{len(pending)}] {ticker:6s} ({tier:10s}) | {mode:6s} | trial {trial}")
        if skip_count:
            print(f"\n  ({skip_count} already-completed run(s) would be skipped)")
        return

    if not pending:
        print("Nothing to do — all runs for these modes/trials are already complete.")
        print(f"Analyze with:\n  cd backend && python -m experiment.analyze {results_path}")
        return

    errors: list[str] = []

    with open(results_path, "a", encoding="utf-8") as out:
        for i, (ticker, tier, mode, trial) in enumerate(pending, 1):
            label = f"[{i:3d}/{len(pending)}] {ticker:6s} ({tier:10s}) | {mode:6s} | trial {trial}"
            print(f"{label} ...", end="", flush=True)

            result = await run_single(ticker, tier, mode, trial, llm)

            if result["error"]:
                status = f"ERROR: {result['error'][:60]}"
                errors.append(f"{ticker}/{mode}/t{trial}: {result['error']}")
            else:
                latency = f"{result['total_latency_s']:.1f}s"
                fc = result.get("fact_check") or {}
                density = fc.get("citation_density")
                density_str = f"density={density:.0%}" if density is not None else "density=N/A"
                resynth_str = " [RESYNTH]" if result["resynth_triggered"] else ""
                status = f"{latency} {density_str}{resynth_str}"

            print(f" {status}")

            out.write(json.dumps(result) + "\n")
            out.flush()

    print(f"\n{'─'*60}")
    print(f"Section complete — {len(pending) - len(errors)}/{len(pending)} runs succeeded")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    print(f"\nResults saved to:\n  {results_path}")
    print(f"\nAnalyze at any time:\n  cd backend && python -m experiment.analyze {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSynth Mode Comparison Experiment")
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of trials per (ticker, mode) pair (default: 1)",
    )
    parser.add_argument(
        "--modes", default="brief,normal,extra",
        help="Comma-separated list of modes to test (default: brief,normal,extra)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to the JSONL results file.  If the file already exists, completed "
             "runs are skipped and new results are appended.  Use the same path across "
             "all sections (e.g. --modes brief then --modes normal) to accumulate "
             "results in one file.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the run plan without executing anything",
    )
    args = parser.parse_args()

    valid_modes = [m for m in args.modes.split(",") if m in MODES]
    invalid = [m for m in args.modes.split(",") if m not in MODES]
    if invalid:
        print(f"Unknown modes ignored: {invalid}")
    if not valid_modes:
        print(f"No valid modes specified.  Choose from: {MODES}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else None

    asyncio.run(main(trials=args.trials, modes=valid_modes, output=output_path, dry_run=args.dry_run))
