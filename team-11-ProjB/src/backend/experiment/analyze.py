#!/usr/bin/env python3
"""
Step 3 — Analyze experiment results and print a structured summary report.

Reads the JSONL produced by run_experiment.py and emits:
  1. Per-mode summary (latency, density, word count, resynth rate)
  2. Citation density by mode × data-richness tier
  3. Latency by mode × tier
  4. Per-node latency breakdown
  5. Re-synthesis trigger rate by tier
  6. Density improvement from resynth (extra mode, triggered runs only)
  7. Data-richness validation (lookup_populated rate)

Also writes a flat CSV to the same directory as the input JSONL so results
can be opened in Excel or loaded into pandas for further analysis.

Usage:
    cd backend
    python -m experiment.analyze experiment/results/experiment_<timestamp>.jsonl
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────

def load_results(*paths: str) -> list[dict]:
    """Load and merge results from one or more JSONL files."""
    results = []
    seen: set[tuple[str, str, int]] = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r["ticker"], r["mode"], r["trial"])
                if key not in seen:
                    seen.add(key)
                    results.append(r)
    return results


def _fmt(values: list[float], decimals: int = 3) -> str:
    if not values:
        return "—"
    mu = statistics.mean(values)
    if len(values) == 1:
        return f"{mu:.{decimals}f}"
    sd = statistics.stdev(values)
    return f"{mu:.{decimals}f} ± {sd:.{decimals}f}"


def _pct_fmt(values: list[float]) -> str:
    if not values:
        return "—"
    mu = statistics.mean(values)
    if len(values) == 1:
        return f"{mu:.1%}"
    sd = statistics.stdev(values)
    return f"{mu:.1%} ± {sd:.1%}"


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "—"
    return f"{numerator}/{denominator} ({numerator/denominator:.1%})"


# ── CSV export ────────────────────────────────────────────────────────

def export_csv(results: list[dict], jsonl_path: str) -> Path:
    csv_path = Path(jsonl_path).with_suffix(".csv")
    fieldnames = [
        "ticker", "tier", "mode", "trial", "timestamp",
        "total_latency_s", "error",
        # fact check
        "fc_citation_density", "fc_tier", "fc_total_claims",
        "fc_verified_count", "fc_unverified_count", "fc_lookup_populated",
        # post fact check
        "pfc_citation_density", "pfc_tier", "pfc_total_claims",
        "pfc_verified_count", "pfc_unverified_count",
        # resynth
        "resynth_triggered", "density_improvement",
        # output size
        "report_word_count", "draft_word_count",
        # per-node latency
        "lat_auditor", "lat_news_hound", "lat_synthesizer",
        "lat_fact_checker", "lat_resynth", "lat_post_fact_checker",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            fc = r.get("fact_check") or {}
            pfc = r.get("post_fact_check") or {}
            nl = r.get("node_latency_s") or {}
            writer.writerow({
                "ticker": r["ticker"],
                "tier": r["tier"],
                "mode": r["mode"],
                "trial": r["trial"],
                "timestamp": r["timestamp"],
                "total_latency_s": r.get("total_latency_s"),
                "error": r.get("error"),
                "fc_citation_density": fc.get("citation_density"),
                "fc_tier": fc.get("tier_label"),
                "fc_total_claims": fc.get("total_claims"),
                "fc_verified_count": fc.get("verified_count"),
                "fc_unverified_count": fc.get("unverified_count"),
                "fc_lookup_populated": fc.get("lookup_populated"),
                "pfc_citation_density": pfc.get("citation_density"),
                "pfc_tier": pfc.get("tier_label"),
                "pfc_total_claims": pfc.get("total_claims"),
                "pfc_verified_count": pfc.get("verified_count"),
                "pfc_unverified_count": pfc.get("unverified_count"),
                "resynth_triggered": r.get("resynth_triggered"),
                "density_improvement": r.get("density_improvement"),
                "report_word_count": r.get("report_word_count"),
                "draft_word_count": r.get("draft_word_count"),
                "lat_auditor": nl.get("auditor"),
                "lat_news_hound": nl.get("news_hound"),
                "lat_synthesizer": nl.get("synthesizer"),
                "lat_fact_checker": nl.get("fact_checker"),
                "lat_resynth": nl.get("resynth"),
                "lat_post_fact_checker": nl.get("post_fact_checker"),
            })
    return csv_path


# ── Main report ───────────────────────────────────────────────────────

TIERS = ["large_cap", "mid_cap", "small_cap"]
TIER_LABELS = {"large_cap": "Large-cap", "mid_cap": "Mid-cap", "small_cap": "Small-cap"}
MODES = ["brief", "normal", "extra", "extra_force"]


def print_section(title: str) -> None:
    print(f"\n{'─'*70}")
    print(title)
    print(f"{'─'*70}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m experiment.analyze <results.jsonl> [results2.jsonl ...]")
        sys.exit(1)

    jsonl_path = sys.argv[1]  # used for CSV output path
    all_results = load_results(*sys.argv[1:])
    ok = [r for r in all_results if not r.get("error")]
    failed = [r for r in all_results if r.get("error")]

    W = 70
    print("=" * W)
    print("FINSYNTH MODE COMPARISON — EXPERIMENT SUMMARY")
    print("=" * W)
    print(f"JSONL: {jsonl_path}")
    print(f"Total runs: {len(all_results)} | Successful: {len(ok)} | Failed: {len(failed)}")

    if failed:
        print(f"\nFailed runs:")
        for r in failed:
            print(f"  {r['ticker']:6s} | {r['mode']:6s} | trial {r['trial']}: {r.get('error','?')}")

    # ── 1. Per-mode summary ───────────────────────────────────────────
    print_section("1. PER-MODE SUMMARY (all tickers combined)")
    hdr = f"{'Mode':<8} {'N':>4}  {'Total(s)':>16}  {'Density':>14}  {'Verified':>10}  {'Unverified':>12}  {'Words':>8}  {'Resynth':>8}"
    print(hdr)
    print("-" * len(hdr))

    for mode in MODES:
        runs = [r for r in ok if r["mode"] == mode]
        latencies = [r["total_latency_s"] for r in runs if r.get("total_latency_s")]
        fcs = [r["fact_check"] for r in runs if r.get("fact_check")]
        densities = [fc["citation_density"] for fc in fcs if fc.get("citation_density") is not None]
        verified = [fc["verified_count"] for fc in fcs]
        unverified = [fc["unverified_count"] for fc in fcs]
        words = [r["report_word_count"] for r in runs if r.get("report_word_count")]
        resynth_n = sum(1 for r in runs if r.get("resynth_triggered"))
        resynth_rate = f"{resynth_n}/{len(runs)}" if runs else "—"
        print(
            f"{mode:<8} {len(runs):>4}  {_fmt(latencies, 1):>16}  "
            f"{_pct_fmt(densities):>14}  {_fmt(verified, 1):>10}  "
            f"{_fmt(unverified, 1):>12}  {_fmt(words, 0):>8}  {resynth_rate:>8}"
        )

    # ── 2. Citation density by mode × tier ───────────────────────────
    print_section("2. CITATION DENSITY BY MODE × DATA-RICHNESS TIER (mean ± std)")
    mode_hdrs = {"brief": "Brief", "normal": "Normal", "extra": "Extra", "extra_force": "ExtraForce"}
    hdr2 = f"{'Tier':<12}" + "".join(f"  {mode_hdrs[m]:>16}" for m in MODES)
    print(hdr2)
    print("-" * len(hdr2))

    for tier in TIERS:
        row = [TIER_LABELS[tier]]
        for mode in MODES:
            runs = [r for r in ok if r["mode"] == mode and r["tier"] == tier and r.get("fact_check")]
            vals = [r["fact_check"]["citation_density"] for r in runs
                    if r["fact_check"].get("citation_density") is not None]
            row.append(_pct_fmt(vals))
        print(f"{row[0]:<12}" + "".join(f"  {v:>16}" for v in row[1:]))

    print("\n  Note: density shown is the *draft* density (pre-resynth) for extra/extra_force,")
    print("  so all four modes are compared on the same draft baseline.")

    # ── 3. Total latency by mode × tier ──────────────────────────────
    print_section("3. TOTAL LATENCY BY MODE × TIER (seconds, mean ± std)")
    hdr3 = f"{'Tier':<12}" + "".join(f"  {mode_hdrs[m]:>16}" for m in MODES)
    print(hdr3)
    print("-" * len(hdr3))

    for tier in TIERS:
        row = [TIER_LABELS[tier]]
        for mode in MODES:
            runs = [r for r in ok if r["mode"] == mode and r["tier"] == tier]
            vals = [r["total_latency_s"] for r in runs if r.get("total_latency_s")]
            row.append(_fmt(vals, 1))
        print(f"{row[0]:<12}" + "".join(f"  {v:>16}" for v in row[1:]))

    # ── 4. Per-node latency breakdown ─────────────────────────────────
    print_section("4. MEAN NODE LATENCY BY MODE (seconds)")
    nodes = ["auditor", "news_hound", "synthesizer", "fact_checker", "resynth", "post_fact_checker"]
    hdr4 = f"{'Node':<22}" + "".join(f"  {mode_hdrs[m]:>12}" for m in MODES)
    print(hdr4)
    print("-" * len(hdr4))

    for node in nodes:
        row = [node]
        for mode in MODES:
            runs = [r for r in ok if r["mode"] == mode and r.get("node_latency_s")]
            vals = [r["node_latency_s"][node] for r in runs if node in r["node_latency_s"]]
            row.append(_fmt(vals, 1) if vals else "—")
        print(f"{row[0]:<22}" + "".join(f"  {v:>12}" for v in row[1:]))

    # ── 5. Resynth trigger rate (extra mode) ─────────────────────────
    print_section("5. RE-SYNTHESIS TRIGGER RATE (extra and extra_force modes)")
    for resynth_mode, label in [("extra", "extra (density-gated)"), ("extra_force", "extra_force (always)")]:
        mode_runs = [r for r in ok if r["mode"] == resynth_mode]
        if not mode_runs:
            continue
        triggered_total = sum(1 for r in mode_runs if r.get("resynth_triggered"))
        print(f"  {label}:")
        print(f"    Overall: {_rate(triggered_total, len(mode_runs))}")
        for tier in TIERS:
            tier_runs = [r for r in mode_runs if r["tier"] == tier]
            triggered = sum(1 for r in tier_runs if r.get("resynth_triggered"))
            print(f"    {TIER_LABELS[tier]:<12}: {_rate(triggered, len(tier_runs))}")

    print()
    print("  extra_force always triggers resynth (when lookup_populated=True).")
    print("  Comparing extra vs extra_force density improvement isolates the value")
    print("  of resynth on already-high-quality drafts.")

    # ── 6. Density improvement from resynth ──────────────────────────
    print_section("6. DENSITY IMPROVEMENT FROM RE-SYNTHESIS (triggered runs only)")
    resynth_runs = [r for r in ok if r["mode"] in ("extra", "extra_force")
                    and r.get("resynth_triggered") and r.get("density_improvement") is not None]

    if resynth_runs:
        orig = [r["fact_check"]["citation_density"] for r in resynth_runs if r.get("fact_check")]
        post = [r["post_fact_check"]["citation_density"] for r in resynth_runs if r.get("post_fact_check")]
        delta = [r["density_improvement"] for r in resynth_runs]
        print(f"  Triggered runs:        {len(resynth_runs)}")
        print(f"  Draft density:         {_pct_fmt(orig)}")
        print(f"  Post-resynth density:  {_pct_fmt(post)}")
        print(f"  Mean improvement:      {_pct_fmt(delta)}")

        print()
        print("  By tier:")
        for tier in TIERS:
            t_runs = [r for r in resynth_runs if r["tier"] == tier]
            if t_runs:
                t_delta = [r["density_improvement"] for r in t_runs]
                print(f"    {TIER_LABELS[tier]:<12}: {_pct_fmt(t_delta)} over {len(t_runs)} triggered run(s)")
    else:
        print("  No re-synthesis was triggered — try extra mode on small-cap tickers.")

    # ── 7. Data-richness validation ───────────────────────────────────
    print_section("7. DATA-RICHNESS VALIDATION (lookup_populated rate by tier)")
    print("  lookup_populated=False means the yfinance snapshot had too few numeric")
    print("  fields for the fact-checker to verify claims — results for those runs")
    print("  should be treated with caution.\n")

    for tier in TIERS:
        # Use brief runs (one mode reference; all modes see the same frozen data)
        brief_runs = [r for r in ok if r["tier"] == tier and r["mode"] == "brief" and r.get("fact_check")]
        if brief_runs:
            populated = sum(1 for r in brief_runs if r["fact_check"].get("lookup_populated"))
            print(f"  {TIER_LABELS[tier]:<12}: {_rate(populated, len(brief_runs))}")

    # ── Confidence caveat ─────────────────────────────────────────────
    print_section("INTERPRETATION NOTES")
    n_per_cell = len(ok) // (len(MODES) * max(len(TIERS), 1)) if ok else 0
    print(f"  Runs per (mode × tier) cell: ~{n_per_cell}")
    print()
    print("  • Density reported is algorithmic (verified numeric claims / total claims).")
    print("    It does NOT measure qualitative accuracy or analytical depth.")
    print("  • 'extra' density above = draft density, not post-resynth, for fair comparison.")
    print("  • Latency is wall-clock on local Ollama.  Hardware load and thermal state")
    print("    introduce noise — trust the mean more than any single run.")
    print("  • For cells with lookup_populated=False the density metric is unreliable;")
    print("    exclude those tickers when comparing fact-check accuracy across modes.")
    print()

    # ── CSV export ────────────────────────────────────────────────────
    csv_path = export_csv(all_results, jsonl_path)
    print(f"  CSV exported to: {csv_path}")
    print("=" * W)


if __name__ == "__main__":
    main()
