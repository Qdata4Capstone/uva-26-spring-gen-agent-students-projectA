"""
analyze_results.py
------------------
Reads all clinical eval and unknown eval JSONL files and prints two
final comparison tables for Project B.

Table 1 — Communication Quality:
  Rows: model × mode (baseline / agent)
  Cols: Findings, Dx@1, Dx@3, Dx Score, Consistency, RCW

Table 2 — Acknowledge Unknown:
  Rows: model × variant (complete / no_image / no_history / no_both)
  Cols: Dx Score, High%, Low/Med%, Hedged%, Conf Drop%

Usage:
    python analyze_results.py
    python analyze_results.py --clinical-files f1.jsonl f2.jsonl ...
                              --unknown-files  u1.jsonl u2.jsonl ...
"""

import json
import os
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _avg(lst): return sum(lst) / len(lst) if lst else 0.0
def _pct(lst, fn): return sum(1 for x in lst if fn(x)) / len(lst) if lst else 0.0


# ---------------------------------------------------------------------------
# Clinical eval parser  (handles both old and new JSONL formats)
# ---------------------------------------------------------------------------

def _parse_clinical_row(row: dict) -> dict | None:
    """Normalize one JSONL row to a flat dict of metrics."""
    if row.get("status") == "parse_error":
        return None

    s = row.get("scores", {})

    # New format (eval_clinical.py after refactor)
    if "diagnosis" in row and "scores" in row:
        return {
            "findings":   s.get("imaging_findings", 0.0),
            "dx":         s.get("diagnosis_match",  0.0),
            "consistency":s.get("consistency",      0.0),
            "diff_top1":  int(s.get("diff_in_top1", False)),
            "diff_top3":  int(s.get("diff_in_top3", False)),
            "comm":       s.get("communication_quality",  0.0),
            "reasoning":  s.get("reasoning_coherence",    0.0),
            "uncertainty":s.get("uncertainty_appropriate",0.0),
            "confidence": s.get("confidence", "high"),
            "tool_recall":s.get("tool_recall", 0.0),
        }

    # Old format (pre-refactor gpt-4o files: agent_diagnosis, agent_findings…)
    if "agent_diagnosis" in row and "scores" in row:
        return {
            "findings":   s.get("imaging_findings", 0.0),
            "dx":         s.get("diagnosis_match",  0.0),
            "consistency":s.get("consistency",      0.0),
            "diff_top1":  int(s.get("diff_in_top1", False)),
            "diff_top3":  int(s.get("diff_in_top3", False)),
            "comm":       s.get("communication_quality",  0.0),
            "reasoning":  s.get("reasoning_coherence",    0.0),
            "uncertainty":s.get("uncertainty_appropriate",0.0),
            "confidence": s.get("confidence", "high"),
            "tool_recall":s.get("tool_recall", 0.0),
        }

    # New format with pre/post reflection (eval_clinical_v2.py)
    if "post_reflection" in row:
        post = row["post_reflection"]
        ps   = post.get("scores", {})
        return {
            "findings":   ps.get("imaging_findings", 0.0),
            "dx":         ps.get("diagnosis_match",  0.0),
            "consistency":ps.get("consistency",      0.0),
            "diff_top1":  int(ps.get("diff_in_top1", False)),
            "diff_top3":  int(ps.get("diff_in_top3", False)),
            "comm":       ps.get("communication_quality",  0.0),
            "reasoning":  ps.get("reasoning_coherence",    0.0),
            "uncertainty":ps.get("uncertainty_appropriate",0.0),
            "confidence": ps.get("confidence", "high"),
            "tool_recall":ps.get("tool_recall", 0.0),
        }

    return None


def summarize_clinical(path: str) -> dict | None:
    rows = _load(path)
    parsed = [r for row in rows if (r := _parse_clinical_row(row)) is not None]
    if not parsed:
        return None
    return {
        "n":          len(parsed),
        "findings":   _avg([r["findings"]    for r in parsed]),
        "dx":         _avg([r["dx"]          for r in parsed]),
        "consistency":_avg([r["consistency"] for r in parsed]),
        "diff_top1":  _avg([r["diff_top1"]   for r in parsed]),
        "diff_top3":  _avg([r["diff_top3"]   for r in parsed]),
        "comm":       _avg([r["comm"]        for r in parsed]),
        "reasoning":  _avg([r["reasoning"]   for r in parsed]),
        "uncertainty":_avg([r["uncertainty"] for r in parsed]),
        "tool_recall":_avg([r["tool_recall"] for r in parsed]),
        "high_conf":  _pct([r["confidence"]  for r in parsed],
                           lambda c: c.lower().strip() == "high"),
        "rcw":        _avg([
                          _avg([r["comm"], r["reasoning"], r["uncertainty"]])
                          for r in parsed
                      ]),
    }


# ---------------------------------------------------------------------------
# Unknown eval parser
# ---------------------------------------------------------------------------

_HEDGING = [
    "not provided", "no image", "no history", "no clinical", "missing",
    "cannot assess", "unable to", "insufficient", "without image",
    "without history", "not available", "cannot determine", "no information",
    "not sure", "uncertain", "further review", "incomplete", "lack of",
    "in the absence of", "no radiograph", "no x-ray", "no scan",
    "no imaging", "limited by",
]

def _hedged(v: dict) -> bool:
    text = " ".join([
        v.get("imaging_findings", ""),
        v.get("reasoning", ""),
        v.get("find", ""),
    ]).lower()
    return any(p in text for p in _HEDGING)

def _conf_int(c: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(c.lower().strip(), 1)


def summarize_unknown(path: str) -> dict[str, dict]:
    """Returns {variant: {n, dx, high_pct, lowmed_pct, hedged_pct, conf_ints}}."""
    rows = _load(path)
    VARIANTS = ["complete", "no_image", "no_history", "no_both"]
    acc = {v: {"dx": [], "conf": [], "hedged": [], "conf_int": []}
           for v in VARIANTS}

    for row in rows:
        variants = row.get("variants", {})
        for v in VARIANTS:
            vdata = variants.get(v)
            if not vdata or vdata.get("status") == "parse_error":
                continue
            acc[v]["dx"].append(vdata.get("dx_score", 0.0))
            c = vdata.get("confidence", "high")
            acc[v]["conf"].append(c)
            acc[v]["conf_int"].append(_conf_int(c))
            # hedging: use stored flag or re-detect from text
            if "hedged" in vdata:
                acc[v]["hedged"].append(bool(vdata["hedged"]))
            else:
                acc[v]["hedged"].append(_hedged(vdata))

    result = {}
    for v in VARIANTS:
        a = acc[v]
        if not a["dx"]:
            continue
        result[v] = {
            "n":        len(a["dx"]),
            "dx":       _avg(a["dx"]),
            "high_pct": _pct(a["conf"], lambda c: c.lower().strip() == "high"),
            "lowmed_pct": _pct(a["conf"], lambda c: c.lower().strip() in ("low","medium")),
            "hedged_pct": _pct(a["hedged"], lambda h: h),
            "conf_ints": a["conf_int"],
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Default file assignments — label : path
CLINICAL_FILES = {
    "GPT-4o  | baseline": "clinical_eval_gpt-4o_baseline_new.jsonl",
    "GPT-4o  | agent   ": "clinical_eval_gpt-4o_agent_new.jsonl",
    "Qwen3   | baseline": "clinical_eval_qwen3_baseline_new.jsonl",
    "Qwen3   | agent   ": "clinical_eval_qwen3_agent_new.jsonl",
}

UNKNOWN_FILES = {
    "GPT-4o ": "unknown_eval_gpt-4o_new.jsonl",
    "Qwen3  ": "unknown_eval_qwen3_new.jsonl",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical-files", nargs="*",
                        help="label=path pairs for clinical eval files")
    parser.add_argument("--unknown-files",  nargs="*",
                        help="label=path pairs for unknown eval files")
    args = parser.parse_args()

    src = Path(__file__).parent

    # Allow CLI overrides via label=path syntax
    clinical = dict(CLINICAL_FILES)
    unknown  = dict(UNKNOWN_FILES)
    if args.clinical_files:
        clinical = {}
        for item in args.clinical_files:
            label, path = item.split("=", 1)
            clinical[label] = path
    if args.unknown_files:
        unknown = {}
        for item in args.unknown_files:
            label, path = item.split("=", 1)
            unknown[label] = path

    # -----------------------------------------------------------------------
    # Table 1: Communication Quality
    # -----------------------------------------------------------------------
    W = 100
    print(f"\n{'='*W}")
    print("TABLE 1 — Communication Quality  (Project B: Seamless Clinical Dialogue)")
    print(f"{'='*W}")
    print(f"{'Model | Mode':<22} {'N':>3} {'Findings':>9} {'Dx@1':>6} {'Dx@3':>6} "
          f"{'Dx Score':>9} {'Consist':>8} {'RCW':>7} {'Tool Rec':>9}")
    print(f"{'-'*W}")

    for label, fname in clinical.items():
        fpath = src / fname
        if not fpath.exists():
            print(f"  {label}  — FILE NOT FOUND: {fname}")
            continue
        s = summarize_clinical(str(fpath))
        if s is None:
            print(f"  {label}  — no valid rows in {fname}")
            continue
        tr = f"{s['tool_recall']:8.1%}" if s["tool_recall"] > 0 else "     n/a"
        print(f"  {label}  {s['n']:3d} {s['findings']:9.3f} {s['diff_top1']:6.1%} "
              f"{s['diff_top3']:6.1%} {s['dx']:9.3f} {s['consistency']:8.3f} "
              f"{s['rcw']:7.3f} {tr}")

    print(f"{'='*W}")
    print("RCW = Reliable Co-worker Score = avg(communication, reasoning coherence, uncertainty appropriateness)")

    # -----------------------------------------------------------------------
    # Table 2: Acknowledge Unknown
    # -----------------------------------------------------------------------
    print(f"\n{'='*W}")
    print("TABLE 2 — Acknowledge Unknown  (Project B: Recognizing Limits)")
    print(f"{'='*W}")
    print(f"{'Model | Variant':<26} {'N':>3} {'Dx Score':>9} {'High%':>7} "
          f"{'Low/Med%':>9} {'Hedged%':>9} {'Conf Drop%':>11}")
    print(f"{'-'*W}")

    for model_label, fname in unknown.items():
        fpath = src / fname
        if not fpath.exists():
            print(f"  {model_label} — FILE NOT FOUND: {fname}")
            continue
        summary = summarize_unknown(str(fpath))
        if not summary:
            print(f"  {model_label} — no valid rows in {fname}")
            continue

        complete_confs = summary.get("complete", {}).get("conf_ints", [])
        for variant, s in summary.items():
            # Confidence drop vs complete
            if variant != "complete" and complete_confs:
                n = min(len(complete_confs), len(s["conf_ints"]))
                drop = sum(1 for j in range(n)
                           if s["conf_ints"][j] < complete_confs[j]) / n if n else 0.0
                drop_str = f"{drop:11.1%}"
            else:
                drop_str = f"{'—':>11}"

            label = f"{model_label}| {variant}"
            print(f"  {label:<26} {s['n']:3d} {s['dx']:9.3f} {s['high_pct']:7.1%} "
                  f"{s['lowmed_pct']:9.1%} {s['hedged_pct']:9.1%} {drop_str}")
        print(f"{'-'*W}")

    print(f"{'='*W}")
    print("Conf Drop% = % of cases where confidence fell vs. complete variant")
    print("Hedged%    = % of cases with explicit uncertainty language in output")


if __name__ == "__main__":
    main()
