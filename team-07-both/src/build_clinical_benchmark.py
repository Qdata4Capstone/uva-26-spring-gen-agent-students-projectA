"""
build_clinical_benchmark.py
----------------------------
Converts eurorad_dataset/extracted_preview.json into a clinical benchmark
JSONL ready for eval_clinical.py.

Each output case contains:
  - case_id, title, patient_age, patient_gender
  - clinical_history           ← given to agent as input
  - images                     ← local paths (relative to this script's dir)
  - image_descriptions         ← figure captions / descriptions
  - gt_imaging_findings        ← ground truth for Step 2 scoring
  - gt_differentials           ← cleaned list for Step 3 scoring
  - gt_final_diagnosis         ← ground truth for Step 4 scoring

Usage:
    python build_clinical_benchmark.py \
        --input  eurorad_dataset/extracted_preview.json \
        --output eurorad_clinical_benchmark.jsonl \
        --min-images 1          # skip cases with no local images
"""

import json
import os
import re
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def clean_final_diagnosis(case: dict) -> str:
    fd = case.get("final_diagnosis", "").strip()
    # Field is sometimes polluted with the full "Differential Diagnosis List:..." text
    if fd and not fd.lower().startswith("differential diagnosis"):
        return fd

    # Fall back: find "Final Diagnosis: ..." inside differential_diagnosis list
    for item in case.get("differential_diagnosis", []):
        item = item.strip()
        if item.lower().startswith("final diagnosis"):
            return re.sub(r"(?i)^final diagnosis\s*:?\s*", "", item).strip()

    # Last resort: scan discussion text
    discussion = case.get("discussion", {})
    text = ""
    if isinstance(discussion, dict):
        text = " ".join(discussion.values())
    else:
        text = str(discussion)
    m = re.search(r"(?i)final diagnosis\s*:?\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()

    return fd  # give up — return whatever was there


def clean_differentials(case: dict) -> list[str]:
    result = []
    for item in case.get("differential_diagnosis", []):
        item = item.strip()
        if not item:
            continue
        if item.lower().startswith("final diagnosis"):
            continue
        if item.lower().startswith("differential diagnosis list"):
            continue
        result.append(item)
    return result


def collect_images(case: dict, base_dir: Path) -> tuple[list[str], list[str]]:
    """Return (absolute_paths, descriptions) for all images that exist on disk."""
    paths, descs = [], []
    for img in case.get("images", []):
        desc = img.get("description", "")
        for f in img.get("files", []):
            full = (base_dir / f).resolve()
            if full.exists():
                paths.append(str(full))
                descs.append(desc)
    return paths, descs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="eurorad_dataset/extracted_preview.json")
    parser.add_argument("--output", default="eurorad_clinical_benchmark.jsonl")
    parser.add_argument("--min-images", type=int, default=1,
                        help="Skip cases with fewer than this many local images")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    base_dir = Path(args.input).parent  # eurorad_dataset/

    with open(args.input) as f:
        raw_cases = json.load(f)

    if args.max_cases:
        raw_cases = raw_cases[:args.max_cases]

    written = skipped = 0
    with open(args.output, "w") as out:
        for case in raw_cases:
            image_paths, image_descs = collect_images(case, base_dir)

            if len(image_paths) < args.min_images:
                skipped += 1
                continue

            final_dx = clean_final_diagnosis(case)
            differentials = clean_differentials(case)

            record = {
                "case_id":            case["case_id"],
                "title":              case.get("title", ""),
                "patient_age":        case.get("patient_age", ""),
                "patient_gender":     case.get("patient_gender", ""),
                "clinical_history":   case.get("clinical_history", ""),
                "images":             image_paths,
                "image_descriptions": image_descs,
                # Ground truth for scoring
                "gt_imaging_findings": case.get("imaging_findings", ""),
                "gt_differentials":    differentials,
                "gt_final_diagnosis":  final_dx,
            }
            out.write(json.dumps(record) + "\n")
            written += 1
            print(f"[{written}] case {case['case_id']} | "
                  f"{len(image_paths)} images | dx: {final_dx[:60]}")

    print(f"\nDone. Written: {written}, Skipped (no images): {skipped}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
