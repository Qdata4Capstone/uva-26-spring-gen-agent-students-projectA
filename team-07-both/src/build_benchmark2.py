"""
build_benchmark2.py
-------------------
Converts eurorad_dataset2/extracted_preview2.json into the same JSONL
format as eurorad_clinical_benchmark.jsonl, then produces:

  eurorad_clinical_benchmark2.jsonl   — new cases only (skips overlap)
  eurorad_clinical_benchmark_all.jsonl — combined (original 63 + new 172)

Image path fix: JSON says "pdf_images/FILENAME"
                actual files live in "eurorad_dataset2/pdf_images2/FILENAME"

Usage:
    python build_benchmark2.py
    python build_benchmark2.py --skip-overlap   # default: skip duplicates
    python build_benchmark2.py --include-overlap # keep duplicates too
"""

import json
import os
import argparse
from pathlib import Path

SRC_DIR   = Path(__file__).parent
NEW_JSON  = SRC_DIR / "eurorad_dataset2" / "extracted_preview2.json"
IMG_DIR   = SRC_DIR / "eurorad_dataset2" / "pdf_images2"
EXISTING  = SRC_DIR / "eurorad_clinical_benchmark.jsonl"
OUT_NEW   = SRC_DIR / "eurorad_clinical_benchmark2.jsonl"
OUT_ALL   = SRC_DIR / "eurorad_clinical_benchmark_all.jsonl"


def fix_image_path(raw_path: str) -> str:
    """Map 'pdf_images/FILENAME' → 'eurorad_dataset2/pdf_images2/FILENAME'."""
    filename = Path(raw_path).name
    return str(Path("eurorad_dataset2") / "pdf_images2" / filename)


def convert_case(raw: dict) -> dict:
    """Convert one raw case to the benchmark JSONL format."""
    # Flatten image paths and descriptions
    image_paths = []
    image_descriptions = []
    for img_entry in raw.get("images", []):
        for fpath in img_entry.get("files", []):
            fixed = fix_image_path(fpath)
            # Only include if the file actually exists
            full = SRC_DIR / fixed
            if full.exists():
                image_paths.append(fixed)
                image_descriptions.append(img_entry.get("description", ""))
            else:
                print(f"  Warning: image not found — {full}")

    return {
        "case_id":           str(raw["case_id"]),
        "title":             raw.get("title", ""),
        "patient_age":       str(raw.get("patient_age", "")),
        "patient_gender":    raw.get("patient_gender", ""),
        "clinical_history":  raw.get("clinical_history", ""),
        "images":            image_paths,
        "image_descriptions":image_descriptions,
        "gt_imaging_findings": raw.get("imaging_findings", ""),
        "gt_differentials":    raw.get("differential_diagnosis", []),
        "gt_final_diagnosis":  raw.get("final_diagnosis", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-overlap", action="store_true",
                        help="Include cases already in the existing benchmark")
    args = parser.parse_args()

    # Load existing case IDs
    existing_ids = set()
    existing_rows = []
    if EXISTING.exists():
        with open(EXISTING) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    existing_ids.add(str(row["case_id"]))
                    existing_rows.append(row)
        print(f"Existing benchmark: {len(existing_rows)} cases")

    # Load and convert new dataset
    with open(NEW_JSON) as f:
        raw_cases = json.load(f)
    print(f"New dataset raw: {len(raw_cases)} cases")

    converted = []
    skipped_overlap = 0
    skipped_no_image = 0

    for raw in raw_cases:
        case_id = str(raw["case_id"])

        if not args.include_overlap and case_id in existing_ids:
            skipped_overlap += 1
            continue

        case = convert_case(raw)

        if not case["images"]:
            skipped_no_image += 1
            print(f"  Skipping case {case_id} — no valid images")
            continue

        converted.append(case)

    print(f"Converted: {len(converted)} cases")
    print(f"Skipped (overlap): {skipped_overlap}")
    print(f"Skipped (no images): {skipped_no_image}")

    # Write new-only benchmark
    with open(OUT_NEW, "w") as f:
        for case in converted:
            f.write(json.dumps(case) + "\n")
    print(f"\nWrote: {OUT_NEW}  ({len(converted)} cases)")

    # Write combined benchmark
    all_rows = existing_rows + converted
    with open(OUT_ALL, "w") as f:
        for case in all_rows:
            f.write(json.dumps(case) + "\n")
    print(f"Wrote: {OUT_ALL}  ({len(all_rows)} cases total)")


if __name__ == "__main__":
    main()
