#!/usr/bin/env python3
"""
Demo runner for the Legal Compliance Agent.
Runs two use cases from specs.md:

  Use Case A: CVPR submission with YouTube data
  Use Case B: GDPR-compliant indoor surveillance processing

Usage:
    python run_demo.py [--use-case A|B] [--dataset-dir DIR] [--output-dir DIR]

Environment:
    ANTHROPIC_API_KEY  — required for LLM controller

Quick test (no LLM, just egoblur on 2 clips):
    python run_demo.py --quick-test
"""

import argparse
import json
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_DIR = "/bigtemp2/tsx4zn/legal/images"
OUTPUT_DIR  = "/bigtemp2/tsx4zn/legal/output"

USE_CASE_A = (
    "I'm writing a CVPR 2026 paper and have video frames downloaded from YouTube. "
    "Outdoor scenes with pedestrians and vehicles. "
    "Please process to comply with CVPR 2026 Open Science Policy — "
    "de-identify faces and license plates, strip EXIF, generate compliance report."
)

USE_CASE_B = (
    "Processing indoor surveillance footage for model training. "
    "Must comply with GDPR. Fixed cameras, indoor pedestrians."
)


def run_quick_test(dataset_dir: str, output_dir: str):
    """
    Quick test: run egoblur directly on first 2 clips, no LLM needed.
    """
    from pathlib import Path
    from agent.tools.egoblur_tool import EgoBlurTool

    logger.info("=== Quick Test Mode (egoblur only, no LLM) ===")
    tool = EgoBlurTool()

    if not tool.is_available:
        logger.error("EgoBlur not available. Check model paths.")
        return

    clips = sorted(Path(dataset_dir).iterdir())[:2]
    for clip in clips:
        if not clip.is_dir():
            continue
        out = Path(output_dir) / "quick_test" / clip.name
        logger.info(f"Processing clip: {clip.name}")
        result = tool.process_clip_dir(str(clip), str(out))
        logger.info(f"  → {result['processed']} images blurred to {out}")

    logger.info(f"Quick test complete. Output: {output_dir}/quick_test/")


def run_use_case(
    use_case_text: str,
    dataset_dir: str,
    output_dir: str,
    api_key: str,
):
    """Run the full agent pipeline for a use case."""
    from agent.agent import LegalComplianceAgent

    agent = LegalComplianceAgent(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        api_key=api_key,
        max_workers=4,
    )

    result = agent.run(use_case_text)

    print("\n" + "=" * 60)
    print("AGENT RUN COMPLETE")
    print("=" * 60)
    print(f"Session ID   : {result['session_id']}")
    print(f"Output Dir   : {result['output_dir']}")
    print(f"Report       : {result['report_path']}")
    print(f"\nSummary:\n  {result['summary']}")
    print("\nPlan:")
    print(json.dumps(result["plan"], indent=2, ensure_ascii=False))
    print("\nRequirements applied:")
    print(json.dumps(result["requirements"], indent=2, ensure_ascii=False))

    ok = sum(1 for p in result["processing_log"] if p.get("status") == "success")
    failed_reid = sum(1 for c in result["critic_results"] if not c.get("passed"))
    print(f"\nProcessing: {ok}/{len(result['processing_log'])} clips succeeded")
    print(f"Re-ID validation: {failed_reid} clip(s) failed threshold")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="Legal Compliance Agent Demo")
    parser.add_argument(
        "--use-case", choices=["A", "B"], default="A",
        help="Use case to run (A=CVPR/YouTube, B=GDPR/surveillance)"
    )
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Run quick test with egoblur only (no API key needed)"
    )
    parser.add_argument("--api-key", default=None, help="Anthropic API key")
    args = parser.parse_args()

    if args.quick_test:
        run_quick_test(args.dataset_dir, args.output_dir)
        return

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error(
            "ANTHROPIC_API_KEY not set. Use --api-key or set env var. "
            "For a test without LLM, use --quick-test."
        )
        sys.exit(1)

    use_case_text = USE_CASE_A if args.use_case == "A" else USE_CASE_B
    logger.info(f"Running Use Case {args.use_case}")
    logger.info(f"User: {use_case_text[:80]}...")

    run_use_case(use_case_text, args.dataset_dir, args.output_dir, api_key)


if __name__ == "__main__":
    main()
