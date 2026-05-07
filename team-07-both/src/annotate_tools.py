"""
annotate_tools.py
-----------------
Adds `expected_tools` and `tool_required` fields to a benchmark JSONL by:
  1. Fast rule-based pass using the question's `categories` field.
  2. GPT-4 verification pass for ambiguous cases.

Usage:
    python annotate_tools.py --input chestagentbench/metadata.jsonl \
                              --output chestagentbench/metadata_annotated.jsonl
    # Skip GPT pass (rules only):
    python annotate_tools.py --input ... --output ... --rules-only
"""

import json
import os
import argparse
import openai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Tool catalogue — matches the exact .name values in medrax/tools/
# ---------------------------------------------------------------------------
TOOL_DESCRIPTIONS = {
    "chest_xray_classifier": (
        "Classifies 18 chest pathologies (e.g. Pneumonia, Effusion, Mass) "
        "and returns probability scores. Use when the question asks about "
        "disease likelihood, classifier confidence, or pathology detection."
    ),
    "chest_xray_report_generator": (
        "Generates a structured radiology report (Findings + Impression). "
        "Use when the question asks what the report says, or requires a "
        "comprehensive written analysis of the X-ray."
    ),
    "chest_xray_expert": (
        "Visual QA model fine-tuned on chest X-rays. Use when the question "
        "asks about specific radiographic features, abnormalities, or "
        "requires image-level description beyond classification."
    ),
    "llava_med_qa": (
        "General medical VQA model. Use when the question requires open-ended "
        "medical image interpretation or explanation of findings."
    ),
    "xray_phrase_grounding": (
        "Localises anatomical phrases to bounding-box regions. Use when the "
        "question asks *where* a finding is, which region shows a finding, "
        "or requires spatial/anatomical localisation."
    ),
    "chest_xray_segmentation": (
        "Segments lung anatomy (lobes, pleura, etc.). Use when the question "
        "involves measuring areas, comparing lobe sizes, or understanding "
        "anatomical boundaries."
    ),
    "GradCAMExplainerTool": (
        "Generates GradCAM heatmaps showing which image regions drove the "
        "classifier's prediction. Use when the question asks why the model "
        "flagged a condition, or which region supports the prediction."
    ),
}

# ---------------------------------------------------------------------------
# Rule-based annotation
# ---------------------------------------------------------------------------
CATEGORY_TO_TOOLS: dict[str, list[str]] = {
    "localization":    ["xray_phrase_grounding"],
    "classification":  ["chest_xray_classifier"],
    "detection":       ["chest_xray_classifier", "chest_xray_expert"],
    "characterization":["chest_xray_expert", "llava_med_qa"],
    "comparison":      ["chest_xray_expert", "llava_med_qa"],
    "diagnosis":       ["chest_xray_classifier", "chest_xray_report_generator"],
    "relationship":    ["chest_xray_expert"],
    "reasoning":       [],   # pure reasoning — no tool required
}

# Questions with these categories need tool use to answer reliably
TOOL_REQUIRED_CATEGORIES = {"localization", "classification", "detection"}


def rule_based_annotation(example: dict) -> tuple[list[str], bool]:
    """Return (expected_tools, tool_required) using category heuristics."""
    raw = example.get("categories", "")
    categories = [c.strip().lower() for c in raw.split(",") if c.strip()]

    tools: set[str] = set()
    for cat in categories:
        tools.update(CATEGORY_TO_TOOLS.get(cat, []))

    tool_required = bool(tools and any(c in TOOL_REQUIRED_CATEGORIES for c in categories))
    return sorted(tools), tool_required


# ---------------------------------------------------------------------------
# GPT-4 verification
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a radiology AI benchmark designer. Given a multiple-choice question
about a chest X-ray, decide which tools (if any) from the list below are
REQUIRED to answer it correctly — i.e. the question cannot be reliably
answered by visual inspection alone.

Available tools:
{tool_list}

Respond with valid JSON only:
{{
  "expected_tools": ["tool_name", ...],   // empty list if none needed
  "tool_required": true | false,          // true if at least one tool is essential
  "reasoning": "one sentence"
}}
"""


def gpt_annotation(example: dict, client: openai.OpenAI) -> tuple[list[str], bool]:
    tool_list = "\n".join(
        f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()
    )
    user_content = (
        f"Question:\n{example['question']}\n\n"
        f"Categories: {example.get('categories', 'unknown')}\n"
        f"Explanation hint: {example.get('explanation', '')}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(tool_list=tool_list)},
                {"role": "user",   "content": user_content},
            ],
            max_completion_tokens=300,
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("expected_tools", []), bool(data.get("tool_required", False))
    except Exception as e:
        print(f"  GPT annotation failed for {example.get('question_id')}: {e}")
        return None, None  # fall back to rule-based result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="chestagentbench/metadata.jsonl")
    parser.add_argument("--output", default="chestagentbench/metadata_annotated.jsonl")
    parser.add_argument("--rules-only", action="store_true",
                        help="Skip GPT verification, use rule-based only")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    client = None
    if not args.rules_only:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set; use --rules-only to skip GPT.")
        kwargs = {}
        if base_url := os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url
        client = openai.OpenAI(api_key=api_key, **kwargs)

    with open(args.input) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    if args.max_cases:
        examples = examples[:args.max_cases]

    annotated = []
    for i, ex in enumerate(examples):
        rule_tools, rule_required = rule_based_annotation(ex)

        if args.rules_only or not rule_required:
            # Unambiguous: no tool needed, or rules-only mode
            ex["expected_tools"] = rule_tools
            ex["tool_required"]  = rule_required
            source = "rules"
        else:
            # Tool-required case: verify with GPT
            gpt_tools, gpt_required = gpt_annotation(ex, client)
            if gpt_tools is not None:
                ex["expected_tools"] = gpt_tools
                ex["tool_required"]  = gpt_required
                source = "gpt"
            else:
                ex["expected_tools"] = rule_tools
                ex["tool_required"]  = rule_required
                source = "rules(fallback)"

        print(f"[{i+1}/{len(examples)}] {ex['question_id']} | "
              f"tools={ex['expected_tools']} required={ex['tool_required']} ({source})")
        annotated.append(ex)

    with open(args.output, "w") as f:
        for ex in annotated:
            f.write(json.dumps(ex) + "\n")

    total = len(annotated)
    n_required = sum(1 for e in annotated if e["tool_required"])
    print(f"\nDone. {total} cases → {args.output}")
    print(f"Tool-required: {n_required}/{total} ({n_required/total:.0%})")


if __name__ == "__main__":
    main()
