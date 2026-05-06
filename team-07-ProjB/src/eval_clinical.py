"""
eval_clinical.py
----------------
Evaluates the MedRAX agent on the clinical reasoning benchmark produced by
build_clinical_benchmark.py.

The agent is forced through a 4-step clinical pipeline per case:
  Step 1 – History intake:      key clinical facts extracted from history
  Step 2 – Imaging analysis:    findings from image + tools
  Step 3 – Differential Dx:     ranked list of up to 5 diagnoses
  Step 4 – Final impression:    single primary diagnosis + confidence

Each step is scored against Eurorad ground truth. Tool use is tracked
separately and scored against the expected tools for the imaging step.

Usage:
    python eval_clinical.py \
        --benchmark eurorad_clinical_benchmark.jsonl \
        --model     gpt-4o \
        --use-agent \
        --max-cases 20
"""

import json
import os
import re
import sys
import base64
import time
import argparse
import openai
import requests
import tempfile
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Clinical prompt
# ---------------------------------------------------------------------------

CLINICAL_SYSTEM = (
    "You are an expert radiologist performing a structured case review. "
    "Follow the 4-step format exactly and respond with valid JSON only."
)

CLINICAL_USER = """\
A patient presents with the following clinical history:
{history}

You have been provided with {n_images} medical image(s) for this case.
Analyze them carefully using available tools.

Respond ONLY with this JSON structure (no extra text):
{{
  "history_summary": ["fact1", "fact2", ...],
  "imaging_findings": "detailed description of what you see in the images",
  "differentials": ["Most likely Dx", "2nd", "3rd", "4th", "5th"],
  "final_diagnosis": "single primary diagnosis",
  "confidence": "high | medium | low",
  "reasoning": "one sentence linking findings to diagnosis"
}}
"""

# Agent mode — tool gathering prompt
TOOL_GATHER_PROMPT = """\
You are an expert radiologist assistant. A chest X-ray case needs analysis.

Image path: {image_path}

Think step by step before using any tool:
1. What clinical information do I need to support or rule out diagnoses?
2. Which tools will provide that information for this specific case?
3. Call the tools you judge most relevant. You may call multiple tools if needed.
4. After each tool result, reason about whether additional tools are needed.

Available tools and what they provide:
- chest_xray_classifier: quantitative probability scores (0.0-1.0) per pathology \
(useful for confirming or ruling out specific conditions)
- chest_xray_report_generator: structured radiology report with findings and impression \
(useful for a second-opinion narrative description)
- grad_cam_explainer: heatmap showing which image regions drove the classifier prediction \
(useful when you want to verify the classifier is attending to the right anatomy)

Do not produce a final diagnosis yet. Focus on gathering evidence through tools.
"""

# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

def gpt_judge(client: openai.OpenAI, prompt: str) -> float:
    """Ask the judge model to score something 0.0-1.0. Returns the float."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical evaluation judge. "
                 "Return a single JSON: {\"score\": <float 0.0-1.0>}"},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=50,
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return float(data.get("score", 0.0))
    except Exception as e:
        print(f"  Judge error: {e}")
        return 0.0


def score_findings(client, agent_findings: str, gt_findings: str) -> float:
    prompt = (
        f"Ground truth imaging findings:\n{gt_findings}\n\n"
        f"Agent imaging findings:\n{agent_findings}\n\n"
        "Score 0.0-1.0: how well does the agent's description capture the key "
        "radiological findings in the ground truth? "
        "1.0 = all key findings present, 0.0 = completely wrong."
    )
    return gpt_judge(client, prompt)


def score_consistency(client, findings: str, diagnosis: str) -> float:
    prompt = (
        f"Imaging findings stated by agent:\n{findings}\n\n"
        f"Final diagnosis stated by agent:\n{diagnosis}\n\n"
        "Score 0.0-1.0: do the stated findings logically support the stated diagnosis? "
        "1.0 = fully consistent, 0.0 = contradictory or unrelated."
    )
    return gpt_judge(client, prompt)


def score_diagnosis_match(client, agent_dx: str, gt_dx: str) -> float:
    prompt = (
        f"Ground truth diagnosis: {gt_dx}\n"
        f"Agent diagnosis:        {agent_dx}\n\n"
        "Score 0.0-1.0: are these semantically equivalent medical diagnoses? "
        "1.0 = same diagnosis (minor wording differences ok), "
        "0.5 = same disease family but less specific, "
        "0.0 = wrong diagnosis."
    )
    return gpt_judge(client, prompt)


def score_communication_quality(client, findings: str, differentials: list, reasoning: str) -> float:
    """Score how well the report communicates findings in an expert-friendly way."""
    diff_str = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(differentials))
    prompt = (
        f"A radiology AI produced the following report:\n\n"
        f"Imaging Findings:\n{findings}\n\n"
        f"Differentials:\n{diff_str}\n\n"
        f"Reasoning:\n{reasoning}\n\n"
        "Score 0.0-1.0 on how useful this report would be to a radiologist co-worker. Consider:\n"
        "- Use of correct radiological terminology\n"
        "- Logical structure (findings lead to differentials lead to reasoning)\n"
        "- Specificity and actionability (avoids vague boilerplate)\n"
        "- Appropriate level of detail — not too sparse, not redundant\n"
        "1.0 = expert-level report a radiologist would trust and act on\n"
        "0.5 = adequate but with notable gaps or imprecision\n"
        "0.0 = unhelpful, misleading, or unprofessional"
    )
    return gpt_judge(client, prompt)


def score_reasoning_coherence(client, history: str, findings: str,
                               differentials: list, diagnosis: str, reasoning: str) -> float:
    """Score whether the reasoning chain reflects sound clinical thinking."""
    diff_str = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(differentials))
    prompt = (
        f"Clinical history: {history}\n\n"
        f"Imaging findings stated by agent:\n{findings}\n\n"
        f"Differential diagnoses:\n{diff_str}\n\n"
        f"Final diagnosis: {diagnosis}\n"
        f"Reasoning: {reasoning}\n\n"
        "Score 0.0-1.0 on the quality of clinical reasoning. Consider:\n"
        "- Does the agent integrate clinical history with imaging findings?\n"
        "- Is the differential list clinically plausible given the findings?\n"
        "- Does the final diagnosis logically follow from findings and differentials?\n"
        "- Is the reasoning free from contradictions or logical jumps?\n"
        "1.0 = sound clinical reasoning throughout\n"
        "0.5 = mostly sound with minor gaps\n"
        "0.0 = flawed or incoherent reasoning"
    )
    return gpt_judge(client, prompt)


def score_uncertainty_appropriateness(confidence: str, diagnosis_score: float) -> float:
    """
    Score whether the expressed confidence is calibrated to actual performance.
    High confidence on a correct answer = 1.0
    Low confidence on a wrong answer = 1.0 (agent knew it didn't know)
    High confidence on a wrong answer = 0.0 (dangerous overconfidence)
    Low confidence on a correct answer = 0.5 (conservative but safe)
    """
    conf_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
    conf = conf_map.get(confidence.lower().strip(), 0.5)
    correct = diagnosis_score >= 0.5

    if correct and conf == 1.0:   return 1.0   # confident and right
    if correct and conf == 0.5:   return 0.8   # correct but hedged — acceptable
    if correct and conf == 0.0:   return 0.5   # correct but said "I don't know" — too conservative
    if not correct and conf == 0.0: return 1.0 # wrong but knew it — epistemic honesty
    if not correct and conf == 0.5: return 0.4 # wrong and uncertain — marginal
    if not correct and conf == 1.0: return 0.0 # wrong and overconfident — dangerous
    return 0.5


# ---------------------------------------------------------------------------
# Differential recall
# ---------------------------------------------------------------------------

def differential_recall(agent_diffs: list[str], gt_dx: str, client: openai.OpenAI) -> dict:
    """Check if gt_dx appears in agent's differential list (top-1, top-3, top-5)."""
    results = {"in_top1": False, "in_top3": False, "in_top5": False, "rank": -1}
    if not agent_diffs or not gt_dx:
        return results

    prompt_tmpl = (
        "Ground truth diagnosis: {gt}\n"
        "Candidate diagnosis:    {cand}\n"
        "Is the candidate semantically equivalent to the ground truth? "
        "Return JSON: {{\"match\": true|false}}"
    )
    for i, cand in enumerate(agent_diffs[:5]):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Medical diagnosis matcher. JSON only."},
                    {"role": "user", "content": prompt_tmpl.format(gt=gt_dx, cand=cand)},
                ],
                max_completion_tokens=20,
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            if data.get("match"):
                results["rank"] = i + 1
                if i == 0: results["in_top1"] = True
                if i < 3:  results["in_top3"] = True
                if i < 5:  results["in_top5"] = True
                break
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images_as_b64(image_paths: list[str]) -> list[str]:
    b64_list = []
    for path in image_paths:
        try:
            with open(path, "rb") as f:
                b64_list.append(base64.b64encode(f.read()).decode("utf-8"))
        except Exception as e:
            print(f"  Failed to load image {path}: {e}")
    return b64_list


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------


_TOOL_FAILURE_MARKERS = ("error", "exception", "traceback", "failed", "invalid", "not supported", "unsupported")

def _extract_tool_outputs(messages: list) -> str:
    """
    Collect ToolMessage contents, skipping empty or error outputs.
    Returns formatted block, or a note if nothing useful was found.
    """
    from langchain_core.messages import ToolMessage
    parts = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            content = content.strip()
            if not content:
                continue
            lower = content.lower()
            if any(marker in lower for marker in _TOOL_FAILURE_MARKERS) and len(content) < 300:
                continue
            parts.append(f"[{msg.name}]\n{content}")
    return "\n\n".join(parts) if parts else "No applicable tool outputs available."


def _parse_json(raw: str) -> dict | None:
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def run_agent(case: dict, agent, use_agent: bool, client: openai.OpenAI) -> tuple[dict | None, list[str], list]:
    """
    Two-mode pipeline (no self-reflection):

    Agent mode:
      Phase 1 — tool gathering: agent calls imaging tools freely.
      Phase 2 — synthesis: GPT produces the clinical JSON using images + history
                + tool outputs combined.

    Non-agent mode:
      Direct GPT call from images + history alone.

    Returns (parsed, tools_called, agent_messages).
    """
    from langchain_core.messages import ToolMessage

    image_paths = case["images"]
    history     = case["clinical_history"]
    n_images    = len(image_paths)

    tools_called   = []
    agent_messages = []
    tool_outputs   = ""

    if use_agent:
        # ------------------------------------------------------------------
        # Phase 1: tool gathering
        # ------------------------------------------------------------------
        pass1_messages = []
        if image_paths:
            pass1_messages.append({"role": "user", "content": f"image_path: {image_paths[0]}"})
        for path in image_paths:
            b64s = load_images_as_b64([path])
            if b64s:
                pass1_messages.append({
                    "role": "user",
                    "content": [{"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{b64s[0]}"}}],
                })
        pass1_messages.append({
            "role": "user",
            "content": [{"type": "text",
                         "text": TOOL_GATHER_PROMPT.format(
                             image_path=image_paths[0] if image_paths else "N/A"
                         )}],
        })

        thread_id   = str(time.time())
        pass1_state = agent.workflow.invoke(
            {"messages": pass1_messages},
            {"configurable": {"thread_id": thread_id}},
        )

        tools_called   = [m.name for m in pass1_state["messages"] if isinstance(m, ToolMessage)]
        agent_messages = pass1_state["messages"]
        tool_outputs   = _extract_tool_outputs(pass1_state["messages"])

    # ------------------------------------------------------------------
    # Phase 2 (both modes): synthesis — tool outputs included when available
    # ------------------------------------------------------------------
    tool_section = f"\n\nSpecialized imaging tool outputs:\n{tool_outputs}" if tool_outputs else ""
    content = [{"type": "text",
                "text": CLINICAL_USER.format(history=history, n_images=n_images) + tool_section}]
    for b64 in load_images_as_b64(image_paths):
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    resp = client.chat.completions.create(
        model=client._model,
        messages=[
            {"role": "system", "content": CLINICAL_SYSTEM},
            {"role": "user",   "content": content},
        ],
        max_completion_tokens=800,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    parsed = _parse_json(resp.choices[0].message.content)
    return parsed, tools_called, agent_messages


# ---------------------------------------------------------------------------
# Modality detection
# ---------------------------------------------------------------------------

_XR_KEYWORDS = {
    "chest x-ray", "chest x ray", "chest radiograph", "chest radiography",
    "cxr", "pa chest", "ap chest", "frontal chest", "plain film", "chest film",
    "plain radiograph", "chest ap", "posteroanterior",
}

def _is_chest_xray_case(case: dict) -> bool:
    """Return True if the case contains at least one chest X-ray image."""
    text = " ".join([
        case.get("title", ""),
        *case.get("image_descriptions", []),
    ]).lower()
    return any(kw in text for kw in _XR_KEYWORDS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="eurorad_clinical_benchmark.jsonl")
    parser.add_argument("--output",    default=None, help="Log file (default: auto-named)")
    parser.add_argument("--model",     default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--use-agent", action="store_true")
    parser.add_argument("--model-dir",
                        default=os.environ.get("MODEL_DIR",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mydownload")))
    parser.add_argument("--prompt-file", default="medrax/docs/system_prompts.txt")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--xr-only", action="store_true",
                        help="Restrict evaluation to chest X-ray cases only")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    # Model client: uses OPENAI_BASE_URL if set (e.g. local vllm serving Qwen)
    openai_kwargs = {"api_key": api_key}
    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    # Judge client: always hits real OpenAI regardless of OPENAI_BASE_URL env var
    judge_api_key = os.getenv("JUDGE_OPENAI_API_KEY", api_key)
    judge_client = openai.OpenAI(
        api_key=judge_api_key,
        base_url="https://api.openai.com/v1",  # explicit override — SDK reads env var otherwise
    )

    # Model client for clinical reasoning (may point to local vllm)
    gpt_client = openai.OpenAI(**openai_kwargs)
    gpt_client._model = args.model

    # Init agent
    agent = None
    if args.use_agent:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from main import initialize_agent
        agent, _ = initialize_agent(
            prompt_file=args.prompt_file,
            tools_to_use=[
                "ChestXRayClassifierTool",
                "ChestXRayReportGeneratorTool",
                "GradCAMExplainerTool",
            ],
            model_dir=args.model_dir,
            temp_dir=os.path.join(args.model_dir, "temp"),
            model=args.model,
            temperature=args.temperature,
            openai_kwargs=openai_kwargs,
        )
        print("Agent initialized.")

    # Load benchmark
    with open(args.benchmark) as f:
        cases = [json.loads(l) for l in f if l.strip()]

    if args.xr_only:
        cases = [c for c in cases if _is_chest_xray_case(c)]
        print(f"XR-only filter applied: {len(cases)} chest X-ray cases retained.")

    if args.max_cases:
        cases = cases[:args.max_cases]

    log_file = args.output or f"clinical_eval_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    def _score_parsed(parsed: dict, case: dict) -> dict:
        """Score a single parsed output dict against ground truth. Returns raw score values."""
        s_findings = score_findings(
            judge_client, parsed.get("imaging_findings", ""), case["gt_imaging_findings"])
        diff_scores = differential_recall(
            parsed.get("differentials", []), case["gt_final_diagnosis"], judge_client)
        s_diagnosis = score_diagnosis_match(
            judge_client, parsed.get("final_diagnosis", ""), case["gt_final_diagnosis"])
        s_consistency = score_consistency(
            judge_client, parsed.get("imaging_findings", ""), parsed.get("final_diagnosis", ""))
        s_communication = score_communication_quality(
            judge_client,
            parsed.get("imaging_findings", ""),
            parsed.get("differentials", []),
            parsed.get("reasoning", ""),
        )
        s_reasoning = score_reasoning_coherence(
            judge_client,
            case["clinical_history"],
            parsed.get("imaging_findings", ""),
            parsed.get("differentials", []),
            parsed.get("final_diagnosis", ""),
            parsed.get("reasoning", ""),
        )
        confidence_str = parsed.get("confidence", "medium")
        s_uncertainty = score_uncertainty_appropriateness(confidence_str, s_diagnosis)
        return {
            "findings": s_findings, "diagnosis": s_diagnosis,
            "consistency": s_consistency, "diff_scores": diff_scores,
            "communication": s_communication, "reasoning_coherence": s_reasoning,
            "uncertainty_appropriate": s_uncertainty, "confidence_raw": confidence_str,
        }

    IMAGING_TOOLS = {"chest_xray_classifier", "chest_xray_report_generator"}

    scores = {
        "findings": [], "consistency": [], "diagnosis": [],
        "diff_top1": [], "diff_top3": [], "tool_recall": [],
        "communication": [], "reasoning_coherence": [], "uncertainty_appropriate": [],
        "confidence_raw": [],
    }

    with open(log_file, "w") as log:
        for i, case in enumerate(cases):
            print(f"\n[{i+1}/{len(cases)}] Case {case['case_id']} — {case['title']}")

            parsed, tools_called, agent_messages = \
                run_agent(case, agent, args.use_agent, gpt_client)

            if parsed is None:
                print("  ✗ Failed to parse agent response.")
                log.write(json.dumps({"case_id": case["case_id"], "status": "parse_error"}) + "\n")
                continue

            s = _score_parsed(parsed, case)

            tool_recall = 0.0
            if args.use_agent:
                tool_recall = len(set(tools_called) & IMAGING_TOOLS) / len(IMAGING_TOOLS)

            scores["findings"].append(s["findings"])
            scores["consistency"].append(s["consistency"])
            scores["diagnosis"].append(s["diagnosis"])
            scores["diff_top1"].append(int(s["diff_scores"]["in_top1"]))
            scores["diff_top3"].append(int(s["diff_scores"]["in_top3"]))
            scores["tool_recall"].append(tool_recall)
            scores["communication"].append(s["communication"])
            scores["reasoning_coherence"].append(s["reasoning_coherence"])
            scores["uncertainty_appropriate"].append(s["uncertainty_appropriate"])
            scores["confidence_raw"].append(s["confidence_raw"])

            entry = {
                "case_id":      case["case_id"],
                "title":        case["title"],
                "gt_diagnosis": case["gt_final_diagnosis"],
                "tools_called": tools_called,
                "diagnosis":    parsed.get("final_diagnosis", ""),
                "differentials": parsed.get("differentials", []),
                "findings":     parsed.get("imaging_findings", ""),
                "confidence":   parsed.get("confidence", ""),
                "scores": {
                    "imaging_findings":        round(s["findings"], 3),
                    "diagnosis_match":         round(s["diagnosis"], 3),
                    "consistency":             round(s["consistency"], 3),
                    "diff_in_top1":            s["diff_scores"]["in_top1"],
                    "diff_in_top3":            s["diff_scores"]["in_top3"],
                    "tool_recall":             round(tool_recall, 3),
                    "communication_quality":   round(s["communication"], 3),
                    "reasoning_coherence":     round(s["reasoning_coherence"], 3),
                    "uncertainty_appropriate": round(s["uncertainty_appropriate"], 3),
                    "confidence":              s["confidence_raw"],
                },
            }
            log.write(json.dumps(entry) + "\n")

            print(f"  Findings: {s['findings']:.2f} | Dx: {s['diagnosis']:.2f} | "
                  f"Consist: {s['consistency']:.2f} | conf={s['confidence_raw']}")
            print(f"  Diff top-1: {s['diff_scores']['in_top1']}  "
                  f"top-3: {s['diff_scores']['in_top3']}  "
                  f"rank: {s['diff_scores']['rank']}")
            if args.use_agent:
                print(f"  Tools: {tools_called}  recall: {tool_recall:.0%}")

    # Summary
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    n = len(scores["findings"])

    conf_buckets = {"high": [], "medium": [], "low": []}
    for i in range(n):
        c = scores["confidence_raw"][i].lower().strip()
        if c in conf_buckets:
            conf_buckets[c].append(scores["diagnosis"][i])

    rcw_score = avg([
        avg(scores["communication"]),
        avg(scores["reasoning_coherence"]),
        avg(scores["uncertainty_appropriate"]),
    ])

    W = 62
    print(f"\n{'='*W}")
    mode = "Agent" if args.use_agent else "Non-agent (baseline)"
    print(f"Clinical Reasoning Evaluation [{mode}] — {n} cases")
    print(f"{'='*W}")
    print(f"{'Metric':<35} {'Score':>8}")
    print(f"{'-'*W}")
    print(f"{'Imaging Findings Score':<35} {avg(scores['findings']):8.3f}")
    print(f"{'Differential Recall@1':<35} {avg(scores['diff_top1']):8.1%}")
    print(f"{'Differential Recall@3':<35} {avg(scores['diff_top3']):8.1%}")
    print(f"{'Final Diagnosis Score':<35} {avg(scores['diagnosis']):8.3f}")
    print(f"{'Consistency Score':<35} {avg(scores['consistency']):8.3f}")
    print(f"{'Uncertainty Appropriateness':<35} {avg(scores['uncertainty_appropriate']):8.3f}")
    if args.use_agent:
        print(f"{'Imaging Tool Recall':<35} {avg(scores['tool_recall']):8.1%}")
    print(f"{'-'*W}")
    print(f"{'Reliable Co-worker Score':<35} {rcw_score:8.3f}")
    print(f"  Communication Quality:      {avg(scores['communication']):8.3f}")
    print(f"  Reasoning Coherence:        {avg(scores['reasoning_coherence']):8.3f}")
    print(f"{'-'*W}")
    print(f"Confidence Calibration")
    for level in ("high", "medium", "low"):
        bucket = conf_buckets[level]
        if bucket:
            print(f"  {level:6s}: {len(bucket):2d} cases  avg diagnosis = {avg(bucket):.3f}")
        else:
            print(f"  {level:6s}:  0 cases")
    print(f"{'='*W}")
    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
