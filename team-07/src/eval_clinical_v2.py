"""
eval_clinical_v2.py
-------------------
Tool-grounded self-reflection pipeline for clinical reasoning evaluation.

Implements three tool-based self-reflection patterns:
  A – Targeted Hypothesis Testing:  classifier is called AFTER the preliminary diagnosis
      to challenge it — does the top classifier score support or contradict the diagnosis?
  B – Spatial Grounding via GradCAM: heatmap is used to verify whether the model's stated
      finding locations match the regions the classifier actually attended to.
  C – Cross-Tool Consistency Check: report generator provides an independent narrative;
      mismatches between its findings and the model's findings trigger explicit reconciliation.

Pipeline per case:
  Pass 1 – Vision-only synthesis:  GPT diagnoses from images + history alone (no tools).
  Pass 2 – Targeted tool calls:    LangChain agent calls all three tools, each directed at
                                   challenging the Pass 1 diagnosis (Patterns A, B, C).
  Pass 3 – Reconciliation:         GPT receives initial diagnosis + per-tool evidence and must
                                   address each contradiction. May revise ANY field.

Key difference from v1:
  v1 calls tools BEFORE diagnosing (tools as data sources).
  v2 calls tools AFTER diagnosing (tools as external validators / falsifiers).

Usage:
    python eval_clinical_v2.py \\
        --benchmark eurorad_clinical_benchmark.jsonl \\
        --model     gpt-4o \\
        --use-agent \\
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
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLINICAL_SYSTEM = (
    "You are an expert radiologist performing a structured case review. "
    "Follow the 4-step format exactly and respond with valid JSON only."
)

# Pass 1: vision-only — no tools, clean baseline
VISION_ONLY_USER = """\
A patient presents with the following clinical history:
{history}

You have been provided with {n_images} chest X-ray image(s).
Analyze the images carefully from your direct visual observations alone.

Respond ONLY with this JSON structure (no extra text):
{{
  "history_summary": ["fact1", "fact2", ...],
  "imaging_findings": "thorough visual description of what you directly observe in the images",
  "differentials": ["Most likely Dx", "2nd", "3rd", "4th", "5th"],
  "final_diagnosis": "single primary diagnosis",
  "confidence": "high | medium | low",
  "reasoning": "one sentence linking findings to diagnosis"
}}
"""

# Pass 2: targeted tool calls — challenge the preliminary diagnosis
# The agent already has the diagnosis; it must now try to falsify it.
TARGETED_TOOL_PROMPT = """\
You are an expert radiologist assistant performing tool-based self-reflection \
on a preliminary clinical assessment.

Image path: {image_path}

Preliminary diagnosis:    {diagnosis}
Preliminary differentials: {differentials}
Preliminary findings:     {findings}

Your job is to challenge this assessment with exactly three targeted tests below. \
Do NOT call ImageVisualizerTool or any tool not listed here.

[TEST A — Hypothesis Testing]
Call chest_xray_classifier.
After the result, note:
  • Is the preliminary diagnosis among the top-scoring predictions?
  • Does any other condition score more than 0.15 higher?

[TEST B — Spatial Grounding]
Call grad_cam_explainer with target_class set to the top-scoring label from Test A
(use the preliminary diagnosis as target_class if the classifier did not return a clear top label).
After the result, note:
  • Does the highlighted anatomy match the region described in the preliminary findings?
    (e.g. if findings say "right lower lobe opacity", does the heatmap attend there?)
  Do NOT call ImageVisualizerTool after this — the text description from GradCAM is sufficient.

[TEST C — Independent Report]
Call chest_xray_report_generator.
After the result, note:
  • Any key finding (consolidation, effusion, pneumothorax, cardiomegaly, mass, opacity)
    mentioned in the report that is ABSENT from the preliminary findings, or vice versa.
  • Whether the report's impression agrees or conflicts with the preliminary diagnosis.

Call only these three tools, in order. After each result write a one-line annotation.
Do NOT produce a revised diagnosis yet — only gather and annotate evidence.
"""

# Pass 3: reconciliation — all fields open for revision
RECONCILIATION_PROMPT = """\
You previously produced a preliminary clinical assessment:
{initial_json}

Three tool-based tests were run to challenge that assessment:

[TEST A — Classifier Scores (Hypothesis Testing)]
{classifier_output}

[TEST B — GradCAM Heatmap (Spatial Grounding)]
{gradcam_output}

[TEST C — Report Generator (Independent Narrative)]
{report_output}

Instructions:
For each tool result, classify agreement as "agreement", "partial", or "contradiction",
then address every contradiction explicitly before finalising your output.

Revision rules:
• Pattern A — If the classifier's top prediction differs from your diagnosis with a gap > 0.15,
  seriously reconsider your diagnosis or re-rank differentials.
• Pattern B — If the GradCAM heatmap highlights a region inconsistent with your stated finding
  location, acknowledge the spatial discrepancy and update imaging_findings accordingly.
• Pattern C — If the report generator describes a clinically significant finding absent from
  your report (or contradicts one of yours), either adopt it or explain why you discount it.

You MAY revise any field: final_diagnosis, differentials, imaging_findings, confidence, reasoning.
Set "revised": true only if at least one field actually changed from the preliminary assessment.

Respond ONLY with this JSON (no extra text):
{{
  "history_summary": ["fact1", ...],
  "imaging_findings": "updated or unchanged description",
  "differentials": ["Most likely Dx", "2nd", "3rd", "4th", "5th"],
  "final_diagnosis": "single primary diagnosis",
  "confidence": "high | medium | low",
  "reasoning": "one sentence linking findings to diagnosis",
  "revised": true,
  "revision_reasons": ["what changed and why", ...],
  "tool_contradictions": {{
    "classifier":       "agreement | partial | contradiction",
    "gradcam":          "agreement | partial | contradiction | unclear",
    "report_generator": "agreement | partial | contradiction"
  }}
}}
"""

# ---------------------------------------------------------------------------
# GPT-4 judge helpers  (identical to v1)
# ---------------------------------------------------------------------------

def gpt_judge(client: openai.OpenAI, prompt: str) -> float:
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
    conf_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
    conf = conf_map.get(confidence.lower().strip(), 0.5)
    correct = diagnosis_score >= 0.5
    if correct and conf == 1.0:    return 1.0
    if correct and conf == 0.5:    return 0.8
    if correct and conf == 0.0:    return 0.5
    if not correct and conf == 0.0: return 1.0
    if not correct and conf == 0.5: return 0.4
    if not correct and conf == 1.0: return 0.0
    return 0.5


def differential_recall(agent_diffs: list[str], gt_dx: str, client: openai.OpenAI) -> dict:
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
# Tool output helpers
# ---------------------------------------------------------------------------

_TOOL_FAILURE_MARKERS = (
    "error", "exception", "traceback", "failed", "invalid", "not supported", "unsupported"
)


def _extract_tool_output_by_name(messages: list, tool_name: str) -> str:
    """Return the content of the first ToolMessage whose name matches tool_name.

    Matching is done after stripping underscores and lowercasing so that
    'grad_cam_explainer' matches 'GradCAMExplainerTool' and vice versa.
    """
    from langchain_core.messages import ToolMessage
    needle = tool_name.lower().replace("_", "")
    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = (getattr(msg, "name", "") or "").lower().replace("_", "")
            if needle in name:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                return content.strip() or "Tool returned empty output."
    return "Tool was not called or produced no output."


def _extract_tool_outputs(messages: list) -> str:
    from langchain_core.messages import ToolMessage
    parts = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            content = content.strip()
            if not content:
                continue
            lower = content.lower()
            if any(m in lower for m in _TOOL_FAILURE_MARKERS) and len(content) < 300:
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


# ---------------------------------------------------------------------------
# Modality detection
# ---------------------------------------------------------------------------

_XR_KEYWORDS = {
    "chest x-ray", "chest x ray", "chest radiograph", "chest radiography",
    "cxr", "pa chest", "ap chest", "frontal chest", "plain film", "chest film",
    "plain radiograph", "chest ap", "posteroanterior",
}


def _is_chest_xray_case(case: dict) -> bool:
    text = " ".join([case.get("title", ""), *case.get("image_descriptions", [])]).lower()
    return any(kw in text for kw in _XR_KEYWORDS)


# ---------------------------------------------------------------------------
# Agent runner v2: tool-grounded self-reflection
# ---------------------------------------------------------------------------

def run_agent_v2(
    case: dict, agent, use_agent: bool, client: openai.OpenAI
) -> tuple[dict | None, dict | None, list[str], list]:
    """
    Three-pass tool-grounded self-reflection pipeline.

    Pass 1 — Vision-only synthesis (no tools):
        GPT reads images + history and produces an initial JSON diagnosis.
        Tools are intentionally withheld so the baseline reflects pure visual reasoning.

    Pass 2 — Targeted tool calls (Patterns A, B, C):
        The LangChain agent receives the Pass 1 diagnosis and is instructed to call
        all three tools specifically to challenge it:
          A) chest_xray_classifier  → does the classifier agree with the diagnosis?
          B) grad_cam_explainer     → does the heatmap match the stated finding locations?
          C) chest_xray_report_generator → does an independent report agree?

    Pass 3 — Reconciliation:
        GPT receives the Pass 1 JSON plus the three per-tool outputs.
        It must classify each as agreement/partial/contradiction and address every
        contradiction. Unlike v1, ALL fields may be revised.

    Returns (initial_parsed, reflected_parsed, tools_called, agent_messages).
    """
    from langchain_core.messages import ToolMessage

    image_paths = case["images"]
    history     = case["clinical_history"]
    n_images    = len(image_paths)

    tools_called   = []
    agent_messages = []

    # ------------------------------------------------------------------
    # Pass 1: Vision-only synthesis — no tools
    # ------------------------------------------------------------------
    content = [{"type": "text",
                "text": VISION_ONLY_USER.format(history=history, n_images=n_images)}]
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
    initial_parsed = _parse_json(resp.choices[0].message.content)

    if not use_agent or initial_parsed is None:
        return initial_parsed, None, tools_called, agent_messages

    # ------------------------------------------------------------------
    # Pass 2: Targeted tool calls — challenge the Pass 1 diagnosis
    # ------------------------------------------------------------------
    diagnosis     = initial_parsed.get("final_diagnosis", "unknown")
    differentials = ", ".join(initial_parsed.get("differentials", []))
    findings      = initial_parsed.get("imaging_findings", "")

    pass2_messages = []
    # Supply images to the agent so vision-capable tools can use them
    for path in image_paths:
        b64s = load_images_as_b64([path])
        if b64s:
            pass2_messages.append({
                "role": "user",
                "content": [{"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64s[0]}"}}],
            })
    # Also supply file path as text for tools that accept a path argument
    if image_paths:
        pass2_messages.append({"role": "user", "content": f"image_path: {image_paths[0]}"})

    pass2_messages.append({
        "role": "user",
        "content": [{"type": "text",
                     "text": TARGETED_TOOL_PROMPT.format(
                         image_path=image_paths[0] if image_paths else "N/A",
                         diagnosis=diagnosis,
                         differentials=differentials,
                         findings=findings,
                     )}],
    })

    thread_id  = str(time.time())
    pass2_state = agent.workflow.invoke(
        {"messages": pass2_messages},
        {"configurable": {"thread_id": thread_id}},
    )

    tools_called   = [m.name for m in pass2_state["messages"] if isinstance(m, ToolMessage)]
    agent_messages = pass2_state["messages"]

    # Extract per-pattern tool outputs for Pass 3
    classifier_output = _extract_tool_output_by_name(agent_messages, "chest_xray_classifier")
    gradcam_output    = _extract_tool_output_by_name(agent_messages, "grad_cam_explainer")
    report_output     = _extract_tool_output_by_name(agent_messages, "chest_xray_report_generator")

    # ------------------------------------------------------------------
    # Pass 3: Reconciliation — may revise any field
    # ------------------------------------------------------------------
    refl_resp = client.chat.completions.create(
        model=client._model,
        messages=[
            {"role": "system", "content": CLINICAL_SYSTEM},
            {"role": "user", "content": RECONCILIATION_PROMPT.format(
                initial_json=json.dumps(initial_parsed, indent=2),
                classifier_output=classifier_output,
                gradcam_output=gradcam_output,
                report_output=report_output,
            )},
        ],
        max_completion_tokens=700,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    reflected_parsed = _parse_json(refl_resp.choices[0].message.content)

    return initial_parsed, reflected_parsed, tools_called, agent_messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark",   default="eurorad_clinical_benchmark.jsonl")
    parser.add_argument("--output",      default=None, help="Log file (default: auto-named)")
    parser.add_argument("--model",       default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--use-agent",   action="store_true")
    parser.add_argument("--model-dir",
                        default=os.environ.get("MODEL_DIR",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mydownload")))
    parser.add_argument("--prompt-file", default="medrax/docs/system_prompts.txt")
    parser.add_argument("--max-cases",   type=int, default=None)
    parser.add_argument("--xr-only",     action="store_true",
                        help="Restrict to chest X-ray cases only")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    openai_kwargs = {"api_key": api_key}
    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    judge_api_key = os.getenv("JUDGE_OPENAI_API_KEY", api_key)
    judge_client = openai.OpenAI(
        api_key=judge_api_key,
        base_url="https://api.openai.com/v1",
    )

    gpt_client = openai.OpenAI(**openai_kwargs)
    gpt_client._model = args.model

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

    with open(args.benchmark) as f:
        cases = [json.loads(l) for l in f if l.strip()]

    if args.xr_only:
        cases = [c for c in cases if _is_chest_xray_case(c)]
        print(f"XR-only filter: {len(cases)} chest X-ray cases retained.")

    if args.max_cases:
        cases = cases[:args.max_cases]

    log_file = (
        args.output or
        f"clinical_eval_v2_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    # All three targeted tools count toward recall
    TARGETED_TOOLS = {"chest_xray_classifier", "chest_xray_report_generator", "grad_cam_explainer"}

    def _score_parsed(parsed: dict, case: dict) -> dict:
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

    def _empty_scores():
        return {
            "findings": [], "consistency": [], "diagnosis": [],
            "diff_top1": [], "diff_top3": [], "tool_recall": [],
            "communication": [], "reasoning_coherence": [], "uncertainty_appropriate": [],
            "confidence_raw": [],
        }

    pre_scores  = _empty_scores()   # Pass 1 output (vision only)
    post_scores = _empty_scores()   # Pass 3 output (after tool-grounded reflection)

    reflection_stats = {
        "total": 0, "revised": 0,
        "classifier_contradictions": 0,
        "gradcam_contradictions":    0,
        "report_contradictions":     0,
        "revision_reasons":          [],
    }

    with open(log_file, "w") as log:
        for i, case in enumerate(cases):
            print(f"\n[{i+1}/{len(cases)}] Case {case['case_id']} — {case['title']}")

            initial_parsed, reflected_parsed, tools_called, agent_messages = \
                run_agent_v2(case, agent, args.use_agent, gpt_client)

            if initial_parsed is None:
                print("  ✗ Failed to parse initial response.")
                log.write(json.dumps({"case_id": case["case_id"], "status": "parse_error"}) + "\n")
                continue

            pre = _score_parsed(initial_parsed, case)

            tool_recall = 0.0
            if args.use_agent:
                called = set(tools_called)
                tool_recall = len(called & TARGETED_TOOLS) / len(TARGETED_TOOLS)

            final_parsed = reflected_parsed if reflected_parsed is not None else initial_parsed
            post = _score_parsed(final_parsed, case)

            # Reflection metadata
            reflection_revised  = False
            revision_reasons    = []
            tool_contradictions = {}
            if reflected_parsed is not None:
                reflection_stats["total"] += 1
                reflection_revised  = bool(reflected_parsed.get("revised", False))
                revision_reasons    = reflected_parsed.get("revision_reasons", [])
                tool_contradictions = reflected_parsed.get("tool_contradictions", {})

                if reflection_revised:
                    reflection_stats["revised"] += 1
                    reflection_stats["revision_reasons"].extend(revision_reasons)

                for tool_key, stat_key in [
                    ("classifier",       "classifier_contradictions"),
                    ("gradcam",          "gradcam_contradictions"),
                    ("report_generator", "report_contradictions"),
                ]:
                    if tool_contradictions.get(tool_key) == "contradiction":
                        reflection_stats[stat_key] += 1

            # Accumulate both pre and post
            for scores, p in [(pre_scores, pre), (post_scores, post)]:
                scores["findings"].append(p["findings"])
                scores["consistency"].append(p["consistency"])
                scores["diagnosis"].append(p["diagnosis"])
                scores["diff_top1"].append(int(p["diff_scores"]["in_top1"]))
                scores["diff_top3"].append(int(p["diff_scores"]["in_top3"]))
                scores["tool_recall"].append(tool_recall)
                scores["communication"].append(p["communication"])
                scores["reasoning_coherence"].append(p["reasoning_coherence"])
                scores["uncertainty_appropriate"].append(p["uncertainty_appropriate"])
                scores["confidence_raw"].append(p["confidence_raw"])

            entry = {
                "case_id":      case["case_id"],
                "title":        case["title"],
                "gt_diagnosis": case["gt_final_diagnosis"],
                "tools_called": tools_called,
                "tool_recall":  tool_recall,
                "pre_reflection": {
                    "diagnosis":    initial_parsed.get("final_diagnosis", ""),
                    "differentials": initial_parsed.get("differentials", []),
                    "findings":     initial_parsed.get("imaging_findings", ""),
                    "confidence":   initial_parsed.get("confidence", ""),
                    "scores": {
                        "imaging_findings":        round(pre["findings"], 3),
                        "diagnosis_match":         round(pre["diagnosis"], 3),
                        "consistency":             round(pre["consistency"], 3),
                        "diff_in_top1":            pre["diff_scores"]["in_top1"],
                        "diff_in_top3":            pre["diff_scores"]["in_top3"],
                        "uncertainty_appropriate": round(pre["uncertainty_appropriate"], 3),
                        "confidence":              pre["confidence_raw"],
                    },
                },
                "post_reflection": {
                    "diagnosis":           final_parsed.get("final_diagnosis", ""),
                    "differentials":       final_parsed.get("differentials", []),
                    "findings":            final_parsed.get("imaging_findings", ""),
                    "confidence":          final_parsed.get("confidence", ""),
                    "revised":             reflection_revised,
                    "revision_reasons":    revision_reasons,
                    "tool_contradictions": tool_contradictions,
                    "scores": {
                        "imaging_findings":        round(post["findings"], 3),
                        "diagnosis_match":         round(post["diagnosis"], 3),
                        "consistency":             round(post["consistency"], 3),
                        "diff_in_top1":            post["diff_scores"]["in_top1"],
                        "diff_in_top3":            post["diff_scores"]["in_top3"],
                        "tool_recall":             round(tool_recall, 3),
                        "communication_quality":   round(post["communication"], 3),
                        "reasoning_coherence":     round(post["reasoning_coherence"], 3),
                        "uncertainty_appropriate": round(post["uncertainty_appropriate"], 3),
                        "confidence":              post["confidence_raw"],
                    },
                },
            }
            log.write(json.dumps(entry) + "\n")

            print(f"  [pre]  Findings: {pre['findings']:.2f} | Dx: {pre['diagnosis']:.2f} | "
                  f"Consist: {pre['consistency']:.2f} | conf={pre['confidence_raw']}")
            if args.use_agent and reflected_parsed is not None:
                marker = (f"REVISED ({len(revision_reasons)} changes)"
                          if reflection_revised else "no change")
                print(f"  [post] Findings: {post['findings']:.2f} | Dx: {post['diagnosis']:.2f} | "
                      f"Consist: {post['consistency']:.2f} | conf={post['confidence_raw']} | {marker}")
                if tool_contradictions:
                    print(f"  Tool agreement — "
                          f"A(classifier):{tool_contradictions.get('classifier','?')}  "
                          f"B(gradcam):{tool_contradictions.get('gradcam','?')}  "
                          f"C(report):{tool_contradictions.get('report_generator','?')}")
            print(f"  Diff top-1: {post['diff_scores']['in_top1']}  "
                  f"top-3: {post['diff_scores']['in_top3']}  "
                  f"rank: {post['diff_scores']['rank']}")
            if args.use_agent:
                print(f"  Tools called: {tools_called}  recall: {tool_recall:.0%}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0
    def delta(post, pre): return f"{post - pre:+.3f}"

    n      = len(pre_scores["findings"])
    scores = post_scores

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
    pre_rcw = avg([
        avg(pre_scores["communication"]),
        avg(pre_scores["reasoning_coherence"]),
        avg(pre_scores["uncertainty_appropriate"]),
    ])

    W = 62
    print(f"\n{'='*W}")
    print(f"Clinical Reasoning Evaluation v2 — Tool-Grounded Reflection — {n} cases")
    print(f"{'='*W}")
    print(f"{'Metric':<35} {'Pre-refl':>8} {'Post-refl':>9} {'Delta':>7}")
    print(f"{'-'*W}")
    print(f"{'Imaging Findings Score':<35} "
          f"{avg(pre_scores['findings']):8.3f} {avg(scores['findings']):9.3f} "
          f"{delta(avg(scores['findings']), avg(pre_scores['findings'])):>7}")
    print(f"{'Differential Recall@1':<35} "
          f"{avg(pre_scores['diff_top1']):8.1%} {avg(scores['diff_top1']):9.1%} "
          f"{delta(avg(scores['diff_top1']), avg(pre_scores['diff_top1'])):>7}")
    print(f"{'Differential Recall@3':<35} "
          f"{avg(pre_scores['diff_top3']):8.1%} {avg(scores['diff_top3']):9.1%} "
          f"{delta(avg(scores['diff_top3']), avg(pre_scores['diff_top3'])):>7}")
    print(f"{'Final Diagnosis Score':<35} "
          f"{avg(pre_scores['diagnosis']):8.3f} {avg(scores['diagnosis']):9.3f} "
          f"{delta(avg(scores['diagnosis']), avg(pre_scores['diagnosis'])):>7}")
    print(f"{'Consistency Score':<35} "
          f"{avg(pre_scores['consistency']):8.3f} {avg(scores['consistency']):9.3f} "
          f"{delta(avg(scores['consistency']), avg(pre_scores['consistency'])):>7}")
    print(f"{'Uncertainty Appropriateness':<35} "
          f"{avg(pre_scores['uncertainty_appropriate']):8.3f} "
          f"{avg(scores['uncertainty_appropriate']):9.3f} "
          f"{delta(avg(scores['uncertainty_appropriate']), avg(pre_scores['uncertainty_appropriate'])):>7}")
    if args.use_agent:
        print(f"{'Targeted Tool Recall (A+B+C)':<35} {avg(scores['tool_recall']):8.1%}")
    print(f"{'-'*W}")
    print(f"{'Reliable Co-worker Score':<35} "
          f"{pre_rcw:8.3f} {rcw_score:9.3f} {delta(rcw_score, pre_rcw):>7}")
    print(f"  Communication Quality:      "
          f"{avg(pre_scores['communication']):8.3f} {avg(scores['communication']):9.3f}")
    print(f"  Reasoning Coherence:        "
          f"{avg(pre_scores['reasoning_coherence']):8.3f} {avg(scores['reasoning_coherence']):9.3f}")
    print(f"{'-'*W}")
    print(f"Confidence Calibration (post-reflection)")
    for level in ("high", "medium", "low"):
        bucket = conf_buckets[level]
        if bucket:
            print(f"  {level:6s}: {len(bucket):2d} cases  avg diagnosis = {avg(bucket):.3f}")
        else:
            print(f"  {level:6s}:  0 cases")
    if args.use_agent and reflection_stats["total"] > 0:
        rev_rate = reflection_stats["revised"] / reflection_stats["total"]
        print(f"{'-'*W}")
        print(f"Tool-Grounded Reflection Summary:")
        print(f"  Cases with reflection:          {reflection_stats['total']}")
        print(f"  Cases revised after reflection: {reflection_stats['revised']} ({rev_rate:.1%})")
        print(f"  Pattern A contradictions (classifier):  {reflection_stats['classifier_contradictions']}")
        print(f"  Pattern B contradictions (gradcam):     {reflection_stats['gradcam_contradictions']}")
        print(f"  Pattern C contradictions (report gen):  {reflection_stats['report_contradictions']}")
    print(f"{'='*W}")
    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
