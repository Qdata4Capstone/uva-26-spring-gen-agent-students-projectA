"""
eval_clinical_v3.py
--------------------
v1 pipeline + one self-verification pass for agent mode.

Phase 1 (agent only) — Tool gathering:
    Agent calls imaging tools freely, same as eval_clinical.py.

Phase 2 (both modes) — Synthesis:
    Model produces an initial clinical JSON from images + history
    + tool outputs (if agent). Identical to eval_clinical.py.
    This output is recorded as pre_reflection.

Phase 3 (agent only) — Self-verification:
    Model receives its Phase 2 JSON + the tool outputs (no images).
    It checks for contradictions between its stated findings/diagnosis
    and the tool evidence, then either confirms or revises.
    This output is recorded as post_reflection.

Non-agent mode: Phase 2 only (no tools, no reflection). Same as v1.

Usage:
    python eval_clinical_v3.py \\
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

# Phase 2: synthesis (same as eval_clinical.py CLINICAL_USER)
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

# Phase 1: tool gathering (same as eval_clinical.py TOOL_GATHER_PROMPT)
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

# Phase 3: self-verification (text-only, no images)
SELF_VERIFY_PROMPT = """\
You produced the following clinical assessment:
{initial_json}

The following specialized imaging tools were run on this case:

[Classifier Output]
{classifier_output}

[Report Generator Output]
{report_output}

Review your assessment against the tool outputs and check:
1. Does the classifier's top prediction align with your final diagnosis or support \
a competing differential?
2. Does the report generator's findings agree with your stated imaging findings? \
Are any significant findings mentioned by the report absent from yours, or vice versa?
3. Is your stated confidence level appropriate given the agreement or disagreement \
between the tools and your assessment?

Revision rules:
- Only revise a field if there is clear, specific evidence from the tools that \
contradicts it. Do not change a correct assessment just because tools are uncertain.
- If the classifier's top label differs from your diagnosis but neither is \
definitively supported, note it in reasoning but keep your diagnosis.
- Set "revised": true only if at least one field (diagnosis, findings, confidence, \
differentials, or reasoning) actually changed.

Respond ONLY with this JSON (no extra text):
{{
  "imaging_findings": "updated or unchanged description",
  "differentials": ["Most likely Dx", "2nd", "3rd", "4th", "5th"],
  "final_diagnosis": "single primary diagnosis",
  "confidence": "high | medium | low",
  "reasoning": "one sentence linking findings to diagnosis",
  "revised": false,
  "revision_reasons": [],
  "tool_agreement": {{
    "classifier": "agreement | partial | contradiction | not_applicable",
    "report_generator": "agreement | partial | contradiction"
  }}
}}
"""

# ---------------------------------------------------------------------------
# Judge helpers  (identical to eval_clinical.py)
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
    return gpt_judge(client, (
        f"Ground truth imaging findings:\n{gt_findings}\n\n"
        f"Agent imaging findings:\n{agent_findings}\n\n"
        "Score 0.0-1.0: how well does the agent's description capture the key "
        "radiological findings in the ground truth? "
        "1.0 = all key findings present, 0.0 = completely wrong."
    ))


def score_consistency(client, findings: str, diagnosis: str) -> float:
    return gpt_judge(client, (
        f"Imaging findings stated by agent:\n{findings}\n\n"
        f"Final diagnosis stated by agent:\n{diagnosis}\n\n"
        "Score 0.0-1.0: do the stated findings logically support the stated diagnosis? "
        "1.0 = fully consistent, 0.0 = contradictory or unrelated."
    ))


def score_diagnosis_match(client, agent_dx: str, gt_dx: str) -> float:
    return gpt_judge(client, (
        f"Ground truth diagnosis: {gt_dx}\n"
        f"Agent diagnosis:        {agent_dx}\n\n"
        "Score 0.0-1.0: are these semantically equivalent medical diagnoses? "
        "1.0 = same diagnosis (minor wording differences ok), "
        "0.5 = same disease family but less specific, "
        "0.0 = wrong diagnosis."
    ))


def score_communication_quality(client, findings: str, differentials: list, reasoning: str) -> float:
    diff_str = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(differentials))
    return gpt_judge(client, (
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
    ))


def score_reasoning_coherence(client, history: str, findings: str,
                               differentials: list, diagnosis: str, reasoning: str) -> float:
    diff_str = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(differentials))
    return gpt_judge(client, (
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
    ))


def score_uncertainty_appropriateness(confidence: str, diagnosis_score: float) -> float:
    conf_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
    conf = conf_map.get(str(confidence).lower().strip(), 0.5)
    correct = diagnosis_score >= 0.5
    if correct and conf == 1.0:    return 1.0
    if correct and conf == 0.5:    return 0.8
    if correct and conf == 0.0:    return 0.5
    if not correct and conf == 0.0: return 1.0
    if not correct and conf == 0.5: return 0.4
    if not correct and conf == 1.0: return 0.0
    return 0.5


def differential_recall(agent_diffs: list, gt_dx: str, client: openai.OpenAI) -> dict:
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

def load_images_as_b64(image_paths: list) -> list:
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
            if any(m in content.lower() for m in _TOOL_FAILURE_MARKERS) and len(content) < 300:
                continue
            parts.append(f"[{msg.name}]\n{content}")
    return "\n\n".join(parts) if parts else ""


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
    "plain radiograph", "chest ap", "posteroanterior", "portable chest",
}


def _is_chest_xray_case(case: dict) -> bool:
    text = " ".join([
        case.get("title", ""),
        *case.get("image_descriptions", []),
    ]).lower()
    return any(kw in text for kw in _XR_KEYWORDS)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_agent_v3(
    case: dict, agent, use_agent: bool, client: openai.OpenAI
) -> tuple:
    """
    Phase 1 (agent only): tool gathering — agent calls imaging tools.
    Phase 2 (both):       synthesis     — model produces initial JSON (pre_reflection).
    Phase 3 (agent only): self-verify   — model checks its output vs tool evidence,
                                          revises if warranted (post_reflection).

    Returns (initial_parsed, reflected_parsed, tools_called, agent_messages).
    reflected_parsed is None in non-agent mode.
    """
    from langchain_core.messages import ToolMessage

    image_paths    = case.get("images", []) or []
    history        = case.get("clinical_history", "")
    n_images       = len(image_paths)
    tools_called   = []
    agent_messages = []
    tool_outputs   = ""

    # ------------------------------------------------------------------
    # Phase 1: tool gathering (agent only)
    # ------------------------------------------------------------------
    if use_agent:
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
    # Phase 2: synthesis (both modes) — identical to eval_clinical.py
    # ------------------------------------------------------------------
    tool_section = f"\n\nSpecialized imaging tool outputs:\n{tool_outputs}" if tool_outputs else ""
    content = [{"type": "text",
                "text": CLINICAL_USER.format(history=history, n_images=n_images) + tool_section}]
    for b64 in load_images_as_b64(image_paths):
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    def _synthesis_call(include_tools: bool):
        ts = f"\n\nSpecialized imaging tool outputs:\n{tool_outputs}" if (tool_outputs and include_tools) else ""
        c = [{"type": "text",
              "text": CLINICAL_USER.format(history=history, n_images=n_images) + ts}]
        for b64 in load_images_as_b64(image_paths):
            c.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return client.chat.completions.create(
            model=client._model,
            messages=[
                {"role": "system", "content": CLINICAL_SYSTEM},
                {"role": "user",   "content": c},
            ],
            max_completion_tokens=800,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

    try:
        resp = _synthesis_call(include_tools=True)
    except openai.BadRequestError as e:
        if "context length" in str(e).lower() or "maximum context" in str(e).lower():
            print("  Context too long with tool outputs — retrying without tools.")
            try:
                resp = _synthesis_call(include_tools=False)
                tool_outputs = ""  # clear so Phase 3 knows tools weren't usable
            except openai.BadRequestError as e2:
                print(f"  Context still too long — skipping case. ({e2})")
                return None, None, tools_called, agent_messages
        else:
            raise

    initial_parsed = _parse_json(resp.choices[0].message.content)

    if not use_agent or initial_parsed is None:
        return initial_parsed, None, tools_called, agent_messages

    # ------------------------------------------------------------------
    # Phase 3: self-verification (agent only, text-only — no images)
    # ------------------------------------------------------------------
    classifier_output = _extract_tool_output_by_name(agent_messages, "chest_xray_classifier")
    report_output     = _extract_tool_output_by_name(agent_messages, "chest_xray_report_generator")

    verify_resp = client.chat.completions.create(
        model=client._model,
        messages=[
            {"role": "system", "content": CLINICAL_SYSTEM},
            {"role": "user", "content": SELF_VERIFY_PROMPT.format(
                initial_json=json.dumps(initial_parsed, indent=2),
                classifier_output=classifier_output,
                report_output=report_output,
            )},
        ],
        max_completion_tokens=700,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    reflected_parsed = _parse_json(verify_resp.choices[0].message.content)

    return initial_parsed, reflected_parsed, tools_called, agent_messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark",   default="eurorad_clinical_benchmark.jsonl")
    parser.add_argument("--output",      default=None)
    parser.add_argument("--model",       default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--use-agent",   action="store_true")
    parser.add_argument("--model-dir",
                        default=os.environ.get("MODEL_DIR",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mydownload")))
    parser.add_argument("--prompt-file", default="medrax/docs/system_prompts.txt")
    parser.add_argument("--max-cases",   type=int, default=None)
    parser.add_argument("--xr-only",     action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    openai_kwargs = {"api_key": api_key}
    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    judge_api_key = os.getenv("JUDGE_OPENAI_API_KEY", api_key)
    judge_client  = openai.OpenAI(api_key=judge_api_key, base_url="https://api.openai.com/v1")

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
        f"clinical_eval_v3_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    IMAGING_TOOLS = {"chest_xray_classifier", "chest_xray_report_generator", "grad_cam_explainer"}

    def _score(parsed: dict, case: dict) -> dict:
        s_findings  = score_findings(judge_client, parsed.get("imaging_findings", ""), case["gt_imaging_findings"])
        diff_scores = differential_recall(parsed.get("differentials", []), case["gt_final_diagnosis"], judge_client)
        s_diagnosis = score_diagnosis_match(judge_client, parsed.get("final_diagnosis", ""), case["gt_final_diagnosis"])
        s_consist   = score_consistency(judge_client, parsed.get("imaging_findings", ""), parsed.get("final_diagnosis", ""))
        s_comm      = score_communication_quality(judge_client, parsed.get("imaging_findings", ""),
                                                  parsed.get("differentials", []), parsed.get("reasoning", ""))
        s_reason    = score_reasoning_coherence(judge_client, case["clinical_history"],
                                                parsed.get("imaging_findings", ""), parsed.get("differentials", []),
                                                parsed.get("final_diagnosis", ""), parsed.get("reasoning", ""))
        conf        = parsed.get("confidence", "medium")
        s_uncert    = score_uncertainty_appropriateness(conf, s_diagnosis)
        return {
            "findings": s_findings, "diagnosis": s_diagnosis, "consistency": s_consist,
            "diff_scores": diff_scores, "communication": s_comm,
            "reasoning_coherence": s_reason, "uncertainty_appropriate": s_uncert,
            "confidence_raw": conf,
        }

    def _empty():
        return {"findings": [], "consistency": [], "diagnosis": [], "diff_top1": [], "diff_top3": [],
                "tool_recall": [], "communication": [], "reasoning_coherence": [],
                "uncertainty_appropriate": [], "confidence_raw": []}

    def _append(scores: dict, s: dict, tool_recall: float):
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

    pre_scores  = _empty()
    post_scores = _empty()

    reflection_stats = {
        "total": 0, "revised": 0,
        "helpful_revisions": 0, "harmful_revisions": 0, "neutral_revisions": 0,
        "classifier_agreement": {"agreement": 0, "partial": 0, "contradiction": 0, "not_applicable": 0},
        "report_agreement":     {"agreement": 0, "partial": 0, "contradiction": 0},
    }

    with open(log_file, "w") as log:
        for i, case in enumerate(cases):
            print(f"\n[{i+1}/{len(cases)}] Case {case['case_id']} — {case['title']}")

            initial_parsed, reflected_parsed, tools_called, agent_messages = \
                run_agent_v3(case, agent, args.use_agent, gpt_client)

            if initial_parsed is None:
                print("  ✗ Failed to parse initial response.")
                log.write(json.dumps({"case_id": case["case_id"], "status": "parse_error"}) + "\n")
                continue

            pre         = _score(initial_parsed, case)
            tool_recall = len({t for t in tools_called} & IMAGING_TOOLS) / len(IMAGING_TOOLS) \
                          if args.use_agent else 0.0

            final_parsed = reflected_parsed if reflected_parsed is not None else initial_parsed
            post         = _score(final_parsed, case)

            reflection_revised = False
            revision_reasons   = []
            tool_agreement     = {}

            if args.use_agent and reflected_parsed is not None:
                reflection_stats["total"] += 1
                reflection_revised = bool(reflected_parsed.get("revised", False))
                revision_reasons   = reflected_parsed.get("revision_reasons", [])
                tool_agreement     = reflected_parsed.get("tool_agreement", {})

                if reflection_revised:
                    reflection_stats["revised"] += 1
                    pre_correct  = pre["diagnosis"] >= 0.5
                    post_correct = post["diagnosis"] >= 0.5
                    if (not pre_correct) and post_correct:
                        reflection_stats["helpful_revisions"] += 1
                    elif pre_correct and (not post_correct):
                        reflection_stats["harmful_revisions"] += 1
                    else:
                        reflection_stats["neutral_revisions"] += 1

                for key, stat_dict in [("classifier", reflection_stats["classifier_agreement"]),
                                       ("report_generator", reflection_stats["report_agreement"])]:
                    verdict = tool_agreement.get(key, "")
                    if verdict in stat_dict:
                        stat_dict[verdict] += 1

            _append(pre_scores,  pre,  tool_recall)
            _append(post_scores, post, tool_recall)

            entry = {
                "case_id":  case["case_id"],
                "title":    case["title"],
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
                    "diagnosis":    final_parsed.get("final_diagnosis", ""),
                    "differentials": final_parsed.get("differentials", []),
                    "findings":     final_parsed.get("imaging_findings", ""),
                    "confidence":   final_parsed.get("confidence", ""),
                    "revised":      reflection_revised,
                    "revision_reasons": revision_reasons,
                    "tool_agreement":   tool_agreement,
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
                marker = (f"REVISED ({len(revision_reasons)} reasons)"
                          if reflection_revised else "no change")
                print(f"  [post] Findings: {post['findings']:.2f} | Dx: {post['diagnosis']:.2f} | "
                      f"Consist: {post['consistency']:.2f} | conf={post['confidence_raw']} | {marker}")
                if tool_agreement:
                    print(f"  Tool agreement — "
                          f"classifier:{tool_agreement.get('classifier','?')}  "
                          f"report:{tool_agreement.get('report_generator','?')}")
            print(f"  Diff top-1: {post['diff_scores']['in_top1']}  "
                  f"top-3: {post['diff_scores']['in_top3']}  "
                  f"rank: {post['diff_scores']['rank']}")
            if args.use_agent:
                print(f"  Tools called: {tools_called}  recall: {tool_recall:.0%}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0
    def delta(post, pre): return f"{post - pre:+.3f}"

    n      = len(pre_scores["findings"])
    rcw    = avg([avg(post_scores["communication"]), avg(post_scores["reasoning_coherence"]),
                  avg(post_scores["uncertainty_appropriate"])])
    pre_rcw = avg([avg(pre_scores["communication"]), avg(pre_scores["reasoning_coherence"]),
                   avg(pre_scores["uncertainty_appropriate"])])

    W = 65
    print(f"\n{'='*W}")
    print(f"Clinical Reasoning Evaluation v3 — Self-Verification — {n} cases")
    print(f"{'='*W}")
    print(f"{'Metric':<38} {'Pre':>7} {'Post':>8} {'Delta':>7}")
    print(f"{'-'*W}")
    print(f"{'Imaging Findings Score':<38} {avg(pre_scores['findings']):7.3f} {avg(post_scores['findings']):8.3f} {delta(avg(post_scores['findings']), avg(pre_scores['findings'])):>7}")
    print(f"{'Differential Recall@1':<38} {avg(pre_scores['diff_top1']):7.1%} {avg(post_scores['diff_top1']):8.1%} {delta(avg(post_scores['diff_top1']), avg(pre_scores['diff_top1'])):>7}")
    print(f"{'Differential Recall@3':<38} {avg(pre_scores['diff_top3']):7.1%} {avg(post_scores['diff_top3']):8.1%} {delta(avg(post_scores['diff_top3']), avg(pre_scores['diff_top3'])):>7}")
    print(f"{'Final Diagnosis Score':<38} {avg(pre_scores['diagnosis']):7.3f} {avg(post_scores['diagnosis']):8.3f} {delta(avg(post_scores['diagnosis']), avg(pre_scores['diagnosis'])):>7}")
    print(f"{'Consistency Score':<38} {avg(pre_scores['consistency']):7.3f} {avg(post_scores['consistency']):8.3f} {delta(avg(post_scores['consistency']), avg(pre_scores['consistency'])):>7}")
    print(f"{'Uncertainty Appropriateness':<38} {avg(pre_scores['uncertainty_appropriate']):7.3f} {avg(post_scores['uncertainty_appropriate']):8.3f} {delta(avg(post_scores['uncertainty_appropriate']), avg(pre_scores['uncertainty_appropriate'])):>7}")
    if args.use_agent:
        print(f"{'Tool Recall':<38} {'':>7} {avg(post_scores['tool_recall']):8.1%}")
    print(f"{'-'*W}")
    print(f"{'Reliable Co-worker Score (RCW)':<38} {pre_rcw:7.3f} {rcw:8.3f} {delta(rcw, pre_rcw):>7}")
    print(f"  Communication Quality:          {avg(pre_scores['communication']):7.3f} {avg(post_scores['communication']):8.3f}")
    print(f"  Reasoning Coherence:            {avg(pre_scores['reasoning_coherence']):7.3f} {avg(post_scores['reasoning_coherence']):8.3f}")

    if args.use_agent and reflection_stats["total"] > 0:
        rev_rate = reflection_stats["revised"] / reflection_stats["total"]
        print(f"{'-'*W}")
        print(f"Self-Verification Summary  ({reflection_stats['total']} agent cases)")
        print(f"  Revised:            {reflection_stats['revised']} ({rev_rate:.1%})")
        print(f"  Helpful revisions:  {reflection_stats['helpful_revisions']}")
        print(f"  Harmful revisions:  {reflection_stats['harmful_revisions']}")
        print(f"  Neutral revisions:  {reflection_stats['neutral_revisions']}")
        print(f"  Classifier agreement: {reflection_stats['classifier_agreement']}")
        print(f"  Report agreement:     {reflection_stats['report_agreement']}")

    print(f"{'='*W}")
    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
