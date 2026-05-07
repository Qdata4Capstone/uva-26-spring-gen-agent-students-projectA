"""
experiments.py — red-team experiment runner and ABATE baseline in one file.

Functions:
  run_experiment           — adaptive red-team experiment (MARSE)
  summarize_experiment     — one-line human-readable summary
  load_experiment_log      — load a saved red-team log from disk
  run_baseline_experiment  — static probe bank experiment (ABATE)
  load_baseline_log        — load a saved baseline log from disk
"""

import json
import os
import random
import re
import time
from datetime import datetime

import config
from agents import get_input_safety_stats, AlignmentBlockedError
from backends import build_target_agent, build_red_team_agent
from red_team import LLMJudge, PROBE_BANK


def _extract_latest_blocked_details(target_agent):
    memory = getattr(target_agent, "memory", None)
    if memory is None or not hasattr(memory, "all"):
        return None, None
    entries = memory.all()
    for entry in reversed(entries):
        content = str(entry.get("content", ""))
        hit = re.search(r"\[blocked_rule=([^;\]]+);matched=([^\]]+)\]", content)
        if hit:
            return hit.group(1), hit.group(2)
    return None, None


def _counter_diff(after_counts, before_counts):
    delta = {}
    keys = set(after_counts) | set(before_counts)
    for key in keys:
        diff = int(after_counts.get(key, 0)) - int(before_counts.get(key, 0))
        if diff > 0:
            delta[key] = diff
    return delta


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) <= 1:
        return 0.0
    mean_val = _mean(values)
    var = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
    return var ** 0.5


def _bootstrap_ci(values, num_samples=500, confidence=0.95, seed=None):
    if not values:
        return {"low": 0.0, "high": 0.0}
    if len(values) == 1:
        val = float(values[0])
        return {"low": val, "high": val}

    confidence = max(0.01, min(0.999, float(confidence)))
    num_samples = max(10, int(num_samples))
    rng = random.Random(seed)
    n = len(values)

    sample_means = []
    for _ in range(num_samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        sample_means.append(_mean(sample))
    sample_means.sort()

    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, min(len(sample_means) - 1, int(alpha * (len(sample_means) - 1))))
    high_idx = max(0, min(len(sample_means) - 1, int((1.0 - alpha) * (len(sample_means) - 1))))
    return {"low": sample_means[low_idx], "high": sample_means[high_idx]}


def _empty_category_bucket(category):
    return {
        "category": category,
        "n_probes": 0,
        "n_violations": 0,
        "violation_rate": 0.0,
        "mean_severity_realized": 0.0,
        "probe_results": [],
    }


def _empty_agent_bucket(agent_name, categories):
    return {
        "agent_name": agent_name,
        "total_probes": 0,
        "total_violations": 0,
        "violation_rate": 0.0,
        "by_category": {category: _empty_category_bucket(category) for category in categories},
        "probe_log": [],
    }


def _date_time_id(prefix):
    # Include microseconds to avoid collisions in rapid back-to-back runs.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{stamp}"


def _primary_violation_detected(score):
    if not isinstance(score, dict):
        return False
    if "reporting_violation_detected" in score:
        return bool(score.get("reporting_violation_detected"))
    return bool(score.get("violation_detected", False))


def _primary_severity(score):
    if not isinstance(score, dict):
        return 0.0
    raw = score.get("reporting_severity", score.get("severity", 0.0))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Red-team experiment (MARSE)
# ---------------------------------------------------------------------------

def run_experiment(
    target_agent_name=config.TARGET_AGENT,
    n_turns=config.MAX_TURNS,
    stop_on_violation=config.VIOLATION_STOPS_EXPERIMENT,
    output_dir=config.EXPERIMENT_OUTPUT_DIR,
    log_filename=config.EXPERIMENT_LOG_FILENAME,
    seed=config.EXPERIMENT_SEED,
    prior_bandit_state=None,
    red_team_domain=None,
    fail_on_runtime_errors=False,
):
    config.validate_runtime_config()

    if seed is not None:
        random.seed(seed)

    pre_safety_stats = get_input_safety_stats()
    experiment_id = _date_time_id("exp")

    target_agent = build_target_agent(target_agent_name, config.TARGET_LLM)
    red_team_agent = build_red_team_agent(
        target_agent,
        config.RED_TEAM_LLM,
        attacker_domain=red_team_domain,
        campaign_id=experiment_id,
        evaluator_seed=seed,
    )
    if prior_bandit_state:
        red_team_agent.load_bandit_state(prior_bandit_state)
    os.makedirs(output_dir, exist_ok=True)

    log_dict = {
        "experiment_id": experiment_id,
        "target_agent": target_agent_name,
        "red_team_domain": red_team_domain,
        "decision_signal": "reporting_first",
        "decision_signal_fallback": "rule",
        "n_turn_scores_logged": 0,
        "n_turn_state_updates_applied": 0,
        "state_update_failures": 0,
        "n_turns_attempted": 0,
        "n_plan_failures": 0,
        "n_target_failures": 0,
        "max_turns": n_turns,
        "stop_on_violation": stop_on_violation,
        "fail_on_runtime_errors": bool(fail_on_runtime_errors),
        "seed": seed,
        "config": config.runtime_config_snapshot(),
        "start_time": time.time(),
        "end_time": None,
        "total_turns": 0,
        "violation_occurred": False,
        "violation_turn": None,
        "turns": [],
    }

    for i in range(n_turns):
        log_dict["n_turns_attempted"] += 1
        print(f"[{target_agent_name}] Turn {i + 1}/{n_turns} ...", flush=True)
        try:
            red_team_agent.perceive(f"Turn {i + 1} of {n_turns} targeting {target_agent_name}")
            plan = red_team_agent.plan([])
        except Exception as e:
            log_dict["n_plan_failures"] += 1
            print(f"Turn {i + 1} skipped: red team LLM error — {e}")
            continue

        blocked_rule = None
        blocked_match = None
        try:
            target_result = target_agent.run(plan["attack"])
            target_result["blocked_at_perceive"] = False
            eval_outcome = target_result
        except AlignmentBlockedError:
            blocked_rule, blocked_match = _extract_latest_blocked_details(target_agent)
            target_result = {
                "plan": "blocked",
                "reflection": "",
                "action_result": {},
                "blocked_at_perceive": True,
                "blocked_rule": blocked_rule,
                "blocked_match": blocked_match,
            }
            eval_outcome = "detected"
        except Exception as e:
            log_dict["n_target_failures"] += 1
            print(f"Turn {i + 1} skipped: target LLM error — {e}")
            continue

        alignment_warning = bool(target_result.get("alignment_warning", False))
        score = red_team_agent.evaluator.evaluate(
            plan["attack"],
            eval_outcome,
            surface=plan["surface"],
            alignment_warning=alignment_warning,
        )
        log_dict["n_turn_scores_logged"] += 1

        act_update_failed = False
        act_error = None
        try:
            red_team_agent.act(plan, outcome=eval_outcome, precomputed_score=score)
            log_dict["n_turn_state_updates_applied"] += 1
        except Exception as e:
            log_dict["state_update_failures"] += 1
            act_update_failed = True
            act_error = str(e)
            print(f"Turn {i + 1}: red team act() error — {e}")

        # Reflect every 3 turns so strategic insights feed back into future plan() calls
        if (i + 1) % 3 == 0:
            try:
                recent = red_team_agent.memory.retrieve("attack", k=5)
                red_team_agent.reflect(recent)
            except Exception as e:
                print(f"Turn {i + 1}: reflect() skipped — {e}")

        turn_log = {
            "turn": i + 1,
            "surface": plan["surface"],
            "attack": plan["attack"],
            "target_result_summary": {
                "plan_snippet": target_result.get("plan", "")[:200],
                "alignment_warning": target_result.get("alignment_warning", False),
                "blocked_at_perceive": target_result.get("blocked_at_perceive", False),
                "blocked_rule": target_result.get("blocked_rule"),
                "blocked_match": target_result.get("blocked_match"),
            },
            "score": score,
            "primary_violation_detected": _primary_violation_detected(score),
            "primary_severity": _primary_severity(score),
            "act_update_failed": act_update_failed,
            "act_error": (act_error[:500] if act_error else None),
        }

        log_dict["turns"].append(turn_log)
        log_dict["total_turns"] += 1

        primary_violated = _primary_violation_detected(score)
        violation_flag = "VIOLATION" if primary_violated else "ok"
        blocked_flag = " [blocked]" if target_result.get("blocked_at_perceive") else ""
        print(f"  surface={plan['surface']}  result={violation_flag}{blocked_flag}", flush=True)

        if primary_violated:
            log_dict["violation_occurred"] = True
            if log_dict["violation_turn"] is None:
                log_dict["violation_turn"] = i + 1
            if stop_on_violation:
                break

    log_dict["end_time"] = time.time()
    log_dict["duration_seconds"] = log_dict["end_time"] - log_dict["start_time"]
    log_dict["bandit_final_state"] = red_team_agent.bandit_state
    log_dict["campaign_summary"] = red_team_agent.campaign_state
    log_dict["discovered_surfaces"] = red_team_agent.campaign_state["discovered_surfaces"]

    blocked_rule_counts = {}
    rule_violation_count = 0
    reporting_violation_count = 0
    for turn in log_dict["turns"]:
        rule = turn.get("target_result_summary", {}).get("blocked_rule")
        if rule:
            blocked_rule_counts[rule] = blocked_rule_counts.get(rule, 0) + 1
        score = turn.get("score", {})
        if bool(score.get("violation_detected", False)):
            rule_violation_count += 1
        if _primary_violation_detected(score):
            reporting_violation_count += 1
    log_dict["blocked_rule_counts"] = blocked_rule_counts
    log_dict["rule_violation_count"] = rule_violation_count
    log_dict["reporting_violation_count"] = reporting_violation_count

    runtime_failures = log_dict["n_plan_failures"] + log_dict["n_target_failures"]
    state_update_failures = int(log_dict.get("state_update_failures", 0))
    if log_dict["total_turns"] == 0 and runtime_failures > 0:
        run_status = "failed"
    elif log_dict["total_turns"] > 0 and state_update_failures >= log_dict["total_turns"]:
        # All completed turns failed to update red-team state, so adaptation metrics are unusable.
        run_status = "failed"
    elif runtime_failures > 0 or state_update_failures > 0:
        run_status = "degraded"
    else:
        run_status = "ok"
    log_dict["run_status"] = run_status

    post_safety_stats = get_input_safety_stats()
    log_dict["input_safety_rule_hits_delta"] = _counter_diff(post_safety_stats, pre_safety_stats)

    output_path = os.path.join(output_dir, log_dict["experiment_id"] + "_" + log_filename)
    with open(output_path, "w") as f:
        f.write(json.dumps(log_dict, indent=2))

    print(f"experiment_id:      {log_dict['experiment_id']}")
    print(f"run_status:         {log_dict['run_status']}")
    print(f"total_turns:        {log_dict['total_turns']}")
    print(f"violation_occurred: {log_dict['violation_occurred']}")
    print(f"violation_turn:     {log_dict['violation_turn']}")
    print(f"output:             {output_path}")

    if fail_on_runtime_errors and log_dict["run_status"] == "failed":
        raise RuntimeError("Experiment failed: no completed turns due to runtime errors.")

    return log_dict


def load_experiment_log(filepath):
    with open(filepath) as f:
        return json.loads(f.read())


def summarize_experiment(log_dict):
    surfaces = ", ".join(log_dict.get("discovered_surfaces", [])) or "none"
    duration = round(log_dict.get("duration_seconds", 0), 1)
    violated = log_dict["violation_occurred"]
    violation_str = f"yes (turn {log_dict['violation_turn']})" if violated else "no"
    return (
        f"Experiment: {log_dict['experiment_id']}\n"
        f"Target agent: {log_dict['target_agent']}\n"
        f"Run status: {log_dict.get('run_status', 'unknown')}\n"
        f"Total turns: {log_dict['total_turns']} / {log_dict['max_turns']}\n"
        f"Violation occurred: {violation_str}\n"
        f"Discovered surfaces: {surfaces}\n"
        f"Duration: {duration}s"
    )


# ---------------------------------------------------------------------------
# ABATE baseline experiment
# ---------------------------------------------------------------------------

def run_baseline_experiment(
    agent_names=None,
    n_probes_per_category=config.BASELINE_N_PROBES_PER_CATEGORY,
    output_dir=config.BASELINE_OUTPUT_DIR,
    log_filename=config.BASELINE_LOG_FILENAME,
    judge_backend=config.BASELINE_LLM_JUDGE_BACKEND,
    n_runs=config.BASELINE_N_RUNS,
    seed=config.BASELINE_SEED,
    randomize_probes=config.BASELINE_RANDOMIZE_PROBES,
    bootstrap_samples=config.BASELINE_BOOTSTRAP_SAMPLES,
    bootstrap_confidence=config.BASELINE_BOOTSTRAP_CONFIDENCE,
):
    normalized_judge_backend = str(judge_backend or "").strip().lower()
    if normalized_judge_backend not in {"openai", "stub"}:
        raise ValueError(
            f"Unsupported baseline judge backend: {judge_backend!r}. "
            "Expected one of: 'openai', 'stub'."
        )
    normalized_fallback = str(getattr(config, "BASELINE_LLM_JUDGE_FALLBACK", "") or "").strip().lower()
    require_openai = (
        config.TARGET_LLM == "openai"
        or normalized_judge_backend == "openai"
        or normalized_fallback == "openai"
    )
    config.validate_runtime_config(require_openai=require_openai)

    if agent_names is None:
        agent_names = ["medical", "financial", "customer_service"]

    os.makedirs(output_dir, exist_ok=True)
    judge = LLMJudge(
        backend=normalized_judge_backend,
        fallback_backend=config.BASELINE_LLM_JUDGE_FALLBACK,
        model=config.BASELINE_LLM_JUDGE_MODEL,
    )

    pre_safety_stats = get_input_safety_stats()

    categories = list(config.BASELINE_CATEGORIES)
    n_runs = max(1, int(n_runs))
    total_probes = n_runs * len(agent_names if agent_names else ["medical", "financial", "customer_service"]) * len(categories) * n_probes_per_category

    print(f"[ABATE] Starting baseline experiment")
    print(f"  agents={agent_names or ['medical', 'financial', 'customer_service']}")
    print(f"  n_runs={n_runs}, probes/category={n_probes_per_category}, categories={len(categories)}")
    print(f"  judge={normalized_judge_backend}, target={config.TARGET_LLM}, seed={seed}")
    print(f"  total judge calls: ~{total_probes}")

    master_log = {
        "experiment_id": _date_time_id("baseline"),
        "start_time": time.time(),
        "agent_names": agent_names,
        "categories": categories,
        "n_probes_per_category": n_probes_per_category,
        "n_runs": n_runs,
        "seed": seed,
        "randomize_probes": bool(randomize_probes),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_confidence": float(bootstrap_confidence),
        "config": config.runtime_config_snapshot(),
        "run_seeds": [],
        "runs": [],
        "results": {agent_name: _empty_agent_bucket(agent_name, categories) for agent_name in agent_names},
        "summary": {},
    }

    per_agent_run_rates = {agent_name: [] for agent_name in agent_names}
    per_category_run_rates = {
        category: {agent_name: [] for agent_name in agent_names}
        for category in categories
    }

    for run_idx in range(n_runs):
        run_seed = None if seed is None else int(seed) + run_idx * int(config.BASELINE_SEED_STRIDE)
        rng = random.Random(run_seed)
        master_log["run_seeds"].append(run_seed)
        run_log = {
            "run_index": run_idx + 1,
            "seed": run_seed,
            "results": {},
        }

        print(f"\n[run {run_idx + 1}/{n_runs}] seed={run_seed}")

        for agent_name in agent_names:
            target_agent = build_target_agent(agent_name, config.TARGET_LLM)
            run_bucket = _empty_agent_bucket(agent_name, categories)

            print(f"  [{agent_name}]")

            for category in categories:
                probe_pool = list(PROBE_BANK[category])
                if randomize_probes:
                    rng.shuffle(probe_pool)
                selected_probes = probe_pool[:n_probes_per_category]

                cat_bucket = _empty_category_bucket(category)
                cat_bucket["n_probes"] = len(selected_probes)

                violations_this_cat = 0
                for probe in selected_probes:
                    blocked_at_perceive = False
                    blocked_rule = None
                    blocked_match = None
                    try:
                        result = target_agent.run(probe["prompt"])
                        agent_response_string = result.get("plan", "")
                    except AlignmentBlockedError:
                        agent_response_string = "BLOCKED_AT_PERCEIVE"
                        blocked_at_perceive = True
                        blocked_rule, blocked_match = _extract_latest_blocked_details(target_agent)

                    if blocked_at_perceive:
                        verdict = {
                            "aligned": True,
                            "confidence": 1.0,
                            "violation_type": None,
                            "reasoning": "Input blocked at perceive layer — correct alignment behavior.",
                            "severity_realized": 0.0,
                        }
                    else:
                        verdict = judge.judge(probe, agent_response_string)

                    probe_result = {
                        "run_index": run_idx + 1,
                        "run_seed": run_seed,
                        "probe_id": probe["id"],
                        "prompt": probe["prompt"][:120],
                        "agent_response": agent_response_string[:300],
                        "verdict": verdict,
                        "blocked_at_perceive": blocked_at_perceive,
                        "blocked_rule": blocked_rule,
                        "blocked_match": blocked_match,
                    }

                    cat_bucket["probe_results"].append(probe_result)
                    run_bucket["probe_log"].append(probe_result)
                    run_bucket["total_probes"] += 1

                    if not verdict["aligned"]:
                        run_bucket["total_violations"] += 1
                        cat_bucket["n_violations"] += 1
                        violations_this_cat += 1

                    status = "BLOCKED" if blocked_at_perceive else ("VIOLATION" if not verdict["aligned"] else "ok")
                    print(f"    {probe['id']:40s} {status}")

                print(f"    -> {category}: {violations_this_cat}/{len(selected_probes)} violations")

                cat_bucket["violation_rate"] = (
                    cat_bucket["n_violations"] / cat_bucket["n_probes"]
                    if cat_bucket["n_probes"] > 0
                    else 0.0
                )
                severity_values = [pr["verdict"]["severity_realized"] for pr in cat_bucket["probe_results"]]
                cat_bucket["mean_severity_realized"] = (
                    sum(severity_values) / len(severity_values)
                    if severity_values
                    else 0.0
                )
                run_bucket["by_category"][category] = cat_bucket

            run_bucket["violation_rate"] = (
                run_bucket["total_violations"] / run_bucket["total_probes"]
                if run_bucket["total_probes"] > 0
                else 0.0
            )
            print(f"  [{agent_name}] done — {run_bucket['total_violations']}/{run_bucket['total_probes']} violations ({run_bucket['violation_rate']:.0%})")
            run_log["results"][agent_name] = run_bucket
            per_agent_run_rates[agent_name].append(run_bucket["violation_rate"])

            for category in categories:
                per_category_run_rates[category][agent_name].append(
                    run_bucket["by_category"][category]["violation_rate"]
                )

            agg_bucket = master_log["results"][agent_name]
            agg_bucket["total_probes"] += run_bucket["total_probes"]
            agg_bucket["total_violations"] += run_bucket["total_violations"]
            agg_bucket["probe_log"].extend(run_bucket["probe_log"])
            for category in categories:
                run_cat = run_bucket["by_category"][category]
                agg_cat = agg_bucket["by_category"][category]
                agg_cat["n_probes"] += run_cat["n_probes"]
                agg_cat["n_violations"] += run_cat["n_violations"]
                agg_cat["probe_results"].extend(run_cat["probe_results"])

        master_log["runs"].append(run_log)

    for agent_name in agent_names:
        bucket = master_log["results"][agent_name]
        bucket["violation_rate"] = (
            bucket["total_violations"] / bucket["total_probes"]
            if bucket["total_probes"] > 0
            else 0.0
        )
        for category in categories:
            cat_bucket = bucket["by_category"][category]
            cat_bucket["violation_rate"] = (
                cat_bucket["n_violations"] / cat_bucket["n_probes"]
                if cat_bucket["n_probes"] > 0
                else 0.0
            )
            severity_values = [pr["verdict"]["severity_realized"] for pr in cat_bucket["probe_results"]]
            cat_bucket["mean_severity_realized"] = (
                sum(severity_values) / len(severity_values)
                if severity_values
                else 0.0
            )

    overall_violation_rates = {
        name: _mean(per_agent_run_rates[name]) for name in agent_names
    }
    overall_violation_std = {
        name: _std(per_agent_run_rates[name]) for name in agent_names
    }
    overall_violation_ci95 = {
        name: _bootstrap_ci(
            per_agent_run_rates[name],
            num_samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=(None if master_log["seed"] is None else int(master_log["seed"]) + 31),
        )
        for name in agent_names
    }

    category_violation_rates = {}
    category_violation_std = {}
    category_violation_ci95 = {}
    for category in categories:
        category_violation_rates[category] = {}
        category_violation_std[category] = {}
        category_violation_ci95[category] = {}
        for name in agent_names:
            vals = per_category_run_rates[category][name]
            category_violation_rates[category][name] = _mean(vals)
            category_violation_std[category][name] = _std(vals)
            category_violation_ci95[category][name] = _bootstrap_ci(
                vals,
                num_samples=bootstrap_samples,
                confidence=bootstrap_confidence,
                seed=(None if master_log["seed"] is None else int(master_log["seed"]) + 79),
            )

    most_vulnerable_agent = max(overall_violation_rates, key=overall_violation_rates.get)
    category_mean_rates = {
        category: _mean(list(category_violation_rates[category].values()))
        for category in categories
    }
    most_breached_category = max(category_mean_rates, key=category_mean_rates.get)
    least_breached_category = min(category_mean_rates, key=category_mean_rates.get)

    blocked_rule_counts = {}
    for agent_name in agent_names:
        for probe_result in master_log["results"][agent_name]["probe_log"]:
            rule = probe_result.get("blocked_rule")
            if rule:
                blocked_rule_counts[rule] = blocked_rule_counts.get(rule, 0) + 1

    post_safety_stats = get_input_safety_stats()
    input_safety_rule_hits_delta = _counter_diff(post_safety_stats, pre_safety_stats)

    master_log["summary"] = {
        "overall_violation_rates": overall_violation_rates,
        "overall_violation_rates_std": overall_violation_std,
        "overall_violation_rates_ci95": overall_violation_ci95,
        "category_violation_rates": category_violation_rates,
        "category_violation_rates_std": category_violation_std,
        "category_violation_rates_ci95": category_violation_ci95,
        "most_vulnerable_agent": most_vulnerable_agent,
        "most_breached_category": most_breached_category,
        "least_breached_category": least_breached_category,
        "blocked_rule_counts": blocked_rule_counts,
        "input_safety_rule_hits_delta": input_safety_rule_hits_delta,
    }

    end_time = time.time()
    master_log["end_time"] = end_time
    master_log["duration_seconds"] = end_time - master_log["start_time"]

    output_path = os.path.join(output_dir, master_log["experiment_id"] + "_" + log_filename)
    with open(output_path, "w") as f:
        f.write(json.dumps(master_log, indent=2))

    print(f"\n[ABATE] Complete — {master_log['duration_seconds']:.1f}s")
    print(f"  experiment_id: {master_log['experiment_id']}")
    s = master_log["summary"]
    for agent_name in agent_names:
        rate = s["overall_violation_rates"].get(agent_name, 0.0)
        ci = s["overall_violation_rates_ci95"].get(agent_name, {})
        print(f"  {agent_name:20s}: {rate:.0%}  95% CI [{ci.get('low', 0):.0%}, {ci.get('high', 0):.0%}]")
    print(f"  most vulnerable: {s['most_vulnerable_agent']}")
    print(f"  most breached category: {s['most_breached_category']}")
    print(f"  output: {output_path}")

    return master_log


def load_baseline_log(filepath):
    with open(filepath) as f:
        return json.loads(f.read())
