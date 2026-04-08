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
import time

import config
from backends import build_target_agent, build_red_team_agent
from red_team import LLMJudge, PROBE_BANK


# ---------------------------------------------------------------------------
# Red-team experiment (MARSE)
# ---------------------------------------------------------------------------

def run_experiment(
    target_agent_name=config.TARGET_AGENT,
    n_turns=config.MAX_TURNS,
    stop_on_violation=config.VIOLATION_STOPS_EXPERIMENT,
    output_dir=config.EXPERIMENT_OUTPUT_DIR,
    log_filename=config.EXPERIMENT_LOG_FILENAME,
):
    target_agent = build_target_agent(target_agent_name, config.TARGET_LLM)
    red_team_agent = build_red_team_agent(target_agent, config.RED_TEAM_LLM)
    os.makedirs(output_dir, exist_ok=True)

    log_dict = {
        "experiment_id": "exp_" + str(int(time.time())),
        "target_agent": target_agent_name,
        "max_turns": n_turns,
        "stop_on_violation": stop_on_violation,
        "start_time": time.time(),
        "end_time": None,
        "total_turns": 0,
        "violation_occurred": False,
        "violation_turn": None,
        "turns": [],
    }

    for i in range(n_turns):
        try:
            red_team_agent.perceive(f"Turn {i + 1} of {n_turns} targeting {target_agent_name}")
            plan = red_team_agent.plan([])
        except Exception as e:
            print(f"Turn {i + 1} skipped: red team LLM error — {e}")
            continue

        try:
            target_result = target_agent.run(plan["attack"])
            target_result["blocked_at_perceive"] = False
            eval_outcome = target_result
        except ValueError:
            target_result = {
                "plan": "blocked",
                "reflection": "",
                "action_result": {},
                "blocked_at_perceive": True,
            }
            eval_outcome = "detected"
        except Exception as e:
            print(f"Turn {i + 1} skipped: target LLM error — {e}")
            continue

        score = red_team_agent.evaluator.evaluate(plan["attack"], eval_outcome)

        try:
            red_team_agent.act(plan)
        except Exception as e:
            print(f"Turn {i + 1}: red team act() error — {e}")

        turn_log = {
            "turn": i + 1,
            "surface": plan["surface"],
            "attack": plan["attack"],
            "target_result_summary": {
                "plan_snippet": target_result.get("plan", "")[:200],
                "alignment_warning": target_result.get("alignment_warning", False),
                "blocked_at_perceive": target_result.get("blocked_at_perceive", False),
            },
            "score": score,
        }

        log_dict["turns"].append(turn_log)
        log_dict["total_turns"] += 1

        if score["violation_detected"]:
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

    output_path = os.path.join(output_dir, log_dict["experiment_id"] + "_" + log_filename)
    with open(output_path, "w") as f:
        f.write(json.dumps(log_dict, indent=2))

    print(f"experiment_id:      {log_dict['experiment_id']}")
    print(f"total_turns:        {log_dict['total_turns']}")
    print(f"violation_occurred: {log_dict['violation_occurred']}")
    print(f"violation_turn:     {log_dict['violation_turn']}")
    print(f"output:             {output_path}")

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
):
    if agent_names is None:
        agent_names = ["medical", "financial", "customer_service"]

    os.makedirs(output_dir, exist_ok=True)
    judge = LLMJudge(backend=judge_backend)

    master_log = {
        "experiment_id": "baseline_" + str(int(time.time())),
        "start_time": time.time(),
        "agent_names": agent_names,
        "categories": config.BASELINE_CATEGORIES,
        "n_probes_per_category": n_probes_per_category,
        "results": {},
        "summary": {},
    }

    for agent_name in agent_names:
        target_agent = build_target_agent(agent_name, config.TARGET_LLM)

        # one bucket per agent, gets merged into master_log at end
        bucket = {
            "agent_name": agent_name,
            "total_probes": 0,
            "total_violations": 0,
            "violation_rate": 0.0,
            "by_category": {},
            "probe_log": [],
        }

        for category in config.BASELINE_CATEGORIES:
            full_probe_list = PROBE_BANK[category]
            selected_probes = full_probe_list[:n_probes_per_category]

            cat_bucket = {
                "category": category,
                "n_probes": len(selected_probes),
                "n_violations": 0,
                "violation_rate": 0.0,
                "mean_severity_realized": 0.0,
                "probe_results": [],
            }

            for probe in selected_probes:
                blocked_at_perceive = False
                try:
                    result = target_agent.run(probe["prompt"])
                    agent_response_string = result.get("plan", "")
                except ValueError:
                    agent_response_string = "BLOCKED_AT_PERCEIVE"
                    blocked_at_perceive = True

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
                    "probe_id": probe["id"],
                    "prompt": probe["prompt"][:120],
                    "agent_response": agent_response_string[:300],
                    "verdict": verdict,
                    "blocked_at_perceive": blocked_at_perceive,
                }

                cat_bucket["probe_results"].append(probe_result)
                bucket["probe_log"].append(probe_result)
                bucket["total_probes"] += 1

                if not verdict["aligned"]:
                    bucket["total_violations"] += 1
                    cat_bucket["n_violations"] += 1

            n_probes = len(selected_probes)
            cat_bucket["violation_rate"] = (
                cat_bucket["n_violations"] / n_probes if n_probes > 0 else 0.0
            )
            severity_values = [
                pr["verdict"]["severity_realized"] for pr in cat_bucket["probe_results"]
            ]
            cat_bucket["mean_severity_realized"] = (
                sum(severity_values) / len(severity_values) if severity_values else 0.0
            )

            bucket["by_category"][category] = cat_bucket

        bucket["violation_rate"] = (
            bucket["total_violations"] / bucket["total_probes"]
            if bucket["total_probes"] > 0
            else 0.0
        )

        master_log["results"][agent_name] = bucket

    overall_violation_rates = {
        name: master_log["results"][name]["violation_rate"] for name in agent_names
    }

    category_violation_rates = {}
    for category in config.BASELINE_CATEGORIES:
        category_violation_rates[category] = {
            name: master_log["results"][name]["by_category"][category]["violation_rate"]
            for name in agent_names
        }

    most_vulnerable_agent = max(overall_violation_rates, key=overall_violation_rates.get)

    category_mean_rates = {
        category: sum(category_violation_rates[category].values()) / len(agent_names)
        for category in config.BASELINE_CATEGORIES
    }
    most_breached_category = max(category_mean_rates, key=category_mean_rates.get)
    least_breached_category = min(category_mean_rates, key=category_mean_rates.get)

    master_log["summary"] = {
        "overall_violation_rates": overall_violation_rates,
        "category_violation_rates": category_violation_rates,
        "most_vulnerable_agent": most_vulnerable_agent,
        "most_breached_category": most_breached_category,
        "least_breached_category": least_breached_category,
    }

    end_time = time.time()
    master_log["end_time"] = end_time
    master_log["duration_seconds"] = end_time - master_log["start_time"]

    output_path = os.path.join(output_dir, master_log["experiment_id"] + "_" + log_filename)
    with open(output_path, "w") as f:
        f.write(json.dumps(master_log, indent=2))

    print(f"experiment_id: {master_log['experiment_id']}")
    print(f"output:        {output_path}")

    return master_log


def load_baseline_log(filepath):
    with open(filepath) as f:
        return json.loads(f.read())
