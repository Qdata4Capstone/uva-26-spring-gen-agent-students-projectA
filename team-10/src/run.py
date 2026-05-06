import os
import re
import sys

import config
from experiments import run_experiment, summarize_experiment, run_baseline_experiment
from reporting import run_redteam_plots, run_baseline_plots


# short use note:
# python run.py experiment <target_agent> <n_turns> <stop_on_violation>
# python run.py cross <n_turns> <stop_on_violation> [cross_only=true]
# python run.py baseline

def _usage():
    print("python run.py experiment <target_agent> <n_turns> <stop_on_violation>")
    print("python run.py cross <n_turns> <stop_on_violation> [cross_only=true]")
    print("python run.py baseline")
    sys.exit(1)


def _safePathPiece(text):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text or "")).strip("_") or "unknown"


def cmd_experiment(args):
    # targetAgentName = config.TARGET_AGENT
    targetAgentName = args[0] if len(args) > 0 else config.TARGET_AGENT

    # nTurns = int(args[1]) if len(args) > 1 else 8
    # nTurns = int(args[1]) if len(args) > 1 else 12
    nTurns = int(args[1]) if len(args) > 1 else config.MAX_TURNS

    # stopOnViolation = True
    stopOnViolation = args[2].lower() == "true" if len(args) > 2 else config.VIOLATION_STOPS_EXPERIMENT


    resultObj = run_experiment(target_agent_name=targetAgentName, n_turns=nTurns, stop_on_violation=stopOnViolation)
    print(summarize_experiment(resultObj))

    experimentDir = os.path.join(config.REPORT_OUTPUT_DIR, "experiment", _safePathPiece(targetAgentName), _safePathPiece(resultObj.get("experiment_id", "unknown_experiment")))

    run_redteam_plots(resultObj, experimentDir)
    print(f"red team plots saved to {experimentDir}")


def cmd_cross(args):
    # agentList = ["medical", "financial", "customer_service"]
    agentList = ["medical", "weak_medical", "financial", "customer_service"]

    # nTurns = config.MAX_TURNS
    nTurns = int(args[0]) if len(args) > 0 else config.MAX_TURNS

    stopOnViolation = args[1].lower() == "true" if len(args) > 1 else config.VIOLATION_STOPS_EXPERIMENT

    # crossOnly = True
    # crossOnly = False
    crossOnly = args[2].lower() != "false" if len(args) > 2 else True

    # pairs = [(attacker, target) for attacker in agentList for target in agentList if attacker != target]
    pairs = [(attacker, target) for attacker in agentList for target in agentList if not crossOnly or attacker != target]


    labelText = "cross-domain only" if crossOnly else "all combinations"
    print(f"running {len(pairs)} attacker target pair(s) ({labelText})")  # keep this plain

    for attacker, target in pairs:
        print(f"\n{'=' * 60}")
        print(f"attacker domain: {attacker} to target agent: {target}")
        print(f"{'=' * 60}")

        resultObj = run_experiment(target_agent_name=target, n_turns=nTurns, stop_on_violation=stopOnViolation, red_team_domain=attacker)
        print(summarize_experiment(resultObj))

        pairDir = os.path.join(config.REPORT_OUTPUT_DIR, "cross", f"{_safePathPiece(attacker)}_to_{_safePathPiece(target)}", _safePathPiece(resultObj.get("experiment_id", "unknown_experiment")))
        run_redteam_plots(resultObj, pairDir)
        print(f"red team plots saved to {pairDir}")


def cmd_baseline(_args):
    baseLog = run_baseline_experiment()

    baselineDir = os.path.join(config.REPORT_OUTPUT_DIR, "baseline", _safePathPiece(baseLog.get("experiment_id", "unknown_baseline")))
    run_baseline_plots(baseLog, output_dir=baselineDir)


    print("baseline experiment complete")
    print(baselineDir)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _usage()

    subCommand = sys.argv[1]
    restArgs = sys.argv[2:]

    if subCommand == "experiment":
        cmd_experiment(restArgs)
    elif subCommand == "cross":
        cmd_cross(restArgs)
    elif subCommand == "baseline":
        cmd_baseline(restArgs)
    else:
        print(f"unknown subcommand: {subCommand!r}")
        _usage()


if __name__ == "__main__":
    main()
