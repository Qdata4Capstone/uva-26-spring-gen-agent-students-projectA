"""
run.py — unified CLI entry point.

Usage:
  python run.py experiment <target_agent> <n_turns> <stop_on_violation>
  python run.py baseline
  python run.py --help

Subcommands:
  experiment   Run an adaptive red-team experiment (MARSE).
               Args: target_agent (medical|financial|customer_service)
                     n_turns      (int, default from config)
                     stop_on_violation (true|false, default from config)
  baseline     Run the static probe bank experiment (ABATE) for all three agents.
"""

import sys

import config
from experiments import run_experiment, summarize_experiment, run_baseline_experiment
from reporting import run_redteam_plots, run_baseline_plots


def _usage():
    print(__doc__.strip())
    sys.exit(1)


def cmd_experiment(args):
    target_agent_name = args[0] if len(args) > 0 else config.TARGET_AGENT
    n_turns = int(args[1]) if len(args) > 1 else config.MAX_TURNS
    stop_on_violation = args[2].lower() == "true" if len(args) > 2 else config.VIOLATION_STOPS_EXPERIMENT

    result = run_experiment(
        target_agent_name=target_agent_name,
        n_turns=n_turns,
        stop_on_violation=stop_on_violation,
    )
    print(summarize_experiment(result))
    run_redteam_plots(result, config.REPORT_OUTPUT_DIR)
    print(f"Red team plots saved to {config.REPORT_OUTPUT_DIR}")


def cmd_baseline(_args):
    log_dict = run_baseline_experiment()
    run_baseline_plots(log_dict, output_dir=config.REPORT_OUTPUT_DIR)
    print("Baseline experiment complete.")
    print(config.REPORT_OUTPUT_DIR)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _usage()

    subcommand = sys.argv[1]
    rest = sys.argv[2:]

    if subcommand == "experiment":
        cmd_experiment(rest)
    elif subcommand == "baseline":
        cmd_baseline(rest)
    else:
        print(f"Unknown subcommand: {subcommand!r}")
        _usage()


if __name__ == "__main__":
    main()
