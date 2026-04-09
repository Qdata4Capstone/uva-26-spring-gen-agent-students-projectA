"""
run.py — unified CLI entry point.

Usage:
  python run.py experiment <target_agent> <n_turns> <stop_on_violation>
  python run.py baseline
  python run.py frr
  python run.py transfer <source_log_path> [top_n]
  python run.py --help

Subcommands:
  experiment   Run an adaptive red-team experiment (MARSE).
               Args: target_agent (medical|financial|customer_service)
                     n_turns      (int, default from config)
                     stop_on_violation (true|false, default from config)
  baseline     Run the static probe bank experiment (ABATE) for all three agents.
  frr          Run the False Refusal Rate experiment on all three agents.
               Sends benign legitimate queries and measures over-refusal rate.
               Target: FRR < 5% per agent.
  transfer     Replay top-N attacks from a MARSE log against all agents.
               Args: source_log_path  path to a campaign_log.json
                     top_n            (int, default from config)
"""

import os
import sys


def _load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


_load_env()

import config
from experiments import (
    run_experiment, summarize_experiment,
    run_baseline_experiment,
    run_frr_experiment,
    run_transfer_experiment,
)
from reporting import run_redteam_plots, run_baseline_plots, run_frr_plots


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


def cmd_frr(_args):
    frr_log = run_frr_experiment()
    run_frr_plots(frr_log, output_dir=config.REPORT_OUTPUT_DIR)
    print("FRR experiment complete.")
    print(config.REPORT_OUTPUT_DIR)


def cmd_transfer(args):
    if not args:
        print("Usage: python run.py transfer <source_log_path> [top_n]")
        _usage()
    source_log_path = args[0]
    top_n = int(args[1]) if len(args) > 1 else config.TRANSFER_TOP_N_ATTACKS
    run_transfer_experiment(source_log_path, top_n=top_n)
    print("Transfer experiment complete.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _usage()

    subcommand = sys.argv[1]
    rest = sys.argv[2:]

    if subcommand == "experiment":
        cmd_experiment(rest)
    elif subcommand == "baseline":
        cmd_baseline(rest)
    elif subcommand == "frr":
        cmd_frr(rest)
    elif subcommand == "transfer":
        cmd_transfer(rest)
    else:
        print(f"Unknown subcommand: {subcommand!r}")
        _usage()


if __name__ == "__main__":
    main()
