"""
reporting.py — all plot and table generation in one file.

Sections:
  Baseline plots    — violation rate by agent/category, severity heatmap, score distribution
  Red-team plots    — cumulative success rate, bandit state, surface violation rate, severity by turn
  Comparison plots  — baseline vs red-team violation rate, efficiency scatter
  Table helpers     — generate_baseline_table, generate_redteam_table (and print variants)
  Convenience       — run_baseline_plots, run_redteam_plots
"""

import os
import math

import matplotlib
import matplotlib.pyplot as plt

from config import (
    MATPLOTLIB_STYLE,
    FIGURE_DPI,
    FIGURE_WIDTH,
    FIGURE_HEIGHT,
    REPORT_OUTPUT_DIR,
)


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


def _apply_style():
    try:
        plt.style.use(MATPLOTLIB_STYLE)
    except OSError:
        plt.style.use("ggplot")


def _ci_halfwidth(mean_value, ci_entry):
    if not isinstance(ci_entry, dict):
        return 0.0
    low = ci_entry.get("low")
    high = ci_entry.get("high")
    if low is None or high is None:
        return 0.0
    try:
        low = float(low)
        high = float(high)
        mean_value = float(mean_value)
    except (TypeError, ValueError):
        return 0.0
    return max(abs(mean_value - low), abs(high - mean_value))


def _redteam_primary_counts(redteam_log):
    turns = list(redteam_log.get("turns", []))
    if turns:
        attempts = len(turns)
        violations = sum(
            1 for turn in turns
            if _primary_violation_detected(turn.get("score", {}))
        )
        return attempts, violations

    attempts = int(redteam_log.get("total_turns", 0) or 0)
    violations = int(redteam_log.get("reporting_violation_count", 0) or 0)
    if attempts > 0:
        return attempts, violations

    # Backward compatibility for older logs that only have campaign summary.
    campaign = redteam_log.get("campaign_summary", {})
    attempts = int(campaign.get("total_attempts", 0) or 0)
    violations = int(campaign.get("total_successes", 0) or 0)
    return attempts, violations


# ---------------------------------------------------------------------------
# Baseline plots (ABATE)
# ---------------------------------------------------------------------------

def plot_violation_rate_by_agent(log_dict, output_dir):
    _apply_style()
    overall = log_dict.get("summary", {}).get("overall_violation_rates", {})
    ci95 = log_dict.get("summary", {}).get("overall_violation_rates_ci95", {})
    agent_names = list(overall.keys())
    rates = [overall[a] for a in agent_names]
    xerr = [_ci_halfwidth(overall[a], ci95.get(a, {})) for a in agent_names]
    cmap = matplotlib.colormaps["tab10"]
    colors = [cmap(i) for i in range(len(agent_names))]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    any_ci = any(v > 0 for v in xerr)
    ax.barh(
        agent_names,
        rates,
        color=colors,
        xerr=(xerr if any_ci else None),
        capsize=(4 if any_ci else 0),
        error_kw={"elinewidth": 1.3} if any_ci else None,
    )
    ax.axvline(x=0.5, color="red", linestyle="--", label="50% threshold")
    ax.set_xlim(0.0, 1.0)
    if any_ci:
        ax.set_title("Overall Alignment Violation Rate by Agent (ABATE Baseline, 95% CI)")
    else:
        ax.set_title("Overall Alignment Violation Rate by Agent (ABATE Baseline)")
    ax.set_xlabel("Violation Rate")
    ax.set_ylabel("Agent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "violation_rate_by_agent.png"), dpi=FIGURE_DPI)
    plt.close(fig)


def plot_violation_rate_by_category(log_dict, output_dir):
    _apply_style()
    categories = log_dict["categories"]
    agent_names = log_dict["agent_names"]
    category_vr = log_dict["summary"]["category_violation_rates"]

    short_labels = [c[:12].replace("_", "\n") for c in categories]
    x = range(len(categories))
    n_agents = len(agent_names)
    bar_width = 0.8 / n_agents
    cmap = matplotlib.colormaps["tab10"]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    for i, agent_name in enumerate(agent_names):
        offsets = [xi + i * bar_width - (n_agents - 1) * bar_width / 2 for xi in x]
        rates = [category_vr[cat].get(agent_name, 0.0) for cat in categories]
        ax.bar(offsets, rates, width=bar_width, color=cmap(i), label=agent_name)

    ax.set_xticks(list(x))
    ax.set_xticklabels(short_labels, rotation=30, ha="right")
    ax.set_title("Violation Rate by Attack Category and Agent (ABATE Baseline)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Violation Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "violation_rate_by_category.png"), dpi=FIGURE_DPI)
    plt.close(fig)


def plot_severity_heatmap(log_dict, output_dir):
    _apply_style()
    agent_names = log_dict["agent_names"]
    categories = log_dict["categories"]

    data = []
    for agent_name in agent_names:
        row = []
        for category in categories:
            val = (
                log_dict["results"][agent_name]["by_category"][category]["mean_severity_realized"]
            )
            row.append(val)
        data.append(row)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_yticks(range(len(agent_names)))
    ax.set_yticklabels(agent_names)

    for i in range(len(agent_names)):
        for j in range(len(categories)):
            ax.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Severity Realized")
    ax.set_title("Mean Realized Severity Heatmap (ABATE Baseline)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "severity_heatmap.png"), dpi=FIGURE_DPI)
    plt.close(fig)


def plot_probe_score_distribution(log_dict, output_dir):
    _apply_style()
    agent_names = log_dict["agent_names"]
    n_agents = len(agent_names)

    fig, axes = plt.subplots(1, n_agents, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    if n_agents == 1:
        axes = [axes]

    for ax, agent_name in zip(axes, agent_names):
        severity_values = []
        for category in log_dict["categories"]:
            for pr in log_dict["results"][agent_name]["by_category"][category]["probe_results"]:
                severity_values.append(pr["verdict"]["severity_realized"])
        ax.hist(severity_values, bins=10, range=(0.0, 1.0))
        ax.set_title(agent_name)
        ax.set_xlabel("Severity Realized")
        ax.set_ylabel("Count")

    fig.suptitle("Distribution of Realized Violation Severity by Agent")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "severity_distribution.png"), dpi=FIGURE_DPI)
    plt.close(fig)


def generate_baseline_table(log_dict):
    header = ["Agent", "Category", "N Probes", "N Violations", "Violation Rate", "Mean Severity"]
    rows = []
    for agent_name in log_dict["agent_names"]:
        for category in log_dict["categories"]:
            cat_data = log_dict["results"][agent_name]["by_category"][category]
            rows.append([
                agent_name,
                category,
                cat_data["n_probes"],
                cat_data["n_violations"],
                f"{cat_data['violation_rate']:.3f}",
                f"{cat_data['mean_severity_realized']:.3f}",
            ])
    return header, rows


def print_baseline_table(log_dict):
    header, rows = generate_baseline_table(log_dict)
    col_widths = [20, 35, 10, 12, 15, 14]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*header))
    print("-" * sum(col_widths))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def run_baseline_plots(log_dict, output_dir=REPORT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plot_violation_rate_by_agent(log_dict, output_dir)
    plot_violation_rate_by_category(log_dict, output_dir)
    plot_severity_heatmap(log_dict, output_dir)
    plot_probe_score_distribution(log_dict, output_dir)
    print_baseline_table(log_dict)


# ---------------------------------------------------------------------------
# Red-team plots (MARSE)
# ---------------------------------------------------------------------------

def plot_attack_success_over_turns(log_dict, output_dir):
    _apply_style()
    turns = log_dict["turns"]
    cumulative_rates = []
    violation_turns = []
    violations_so_far = 0

    for i, turn in enumerate(turns):
        if _primary_violation_detected(turn.get("score", {})):
            violations_so_far += 1
            violation_turns.append(i + 1)
        cumulative_rates.append(violations_so_far / (i + 1))

    x = list(range(1, len(turns) + 1))
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax.plot(x, cumulative_rates, linewidth=2)

    if violation_turns:
        viol_rates = [cumulative_rates[t - 1] for t in violation_turns]
        ax.scatter(violation_turns, viol_rates, color="red", zorder=5, label="Violation")
        ax.legend()

    ax.set_title("Cumulative Reporting Violation Rate Over Turns (Primary Metric)")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Cumulative Success Rate")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "attack_success_over_turns.png"), dpi=FIGURE_DPI)
    plt.close(fig)


def plot_bandit_state(log_dict, output_dir):
    _apply_style()
    bandit = log_dict.get("bandit_final_state", {})
    surfaces = list(bandit.keys())
    attempts = [bandit[s].get("attempts", 0) for s in surfaces]
    successes = [bandit[s].get("successes", 0) for s in surfaces]

    x = range(len(surfaces))
    bar_width = 0.35
    cmap = matplotlib.colormaps["tab10"]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax.bar([xi - bar_width / 2 for xi in x], attempts, width=bar_width, color=cmap(0), label="attempts")
    ax.bar([xi + bar_width / 2 for xi in x], successes, width=bar_width, color=cmap(1), label="successes")
    ax.set_xticks(list(x))
    ax.set_xticklabels(surfaces, rotation=20, ha="right")
    ax.set_title("Bandit State (Rule Metric): Attempts vs Successes by Attack Surface")
    ax.set_xlabel("Attack Surface")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bandit_state.png"), dpi=FIGURE_DPI)
    plt.close(fig)


def plot_surface_violation_rate(log_dict, output_dir):
    _apply_style()
    turns = list(log_dict.get("turns", []))
    per_surface = {}
    for turn in turns:
        surface = turn.get("surface")
        if not surface:
            continue
        bucket = per_surface.setdefault(surface, {"attempts": 0, "violations": 0})
        bucket["attempts"] += 1
        if _primary_violation_detected(turn.get("score", {})):
            bucket["violations"] += 1

    # Backward compatibility for older logs that may not have turn-level scores.
    if not per_surface:
        bandit = log_dict.get("bandit_final_state", {})
        for surface, state in bandit.items():
            attempts = int(state.get("attempts", 0))
            successes = int(state.get("successes", 0))
            if attempts > 0:
                per_surface[surface] = {"attempts": attempts, "violations": successes}

    surface_rates = {}
    for surface, bucket in per_surface.items():
        attempts = bucket["attempts"]
        if attempts > 0:
            surface_rates[surface] = bucket["violations"] / attempts

    sorted_surfaces = sorted(surface_rates, key=surface_rates.get, reverse=True)
    rates = [surface_rates[s] for s in sorted_surfaces]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    bars = ax.barh(sorted_surfaces, rates)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.2f}",
            va="center",
        )
    ax.set_xlim(0.0, 1.1)
    ax.set_title("Primary Violation Rate by Attack Surface (Reporting-First)")
    ax.set_xlabel("Primary Violation Rate (violations / attempts)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "surface_violation_rate.png"), dpi=FIGURE_DPI)
    plt.close(fig)


def plot_novelty_over_turns(log_dict, output_dir):
    _apply_style()
    turns = log_dict["turns"]
    x = [t["turn"] for t in turns]
    y = [_primary_severity(t.get("score", {})) for t in turns]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax.plot(x, y, linewidth=2)
    ax.fill_between(x, y, alpha=0.2)
    ax.set_title("Primary Severity Score by Turn (Reporting-First)")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Severity Score")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "severity_by_turn.png"), dpi=FIGURE_DPI)
    plt.close(fig)


def generate_redteam_table(log_dict):
    header = ["Turn", "Surface", "Primary Violation", "Primary Severity", "Blocked at Perceive"]
    rows = []
    for turn in log_dict["turns"]:
        score = turn.get("score", {})
        rows.append([
            turn["turn"],
            turn["surface"],
            "Yes" if _primary_violation_detected(score) else "No",
            f"{_primary_severity(score):.2f}",
            "Yes" if turn["target_result_summary"].get("blocked_at_perceive", False) else "No",
        ])
    return header, rows


def print_redteam_table(log_dict):
    header, rows = generate_redteam_table(log_dict)
    col_widths = [6, 30, 20, 10, 20]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*header))
    print("-" * sum(col_widths))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def run_redteam_plots(log_dict, output_dir=REPORT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plot_attack_success_over_turns(log_dict, output_dir)
    plot_bandit_state(log_dict, output_dir)
    plot_surface_violation_rate(log_dict, output_dir)
    plot_novelty_over_turns(log_dict, output_dir)
    print_redteam_table(log_dict)


# ---------------------------------------------------------------------------
# Comparison plots (baseline vs red-team)
# ---------------------------------------------------------------------------

def plot_baseline_vs_redteam_violation_rate(baseline_log, redteam_log, output_dir):
    _apply_style()
    agent_names = list(baseline_log.get("summary", {}).get("overall_violation_rates", {}).keys())

    total_attempts, primary_violations = _redteam_primary_counts(redteam_log)
    redteam_rate = primary_violations / total_attempts if total_attempts > 0 else 0.0
    redteam_target = redteam_log.get("target_agent", None)

    bar_width = 0.35
    x = range(len(agent_names))
    baseline_rates = [
        baseline_log["summary"]["overall_violation_rates"].get(a, 0.0) for a in agent_names
    ]
    redteam_rates = [
        redteam_rate if a == redteam_target else 0.0 for a in agent_names
    ]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax.bar(
        [xi - bar_width / 2 for xi in x],
        baseline_rates,
        width=bar_width,
        color="steelblue",
        label="ABATE Baseline",
    )
    ax.bar(
        [xi + bar_width / 2 for xi in x],
        redteam_rates,
        width=bar_width,
        color="firebrick",
        label="Red Team",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(agent_names)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Baseline vs Red Team Violation Rate by Agent (Reporting-First)")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Violation Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "baseline_vs_redteam.png"), dpi=FIGURE_DPI)
    plt.close(fig)


def plot_efficiency_comparison(baseline_log, redteam_log, output_dir):
    _apply_style()
    baseline_probes = sum(
        res["total_probes"]
        for res in baseline_log.get("results", {}).values()
    )
    baseline_violations = sum(
        res["total_violations"]
        for res in baseline_log.get("results", {}).values()
    )

    redteam_probes, redteam_violations = _redteam_primary_counts(redteam_log)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax.scatter(
        [baseline_probes],
        [baseline_violations],
        color="steelblue",
        s=150,
        marker="o",
        label="ABATE Baseline",
        zorder=5,
    )
    ax.scatter(
        [redteam_probes],
        [redteam_violations],
        color="firebrick",
        s=200,
        marker="*",
        label="Red Team",
        zorder=5,
    )
    ax.annotate(
        "ABATE Baseline",
        (baseline_probes, baseline_violations),
        textcoords="offset points",
        xytext=(8, 4),
    )
    ax.annotate(
        "Red Team",
        (redteam_probes, redteam_violations),
        textcoords="offset points",
        xytext=(8, 4),
    )
    ax.set_title("Efficiency: Violations Found vs Probes Used")
    ax.set_xlabel("Total Probes / Attacks Used")
    ax.set_ylabel("Violations Found (Primary Metric)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "efficiency_comparison.png"), dpi=FIGURE_DPI)
    plt.close(fig)
