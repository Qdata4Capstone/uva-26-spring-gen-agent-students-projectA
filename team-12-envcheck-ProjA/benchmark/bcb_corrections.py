"""Per-task correct_solution substitutions for BigCodeBench cases.

Each entry is task_id → (old_substring, new_substring). build_candidates.py
applies a single str.replace() on canonical_solution to produce
correct_solution — code that uses the alternative API and works in bad_env.

Substitutions must match the canonical exactly; if a replacement doesn't
change anything, build_candidates.py raises (so renames in BCB upstream
get caught).
"""

# Common pattern for sns.histplot → sns.distplot:
#   - histplot defaults to kde=False; distplot defaults to kde=True.
#     So when canonical has no `kde=` or `kde=False`, we explicitly add
#     `kde=False` to the distplot call to match behavior.
#   - histplot signature uses (data=, x=) for DataFrame; distplot takes a
#     1-D array, so we extract df[col] manually.

CORRECTIONS: dict[str, tuple[str, str]] = {
    # sns.displot → sns.distplot (returns Axes directly, no .ax attribute)
    "BigCodeBench/43": (
        "        plot = sns.displot(df[col], bins=10)\n"
        "        plots.append(plot.ax)\n",
        "        plot = sns.distplot(df[col], bins=10, kde=False)\n"
        "        plots.append(plot)\n",
    ),

    # sns.histplot → sns.distplot
    "BigCodeBench/53": (
        'sns.histplot(data=df, x="Age")',
        'sns.distplot(df["Age"], kde=False)',
    ),
    "BigCodeBench/62": (
        "sns.histplot(from_user_values, color=color)",
        "sns.distplot(from_user_values, color=color, kde=False)",
    ),
    "BigCodeBench/68": (
        "ax = sns.histplot(data=df, x='Age', kde=True)",
        "ax = sns.distplot(df['Age'])",
    ),
    "BigCodeBench/71": (
        "plot = sns.histplot(df['mean'], kde=True)",
        "plot = sns.distplot(df['mean'])",
    ),
    "BigCodeBench/196": (
        "plot = sns.histplot(random_numbers, kde=False)",
        "plot = sns.distplot(random_numbers, kde=False)",
    ),
    "BigCodeBench/230": (
        "sns.histplot(df['Score'], bins=10)",
        "sns.distplot(df['Score'], bins=10, kde=False)",
    ),
    "BigCodeBench/307": (
        "plot = sns.histplot(data)",
        "plot = sns.distplot(data, kde=False)",
    ),
    "BigCodeBench/530": (
        'ax = sns.histplot(duplicates_df["age"], bins=bins)',
        'ax = sns.distplot(duplicates_df["age"], bins=bins, kde=False)',
    ),
    "BigCodeBench/537": (
        'ax = sns.histplot(data=df, x="age", bins=30, kde=True)',
        'ax = sns.distplot(df["age"], bins=30)',
    ),
    "BigCodeBench/916": (
        "histplot_ax = sns.histplot(df['closing_price'], kde=True, ax=axes[1])",
        # distplot doesn't auto-set ylabel; test checks 'Count' is in ylabel.
        "histplot_ax = sns.distplot(df['closing_price'], ax=axes[1])\n"
        "    histplot_ax.set_ylabel('Count')",
    ),
    "BigCodeBench/1024": (
        "plot = sns.histplot(df.values.flatten(), bins=bin_edges, kde=False)",
        "plot = sns.distplot(df.values.flatten(), bins=bin_edges, kde=False)",
    ),

    # OneHotEncoder(sparse=) → OneHotEncoder(sparse_output=)
    "BigCodeBench/686": (
        "encoder = OneHotEncoder(sparse=False)",
        "encoder = OneHotEncoder(sparse_output=False)",
    ),
}


def apply_correction(task_id: str, canonical: str) -> str:
    """Return correct_solution for a BCB task. Raises if no rule or no match."""
    if task_id not in CORRECTIONS:
        raise KeyError(f"No correction defined for {task_id}")
    old, new = CORRECTIONS[task_id]
    if old not in canonical:
        raise ValueError(
            f"Correction old-substring not found in canonical for {task_id}.\n"
            f"  expected: {old!r}\n"
            f"  canonical:\n{canonical}"
        )
    return canonical.replace(old, new)
