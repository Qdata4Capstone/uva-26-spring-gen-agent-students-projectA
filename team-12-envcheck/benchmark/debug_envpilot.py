"""Run EnvPilot once on a single case and dump every node's state output.

Usage:
    set -a; . .env.local; set +a
    uv run python benchmark/debug_envpilot.py bcb_002
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from runner_utils import build_env, load_candidates


def main() -> int:
    case_id = sys.argv[1] if len(sys.argv) > 1 else "bcb_002"
    cases = load_candidates()
    case = next((c for c in cases if c["case_id"] == case_id), None)
    if not case:
        print(f"No case_id={case_id}", file=sys.stderr)
        return 1

    bad_env_path = build_env(case["bad_env_pip"], python_version=case.get("bad_python"))
    print(f"bad_env: {bad_env_path}\n")

    from envcheck.agent.graph import build_graph, get_default_initial_state
    from envcheck.agent.nodes import reset_metrics, get_metrics

    reset_metrics()
    entry = case.get("entry_point") or "task_func"
    code_prompt = (case.get("code_prompt") or "").rstrip()
    augmented_task = (
        f"{case['instruct_prompt']}\n\n"
        f"Define a Python function named `{entry}`. "
        f"You must use exactly these imports and this signature:\n\n"
        f"```python\n{code_prompt}\n```"
    )
    initial = get_default_initial_state(
        task_description=augmented_task,
        env_path=str(bad_env_path),
    )

    app = build_graph()

    # Stream — yields one update per node executed
    last_state: dict = {}
    for i, update in enumerate(app.stream(initial, {"recursion_limit": 60})):
        # update is {node_name: state_diff}
        for node_name, diff in update.items():
            print(f"========= STEP {i} :: {node_name} =========")
            for key, val in diff.items():
                if key == "messages":
                    continue  # noisy
                printable = val
                if isinstance(val, str) and len(val) > 600:
                    printable = val[:300] + f"\n  ...[{len(val)-600} chars omitted]...\n" + val[-300:]
                if isinstance(val, dict):
                    printable = json.dumps(val, indent=2, default=str)[:1500]
                if isinstance(val, list) and len(val) > 4:
                    printable = f"[list of {len(val)} items, first: {val[0] if val else None}]"
                print(f"  {key}: {printable}")
            print()
            last_state.update(diff)

    print("=" * 60)
    print("Final metrics:", get_metrics())
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
