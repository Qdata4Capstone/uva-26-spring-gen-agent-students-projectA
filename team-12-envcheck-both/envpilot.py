"""
EnvPilot CLI — Interactive entry point for the proactive coding agent.

Runs the full 5-phase LangGraph workflow:
  1. Analysis & Uncertainty Assessment
  2. Environment Probing
  3. Knowledge Alignment (+ optional web search)
  4. Pre-flight Verification
  5. Final One-Pass Code Generation

Usage:
    python envpilot.py "Create a numpy script that does trapezoid integration"
    python envpilot.py --env ./environments/case_numpy_2x "Create a numpy script..."
    python envpilot.py --verbose "..."
"""

import argparse
import json
import logging
import sys
import textwrap


def main():
    parser = argparse.ArgumentParser(
        description="EnvPilot — Proactive coding agent with environment awareness",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="The coding task description",
    )
    parser.add_argument(
        "--env",
        default="",
        help="Path to a virtual environment to target",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if not args.task:
        # Interactive mode
        print("=" * 60)
        print("  EnvPilot — Proactive Coding Agent")
        print("  Plan Before Coding; Verify Before Executing")
        print("=" * 60)
        print()
        task = input("Describe your coding task:\n> ").strip()
        if not task:
            print("No task provided. Exiting.")
            sys.exit(0)
    else:
        task = args.task

    print(f"\n{'=' * 60}")
    print(f"  Task: {task[:80]}{'...' if len(task) > 80 else ''}")
    if args.env:
        print(f"  Environment: {args.env}")
    print(f"{'=' * 60}\n")

    # Lazy imports to keep CLI startup fast
    from envcheck.agent.graph import build_graph, get_default_initial_state

    print("[*] Building EnvPilot agent graph...")
    app = build_graph()

    initial_state = get_default_initial_state(
        task_description=task,
        env_path=args.env,
    )

    print("[*] Running 5-phase workflow...\n")
    phase_names = {
        "analysis_complete": "Phase 1: Analysis Complete",
        "env_probed": "Phase 2: Environment Probed",
        "kb_queried": "Phase 3: Knowledge Base Queried",
        "web_searched": "Phase 3b: Web Search Complete",
        "kb_updated": "Phase 3c: Knowledge Base Updated",
        "preflight_passed": "Phase 4: Preflight PASSED",
        "preflight_failed": "Phase 4: Preflight FAILED",
        "complete": "Phase 5: Code Generation Complete",
    }

    try:
        result = app.invoke(initial_state)
    except Exception as e:
        print(f"\n[ERROR] Agent execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Display results
    phase = result.get("phase", "unknown")
    print(f"\n{'=' * 60}")
    print(f"  Status: {phase_names.get(phase, phase)}")
    print(f"  Uncertainty Score: {result.get('uncertainty_score', '?')}%")
    print(f"  Preflight Attempts: {result.get('preflight_attempts', 0)}")
    print(f"  KB Rules Found: {len(result.get('kb_results', []))}")
    print(f"{'=' * 60}\n")

    if result.get("error"):
        print(f"[WARNING] Error during execution: {result['error']}\n")

    final_code = result.get("final_code", "")
    if final_code:
        print("Generated Code:")
        print("-" * 60)
        print(final_code)
        print("-" * 60)

        # Optionally save to file
        print(f"\nSave to file? [Enter filename or press Enter to skip]")
        try:
            filename = input("> ").strip()
            if filename:
                from pathlib import Path
                Path(filename).write_text(final_code)
                print(f"Saved to {filename}")
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        print("[!] No code was generated.")
        if result.get("messages"):
            last_msg = result["messages"][-1]
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            print(f"\nLast message:\n{textwrap.indent(content[:500], '  ')}")


if __name__ == "__main__":
    main()
