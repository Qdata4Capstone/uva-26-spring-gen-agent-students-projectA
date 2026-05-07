"""
EnvCheck Preflight Runner — Execute isolated smoke tests in a target environment.

Runs a minimal Python script inside a virtual environment to verify
that a proposed code plan works before generating the full project.

Usage:
    from envcheck.preflight_runner import run_preflight

    result = run_preflight(
        code='import numpy; print(numpy.__version__)',
        env_path='./environments/case_numpy_2x',
    )
    print(result.success, result.stdout)
"""

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PreflightResult:
    """Result of a preflight test execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float


def run_preflight(
    code: str,
    env_path: str | Path,
    timeout: int = 15,
) -> PreflightResult:
    """Run a preflight smoke test in an isolated environment.

    Writes the code to a temporary file and executes it using the
    Python binary from the specified virtual environment.

    Args:
        code: Python source code to execute.
        env_path: Path to the virtual environment root.
        timeout: Maximum execution time in seconds.

    Returns:
        PreflightResult with success status, output, and timing.
    """
    env_path = Path(env_path)
    python_bin = env_path / "bin" / "python"

    if not python_bin.exists():
        return PreflightResult(
            success=False,
            stdout="",
            stderr=f"Python binary not found at {python_bin}",
            exit_code=-1,
            duration_ms=0.0,
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="envcheck_preflight_", delete=False
    ) as f:
        f.write(code)
        script_path = Path(f.name)

    # Force a non-interactive matplotlib backend. Without this, importing
    # matplotlib (transitively via seaborn etc.) on the python-build-standalone
    # Python 3.8 build we use for legacy envs crashes on Tk init
    # (`module '_tkinter' has no attribute '__file__'`), masking the *real*
    # breaking-change error the smoke test was meant to surface.
    env = {**os.environ, "MPLBACKEND": "Agg"}

    start = time.perf_counter()
    try:
        result = subprocess.run(
            [str(python_bin), str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        return PreflightResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            duration_ms=duration_ms,
        )
    except subprocess.TimeoutExpired:
        duration_ms = (time.perf_counter() - start) * 1000
        return PreflightResult(
            success=False,
            stdout="",
            stderr=f"Preflight test timed out after {timeout}s",
            exit_code=-2,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        return PreflightResult(
            success=False,
            stdout="",
            stderr=f"Preflight execution error: {e}",
            exit_code=-3,
            duration_ms=duration_ms,
        )
    finally:
        script_path.unlink(missing_ok=True)


def to_dict(result: PreflightResult) -> dict:
    """Convert a PreflightResult to a serializable dict."""
    return {
        "success": result.success,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "duration_ms": round(result.duration_ms, 1),
    }
