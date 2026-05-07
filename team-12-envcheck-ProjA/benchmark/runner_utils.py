"""Shared utilities for benchmark scripts.

Functions for:
  - Building isolated venvs from a pip-spec list (using uv)
  - Composing a runnable test script from a candidate's fields
  - Running a script inside a specific venv with timeout/resource limits

Conventions:
  - Venvs are cached under benchmark/envs/<key>/. `key` is a stable hash of
    the sorted pip-spec list, so cases sharing identical env specs reuse one
    venv. Pass force_rebuild=True to recreate.
  - Runs always pass `-q` to the script and capture stdout/stderr.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BENCHMARK_DIR = Path(__file__).parent
ENVS_DIR = BENCHMARK_DIR / "envs"
ENVS_DIR.mkdir(exist_ok=True)


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.exit_code == 0


def _env_key(pip_lines: list[str], python_version: str = "") -> str:
    """Stable short hash for (Python version, pip spec) — dedupes venvs."""
    canonical = (python_version or "default") + "\n" + "\n".join(sorted(pip_lines))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def env_path_for(pip_lines: list[str], python_version: str = "") -> Path:
    return ENVS_DIR / _env_key(pip_lines, python_version)


def venv_python(env_dir: Path) -> Path:
    return env_dir / "bin" / "python"


# Default Python for all benchmark venvs. 3.11 has the broadest wheel coverage
# for both modern packages (numpy 2.x, pandas 2.x) and slightly older ones
# (pandas 1.5.x, scipy 1.10+) that we need for legacy snapshots.
DEFAULT_PYTHON = "3.11"


def build_env(pip_lines: list[str], force_rebuild: bool = False,
              python_version: Optional[str] = None,
              quiet: bool = True) -> Path:
    """Create venv (if missing) and install pip_lines into it. Returns env_dir.

    Caches by sorted-pip-spec hash so identical specs share one venv.
    Pre-installs setuptools+wheel; if the main install fails (e.g. older
    sdists missing pkg_resources as a build dep), retries with
    --no-build-isolation.
    """
    env_dir = env_path_for(pip_lines, python_version or DEFAULT_PYTHON)
    marker = env_dir / ".envcheck_ready"

    if env_dir.exists() and marker.exists() and not force_rebuild:
        return env_dir

    if env_dir.exists():
        shutil.rmtree(env_dir)

    # Create venv (default Python 3.11 unless overridden)
    py_ver = python_version or DEFAULT_PYTHON
    venv_cmd = ["uv", "venv", "--python", py_ver, str(env_dir)]
    subprocess.run(venv_cmd, check=True, capture_output=quiet)

    py = str(venv_python(env_dir))

    # Bootstrap: install setuptools + wheel so older sdists can build
    # against pkg_resources without --no-build-isolation gymnastics.
    subprocess.run(
        ["uv", "pip", "install", "--python", py, "setuptools", "wheel"],
        check=True, capture_output=quiet,
    )

    # First attempt: normal install with build isolation
    install_cmd = ["uv", "pip", "install", "--python", py, *pip_lines]
    result = subprocess.run(install_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Retry with --no-build-isolation; older sdists (e.g. pandas 1.5.x
        # on Python 3.12) need the venv's setuptools to provide pkg_resources.
        retry_cmd = [
            "uv", "pip", "install", "--python", py,
            "--no-build-isolation",
            *pip_lines,
        ]
        result2 = subprocess.run(retry_cmd, capture_output=True, text=True)
        if result2.returncode != 0:
            raise RuntimeError(
                f"uv pip install failed for {pip_lines}\n"
                f"--- first attempt ---\n{result.stderr}\n"
                f"--- retry with --no-build-isolation ---\n{result2.stderr}"
            )

    # Mark ready
    marker.write_text(json.dumps({"pip_lines": pip_lines}, indent=2))
    return env_dir


def compose_test_script(case: dict, user_code: Optional[str] = None) -> str:
    """Glue code_prompt + (user_code or canonical_solution) + test into a
    standalone runnable script.

    `user_code` lets the caller substitute model-generated code in place of
    the canonical. The function body is the part after the `def task_func(...)`
    signature line in code_prompt; the canonical/user code is the function body
    (already indented).
    """
    body = user_code if user_code is not None else case["canonical_solution"]
    return (
        case["code_prompt"]
        + body
        + "\n\n"
        + case["test"]
        + "\n\n"
        + "if __name__ == '__main__':\n"
        + "    import unittest\n"
        + "    unittest.main(verbosity=2, exit=True)\n"
    )


def run_in_env(env_dir: Path, script: str, timeout_s: int = 60) -> RunResult:
    """Write script to a tempfile, run with venv's python, capture output."""
    import os
    import time
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, dir="/tmp"
    ) as f:
        f.write(script)
        script_path = f.name

    py = str(venv_python(env_dir))
    # Force non-interactive matplotlib backend; the python-build-standalone
    # Python 3.8 build we use for legacy envs has a broken _tkinter, so the
    # default TkAgg backend can't import. Agg is safe for plt.show()/savefig.
    env = {**os.environ, "MPLBACKEND": "Agg"}
    start = time.time()
    try:
        proc = subprocess.run(
            [py, script_path],
            capture_output=True, text=True,
            timeout=timeout_s,
            env=env,
        )
        return RunResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            timed_out=False,
            duration_s=time.time() - start,
        )
    except subprocess.TimeoutExpired as e:
        return RunResult(
            exit_code=-1,
            stdout=(e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            stderr=(e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
            timed_out=True,
            duration_s=timeout_s,
        )
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass


def load_candidates(path: Optional[Path] = None) -> list[dict]:
    p = path or (BENCHMARK_DIR / "candidates.json")
    return json.loads(p.read_text())
