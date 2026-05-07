"""Build benchmark/candidates.json from two sources:
  1. BigCodeBench v0.1.4 — directed regex search for known breaking-change patterns
  2. benchmark/manual_cases.py — hand-written REMOVAL-direction cases

Excludes deprecation-only / FutureWarning-only patterns (e.g. pd.applymap),
which are not hard env-conflict errors.

Run: `uv run --with datasets python benchmark/build_candidates.py`
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from bcb_corrections import apply_correction
from manual_cases import CASES as MANUAL_CASES

OUT_PATH = Path(__file__).parent / "candidates.json"

# Tasks to drop because their tests are unsolvable with any alternative API
# (e.g. hardcoded RNG outputs that depend on default_rng's PCG64 algorithm —
# RandomState's Mersenne Twister produces different values, so test never passes).
DROP_TASK_IDS = {
    "BigCodeBench/758",  # np_default_rng: test_case_6 assert_frame_equal vs hardcoded default_rng output
    "BigCodeBench/66",   # sns_distplot: seaborn 0.14 doesn't exist on PyPI; distplot still in 0.13.2 (deprecation only, no hard removal)
    "BigCodeBench/196",  # sns_histplot: test asserts ax.containers[0].datavalues — histplot-specific attr, no distplot equivalent
    "BigCodeBench/307",  # sns_histplot: test asserts str(type(plot))=='matplotlib.axes._axes.Axes' but old matplotlib 3.2.2 returns AxesSubplot
}

# Skip entire rules — usually because there's no Python version we can build them on.
SKIP_RULES = {
    "scipy_fft",  # bad version needs scipy 1.3.3 → Python 3.7, which uv no longer ships
}

# Tasks where the alternative API likely works but the test has tight numeric
# / plot-artifact-count checks that may behave differently across API variants.
# Worth flagging for runtime verification before relying on them.
NEEDS_RUNTIME_VERIFY: dict[str, str] = {
    "BigCodeBench/307": "len(patches)==5; sns.distplot default Sturges binning may give different bin count vs sns.histplot",
    "BigCodeBench/66":  "sns.distplot deprecated 0.11; verify whether removed in 0.14 (still warning-only as of 0.13.x)",
    "BigCodeBench/686": "OneHotEncoder sparse= deprecated 1.2; verify whether removed in 1.6",
}

# Patterns kept to "hard error on bad_version" only — no warning-only cases.
PATTERNS = [
    ("sns_histplot",   r"\bsns\.histplot\b",   "seaborn", "0.10.1", "0.13.2", "AttributeError",
     "sns.histplot added in seaborn 0.11", "introduction", ""),
    ("sns_displot",    r"\bsns\.displot\b",    "seaborn", "0.10.1", "0.13.2", "AttributeError",
     "sns.displot added in seaborn 0.11", "introduction", ""),
    ("sns_distplot",   r"\bsns\.distplot\b",   "seaborn", "0.14.0", "0.11.2", "AttributeError",
     "sns.distplot deprecated 0.11; verify whether removed in 0.14", "removal", "needs_runtime_verify"),
    ("scipy_fft",      r"from scipy\.fft import|scipy\.fft\.(?:fft|ifft|rfft|irfft|fft2)",
     "scipy", "1.3.3", "1.11.4", "ModuleNotFoundError",
     "scipy.fft module added in SciPy 1.4", "introduction", ""),
    ("np_default_rng", r"np\.random\.default_rng\b", "numpy", "1.16.6", "1.26.4", "AttributeError",
     "np.random.default_rng added in NumPy 1.17", "introduction", ""),
    ("skl_OHE_sparse", r"OneHotEncoder\([^)]*\bsparse\s*=", "scikit-learn", "1.6.0", "1.1.3", "TypeError",
     "OneHotEncoder sparse= deprecated 1.2; verify whether removed in 1.6", "removal", "needs_runtime_verify"),
]

LIB_KEYS = {
    "numpy": ["numpy"],
    "pandas": ["pandas"],
    "scikit-learn": ["sklearn", "scikit-learn"],
    "scipy": ["scipy"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "Pillow": ["pillow", "pil"],
    "flask": ["flask"],
}


# Snapshots of mutually-compatible library versions for env construction.
# Each case's bad_env / good_env is built from one snapshot + the case's own
# bad_version / good_version override on the library_under_test, so we control
# all peer versions instead of letting uv pull in new versions that may
# (a) introduce a *second* breaking change confounding the test, or
# (b) be incompatible with the pinned bad/good version.

LEGACY_2019_SNAPSHOT = {  # ~late 2019 — used only for scipy_fft (scipy 1.3.3 needs Python 3.7)
    "numpy": "1.17.5",
    "pandas": "0.25.3",
    "scipy": "1.3.3",
    "matplotlib": "3.1.3",
    "seaborn": "0.10.1",
    "scikit-learn": "0.22.2.post1",
    "Pillow": "7.0.0",
    "flask": "1.1.4",
    "requests": "2.22.0",
    "beautifulsoup4": "4.8.2",
}

LEGACY_2020_SNAPSHOT = {  # ~mid 2020 — used for sns_histplot/sns_displot intro (Python 3.8)
    "numpy": "1.18.5",
    "pandas": "1.0.5",
    "scipy": "1.4.1",
    "matplotlib": "3.2.2",
    "seaborn": "0.10.1",
    "scikit-learn": "0.23.2",
    "Pillow": "7.2.0",
    "flask": "2.0.3",
    "requests": "2.25.1",
    "beautifulsoup4": "4.9.3",
}

LEGACY_RECENT_SNAPSHOT = {  # ~early 2023 — Python 3.11 compatible, used for removal good_env
    "numpy": "1.24.4",
    "pandas": "1.5.3",
    "scipy": "1.10.1",
    "matplotlib": "3.7.5",
    "seaborn": "0.12.2",
    "scikit-learn": "1.1.3",
    "Pillow": "9.5.0",
    "flask": "2.2.5",
    "requests": "2.31.0",
    "beautifulsoup4": "4.12.2",
}

CURRENT_SNAPSHOT = {  # ~2024 — latest stable, Python 3.11 compatible
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "scipy": "1.13.1",
    "matplotlib": "3.8.4",
    "seaborn": "0.13.2",
    "scikit-learn": "1.4.2",
    "Pillow": "10.4.0",
    "flask": "3.0.3",
    "requests": "2.32.3",
    "beautifulsoup4": "4.12.3",
}

# Per-rule env spec: (bad_snapshot, good_snapshot, bad_python, good_python).
# Rules not in this map fall through to direction-based defaults below.
RULE_ENV_SPEC: dict[tuple[str, str], tuple[dict, dict, str, str]] = {
    # Introduction direction — needs ancient peers, hence old Python
    ("sns_histplot",  "introduction"): (LEGACY_2020_SNAPSHOT, CURRENT_SNAPSHOT, "3.8",  "3.11"),
    ("sns_displot",   "introduction"): (LEGACY_2020_SNAPSHOT, CURRENT_SNAPSHOT, "3.8",  "3.11"),
    ("scipy_fft",     "introduction"): (LEGACY_2019_SNAPSHOT, CURRENT_SNAPSHOT, "3.7",  "3.11"),
}

# Defaults: removal cases — bad=current, good=2023-era; both Python 3.11
DEFAULT_REMOVAL_SPEC = (CURRENT_SNAPSHOT, LEGACY_RECENT_SNAPSHOT, "3.11", "3.11")
DEFAULT_INTRO_SPEC = (LEGACY_RECENT_SNAPSHOT, CURRENT_SNAPSHOT, "3.11", "3.11")

# Transitive ABI deps — pandas wheels are compiled against a specific numpy
# range, so when we pin pandas we must also pin numpy from the same snapshot
# (otherwise uv pulls latest numpy → ABI mismatch at import).
TRANSITIVE_DEPS = {
    "pandas":       ["numpy"],
    "scipy":        ["numpy"],
    "matplotlib":   ["numpy"],
    "scikit-learn": ["numpy", "scipy"],
    "seaborn":      ["numpy", "pandas", "matplotlib"],
}

# task.libs raw name → canonical pip distribution name (subset that needs renaming)
LIB_NAME_TO_PIP = {
    "sklearn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "pil": "Pillow",
    "pillow": "Pillow",
    "image": "Pillow",
    "bs4": "beautifulsoup4",
    "beautifulsoup4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "yaml": "PyYAML",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
}

# Names that are part of the Python standard library — never install via pip.
STDLIB_MODULES = {
    "abc", "argparse", "array", "ast", "asyncio", "base64", "binascii",
    "bisect", "bz2", "calendar", "codecs", "collections", "concurrent",
    "contextlib", "copy", "csv", "ctypes", "dataclasses", "datetime",
    "decimal", "difflib", "doctest", "email", "enum", "errno", "fractions",
    "functools", "gc", "getopt", "getpass", "gettext", "glob", "gzip",
    "hashlib", "heapq", "hmac", "html", "http", "importlib", "inspect",
    "io", "ipaddress", "itertools", "json", "keyword", "locale", "logging",
    "lzma", "math", "mimetypes", "multiprocessing", "operator", "os",
    "pathlib", "pickle", "platform", "pprint", "queue", "random", "re",
    "secrets", "select", "shutil", "signal", "smtplib", "socket", "sqlite3",
    "ssl", "stat", "statistics", "string", "struct", "subprocess", "sys",
    "tarfile", "tempfile", "textwrap", "threading", "time", "timeit",
    "traceback", "types", "typing", "unittest", "urllib", "uuid", "warnings",
    "weakref", "xml", "zipfile", "zoneinfo",
}


def _resolve_pip_name(lib: str) -> str | None:
    """Return canonical pip distribution name, or None if stdlib / unknown."""
    norm = str(lib).lower()
    if norm in STDLIB_MODULES:
        return None
    # Known rename map
    if norm in LIB_NAME_TO_PIP:
        return LIB_NAME_TO_PIP[norm]
    # Default: assume the task.libs name == pip name (case-insensitive)
    return norm


def _expand_libs(task_libs: list[str], lib_under_test: str) -> set[str]:
    """Expand task libs with transitive ABI deps (e.g. pandas → numpy).

    Returns canonical pip names. Stdlib entries are dropped here.
    """
    expanded: set[str] = set()
    seen: set[str] = set()

    def _add_with_deps(name: str) -> None:
        if not name or name in seen:
            return
        seen.add(name)
        expanded.add(name)
        for dep in TRANSITIVE_DEPS.get(name, []):
            _add_with_deps(dep)

    for lib in task_libs:
        norm = _resolve_pip_name(lib)
        if norm:
            _add_with_deps(norm)
    _add_with_deps(lib_under_test)
    return expanded


def _make_env(snapshot: dict[str, str], task_libs: list[str],
              lib_under_test: str, version_override: str) -> list[str]:
    """Build a list of pip-spec strings for one env.

    For each task lib (and ABI-transitive deps): pin to snapshot version if
    listed there; otherwise install unpinned (third-party libs we don't
    control, e.g. regex, pytz). Stdlib is skipped. lib_under_test is always
    pinned to version_override.
    """
    pkgs: dict[str, str] = {}
    for name in _expand_libs(task_libs, lib_under_test):
        pkgs[name] = snapshot.get(name, "")  # "" = unpinned
    pkgs[lib_under_test] = version_override
    return [
        f"{name}=={ver}" if ver else name
        for name, ver in sorted(pkgs.items())
    ]


def _bad_env_for(rule_label: str, direction: str, task_libs: list[str],
                 lib_under_test: str, bad_version: str) -> tuple[list[str], str]:
    """Pick per-rule bad-env snapshot + Python version, return (pip_lines, python)."""
    spec = RULE_ENV_SPEC.get((rule_label, direction))
    if spec is None:
        spec = DEFAULT_INTRO_SPEC if direction == "introduction" else DEFAULT_REMOVAL_SPEC
    bad_snap, _good_snap, bad_python, _good_python = spec
    bad_env = _make_env(bad_snap, task_libs, lib_under_test, bad_version)
    return bad_env, bad_python


def search_bigcodebench() -> list[dict]:
    print("Loading BigCodeBench v0.1.4 …", flush=True)
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    print(f"  loaded {len(ds)} tasks", flush=True)

    hits: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for t in ds:
        canon = t.get("canonical_solution") or ""
        code_prompt = t.get("code_prompt") or ""
        full = code_prompt + "\n" + canon
        try:
            libs = ast.literal_eval(t.get("libs") or "[]")
        except Exception:
            libs = []
        libs_lc = [str(x).lower() for x in libs]

        for label, regex, lib, bad, good, err, reason, kind, note in PATTERNS:
            if label in SKIP_RULES:
                continue
            if not any(any(k in l for k in LIB_KEYS[lib]) for l in libs_lc):
                continue
            m = re.search(regex, full)
            if not m:
                continue
            if t["task_id"] in DROP_TASK_IDS:
                continue
            key = (t["task_id"], label)
            if key in seen:
                continue
            seen.add(key)
            start = full.rfind("\n", 0, m.start()) + 1
            end = full.find("\n", m.end())
            if end == -1:
                end = len(full)
            evidence = full[start:end].strip()[:200]

            effective_note = NEEDS_RUNTIME_VERIFY.get(t["task_id"], note)
            bad_env_pip, bad_python = _bad_env_for(label, kind, libs, lib, bad)

            canonical = t.get("canonical_solution") or ""
            try:
                correct_solution = apply_correction(t["task_id"], canonical)
            except (KeyError, ValueError) as e:
                raise RuntimeError(
                    f"Missing or invalid correction for {t['task_id']} ({label}): {e}"
                )

            hits.append({
                "task_id": t["task_id"],
                "libs": libs,
                "library_under_test": lib,
                "bad_version": bad,
                "good_version": good,  # documentation only — no good_env built
                "bad_env_pip": bad_env_pip,
                "bad_python": bad_python,
                "error_type": err,
                "kind": kind,
                "rule_label": label,
                "reason": reason,
                "evidence_line": evidence,
                "note": effective_note,
                "instruct_prompt": t.get("instruct_prompt") or "",
                "code_prompt": t.get("code_prompt") or "",
                "canonical_solution": canonical,
                "correct_solution": correct_solution,
                "test": t.get("test") or "",
                "entry_point": t.get("entry_point") or "task_func",
                "verified": False,
            })
    return hits


def main() -> None:
    bcb_hits = search_bigcodebench()
    print(f"BigCodeBench hits: {len(bcb_hits)}", flush=True)

    candidates: list[dict] = []

    # 1) BCB hits, with sequential case_ids
    for i, h in enumerate(bcb_hits, start=1):
        candidates.append({"case_id": f"bcb_{i:03d}", **h})

    # 2) Manual cases (already have case_id like manual_001) — compute envs too
    for c in MANUAL_CASES:
        c = dict(c)
        bad_env_pip, bad_python = _bad_env_for(
            c["rule_label"], c["kind"], c["libs"], c["library_under_test"],
            c["bad_version"],
        )
        c["bad_env_pip"] = bad_env_pip
        c["bad_python"] = bad_python
        if "correct_solution" not in c:
            raise RuntimeError(
                f"Manual case {c['case_id']} is missing 'correct_solution' field"
            )
        candidates.append(c)

    OUT_PATH.write_text(json.dumps(candidates, indent=2))

    print(f"\nWrote {len(candidates)} cases to {OUT_PATH}")
    print(f"  bcb     : {sum(1 for c in candidates if c['case_id'].startswith('bcb_'))}")
    print(f"  manual  : {sum(1 for c in candidates if c['case_id'].startswith('manual_'))}")
    print()
    print("=== Distribution ===")
    print(f"  by library : {dict(Counter(c['library_under_test'] for c in candidates))}")
    print(f"  by kind    : {dict(Counter(c['kind'] for c in candidates))}")
    print(f"  by error   : {dict(Counter(c['error_type'] for c in candidates))}")
    print(f"  by rule    : {dict(Counter(c['rule_label'] for c in candidates))}")
    print(f"  needs_verify: {sum(1 for c in candidates if c.get('note'))}")


if __name__ == "__main__":
    main()
