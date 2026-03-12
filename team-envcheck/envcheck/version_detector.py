"""
EnvCheck Version Detector — Read installed package versions from a target environment.

Given a path to a Python virtual environment, this module can:
  1. List all installed packages and their versions
  2. Look up the version of a specific package
  3. Compare versions against breaking change boundaries

Uses importlib.metadata via the target environment's site-packages.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Mapping from PyPI package names to importable module names
# (only needed when they differ)
PACKAGE_TO_MODULE: dict[str, str] = {
    "scikit-learn": "sklearn",
    "pillow": "PIL",
    "python-dateutil": "dateutil",
    "pyyaml": "yaml",
    "beautifulsoup4": "bs4",
}

# Reverse: module names to PyPI package names
MODULE_TO_PACKAGE: dict[str, str] = {v: k for k, v in PACKAGE_TO_MODULE.items()}
# Also add common identity mappings
MODULE_TO_PACKAGE.update({
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "networkx": "networkx",
    "pydantic": "pydantic",
    "sklearn": "scikit-learn",
})


@dataclass
class InstalledPackage:
    """Information about an installed package."""
    name: str        # PyPI name (e.g., "scikit-learn")
    version: str     # Version string (e.g., "1.8.0")


def get_installed_packages(env_path: str | Path) -> dict[str, InstalledPackage]:
    """Get all installed packages in a virtual environment.

    Args:
        env_path: Path to the virtual environment root (e.g., ./environments/case_numpy_2x)

    Returns:
        Dictionary mapping normalized package names to InstalledPackage objects.
    """
    env_path = Path(env_path)
    python_bin = env_path / "bin" / "python"

    if not python_bin.exists():
        return {}

    # Use pip list --format=json via the environment's Python
    result = subprocess.run(
        [str(python_bin), "-m", "pip", "list", "--format=json"],
        capture_output=True, text=True, timeout=30,
    )

    if result.returncode != 0:
        # Fallback: use importlib.metadata directly
        return _get_packages_via_metadata(python_bin)

    import json
    try:
        packages = json.loads(result.stdout)
    except json.JSONDecodeError:
        return _get_packages_via_metadata(python_bin)

    installed = {}
    for pkg in packages:
        name = pkg["name"].lower()
        installed[name] = InstalledPackage(
            name=pkg["name"],
            version=pkg["version"],
        )
    return installed


def _get_packages_via_metadata(python_bin: Path) -> dict[str, InstalledPackage]:
    """Fallback: get packages using importlib.metadata in the target environment."""
    script = (
        "import importlib.metadata, json; "
        "pkgs = [{'name': d.metadata['Name'], 'version': d.metadata['Version']} "
        "for d in importlib.metadata.distributions()]; "
        "print(json.dumps(pkgs))"
    )
    result = subprocess.run(
        [str(python_bin), "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return {}

    import json
    try:
        packages = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}

    installed = {}
    for pkg in packages:
        name = pkg["name"].lower()
        installed[name] = InstalledPackage(
            name=pkg["name"],
            version=pkg["version"],
        )
    return installed


def get_package_version(env_path: str | Path, package_name: str) -> Optional[str]:
    """Get the version of a specific package in the environment.

    Args:
        env_path: Path to the virtual environment
        package_name: PyPI package name (e.g., "numpy", "scikit-learn")

    Returns:
        Version string, or None if not installed.
    """
    packages = get_installed_packages(env_path)
    normalized = package_name.lower().replace("-", "-")
    pkg = packages.get(normalized)
    return pkg.version if pkg else None


def resolve_module_to_package(module_name: str) -> str:
    """Map an importable module name to its PyPI package name.

    Args:
        module_name: The top-level module name (e.g., "sklearn", "numpy")

    Returns:
        The PyPI package name (e.g., "scikit-learn", "numpy")
    """
    # Check explicit mapping
    top_level = module_name.split(".")[0]
    if top_level in MODULE_TO_PACKAGE:
        return MODULE_TO_PACKAGE[top_level]
    # Default: assume module name == package name
    return top_level


def compare_versions(installed: str, boundary: str) -> int:
    """Compare two version strings.

    Returns:
        -1 if installed < boundary
         0 if installed == boundary
         1 if installed > boundary
    """
    from packaging.version import Version, InvalidVersion

    try:
        v_installed = Version(installed)
        v_boundary = Version(boundary)
    except InvalidVersion:
        # Fallback to basic string comparison
        return (installed > boundary) - (installed < boundary)

    if v_installed < v_boundary:
        return -1
    elif v_installed > v_boundary:
        return 1
    return 0


def is_version_affected(installed_version: str, removed_in: str) -> bool:
    """Check if an installed version is affected by a breaking change.

    A version is affected if it is >= the version where the API was removed.

    Args:
        installed_version: The installed version string
        removed_in: The version where the API was removed (from knowledge base)

    Returns:
        True if the installed version is affected (API was removed).
    """
    if removed_in == "N/A":
        # Special case: this rule always applies (reverse-compat)
        # The scanner will need to check version direction differently
        return True

    return compare_versions(installed_version, removed_in) >= 0
