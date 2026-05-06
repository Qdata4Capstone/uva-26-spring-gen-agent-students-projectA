"""
EnvCheck Scanner — Core diagnostic engine.

Ties together the parser, version detector, and knowledge base to produce
a structured diagnostic report for a set of Python source files against
a target environment.

Usage:
    from envcheck.scanner import scan_file, scan_project

    report = scan_file("test.py", env_path="./environments/case_numpy_2x/")
    print(report)

    report = scan_project(source_dir="./test_cases/generated/", env_path="./environments/case_numpy_2x/")
    print(report)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from envcheck.knowledge_base import (
    BREAKING_CHANGES,
    BreakingChangeRule,
    PatternType,
    Severity,
)
from envcheck.parser import (
    AttributeAccess,
    ImportInfo,
    MethodCall,
    ParseResult,
    parse_file,
    parse_source,
)
from envcheck.version_detector import (
    get_installed_packages,
    is_version_affected,
    resolve_module_to_package,
    InstalledPackage,
)


@dataclass
class Finding:
    """A single diagnostic finding — a detected breaking change in source code."""
    filepath: str
    lineno: int
    col_offset: int
    rule: BreakingChangeRule
    matched_code: str          # The actual code pattern that matched
    installed_version: str     # The version installed in the environment
    severity: Severity = Severity.ERROR

    def __str__(self) -> str:
        return (
            f"{self.filepath}:{self.lineno} [{self.severity.value.upper()}] "
            f"{self.rule.rule_id}: {self.matched_code}\n"
            f"  {self.rule.description}\n"
            f"  Old: {self.rule.old_api}\n"
            f"  New: {self.rule.new_api}\n"
            f"  Installed: {self.rule.library} {self.installed_version} "
            f"(removed in {self.rule.removed_in})"
        )


@dataclass
class ScanReport:
    """Complete scan report for one or more source files."""
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    scan_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def total_findings(self) -> int:
        return len(self.findings)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    def __str__(self) -> str:
        lines = []
        lines.append(f"EnvCheck Scan Report")
        lines.append(f"{'=' * 60}")
        lines.append(f"Files scanned: {self.files_scanned}")
        lines.append(f"Findings: {self.total_findings} "
                      f"({self.error_count} errors, {self.warning_count} warnings)")
        lines.append(f"Scan time: {self.scan_time_ms:.1f}ms")

        if self.errors:
            lines.append(f"\nParse errors:")
            for err in self.errors:
                lines.append(f"  ⚠ {err}")

        if self.findings:
            lines.append(f"\nFindings:")
            lines.append(f"{'-' * 60}")
            for finding in self.findings:
                lines.append(str(finding))
                lines.append("")

        if not self.findings:
            lines.append(f"\n✅ No breaking changes detected.")

        return "\n".join(lines)


def _match_attribute_rules(
    parse_result: ParseResult,
    rules: list[BreakingChangeRule],
    packages: dict[str, InstalledPackage],
) -> list[Finding]:
    """Match ATTRIBUTE pattern rules against parsed attribute accesses.

    Looks for patterns like `np.trapz` where `np` is an alias for `numpy`
    and `trapz` is a removed attribute.
    """
    findings = []

    # Build reverse alias map: alias → module name
    # e.g., {"np": "numpy", "nx": "networkx"}
    alias_to_module = {}
    for alias, full_path in parse_result.alias_map.items():
        # For `import numpy as np`: alias_map has {"np": "numpy"}
        # For `from X import Y`: alias_map has {"Y": "X.Y"} — skip these
        parts = full_path.split(".")
        if len(parts) == 1 or alias == parts[0]:
            alias_to_module[alias] = full_path

    for access in parse_result.attribute_accesses:
        # Resolve the object name to its actual module
        resolved_module = alias_to_module.get(access.object_name, access.object_name)

        for rule in rules:
            if rule.pattern_type != PatternType.ATTRIBUTE:
                continue

            # Check if this access matches: module matches AND attribute matches
            if resolved_module == rule.module_path and access.attribute == rule.symbol:
                # Check if the library is installed and the version is affected
                package_name = resolve_module_to_package(rule.module_path).lower()
                pkg = packages.get(package_name)
                if pkg and is_version_affected(pkg.version, rule.removed_in):
                    findings.append(Finding(
                        filepath=parse_result.filepath,
                        lineno=access.lineno,
                        col_offset=access.col_offset,
                        rule=rule,
                        matched_code=access.full_chain,
                        installed_version=pkg.version,
                        severity=rule.severity,
                    ))

    return findings


def _match_import_rules(
    parse_result: ParseResult,
    rules: list[BreakingChangeRule],
    packages: dict[str, InstalledPackage],
) -> list[Finding]:
    """Match IMPORT pattern rules against parsed import statements.

    Looks for patterns like `from scipy.integrate import cumtrapz`.
    """
    findings = []

    for imp in parse_result.imports:
        if not imp.is_from_import:
            continue

        for rule in rules:
            if rule.pattern_type != PatternType.IMPORT:
                continue

            # Check if module and imported name match
            if imp.module == rule.module_path and imp.name == rule.symbol:
                # Check version
                package_name = resolve_module_to_package(rule.module_path).lower()
                pkg = packages.get(package_name)
                if pkg and is_version_affected(pkg.version, rule.removed_in):
                    findings.append(Finding(
                        filepath=parse_result.filepath,
                        lineno=imp.lineno,
                        col_offset=imp.col_offset,
                        rule=rule,
                        matched_code=f"from {imp.module} import {imp.name}",
                        installed_version=pkg.version,
                        severity=rule.severity,
                    ))

    return findings


def _match_method_call_rules(
    parse_result: ParseResult,
    rules: list[BreakingChangeRule],
    packages: dict[str, InstalledPackage],
) -> list[Finding]:
    """Match METHOD_CALL pattern rules against parsed method calls.

    Looks for patterns like `df.fillna(method="ffill")` where a specific
    keyword argument triggers the breaking change.
    """
    findings = []

    for call in parse_result.method_calls:
        for rule in rules:
            if rule.pattern_type != PatternType.METHOD_CALL:
                continue

            # Check if method name matches
            if call.method_name != rule.symbol:
                continue

            # Check if the problematic kwargs are present
            if rule.method_kwargs:
                kwarg_match = any(
                    k in call.keyword_args for k in rule.method_kwargs
                )
                if not kwarg_match:
                    continue

            # Check version
            package_name = rule.library.lower()
            pkg = packages.get(package_name)
            if pkg and is_version_affected(pkg.version, rule.removed_in):
                kwargs_str = ", ".join(f"{k}={v}" for k, v in call.keyword_args.items())
                findings.append(Finding(
                    filepath=parse_result.filepath,
                    lineno=call.lineno,
                    col_offset=call.col_offset,
                    rule=rule,
                    matched_code=f"{call.object_name}.{call.method_name}({kwargs_str})",
                    installed_version=pkg.version,
                    severity=rule.severity,
                ))

    return findings


def _match_method_access_rules(
    parse_result: ParseResult,
    rules: list[BreakingChangeRule],
    packages: dict[str, InstalledPackage],
) -> list[Finding]:
    """Match METHOD_ACCESS pattern rules against parsed method calls and attribute accesses.

    Looks for patterns like `df.mad()` or `user.model_dump()` where the method
    itself doesn't exist in the installed version.
    """
    findings = []

    # Check method calls (obj.method())
    for call in parse_result.method_calls:
        for rule in rules:
            if rule.pattern_type != PatternType.METHOD_ACCESS:
                continue

            if call.method_name != rule.symbol:
                continue

            # For method access rules, we need to infer the object type
            # Heuristic: check if the file imports the relevant library
            package_name = rule.library.lower()
            pkg = packages.get(package_name)
            if not pkg:
                continue

            # Check if the relevant library is imported in this file
            module_top = resolve_module_to_package(rule.module_path)
            lib_imported = any(
                imp.module.split(".")[0] in (rule.module_path, rule.module_path.split(".")[0])
                or imp.name.split(".")[0] in (rule.module_path, rule.module_path.split(".")[0])
                for imp in parse_result.imports
            )
            if not lib_imported:
                continue

            if is_version_affected(pkg.version, rule.removed_in):
                findings.append(Finding(
                    filepath=parse_result.filepath,
                    lineno=call.lineno,
                    col_offset=call.col_offset,
                    rule=rule,
                    matched_code=f"{call.object_name}.{call.method_name}()",
                    installed_version=pkg.version,
                    severity=rule.severity,
                ))

    return findings


def scan_source(
    source_code: str,
    env_path: str | Path,
    filepath: str = "<string>",
) -> ScanReport:
    """Scan a Python source string against a target environment.

    Args:
        source_code: Python source code to scan.
        env_path: Path to the target virtual environment.
        filepath: Source file path (for reporting).

    Returns:
        ScanReport with all detected breaking changes.
    """
    start = time.perf_counter()
    report = ScanReport()

    # Parse the source code
    parse_result = parse_source(source_code, filepath=filepath)
    if parse_result.parse_errors:
        report.errors.extend(parse_result.parse_errors)

    # Get installed packages
    packages = get_installed_packages(env_path)

    # Get all rules
    rules = BREAKING_CHANGES

    # Run all matchers
    report.findings.extend(_match_attribute_rules(parse_result, rules, packages))
    report.findings.extend(_match_import_rules(parse_result, rules, packages))
    report.findings.extend(_match_method_call_rules(parse_result, rules, packages))
    report.findings.extend(_match_method_access_rules(parse_result, rules, packages))

    # Sort findings by line number
    report.findings.sort(key=lambda f: (f.filepath, f.lineno))

    report.files_scanned = 1
    report.scan_time_ms = (time.perf_counter() - start) * 1000

    return report


def scan_file(
    filepath: str | Path,
    env_path: str | Path,
) -> ScanReport:
    """Scan a single Python file against a target environment.

    Args:
        filepath: Path to the .py file to scan.
        env_path: Path to the target virtual environment.

    Returns:
        ScanReport with all detected breaking changes.
    """
    filepath = Path(filepath)
    try:
        source_code = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        report = ScanReport()
        report.errors.append(f"Could not read {filepath}: {e}")
        return report

    return scan_source(source_code, env_path, filepath=str(filepath))


def scan_project(
    source_dir: str | Path,
    env_path: str | Path,
) -> ScanReport:
    """Scan all Python files in a directory against a target environment.

    Args:
        source_dir: Directory containing .py files to scan.
        env_path: Path to the target virtual environment.

    Returns:
        Combined ScanReport for all files.
    """
    start = time.perf_counter()
    source_dir = Path(source_dir)
    combined = ScanReport()

    py_files = sorted(source_dir.glob("**/*.py"))
    if not py_files:
        combined.errors.append(f"No .py files found in {source_dir}")
        return combined

    # Get installed packages once (shared across all files)
    packages = get_installed_packages(env_path)

    for py_file in py_files:
        parse_result = parse_file(str(py_file))
        if parse_result.parse_errors:
            combined.errors.extend(parse_result.parse_errors)

        rules = BREAKING_CHANGES
        combined.findings.extend(_match_attribute_rules(parse_result, rules, packages))
        combined.findings.extend(_match_import_rules(parse_result, rules, packages))
        combined.findings.extend(_match_method_call_rules(parse_result, rules, packages))
        combined.findings.extend(_match_method_access_rules(parse_result, rules, packages))
        combined.files_scanned += 1

    # Sort findings
    combined.findings.sort(key=lambda f: (f.filepath, f.lineno))
    combined.scan_time_ms = (time.perf_counter() - start) * 1000

    return combined
