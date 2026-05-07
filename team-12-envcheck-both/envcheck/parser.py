"""
EnvCheck AST Parser — Extract imports, attribute accesses, and method calls.

Uses Python's ast module to parse source files and extract structured
information about:
  1. Import statements (import X, from X import Y)
  2. Attribute access patterns (module.attr)
  3. Method/function calls (obj.method(...), including keyword args)

This module is purely a parser — it does NOT do any matching against
the knowledge base. That's the scanner's job.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ImportInfo:
    """An import statement found in source code."""
    module: str              # e.g., "scipy.integrate"
    name: str                # e.g., "cumtrapz" (for `from X import Y`)
    alias: Optional[str]     # e.g., "np" (for `import numpy as np`)
    lineno: int
    col_offset: int
    is_from_import: bool     # True for `from X import Y`, False for `import X`


@dataclass
class AttributeAccess:
    """An attribute access found in source code (e.g., np.trapz)."""
    object_name: str         # e.g., "np" — the variable/module being accessed
    attribute: str           # e.g., "trapz" — the attribute being accessed
    lineno: int
    col_offset: int
    full_chain: str          # e.g., "np.trapz" — reconstructed dotted name


@dataclass
class MethodCall:
    """A method call found in source code (e.g., df.fillna(method='ffill'))."""
    object_name: str         # e.g., "df" — the variable the method is called on
    method_name: str         # e.g., "fillna"
    lineno: int
    col_offset: int
    keyword_args: dict[str, Optional[str]]  # e.g., {"method": "ffill"}
    positional_arg_count: int


@dataclass
class ParseResult:
    """Complete parse result for a single source file."""
    filepath: str
    imports: list[ImportInfo] = field(default_factory=list)
    attribute_accesses: list[AttributeAccess] = field(default_factory=list)
    method_calls: list[MethodCall] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)

    # Mapping from alias to module name (e.g., {"np": "numpy", "pd": "pandas"})
    alias_map: dict[str, str] = field(default_factory=dict)


def _get_dotted_name(node: ast.AST) -> Optional[str]:
    """Reconstruct a dotted name from an AST node (e.g., np.linalg.solve → 'np.linalg.solve')."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        parent = _get_dotted_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
    return None


def _get_root_name(node: ast.AST) -> Optional[str]:
    """Get the root (leftmost) name from a dotted attribute chain or subscript."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return _get_root_name(node.value)
    elif isinstance(node, ast.Subscript):
        # Handle df["A"].method() — root is still "df"
        return _get_root_name(node.value)
    elif isinstance(node, ast.Call):
        # Handle chained calls like obj.method1().method2()
        return _get_root_name(node.func)
    return None


def _get_keyword_value(node: ast.keyword) -> Optional[str]:
    """Extract a string representation of a keyword argument value."""
    if isinstance(node.value, ast.Constant):
        return str(node.value.value)
    elif isinstance(node.value, ast.Name):
        return node.value.id
    return None


class _SourceVisitor(ast.NodeVisitor):
    """AST visitor that collects imports, attribute accesses, and method calls."""

    def __init__(self):
        self.imports: list[ImportInfo] = []
        self.attribute_accesses: list[AttributeAccess] = []
        self.method_calls: list[MethodCall] = []
        self.alias_map: dict[str, str] = {}

    def visit_Import(self, node: ast.Import):
        """Handle `import X` and `import X as Y`."""
        for alias in node.names:
            info = ImportInfo(
                module=alias.name,
                name=alias.name,
                alias=alias.asname,
                lineno=node.lineno,
                col_offset=node.col_offset,
                is_from_import=False,
            )
            self.imports.append(info)
            # Track alias mapping: `import numpy as np` → {"np": "numpy"}
            effective_name = alias.asname if alias.asname else alias.name
            self.alias_map[effective_name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Handle `from X import Y` and `from X import Y as Z`."""
        module = node.module or ""
        for alias in node.names:
            info = ImportInfo(
                module=module,
                name=alias.name,
                alias=alias.asname,
                lineno=node.lineno,
                col_offset=node.col_offset,
                is_from_import=True,
            )
            self.imports.append(info)
            # Track: `from sklearn.datasets import load_boston` → {"load_boston": "sklearn.datasets.load_boston"}
            effective_name = alias.asname if alias.asname else alias.name
            self.alias_map[effective_name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Handle attribute access like `np.trapz`, `nx.write_gpickle`."""
        full_chain = _get_dotted_name(node)
        root_name = _get_root_name(node)
        if full_chain and root_name:
            access = AttributeAccess(
                object_name=root_name,
                attribute=node.attr,
                lineno=node.lineno,
                col_offset=node.col_offset,
                full_chain=full_chain,
            )
            self.attribute_accesses.append(access)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Handle function/method calls like `df.fillna(method='ffill')`."""
        if isinstance(node.func, ast.Attribute):
            root_name = _get_root_name(node.func.value)
            if root_name:
                kwargs = {}
                for kw in node.keywords:
                    if kw.arg:  # Skip **kwargs
                        kwargs[kw.arg] = _get_keyword_value(kw)

                call = MethodCall(
                    object_name=root_name,
                    method_name=node.func.attr,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    keyword_args=kwargs,
                    positional_arg_count=len(node.args),
                )
                self.method_calls.append(call)
        self.generic_visit(node)


def parse_source(source_code: str, filepath: str = "<string>") -> ParseResult:
    """Parse a Python source string and extract all relevant code patterns.

    Args:
        source_code: The Python source code to parse.
        filepath: The path to the source file (for error reporting).

    Returns:
        ParseResult with all extracted imports, accesses, and calls.
    """
    result = ParseResult(filepath=filepath)

    try:
        tree = ast.parse(source_code, filename=filepath)
    except SyntaxError as e:
        result.parse_errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
        return result

    visitor = _SourceVisitor()
    visitor.visit(tree)

    result.imports = visitor.imports
    result.attribute_accesses = visitor.attribute_accesses
    result.method_calls = visitor.method_calls
    result.alias_map = visitor.alias_map

    return result


def parse_file(filepath: str | Path) -> ParseResult:
    """Parse a Python file and extract all relevant code patterns.

    Args:
        filepath: Path to the .py file to parse.

    Returns:
        ParseResult with all extracted imports, accesses, and calls.
    """
    filepath = Path(filepath)
    try:
        source_code = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        result = ParseResult(filepath=str(filepath))
        result.parse_errors.append(f"Could not read file: {e}")
        return result

    return parse_source(source_code, filepath=str(filepath))
