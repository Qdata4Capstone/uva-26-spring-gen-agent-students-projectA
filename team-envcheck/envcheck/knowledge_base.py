"""
EnvCheck Knowledge Base — Registry of known breaking API changes.

Each entry describes a specific API that was removed/renamed in a library version,
along with pattern matching rules for detecting usage in source code.

The knowledge base is designed to be extensible: add new entries by appending
to the BREAKING_CHANGES list or calling register_breaking_change().

Pattern types:
  - "attribute": Matches module.attribute access (e.g., np.trapz)
  - "import": Matches import statements (e.g., from scipy.integrate import cumtrapz)
  - "method_call": Matches method calls on objects (e.g., df.fillna(method=...))
  - "method_access": Matches method/attribute access on objects (e.g., df.mad())
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PatternType(Enum):
    """How the scanner should look for this breaking change in source code."""
    ATTRIBUTE = "attribute"       # module.attr (e.g., np.trapz)
    IMPORT = "import"             # from module import name
    METHOD_CALL = "method_call"   # obj.method(specific_kwarg=...)
    METHOD_ACCESS = "method_access"  # obj.method() — method doesn't exist


class Severity(Enum):
    """How severe this breaking change is."""
    ERROR = "error"      # Will definitely crash at runtime
    WARNING = "warning"  # May crash depending on usage


@dataclass
class BreakingChangeRule:
    """A single rule describing a breaking API change.

    Attributes:
        rule_id: Unique identifier for this rule
        library: PyPI package name (e.g., "numpy", "pandas")
        removed_in: Version where the API was removed (e.g., "2.0.0")
        pattern_type: How to detect this in source code
        module_path: The module path for imports (e.g., "numpy", "scipy.integrate")
        symbol: The function/attribute name that was removed (e.g., "trapz")
        old_api: Human-readable old API usage (e.g., "np.trapz(y, x)")
        new_api: Human-readable replacement (e.g., "np.trapezoid(y, x)")
        error_type: Expected error type (e.g., "AttributeError")
        description: Human-readable explanation
        severity: How severe this is
        method_kwargs: For METHOD_CALL type — specific kwargs that trigger the break
        base_type_hint: For METHOD_CALL/METHOD_ACCESS — hint about what type the object is
    """
    rule_id: str
    library: str
    removed_in: str
    pattern_type: PatternType
    module_path: str
    symbol: str
    old_api: str
    new_api: str
    error_type: str
    description: str
    severity: Severity = Severity.ERROR
    method_kwargs: Optional[dict] = None
    base_type_hint: Optional[str] = None


# =============================================================================
# KNOWLEDGE BASE — All known breaking changes
# =============================================================================

BREAKING_CHANGES: list[BreakingChangeRule] = []


def register_breaking_change(rule: BreakingChangeRule) -> None:
    """Register a new breaking change rule in the knowledge base."""
    BREAKING_CHANGES.append(rule)


def get_rules_for_library(library: str) -> list[BreakingChangeRule]:
    """Get all breaking change rules for a specific library."""
    return [r for r in BREAKING_CHANGES if r.library == library]


def get_all_libraries() -> set[str]:
    """Get all library names that have registered rules."""
    return {r.library for r in BREAKING_CHANGES}


# =============================================================================
# NumPy 2.0 breaking changes
# =============================================================================
register_breaking_change(BreakingChangeRule(
    rule_id="numpy-trapz-removed",
    library="numpy",
    removed_in="2.0.0",
    pattern_type=PatternType.ATTRIBUTE,
    module_path="numpy",
    symbol="trapz",
    old_api="np.trapz(y, x)",
    new_api="np.trapezoid(y, x)",
    error_type="AttributeError",
    description="np.trapz was removed in NumPy 2.0. Use np.trapezoid instead.",
))

register_breaking_change(BreakingChangeRule(
    rule_id="numpy-infty-removed",
    library="numpy",
    removed_in="2.0.0",
    pattern_type=PatternType.ATTRIBUTE,
    module_path="numpy",
    symbol="infty",
    old_api="np.infty",
    new_api="np.inf",
    error_type="AttributeError",
    description="np.infty alias was removed in NumPy 2.0. Use np.inf instead.",
))

register_breaking_change(BreakingChangeRule(
    rule_id="numpy-bool-removed",
    library="numpy",
    removed_in="2.0.0",
    pattern_type=PatternType.ATTRIBUTE,
    module_path="numpy",
    symbol="bool",
    old_api="np.bool",
    new_api="np.bool_",
    error_type="AttributeError",
    description="np.bool was removed in NumPy 2.0. Use np.bool_ or Python bool.",
))

register_breaking_change(BreakingChangeRule(
    rule_id="numpy-int-removed",
    library="numpy",
    removed_in="2.0.0",
    pattern_type=PatternType.ATTRIBUTE,
    module_path="numpy",
    symbol="int",
    old_api="np.int",
    new_api="np.int_",
    error_type="AttributeError",
    description="np.int was removed in NumPy 2.0. Use np.int_ or Python int.",
))

register_breaking_change(BreakingChangeRule(
    rule_id="numpy-float-removed",
    library="numpy",
    removed_in="2.0.0",
    pattern_type=PatternType.ATTRIBUTE,
    module_path="numpy",
    symbol="float",
    old_api="np.float",
    new_api="np.float64",
    error_type="AttributeError",
    description="np.float was removed in NumPy 2.0. Use np.float64 or Python float.",
))

register_breaking_change(BreakingChangeRule(
    rule_id="numpy-complex-removed",
    library="numpy",
    removed_in="2.0.0",
    pattern_type=PatternType.ATTRIBUTE,
    module_path="numpy",
    symbol="complex",
    old_api="np.complex",
    new_api="np.complex128",
    error_type="AttributeError",
    description="np.complex was removed in NumPy 2.0. Use np.complex128 or Python complex.",
))

register_breaking_change(BreakingChangeRule(
    rule_id="numpy-object-removed",
    library="numpy",
    removed_in="2.0.0",
    pattern_type=PatternType.ATTRIBUTE,
    module_path="numpy",
    symbol="object",
    old_api="np.object",
    new_api="np.object_",
    error_type="AttributeError",
    description="np.object was removed in NumPy 2.0. Use np.object_ or Python object.",
))

register_breaking_change(BreakingChangeRule(
    rule_id="numpy-str-removed",
    library="numpy",
    removed_in="2.0.0",
    pattern_type=PatternType.ATTRIBUTE,
    module_path="numpy",
    symbol="str",
    old_api="np.str",
    new_api="np.str_",
    error_type="AttributeError",
    description="np.str was removed in NumPy 2.0. Use np.str_ or Python str.",
))

# =============================================================================
# SciPy 1.14 breaking changes
# =============================================================================
register_breaking_change(BreakingChangeRule(
    rule_id="scipy-cumtrapz-removed",
    library="scipy",
    removed_in="1.14.0",
    pattern_type=PatternType.IMPORT,
    module_path="scipy.integrate",
    symbol="cumtrapz",
    old_api="from scipy.integrate import cumtrapz",
    new_api="from scipy.integrate import cumulative_trapezoid",
    error_type="ImportError",
    description="cumtrapz was removed in SciPy 1.14. Use cumulative_trapezoid.",
))

register_breaking_change(BreakingChangeRule(
    rule_id="scipy-simps-removed",
    library="scipy",
    removed_in="1.14.0",
    pattern_type=PatternType.IMPORT,
    module_path="scipy.integrate",
    symbol="simps",
    old_api="from scipy.integrate import simps",
    new_api="from scipy.integrate import simpson",
    error_type="ImportError",
    description="simps was removed in SciPy 1.14. Use simpson.",
))

# =============================================================================
# scikit-learn 1.2 breaking changes
# =============================================================================
register_breaking_change(BreakingChangeRule(
    rule_id="sklearn-load-boston-removed",
    library="scikit-learn",
    removed_in="1.2.0",
    pattern_type=PatternType.IMPORT,
    module_path="sklearn.datasets",
    symbol="load_boston",
    old_api="from sklearn.datasets import load_boston",
    new_api="from sklearn.datasets import fetch_california_housing",
    error_type="ImportError",
    description="load_boston was removed in scikit-learn 1.2 due to ethical concerns. Use fetch_california_housing.",
))

# =============================================================================
# Pandas 2.0+ breaking changes
# =============================================================================
register_breaking_change(BreakingChangeRule(
    rule_id="pandas-fillna-method-removed",
    library="pandas",
    removed_in="2.2.0",
    pattern_type=PatternType.METHOD_CALL,
    module_path="pandas",
    symbol="fillna",
    old_api='df.fillna(method="ffill")',
    new_api="df.ffill()",
    error_type="TypeError",
    description="fillna(method=...) keyword removed in pandas 2.2. Use df.ffill() or df.bfill() directly.",
    method_kwargs={"method": None},  # Any value of 'method' kwarg triggers this
    base_type_hint="pandas.DataFrame",
))

register_breaking_change(BreakingChangeRule(
    rule_id="pandas-mad-removed",
    library="pandas",
    removed_in="2.0.0",
    pattern_type=PatternType.METHOD_ACCESS,
    module_path="pandas",
    symbol="mad",
    old_api='df["col"].mad()',
    new_api='(df["col"] - df["col"].mean()).abs().mean()',
    error_type="AttributeError",
    description=".mad() was removed in pandas 2.0. Compute manually.",
    base_type_hint="pandas.DataFrame",
))

# =============================================================================
# NetworkX 3.0 breaking changes
# =============================================================================
register_breaking_change(BreakingChangeRule(
    rule_id="networkx-write-gpickle-removed",
    library="networkx",
    removed_in="3.0.0",
    pattern_type=PatternType.ATTRIBUTE,
    module_path="networkx",
    symbol="write_gpickle",
    old_api='nx.write_gpickle(G, "file.gpickle")',
    new_api='pickle.dump(G, open("file.pickle", "wb"))',
    error_type="AttributeError",
    description="write_gpickle removed in NetworkX 3.0 due to security concerns. Use standard pickle.",
))

register_breaking_change(BreakingChangeRule(
    rule_id="networkx-read-gpickle-removed",
    library="networkx",
    removed_in="3.0.0",
    pattern_type=PatternType.ATTRIBUTE,
    module_path="networkx",
    symbol="read_gpickle",
    old_api='nx.read_gpickle("file.gpickle")',
    new_api='pickle.load(open("file.pickle", "rb"))',
    error_type="AttributeError",
    description="read_gpickle removed in NetworkX 3.0 due to security concerns. Use standard pickle.",
))

# =============================================================================
# Pandas reverse-compat: APIs that DON'T exist in old versions
# =============================================================================
register_breaking_change(BreakingChangeRule(
    rule_id="pandas-map-not-in-1x",
    library="pandas",
    removed_in="N/A",  # Not removed — it was ADDED in 2.1
    pattern_type=PatternType.METHOD_ACCESS,
    module_path="pandas",
    symbol="map",
    old_api="df.map(func)  # Only works in pandas >= 2.1",
    new_api="df.applymap(func)  # Works in pandas < 2.1",
    error_type="AttributeError",
    description="DataFrame.map() was added in pandas 2.1. In pandas 1.x, only applymap() exists.",
    base_type_hint="pandas.DataFrame",
))

# =============================================================================
# Pydantic V1 vs V2
# =============================================================================
register_breaking_change(BreakingChangeRule(
    rule_id="pydantic-model-dump-not-in-v1",
    library="pydantic",
    removed_in="N/A",  # Not removed — model_dump() was ADDED in V2
    pattern_type=PatternType.METHOD_ACCESS,
    module_path="pydantic",
    symbol="model_dump",
    old_api="user.model_dump()  # Only works in pydantic >= 2.0",
    new_api="user.dict()  # Works in pydantic < 2.0",
    error_type="AttributeError",
    description="Pydantic V2 introduced model_dump() to replace dict(). In V1, only .dict() exists.",
    base_type_hint="pydantic.BaseModel",
))

register_breaking_change(BreakingChangeRule(
    rule_id="pydantic-model-validate-not-in-v1",
    library="pydantic",
    removed_in="N/A",
    pattern_type=PatternType.METHOD_ACCESS,
    module_path="pydantic",
    symbol="model_validate",
    old_api="User.model_validate(data)  # Only works in pydantic >= 2.0",
    new_api="User.parse_obj(data)  # Works in pydantic < 2.0",
    error_type="AttributeError",
    description="Pydantic V2 introduced model_validate() to replace parse_obj(). In V1, only parse_obj() exists.",
    base_type_hint="pydantic.BaseModel",
))
