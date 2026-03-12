"""
EnvCheck Live Demo — Test Case Definitions

Each case represents a scenario where an LLM would generate code using outdated APIs
that crash in the specified environment. EnvCheck should detect these BEFORE runtime.

Structure per case:
    - id: Unique identifier
    - library: Target library name
    - problem: The prompt given to the LLM
    - environment: Exact pip install command (use with uv)
    - broken_code: What the LLM will almost certainly generate
    - fixed_code: The correct modern equivalent
    - expected_error: The error type and message when broken_code runs
    - breaking_changes: List of specific API changes (old → new, version boundary)
"""

from dataclasses import dataclass, field


@dataclass
class BreakingChange:
    """A single API change that breaks backward compatibility."""
    old_api: str          # e.g. "np.trapz(y, x)"
    new_api: str          # e.g. "np.trapezoid(y, x)"
    removed_in: str       # e.g. "2.0.0"
    error_type: str       # e.g. "AttributeError"
    description: str      # Human-readable explanation


@dataclass
class TestCase:
    """A complete test case for the EnvCheck live demo."""
    id: str
    library: str
    problem: str
    environment: str          # pip install command
    broken_code: str          # LLM-generated (will crash)
    fixed_code: str           # Correct modern version
    expected_error: str       # Error message pattern
    breaking_changes: list[BreakingChange] = field(default_factory=list)


# =============================================================================
# CASE 1: NumPy 2.0 — np.trapz removed, np.infty removed
# =============================================================================
CASE_NUMPY = TestCase(
    id="numpy_2x",
    library="numpy",
    problem=(
        "Write a complete Python script from scratch using ONLY numpy. The script must: "
        "1) Create an array x from 0 to 10. "
        "2) Create y = x**2. "
        "3) Calculate the area under the curve y with respect to x using numpy's built-in trapezoidal rule function. "
        "4) Create a new array by appending numpy's explicit verbose infinity alias to x, and print it."
    ),
    environment='pip install "numpy>=2.0.0"',
    broken_code='''\
import numpy as np

x = np.linspace(0, 10, 100)
y = x ** 2

# Calculate area under curve using trapezoidal rule
area = np.trapz(y, x)
print(f"Area under curve: {area}")

# Append infinity alias
x_with_inf = np.append(x, np.infty)
print(f"Array with infinity: {x_with_inf}")
''',
    fixed_code='''\
import numpy as np

x = np.linspace(0, 10, 100)
y = x ** 2

# Calculate area under curve using trapezoidal rule (NumPy 2.0+)
area = np.trapezoid(y, x)
print(f"Area under curve: {area}")

# Append infinity (NumPy 2.0+: use np.inf, not np.infty)
x_with_inf = np.append(x, np.inf)
print(f"Array with infinity: {x_with_inf}")
''',
    expected_error="AttributeError: module 'numpy' has no attribute 'trapz'",
    breaking_changes=[
        BreakingChange(
            old_api="np.trapz(y, x)",
            new_api="np.trapezoid(y, x)",
            removed_in="2.0.0",
            error_type="AttributeError",
            description="np.trapz was removed in NumPy 2.0. Use np.trapezoid instead.",
        ),
        BreakingChange(
            old_api="np.infty",
            new_api="np.inf",
            removed_in="2.0.0",
            error_type="AttributeError",
            description="np.infty alias was removed in NumPy 2.0. Use np.inf instead.",
        ),
    ],
)

# =============================================================================
# CASE 2: SciPy 1.14 — cumtrapz/simps removed
# =============================================================================
CASE_SCIPY = TestCase(
    id="scipy_114",
    library="scipy",
    problem=(
        "Write a complete Python script from scratch using scipy. The script should: "
        "1) Generate a dummy signal array of 100 ones. "
        "2) Calculate the cumulative integral of this signal using scipy's standard "
        "cumulative trapezoidal rule function from the integrate module. "
        "3) Calculate the area under the curve using Simpson's rule from the integrate module. "
        "Print both results."
    ),
    environment='pip install "scipy>=1.14.0"',
    broken_code='''\
import numpy as np
from scipy.integrate import cumtrapz, simps

signal = np.ones(100)

# Cumulative integral
cumulative = cumtrapz(signal)
print(f"Cumulative integral: {cumulative[:5]}...")

# Simpson's rule
area = simps(signal)
print(f"Area (Simpson's): {area}")
''',
    fixed_code='''\
import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson

signal = np.ones(100)

# Cumulative integral (SciPy 1.14+)
cumulative = cumulative_trapezoid(signal)
print(f"Cumulative integral: {cumulative[:5]}...")

# Simpson's rule (SciPy 1.14+)
area = simpson(signal)
print(f"Area (Simpson's): {area}")
''',
    expected_error="ImportError: cannot import name 'cumtrapz' from 'scipy.integrate'",
    breaking_changes=[
        BreakingChange(
            old_api="from scipy.integrate import cumtrapz",
            new_api="from scipy.integrate import cumulative_trapezoid",
            removed_in="1.14.0",
            error_type="ImportError",
            description="cumtrapz was removed in SciPy 1.14. Use cumulative_trapezoid.",
        ),
        BreakingChange(
            old_api="from scipy.integrate import simps",
            new_api="from scipy.integrate import simpson",
            removed_in="1.14.0",
            error_type="ImportError",
            description="simps was removed in SciPy 1.14. Use simpson.",
        ),
    ],
)

# =============================================================================
# CASE 3: scikit-learn 1.2 — load_boston removed
# =============================================================================
CASE_SKLEARN = TestCase(
    id="sklearn_12",
    library="scikit-learn",
    problem=(
        "Write a complete Python script from scratch using scikit-learn and numpy. "
        "1) Load the classic Boston Housing dataset from sklearn.datasets. "
        "2) Print the shape of the feature matrix and the target array. "
        "3) Train a simple LinearRegression model and print the R² score."
    ),
    environment='pip install "scikit-learn>=1.2.0"',
    broken_code='''\
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

# Load Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train a simple linear regression
model = LinearRegression()
model.fit(X, y)
score = model.score(X, y)
print(f"R² score: {score:.4f}")
''',
    fixed_code='''\
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Load California Housing dataset (sklearn 1.2+: load_boston removed)
housing = fetch_california_housing()
X, y = housing.data, housing.target
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train a simple linear regression
model = LinearRegression()
model.fit(X, y)
score = model.score(X, y)
print(f"R² score: {score:.4f}")
''',
    expected_error="ImportError: `load_boston` has been removed from scikit-learn since version 1.2",
    breaking_changes=[
        BreakingChange(
            old_api="from sklearn.datasets import load_boston",
            new_api="from sklearn.datasets import fetch_california_housing",
            removed_in="1.2.0",
            error_type="ImportError",
            description="load_boston was removed in scikit-learn 1.2 due to ethical concerns. Use fetch_california_housing.",
        ),
    ],
)

# =============================================================================
# CASE 4: Pandas 2.2 — fillna(method=) removed, .mad() removed
# =============================================================================
CASE_PANDAS_22 = TestCase(
    id="pandas_22",
    library="pandas",
    problem=(
        "Write a complete Python script from scratch using pandas. "
        "1) Create a DataFrame with a single column 'A' containing the values: [1.0, None, 3.0, None, 5.0]. "
        "2) Fill the missing (None) values by explicitly passing the forward fill method argument into the fillna function. "
        "3) Calculate the Mean Absolute Deviation (MAD) of the filled column 'A' using pandas' built-in mad function. "
        "Print the filled DataFrame and the MAD value."
    ),
    environment='pip install "pandas>=2.2.0"',
    broken_code='''\
import pandas as pd

df = pd.DataFrame({"A": [1.0, None, 3.0, None, 5.0]})

# Forward fill missing values
df = df.fillna(method="ffill")
print("Filled DataFrame:")
print(df)

# Calculate Mean Absolute Deviation
mad_value = df["A"].mad()
print(f"MAD: {mad_value}")
''',
    fixed_code='''\
import pandas as pd

df = pd.DataFrame({"A": [1.0, None, 3.0, None, 5.0]})

# Forward fill missing values (pandas 2.2+)
df = df.ffill()
print("Filled DataFrame:")
print(df)

# Calculate Mean Absolute Deviation manually (pandas 2.2+: .mad() removed)
mad_value = (df["A"] - df["A"].mean()).abs().mean()
print(f"MAD: {mad_value}")
''',
    expected_error="ValueError: ffill/bfill method is no longer supported in fillna",
    breaking_changes=[
        BreakingChange(
            old_api='df.fillna(method="ffill")',
            new_api="df.ffill()",
            removed_in="2.2.0",
            error_type="ValueError",
            description="fillna(method=...) keyword removed in pandas 2.2. Use df.ffill() or df.bfill() directly.",
        ),
        BreakingChange(
            old_api='df["A"].mad()',
            new_api='(df["A"] - df["A"].mean()).abs().mean()',
            removed_in="2.0.0",
            error_type="AttributeError",
            description=".mad() was removed in pandas 2.0. Compute manually.",
        ),
    ],
)

# =============================================================================
# CASE 5: NetworkX 3.0 — gpickle read/write removed
# =============================================================================
CASE_NETWORKX = TestCase(
    id="networkx_3x",
    library="networkx",
    problem=(
        "Write a complete Python script from scratch using networkx. "
        "1) Generate a random Erdos-Renyi graph with 20 nodes and a 0.5 edge probability. "
        "2) Save this graph to a local file named 'graph.gpickle' using networkx's built-in gpickle writing function. "
        "3) Load the graph back from 'graph.gpickle' into a new variable using networkx's built-in gpickle reading function. "
        "Print the number of edges in the loaded graph."
    ),
    environment='pip install "networkx>=3.0.0"',
    broken_code='''\
import networkx as nx

# Generate random graph
G = nx.erdos_renyi_graph(20, 0.5)

# Save to gpickle
nx.write_gpickle(G, "graph.gpickle")
print("Graph saved to graph.gpickle")

# Load back
G_loaded = nx.read_gpickle("graph.gpickle")
print(f"Number of edges: {G_loaded.number_of_edges()}")
''',
    fixed_code='''\
import pickle
import networkx as nx

# Generate random graph
G = nx.erdos_renyi_graph(20, 0.5)

# Save using standard pickle (networkx 3.0+: gpickle removed)
with open("graph.pickle", "wb") as f:
    pickle.dump(G, f)
print("Graph saved to graph.pickle")

# Load back
with open("graph.pickle", "rb") as f:
    G_loaded = pickle.load(f)
print(f"Number of edges: {G_loaded.number_of_edges()}")
''',
    expected_error="AttributeError: module 'networkx' has no attribute 'write_gpickle'",
    breaking_changes=[
        BreakingChange(
            old_api='nx.write_gpickle(G, "graph.gpickle")',
            new_api='pickle.dump(G, open("graph.pickle", "wb"))',
            removed_in="3.0.0",
            error_type="AttributeError",
            description="write_gpickle/read_gpickle removed in NetworkX 3.0 due to security concerns. Use standard pickle.",
        ),
    ],
)

# =============================================================================
# CASE 6: Pandas 1.5.3 (pinned old) — DataFrame.map() doesn't exist yet
# =============================================================================
CASE_PANDAS_15 = TestCase(
    id="pandas_15",
    library="pandas",
    problem=(
        "Write a complete Python script from scratch using pandas. "
        "1) Create a DataFrame with two columns 'A' and 'B', both containing the numbers [1, 2, 3]. "
        "2) Apply a lambda function to square every single element in the entire DataFrame "
        "using pandas' element-wise mapping function. Print the resulting DataFrame."
    ),
    environment='pip install "pandas==1.5.3" "numpy<2.0.0" --python 3.11',
    broken_code='''\
import pandas as pd

df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]})

# Square every element using element-wise mapping
result = df.map(lambda x: x ** 2)
print(result)
''',
    fixed_code='''\
import pandas as pd

df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]})

# Square every element (pandas 1.x: use applymap, not map)
result = df.applymap(lambda x: x ** 2)
print(result)
''',
    expected_error="AttributeError: 'DataFrame' object has no attribute 'map'",
    breaking_changes=[
        BreakingChange(
            old_api="df.applymap(func)",
            new_api="df.map(func)",
            removed_in="N/A (map added in 2.1, applymap removed in 2.1)",
            error_type="AttributeError",
            description="DataFrame.map() was added in pandas 2.1 as replacement for applymap(). In pandas 1.x, only applymap() exists.",
        ),
    ],
)

# =============================================================================
# CASE 7: Pydantic V1 — model_dump() doesn't exist
# =============================================================================
CASE_PYDANTIC = TestCase(
    id="pydantic_v1",
    library="pydantic",
    problem=(
        "Write a complete Python script from scratch using pydantic. "
        "Define a User model with id (integer) and name (string). "
        "Instantiate a user with id=1 and name='Alice'. "
        "Then, export the model to a standard Python dictionary using pydantic's built-in "
        "dictionary export method and print the dictionary."
    ),
    environment='pip install "pydantic==1.10.15"',
    broken_code='''\
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str

user = User(id=1, name="Alice")

# Export to dictionary
user_dict = user.model_dump()
print(user_dict)
''',
    fixed_code='''\
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str

user = User(id=1, name="Alice")

# Export to dictionary (Pydantic V1: use .dict(), not .model_dump())
user_dict = user.dict()
print(user_dict)
''',
    expected_error="AttributeError: 'User' object has no attribute 'model_dump'",
    breaking_changes=[
        BreakingChange(
            old_api="user.dict()",
            new_api="user.model_dump()",
            removed_in="N/A (.dict() deprecated in V2, .model_dump() doesn't exist in V1)",
            error_type="AttributeError",
            description="Pydantic V2 introduced model_dump() to replace dict(). In V1 environments, only .dict() exists.",
        ),
    ],
)


# =============================================================================
# ALL CASES REGISTRY
# =============================================================================
ALL_CASES: list[TestCase] = [
    CASE_NUMPY,
    CASE_SCIPY,
    CASE_SKLEARN,
    CASE_PANDAS_22,
    CASE_NETWORKX,
    CASE_PANDAS_15,
    CASE_PYDANTIC,
]


def print_case_summary():
    """Print a summary table of all test cases."""
    print(f"{'ID':<16} {'Library':<12} {'Env Version':<28} {'Error Type':<20} {'# Changes'}")
    print("-" * 90)
    for case in ALL_CASES:
        env_short = case.environment.replace("pip install ", "")
        err_type = case.breaking_changes[0].error_type if case.breaking_changes else "Unknown"
        print(f"{case.id:<16} {case.library:<12} {env_short:<28} {err_type:<20} {len(case.breaking_changes)}")


if __name__ == "__main__":
    print_case_summary()
