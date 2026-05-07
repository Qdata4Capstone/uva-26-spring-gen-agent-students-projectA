"""Hand-written breaking-change cases in BigCodeBench format.

Each case mirrors a BigCodeBench candidate (task_id, libs, instruct_prompt,
code_prompt, canonical_solution, test, entry_point) plus our benchmark fields
(library_under_test, bad_version, good_version, error_type, kind, rule_label,
reason, evidence_line, note, verified).

All cases are REMOVAL-direction: canonical_solution uses an API that was
genuinely removed in bad_version (so canonical+test crashes with the
documented error_type when run in a bad_version venv).
"""

CASES = []


def _add(**kwargs):
    kwargs.setdefault("note", "")
    kwargs.setdefault("verified", False)
    kwargs.setdefault("entry_point", "task_func")
    CASES.append(kwargs)


# ===== NumPy 2.0 removals =====
# Note: manual_001 (np.trapz) and manual_003 (np.in1d) were dropped after
# verify_ground_truth showed they only emit DeprecationWarning in numpy 2.0.x —
# not actually removed yet, so canonical+test still passes in bad_env.

_add(
    case_id="manual_002",
    task_id="Manual/002",
    libs=["numpy"],
    library_under_test="numpy",
    bad_version="2.0.2", good_version="1.26.4",
    error_type="AttributeError", kind="removal", rule_label="np_product",
    reason="np.product removed in NumPy 2.0; use np.prod",
    evidence_line="    return float(np.product(arr))",
    instruct_prompt=(
        "Given a sequence of numbers, return the product of all elements as a float."
    ),
    code_prompt="import numpy as np\n\ndef task_func(arr):\n",
    canonical_solution="    return float(np.product(arr))\n",
    correct_solution="    return float(np.prod(arr))\n",
    test='''import unittest

class TestCases(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(task_func([1, 2, 3, 4]), 24.0)

    def test_with_one(self):
        self.assertAlmostEqual(task_func([5, 1, 2]), 10.0)

    def test_with_float(self):
        self.assertAlmostEqual(task_func([0.5, 0.5, 4.0]), 1.0)

    def test_single(self):
        self.assertAlmostEqual(task_func([7]), 7.0)
''',
)

_add(
    case_id="manual_004",
    task_id="Manual/004",
    libs=["numpy"],
    library_under_test="numpy",
    bad_version="2.0.2", good_version="1.26.4",
    error_type="AttributeError", kind="removal", rule_label="np_NaN",
    reason="np.NaN removed in NumPy 2.0; use np.nan",
    evidence_line="    return np.array([np.NaN if v is None else v for v in values], dtype=float)",
    instruct_prompt=(
        "Convert a sequence that may contain None values into a numpy float array, "
        "replacing every None with NaN. Return the array."
    ),
    code_prompt="import numpy as np\n\ndef task_func(values):\n",
    canonical_solution=(
        "    return np.array([np.NaN if v is None else v for v in values], dtype=float)\n"
    ),
    correct_solution=(
        "    return np.array([np.nan if v is None else v for v in values], dtype=float)\n"
    ),
    test='''import unittest
import numpy as np

class TestCases(unittest.TestCase):
    def test_basic(self):
        result = task_func([1, None, 3.0])
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result[0], 1.0)
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(result[2], 3.0)

    def test_all_none(self):
        result = task_func([None, None])
        self.assertTrue(np.isnan(result).all())

    def test_no_none(self):
        result = task_func([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
''',
)

_add(
    case_id="manual_005",
    task_id="Manual/005",
    libs=["numpy"],
    library_under_test="numpy",
    bad_version="2.0.2", good_version="1.26.4",
    error_type="AttributeError", kind="removal", rule_label="np_alltrue",
    reason="np.alltrue removed in NumPy 2.0; use np.all",
    evidence_line="    return bool(np.alltrue(np.asarray(values) > threshold))",
    instruct_prompt=(
        "Return True if every element in `values` is strictly greater than `threshold`, "
        "otherwise False."
    ),
    code_prompt="import numpy as np\n\ndef task_func(values, threshold):\n",
    canonical_solution=(
        "    return bool(np.alltrue(np.asarray(values) > threshold))\n"
    ),
    correct_solution=(
        "    return bool(np.all(np.asarray(values) > threshold))\n"
    ),
    test='''import unittest

class TestCases(unittest.TestCase):
    def test_all_above(self):
        self.assertTrue(task_func([2, 3, 4], 1))

    def test_one_below(self):
        self.assertFalse(task_func([2, 3, 0], 1))

    def test_equal_threshold(self):
        self.assertFalse(task_func([1, 1, 1], 1))

    def test_empty(self):
        self.assertTrue(task_func([], 0))
''',
)


# ===== Pandas 2.0 removals =====

_add(
    case_id="manual_006",
    task_id="Manual/006",
    libs=["pandas"],
    library_under_test="pandas",
    bad_version="2.2.2", good_version="1.5.3",
    error_type="AttributeError", kind="removal", rule_label="pd_append",
    reason="DataFrame.append removed in pandas 2.0; use pd.concat",
    evidence_line="        df = df.append(record, ignore_index=True)",
    instruct_prompt=(
        "Build a DataFrame from a list of dict records. Each record is a dict "
        "mapping column name to value. Append the records one by one and return "
        "the final DataFrame."
    ),
    code_prompt="import pandas as pd\n\ndef task_func(records):\n",
    canonical_solution=(
        "    df = pd.DataFrame()\n"
        "    for record in records:\n"
        "        df = df.append(record, ignore_index=True)\n"
        "    return df\n"
    ),
    correct_solution=(
        "    df = pd.DataFrame()\n"
        "    for record in records:\n"
        "        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)\n"
        "    return df\n"
    ),
    test='''import unittest
import pandas as pd

class TestCases(unittest.TestCase):
    def test_basic(self):
        records = [{"name": "a", "value": 1}, {"name": "b", "value": 2}]
        df = task_func(records)
        self.assertEqual(len(df), 2)
        self.assertEqual(set(df.columns), {"name", "value"})
        self.assertEqual(df.iloc[0]["name"], "a")
        self.assertEqual(df.iloc[1]["value"], 2)

    def test_empty(self):
        df = task_func([])
        self.assertEqual(len(df), 0)

    def test_single(self):
        df = task_func([{"a": 1, "b": 2}])
        self.assertEqual(df.iloc[0]["a"], 1)
        self.assertEqual(df.iloc[0]["b"], 2)
''',
)

_add(
    case_id="manual_007",
    task_id="Manual/007",
    libs=["pandas"],
    library_under_test="pandas",
    bad_version="2.2.2", good_version="1.5.3",
    error_type="AttributeError", kind="removal", rule_label="pd_iteritems",
    reason="DataFrame.iteritems removed in pandas 2.0; use df.items()",
    evidence_line="    for col_name, col_data in df.iteritems():",
    instruct_prompt=(
        "Given a DataFrame with numeric columns, compute the mean of each column "
        "and return a dict mapping column name to its mean."
    ),
    code_prompt="import pandas as pd\n\ndef task_func(df):\n",
    canonical_solution=(
        "    result = {}\n"
        "    for col_name, col_data in df.iteritems():\n"
        "        result[col_name] = float(col_data.mean())\n"
        "    return result\n"
    ),
    correct_solution=(
        "    result = {}\n"
        "    for col_name, col_data in df.items():\n"
        "        result[col_name] = float(col_data.mean())\n"
        "    return result\n"
    ),
    test='''import unittest
import pandas as pd

class TestCases(unittest.TestCase):
    def test_basic(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = task_func(df)
        self.assertAlmostEqual(result["a"], 2.0)
        self.assertAlmostEqual(result["b"], 5.0)

    def test_single_col(self):
        df = pd.DataFrame({"x": [10, 20, 30]})
        result = task_func(df)
        self.assertAlmostEqual(result["x"], 20.0)

    def test_floats(self):
        df = pd.DataFrame({"v": [0.5, 1.5]})
        result = task_func(df)
        self.assertAlmostEqual(result["v"], 1.0)
''',
)

_add(
    case_id="manual_008",
    task_id="Manual/008",
    libs=["pandas", "io"],
    library_under_test="pandas",
    bad_version="2.2.2", good_version="1.5.3",
    error_type="TypeError", kind="removal", rule_label="pd_squeeze_arg",
    reason="pd.read_csv squeeze= removed in pandas 2.0; use .squeeze() method",
    evidence_line="    return pd.read_csv(io.StringIO(csv_text), squeeze=True)",
    instruct_prompt=(
        "Given a CSV string containing a single column of numbers, parse it and "
        "return a pandas Series of the values (not a DataFrame)."
    ),
    code_prompt="import io\nimport pandas as pd\n\ndef task_func(csv_text):\n",
    canonical_solution=(
        "    return pd.read_csv(io.StringIO(csv_text), squeeze=True)\n"
    ),
    correct_solution=(
        # squeeze() with no axis collapses both dims, returning scalar for 1-row CSV.
        # squeeze("columns") collapses only the column dim, always yielding a Series.
        "    return pd.read_csv(io.StringIO(csv_text)).squeeze(\"columns\")\n"
    ),
    test='''import unittest
import pandas as pd

class TestCases(unittest.TestCase):
    def test_basic(self):
        s = task_func("v\\n1\\n2\\n3\\n")
        self.assertIsInstance(s, pd.Series)
        self.assertEqual(list(s), [1, 2, 3])

    def test_single_value(self):
        s = task_func("x\\n42\\n")
        self.assertIsInstance(s, pd.Series)
        self.assertEqual(s.iloc[0], 42)
''',
)


# ===== Pillow 10.0 removal =====

_add(
    case_id="manual_009",
    task_id="Manual/009",
    libs=["PIL"],
    library_under_test="Pillow",
    bad_version="10.4.0", good_version="9.5.0",
    error_type="AttributeError", kind="removal", rule_label="pil_antialias",
    reason="Image.ANTIALIAS removed in Pillow 10.0; use Image.LANCZOS",
    evidence_line="    return img.resize(size, Image.ANTIALIAS)",
    instruct_prompt=(
        "Resize a PIL Image to the target (width, height) using high-quality "
        "antialiasing. Return the resized image."
    ),
    code_prompt="from PIL import Image\n\ndef task_func(img, size):\n",
    canonical_solution=(
        "    return img.resize(size, Image.ANTIALIAS)\n"
    ),
    correct_solution=(
        "    return img.resize(size, Image.LANCZOS)\n"
    ),
    test='''import unittest
from PIL import Image

class TestCases(unittest.TestCase):
    def test_resize_smaller(self):
        img = Image.new("RGB", (100, 50), color="red")
        out = task_func(img, (50, 25))
        self.assertEqual(out.size, (50, 25))

    def test_resize_larger(self):
        img = Image.new("RGB", (10, 10), color="blue")
        out = task_func(img, (40, 40))
        self.assertEqual(out.size, (40, 40))

    def test_returns_image(self):
        img = Image.new("L", (8, 8), color=128)
        out = task_func(img, (4, 4))
        self.assertIsInstance(out, Image.Image)
''',
)


# ===== Flask 2.3 removal =====

_add(
    case_id="manual_010",
    task_id="Manual/010",
    libs=["flask"],
    library_under_test="flask",
    bad_version="3.0.3", good_version="2.2.5",
    error_type="ImportError", kind="removal", rule_label="flask_markup",
    reason="flask.Markup removed in Flask 2.3; use markupsafe.Markup",
    evidence_line="    from flask import Markup",
    instruct_prompt=(
        "Wrap a string of HTML in a Markup object so that it is treated as "
        "template-safe. Return the Markup-wrapped value."
    ),
    code_prompt="def task_func(html):\n",
    canonical_solution=(
        "    from flask import Markup\n"
        "    return Markup(html)\n"
    ),
    correct_solution=(
        "    from markupsafe import Markup\n"
        "    return Markup(html)\n"
    ),
    test='''import unittest

class TestCases(unittest.TestCase):
    def test_basic(self):
        result = task_func("<b>hello</b>")
        self.assertEqual(str(result), "<b>hello</b>")

    def test_striptags(self):
        result = task_func("<p>world</p>")
        self.assertEqual(result.striptags(), "world")

    def test_safe_concat(self):
        result = task_func("<i>x</i>")
        # Markup objects are str subclasses
        self.assertIsInstance(result, str)
''',
)


# ===== NumPy 1.24 type-alias removals =====

_add(
    case_id="manual_011",
    task_id="Manual/011",
    libs=["numpy"],
    library_under_test="numpy",
    bad_version="1.26.4", good_version="1.23.5",
    error_type="AttributeError", kind="removal", rule_label="np_float_alias",
    reason="np.float removed in NumPy 1.24; use built-in float or np.float64",
    evidence_line="    return np.array(values, dtype=np.float)",
    instruct_prompt=(
        "Convert a list of numeric strings or numbers into a numpy float array. "
        "Return the array."
    ),
    code_prompt="import numpy as np\n\ndef task_func(values):\n",
    canonical_solution="    return np.array(values, dtype=np.float)\n",
    correct_solution="    return np.array(values, dtype=float)\n",
    test='''import unittest
import numpy as np

class TestCases(unittest.TestCase):
    def test_strings(self):
        result = task_func(["1", "2.5", "3"])
        np.testing.assert_array_equal(result, np.array([1.0, 2.5, 3.0]))
        self.assertEqual(result.dtype, np.float64)

    def test_ints(self):
        result = task_func([1, 2, 3])
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_mixed(self):
        result = task_func([0.5, 1, "2.5"])
        np.testing.assert_array_equal(result, np.array([0.5, 1.0, 2.5]))
''',
)


# ===== Pandas 2.0 read_csv arg removals =====

_add(
    case_id="manual_012",
    task_id="Manual/012",
    libs=["pandas", "io"],
    library_under_test="pandas",
    bad_version="2.2.2", good_version="1.5.3",
    error_type="TypeError", kind="removal", rule_label="pd_error_bad_lines",
    reason="pd.read_csv error_bad_lines= removed in pandas 2.0; use on_bad_lines='skip'",
    evidence_line="    return pd.read_csv(io.StringIO(csv_text), error_bad_lines=False)",
    instruct_prompt=(
        "Parse a CSV string that may contain malformed lines. Skip bad lines "
        "silently and return the resulting DataFrame."
    ),
    code_prompt="import io\nimport pandas as pd\n\ndef task_func(csv_text):\n",
    canonical_solution=(
        "    return pd.read_csv(io.StringIO(csv_text), error_bad_lines=False)\n"
    ),
    correct_solution=(
        "    return pd.read_csv(io.StringIO(csv_text), on_bad_lines='skip')\n"
    ),
    test='''import unittest
import pandas as pd

class TestCases(unittest.TestCase):
    def test_clean(self):
        df = task_func("a,b\\n1,2\\n3,4\\n")
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["a", "b"])

    def test_with_bad_line(self):
        df = task_func("a,b\\n1,2\\n3,4,5\\n6,7\\n")
        # Bad line (3,4,5) skipped; 2 good rows remain
        self.assertEqual(len(df), 2)
''',
)


# ===== sklearn 1.2 linear-model normalize= removal =====

_add(
    case_id="manual_013",
    task_id="Manual/013",
    libs=["numpy", "sklearn"],
    library_under_test="scikit-learn",
    bad_version="1.4.2", good_version="1.1.3",
    error_type="TypeError", kind="removal", rule_label="skl_lr_normalize",
    reason="LinearRegression normalize= removed in sklearn 1.2; use a Pipeline with StandardScaler",
    evidence_line="    model = LinearRegression(normalize=True)",
    instruct_prompt=(
        "Fit a linear regression on (X, y) with input features pre-scaled "
        "(standardized) so the regression coefficients are on a comparable scale. "
        "Return the predictions for X_test."
    ),
    code_prompt=(
        "import numpy as np\n"
        "from sklearn.linear_model import LinearRegression\n\n"
        "def task_func(X, y, X_test):\n"
    ),
    canonical_solution=(
        "    model = LinearRegression(normalize=True)\n"
        "    model.fit(X, y)\n"
        "    return model.predict(X_test)\n"
    ),
    correct_solution=(
        "    from sklearn.preprocessing import StandardScaler\n"
        "    from sklearn.pipeline import make_pipeline\n"
        "    pipe = make_pipeline(StandardScaler(), LinearRegression())\n"
        "    pipe.fit(X, y)\n"
        "    return pipe.predict(X_test)\n"
    ),
    test='''import unittest
import numpy as np

class TestCases(unittest.TestCase):
    def test_simple_linear(self):
        # y = 2x, prediction at x=5 should be ~10
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        X_test = np.array([[5.0]])
        result = task_func(X, y, X_test)
        self.assertEqual(result.shape, (1,))
        self.assertAlmostEqual(float(result[0]), 10.0, places=2)

    def test_two_features(self):
        # y = x1 + x2
        X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        X_test = np.array([[5.0, 5.0]])
        result = task_func(X, y, X_test)
        self.assertAlmostEqual(float(result[0]), 10.0, places=1)
''',
)


# ===== matplotlib 3.9 register_cmap removal =====

_add(
    case_id="manual_014",
    task_id="Manual/014",
    libs=["matplotlib"],
    library_under_test="matplotlib",
    bad_version="3.9.2", good_version="3.8.4",
    error_type="ImportError", kind="removal", rule_label="mpl_register_cmap",
    reason="matplotlib.cm.register_cmap removed in matplotlib 3.9; use matplotlib.colormaps.register",
    evidence_line="    from matplotlib.cm import register_cmap",
    instruct_prompt=(
        "Register a custom colormap built from the given color list under the "
        "given name, then return the registered colormap object via "
        "matplotlib.pyplot.get_cmap."
    ),
    code_prompt=(
        "import matplotlib.pyplot as plt\n"
        "from matplotlib.colors import LinearSegmentedColormap\n\n"
        "def task_func(name, colors):\n"
    ),
    canonical_solution=(
        "    from matplotlib.cm import register_cmap\n"
        "    cmap = LinearSegmentedColormap.from_list(name, colors)\n"
        "    register_cmap(name=name, cmap=cmap)\n"
        "    return plt.get_cmap(name)\n"
    ),
    correct_solution=(
        "    import matplotlib as mpl\n"
        "    cmap = LinearSegmentedColormap.from_list(name, colors)\n"
        "    mpl.colormaps.register(cmap, name=name)\n"
        "    return plt.get_cmap(name)\n"
    ),
    test='''import unittest
from matplotlib.colors import Colormap

class TestCases(unittest.TestCase):
    def test_basic(self):
        cmap = task_func("test_cmap_basic", ["red", "blue"])
        self.assertIsInstance(cmap, Colormap)
        self.assertEqual(cmap.name, "test_cmap_basic")

    def test_three_colors(self):
        cmap = task_func("test_cmap_three", ["red", "green", "blue"])
        self.assertIsInstance(cmap, Colormap)
        # Check colormap is callable and produces colors
        c0 = cmap(0.0)
        c1 = cmap(1.0)
        self.assertEqual(len(c0), 4)  # RGBA
        self.assertNotEqual(c0, c1)
''',
)


# ===== scipy 1.12 signal.gaussian removal =====

_add(
    case_id="manual_015",
    task_id="Manual/015",
    libs=["numpy", "scipy"],
    library_under_test="scipy",
    bad_version="1.13.1", good_version="1.11.4",
    error_type="ImportError", kind="removal", rule_label="scipy_signal_gaussian",
    reason="scipy.signal.gaussian removed in SciPy 1.12; use scipy.signal.windows.gaussian",
    evidence_line="    from scipy.signal import gaussian",
    instruct_prompt=(
        "Build a Gaussian window of length M with standard deviation std, "
        "and return it as a numpy array."
    ),
    code_prompt="import numpy as np\n\ndef task_func(M, std):\n",
    canonical_solution=(
        "    from scipy.signal import gaussian\n"
        "    return gaussian(M, std)\n"
    ),
    correct_solution=(
        "    from scipy.signal.windows import gaussian\n"
        "    return gaussian(M, std)\n"
    ),
    test='''import unittest
import numpy as np

class TestCases(unittest.TestCase):
    def test_length(self):
        w = task_func(11, 2.0)
        self.assertEqual(len(w), 11)

    def test_peak_at_center(self):
        w = task_func(11, 2.0)
        # Gaussian window peaks at the center
        self.assertEqual(np.argmax(w), 5)
        self.assertAlmostEqual(float(w[5]), 1.0, places=5)

    def test_symmetric(self):
        w = task_func(9, 1.5)
        np.testing.assert_array_almost_equal(w, w[::-1])
''',
)


if __name__ == "__main__":
    print(f"Defined {len(CASES)} manual cases:")
    for c in CASES:
        print(f"  {c['case_id']:14s} {c['task_id']:14s} "
              f"{c['library_under_test']:12s} {c['rule_label']}")
