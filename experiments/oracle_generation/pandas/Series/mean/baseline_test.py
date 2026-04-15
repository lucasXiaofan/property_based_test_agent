"""
Baseline property-based tests for pandas.Series.mean.

Properties tested:
  1. Basic correctness against numpy reference
  2. Single-element series: mean equals that element
  3. Constant series: mean equals the constant
  4. skipna=True (default): NaN values are excluded
  5. skipna=False: any NaN in data yields NaN result
  6. All-NaN series with skipna=True: returns NaN
  7. Linearity – scaling: mean(k*s) == k * mean(s)
  8. Linearity – translation: mean(s + c) == mean(s) + c
  9. Boolean series: treated as 0/1 integers
 10. Return value is a scalar (not a Series)
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, from_dtype


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

finite_floats = st.floats(
    allow_nan=False,
    allow_infinity=False,
    min_value=-1e10,
    max_value=1e10,
)

floats_with_nan = st.floats(
    allow_nan=True,
    allow_infinity=False,
    min_value=-1e10,
    max_value=1e10,
)


def numeric_series(min_size=1, max_size=50, allow_nan=False):
    """Strategy producing a pd.Series of finite floats (optionally with NaN)."""
    element_strategy = floats_with_nan if allow_nan else finite_floats
    return st.lists(element_strategy, min_size=min_size, max_size=max_size).map(
        pd.Series
    )


# ---------------------------------------------------------------------------
# Property 1 – Result matches numpy.mean on non-NaN data
# ---------------------------------------------------------------------------

@given(numeric_series(min_size=1))
@settings(max_examples=200)
def test_mean_matches_numpy_no_nan(s):
    """Series.mean() on finite data should equal numpy.mean."""
    result = s.mean()
    expected = np.mean(s.values)
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12), (
        f"mean={result}, numpy={expected}"
    )


# ---------------------------------------------------------------------------
# Property 2 – Single element
# ---------------------------------------------------------------------------

@given(finite_floats)
def test_mean_single_element(value):
    """mean of a single-element series equals that element."""
    s = pd.Series([value])
    result = s.mean()
    assert math.isclose(result, value, rel_tol=1e-9, abs_tol=1e-12), (
        f"value={value}, mean={result}"
    )


# ---------------------------------------------------------------------------
# Property 3 – Constant series
# ---------------------------------------------------------------------------

@given(
    finite_floats,
    st.integers(min_value=1, max_value=50),
)
def test_mean_constant_series(value, n):
    """mean of a constant series equals that constant."""
    s = pd.Series([value] * n)
    result = s.mean()
    assert math.isclose(result, value, rel_tol=1e-9, abs_tol=1e-12), (
        f"constant={value}, n={n}, mean={result}"
    )


# ---------------------------------------------------------------------------
# Property 4 – skipna=True excludes NaN (default behaviour)
# ---------------------------------------------------------------------------

@given(
    st.lists(finite_floats, min_size=1, max_size=40),
    st.lists(st.just(float("nan")), min_size=1, max_size=10),
)
def test_mean_skipna_true_ignores_nan(finite_vals, nan_vals):
    """Adding NaN values should not change mean when skipna=True."""
    s_pure = pd.Series(finite_vals)
    s_with_nan = pd.Series(finite_vals + nan_vals)
    result_pure = s_pure.mean(skipna=True)
    result_with_nan = s_with_nan.mean(skipna=True)
    assert math.isclose(result_pure, result_with_nan, rel_tol=1e-9, abs_tol=1e-12), (
        f"pure mean={result_pure}, with-NaN mean={result_with_nan}"
    )


# ---------------------------------------------------------------------------
# Property 5 – skipna=False: any NaN → result is NaN
# ---------------------------------------------------------------------------

@given(
    st.lists(finite_floats, min_size=0, max_size=40),
    st.lists(st.just(float("nan")), min_size=1, max_size=10),
)
def test_mean_skipna_false_propagates_nan(finite_vals, nan_vals):
    """When skipna=False, any NaN in the series makes mean return NaN."""
    s = pd.Series(finite_vals + nan_vals)
    result = s.mean(skipna=False)
    assert math.isnan(result), f"Expected NaN, got {result}"


# ---------------------------------------------------------------------------
# Property 6 – All-NaN series with skipna=True returns NaN
# ---------------------------------------------------------------------------

@given(st.integers(min_value=1, max_value=20))
def test_mean_all_nan_skipna_true_returns_nan(n):
    """mean of an all-NaN series with skipna=True should return NaN."""
    s = pd.Series([float("nan")] * n)
    result = s.mean(skipna=True)
    assert math.isnan(result), f"Expected NaN for all-NaN series, got {result}"


# ---------------------------------------------------------------------------
# Property 7 – Scaling: mean(k*s) == k * mean(s)
# ---------------------------------------------------------------------------

@given(
    numeric_series(min_size=1),
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_mean_scaling(s, k):
    """mean(k*s) should equal k * mean(s) for finite scalars."""
    assume(not s.empty)
    result = (k * s).mean()
    expected = k * s.mean()
    # Use absolute tolerance to handle near-zero products
    assert math.isclose(result, expected, rel_tol=1e-6, abs_tol=1e-9), (
        f"k={k}, mean(k*s)={result}, k*mean(s)={expected}"
    )


# ---------------------------------------------------------------------------
# Property 8 – Translation: mean(s + c) == mean(s) + c
# ---------------------------------------------------------------------------

@given(
    numeric_series(min_size=1),
    finite_floats,
)
@settings(max_examples=200)
def test_mean_translation(s, c):
    """mean(s + c) should equal mean(s) + c."""
    assume(not s.empty)
    result = (s + c).mean()
    expected = s.mean() + c
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12), (
        f"c={c}, mean(s+c)={result}, mean(s)+c={expected}"
    )


# ---------------------------------------------------------------------------
# Property 9 – Boolean series treated as 0/1
# ---------------------------------------------------------------------------

@given(st.lists(st.booleans(), min_size=1, max_size=50))
def test_mean_boolean_series(bools):
    """mean of a boolean series equals the proportion of True values."""
    s = pd.Series(bools)
    result = s.mean()
    expected = sum(bools) / len(bools)
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12), (
        f"bools={bools}, mean={result}, expected={expected}"
    )


# ---------------------------------------------------------------------------
# Property 10 – Return value is a scalar
# ---------------------------------------------------------------------------

@given(numeric_series(min_size=1))
def test_mean_returns_scalar(s):
    """Series.mean() should return a Python/numpy scalar, not a Series."""
    result = s.mean()
    assert not isinstance(result, pd.Series), (
        f"Expected scalar, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# Property 11 – Mean is bounded by [min, max] of the series
# ---------------------------------------------------------------------------

@given(numeric_series(min_size=1))
def test_mean_bounded_by_min_max(s):
    """mean must lie within [min, max] of the series values."""
    result = s.mean()
    lower = float(s.min())
    upper = float(s.max())
    tol = max(1e-9, 1e-12 * max(abs(lower), abs(upper), abs(float(result)), 1.0))
    assert lower - tol <= result <= upper + tol, (
        f"mean={result} outside [{lower}, {upper}] with tol={tol}"
    )


# ---------------------------------------------------------------------------
# Property 12 – Mean of concatenation
# ---------------------------------------------------------------------------

@given(
    st.lists(finite_floats, min_size=1, max_size=30),
    st.lists(finite_floats, min_size=1, max_size=30),
)
def test_mean_concatenation(vals_a, vals_b):
    """mean(concat(a, b)) equals weighted average of mean(a) and mean(b)."""
    a = pd.Series(vals_a)
    b = pd.Series(vals_b)
    combined = pd.concat([a, b], ignore_index=True)
    result = combined.mean()
    expected = (a.sum() + b.sum()) / (len(a) + len(b))
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12), (
        f"combined mean={result}, expected={expected}"
    )
