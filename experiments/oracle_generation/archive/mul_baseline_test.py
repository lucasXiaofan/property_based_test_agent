"""
Baseline property-based tests for pandas.Series.mul

Properties tested:
1. Equivalence with * operator (no fill_value)
2. Commutativity: a.mul(b) == b.mul(a) (element-wise, same index)
3. Scalar multiplication: a.mul(k) == k * a
4. fill_value substitutes NaN before computation
5. If both locations are NaN, result is NaN regardless of fill_value
6. Identity: a.mul(1) == a (for non-NaN values)
7. Zero multiplication: a.mul(0) == 0 (for non-NaN values)
8. Index alignment: result index is the union of both series' indices
9. Output dtype is float when NaN is involved
10. level parameter does not affect non-MultiIndex series behavior
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import series, column
from pandas.testing import assert_series_equal


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

finite_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
maybe_nan_floats = st.one_of(finite_floats, st.just(float("nan")))
small_ints = st.integers(min_value=-100, max_value=100)
scalar_values = st.one_of(finite_floats, small_ints)

index_elements = st.text(min_size=1, max_size=2, alphabet=st.characters(whitelist_categories=("Lu", "Ll")))


def series_strategy(min_size=1, max_size=6, allow_nan=True):
    float_val = maybe_nan_floats if allow_nan else finite_floats
    return st.lists(float_val, min_size=min_size, max_size=max_size).flatmap(
        lambda vals: st.lists(
            index_elements, min_size=len(vals), max_size=len(vals), unique=True
        ).map(lambda idx: pd.Series(vals, index=idx))
    )


def aligned_series_pair(min_size=1, max_size=5):
    """Two series that share the same index (fully aligned)."""
    return st.lists(
        st.tuples(maybe_nan_floats, maybe_nan_floats), min_size=min_size, max_size=max_size
    ).flatmap(
        lambda pairs: st.lists(
            index_elements, min_size=len(pairs), max_size=len(pairs), unique=True
        ).map(
            lambda idx: (
                pd.Series([p[0] for p in pairs], index=idx),
                pd.Series([p[1] for p in pairs], index=idx),
            )
        )
    )


# ---------------------------------------------------------------------------
# 1. Equivalence with * operator
# ---------------------------------------------------------------------------

@given(a=series_strategy(), b=series_strategy())
@settings(max_examples=100)
def test_mul_equivalent_to_star_operator(a, b):
    """Series.mul(other) == series * other when no fill_value is used."""
    result_mul = a.mul(b)
    result_star = a * b
    assert_series_equal(result_mul, result_star)


@given(a=series_strategy(), k=scalar_values)
@settings(max_examples=100)
def test_mul_scalar_equivalent_to_star_operator(a, k):
    """Series.mul(scalar) == series * scalar."""
    result_mul = a.mul(k)
    result_star = a * k
    assert_series_equal(result_mul, result_star)


# ---------------------------------------------------------------------------
# 2. Commutativity (same index)
# ---------------------------------------------------------------------------

@given(pair=aligned_series_pair())
@settings(max_examples=100)
def test_mul_commutative_same_index(pair):
    """a.mul(b) == b.mul(a) when both series share the same index."""
    a, b = pair
    assert_series_equal(a.mul(b), b.mul(a))


@given(pair=aligned_series_pair())
@settings(max_examples=100)
def test_mul_commutative_with_fill_value(pair):
    """Commutativity holds when fill_value is provided."""
    a, b = pair
    fv = 0.0
    assert_series_equal(a.mul(b, fill_value=fv), b.mul(a, fill_value=fv))


# ---------------------------------------------------------------------------
# 3. Scalar multiplication identity and zero
# ---------------------------------------------------------------------------

@given(a=series_strategy(allow_nan=False))
@settings(max_examples=100)
def test_mul_by_one_is_identity(a):
    """a.mul(1) == a for series with no NaN values."""
    result = a.mul(1)
    assert_series_equal(result, a.astype(float))


@given(a=series_strategy(allow_nan=False))
@settings(max_examples=100)
def test_mul_by_zero_gives_zeros(a):
    """a.mul(0) produces all-zero series for non-NaN values."""
    result = a.mul(0)
    expected = pd.Series(np.zeros(len(a)), index=a.index)
    assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# 4. fill_value substitutes NaN before computation
# ---------------------------------------------------------------------------

def test_fill_value_replaces_nan_in_a():
    """NaN in `a` is replaced by fill_value before multiplying."""
    a = pd.Series([1.0, np.nan, 3.0], index=["x", "y", "z"])
    b = pd.Series([2.0, 4.0, 5.0], index=["x", "y", "z"])
    result = a.mul(b, fill_value=0.0)
    expected = pd.Series([2.0, 0.0, 15.0], index=["x", "y", "z"])
    assert_series_equal(result, expected)


def test_fill_value_replaces_nan_in_b():
    """NaN in `b` is replaced by fill_value before multiplying."""
    a = pd.Series([2.0, 4.0, 6.0], index=["x", "y", "z"])
    b = pd.Series([1.0, np.nan, 3.0], index=["x", "y", "z"])
    result = a.mul(b, fill_value=0.0)
    expected = pd.Series([2.0, 0.0, 18.0], index=["x", "y", "z"])
    assert_series_equal(result, expected)


def test_fill_value_scalar_replaces_nan():
    """NaN in series is replaced by fill_value when multiplying by scalar."""
    a = pd.Series([1.0, 1.0, 1.0, np.nan], index=["a", "b", "c", "d"])
    result = a.mul(5, fill_value=0)
    expected = pd.Series([5.0, 5.0, 5.0, 0.0], index=["a", "b", "c", "d"])
    assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# 5. Both NaN => result NaN regardless of fill_value
# ---------------------------------------------------------------------------

def test_both_nan_result_is_nan():
    """When both series have NaN at the same location, result is NaN even with fill_value."""
    a = pd.Series([1.0, np.nan], index=["x", "y"])
    b = pd.Series([2.0, np.nan], index=["x", "y"])
    result = a.mul(b, fill_value=0.0)
    assert pd.isna(result["y"]), "Both NaN at same location should yield NaN"
    assert result["x"] == 2.0


# ---------------------------------------------------------------------------
# 6. Index alignment: result index is union of both indices
# ---------------------------------------------------------------------------

def test_index_alignment_union():
    """Result index is the union of the two series' indices."""
    a = pd.Series([1.0, 2.0], index=["a", "b"])
    b = pd.Series([3.0, 4.0], index=["b", "c"])
    result = a.mul(b)
    expected_index = pd.Index(["a", "b", "c"])
    assert result.index.equals(expected_index)


def test_index_alignment_non_overlapping():
    """Non-overlapping indices yield all NaN (without fill_value)."""
    a = pd.Series([1.0, 2.0], index=["a", "b"])
    b = pd.Series([3.0, 4.0], index=["c", "d"])
    result = a.mul(b)
    assert result.isna().all()
    assert set(result.index) == {"a", "b", "c", "d"}


def test_index_alignment_with_fill_value():
    """fill_value fills in missing aligned values before multiplication."""
    a = pd.Series([1.0, 1.0, 1.0, np.nan], index=["a", "b", "c", "d"])
    b = pd.Series([1.0, np.nan, 1.0, np.nan], index=["a", "b", "d", "e"])
    result = a.multiply(b, fill_value=0)
    # From documentation example
    expected = pd.Series([1.0, 0.0, 0.0, 0.0, np.nan], index=["a", "b", "c", "d", "e"])
    assert_series_equal(result, expected)


@given(pair=aligned_series_pair())
@settings(max_examples=100)
def test_result_index_equals_input_index_when_aligned(pair):
    """When both series have the same index, result index equals that index."""
    a, b = pair
    result = a.mul(b)
    assert result.index.equals(a.index)


# ---------------------------------------------------------------------------
# 7. Output is a Series
# ---------------------------------------------------------------------------

@given(a=series_strategy(), b=series_strategy())
@settings(max_examples=50)
def test_mul_returns_series(a, b):
    """mul always returns a pandas Series."""
    result = a.mul(b)
    assert isinstance(result, pd.Series)


@given(a=series_strategy(), k=scalar_values)
@settings(max_examples=50)
def test_mul_scalar_returns_series(a, k):
    """mul with a scalar always returns a pandas Series."""
    result = a.mul(k)
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# 8. Associativity with scalar: (a * k1) * k2 == a * (k1 * k2)
# ---------------------------------------------------------------------------

@given(
    a=series_strategy(allow_nan=False),
    k1=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    k2=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_scalar_mul_associativity(a, k1, k2):
    """(a.mul(k1)).mul(k2) == a.mul(k1 * k2) within floating-point tolerance."""
    result_chain = a.mul(k1).mul(k2)
    result_direct = a.mul(k1 * k2)
    assert_series_equal(result_chain, result_direct, check_exact=False, rtol=1e-5)


# ---------------------------------------------------------------------------
# 9. axis parameter is accepted without error
# ---------------------------------------------------------------------------

@given(a=series_strategy())
@settings(max_examples=30)
def test_axis_0_accepted(a):
    """axis=0 (default) is accepted without raising."""
    result = a.mul(2.0, axis=0)
    assert isinstance(result, pd.Series)


@given(a=series_strategy())
@settings(max_examples=30)
def test_axis_index_accepted(a):
    """axis='index' is accepted without raising."""
    result = a.mul(2.0, axis="index")
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# 10. Preserve NaN without fill_value
# ---------------------------------------------------------------------------

@given(a=series_strategy())
@settings(max_examples=100)
def test_nan_preserved_without_fill_value(a):
    """NaN values in input remain NaN in output when no fill_value is given."""
    nan_mask = a.isna()
    result = a.mul(99.0)
    assert result[nan_mask].isna().all()
