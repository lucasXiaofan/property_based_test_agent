from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st


@given(
    data=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=50)
)
@settings(max_examples=100)
def test_mean_basic_numeric(data):
    s = pd.Series(data)
    result = s.mean()
    expected = sum(data) / len(data)
    assert abs(result - expected) < 1e-9


@given(
    data=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    )
)
@settings(max_examples=100)
def test_mean_returns_scalar(data):
    s = pd.Series(data)
    result = s.mean()
    assert isinstance(result, (int, float, np.floating, np.integer))


@given(
    data=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    )
)
@settings(max_examples=100)
def test_mean_float_result_type(data):
    s = pd.Series(data)
    result = s.mean()
    assert isinstance(result, float)


@given(
    data=st.lists(
        st.one_of(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            st.none(),
        ),
        min_size=1,
        max_size=50,
    )
)
@settings(max_examples=100)
def test_mean_skipna_true_ignores_na(data):
    s = pd.Series(data, dtype="float64")
    result = s.mean(skipna=True)
    non_na = [x for x in data if x is not None]
    if len(non_na) > 0:
        expected = sum(non_na) / len(non_na)
        assert abs(result - expected) < 1e-9
    else:
        assert np.isnan(result)


@given(
    data=st.lists(
        st.one_of(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            st.none(),
        ),
        min_size=1,
        max_size=50,
    )
)
@settings(max_examples=100)
def test_mean_skipna_false_with_na_returns_nan(data):
    s = pd.Series(data, dtype="float64")
    has_na = any(x is None for x in data)
    if has_na:
        result = s.mean(skipna=False)
        assert np.isnan(result)


@given(data=st.lists(st.booleans(), min_size=1, max_size=50))
@settings(max_examples=100)
def test_mean_boolean_treated_as_numeric(data):
    s = pd.Series(data)
    result = s.mean()
    expected = sum(int(x) for x in data) / len(data)
    assert abs(result - expected) < 1e-9


# ============================================================================
# NEW TESTS: Edge cases and non-happy-path cases from documentation
# ============================================================================

def test_mean_empty_series_returns_nan():
    """Empty Series should return NaN (no values to compute mean)."""
    s = pd.Series([], dtype=float)
    result = s.mean()
    assert np.isnan(result)


def test_mean_single_element():
    """Mean of single element should return that element."""
    s = pd.Series([42])
    assert s.mean() == 42.0


def test_mean_all_null_with_skipna_true():
    """All null values with skipna=True should return NaN."""
    s = pd.Series([None, None, None], dtype=float)
    result = s.mean(skipna=True)
    assert np.isnan(result)


def test_mean_axis_parameter_defaults_to_zero():
    """axis parameter should work for Series (defaults to 0, unused)."""
    s = pd.Series([1, 2, 3])
    assert s.mean(axis=0) == 2.0


def test_mean_numeric_only_true_with_numeric_series():
    """numeric_only=True should work with numeric Series."""
    s = pd.Series([1.0, 2.0, 3.0])
    result = s.mean(numeric_only=True)
    assert result == 2.0


def test_mean_numeric_only_false_includes_all_numeric():
    """numeric_only=False (default) should include numeric columns."""
    s = pd.Series([1, 2, 3])
    result = s.mean(numeric_only=False)
    assert result == 2.0


def test_mean_mixed_int_float_types():
    """Mean should work correctly with mixed int and float types."""
    s = pd.Series([1, 2.5, 3, 4.5])
    expected = (1 + 2.5 + 3 + 4.5) / 4
    assert abs(s.mean() - expected) < 1e-9


def test_mean_with_dtype_object_containing_numeric_strings():
    """Series with object dtype containing numeric strings should fail or handle appropriately."""
    s = pd.Series(["1", "2", "3"], dtype=object)
    with pytest.raises(TypeError):
        s.mean()


def test_mean_negative_values():
    """Mean should work correctly with negative numbers."""
    s = pd.Series([-5, -1, 0, 1, 5])
    assert s.mean() == 0.0


def test_mean_very_large_values():
    """Mean should handle very large values without overflow."""
    s = pd.Series([1e300, 1e300, 1e300])
    assert s.mean() == 1e300


def test_mean_very_small_values():
    """Mean should handle very small values correctly."""
    s = pd.Series([1e-300, 1e-300, 1e-300])
    assert s.mean() == 1e-300


def test_mean_integer_series_returns_float():
    """Integer Series should return float type for mean."""
    s = pd.Series([1, 2, 3])
    result = s.mean()
    assert isinstance(result, float)


def test_mean_datetimelike_returns_timedelta_mean():
    """Mean of timedelta Series should return timedelta."""
    s = pd.Series(pd.to_timedelta(["1 days", "2 days", "3 days"]))
    result = s.mean()
    assert isinstance(result, pd.Timedelta)
    assert result == pd.Timedelta("2 days")
