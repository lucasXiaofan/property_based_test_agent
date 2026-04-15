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
