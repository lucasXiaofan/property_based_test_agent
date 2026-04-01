import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st


FINITE_FLOATS = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

SMALL_INTS = st.integers(min_value=-1000, max_value=1000)


@given(st.lists(SMALL_INTS, min_size=1, max_size=20))
def test_mean_matches_manual_average_for_integer_series(values):
    s = pd.Series(values)

    result = s.mean()

    assert result == pytest.approx(sum(values) / len(values))
    assert isinstance(result, float)


@given(st.lists(st.booleans(), min_size=1, max_size=20))
def test_mean_of_booleans_matches_fraction_of_true(values):
    s = pd.Series(values)

    result = s.mean()

    assert result == pytest.approx(sum(values) / len(values))


@given(st.lists(st.one_of(FINITE_FLOATS, st.none()), min_size=1, max_size=20))
def test_mean_skipna_true_matches_dropna_mean(values):
    s = pd.Series(values, dtype="float64")
    non_null = [value for value in values if value is not None]

    result = s.mean(skipna=True)

    if non_null:
        assert result == pytest.approx(sum(non_null) / len(non_null))
    else:
        assert np.isnan(result)


@given(
    st.lists(FINITE_FLOATS, min_size=1, max_size=20),
    st.integers(min_value=1, max_value=5),
)
def test_appending_only_missing_values_does_not_change_skipna_true_mean(values, na_count):
    base = pd.Series(values, dtype="float64")
    with_na = pd.Series(values + [None] * na_count, dtype="float64")

    assert with_na.mean(skipna=True) == pytest.approx(base.mean(skipna=True))


@given(
    st.lists(FINITE_FLOATS, min_size=1, max_size=20),
    st.integers(min_value=1, max_value=5),
)
def test_appending_missing_values_forces_nan_when_skipna_false(values, na_count):
    s = pd.Series(values + [None] * na_count, dtype="float64")

    result = s.mean(skipna=False)

    assert np.isnan(result)


@given(st.integers(min_value=1, max_value=20))
def test_all_na_series_mean_is_nan(n):
    s = pd.Series([None] * n, dtype="float64")

    assert np.isnan(s.mean(skipna=True))
    assert np.isnan(s.mean(skipna=False))


@given(FINITE_FLOATS)
def test_single_element_series_mean_equals_element(value):
    s = pd.Series([value])

    assert s.mean() == value


@given(
    st.lists(st.one_of(FINITE_FLOATS, st.none()), min_size=1, max_size=20),
    st.data(),
)
def test_mean_is_order_invariant_even_with_missing_values(values, data):
    permuted = data.draw(st.permutations(values))
    left = pd.Series(values, dtype="float64")
    right = pd.Series(permuted, dtype="float64")

    left_result = left.mean(skipna=True)
    right_result = right.mean(skipna=True)

    if np.isnan(left_result):
        assert np.isnan(right_result)
    else:
        assert left_result == pytest.approx(right_result)


@given(st.lists(st.booleans(), min_size=1, max_size=20))
def test_boolean_series_and_integer_encoding_have_same_mean(values):
    bool_series = pd.Series(values)
    int_series = bool_series.astype(int)

    assert bool_series.mean() == pytest.approx(int_series.mean())


@given(st.lists(st.one_of(FINITE_FLOATS, st.none()), min_size=1, max_size=20))
def test_mean_does_not_mutate_original_series(values):
    s = pd.Series(values, dtype="float64")
    before = s.copy(deep=True)

    s.mean(skipna=True)
    s.mean(skipna=False)

    pd.testing.assert_series_equal(s, before)
