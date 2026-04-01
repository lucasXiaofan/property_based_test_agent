import pandas as pd
from hypothesis import given, strategies as st
from pandas.testing import assert_series_equal


small_ints = st.integers(min_value=-7, max_value=7)
finite_scalars = st.integers(min_value=-5, max_value=5).map(float)
nonzero_scalars = st.integers(min_value=-5, max_value=5).filter(lambda x: x != 0).map(float)
fill_scalars = st.integers(min_value=-3, max_value=3).map(float)
nan_or_int = st.one_of(st.just(float("nan")), small_ints.map(float))


@st.composite
def numeric_series(draw, min_size=1, max_size=6):
    values = draw(
        st.lists(small_ints, min_size=min_size, max_size=max_size).filter(
            lambda xs: len(xs) > 0
        )
    )
    return pd.Series(values, dtype=float)


@st.composite
def numeric_series_with_nan(draw, min_size=1, max_size=6):
    values = draw(st.lists(nan_or_int, min_size=min_size, max_size=max_size))
    return pd.Series(values, dtype=float)


@st.composite
def same_length_series_pair(draw):
    size = draw(st.integers(min_value=1, max_value=6))
    left = draw(st.lists(small_ints, min_size=size, max_size=size))
    right = draw(st.lists(small_ints, min_size=size, max_size=size))
    return pd.Series(left, dtype=float), pd.Series(right, dtype=float)


@st.composite
def aligned_nan_series_pair(draw):
    size = draw(st.integers(min_value=1, max_value=6))
    left = draw(st.lists(nan_or_int, min_size=size, max_size=size))
    right = draw(st.lists(nan_or_int, min_size=size, max_size=size))
    return pd.Series(left, dtype=float), pd.Series(right, dtype=float)


@st.composite
def partial_overlap_series_pair(draw):
    a = draw(small_ints)
    b = draw(small_ints)
    c = draw(small_ints)
    d = draw(small_ints)
    left = pd.Series({"a": float(a), "b": float(b), "c": float(c)})
    right = pd.Series({"a": float(a if draw(st.booleans()) else d), "d": float(d)})
    return left, right


@given(series=numeric_series(), scalar=finite_scalars)
def test_mul_scalar_matches_star_and_preserves_shape(series, scalar):
    result = series.mul(scalar)
    expected = series * scalar

    assert_series_equal(result, expected)
    assert result.index.equals(series.index)
    assert len(result) == len(series)


@given(pair=same_length_series_pair())
def test_mul_series_same_index_matches_star(pair):
    left, right = pair
    result = left.mul(right)

    assert_series_equal(result, left * right)


@given(pair=same_length_series_pair())
def test_mul_axis_index_matches_axis_zero(pair):
    left, right = pair

    assert_series_equal(left.mul(right, axis=0), left.mul(right, axis="index"))


@given(series=numeric_series(), scalar=finite_scalars)
def test_mul_scalar_matches_rmul(series, scalar):
    assert_series_equal(series.mul(scalar), series.rmul(scalar))


@given(series=numeric_series())
def test_mul_zero_scalar_produces_zero_for_non_missing_values(series):
    result = series.mul(0.0)
    expected = pd.Series(0.0, index=series.index)

    assert_series_equal(result, expected)


@given(series=numeric_series_with_nan(), scalar=nonzero_scalars)
def test_fill_value_replaces_existing_nan_before_scalar_multiplication(series, scalar):
    result = series.mul(scalar, fill_value=0.0)
    expected = series.fillna(0.0) * scalar

    assert_series_equal(result, expected)


@given(series=numeric_series_with_nan())
def test_nan_times_zero_stays_nan_without_fill_value(series):
    result = series.mul(0.0)
    expected = series * 0.0

    assert_series_equal(result, expected)


@given(pair=aligned_nan_series_pair(), fill_value=fill_scalars)
def test_both_missing_positions_remain_missing_even_with_fill(pair, fill_value):
    left, right = pair
    result = left.mul(right, fill_value=fill_value)
    both_missing = left.isna() & right.isna()

    assert result[both_missing].isna().all()


@given(pair=partial_overlap_series_pair())
def test_partial_overlap_without_fill_leaves_non_overlaps_missing(pair):
    left, right = pair
    result = left.mul(right)
    overlap = left.index.intersection(right.index)
    non_overlap = result.index.difference(overlap)

    assert result.index.equals(left.index.union(right.index))
    assert result[non_overlap].isna().all()
    assert_series_equal(result[overlap], (left * right)[overlap])


@given(pair=partial_overlap_series_pair(), fill_value=fill_scalars)
def test_partial_overlap_with_fill_uses_fill_for_one_sided_missing(pair, fill_value):
    left, right = pair
    result = left.mul(right, fill_value=fill_value)
    expected = pd.Series(
        {
            "a": left["a"] * right["a"],
            "b": left["b"] * fill_value,
            "c": left["c"] * fill_value,
            "d": fill_value * right["d"],
        },
        dtype=float,
    )
    expected = expected.reindex(result.index)

    assert_series_equal(result, expected)


@given(series=numeric_series_with_nan(), scalar=nonzero_scalars)
def test_mul_does_not_mutate_input_series(series, scalar):
    original = series.copy(deep=True)
    _ = series.mul(scalar, fill_value=0.0)

    assert_series_equal(series, original)
