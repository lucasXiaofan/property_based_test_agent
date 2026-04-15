import hypothesis
import pandas as pd
import pytest
from hypothesis import given, settings, assume, example
import hypothesis.strategies as st


@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=10,
    ),
    scalar=st.floats(
        allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
    ),
)
@example(data=[1.0, 2.0, 3.0], scalar=5.0)
@example(data=[], scalar=1.0)
@settings(max_examples=100)
def test_mul_scalar_equiv_to_star_operator(data, scalar):
    if len(data) == 0:
        return
    s = pd.Series(data, dtype=float)
    result = s.mul(scalar)
    expected = s * scalar
    assert result.equals(expected), (
        f"Expected {expected.tolist()}, got {result.tolist()}"
    )


@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=10,
    ),
    other=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=10,
    ),
)
@example(data=[1.0, 2.0], other=[3.0, 4.0])
@settings(max_examples=100)
def test_mul_series_equiv_to_star_operator(data, other):
    s1 = pd.Series(data, dtype=float)
    s2 = pd.Series(other, dtype=float)
    result = s1.mul(s2)
    expected = s1 * s2
    assert result.equals(expected), (
        f"Expected {expected.tolist()}, got {result.tolist()}"
    )


@given(
    data=st.lists(
        st.one_of(
            st.floats(
                allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
            ),
            st.just(float("nan")),
        ),
        min_size=1,
        max_size=10,
    ),
    scalar=st.floats(
        allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
    ),
)
@example(data=[1.0, float("nan"), 3.0], scalar=5.0)
@settings(max_examples=100)
def test_fill_value_replaces_nan_before_multiply(data, scalar):
    s = pd.Series(data, dtype=float)
    result = s.mul(scalar, fill_value=0.0)
    nan_indices = s[s.isna()].index
    for idx in nan_indices:
        assert result[idx] == 0.0 * scalar, (
            f"Expected 0.0 at index {idx}, got {result[idx]}"
        )


@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=10,
    ),
)
@example(data=[1.0, 2.0, 3.0])
@settings(max_examples=100)
def test_multiply_by_zero_gives_zeros(data):
    s = pd.Series(data, dtype=float)
    result = s.mul(0.0)
    assert (result == 0.0).all(), f"Expected all zeros, got {result.tolist()}"


@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=10,
    ),
    scalar=st.floats(
        allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
    ),
)
@example(data=[1.0, 2.0, 3.0], scalar=5.0)
@settings(max_examples=100)
def test_scalar_multiply_length_preserved(data, scalar):
    s = pd.Series(data, dtype=float)
    result = s.mul(scalar)
    assert len(result) == len(s), f"Expected length {len(s)}, got {len(result)}"


@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=10,
    ),
    scalar=st.floats(
        allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
    ),
)
@example(data=[1.0, 2.0, 3.0], scalar=5.0)
@settings(max_examples=100)
def test_index_preserved_for_scalar(data, scalar):
    s = pd.Series(
        data,
        dtype=float,
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"][: len(data)],
    )
    result = s.mul(scalar)
    assert result.index.equals(s.index), (
        f"Expected index {s.index.tolist()}, got {result.index.tolist()}"
    )


@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=10,
    ),
    scalar=st.floats(
        allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
    ),
)
@example(data=[1.0, 2.0, 3.0], scalar=5.0)
@settings(max_examples=100)
def test_commutativity_with_scalar(data, scalar):
    s = pd.Series(data, dtype=float)
    result = s.mul(scalar)
    expected = s.rmul(scalar)
    assert result.equals(expected), (
        f"Expected {expected.tolist()}, got {result.tolist()}"
    )


@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=10,
    ),
)
@example(data=[1.0, 2.0])
@settings(max_examples=100)
def test_result_is_series(data):
    s = pd.Series(data, dtype=float)
    result = s.mul(2.0)
    assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"


@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=2,
        max_size=10,
    ),
)
@example(data=[1.0, 2.0, 3.0])
@settings(max_examples=100)
def test_element_wise_correctness(data):
    s1 = pd.Series(data, dtype=float)
    s2 = pd.Series(data, dtype=float)
    result = s1.mul(s2)
    for i in range(len(s1)):
        assert result.iloc[i] == s1.iloc[i] * s2.iloc[i], (
            f"Mismatch at index {i}: {result.iloc[i]} != {s1.iloc[i]} * {s2.iloc[i]}"
        )
