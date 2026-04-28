import hypothesis
import pandas as pd
import pytest
from hypothesis import given, settings, assume, example
import hypothesis.strategies as st
import numpy as np


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


# ======= IMPROVED TESTS: Non-happy-path and edge cases =======

# Test fill_value with Series alignment (from docs: fill_value affects alignment)
@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=5,
    ),
    other_data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=5,
    ),
)
@example(data=[1.0, 1.0, 1.0], other_data=[1.0, 1.0])
@settings(max_examples=50)
def test_fill_value_with_series_alignment(data, other_data):
    """Test fill_value handles Series with misaligned indexes correctly."""
    s1 = pd.Series(data, index=list("abcdefghij")[: len(data)])
    s2 = pd.Series(other_data, index=list("abdefg")[: len(other_data)])
    result = s1.mul(s2, fill_value=0.0)
    assert result.loc["a"] == s1.loc["a"] * s2.loc["a"]


# Test NaN propagation in Series multiplication (NaN * value = NaN)
def test_nan_propagates_in_series_multiplication():
    """NaN values should propagate in Series multiplication."""
    s1 = pd.Series([1.0, np.nan, 3.0])
    s2 = pd.Series([2.0, 2.0, 2.0])
    result = s1.mul(s2)
    assert pd.isna(result.iloc[1]), f"Expected NaN at index 1, got {result.iloc[1]}"
    assert result.iloc[0] == 2.0
    assert result.iloc[2] == 6.0


# Test fill_value with non-zero value replaces NaN before multiply
def test_nan_filled_before_multiply_with_nonzero_fill():
    """fill_value=2.0 should replace NaN before multiplication."""
    s = pd.Series([1.0, np.nan, 3.0])
    result = s.mul(2.0, fill_value=2.0)
    assert result.iloc[0] == 2.0
    assert result.iloc[1] == 4.0  # fill_value (2.0) * 2.0 = 4.0
    assert result.iloc[2] == 6.0


# Test integer Series multiplication
def test_integer_series_multiplication():
    """Test multiplication with integer Series returns float by default."""
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series([4, 5, 6])
    result = s1.mul(s2)
    assert result.tolist() == [4, 10, 18]


# Test axis parameter exists and works (even if unused for Series)
def test_axis_parameter_accepted():
    """Axis parameter should be accepted for DataFrame compatibility."""
    s = pd.Series([1.0, 2.0, 3.0])
    result = s.mul(2.0, axis=0)
    assert result.tolist() == [2.0, 4.0, 6.0]


# Test with level parameter for MultiIndex
def test_level_parameter_with_multiindex():
    """Test level parameter works with MultiIndex."""
    idx = pd.MultiIndex.from_tuples([(1, "a"), (1, "b"), (2, "a")])
    s1 = pd.Series([1.0, 2.0, 3.0], index=idx)
    s2 = pd.Series([4.0, 5.0, 6.0], index=idx)
    result = s1.mul(s2, level=0)
    expected = pd.Series([4.0, 10.0, 18.0], index=idx)
    assert result.equals(expected)


# Test multiply with Series having different indexes
def test_series_multiplication_misaligned_indexes():
    """Test multiplication with misaligned Series indexes produces NaN for non-matching."""
    s1 = pd.Series([1.0, 2.0], index=["a", "b"])
    s2 = pd.Series([3.0, 4.0], index=["b", "c"])
    result = s1.mul(s2)
    assert pd.isna(result.loc["a"])  # a is only in s1, no match in s2
    assert result.loc["b"] == 2.0 * 3.0  # b matches b
    assert pd.isna(result.loc["c"])  # c is only in s2


# Test with None as other parameter - raises TypeError
def test_none_as_other_raises_or_returns_nan():
    """Multiplying by None raises TypeError."""
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        s.mul(None)


# Test fill_value with both Series having NaN at same position
def test_fill_value_both_missing_gives_nan():
    """When both Series have NaN at same position and fill_value is used, result is NaN."""
    s1 = pd.Series([1.0, np.nan, 3.0], index=["a", "b", "c"])
    s2 = pd.Series([2.0, np.nan, 4.0], index=["a", "b", "c"])
    result = s1.mul(s2, fill_value=0.0)
    assert pd.isna(result.loc["b"]), f"Expected NaN at b, got {result.loc['b']}"
    assert result.loc["a"] == 2.0
    assert result.loc["c"] == 12.0


# Test negative multiplication
def test_negative_scalar_multiplication():
    """Test multiplication with negative scalar."""
    s = pd.Series([1.0, 2.0, 3.0])
    result = s.mul(-2.0)
    assert result.tolist() == [-2.0, -4.0, -6.0]


# Test with float16 dtype
def test_float16_dtype():
    """Test multiplication preserves float16 dtype."""
    s = pd.Series([1.0, 2.0, 3.0], dtype=np.float16)
    result = s.mul(2.0)
    assert result.dtype == np.float16


# Test empty Series with scalar
def test_empty_series_with_scalar():
    """Test empty Series multiplication returns empty Series."""
    s = pd.Series([], dtype=float)
    result = s.mul(5.0)
    assert len(result) == 0
    assert isinstance(result, pd.Series)
