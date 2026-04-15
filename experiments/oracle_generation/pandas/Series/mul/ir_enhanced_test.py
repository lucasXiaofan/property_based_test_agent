import hypothesis
import pandas as pd
import pytest
from hypothesis import given, settings, assume, example
import hypothesis.strategies as st


# ==================== BASELINE TESTS ====================


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
    """BASELINE: Test mul is equivalent to * operator with scalar"""
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
    """BASELINE: Test mul is equivalent to * operator with series"""
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
    """BASELINE: Test fill_value replaces NaN before multiplication"""
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
    """BASELINE: Test multiplying by zero gives zeros"""
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
    """BASELINE: Test scalar multiplication preserves length"""
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
    """BASELINE: Test index is preserved for scalar multiplication"""
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
    """BASELINE: Test mul is commutative with rmul for scalars"""
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
    """BASELINE: Test result is a Series"""
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
    """BASELINE: Test element-wise multiplication correctness"""
    s1 = pd.Series(data, dtype=float)
    s2 = pd.Series(data, dtype=float)
    result = s1.mul(s2)
    for i in range(len(s1)):
        assert result.iloc[i] == s1.iloc[i] * s2.iloc[i], (
            f"Mismatch at index {i}: {result.iloc[i]} != {s1.iloc[i]} * {s2.iloc[i]}"
        )


# ==================== NEW TESTS FROM IR ====================


@given(
    self_data=st.lists(
        st.one_of(
            st.floats(
                allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
            ),
            st.just(float("nan")),
        ),
        min_size=2,
        max_size=5,
    ),
    other_data=st.lists(
        st.one_of(
            st.floats(
                allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
            ),
            st.just(float("nan")),
        ),
        min_size=2,
        max_size=5,
    ),
)
@example(self_data=[1.0, float("nan")], other_data=[float("nan"), 2.0])
@settings(max_examples=100)
def test_both_nan_stays_nan_regardless_fill_value(self_data, other_data):
    """IR ENHANCED: If both corresponding Series locations are NaN, result is NaN regardless of fill_value"""
    s1 = pd.Series(
        self_data, dtype=float, index=["a", "b", "c", "d", "e"][: len(self_data)]
    )
    s2 = pd.Series(
        other_data, dtype=float, index=["a", "b", "c", "d", "e"][: len(other_data)]
    )
    result = s1.mul(s2, fill_value=0.0)
    common_index = s1.index.intersection(s2.index)
    for idx in common_index:
        if pd.isna(s1[idx]) and pd.isna(s2[idx]):
            assert pd.isna(result[idx]), (
                f"Expected NaN at {idx} when both inputs are NaN, got {result[idx]}"
            )


@given(
    self_index=st.lists(
        st.sampled_from(["a", "b", "c", "d", "e"]), min_size=2, max_size=3, unique=True
    ),
    self_values=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=2,
        max_size=3,
    ),
    other_index=st.lists(
        st.sampled_from(["a", "b", "c", "d", "e", "f", "g"]),
        min_size=2,
        max_size=3,
        unique=True,
    ),
    other_values=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=2,
        max_size=3,
    ),
)
@example(
    self_index=["a", "c"],
    self_values=[1.0, 2.0],
    other_index=["b", "d"],
    other_values=[3.0, 4.0],
)
@settings(max_examples=100)
def test_non_overlapping_index_with_fill_value_none_is_nan(
    self_index, self_values, other_index, other_values
):
    """IR ENHANCED: Non-overlapping index elements are NaN when fill_value=None"""
    assume(len(self_index) == len(self_values))
    assume(len(other_index) == len(other_values))
    s1 = pd.Series(dict(zip(self_index, self_values)))
    s2 = pd.Series(dict(zip(other_index, other_values)))
    result = s1.mul(s2, fill_value=None)
    non_overlapping = result[~result.index.isin(s1.index.intersection(s2.index))]
    assert non_overlapping.isna().all(), (
        f"Expected NaN for non-overlapping indices, got {non_overlapping.tolist()}"
    )


@given(
    self_data=st.lists(
        st.one_of(
            st.floats(
                allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
            ),
            st.just(float("nan")),
        ),
        min_size=3,
        max_size=5,
    ),
    scalar=st.floats(
        allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
    ),
)
@example(self_data=[1.0, float("nan"), 3.0], scalar=5.0)
@settings(max_examples=100)
def test_non_overlapping_index_with_fill_value_zero_is_zero(self_data, scalar):
    """IR ENHANCED: NaN values in self are filled with fill_value before multiplying with scalar"""
    assume(scalar != 0.0)
    s1 = pd.Series(
        self_data, dtype=float, index=["a", "b", "c", "d", "e"][: len(self_data)]
    )
    result = s1.mul(scalar, fill_value=0.0)
    nan_mask = s1.isna()
    assert result[nan_mask].eq(0.0).all(), (
        f"Expected 0.0 at NaN positions, got {result[nan_mask].tolist()}"
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
def test_fill_value_with_finite_float(data, scalar):
    """IR ENHANCED: Test fill_value can be any finite float"""
    s = pd.Series(data, dtype=float)
    result = s.mul(scalar, fill_value=0.5)
    nan_indices = s[s.isna()].index
    for idx in nan_indices:
        assert result[idx] == 0.5 * scalar, (
            f"Expected {0.5 * scalar} at index {idx}, got {result[idx]}"
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
def test_axis_parameter_accepted(data):
    """IR ENHANCED: Test axis parameter is accepted (for DataFrame compatibility)"""
    s = pd.Series(data, dtype=float)
    result = s.mul(2.0, axis=0)
    expected = s * 2.0
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
@example(data=[1.0, 2.0, 3.0])
@settings(max_examples=100)
def test_axis_string_index_accepted(data):
    """IR ENHANCED: Test axis='index' parameter is accepted"""
    s = pd.Series(data, dtype=float)
    result = s.mul(2.0, axis="index")
    expected = s * 2.0
    assert result.equals(expected), (
        f"Expected {expected.tolist()}, got {result.tolist()}"
    )
