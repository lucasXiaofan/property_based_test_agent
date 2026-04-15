import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st


INT_VALUES = st.lists(
    st.integers(min_value=-10**6, max_value=10**6),
    min_size=0,
    max_size=8,
)
NONNEGATIVE_INT_VALUES = st.lists(
    st.integers(min_value=0, max_value=10**6),
    min_size=0,
    max_size=8,
)
STRING_VALUES = st.lists(st.text(max_size=8), min_size=0, max_size=8)
FLOAT_VALUES = st.lists(
    st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=-10**6,
        max_value=10**6,
    ),
    min_size=0,
    max_size=8,
)


@given(INT_VALUES)
def test_astype_float_preserves_order_length_and_numeric_values(values):
    idx = pd.Index(values, dtype="int64")

    result = idx.astype("float", copy=True)

    assert isinstance(result, pd.Index)
    assert result.dtype == np.dtype("float64")
    assert result is not idx
    assert len(result) == len(idx)
    assert list(result) == [float(v) for v in values]


@given(INT_VALUES)
def test_numpy_float64_dtype_object_matches_string_alias(values):
    idx = pd.Index(values, dtype="int64")

    via_string = idx.astype("float", copy=True)
    via_numpy_dtype = idx.astype(np.dtype("float64"), copy=True)

    assert via_numpy_dtype.dtype == np.dtype("float64")
    assert via_numpy_dtype.equals(via_string)


@given(STRING_VALUES)
def test_astype_object_preserves_string_values(values):
    idx = pd.Index(values)

    result = idx.astype("object", copy=True)

    assert result.dtype == np.dtype("O")
    assert result is not idx
    assert list(result) == values


def test_empty_index_cast_to_float_stays_empty():
    idx = pd.Index([], dtype="int64")

    result = idx.astype("float", copy=True)

    assert isinstance(result, pd.Index)
    assert result.dtype == np.dtype("float64")
    assert result.empty
    assert result is not idx


@given(INT_VALUES)
def test_copy_false_with_same_dtype_can_return_original_index(values):
    idx = pd.Index(values, dtype="int64")

    result = idx.astype("int64", copy=False)

    assert result is idx


@given(INT_VALUES)
def test_copy_false_with_changed_dtype_returns_distinct_index(values):
    idx = pd.Index(values, dtype="int64")

    result = idx.astype("float", copy=False)

    assert result.dtype == np.dtype("float64")
    assert result is not idx
    assert list(result) == [float(v) for v in values]


@given(STRING_VALUES)
def test_incompatible_datetime_cast_raises_type_error(values):
    assume(values)
    idx = pd.Index(values)

    # Some strings coerce successfully (e.g. "" -> NaT), while others raise
    # parsing-related exceptions. Both outcomes are valid for this broad input set.
    try:
        result = idx.astype("datetime64[ns]", copy=True)
    except (TypeError, ValueError):
        return
    assert result.dtype == np.dtype("datetime64[ns]")
    assert len(result) == len(idx)


@given(INT_VALUES)
def test_all_signed_integer_aliases_normalize_to_int64(values):
    assume(values)
    idx = pd.Index(values, dtype="int64")

    for dtype in ("int8", "int16", "int32", "int64"):
        result = idx.astype(dtype, copy=True)
        expected = np.asarray(values, dtype="int64").astype(dtype).tolist()
        assert result.dtype == np.dtype(dtype)
        assert list(result) == expected


@given(NONNEGATIVE_INT_VALUES)
def test_all_unsigned_integer_aliases_normalize_to_uint64(values):
    assume(values)
    idx = pd.Index(values, dtype="int64")

    for dtype in ("uint8", "uint16", "uint32", "uint64"):
        result = idx.astype(dtype, copy=True)
        expected = np.asarray(values, dtype="int64").astype(dtype).tolist()
        assert result.dtype == np.dtype(dtype)
        assert list(result) == expected


@given(FLOAT_VALUES)
def test_float_index_to_object_preserves_exact_iteration_order(values):
    idx = pd.Index(values, dtype="float64")

    result = idx.astype("object", copy=True)

    assert result.dtype == np.dtype("O")
    assert list(result) == list(idx)
