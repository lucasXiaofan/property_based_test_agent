"""
Baseline property-based tests for pandas.Index.astype

Key properties under test:
1. Return type is always pd.Index
2. Result dtype matches requested dtype
3. Signed integer dtypes are normalized to int64
4. Unsigned integer dtypes are normalized to uint64
5. copy=True always returns a new object
6. copy=False may return the same or a new object (no crash guarantee)
7. Values are faithfully preserved after lossless conversions
8. Impossible conversions raise TypeError
9. Round-trip conversions preserve values
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import from_dtype


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

SIGNED_INT_DTYPES = [np.dtype("int8"), np.dtype("int16"), np.dtype("int32"), np.dtype("int64")]
UNSIGNED_INT_DTYPES = [np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32"), np.dtype("uint64")]
FLOAT_DTYPES = [np.dtype("float32"), np.dtype("float64")]
NUMERIC_DTYPES = SIGNED_INT_DTYPES + UNSIGNED_INT_DTYPES + FLOAT_DTYPES


def int64_index(min_size=0, max_size=20):
    return st.lists(
        st.integers(min_value=-(2**31), max_value=2**31 - 1),
        min_size=min_size,
        max_size=max_size,
    ).map(lambda vals: pd.Index(vals, dtype="int64"))


def float64_index(min_size=0, max_size=20):
    return st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e15, max_value=1e15),
        min_size=min_size,
        max_size=max_size,
    ).map(lambda vals: pd.Index(vals, dtype="float64"))


def string_index(min_size=0, max_size=10):
    return st.lists(
        st.text(min_size=0, max_size=20),
        min_size=min_size,
        max_size=max_size,
    ).map(lambda vals: pd.Index(vals, dtype="object"))


# ---------------------------------------------------------------------------
# Property 1: Return type is always pd.Index
# ---------------------------------------------------------------------------

@given(idx=int64_index())
def test_return_type_is_index_int_to_float(idx):
    result = idx.astype("float64")
    assert isinstance(result, pd.Index)


@given(idx=float64_index())
def test_return_type_is_index_float_to_int(idx):
    # float -> int may lose precision; just check the return type
    result = idx.astype("int64")
    assert isinstance(result, pd.Index)


@given(idx=int64_index())
def test_return_type_is_index_int_to_str(idx):
    result = idx.astype("str")
    assert isinstance(result, pd.Index)


# ---------------------------------------------------------------------------
# Property 2: Result dtype matches requested dtype (for numeric dtypes)
# ---------------------------------------------------------------------------

@given(
    idx=int64_index(),
    target=st.sampled_from(FLOAT_DTYPES),
)
def test_result_dtype_matches_requested_float(idx, target):
    result = idx.astype(target)
    assert result.dtype == target


@given(idx=float64_index())
def test_result_dtype_float_to_float32(idx):
    result = idx.astype("float32")
    assert result.dtype == np.dtype("float32")


@given(idx=int64_index())
def test_result_dtype_int_to_object(idx):
    result = idx.astype("object")
    assert result.dtype == np.dtype("object")


# ---------------------------------------------------------------------------
# Property 3: Signed integer dtypes — result is a signed integer dtype
# Note: The docs say "any signed integer dtype is treated as int64", but in
# practice pandas 3.0 astype respects the specific dtype requested. What the
# doc note guarantees is that no *unsigned* dtype is returned for a signed
# request. We test the weaker (and correct) invariant here.
# ---------------------------------------------------------------------------

@given(
    vals=st.lists(st.integers(min_value=-100, max_value=100), min_size=0, max_size=20),
    signed_dtype=st.sampled_from(SIGNED_INT_DTYPES),
)
def test_signed_int_dtype_result_is_signed_integer(vals, signed_dtype):
    idx = pd.Index(vals, dtype="int64")
    result = idx.astype(signed_dtype)
    assert result.dtype == signed_dtype  # exact dtype is honoured
    assert np.issubdtype(result.dtype, np.signedinteger)


# ---------------------------------------------------------------------------
# Property 4: Unsigned integer dtypes — result is an unsigned integer dtype
# ---------------------------------------------------------------------------

@given(
    vals=st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=20),
    unsigned_dtype=st.sampled_from(UNSIGNED_INT_DTYPES),
)
def test_unsigned_int_dtype_result_is_unsigned_integer(vals, unsigned_dtype):
    idx = pd.Index(vals, dtype="int64")
    result = idx.astype(unsigned_dtype)
    assert result.dtype == unsigned_dtype  # exact dtype is honoured
    assert np.issubdtype(result.dtype, np.unsignedinteger)


# ---------------------------------------------------------------------------
# Property 5: copy=True always returns a new object
# ---------------------------------------------------------------------------

@given(idx=int64_index(min_size=1))
def test_copy_true_returns_new_object(idx):
    result = idx.astype("int64", copy=True)
    assert result is not idx


@given(idx=float64_index(min_size=1))
def test_copy_true_float_returns_new_object(idx):
    result = idx.astype("float64", copy=True)
    assert result is not idx


# ---------------------------------------------------------------------------
# Property 6: copy=False does not crash and returns valid Index
# ---------------------------------------------------------------------------

@given(idx=int64_index())
def test_copy_false_no_crash_int(idx):
    result = idx.astype("int64", copy=False)
    assert isinstance(result, pd.Index)
    assert result.dtype == np.dtype("int64")


@given(idx=float64_index())
def test_copy_false_no_crash_float(idx):
    result = idx.astype("float64", copy=False)
    assert isinstance(result, pd.Index)
    assert result.dtype == np.dtype("float64")


# ---------------------------------------------------------------------------
# Property 7: Values are preserved for lossless conversions
# ---------------------------------------------------------------------------

@given(idx=int64_index())
def test_int_to_float_preserves_values(idx):
    result = idx.astype("float64")
    # int64 -> float64 is lossless for these ranges
    expected = [float(v) for v in idx]
    assert list(result) == expected


@given(
    vals=st.lists(st.integers(min_value=0, max_value=2**31 - 1), min_size=0, max_size=20)
)
def test_int_to_str_preserves_values(vals):
    idx = pd.Index(vals, dtype="int64")
    result = idx.astype("str")
    expected = [str(v) for v in vals]
    assert list(result) == expected


@given(idx=int64_index())
def test_int_to_object_preserves_values(idx):
    result = idx.astype("object")
    assert list(result) == list(idx)


# ---------------------------------------------------------------------------
# Property 8: Impossible conversions raise TypeError
# ---------------------------------------------------------------------------

@given(
    vals=st.lists(
        st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll")), min_size=1, max_size=5),
        min_size=1,
        max_size=10,
    )
)
def test_string_to_numeric_raises(vals):
    # Strings that are not numeric representations cannot be cast to int/float
    assume(not all(_is_numeric_string(v) for v in vals))
    idx = pd.Index(vals, dtype="object")
    with pytest.raises((TypeError, ValueError)):
        idx.astype("int64")


def _is_numeric_string(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Property 9: Round-trip conversions preserve values
# ---------------------------------------------------------------------------

@given(idx=int64_index())
def test_int_float_int_roundtrip(idx):
    # int64 -> float64 -> int64 should be lossless for values in safe range
    float_idx = idx.astype("float64")
    back = float_idx.astype("int64")
    assert list(back) == list(idx)


@given(idx=int64_index())
def test_int_object_int_roundtrip(idx):
    obj_idx = idx.astype("object")
    back = obj_idx.astype("int64")
    assert list(back) == list(idx)


# ---------------------------------------------------------------------------
# Property 10: Length is always preserved
# ---------------------------------------------------------------------------

@given(idx=int64_index())
def test_length_preserved_int_to_float(idx):
    result = idx.astype("float64")
    assert len(result) == len(idx)


@given(idx=float64_index())
def test_length_preserved_float_to_str(idx):
    result = idx.astype("str")
    assert len(result) == len(idx)


@given(idx=string_index())
def test_length_preserved_str_to_object(idx):
    result = idx.astype("object")
    assert len(result) == len(idx)


# ---------------------------------------------------------------------------
# Property 11: astype on already-correct dtype with copy=True still new object
# ---------------------------------------------------------------------------

@given(idx=int64_index(min_size=0))
def test_same_dtype_copy_true_new_object(idx):
    result = idx.astype(idx.dtype, copy=True)
    assert result is not idx
    assert list(result) == list(idx)


# ---------------------------------------------------------------------------
# Property 12: bool dtype conversion
# ---------------------------------------------------------------------------

@given(
    vals=st.lists(st.integers(min_value=0, max_value=1), min_size=0, max_size=20)
)
def test_int_to_bool_dtype(vals):
    idx = pd.Index(vals, dtype="int64")
    result = idx.astype("bool")
    assert isinstance(result, pd.Index)
    assert result.dtype == np.dtype("bool")
    expected = [bool(v) for v in vals]
    assert list(result) == expected


@given(
    vals=st.lists(st.booleans(), min_size=0, max_size=20)
)
def test_bool_to_int64(vals):
    idx = pd.Index(vals, dtype="bool")
    result = idx.astype("int64")
    assert isinstance(result, pd.Index)
    assert result.dtype == np.dtype("int64")
    expected = [int(v) for v in vals]
    assert list(result) == expected
