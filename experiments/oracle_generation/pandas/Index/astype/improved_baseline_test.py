import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
import hypothesis.strategies as st


@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.integers(min_value=-(10**9), max_value=10**9), min_size=1, max_size=20
        ),
    ),
    dtype=st.sampled_from(["float", "int64", "float64", "int32", "object"]),
    copy=st.booleans(),
)
@settings(max_examples=100)
def test_astype_returns_index(idx, dtype, copy):
    result = idx.astype(dtype, copy=copy)
    assert isinstance(result, pd.Index)


@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.integers(min_value=-(10**9), max_value=10**9), min_size=1, max_size=20
        ),
    )
)
@settings(max_examples=50)
def test_astype_int_to_float_values_preserved(idx):
    result = idx.astype("float")
    expected = [float(v) for v in idx]
    assert list(result) == expected


@given(
    idx=st.builds(
        pd.Index,
        st.lists(st.integers(min_value=0, max_value=10**9), min_size=1, max_size=20),
    ),
    dtype=st.sampled_from(
        ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
    ),
)
@settings(max_examples=50)
def test_astype_int_dtypes_accepted(idx, dtype):
    result = idx.astype(dtype)
    assert isinstance(result, pd.Index)


@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.integers(min_value=-(10**9), max_value=10**9), min_size=1, max_size=20
        ),
    )
)
@settings(max_examples=50)
def test_astype_copy_true_returns_new_object(idx):
    result = idx.astype("float", copy=True)
    assert result is not idx


@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.integers(min_value=-(10**9), max_value=10**9), min_size=1, max_size=20
        ),
    )
)
@settings(max_examples=50)
def test_astype_preserves_length(idx):
    result = idx.astype("float")
    assert len(result) == len(idx)


@given(idx=st.builds(pd.Index, st.just([])))
@settings(max_examples=10)
def test_astype_empty_index(idx):
    result = idx.astype("float")
    assert len(result) == 0 and isinstance(result, pd.Index)


@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.floats(
                allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9
            ),
            min_size=1,
            max_size=20,
        ),
    )
)
@settings(max_examples=50)
def test_astype_float_to_int(idx):
    assume(not any(np.isnan(x) for x in idx))
    result = idx.astype("int64")
    assert result.dtype == np.dtype("int64")


# === NEW TESTS FOR EDGE CASES AND NON-HAPPY-PATH BEHAVIOR ===

# Test: small int dtype is preserved in output (int8 stays int8)
@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=1,
            max_size=20,
        ),
    ),
)
@settings(max_examples=30)
def test_astype_int8_preserves_small_dtype(idx):
    result = idx.astype("int8")
    assert result.dtype == np.dtype("int8")


# Test: copy=False returns same object when dtype is unchanged
@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.integers(min_value=-(10**9), max_value=10**9), min_size=1, max_size=20
        ),
    )
)
@settings(max_examples=30)
def test_astype_copy_false_returns_same_object_when_dtype_same(idx):
    result = idx.astype(idx.dtype, copy=False)
    assert result is idx


# Test: copy=False with compatible dtype returns new object (not same)
@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.integers(min_value=-(10**9), max_value=10**9), min_size=1, max_size=20
        ),
    )
)
@settings(max_examples=30)
def test_astype_copy_false_with_different_dtype_returns_new_object(idx):
    result = idx.astype("float64", copy=False)
    assert result is not idx


# Test: TypeError/ValueError raised when conversion is impossible (non-numeric string to int)
def test_astype_impossible_conversion_raises_error():
    idx = pd.Index(["a", "b", "c"])
    try:
        idx.astype("int64")
        assert False, "Expected TypeError or ValueError"
    except (TypeError, ValueError):
        pass


# Test: TypeError/ValueError when converting mixed string/numeric to numeric
def test_astype_mixed_string_numeric_to_int_raises_error():
    idx = pd.Index(["1", "2", "a"])
    try:
        idx.astype("int64")
        assert False, "Expected TypeError or ValueError"
    except (TypeError, ValueError):
        pass


# Test: float with NaN cannot be converted to integer
def test_astype_float_with_nan_to_int_raises_type_error():
    idx = pd.Index([1.0, 2.0, np.nan])
    try:
        idx.astype("int64")
        assert False, "Expected TypeError"
    except (TypeError, ValueError):
        pass


# Test: float with infinity cannot be converted to integer
def test_astype_float_with_inf_to_int_raises_type_error():
    idx = pd.Index([1.0, 2.0, np.inf])
    try:
        idx.astype("int64")
        assert False, "Expected TypeError"
    except (TypeError, ValueError):
        pass


# Test: Boolean index conversion to int
@given(
    idx=st.builds(
        pd.Index,
        st.lists(st.booleans(), min_size=1, max_size=20),
    )
)
@settings(max_examples=30)
def test_astype_bool_to_int(idx):
    result = idx.astype("int64")
    expected = [int(b) for b in idx]
    assert list(result) == expected


# Test: Boolean index conversion to float
@given(
    idx=st.builds(
        pd.Index,
        st.lists(st.booleans(), min_size=1, max_size=20),
    )
)
@settings(max_examples=30)
def test_astype_bool_to_float(idx):
    result = idx.astype("float64")
    expected = [float(b) for b in idx]
    assert list(result) == expected


# Test: String index to object dtype works
@given(
    idx=st.builds(
        pd.Index,
        st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
    )
)
@settings(max_examples=30)
def test_astype_string_to_object(idx):
    result = idx.astype("object")
    assert list(result) == list(idx)


# Test: String index to string dtype works
@given(
    idx=st.builds(
        pd.Index,
        st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
    )
)
@settings(max_examples=30)
def test_astype_string_to_string(idx):
    result = idx.astype("str")
    assert list(result) == [str(x) for x in idx]


# Test: datetime index conversion to int (unix timestamp)
def test_astype_datetime_to_int():
    idx = pd.Index(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    result = idx.astype("int64")
    assert result.dtype == np.dtype("int64")


# Test: datetime index conversion to datetime with different resolution
def test_astype_datetime_to_datetime_ns():
    idx = pd.Index(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    result = idx.astype("datetime64[ns]")
    assert result.dtype == np.dtype("datetime64[ns]")


# Test: timedelta index conversion to int
def test_astype_timedelta_to_int():
    idx = pd.Index(pd.to_timedelta(["1 days", "2 days", "3 days"]))
    result = idx.astype("int64")
    assert result.dtype == np.dtype("int64")


# Test: IntIndex categorical dtype
def test_astype_to_categorical():
    idx = pd.Index([1, 2, 3])
    result = idx.astype("category")
    assert result.dtype == "category"


# Test: MultiIndex conversion is not supported - raises TypeError
def test_astype_multiindex_raises_type_error():
    idx = pd.MultiIndex.from_tuples([(1, 2), (3, 4)])
    try:
        idx.astype("int64")
        assert False, "Expected TypeError"
    except TypeError:
        pass
