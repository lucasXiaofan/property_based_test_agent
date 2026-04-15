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
