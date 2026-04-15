import numpy as np
import pandas as pd
from hypothesis import given, settings, assume, example
import hypothesis.strategies as st


# ==================== BASELINE TEST CASES ====================


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


# ==================== NEW TEST CASES INSPIRED BY IR ====================


@given(idx=st.builds(pd.Index, st.lists(st.text(max_size=10), min_size=1, max_size=20)))
@settings(max_examples=50)
def test_astype_string_index_to_object(idx):
    result = idx.astype("object")
    assert result.dtype == np.dtype("O")


@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.integers(min_value=-(10**9), max_value=10**9), min_size=1, max_size=20
        ),
    )
)
@settings(max_examples=50)
def test_astype_numpy_dtype_object_accepted(idx):
    result = idx.astype(np.dtype("float64"))
    assert result.dtype == np.dtype("float64")


@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.integers(min_value=-(10**9), max_value=10**9), min_size=1, max_size=20
        ),
    )
)
@settings(max_examples=50)
def test_astype_copy_false_returns_same_object_when_dtype_same(idx):
    result = idx.astype(idx.dtype, copy=False)
    assert result is idx


@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.integers(min_value=-(10**9), max_value=10**9), min_size=1, max_size=20
        ),
    )
)
@settings(max_examples=50)
def test_astype_copy_false_returns_new_object_when_dtype_differs(idx):
    result = idx.astype("float", copy=False)
    assert result is not idx


@example(idx=pd.Index(["a", "b", "c"]))
@example(idx=pd.Index(["x", "y", "z"]))
@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.text(
                max_size=10,
                alphabet=st.characters(whitelist_categories=["Lu", "Ll"]),
                min_size=1,
            ),
            min_size=1,
            max_size=5,
        ),
    )
)
@settings(max_examples=30)
def test_astype_incompatible_dtype_raises_type_error(idx):
    try:
        result = idx.astype("datetime64[ns]")
    except (TypeError, ValueError):
        pass
    else:
        assert False, (
            "Expected TypeError or ValueError for incompatible dtype conversion"
        )


@given(
    idx=st.builds(
        pd.Index,
        st.lists(st.integers(min_value=0, max_value=255), min_size=1, max_size=20),
    )
)
@settings(max_examples=50)
def test_astype_uint8_values_preserved(idx):
    result = idx.astype("uint8")
    assert list(result) == [int(v) for v in idx]


@given(
    idx=st.builds(
        pd.Index,
        st.lists(st.integers(min_value=-128, max_value=127), min_size=1, max_size=20),
    )
)
@settings(max_examples=50)
def test_astype_int8_values_preserved(idx):
    result = idx.astype("int8")
    assert list(result) == [int(v) for v in idx]


@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.floats(
                min_value=-1e300, max_value=1e300, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=10,
        ),
    )
)
@settings(max_examples=30)
def test_astype_large_float_to_object(idx):
    result = idx.astype("object")
    assert result.dtype == np.dtype("O")
    assert list(result) == list(idx)


@given(
    idx=st.builds(
        pd.Index,
        st.lists(
            st.integers(min_value=-(10**9), max_value=10**9), min_size=1, max_size=20
        ),
    )
)
@settings(max_examples=50)
def test_astype_preserves_order(idx):
    result = idx.astype("float")
    assert list(result) == list(idx)


@given(idx=st.builds(pd.Index, st.just([])))
@settings(max_examples=10)
def test_astype_empty_index_dtype_preserved(idx):
    result = idx.astype("int64")
    assert len(result) == 0
    assert result.dtype == np.dtype("int64")
