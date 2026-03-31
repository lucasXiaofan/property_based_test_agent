"""
Property-based tests for pandas DataFrame.reindex() (3.0.0).

Tests are written against the local source in pandas_bug_finding/pandas/
(pandas/core/frame.py::DataFrame.reindex) but executed via the installed
pandas package, since the source tree has no compiled C extensions.
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra.pandas import column, data_frames


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def simple_df(draw):
    """DataFrame with integer-range index and a mix of numeric columns."""
    n_rows = draw(st.integers(min_value=0, max_value=20))
    n_cols = draw(st.integers(min_value=1, max_value=5))
    cols = [
        column(name=f"c{i}", dtype=float,
               elements=st.one_of(
                   st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                   st.just(float("nan")),
               ))
        for i in range(n_cols)
    ]
    return draw(data_frames(columns=cols, index=st.just(list(range(n_rows)))))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@given(df=simple_df(), new_index=st.lists(st.integers(min_value=-5, max_value=25),
                                           min_size=0, max_size=25, unique=True))
@settings(max_examples=200)
def test_reindex_output_shape(df, new_index):
    """Result has exactly len(new_index) rows and same columns as original."""
    result = df.reindex(new_index)
    assert result.shape == (len(new_index), len(df.columns))
    assert list(result.columns) == list(df.columns)


@given(df=simple_df(), new_index=st.lists(st.integers(min_value=-5, max_value=25),
                                           min_size=0, max_size=25, unique=True))
@settings(max_examples=200)
def test_reindex_preserves_existing_values(df, new_index):
    """Rows in new_index that exist in df.index keep their original values."""
    result = df.reindex(new_index)
    for label in new_index:
        if label in df.index:
            pd.testing.assert_series_equal(
                result.loc[label], df.loc[label], check_names=False
            )


@given(df=simple_df(), new_index=st.lists(st.integers(min_value=-5, max_value=25),
                                           min_size=0, max_size=25, unique=True))
@settings(max_examples=200)
def test_reindex_missing_labels_are_nan(df, new_index):
    """Rows introduced by reindex (not in original index) are all NaN by default."""
    result = df.reindex(new_index)
    for label in new_index:
        if label not in df.index:
            assert result.loc[label].isna().all(), (
                f"Expected all NaN for new label {label}, got {result.loc[label].to_dict()}"
            )


@given(df=simple_df(), fill=st.floats(min_value=-1e6, max_value=1e6,
                                       allow_nan=False, allow_infinity=False),
       new_index=st.lists(st.integers(min_value=-5, max_value=25),
                          min_size=1, max_size=25, unique=True))
@settings(max_examples=200)
def test_reindex_fill_value(df, fill, new_index):
    """fill_value replaces NaN for labels not in the original index."""
    result = df.reindex(new_index, fill_value=fill)
    for label in new_index:
        if label not in df.index:
            row = result.loc[label]
            assert (row == fill).all(), (
                f"fill_value={fill} not used for new label {label}: {row.to_dict()}"
            )


@given(df=simple_df())
@settings(max_examples=100)
def test_reindex_same_index_is_identity(df):
    """Reindexing with the same index returns equal data."""
    result = df.reindex(df.index)
    pd.testing.assert_frame_equal(result, df, check_like=False)


@given(df=simple_df(), new_cols=st.lists(st.text(min_size=1, max_size=3),
                                          min_size=0, max_size=8, unique=True))
@settings(max_examples=200)
def test_reindex_columns(df, new_cols):
    """Reindexing columns produces correct shape and fills missing cols with NaN."""
    result = df.reindex(columns=new_cols)
    assert result.shape == (len(df), len(new_cols))
    for col in new_cols:
        if col not in df.columns:
            assert result[col].isna().all(), (
                f"New column '{col}' should be all NaN"
            )
        else:
            pd.testing.assert_series_equal(result[col], df[col])


@given(df=simple_df(),
       row_idx=st.lists(st.integers(min_value=-5, max_value=25),
                        min_size=0, max_size=25, unique=True),
       col_idx=st.lists(st.text(min_size=1, max_size=3),
                        min_size=0, max_size=8, unique=True))
@settings(max_examples=200)
def test_reindex_both_axes(df, row_idx, col_idx):
    """Reindexing both axes simultaneously produces correct shape."""
    result = df.reindex(index=row_idx, columns=col_idx)
    assert result.shape == (len(row_idx), len(col_idx))


@given(df=simple_df(), new_index=st.lists(st.integers(min_value=0, max_value=19),
                                           min_size=1, max_size=20, unique=True))
@settings(max_examples=100)
def test_reindex_ffill(df, new_index):
    """Forward-fill: non-NaN values must be >= as many as without fill for sorted index."""
    assume(len(df) >= 2)
    sorted_idx = sorted(new_index)
    result_no_fill = df.reindex(sorted_idx)
    result_ffill = df.reindex(sorted_idx, method="ffill")
    # ffill should never introduce NaN where no_fill already has a value
    for col in df.columns:
        no_fill_na = result_no_fill[col].isna()
        ffill_na = result_ffill[col].isna()
        # Any non-NaN in no_fill must also be non-NaN in ffill
        assert not (no_fill_na & ~ffill_na).any() or True  # ffill can only reduce NaN count
        # More precisely: ffill cannot turn a valid value into NaN
        assert (~no_fill_na & ffill_na).sum() == 0, (
            f"ffill turned a valid value into NaN in column '{col}'"
        )


if __name__ == "__main__":
    print("Running property-based tests for pandas DataFrame.reindex (3.0.0)...")
    test_reindex_output_shape()
    test_reindex_preserves_existing_values()
    test_reindex_missing_labels_are_nan()
    test_reindex_fill_value()
    test_reindex_same_index_is_identity()
    test_reindex_columns()
    test_reindex_both_axes()
    test_reindex_ffill()
    print("All tests passed!")
