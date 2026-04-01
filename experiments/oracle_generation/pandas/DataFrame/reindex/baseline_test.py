"""
Baseline property-based tests for pandas.DataFrame.reindex.

Properties tested:
1. Output index equals the requested new index
2. Values present in both original and new index are preserved
3. Values absent from original index are filled with NaN (default)
4. fill_value replaces NaN for missing entries
5. Column reindexing: same rules apply to columns axis
6. Both axes reindexed simultaneously
7. axis-style calling convention (labels + axis=) matches keyword convention
8. ffill/bfill methods propagate values on monotonic index
9. limit parameter caps consecutive fill count
10. Reindexing with the same index returns equal values
11. Result shape: len(new_index) rows × original columns (or len(new_columns) columns)
12. MultiIndex level broadcasting
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, indexes, range_indexes


# ---------------------------------------------------------------------------
# Helpers / shared strategies
# ---------------------------------------------------------------------------

int_or_float = st.one_of(st.integers(-100, 100), st.floats(-1e6, 1e6, allow_nan=False))

small_index_elements = st.one_of(
    st.integers(0, 20),
    st.text(alphabet="abcdefghij", min_size=1, max_size=3),
)


def make_simple_df_strategy(min_rows=1, max_rows=8, min_cols=1, max_cols=4):
    """Strategy producing DataFrames with integer/float values and simple labels."""
    return data_frames(
        columns=[
            column(f"c{i}", elements=st.floats(-100, 100, allow_nan=False))
            for i in range(max_cols)
        ],
        index=range_indexes(min_size=min_rows, max_size=max_rows),
    )


# ---------------------------------------------------------------------------
# 1. Output index equals the requested new index
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column("v", elements=st.floats(-10, 10, allow_nan=False))],
        index=indexes(elements=small_index_elements, min_size=0, max_size=6, unique=True),
    ),
    new_labels=st.lists(small_index_elements, min_size=0, max_size=8),
)
@settings(max_examples=100)
def test_output_index_equals_new_index(df, new_labels):
    new_idx = pd.Index(new_labels)
    result = df.reindex(new_idx)
    pd.testing.assert_index_equal(result.index, new_idx)


# ---------------------------------------------------------------------------
# 2. Values present in both original and new index are preserved (no fill)
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column("v", elements=st.floats(-10, 10, allow_nan=False))],
        index=indexes(elements=small_index_elements, min_size=1, max_size=6, unique=True),
    ),
)
@settings(max_examples=100)
def test_existing_labels_values_preserved(df):
    """Reindexing with a subset of the existing index preserves values."""
    assume(len(df) > 0)
    subset = df.index.tolist()
    # Shuffle via reverse to avoid same-object identity issues
    new_idx = pd.Index(subset[::-1])
    result = df.reindex(new_idx)
    for label in new_idx:
        assert result.loc[label, "v"] == df.loc[label, "v"]


# ---------------------------------------------------------------------------
# 3. Missing labels filled with NaN by default
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column("v", elements=st.floats(-10, 10, allow_nan=False))],
        index=indexes(elements=st.integers(0, 10), min_size=1, max_size=5, unique=True),
    ),
    extra=st.lists(st.integers(50, 100), min_size=1, max_size=4),
)
@settings(max_examples=100)
def test_missing_labels_filled_with_nan(df, extra):
    """Labels not in original index produce NaN rows."""
    new_idx = pd.Index(extra)
    result = df.reindex(new_idx)
    assert result.isna().all(axis=None)


# ---------------------------------------------------------------------------
# 4. fill_value replaces NaN for missing entries
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column("v", elements=st.floats(-10, 10, allow_nan=False))],
        index=indexes(elements=st.integers(0, 10), min_size=1, max_size=5, unique=True),
    ),
    extra=st.lists(st.integers(50, 100), min_size=1, max_size=4),
    fill_val=st.floats(-999, 999, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_fill_value_used_for_missing(df, extra, fill_val):
    new_idx = pd.Index(extra)
    result = df.reindex(new_idx, fill_value=fill_val)
    assert (result["v"] == fill_val).all()


# ---------------------------------------------------------------------------
# 5. Result shape — row reindex
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column(f"c{i}", elements=st.floats(-10, 10, allow_nan=False)) for i in range(3)],
        index=range_indexes(min_size=1, max_size=6),
    ),
    new_labels=st.lists(st.integers(0, 20), min_size=0, max_size=8),
)
@settings(max_examples=100)
def test_result_shape_row_reindex(df, new_labels):
    new_idx = pd.Index(new_labels)
    result = df.reindex(new_idx)
    assert result.shape == (len(new_idx), df.shape[1])


# ---------------------------------------------------------------------------
# 6. Column reindexing — output columns equal requested columns
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column(f"col{i}", elements=st.floats(-10, 10, allow_nan=False)) for i in range(4)],
        index=range_indexes(min_size=1, max_size=5),
    ),
    new_cols=st.lists(
        st.sampled_from([f"col{i}" for i in range(4)] + ["extra1", "extra2"]),
        min_size=1, max_size=6,
    ),
)
@settings(max_examples=100)
def test_column_reindex_output_columns(df, new_cols):
    new_col_idx = pd.Index(new_cols)
    result = df.reindex(columns=new_col_idx)
    pd.testing.assert_index_equal(result.columns, new_col_idx)


# ---------------------------------------------------------------------------
# 7. Column reindex — existing columns preserved, new columns are NaN
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column(f"col{i}", elements=st.floats(-10, 10, allow_nan=False)) for i in range(3)],
        index=range_indexes(min_size=1, max_size=5),
    ),
)
@settings(max_examples=100)
def test_column_reindex_existing_preserved_new_nan(df):
    existing = df.columns.tolist()
    new_cols = existing + ["new_col_x"]
    result = df.reindex(columns=new_cols)
    # Existing columns unchanged
    for col in existing:
        pd.testing.assert_series_equal(result[col], df[col])
    # New column is all NaN
    assert result["new_col_x"].isna().all()


# ---------------------------------------------------------------------------
# 8. axis-style calling convention matches keyword convention
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column(f"col{i}", elements=st.floats(-10, 10, allow_nan=False)) for i in range(3)],
        index=indexes(elements=st.integers(0, 10), min_size=1, max_size=5, unique=True),
    ),
    new_labels=st.lists(st.integers(0, 15), min_size=1, max_size=6),
)
@settings(max_examples=100)
def test_axis_style_vs_keyword_index(df, new_labels):
    """labels + axis='index' must equal index=labels."""
    new_idx = pd.Index(new_labels)
    result_kw = df.reindex(index=new_idx)
    result_ax = df.reindex(new_idx, axis="index")
    pd.testing.assert_frame_equal(result_kw, result_ax)


@given(
    df=data_frames(
        columns=[column(f"col{i}", elements=st.floats(-10, 10, allow_nan=False)) for i in range(3)],
        index=range_indexes(min_size=1, max_size=5),
    ),
    new_cols=st.lists(
        st.sampled_from([f"col{i}" for i in range(3)] + ["extra"]),
        min_size=1, max_size=5,
    ),
)
@settings(max_examples=100)
def test_axis_style_vs_keyword_columns(df, new_cols):
    """labels + axis='columns' must equal columns=labels."""
    new_col_idx = pd.Index(new_cols)
    result_kw = df.reindex(columns=new_col_idx)
    result_ax = df.reindex(new_col_idx, axis="columns")
    pd.testing.assert_frame_equal(result_kw, result_ax)


# ---------------------------------------------------------------------------
# 9. Reindexing with same index returns identical values
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column("v", elements=st.floats(-10, 10, allow_nan=False))],
        index=indexes(elements=small_index_elements, min_size=1, max_size=6, unique=True),
    ),
)
@settings(max_examples=100)
def test_reindex_same_index_returns_equal_values(df):
    result = df.reindex(df.index)
    pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# 10. ffill method on monotonic index propagates last valid forward
# ---------------------------------------------------------------------------

def test_ffill_propagates_forward():
    """Forward-fill: NaN gaps after a valid value are filled with that value."""
    date_idx = pd.date_range("2010-01-01", periods=4, freq="D")
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]}, index=date_idx)
    new_idx = pd.date_range("2009-12-31", periods=7, freq="D")
    result = df.reindex(new_idx, method="ffill")
    # 2009-12-31 is before the data, so still NaN
    assert np.isnan(result.loc["2009-12-31", "v"])
    # Days within original range keep ffill
    assert result.loc["2010-01-03", "v"] == 3.0
    assert result.loc["2010-01-04", "v"] == 4.0
    # Day after last entry is ffilled with last value
    assert result.loc["2010-01-05", "v"] == 4.0


# ---------------------------------------------------------------------------
# 11. bfill method on monotonic index back-fills from next valid value
# ---------------------------------------------------------------------------

def test_bfill_propagates_backward():
    date_idx = pd.date_range("2010-01-02", periods=3, freq="D")
    df = pd.DataFrame({"v": [10.0, 20.0, 30.0]}, index=date_idx)
    new_idx = pd.date_range("2010-01-01", periods=5, freq="D")
    result = df.reindex(new_idx, method="bfill")
    # 2010-01-01 is before data — back-fill with first valid (10.0)
    assert result.loc["2010-01-01", "v"] == 10.0
    # Dates after the last entry have no next value → NaN
    assert np.isnan(result.loc["2010-01-05", "v"])


# ---------------------------------------------------------------------------
# 12. limit parameter caps number of consecutive fills
# ---------------------------------------------------------------------------

def test_ffill_limit_caps_consecutive_fill():
    # Sparse original index; dense new index exposes gaps where limit applies.
    # method+limit in reindex applies to new positions NOT in the original index.
    idx = pd.Index([0, 5])
    df = pd.DataFrame({"v": [1.0, 5.0]}, index=idx)
    new_idx = pd.Index([0, 1, 2, 3, 4, 5])
    result = df.reindex(new_idx, method="ffill", limit=2)
    # position 0 is exact match
    assert result.loc[0, "v"] == 1.0
    # positions 1, 2 are within limit → filled with 1.0
    assert result.loc[1, "v"] == 1.0
    assert result.loc[2, "v"] == 1.0
    # positions 3, 4 exceed limit → NaN
    assert np.isnan(result.loc[3, "v"])
    assert np.isnan(result.loc[4, "v"])
    # position 5 is exact match → 5.0
    assert result.loc[5, "v"] == 5.0


# ---------------------------------------------------------------------------
# 13. bfill limit caps consecutive backward fill
# ---------------------------------------------------------------------------

def test_bfill_limit_caps_consecutive_fill():
    # Sparse original index; dense new index exposes gaps where limit applies.
    idx = pd.Index([0, 5])
    df = pd.DataFrame({"v": [1.0, 5.0]}, index=idx)
    new_idx = pd.Index([0, 1, 2, 3, 4, 5])
    result = df.reindex(new_idx, method="bfill", limit=2)
    # position 5 is exact match → 5.0
    assert result.loc[5, "v"] == 5.0
    # positions 3, 4 are within limit → back-filled with 5.0
    assert result.loc[4, "v"] == 5.0
    assert result.loc[3, "v"] == 5.0
    # positions 1, 2 exceed limit → NaN
    assert np.isnan(result.loc[1, "v"])
    assert np.isnan(result.loc[2, "v"])
    # position 0 is exact match → 1.0
    assert result.loc[0, "v"] == 1.0


# ---------------------------------------------------------------------------
# 14. fill_value does not affect rows that exist in original index
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column("v", elements=st.floats(-10, 10, allow_nan=False))],
        index=indexes(elements=st.integers(0, 10), min_size=1, max_size=5, unique=True),
    ),
    fill_val=st.floats(-999, 999, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_fill_value_does_not_affect_existing_rows(df, fill_val):
    assume(len(df) > 0)
    # Use only existing labels → no missing rows → fill_value irrelevant
    result = df.reindex(df.index, fill_value=fill_val)
    pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# 15. Both index and columns can be reindexed simultaneously
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column(f"col{i}", elements=st.floats(-10, 10, allow_nan=False)) for i in range(3)],
        index=indexes(elements=st.integers(0, 10), min_size=1, max_size=5, unique=True),
    ),
    new_row_labels=st.lists(st.integers(0, 12), min_size=1, max_size=5),
    new_col_labels=st.lists(
        st.sampled_from([f"col{i}" for i in range(3)] + ["extra"]),
        min_size=1, max_size=4,
    ),
)
@settings(max_examples=100)
def test_simultaneous_index_and_column_reindex_shape(df, new_row_labels, new_col_labels):
    new_idx = pd.Index(new_row_labels)
    new_cols = pd.Index(new_col_labels)
    result = df.reindex(index=new_idx, columns=new_cols)
    assert result.shape == (len(new_idx), len(new_cols))


# ---------------------------------------------------------------------------
# 16. Empty new index produces empty DataFrame with correct columns
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column(f"c{i}", elements=st.floats(-10, 10, allow_nan=False)) for i in range(3)],
        index=range_indexes(min_size=1, max_size=5),
    ),
)
@settings(max_examples=50)
def test_empty_new_index_produces_empty_dataframe(df):
    result = df.reindex(pd.Index([]))
    assert len(result) == 0
    pd.testing.assert_index_equal(result.columns, df.columns)


# ---------------------------------------------------------------------------
# 17. Reindexing with superset of original index: original rows intact
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column("v", elements=st.floats(-10, 10, allow_nan=False))],
        index=indexes(elements=st.integers(0, 5), min_size=1, max_size=4, unique=True),
    ),
    extra=st.lists(st.integers(100, 110), min_size=1, max_size=3),
)
@settings(max_examples=100)
def test_superset_reindex_original_rows_intact(df, extra):
    # Deduplicate extra to avoid multi-row .loc returns
    extra = list(dict.fromkeys(extra))
    new_labels = df.index.tolist() + extra
    new_idx = pd.Index(new_labels)
    result = df.reindex(new_idx)
    for label in df.index:
        assert result.loc[label, "v"] == df.loc[label, "v"]
    for label in extra:
        assert np.isnan(result.loc[label, "v"])


# ---------------------------------------------------------------------------
# 18. Numeric axis argument (0 / 1) equivalent to string ('index'/'columns')
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column(f"col{i}", elements=st.floats(-10, 10, allow_nan=False)) for i in range(3)],
        index=indexes(elements=st.integers(0, 10), min_size=1, max_size=5, unique=True),
    ),
    new_labels=st.lists(st.integers(0, 12), min_size=1, max_size=5),
)
@settings(max_examples=80)
def test_numeric_axis_0_equals_index_string(df, new_labels):
    new_idx = pd.Index(new_labels)
    result_num = df.reindex(new_idx, axis=0)
    result_str = df.reindex(new_idx, axis="index")
    pd.testing.assert_frame_equal(result_num, result_str)


@given(
    df=data_frames(
        columns=[column(f"col{i}", elements=st.floats(-10, 10, allow_nan=False)) for i in range(3)],
        index=range_indexes(min_size=1, max_size=5),
    ),
    new_cols=st.lists(
        st.sampled_from([f"col{i}" for i in range(3)] + ["extra"]),
        min_size=1, max_size=4,
    ),
)
@settings(max_examples=80)
def test_numeric_axis_1_equals_columns_string(df, new_cols):
    new_col_idx = pd.Index(new_cols)
    result_num = df.reindex(new_col_idx, axis=1)
    result_str = df.reindex(new_col_idx, axis="columns")
    pd.testing.assert_frame_equal(result_num, result_str)


# ---------------------------------------------------------------------------
# 19. nearest method on monotonic index
# ---------------------------------------------------------------------------

def test_nearest_method_picks_closest_value():
    idx = pd.to_datetime(["2010-01-01", "2010-01-03", "2010-01-05"])
    df = pd.DataFrame({"v": [1.0, 3.0, 5.0]}, index=idx)
    new_idx = pd.to_datetime(["2010-01-02"])
    result = df.reindex(new_idx, method="nearest")
    # 2010-01-02 is equidistant from 01-01 and 01-03; pandas picks the earlier one
    assert result.loc["2010-01-02", "v"] in (1.0, 3.0)


# ---------------------------------------------------------------------------
# 20. MultiIndex level: reindex along a level
# ---------------------------------------------------------------------------

def test_multiindex_level_reindex():
    arrays = [["bar", "bar", "baz", "baz"], ["one", "two", "one", "two"]]
    idx = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
    df = pd.DataFrame({"v": [1, 2, 3, 4]}, index=idx)
    result = df.reindex(["bar"], level="first")
    assert list(result.index.get_level_values("first")) == ["bar", "bar"]
    assert list(result["v"]) == [1, 2]


# ---------------------------------------------------------------------------
# 21. Method=None (default) does not fill; original NaNs stay NaN
# ---------------------------------------------------------------------------

def test_method_none_does_not_fill():
    idx = pd.RangeIndex(5)
    df = pd.DataFrame({"v": [1.0, np.nan, 3.0, np.nan, 5.0]}, index=idx)
    result = df.reindex(pd.RangeIndex(5), method=None)
    assert np.isnan(result.loc[1, "v"])
    assert np.isnan(result.loc[3, "v"])


# ---------------------------------------------------------------------------
# 22. Reindex is idempotent: applying twice with same index is same as once
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[column("v", elements=st.floats(-10, 10, allow_nan=False))],
        index=indexes(elements=st.integers(0, 10), min_size=1, max_size=5, unique=True),
    ),
    new_labels=st.lists(st.integers(0, 12), min_size=1, max_size=6),
)
@settings(max_examples=80)
def test_reindex_idempotent(df, new_labels):
    new_idx = pd.Index(new_labels)
    once = df.reindex(new_idx)
    twice = once.reindex(new_idx)
    pd.testing.assert_frame_equal(once, twice)


# ---------------------------------------------------------------------------
# 23. pad/ffill aliases are equivalent
# ---------------------------------------------------------------------------

def test_pad_and_ffill_are_equivalent():
    idx = pd.date_range("2010-01-01", periods=4, freq="D")
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    new_idx = pd.date_range("2009-12-30", periods=8, freq="D")
    result_pad = df.reindex(new_idx, method="pad")
    result_ffill = df.reindex(new_idx, method="ffill")
    pd.testing.assert_frame_equal(result_pad, result_ffill)


# ---------------------------------------------------------------------------
# 24. backfill/bfill aliases are equivalent
# ---------------------------------------------------------------------------

def test_backfill_and_bfill_are_equivalent():
    idx = pd.date_range("2010-01-01", periods=4, freq="D")
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    new_idx = pd.date_range("2009-12-30", periods=8, freq="D")
    result_backfill = df.reindex(new_idx, method="backfill")
    result_bfill = df.reindex(new_idx, method="bfill")
    pd.testing.assert_frame_equal(result_backfill, result_bfill)
