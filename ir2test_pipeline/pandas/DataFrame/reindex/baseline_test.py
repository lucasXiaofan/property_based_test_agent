"""
Baseline property-based tests for pandas DataFrame.reindex (3.0.0).

Docs: https://pandas.pydata.org/pandas-docs/version/3.0.0/reference/api/pandas.DataFrame.reindex.html
Tools: pytest, hypothesis
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, indexes, range_indexes


# ── helpers ───────────────────────────────────────────────────────────────────

def _int_df(min_size=1, max_size=6):
    return data_frames(
        columns=[column("a", dtype=float), column("b", dtype=float)],
        index=indexes(dtype=int, min_size=min_size, max_size=max_size),
    )


def _single_col_df(min_size=1, max_size=6):
    return data_frames(
        columns=[column("x", dtype=float)],
        index=indexes(dtype=int, min_size=min_size, max_size=max_size),
    )


# ── P1: Missing row labels receive NaN by default ────────────────────────────

@given(
    df=_single_col_df(),
    extra=st.lists(
        st.integers(min_value=10_000, max_value=20_000),
        min_size=1, max_size=3, unique=True,
    ),
)
def test_missing_row_labels_get_nan(df, extra):
    """Labels not in the original index receive NaN by default."""
    new_index = list(df.index) + extra
    result = df.reindex(new_index)
    for label in extra:
        assert result.loc[label].isna().all(), (
            f"Label {label} not in original → expected NaN, got {result.loc[label].tolist()}"
        )


# ── P2: reindex always returns a new DataFrame object ────────────────────────

@given(df=_single_col_df())
def test_reindex_returns_new_dataframe_object(df):
    """reindex always returns a new object, not a view of the original."""
    result = df.reindex(df.index)
    assert isinstance(result, pd.DataFrame)
    assert result is not df


# ── P3: Return type is always DataFrame ──────────────────────────────────────

@given(
    df=_int_df(),
    new_labels=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=6),
)
def test_return_type_is_always_dataframe(df, new_labels):
    """reindex returns a DataFrame regardless of index contents."""
    result = df.reindex(new_labels)
    assert isinstance(result, pd.DataFrame)


# ── P4: Existing values are preserved ────────────────────────────────────────

@given(df=_single_col_df(min_size=2, max_size=8))
def test_existing_labels_preserve_values(df):
    """Values for labels present in both old and new index are preserved exactly."""
    assume(len(df) >= 2)
    subset = df.index[: max(1, len(df) // 2)]
    result = df.reindex(subset)
    for label in subset:
        pd.testing.assert_series_equal(result.loc[label], df.loc[label])


# ── P5: fill_value replaces NaN for new (missing) labels ─────────────────────

@given(
    df=_single_col_df(),
    fill_val=st.integers(min_value=-100, max_value=100),
    extra=st.lists(
        st.integers(min_value=50_000, max_value=60_000),
        min_size=1, max_size=3, unique=True,
    ),
)
def test_fill_value_used_for_missing_row_labels(df, fill_val, extra):
    """fill_value is placed at rows whose label was not in the original index."""
    new_index = list(df.index) + extra
    result = df.reindex(new_index, fill_value=fill_val)
    for label in extra:
        assert (result.loc[label] == fill_val).all(), (
            f"Expected fill_value={fill_val} at label {label}"
        )


# ── P6: Column reindexing — new column is NaN-filled ─────────────────────────

@given(
    df=data_frames(
        columns=[column("a", dtype=float), column("b", dtype=float)],
        index=range_indexes(min_size=1, max_size=5),
    )
)
def test_new_column_label_gets_nan(df):
    """Reindexing with an unknown column name produces a NaN-filled column."""
    result = df.reindex(columns=["a", "b", "new_col"])
    assert "new_col" in result.columns
    assert result["new_col"].isna().all()


# ── P7: Column reindexing — removed column disappears ────────────────────────

@given(
    df=data_frames(
        columns=[column("a", dtype=float), column("b", dtype=float)],
        index=range_indexes(min_size=1, max_size=5),
    )
)
def test_removed_column_absent_from_result(df):
    """Columns not in the new column list are dropped from the result."""
    result = df.reindex(columns=["a"])
    assert list(result.columns) == ["a"]
    assert "b" not in result.columns


# ── P8: labels + axis='columns' equivalent to columns= ───────────────────────

@given(
    df=data_frames(
        columns=[column("a", dtype=float), column("b", dtype=float)],
        index=range_indexes(min_size=1, max_size=5),
    )
)
def test_labels_axis_columns_equals_columns_kwarg(df):
    """df.reindex(cols, axis='columns') == df.reindex(columns=cols)."""
    new_cols = ["a", "new_col"]
    result_kw = df.reindex(columns=new_cols)
    result_axis = df.reindex(new_cols, axis="columns")
    pd.testing.assert_frame_equal(result_kw, result_axis)


# ── P9: labels + axis=0 equivalent to index= ──────────────────────────────────

@given(
    df=_single_col_df(),
    new_labels=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=5),
)
def test_labels_axis_0_equals_index_kwarg(df, new_labels):
    """df.reindex(labels, axis=0) == df.reindex(index=labels)."""
    result_kw = df.reindex(index=new_labels)
    result_axis = df.reindex(new_labels, axis=0)
    pd.testing.assert_frame_equal(result_kw, result_axis)


# ── P10: Result shape equals (len(new_index), len(new_cols)) ─────────────────

@given(
    df=data_frames(
        columns=[column("a", dtype=float), column("b", dtype=float)],
        index=range_indexes(min_size=1, max_size=6),
    ),
    new_index=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=8),
    new_cols=st.lists(
        st.sampled_from(["a", "b", "c", "d"]),
        min_size=1, max_size=4, unique=True,
    ),
)
def test_result_shape_matches_new_index_and_columns(df, new_index, new_cols):
    """Output shape is (len(new_index), len(new_cols)) after simultaneous reindex."""
    result = df.reindex(index=new_index, columns=new_cols)
    assert result.shape == (len(new_index), len(new_cols))


# ── P11: Reindex with same index reproduces the DataFrame ────────────────────

@given(df=_single_col_df())
def test_reindex_same_index_reproduces_values(df):
    """Reindexing with an identical index leaves all values unchanged."""
    result = df.reindex(df.index)
    pd.testing.assert_frame_equal(result, df)


# ── P12: fill_value applies to new column labels too ─────────────────────────

@given(
    df=data_frames(
        columns=[column("a", dtype=float)],
        index=range_indexes(min_size=1, max_size=5),
    ),
    fill_val=st.integers(min_value=-10, max_value=10),
)
def test_fill_value_applies_to_new_columns(df, fill_val):
    """fill_value fills newly introduced column positions, not only row positions."""
    result = df.reindex(columns=["a", "new_col"], fill_value=fill_val)
    assert (result["new_col"] == fill_val).all()


# ── P13: Forward fill (ffill) propagates last valid observation ───────────────

def test_ffill_propagates_last_valid_observation():
    """method='ffill' (or 'pad') fills gaps with the preceding valid value."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=idx)
    new_idx = pd.date_range("2019-12-31", periods=6, freq="D")
    result = df.reindex(new_idx, method="ffill")
    # Before first original date → no prior value → NaN
    assert np.isnan(result.loc["2019-12-31", "v"])
    # Dates present in original → original values
    assert result.loc["2020-01-01", "v"] == 1.0
    assert result.loc["2020-01-03", "v"] == 3.0
    # After last original date → filled from last valid
    assert result.loc["2020-01-04", "v"] == 3.0


def test_pad_is_alias_for_ffill():
    """method='pad' produces the same result as method='ffill'."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=idx)
    new_idx = pd.date_range("2019-12-31", periods=6, freq="D")
    result_ffill = df.reindex(new_idx, method="ffill")
    result_pad = df.reindex(new_idx, method="pad")
    pd.testing.assert_frame_equal(result_ffill, result_pad)


# ── P14: Backward fill (bfill) uses the next valid observation ────────────────

def test_bfill_uses_next_valid_observation():
    """method='bfill' (or 'backfill') fills gaps with the following valid value."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"v": [10.0, 20.0, 30.0]}, index=idx)
    new_idx = pd.date_range("2019-12-31", periods=5, freq="D")
    result = df.reindex(new_idx, method="bfill")
    # Before first original date → bfill from first value
    assert result.loc["2019-12-31", "v"] == 10.0
    # After last original date → no next value → NaN
    assert np.isnan(result.loc["2020-01-04", "v"])


def test_backfill_is_alias_for_bfill():
    """method='backfill' produces the same result as method='bfill'."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=idx)
    new_idx = pd.date_range("2019-12-31", periods=6, freq="D")
    result_bfill = df.reindex(new_idx, method="bfill")
    result_backfill = df.reindex(new_idx, method="backfill")
    pd.testing.assert_frame_equal(result_bfill, result_backfill)


# ── P15: Pre-existing NaN is NOT filled by method ────────────────────────────

def test_method_does_not_fill_preexisting_nan():
    """Filling methods propagate across new index positions but skip existing NaN in source."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    df = pd.DataFrame({"v": [1.0, np.nan, 3.0, 4.0]}, index=idx)
    result = df.reindex(idx, method="ffill")
    # 2020-01-02 was NaN in the original — reindex on same index doesn't fill it
    assert np.isnan(result.loc["2020-01-02", "v"])


# ── P16: Nearest fill selects the closer neighbor ────────────────────────────

def test_nearest_fill_selects_closer_neighbor():
    """method='nearest' assigns the value from the nearest index position."""
    idx = pd.to_datetime(["2020-01-01", "2020-01-10"])
    df = pd.DataFrame({"v": [1.0, 10.0]}, index=idx)
    # 2020-01-02 is 1 day from 2020-01-01 and 8 days from 2020-01-10 → picks 1.0
    result = df.reindex(
        pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-10"]),
        method="nearest",
    )
    assert result.loc["2020-01-02", "v"] == 1.0
    assert result.loc["2020-01-10", "v"] == 10.0


# ── P17: limit constrains consecutive fill count ──────────────────────────────

def test_limit_caps_consecutive_fills():
    """limit=N stops filling after N consecutive missing positions."""
    idx = pd.RangeIndex(0, 3)
    df = pd.DataFrame({"v": [1.0, np.nan, np.nan]}, index=idx)
    new_idx = pd.RangeIndex(0, 6)
    result = df.reindex(new_idx, method="ffill", limit=1)
    # position 1 is the first consecutive gap → filled
    assert result.loc[1, "v"] == 1.0
    # position 2 is the second consecutive gap → limit exceeded → NaN
    assert np.isnan(result.loc[2, "v"])
    # positions 3-5 are new labels past end of original — also NaN
    assert np.isnan(result.loc[5, "v"])


# ── P18: method requires monotonically ordered index ─────────────────────────

def test_method_raises_on_non_monotonic_index():
    """Specifying a fill method on a non-monotonic index raises an error."""
    df = pd.DataFrame({"v": [1, 2, 3]}, index=[3, 1, 2])
    with pytest.raises((ValueError, KeyError)):
        df.reindex([1, 2, 3, 4], method="ffill")


# ── P19: tolerance restricts inexact matching distance ───────────────────────

def test_tolerance_within_limit_matches():
    """Labels within tolerance of an original label are filled from that label."""
    idx = pd.to_datetime(["2020-01-01", "2020-01-10"])
    df = pd.DataFrame({"v": [1.0, 10.0]}, index=idx)
    new_idx = pd.to_datetime(["2020-01-02", "2020-01-10"])
    result = df.reindex(new_idx, method="nearest", tolerance=pd.Timedelta("2D"))
    assert result.loc["2020-01-02", "v"] == 1.0


def test_tolerance_outside_limit_gives_nan():
    """Labels farther than tolerance from any original label receive NaN."""
    idx = pd.to_datetime(["2020-01-01", "2020-01-10"])
    df = pd.DataFrame({"v": [1.0, 10.0]}, index=idx)
    # 2020-01-05 is 4 days from nearest → outside tolerance of 1 day
    new_idx = pd.to_datetime(["2020-01-05", "2020-01-10"])
    result = df.reindex(new_idx, method="nearest", tolerance=pd.Timedelta("1D"))
    assert np.isnan(result.loc["2020-01-05", "v"])


# ── P20: MultiIndex level reindexing ─────────────────────────────────────────

def test_multiindex_level_reindex_preserves_matching_rows():
    """level parameter broadcasts reindex over one level of a MultiIndex."""
    arrays = [["a", "a", "b", "b"], [1, 2, 1, 2]]
    idx = pd.MultiIndex.from_arrays(arrays, names=["letter", "number"])
    df = pd.DataFrame({"v": [10, 20, 30, 40]}, index=idx)
    result = df.reindex(["a", "b"], level="letter")
    # All original rows for letters 'a' and 'b' should be present
    assert set(result.index.get_level_values("letter")) == {"a", "b"}
    assert result.loc[("a", 1), "v"] == 10
    assert result.loc[("b", 2), "v"] == 40


def test_multiindex_level_reindex_adds_nan_for_new_level_value():
    """A new level value not present in the MultiIndex produces NaN rows."""
    arrays = [["a", "a"], [1, 2]]
    idx = pd.MultiIndex.from_arrays(arrays, names=["letter", "number"])
    df = pd.DataFrame({"v": [10, 20]}, index=idx)
    result = df.reindex(["a", "c"], level="letter")
    c_rows = result.xs("c", level="letter", drop_level=False)
    assert c_rows["v"].isna().all()
