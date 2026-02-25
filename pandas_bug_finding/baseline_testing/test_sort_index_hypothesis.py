"""Property-based tests for pandas.DataFrame.sort_index.

Clauses tested:
  C1  - sort_index() returns a new DataFrame when inplace=False.
  C2  - sort_index(inplace=True) returns None.
  C3  - Default sort is ascending.
  C4  - axis=1 sorts column labels, not row labels.
  C5  - ascending=False produces descending label order.
  C6  - na_position='last' puts NaN index values at the end.
  C7  - na_position='first' puts NaN index values at the beginning.
  C8  - ignore_index=True relabels the result axis as 0, 1, ..., n-1.
  C9  - key function is applied to index values before sorting.
  C10 - mergesort and stable are the only stable sort algorithms.
  C11 - sort_remaining=True sorts remaining MultiIndex levels.
  C12 - level parameter sorts by specified index level for MultiIndex.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, indexes, range_indexes


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_int_indexed_dfs = data_frames(
    columns=[
        column("a", elements=st.integers(-100, 100)),
        column("b", elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
    ],
    index=indexes(elements=st.integers(-1000, 1000), min_size=0, max_size=20, unique=True),
)

_str_indexed_dfs = data_frames(
    columns=[
        column("x", elements=st.integers(0, 100)),
        column("y", elements=st.integers(0, 100)),
    ],
    index=indexes(
        elements=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", max_size=4, min_size=1),
        min_size=0,
        max_size=15,
        unique=True,
    ),
)


# ---------------------------------------------------------------------------
# C1: sort_index() returns a new DataFrame when inplace=False
# ---------------------------------------------------------------------------

@given(df=_int_indexed_dfs)
@settings(max_examples=100)
def test_c1_returns_new_dataframe(df):
    """C1: sort_index() returns a new DataFrame, not the original object."""
    result = df.sort_index()
    assert isinstance(result, pd.DataFrame), (
        f"Expected DataFrame, got {type(result)}"
    )
    assert result is not df, "sort_index() should return a new object, not the original"


# ---------------------------------------------------------------------------
# C2: sort_index(inplace=True) returns None
# ---------------------------------------------------------------------------

@given(df=_int_indexed_dfs)
@settings(max_examples=100)
def test_c2_inplace_returns_none(df):
    """C2: sort_index(inplace=True) must return None."""
    df_copy = df.copy()
    result = df_copy.sort_index(inplace=True)
    assert result is None, (
        f"sort_index(inplace=True) should return None, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# C3: Default sort is ascending
# ---------------------------------------------------------------------------

@given(df=_int_indexed_dfs)
@settings(max_examples=100)
def test_c3_default_ascending_sort(df):
    """C3: sort_index() with defaults produces monotonically increasing index."""
    result = df.sort_index()
    if len(result) > 1:
        assert result.index.is_monotonic_increasing, (
            f"Default sort_index() must produce ascending index; got {list(result.index)}"
        )


@given(df=_str_indexed_dfs)
@settings(max_examples=100)
def test_c3_default_ascending_sort_string_index(df):
    """C3: Default sort_index() on string-indexed DataFrame is ascending."""
    result = df.sort_index()
    idx_vals = list(result.index)
    assert idx_vals == sorted(idx_vals), (
        f"Expected sorted index, got {idx_vals}"
    )


# ---------------------------------------------------------------------------
# C4: axis=1 sorts column labels
# ---------------------------------------------------------------------------

@given(df=_int_indexed_dfs)
@settings(max_examples=100)
def test_c4_axis1_sorts_columns(df):
    """C4: sort_index(axis=1) sorts the column labels, not the row index."""
    result = df.sort_index(axis=1)
    col_list = list(result.columns)
    assert col_list == sorted(col_list), (
        f"sort_index(axis=1) should sort columns; got {col_list}"
    )
    # Row index must be unchanged
    pd.testing.assert_index_equal(result.index, df.index)


@given(df=_int_indexed_dfs)
@settings(max_examples=100)
def test_c4_axis_columns_name_equivalent(df):
    """C4: axis='columns' is equivalent to axis=1."""
    result_int = df.sort_index(axis=1)
    result_name = df.sort_index(axis="columns")
    pd.testing.assert_frame_equal(result_int, result_name)


# ---------------------------------------------------------------------------
# C5: ascending=False produces descending label order
# ---------------------------------------------------------------------------

@given(df=_int_indexed_dfs)
@settings(max_examples=100)
def test_c5_descending_sort(df):
    """C5: sort_index(ascending=False) produces monotonically decreasing index."""
    result = df.sort_index(ascending=False)
    if len(result) > 1:
        assert result.index.is_monotonic_decreasing, (
            f"sort_index(ascending=False) must produce descending index; got {list(result.index)}"
        )


@given(df=_int_indexed_dfs)
@settings(max_examples=100)
def test_c5_ascending_false_reverses_ascending_true(df):
    """C5: descending sort is the reverse of ascending sort."""
    asc = df.sort_index(ascending=True)
    desc = df.sort_index(ascending=False)
    pd.testing.assert_frame_equal(desc, asc[::-1])


# ---------------------------------------------------------------------------
# C6: na_position='last' puts NaN index values at the end
# ---------------------------------------------------------------------------

def test_c6_na_position_last_puts_nan_at_end():
    """C6: na_position='last' (default) places NaN index values at the end."""
    df = pd.DataFrame({"val": [10, 20, 30]}, index=[1.0, float("nan"), 2.0])
    result = df.sort_index(na_position="last")
    assert pd.isna(result.index[-1]), (
        f"Last index value should be NaN with na_position='last'; got {result.index[-1]}"
    )
    assert not pd.isna(result.index[0]), "First index should not be NaN"


def test_c6_na_position_last_is_default():
    """C6: na_position defaults to 'last'."""
    df = pd.DataFrame({"val": [10, 20, 30]}, index=[1.0, float("nan"), 2.0])
    result_default = df.sort_index()
    result_last = df.sort_index(na_position="last")
    pd.testing.assert_frame_equal(result_default, result_last)


# ---------------------------------------------------------------------------
# C7: na_position='first' puts NaN index values at the beginning
# ---------------------------------------------------------------------------

def test_c7_na_position_first_puts_nan_at_start():
    """C7: na_position='first' places NaN index values at the beginning."""
    df = pd.DataFrame({"val": [10, 20, 30]}, index=[1.0, float("nan"), 2.0])
    result = df.sort_index(na_position="first")
    assert pd.isna(result.index[0]), (
        f"First index value should be NaN with na_position='first'; got {result.index[0]}"
    )
    assert not pd.isna(result.index[-1]), "Last index should not be NaN"


def test_c7_na_first_vs_last_are_opposites():
    """C7: na_position='first' and 'last' place NaN at opposite ends."""
    df = pd.DataFrame({"val": [10, 20, 30]}, index=[1.0, float("nan"), 2.0])
    first = df.sort_index(na_position="first")
    last = df.sort_index(na_position="last")
    assert pd.isna(first.index[0])
    assert pd.isna(last.index[-1])
    assert not pd.isna(first.index[-1])
    assert not pd.isna(last.index[0])


# ---------------------------------------------------------------------------
# C8: ignore_index=True relabels axis as 0..n-1
# ---------------------------------------------------------------------------

@given(df=_int_indexed_dfs)
@settings(max_examples=100)
def test_c8_ignore_index_relabels_0_to_n(df):
    """C8: sort_index(ignore_index=True) produces RangeIndex 0..n-1."""
    result = df.sort_index(ignore_index=True)
    expected_index = list(range(len(df)))
    assert list(result.index) == expected_index, (
        f"Expected 0..{len(df)-1} index, got {list(result.index)}"
    )


@given(df=_int_indexed_dfs)
@settings(max_examples=100)
def test_c8_ignore_index_false_preserves_index(df):
    """C8: sort_index(ignore_index=False) preserves the original label values."""
    result = df.sort_index(ignore_index=False)
    # Labels in result must be a permutation of df.index
    assert sorted(result.index.tolist()) == sorted(df.index.tolist()), (
        "ignore_index=False must keep original index labels"
    )


# ---------------------------------------------------------------------------
# C9: key function is applied to index values before sorting
# ---------------------------------------------------------------------------

def test_c9_key_function_changes_sort_order():
    """C9: key function modifies sort key; case-insensitive sort example."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=["A", "b", "C", "d"])
    result = df.sort_index(key=lambda x: x.str.lower())
    # With case-insensitive sort: A(a), b(b), C(c), d(d) — already in order
    assert list(result.index) == ["A", "b", "C", "d"], (
        f"key=str.lower() should produce case-insensitive order; got {list(result.index)}"
    )


def test_c9_key_function_reverse_sort():
    """C9: key function that negates integers reverses sort order."""
    df = pd.DataFrame({"v": [10, 20, 30]}, index=[1, 2, 3])
    result = df.sort_index(key=lambda x: -x)
    assert list(result.index) == [3, 2, 1], (
        f"key=-x should reverse sort; got {list(result.index)}"
    )


# ---------------------------------------------------------------------------
# C10: mergesort/stable are the only stable algorithms
# (stable sort preserves relative order of equal-key elements)
# ---------------------------------------------------------------------------

def test_c10_stable_sort_preserves_relative_order():
    """C10: 'stable' kind preserves original order for ties in sort key."""
    # MultiIndex with duplicate first level — original order should be kept
    idx = pd.MultiIndex.from_tuples(
        [(1, "b"), (1, "a"), (2, "c")], names=["lvl0", "lvl1"]
    )
    df = pd.DataFrame({"x": [10, 20, 30]}, index=idx)
    # Sort only by level 0 with kind='stable' and sort_remaining=False
    result = df.sort_index(level=0, sort_remaining=False, kind="stable")
    # Level-0 ties (1, "b") and (1, "a") must preserve original order
    level1_for_1 = [v for k, v in zip(result.index.get_level_values(0),
                                       result.index.get_level_values(1)) if k == 1]
    assert level1_for_1 == ["b", "a"], (
        f"Stable sort must preserve relative order for ties; got {level1_for_1}"
    )


# ---------------------------------------------------------------------------
# C11: sort_remaining=True sorts remaining MultiIndex levels
# ---------------------------------------------------------------------------

def test_c11_sort_remaining_true_sorts_other_levels():
    """C11: sort_remaining=True also sorts non-specified MultiIndex levels."""
    idx = pd.MultiIndex.from_tuples(
        [(2, 2), (2, 1), (1, 0)], names=["a", "b"]
    )
    df = pd.DataFrame({"x": [10, 20, 30]}, index=idx)
    result = df.sort_index(level=0, sort_remaining=True)
    expected = pd.MultiIndex.from_tuples(
        [(1, 0), (2, 1), (2, 2)], names=["a", "b"]
    )
    pd.testing.assert_index_equal(result.index, expected)


def test_c11_sort_remaining_false_skips_other_levels():
    """C11: sort_remaining=False skips secondary level sorting."""
    idx = pd.MultiIndex.from_tuples(
        [(2, 2), (2, 1), (1, 0)], names=["a", "b"]
    )
    df = pd.DataFrame({"x": [10, 20, 30]}, index=idx)
    result = df.sort_index(level=0, sort_remaining=False)
    # Primary level must be sorted; secondary level within group may not be
    level0 = list(result.index.get_level_values(0))
    assert level0 == sorted(level0), f"Primary level must be sorted; got {level0}"
    # Secondary level within group for a=2: must be original order [2, 1]
    level1_for_2 = [v for k, v in zip(result.index.get_level_values(0),
                                        result.index.get_level_values(1)) if k == 2]
    assert level1_for_2 == [2, 1], (
        f"sort_remaining=False should not sort secondary levels; got {level1_for_2}"
    )


# ---------------------------------------------------------------------------
# C12: level parameter sorts by specified index level for MultiIndex
# ---------------------------------------------------------------------------

def test_c12_level_sorts_by_specified_level():
    """C12: level parameter causes sort by that MultiIndex level."""
    idx = pd.MultiIndex.from_tuples(
        [(3, "b"), (1, "a"), (2, "c")], names=["x", "y"]
    )
    df = pd.DataFrame({"val": [30, 10, 20]}, index=idx)
    result = df.sort_index(level=0)
    lvl0_vals = list(result.index.get_level_values(0))
    assert lvl0_vals == sorted(lvl0_vals), (
        f"sort_index(level=0) must sort by level 0; got {lvl0_vals}"
    )


def test_c12_level_by_name():
    """C12: level parameter accepts level name as well as integer."""
    idx = pd.MultiIndex.from_tuples(
        [(3, "b"), (1, "a"), (2, "c")], names=["x", "y"]
    )
    df = pd.DataFrame({"val": [30, 10, 20]}, index=idx)
    result_int = df.sort_index(level=0)
    result_name = df.sort_index(level="x")
    pd.testing.assert_frame_equal(result_int, result_name)
