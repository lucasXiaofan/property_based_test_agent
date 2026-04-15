"""
Baseline property-based tests for pandas.DataFrame.groupby.

Properties tested:
1. Return type is DataFrameGroupBy
2. Group count matches distinct key values (with dropna=True/False)
3. Partition property: rows in each group share the same key value
4. Completeness: every row belongs to exactly one group
5. sort=True produces lexicographically sorted group keys
6. sort=False preserves first-appearance order of group keys
7. as_index=False produces RangeIndex (no group labels in index)
8. as_index=True places group labels in the index
9. observed=True on Categorical grouper only shows observed categories
10. observed=False on Categorical grouper shows all categories (even empty)
11. dropna=True excludes NA keys; dropna=False includes NA keys as one group
12. level parameter works on MultiIndex DataFrames
13. Aggregation result row count equals number of groups
14. group_keys=True / False affects index when using apply
15. Grouping by multiple columns
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_int_col(name):
    return column(name, elements=st.integers(min_value=0, max_value=3))


def small_str_col(name):
    return column(name, elements=st.sampled_from(["a", "b", "c"]))


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[small_int_col("key"), small_int_col("val")],
        index=range_indexes(min_size=1, max_size=20),
    )
)
def test_groupby_returns_dataframegroupby(df):
    result = df.groupby("key")
    assert isinstance(result, pd.core.groupby.DataFrameGroupBy)


# ---------------------------------------------------------------------------
# 2. Group count matches distinct key values (dropna=True)
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[small_str_col("key"), small_int_col("val")],
        index=range_indexes(min_size=1, max_size=30),
    )
)
def test_group_count_equals_distinct_keys_dropna_true(df):
    grp = df.groupby("key", dropna=True)
    expected = df["key"].dropna().nunique()
    assert len(grp) == expected


# ---------------------------------------------------------------------------
# 3. Partition: rows in each group share the same key
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[small_str_col("key"), small_int_col("val")],
        index=range_indexes(min_size=1, max_size=30),
    )
)
def test_partition_rows_share_same_key(df):
    for name, group in df.groupby("key", dropna=True):
        assert (group["key"] == name).all()


# ---------------------------------------------------------------------------
# 4. Completeness: every row belongs to exactly one group (dropna=True drops NA rows)
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[small_str_col("key"), small_int_col("val")],
        index=range_indexes(min_size=1, max_size=30),
    )
)
def test_completeness_all_non_na_rows_covered(df):
    non_na_idx = df.index[df["key"].notna()]
    covered = pd.Index([], dtype=df.index.dtype)
    for _, group in df.groupby("key", dropna=True):
        covered = covered.append(group.index)
    assert set(covered) == set(non_na_idx)


@given(
    df=data_frames(
        columns=[small_str_col("key"), small_int_col("val")],
        index=range_indexes(min_size=1, max_size=30),
    )
)
def test_completeness_all_rows_covered_dropna_false(df):
    covered = pd.Index([], dtype=df.index.dtype)
    for _, group in df.groupby("key", dropna=False):
        covered = covered.append(group.index)
    assert set(covered) == set(df.index)


# ---------------------------------------------------------------------------
# 5. sort=True → group keys are in sorted order
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[small_str_col("key"), small_int_col("val")],
        index=range_indexes(min_size=2, max_size=30),
    )
)
def test_sort_true_keys_are_sorted(df):
    assume(df["key"].nunique() >= 2)
    keys = [name for name, _ in df.groupby("key", sort=True, dropna=True)]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# 6. sort=False → group keys appear in first-occurrence order
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[small_str_col("key"), small_int_col("val")],
        index=range_indexes(min_size=2, max_size=30),
    )
)
def test_sort_false_keys_in_first_occurrence_order(df):
    assume(df["key"].notna().any())
    seen = []
    for k in df["key"]:
        if pd.notna(k) and k not in seen:
            seen.append(k)
    keys = [name for name, _ in df.groupby("key", sort=False, dropna=True)]
    assert keys == seen


# ---------------------------------------------------------------------------
# 7. as_index=False → result index is RangeIndex
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[small_str_col("key"), small_int_col("val")],
        index=range_indexes(min_size=1, max_size=30),
    )
)
def test_as_index_false_result_has_range_index(df):
    result = df.groupby("key", as_index=False, dropna=True).sum()
    assert isinstance(result.index, pd.RangeIndex)


# ---------------------------------------------------------------------------
# 8. as_index=True → group label appears in result index
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[small_str_col("key"), small_int_col("val")],
        index=range_indexes(min_size=1, max_size=30),
    )
)
def test_as_index_true_group_label_in_index(df):
    result = df.groupby("key", as_index=True, dropna=True).sum()
    assert result.index.name == "key"
    # Every index value should have been a key in the original DataFrame
    for idx_val in result.index:
        assert idx_val in df["key"].values


# ---------------------------------------------------------------------------
# 9. observed=True on Categorical only shows observed categories
# ---------------------------------------------------------------------------

@given(
    cats=st.lists(
        st.sampled_from(["x", "y", "z"]),
        min_size=2,
        max_size=20,
    )
)
def test_observed_true_only_observed_categories(cats):
    all_cats = ["x", "y", "z"]
    df = pd.DataFrame(
        {"key": pd.Categorical(cats, categories=all_cats), "val": range(len(cats))}
    )
    result = df.groupby("key", observed=True).sum()
    observed_cats = set(cats)
    assert set(result.index) == observed_cats


# ---------------------------------------------------------------------------
# 10. observed=False shows all categories including empty ones
# ---------------------------------------------------------------------------

def test_observed_false_shows_all_categories():
    all_cats = ["x", "y", "z"]
    df = pd.DataFrame(
        {
            "key": pd.Categorical(["x", "x"], categories=all_cats),
            "val": [1, 2],
        }
    )
    result = df.groupby("key", observed=False).sum()
    assert set(result.index) == set(all_cats)


# ---------------------------------------------------------------------------
# 11a. dropna=True excludes NA keys
# ---------------------------------------------------------------------------

def test_dropna_true_excludes_na_keys():
    df = pd.DataFrame({"key": ["a", None, "b", "a"], "val": [1, 2, 3, 4]})
    result = df.groupby("key", dropna=True).sum()
    assert None not in result.index
    assert np.nan not in result.index
    assert set(result.index) == {"a", "b"}


# ---------------------------------------------------------------------------
# 11b. dropna=False includes NA keys as one group
# ---------------------------------------------------------------------------

def test_dropna_false_includes_na_as_group():
    df = pd.DataFrame({"key": ["a", None, "b", "a"], "val": [1, 2, 3, 4]})
    result = df.groupby("key", dropna=False).sum()
    # NA group should be present
    assert any(pd.isna(k) for k in result.index)
    assert "a" in result.index and "b" in result.index


def test_dropna_false_na_group_sum():
    df = pd.DataFrame({"key": ["a", None, "b"], "val": [1, 99, 3]})
    result = df.groupby("key", dropna=False).sum()
    na_row = result.loc[result.index.isna()]
    assert int(na_row["val"].iloc[0]) == 99


# ---------------------------------------------------------------------------
# 12. level parameter on MultiIndex
# ---------------------------------------------------------------------------

def test_groupby_level_multiindex():
    arrays = [
        ["Falcon", "Falcon", "Parrot", "Parrot"],
        ["Captive", "Wild", "Captive", "Wild"],
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=("Animal", "Type"))
    df = pd.DataFrame({"Speed": [390.0, 350.0, 30.0, 20.0]}, index=index)

    result_level0 = df.groupby(level=0).mean()
    assert set(result_level0.index) == {"Falcon", "Parrot"}
    assert pytest.approx(result_level0.loc["Falcon", "Speed"]) == 370.0
    assert pytest.approx(result_level0.loc["Parrot", "Speed"]) == 25.0

    result_level_name = df.groupby(level="Type").mean()
    assert set(result_level_name.index) == {"Captive", "Wild"}
    assert pytest.approx(result_level_name.loc["Captive", "Speed"]) == 210.0
    assert pytest.approx(result_level_name.loc["Wild", "Speed"]) == 185.0


# ---------------------------------------------------------------------------
# 13. Aggregation result row count equals number of groups
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[small_str_col("key"), small_int_col("val")],
        index=range_indexes(min_size=1, max_size=30),
    )
)
def test_agg_result_row_count_equals_group_count(df):
    grp = df.groupby("key", dropna=True)
    result = grp.sum()
    assert len(result) == len(grp)


# ---------------------------------------------------------------------------
# 14. group_keys parameter affects index in apply
# ---------------------------------------------------------------------------

def test_group_keys_true_adds_group_label_to_index():
    df = pd.DataFrame(
        {"Animal": ["Falcon", "Falcon", "Parrot", "Parrot"], "Speed": [380.0, 370.0, 24.0, 26.0]}
    )
    result = df.groupby("Animal", group_keys=True)[["Speed"]].apply(lambda x: x)
    assert result.index.nlevels == 2
    assert result.index.names[0] == "Animal"


def test_group_keys_false_no_extra_level_in_index():
    df = pd.DataFrame(
        {"Animal": ["Falcon", "Falcon", "Parrot", "Parrot"], "Speed": [380.0, 370.0, 24.0, 26.0]}
    )
    result = df.groupby("Animal", group_keys=False)[["Speed"]].apply(lambda x: x)
    assert result.index.nlevels == 1


# ---------------------------------------------------------------------------
# 15. Grouping by multiple columns
# ---------------------------------------------------------------------------

@given(
    df=data_frames(
        columns=[small_str_col("k1"), small_str_col("k2"), small_int_col("val")],
        index=range_indexes(min_size=1, max_size=30),
    )
)
def test_groupby_multiple_columns_partition(df):
    for (k1, k2), group in df.groupby(["k1", "k2"], dropna=True):
        assert (group["k1"] == k1).all()
        assert (group["k2"] == k2).all()


@given(
    df=data_frames(
        columns=[small_str_col("k1"), small_str_col("k2"), small_int_col("val")],
        index=range_indexes(min_size=1, max_size=30),
    )
)
def test_groupby_multiple_columns_result_has_multiindex(df):
    result = df.groupby(["k1", "k2"], dropna=True).sum()
    assert isinstance(result.index, pd.MultiIndex)
