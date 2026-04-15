"""
IR-enhanced test cases for pandas DataFrame.groupby.
- Baseline tests: tests from baseline_test.py
- New tests: tests inspired by ir_v2.json focusing on high-stakes edge cases
"""

import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis.strategies import integers, floats, lists, sampled_from
from hypothesis.extra.pandas import data_frames, column


# =============================================================================
# BASELINE TESTS (from baseline_test.py)
# =============================================================================


@given(
    data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=float),
            column("C", dtype=str),
        ],
    )
)
@settings(max_examples=30)
def test_groupby_basic(df):
    """BASELINE: Test basic groupby returns a DataFrameGroupBy object."""
    assume(len(df) > 0)
    result = df.groupby(by=["A"])
    assert hasattr(result, "groups")
    assert hasattr(result, "ngroups")


@given(
    data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=float),
        ],
    )
)
@settings(max_examples=30)
def test_groupby_as_index_true(df):
    """BASELINE: Test groupby with as_index=True returns grouped data with group labels as index."""
    assume(len(df) > 0)
    result = df.groupby(by=["A"], as_index=True)
    df_agg = result.mean()
    assume(len(df_agg) > 0)
    assert df_agg.index.name == "A"


@given(
    data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=int),
            column("C", dtype=float),
        ],
    )
)
@settings(max_examples=30)
def test_groupby_as_index_false(df):
    """BASELINE: Test groupby with as_index=False returns DataFrame with group labels as column."""
    assume(len(df) > 0)
    result = df.groupby(by=["A", "B"], as_index=False)
    df_agg = result.mean()
    assume(len(df_agg) > 0)
    assert "A" in df_agg.columns
    assert "B" in df_agg.columns


@given(
    data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=float),
        ],
    )
)
@settings(max_examples=30)
def test_groupby_sort_true(df):
    """BASELINE: Test groupby with sort=True sorts group keys."""
    assume(len(df) > 0)
    result = df.groupby(by=["A"], sort=True)
    groups = list(result.groups.keys())
    if len(groups) > 1:
        assert groups == sorted(groups)


@given(
    data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=float),
        ],
    )
)
@settings(max_examples=30)
def test_groupby_sort_false(df):
    """BASELINE: Test groupby with sort=False preserves original order."""
    assume(len(df) > 0)
    original_order = df["A"].tolist()
    result = df.groupby(by=["A"], sort=False)
    assume(result.ngroups > 0)
    first_group_key = list(result.groups.keys())[0]
    first_group_indices = result.groups[first_group_key]
    original_indices = [i for i, v in enumerate(original_order) if v == first_group_key]
    assert first_group_indices.tolist() == original_indices


def test_groupby_dropna_true():
    """BASELINE: Test groupby with dropna=True drops NA values from group keys."""
    df_with_na = pd.DataFrame(
        {
            "A": [1, 2, None, 3, None],
            "B": [10, 20, 30, 40, 50],
        }
    )
    result = df_with_na.groupby(by=["A"], dropna=True)
    groups = result.groups
    assert None not in groups
    assert 1 in groups
    assert 2 in groups
    assert 3 in groups


def test_groupby_dropna_false():
    """BASELINE: Test groupby with dropna=False treats NA as a separate group."""
    df_with_na = pd.DataFrame(
        {
            "A": [1, 2, None, 3, None],
            "B": [10, 20, 30, 40, 50],
        }
    )
    result = df_with_na.groupby(by=["A"], dropna=False)
    groups = result.groups
    assert any(pd.isna(k) for k in groups.keys())


@given(
    data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=int),
            column("C", dtype=float),
        ],
    )
)
@settings(max_examples=30)
def test_groupby_multiple_columns(df):
    """BASELINE: Test groupby with multiple columns."""
    assume(len(df) > 0)
    result = df.groupby(by=["A", "B"])
    assert result.ngroups >= 0


@given(
    data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=float),
        ],
    )
)
@settings(max_examples=30)
def test_groupby_with_series(df):
    """BASELINE: Test groupby using a Series as the by parameter."""
    assume(len(df) > 0)
    by_series = df["A"]
    result = df.groupby(by=by_series)
    assert result.ngroups >= 0


@given(
    data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=float),
        ],
    )
)
@settings(max_examples=30)
def test_groupby_with_dict(df):
    """BASELINE: Test groupby using a dict as the by parameter."""
    assume(len(df) > 0)
    unique_vals = df["A"].unique()
    by_dict = {v: v % 2 for v in unique_vals if pd.notna(v)}
    assume(len(by_dict) > 0)
    result = df.groupby(by=by_dict)
    assert result.ngroups >= 0


@given(
    data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=float),
        ],
    )
)
@settings(max_examples=30)
def test_groupby_with_function(df):
    """BASELINE: Test groupby using a function as the by parameter."""
    assume(len(df) > 0)
    result = df.groupby(by=lambda x: x % 2)
    assert result.ngroups >= 0


@given(
    data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=float),
        ],
    )
)
@settings(max_examples=30)
def test_groupby_preserves_order_within_group(df):
    """BASELINE: Test that groupby preserves the order of observations within each group."""
    assume(len(df) > 0)
    result = df.groupby(by=["A"])
    for group_key, group_indices in result.groups.items():
        original_order = df.loc[group_indices, "A"].tolist()
        assert original_order == list(sorted(original_order))


@given(
    data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=float),
        ],
    )
)
@settings(max_examples=30)
def test_groupby_ngroups(df):
    """BASELINE: Test that ngroups returns the number of unique groups."""
    assume(len(df) > 0)
    result = df.groupby(by=["A"])
    unique_values = df["A"].dropna().nunique()
    assert result.ngroups == unique_values


@given(
    data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=float),
        ],
    )
)
@settings(max_examples=30)
def test_groupby_groups_attribute(df):
    """BASELINE: Test that groups attribute returns a dict mapping group names to row indices."""
    assume(len(df) > 0)
    result = df.groupby(by=["A"])
    groups = result.groups
    assert isinstance(groups, dict)
    for group_key, indices in groups.items():
        assert len(indices) > 0


def test_groupby_level_parameter():
    """BASELINE: Test groupby with level parameter on MultiIndex."""
    arrays = [
        ["Falcon", "Falcon", "Parrot", "Parrot", "Cat", "Cat"],
        ["Captive", "Wild", "Captive", "Wild", "Captive", "Wild"],
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=("Animal", "Type"))
    df = pd.DataFrame(
        {"Max Speed": [390.0, 350.0, 30.0, 20.0, 25.0, 15.0]}, index=index
    )
    result = df.groupby(level=0)
    assert result.ngroups == 3
    result_level1 = df.groupby(level=1)
    assert result_level1.ngroups == 2


def test_groupby_group_keys_true():
    """BASELINE: Test groupby with group_keys=True includes group keys in result index."""
    df = pd.DataFrame(
        {
            "Animal": ["Falcon", "Falcon", "Parrot", "Parrot"],
            "Max Speed": [380.0, 370.0, 24.0, 26.0],
        }
    )
    result = df.groupby("Animal", group_keys=True)[["Max Speed"]].apply(lambda x: x)
    assert "Animal" in result.index.names or "Animal" in result.index.get_level_values(
        0
    )


def test_groupby_group_keys_false():
    """BASELINE: Test groupby with group_keys=False excludes group keys from result index."""
    df = pd.DataFrame(
        {
            "Animal": ["Falcon", "Falcon", "Parrot", "Parrot"],
            "Max Speed": [380.0, 370.0, 24.0, 26.0],
        }
    )
    result = df.groupby("Animal", group_keys=False)[["Max Speed"]].apply(lambda x: x)
    assert "Animal" not in result.index.names


# =============================================================================
# NEW TESTS (inspired by ir_v2.json - high-stakes edge cases)
# =============================================================================


def test_groupby_categorical_observed_true():
    """
    NEW (IR): Test observed=True only shows observed values for categorical groupers.
    IR post_condition: only_observed_categories_in_groups_when_observed_true
    """
    df = pd.DataFrame(
        {"key": pd.Categorical(["a", "b"], categories=["a", "b", "c"]), "val": [1, 2]}
    )
    result = df.groupby(by=["key"], observed=True)
    assert set(result.groups.keys()) == {"a", "b"}


def test_groupby_categorical_observed_false():
    """
    NEW (IR): Test observed=False shows all categories including unobserved ones.
    IR post_condition: all_categories_in_groups_when_observed_false
    """
    df = pd.DataFrame(
        {"key": pd.Categorical(["a", "b"], categories=["a", "b", "c"]), "val": [1, 2]}
    )
    result = df.groupby(by=["key"], observed=False)
    assert set(result.groups.keys()) == {"a", "b", "c"}


def test_groupby_all_rows_covered_no_na():
    """
    NEW (IR): Test all rows are covered by groups when dropna=True.
    IR post_condition: all_rows_covered_by_groups_when_no_na
    """
    df = pd.DataFrame({"key": ["a", "b", "a", "b", "a"], "val": [1, 2, 3, 4, 5]})
    result = df.groupby(by=["key"], dropna=True)
    total_rows = sum(len(grp) for _, grp in result)
    assert total_rows == len(df)


def test_groupby_all_rows_covered_with_na():
    """
    NEW (IR): Test all rows including NA are covered when dropna=False.
    IR post_condition: all_rows_including_na_covered_when_dropna_false
    """
    df = pd.DataFrame({"key": ["a", "b", None, "a", None], "val": [1, 2, 3, 4, 5]})
    result = df.groupby(by=["key"], dropna=False)
    total_rows = sum(len(grp) for _, grp in result)
    assert total_rows == len(df)


def test_groupby_level_by_name():
    """
    NEW (IR): Test groupby with level parameter using level name.
    IR post_condition: level_name_groupby_groups_by_named_level
    """
    arrays = [
        ["Falcon", "Falcon", "Parrot", "Parrot"],
        ["Captive", "Wild", "Captive", "Wild"],
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=("Animal", "Type"))
    df = pd.DataFrame({"Max Speed": [390.0, 350.0, 30.0, 20.0]}, index=index)
    result = df.groupby(level="Animal")
    assert set(result.groups.keys()) == {"Falcon", "Parrot"}


def test_groupby_by_function_on_index():
    """
    NEW (IR): Test groupby using a function on the index values.
    IR post_condition: by_function_groups_by_index_mapped_values
    """
    df = pd.DataFrame({"val": [10, 20, 30, 40, 50]}, index=[0, 1, 2, 3, 4])
    result = df.groupby(by=lambda x: x % 2)
    expected_keys = {0, 1}
    assert set(result.groups.keys()) == expected_keys


def test_groupby_group_keys_apply_result_index():
    """
    NEW (IR): Test group_keys=True adds keys to apply result index.
    IR post_condition: group_keys_true_adds_keys_to_apply_result_index
    """
    df = pd.DataFrame({"key": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]})
    result = df.groupby(by=["key"], group_keys=True)["val"].apply(lambda x: x)
    assert isinstance(result.index, pd.MultiIndex)


def test_groupby_group_keys_apply_result_index_false():
    """
    NEW (IR): Test group_keys=False omits keys from apply result index.
    IR post_condition: group_keys_false_omits_keys_from_apply_result_index
    """
    df = pd.DataFrame({"key": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]})
    result = df.groupby(by=["key"], group_keys=False)["val"].apply(lambda x: x)
    assert not isinstance(result.index, pd.MultiIndex)


def test_groupby_empty_dataframe():
    """
    NEW (IR): Test groupby on empty DataFrame handles gracefully.
    IR implicit: edge case handling
    """
    df = pd.DataFrame({"key": [], "val": []})
    result = df.groupby(by=["key"])
    assert result.ngroups == 0


def test_groupby_single_row_dataframe():
    """
    NEW (IR): Test groupby on single row DataFrame.
    IR implicit: edge case handling
    """
    df = pd.DataFrame({"key": ["a"], "val": [1]})
    result = df.groupby(by=["key"])
    assert result.ngroups == 1


def test_groupby_all_same_key():
    """
    NEW (IR): Test groupby when all rows have the same key.
    IR implicit: edge case handling - potential performance issue
    """
    df = pd.DataFrame({"key": ["a"] * 10, "val": list(range(10))})
    result = df.groupby(by=["key"])
    assert result.ngroups == 1
    assert sum(len(grp) for _, grp in result) == 10


def test_groupby_each_row_unique_key():
    """
    NEW (IR): Test groupby when each row has a unique key.
    IR implicit: edge case handling - potential memory issue
    """
    df = pd.DataFrame({"key": list(range(10)), "val": list(range(10))})
    result = df.groupby(by=["key"])
    assert result.ngroups == 10


def test_groupby_preserves_row_order_within_group():
    """
    NEW (IR): Test that row order within each group is preserved from original DataFrame.
    IR post_condition: row_order_preserved_within_each_group
    """
    df = pd.DataFrame({"key": ["a", "b", "a", "b", "a"], "val": [1, 2, 3, 4, 5]})
    result = df.groupby(by=["key"])
    for group_key, group_indices in result.groups.items():
        grouped_indices = list(group_indices)
        original_indices_in_order = [i for i in df.index if i in set(group_indices)]
        assert grouped_indices == original_indices_in_order


def test_groupby_by_series():
    """
    NEW (IR): Test groupby using a Series as the by parameter.
    IR pre_condition: by_series_as_grouper
    """
    df = pd.DataFrame({"key": ["a", "b", "a", "b"], "val": [1, 2, 3, 4]})
    by_series = pd.Series([1, 2, 1, 2])
    result = df.groupby(by=by_series)
    assert result.ngroups == 2


def test_groupby_na_special_handling():
    """
    NEW (IR): Test that NA values are handled specially (collapsed to single group).
    IR note: "any NA values will be collapsed to a single group, regardless of how they compare"
    """
    df = pd.DataFrame({"key": [None, None, None, None], "val": [1, 2, 3, 4]})
    result = df.groupby(by=["key"], dropna=True)
    assert result.ngroups == 0


def test_groupby_integer_and_string_keys():
    """
    NEW (IR): Test groupby with mixed integer and string keys.
    IR implicit: different key types
    """
    df = pd.DataFrame({"key": [1, "a", 2, "b", 1], "val": [1, 2, 3, 4, 5]})
    result = df.groupby(by=["key"])
    assert result.ngroups == 4


def test_groupby_tuple_key():
    """
    NEW (IR): Test groupby with a tuple as the key (should be treated as single key).
    IR note: "Notice that a tuple is interpreted as a (single) key"
    """
    df = pd.DataFrame(
        {"key": [("a", 1), ("a", 1), ("b", 2), ("b", 2)], "val": [1, 2, 3, 4]}
    )
    result = df.groupby(by=["key"])
    assert result.ngroups == 2


def test_groupby_sql_style_output():
    """
    NEW (IR): Test SQL-style output with as_index=False returns DataFrame with columns.
    IR post_condition: sql_style_output_when_as_index_false
    """
    df = pd.DataFrame({"key": ["a", "b", "a", "b"], "val": [1, 2, 3, 4]})
    agg = df.groupby(by=["key"], as_index=False).sum()
    assert "key" in agg.columns
    assert isinstance(agg.index, pd.RangeIndex)


def test_groupby_group_count_equals_unique():
    """
    NEW (IR): Test group count equals unique non-NA key values.
    IR post_condition: group_count_equals_unique_non_na_key_values
    """
    df = pd.DataFrame({"key": ["a", "b", "a", "b", "c"], "val": [1, 2, 3, 4, 5]})
    result = df.groupby(by=["key"], dropna=True)
    assert result.ngroups == df["key"].nunique(dropna=True)
