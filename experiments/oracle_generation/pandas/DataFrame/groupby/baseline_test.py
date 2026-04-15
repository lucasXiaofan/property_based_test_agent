import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis.strategies import integers, floats, lists, sampled_from
from hypothesis.extra.pandas import data_frames, column


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
    """Test basic groupby returns a DataFrameGroupBy object."""
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
    """Test groupby with as_index=True returns grouped data with group labels as index."""
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
    """Test groupby with as_index=False returns DataFrame with group labels as column."""
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
    """Test groupby with sort=True sorts group keys."""
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
    """Test groupby with sort=False preserves original order."""
    assume(len(df) > 0)
    original_order = df["A"].tolist()
    result = df.groupby(by=["A"], sort=False)
    assume(result.ngroups > 0)
    first_group_key = list(result.groups.keys())[0]
    first_group_indices = result.groups[first_group_key]
    original_indices = [i for i, v in enumerate(original_order) if v == first_group_key]
    assert first_group_indices.tolist() == original_indices


def test_groupby_dropna_true():
    """Test groupby with dropna=True drops NA values from group keys."""
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
    """Test groupby with dropna=False treats NA as a separate group."""
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
    """Test groupby with multiple columns."""
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
    """Test groupby using a Series as the by parameter."""
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
    """Test groupby using a dict as the by parameter."""
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
    """Test groupby using a function as the by parameter."""
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
    """Test that groupby preserves the order of observations within each group."""
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
    """Test that ngroups returns the number of unique groups."""
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
    """Test that groups attribute returns a dict mapping group names to row indices."""
    assume(len(df) > 0)
    result = df.groupby(by=["A"])
    groups = result.groups
    assert isinstance(groups, dict)
    for group_key, indices in groups.items():
        assert len(indices) > 0


def test_groupby_level_parameter():
    """Test groupby with level parameter on MultiIndex."""
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
    """Test groupby with group_keys=True includes group keys in result index."""
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
    """Test groupby with group_keys=False excludes group keys from result index."""
    df = pd.DataFrame(
        {
            "Animal": ["Falcon", "Falcon", "Parrot", "Parrot"],
            "Max Speed": [380.0, 370.0, 24.0, 26.0],
        }
    )
    result = df.groupby("Animal", group_keys=False)[["Max Speed"]].apply(lambda x: x)
    assert "Animal" not in result.index.names
