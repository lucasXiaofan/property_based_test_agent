import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis.healthcheck import HealthCheck
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
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
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


# ============================================================================
# NEW TESTS - Improvements based on documentation analysis
# ============================================================================

def test_groupby_by_and_level_parameter():
    """Test that using both 'by' and 'level' parameters uses 'by' and ignores level."""
    arrays = [
        ["Falcon", "Falcon", "Parrot", "Parrot"],
        ["Captive", "Wild", "Captive", "Wild"],
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=("Animal", "Type"))
    df = pd.DataFrame(
        {"Max Speed": [390.0, 350.0, 30.0, 20.0]}, index=index
    )
    result = df.groupby(by=["Animal"], level=0)
    assert result.ngroups == 2


def test_groupby_empty_dataframe():
    """Test groupby on an empty DataFrame returns valid groupby object."""
    df = pd.DataFrame({"A": [], "B": []})
    result = df.groupby(by=["A"])
    assert hasattr(result, "groups")
    assert result.ngroups == 0


def test_groupby_empty_dataframe_with_level():
    """Test groupby with level on empty DataFrame with MultiIndex."""
    arrays = [[], []]
    index = pd.MultiIndex.from_arrays(arrays, names=("A", "B"))
    df = pd.DataFrame({"C": []}, index=index)
    result = df.groupby(level=0)
    assert result.ngroups == 0


def test_groupby_categorical_observed_true():
    """Test groupby with observed=True on Categorical data (default behavior)."""
    df = pd.DataFrame({
        "A": pd.Categorical(["a", "a", "b", "b", "c"]),
        "B": [1, 2, 3, 4, 5],
    })
    result = df.groupby(by=["A"], observed=True)
    groups = list(result.groups.keys())
    assert len(groups) == 3
    assert "a" in groups
    assert "b" in groups
    assert "c" in groups


def test_groupby_categorical_observed_false():
    """Test groupby with observed=False shows all categories including unobserved."""
    df = pd.DataFrame({
        "A": pd.Categorical(["a", "a", "b", "b"], categories=["a", "b", "c"]),
        "B": [1, 2, 3, 4],
    })
    result = df.groupby(by=["A"], observed=False)
    groups = list(result.groups.keys())
    assert len(groups) == 3


def test_groupby_tuple_as_single_key():
    """Test that tuple is interpreted as a single key, not multiple columns."""
    df = pd.DataFrame({
        "A": [(1, 2), (1, 2), (3, 4)],
        "B": [1, 2, 3],
    })
    result = df.groupby(by=["A"])
    assert result.ngroups == 2


def test_groupby_hash_based_equal_objects_same_group():
    """Test that objects comparing as equal are in the same group (hash-based)."""
    class HashableValue:
        def __init__(self, value):
            self.value = value
        def __hash__(self):
            return hash(self.value)
        def __eq__(self, other):
            return self.value == other.value

    df = pd.DataFrame({
        "A": [HashableValue(1), HashableValue(1), HashableValue(2)],
        "B": [1, 2, 3],
    })
    result = df.groupby(by=["A"])
    assert result.ngroups == 2


def test_groupby_na_values_collapsed_to_single_group():
    """Test that NA values are collapsed to a single group regardless of comparison."""
    df = pd.DataFrame({
        "A": [float('nan'), float('nan'), float('nan')],
        "B": [1, 2, 3],
    })
    result = df.groupby(by=["A"])
    assert result.ngroups == 1


def test_groupby_single_column_string_vs_list():
    """Test that single column string 'A' is equivalent to ['A']."""
    df = pd.DataFrame({
        "A": [1, 1, 2],
        "B": [10, 20, 30],
    })
    result_string = df.groupby(by="A")
    result_list = df.groupby(by=["A"])
    assert result_string.ngroups == result_list.ngroups


def test_groupby_mixed_na_and_regular_keys():
    """Test groupby with mix of NA and regular keys using dropna behavior."""
    df = pd.DataFrame({
        "A": [1, 2, None, 3, None, 1],
        "B": [1, 2, 3, 4, 5, 6],
    })
    result_dropna = df.groupby(by=["A"], dropna=True)
    result_keepna = df.groupby(by=["A"], dropna=False)
    assert result_dropna.ngroups < result_keepna.ngroups


def test_groupby_with_explicit_grouper():
    """Test groupby using pd.Grouper for frequency-based grouping."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
        "value": [1, 2, 3, 4],
    })
    result = df.groupby(pd.Grouper(key="date", freq="2D")).sum()
    assert len(result) == 2


def test_groupby_getitem_returns_subgroup():
    """Test that groupby[column] returns a DataFrameGroupBy for that column."""
    df = pd.DataFrame({
        "A": [1, 1, 2],
        "B": [10, 20, 30],
        "C": [100, 200, 300],
    })
    result = df.groupby(by=["A"])["B"]
    assert hasattr(result, "agg")


def test_groupby_iteration():
    """Test that we can iterate over groupby groups."""
    df = pd.DataFrame({
        "A": [1, 1, 2, 2],
        "B": [10, 20, 30, 40],
    })
    groups = list(df.groupby(by=["A"]))
    assert len(groups) == 2
    for key, group_df in groups:
        assert isinstance(group_df, pd.DataFrame)
        assert len(group_df) > 0


def test_groupby_size_returns_series():
    """Test that groupby.size() returns a Series with group counts."""
    df = pd.DataFrame({
        "A": [1, 1, 2, 2, 3],
        "B": [10, 20, 30, 40, 50],
    })
    result = df.groupby(by=["A"]).size()
    assert isinstance(result, pd.Series)
    assert result.sum() == len(df)


def test_groupby_first_last():
    """Test groupby.first() and groupby.last() return first/last values per group."""
    df = pd.DataFrame({
        "A": [1, 1, 2, 2],
        "B": [10, 20, 30, 40],
        "C": ["a", "b", "c", "d"],
    })
    first_result = df.groupby(by=["A"]).first()
    last_result = df.groupby(by=["A"]).last()
    assert len(first_result) == 2
    assert len(last_result) == 2


def test_groupby_nunique():
    """Test groupby.nunique() returns count of unique values per group."""
    df = pd.DataFrame({
        "A": [1, 1, 1, 2, 2],
        "B": [1, 1, 2, 2, 2],
    })
    result = df.groupby(by=["A"]).nunique()
    assert result.loc[1] == 2
    assert result.loc[2] == 1


def test_groupby_count():
    """Test groupby.count() returns non-NA count per group."""
    df = pd.DataFrame({
        "A": [1, 1, 2, 2],
        "B": [10, None, 30, 40],
        "C": [None, 20, 30, None],
    })
    result = df.groupby(by=["A"]).count()
    assert result.loc[1, "B"] == 1
    assert result.loc[1, "C"] == 1
    assert result.loc[2, "B"] == 2


def test_groupby_sum_with_nan():
    """Test groupby.sum() handles NaN values correctly."""
    df = pd.DataFrame({
        "A": [1, 1, 2],
        "B": [10, float('nan'), 30],
    })
    result = df.groupby(by=["A"]).sum()
    assert result.loc[1] == 10
    assert result.loc[2] == 30
