"""
Property-based and concrete tests for pandas.DataFrame.loc

Clause coverage:
  C1 - loc is primarily label-based (not position-based)
  C2 - Integer inputs are labels, never positions
  C3 - Label slices include BOTH start and stop
  C4 - Single label returns Series
  C5 - List of labels returns DataFrame
  C6 - KeyError raised for missing labels
  C7 - Boolean array selects rows where True; NA -> False
  C8 - Callable is called with the DataFrame; result used for indexing
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def string_df():
    return pd.DataFrame(
        [[1, 2], [4, 5], [7, 8]],
        index=["cobra", "viper", "sidewinder"],
        columns=["max_speed", "shield"],
    )


@pytest.fixture
def int_label_df():
    """DataFrame whose index labels are integers that do NOT match their positions."""
    return pd.DataFrame(
        {"value": [100, 200, 300]},
        index=[10, 20, 30],
    )


# ---------------------------------------------------------------------------
# C1 – label-based access (not position-based)
# ---------------------------------------------------------------------------

def test_c1_string_label_returns_correct_row(string_df):
    """loc with a string label returns the row matching that label."""
    result = string_df.loc["cobra"]
    assert result["max_speed"] == 1
    assert result["shield"] == 2


def test_c1_label_access_on_reversed_df(string_df):
    """After reversing row order, loc still finds by label, not position."""
    df = string_df.iloc[::-1].copy()  # sidewinder at pos 0, cobra at pos 2
    result = df.loc["cobra"]
    assert result["max_speed"] == 1


def test_c1_string_label_is_not_iloc(string_df):
    """loc['viper'] returns viper's data, not iloc[0] data."""
    result = string_df.loc["viper"]
    # viper is at position 1; if loc used positions, label 'viper' would TypeError
    assert result["max_speed"] == 4


# ---------------------------------------------------------------------------
# C2 – integers are labels, not positions
# ---------------------------------------------------------------------------

def test_c2_integer_label_returns_correct_row(int_label_df):
    """loc[10] with index=[10,20,30] returns the row labelled 10."""
    result = int_label_df.loc[10]
    assert result["value"] == 100


def test_c2_integer_label_second_row(int_label_df):
    """loc[20] returns the row labelled 20 (position 1), not position 20."""
    result = int_label_df.loc[20]
    assert result["value"] == 200


def test_c2_missing_integer_raises_keyerror(int_label_df):
    """An integer not present in the index raises KeyError, even if valid as position."""
    with pytest.raises(KeyError):
        int_label_df.loc[0]   # 0 is not in index [10,20,30]


# ---------------------------------------------------------------------------
# C3 – label slices include BOTH start and stop
# ---------------------------------------------------------------------------

def test_c3_slice_includes_stop_label(string_df):
    """Slice 'cobra':'viper' includes 'viper' (stop is inclusive)."""
    result = string_df.loc["cobra":"viper"]
    assert "viper" in result.index


def test_c3_slice_includes_start_label(string_df):
    """Slice 'cobra':'viper' includes 'cobra' (start is inclusive)."""
    result = string_df.loc["cobra":"viper"]
    assert "cobra" in result.index


def test_c3_full_slice_returns_all_three_rows(string_df):
    """Slice 'cobra':'sidewinder' returns all 3 rows (both endpoints included)."""
    result = string_df.loc["cobra":"sidewinder"]
    assert len(result) == 3
    assert list(result.index) == ["cobra", "viper", "sidewinder"]


def test_c3_slice_length_two(string_df):
    """Slice 'cobra':'viper' returns exactly 2 rows."""
    result = string_df.loc["cobra":"viper"]
    assert len(result) == 2


# ---------------------------------------------------------------------------
# C4 – single label returns Series
# ---------------------------------------------------------------------------

def test_c4_single_row_label_returns_series(string_df):
    """Single row label returns a pd.Series."""
    result = string_df.loc["viper"]
    assert isinstance(result, pd.Series)


def test_c4_series_has_correct_name(string_df):
    """Returned Series has the row label as its name."""
    result = string_df.loc["viper"]
    assert result.name == "viper"


def test_c4_series_index_is_columns(string_df):
    """Returned Series index matches the DataFrame columns."""
    result = string_df.loc["sidewinder"]
    assert list(result.index) == ["max_speed", "shield"]


def test_c4_series_values_are_correct(string_df):
    """Single label returns correct values."""
    result = string_df.loc["sidewinder"]
    assert result["max_speed"] == 7
    assert result["shield"] == 8


# ---------------------------------------------------------------------------
# C5 – list of labels returns DataFrame
# ---------------------------------------------------------------------------

def test_c5_list_of_labels_returns_dataframe(string_df):
    """A list of labels returns a pd.DataFrame."""
    result = string_df.loc[["viper", "sidewinder"]]
    assert isinstance(result, pd.DataFrame)


def test_c5_single_item_list_returns_dataframe(string_df):
    """A single-item list still returns a DataFrame (not a Series)."""
    result = string_df.loc[["cobra"]]
    assert isinstance(result, pd.DataFrame)


def test_c5_list_preserves_row_order(string_df):
    """List access preserves the requested label order."""
    result = string_df.loc[["sidewinder", "cobra"]]
    assert list(result.index) == ["sidewinder", "cobra"]


def test_c5_list_returns_correct_shape(string_df):
    """List of 2 labels returns a DataFrame with 2 rows."""
    result = string_df.loc[["viper", "sidewinder"]]
    assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# C6 – KeyError on missing label
# ---------------------------------------------------------------------------

def test_c6_missing_string_label_raises_keyerror(string_df):
    """Accessing a non-existent string label raises KeyError."""
    with pytest.raises(KeyError):
        string_df.loc["python"]


def test_c6_missing_label_in_list_raises_keyerror(string_df):
    """A list containing a missing label raises KeyError."""
    with pytest.raises(KeyError):
        string_df.loc[["cobra", "missing_snake"]]


def test_c6_missing_integer_label_raises_keyerror(int_label_df):
    """Accessing an integer label not in the index raises KeyError."""
    with pytest.raises(KeyError):
        int_label_df.loc[99]


# ---------------------------------------------------------------------------
# C7 – Boolean array selects rows where True
# ---------------------------------------------------------------------------

def test_c7_bool_list_selects_true_rows(string_df):
    """Boolean list [T,F,T] returns rows 0 and 2."""
    result = string_df.loc[[True, False, True]]
    assert "cobra" in result.index
    assert "sidewinder" in result.index
    assert "viper" not in result.index


def test_c7_all_false_returns_empty(string_df):
    """All-False boolean array returns an empty DataFrame."""
    result = string_df.loc[[False, False, False]]
    assert len(result) == 0


def test_c7_all_true_returns_all_rows(string_df):
    """All-True boolean array returns all rows."""
    result = string_df.loc[[True, True, True]]
    assert len(result) == 3


def test_c7_bool_series_filters_by_condition(string_df):
    """Boolean Series from condition filters correctly."""
    result = string_df.loc[string_df["shield"] > 4]
    assert "viper" in result.index
    assert "sidewinder" in result.index
    assert "cobra" not in result.index


def test_c7_bool_inverted_mask_behavior(string_df):
    """True rows are included, False rows are excluded (not inverted)."""
    mask = [True, False, False]
    result = string_df.loc[mask]
    assert list(result.index) == ["cobra"]


# ---------------------------------------------------------------------------
# C8 – Callable receives the DataFrame and its result is used for indexing
# ---------------------------------------------------------------------------

def test_c8_callable_filters_by_condition(string_df):
    """Callable using a condition returns the matching row."""
    result = string_df.loc[lambda df: df["shield"] == 8]
    assert len(result) == 1
    assert "sidewinder" in result.index


def test_c8_callable_receives_dataframe(string_df):
    """Callable receives the full DataFrame (not a column or something else)."""
    received = []

    def capture(df):
        received.append(df)
        return [True, False, True]

    string_df.loc[capture]
    assert len(received) == 1
    assert isinstance(received[0], pd.DataFrame)
    assert received[0].shape == string_df.shape


def test_c8_callable_result_used_for_selection(string_df):
    """Rows selected match what the callable returns."""
    # Lambda returns mask selecting only sidewinder
    result = string_df.loc[lambda df: df["max_speed"] > 5]
    assert "sidewinder" in result.index
    assert "cobra" not in result.index
    assert "viper" not in result.index


def test_c8_callable_bool_list_result(string_df):
    """Callable returning bool list selects the right rows."""
    result = string_df.loc[lambda df: [False, True, False]]
    assert list(result.index) == ["viper"]
