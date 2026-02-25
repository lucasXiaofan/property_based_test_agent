"""
Property-based and concrete tests for pandas.DataFrame.iloc

Clause coverage:
  C1 - iloc is purely integer-location based (not label-based)
  C2 - Single integer returns Series
  C3 - List/array of integers returns DataFrame
  C4 - Slice with ints returns DataFrame
  C5 - Boolean array selects rows where True
  C6 - Callable is called with the DataFrame; result is used for indexing
  C7 - Tuple indexes both rows and columns simultaneously
  C8 - Out-of-bounds integer raises IndexError
  C9 - Slice indexers allow out-of-bounds without raising IndexError
  C10 - Scalar (single row + single col) indexing returns a scalar value
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Standard 3x2 DataFrame with default RangeIndex."""
    return pd.DataFrame(
        [[1, 4], [2, 5], [3, 6]],
        columns=["a", "b"],
    )


@pytest.fixture
def non_default_index_df():
    """DataFrame whose index labels don't match their integer positions."""
    return pd.DataFrame(
        {"value": [100, 200, 300]},
        index=[10, 20, 30],
    )


@pytest.fixture
def wide_df():
    """4-row, 4-column DataFrame for two-axis indexing tests."""
    return pd.DataFrame(
        np.arange(16).reshape(4, 4),
        columns=["w", "x", "y", "z"],
    )


# ---------------------------------------------------------------------------
# C1 – position-based, not label-based
# ---------------------------------------------------------------------------

def test_c1_integer_is_position_not_label(non_default_index_df):
    """iloc[0] returns the FIRST row regardless of index labels."""
    result = non_default_index_df.iloc[0]
    assert result["value"] == 100


def test_c1_integer_position_ignores_label_mismatch(non_default_index_df):
    """iloc[1] returns row at position 1, whose label is 20, not position 20."""
    result = non_default_index_df.iloc[1]
    assert result["value"] == 200


def test_c1_last_position_via_negative_index(non_default_index_df):
    """iloc[-1] returns the last row by position."""
    result = non_default_index_df.iloc[-1]
    assert result["value"] == 300


# ---------------------------------------------------------------------------
# C2 – single integer returns Series
# ---------------------------------------------------------------------------

def test_c2_single_int_returns_series(sample_df):
    """iloc[0] returns a pd.Series."""
    result = sample_df.iloc[0]
    assert isinstance(result, pd.Series)


def test_c2_series_index_is_columns(sample_df):
    """Series returned by single-int iloc has the DataFrame's columns as its index."""
    result = sample_df.iloc[0]
    assert list(result.index) == ["a", "b"]


def test_c2_series_values_are_correct(sample_df):
    """Single-int iloc returns the correct row values."""
    result = sample_df.iloc[1]
    assert result["a"] == 2
    assert result["b"] == 5


def test_c2_single_int_middle_row(sample_df):
    """iloc[1] returns a Series (not a DataFrame)."""
    result = sample_df.iloc[1]
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# C3 – list of integers returns DataFrame
# ---------------------------------------------------------------------------

def test_c3_list_of_ints_returns_dataframe(sample_df):
    """iloc[[0, 2]] returns a pd.DataFrame."""
    result = sample_df.iloc[[0, 2]]
    assert isinstance(result, pd.DataFrame)


def test_c3_single_element_list_returns_dataframe(sample_df):
    """iloc[[0]] (single-item list) returns a DataFrame, not a Series."""
    result = sample_df.iloc[[0]]
    assert isinstance(result, pd.DataFrame)


def test_c3_list_preserves_order(sample_df):
    """List access preserves the requested position order."""
    result = sample_df.iloc[[2, 0]]
    assert result.iloc[0]["a"] == 3  # position 2 first
    assert result.iloc[1]["a"] == 1  # position 0 second


def test_c3_list_returns_correct_shape(sample_df):
    """List of 2 integers returns a DataFrame with 2 rows."""
    result = sample_df.iloc[[0, 1]]
    assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# C4 – slice returns DataFrame
# ---------------------------------------------------------------------------

def test_c4_slice_returns_dataframe(sample_df):
    """iloc[0:2] returns a pd.DataFrame."""
    result = sample_df.iloc[0:2]
    assert isinstance(result, pd.DataFrame)


def test_c4_slice_stop_is_exclusive(sample_df):
    """iloc[0:2] returns rows at positions 0 and 1 (stop is exclusive like Python slices)."""
    result = sample_df.iloc[0:2]
    assert len(result) == 2
    assert result.iloc[0]["a"] == 1
    assert result.iloc[1]["a"] == 2


def test_c4_full_slice_returns_all_rows(sample_df):
    """iloc[:] returns all rows."""
    result = sample_df.iloc[:]
    assert len(result) == 3


def test_c4_step_slice(sample_df):
    """iloc[::2] returns every other row."""
    result = sample_df.iloc[::2]
    assert len(result) == 2
    assert result.iloc[0]["a"] == 1
    assert result.iloc[1]["a"] == 3


# ---------------------------------------------------------------------------
# C5 – boolean array selects rows where True
# ---------------------------------------------------------------------------

def test_c5_bool_list_selects_true_rows(sample_df):
    """Boolean list [T,F,T] returns rows at positions 0 and 2."""
    result = sample_df.iloc[[True, False, True]]
    assert len(result) == 2
    assert result.iloc[0]["a"] == 1
    assert result.iloc[1]["a"] == 3


def test_c5_all_false_returns_empty(sample_df):
    """All-False boolean array returns an empty DataFrame."""
    result = sample_df.iloc[[False, False, False]]
    assert len(result) == 0
    assert isinstance(result, pd.DataFrame)


def test_c5_all_true_returns_all_rows(sample_df):
    """All-True boolean array returns all rows."""
    result = sample_df.iloc[[True, True, True]]
    assert len(result) == 3


def test_c5_bool_array_false_rows_excluded(sample_df):
    """Rows with False are excluded from the result."""
    result = sample_df.iloc[[False, True, False]]
    assert len(result) == 1
    assert result.iloc[0]["a"] == 2


# ---------------------------------------------------------------------------
# C6 – callable is called with the DataFrame
# ---------------------------------------------------------------------------

def test_c6_callable_filters_rows(sample_df):
    """Callable using a condition returns the matching rows."""
    result = sample_df.iloc[lambda df: [0, 2]]
    assert len(result) == 2


def test_c6_callable_receives_dataframe(sample_df):
    """Callable receives the full DataFrame as its argument."""
    received = []

    def capture(df):
        received.append(df)
        return [0]

    sample_df.iloc[capture]
    assert len(received) == 1
    assert isinstance(received[0], pd.DataFrame)
    assert received[0].shape == sample_df.shape


def test_c6_callable_result_used_for_selection(sample_df):
    """Rows selected match what the callable returns."""
    result = sample_df.iloc[lambda df: [1]]
    assert result.iloc[0]["a"] == 2


def test_c6_callable_bool_mask(sample_df):
    """Callable returning boolean array selects the right rows."""
    result = sample_df.iloc[lambda df: [False, True, False]]
    assert len(result) == 1
    assert result.iloc[0]["a"] == 2


# ---------------------------------------------------------------------------
# C7 – tuple indexes both rows and columns simultaneously
# ---------------------------------------------------------------------------

def test_c7_tuple_selects_row_and_col(wide_df):
    """iloc[0, 1] selects row 0 and column 1."""
    result = wide_df.iloc[0, 1]
    assert result == wide_df.iloc[0]["x"]


def test_c7_tuple_row_list_col_list(wide_df):
    """iloc[[0, 1], [0, 2]] returns a 2x2 DataFrame."""
    result = wide_df.iloc[[0, 1], [0, 2]]
    assert result.shape == (2, 2)


def test_c7_tuple_slice_rows_single_col(wide_df):
    """iloc[0:2, 1] returns a Series of column 'x' for rows 0-1."""
    result = wide_df.iloc[0:2, 1]
    assert isinstance(result, pd.Series)
    assert len(result) == 2


def test_c7_tuple_columns_match_requested_positions(wide_df):
    """Columns in result match the requested column positions."""
    result = wide_df.iloc[[0], [1, 3]]
    assert list(result.columns) == ["x", "z"]


# ---------------------------------------------------------------------------
# C8 – out-of-bounds integer raises IndexError
# ---------------------------------------------------------------------------

def test_c8_out_of_bounds_raises_indexerror(sample_df):
    """iloc[100] raises IndexError on a 3-row DataFrame."""
    with pytest.raises(IndexError):
        sample_df.iloc[100]


def test_c8_negative_out_of_bounds_raises_indexerror(sample_df):
    """iloc[-100] raises IndexError on a 3-row DataFrame."""
    with pytest.raises(IndexError):
        sample_df.iloc[-100]


def test_c8_list_out_of_bounds_raises_indexerror(sample_df):
    """iloc[[0, 999]] raises IndexError when one element is out-of-bounds."""
    with pytest.raises(IndexError):
        sample_df.iloc[[0, 999]]


def test_c8_in_bounds_does_not_raise(sample_df):
    """iloc[2] does not raise for a 3-row DataFrame (last valid position)."""
    result = sample_df.iloc[2]
    assert result["a"] == 3


# ---------------------------------------------------------------------------
# C9 – slice indexers allow out-of-bounds without raising IndexError
# ---------------------------------------------------------------------------

def test_c9_slice_with_large_stop_does_not_raise(sample_df):
    """iloc[0:1000] does not raise IndexError even though 1000 >> len(df)."""
    result = sample_df.iloc[0:1000]
    assert len(result) == 3


def test_c9_slice_with_large_start_returns_empty(sample_df):
    """iloc[1000:] does not raise and returns an empty DataFrame."""
    result = sample_df.iloc[1000:]
    assert len(result) == 0
    assert isinstance(result, pd.DataFrame)


def test_c9_slice_negative_large_start_does_not_raise(sample_df):
    """iloc[-1000:] does not raise and clamps to beginning."""
    result = sample_df.iloc[-1000:]
    assert len(result) == 3


# ---------------------------------------------------------------------------
# C10 – scalar (row + col) indexing returns a scalar value
# ---------------------------------------------------------------------------

def test_c10_scalar_indexing_returns_scalar(sample_df):
    """iloc[0, 0] returns a scalar, not a Series or DataFrame."""
    result = sample_df.iloc[0, 0]
    assert not isinstance(result, (pd.Series, pd.DataFrame))


def test_c10_scalar_value_is_correct(sample_df):
    """iloc[1, 1] returns the correct scalar value."""
    result = sample_df.iloc[1, 1]
    assert result == 5


def test_c10_scalar_negative_indices(sample_df):
    """iloc[-1, -1] returns the scalar at the last row/column."""
    result = sample_df.iloc[-1, -1]
    assert result == 6
