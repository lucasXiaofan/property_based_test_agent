"""Property-based tests for pandas.DataFrame.assign.

Clauses tested:
  C1 - assign returns a new DataFrame with all original columns plus new ones.
  C2 - Existing columns that are re-assigned will be overwritten.
  C3 - The original DataFrame is not modified by assign.
  C4 - Callable values are evaluated on the DataFrame.
  C5 - Non-callable values are directly assigned.
  C6 - Items are computed and assigned into df in order.
  C7 - Later items in kwargs may refer to newly created or modified columns.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_dfs = data_frames(
    columns=[
        column("a", elements=st.integers(-100, 100)),
        column("b", elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
        column("c", elements=st.text(max_size=5)),
    ],
    index=range_indexes(min_size=0, max_size=20),
)

_int_vals = st.integers(min_value=-100, max_value=100)


# ---------------------------------------------------------------------------
# C1: assign returns new DataFrame with all original columns plus new ones
# ---------------------------------------------------------------------------

@given(df=_dfs, val=_int_vals)
@settings(max_examples=100)
def test_c1_returns_new_df_with_original_plus_new(df, val):
    """C1: assign returns a new object with all original + new columns."""
    result = df.assign(new_col=val)
    assert isinstance(result, pd.DataFrame)
    # All original columns present in result
    for col in df.columns:
        assert col in result.columns, f"Missing original column {col!r} in result"
    # New column present
    assert "new_col" in result.columns
    # Result is a distinct object
    assert result is not df


# ---------------------------------------------------------------------------
# C2: Existing columns are overwritten
# ---------------------------------------------------------------------------

@given(df=_dfs, val=_int_vals)
@settings(max_examples=100)
def test_c2_existing_column_overwritten(df, val):
    """C2: assign overwrites existing columns when re-assigned."""
    result = df.assign(a=val)
    assert "a" in result.columns
    assert (result["a"] == val).all(), (
        f"Expected all values of 'a' to be {val}, got {result['a'].tolist()}"
    )


# ---------------------------------------------------------------------------
# C3: Original DataFrame is not modified
# ---------------------------------------------------------------------------

@given(df=_dfs, val=_int_vals)
@settings(max_examples=100)
def test_c3_original_not_modified(df, val):
    """C3: assign does not modify the original DataFrame."""
    original = df.copy(deep=True)
    original_cols = set(df.columns)
    df.assign(new_col=val)
    # Column set unchanged
    assert set(df.columns) == original_cols, (
        f"assign modified original df columns: {set(df.columns)} vs {original_cols}"
    )
    # Values unchanged
    pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# C4: Callable values are evaluated on the DataFrame
# ---------------------------------------------------------------------------

@given(df=_dfs)
@settings(max_examples=100)
def test_c4_callable_evaluated_on_dataframe(df):
    """C4: Callable values are evaluated on the DataFrame to produce new column."""
    result = df.assign(double_a=lambda x: x["a"] * 2)
    expected = df["a"] * 2
    pd.testing.assert_series_equal(result["double_a"], expected, check_names=False)


# ---------------------------------------------------------------------------
# C5: Non-callable values are directly assigned
# ---------------------------------------------------------------------------

@given(df=_dfs, val=_int_vals)
@settings(max_examples=100)
def test_c5_noncallable_directly_assigned(df, val):
    """C5: Non-callable (scalar) values are directly assigned as new column."""
    result = df.assign(fixed_col=val)
    assert "fixed_col" in result.columns
    assert (result["fixed_col"] == val).all(), (
        f"Expected all values of 'fixed_col' to be {val}, got {result['fixed_col'].tolist()}"
    )


# ---------------------------------------------------------------------------
# C6: Items are computed and assigned in order
# ---------------------------------------------------------------------------

@given(df=_dfs)
@settings(max_examples=100)
def test_c6_items_computed_in_order(df):
    """C6: Items computed and assigned in order; later items see earlier results."""
    # doubled_a must be created before quadrupled_a can reference it
    result = df.assign(
        doubled_a=lambda x: x["a"] * 2,
        quadrupled_a=lambda x: x["doubled_a"] * 2,
    )
    assert "doubled_a" in result.columns
    assert "quadrupled_a" in result.columns
    # quadrupled_a must equal 4x original 'a'
    pd.testing.assert_series_equal(
        result["quadrupled_a"], df["a"] * 4, check_names=False
    )


# ---------------------------------------------------------------------------
# C7: Later items may refer to newly created or modified columns
# ---------------------------------------------------------------------------

@given(df=_dfs)
@settings(max_examples=100)
def test_c7_later_items_see_modified_columns(df):
    """C7: Later items can refer to columns modified by earlier assignments."""
    # First overwrite 'a' with 10x its value, then derive from the modified 'a'
    result = df.assign(
        a=lambda x: x["a"] * 10,
        derived=lambda x: x["a"] + 1,
    )
    expected_modified_a = df["a"] * 10
    expected_derived = expected_modified_a + 1
    pd.testing.assert_series_equal(
        result["derived"], expected_derived, check_names=False
    )
