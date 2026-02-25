"""Property-based tests for pandas.DataFrame.head.

Clauses tested:
  C1 - head() returns the first n rows.
  C2 - Default value of n is 5.
  C3 - head() returns the same type as caller (DataFrame).
  C4 - head() behavior mirrors df[:n] for all n.
  C5 - n=0 returns an empty object.
  C6 - Negative n returns all rows except the last |n| rows.
  C7 - n > len(df) returns all rows without error.
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
    index=range_indexes(min_size=0, max_size=30),
)

_positive_n = st.integers(min_value=1, max_value=50)
_nonneg_n = st.integers(min_value=0, max_value=50)
_negative_n = st.integers(min_value=-50, max_value=-1)
_any_n = st.integers(min_value=-50, max_value=50)


# ---------------------------------------------------------------------------
# C1: head(n) returns exactly min(n, len(df)) rows for positive n
# ---------------------------------------------------------------------------

@given(df=_dfs, n=_positive_n)
@settings(max_examples=100)
def test_c1_head_returns_first_n_rows(df, n):
    """C1: head(n) returns first n rows (positive n)."""
    result = df.head(n)
    expected_len = min(n, len(df))
    assert len(result) == expected_len, (
        f"Expected {expected_len} rows, got {len(result)} (n={n}, df_len={len(df)})"
    )
    # Values match first n rows
    pd.testing.assert_frame_equal(result, df.iloc[:n])


# ---------------------------------------------------------------------------
# C2: Default n=5
# ---------------------------------------------------------------------------

@given(df=_dfs)
@settings(max_examples=100)
def test_c2_default_n_is_5(df):
    """C2: head() with no argument returns same as head(5)."""
    pd.testing.assert_frame_equal(df.head(), df.head(5))


# ---------------------------------------------------------------------------
# C3: Return type is DataFrame
# ---------------------------------------------------------------------------

@given(df=_dfs, n=_any_n)
@settings(max_examples=100)
def test_c3_returns_dataframe_type(df, n):
    """C3: head() always returns a DataFrame."""
    result = df.head(n)
    assert isinstance(result, pd.DataFrame), (
        f"Expected DataFrame, got {type(result)} (n={n})"
    )


# ---------------------------------------------------------------------------
# C4: Equivalence with df[:n]
# ---------------------------------------------------------------------------

@given(df=_dfs, n=_any_n)
@settings(max_examples=100)
def test_c4_equivalent_to_slice(df, n):
    """C4: df.head(n) equals df[:n] for all integer n."""
    pd.testing.assert_frame_equal(df.head(n), df[:n])


# ---------------------------------------------------------------------------
# C5: n=0 returns empty DataFrame
# ---------------------------------------------------------------------------

@given(df=_dfs)
@settings(max_examples=100)
def test_c5_zero_returns_empty(df):
    """C5: head(0) returns an empty DataFrame."""
    result = df.head(0)
    assert len(result) == 0, f"Expected empty DataFrame, got {len(result)} rows"
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# C6: Negative n returns all rows except the last |n|
# ---------------------------------------------------------------------------

@given(df=_dfs, n=_negative_n)
@settings(max_examples=100)
def test_c6_negative_n_excludes_last_rows(df, n):
    """C6: head(n) with negative n equals df[:-n] (all except last |n|)."""
    result = df.head(n)
    expected_len = max(0, len(df) + n)  # equivalent to max(0, len(df) - |n|)
    assert len(result) == expected_len, (
        f"Expected {expected_len} rows, got {len(result)} (n={n}, df_len={len(df)})"
    )
    pd.testing.assert_frame_equal(result, df[:n])


# ---------------------------------------------------------------------------
# C7: n > len(df) returns all rows
# ---------------------------------------------------------------------------

@given(df=_dfs, extra=st.integers(min_value=1, max_value=100))
@settings(max_examples=100)
def test_c7_overflow_returns_all_rows(df, extra):
    """C7: head(n) where n > len(df) returns all rows without error."""
    n = len(df) + extra
    result = df.head(n)
    assert len(result) == len(df), (
        f"Expected all {len(df)} rows, got {len(result)} (n={n})"
    )
    pd.testing.assert_frame_equal(result, df)
