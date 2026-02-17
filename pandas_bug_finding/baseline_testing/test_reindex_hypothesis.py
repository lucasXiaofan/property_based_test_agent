"""
Property-Based Testing Script for pandas DataFrame.reindex()
==============================================================

This script comprehensively tests the pandas DataFrame.reindex() method using
property-based testing with Hypothesis. It aims to discover bugs by testing both
explicit properties claimed in the docstring and implicit properties that are
expected but not documented.

EXPLICIT PROPERTIES (from docstring):
1. Returns a DataFrame
2. Places NA/NaN in locations having no value in the previous index
3. A new object is produced (not the same object as input)
4. `labels` parameter conforms the axis specified by `axis` to new labels
5. `index` parameter sets new labels for the row index
6. `columns` parameter sets new labels for the columns
7. `axis` parameter targets axis by name ('index', 'columns') or number (0, 1)
8. `method` parameter fills holes: None (default), 'pad'/'ffill', 'backfill'/'bfill', 'nearest'
   - Only applicable with monotonically increasing/decreasing index
9. `fill_value` parameter sets value for missing locations (default NaN, any compatible value)
10. `limit` parameter caps maximum consecutive forward/backward fills
11. `tolerance` parameter sets maximum distance for inexact matches
12. `level` parameter broadcasts across a MultiIndex level
13. Two calling conventions: (index=..., columns=...) vs (labels, axis=...)
14. Filling during reindex does NOT look at DataFrame values, only compares indexes
    (pre-existing NaN in original data is NOT filled by method)

IMPLICIT PROPERTIES (not explicitly documented but expected):
1. Original DataFrame is not modified (immutability)
2. Reindexing to the same index returns an equivalent DataFrame
3. Result shape = (len(new_index), original_cols) for index reindex, etc.
4. Data values at matching index positions are preserved exactly
5. Preserves column dtypes where no new NaN is introduced (or upcasts to nullable)
6. Works with various index types: integer, string, datetime
7. method='ffill' and method='pad' produce identical results
8. method='bfill' and method='backfill' produce identical results
9. Handles empty DataFrames correctly
10. Reindexing columns preserves row count
11. Reindexing with duplicate labels in new index duplicates rows
12. Reindexing then reindexing back to original recovers matching values
13. Order of result matches order of new index/columns
14. fill_value only fills positions from new labels not in original, not pre-existing NaN
15. limit=0 means no filling at all (same as method=None)
16. Column reindex with fill_value fills missing columns with that value

PANDAS VERSION:
This script targets pandas 3.0.0.

HOW TO RUN:
-----------
Using uv:
    uv run pytest pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py -v

With Hypothesis statistics:
    uv run pytest pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py -v --hypothesis-show-statistics

Verbose output:
    uv run pytest pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py -vv -s

Run specific test class:
    uv run pytest pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py::TestExplicitProperties -v

FAILED test_reindex_hypothesis.py::TestMethodParameter::test_method_with_non_monotonic_raises - Failed: DID NOT RAISE <class 'ValueError'>
FAILED test_reindex_hypothesis.py::TestLimitToleranceInteraction::test_limit_zero_means_no_fill - ValueError: Limit must be greater than 0
=========== 2 failed, 61 passed in 9.58s ============
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import copy


# ==============================================================================
# HYPOTHESIS STRATEGIES
# ==============================================================================

@st.composite
def string_index_labels(draw, min_size=0, max_size=10):
    """Generate a list of unique string labels for use as index."""
    labels = draw(st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1, max_size=8
        ),
        min_size=min_size, max_size=max_size, unique=True
    ))
    return labels


@st.composite
def integer_index_labels(draw, min_size=0, max_size=10):
    """Generate a list of unique integer labels for use as index."""
    labels = draw(st.lists(
        st.integers(min_value=-100, max_value=100),
        min_size=min_size, max_size=max_size, unique=True
    ))
    return labels


@st.composite
def numeric_dataframes_with_str_index(draw, min_rows=1, max_rows=10, min_cols=1, max_cols=5):
    """Generate DataFrames with string index and numeric data."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    idx_labels = draw(st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll')),
            min_size=1, max_size=6
        ),
        min_size=n_rows, max_size=n_rows, unique=True
    ))
    col_names = [f"col_{i}" for i in range(n_cols)]

    data = {}
    for col_name in col_names:
        values = draw(st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            min_size=n_rows, max_size=n_rows
        ))
        data[col_name] = values

    return pd.DataFrame(data, index=pd.Index(idx_labels))


@st.composite
def numeric_dataframes_with_int_index(draw, min_rows=1, max_rows=10, min_cols=1, max_cols=5):
    """Generate DataFrames with integer index and numeric data."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    idx_labels = draw(st.lists(
        st.integers(min_value=0, max_value=50),
        min_size=n_rows, max_size=n_rows, unique=True
    ))
    col_names = [f"col_{i}" for i in range(n_cols)]

    data = {}
    for col_name in col_names:
        values = draw(st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            min_size=n_rows, max_size=n_rows
        ))
        data[col_name] = values

    return pd.DataFrame(data, index=pd.Index(idx_labels))


@st.composite
def numeric_dataframes_with_nan(draw, min_rows=2, max_rows=10, min_cols=1, max_cols=4):
    """Generate DataFrames with numeric data that may contain NaN."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    idx_labels = list(range(n_rows))
    col_names = [f"col_{i}" for i in range(n_cols)]

    data = {}
    for col_name in col_names:
        values = draw(st.lists(
            st.one_of(
                st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                st.just(np.nan),
            ),
            min_size=n_rows, max_size=n_rows
        ))
        data[col_name] = values

    return pd.DataFrame(data, index=pd.Index(idx_labels))


@st.composite
def monotonic_int_index_dataframes(draw, min_rows=3, max_rows=15):
    """Generate DataFrames with sorted integer index (for method= tests)."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=1, max_value=3))

    # Generate sorted unique integers
    idx_labels = sorted(draw(st.lists(
        st.integers(min_value=0, max_value=100),
        min_size=n_rows, max_size=n_rows, unique=True
    )))
    col_names = [f"col_{i}" for i in range(n_cols)]

    data = {}
    for col_name in col_names:
        values = draw(st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            min_size=n_rows, max_size=n_rows
        ))
        data[col_name] = values

    return pd.DataFrame(data, index=pd.Index(idx_labels))


# ==============================================================================
# TEST CLASS 1: EXPLICIT PROPERTIES FROM DOCSTRING
# ==============================================================================

class TestExplicitProperties:
    """Test properties explicitly claimed in the DataFrame.reindex() docstring."""

    @given(df=numeric_dataframes_with_str_index(),
           new_labels=string_index_labels(min_size=1, max_size=8))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_returns_dataframe(self, df, new_labels):
        """reindex always returns a DataFrame."""
        result = df.reindex(new_labels)
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframes_with_str_index(),
           new_labels=string_index_labels(min_size=1, max_size=8))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_new_labels_not_in_original_get_nan(self, df, new_labels):
        """Labels in new index not present in original get NaN values."""
        result = df.reindex(new_labels)
        missing_labels = [l for l in new_labels if l not in df.index]
        for label in missing_labels:
            assert result.loc[label].isna().all(), \
                f"Label '{label}' not in original should have all NaN, got {result.loc[label]}"

    @given(df=numeric_dataframes_with_str_index())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_produces_new_object(self, df):
        """reindex produces a new object (not the same reference)."""
        result = df.reindex(df.index)
        assert result is not df

    @given(df=numeric_dataframes_with_str_index(),
           new_labels=string_index_labels(min_size=1, max_size=6))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_index_parameter(self, df, new_labels):
        """index= parameter sets new row labels."""
        result = df.reindex(index=new_labels)
        assert list(result.index) == new_labels

    @given(df=numeric_dataframes_with_str_index(min_cols=2))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_columns_parameter(self, df):
        """columns= parameter sets new column labels."""
        orig_cols = list(df.columns)
        new_cols = orig_cols[:1] + ['NEW_COL']
        result = df.reindex(columns=new_cols)
        assert list(result.columns) == new_cols
        # NEW_COL should be all NaN
        assert result['NEW_COL'].isna().all()

    @given(df=numeric_dataframes_with_str_index(),
           new_labels=string_index_labels(min_size=1, max_size=6))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_axis_0_same_as_index(self, df, new_labels):
        """labels with axis=0 produces same result as index= parameter."""
        result_axis = df.reindex(new_labels, axis=0)
        result_index = df.reindex(index=new_labels)
        pd.testing.assert_frame_equal(result_axis, result_index)

    @given(df=numeric_dataframes_with_str_index(min_cols=2))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_axis_1_same_as_columns(self, df):
        """labels with axis=1 produces same result as columns= parameter."""
        new_cols = list(df.columns[:1]) + ['NEW_COL']
        result_axis = df.reindex(new_cols, axis=1)
        result_cols = df.reindex(columns=new_cols)
        pd.testing.assert_frame_equal(result_axis, result_cols)

    @given(df=numeric_dataframes_with_str_index())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_axis_names_work(self, df):
        """axis='index' and axis='columns' work like axis=0 and axis=1."""
        new_idx = list(df.index)
        result_name = df.reindex(new_idx, axis='index')
        result_num = df.reindex(new_idx, axis=0)
        pd.testing.assert_frame_equal(result_name, result_num)

    @given(df=numeric_dataframes_with_str_index(),
           new_labels=string_index_labels(min_size=1, max_size=6))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_fill_value_parameter(self, df, new_labels):
        """fill_value fills missing positions with the given value."""
        fill_val = -999.0
        result = df.reindex(new_labels, fill_value=fill_val)
        missing_labels = [l for l in new_labels if l not in df.index]
        for label in missing_labels:
            assert (result.loc[label] == fill_val).all(), \
                f"Missing label '{label}' should have fill_value={fill_val}"

    @given(df=monotonic_int_index_dataframes())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_method_ffill(self, df):
        """method='ffill' forward fills gaps from the last valid value."""
        idx = list(df.index)
        # Insert a new label between existing ones
        if len(idx) >= 2:
            gap_label = (idx[0] + idx[1]) // 2
            if gap_label not in idx and gap_label != idx[0]:
                new_idx = sorted(idx + [gap_label])
                result = df.reindex(new_idx, method='ffill')
                # gap_label should have been forward-filled from idx[0]
                expected_source = max(l for l in idx if l <= gap_label)
                for col in df.columns:
                    assert result.loc[gap_label, col] == df.loc[expected_source, col]

    @given(df=monotonic_int_index_dataframes())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_method_bfill(self, df):
        """method='bfill' backward fills gaps from the next valid value."""
        idx = list(df.index)
        if len(idx) >= 2:
            gap_label = (idx[0] + idx[1]) // 2
            if gap_label not in idx and gap_label != idx[1]:
                new_idx = sorted(idx + [gap_label])
                result = df.reindex(new_idx, method='bfill')
                expected_source = min(l for l in idx if l >= gap_label)
                for col in df.columns:
                    assert result.loc[gap_label, col] == df.loc[expected_source, col]

    @given(df=monotonic_int_index_dataframes())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_ffill_pad_equivalence(self, df):
        """method='ffill' and method='pad' produce identical results."""
        idx = list(df.index)
        if len(idx) >= 2:
            gap_label = (idx[0] + idx[1]) // 2
            if gap_label not in idx:
                new_idx = sorted(idx + [gap_label])
                result_ffill = df.reindex(new_idx, method='ffill')
                result_pad = df.reindex(new_idx, method='pad')
                pd.testing.assert_frame_equal(result_ffill, result_pad)

    @given(df=monotonic_int_index_dataframes())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_bfill_backfill_equivalence(self, df):
        """method='bfill' and method='backfill' produce identical results."""
        idx = list(df.index)
        if len(idx) >= 2:
            gap_label = (idx[0] + idx[1]) // 2
            if gap_label not in idx:
                new_idx = sorted(idx + [gap_label])
                result_bfill = df.reindex(new_idx, method='bfill')
                result_backfill = df.reindex(new_idx, method='backfill')
                pd.testing.assert_frame_equal(result_bfill, result_backfill)

    @given(df=monotonic_int_index_dataframes())
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_limit_caps_fill(self, df):
        """limit parameter caps the number of consecutive fills."""
        idx = sorted(list(df.index))
        assume(len(idx) >= 2)
        # Create multiple gap labels after the last index value
        max_val = max(idx)
        new_idx = idx + [max_val + 1, max_val + 2, max_val + 3]
        result_limit1 = df.reindex(new_idx, method='ffill', limit=1)
        # Only max_val+1 should be filled; max_val+2 and max_val+3 should be NaN
        for col in df.columns:
            assert not pd.isna(result_limit1.loc[max_val + 1, col]), \
                "First gap should be filled with limit=1"
            assert pd.isna(result_limit1.loc[max_val + 2, col]), \
                "Second gap should be NaN with limit=1"

    @given(df=numeric_dataframes_with_nan())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_method_does_not_fill_original_nan(self, df):
        """Filling during reindex does NOT fill pre-existing NaN values in the data."""
        # Use exact same index so no new labels are introduced
        original_nan_mask = df.isna()
        if not original_nan_mask.any().any():
            return  # no NaN to test
        # Sort index for monotonic requirement
        df_sorted = df.sort_index()
        result = df_sorted.reindex(df_sorted.index, method='ffill')
        # Pre-existing NaN in original data should remain NaN
        for col in df_sorted.columns:
            for idx_label in df_sorted.index:
                if pd.isna(df_sorted.loc[idx_label, col]):
                    assert pd.isna(result.loc[idx_label, col]), \
                        f"Pre-existing NaN at ({idx_label}, {col}) should not be filled"


# ==============================================================================
# TEST CLASS 2: IMPLICIT BASIC BEHAVIOR
# ==============================================================================

class TestImplicitBasicBehavior:
    """Test implicit behavioral properties of reindex."""

    @given(df=numeric_dataframes_with_str_index(),
           new_labels=string_index_labels(min_size=1, max_size=6))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_immutability(self, df, new_labels):
        """reindex does not modify the original DataFrame."""
        df_copy = df.copy(deep=True)
        df.reindex(new_labels)
        pd.testing.assert_frame_equal(df, df_copy)

    @given(df=numeric_dataframes_with_str_index())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_same_index_preserves_data(self, df):
        """Reindexing to the same index preserves all data."""
        result = df.reindex(df.index)
        pd.testing.assert_frame_equal(result, df)

    @given(df=numeric_dataframes_with_str_index(),
           new_labels=string_index_labels(min_size=1, max_size=8))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_result_shape_index_reindex(self, df, new_labels):
        """Result has len(new_labels) rows and same number of columns."""
        result = df.reindex(new_labels)
        assert result.shape == (len(new_labels), len(df.columns))

    @given(df=numeric_dataframes_with_str_index(),
           new_labels=string_index_labels(min_size=1, max_size=6))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_matching_labels_preserve_values(self, df, new_labels):
        """Data at labels present in both old and new index is preserved."""
        result = df.reindex(new_labels)
        common_labels = [l for l in new_labels if l in df.index]
        for label in common_labels:
            pd.testing.assert_series_equal(
                result.loc[label], df.loc[label], check_names=False
            )

    @given(df=numeric_dataframes_with_str_index(),
           new_labels=string_index_labels(min_size=1, max_size=6))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_result_index_matches_new_labels(self, df, new_labels):
        """Result index exactly matches the new labels in order."""
        result = df.reindex(new_labels)
        assert list(result.index) == new_labels

    @given(df=numeric_dataframes_with_str_index(min_cols=2))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_column_reindex_preserves_rows(self, df):
        """Reindexing columns does not change the number of rows."""
        new_cols = list(df.columns[:1]) + ['EXTRA']
        result = df.reindex(columns=new_cols)
        assert len(result) == len(df)

    @given(df=numeric_dataframes_with_str_index(min_cols=2))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_column_reindex_result_columns_match(self, df):
        """Column reindex result has exactly the requested columns in order."""
        new_cols = list(df.columns[:1]) + ['EXTRA']
        result = df.reindex(columns=new_cols)
        assert list(result.columns) == new_cols

    @given(df=numeric_dataframes_with_str_index())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_deterministic(self, df):
        """Same reindex call produces identical results."""
        new_idx = list(df.index) + ['ZZZZ']
        result1 = df.reindex(new_idx)
        result2 = df.reindex(new_idx)
        pd.testing.assert_frame_equal(result1, result2)

    @given(df=numeric_dataframes_with_int_index(),
           new_labels=integer_index_labels(min_size=1, max_size=6))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_works_with_integer_index(self, df, new_labels):
        """reindex works with integer index labels."""
        result = df.reindex(new_labels)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (len(new_labels), len(df.columns))


# ==============================================================================
# TEST CLASS 3: FILL VALUE PROPERTIES
# ==============================================================================

class TestFillValueProperties:
    """Test fill_value parameter behavior."""

    @given(df=numeric_dataframes_with_str_index())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_fill_value_zero(self, df):
        """fill_value=0 fills new positions with 0."""
        new_idx = list(df.index) + ['MISSING_LABEL']
        result = df.reindex(new_idx, fill_value=0)
        assert (result.loc['MISSING_LABEL'] == 0).all()

    @given(df=numeric_dataframes_with_str_index(),
           fill_val=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e4, max_value=1e4))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_fill_value_arbitrary(self, df, fill_val):
        """fill_value with arbitrary float works correctly."""
        new_idx = list(df.index) + ['MISSING_LABEL']
        result = df.reindex(new_idx, fill_value=fill_val)
        assert (result.loc['MISSING_LABEL'] == fill_val).all()

    @given(df=numeric_dataframes_with_nan())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_fill_value_does_not_fill_original_nan(self, df):
        """fill_value only fills positions from new labels, NOT pre-existing NaN."""
        original_nan_mask = df.isna()
        if not original_nan_mask.any().any():
            return
        # Reindex to same index with fill_value — should not affect pre-existing NaN
        result = df.reindex(df.index, fill_value=-999)
        for col in df.columns:
            for idx_label in df.index:
                if pd.isna(df.loc[idx_label, col]):
                    assert pd.isna(result.loc[idx_label, col]), \
                        f"Pre-existing NaN at ({idx_label}, {col}) should not be filled by fill_value"

    @given(df=numeric_dataframes_with_str_index(min_cols=2))
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_fill_value_for_new_columns(self, df):
        """fill_value fills new columns introduced by column reindex."""
        fill_val = -42.0
        new_cols = list(df.columns) + ['NEW_COL']
        result = df.reindex(columns=new_cols, fill_value=fill_val)
        assert (result['NEW_COL'] == fill_val).all()

    @given(df=numeric_dataframes_with_str_index())
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_fill_value_string(self, df):
        """fill_value with a string fills missing positions."""
        new_idx = list(df.index) + ['MISSING']
        result = df.reindex(new_idx, fill_value='FILLED')
        assert (result.loc['MISSING'] == 'FILLED').all()

    @given(df=numeric_dataframes_with_str_index(),
           new_labels=string_index_labels(min_size=1, max_size=4))
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_default_fill_is_nan(self, df, new_labels):
        """Default fill_value is NaN."""
        result = df.reindex(new_labels)
        missing_labels = [l for l in new_labels if l not in df.index]
        for label in missing_labels:
            assert result.loc[label].isna().all()


# ==============================================================================
# TEST CLASS 4: METHOD PARAMETER (FILLING LOGIC)
# ==============================================================================

class TestMethodParameter:
    """Test method parameter for forward/backward/nearest filling."""

    @given(df=monotonic_int_index_dataframes())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_method_none_no_fill(self, df):
        """method=None (default) does not fill any gaps."""
        idx = sorted(list(df.index))
        max_val = max(idx)
        new_idx = idx + [max_val + 1]
        result = df.reindex(new_idx)
        for col in df.columns:
            assert pd.isna(result.loc[max_val + 1, col])

    @given(df=monotonic_int_index_dataframes())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_method_nearest(self, df):
        """method='nearest' fills with nearest valid value."""
        idx = sorted(list(df.index))
        assume(len(idx) >= 2)
        # Insert a gap label very close to idx[0]
        gap_label = idx[0] - 1
        if gap_label >= 0 and gap_label not in idx:
            new_idx = sorted([gap_label] + idx)
            result = df.reindex(new_idx, method='nearest')
            # gap_label is closest to idx[0]
            for col in df.columns:
                assert result.loc[gap_label, col] == df.loc[idx[0], col]

    @given(df=monotonic_int_index_dataframes())
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_method_with_non_monotonic_raises(self, df):
        """method= with non-monotonic index raises ValueError."""
        # Shuffle the index to make it non-monotonic
        shuffled_idx = list(df.index)
        if len(shuffled_idx) >= 2:
            shuffled_idx[0], shuffled_idx[-1] = shuffled_idx[-1], shuffled_idx[0]
            df_shuffled = df.reindex(shuffled_idx)
            # Now try to use method on non-monotonic index
            new_idx = shuffled_idx + [max(shuffled_idx) + 1]
            with pytest.raises(ValueError):
                df_shuffled.reindex(new_idx, method='ffill')

    @given(df=monotonic_int_index_dataframes())
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_ffill_does_not_look_forward(self, df):
        """ffill only looks backward, not forward. A label before all originals gets NaN."""
        idx = sorted(list(df.index))
        min_val = min(idx)
        if min_val > 0:
            new_idx = [min_val - 1] + idx
            result = df.reindex(new_idx, method='ffill')
            # Label before all originals: nothing to forward-fill from
            for col in df.columns:
                assert pd.isna(result.loc[min_val - 1, col])

    @given(df=monotonic_int_index_dataframes())
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_bfill_does_not_look_backward(self, df):
        """bfill only looks forward, not backward. A label after all originals gets NaN."""
        idx = sorted(list(df.index))
        max_val = max(idx)
        new_idx = idx + [max_val + 1]
        result = df.reindex(new_idx, method='bfill')
        for col in df.columns:
            assert pd.isna(result.loc[max_val + 1, col])


# ==============================================================================
# TEST CLASS 5: TOLERANCE PARAMETER
# ==============================================================================

class TestToleranceParameter:
    """Test tolerance parameter for inexact matching."""

    @given(df=monotonic_int_index_dataframes())
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_tolerance_within_range_fills(self, df):
        """Labels within tolerance distance are filled."""
        idx = sorted(list(df.index))
        assume(len(idx) >= 2)
        max_val = max(idx)
        gap_label = max_val + 1
        new_idx = idx + [gap_label]
        result = df.reindex(new_idx, method='nearest', tolerance=2)
        # gap_label is within tolerance=2 of max_val
        for col in df.columns:
            assert not pd.isna(result.loc[gap_label, col])

    @given(df=monotonic_int_index_dataframes())
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_tolerance_outside_range_stays_nan(self, df):
        """Labels outside tolerance distance remain NaN."""
        idx = sorted(list(df.index))
        max_val = max(idx)
        gap_label = max_val + 100
        new_idx = idx + [gap_label]
        result = df.reindex(new_idx, method='nearest', tolerance=1)
        # gap_label is outside tolerance=1
        for col in df.columns:
            assert pd.isna(result.loc[gap_label, col])


# ==============================================================================
# TEST CLASS 6: BOTH INDEX AND COLUMNS SIMULTANEOUSLY
# ==============================================================================

class TestSimultaneousReindex:
    """Test reindexing both index and columns at the same time."""

    @given(df=numeric_dataframes_with_str_index(min_rows=2, min_cols=2))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_reindex_both_axes(self, df):
        """Reindexing both index and columns simultaneously works."""
        new_idx = list(df.index[:1]) + ['NEW_ROW']
        new_cols = list(df.columns[:1]) + ['NEW_COL']
        result = df.reindex(index=new_idx, columns=new_cols)
        assert result.shape == (len(new_idx), len(new_cols))
        assert list(result.index) == new_idx
        assert list(result.columns) == new_cols

    @given(df=numeric_dataframes_with_str_index(min_rows=2, min_cols=2))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_both_axes_fill_value(self, df):
        """fill_value applies to both new rows and new columns."""
        new_idx = list(df.index[:1]) + ['NEW_ROW']
        new_cols = list(df.columns[:1]) + ['NEW_COL']
        result = df.reindex(index=new_idx, columns=new_cols, fill_value=-1.0)
        # New row should have fill_value in all columns
        first_col = new_cols[0]
        assert result.loc['NEW_ROW', 'NEW_COL'] == -1.0
        # New column for existing row should have fill_value
        existing_row = new_idx[0]
        assert result.loc[existing_row, 'NEW_COL'] == -1.0

    @given(df=numeric_dataframes_with_str_index(min_rows=2, min_cols=2))
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_both_axes_preserves_intersection(self, df):
        """Values at the intersection of old and new labels are preserved."""
        new_idx = list(df.index[:1]) + ['NEW_ROW']
        new_cols = list(df.columns[:1]) + ['NEW_COL']
        result = df.reindex(index=new_idx, columns=new_cols)
        # The intersection cell should match original
        existing_row = new_idx[0]
        existing_col = new_cols[0]
        assert result.loc[existing_row, existing_col] == df.loc[existing_row, existing_col]


# ==============================================================================
# TEST CLASS 7: EDGE CASES AND BOUNDARY CONDITIONS
# ==============================================================================

class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    @given(params=st.just({}))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_empty_dataframe_reindex(self, params):
        """Reindexing an empty DataFrame returns an empty-like DataFrame."""
        df = pd.DataFrame({'a': pd.Series([], dtype='float64')})
        result = df.reindex([0, 1, 2])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result['a'].isna().all()

    def test_reindex_to_empty_index(self):
        """Reindexing to an empty index returns a zero-row DataFrame."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = df.reindex([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ['a']

    @given(df=numeric_dataframes_with_str_index())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_duplicate_labels_in_new_index(self, df):
        """Reindexing with duplicate labels in new index duplicates rows."""
        assume(len(df.index) > 0)
        first_label = df.index[0]
        new_idx = [first_label, first_label]
        result = df.reindex(new_idx)
        assert len(result) == 2
        pd.testing.assert_series_equal(
            result.iloc[0], result.iloc[1], check_names=False
        )

    def test_single_row_dataframe(self):
        """Single-row DataFrame reindex works."""
        df = pd.DataFrame({'a': [1.0], 'b': [2.0]}, index=['x'])
        result = df.reindex(['x', 'y'])
        assert len(result) == 2
        assert result.loc['x', 'a'] == 1.0
        assert pd.isna(result.loc['y', 'a'])

    def test_single_column_dataframe(self):
        """Single-column DataFrame reindex works."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        result = df.reindex(columns=['a', 'b'])
        assert list(result.columns) == ['a', 'b']
        assert result['b'].isna().all()

    @given(df=numeric_dataframes_with_int_index())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_reindex_preserves_column_dtypes_when_no_nan(self, df):
        """When no new NaN is introduced (same index), dtypes are preserved."""
        result = df.reindex(df.index)
        for col in df.columns:
            assert result[col].dtype == df[col].dtype, \
                f"Column {col}: expected {df[col].dtype}, got {result[col].dtype}"

    def test_multiindex_level_parameter(self):
        """level parameter works with MultiIndex."""
        idx = pd.MultiIndex.from_tuples([
            ('a', 1), ('a', 2), ('b', 1), ('b', 2)
        ], names=['first', 'second'])
        df = pd.DataFrame({'val': [10, 20, 30, 40]}, index=idx)
        result = df.reindex(['a', 'b', 'c'], level='first')
        assert isinstance(result, pd.DataFrame)

    def test_datetime_index_reindex(self):
        """Reindex works with datetime index."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        df = pd.DataFrame({'val': [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
        new_dates = pd.date_range('2019-12-30', periods=8, freq='D')
        result = df.reindex(new_dates)
        assert len(result) == 8
        # First two days should be NaN (before original range)
        assert result.iloc[0].isna().all()
        assert result.iloc[1].isna().all()
        # 2020-01-01 should have value 1.0
        assert result.loc['2020-01-01', 'val'] == 1.0

    def test_datetime_index_with_method_ffill(self):
        """ffill with datetime index works correctly."""
        dates = pd.date_range('2020-01-01', periods=3, freq='2D')
        df = pd.DataFrame({'val': [10.0, 20.0, 30.0]}, index=dates)
        new_dates = pd.date_range('2020-01-01', periods=5, freq='D')
        result = df.reindex(new_dates, method='ffill')
        # 2020-01-02 should be forward-filled from 2020-01-01
        assert result.loc['2020-01-02', 'val'] == 10.0

    def test_mixed_dtype_columns(self):
        """Reindexing preserves mixed dtype columns."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'str_col': ['a', 'b', 'c'],
            'float_col': [1.1, 2.2, 3.3],
        })
        result = df.reindex([0, 1, 2, 3])
        assert len(result) == 4
        assert result.loc[0, 'int_col'] == 1
        assert result.loc[0, 'str_col'] == 'a'

    def test_boolean_dtype_column(self):
        """Reindex works with boolean dtype columns."""
        df = pd.DataFrame({'flag': [True, False, True]}, index=[0, 1, 2])
        result = df.reindex([0, 1, 2, 3])
        assert len(result) == 4
        assert result.loc[0, 'flag'] == True

    def test_nullable_int_dtype(self):
        """Reindex works with nullable Int64 dtype."""
        df = pd.DataFrame({'val': pd.array([1, 2, 3], dtype='Int64')}, index=[0, 1, 2])
        result = df.reindex([0, 1, 2, 3])
        assert len(result) == 4
        assert pd.isna(result.loc[3, 'val'])

    def test_categorical_column(self):
        """Reindex works with categorical columns."""
        df = pd.DataFrame({
            'cat': pd.Categorical(['a', 'b', 'c'])
        }, index=[0, 1, 2])
        result = df.reindex([0, 1, 2, 3])
        assert len(result) == 4


# ==============================================================================
# TEST CLASS 8: ROUNDTRIP AND COMPOSABILITY
# ==============================================================================

class TestRoundtripAndComposability:
    """Test roundtrip reindexing and composability properties."""

    @given(df=numeric_dataframes_with_str_index(min_rows=2, max_rows=6))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_reindex_roundtrip_preserves_matching_values(self, df):
        """Reindex to superset, then back to original, preserves values."""
        original_idx = list(df.index)
        superset_idx = original_idx + ['EXTRA1', 'EXTRA2']
        result_expanded = df.reindex(superset_idx)
        result_contracted = result_expanded.reindex(original_idx)
        pd.testing.assert_frame_equal(result_contracted, df)

    @given(df=numeric_dataframes_with_str_index(min_rows=2, max_rows=6))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_reindex_subset_then_superset(self, df):
        """Reindex to subset then back to original: subset values match, others are NaN."""
        assume(len(df.index) >= 2)
        original_idx = list(df.index)
        subset_idx = original_idx[:1]
        result_subset = df.reindex(subset_idx)
        result_back = result_subset.reindex(original_idx)
        # Subset labels should match
        for label in subset_idx:
            pd.testing.assert_series_equal(
                result_back.loc[label], df.loc[label], check_names=False
            )
        # Other labels should be NaN
        for label in original_idx[1:]:
            assert result_back.loc[label].isna().all()

    @given(df=numeric_dataframes_with_str_index(min_cols=2, max_cols=4))
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_column_reindex_roundtrip(self, df):
        """Reindex columns to superset then back preserves original data."""
        original_cols = list(df.columns)
        superset_cols = original_cols + ['EXTRA_COL']
        result_expanded = df.reindex(columns=superset_cols)
        result_contracted = result_expanded.reindex(columns=original_cols)
        pd.testing.assert_frame_equal(result_contracted, df)

    @given(df=numeric_dataframes_with_str_index(min_rows=2))
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_idempotent_reindex(self, df):
        """Reindexing to the same target twice is idempotent."""
        new_idx = list(df.index) + ['EXTRA']
        result1 = df.reindex(new_idx)
        result2 = result1.reindex(new_idx)
        pd.testing.assert_frame_equal(result1, result2)

    @given(df=numeric_dataframes_with_str_index(min_rows=2))
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_reindex_order_matters(self, df):
        """The order of labels in new index determines the row order of the result."""
        assume(len(df.index) >= 2)
        idx = list(df.index)
        reversed_idx = list(reversed(idx))
        result_orig = df.reindex(idx)
        result_rev = df.reindex(reversed_idx)
        # Reversed result should have reversed row order
        assert list(result_rev.index) == reversed_idx
        for i, label in enumerate(reversed_idx):
            pd.testing.assert_series_equal(
                result_rev.iloc[i], result_orig.loc[label], check_names=False
            )


# ==============================================================================
# TEST CLASS 9: LIMIT AND TOLERANCE INTERACTION
# ==============================================================================

class TestLimitToleranceInteraction:
    """Test limit and tolerance interactions with method parameter."""

    def test_limit_zero_means_no_fill(self):
        """limit=0 with method= means no filling occurs."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]}, index=[0, 2, 4])
        result = df.reindex([0, 1, 2, 3, 4], method='ffill', limit=0)
        assert pd.isna(result.loc[1, 'a'])
        assert pd.isna(result.loc[3, 'a'])

    def test_limit_greater_than_gap(self):
        """limit larger than gap size fills everything."""
        df = pd.DataFrame({'a': [1.0, 2.0]}, index=[0, 5])
        new_idx = [0, 1, 2, 3, 4, 5]
        result = df.reindex(new_idx, method='ffill', limit=10)
        # All gaps between 0 and 5 should be filled with 1.0
        for i in range(1, 5):
            assert result.loc[i, 'a'] == 1.0

    def test_tolerance_with_ffill(self):
        """tolerance limits how far ffill can look."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]}, index=[0, 10, 20])
        new_idx = [0, 1, 5, 10, 11, 15, 20]
        result = df.reindex(new_idx, method='ffill', tolerance=2)
        # 1 is within tolerance of 0 → filled
        assert result.loc[1, 'a'] == 1.0
        # 5 is NOT within tolerance of 0 → NaN
        assert pd.isna(result.loc[5, 'a'])
        # 11 is within tolerance of 10 → filled
        assert result.loc[11, 'a'] == 2.0
        # 15 is NOT within tolerance of 10 → NaN
        assert pd.isna(result.loc[15, 'a'])

    def test_tolerance_with_bfill(self):
        """tolerance limits how far bfill can look."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]}, index=[0, 10, 20])
        new_idx = [0, 5, 9, 10, 15, 19, 20]
        result = df.reindex(new_idx, method='bfill', tolerance=2)
        # 9 is within tolerance of 10 → filled
        assert result.loc[9, 'a'] == 2.0
        # 5 is NOT within tolerance of 10 → NaN
        assert pd.isna(result.loc[5, 'a'])
        # 19 is within tolerance of 20 → filled
        assert result.loc[19, 'a'] == 3.0
        # 15 is NOT within tolerance of 20 → NaN
        assert pd.isna(result.loc[15, 'a'])

    def test_limit_and_tolerance_combined(self):
        """limit and tolerance can be used together."""
        df = pd.DataFrame({'a': [1.0, 2.0]}, index=[0, 10])
        new_idx = [0, 1, 2, 3, 10]
        result = df.reindex(new_idx, method='ffill', limit=2, tolerance=5)
        # 1: within tolerance and within limit → filled
        assert result.loc[1, 'a'] == 1.0
        # 2: within tolerance and within limit → filled
        assert result.loc[2, 'a'] == 1.0
        # 3: within tolerance but exceeds limit=2 → NaN
        assert pd.isna(result.loc[3, 'a'])
