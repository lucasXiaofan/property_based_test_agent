"""
Property-Based Testing Script for pandas.DataFrame.round()
===========================================================

This script comprehensively tests the pandas DataFrame.round() method using
property-based testing with Hypothesis. It aims to discover bugs by testing both
explicit properties claimed in the docstring and implicit properties that are
expected but not documented.

EXPLICIT PROPERTIES (from docstring):
1. Returns a DataFrame
2. Accepts int, dict, or Series for decimals parameter
3. Int rounds all columns to same number of places
4. Dict/Series round to variable numbers of places
5. Column names should be in keys (dict) or index (Series)
6. Columns not in decimals are left as is
7. Elements of decimals not in columns are ignored
8. *args and **kwargs accepted for numpy compatibility (no effect)
9. Only affects specified columns

IMPLICIT PROPERTIES (not explicitly documented but expected):
1. Non-numeric columns are left unchanged (no errors)
2. Returns new DataFrame (immutability - doesn't modify original)
3. Preserves DataFrame shape (rows, columns)
4. Preserves index and column names
5. Handles NaN values correctly (NaN remains NaN)
6. Handles infinity values correctly (inf remains inf)
7. Negative decimals round to powers of 10 (numpy-compatible)
8. Works with nullable dtypes (Int64, Float64, pd.NA)
9. Handles empty DataFrames correctly
10. Handles MultiIndex columns correctly
11. Handles duplicate column names correctly
12. Idempotent for integer decimals (round(round(x)) == round(x))

PANDAS VERSION:
This script was developed using pandas 2.2.3 documentation from Context7.

HOW TO RUN:
-----------
Using uv:
    uv run pytest pandas_bug_finding/baseline_testing/pandas_round_all_properties_testing_script.py -v

With specific settings:
    uv run pytest pandas_bug_finding/baseline_testing/pandas_round_all_properties_testing_script.py -v --hypothesis-show-statistics

For verbose output:
    uv run pytest pandas_bug_finding/baseline_testing/pandas_round_all_properties_testing_script.py -vv -s

Run specific test class:
    uv run pytest pandas_bug_finding/baseline_testing/pandas_round_all_properties_testing_script.py::TestExplicitProperties -v

"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, assume, example
from hypothesis.extra.pandas import column, data_frames, range_indexes
from typing import Union
import copy


# ==============================================================================
# HYPOTHESIS STRATEGIES
# ==============================================================================

@st.composite
def numeric_dataframes(draw, min_rows=0, max_rows=20, min_cols=1, max_cols=10):
    """Generate DataFrames with numeric columns including edge cases."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    # Generate column names (may include duplicates)
    allow_duplicates = draw(st.booleans())
    if allow_duplicates:
        col_names = draw(st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=5),
            min_size=n_cols, max_size=n_cols
        ))
    else:
        col_names = [f"col_{i}" for i in range(n_cols)]

    # Generate numeric dtypes for each column
    dtypes = draw(st.lists(
        st.sampled_from([np.float64, np.float32, np.int64, np.int32]),
        min_size=n_cols, max_size=n_cols
    ))

    # Build DataFrame column by column
    data = {}
    for i, (col_name, dtype) in enumerate(zip(col_names, dtypes)):
        if np.issubdtype(dtype, np.floating):
            # Include special values: NaN, inf, -inf
            values = draw(st.lists(
                st.one_of(
                    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
                    st.just(np.nan),
                    st.just(np.inf),
                    st.just(-np.inf)
                ),
                min_size=n_rows, max_size=n_rows
            ))
        else:
            values = draw(st.lists(
                st.integers(min_value=-1000000, max_value=1000000),
                min_size=n_rows, max_size=n_rows
            ))

        # Handle duplicate column names by appending to existing list
        if col_name in data:
            # For duplicate columns, create unique key
            key = f"{col_name}_{i}"
            data[key] = values
        else:
            data[col_name] = values

    df = pd.DataFrame(data)

    # Rename columns back if we used unique keys
    if allow_duplicates:
        df.columns = col_names

    return df


@st.composite
def mixed_dataframes(draw):
    """Generate DataFrames with mixed numeric and non-numeric columns."""
    n_rows = draw(st.integers(min_value=1, max_value=10))
    n_numeric_cols = draw(st.integers(min_value=1, max_value=5))
    n_nonnumeric_cols = draw(st.integers(min_value=1, max_value=3))

    data = {}

    # Add numeric columns
    for i in range(n_numeric_cols):
        data[f"num_{i}"] = draw(st.lists(
            st.floats(allow_nan=True, allow_infinity=True, min_value=-1e6, max_value=1e6),
            min_size=n_rows, max_size=n_rows
        ))

    # Add non-numeric columns
    for i in range(n_nonnumeric_cols):
        col_type = draw(st.sampled_from(['string', 'datetime', 'boolean']))
        if col_type == 'string':
            data[f"str_{i}"] = draw(st.lists(
                st.text(min_size=1, max_size=10),
                min_size=n_rows, max_size=n_rows
            ))
        elif col_type == 'datetime':
            data[f"dt_{i}"] = pd.date_range('2020-01-01', periods=n_rows)
        else:
            data[f"bool_{i}"] = draw(st.lists(
                st.booleans(),
                min_size=n_rows, max_size=n_rows
            ))

    return pd.DataFrame(data)


@st.composite
def nullable_dtype_dataframes(draw):
    """Generate DataFrames with nullable dtypes (Int64, Float64)."""
    n_rows = draw(st.integers(min_value=1, max_value=15))
    n_cols = draw(st.integers(min_value=1, max_value=5))

    data = {}
    for i in range(n_cols):
        dtype = draw(st.sampled_from(['Int64', 'Float64']))
        if dtype == 'Int64':
            values = draw(st.lists(
                st.one_of(st.integers(min_value=-1000, max_value=1000), st.just(pd.NA)),
                min_size=n_rows, max_size=n_rows
            ))
        else:  # Float64
            values = draw(st.lists(
                st.one_of(
                    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                    st.just(pd.NA)
                ),
                min_size=n_rows, max_size=n_rows
            ))
        data[f"col_{i}"] = pd.array(values, dtype=dtype)

    return pd.DataFrame(data)


@st.composite
def decimal_places(draw, max_value=10):
    """Generate decimal places including negative values."""
    return draw(st.integers(min_value=-5, max_value=max_value))


# ==============================================================================
# TEST CLASS 1: EXPLICIT PROPERTIES FROM DOCSTRING
# ==============================================================================

class TestExplicitProperties:
    """Test properties explicitly claimed in the DataFrame.round() docstring."""

    @given(df=numeric_dataframes(), decimals=st.integers(min_value=0, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_returns_dataframe(self, df, decimals):
        """Property: round() returns a DataFrame."""
        result = df.round(decimals)
        assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result)}"

    @given(df=numeric_dataframes(min_cols=2, max_cols=5))
    @settings(max_examples=50, deadline=None)
    def test_accepts_int_parameter(self, df):
        """Property: round() accepts int for decimals parameter."""
        decimals = 2
        result = df.round(decimals)
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframes(min_cols=2, max_cols=5))
    @settings(max_examples=50, deadline=None)
    def test_accepts_dict_parameter(self, df):
        """Property: round() accepts dict for decimals parameter."""
        # Create dict with subset of columns
        decimals_dict = {col: 1 for col in df.columns[:2]}
        result = df.round(decimals_dict)
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframes(min_cols=2, max_cols=5))
    @settings(max_examples=50, deadline=None)
    def test_accepts_series_parameter(self, df):
        """Property: round() accepts Series for decimals parameter."""
        # Create Series with column names as index
        decimals_series = pd.Series([1, 2], index=list(df.columns[:2]))
        result = df.round(decimals_series)
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframes(min_cols=3, max_cols=5), decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=50, deadline=None)
    def test_int_rounds_all_columns_uniformly(self, df, decimals):
        """Property: int parameter rounds all numeric columns to same precision."""
        result = df.round(decimals)

        # For each numeric column, check that values are rounded to same precision
        for col in df.select_dtypes(include=[np.number]).columns:
            for i in range(len(df)):
                original = df[col].iloc[i]
                rounded = result[col].iloc[i]

                if pd.isna(original):
                    assert pd.isna(rounded)
                elif np.isinf(original):
                    assert rounded == original
                else:
                    expected = round(float(original), decimals)
                    if not np.isclose(rounded, expected, rtol=1e-9, atol=1e-9):
                        # Allow for floating point precision issues
                        assert abs(rounded - expected) < 10 ** (-(decimals + 1))

    @given(df=numeric_dataframes(min_cols=3, max_cols=5))
    @settings(max_examples=50, deadline=None)
    def test_dict_rounds_specified_columns_variably(self, df):
        """Property: dict parameter rounds different columns to different precisions."""
        # Select subset of columns with different decimal places
        col_list = list(df.select_dtypes(include=[np.number]).columns)
        if len(col_list) < 2:
            assume(False)

        decimals_dict = {col_list[0]: 0, col_list[1]: 3}
        result = df.round(decimals_dict)

        # Check first column rounded to 0 decimals
        for i in range(len(df)):
            original = df[col_list[0]].iloc[i]
            rounded = result[col_list[0]].iloc[i]
            if not pd.isna(original) and not np.isinf(original):
                expected = round(float(original), 0)
                assert np.isclose(rounded, expected, rtol=1e-9, atol=1e-9)

    @given(df=numeric_dataframes(min_cols=3, max_cols=5))
    @settings(max_examples=30, deadline=None)
    def test_columns_not_in_decimals_unchanged(self, df):
        """Property: columns not in decimals dict/Series are left as is."""
        col_list = list(df.select_dtypes(include=[np.number]).columns)
        if len(col_list) < 2:
            assume(False)

        # Round only first column
        decimals_dict = {col_list[0]: 2}
        result = df.round(decimals_dict)

        # Other columns should be unchanged
        for col in col_list[1:]:
            pd.testing.assert_series_equal(result[col], df[col], check_names=True)

    @given(df=numeric_dataframes(min_cols=2, max_cols=4))
    @settings(max_examples=30, deadline=None)
    def test_ignores_decimals_for_nonexistent_columns(self, df):
        """Property: elements of decimals not in DataFrame columns are ignored."""
        # Include a column name that doesn't exist
        decimals_dict = {'nonexistent_column': 5, list(df.columns)[0]: 2}
        result = df.round(decimals_dict)

        # Should not raise error, just ignore the nonexistent column
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframes(), decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_accepts_args_kwargs_for_numpy_compatibility(self, df, decimals):
        """Property: *args and **kwargs accepted but have no effect."""
        result1 = df.round(decimals)
        result2 = df.round(decimals, extra_arg=True)
        result3 = df.round(decimals, out=None)

        # Results should be identical regardless of extra args
        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result1, result3)


# ==============================================================================
# TEST CLASS 2: IMPLICIT PROPERTIES - BASIC BEHAVIOR
# ==============================================================================

class TestImplicitBasicBehavior:
    """Test implicit properties about basic behavior not explicitly documented."""

    @given(df=mixed_dataframes(), decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=50, deadline=None)
    def test_nonnumeric_columns_unchanged(self, df, decimals):
        """Implicit Property 1: Non-numeric columns are left unchanged (no errors)."""
        result = df.round(decimals)

        # Non-numeric columns should be identical
        for col in df.select_dtypes(exclude=[np.number]).columns:
            pd.testing.assert_series_equal(result[col], df[col], check_names=True)

    @given(df=numeric_dataframes(), decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=50, deadline=None)
    def test_immutability_returns_new_dataframe(self, df, decimals):
        """Implicit Property 2: Returns new DataFrame (doesn't modify original)."""
        original_copy = df.copy()
        result = df.round(decimals)

        # Original DataFrame should be unchanged
        pd.testing.assert_frame_equal(df, original_copy)

        # Result should be a different object
        assert result is not df

    @given(df=numeric_dataframes(), decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=50, deadline=None)
    def test_preserves_dataframe_shape(self, df, decimals):
        """Implicit Property 3: Preserves DataFrame shape (rows, columns)."""
        result = df.round(decimals)
        assert result.shape == df.shape

    @given(df=numeric_dataframes(), decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=50, deadline=None)
    def test_preserves_index_and_columns(self, df, decimals):
        """Implicit Property 4: Preserves index and column names."""
        result = df.round(decimals)

        pd.testing.assert_index_equal(result.index, df.index)
        pd.testing.assert_index_equal(result.columns, df.columns)

    @given(decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_handles_nan_correctly(self, decimals):
        """Implicit Property 5: NaN values remain NaN after rounding."""
        df = pd.DataFrame({
            'A': [1.23, np.nan, 4.56],
            'B': [np.nan, np.nan, 7.89]
        })
        result = df.round(decimals)

        # NaN should stay NaN
        assert pd.isna(result['A'].iloc[1])
        assert pd.isna(result['B'].iloc[0])
        assert pd.isna(result['B'].iloc[1])

    @given(decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_handles_infinity_correctly(self, decimals):
        """Implicit Property 6: Infinity values remain infinity after rounding."""
        df = pd.DataFrame({
            'A': [1.23, np.inf, -np.inf],
            'B': [np.inf, -np.inf, 4.56]
        })
        result = df.round(decimals)

        # Infinity should stay infinity
        assert result['A'].iloc[1] == np.inf
        assert result['A'].iloc[2] == -np.inf
        assert result['B'].iloc[0] == np.inf
        assert result['B'].iloc[1] == -np.inf


# ==============================================================================
# TEST CLASS 3: IMPLICIT PROPERTIES - ADVANCED BEHAVIOR
# ==============================================================================

class TestImplicitAdvancedBehavior:
    """Test advanced implicit properties and edge cases."""

    @given(df=numeric_dataframes(min_cols=1, max_cols=5))
    @settings(max_examples=40, deadline=None)
    def test_negative_decimals_round_to_powers_of_10(self, df):
        """Implicit Property 7: Negative decimals round to powers of 10 (numpy-like)."""
        negative_decimals = -1
        result = df.round(negative_decimals)

        # With decimals=-1, should round to nearest 10
        for col in df.select_dtypes(include=[np.number]).columns:
            for i in range(len(df)):
                original = df[col].iloc[i]
                rounded = result[col].iloc[i]

                if pd.isna(original) or np.isinf(original):
                    continue

                # Should be divisible by 10
                expected = round(float(original), negative_decimals)
                assert np.isclose(rounded, expected, rtol=1e-9, atol=1e-9)

    @given(df=nullable_dtype_dataframes(), decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=40, deadline=None)
    def test_works_with_nullable_dtypes(self, df, decimals):
        """Implicit Property 8: Works with nullable dtypes (Int64, Float64, pd.NA)."""
        result = df.round(decimals)

        # Should return DataFrame without errors
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

        # pd.NA should remain pd.NA
        for col in df.columns:
            for i in range(len(df)):
                if pd.isna(df[col].iloc[i]):
                    assert pd.isna(result[col].iloc[i])

    @given(decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_handles_empty_dataframe(self, decimals):
        """Implicit Property 9: Handles empty DataFrames correctly."""
        # Empty with columns
        df1 = pd.DataFrame(columns=['A', 'B', 'C'])
        result1 = df1.round(decimals)
        assert result1.shape == (0, 3)
        assert list(result1.columns) == ['A', 'B', 'C']

        # Completely empty
        df2 = pd.DataFrame()
        result2 = df2.round(decimals)
        assert result2.shape == (0, 0)

    @given(decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_handles_multiindex_columns(self, decimals):
        """Implicit Property 10: Handles MultiIndex columns correctly."""
        arrays = [
            ['A', 'A', 'B', 'B'],
            ['one', 'two', 'one', 'two']
        ]
        columns = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])

        df = pd.DataFrame(np.random.randn(5, 4), columns=columns)
        result = df.round(decimals)

        # Should preserve MultiIndex structure
        assert isinstance(result.columns, pd.MultiIndex)
        pd.testing.assert_index_equal(result.columns, df.columns)

    @given(decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_handles_duplicate_column_names(self, decimals):
        """Implicit Property 11: Handles duplicate column names correctly."""
        # Create DataFrame with duplicate column names
        df = pd.DataFrame({
            'A': [1.234, 5.678],
            'B': [2.345, 6.789]
        })
        df.columns = ['A', 'A']  # Duplicate names

        result = df.round(decimals)

        # Should still work and preserve structure
        assert result.shape == df.shape
        assert list(result.columns) == ['A', 'A']

    @given(df=numeric_dataframes(min_cols=2, max_cols=5), decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=50, deadline=None)
    def test_idempotent_for_integer_decimals(self, df, decimals):
        """Implicit Property 12: Idempotent - round(round(x)) == round(x)."""
        result1 = df.round(decimals)
        result2 = result1.round(decimals)

        # Rounding twice should give same result as rounding once
        pd.testing.assert_frame_equal(result1, result2, check_dtype=False, atol=1e-10)


# ==============================================================================
# TEST CLASS 4: EDGE CASES AND BOUNDARY CONDITIONS
# ==============================================================================

class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    @given(decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=20, deadline=None)
    def test_very_large_numbers(self, decimals):
        """Test rounding with very large numbers."""
        df = pd.DataFrame({
            'A': [1.23e100, 4.56e200, 7.89e-100],
            'B': [9.99e150, 1.11e-150, 2.22e50]
        })
        result = df.round(decimals)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    @given(decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=20, deadline=None)
    def test_very_small_numbers(self, decimals):
        """Test rounding with very small numbers near zero."""
        df = pd.DataFrame({
            'A': [1e-10, 1e-15, 1e-20],
            'B': [-1e-10, -1e-15, -1e-20]
        })
        result = df.round(decimals)

        # Very small numbers should round to zero with limited decimals
        assert isinstance(result, pd.DataFrame)

    def test_maximum_decimal_places(self):
        """Test with very large decimal places value."""
        df = pd.DataFrame({'A': [1.123456789012345], 'B': [9.987654321098765]})
        result = df.round(15)

        assert isinstance(result, pd.DataFrame)

    @given(decimals=st.integers(min_value=-10, max_value=-1))
    @settings(max_examples=20, deadline=None)
    def test_large_negative_decimals(self, decimals):
        """Test with large negative decimal values."""
        df = pd.DataFrame({'A': [123456.789, 987654.321], 'B': [555555.555, 111111.111]})
        result = df.round(decimals)

        assert isinstance(result, pd.DataFrame)

    def test_single_row_dataframe(self):
        """Test with single-row DataFrame."""
        df = pd.DataFrame({'A': [1.234], 'B': [5.678], 'C': [9.012]})
        result = df.round(2)

        assert result.shape == (1, 3)
        assert np.isclose(result['A'].iloc[0], 1.23)

    def test_single_column_dataframe(self):
        """Test with single-column DataFrame."""
        df = pd.DataFrame({'A': [1.234, 5.678, 9.012]})
        result = df.round(1)

        assert result.shape == (3, 1)
        assert np.isclose(result['A'].iloc[0], 1.2)

    def test_all_nan_dataframe(self):
        """Test with DataFrame containing only NaN values."""
        df = pd.DataFrame({'A': [np.nan, np.nan], 'B': [np.nan, np.nan]})
        result = df.round(2)

        assert result.shape == (2, 2)
        assert result.isna().all().all()

    def test_all_infinity_dataframe(self):
        """Test with DataFrame containing only infinity values."""
        df = pd.DataFrame({
            'A': [np.inf, -np.inf, np.inf],
            'B': [-np.inf, np.inf, -np.inf]
        })
        result = df.round(2)

        assert result.shape == (3, 2)
        # Infinities should remain
        assert result['A'].iloc[0] == np.inf
        assert result['A'].iloc[1] == -np.inf


# ==============================================================================
# TEST CLASS 5: PARAMETER FORMAT CONSISTENCY
# ==============================================================================

class TestParameterFormatConsistency:
    """Test that different parameter formats produce consistent results."""

    @given(df=numeric_dataframes(min_cols=3, max_cols=5), decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=40, deadline=None)
    def test_int_vs_dict_consistency(self, df, decimals):
        """Test that int and equivalent dict produce same results."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            assume(False)

        # Round with int
        result_int = df.round(decimals)

        # Round with dict (all columns same decimals)
        decimals_dict = {col: decimals for col in numeric_cols}
        result_dict = df.round(decimals_dict)

        # Results should be identical
        pd.testing.assert_frame_equal(result_int, result_dict, check_dtype=False)

    @given(df=numeric_dataframes(min_cols=3, max_cols=5), decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=40, deadline=None)
    def test_dict_vs_series_consistency(self, df, decimals):
        """Test that dict and Series with same values produce same results."""
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns[:3])
        if len(numeric_cols) < 2:
            assume(False)

        # Round with dict
        decimals_dict = {col: decimals for col in numeric_cols}
        result_dict = df.round(decimals_dict)

        # Round with Series
        decimals_series = pd.Series([decimals] * len(numeric_cols), index=numeric_cols)
        result_series = df.round(decimals_series)

        # Results should be identical
        pd.testing.assert_frame_equal(result_dict, result_series, check_dtype=False)

    @given(df=numeric_dataframes(min_cols=2, max_cols=4))
    @settings(max_examples=30, deadline=None)
    def test_partial_dict_vs_full_dict(self, df):
        """Test that partial dict only affects specified columns."""
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) < 2:
            assume(False)

        # Round only first column
        decimals_partial = {numeric_cols[0]: 2}
        result_partial = df.round(decimals_partial)

        # First column should be rounded
        for i in range(len(df)):
            original = df[numeric_cols[0]].iloc[i]
            rounded = result_partial[numeric_cols[0]].iloc[i]
            if not pd.isna(original) and not np.isinf(original):
                expected = round(float(original), 2)
                assert np.isclose(rounded, expected, rtol=1e-9, atol=1e-9)

        # Other columns should be unchanged
        for col in numeric_cols[1:]:
            pd.testing.assert_series_equal(result_partial[col], df[col], check_names=True)


# ==============================================================================
# TEST CLASS 6: MATHEMATICAL PROPERTIES
# ==============================================================================

class TestMathematicalProperties:
    """Test mathematical properties and invariants."""

    @given(df=numeric_dataframes(), decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=50, deadline=None)
    def test_rounded_values_within_bounds(self, df, decimals):
        """Test that rounded values stay within reasonable bounds of original."""
        result = df.round(decimals)

        for col in df.select_dtypes(include=[np.number]).columns:
            for i in range(len(df)):
                original = df[col].iloc[i]
                rounded = result[col].iloc[i]

                if pd.isna(original) or np.isinf(original):
                    continue

                # Rounded value should be within 0.5 * 10^(-decimals) of original
                tolerance = 0.5 * (10 ** (-decimals))
                assert abs(rounded - original) <= tolerance + 1e-10

    @given(df=numeric_dataframes(), decimals=st.integers(min_value=1, max_value=5))
    @settings(max_examples=40, deadline=None)
    def test_rounding_reduces_precision(self, df, decimals):
        """Test that rounding to N decimals doesn't create values with >N decimals."""
        result = df.round(decimals)

        for col in result.select_dtypes(include=[np.number]).columns:
            for i in range(len(result)):
                value = result[col].iloc[i]

                if pd.isna(value) or np.isinf(value):
                    continue

                # Check that value has at most 'decimals' decimal places
                # (within floating point precision)
                scaled = value * (10 ** decimals)
                assert np.isclose(scaled, round(scaled), rtol=1e-9, atol=1e-9)

    def test_rounding_symmetry(self):
        """Test that rounding is symmetric around zero."""
        df = pd.DataFrame({
            'A': [1.235, -1.235, 2.675, -2.675],
            'B': [0.005, -0.005, 0.015, -0.015]
        })
        result = df.round(2)

        # Symmetric values should round symmetrically
        assert result['A'].iloc[0] == -result['A'].iloc[1]
        assert result['B'].iloc[0] == -result['B'].iloc[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
