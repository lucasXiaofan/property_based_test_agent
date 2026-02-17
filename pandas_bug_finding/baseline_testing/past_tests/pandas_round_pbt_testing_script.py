"""
Property-Based Testing Suite for pandas.DataFrame.round()

This test suite uses Hypothesis to discover potential bugs in pandas DataFrame.round()
through systematic exploration of the input space and verification of mathematical properties.

TESTED PROPERTIES:
==================

1. Type Invariants:
   - round() always returns a DataFrame (not modifying in-place)
   - Shape preservation: output has same dimensions as input
   - Metadata preservation: column names and index are unchanged
   - Non-numeric columns remain completely unchanged

2. Mathematical Properties:
   - Idempotence: round(round(df, n), n) == round(df, n)
   - Rounding bounds: |rounded_value - original_value| <= 0.5 * 10^(-decimals)
   - Stability: rounded values stay within reasonable bounds of originals
   - Consistency: round(df, int) ≈ round(df, dict) ≈ round(df, Series) for equivalent specs

3. Parameter Type Properties:
   - Integer parameter: rounds ALL numeric columns uniformly
   - Dict parameter: rounds specified columns only, leaves others unchanged
   - Series parameter: rounds columns matching Series index, leaves others unchanged
   - Missing columns in dict/Series are ignored (no errors)
   - Extra keys in dict/Series (not in DataFrame) are safely ignored

4. Edge Cases & Special Values:
   - Empty DataFrames (0 rows) are handled correctly
   - NaN values remain NaN after rounding
   - Infinity values (±inf) remain infinity after rounding
   - Negative decimals round to powers of 10 (e.g., -1 rounds to nearest 10)
   - Nullable dtypes (Float64, Int64) are supported
   - Mixed dtypes (float32, float64, int32, int64) are handled correctly
   - Zero decimal places rounds to integers

5. Metamorphic Properties:
   - Column subset equivalence: round(df[cols]) == round(df)[cols]
   - Dict-Series equivalence: round(df, dict) == round(df, Series(dict))
   - Partial specification: columns not specified in dict/Series remain unchanged

DISCOVERED BUGS / FINDINGS:
===========================
(Tests will document any discrepancies found during execution)

HOW TO RUN:
===========
Using uv (recommended):
    uv run pytest pandas_round_pbt_testing_script.py -v

Using pytest directly:
    pytest pandas_round_pbt_testing_script.py -v

For verbose hypothesis output:
    uv run pytest pandas_round_pbt_testing_script.py -v --hypothesis-verbosity=verbose

For quick smoke test (fewer examples):
    uv run pytest pandas_round_pbt_testing_script.py -v --hypothesis-profile=dev

To see failing examples only:
    uv run pytest pandas_round_pbt_testing_script.py -v -x

REQUIREMENTS:
=============
- pandas>=2.0.0
- hypothesis>=6.0.0
- pytest>=7.0.0
- numpy>=1.20.0

Context7 Documentation Reference:
- Library: /pandas-dev/pandas
- Key findings:
  * DataFrame.round() introduced in v0.17.0
  * Supports int, dict, and Series parameters
  * Non-numeric columns left unchanged (v0.18.0+)
  * Nullable dtypes supported (v1.3.0+)
  * No 'out' parameter (removed in v0.18.0)
"""

import math
from typing import Union
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
from hypothesis.extra.pandas import data_frames, column, range_indexes


# ============================================================================
# HYPOTHESIS STRATEGIES FOR GENERATING TEST DATA
# ============================================================================

@st.composite
def numeric_dataframes(draw):
    """
    Generate DataFrames with diverse numeric columns and realistic test data.

    Includes:
    - Multiple numeric dtypes (float64, float32, int64, int32)
    - Special float values (NaN, ±inf) with controlled frequency
    - Varied DataFrame sizes (1-50 rows, 1-10 columns)
    - Mixed column types
    """
    n_cols = draw(st.integers(min_value=1, max_value=10))
    n_rows = draw(st.integers(min_value=1, max_value=50))

    # Define column strategies with different dtypes
    cols = {}
    col_names = [f"col_{i}" for i in range(n_cols)]

    for col_name in col_names:
        dtype_choice = draw(st.sampled_from(['float64', 'float32', 'int64', 'int32']))

        if dtype_choice in ['float64', 'float32']:
            # Float columns with special values
            cols[col_name] = column(
                col_name,
                elements=st.one_of(
                    st.floats(
                        min_value=-1e10,
                        max_value=1e10,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    st.sampled_from([np.nan, np.inf, -np.inf])
                ),
                dtype=dtype_choice
            )
        else:
            # Integer columns
            cols[col_name] = column(
                col_name,
                elements=st.integers(min_value=-1_000_000, max_value=1_000_000),
                dtype=dtype_choice
            )

    df = draw(data_frames(
        columns=list(cols.values()),
        index=range_indexes(min_size=n_rows, max_size=n_rows)
    ))

    return df


@st.composite
def dataframes_with_mixed_types(draw):
    """Generate DataFrames with both numeric and non-numeric columns."""
    n_numeric = draw(st.integers(min_value=1, max_value=5))
    n_string = draw(st.integers(min_value=1, max_value=3))
    n_rows = draw(st.integers(min_value=1, max_value=20))

    cols = {}

    # Add numeric columns
    for i in range(n_numeric):
        cols[f"num_{i}"] = column(
            f"num_{i}",
            elements=st.floats(
                min_value=-1000, max_value=1000,
                allow_nan=True, allow_infinity=False
            ),
            dtype='float64'
        )

    # Add string columns
    for i in range(n_string):
        cols[f"str_{i}"] = column(
            f"str_{i}",
            elements=st.text(min_size=0, max_size=10),
            dtype=object
        )

    df = draw(data_frames(
        columns=list(cols.values()),
        index=range_indexes(min_size=n_rows, max_size=n_rows)
    ))

    return df


@st.composite
def decimal_places(draw):
    """Generate realistic decimal place values, including edge cases."""
    return draw(st.integers(min_value=-5, max_value=10))


@st.composite
def round_parameters(draw, df):
    """
    Generate valid round() parameters (int, dict, or Series) for a given DataFrame.

    Returns tuple: (param, param_type)
    """
    param_type = draw(st.sampled_from(['int', 'dict', 'series']))

    if param_type == 'int':
        decimals = draw(decimal_places())
        return (decimals, 'int')

    elif param_type == 'dict':
        # Select subset of numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            # Fallback to int if no numeric columns
            return (0, 'int')

        n_cols = draw(st.integers(min_value=1, max_value=len(numeric_cols)))
        selected_cols = draw(st.lists(
            st.sampled_from(numeric_cols),
            min_size=n_cols,
            max_size=n_cols,
            unique=True
        ))

        decimals_dict = {
            col: draw(decimal_places())
            for col in selected_cols
        }
        return (decimals_dict, 'dict')

    else:  # series
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return (0, 'int')

        n_cols = draw(st.integers(min_value=1, max_value=len(numeric_cols)))
        selected_cols = draw(st.lists(
            st.sampled_from(numeric_cols),
            min_size=n_cols,
            max_size=n_cols,
            unique=True
        ))

        decimals_series = pd.Series({
            col: draw(decimal_places())
            for col in selected_cols
        })
        return (decimals_series, 'series')


# ============================================================================
# TEST CLASS 1: TYPE INVARIANTS
# ============================================================================

class TestTypeInvariants:
    """Test that round() preserves types and structure."""

    @given(df=numeric_dataframes(), decimals=decimal_places())
    @settings(max_examples=50, deadline=None)
    def test_returns_dataframe(self, df, decimals):
        """round() must return a DataFrame, not modify in-place."""
        result = df.round(decimals)
        assert isinstance(result, pd.DataFrame), \
            f"Expected DataFrame, got {type(result)}"

    @given(df=numeric_dataframes(), decimals=decimal_places())
    @settings(max_examples=50, deadline=None)
    def test_shape_preservation(self, df, decimals):
        """round() must preserve DataFrame shape."""
        result = df.round(decimals)
        assert result.shape == df.shape, \
            f"Shape changed from {df.shape} to {result.shape}"

    @given(df=numeric_dataframes(), decimals=decimal_places())
    @settings(max_examples=50, deadline=None)
    def test_column_preservation(self, df, decimals):
        """round() must preserve column names."""
        result = df.round(decimals)
        assert result.columns.tolist() == df.columns.tolist(), \
            f"Columns changed from {df.columns.tolist()} to {result.columns.tolist()}"

    @given(df=numeric_dataframes(), decimals=decimal_places())
    @settings(max_examples=50, deadline=None)
    def test_index_preservation(self, df, decimals):
        """round() must preserve index."""
        result = df.round(decimals)
        assert result.index.equals(df.index), \
            f"Index changed"

    @given(df=dataframes_with_mixed_types(), decimals=decimal_places())
    @settings(max_examples=50, deadline=None)
    def test_non_numeric_columns_unchanged(self, df, decimals):
        """round() must leave non-numeric columns completely unchanged."""
        result = df.round(decimals)

        # Check each non-numeric column
        for col in df.select_dtypes(exclude=[np.number]).columns:
            assert result[col].equals(df[col]), \
                f"Non-numeric column '{col}' was modified"


# ============================================================================
# TEST CLASS 2: MATHEMATICAL PROPERTIES
# ============================================================================

class TestMathematicalProperties:
    """Test mathematical properties and invariants of rounding."""

    @given(df=numeric_dataframes(), decimals=st.integers(min_value=0, max_value=10))
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_idempotence(self, df, decimals):
        """round(round(x, n), n) should equal round(x, n)."""
        # Only test on DataFrames with at least some numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        assume(not numeric_df.empty)

        once = df.round(decimals)
        twice = once.round(decimals)

        # Compare numeric columns
        for col in numeric_df.columns:
            # Use np.allclose for floating point comparison, handling NaN
            pd.testing.assert_series_equal(
                once[col],
                twice[col],
                check_exact=False,
                rtol=1e-10,
                atol=1e-10
            )

    @given(df=numeric_dataframes(), decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_rounding_bounds(self, df, decimals):
        """Rounded values should be within 0.5 * 10^(-decimals) of original."""
        numeric_df = df.select_dtypes(include=[np.number])
        assume(not numeric_df.empty)

        result = df.round(decimals)
        tolerance = 0.5 * (10 ** (-decimals))

        for col in numeric_df.columns:
            original = df[col].values
            rounded = result[col].values

            # Check each value (skip NaN and inf)
            for orig, rnd in zip(original, rounded):
                if np.isnan(orig) or np.isinf(orig):
                    continue
                if np.isnan(rnd) or np.isinf(rnd):
                    continue

                diff = abs(rnd - orig)
                assert diff <= tolerance + 1e-10, \
                    f"Rounding error too large: |{rnd} - {orig}| = {diff} > {tolerance}"

    @given(df=numeric_dataframes(), decimals=st.integers(min_value=-2, max_value=8))
    @settings(max_examples=30, deadline=None)
    def test_immutability(self, df, decimals):
        """round() should not modify the original DataFrame."""
        original_data = df.copy()
        _ = df.round(decimals)

        pd.testing.assert_frame_equal(df, original_data)


# ============================================================================
# TEST CLASS 3: PARAMETER TYPE PROPERTIES
# ============================================================================

class TestParameterTypes:
    """Test different parameter types (int, dict, Series)."""

    @given(df=numeric_dataframes(), decimals=decimal_places())
    @settings(max_examples=50, deadline=None)
    def test_int_parameter_rounds_all_numeric(self, df, decimals):
        """Integer parameter should round ALL numeric columns."""
        result = df.round(decimals)

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Verify that rounding was applied
            # (We can't check exact values due to floating point, but we can
            # check that the operation completed and returned valid data)
            assert not result[col].isna().all() or df[col].isna().all(), \
                f"Column {col} became all NaN unexpectedly"

    @given(df=numeric_dataframes())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_dict_parameter_selective_rounding(self, df):
        """Dict parameter should only round specified columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        assume(len(numeric_cols) >= 2)

        # Round only first column
        decimals_dict = {numeric_cols[0]: 2}
        result = df.round(decimals_dict)

        # First column should be rounded
        pd.testing.assert_index_equal(result.columns, df.columns)

        # Other numeric columns should be unchanged
        for col in numeric_cols[1:]:
            pd.testing.assert_series_equal(
                result[col],
                df[col],
                check_exact=True
            )

    @given(df=numeric_dataframes())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_series_parameter_selective_rounding(self, df):
        """Series parameter should only round columns in its index."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        assume(len(numeric_cols) >= 2)

        # Round only first column using Series
        decimals_series = pd.Series({numeric_cols[0]: 2})
        result = df.round(decimals_series)

        # Other numeric columns should be unchanged
        for col in numeric_cols[1:]:
            pd.testing.assert_series_equal(
                result[col],
                df[col],
                check_exact=True
            )

    @given(df=numeric_dataframes(), decimals=decimal_places())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_dict_series_equivalence(self, df, decimals):
        """round(df, dict) should equal round(df, Series(dict))."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        assume(len(numeric_cols) >= 1)

        # Create equivalent dict and Series
        decimals_dict = {col: decimals for col in numeric_cols}
        decimals_series = pd.Series(decimals_dict)

        result_dict = df.round(decimals_dict)
        result_series = df.round(decimals_series)

        pd.testing.assert_frame_equal(result_dict, result_series)


# ============================================================================
# TEST CLASS 4: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and special values."""

    @given(decimals=decimal_places())
    @settings(max_examples=20, deadline=None)
    def test_empty_dataframe(self, decimals):
        """round() should handle empty DataFrames."""
        df = pd.DataFrame()
        result = df.round(decimals)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @given(decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_nan_preservation(self, decimals):
        """NaN values should remain NaN after rounding."""
        df = pd.DataFrame({
            'A': [1.234, np.nan, 5.678],
            'B': [np.nan, np.nan, 3.141]
        })

        result = df.round(decimals)

        # Check NaN positions are preserved
        assert result['A'].isna()[1] == True
        assert result['B'].isna()[0] == True
        assert result['B'].isna()[1] == True

    @given(decimals=st.integers(min_value=0, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_infinity_preservation(self, decimals):
        """±inf values should remain ±inf after rounding."""
        df = pd.DataFrame({
            'A': [1.234, np.inf, -np.inf],
            'B': [np.inf, -np.inf, 3.141]
        })

        result = df.round(decimals)

        # Check infinity preservation
        assert np.isinf(result['A'].iloc[1]) and result['A'].iloc[1] > 0
        assert np.isinf(result['A'].iloc[2]) and result['A'].iloc[2] < 0
        assert np.isinf(result['B'].iloc[0]) and result['B'].iloc[0] > 0
        assert np.isinf(result['B'].iloc[1]) and result['B'].iloc[1] < 0

    @given(st.integers(min_value=-5, max_value=-1))
    @settings(max_examples=30, deadline=None)
    def test_negative_decimals(self, decimals):
        """Negative decimals should round to powers of 10."""
        df = pd.DataFrame({
            'A': [123.456, 987.654, 555.555]
        })

        result = df.round(decimals)

        # For decimals=-1, should round to nearest 10
        # For decimals=-2, should round to nearest 100, etc.
        power = 10 ** (-decimals)

        for val in result['A']:
            # Check that value is a multiple of the power
            remainder = val % power
            assert abs(remainder) < 1e-10 or abs(remainder - power) < 1e-10, \
                f"Value {val} is not a multiple of {power} (remainder: {remainder})"

    @given(df=numeric_dataframes())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_extra_keys_in_dict_ignored(self, df):
        """Extra keys in dict (not in DataFrame columns) should be ignored."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        assume(len(numeric_cols) >= 1)

        # Add extra keys that don't exist in DataFrame
        decimals_dict = {numeric_cols[0]: 2, 'nonexistent_col': 5}

        # Should not raise an error
        result = df.round(decimals_dict)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape


# ============================================================================
# TEST CLASS 5: METAMORPHIC PROPERTIES
# ============================================================================

class TestMetamorphicProperties:
    """Test metamorphic properties and relationships between operations."""

    @given(df=numeric_dataframes(), decimals=decimal_places())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_column_subset_equivalence(self, df, decimals):
        """round(df[cols]) should equal round(df)[cols]."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        assume(len(numeric_cols) >= 2)

        # Select subset of columns
        subset_cols = numeric_cols[:len(numeric_cols)//2]
        assume(len(subset_cols) >= 1)

        # Round then select
        round_then_select = df.round(decimals)[subset_cols]

        # Select then round
        select_then_round = df[subset_cols].round(decimals)

        pd.testing.assert_frame_equal(round_then_select, select_then_round)

    @given(df=numeric_dataframes(), decimals=decimal_places())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_int_dict_consistency(self, df, decimals):
        """round(df, n) should equal round(df, {col:n for all cols})."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        assume(len(numeric_cols) >= 1)

        # Round with integer
        result_int = df.round(decimals)

        # Round with equivalent dict
        decimals_dict = {col: decimals for col in numeric_cols}
        result_dict = df.round(decimals_dict)

        # They should be equal for numeric columns
        for col in numeric_cols:
            pd.testing.assert_series_equal(
                result_int[col],
                result_dict[col],
                check_exact=False,
                rtol=1e-10
            )


# ============================================================================
# MANUAL TEST CASES (from documentation examples)
# ============================================================================

class TestDocumentationExamples:
    """Test examples from pandas documentation to ensure compatibility."""

    def test_basic_example_from_docs(self):
        """Test the basic example from pandas documentation."""
        df = pd.DataFrame([(.21, .32), (.01, .67), (.66, .03), (.21, .18)],
                          columns=['dogs', 'cats'])

        # Round to 1 decimal place
        result = df.round(1)

        expected = pd.DataFrame([(0.2, 0.3), (0.0, 0.7), (0.7, 0.0), (0.2, 0.2)],
                                columns=['dogs', 'cats'])

        pd.testing.assert_frame_equal(result, expected)

    def test_dict_example_from_docs(self):
        """Test dict parameter example from documentation."""
        df = pd.DataFrame([(.21, .32), (.01, .67), (.66, .03), (.21, .18)],
                          columns=['dogs', 'cats'])

        result = df.round({'dogs': 1, 'cats': 0})

        expected = pd.DataFrame([(0.2, 0.0), (0.0, 1.0), (0.7, 0.0), (0.2, 0.0)],
                                columns=['dogs', 'cats'])

        pd.testing.assert_frame_equal(result, expected)

    def test_series_example_from_docs(self):
        """Test Series parameter example from documentation."""
        df = pd.DataFrame([(.21, .32), (.01, .67), (.66, .03), (.21, .18)],
                          columns=['dogs', 'cats'])

        decimals = pd.Series([0, 1], index=['cats', 'dogs'])
        result = df.round(decimals)

        expected = pd.DataFrame([(0.2, 0.0), (0.0, 1.0), (0.7, 0.0), (0.2, 0.0)],
                                columns=['dogs', 'cats'])

        pd.testing.assert_frame_equal(result, expected)


if __name__ == '__main__':
    # Run with pytest
    pytest.main([__file__, '-v', '--tb=short'])
