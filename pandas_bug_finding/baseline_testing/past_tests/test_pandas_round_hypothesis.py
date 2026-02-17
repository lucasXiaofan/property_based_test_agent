"""
Property-based tests for pandas DataFrame.round() using Hypothesis.

This test suite aims to find bugs in the pandas DataFrame.round() method by
exploring edge cases and various input combinations.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes


# Strategy for decimal places (reasonable range)
decimals_strategy = st.integers(min_value=-10, max_value=10)

# Strategy for column names
column_names_strategy = st.lists(
    st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    min_size=1,
    max_size=5,
    unique=True
)

# Strategy for numeric dataframes
@st.composite
def numeric_dataframe(draw):
    """Generate a DataFrame with numeric columns."""
    col_names = draw(column_names_strategy)
    num_rows = draw(st.integers(min_value=1, max_value=20))

    data = {}
    for col in col_names:
        dtype = draw(st.sampled_from(['float64', 'int64', 'float32', 'int32']))
        if 'float' in dtype:
            data[col] = draw(st.lists(
                st.floats(min_value=-1e10, max_value=1e10, allow_nan=True, allow_infinity=True),
                min_size=num_rows,
                max_size=num_rows
            ))
        else:
            data[col] = draw(st.lists(
                st.integers(min_value=-1000000, max_value=1000000),
                min_size=num_rows,
                max_size=num_rows
            ))

    return pd.DataFrame(data)


class TestDataFrameRoundInt:
    """Test DataFrame.round() with integer decimals parameter."""

    @given(df=numeric_dataframe(), decimals=decimals_strategy)
    @settings(max_examples=100, deadline=None)
    def test_round_int_returns_dataframe(self, df, decimals):
        """Test that round with int decimals returns a DataFrame."""
        result = df.round(decimals)
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframe(), decimals=decimals_strategy)
    @settings(max_examples=100, deadline=None)
    def test_round_int_preserves_shape(self, df, decimals):
        """Test that rounding preserves the DataFrame shape."""
        result = df.round(decimals)
        assert result.shape == df.shape

    @given(df=numeric_dataframe(), decimals=decimals_strategy)
    @settings(max_examples=100, deadline=None)
    def test_round_int_preserves_columns(self, df, decimals):
        """Test that rounding preserves column names and order."""
        result = df.round(decimals)
        assert list(result.columns) == list(df.columns)

    @given(df=numeric_dataframe(), decimals=decimals_strategy)
    @settings(max_examples=100, deadline=None)
    def test_round_int_preserves_index(self, df, decimals):
        """Test that rounding preserves the index."""
        result = df.round(decimals)
        assert result.index.equals(df.index)

    @given(df=numeric_dataframe())
    @settings(max_examples=50, deadline=None)
    def test_round_zero_decimals_removes_fractional_part(self, df):
        """Test that round(0) removes fractional parts for float columns."""
        result = df.round(0)

        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                # Check that non-NaN values have no fractional part
                non_nan_result = result[col].dropna()
                for val in non_nan_result:
                    if not np.isinf(val):
                        assert val == np.floor(val) or val == np.ceil(val)

    @given(df=numeric_dataframe(), decimals=st.integers(min_value=1, max_value=5))
    @settings(max_examples=50, deadline=None)
    def test_round_idempotent(self, df, decimals):
        """Test that rounding twice gives the same result as rounding once."""
        result1 = df.round(decimals)
        result2 = result1.round(decimals)

        # Should be equal (allowing for floating point precision)
        pd.testing.assert_frame_equal(result1, result2)


class TestDataFrameRoundDict:
    """Test DataFrame.round() with dict decimals parameter."""

    @given(df=numeric_dataframe())
    @settings(max_examples=50, deadline=None)
    def test_round_dict_subset_columns(self, df):
        """Test rounding with dict specifying only some columns."""
        # Select a random subset of columns
        if len(df.columns) > 1:
            cols_to_round = list(df.columns[:len(df.columns)//2])
        else:
            cols_to_round = list(df.columns)

        decimals_dict = {col: 2 for col in cols_to_round}
        result = df.round(decimals_dict)

        # Columns not in dict should be unchanged
        for col in df.columns:
            if col not in decimals_dict:
                pd.testing.assert_series_equal(result[col], df[col])

    @given(df=numeric_dataframe())
    @settings(max_examples=50, deadline=None)
    def test_round_dict_ignores_nonexistent_columns(self, df):
        """Test that dict entries for non-existent columns are ignored."""
        decimals_dict = {'nonexistent_col': 2, 'another_fake_col': 3}

        # Add one real column if it exists
        if len(df.columns) > 0:
            decimals_dict[df.columns[0]] = 1

        # Should not raise an error
        result = df.round(decimals_dict)
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframe())
    @settings(max_examples=50, deadline=None)
    def test_round_dict_different_decimals_per_column(self, df):
        """Test that different columns can have different decimal places."""
        decimals_dict = {col: i for i, col in enumerate(df.columns)}
        result = df.round(decimals_dict)

        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)


class TestDataFrameRoundSeries:
    """Test DataFrame.round() with Series decimals parameter."""

    @given(df=numeric_dataframe())
    @settings(max_examples=50, deadline=None)
    def test_round_series_basic(self, df):
        """Test rounding with Series decimals parameter."""
        decimals_series = pd.Series({col: 2 for col in df.columns})
        result = df.round(decimals_series)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    @given(df=numeric_dataframe())
    @settings(max_examples=50, deadline=None)
    def test_round_series_subset_columns(self, df):
        """Test rounding with Series specifying only some columns."""
        if len(df.columns) > 1:
            cols_to_round = list(df.columns[:len(df.columns)//2])
        else:
            cols_to_round = list(df.columns)

        decimals_series = pd.Series({col: 2 for col in cols_to_round})
        result = df.round(decimals_series)

        # Columns not in series should be unchanged
        for col in df.columns:
            if col not in cols_to_round:
                pd.testing.assert_series_equal(result[col], df[col])

    @given(df=numeric_dataframe())
    @settings(max_examples=50, deadline=None)
    def test_round_series_ignores_nonexistent_columns(self, df):
        """Test that Series entries for non-existent columns are ignored."""
        decimals_series = pd.Series({'nonexistent_col': 2, 'another_fake_col': 3})

        # Add one real column if it exists
        if len(df.columns) > 0:
            decimals_series[df.columns[0]] = 1

        # Should not raise an error
        result = df.round(decimals_series)
        assert isinstance(result, pd.DataFrame)


class TestDataFrameRoundEdgeCases:
    """Test edge cases and potential bug scenarios."""

    def test_round_empty_dataframe(self):
        """Test rounding an empty DataFrame."""
        df = pd.DataFrame()
        result = df.round(2)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (0, 0)

    def test_round_with_nan_values(self):
        """Test rounding with NaN values."""
        df = pd.DataFrame({'a': [1.234, np.nan, 3.456], 'b': [np.nan, 2.345, np.nan]})
        result = df.round(2)

        # NaN values should remain NaN
        assert pd.isna(result.loc[1, 'a'])
        assert pd.isna(result.loc[0, 'b'])
        assert pd.isna(result.loc[2, 'b'])

    def test_round_with_inf_values(self):
        """Test rounding with infinity values."""
        df = pd.DataFrame({'a': [1.234, np.inf, -np.inf], 'b': [np.inf, 2.345, -np.inf]})
        result = df.round(2)

        # Infinity values should remain infinity
        assert np.isinf(result.loc[1, 'a']) and result.loc[1, 'a'] > 0
        assert np.isinf(result.loc[2, 'a']) and result.loc[2, 'a'] < 0
        assert np.isinf(result.loc[0, 'b']) and result.loc[0, 'b'] > 0

    def test_round_negative_decimals(self):
        """Test rounding with negative decimal places."""
        df = pd.DataFrame({'a': [123.456, 234.567], 'b': [345.678, 456.789]})
        result = df.round(-1)

        # Should round to nearest 10
        assert isinstance(result, pd.DataFrame)

    def test_round_very_large_decimals(self):
        """Test rounding with very large decimal parameter."""
        df = pd.DataFrame({'a': [1.234567890123456], 'b': [2.345678901234567]})
        result = df.round(15)

        assert isinstance(result, pd.DataFrame)

    def test_round_mixed_dtypes(self):
        """Test rounding DataFrame with mixed numeric and non-numeric types."""
        df = pd.DataFrame({
            'float_col': [1.234, 2.345, 3.456],
            'int_col': [1, 2, 3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })

        result = df.round(2)

        # Non-numeric columns should be unchanged
        pd.testing.assert_series_equal(result['str_col'], df['str_col'])
        pd.testing.assert_series_equal(result['bool_col'], df['bool_col'])

    def test_round_with_multiindex_columns(self):
        """Test rounding with MultiIndex columns."""
        df = pd.DataFrame(
            [[1.234, 2.345], [3.456, 4.567]],
            columns=pd.MultiIndex.from_tuples([('A', 'a'), ('B', 'b')])
        )
        result = df.round(2)

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.columns, pd.MultiIndex)

    @given(df=numeric_dataframe())
    @settings(max_examples=50, deadline=None)
    def test_round_does_not_modify_original(self, df):
        """Test that round() does not modify the original DataFrame."""
        df_copy = df.copy()
        _ = df.round(2)

        pd.testing.assert_frame_equal(df, df_copy)

    def test_round_args_kwargs_compatibility(self):
        """Test that *args and **kwargs are accepted for numpy compatibility."""
        df = pd.DataFrame({'a': [1.234, 2.345], 'b': [3.456, 4.567]})

        # These should not raise errors
        result1 = df.round(2, 'extra_arg')
        result2 = df.round(2, extra_kwarg='value')

        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)

    def test_round_with_duplicate_column_names(self):
        """Test rounding with duplicate column names."""
        df = pd.DataFrame([[1.234, 2.345], [3.456, 4.567]], columns=['a', 'a'])
        result = df.round(2)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    @given(
        decimals=st.one_of(
            st.none(),
            st.floats(min_value=-100, max_value=100, allow_nan=True, allow_infinity=True)
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_round_invalid_decimals_type(self, decimals):
        """Test that invalid decimals types raise appropriate errors."""
        df = pd.DataFrame({'a': [1.234, 2.345], 'b': [3.456, 4.567]})

        # None, NaN, or Inf decimals should raise TypeError or ValueError
        if decimals is None or (isinstance(decimals, float) and (np.isnan(decimals) or np.isinf(decimals))):
            with pytest.raises((TypeError, ValueError)):
                df.round(decimals)


class TestDataFrameRoundInvariants:
    """Test mathematical invariants and properties of rounding."""

    @given(df=numeric_dataframe(), decimals=st.integers(min_value=0, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_round_bounds(self, df, decimals):
        """Test that rounded values are bounded by floor and ceiling."""
        result = df.round(decimals)

        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                for orig, rounded in zip(df[col], result[col]):
                    if pd.notna(orig) and pd.notna(rounded) and np.isfinite(orig) and np.isfinite(rounded):
                        # Rounded value should be close to original
                        assert abs(rounded - orig) <= 0.5 * (10 ** -decimals) + 1e-10

    @given(df=numeric_dataframe())
    @settings(max_examples=30, deadline=None)
    def test_round_consistency_int_vs_dict(self, df):
        """Test that round(n) is equivalent to round({col: n for col in cols})."""
        decimals = 2
        result_int = df.round(decimals)
        result_dict = df.round({col: decimals for col in df.columns})

        pd.testing.assert_frame_equal(result_int, result_dict)

    @given(df=numeric_dataframe())
    @settings(max_examples=30, deadline=None)
    def test_round_consistency_dict_vs_series(self, df):
        """Test that dict and Series decimals produce the same result."""
        decimals_dict = {col: 2 for col in df.columns}
        decimals_series = pd.Series(decimals_dict)

        result_dict = df.round(decimals_dict)
        result_series = df.round(decimals_series)

        pd.testing.assert_frame_equal(result_dict, result_series)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
