"""
Property-based testing for pandas DataFrame.round() function.

This script uses Hypothesis to find potential bugs in pandas' round function by testing
implicit expectations and edge cases not covered in the docstring, including:
- Different numeric data types (int, float, complex, etc.)
- Edge cases (NaN, inf, very small/large numbers)
- Consistency with numpy.round behavior
- Mixed numeric and non-numeric columns
- Various decimals parameter formats
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.extra import pandas as pd_st, numpy as np_st
import warnings


# Custom strategies for generating diverse test data
@st.composite
def numeric_dtypes(draw):
    """Generate various numeric data types."""
    return draw(st.sampled_from([
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
        float, int
    ]))


@st.composite
def dataframe_with_numeric_columns(draw, min_cols=1, max_cols=5, min_rows=1, max_rows=10):
    """Generate DataFrames with numeric columns of various types."""
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    columns = [f'col_{i}' for i in range(n_cols)]
    data = {}

    for col in columns:
        dtype = draw(numeric_dtypes())
        if dtype in [float, np.float16, np.float32, np.float64]:
            # Include special float values
            values = draw(st.lists(
                st.one_of(
                    st.floats(min_value=-1e10, max_value=1e10, allow_nan=True, allow_infinity=True),
                    st.just(np.nan),
                    st.just(np.inf),
                    st.just(-np.inf),
                    st.floats(min_value=-1e-10, max_value=1e-10),  # Very small numbers
                ),
                min_size=n_rows,
                max_size=n_rows
            ))
        else:
            # Integer types
            if dtype in [np.int8, np.uint8]:
                min_val, max_val = -128, 127 if dtype == np.int8 else 0, 255
            elif dtype in [np.int16, np.uint16]:
                min_val, max_val = -32768, 32767 if dtype == np.int16 else 0, 65535
            else:
                min_val, max_val = -1000000, 1000000

            values = draw(st.lists(
                st.integers(min_value=min_val, max_value=max_val),
                min_size=n_rows,
                max_size=n_rows
            ))

        data[col] = values

    return pd.DataFrame(data)


@st.composite
def dataframe_with_mixed_types(draw, min_cols=2, max_cols=5):
    """Generate DataFrames with both numeric and non-numeric columns."""
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    n_rows = draw(st.integers(min_value=1, max_value=10))

    data = {}
    numeric_cols = []

    for i in range(n_cols):
        col_name = f'col_{i}'
        is_numeric = draw(st.booleans())

        if is_numeric:
            numeric_cols.append(col_name)
            values = draw(st.lists(
                st.floats(min_value=-1000, max_value=1000, allow_nan=True),
                min_size=n_rows,
                max_size=n_rows
            ))
        else:
            values = draw(st.lists(
                st.one_of(st.text(min_size=1, max_size=10), st.just(None)),
                min_size=n_rows,
                max_size=n_rows
            ))

        data[col_name] = values

    return pd.DataFrame(data), numeric_cols


@st.composite
def decimals_parameter(draw, columns):
    """Generate various forms of decimals parameter."""
    param_type = draw(st.sampled_from(['int', 'dict', 'series']))

    if param_type == 'int':
        return draw(st.integers(min_value=-10, max_value=10))
    elif param_type == 'dict':
        # Select subset of columns
        n_keys = draw(st.integers(min_value=0, max_value=len(columns)))
        selected_cols = draw(st.lists(
            st.sampled_from(columns),
            min_size=n_keys,
            max_size=n_keys,
            unique=True
        ))
        return {col: draw(st.integers(min_value=-5, max_value=5)) for col in selected_cols}
    else:  # series
        n_keys = draw(st.integers(min_value=1, max_value=min(len(columns), 3)))
        selected_cols = draw(st.lists(
            st.sampled_from(columns),
            min_size=n_keys,
            max_size=n_keys,
            unique=True
        ))
        values = [draw(st.integers(min_value=-5, max_value=5)) for _ in selected_cols]
        return pd.Series(values, index=selected_cols)


# Property-based tests

@given(df=dataframe_with_numeric_columns())
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_round_returns_dataframe(df):
    """Test that round always returns a DataFrame."""
    result = df.round()
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape


@given(df=dataframe_with_numeric_columns(), decimals=st.integers(min_value=0, max_value=10))
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_round_preserves_shape(df, decimals):
    """Test that rounding preserves DataFrame shape."""
    result = df.round(decimals)
    assert result.shape == df.shape
    assert list(result.columns) == list(df.columns)
    assert list(result.index) == list(df.index)


@given(df=dataframe_with_numeric_columns())
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_round_consistency_with_numpy(df):
    """Test that pandas round is consistent with numpy round for numeric columns."""
    decimals = 2
    df_rounded = df.round(decimals)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Compare with numpy round
            expected = np.round(df[col].values, decimals)
            actual = df_rounded[col].values

            # Use pandas testing utilities for NaN-aware comparison
            try:
                pd.testing.assert_numpy_array_equal(actual, expected)
            except AssertionError:
                # Check if both are NaN or both are equal
                mask_nan = np.isnan(actual) & np.isnan(expected)
                mask_equal = actual == expected
                assert np.all(mask_nan | mask_equal), f"Mismatch in column {col}"


@given(dataframe_with_mixed_types())
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_round_with_mixed_types(args):
    """Test that non-numeric columns are left unchanged."""
    df, numeric_cols = args
    decimals = 2

    result = df.round(decimals)

    # Check that non-numeric columns are unchanged
    for col in df.columns:
        if col not in numeric_cols:
            pd.testing.assert_series_equal(result[col], df[col])


@given(df=dataframe_with_numeric_columns())
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_round_with_dict_decimals(df):
    """Test rounding with dictionary decimals parameter."""
    assume(len(df.columns) > 0)

    # Create dict with some columns
    decimals_dict = {col: i % 5 for i, col in enumerate(df.columns[:len(df.columns)//2 + 1])}

    result = df.round(decimals_dict)

    # Verify columns in dict are rounded
    for col, dec in decimals_dict.items():
        if col in df.columns:
            expected = np.round(df[col].values, dec)
            actual = result[col].values

            # NaN-aware comparison
            mask_nan = np.isnan(actual) & np.isnan(expected)
            mask_equal = np.isclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=False)
            assert np.all(mask_nan | mask_equal), f"Mismatch in column {col}"

    # Verify columns not in dict are unchanged
    for col in df.columns:
        if col not in decimals_dict:
            pd.testing.assert_series_equal(result[col], df[col])


@given(df=dataframe_with_numeric_columns())
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_round_with_series_decimals(df):
    """Test rounding with Series decimals parameter."""
    assume(len(df.columns) > 0)

    # Create Series with some columns
    cols_subset = list(df.columns[:len(df.columns)//2 + 1])
    decimals_series = pd.Series([i % 5 for i in range(len(cols_subset))], index=cols_subset)

    result = df.round(decimals_series)

    # Verify columns in series are rounded
    for col in decimals_series.index:
        if col in df.columns:
            dec = decimals_series[col]
            expected = np.round(df[col].values, dec)
            actual = result[col].values

            # NaN-aware comparison
            mask_nan = np.isnan(actual) & np.isnan(expected)
            mask_equal = np.isclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=False)
            assert np.all(mask_nan | mask_equal), f"Mismatch in column {col}"


@given(df=dataframe_with_numeric_columns(), decimals=st.integers(min_value=-5, max_value=0))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_round_with_negative_decimals(df):
    """Test rounding with negative decimal places (rounding to tens, hundreds, etc.)."""
    result = df.round(decimals)

    # Verify shape preserved
    assert result.shape == df.shape

    # Verify consistency with numpy
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            expected = np.round(df[col].values, decimals)
            actual = result[col].values

            mask_nan = np.isnan(actual) & np.isnan(expected)
            mask_equal = np.isclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=False)
            assert np.all(mask_nan | mask_equal), f"Mismatch in column {col} with decimals={decimals}"


@given(df=dataframe_with_numeric_columns())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_round_idempotency(df):
    """Test that rounding twice with same decimals gives same result."""
    decimals = 2

    result1 = df.round(decimals)
    result2 = result1.round(decimals)

    pd.testing.assert_frame_equal(result1, result2)


@given(df=dataframe_with_numeric_columns(), decimals=st.integers(min_value=0, max_value=10))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_round_with_integer_columns(df, decimals):
    """Test that rounding integer columns works correctly."""
    # Filter to only integer columns
    int_df = df.select_dtypes(include=['int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'])

    if len(int_df.columns) > 0:
        result = int_df.round(decimals)

        # Integer values should remain unchanged when rounded with positive decimals
        pd.testing.assert_frame_equal(result, int_df)


@given(df=dataframe_with_numeric_columns())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_round_preserves_nan_and_inf(df):
    """Test that NaN and Inf values are preserved after rounding."""
    result = df.round(2)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check NaN preservation
            assert np.array_equal(np.isnan(df[col]), np.isnan(result[col])), \
                f"NaN not preserved in column {col}"

            # Check Inf preservation
            assert np.array_equal(np.isinf(df[col]), np.isinf(result[col])), \
                f"Inf not preserved in column {col}"


@given(df=dataframe_with_numeric_columns())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_round_does_not_modify_original(df):
    """Test that rounding does not modify the original DataFrame."""
    df_copy = df.copy()

    result = df.round(2)

    pd.testing.assert_frame_equal(df, df_copy)


@given(df=dataframe_with_numeric_columns(), decimals=st.integers(min_value=0, max_value=5))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_round_precision_bounds(df, decimals):
    """Test that rounded values have at most the specified decimal places."""
    result = df.round(decimals)

    for col in result.columns:
        if pd.api.types.is_float_dtype(result[col]):
            for val in result[col].dropna():
                if np.isfinite(val):
                    # Check decimal places
                    str_val = f"{val:.{decimals + 5}f}"
                    decimal_part = str_val.split('.')[1] if '.' in str_val else ''

                    # Count non-zero digits after the decimal point beyond specified decimals
                    if len(decimal_part) > decimals:
                        trailing = decimal_part[decimals:]
                        # Allow for floating point representation errors
                        trailing_int = int(trailing)
                        assert trailing_int < 10**(len(trailing) - 1), \
                            f"Value {val} has more than {decimals} significant decimal places"


# Edge case tests using example-based testing

def test_empty_dataframe():
    """Test rounding an empty DataFrame."""
    df = pd.DataFrame()
    result = df.round(2)
    assert result.empty
    assert isinstance(result, pd.DataFrame)


def test_round_with_zero_decimals():
    """Test rounding to zero decimal places."""
    df = pd.DataFrame({'a': [1.5, 2.5, 3.5, 4.5]})
    result = df.round(0)

    # Check consistency with numpy round (banker's rounding)
    expected = np.round(df['a'].values, 0)
    np.testing.assert_array_equal(result['a'].values, expected)


def test_round_with_very_large_decimals():
    """Test rounding with very large decimal parameter."""
    df = pd.DataFrame({'a': [1.23456789, 2.3456789]})
    result = df.round(50)

    # Should be same as original for large decimals
    pd.testing.assert_frame_equal(result, df)


def test_round_with_complex_numbers():
    """Test that complex numbers are handled appropriately."""
    df = pd.DataFrame({'a': [1+2j, 3+4j]})

    # Complex numbers should not be rounded (non-numeric for round purposes)
    try:
        result = df.round(2)
        # If no error, check that values are unchanged
        pd.testing.assert_frame_equal(result, df)
    except (TypeError, AttributeError):
        # Expected behavior - complex numbers can't be rounded
        pass


def test_round_with_datetime_columns():
    """Test that datetime columns are left unchanged."""
    df = pd.DataFrame({
        'numeric': [1.234, 5.678],
        'datetime': pd.date_range('2020-01-01', periods=2)
    })

    result = df.round(2)

    # Numeric column should be rounded
    assert list(result['numeric']) == [1.23, 5.68]

    # Datetime column should be unchanged
    pd.testing.assert_series_equal(result['datetime'], df['datetime'])


def test_round_with_categorical_columns():
    """Test that categorical columns are left unchanged."""
    df = pd.DataFrame({
        'numeric': [1.234, 5.678],
        'category': pd.Categorical(['a', 'b'])
    })

    result = df.round(2)

    # Numeric column should be rounded
    assert list(result['numeric']) == [1.23, 5.68]

    # Categorical column should be unchanged
    pd.testing.assert_series_equal(result['category'], df['category'])


def test_round_bankers_rounding():
    """Test that pandas uses banker's rounding (round half to even) like numpy."""
    df = pd.DataFrame({'a': [0.5, 1.5, 2.5, 3.5, 4.5]})
    result = df.round(0)

    # Banker's rounding: 0.5->0, 1.5->2, 2.5->2, 3.5->4, 4.5->4
    expected = np.round(df['a'].values, 0)
    np.testing.assert_array_equal(result['a'].values, expected)


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
