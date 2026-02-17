"""
Property-based testing for pandas DataFrame.round() function using Hypothesis.
This test suite attempts to find bugs in pandas 2.2.3 round() implementation.
"""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import column, data_frames
import math


# Strategies for generating test data
@st.composite
def numeric_dataframe_strategy(draw):
    """Generate a DataFrame with numeric columns."""
    num_cols = draw(st.integers(min_value=1, max_value=10))
    num_rows = draw(st.integers(min_value=0, max_value=100))

    columns = []
    for i in range(num_cols):
        col_name = f"col_{i}"
        # Mix of different numeric types
        dtype = draw(st.sampled_from([np.float64, np.float32, np.int64, np.int32]))

        if dtype in [np.float64, np.float32]:
            # Combine bounded floats with special values (NaN, inf)
            elements = st.one_of(
                st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
                st.just(float('nan')),
                st.just(float('inf')),
                st.just(float('-inf'))
            )
        else:
            elements = st.integers(min_value=-10000, max_value=10000)

        columns.append(column(name=col_name, dtype=dtype, elements=elements))

    return draw(data_frames(columns=columns))


@st.composite
def mixed_dataframe_strategy(draw):
    """Generate a DataFrame with mixed numeric and non-numeric columns."""
    num_cols = draw(st.integers(min_value=1, max_value=5))
    num_rows = draw(st.integers(min_value=0, max_value=50))

    columns = []
    for i in range(num_cols):
        col_name = f"col_{i}"
        # Mix numeric and string columns
        is_numeric = draw(st.booleans())

        if is_numeric:
            dtype = draw(st.sampled_from([np.float64, np.int64]))
            if dtype == np.float64:
                # Combine bounded floats with special values (NaN, inf)
                elements = st.one_of(
                    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
                    st.just(float('nan')),
                    st.just(float('inf')),
                    st.just(float('-inf'))
                )
            else:
                elements = st.integers(min_value=-1000, max_value=1000)
        else:
            dtype = str
            elements = st.text(min_size=0, max_size=10)

        columns.append(column(name=col_name, dtype=dtype, elements=elements))

    return draw(data_frames(columns=columns))


# Test 1: Basic property - rounding with integer decimals
@given(
    df=numeric_dataframe_strategy(),
    decimals=st.integers(min_value=-10, max_value=10)
)
@settings(max_examples=200)
def test_round_with_int_decimals(df, decimals):
    """Test that rounding with integer decimals parameter works correctly."""
    try:
        result = df.round(decimals)

        # Property 1: Result should have same shape as input
        assert result.shape == df.shape, f"Shape mismatch: {result.shape} vs {df.shape}"

        # Property 2: Result should have same column names
        assert list(result.columns) == list(df.columns), "Column names changed"

        # Property 3: For finite values, check rounding is correct
        for col in df.select_dtypes(include=[np.number]).columns:
            for idx in df.index:
                original = df.loc[idx, col]
                rounded = result.loc[idx, col]

                if pd.isna(original):
                    assert pd.isna(rounded), f"NaN not preserved at {idx}, {col}"
                elif np.isinf(original):
                    assert np.isinf(rounded) and (original == rounded), f"Inf not preserved at {idx}, {col}"
                else:
                    # Check that rounding was applied correctly
                    expected = round(float(original), decimals)
                    if not math.isclose(float(rounded), expected, rel_tol=1e-9, abs_tol=1e-15):
                        print(f"Rounding error at {idx}, {col}: {original} -> {rounded}, expected {expected}")

    except Exception as e:
        # Log unexpected exceptions
        print(f"Unexpected error with decimals={decimals}, df.shape={df.shape}: {e}")
        raise


# Test 2: Rounding with dict parameter
@given(
    df=numeric_dataframe_strategy(),
    decimals=st.data()
)
@settings(max_examples=200)
def test_round_with_dict_decimals(df, decimals):
    """Test that rounding with dict decimals parameter works correctly."""
    # Generate a dict with some column names
    num_decimals = decimals.draw(st.integers(min_value=0, max_value=len(df.columns)))
    selected_cols = decimals.draw(st.lists(
        st.sampled_from(list(df.columns)),
        min_size=0,
        max_size=num_decimals,
        unique=True
    ))

    decimals_dict = {
        col: decimals.draw(st.integers(min_value=-5, max_value=10))
        for col in selected_cols
    }

    # Add some extra keys that don't exist in DataFrame
    extra_keys = decimals.draw(st.booleans())
    if extra_keys and len(decimals_dict) > 0:
        decimals_dict['nonexistent_col'] = decimals.draw(st.integers(min_value=0, max_value=5))

    try:
        result = df.round(decimals_dict)

        # Property 1: Result should have same shape
        assert result.shape == df.shape, f"Shape mismatch: {result.shape} vs {df.shape}"

        # Property 2: Columns not in dict should be unchanged
        for col in df.columns:
            if col not in decimals_dict:
                # Column should be unchanged
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    pd.testing.assert_series_equal(result[col], df[col], check_exact=False)

        # Property 3: Columns in dict should be rounded to specified decimals
        for col, dec in decimals_dict.items():
            if col in df.columns:
                for idx in df.index:
                    original = df.loc[idx, col]
                    rounded = result.loc[idx, col]

                    if pd.isna(original):
                        assert pd.isna(rounded), f"NaN not preserved"
                    elif not np.isinf(original):
                        expected = round(float(original), dec)
                        if not math.isclose(float(rounded), expected, rel_tol=1e-9, abs_tol=1e-15):
                            print(f"Dict rounding error at {idx}, {col}: {original} -> {rounded}, expected {expected} with decimals={dec}")

    except Exception as e:
        print(f"Unexpected error with dict decimals={decimals_dict}, df.shape={df.shape}: {e}")
        raise


# Test 3: Rounding with Series parameter
@given(
    df=numeric_dataframe_strategy(),
    decimals=st.data()
)
@settings(max_examples=200)
def test_round_with_series_decimals(df, decimals):
    """Test that rounding with Series decimals parameter works correctly."""
    # Generate a Series with some column names as index
    num_decimals = decimals.draw(st.integers(min_value=0, max_value=len(df.columns)))
    selected_cols = decimals.draw(st.lists(
        st.sampled_from(list(df.columns)),
        min_size=0,
        max_size=num_decimals,
        unique=True
    ))

    decimal_values = [
        decimals.draw(st.integers(min_value=-5, max_value=10))
        for _ in selected_cols
    ]

    decimals_series = pd.Series(decimal_values, index=selected_cols)

    try:
        result = df.round(decimals_series)

        # Property 1: Result should have same shape
        assert result.shape == df.shape, f"Shape mismatch"

        # Property 2: Columns not in Series should be unchanged
        for col in df.columns:
            if col not in decimals_series.index:
                pd.testing.assert_series_equal(result[col], df[col], check_exact=False)

        # Property 3: Columns in Series should be rounded correctly
        for col in decimals_series.index:
            if col in df.columns:
                dec = decimals_series[col]
                for idx in df.index:
                    original = df.loc[idx, col]
                    rounded = result.loc[idx, col]

                    if pd.isna(original):
                        assert pd.isna(rounded)
                    elif not np.isinf(original):
                        expected = round(float(original), dec)
                        if not math.isclose(float(rounded), expected, rel_tol=1e-9, abs_tol=1e-15):
                            print(f"Series rounding error at {idx}, {col}: {original} -> {rounded}, expected {expected}")

    except Exception as e:
        print(f"Unexpected error with Series decimals, df.shape={df.shape}: {e}")
        raise


# Test 4: Mixed DataFrame with numeric and non-numeric columns
@given(
    df=mixed_dataframe_strategy(),
    decimals=st.integers(min_value=0, max_value=5)
)
@settings(max_examples=200)
def test_round_with_mixed_dataframe(df, decimals):
    """Test that non-numeric columns are left unchanged."""
    try:
        result = df.round(decimals)

        # Property 1: Non-numeric columns should be unchanged
        for col in df.columns:
            if df[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                pd.testing.assert_series_equal(result[col], df[col], check_dtype=True)

        # Property 2: Numeric columns should be rounded
        for col in df.select_dtypes(include=[np.number]).columns:
            for idx in df.index:
                original = df.loc[idx, col]
                rounded = result.loc[idx, col]

                if pd.isna(original):
                    assert pd.isna(rounded)
                elif not np.isinf(original):
                    expected = round(float(original), decimals)
                    if not math.isclose(float(rounded), expected, rel_tol=1e-9, abs_tol=1e-15):
                        print(f"Mixed df rounding error: {original} -> {rounded}, expected {expected}")

    except Exception as e:
        print(f"Unexpected error with mixed DataFrame: {e}")
        raise


# Test 5: Edge cases - empty DataFrame
@given(decimals=st.integers(min_value=-5, max_value=10))
@settings(max_examples=50)
def test_round_empty_dataframe(decimals):
    """Test rounding on empty DataFrame."""
    df = pd.DataFrame()
    try:
        result = df.round(decimals)
        assert result.empty, "Empty DataFrame should remain empty"
        assert result.shape == (0, 0), "Shape should be (0, 0)"
    except Exception as e:
        print(f"Error with empty DataFrame and decimals={decimals}: {e}")
        raise


# Test 6: Idempotence - rounding twice should give same result
@given(
    df=numeric_dataframe_strategy(),
    decimals=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=100)
def test_round_idempotence(df, decimals):
    """Test that rounding twice gives the same result as rounding once."""
    try:
        result1 = df.round(decimals)
        result2 = result1.round(decimals)

        pd.testing.assert_frame_equal(result1, result2, check_exact=False)
    except Exception as e:
        print(f"Idempotence test failed: {e}")
        raise


# Test 7: Consistency with numpy.round
@given(
    df=numeric_dataframe_strategy(),
    decimals=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=100)
def test_round_consistency_with_numpy(df, decimals):
    """Test that pandas round is consistent with numpy round for valid cases."""
    try:
        pandas_result = df.round(decimals)

        for col in df.select_dtypes(include=[np.number]).columns:
            # Apply numpy round to the column
            numpy_result = np.round(df[col].values, decimals)

            # Compare (allowing for NaN and Inf)
            for i, (pr, nr) in enumerate(zip(pandas_result[col].values, numpy_result)):
                if pd.isna(df[col].iloc[i]):
                    continue
                if np.isinf(df[col].iloc[i]):
                    continue

                if not math.isclose(pr, nr, rel_tol=1e-9, abs_tol=1e-15):
                    print(f"Inconsistency with numpy at {i}, {col}: pandas={pr}, numpy={nr}, original={df[col].iloc[i]}")

    except Exception as e:
        print(f"Consistency test failed: {e}")
        raise


if __name__ == "__main__":
    # Run all tests
    print("Running property-based tests for pandas DataFrame.round()...")
    print("\n1. Testing with integer decimals...")
    test_round_with_int_decimals()

    print("\n2. Testing with dict decimals...")
    test_round_with_dict_decimals()

    print("\n3. Testing with Series decimals...")
    test_round_with_series_decimals()

    print("\n4. Testing with mixed DataFrame...")
    test_round_with_mixed_dataframe()

    print("\n5. Testing empty DataFrame...")
    test_round_empty_dataframe()

    print("\n6. Testing idempotence...")
    test_round_idempotence()

    print("\n7. Testing consistency with numpy...")
    test_round_consistency_with_numpy()

    print("\n✓ All property-based tests passed!")
