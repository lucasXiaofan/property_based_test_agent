"""
Test script for pandas DataFrame.round() function.
Focus: Edge cases and implicit behaviors not covered in the docstring.
Goal: Find inconsistencies with numpy and test different data types.
"""

import pandas as pd
import numpy as np
import warnings


def test_negative_decimals():
    """Test rounding with negative decimals (rounding to tens, hundreds, etc.)"""
    print("\n=== Test: Negative Decimals ===")
    df = pd.DataFrame({'A': [123.456, 789.123, 456.789]})

    # Round to nearest 10
    result = df.round(-1)
    print(f"Original: {df['A'].tolist()}")
    print(f"Round to -1: {result['A'].tolist()}")

    # Compare with numpy
    np_result = np.round(df['A'].values, -1)
    print(f"NumPy result: {np_result.tolist()}")
    assert np.allclose(result['A'].values, np_result), "Mismatch with numpy for negative decimals"

    # Round to nearest 100
    result_100 = df.round(-2)
    print(f"Round to -2: {result_100['A'].tolist()}")


def test_complex_numbers():
    """Test rounding with complex numbers"""
    print("\n=== Test: Complex Numbers ===")
    df = pd.DataFrame({'A': [1.234 + 2.567j, 3.891 + 4.123j]})

    try:
        result = df.round(2)
        print(f"Original: {df['A'].tolist()}")
        print(f"Rounded: {result['A'].tolist()}")

        # Check if both real and imaginary parts are rounded
        for orig, rounded in zip(df['A'], result['A']):
            print(f"  {orig} -> {rounded}")
            assert round(orig.real, 2) == rounded.real, "Real part not rounded correctly"
            assert round(orig.imag, 2) == rounded.imag, "Imaginary part not rounded correctly"
    except Exception as e:
        print(f"Error with complex numbers: {type(e).__name__}: {e}")


def test_integer_columns():
    """Test rounding with integer columns - should they be unchanged?"""
    print("\n=== Test: Integer Columns ===")
    df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]})
    print(f"Original dtypes: {df.dtypes.to_dict()}")

    result = df.round(2)
    print(f"After round(2) dtypes: {result.dtypes.to_dict()}")
    print(f"Values unchanged: {df.equals(result)}")

    # Check if dtype is preserved
    assert df['A'].dtype == result['A'].dtype, "Integer dtype not preserved"


def test_negative_decimals_on_integers():
    """Test negative decimals on integer columns"""
    print("\n=== Test: Negative Decimals on Integers ===")
    df = pd.DataFrame({'A': [123, 789, 456]})
    print(f"Original (int): {df['A'].tolist()}")

    result = df.round(-1)
    print(f"Round to -1: {result['A'].tolist()}")
    print(f"Dtype after rounding: {result['A'].dtype}")

    # Compare with numpy
    np_result = np.round(df['A'].values, -1)
    print(f"NumPy result: {np_result.tolist()}")


def test_nan_and_inf():
    """Test rounding with NaN and inf values"""
    print("\n=== Test: NaN and Inf Values ===")
    df = pd.DataFrame({
        'A': [1.234, np.nan, 3.456],
        'B': [np.inf, 2.567, -np.inf],
        'C': [1.111, np.nan, np.inf]
    })

    print("Original:")
    print(df)

    result = df.round(2)
    print("\nRounded:")
    print(result)

    # Check that NaN and inf are preserved
    assert pd.isna(result.loc[1, 'A']), "NaN not preserved"
    assert np.isinf(result.loc[0, 'B']), "Inf not preserved"
    assert np.isinf(result.loc[2, 'B']) and result.loc[2, 'B'] < 0, "-Inf not preserved"


def test_negative_zero():
    """Test rounding with negative zero"""
    print("\n=== Test: Negative Zero ===")
    df = pd.DataFrame({'A': [-0.0, -0.001, -0.0001]})

    print("Original:")
    print(df)
    print(f"Sign bits: {[np.signbit(x) for x in df['A']]}")

    result = df.round(2)
    print("\nRounded:")
    print(result)
    print(f"Sign bits after rounding: {[np.signbit(x) for x in result['A']]}")

    # Compare with numpy
    np_result = np.round(df['A'].values, 2)
    print(f"NumPy sign bits: {[np.signbit(x) for x in np_result]}")


def test_bankers_rounding():
    """Test half-to-even rounding (banker's rounding)"""
    print("\n=== Test: Banker's Rounding (Round Half to Even) ===")
    df = pd.DataFrame({
        'A': [0.5, 1.5, 2.5, 3.5, 4.5],
        'B': [0.25, 0.35, 0.45, 0.55, 0.65]
    })

    print("Original:")
    print(df)

    result_0 = df.round(0)
    print("\nRounded to 0 decimals (should round .5 to nearest even):")
    print(result_0)

    result_1 = df.round(1)
    print("\nRounded to 1 decimal:")
    print(result_1)

    # Compare with numpy (numpy also uses round half to even)
    np_result = np.round(df['A'].values, 0)
    print(f"\nNumPy result for column A: {np_result}")
    assert np.allclose(result_0['A'].values, np_result), "Pandas doesn't match numpy's banker's rounding"


def test_nullable_dtypes():
    """Test rounding with nullable dtypes (Int64, Float64)"""
    print("\n=== Test: Nullable Dtypes ===")
    df = pd.DataFrame({
        'A': pd.array([1.234, 2.567, pd.NA], dtype='Float64'),
        'B': pd.array([10, 20, pd.NA], dtype='Int64')
    })

    print("Original:")
    print(df)
    print(f"Dtypes: {df.dtypes.to_dict()}")

    result = df.round(1)
    print("\nRounded:")
    print(result)
    print(f"Dtypes after rounding: {result.dtypes.to_dict()}")

    # Check that pd.NA is preserved
    assert pd.isna(result.loc[2, 'A']), "pd.NA not preserved in Float64"


def test_mixed_column_types():
    """Test DataFrame with mixed numeric and non-numeric columns"""
    print("\n=== Test: Mixed Column Types ===")
    df = pd.DataFrame({
        'float': [1.234, 2.567],
        'int': [10, 20],
        'string': ['a', 'b'],
        'bool': [True, False],
        'datetime': pd.to_datetime(['2020-01-01', '2020-01-02'])
    })

    print("Original:")
    print(df)
    print(f"Dtypes: {df.dtypes.to_dict()}")

    result = df.round(2)
    print("\nAfter round(2):")
    print(result)
    print(f"Dtypes after rounding: {result.dtypes.to_dict()}")

    # Check that non-numeric columns are unchanged
    assert result['string'].equals(df['string']), "String column changed"
    assert result['bool'].equals(df['bool']), "Bool column changed"
    assert result['datetime'].equals(df['datetime']), "Datetime column changed"


def test_categorical_columns():
    """Test rounding with categorical columns"""
    print("\n=== Test: Categorical Columns ===")
    df = pd.DataFrame({
        'A': [1.234, 2.567, 3.891],
        'B': pd.Categorical(['cat', 'dog', 'cat'])
    })

    print("Original:")
    print(df)
    print(f"Dtypes: {df.dtypes.to_dict()}")

    result = df.round(2)
    print("\nRounded:")
    print(result)
    print(f"Dtypes after rounding: {result.dtypes.to_dict()}")

    # Check that categorical is preserved
    assert isinstance(result['B'].dtype, pd.CategoricalDtype), "Categorical dtype not preserved"


def test_empty_dataframe():
    """Test rounding with empty DataFrame"""
    print("\n=== Test: Empty DataFrame ===")
    df = pd.DataFrame({'A': [], 'B': []})

    print(f"Original shape: {df.shape}")
    result = df.round(2)
    print(f"After round(2) shape: {result.shape}")

    assert result.shape == df.shape, "Shape changed for empty DataFrame"


def test_very_large_decimals():
    """Test rounding with very large decimal values"""
    print("\n=== Test: Very Large Decimals ===")
    df = pd.DataFrame({'A': [1.23456789012345]})

    print(f"Original: {df['A'].tolist()}")

    for decimals in [0, 5, 10, 15, 20]:
        result = df.round(decimals)
        print(f"Round to {decimals}: {result['A'].tolist()}")


def test_dict_with_invalid_columns():
    """Test dict parameter with invalid column names"""
    print("\n=== Test: Dict with Invalid Column Names ===")
    df = pd.DataFrame({'A': [1.234, 2.567], 'B': [3.891, 4.123]})

    print("Original:")
    print(df)

    # Dict with some invalid columns
    result = df.round({'A': 1, 'C': 2, 'D': 0})
    print("\nRounded with {'A': 1, 'C': 2, 'D': 0}:")
    print(result)

    # Check that B is unchanged and A is rounded
    assert round(df.loc[0, 'A'], 1) == result.loc[0, 'A'], "Column A not rounded correctly"
    assert df['B'].equals(result['B']), "Column B should be unchanged"


def test_series_with_mismatched_index():
    """Test Series parameter with non-matching index"""
    print("\n=== Test: Series with Mismatched Index ===")
    df = pd.DataFrame({'A': [1.234, 2.567], 'B': [3.891, 4.123]})

    print("Original:")
    print(df)

    # Series with partial match
    decimals = pd.Series([1, 2], index=['A', 'C'])
    result = df.round(decimals)
    print("\nRounded with Series(index=['A', 'C']):")
    print(result)

    # Check behavior
    assert round(df.loc[0, 'A'], 1) == result.loc[0, 'A'], "Column A not rounded correctly"


def test_no_mutation():
    """Test that round() doesn't mutate the original DataFrame"""
    print("\n=== Test: No Mutation ===")
    df = pd.DataFrame({'A': [1.234, 2.567], 'B': [3.891, 4.123]})
    original_values = df.copy()

    print("Original:")
    print(df)

    result = df.round(1)

    print("\nAfter calling round(1), original DataFrame:")
    print(df)

    assert df.equals(original_values), "Original DataFrame was mutated!"


def test_object_dtype_with_numbers():
    """Test object dtype containing numeric values"""
    print("\n=== Test: Object Dtype with Numbers ===")
    df = pd.DataFrame({'A': [1.234, 2.567, 'string']})

    print("Original:")
    print(df)
    print(f"Dtype: {df['A'].dtype}")

    try:
        result = df.round(2)
        print("\nRounded:")
        print(result)
        print(f"Dtype after rounding: {result['A'].dtype}")
    except Exception as e:
        print(f"Error with object dtype: {type(e).__name__}: {e}")


def test_zero_decimals():
    """Test rounding to zero decimals"""
    print("\n=== Test: Zero Decimals ===")
    df = pd.DataFrame({'A': [1.4, 1.5, 1.6, 2.4, 2.5, 2.6]})

    print("Original:")
    print(df)

    result = df.round(0)
    print("\nRounded to 0 decimals:")
    print(result)

    # Compare with numpy
    np_result = np.round(df['A'].values, 0)
    print(f"NumPy result: {np_result}")
    assert np.allclose(result['A'].values, np_result), "Mismatch with numpy for 0 decimals"


def test_multiindex_dataframe():
    """Test rounding with MultiIndex DataFrame"""
    print("\n=== Test: MultiIndex DataFrame ===")
    arrays = [['bar', 'bar', 'baz', 'baz'], ['one', 'two', 'one', 'two']]
    index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
    df = pd.DataFrame({'A': [1.234, 2.567, 3.891, 4.123]}, index=index)

    print("Original:")
    print(df)

    result = df.round(2)
    print("\nRounded:")
    print(result)

    assert isinstance(result.index, pd.MultiIndex), "MultiIndex not preserved"


def test_very_small_numbers():
    """Test rounding very small numbers (near machine epsilon)"""
    print("\n=== Test: Very Small Numbers ===")
    df = pd.DataFrame({
        'A': [1e-10, 1e-15, 1e-20, 1.23456789e-8]
    })

    print("Original:")
    print(df)

    for decimals in [5, 10, 15, 20]:
        result = df.round(decimals)
        print(f"\nRounded to {decimals} decimals:")
        print(result)


def test_float32_vs_float64():
    """Test rounding with float32 vs float64"""
    print("\n=== Test: Float32 vs Float64 ===")
    df32 = pd.DataFrame({'A': [1.234567, 2.345678]}, dtype=np.float32)
    df64 = pd.DataFrame({'A': [1.234567, 2.345678]}, dtype=np.float64)

    print("Float32 original:")
    print(df32)
    print(f"Dtype: {df32['A'].dtype}")

    print("\nFloat64 original:")
    print(df64)
    print(f"Dtype: {df64['A'].dtype}")

    result32 = df32.round(3)
    result64 = df64.round(3)

    print("\nFloat32 rounded:")
    print(result32)
    print(f"Dtype: {result32['A'].dtype}")

    print("\nFloat64 rounded:")
    print(result64)
    print(f"Dtype: {result64['A'].dtype}")

    # Check if dtypes are preserved
    assert df32['A'].dtype == result32['A'].dtype, "Float32 dtype not preserved"
    assert df64['A'].dtype == result64['A'].dtype, "Float64 dtype not preserved"


def test_string_decimals_parameter():
    """Test passing invalid types for decimals parameter"""
    print("\n=== Test: Invalid Decimals Parameter ===")
    df = pd.DataFrame({'A': [1.234, 2.567]})

    # Test with string
    try:
        result = df.round("2")
        print(f"round('2') succeeded: {result}")
    except Exception as e:
        print(f"round('2') raised {type(e).__name__}: {e}")

    # Test with None
    try:
        result = df.round(None)
        print(f"round(None) succeeded: {result}")
    except Exception as e:
        print(f"round(None) raised {type(e).__name__}: {e}")

    # Test with float
    try:
        result = df.round(2.5)
        print(f"round(2.5) succeeded: {result}")
    except Exception as e:
        print(f"round(2.5) raised {type(e).__name__}: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing pandas DataFrame.round() - Edge Cases & Implicit Behaviors")
    print("=" * 60)

    tests = [
        test_negative_decimals,
        test_complex_numbers,
        test_integer_columns,
        test_negative_decimals_on_integers,
        test_nan_and_inf,
        test_negative_zero,
        test_bankers_rounding,
        test_nullable_dtypes,
        test_mixed_column_types,
        test_categorical_columns,
        test_empty_dataframe,
        test_very_large_decimals,
        test_dict_with_invalid_columns,
        test_series_with_mismatched_index,
        test_no_mutation,
        test_object_dtype_with_numbers,
        test_zero_decimals,
        test_multiindex_dataframe,
        test_very_small_numbers,
        test_float32_vs_float64,
        test_string_decimals_parameter,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print(f"✓ {test.__name__} passed")
        except AssertionError as e:
            failed += 1
            print(f"✗ {test.__name__} failed: {e}")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} error: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)


if __name__ == "__main__":
    main()
