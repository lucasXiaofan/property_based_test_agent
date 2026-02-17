"""
Test script for pandas DataFrame.round() - Creative Edge Cases
Focus: Unusual data types and scenarios outside typical numeric testing.
Goal: Think outside the box - test object types, mixed data, special cases.
"""

import pandas as pd
import numpy as np
import sys
from decimal import Decimal
from fractions import Fraction
import warnings


def print_test_header(test_name):
    """Print formatted test header"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print('='*80)


def test_object_dtype_with_mixed_content():
    """Test object dtype containing various types"""
    print_test_header("Object Dtype with Mixed Content")

    # Pure strings
    df1 = pd.DataFrame({'A': ['hello', 'world', 'test']})
    print("\n1. Pure strings:")
    print(f"  Original: {df1['A'].tolist()}")
    try:
        result = df1.round(2)
        print(f"  After round(2): {result['A'].tolist()}")
        print(f"  ✓ No error, strings preserved: {df1['A'].equals(result['A'])}")
    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")

    # Mixed numbers and strings
    df2 = pd.DataFrame({'A': [1.234, 'string', 3.456]})
    print("\n2. Mixed numbers and strings:")
    print(f"  Original: {df2['A'].tolist()}, dtype: {df2['A'].dtype}")
    try:
        result = df2.round(2)
        print(f"  After round(2): {result['A'].tolist()}")
        print(f"  ✗ UNEXPECTED: Should have raised error or kept as-is")
    except Exception as e:
        print(f"  ✓ Error raised: {type(e).__name__}: {e}")

    # None values in object dtype
    df3 = pd.DataFrame({'A': [1.234, None, 3.456]})
    print("\n3. Object dtype with None:")
    print(f"  Original: {df3['A'].tolist()}, dtype: {df3['A'].dtype}")
    try:
        result = df3.round(2)
        print(f"  After round(2): {result['A'].tolist()}")
        print(f"  None preserved: {result['A'].iloc[1] is None or pd.isna(result['A'].iloc[1])}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

    # Lists in cells
    df4 = pd.DataFrame({'A': [[1.234, 2.567], [3.891, 4.123]]})
    print("\n4. Lists in cells:")
    print(f"  Original: {df4['A'].tolist()}")
    try:
        result = df4.round(2)
        print(f"  After round(2): {result['A'].tolist()}")
        print(f"  ✗ UNEXPECTED: Lists in cells should not be rounded")
    except Exception as e:
        print(f"  ✓ Error raised: {type(e).__name__}: {e}")

    # Dicts in cells
    df5 = pd.DataFrame({'A': [{'value': 1.234}, {'value': 2.567}]})
    print("\n5. Dicts in cells:")
    print(f"  Original: {df5['A'].tolist()}")
    try:
        result = df5.round(2)
        print(f"  After round(2): {result['A'].tolist()}")
        print(f"  Dicts preserved: {df5['A'].equals(result['A'])}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")


def test_boolean_data():
    """Test rounding boolean data"""
    print_test_header("Boolean Data")

    df = pd.DataFrame({'A': [True, False, True, False]})
    print(f"Original: {df['A'].tolist()}, dtype: {df['A'].dtype}")

    try:
        result = df.round(2)
        print(f"After round(2): {result['A'].tolist()}, dtype: {result['A'].dtype}")
        print(f"Values unchanged: {df['A'].equals(result['A'])}")
        print(f"Dtype preserved: {df['A'].dtype == result['A'].dtype}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    # Boolean with nullable dtype
    df_nullable = pd.DataFrame({'A': pd.array([True, False, pd.NA], dtype='boolean')})
    print(f"\nNullable boolean: {df_nullable['A'].tolist()}, dtype: {df_nullable['A'].dtype}")
    try:
        result = df_nullable.round(2)
        print(f"After round(2): {result['A'].tolist()}, dtype: {result['A'].dtype}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_string_numbers():
    """Test strings that look like numbers"""
    print_test_header("String Numbers")

    df = pd.DataFrame({'A': ['1.234', '2.567', '3.891']})
    print(f"Original: {df['A'].tolist()}, dtype: {df['A'].dtype}")

    try:
        result = df.round(2)
        print(f"After round(2): {result['A'].tolist()}")
        print(f"Strings preserved: {df['A'].equals(result['A'])}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    # Try with StringDtype
    df_string = pd.DataFrame({'A': pd.array(['1.234', '2.567', '3.891'], dtype='string')})
    print(f"\nStringDtype: {df_string['A'].tolist()}, dtype: {df_string['A'].dtype}")
    try:
        result = df_string.round(2)
        print(f"After round(2): {result['A'].tolist()}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_decimal_objects():
    """Test Decimal objects from decimal module"""
    print_test_header("Decimal Objects")

    df = pd.DataFrame({'A': [Decimal('1.234'), Decimal('2.567'), Decimal('3.891')]})
    print(f"Original: {df['A'].tolist()}, dtype: {df['A'].dtype}")

    try:
        result = df.round(2)
        print(f"After round(2): {result['A'].tolist()}")
        print(f"Result dtype: {result['A'].dtype}")
        # Check if Decimals were converted or preserved
        print(f"Still Decimal objects: {isinstance(result['A'].iloc[0], Decimal)}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_fraction_objects():
    """Test Fraction objects from fractions module"""
    print_test_header("Fraction Objects")

    df = pd.DataFrame({'A': [Fraction(1, 3), Fraction(2, 3), Fraction(5, 6)]})
    print(f"Original: {df['A'].tolist()}, dtype: {df['A'].dtype}")
    print(f"As decimals: {[float(x) for x in df['A']]}")

    try:
        result = df.round(2)
        print(f"After round(2): {result['A'].tolist()}")
        print(f"Result dtype: {result['A'].dtype}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_datetime_in_dataframe():
    """Test datetime columns in DataFrame.round()"""
    print_test_header("Datetime Columns")

    df = pd.DataFrame({
        'numeric': [1.234, 2.567],
        'datetime': pd.to_datetime(['2020-01-01 12:34:56.789', '2020-12-31 23:59:59.999'])
    })
    print("Original:")
    print(df)
    print(f"Dtypes: {df.dtypes.to_dict()}")

    try:
        result = df.round(2)
        print("\nAfter round(2):")
        print(result)
        print(f"Datetime preserved: {df['datetime'].equals(result['datetime'])}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_timedelta_in_dataframe():
    """Test timedelta columns"""
    print_test_header("Timedelta Columns")

    df = pd.DataFrame({
        'numeric': [1.234, 2.567],
        'timedelta': pd.to_timedelta(['1 days 2 hours 30 minutes', '3 days 4 hours 15 minutes'])
    })
    print("Original:")
    print(df)

    try:
        result = df.round(2)
        print("\nAfter round(2):")
        print(result)
        print(f"Timedelta preserved: {df['timedelta'].equals(result['timedelta'])}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_categorical_with_numbers():
    """Test categorical data with numeric categories"""
    print_test_header("Categorical with Numbers")

    df = pd.DataFrame({
        'A': pd.Categorical([1.234, 2.567, 1.234, 2.567])
    })
    print(f"Original: {df['A'].tolist()}, dtype: {df['A'].dtype}")
    print(f"Categories: {df['A'].cat.categories.tolist()}")

    try:
        result = df.round(2)
        print(f"After round(2): {result['A'].tolist()}")
        print(f"Result dtype: {result['A'].dtype}")
        if isinstance(result['A'].dtype, pd.CategoricalDtype):
            print(f"Categories after: {result['A'].cat.categories.tolist()}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_period_dtype():
    """Test Period dtype"""
    print_test_header("Period Dtype")

    df = pd.DataFrame({
        'numeric': [1.234, 2.567, 3.891],
        'period': pd.period_range('2020-01', periods=3, freq='M')
    })
    print("Original:")
    print(df)
    print(f"Dtypes: {df.dtypes.to_dict()}")

    try:
        result = df.round(2)
        print("\nAfter round(2):")
        print(result)
        print(f"Period preserved: {df['period'].equals(result['period'])}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_interval_dtype():
    """Test Interval dtype"""
    print_test_header("Interval Dtype")

    df = pd.DataFrame({
        'numeric': [1.234, 2.567, 3.891],
        'interval': pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2), (2, 3)])
    })
    print("Original:")
    print(df)
    print(f"Dtypes: {df.dtypes.to_dict()}")

    try:
        result = df.round(2)
        print("\nAfter round(2):")
        print(result)
        print(f"Interval preserved: {df['interval'].equals(result['interval'])}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_sparse_arrays():
    """Test sparse arrays"""
    print_test_header("Sparse Arrays")

    df = pd.DataFrame({
        'A': pd.arrays.SparseArray([1.234, 0, 0, 2.567, 0, 0, 3.891])
    })
    print(f"Original: {df['A'].tolist()}")
    print(f"Dtype: {df['A'].dtype}")
    print(f"Sparsity: {1 - df['A'].sparse.density:.2%}")

    try:
        result = df.round(2)
        print(f"After round(2): {result['A'].tolist()}")
        print(f"Result dtype: {result['A'].dtype}")
        if hasattr(result['A'], 'sparse'):
            print(f"Still sparse: True, density: {result['A'].sparse.density:.2%}")
        else:
            print(f"Still sparse: False")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_multiindex_columns():
    """Test DataFrame with MultiIndex columns"""
    print_test_header("MultiIndex Columns")

    df = pd.DataFrame(
        [[1.234, 2.567], [3.891, 4.123]],
        columns=pd.MultiIndex.from_tuples([('A', 'x'), ('A', 'y')])
    )
    print("Original:")
    print(df)
    print(f"Column names: {df.columns.tolist()}")

    try:
        result = df.round(2)
        print("\nAfter round(2):")
        print(result)
        print(f"MultiIndex preserved: {isinstance(result.columns, pd.MultiIndex)}")
        print(f"Column structure preserved: {df.columns.equals(result.columns)}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_special_column_names():
    """Test DataFrames with unusual column names"""
    print_test_header("Special Column Names")

    # Unicode and special characters
    df1 = pd.DataFrame({
        '中文': [1.234, 2.567],
        '🚀': [3.891, 4.123],
        'with spaces': [5.678, 6.789],
        '': [7.890, 8.901]  # Empty string
    })
    print("\n1. Unicode and special characters:")
    print(f"  Columns: {df1.columns.tolist()}")
    try:
        result = df1.round(2)
        print(f"  ✓ Success, columns preserved: {df1.columns.equals(result.columns)}")
    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")

    # Very long column name
    long_name = 'A' * 1000
    df2 = pd.DataFrame({long_name: [1.234, 2.567]})
    print(f"\n2. Very long column name (length={len(long_name)}):")
    try:
        result = df2.round(2)
        print(f"  ✓ Success")
    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")

    # Column names that are numbers
    df3 = pd.DataFrame({1: [1.234, 2.567], 2.5: [3.891, 4.123]})
    print(f"\n3. Numeric column names: {df3.columns.tolist()}")
    try:
        result = df3.round(2)
        print(f"  ✓ Success, columns: {result.columns.tolist()}")
    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")


def test_dict_series_edge_cases():
    """Test edge cases with dict/Series parameters"""
    print_test_header("Dict/Series Parameter Edge Cases")

    df = pd.DataFrame({'A': [1.234, 2.567], 'B': [3.891, 4.123]})

    # Empty dict
    print("\n1. Empty dict:")
    try:
        result = df.round({})
        print(f"  Result equals original: {df.equals(result)}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

    # Dict with None values
    print("\n2. Dict with None values:")
    try:
        result = df.round({'A': None, 'B': 2})
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

    # Series with duplicate index
    print("\n3. Series with duplicate index:")
    try:
        decimals = pd.Series([1, 2, 3], index=['A', 'A', 'B'])
        result = df.round(decimals)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

    # Empty Series
    print("\n4. Empty Series:")
    try:
        result = df.round(pd.Series([], dtype=int))
        print(f"  Result equals original: {df.equals(result)}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

    # Dict with negative values
    print("\n5. Dict with negative decimals:")
    try:
        result = df.round({'A': -1, 'B': 2})
        print(f"  Result:\n{result}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")


def test_idempotency():
    """Test that round(round(x)) == round(x)"""
    print_test_header("Idempotency: round(round(x)) == round(x)")

    df = pd.DataFrame({'A': [1.23456789, 2.34567890, 3.45678901]})
    print(f"Original: {df['A'].tolist()}")

    result1 = df.round(2)
    result2 = result1.round(2)
    result3 = result2.round(2)

    print(f"After 1st round(2): {result1['A'].tolist()}")
    print(f"After 2nd round(2): {result2['A'].tolist()}")
    print(f"After 3rd round(2): {result3['A'].tolist()}")

    assert result1.equals(result2), "BUG: Not idempotent after 2nd application"
    assert result2.equals(result3), "BUG: Not idempotent after 3rd application"
    print("✓ Idempotency test passed")


def test_subnormal_numbers():
    """Test subnormal (denormalized) floating point numbers"""
    print_test_header("Subnormal Numbers")

    # Smallest positive float64 subnormal number
    subnormal = np.finfo(np.float64).tiny / 2
    df = pd.DataFrame({'A': [subnormal, -subnormal, 1.234]})

    print(f"Original: {df['A'].tolist()}")
    print(f"Is subnormal: {[x < np.finfo(np.float64).tiny and x != 0 for x in df['A']]}")

    try:
        result = df.round(2)
        print(f"After round(2): {result['A'].tolist()}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_memory_order():
    """Test arrays with different memory layout (C vs Fortran order)"""
    print_test_header("Memory Layout (C vs Fortran order)")

    arr_c = np.array([[1.234, 2.567], [3.891, 4.123]], order='C')
    arr_f = np.array([[1.234, 2.567], [3.891, 4.123]], order='F')

    df_c = pd.DataFrame(arr_c)
    df_f = pd.DataFrame(arr_f)

    print(f"C-order flags: {arr_c.flags}")
    print(f"F-order flags: {arr_f.flags}")

    result_c = df_c.round(2)
    result_f = df_f.round(2)

    print(f"Results equal: {result_c.equals(result_f)}")
    print(f"✓ Memory order doesn't affect results")


def test_bytes_data():
    """Test bytes and bytearray data"""
    print_test_header("Bytes and Bytearray Data")

    df = pd.DataFrame({'A': [b'1.234', b'2.567', b'3.891']})
    print(f"Original: {df['A'].tolist()}, dtype: {df['A'].dtype}")

    try:
        result = df.round(2)
        print(f"After round(2): {result['A'].tolist()}")
        print(f"Bytes preserved: {df['A'].equals(result['A'])}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def test_copy_vs_view():
    """Test if round() returns a copy or view"""
    print_test_header("Copy vs View Behavior")

    df = pd.DataFrame({'A': [1.234, 2.567], 'B': [3.891, 4.123]})
    result = df.round(2)

    # Modify result
    original_value = result.loc[0, 'A']
    result.loc[0, 'A'] = 999.999

    if df.loc[0, 'A'] == 999.999:
        print("✗ BUG: round() returned a view, original DataFrame was modified!")
        print(f"Original df after modifying result: {df['A'].tolist()}")
    else:
        print("✓ round() returns a copy, original DataFrame is safe")


def test_column_order_preservation():
    """Test that column order is preserved"""
    print_test_header("Column Order Preservation")

    # Create DataFrame with specific column order
    df = pd.DataFrame({'Z': [1.234], 'A': [2.567], 'M': [3.891], 'B': [4.123]})
    original_order = df.columns.tolist()

    result = df.round(2)
    result_order = result.columns.tolist()

    print(f"Original order: {original_order}")
    print(f"Result order:   {result_order}")
    print(f"✓ Order preserved: {original_order == result_order}")


def test_index_preservation():
    """Test various index types are preserved"""
    print_test_header("Index Preservation")

    # String index
    df1 = pd.DataFrame({'A': [1.234, 2.567]}, index=['row1', 'row2'])
    result1 = df1.round(2)
    print(f"String index preserved: {df1.index.equals(result1.index)}")

    # DatetimeIndex
    df2 = pd.DataFrame({'A': [1.234, 2.567]},
                       index=pd.date_range('2020-01-01', periods=2))
    result2 = df2.round(2)
    print(f"DatetimeIndex preserved: {df2.index.equals(result2.index)}")

    # MultiIndex
    idx = pd.MultiIndex.from_tuples([('A', 1), ('A', 2)])
    df3 = pd.DataFrame({'A': [1.234, 2.567]}, index=idx)
    result3 = df3.round(2)
    print(f"MultiIndex preserved: {df3.index.equals(result3.index)}")

    print("✓ All index types preserved")


def main():
    """Run all creative edge case tests"""
    print("=" * 80)
    print("Testing pandas DataFrame.round() - Creative Edge Cases")
    print("Focus: Unusual data types and outside-the-box scenarios")
    print("=" * 80)

    tests = [
        test_object_dtype_with_mixed_content,
        test_boolean_data,
        test_string_numbers,
        test_decimal_objects,
        test_fraction_objects,
        test_datetime_in_dataframe,
        test_timedelta_in_dataframe,
        test_categorical_with_numbers,
        test_period_dtype,
        test_interval_dtype,
        test_sparse_arrays,
        test_multiindex_columns,
        test_special_column_names,
        test_dict_series_edge_cases,
        test_idempotency,
        test_subnormal_numbers,
        test_memory_order,
        test_bytes_data,
        test_copy_vs_view,
        test_column_order_preservation,
        test_index_preservation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print(f"\n✓ {test.__name__} completed")
        except AssertionError as e:
            failed += 1
            print(f"\n✗ {test.__name__} FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test.__name__} ERROR: {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
    print(f"Test Summary: {passed} completed, {failed} failed out of {passed + failed} tests")
    print("=" * 80)

    return failed


if __name__ == "__main__":
    num_failed = main()
    sys.exit(min(num_failed, 1))
