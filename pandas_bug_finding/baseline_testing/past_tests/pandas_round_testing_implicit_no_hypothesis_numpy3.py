"""
Test script for pandas DataFrame.round() vs numpy.round() with unusual dtypes.
Focus: Object dtype and other unusual types - compare pandas with numpy behavior.
Goal: Find inconsistencies in how pandas and numpy handle non-standard data types.
"""

import pandas as pd
import numpy as np
import sys
from decimal import Decimal
from fractions import Fraction


def print_separator(char='=', length=80):
    print(char * length)


def compare_round_behavior(test_name, arr, df, decimals):
    """Compare pandas and numpy round behavior"""
    print(f"\nTest: {test_name}")
    print(f"Array dtype: {arr.dtype}, DataFrame dtype: {df.dtypes.tolist()}")
    print(f"Data: {arr.tolist() if arr.ndim == 1 else arr}")

    pandas_error = None
    numpy_error = None
    pandas_result = None
    numpy_result = None

    # Try NumPy
    try:
        numpy_result = np.round(arr, decimals)
        print(f"✓ NumPy succeeded: {numpy_result}")
    except Exception as e:
        numpy_error = (type(e).__name__, str(e))
        print(f"✓ NumPy raised: {numpy_error[0]}: {numpy_error[1]}")

    # Try Pandas
    try:
        pandas_result = df.round(decimals)
        print(f"✓ Pandas succeeded: {pandas_result.values}")
    except Exception as e:
        pandas_error = (type(e).__name__, str(e))
        print(f"✓ Pandas raised: {pandas_error[0]}: {pandas_error[1]}")

    # Compare results
    if pandas_error and numpy_error:
        if pandas_error[0] == numpy_error[0]:
            print("✓ CONSISTENT: Both raised same error type")
            return True
        else:
            print(f"✗ BUG: Different errors! Pandas: {pandas_error[0]}, NumPy: {numpy_error[0]}")
            return False
    elif pandas_error and not numpy_error:
        print(f"✗ BUG: Pandas raised {pandas_error[0]}, but NumPy succeeded!")
        return False
    elif not pandas_error and numpy_error:
        print(f"✗ BUG: NumPy raised {numpy_error[0]}, but Pandas succeeded!")
        return False
    else:
        # Both succeeded - compare results
        try:
            if np.array_equal(pandas_result.values, numpy_result, equal_nan=True):
                print("✓ CONSISTENT: Results match exactly")
                return True
            else:
                print(f"✗ BUG: Results differ!")
                print(f"  Pandas: {pandas_result.values}")
                print(f"  NumPy:  {numpy_result}")
                return False
        except Exception as e:
            print(f"✗ Error comparing: {e}")
            return False


def test_object_dtype_pure_strings():
    """Test object dtype with pure strings"""
    print_separator()
    print("TEST 1: Object Dtype - Pure Strings")
    print_separator()

    arr = np.array(['hello', 'world', 'test'], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Pure strings", arr, df, 2)


def test_object_dtype_mixed_numeric_string():
    """Test object dtype with mixed numbers and strings"""
    print_separator()
    print("TEST 2: Object Dtype - Mixed Numbers and Strings")
    print_separator()

    arr = np.array([1.234, 'string', 3.456], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Mixed numeric and string", arr, df, 2)


def test_object_dtype_pure_floats():
    """Test object dtype containing only floats"""
    print_separator()
    print("TEST 3: Object Dtype - Pure Floats (stored as objects)")
    print_separator()

    arr = np.array([1.234, 2.567, 3.891], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Pure floats as objects", arr, df, 2)


def test_object_dtype_pure_ints():
    """Test object dtype containing only integers"""
    print_separator()
    print("TEST 4: Object Dtype - Pure Integers (stored as objects)")
    print_separator()

    arr = np.array([1, 2, 3], dtype=object)
    df = pd.DataFrame({'A': arr})

    result = compare_round_behavior("Pure ints as objects", arr, df, 2)

    # Also test with negative decimals
    print("\nWith negative decimals:")
    arr2 = np.array([123, 456, 789], dtype=object)
    df2 = pd.DataFrame({'A': arr2})
    result2 = compare_round_behavior("Ints with decimals=-1", arr2, df2, -1)

    return result and result2


def test_object_dtype_none_values():
    """Test object dtype with None values"""
    print_separator()
    print("TEST 5: Object Dtype - None Values")
    print_separator()

    arr = np.array([1.234, None, 3.456], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Object with None", arr, df, 2)


def test_object_dtype_nan_values():
    """Test object dtype with NaN values"""
    print_separator()
    print("TEST 6: Object Dtype - NaN Values")
    print_separator()

    arr = np.array([1.234, np.nan, 3.456], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Object with NaN", arr, df, 2)


def test_object_dtype_lists():
    """Test object dtype with lists"""
    print_separator()
    print("TEST 7: Object Dtype - Lists in Cells")
    print_separator()

    arr = np.array([[1.234, 2.567], [3.891, 4.123]], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Lists in cells", arr, df, 2)


def test_object_dtype_dicts():
    """Test object dtype with dicts"""
    print_separator()
    print("TEST 8: Object Dtype - Dicts in Cells")
    print_separator()

    arr = np.array([{'value': 1.234}, {'value': 2.567}], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Dicts in cells", arr, df, 2)


def test_object_dtype_decimal():
    """Test object dtype with Decimal objects"""
    print_separator()
    print("TEST 9: Object Dtype - Decimal Objects")
    print_separator()

    arr = np.array([Decimal('1.234'), Decimal('2.567'), Decimal('3.891')], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Decimal objects", arr, df, 2)


def test_object_dtype_fraction():
    """Test object dtype with Fraction objects"""
    print_separator()
    print("TEST 10: Object Dtype - Fraction Objects")
    print_separator()

    arr = np.array([Fraction(1, 3), Fraction(2, 3), Fraction(5, 6)], dtype=object)
    df = pd.DataFrame({'A': arr})

    print(f"As decimals: {[float(x) for x in arr]}")
    return compare_round_behavior("Fraction objects", arr, df, 2)


def test_object_dtype_boolean():
    """Test object dtype with boolean values"""
    print_separator()
    print("TEST 11: Object Dtype - Boolean Values")
    print_separator()

    arr = np.array([True, False, True], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Booleans as objects", arr, df, 2)


def test_object_dtype_mixed_numeric_types():
    """Test object dtype with mixed int and float"""
    print_separator()
    print("TEST 12: Object Dtype - Mixed Int and Float")
    print_separator()

    arr = np.array([1, 2.5, 3, 4.7], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Mixed int and float", arr, df, 1)


def test_object_dtype_complex():
    """Test object dtype with complex numbers"""
    print_separator()
    print("TEST 13: Object Dtype - Complex Numbers")
    print_separator()

    arr = np.array([1.234+2.567j, 3.891+4.123j], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Complex as objects", arr, df, 2)


def test_object_dtype_bytes():
    """Test object dtype with bytes"""
    print_separator()
    print("TEST 14: Object Dtype - Bytes")
    print_separator()

    arr = np.array([b'1.234', b'2.567', b'3.891'], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Bytes", arr, df, 2)


def test_object_dtype_empty_strings():
    """Test object dtype with empty strings"""
    print_separator()
    print("TEST 15: Object Dtype - Empty Strings")
    print_separator()

    arr = np.array(['', '', ''], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Empty strings", arr, df, 2)


def test_object_dtype_inf():
    """Test object dtype with infinity"""
    print_separator()
    print("TEST 16: Object Dtype - Infinity")
    print_separator()

    arr = np.array([np.inf, -np.inf, 1.234], dtype=object)
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Inf as objects", arr, df, 2)


def test_object_dtype_negative_zero():
    """Test object dtype with negative zero"""
    print_separator()
    print("TEST 17: Object Dtype - Negative Zero")
    print_separator()

    arr = np.array([-0.0, 0.0, -0.001], dtype=object)
    df = pd.DataFrame({'A': arr})

    result = compare_round_behavior("Negative zero", arr, df, 2)

    # Check sign bits if both succeeded
    print("\nChecking sign bits:")
    try:
        np_result = np.round(arr, 2)
        pd_result = df.round(2).values

        np_signs = [np.signbit(float(x)) if isinstance(x, (int, float)) else None for x in np_result]
        pd_signs = [np.signbit(float(x)) if isinstance(x, (int, float)) else None for x in pd_result]

        print(f"NumPy sign bits: {np_signs}")
        print(f"Pandas sign bits: {pd_signs}")

        if np_signs == pd_signs:
            print("✓ Sign bits match")
        else:
            print("✗ BUG: Sign bits differ!")
            return False
    except:
        pass

    return result


def test_string_dtype():
    """Test pandas StringDtype (not available in numpy)"""
    print_separator()
    print("TEST 18: Pandas StringDtype (pandas-specific)")
    print_separator()

    df = pd.DataFrame({'A': pd.array(['1.234', '2.567', '3.891'], dtype='string')})
    print(f"DataFrame dtype: {df['A'].dtype}")
    print(f"Data: {df['A'].tolist()}")

    try:
        result = df.round(2)
        print(f"✓ Pandas succeeded: {result['A'].tolist()}")
        print(f"Strings preserved: {df['A'].equals(result['A'])}")
    except Exception as e:
        print(f"✓ Pandas raised: {type(e).__name__}: {e}")

    print("\nNote: NumPy doesn't have StringDtype, so no comparison possible")
    return True


def test_boolean_dtype():
    """Test boolean arrays"""
    print_separator()
    print("TEST 19: Boolean Dtype")
    print_separator()

    arr = np.array([True, False, True, False])
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Standard boolean", arr, df, 2)


def test_nullable_boolean_dtype():
    """Test pandas nullable boolean (not in numpy)"""
    print_separator()
    print("TEST 20: Nullable Boolean Dtype (pandas-specific)")
    print_separator()

    df = pd.DataFrame({'A': pd.array([True, False, pd.NA], dtype='boolean')})
    print(f"DataFrame dtype: {df['A'].dtype}")
    print(f"Data: {df['A'].tolist()}")

    try:
        result = df.round(2)
        print(f"✓ Pandas succeeded: {result['A'].tolist()}")
    except Exception as e:
        print(f"✓ Pandas raised: {type(e).__name__}: {e}")

    print("\nNote: NumPy doesn't have nullable boolean, so no comparison possible")
    return True


def test_datetime_dtype():
    """Test datetime arrays"""
    print_separator()
    print("TEST 21: Datetime Dtype")
    print_separator()

    arr = np.array(['2020-01-01', '2020-12-31'], dtype='datetime64[D]')
    df = pd.DataFrame({'A': pd.to_datetime(arr)})

    print(f"NumPy dtype: {arr.dtype}, Pandas dtype: {df['A'].dtype}")
    print(f"Data: {arr}")

    numpy_error = None
    pandas_error = None

    try:
        numpy_result = np.round(arr, 2)
        print(f"✓ NumPy succeeded: {numpy_result}")
    except Exception as e:
        numpy_error = (type(e).__name__, str(e))
        print(f"✓ NumPy raised: {numpy_error[0]}: {numpy_error[1]}")

    try:
        pandas_result = df.round(2)
        print(f"✓ Pandas succeeded, datetime preserved: {df['A'].equals(pandas_result['A'])}")
    except Exception as e:
        pandas_error = (type(e).__name__, str(e))
        print(f"✓ Pandas raised: {pandas_error[0]}: {pandas_error[1]}")

    if (pandas_error and numpy_error and pandas_error[0] == numpy_error[0]):
        print("✓ CONSISTENT: Both raised same error")
        return True
    elif (not pandas_error and not numpy_error):
        print("✓ CONSISTENT: Both succeeded")
        return True
    else:
        print("✗ INCONSISTENT: Different behavior!")
        return False


def test_timedelta_dtype():
    """Test timedelta arrays"""
    print_separator()
    print("TEST 22: Timedelta Dtype")
    print_separator()

    arr = np.array([1, 2, 3], dtype='timedelta64[D]')
    df = pd.DataFrame({'A': pd.to_timedelta(arr, unit='D')})

    print(f"NumPy dtype: {arr.dtype}, Pandas dtype: {df['A'].dtype}")
    print(f"Data: {arr}")

    numpy_error = None
    pandas_error = None

    try:
        numpy_result = np.round(arr, 2)
        print(f"✓ NumPy succeeded: {numpy_result}")
    except Exception as e:
        numpy_error = (type(e).__name__, str(e))
        print(f"✓ NumPy raised: {numpy_error[0]}: {numpy_error[1]}")

    try:
        pandas_result = df.round(2)
        print(f"✓ Pandas succeeded, timedelta preserved: {df['A'].equals(pandas_result['A'])}")
    except Exception as e:
        pandas_error = (type(e).__name__, str(e))
        print(f"✓ Pandas raised: {pandas_error[0]}: {pandas_error[1]}")

    if (pandas_error and numpy_error and pandas_error[0] == numpy_error[0]):
        print("✓ CONSISTENT: Both raised same error")
        return True
    elif (not pandas_error and not numpy_error):
        print("✓ CONSISTENT: Both succeeded")
        return True
    else:
        print("✗ INCONSISTENT: Different behavior!")
        return False


def test_unicode_dtype():
    """Test Unicode string dtype"""
    print_separator()
    print("TEST 23: Unicode String Dtype")
    print_separator()

    arr = np.array(['中文', '🚀', 'test'], dtype='U10')
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Unicode strings", arr, df, 2)


def test_structured_array():
    """Test structured numpy array"""
    print_separator()
    print("TEST 24: Structured Array (numpy-specific)")
    print_separator()

    arr = np.array([(1.234, 'a'), (2.567, 'b')],
                   dtype=[('value', 'f8'), ('label', 'U1')])

    print(f"Structured array dtype: {arr.dtype}")
    print(f"Data: {arr}")

    try:
        numpy_result = np.round(arr, 2)
        print(f"✓ NumPy succeeded: {numpy_result}")
    except Exception as e:
        print(f"✓ NumPy raised: {type(e).__name__}: {e}")

    # Create DataFrame from structured array
    df = pd.DataFrame(arr)
    print(f"\nDataFrame from structured array:")
    print(df)
    print(f"Dtypes: {df.dtypes.to_dict()}")

    try:
        pandas_result = df.round(2)
        print(f"✓ Pandas succeeded:")
        print(pandas_result)
    except Exception as e:
        print(f"✓ Pandas raised: {type(e).__name__}: {e}")

    return True


def test_void_dtype():
    """Test void dtype"""
    print_separator()
    print("TEST 25: Void Dtype")
    print_separator()

    arr = np.array([b'1234', b'5678'], dtype='V4')
    df = pd.DataFrame({'A': arr})

    return compare_round_behavior("Void dtype", arr, df, 2)


def main():
    """Run all tests"""
    print_separator('=')
    print("pandas DataFrame.round() vs numpy.round()")
    print("Testing Object Dtype and Unusual Types")
    print_separator('=')

    tests = [
        test_object_dtype_pure_strings,
        test_object_dtype_mixed_numeric_string,
        test_object_dtype_pure_floats,
        test_object_dtype_pure_ints,
        test_object_dtype_none_values,
        test_object_dtype_nan_values,
        test_object_dtype_lists,
        test_object_dtype_dicts,
        test_object_dtype_decimal,
        test_object_dtype_fraction,
        test_object_dtype_boolean,
        test_object_dtype_mixed_numeric_types,
        test_object_dtype_complex,
        test_object_dtype_bytes,
        test_object_dtype_empty_strings,
        test_object_dtype_inf,
        test_object_dtype_negative_zero,
        test_string_dtype,
        test_boolean_dtype,
        test_nullable_boolean_dtype,
        test_datetime_dtype,
        test_timedelta_dtype,
        test_unicode_dtype,
        test_structured_array,
        test_void_dtype,
    ]

    passed = 0
    failed = 0
    bugs_found = []

    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✓✓✓ {test.__name__} - CONSISTENT\n")
            else:
                failed += 1
                bugs_found.append(test.__name__)
                print(f"✗✗✗ {test.__name__} - INCONSISTENCY FOUND\n")
        except Exception as e:
            failed += 1
            bugs_found.append(test.__name__)
            print(f"✗✗✗ {test.__name__} - ERROR: {type(e).__name__}: {e}\n")

    print_separator('=')
    print(f"FINAL SUMMARY: {passed} passed, {failed} failed out of {passed + failed} tests")
    if bugs_found:
        print(f"\nBugs/Inconsistencies found in:")
        for i, bug in enumerate(bugs_found, 1):
            print(f"  {i}. {bug}")
    else:
        print("\n🎉 No inconsistencies found!")
    print_separator('=')

    return len(bugs_found)


if __name__ == "__main__":
    num_bugs = main()
    sys.exit(min(num_bugs, 1))
