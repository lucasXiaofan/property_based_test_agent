"""
Test script for pandas DataFrame.round() vs numpy.round() consistency.
Focus: Comparing behavior including both results AND error handling.
Goal: Find inconsistencies where pandas and numpy behave differently.
"""

import pandas as pd
import numpy as np
import sys
from typing import Any, Callable


class TestResult:
    """Store test results for comparison"""
    def __init__(self, name: str):
        self.name = name
        self.pandas_error = None
        self.numpy_error = None
        self.pandas_result = None
        self.numpy_result = None
        self.passed = False
        self.message = ""


def compare_behavior(test_name: str, pandas_func: Callable, numpy_func: Callable) -> TestResult:
    """
    Compare pandas and numpy behavior for both results and errors.
    Returns TestResult with detailed comparison.
    """
    result = TestResult(test_name)

    # Try pandas
    try:
        result.pandas_result = pandas_func()
    except Exception as e:
        result.pandas_error = (type(e).__name__, str(e))

    # Try numpy
    try:
        result.numpy_result = numpy_func()
    except Exception as e:
        result.numpy_error = (type(e).__name__, str(e))

    # Compare results
    if result.pandas_error and result.numpy_error:
        # Both raised errors - check if they're the same type
        if result.pandas_error[0] == result.numpy_error[0]:
            result.passed = True
            result.message = f"Both raised {result.pandas_error[0]}"
        else:
            result.passed = False
            result.message = f"INCONSISTENT ERRORS: pandas raised {result.pandas_error[0]}, numpy raised {result.numpy_error[0]}"
    elif result.pandas_error and not result.numpy_error:
        result.passed = False
        result.message = f"BUG: pandas raised {result.pandas_error[0]}, but numpy succeeded with {result.numpy_result}"
    elif not result.pandas_error and result.numpy_error:
        result.passed = False
        result.message = f"BUG: numpy raised {result.numpy_error[0]}, but pandas succeeded with {result.pandas_result}"
    else:
        # Both succeeded - compare results
        try:
            if isinstance(result.pandas_result, pd.DataFrame):
                pandas_values = result.pandas_result.values
            else:
                pandas_values = result.pandas_result

            if np.array_equal(pandas_values, result.numpy_result, equal_nan=True):
                result.passed = True
                result.message = "Results match"
            elif np.allclose(pandas_values, result.numpy_result, equal_nan=True, rtol=1e-10):
                result.passed = True
                result.message = "Results match (within tolerance)"
            else:
                result.passed = False
                result.message = f"BUG: Results differ - pandas: {pandas_values}, numpy: {result.numpy_result}"
        except Exception as e:
            result.passed = False
            result.message = f"Error comparing results: {e}"

    return result


def test_basic_float_rounding():
    """Test basic float rounding"""
    print("\n=== Test 1: Basic Float Rounding ===")
    arr = np.array([[1.234, 2.567], [3.891, 4.123]])
    df = pd.DataFrame(arr, columns=['A', 'B'])

    for decimals in [0, 1, 2, 3]:
        result = compare_behavior(
            f"Basic float with {decimals} decimals",
            lambda d=decimals: df.round(d).values,
            lambda d=decimals: np.round(arr, d)
        )
        print(f"  decimals={decimals}: {result.message}")
        if not result.passed:
            print(f"    Pandas: {result.pandas_result}")
            print(f"    NumPy:  {result.numpy_result}")
            return False
    return True


def test_negative_decimals():
    """Test negative decimals (rounding to tens, hundreds)"""
    print("\n=== Test 2: Negative Decimals ===")
    arr = np.array([[123.456, 789.123], [456.789, 321.654]])
    df = pd.DataFrame(arr, columns=['A', 'B'])

    bugs_found = []
    for decimals in [-1, -2, -3]:
        result = compare_behavior(
            f"Negative decimals {decimals}",
            lambda d=decimals: df.round(d).values,
            lambda d=decimals: np.round(arr, d)
        )
        print(f"  decimals={decimals}: {result.message}")
        if not result.passed:
            print(f"    Pandas: {result.pandas_result}")
            print(f"    NumPy:  {result.numpy_result}")
            bugs_found.append(result)

    return len(bugs_found) == 0


def test_integer_arrays():
    """Test integer arrays"""
    print("\n=== Test 3: Integer Arrays ===")
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    df = pd.DataFrame(arr, columns=['A', 'B'])

    bugs_found = []
    for decimals in [0, 1, -1]:
        result = compare_behavior(
            f"Integer with decimals={decimals}",
            lambda d=decimals: df.round(d).values,
            lambda d=decimals: np.round(arr, d)
        )
        print(f"  decimals={decimals}: {result.message}")
        if not result.passed:
            print(f"    Pandas: {result.pandas_result}")
            print(f"    NumPy:  {result.numpy_result}")
            print(f"    Pandas dtype: {df.round(decimals).values.dtype}")
            print(f"    NumPy dtype:  {np.round(arr, decimals).dtype}")
            bugs_found.append(result)

    return len(bugs_found) == 0


def test_negative_decimals_on_integers():
    """Test negative decimals on integers - important edge case"""
    print("\n=== Test 4: Negative Decimals on Integers ===")
    arr = np.array([[123, 789], [456, 321]], dtype=np.int64)
    df = pd.DataFrame(arr, columns=['A', 'B'])

    bugs_found = []
    for decimals in [-1, -2]:
        result = compare_behavior(
            f"Integer with negative decimals={decimals}",
            lambda d=decimals: df.round(d).values,
            lambda d=decimals: np.round(arr, d)
        )
        print(f"  decimals={decimals}: {result.message}")
        if not result.passed:
            print(f"    Pandas result: {result.pandas_result}")
            print(f"    NumPy result:  {result.numpy_result}")
            print(f"    Pandas dtype: {df.round(decimals).values.dtype}")
            print(f"    NumPy dtype:  {np.round(arr, decimals).dtype}")
            bugs_found.append(result)

    return len(bugs_found) == 0


def test_nan_values():
    """Test NaN handling"""
    print("\n=== Test 5: NaN Values ===")
    arr = np.array([[1.234, np.nan], [np.nan, 4.567]])
    df = pd.DataFrame(arr, columns=['A', 'B'])

    result = compare_behavior(
        "NaN values",
        lambda: df.round(2).values,
        lambda: np.round(arr, 2)
    )
    print(f"  {result.message}")
    if not result.passed:
        print(f"    Pandas: {result.pandas_result}")
        print(f"    NumPy:  {result.numpy_result}")
    return result.passed


def test_inf_values():
    """Test infinity handling"""
    print("\n=== Test 6: Infinity Values ===")
    arr = np.array([[np.inf, -np.inf], [1.234, 5.678]])
    df = pd.DataFrame(arr, columns=['A', 'B'])

    result = compare_behavior(
        "Inf values",
        lambda: df.round(2).values,
        lambda: np.round(arr, 2)
    )
    print(f"  {result.message}")
    if not result.passed:
        print(f"    Pandas: {result.pandas_result}")
        print(f"    NumPy:  {result.numpy_result}")
    return result.passed


def test_negative_zero():
    """Test negative zero preservation"""
    print("\n=== Test 7: Negative Zero ===")
    arr = np.array([[-0.0, -0.001], [-0.0001, 0.0001]])
    df = pd.DataFrame(arr, columns=['A', 'B'])

    result = compare_behavior(
        "Negative zero",
        lambda: df.round(2).values,
        lambda: np.round(arr, 2)
    )

    # Check sign bits
    pandas_signs = np.signbit(result.pandas_result) if result.pandas_result is not None else None
    numpy_signs = np.signbit(result.numpy_result) if result.numpy_result is not None else None

    print(f"  {result.message}")
    if pandas_signs is not None and numpy_signs is not None:
        if not np.array_equal(pandas_signs, numpy_signs):
            print(f"    BUG: Sign bits differ!")
            print(f"    Pandas signs: {pandas_signs}")
            print(f"    NumPy signs:  {numpy_signs}")
            return False
    return result.passed


def test_bankers_rounding():
    """Test round-half-to-even (banker's rounding)"""
    print("\n=== Test 8: Banker's Rounding (Round Half to Even) ===")
    arr = np.array([[0.5, 1.5, 2.5, 3.5, 4.5]])
    df = pd.DataFrame(arr, columns=['A', 'B', 'C', 'D', 'E'])

    result = compare_behavior(
        "Banker's rounding",
        lambda: df.round(0).values,
        lambda: np.round(arr, 0)
    )
    print(f"  {result.message}")
    if not result.passed:
        print(f"    Pandas: {result.pandas_result}")
        print(f"    NumPy:  {result.numpy_result}")
        print("    Expected: [0, 2, 2, 4, 4] (round half to even)")
    return result.passed


def test_complex_numbers():
    """Test complex number rounding"""
    print("\n=== Test 9: Complex Numbers ===")
    arr = np.array([[1.234 + 2.567j, 3.891 + 4.123j]])
    df = pd.DataFrame(arr, columns=['A', 'B'])

    result = compare_behavior(
        "Complex numbers",
        lambda: df.round(2).values,
        lambda: np.round(arr, 2)
    )
    print(f"  {result.message}")
    if result.pandas_error:
        print(f"    Pandas error: {result.pandas_error}")
    if result.numpy_error:
        print(f"    NumPy error: {result.numpy_error}")
    if not result.passed:
        if result.pandas_result is not None:
            print(f"    Pandas: {result.pandas_result}")
        if result.numpy_result is not None:
            print(f"    NumPy:  {result.numpy_result}")
    return result.passed


def test_very_large_decimals():
    """Test very large decimal parameter"""
    print("\n=== Test 10: Very Large Decimals ===")
    arr = np.array([[1.23456789012345, 2.34567890123456]])
    df = pd.DataFrame(arr, columns=['A', 'B'])

    bugs_found = []
    for decimals in [10, 15, 20, 50]:
        result = compare_behavior(
            f"Large decimals={decimals}",
            lambda d=decimals: df.round(d).values,
            lambda d=decimals: np.round(arr, d)
        )
        print(f"  decimals={decimals}: {result.message}")
        if not result.passed:
            bugs_found.append(result)

    return len(bugs_found) == 0


def test_float32_precision():
    """Test float32 precision"""
    print("\n=== Test 11: Float32 Precision ===")
    arr = np.array([[1.234567, 2.345678]], dtype=np.float32)
    df = pd.DataFrame(arr, columns=['A', 'B'])

    result = compare_behavior(
        "Float32",
        lambda: df.round(3).values,
        lambda: np.round(arr, 3)
    )
    print(f"  {result.message}")
    if not result.passed:
        print(f"    Pandas: {result.pandas_result} (dtype: {df.round(3).values.dtype})")
        print(f"    NumPy:  {result.numpy_result} (dtype: {np.round(arr, 3).dtype})")
    return result.passed


def test_very_small_numbers():
    """Test very small numbers near machine epsilon"""
    print("\n=== Test 12: Very Small Numbers ===")
    arr = np.array([[1e-10, 1e-15], [1e-20, 1.23456789e-8]])
    df = pd.DataFrame(arr, columns=['A', 'B'])

    bugs_found = []
    for decimals in [10, 15, 20]:
        result = compare_behavior(
            f"Small numbers with decimals={decimals}",
            lambda d=decimals: df.round(d).values,
            lambda d=decimals: np.round(arr, d)
        )
        print(f"  decimals={decimals}: {result.message}")
        if not result.passed:
            bugs_found.append(result)

    return len(bugs_found) == 0


def test_invalid_decimals_type():
    """Test invalid decimals parameter types"""
    print("\n=== Test 13: Invalid Decimals Type ===")
    arr = np.array([[1.234, 2.567]])
    df = pd.DataFrame(arr, columns=['A', 'B'])

    bugs_found = []

    # Test with string
    print("  Testing string '2':")
    result = compare_behavior(
        "String decimals",
        lambda: df.round("2"),
        lambda: np.round(arr, "2")
    )
    print(f"    {result.message}")
    if not result.passed:
        bugs_found.append(result)
        if result.pandas_error:
            print(f"    Pandas: {result.pandas_error}")
        if result.numpy_error:
            print(f"    NumPy:  {result.numpy_error}")

    # Test with float
    print("  Testing float 2.5:")
    result = compare_behavior(
        "Float decimals",
        lambda: df.round(2.5),
        lambda: np.round(arr, 2.5)
    )
    print(f"    {result.message}")
    if not result.passed:
        bugs_found.append(result)
        if result.pandas_error:
            print(f"    Pandas: {result.pandas_error}")
        if result.numpy_error:
            print(f"    NumPy:  {result.numpy_error}")

    # Test with None
    print("  Testing None:")
    result = compare_behavior(
        "None decimals",
        lambda: df.round(None),
        lambda: np.round(arr, None)
    )
    print(f"    {result.message}")
    if not result.passed:
        bugs_found.append(result)
        if result.pandas_error:
            print(f"    Pandas: {result.pandas_error}")
        if result.numpy_error:
            print(f"    NumPy:  {result.numpy_error}")

    return len(bugs_found) == 0


def test_empty_array():
    """Test empty array"""
    print("\n=== Test 14: Empty Array ===")
    arr = np.array([]).reshape(0, 2)
    df = pd.DataFrame(arr, columns=['A', 'B'])

    result = compare_behavior(
        "Empty array",
        lambda: df.round(2).values,
        lambda: np.round(arr, 2)
    )
    print(f"  {result.message}")
    if not result.passed:
        print(f"    Pandas shape: {result.pandas_result.shape if result.pandas_result is not None else 'N/A'}")
        print(f"    NumPy shape:  {result.numpy_result.shape if result.numpy_result is not None else 'N/A'}")
    return result.passed


def test_single_value():
    """Test single value"""
    print("\n=== Test 15: Single Value ===")
    arr = np.array([[1.23456]])
    df = pd.DataFrame(arr, columns=['A'])

    result = compare_behavior(
        "Single value",
        lambda: df.round(2).values,
        lambda: np.round(arr, 2)
    )
    print(f"  {result.message}")
    return result.passed


def test_large_array():
    """Test large array"""
    print("\n=== Test 16: Large Array ===")
    np.random.seed(42)
    arr = np.random.randn(100, 10)
    df = pd.DataFrame(arr)

    result = compare_behavior(
        "Large array (100x10)",
        lambda: df.round(2).values,
        lambda: np.round(arr, 2)
    )
    print(f"  {result.message}")
    return result.passed


def test_various_dtypes():
    """Test various numpy dtypes"""
    print("\n=== Test 17: Various Dtypes ===")

    bugs_found = []

    # int32
    arr_int32 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    df_int32 = pd.DataFrame(arr_int32)
    result = compare_behavior(
        "int32",
        lambda: df_int32.round(0).values,
        lambda: np.round(arr_int32, 0)
    )
    print(f"  int32: {result.message}")
    if not result.passed:
        bugs_found.append(result)
        print(f"    Pandas dtype: {df_int32.round(0).values.dtype}")
        print(f"    NumPy dtype:  {np.round(arr_int32, 0).dtype}")

    # uint8
    arr_uint8 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    df_uint8 = pd.DataFrame(arr_uint8)
    result = compare_behavior(
        "uint8",
        lambda: df_uint8.round(0).values,
        lambda: np.round(arr_uint8, 0)
    )
    print(f"  uint8: {result.message}")
    if not result.passed:
        bugs_found.append(result)
        print(f"    Pandas dtype: {df_uint8.round(0).values.dtype}")
        print(f"    NumPy dtype:  {np.round(arr_uint8, 0).dtype}")

    # float16
    arr_float16 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16)
    df_float16 = pd.DataFrame(arr_float16)
    result = compare_behavior(
        "float16",
        lambda: df_float16.round(0).values,
        lambda: np.round(arr_float16, 0)
    )
    print(f"  float16: {result.message}")
    if not result.passed:
        bugs_found.append(result)
        print(f"    Pandas: {df_float16.round(0).values}")
        print(f"    NumPy:  {np.round(arr_float16, 0)}")

    return len(bugs_found) == 0


def test_mixed_signs():
    """Test mixed positive and negative numbers"""
    print("\n=== Test 18: Mixed Signs ===")
    arr = np.array([[-1.5, -0.5, 0.5, 1.5], [-2.5, -1.5, 1.5, 2.5]])
    df = pd.DataFrame(arr)

    result = compare_behavior(
        "Mixed signs with banker's rounding",
        lambda: df.round(0).values,
        lambda: np.round(arr, 0)
    )
    print(f"  {result.message}")
    if not result.passed:
        print(f"    Pandas: {result.pandas_result}")
        print(f"    NumPy:  {result.numpy_result}")
    return result.passed


def test_all_zeros():
    """Test array of all zeros"""
    print("\n=== Test 19: All Zeros ===")
    arr = np.array([[0.0, 0.0], [0.0, 0.0]])
    df = pd.DataFrame(arr)

    result = compare_behavior(
        "All zeros",
        lambda: df.round(2).values,
        lambda: np.round(arr, 2)
    )
    print(f"  {result.message}")
    return result.passed


def test_extreme_values():
    """Test extreme values"""
    print("\n=== Test 20: Extreme Values ===")
    arr = np.array([[1e100, 1e-100], [1e200, 1e-200]])
    df = pd.DataFrame(arr)

    result = compare_behavior(
        "Extreme values",
        lambda: df.round(2).values,
        lambda: np.round(arr, 2)
    )
    print(f"  {result.message}")
    if not result.passed:
        print(f"    Pandas: {result.pandas_result}")
        print(f"    NumPy:  {result.numpy_result}")
    return result.passed


def main():
    """Run all tests"""
    print("=" * 80)
    print("Testing pandas DataFrame.round() vs numpy.round() Consistency")
    print("Comparing both RESULTS and ERROR BEHAVIOR")
    print("=" * 80)

    tests = [
        test_basic_float_rounding,
        test_negative_decimals,
        test_integer_arrays,
        test_negative_decimals_on_integers,
        test_nan_values,
        test_inf_values,
        test_negative_zero,
        test_bankers_rounding,
        test_complex_numbers,
        test_very_large_decimals,
        test_float32_precision,
        test_very_small_numbers,
        test_invalid_decimals_type,
        test_empty_array,
        test_single_value,
        test_large_array,
        test_various_dtypes,
        test_mixed_signs,
        test_all_zeros,
        test_extreme_values,
    ]

    passed = 0
    failed = 0
    bugs_found = []

    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✓ {test.__name__} - CONSISTENT")
            else:
                failed += 1
                bugs_found.append(test.__name__)
                print(f"✗ {test.__name__} - INCONSISTENCY FOUND")
        except Exception as e:
            failed += 1
            bugs_found.append(test.__name__)
            print(f"✗ {test.__name__} - ERROR: {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
    print(f"Test Summary: {passed} passed, {failed} failed out of {passed + failed} tests")
    if bugs_found:
        print(f"\nBugs/Inconsistencies found in:")
        for bug in bugs_found:
            print(f"  - {bug}")
    else:
        print("\nNo inconsistencies found! pandas.round() matches numpy.round() behavior.")
    print("=" * 80)

    return len(bugs_found)


if __name__ == "__main__":
    num_bugs = main()
    sys.exit(min(num_bugs, 1))  # Exit with 1 if any bugs found, 0 otherwise
