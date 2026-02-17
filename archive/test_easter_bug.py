"""
Property-based testing for dateutil.easter using hypothesis and icontract.
This test file demonstrates the bug where easter() returns invalid dates for certain years.
"""

import sys
from pathlib import Path

# Add the dateutil src directory to the path
dateutil_src = Path(__file__).parent / "dateutil" / "src"
sys.path.insert(0, str(dateutil_src))

import datetime
from hypothesis import given, strategies as st, settings, example
from icontract import require, ensure, ViolationError
from dateutil import easter


# Define contracts using icontract
@ensure(lambda result: result.month in [3, 4], "Easter must fall in March or April")
@ensure(lambda result: result.weekday() == 6, "Easter must fall on Sunday (weekday=6)")
@ensure(
    lambda result: (result.month == 3 and result.day >= 22) or result.month == 4,
    "Easter cannot be before March 22"
)
@ensure(
    lambda result: (result.month == 4 and result.day <= 25) or result.month == 3,
    "Easter cannot be after April 25"
)
def easter_with_invariants(year: int, method: int) -> datetime.date:
    """
    Wrapper around easter() that enforces calendar invariants using icontract.

    Easter date invariants:
    1. Must fall between March 22 and April 25 (inclusive)
    2. Must fall on a Sunday (weekday() == 6)
    3. Must be in March or April only

    Args:
        year: Year to compute Easter for (1583-4099 for methods 2 and 3)
        method: Easter calculation method (1=Julian, 2=Orthodox, 3=Western)

    Returns:
        datetime.date: The date of Easter Sunday for the given year

    Raises:
        ViolationError: If the computed Easter date violates any invariant
    """
    return easter.easter(year, method)


@given(st.integers(min_value=1583, max_value=4099))
@settings(max_examples=1000)
@example(2480)  # Specific failing case from bug report
def test_easter_date_invariants(year):
    """
    Property-based test that Easter always falls within valid calendar constraints.

    This test verifies that for all years in the valid range (1583-4099),
    Easter dates satisfy the fundamental calendar invariants.
    """
    for method in [1, 2, 3]:
        try:
            easter_date = easter_with_invariants(year, method)

            # Additional explicit assertions for clarity
            assert easter_date.month in [3, 4], \
                f"Easter {year} (method {method}): {easter_date} - Month {easter_date.month} is not March or April"

            assert easter_date.weekday() == 6, \
                f"Easter {year} (method {method}): {easter_date} - Weekday {easter_date.weekday()} is not Sunday"

            # Check date range: March 22 - April 25
            if easter_date.month == 3:
                assert easter_date.day >= 22, \
                    f"Easter {year} (method {method}): {easter_date} - March date before 22nd"
            elif easter_date.month == 4:
                assert easter_date.day <= 25, \
                    f"Easter {year} (method {method}): {easter_date} - April date after 25th"

        except ViolationError as e:
            print(f"\n{'='*70}")
            print(f"INVARIANT VIOLATION DETECTED!")
            print(f"{'='*70}")
            print(f"Year: {year}")
            print(f"Method: {method} ({'Julian' if method == 1 else 'Orthodox' if method == 2 else 'Western'})")
            print(f"Error: {e}")
            print(f"{'='*70}\n")
            raise


def test_specific_bug_case_2480():
    """
    Test the specific bug case from the bug report: year 2480.

    Expected behavior:
    - Orthodox (method 2): Should be in March or April, on a Sunday
    - Julian (method 1): Should be on a Sunday

    Actual behavior (buggy):
    - Orthodox: Returns May 5, 2480 (outside valid range)
    - Julian: Returns April 19, 2480 on Friday (not Sunday)
    """
    year = 2480

    print(f"\n{'='*70}")
    print(f"Testing Year {year} - Known Bug Cases")
    print(f"{'='*70}\n")

    # Test Orthodox method (method 2)
    try:
        orthodox_easter = easter_with_invariants(year, method=2)
        print(f"Orthodox Easter {year}: {orthodox_easter} ✓")
    except ViolationError as e:
        orthodox_easter = easter.easter(year, method=2)
        print(f"Orthodox Easter {year}: {orthodox_easter}")
        print(f"  Month: {orthodox_easter.month} (expected: 3 or 4) ✗")
        print(f"  Weekday: {orthodox_easter.weekday()} (expected: 6 for Sunday)")
        print(f"  VIOLATION: {e}\n")
        raise

    # Test Julian method (method 1)
    try:
        julian_easter = easter_with_invariants(year, method=1)
        print(f"Julian Easter {year}: {julian_easter} ✓")
    except ViolationError as e:
        julian_easter = easter.easter(year, method=1)
        print(f"Julian Easter {year}: {julian_easter}")
        print(f"  Weekday: {julian_easter.weekday()} (expected: 6 for Sunday) ✗")
        print(f"  VIOLATION: {e}\n")
        raise

    # Test Western method (method 3) - for comparison
    try:
        western_easter = easter_with_invariants(year, method=3)
        print(f"Western Easter {year}: {western_easter} ✓")
    except ViolationError as e:
        western_easter = easter.easter(year, method=3)
        print(f"Western Easter {year}: {western_easter}")
        print(f"  VIOLATION: {e}\n")
        raise


def demonstrate_bug():
    """
    Demonstrate the bug without contracts to show raw output.
    """
    year = 2480

    print(f"\n{'='*70}")
    print(f"Bug Demonstration - Year {year} (Raw Output)")
    print(f"{'='*70}\n")

    orthodox_easter = easter.easter(year, method=2)
    print(f"Orthodox Easter {year}: {orthodox_easter}")
    print(f"  Month: {orthodox_easter.month} (should be 3 or 4)")
    print(f"  Weekday: {orthodox_easter.weekday()} (should be 6 for Sunday)")
    print(f"  Day name: {orthodox_easter.strftime('%A')}")

    julian_easter = easter.easter(year, method=1)
    print(f"\nJulian Easter {year}: {julian_easter}")
    print(f"  Month: {julian_easter.month} (should be 3 or 4)")
    print(f"  Weekday: {julian_easter.weekday()} (should be 6 for Sunday)")
    print(f"  Day name: {julian_easter.strftime('%A')}")

    western_easter = easter.easter(year, method=3)
    print(f"\nWestern Easter {year}: {western_easter}")
    print(f"  Month: {western_easter.month} (should be 3 or 4)")
    print(f"  Weekday: {western_easter.weekday()} (should be 6 for Sunday)")
    print(f"  Day name: {western_easter.strftime('%A')}")

    print(f"\n{'='*70}")
    print("BUGS DETECTED:")
    print(f"{'='*70}")

    if orthodox_easter.month not in [3, 4]:
        print(f"✗ Orthodox method returns {orthodox_easter.strftime('%B')} (month {orthodox_easter.month})")
        print(f"  Easter MUST be in March or April only")

    if julian_easter.weekday() != 6:
        print(f"✗ Julian method returns {julian_easter.strftime('%A')} (weekday {julian_easter.weekday()})")
        print(f"  Easter MUST be on Sunday (weekday 6)")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    # First, demonstrate the raw bug
    demonstrate_bug()

    # Then test with contracts to show violations
    print("\n" + "="*70)
    print("Running tests with icontract invariants...")
    print("="*70)

    try:
        test_specific_bug_case_2480()
    except ViolationError:
        print("\nTest failed as expected due to invariant violations!")

    # Run property-based tests
    print("\n" + "="*70)
    print("Running property-based tests with hypothesis...")
    print("="*70)

    try:
        test_easter_date_invariants()
        print("\nAll property-based tests passed!")
    except (AssertionError, ViolationError) as e:
        print(f"\nProperty-based test failed: {e}")
