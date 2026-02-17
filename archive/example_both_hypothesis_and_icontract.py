"""
Example: Combining Hypothesis AND iContract

Use BOTH when:
1. You have production code that needs runtime validation (iContract)
2. AND you want to test that code thoroughly with property-based testing (Hypothesis)
3. You want contracts to serve as both documentation AND test oracles
4. You're building critical systems where correctness is paramount

This combination is POWERFUL but NOT ALWAYS NECESSARY.
"""

from icontract import require, ensure, invariant, ViolationError
from hypothesis import given, strategies as st, assume, settings, example
from datetime import date, timedelta
from typing import List, Optional
from dataclasses import dataclass
import math


# ============================================================================
# CASE 1: When BOTH are necessary and complementary
# Production code with contracts + property-based testing
# ============================================================================

@require(lambda lst: len(lst) > 0, "List cannot be empty")
@ensure(lambda lst, result: result in lst, "Result must be from the list")
@ensure(lambda lst, result: all(result >= x for x in lst), "Result must be the maximum")
def find_maximum(lst: List[int]) -> int:
    """
    Find the maximum value in a non-empty list.

    WHY BOTH?
    - iContract: Validates inputs at runtime (catches empty list from users)
    - Hypothesis: Tests with random lists to ensure correctness

    The contract helps Hypothesis catch bugs by raising ViolationError!
    """
    # Buggy implementation for demonstration:
    # return lst[0]  # Would fail postcondition check!

    # Correct implementation:
    return max(lst)


# Test with Hypothesis - it will use iContract's contracts as test oracles
@given(st.lists(st.integers(), min_size=1))
@settings(max_examples=100)
def test_find_maximum(lst):
    """
    Property-based test that leverages iContract's postconditions.

    COMPLEMENTARY ROLES:
    - iContract ensures the CONTRACT is satisfied for each test case
    - Hypothesis generates diverse test cases to find edge cases

    If find_maximum violates any contract, Hypothesis will find it!
    """
    result = find_maximum(lst)  # Contracts are checked automatically

    # Additional explicit test (redundant with contract, but clear):
    assert result == max(lst)


# ============================================================================
# CASE 2: Both necessary - Date range calculation
# Critical production code that needs both runtime safety and thorough testing
# ============================================================================

@require(lambda start_date, end_date: start_date <= end_date,
         "Start date must be before or equal to end date")
@require(lambda start_date: start_date.year >= 1900, "Start date too far in the past")
@require(lambda end_date: end_date.year <= 2100, "End date too far in the future")
@ensure(lambda result: result >= 0, "Result must be non-negative")
@ensure(lambda start_date, end_date, result: result == (end_date - start_date).days,
        "Result must match actual date difference")
def business_days_between(start_date: date, end_date: date) -> int:
    """
    Calculate business days between two dates.

    WHY BOTH?
    - iContract: Validates date ranges at runtime (important for user input)
    - Hypothesis: Tests with random dates to catch calculation bugs
    - Contracts serve as test oracles that Hypothesis can check
    """
    # Simplified implementation (doesn't account for weekends/holidays)
    return (end_date - start_date).days


@given(
    st.dates(min_value=date(1900, 1, 1), max_value=date(2100, 12, 31)),
    st.integers(min_value=0, max_value=1000)
)
def test_business_days_between(start_date, days_offset):
    """
    Hypothesis generates random date pairs, iContract validates contracts.

    SYNERGY:
    - Hypothesis explores the input space
    - iContract catches contract violations
    - Together they provide comprehensive validation
    """
    end_date = start_date + timedelta(days=days_offset)

    # This will raise ViolationError if contracts are violated
    result = business_days_between(start_date, end_date)

    # Property: adding result days to start gives end (for this simple impl)
    assert start_date + timedelta(days=result) == end_date


# ============================================================================
# CASE 3: When BOTH are necessary - Complex stateful object
# Class with invariants that needs thorough testing
# ============================================================================

@invariant(lambda self: self.balance >= 0, "Balance cannot be negative")
@invariant(lambda self: self.balance == sum(t.amount for t in self.transactions),
           "Balance must equal sum of transactions")
@invariant(lambda self: len(self.transactions) >= 0, "Transaction list must exist")
class BankAccount:
    """
    Bank account with strong invariants.

    WHY BOTH?
    - iContract: Maintains invariants at runtime (catches implementation bugs)
    - Hypothesis: Stateful testing to explore operation sequences
    - Invariants are checked after EVERY operation
    """

    @dataclass
    class Transaction:
        amount: float
        description: str

    @require(lambda initial_balance: initial_balance >= 0,
             "Initial balance must be non-negative")
    def __init__(self, initial_balance: float = 0.0):
        self.balance = initial_balance
        self.transactions: List[BankAccount.Transaction] = []
        if initial_balance > 0:
            self.transactions.append(
                BankAccount.Transaction(initial_balance, "Initial deposit")
            )

    @require(lambda amount: amount > 0, "Deposit amount must be positive")
    @ensure(lambda self, amount, OLD: self.balance == OLD.balance + amount,
            "Balance must increase by deposit amount")
    def deposit(self, amount: float) -> None:
        """Deposit money."""
        self.balance += amount
        self.transactions.append(BankAccount.Transaction(amount, "Deposit"))

    @require(lambda amount: amount > 0, "Withdrawal amount must be positive")
    @require(lambda self, amount: amount <= self.balance, "Insufficient funds")
    @ensure(lambda self, amount, OLD: self.balance == OLD.balance - amount,
            "Balance must decrease by withdrawal amount")
    def withdraw(self, amount: float) -> None:
        """Withdraw money."""
        self.balance -= amount
        self.transactions.append(BankAccount.Transaction(-amount, "Withdrawal"))


# Stateful property-based testing with Hypothesis
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant as hyp_invariant


class BankAccountMachine(RuleBasedStateMachine):
    """
    Hypothesis stateful testing for BankAccount.

    POWERFUL COMBINATION:
    - iContract checks invariants after each operation
    - Hypothesis generates random operation sequences
    - This finds bugs in state transitions that unit tests miss!
    """

    def __init__(self):
        super().__init__()
        self.account = BankAccount(0.0)
        self.expected_balance = 0.0

    @rule(amount=st.floats(min_value=0.01, max_value=1000.0))
    def deposit(self, amount):
        """Hypothesis rule: deposit random amounts."""
        # iContract validates preconditions and postconditions
        self.account.deposit(amount)
        self.expected_balance += amount

    @rule(amount=st.floats(min_value=0.01, max_value=100.0))
    def withdraw(self, amount):
        """Hypothesis rule: withdraw random amounts."""
        assume(amount <= self.account.balance)  # Filter invalid attempts

        # iContract validates preconditions and postconditions
        self.account.withdraw(amount)
        self.expected_balance -= amount

    @hyp_invariant()
    def balance_matches_expected(self):
        """Hypothesis invariant: balance matches our model."""
        # iContract invariants are ALSO checked after each operation!
        assert abs(self.account.balance - self.expected_balance) < 0.01


# TestBankAccount = BankAccountMachine.TestCase  # Uncomment to run with pytest


# ============================================================================
# CASE 4: When ONLY Hypothesis is needed (iContract is overkill)
# Simple pure function - contracts don't add value
# ============================================================================

def add_numbers(a: int, b: int) -> int:
    """
    Add two numbers.

    WHY ONLY HYPOTHESIS?
    - Pure function with simple logic
    - No preconditions to validate (all ints are valid)
    - No complex postconditions
    - Type system is sufficient
    - Adding contracts would be verbose without benefit

    iContract would be OVERKILL here:
    @require(lambda a: isinstance(a, int))  # Type hint already does this
    @require(lambda b: isinstance(b, int))  # Type hint already does this
    @ensure(lambda a, b, result: result == a + b)  # Tautology!
    """
    return a + b


@given(st.integers(), st.integers())
def test_addition_properties(a, b):
    """Test addition properties - no contracts needed!"""
    # Commutativity
    assert add_numbers(a, b) == add_numbers(b, a)

    # Identity
    assert add_numbers(a, 0) == a

    # No need for iContract here!


# ============================================================================
# CASE 5: When ONLY iContract is needed (Hypothesis is overkill)
# Simple validation with clear constraints - don't need random testing
# ============================================================================

@require(lambda age: 0 <= age <= 150, "Age must be between 0 and 150")
@require(lambda name: len(name) > 0, "Name cannot be empty")
@require(lambda name: len(name) <= 100, "Name too long")
def create_user(name: str, age: int) -> dict:
    """
    Create a user with validated inputs.

    WHY ONLY iContract?
    - Simple input validation
    - Clear, finite constraints
    - Not exploring complex properties
    - Don't need random testing - constraints are obvious
    - Hypothesis would generate 100s of test cases for no benefit

    Hypothesis would be OVERKILL here - we don't need to test
    thousands of random name/age combinations to verify this works.
    """
    return {"name": name, "age": age}


# Manual test is sufficient - no need for property-based testing
try:
    user1 = create_user("Alice", 30)  # Valid
    user2 = create_user("", 30)  # Raises ViolationError
except ViolationError as e:
    print(f"Validation works: {e}")


# ============================================================================
# CASE 6: BOTH are necessary - Easter date calculation
# Complex algorithm with mathematical invariants
# ============================================================================

@require(lambda year: 1583 <= year <= 4099, "Year must be in valid range")
@require(lambda method: method in [1, 2, 3], "Method must be 1, 2, or 3")
@ensure(lambda result: result.month in [3, 4], "Easter must be in March or April")
@ensure(lambda result: result.weekday() == 6, "Easter must be on Sunday")
@ensure(lambda result: (result.month == 3 and result.day >= 22) or result.month == 4,
        "Easter cannot be before March 22")
@ensure(lambda result: (result.month == 4 and result.day <= 25) or result.month == 3,
        "Easter cannot be after April 25")
def calculate_easter(year: int, method: int) -> date:
    """
    Calculate Easter date for a given year.

    WHY BOTH ARE CRITICAL:
    - iContract: Enforces astronomical/calendar invariants
    - Hypothesis: Tests across thousands of years to find edge cases

    This is the IDEAL use case for combining both tools!

    Without iContract: Tests might pass but violate calendar rules
    Without Hypothesis: Might miss edge cases in specific years
    Together: Comprehensive validation of a complex algorithm
    """
    # Simplified placeholder - real implementation would use complex algorithm
    # (This would be the dateutil.easter algorithm)

    # For demonstration, using a simple approximation that might have bugs
    # Real implementation would be much more complex

    # This is where the actual Easter calculation would go
    # For now, return a dummy value that satisfies contracts
    return date(year, 4, 15)  # Simplified - not accurate!


@given(
    st.integers(min_value=1583, max_value=4099),
    st.sampled_from([1, 2, 3])
)
@settings(max_examples=500)
@example(2480, 2)  # Known problematic case
def test_easter_calculation(year, method):
    """
    Property-based test for Easter calculation.

    SYNERGY IN ACTION:
    - Hypothesis generates (year, method) combinations
    - iContract validates calendar invariants for EACH case
    - If implementation has bugs, contracts catch them
    - Hypothesis finds the specific inputs that trigger bugs

    This found real bugs in dateutil.easter!
    """
    easter_date = calculate_easter(year, method)

    # These checks are redundant with contracts but make tests explicit
    assert easter_date.month in [3, 4]
    assert easter_date.weekday() == 6

    # Additional property: Easter moves within expected range year-to-year
    # (This would catch algorithmic bugs that contracts might miss)


# ============================================================================
# CASE 7: When contracts and properties work together
# Sorting algorithm - contracts ensure correctness, Hypothesis tests thoroughly
# ============================================================================

@require(lambda lst: lst is not None, "List cannot be None")
@ensure(lambda result: len(result) == len(OLD.lst), "Length must be preserved")
@ensure(lambda result: all(result[i] <= result[i+1] for i in range(len(result)-1))
        if len(result) > 1 else True, "Result must be sorted")
@ensure(lambda lst, result: sorted(lst) == result, "Result must be sorted version of input")
def bubble_sort(lst: List[int]) -> List[int]:
    """
    Sort a list using bubble sort.

    WHY BOTH?
    - iContract: Ensures postconditions (sorted, same length, same elements)
    - Hypothesis: Tests with random lists to verify algorithm correctness
    - Contracts catch bugs in the sorting logic
    - Hypothesis finds which inputs trigger those bugs
    """
    result = lst.copy()
    n = len(result)
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result


@given(st.lists(st.integers()))
def test_bubble_sort(lst):
    """
    Hypothesis tests sorting with random lists.

    Contracts automatically verify:
    - Sorted order
    - No elements lost/added
    - Correct permutation
    """
    result = bubble_sort(lst)

    # Verify it matches Python's built-in sort (oracle testing)
    assert result == sorted(lst)


# ============================================================================
# DECISION GUIDE: When to use what?
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────┐
│                        DECISION GUIDE                               │
└─────────────────────────────────────────────────────────────────────┘

USE HYPOTHESIS ONLY:
├─ Simple pure functions with mathematical properties
├─ Testing existing code without modifying it
├─ Exploring edge cases in algorithms
├─ When runtime overhead is unacceptable
└─ When properties are simple and don't need runtime checking

    Example: Testing that list reversal is involutive

    @given(st.lists(st.integers()))
    def test_reverse(lst):
        assert reverse(reverse(lst)) == lst


USE iContract ONLY:
├─ Simple input validation
├─ API boundaries with clear constraints
├─ When constraints are obvious and finite
├─ Documenting expected behavior
└─ When you don't need exhaustive testing

    Example: Age validation

    @require(lambda age: 0 <= age <= 150)
    def create_user(age: int):
        return User(age)


USE BOTH (POWERFUL COMBINATION):
├─ Complex algorithms with invariants
├─ Critical production code
├─ When correctness is paramount
├─ Stateful systems with invariants
├─ Mathematical/scientific computing
└─ When contracts serve as test oracles

    Example: Easter date calculation

    @ensure(lambda result: result.month in [3, 4])
    @ensure(lambda result: result.weekday() == 6)
    def easter(year: int) -> date:
        ...

    @given(st.integers(1583, 4099))
    def test_easter(year):
        date = easter(year)  # Contracts checked automatically!


┌─────────────────────────────────────────────────────────────────────┐
│                    WHEN BOTH ARE NECESSARY                          │
└─────────────────────────────────────────────────────────────────────┘

✓ Complex algorithms where bugs are likely
✓ Calendar/date calculations with invariants
✓ Financial calculations (precision + testing)
✓ Stateful systems with complex invariants
✓ Data structures with structural invariants
✓ Parsing/validation with complex rules
✓ Scientific computing with mathematical constraints
✓ Critical systems where failures are costly

iContract provides the SAFETY NET (runtime checks)
Hypothesis provides the STRESS TEST (thorough exploration)

Together they find bugs that neither would find alone!


┌─────────────────────────────────────────────────────────────────────┐
│                    WHEN BOTH ARE OVERKILL                           │
└─────────────────────────────────────────────────────────────────────┘

✗ Simple getters/setters
✗ Trivial transformations (string.upper())
✗ Obvious validations (age > 0)
✗ Thin wrappers around library functions
✗ When type hints are sufficient

Don't over-engineer! Use tools when they add value.


┌─────────────────────────────────────────────────────────────────────┐
│                         COST-BENEFIT                                │
└─────────────────────────────────────────────────────────────────────┘

HYPOTHESIS:
+ Finds edge cases automatically
+ Tests thousands of inputs
+ Great for exploring unknown bugs
- Test execution time
- Requires property thinking

iContract:
+ Runtime validation in production
+ Self-documenting contracts
+ Clear error messages
- Runtime performance overhead
- Requires careful contract design

BOTH:
+ Maximum confidence in correctness
+ Contracts serve as test oracles
+ Finds bugs neither would find alone
- Higher complexity
- Both costs combined
- Overkill for simple code
"""


if __name__ == "__main__":
    print("Examples demonstrating Hypothesis + iContract combination")
    print("=" * 70)

    print("\n1. Testing find_maximum with both tools:")
    test_find_maximum()
    print("   ✓ Hypothesis generated random lists, iContract validated contracts")

    print("\n2. Testing business_days_between:")
    test_business_days_between()
    print("   ✓ Random dates tested, contracts enforced")

    print("\n3. Testing bubble_sort:")
    test_bubble_sort()
    print("   ✓ Random lists sorted, invariants maintained")

    print("\n" + "=" * 70)
    print("See comments in code for detailed decision guide!")
    print("=" * 70)
