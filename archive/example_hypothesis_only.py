"""
Example: Property-based testing with Hypothesis ONLY

Use Hypothesis alone when:
1. You're writing tests (not production code)
2. You want to explore edge cases automatically
3. You need to verify properties across many inputs
4. Runtime contract checking overhead is unacceptable
5. You're testing pure functions with clear mathematical properties

Hypothesis is a TESTING tool - it generates random inputs to find bugs.
"""

from hypothesis import given, strategies as st, assume, example
from datetime import date, timedelta


# ============================================================================
# Example 1: Simple mathematical property
# Hypothesis is perfect for this - we just need test-time validation
# ============================================================================

def reverse_list(lst: list) -> list:
    """Reverse a list."""
    return lst[::-1]


@given(st.lists(st.integers()))
def test_reverse_is_involutive(lst):
    """Property: reversing twice returns the original list."""
    # Hypothesis generates many random lists to test this property
    assert reverse_list(reverse_list(lst)) == lst


@given(st.lists(st.integers()))
def test_reverse_preserves_length(lst):
    """Property: reversing doesn't change the length."""
    assert len(reverse_list(lst)) == len(lst)


# ============================================================================
# Example 2: String manipulation
# Hypothesis alone is sufficient - no need for runtime contract checking
# ============================================================================

def title_case(s: str) -> str:
    """Convert string to title case."""
    return s.title()


@given(st.text())
def test_title_case_first_char_uppercase(s):
    """Property: title case makes first character of each word uppercase."""
    assume(len(s) > 0 and s[0].isalpha())  # Filter to relevant inputs
    result = title_case(s)
    assert result[0].isupper() or not result[0].isalpha()


# ============================================================================
# Example 3: Arithmetic properties
# Testing commutative, associative properties - perfect for Hypothesis only
# ============================================================================

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@given(st.integers(), st.integers())
def test_addition_is_commutative(a, b):
    """Property: a + b = b + a"""
    assert add(a, b) == add(b, a)


@given(st.integers(), st.integers(), st.integers())
def test_addition_is_associative(a, b, c):
    """Property: (a + b) + c = a + (b + c)"""
    assert add(add(a, b), c) == add(a, add(b, c))


# ============================================================================
# Example 4: Date calculations
# Hypothesis can test properties without needing runtime contracts
# ============================================================================

def days_between(start: date, end: date) -> int:
    """Calculate number of days between two dates."""
    return (end - start).days


@given(st.dates(), st.integers(min_value=0, max_value=365))
def test_adding_days_commutes_with_subtraction(start_date, days_to_add):
    """Property: if we add N days, days_between should return N."""
    end_date = start_date + timedelta(days=days_to_add)
    assert days_between(start_date, end_date) == days_to_add


@given(st.dates(), st.dates())
def test_days_between_is_antisymmetric(date1, date2):
    """Property: days_between(a, b) = -days_between(b, a)"""
    assert days_between(date1, date2) == -days_between(date2, date1)


# ============================================================================
# Example 5: Collection operations
# Hypothesis excels at testing collection invariants
# ============================================================================

def unique_elements(lst: list) -> list:
    """Return unique elements preserving order."""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


@given(st.lists(st.integers()))
def test_unique_elements_no_duplicates(lst):
    """Property: result has no duplicates."""
    result = unique_elements(lst)
    assert len(result) == len(set(result))


@given(st.lists(st.integers()))
def test_unique_elements_preserves_order(lst):
    """Property: first occurrence order is preserved."""
    result = unique_elements(lst)
    # All elements in result should appear in same order as in lst
    lst_filtered = [x for x in lst if x in result]
    seen = set()
    expected = []
    for x in lst_filtered:
        if x not in seen:
            seen.add(x)
            expected.append(x)
    assert result == expected


@given(st.lists(st.integers()))
def test_unique_elements_subset(lst):
    """Property: result is a subset of input."""
    result = unique_elements(lst)
    assert all(item in lst for item in result)


# ============================================================================
# Example 6: Stateful testing
# Hypothesis can test stateful systems using rule-based testing
# ============================================================================

from hypothesis.stateful import RuleBasedStateMachine, rule, invariant


class Stack:
    """Simple stack implementation."""
    def __init__(self):
        self.data = []

    def push(self, item):
        self.data.append(item)

    def pop(self):
        if not self.data:
            raise IndexError("pop from empty stack")
        return self.data.pop()

    def is_empty(self):
        return len(self.data) == 0

    def size(self):
        return len(self.data)


class StackMachine(RuleBasedStateMachine):
    """Test stack using stateful property-based testing."""

    def __init__(self):
        super().__init__()
        self.stack = Stack()
        self.model = []  # Python list as reference model

    @rule(value=st.integers())
    def push(self, value):
        """Push a value onto the stack."""
        self.stack.push(value)
        self.model.append(value)

    @rule()
    def pop(self):
        """Pop a value from the stack if not empty."""
        assume(not self.stack.is_empty())
        stack_value = self.stack.pop()
        model_value = self.model.pop()
        assert stack_value == model_value

    @invariant()
    def size_matches_model(self):
        """Invariant: stack size always matches model."""
        assert self.stack.size() == len(self.model)

    @invariant()
    def empty_consistent(self):
        """Invariant: is_empty() is consistent with size()."""
        assert self.stack.is_empty() == (self.stack.size() == 0)


# To run: TestStack = StackMachine.TestCase


# ============================================================================
# SUMMARY: When to use Hypothesis ONLY
# ============================================================================
"""
Use Hypothesis alone when:

✓ Writing TESTS for existing code
✓ Exploring edge cases and finding bugs
✓ Testing mathematical properties (commutativity, associativity, etc.)
✓ Testing pure functions
✓ Verifying invariants across many random inputs
✓ Stateful property testing
✓ No need for runtime validation in production
✓ Performance is critical (no runtime overhead)

Do NOT use for:
✗ Runtime validation in production code
✗ Enforcing preconditions on function inputs
✗ Documenting contracts in the code itself
✗ Defensive programming in libraries
✗ Providing clear error messages for invalid inputs

Hypothesis generates test cases; it doesn't protect production code at runtime.
"""


if __name__ == "__main__":
    import pytest

    print("Running Hypothesis-only examples...")
    print("\nThese are TESTS that explore properties with random inputs.")
    print("Hypothesis is a testing framework, not a runtime validation tool.\n")

    # Run a few tests manually to demonstrate
    test_reverse_is_involutive()
    test_addition_is_commutative()
    test_unique_elements_no_duplicates()

    print("✓ All example tests passed!")
    print("\nRun with pytest for full output: pytest example_hypothesis_only.py -v")
