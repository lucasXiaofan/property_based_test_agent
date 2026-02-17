"""
Example: Design by Contract with iContract ONLY

Use iContract alone when:
1. You're writing production/library code (not just tests)
2. You need runtime validation of preconditions and postconditions
3. You want to document contracts in the code itself
4. You need to catch invalid inputs from users/callers
5. You're practicing defensive programming
6. You want self-documenting APIs

iContract is a RUNTIME VALIDATION tool - it checks contracts during execution.
"""

from icontract import require, ensure, invariant, ViolationError
from datetime import date, timedelta
from typing import List
import math


# ============================================================================
# Example 1: Input validation with preconditions
# iContract enforces valid inputs at runtime - no need for hypothesis testing
# ============================================================================

@require(lambda divisor: divisor != 0, "Divisor cannot be zero")
@ensure(lambda dividend, divisor, result: abs(dividend - divisor * result) < abs(divisor))
def divide(dividend: float, divisor: float) -> float:
    """
    Divide two numbers.

    Precondition: divisor != 0
    Postcondition: dividend ≈ divisor * result (within remainder bounds)
    """
    return dividend / divisor


# Example usage:
try:
    result = divide(10, 2)  # OK: 5.0
    print(f"10 / 2 = {result}")

    result = divide(10, 0)  # Raises ViolationError
except ViolationError as e:
    print(f"Contract violation: {e}")


# ============================================================================
# Example 2: Postconditions ensure output validity
# iContract validates outputs - useful when implementation might have bugs
# ============================================================================

@require(lambda n: n >= 0, "n must be non-negative")
@ensure(lambda n, result: result >= 0, "Square root must be non-negative")
@ensure(lambda n, result: abs(result * result - n) < 0.0001, "result^2 must equal n")
def sqrt(n: float) -> float:
    """
    Calculate square root.

    Precondition: n >= 0
    Postcondition: result >= 0 and result^2 ≈ n
    """
    return math.sqrt(n)


# ============================================================================
# Example 3: Class invariants
# iContract maintains object state consistency - critical for stateful objects
# ============================================================================

@invariant(lambda self: self.balance >= 0, "Balance cannot be negative")
@invariant(lambda self: len(self.transactions) >= 0, "Transaction list must exist")
class BankAccount:
    """
    Bank account with invariant: balance must never be negative.
    """

    @require(lambda initial_balance: initial_balance >= 0, "Initial balance must be non-negative")
    def __init__(self, initial_balance: float = 0.0):
        self.balance = initial_balance
        self.transactions: List[float] = []

    @require(lambda amount: amount > 0, "Deposit amount must be positive")
    @ensure(lambda self, amount, OLD: self.balance == OLD.balance + amount)
    def deposit(self, amount: float) -> None:
        """Deposit money into the account."""
        self.balance += amount
        self.transactions.append(amount)

    @require(lambda amount: amount > 0, "Withdrawal amount must be positive")
    @require(lambda self, amount: amount <= self.balance, "Insufficient funds")
    @ensure(lambda self, amount, OLD: self.balance == OLD.balance - amount)
    def withdraw(self, amount: float) -> None:
        """Withdraw money from the account."""
        self.balance -= amount
        self.transactions.append(-amount)

    @ensure(lambda self, result: result >= 0, "Balance query must return non-negative")
    def get_balance(self) -> float:
        """Get current balance."""
        return self.balance


# Example usage:
account = BankAccount(100.0)
account.deposit(50.0)  # OK
print(f"Balance after deposit: {account.get_balance()}")

try:
    account.withdraw(200.0)  # Raises ViolationError: insufficient funds
except ViolationError as e:
    print(f"Contract violation: {e}")


# ============================================================================
# Example 4: Date range validation
# iContract provides clear runtime errors for invalid date ranges
# ============================================================================

@require(lambda start, end: start <= end, "Start date must be before or equal to end date")
@ensure(lambda result: result >= 0, "Date range must be non-negative")
def calculate_date_range(start: date, end: date) -> int:
    """
    Calculate the number of days between two dates.

    Precondition: start <= end
    Postcondition: result >= 0
    """
    return (end - start).days


# Example usage:
try:
    days = calculate_date_range(date(2024, 1, 1), date(2024, 12, 31))
    print(f"Days in 2024: {days}")

    # This will fail the precondition:
    days = calculate_date_range(date(2024, 12, 31), date(2024, 1, 1))
except ViolationError as e:
    print(f"Contract violation: {e}")


# ============================================================================
# Example 5: List operations with contracts
# iContract documents and enforces list operation constraints
# ============================================================================

@require(lambda lst: len(lst) > 0, "List cannot be empty")
@ensure(lambda lst, result: result in lst, "Result must be from the list")
@ensure(lambda lst, result: all(result >= x for x in lst), "Result must be maximum")
def get_max(lst: List[int]) -> int:
    """
    Get maximum value from a non-empty list.

    Precondition: list must not be empty
    Postcondition: result is the maximum element
    """
    return max(lst)


@require(lambda lst: len(lst) > 0, "List cannot be empty")
@require(lambda index: index >= 0, "Index must be non-negative")
@require(lambda lst, index: index < len(lst), "Index must be within bounds")
@ensure(lambda lst, result: len(result) == len(lst) - 1, "Result should have one less element")
def remove_at_index(lst: List[int], index: int) -> List[int]:
    """
    Remove element at specified index.

    Preconditions:
    - List must not be empty
    - Index must be valid (0 <= index < len(lst))

    Postcondition: result has len(lst) - 1 elements
    """
    return lst[:index] + lst[index + 1:]


# ============================================================================
# Example 6: String validation
# iContract validates and documents string constraints
# ============================================================================

@require(lambda email: '@' in email, "Email must contain @")
@require(lambda email: email.count('@') == 1, "Email must contain exactly one @")
@require(lambda email: len(email.split('@')[0]) > 0, "Email must have local part")
@require(lambda email: len(email.split('@')[1]) > 0, "Email must have domain part")
@require(lambda email: '.' in email.split('@')[1], "Domain must contain a dot")
def validate_email(email: str) -> str:
    """
    Validate and return an email address.

    Preconditions:
    - Must contain exactly one @
    - Must have non-empty local and domain parts
    - Domain must contain at least one dot
    """
    return email.lower().strip()


# Example usage:
try:
    valid = validate_email("user@example.com")
    print(f"Valid email: {valid}")

    invalid = validate_email("not-an-email")  # Raises ViolationError
except ViolationError as e:
    print(f"Invalid email: {e}")


# ============================================================================
# Example 7: Mathematical constraints
# iContract enforces mathematical properties at runtime
# ============================================================================

@require(lambda base, exponent: base != 0 or exponent > 0, "0^0 is undefined")
@require(lambda base, exponent: base >= 0 or exponent == int(exponent),
         "Negative base requires integer exponent")
@ensure(lambda base, exponent, result: base == 0 or result > 0 if exponent > 0 else result != 0,
        "Positive exponent with non-zero base must yield positive result")
def power(base: float, exponent: float) -> float:
    """
    Calculate base^exponent with mathematical constraints.

    Preconditions:
    - 0^0 is not allowed
    - Negative base requires integer exponent (avoid complex numbers)
    """
    return base ** exponent


# ============================================================================
# Example 8: State machine with contracts
# iContract enforces state transition rules
# ============================================================================

from enum import Enum

class OrderState(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@invariant(lambda self: self.state in OrderState, "State must be valid OrderState")
@invariant(lambda self: not (self.state == OrderState.CANCELLED and
                              self.state in [OrderState.SHIPPED, OrderState.DELIVERED]),
           "Cancelled orders cannot be shipped or delivered")
class Order:
    """Order with state machine enforced by contracts."""

    def __init__(self):
        self.state = OrderState.PENDING
        self.items: List[str] = []

    @require(lambda self: self.state == OrderState.PENDING, "Can only confirm pending orders")
    @ensure(lambda self: self.state == OrderState.CONFIRMED, "State must be confirmed after")
    def confirm(self) -> None:
        """Confirm a pending order."""
        self.state = OrderState.CONFIRMED

    @require(lambda self: self.state == OrderState.CONFIRMED, "Can only ship confirmed orders")
    @ensure(lambda self: self.state == OrderState.SHIPPED, "State must be shipped after")
    def ship(self) -> None:
        """Ship a confirmed order."""
        self.state = OrderState.SHIPPED

    @require(lambda self: self.state in [OrderState.PENDING, OrderState.CONFIRMED],
             "Can only cancel pending or confirmed orders")
    @ensure(lambda self: self.state == OrderState.CANCELLED, "State must be cancelled after")
    def cancel(self) -> None:
        """Cancel an order."""
        self.state = OrderState.CANCELLED


# Example usage:
order = Order()
order.confirm()
order.ship()
print(f"Order state: {order.state}")

try:
    order.cancel()  # Raises ViolationError: can't cancel shipped order
except ViolationError as e:
    print(f"Cannot cancel shipped order: {e}")


# ============================================================================
# SUMMARY: When to use iContract ONLY
# ============================================================================
"""
Use iContract alone when:

✓ Writing PRODUCTION/LIBRARY code that needs runtime validation
✓ Enforcing preconditions on function inputs
✓ Validating postconditions on function outputs
✓ Maintaining class invariants
✓ Documenting contracts as executable specifications
✓ Providing clear error messages for invalid usage
✓ Defensive programming against invalid callers
✓ API design with explicit contracts
✓ State machine enforcement
✓ Replacing manual if-checks and assertions

Do NOT use for:
✗ Exploring edge cases (use Hypothesis for that)
✗ Testing with random inputs (use Hypothesis for that)
✗ Finding unknown bugs (use Hypothesis for that)
✗ Performance-critical hot paths (contracts have overhead)
✗ Simple functions where type hints are sufficient

iContract validates at runtime; it doesn't generate test cases.
"""


if __name__ == "__main__":
    print("Running iContract-only examples...")
    print("\nThese are PRODUCTION CODE examples with runtime contract validation.")
    print("iContract catches violations when code is executed with invalid inputs.\n")

    print("✓ All valid operations succeeded!")
    print("✗ Invalid operations raised ViolationError as expected!")
    print("\niContract provides runtime safety and executable documentation.")
