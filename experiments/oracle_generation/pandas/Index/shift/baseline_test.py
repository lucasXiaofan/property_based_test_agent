"""
Baseline property-based tests for pandas.Index.shift.

Properties tested:
1.  Length is preserved after shifting (DatetimeIndex)
2.  Length is preserved after shifting (TimedeltaIndex)
3.  Return type is DatetimeIndex after shifting a DatetimeIndex
4.  Return type is TimedeltaIndex after shifting a TimedeltaIndex
5.  shift(0) returns equal index values (DatetimeIndex)
6.  shift(0) returns equal index values (TimedeltaIndex)
7.  Roundtrip: shift(+n) then shift(-n) returns original (DatetimeIndex)
8.  Roundtrip: shift(+n) then shift(-n) returns original (TimedeltaIndex)
9.  Additivity: shift(n+m) == shift(n).shift(m) for DatetimeIndex (default freq)
10. Explicit freq shifts each element by exactly periods * freq
11. Default freq (freq=None) uses the index's own freq attribute
12. Result freq is None when an explicit freq argument is provided
13. Result freq is preserved when no explicit freq argument is given
14. Raises an error for non-datetime-like Index (regular integer Index)
15. Positive periods shifts all timestamps forward
16. Negative periods shifts all timestamps backward
17. String freq alias 'D' gives same result as pd.Timedelta('1D')
18. Default periods=1 matches shift(1)
19. PeriodIndex: type and length preserved after shift
20. PeriodIndex: shift(0) returns equal index
21. PeriodIndex: positive shift moves each period forward by the expected amount
22. PeriodIndex: roundtrip shift(n) then shift(-n) returns original
"""

from datetime import datetime

import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helpers / shared strategies
# ---------------------------------------------------------------------------

def datetime_range_strategy(min_periods=2, max_periods=10):
    """Strategy producing a DatetimeIndex created via pd.date_range."""
    return st.builds(
        lambda start, periods, freq: pd.date_range(start=start, periods=periods, freq=freq),
        start=st.datetimes(
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2018, 1, 1),
        ),
        periods=st.integers(min_periods, max_periods),
        freq=st.sampled_from(["D", "h", "min", "s", "MS"]),
    )


def timedelta_range_strategy(min_periods=2, max_periods=10):
    """Strategy producing a TimedeltaIndex created via pd.timedelta_range."""
    return st.builds(
        lambda periods, freq: pd.timedelta_range(start="1 day", periods=periods, freq=freq),
        periods=st.integers(min_periods, max_periods),
        freq=st.sampled_from(["D", "h", "min", "s"]),
    )


# ---------------------------------------------------------------------------
# 1. Length is preserved after shifting (DatetimeIndex)
# ---------------------------------------------------------------------------

@given(
    idx=datetime_range_strategy(),
    periods=st.integers(-5, 5),
)
@settings(max_examples=100)
def test_length_preserved_datetimeindex(idx, periods):
    result = idx.shift(periods)
    assert len(result) == len(idx)


# ---------------------------------------------------------------------------
# 2. Length is preserved after shifting (TimedeltaIndex)
# ---------------------------------------------------------------------------

@given(
    idx=timedelta_range_strategy(),
    periods=st.integers(-5, 5),
)
@settings(max_examples=100)
def test_length_preserved_timedeltaindex(idx, periods):
    result = idx.shift(periods)
    assert len(result) == len(idx)


# ---------------------------------------------------------------------------
# 3. Return type is DatetimeIndex after shifting a DatetimeIndex
# ---------------------------------------------------------------------------

@given(
    idx=datetime_range_strategy(),
    periods=st.integers(-5, 5),
)
@settings(max_examples=100)
def test_return_type_datetimeindex(idx, periods):
    result = idx.shift(periods)
    assert isinstance(result, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# 4. Return type is TimedeltaIndex after shifting a TimedeltaIndex
# ---------------------------------------------------------------------------

@given(
    idx=timedelta_range_strategy(),
    periods=st.integers(-5, 5),
)
@settings(max_examples=100)
def test_return_type_timedeltaindex(idx, periods):
    result = idx.shift(periods)
    assert isinstance(result, pd.TimedeltaIndex)


# ---------------------------------------------------------------------------
# 5. shift(0) returns equal index values (DatetimeIndex)
# ---------------------------------------------------------------------------

@given(idx=datetime_range_strategy())
@settings(max_examples=100)
def test_zero_shift_identity_datetimeindex(idx):
    result = idx.shift(0)
    pd.testing.assert_index_equal(result, idx)


# ---------------------------------------------------------------------------
# 6. shift(0) returns equal index values (TimedeltaIndex)
# ---------------------------------------------------------------------------

@given(idx=timedelta_range_strategy())
@settings(max_examples=100)
def test_zero_shift_identity_timedeltaindex(idx):
    result = idx.shift(0)
    pd.testing.assert_index_equal(result, idx)


# ---------------------------------------------------------------------------
# 7. Roundtrip: shift(+n) then shift(-n) returns original (DatetimeIndex)
# ---------------------------------------------------------------------------

@given(
    idx=datetime_range_strategy(),
    n=st.integers(1, 5),
)
@settings(max_examples=100)
def test_roundtrip_shift_datetimeindex(idx, n):
    shifted_forward = idx.shift(n)
    shifted_back = shifted_forward.shift(-n)
    pd.testing.assert_index_equal(shifted_back, idx)


# ---------------------------------------------------------------------------
# 8. Roundtrip: shift(+n) then shift(-n) returns original (TimedeltaIndex)
# ---------------------------------------------------------------------------

@given(
    idx=timedelta_range_strategy(),
    n=st.integers(1, 5),
)
@settings(max_examples=100)
def test_roundtrip_shift_timedeltaindex(idx, n):
    shifted_forward = idx.shift(n)
    shifted_back = shifted_forward.shift(-n)
    pd.testing.assert_index_equal(shifted_back, idx)


# ---------------------------------------------------------------------------
# 9. Additivity: shift(n+m) == shift(n).shift(m) for DatetimeIndex (default freq)
# ---------------------------------------------------------------------------

@given(
    idx=datetime_range_strategy(),
    n=st.integers(-3, 3),
    m=st.integers(-3, 3),
)
@settings(max_examples=100)
def test_additivity_datetimeindex_default_freq(idx, n, m):
    result_combined = idx.shift(n + m)
    result_sequential = idx.shift(n).shift(m)
    pd.testing.assert_index_equal(result_combined, result_sequential)


# ---------------------------------------------------------------------------
# 10. Explicit freq shifts each element by exactly periods * freq
# ---------------------------------------------------------------------------

def test_explicit_freq_shifts_each_element_by_correct_amount():
    """Each element in result should be exactly periods * freq ahead of original."""
    idx = pd.date_range("2011-01-01", periods=5, freq="MS")
    periods = 10
    result = idx.shift(periods, freq="D")
    expected_delta = pd.Timedelta(days=10)
    for orig, shifted in zip(idx, result):
        assert shifted == orig + expected_delta


# ---------------------------------------------------------------------------
# 11. Default freq (freq=None) uses the index's own freq attribute
# ---------------------------------------------------------------------------

def test_default_freq_uses_index_freq():
    """shift() without freq uses the index's own freq (month start here)."""
    idx = pd.date_range("2011-01-01", periods=5, freq="MS")
    result = idx.shift(10)
    # 10 month-starts forward from Jan 2011 → Nov 2011, Dec 2011, Jan 2012 …
    assert result[0] == pd.Timestamp("2011-11-01")
    assert result[1] == pd.Timestamp("2011-12-01")
    assert result[2] == pd.Timestamp("2012-01-01")


# ---------------------------------------------------------------------------
# 12. Result freq is None when an explicit freq argument is provided
# ---------------------------------------------------------------------------

def test_result_freq_none_when_explicit_freq_given():
    """When freq is explicitly provided, the result's freq attribute should be None."""
    idx = pd.date_range("2011-01-01", periods=5, freq="MS")
    result = idx.shift(10, freq="D")
    assert result.freq is None


# ---------------------------------------------------------------------------
# 13. Result freq is preserved when no explicit freq argument is given
# ---------------------------------------------------------------------------

@given(idx=datetime_range_strategy())
@settings(max_examples=100)
def test_result_freq_preserved_with_default_freq(idx):
    """Without an explicit freq, result freq equals the original freq."""
    n = 3
    result = idx.shift(n)
    assert result.freq == idx.freq


# ---------------------------------------------------------------------------
# 14. Raises an error for non-datetime-like Index (regular integer Index)
# ---------------------------------------------------------------------------

def test_raises_for_integer_index():
    """Index.shift is not implemented for non-datetime-like indexes."""
    idx = pd.Index([1, 2, 3, 4, 5])
    with pytest.raises((NotImplementedError, TypeError)):
        idx.shift(1)


def test_raises_for_string_index():
    """Index.shift is not implemented for string indexes."""
    idx = pd.Index(["a", "b", "c"])
    with pytest.raises((NotImplementedError, TypeError)):
        idx.shift(1)


# ---------------------------------------------------------------------------
# 15. Positive periods shifts all timestamps forward
# ---------------------------------------------------------------------------

@given(
    idx=datetime_range_strategy(),
    n=st.integers(1, 5),
)
@settings(max_examples=100)
def test_positive_periods_shifts_forward(idx, n):
    """After shift(+n), every element is strictly greater than the original."""
    result = idx.shift(n)
    assert all(shifted > orig for shifted, orig in zip(result, idx))


# ---------------------------------------------------------------------------
# 16. Negative periods shifts all timestamps backward
# ---------------------------------------------------------------------------

@given(
    idx=datetime_range_strategy(),
    n=st.integers(1, 5),
)
@settings(max_examples=100)
def test_negative_periods_shifts_backward(idx, n):
    """After shift(-n), every element is strictly less than the original."""
    result = idx.shift(-n)
    assert all(shifted < orig for shifted, orig in zip(result, idx))


# ---------------------------------------------------------------------------
# 17. String freq alias 'D' gives same result as pd.Timedelta('1D')
# ---------------------------------------------------------------------------

def test_string_freq_alias_equals_timedelta_object():
    """Using 'D' as a freq string is equivalent to pd.Timedelta('1D')."""
    idx = pd.date_range("2011-01-01", periods=5, freq="MS")
    result_str = idx.shift(3, freq="D")
    result_obj = idx.shift(3, freq=pd.Timedelta("1D"))
    pd.testing.assert_index_equal(result_str, result_obj)


# ---------------------------------------------------------------------------
# 18. Default periods=1 is equivalent to shift(1)
# ---------------------------------------------------------------------------

@given(idx=datetime_range_strategy())
@settings(max_examples=100)
def test_default_periods_equals_one(idx):
    result_default = idx.shift()
    result_explicit = idx.shift(1)
    pd.testing.assert_index_equal(result_default, result_explicit)


# ---------------------------------------------------------------------------
# 19. PeriodIndex: type and length preserved after shift
# ---------------------------------------------------------------------------

def test_period_index_type_and_length_preserved():
    idx = pd.period_range("2011-01", periods=5, freq="M")
    result = idx.shift(3)
    assert isinstance(result, pd.PeriodIndex)
    assert len(result) == len(idx)


# ---------------------------------------------------------------------------
# 20. PeriodIndex: shift(0) returns equal index
# ---------------------------------------------------------------------------

def test_period_index_zero_shift_identity():
    idx = pd.period_range("2011-01", periods=5, freq="M")
    result = idx.shift(0)
    pd.testing.assert_index_equal(result, idx)


# ---------------------------------------------------------------------------
# 21. PeriodIndex: positive shift moves each period forward by the expected amount
# ---------------------------------------------------------------------------

def test_period_index_positive_shift_moves_forward():
    idx = pd.period_range("2011-01", periods=5, freq="M")
    n = 3
    result = idx.shift(n)
    for orig, shifted in zip(idx, result):
        assert shifted == orig + n


# ---------------------------------------------------------------------------
# 22. PeriodIndex: roundtrip shift(n) then shift(-n) returns original
# ---------------------------------------------------------------------------

def test_period_index_roundtrip_shift():
    idx = pd.period_range("2011-01", periods=5, freq="M")
    shifted = idx.shift(5)
    back = shifted.shift(-5)
    pd.testing.assert_index_equal(back, idx)
