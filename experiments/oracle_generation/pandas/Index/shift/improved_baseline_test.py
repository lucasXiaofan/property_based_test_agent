import pandas as pd
import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st


@given(
    start=st.sampled_from(["2011-01-01", "2020-01-01", "2023-06-15"]),
    periods=st.integers(min_value=1, max_value=10),
    freq=st.sampled_from(["D", "W", "MS", "h"]),
)
@settings(max_examples=50)
def test_shift_with_explicit_freq(start, periods, freq):
    """Test shift with explicit freq parameter."""
    index = pd.date_range(start, periods=periods, freq=freq)
    result = index.shift(periods, freq="D")

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)
    assert all(
        result[i] == index[i] + pd.Timedelta(days=periods) for i in range(len(index))
    )


@given(
    start=st.sampled_from(["2011-01-01", "2020-01-01"]),
    periods=st.integers(min_value=1, max_value=12),
)
@settings(max_examples=30)
def test_shift_with_freq_none(start, periods):
    """Test shift with freq=None uses index's own freq."""
    index = pd.date_range(start, periods=periods, freq="MS")
    result = index.shift(periods)

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)
    assert result.freq == index.freq
    assert all(result[i] == index[i] + periods * index.freq for i in range(len(index)))


@given(
    start=st.sampled_from(["2011-01-01", "2020-01-01"]),
    periods=st.integers(min_value=1, max_value=10),
    freq=st.sampled_from(["D", "W", "ME"]),
)
@settings(max_examples=30)
def test_shift_positive_periods(start, periods, freq):
    """Test shift with positive periods moves forward."""
    index = pd.date_range(start, periods=periods, freq=freq)
    result = index.shift(periods, freq=freq)

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)
    assert all(result[i] > index[i] for i in range(len(index)))


@given(
    start=st.sampled_from(["2011-01-01", "2020-01-01"]),
    periods=st.integers(min_value=1, max_value=10),
    freq=st.sampled_from(["D", "W", "ME"]),
)
@settings(max_examples=30)
def test_shift_negative_periods(start, periods, freq):
    """Test shift with negative periods moves backward."""
    index = pd.date_range(start, periods=periods, freq=freq)
    result = index.shift(-periods, freq=freq)

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)
    assert all(result[i] < index[i] for i in range(len(index)))


@given(
    start=st.sampled_from(["2011-01-01", "2020-01-01"]),
    periods=st.integers(min_value=1, max_value=10),
    freq=st.sampled_from(["D", "W", "ME"]),
)
@settings(max_examples=30)
def test_shift_zero_periods_equals_self(start, periods, freq):
    """Test that shifting by zero periods returns the same index."""
    index = pd.date_range(start, periods=periods, freq=freq)
    result = index.shift(0, freq=freq)

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)
    assert result.equals(index)


@given(
    start=st.sampled_from(["2011-01-01", "2020-01-01"]),
    periods=st.integers(min_value=1, max_value=5),
    freq=st.sampled_from(["D", "W"]),
)
@settings(max_examples=20)
def test_shift_roundtrip_positive_negative(start, periods, freq):
    """Test that shifting forward then backward returns original index."""
    index = pd.date_range(start, periods=periods, freq=freq)
    shifted = index.shift(periods, freq=freq)
    roundtrip = shifted.shift(-periods, freq=freq)

    assert roundtrip.equals(index)


def test_shift_datetime_index_returns_index():
    """Test that shift on DatetimeIndex returns a DatetimeIndex."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")
    result = index.shift(1, freq="D")

    assert isinstance(result, pd.Index)
    assert isinstance(result, pd.DatetimeIndex)


def test_shift_with_timedelta_freq():
    """Test shift with Timedelta as freq parameter."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")
    result = index.shift(2, freq=pd.Timedelta(days=3))

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)
    assert all(result[i] == index[i] + pd.Timedelta(days=6) for i in range(len(index)))


def test_shift_with_dateoffset_freq():
    """Test shift with DateOffset as freq parameter."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")
    result = index.shift(2, freq=pd.DateOffset(days=3))

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)


# ============================================================================
# NEW TESTS ADDED FOR IMPROVED COVERAGE
# ============================================================================

def test_shift_period_index():
    """Test shift on PeriodIndex - documented as supported index type."""
    index = pd.period_range("2011-01", periods=5, freq="M")
    result = index.shift(2)

    assert isinstance(result, pd.Index)
    assert isinstance(result, pd.PeriodIndex)
    assert len(result) == len(index)


def test_shift_period_index_with_freq_none():
    """Test PeriodIndex.shift with freq=None uses index's own freq."""
    index = pd.period_range("2011-01", periods=5, freq="M")
    result = index.shift(2)

    assert isinstance(result, pd.PeriodIndex)
    assert len(result) == len(index)
    assert result.freq == index.freq


def test_shift_timedelta_index():
    """Test shift on TimedeltaIndex - documented as supported index type."""
    index = pd.timedelta_range(start="1 days", periods=5, freq="D")
    result = index.shift(2, freq="D")

    assert isinstance(result, pd.Index)
    assert isinstance(result, pd.TimedeltaIndex)
    assert len(result) == len(index)


def test_shift_timedelta_index_with_freq_none():
    """Test TimedeltaIndex.shift with freq=None uses index's own freq."""
    index = pd.timedelta_range(start="1 days", periods=5, freq="D")
    result = index.shift(2)

    assert isinstance(result, pd.TimedeltaIndex)
    assert len(result) == len(index)
    assert result.freq == index.freq


def test_shift_non_datetime_index_raises():
    """Test that shift on non-datetime index raises NotImplementedError."""
    index = pd.Index([1, 2, 3, 4, 5])

    with pytest.raises(NotImplementedError):
        index.shift(1)


def test_shift_empty_datetime_index():
    """Test shift on empty DatetimeIndex returns empty index."""
    index = pd.DatetimeIndex([])
    result = index.shift(1, freq="D")

    assert isinstance(result, pd.DatetimeIndex)
    assert len(result) == 0


def test_shift_invalid_freq_type():
    """Test that invalid freq type raises appropriate error."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")

    with pytest.raises((TypeError, ValueError)):
        index.shift(1, freq=123)


def test_shift_large_periods():
    """Test shift with large number of periods."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")
    result = index.shift(1000, freq="D")

    assert isinstance(result, pd.DatetimeIndex)
    assert len(result) == len(index)
    assert all(result[i] == index[i] + pd.Timedelta(days=1000) for i in range(len(index)))


def test_shift_negative_large_periods():
    """Test shift with large negative periods."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")
    result = index.shift(-1000, freq="D")

    assert isinstance(result, pd.DatetimeIndex)
    assert len(result) == len(index)
    assert all(result[i] == index[i] - pd.Timedelta(days=1000) for i in range(len(index)))


def test_shift_period_index_negative_periods():
    """Test PeriodIndex.shift with negative periods."""
    index = pd.period_range("2011-01", periods=5, freq="M")
    result = index.shift(-2)

    assert isinstance(result, pd.PeriodIndex)
    assert len(result) == len(index)


def test_shift_preserves_index_name():
    """Test that shift preserves the index name."""
    index = pd.date_range("2011-01-01", periods=5, freq="D", name="my_index")
    result = index.shift(1, freq="D")

    assert result.name == "my_index"


def test_shift_with_different_freq():
    """Test shift with a different freq than the index's own freq."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")
    result = index.shift(1, freq="h")

    assert isinstance(result, pd.DatetimeIndex)
    assert len(result) == len(index)


def test_shift_result_freq_preserved_when_freq_none():
    """Test that result freq is preserved when freq=None."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")
    result = index.shift(1)

    assert result.freq == index.freq


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
