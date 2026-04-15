import pandas as pd
import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st


# ==================== BASELINE TESTS ====================


@given(
    start=st.sampled_from(["2011-01-01", "2020-01-01", "2023-06-15"]),
    periods=st.integers(min_value=1, max_value=10),
    freq=st.sampled_from(["D", "W", "MS", "h"]),
)
@settings(max_examples=50)
def test_shift_with_explicit_freq(start, periods, freq):
    """BASELINE: Test shift with explicit freq parameter."""
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
    """BASELINE: Test shift with freq=None uses index's own freq."""
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
    """BASELINE: Test shift with positive periods moves forward."""
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
    """BASELINE: Test shift with negative periods moves backward."""
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
    """BASELINE: Test that shifting by zero periods returns the same index."""
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
    """BASELINE: Test that shifting forward then backward returns original index."""
    index = pd.date_range(start, periods=periods, freq=freq)
    shifted = index.shift(periods, freq=freq)
    roundtrip = shifted.shift(-periods, freq=freq)

    assert roundtrip.equals(index)


def test_shift_datetime_index_returns_index():
    """BASELINE: Test that shift on DatetimeIndex returns a DatetimeIndex."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")
    result = index.shift(1, freq="D")

    assert isinstance(result, pd.Index)
    assert isinstance(result, pd.DatetimeIndex)


def test_shift_with_timedelta_freq():
    """BASELINE: Test shift with Timedelta as freq parameter."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")
    result = index.shift(2, freq=pd.Timedelta(days=3))

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)
    assert all(result[i] == index[i] + pd.Timedelta(days=6) for i in range(len(index)))


def test_shift_with_dateoffset_freq():
    """BASELINE: Test shift with DateOffset as freq parameter."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")
    result = index.shift(2, freq=pd.DateOffset(days=3))

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)


# ==================== NEW IR-ENHANCED TESTS ====================


@given(
    periods=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=30)
def test_shift_timedelta_index(periods):
    """IR-ENHANCED (NEW): Test shift on TimedeltaIndex - one of the datetime-like index classes."""
    index = pd.timedelta_range("0 days", periods=periods, freq="D")
    result = index.shift(periods)

    assert isinstance(result, pd.Index)
    assert isinstance(result, pd.TimedeltaIndex)
    assert len(result) == len(index)
    assert all(
        result[i] == index[i] + pd.Timedelta(days=periods) for i in range(len(index))
    )


@given(
    periods=st.integers(min_value=1, max_value=12),
)
@settings(max_examples=30)
def test_shift_period_index(periods):
    """IR-ENHANCED (NEW): Test shift on PeriodIndex - one of the datetime-like index classes."""
    index = pd.period_range("2011-01", periods=periods, freq="M")
    result = index.shift(periods)

    assert isinstance(result, pd.Index)
    assert isinstance(result, pd.PeriodIndex)
    assert len(result) == len(index)


@given(
    start=st.sampled_from(["2011-01-01", "2020-01-01"]),
    periods=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=30)
def test_shift_with_explicit_freq_loses_freq_attribute(start, periods):
    """IR-ENHANCED (NEW): When explicit freq is provided, result.freq should be None."""
    index = pd.date_range(start, periods=periods, freq="MS")
    result = index.shift(periods, freq="D")

    assert isinstance(result, pd.Index)
    assert result.freq is None


@given(
    start=st.sampled_from(["2011-01-01", "2020-01-01"]),
    periods=st.integers(min_value=1, max_value=10),
    freq=st.sampled_from(["D", "W", "MS", "h", "min", "s"]),
)
@settings(max_examples=50)
def test_shift_various_freq_aliases(start, periods, freq):
    """IR-ENHANCED (NEW): Test shift with various freq alias strings."""
    index = pd.date_range(start, periods=periods, freq="D")
    result = index.shift(periods, freq=freq)

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)


@given(
    periods=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=20)
def test_shift_roundtrip_with_freq_none(periods):
    """IR-ENHANCED (NEW): Test roundtrip when using freq=None (uses index's own freq)."""
    index = pd.date_range("2011-01-01", periods=periods, freq="MS")
    shifted = index.shift(periods)
    roundtrip = shifted.shift(-periods)

    assert roundtrip.freq == index.freq


@given(
    periods=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=20)
def test_shift_with_timedelta_as_freq_parameter(periods):
    """IR-ENHANCED (NEW): Test shift with pd.Timedelta as freq argument."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")
    result = index.shift(periods, freq=pd.Timedelta(days=periods))

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)
    expected_shift = periods * periods
    assert all(
        result[i] == index[i] + pd.Timedelta(days=expected_shift)
        for i in range(len(index))
    )


@given(
    periods=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=20)
def test_shift_with_dateoffset_as_freq_parameter(periods):
    """IR-ENHANCED (NEW): Test shift with pd.DateOffset as freq argument."""
    index = pd.date_range("2011-01-01", periods=5, freq="D")
    result = index.shift(periods, freq=pd.DateOffset(days=periods))

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)


@given(
    periods=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=10)
def test_shift_large_period_values(periods):
    """IR-ENHANCED (NEW): Test shift with large period values (up to 100)."""
    index = pd.date_range("2011-01-01", periods=3, freq="D")
    result = index.shift(100, freq="D")

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)
    assert all(
        result[i] == index[i] + pd.Timedelta(days=100) for i in range(len(index))
    )


@given(
    periods=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=10)
def test_shift_large_negative_period_values(periods):
    """IR-ENHANCED (NEW): Test shift with large negative period values (down to -100)."""
    index = pd.date_range("2011-01-01", periods=10, freq="D")
    result = index.shift(-100, freq="D")

    assert isinstance(result, pd.Index)
    assert len(result) == len(index)
    assert all(result[i] < index[i] for i in range(len(index)))


@given(
    periods=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=20)
def test_shift_preserves_index_type_for_datetimeindex(periods):
    """IR-ENHANCED (NEW): Test that shift preserves DatetimeIndex type when freq=None."""
    index = pd.date_range("2011-01-01", periods=periods, freq="D")
    result = index.shift(periods)

    assert isinstance(result, pd.DatetimeIndex)


@given(
    periods=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=20)
def test_shift_preserves_index_type_for_timedelta_index(periods):
    """IR-ENHANCED (NEW): Test that shift preserves TimedeltaIndex type when freq=None."""
    index = pd.timedelta_range("0 days", periods=periods, freq="D")
    result = index.shift(periods)

    assert isinstance(result, pd.TimedeltaIndex)


@given(
    periods=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=20)
def test_shift_preserves_index_type_for_period_index(periods):
    """IR-ENHANCED (NEW): Test that shift preserves PeriodIndex type when freq=None."""
    index = pd.period_range("2011-01", periods=periods, freq="M")
    result = index.shift(periods)

    assert isinstance(result, pd.PeriodIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
