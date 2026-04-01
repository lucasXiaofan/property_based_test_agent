import pandas as pd
import pytest
from hypothesis import given, strategies as st


FIXED_FREQ_AND_DELTA = st.sampled_from(
    [
        ("D", pd.Timedelta(days=1)),
        ("h", pd.Timedelta(hours=1)),
        ("min", pd.Timedelta(minutes=1)),
        ("s", pd.Timedelta(seconds=1)),
    ]
)


@given(periods=st.just(0), n=st.integers(min_value=1, max_value=8))
def test_zero_shift_is_identity_for_freq_aware_datetime_index(periods, n):
    idx = pd.date_range("2011-01-01", periods=n, freq="D")

    result = idx.shift(periods)

    pd.testing.assert_index_equal(result, idx)
    assert result.freq == idx.freq


@given(periods=st.integers(min_value=1, max_value=20))
def test_freqless_datetime_index_rejects_nonzero_shift(periods):
    idx = pd.DatetimeIndex(["2011-01-01", "2011-01-03"])

    with pytest.raises(pd.errors.NullFrequencyError):
        idx.shift(periods)


@given(periods=st.just(0))
def test_freqless_datetime_index_allows_zero_only(periods):
    idx = pd.DatetimeIndex(["2011-01-01", "2011-01-03"])

    pd.testing.assert_index_equal(idx.shift(periods), idx)


@given(periods=st.integers(min_value=1, max_value=20))
def test_freqless_timedelta_index_rejects_nonzero_shift(periods):
    idx = pd.TimedeltaIndex(["0 days", "2 days"])

    with pytest.raises(pd.errors.NullFrequencyError):
        idx.shift(periods)


@given(periods=st.integers(min_value=-12, max_value=12).filter(lambda x: x != 0))
def test_period_index_rejects_explicit_freq(periods):
    idx = pd.period_range("2011-01", periods=4, freq="M")

    with pytest.raises(TypeError):
        idx.shift(periods, freq="D")


@given(periods=st.integers(min_value=-12, max_value=12).filter(lambda x: x != 0))
def test_timedelta_index_rejects_dateoffset_freq(periods):
    idx = pd.timedelta_range("0 days", periods=4, freq="D")

    with pytest.raises(TypeError):
        idx.shift(periods, freq=pd.DateOffset(days=2))


@given(
    n=st.integers(min_value=1, max_value=8),
    periods=st.integers(min_value=-20, max_value=20),
)
def test_period_index_shift_matches_period_arithmetic(n, periods):
    idx = pd.period_range("2011-01", periods=n, freq="M")

    result = idx.shift(periods)
    expected = idx + periods

    pd.testing.assert_index_equal(result, expected)
    assert isinstance(result, pd.PeriodIndex)


@given(
    n=st.integers(min_value=1, max_value=8),
    periods=st.integers(min_value=-20, max_value=20),
)
def test_timedelta_index_shift_matches_period_arithmetic(n, periods):
    idx = pd.timedelta_range("0 days", periods=n, freq="D")

    result = idx.shift(periods)
    expected = idx + periods * idx.freq

    pd.testing.assert_index_equal(result, expected)
    assert isinstance(result, pd.TimedeltaIndex)


@given(
    n=st.integers(min_value=1, max_value=8),
    periods=st.integers(min_value=-20, max_value=20),
    freq_and_delta=FIXED_FREQ_AND_DELTA,
)
def test_datetime_explicit_fixed_freq_matches_elementwise_addition(
    n, periods, freq_and_delta
):
    freq, delta = freq_and_delta
    idx = pd.date_range("2011-01-01", periods=n, freq="D", tz="UTC")

    result = idx.shift(periods, freq=freq)
    expected = idx + (periods * delta)

    pd.testing.assert_index_equal(result, expected)
    assert result.tz == idx.tz


@given(
    n=st.integers(min_value=1, max_value=8),
    periods=st.integers(min_value=-20, max_value=20),
)
def test_monthstart_none_freq_uses_index_frequency(n, periods):
    idx = pd.date_range("2011-01-01", periods=n, freq="MS")

    result = idx.shift(periods)
    expected = idx + periods * idx.freq

    pd.testing.assert_index_equal(result, expected)
    assert result.freq == idx.freq


@given(
    n=st.integers(min_value=1, max_value=8),
    periods=st.integers(min_value=-20, max_value=20).filter(lambda x: x != 0),
)
def test_monthstart_explicit_daily_freq_uses_day_arithmetic_and_drops_freq(n, periods):
    # Month boundaries make it easier to catch implementations that accidentally reuse idx.freq.
    idx = pd.date_range("2011-01-01", periods=n, freq="MS")

    result = idx.shift(periods, freq="D")
    expected = idx + pd.to_timedelta(periods, unit="D")

    pd.testing.assert_index_equal(result, expected)
    assert result.freq is None


@given(
    n=st.integers(min_value=1, max_value=8),
    periods=st.integers(min_value=-20, max_value=20),
    freq_and_delta=FIXED_FREQ_AND_DELTA,
)
def test_datetime_shift_roundtrip_with_explicit_fixed_freq(n, periods, freq_and_delta):
    freq, _ = freq_and_delta
    idx = pd.date_range("2011-01-01", periods=n, freq="D")

    result = idx.shift(periods, freq=freq).shift(-periods, freq=freq)

    pd.testing.assert_index_equal(result, idx)


@given(
    n=st.integers(min_value=1, max_value=8),
    periods=st.integers(min_value=-20, max_value=20),
    freq_and_delta=FIXED_FREQ_AND_DELTA,
)
def test_datetime_shift_alias_and_timedelta_freq_agree(n, periods, freq_and_delta):
    freq, delta = freq_and_delta
    idx = pd.date_range("2011-01-01", periods=n, freq="D")

    by_alias = idx.shift(periods, freq=freq)
    by_timedelta = idx.shift(periods, freq=delta)

    pd.testing.assert_index_equal(by_alias, by_timedelta)


@given(
    n=st.integers(min_value=1, max_value=8),
    periods=st.integers(min_value=-20, max_value=20),
)
def test_datetime_explicit_matching_freq_agrees_with_implicit_shift(n, periods):
    idx = pd.date_range("2011-01-01", periods=n, freq="D")

    implicit = idx.shift(periods)
    explicit = idx.shift(periods, freq="D")

    pd.testing.assert_index_equal(explicit, implicit)
