"""
Baseline property-based tests for pandas.Series.str.match

Key semantics:
  - Uses re.match (anchored at the START of the string), not re.search.
  - Always operates in regex mode (no regex= parameter).
  - Parameters: pat, case (default True), flags (default 0), na (default depends on dtype).
"""
import re

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, series


# ---------------------------------------------------------------------------
# Helpers / shared strategies
# ---------------------------------------------------------------------------

text_elements = st.one_of(
    st.text(min_size=0, max_size=20),
    st.none(),
)

nonempty_text = st.text(min_size=1, max_size=20)

# Simple regex patterns that are syntactically valid and safe
safe_regex_pat = st.one_of(
    st.just("a"),
    st.just("\\d"),
    st.just("\\w"),
    st.just("\\s"),
    st.just("a|b"),
    st.just("a+"),
    st.just("a*"),
    st.just("a?"),
    st.just("[abc]"),
    st.just("[0-9]"),
    st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
        min_size=1,
        max_size=4,
    ),
)

str_series = st.lists(text_elements, min_size=1, max_size=10).map(pd.Series)


def _is_null(val):
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return False


# ---------------------------------------------------------------------------
# P1: Return type is always a Series of booleans with same length
# ---------------------------------------------------------------------------

@given(s=str_series, pat=safe_regex_pat)
@settings(max_examples=100)
def test_return_type_and_length(s, pat):
    """Result is a boolean Series with the same length as input."""
    result = s.str.match(pat, na=False)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    assert result.dtype == bool


# ---------------------------------------------------------------------------
# P2: Match correctness — result[i] == (re.match(pat, s[i]) is not None)
# ---------------------------------------------------------------------------

@given(s=str_series, pat=safe_regex_pat)
@settings(max_examples=100)
def test_match_correctness(s, pat):
    """result[i] matches re.match(pat, s[i]) is not None for non-null elements."""
    result = s.str.match(pat, na=False)
    for val, res in zip(s, result):
        if _is_null(val):
            assert res is False or res == False
        else:
            expected = re.match(pat, str(val)) is not None
            assert res == expected


# ---------------------------------------------------------------------------
# P3: case=True (default) — result is case-sensitive
# ---------------------------------------------------------------------------

@given(
    s=str_series,
    base=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu")),
        min_size=1,
        max_size=5,
    ),
)
@settings(max_examples=80)
def test_case_sensitive_default(s, base):
    """With case=True (default), uppercase and lowercase patterns differ."""
    upper_pat = base.upper()
    lower_pat = base.lower()
    if upper_pat == lower_pat:
        return  # no case difference to test
    result_upper = s.str.match(upper_pat, case=True, na=False)
    result_lower = s.str.match(lower_pat, case=True, na=False)
    for val, ru, rl in zip(s, result_upper, result_lower):
        if _is_null(val):
            continue
        assert ru == (re.match(upper_pat, str(val)) is not None)
        assert rl == (re.match(lower_pat, str(val)) is not None)


# ---------------------------------------------------------------------------
# P4: case=False — case-insensitive matching
# ---------------------------------------------------------------------------

@given(s=str_series, pat=safe_regex_pat)
@settings(max_examples=80)
def test_case_insensitive(s, pat):
    """With case=False, result[i] matches re.match(pat, s[i], re.IGNORECASE)."""
    result = s.str.match(pat, case=False, na=False)
    for val, res in zip(s, result):
        if _is_null(val):
            assert res is False or res == False
        else:
            expected = re.match(pat, str(val), re.IGNORECASE) is not None
            assert res == expected


# ---------------------------------------------------------------------------
# P5: na parameter controls fill value for missing entries
# ---------------------------------------------------------------------------

@given(
    s=str_series,
    pat=safe_regex_pat,
    na_val=st.one_of(st.just(True), st.just(False)),
)
@settings(max_examples=80)
def test_na_fill_value(s, pat, na_val):
    """Missing values are replaced by the scalar `na` fill value."""
    values = [None] + [v for v in s]
    s_with_nan = pd.Series(values, dtype=object)
    result = s_with_nan.str.match(pat, na=na_val)
    # First element is None/NaN → result should equal na_val
    assert result.iloc[0] == na_val


# ---------------------------------------------------------------------------
# P6: flags=re.IGNORECASE equivalent to case=False
# ---------------------------------------------------------------------------

@given(s=str_series, pat=safe_regex_pat)
@settings(max_examples=80)
def test_flags_ignorecase_equivalent(s, pat):
    """flags=re.IGNORECASE and case=False should give identical results."""
    # Do not pass case= when using flags=re.IGNORECASE; pandas infers case from flags.
    result_flags = s.str.match(pat, flags=re.IGNORECASE, na=False)
    result_case = s.str.match(pat, case=False, na=False)
    pd.testing.assert_series_equal(result_flags, result_case)


# ---------------------------------------------------------------------------
# P7: Empty pattern always matches every non-null string
# ---------------------------------------------------------------------------

@given(s=str_series)
@settings(max_examples=50)
def test_empty_pattern_always_matches(s):
    """re.match('', s) is never None, so every non-null element returns True."""
    result = s.str.match("", na=False)
    for val, res in zip(s, result):
        if _is_null(val):
            assert res is False or res == False
        else:
            assert res is True or res == True


# ---------------------------------------------------------------------------
# P8: Impossible pattern returns all False for non-null elements
# ---------------------------------------------------------------------------

@given(s=str_series)
@settings(max_examples=50)
def test_impossible_pattern_all_false(s):
    """A contradiction pattern that can never match returns False for all non-null."""
    pat = "(?!x)x"  # never matches
    result = s.str.match(pat, na=False)
    for val, res in zip(s, result):
        if _is_null(val):
            assert res is False or res == False
        else:
            assert res is False or res == False


# ---------------------------------------------------------------------------
# P9: match is anchored at start (unlike contains which uses re.search)
# ---------------------------------------------------------------------------

@given(s=st.lists(st.text(min_size=2, max_size=15), min_size=1, max_size=8).map(pd.Series))
@settings(max_examples=80)
def test_match_anchored_at_start(s):
    """match only checks the start of the string; a mid-string pat returns False."""
    # Build strings that contain "zzz" but never at the start
    strings = ["a" + v + "zzz" for v in s]
    ser = pd.Series(strings)
    # "zzz" is never at position 0
    result = ser.str.match("zzz", na=False)
    for res in result:
        assert res is False or res == False

    # Verify the same pattern IS found by contains (which uses re.search)
    result_contains = ser.str.contains("zzz", regex=True, na=False)
    for res in result_contains:
        assert res is True or res == True


# ---------------------------------------------------------------------------
# P10: Idempotency — calling match twice with same args gives same result
# ---------------------------------------------------------------------------

@given(s=str_series, pat=safe_regex_pat)
@settings(max_examples=50)
def test_idempotency(s, pat):
    """Calling match twice with identical arguments produces the same result."""
    r1 = s.str.match(pat, na=False)
    r2 = s.str.match(pat, na=False)
    pd.testing.assert_series_equal(r1, r2)


# ---------------------------------------------------------------------------
# P11: Index input mirrors Series behaviour
# ---------------------------------------------------------------------------

@given(
    values=st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=8),
    pat=safe_regex_pat,
)
@settings(max_examples=60)
def test_index_behavior_matches_series(values, pat):
    """pd.Index.str.match should agree with pd.Series.str.match element-wise."""
    s = pd.Series(values)
    idx = pd.Index(values)
    result_series = s.str.match(pat, na=False)
    result_index = idx.str.match(pat, na=False)
    assert list(result_series) == list(result_index)


# ---------------------------------------------------------------------------
# P12: All NaN with na=True returns all True
# ---------------------------------------------------------------------------

def test_all_nan_na_true():
    """When every element is NaN and na=True, every result is True."""
    s = pd.Series([np.nan, np.nan, np.nan], dtype=object)
    result = s.str.match("anything", na=True)
    assert list(result) == [True, True, True]


# ---------------------------------------------------------------------------
# P13: All NaN with na=False returns all False
# ---------------------------------------------------------------------------

def test_all_nan_na_false():
    """When every element is NaN and na=False, every result is False."""
    s = pd.Series([np.nan, np.nan, np.nan], dtype=object)
    result = s.str.match("anything", na=False)
    assert list(result) == [False, False, False]


# ---------------------------------------------------------------------------
# P14: match vs fullmatch — match permits trailing characters
# ---------------------------------------------------------------------------

def test_match_vs_fullmatch_trailing_chars():
    """match allows trailing characters after the pattern; fullmatch does not."""
    ser = pd.Series(["ab", "abc", "a"])
    # Pattern "a" should match all (all start with 'a')
    result_match = ser.str.match("a")
    assert list(result_match) == [True, True, True]
    # fullmatch "a" only matches the string that is exactly "a"
    result_fullmatch = ser.str.fullmatch("a")
    assert list(result_fullmatch) == [False, False, True]


# ---------------------------------------------------------------------------
# P15: Docstring example — basic correctness smoke test
# ---------------------------------------------------------------------------

def test_docstring_example():
    """Reproduces the example from the official pandas documentation."""
    ser = pd.Series(["horse", "eagle", "donkey"])
    result = ser.str.match("e")
    expected = pd.Series([False, True, False])
    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# P16: Compiled regex pattern behaves identically to string pattern
# ---------------------------------------------------------------------------

@given(s=str_series, pat=safe_regex_pat)
@settings(max_examples=60)
def test_compiled_regex_same_as_string(s, pat):
    """A compiled re.Pattern passed as pat gives the same result as the string."""
    compiled = re.compile(pat)
    result_str = s.str.match(pat, na=False)
    result_compiled = s.str.match(compiled, na=False)
    pd.testing.assert_series_equal(result_str, result_compiled)
