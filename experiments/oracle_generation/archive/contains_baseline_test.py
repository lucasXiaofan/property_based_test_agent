"""
Baseline property-based tests for pandas.Series.str.contains
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

literal_pat = st.text(min_size=0, max_size=5)

# Simple regex patterns that are syntactically valid
safe_regex_pat = st.one_of(
    st.just(""),
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
    st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")), min_size=1, max_size=4),
)

str_series = st.lists(text_elements, min_size=1, max_size=10).map(pd.Series)


# ---------------------------------------------------------------------------
# P1: Return type is always a Series of booleans (or NA) with same length
# ---------------------------------------------------------------------------

@given(s=str_series, pat=literal_pat)
@settings(max_examples=100)
def test_return_type_and_length(s, pat):
    """Result is a boolean Series with the same length as input."""
    result = s.str.contains(pat, regex=False, na=False)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    assert result.dtype == bool


# ---------------------------------------------------------------------------
# P2: Literal match correctness — result[i] is True iff pat in s[i]
# ---------------------------------------------------------------------------

@given(s=str_series, pat=literal_pat)
@settings(max_examples=100)
def test_literal_match_correctness(s, pat):
    """With regex=False, result matches Python's `in` operator on non-null elements."""
    result = s.str.contains(pat, regex=False, na=False)
    for val, res in zip(s, result):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            # na=False means missing values → False
            assert res is False or res == False
        else:
            assert res == (pat in str(val))


# ---------------------------------------------------------------------------
# P3: case=True (default) — result is case-sensitive
# ---------------------------------------------------------------------------

@given(s=str_series, base=nonempty_text)
@settings(max_examples=80)
def test_case_sensitive_default(s, base):
    """With case=True (default), uppercase and lowercase patterns differ."""
    upper_pat = base.upper()
    lower_pat = base.lower()
    if upper_pat == lower_pat:
        return  # nothing to test when there is no case difference
    result_upper = s.str.contains(upper_pat, case=True, regex=False, na=False)
    result_lower = s.str.contains(lower_pat, case=True, regex=False, na=False)
    # They may differ — the key invariant is each is self-consistent
    for val, ru, rl in zip(s, result_upper, result_lower):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        assert ru == (upper_pat in str(val))
        assert rl == (lower_pat in str(val))


# ---------------------------------------------------------------------------
# P4: case=False — case-insensitive matching
# ---------------------------------------------------------------------------

@given(s=str_series, pat=literal_pat)
@settings(max_examples=80)
def test_case_insensitive_literal(s, pat):
    """With case=False and regex=False, search is case-insensitive."""
    result = s.str.contains(pat, case=False, regex=False, na=False)
    for val, res in zip(s, result):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            assert res is False or res == False
        else:
            assert res == (pat.lower() in str(val).lower())


# ---------------------------------------------------------------------------
# P5: na parameter controls fill value for missing entries
# ---------------------------------------------------------------------------

@given(
    s=str_series,
    pat=literal_pat,
    na_val=st.one_of(st.just(True), st.just(False)),
)
@settings(max_examples=80)
def test_na_fill_value(s, pat, na_val):
    """Missing values are replaced by the boolean `na` fill value."""
    # Build an object-dtype series with a forced NaN at position 0 so the
    # .str accessor is always available and NA handling is unambiguous.
    values = [None] + [v for v in s]
    s_with_nan = pd.Series(values, dtype=object)
    result = s_with_nan.str.contains(pat, regex=False, na=na_val)
    # First element is NaN → its result should equal na_val
    assert result.iloc[0] == na_val


# ---------------------------------------------------------------------------
# P6: regex=True — pattern treated as regular expression
# ---------------------------------------------------------------------------

@given(s=str_series, pat=safe_regex_pat)
@settings(max_examples=80)
def test_regex_match_correctness(s, pat):
    """With regex=True, result[i] matches re.search(pat, s[i]) is not None."""
    result = s.str.contains(pat, regex=True, na=False)
    for val, res in zip(s, result):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            assert res is False or res == False
        else:
            expected = re.search(pat, str(val)) is not None
            assert res == expected


# ---------------------------------------------------------------------------
# P7: flags=re.IGNORECASE equivalent to case=False for regex mode
# ---------------------------------------------------------------------------

@given(s=str_series, pat=safe_regex_pat)
@settings(max_examples=80)
def test_flags_ignorecase_equivalent(s, pat):
    """flags=re.IGNORECASE and case=False should give identical results in regex mode."""
    result_flags = s.str.contains(pat, case=True, flags=re.IGNORECASE, regex=True, na=False)
    result_case = s.str.contains(pat, case=False, regex=True, na=False)
    pd.testing.assert_series_equal(result_flags, result_case)


# ---------------------------------------------------------------------------
# P8: Empty pattern always matches (every string contains the empty string)
# ---------------------------------------------------------------------------

@given(s=str_series)
@settings(max_examples=50)
def test_empty_pattern_always_matches(s):
    """An empty literal pattern is contained in every non-null string."""
    result = s.str.contains("", regex=False, na=False)
    for val, res in zip(s, result):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            assert res is False or res == False
        else:
            assert res is True or res == True


# ---------------------------------------------------------------------------
# P9: Pattern that matches nothing returns all False for non-null
# ---------------------------------------------------------------------------

@given(s=str_series)
@settings(max_examples=50)
def test_impossible_pattern_all_false(s):
    """A regex pattern that can never match returns False for all non-null elements."""
    pat = "(?!x)x"  # contradiction — never matches
    result = s.str.contains(pat, regex=True, na=False)
    for val, res in zip(s, result):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            assert res is False or res == False
        else:
            assert res is False or res == False


# ---------------------------------------------------------------------------
# P10: regex=False vs regex=True with a literal (non-special) pattern
# ---------------------------------------------------------------------------

@given(
    s=str_series,
    pat=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
        min_size=1,
        max_size=5,
    ),
)
@settings(max_examples=80)
def test_regex_false_vs_true_plain_literal(s, pat):
    """For alphanumeric patterns (no regex meta-chars), regex=True and regex=False agree."""
    result_literal = s.str.contains(pat, regex=False, na=False)
    result_regex = s.str.contains(pat, regex=True, na=False)
    pd.testing.assert_series_equal(result_literal, result_regex)


# ---------------------------------------------------------------------------
# P11: Idempotency — calling contains twice with same args gives same result
# ---------------------------------------------------------------------------

@given(s=str_series, pat=literal_pat)
@settings(max_examples=50)
def test_idempotency(s, pat):
    """Calling contains twice with identical arguments produces the same result."""
    r1 = s.str.contains(pat, regex=False, na=False)
    r2 = s.str.contains(pat, regex=False, na=False)
    pd.testing.assert_series_equal(r1, r2)


# ---------------------------------------------------------------------------
# P12: OR regex equivalent to logical OR of individual contains
# ---------------------------------------------------------------------------

@given(
    s=str_series,
    pat1=st.text(alphabet="abcde", min_size=1, max_size=3),
    pat2=st.text(alphabet="abcde", min_size=1, max_size=3),
)
@settings(max_examples=60)
def test_or_regex(s, pat1, pat2):
    """s.str.contains('a|b') == s.str.contains('a') | s.str.contains('b')."""
    combined = s.str.contains(f"{pat1}|{pat2}", regex=True, na=False)
    left = s.str.contains(pat1, regex=True, na=False)
    right = s.str.contains(pat2, regex=True, na=False)
    pd.testing.assert_series_equal(combined, left | right)


# ---------------------------------------------------------------------------
# P13: Index input mirrors Series behaviour
# ---------------------------------------------------------------------------

@given(values=st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=8), pat=literal_pat)
@settings(max_examples=60)
def test_index_behavior_matches_series(values, pat):
    """pd.Index.str.contains should agree with pd.Series.str.contains element-wise."""
    s = pd.Series(values)
    idx = pd.Index(values)
    result_series = s.str.contains(pat, regex=False, na=False)
    result_index = idx.str.contains(pat, regex=False, na=False)
    assert list(result_series) == list(result_index)


# ---------------------------------------------------------------------------
# P14: Series of all NaN with na=True returns all True
# ---------------------------------------------------------------------------

def test_all_nan_na_true():
    """When every element is NaN and na=True, every result is True."""
    # Must use object dtype so .str accessor is available
    s = pd.Series([np.nan, np.nan, np.nan], dtype=object)
    result = s.str.contains("anything", regex=False, na=True)
    assert list(result) == [True, True, True]


# ---------------------------------------------------------------------------
# P15: Series of all NaN with na=False returns all False
# ---------------------------------------------------------------------------

def test_all_nan_na_false():
    """When every element is NaN and na=False, every result is False."""
    s = pd.Series([np.nan, np.nan, np.nan], dtype=object)
    result = s.str.contains("anything", regex=False, na=False)
    assert list(result) == [False, False, False]
