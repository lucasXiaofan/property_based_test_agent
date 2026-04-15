import pandas as pd
import re
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ==========================
# BASELINE TESTS (from baseline_test.py)
# ==========================


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_contains_literal_pattern(data):
    """Baseline: Test literal pattern matching with regex=False"""
    s = pd.Series(data)
    pat = "og"
    result = s.str.contains(pat, regex=False)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    for i in range(len(s)):
        assert result.iloc[i] == (pat in s.iloc[i])


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_contains_regex_pattern(data):
    """Baseline: Test regex pattern with \\d"""
    s = pd.Series(data)
    pat = r"\d"
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_contains_case_sensitive(data):
    """Baseline: Test case sensitivity"""
    s = pd.Series(data)
    pat = "oG"
    result = s.str.contains(pat, case=True, regex=True)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_contains_alternation_regex(data):
    """Baseline: Test regex alternation with |"""
    s = pd.Series(data)
    pat = "house|dog"
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_contains_with_flags_ignorecase(data):
    """Baseline: Test IGNORECASE flag"""
    s = pd.Series(data)
    pat = "PARROT"
    result = s.str.contains(pat, flags=re.IGNORECASE, regex=True)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i], re.IGNORECASE))


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_contains_case_insensitive(data):
    """Baseline: Test case=False for case insensitive matching"""
    s = pd.Series(data)
    pat = "og"
    result = s.str.contains(pat, case=False, regex=False)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    for i in range(len(s)):
        assert result.iloc[i] == (pat.lower() in s.iloc[i].lower())


@given(
    data=st.lists(
        st.one_of(st.text(min_size=0, max_size=20), st.none()), min_size=2, max_size=10
    )
)
@settings(max_examples=100)
def test_contains_with_nan_na_false(data):
    """Baseline: Test NaN handling with na=False"""
    assume(any(pd.isna(x) for x in data))
    s = pd.Series(data)
    pat = "og"
    result = s.str.contains(pat, regex=False, na=False)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            assert result.iloc[i] == False


@given(
    data=st.lists(
        st.one_of(st.text(min_size=0, max_size=20), st.none()), min_size=2, max_size=10
    )
)
@settings(max_examples=100)
def test_contains_with_nan_na_true(data):
    """Baseline: Test NaN handling with na=True"""
    assume(any(pd.isna(x) for x in data))
    s = pd.Series(data)
    pat = "og"
    result = s.str.contains(pat, regex=False, na=True)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            assert result.iloc[i] == True


# ==========================
# NEW TESTS (from IR - high-stakes edge cases)
# ==========================


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_regex_metachar_dot_is_literal_when_regex_false(data):
    """[IR NEW] When regex=False, dot (.) should be treated as literal, not regex any-char"""
    s = pd.Series(data)
    pat = ".0"
    result = s.str.contains(pat, regex=False)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == (pat in s.iloc[i])


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_regex_metachar_dot_is_wildcard_when_regex_true(data):
    """[IR NEW] When regex=True, dot (.) matches any character - demonstrates the difference"""
    s = pd.Series(data)
    pat = ".0"
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(data=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_start_anchor_regex(data):
    """[IR NEW] Test ^ anchor - should match at string start"""
    s = pd.Series(data)
    pat = r"^A"
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(data=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_end_anchor_regex(data):
    """[IR NEW] Test $ anchor - should match at string end"""
    s = pd.Series(data)
    pat = r"Z$"
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_word_boundary_regex(data):
    """[IR NEW] Test \\b word boundary"""
    s = pd.Series(data)
    pat = r"\bword\b"
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_w_plus_regex_metachar(data):
    """[IR NEW] Test \\w+ metachar - matches word characters"""
    s = pd.Series(data)
    pat = r"\w+"
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_dot_star_greedy_regex(data):
    """[IR NEW] Test .* greedy matching"""
    s = pd.Series(data)
    pat = r".*abc"
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_character_class_regex(data):
    """[IR NEW] Test [a-z]+ character class"""
    s = pd.Series(data)
    pat = r"[a-z]+"
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_digit_class_regex(data):
    """[IR NEW] Test [0-9] character class"""
    s = pd.Series(data)
    pat = r"[0-9]"
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(
    data=st.lists(
        st.one_of(st.text(min_size=0, max_size=20), st.none()), min_size=2, max_size=10
    )
)
@settings(max_examples=100)
def test_ir_nan_with_regex_pattern(data):
    """[IR NEW] Test NaN handling with regex pattern instead of literal"""
    assume(any(pd.isna(x) for x in data))
    s = pd.Series(data)
    pat = r"\d"
    result = s.str.contains(pat, regex=True, na=False)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            assert result.iloc[i] == False


@given(
    data=st.lists(
        st.one_of(st.text(min_size=0, max_size=20), st.none()), min_size=2, max_size=10
    )
)
@settings(max_examples=100)
def test_ir_nan_with_alternation_regex(data):
    """[IR NEW] Test NaN handling with alternation regex pattern"""
    assume(any(pd.isna(x) for x in data))
    s = pd.Series(data)
    pat = r"cat|dog"
    result = s.str.contains(pat, regex=True, na=True)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            assert result.iloc[i] == True


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_case_and_flags_ignorecase_both_set(data):
    """[IR NEW] Edge case: both case=True and flags=IGNORECASE - flags should take precedence"""
    s = pd.Series(data)
    pat = "test"
    result = s.str.contains(pat, case=True, flags=re.IGNORECASE, regex=True)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i], re.IGNORECASE))


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_multichar_alternation_regex(data):
    """[IR NEW] Test multiple alternations: a|b|c pattern"""
    s = pd.Series(data)
    pat = r"cat|dog|bird"
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_empty_string_pattern_literal(data):
    """[IR NEW] Edge case: empty string as literal pattern"""
    s = pd.Series(data)
    pat = ""
    result = s.str.contains(pat, regex=False)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == (pat in s.iloc[i])


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_empty_string_pattern_regex(data):
    """[IR NEW] Edge case: empty string as regex pattern"""
    s = pd.Series(data)
    pat = ""
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_backslash_in_pattern_regex(data):
    """[IR NEW] Edge case: backslash in regex pattern"""
    s = pd.Series(data)
    pat = r"\\"
    result = s.str.contains(pat, regex=True)
    assert isinstance(result, pd.Series)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_ir_na_as_none_default(data):
    """[IR NEW] Edge case: using None as na parameter"""
    s = pd.Series(data)
    pat = "test"
    result = s.str.contains(pat, regex=False, na=None)
    assert isinstance(result, pd.Series)
