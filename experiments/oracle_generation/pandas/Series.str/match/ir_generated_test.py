"""
Property-based tests for Series.str.match generated from IR v2.

Target: pandas.Series.str.match(pat, case=<no_default>, flags=<no_default>, na=<no_default>)
Key semantics: anchored at string START (re.match), not anywhere (re.search).
claude code generated
"""
import re

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Alphanumeric text — safe to use literally as a regex pattern
_alphanum_text = st.text(
    alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Nd"]),
    min_size=1,
    max_size=4,
)

_any_text = st.text(min_size=0, max_size=10)

_words_mixed_case = st.lists(
    st.text(alphabet="abcABCDEF123 ", min_size=0, max_size=8),
    min_size=1,
    max_size=15,
)


# ===========================================================================
# Group 1 – Return type and structure
# ===========================================================================


@given(
    lst=st.lists(_any_text, min_size=1, max_size=20),
    pat=_alphanum_text,
)
@settings(max_examples=60)
def test_returns_bool_dtype_no_nulls(lst, pat):
    """object-dtype series with no NULLs → result dtype is bool."""
    s = pd.Series(lst)
    result = s.str.match(pat)
    assert result.dtype == bool


@given(
    lst=st.lists(_any_text, min_size=1, max_size=20),
    pat=_alphanum_text,
)
@settings(max_examples=60)
def test_result_length_equals_input_length(lst, pat):
    """Output length must equal input length for all inputs."""
    s = pd.Series(lst)
    result = s.str.match(pat)
    assert len(result) == len(s)


@given(
    lst=st.lists(_any_text, min_size=1, max_size=20),
    pat=_alphanum_text,
)
@settings(max_examples=60)
def test_non_null_inputs_produce_no_nulls(lst, pat):
    """No NULLs in input → no NULLs in output."""
    s = pd.Series(lst)
    result = s.str.match(pat)
    assert result.notna().all()


# ===========================================================================
# Group 2 – Correctness: result agrees with re.match (anchored at start)
# ===========================================================================


@given(
    lst=st.lists(_any_text, min_size=1, max_size=20),
    pat=st.sampled_from(
        ["^[a-z]", "^[A-Z]", "^[0-9]", r"^\w", r"^\d", "[a-z]", "[0-9]", "a"]
    ),
)
@settings(max_examples=80)
def test_result_agrees_with_re_match(lst, pat):
    """Every element must agree with bool(re.match(pat, s)) — the differential oracle."""
    s = pd.Series(lst)
    result = s.str.match(pat)
    for i, val in enumerate(s):
        expected = bool(re.match(pat, val))
        assert result.iloc[i] == expected, (
            f"Mismatch at index {i}: str.match={result.iloc[i]}, "
            f"re.match={expected}, val={val!r}, pat={pat!r}"
        )


def test_documented_example_single_char():
    """Exact documented example: pat='e' on ['horse','eagle','donkey'] → [F,T,F]."""
    s = pd.Series(["horse", "eagle", "donkey"])
    result = s.str.match("e")
    assert list(result) == [False, True, False]


# ===========================================================================
# Group 3 – Anchoring: match ≠ search (highest bug-surface area)
# ===========================================================================


def test_match_is_anchored_not_a_search():
    """
    'e' appears in 'horse' (position 1) and 'donkey' (position 1),
    but match must only return True for strings that START with 'e'.
    Bug: confusing str.match (re.match) with str.contains (re.search).
    """
    s = pd.Series(["horse", "eagle", "donkey"])
    result = s.str.match("e")
    assert list(result) == [False, True, False]


def test_pattern_without_caret_still_anchored():
    """
    Even without '^', str.match anchors at position 0.
    'b' matches 'bcd' and 'bcda' but NOT 'abcd'.
    Bug: implementation using re.search would wrongly match 'abcd'.
    """
    s = pd.Series(["abcd", "bcd", "bcda"])
    result = s.str.match("b")
    assert list(result) == [False, True, True]


def test_match_vs_contains_mid_string():
    """
    Strings that contain the pattern only mid-string must NOT match.
    '1abc' starts with a digit, so '[a-z]' should not match it.
    """
    s = pd.Series(["1abc", "abc1", " abc", "abc"])
    result = s.str.match("[a-z]")
    assert list(result) == [False, True, False, True]


@given(
    lst=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
    pat=st.sampled_from(["[a-z]", "[0-9]", ".", r"\w"]),
)
@settings(max_examples=80)
def test_match_true_implies_contains_true(lst, pat):
    """
    Metamorphic invariant: if str.match=True then str.contains must also be True,
    because re.match success at position 0 implies re.search success.
    Bug: str.match returning True when re.search would return False.
    """
    s = pd.Series(lst)
    match_r = s.str.match(pat, na=False)
    contains_r = s.str.contains(pat, na=False)
    # match is strictly narrower than contains
    false_positive = match_r & ~contains_r
    assert false_positive.sum() == 0, (
        f"match=True but contains=False for: "
        f"{s[false_positive].tolist()}"
    )


# ===========================================================================
# Group 4 – NULL handling (na= parameter and dtype-specific propagation)
# ===========================================================================


@given(
    lst=st.lists(
        st.one_of(_any_text, st.just(float("nan"))),
        min_size=2,
        max_size=20,
    ),
    pat=_alphanum_text,
)
@settings(max_examples=60)
def test_object_dtype_null_propagates_as_nan_by_default(lst, pat):
    """object-dtype NaN positions → NaN in result (default na propagation).
    Must use explicit dtype=object; pandas 3.0 infers 'str' for mixed string/NaN lists.
    """
    assume(any(isinstance(v, float) and np.isnan(v) for v in lst))
    s = pd.Series(lst, dtype=object)
    result = s.str.match(pat)
    for i, val in enumerate(s):
        if pd.isna(val):
            assert pd.isna(result.iloc[i]), (
                f"Expected NaN at index {i}, got {result.iloc[i]!r}"
            )


@given(
    lst=st.lists(
        st.one_of(_any_text, st.just(pd.NA)),
        min_size=2,
        max_size=20,
    ),
    pat=_alphanum_text,
)
@settings(max_examples=60)
def test_string_dtype_null_propagates_as_pd_na_by_default(lst, pat):
    """StringDtype pd.NA positions → pd.NA in result (default na propagation)."""
    assume(any(v is pd.NA for v in lst))
    s = pd.Series(lst, dtype="string")
    result = s.str.match(pat)
    for i in range(len(s)):
        if s.iloc[i] is pd.NA:
            assert result.iloc[i] is pd.NA, (
                f"Expected pd.NA at index {i}, got {result.iloc[i]!r}"
            )


@given(
    lst=st.lists(
        st.one_of(_any_text, st.just(float("nan"))),
        min_size=2,
        max_size=20,
    ),
    pat=_alphanum_text,
)
@settings(max_examples=60)
def test_na_false_fills_null_positions_with_false(lst, pat):
    """na=False → NaN positions in result become False."""
    assume(any(isinstance(v, float) and np.isnan(v) for v in lst))
    s = pd.Series(lst, dtype=object)
    result = s.str.match(pat, na=False)
    for i, val in enumerate(s):
        if pd.isna(val):
            assert result.iloc[i] == False, (
                f"Expected False at index {i}, got {result.iloc[i]!r}"
            )


@given(
    lst=st.lists(
        st.one_of(_any_text, st.just(float("nan"))),
        min_size=2,
        max_size=20,
    ),
    pat=_alphanum_text,
)
@settings(max_examples=60)
def test_na_true_fills_null_positions_with_true(lst, pat):
    """na=True → NaN positions in result become True."""
    assume(any(isinstance(v, float) and np.isnan(v) for v in lst))
    s = pd.Series(lst, dtype=object)
    result = s.str.match(pat, na=True)
    for i, val in enumerate(s):
        if pd.isna(val):
            assert result.iloc[i] == True, (
                f"Expected True at index {i}, got {result.iloc[i]!r}"
            )


def test_all_null_series_na_false_gives_all_false():
    """All-NaN series with na=False → all False."""
    s = pd.Series([np.nan, np.nan, np.nan], dtype=object)
    result = s.str.match("abc", na=False)
    assert list(result) == [False, False, False]


def test_all_null_series_na_true_gives_all_true():
    """All-NaN series with na=True → all True."""
    s = pd.Series([np.nan, np.nan, np.nan], dtype=object)
    result = s.str.match("abc", na=True)
    assert list(result) == [True, True, True]


def test_string_dtype_na_false_fills_pd_na_with_false():
    """StringDtype with na=False: pd.NA positions → False (explicit override)."""
    s = pd.Series(["abc", pd.NA, "xyz"], dtype="string")
    result = s.str.match("^a", na=False)
    assert result.iloc[0] == True
    assert result.iloc[1] == False
    assert result.iloc[2] == False


def test_na_parameter_does_not_affect_non_null_results():
    """
    na= must only affect null positions, not non-null ones.
    Bug surface: na parameter leaking into non-null computations.
    """
    s = pd.Series(["abc", np.nan, "xyz"], dtype=object)
    result_na_false = s.str.match("^a", na=False)
    result_na_true = s.str.match("^a", na=True)
    # Non-null positions identical regardless of na value
    assert result_na_false.iloc[0] == result_na_true.iloc[0]
    assert result_na_false.iloc[2] == result_na_true.iloc[2]
    # Null position controlled by na=
    assert result_na_false.iloc[1] == False
    assert result_na_true.iloc[1] == True


# ===========================================================================
# Group 5 – Case sensitivity (case= parameter)
# ===========================================================================


def test_case_true_is_case_sensitive_by_default():
    """Default (case=True) must NOT match uppercase when pattern is lowercase."""
    s = pd.Series(["Hello", "hello", "HELLO"])
    result = s.str.match("^h")
    assert list(result) == [False, True, False]


def test_case_false_matches_all_cases():
    """case=False must match uppercase, lowercase, and mixed with a lowercase pattern."""
    s = pd.Series(["Hello", "hello", "HELLO"])
    result = s.str.match("^h", case=False)
    assert list(result) == [True, True, True]


@given(
    words=st.lists(
        st.text(alphabet="abcdefABCDEF123", min_size=1, max_size=8),
        min_size=1,
        max_size=15,
    )
)
@settings(max_examples=60)
def test_case_false_agrees_with_re_ignorecase(words):
    """case=False must match re.match(pat, s, re.IGNORECASE) for all elements."""
    pat = "^[a-f]"
    s = pd.Series(words)
    result = s.str.match(pat, case=False)
    for i, w in enumerate(words):
        expected = bool(re.match(pat, w, re.IGNORECASE))
        assert result.iloc[i] == expected, (
            f"Mismatch at index {i}: val={w!r}, got={result.iloc[i]}, want={expected}"
        )


# ===========================================================================
# Group 6 – flags= parameter
# ===========================================================================


@given(
    words=st.lists(
        st.text(alphabet="abcABC123", min_size=0, max_size=8),
        min_size=1,
        max_size=15,
    )
)
@settings(max_examples=60)
def test_flags_ignorecase_agrees_with_re_ignorecase(words):
    """flags=re.IGNORECASE must agree with re.match(pat, s, re.IGNORECASE)."""
    pat = "^[a-c]"
    s = pd.Series(words)
    result = s.str.match(pat, flags=re.IGNORECASE)
    for i, w in enumerate(words):
        expected = bool(re.match(pat, w, re.IGNORECASE))
        assert result.iloc[i] == expected


def test_flags_ignorecase_produces_same_result_as_case_false():
    """
    flags=re.IGNORECASE and case=False should be functionally equivalent.
    Bug surface: the two code paths handling case-insensitivity diverging.
    """
    s = pd.Series(["Apple", "apple", "APPLE", "123apple"])
    pat = "^apple"
    result_case = s.str.match(pat, case=False)
    result_flags = s.str.match(pat, flags=re.IGNORECASE)
    pd.testing.assert_series_equal(result_case, result_flags)


# ===========================================================================
# Group 7 – Compiled regex
# ===========================================================================


@given(
    lst=st.lists(_any_text, min_size=1, max_size=20),
)
@settings(max_examples=60)
def test_compiled_regex_agrees_with_string_pattern(lst):
    """Compiled re.compile(pat) and string pat must yield identical results."""
    pat_str = "^[a-z]"
    pat_obj = re.compile(pat_str)
    s = pd.Series(lst)
    result_str = s.str.match(pat_str)
    result_compiled = s.str.match(pat_obj)
    pd.testing.assert_series_equal(result_str, result_compiled)


def test_compiled_regex_with_ignorecase_flag():
    """Compiled regex with re.IGNORECASE embedded should match case-insensitively."""
    s = pd.Series(["Hello", "hello", "HELLO", "xyz"])
    pat = re.compile("^h", re.IGNORECASE)
    result = s.str.match(pat)
    assert list(result) == [True, True, True, False]


# ===========================================================================
# Group 8 – Edge cases and degenerate inputs
# ===========================================================================


def test_empty_string_with_char_pattern():
    """Empty string never matches a character-class pattern."""
    s = pd.Series(["", "a", ""])
    result = s.str.match("[a-z]")
    assert list(result) == [False, True, False]


def test_empty_string_with_empty_pattern():
    """Empty pattern '' matches the start of every string (including empty)."""
    s = pd.Series(["", "abc", "xyz"])
    result = s.str.match("")
    assert list(result) == [True, True, True]


@given(lst=st.lists(_any_text, min_size=1, max_size=20))
@settings(max_examples=50)
def test_caret_only_matches_all_strings(lst):
    """
    '^' matches the empty-string anchor at position 0 for every string.
    All results must be True (no NULLs in input).
    """
    s = pd.Series(lst)
    result = s.str.match("^", na=False)
    assert result.all()


def test_single_element_series():
    """Minimal input: single-element series returns single-element result."""
    s = pd.Series(["abc"])
    result = s.str.match("^a")
    assert len(result) == 1
    assert result.iloc[0] == True


def test_single_null_series_na_false():
    """Single-element all-null series with na=False returns [False]."""
    s = pd.Series([np.nan], dtype=object)
    result = s.str.match("^a", na=False)
    assert len(result) == 1
    assert result.iloc[0] == False


@given(
    lst=st.lists(_any_text, min_size=1, max_size=20),
    pat=_alphanum_text,
)
@settings(max_examples=50)
def test_idempotent_double_call(lst, pat):
    """Calling str.match twice on the same series yields identical results (no mutation)."""
    s = pd.Series(lst)
    result1 = s.str.match(pat)
    result2 = s.str.match(pat)
    pd.testing.assert_series_equal(result1, result2)


def test_mixed_nulls_and_non_nulls_object_dtype():
    """Mixed series: non-null positions correct, null positions NaN."""
    s = pd.Series(["abc", np.nan, "xyz", np.nan], dtype=object)
    result = s.str.match("^a")
    assert result.iloc[0] == True
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == False
    assert pd.isna(result.iloc[3])


@given(
    lst=st.lists(_any_text, min_size=1, max_size=20),
    pat=_alphanum_text,
)
@settings(max_examples=60)
def test_string_dtype_non_null_positions_are_bool(lst, pat):
    """
    StringDtype result: non-null positions are True/False (not pd.NA or other).
    Bug surface: dtype coercion returning non-bool sentinels for non-null rows.
    """
    s = pd.Series(lst, dtype="string")
    result = s.str.match(pat)
    for i in range(len(s)):
        r = result.iloc[i]
        if s.iloc[i] is not pd.NA:
            assert r in (True, False), f"Expected bool at index {i}, got {r!r}"
