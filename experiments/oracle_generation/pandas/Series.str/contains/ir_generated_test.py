"""
Property-based tests for pandas.Series.str.contains.
Generated from IR v2.

Focus: implementation bugs, not just happy-path confirmation.
Oracle strategy: re.search / Python 'in' operator as ground truth.

claude code generated
"""
import re

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

alphanumeric_text = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
    min_size=1,
    max_size=5,
)

non_null_str_series = st.builds(
    pd.Series,
    st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10),
)


@st.composite
def nullable_str_series(draw):
    """Series with at least one None/NaN element (avoids slow filter)."""
    elems = draw(st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=9))
    null_pos = draw(st.integers(min_value=0, max_value=len(elems)))
    elems.insert(null_pos, None)
    return pd.Series(elems)


# ---------------------------------------------------------------------------
# Group 1: Return type and shape invariants
# ---------------------------------------------------------------------------

@given(s=non_null_str_series, pat=alphanumeric_text)
def test_return_type_is_series(s, pat):
    result = s.str.contains(pat, regex=False)
    assert isinstance(result, pd.Series)


@given(s=non_null_str_series, pat=alphanumeric_text)
def test_result_length_equals_input_length(s, pat):
    result = s.str.contains(pat, regex=False)
    assert len(result) == len(s)


@given(s=nullable_str_series(), pat=alphanumeric_text)
def test_result_length_with_nan_input(s, pat):
    result = s.str.contains(pat, regex=False, na=False)
    assert len(result) == len(s)


# ---------------------------------------------------------------------------
# Group 2: Literal match (regex=False)
# ---------------------------------------------------------------------------

@given(s=non_null_str_series, pat=alphanumeric_text)
def test_literal_match_correctness(s, pat):
    """Each result must equal Python's 'in' operator for the same string."""
    result = s.str.contains(pat, regex=False, case=True)
    for i in range(len(s)):
        assert result.iloc[i] == (pat in s.iloc[i])


def test_regex_false_dot_is_literal():
    """
    Bug surface: if the impl calls re.search without re.escape(pat), '.'
    acts as a wildcard and 'axb' would incorrectly match pattern 'a.b'.
    """
    s = pd.Series(["a.b", "axb", "ab", "a.bc"])
    result = s.str.contains("a.b", regex=False)
    assert result.tolist() == [True, False, False, True]


@given(s=non_null_str_series)
def test_regex_false_dot_not_wildcard(s):
    """Property version: '.' in pattern must only match the literal dot character."""
    result = s.str.contains(".", regex=False)
    for i in range(len(s)):
        assert result.iloc[i] == ("." in s.iloc[i])


@given(s=non_null_str_series)
def test_regex_false_star_not_quantifier(s):
    """'*' is a quantifier in regex; as a literal it must only match the '*' char."""
    result = s.str.contains("*", regex=False)
    for i in range(len(s)):
        assert result.iloc[i] == ("*" in s.iloc[i])


@given(s=non_null_str_series)
def test_regex_false_empty_pattern_matches_everything(s):
    """Empty string is a substring of every string, including the empty string."""
    result = s.str.contains("", regex=False)
    assert result.all()


def test_regex_false_caret_dollar_are_literal():
    """
    '^' and '$' are anchors in regex mode; with regex=False they must be
    matched as literal characters, not anchors.
    """
    s = pd.Series(["^abc", "abc$", "abc", "^abc$"])

    result_caret = s.str.contains("^abc", regex=False)
    assert result_caret.tolist() == [True, False, False, True]

    result_dollar = s.str.contains("abc$", regex=False)
    assert result_dollar.tolist() == [False, True, False, True]


def test_regex_false_vs_regex_true_diverge_on_metachar():
    """
    Regression guard: regex=False and regex=True must differ when pat
    contains metacharacters.  If they agree here, regex=False is broken.
    """
    s = pd.Series(["100", "abc", "1a0", "no_match"])
    pat = r"\d+"

    result_re = s.str.contains(pat, regex=True)
    result_lit = s.str.contains(pat, regex=False)

    # regex=True: strings with at least one digit sequence match
    assert result_re.tolist() == [True, False, True, False]
    # regex=False: no element contains the literal backslash-d-plus
    assert result_lit.tolist() == [False, False, False, False]


def test_regex_false_dot_star_literal_vs_regex():
    """
    '.*' as regex matches every string; as a literal it only matches the
    two-character sequence '.*'.  This is a common confusion bug target.
    """
    s = pd.Series([".*", "hello", "x.*y"])

    result_lit = s.str.contains(".*", regex=False)
    assert result_lit.tolist() == [True, False, True]

    result_re = s.str.contains(".*", regex=True)
    assert result_re.all()


# ---------------------------------------------------------------------------
# Group 3: Regex match (regex=True)
# ---------------------------------------------------------------------------

@given(
    s=non_null_str_series,
    pat=st.sampled_from([r"\d", "[a-z]+", r"\w+", ".0", "^[A-Z]"]),
)
def test_regex_match_agrees_with_re_search(s, pat):
    """Oracle: re.search must agree with str.contains for every element."""
    result = s.str.contains(pat, regex=True, case=True, flags=0)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(
    s=non_null_str_series,
    pat=st.sampled_from(["house|dog", "a|b", "cat|dog", "[0-9]|[a-z]"]),
)
def test_regex_alternation_agrees_with_re_search(s, pat):
    """Alternation (|) must match either branch, consistent with re.search."""
    result = s.str.contains(pat, regex=True, case=True, flags=0)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i]))


@given(s=non_null_str_series)
def test_regex_true_empty_pattern_matches_all(s):
    """re.search('', s) is always truthy; str.contains('', regex=True) must be all-True."""
    result = s.str.contains("", regex=True)
    assert result.all()


# ---------------------------------------------------------------------------
# Group 4: case parameter
# ---------------------------------------------------------------------------

@given(s=non_null_str_series, pat=alphanumeric_text)
def test_case_false_literal_uses_lowercased_comparison(s, pat):
    """case=False must compare lowercased versions of both pat and element."""
    result = s.str.contains(pat, case=False, regex=False)
    for i in range(len(s)):
        assert result.iloc[i] == (pat.lower() in s.iloc[i].lower())


@given(s=non_null_str_series, pat=alphanumeric_text)
def test_case_false_regex_true_agrees_with_re_ignorecase(s, pat):
    """case=False with regex=True must produce re.IGNORECASE semantics."""
    result = s.str.contains(pat, case=False, regex=True)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i], re.IGNORECASE))


def test_case_false_finds_uppercase_value_in_element():
    s = pd.Series(["Hello", "WORLD", "python"])
    assert s.str.contains("hello", case=False).tolist() == [True, False, False]
    assert s.str.contains("world", case=False).tolist() == [False, True, False]


@given(s=non_null_str_series, pat=alphanumeric_text)
def test_case_insensitive_superset_of_case_sensitive(s, pat):
    """
    Metamorphic: every element matched by case=True must also be matched
    by case=False.  Case-insensitive search is strictly no stricter.
    """
    res_sens = s.str.contains(pat, case=True, regex=False)
    res_insens = s.str.contains(pat, case=False, regex=False)
    for i in range(len(s)):
        if res_sens.iloc[i]:
            assert res_insens.iloc[i]


@given(s=non_null_str_series, pat=alphanumeric_text)
def test_case_false_literal_equals_case_false_regex_for_alnum(s, pat):
    """
    Metamorphic: for purely alphanumeric patterns (no metacharacters),
    regex=True and regex=False with case=False must agree.
    """
    res_lit = s.str.contains(pat, case=False, regex=False)
    res_re = s.str.contains(pat, case=False, regex=True)
    pd.testing.assert_series_equal(res_lit, res_re, check_names=False, check_dtype=False)


# ---------------------------------------------------------------------------
# Group 5: flags parameter
# ---------------------------------------------------------------------------

@given(s=non_null_str_series, pat=alphanumeric_text)
def test_flags_ignorecase_agrees_with_re_ignorecase_oracle(s, pat):
    """flags=re.IGNORECASE must produce re.IGNORECASE semantics in the oracle."""
    result = s.str.contains(pat, flags=re.IGNORECASE, regex=True, case=True)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i], re.IGNORECASE))


@given(s=non_null_str_series, pat=alphanumeric_text)
def test_flags_ignorecase_with_case_true_equals_case_false_flags_zero(s, pat):
    """
    Metamorphic: flags=re.IGNORECASE (case=True) must give same result as
    case=False (flags=0).  Both paths must converge to case-insensitive matching.
    """
    res_via_flags = s.str.contains(pat, case=True, flags=re.IGNORECASE, regex=True)
    res_via_case = s.str.contains(pat, case=False, flags=0, regex=True)
    pd.testing.assert_series_equal(res_via_flags, res_via_case, check_names=False, check_dtype=False)


# ---------------------------------------------------------------------------
# Group 6: na parameter
# ---------------------------------------------------------------------------

@given(s=nullable_str_series(), pat=alphanumeric_text)
def test_na_false_fills_nan_positions_with_false(s, pat):
    result = s.str.contains(pat, regex=False, na=False)
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            assert result.iloc[i] == False


@given(s=nullable_str_series(), pat=alphanumeric_text)
def test_na_true_fills_nan_positions_with_true(s, pat):
    result = s.str.contains(pat, regex=False, na=True)
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            assert result.iloc[i] == True


@given(s=nullable_str_series(), pat=alphanumeric_text)
def test_na_param_does_not_affect_non_null_elements(s, pat):
    """na only governs missing-value positions; non-null results must be invariant."""
    res_na_false = s.str.contains(pat, regex=False, na=False)
    res_na_true = s.str.contains(pat, regex=False, na=True)
    for i in range(len(s)):
        if not pd.isna(s.iloc[i]):
            assert res_na_false.iloc[i] == res_na_true.iloc[i]


def test_na_default_propagates_nan_in_result():
    """When na is omitted, NaN input positions must produce NaN (not False/True) in output."""
    s = pd.Series(["dog", None, "cat"])
    result = s.str.contains("dog")
    assert result.iloc[0] == True
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == False


def test_na_false_result_contains_no_nans():
    s = pd.Series(["a", None, "b", None])
    result = s.str.contains("a", na=False)
    assert not result.isna().any()


def test_na_true_result_contains_no_nans():
    s = pd.Series(["a", None, "b", None])
    result = s.str.contains("a", na=True)
    assert not result.isna().any()


# ---------------------------------------------------------------------------
# Group 7: Interaction / adversarial multi-parameter tests
# ---------------------------------------------------------------------------

def test_case_na_regex_combined():
    """
    Adversarial combination: case=False + na=True + regex=True.
    NaN positions must become True; non-NaN positions must use case-insensitive matching.
    """
    s = pd.Series(["Hello", None, "WORLD"])
    result = s.str.contains("hello", case=False, regex=True, na=True)
    assert result.tolist() == [True, True, False]


def test_flags_and_case_false_combined_does_not_raise():
    """
    Combining case=False with flags=re.IGNORECASE is redundant but must not
    raise an error and must still produce case-insensitive results.
    """
    s = pd.Series(["Hello", "WORLD", "python"])
    result = s.str.contains("hello", case=False, flags=re.IGNORECASE, regex=True)
    assert result.iloc[0] == True


@given(
    s=non_null_str_series,
    pat=st.sampled_from(["house|dog", "a|b", "cat|dog"]),
)
def test_alternation_with_case_false(s, pat):
    """Alternation branches must all be matched case-insensitively when case=False."""
    result = s.str.contains(pat, case=False, regex=True)
    for i in range(len(s)):
        assert result.iloc[i] == bool(re.search(pat, s.iloc[i], re.IGNORECASE))


@given(s=nullable_str_series(), pat=alphanumeric_text)
def test_na_non_null_elements_still_obey_literal_oracle(s, pat):
    """Non-null elements must satisfy the literal oracle even when NaN neighbours exist."""
    result = s.str.contains(pat, regex=False, case=True, na=False)
    for i in range(len(s)):
        if not pd.isna(s.iloc[i]):
            assert result.iloc[i] == (pat in s.iloc[i])


@given(s=non_null_str_series, pat=alphanumeric_text)
def test_regex_false_dotted_pattern_is_still_literal(s, pat):
    """
    Pattern 'X.Y' with regex=False must match only the literal dot, not any char.
    Bug surface: re.search without re.escape would treat the dot as a wildcard.
    """
    dotted_pat = pat + "." + pat
    result = s.str.contains(dotted_pat, regex=False)
    for i in range(len(s)):
        assert result.iloc[i] == (dotted_pat in s.iloc[i])


def test_regex_true_default_not_literal():
    """
    Default regex=True means metacharacters are active.
    Verify that omitting regex=False triggers regex behaviour.
    """
    s = pd.Series(["abc123", "abcdef", "123"])
    # \d+ matches sequences of digits
    result = s.str.contains(r"\d+")  # regex=True by default
    assert result.tolist() == [True, False, True]


def test_case_true_does_not_match_wrong_case():
    """
    Regression: case=True (default) must NOT match elements that differ in case.
    """
    s = pd.Series(["Hello", "HELLO", "hello"])
    result = s.str.contains("hello", case=True, regex=False)
    assert result.tolist() == [False, False, True]


@given(s=nullable_str_series(), pat=alphanumeric_text)
def test_na_all_nan_positions_covered(s, pat):
    """Every NaN position must be filled with the na value — no position left as NaN."""
    for na_val in (True, False):
        result = s.str.contains(pat, regex=False, na=na_val)
        nan_mask = s.isna()
        assert (result[nan_mask] == na_val).all()
