import pandas as pd
import re
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


@given(data=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10))
@settings(max_examples=100)
def test_contains_literal_pattern(data):
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
    assume(any(pd.isna(x) for x in data))
    s = pd.Series(data)
    pat = "og"
    result = s.str.contains(pat, regex=False, na=True)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            assert result.iloc[i] == True
