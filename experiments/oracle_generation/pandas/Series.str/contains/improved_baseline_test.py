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


# NEW: Test default na behavior for object dtype - defaults to False in pandas 3.0
def test_contains_default_na_object_dtype():
    # For object dtype, default na is False (as of pandas 3.0)
    s = pd.Series(["dog", "cat", None, "mouse"])
    result = s.str.contains("og", regex=False)
    assert result.iloc[2] == False


# NEW: Test na parameter with pandas.NA for nullable StringDtype
def test_contains_na_pandas_NA_nullable_string():
    s = pd.Series(["dog", "cat", None, "mouse"], dtype="string")
    result = s.str.contains("og", regex=False)
    assert result.iloc[2] is pd.NA


# NEW: Test empty pattern - should match any string
def test_contains_empty_pattern():
    s = pd.Series(["anything", "hello", "world"])
    result = s.str.contains("", regex=False)
    assert result.all()


# NEW: Test pattern with only regex special characters (.)
def test_contains_dot_regex_special_char():
    # '.' in regex matches any character
    s = pd.Series(["40", "40.0", "41", "41.0", "35"])
    result = s.str.contains(".0", regex=True)
    expected = pd.Series([True, True, False, True, False])
    assert result.equals(expected)


# NEW: Test na=None is treated as False (not propagating NaN)
def test_contains_na_none_treated_as_false():
    s = pd.Series(["dog", None, "cat"])
    result = s.str.contains("og", regex=False, na=None)
    assert result.iloc[1] == False


# NEW: Test case=True with flags=re.IGNORECASE - flags should override case
def test_contains_case_with_flags_override():
    # When both case and flags are provided, flags takes precedence
    s = pd.Series(["DOG", "dog", "Dog"])
    result = s.str.contains("dog", case=True, flags=re.IGNORECASE, regex=True)
    assert result.all()


# NEW: Test regex pattern with quantifiers (*, +)
def test_contains_regex_quantifiers():
    s = pd.Series(["aaa", "aab", "baa", "bbb"])
    result = s.str.contains("a+", regex=True)
    assert result.iloc[0] == True
    assert result.iloc[1] == True
    assert result.iloc[2] == True
    assert result.iloc[3] == False


# NEW: Test with entirely empty series
def test_contains_empty_series():
    s = pd.Series([], dtype=object)
    result = s.str.contains("test", regex=False)
    assert len(result) == 0


# NEW: Test na parameter with explicit np.nan - in pandas 3.0 this behaves like False
def test_contains_na_nan_explicit():
    s = pd.Series(["dog", None, "cat"])
    result = s.str.contains("og", regex=False, na=float('nan'))
    # In pandas 3.0, np.nan is converted to False for boolean Series
    assert result.iloc[1] == False


# NEW: Test Unicode patterns
def test_contains_unicode_pattern():
    s = pd.Series(["können", "kaninchen", "français"])
    result = s.str.contains("ö", regex=False)
    assert result.iloc[0] == True
    assert result.iloc[1] == False
    assert result.iloc[2] == False


# NEW: Test with mixed case and regex pattern
def test_contains_regex_case_false():
    s = pd.Series(["DOG", "dog", "Dog", "DOG"])
    result = s.str.contains("dog", case=False, regex=True)
    assert result.all()


# NEW: Test pattern that uses regex anchors (^, $)
def test_contains_regex_anchors():
    s = pd.Series(["hello", "helloworld", "ohello", "hell"])
    result = s.str.contains("^hello", regex=True)
    assert result.iloc[0] == True
    assert result.iloc[1] == True
    assert result.iloc[2] == False
    assert result.iloc[3] == False
