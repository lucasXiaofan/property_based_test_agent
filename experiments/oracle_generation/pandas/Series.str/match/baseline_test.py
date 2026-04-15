import hypothesis
import pandas as pd
import pytest
import re
from hypothesis import given, settings, assume
from hypothesis import strategies as st


@given(data=st.lists(st.text(min_size=0), min_size=1, max_size=20))
@settings(max_examples=100)
def test_match_returns_boolean_series(data):
    """Test that match returns a Series of boolean values."""
    ser = pd.Series(data)
    result = ser.str.match("test")
    assert result.dtype == bool


@given(data=st.lists(st.text(min_size=0), min_size=1, max_size=20))
@settings(max_examples=100)
def test_result_same_length_as_input(data):
    """Test that result has same length as input."""
    ser = pd.Series(data)
    result = ser.str.match("test")
    assert len(result) == len(ser)


@given(
    data=st.lists(st.text(min_size=0), min_size=1, max_size=20),
    pat=st.text(
        alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Nd"]),
        min_size=1,
        max_size=5,
    ),
)
@settings(max_examples=100)
def test_matches_start_of_string_only(data, pat):
    """Test that match only checks the start of the string."""
    ser = pd.Series(data)
    result = ser.str.match(pat)
    for i in range(len(ser)):
        expected = bool(re.match(pat, ser.iloc[i]))
        assert result.iloc[i] == expected


@given(data=st.lists(st.text(min_size=0), min_size=1, max_size=20))
@settings(max_examples=100)
def test_case_sensitive_by_default(data):
    """Test that case sensitive matching is the default."""
    ser = pd.Series(data)
    result = ser.str.match("Test")
    for i in range(len(ser)):
        expected = bool(re.match("Test", ser.iloc[i]))
        assert result.iloc[i] == expected


@given(data=st.lists(st.text(min_size=0), min_size=1, max_size=20))
@settings(max_examples=100)
def test_case_insensitive_matching(data):
    """Test case insensitive matching with case=False."""
    ser = pd.Series(data)
    result = ser.str.match("test", case=False)
    for i in range(len(ser)):
        expected = bool(re.match("test", ser.iloc[i], re.IGNORECASE))
        assert result.iloc[i] == expected


@given(data=st.lists(st.text(min_size=0), min_size=1, max_size=20))
@settings(max_examples=100)
def test_flags_ignorecase(data):
    """Test re.IGNORECASE flag."""
    ser = pd.Series(data)
    result = ser.str.match("test", flags=re.IGNORECASE)
    for i in range(len(ser)):
        expected = bool(re.match("test", ser.iloc[i], re.IGNORECASE))
        assert result.iloc[i] == expected


@given(
    data=st.lists(st.one_of(st.text(min_size=0), st.none()), min_size=1, max_size=20)
)
@settings(max_examples=100)
def test_na_fill_object_dtype_default(data):
    """Test that NaN values result in NaN for object dtype."""
    ser = pd.Series(data, dtype="object")
    result = ser.str.match("test")
    for i in range(len(ser)):
        if pd.isna(ser.iloc[i]):
            assert pd.isna(result.iloc[i])


@given(
    data=st.lists(
        st.one_of(st.text(min_size=0), st.just(pd.NA)), min_size=1, max_size=20
    )
)
@settings(max_examples=100)
def test_na_fill_string_dtype_default(data):
    """Test that pd.NA values result in pd.NA for StringDtype."""
    ser = pd.Series(data, dtype="string")
    result = ser.str.match("test")
    for i in range(len(ser)):
        if ser.iloc[i] is pd.NA:
            assert result.iloc[i] is pd.NA


@given(
    data=st.lists(st.one_of(st.text(min_size=0), st.none()), min_size=1, max_size=20)
)
@settings(max_examples=100)
def test_na_fill_custom_value_false(data):
    """Test custom na value (False) for object dtype."""
    ser = pd.Series(data, dtype="object")
    result = ser.str.match("test", na=False)
    for i in range(len(ser)):
        if pd.isna(ser.iloc[i]):
            assert result.iloc[i] is False or result.iloc[i] == False


@given(data=st.lists(st.text(min_size=0), min_size=1, max_size=20))
@settings(max_examples=100)
def test_compiled_regex(data):
    """Test with compiled regex pattern."""
    ser = pd.Series(data)
    pattern = re.compile("^test")
    result = ser.str.match(pattern)
    for i in range(len(ser)):
        expected = bool(pattern.match(ser.iloc[i]))
        assert result.iloc[i] == expected


def test_example_from_docstring():
    """Test the example from the pandas documentation."""
    ser = pd.Series(["horse", "eagle", "donkey"])
    result = ser.str.match("e")
    expected = pd.Series([False, True, False])
    assert list(result) == list(expected)
