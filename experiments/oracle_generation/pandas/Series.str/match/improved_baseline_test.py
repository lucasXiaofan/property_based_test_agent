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


# ============================================================
# NEW TESTS: Edge cases and non-happy-path scenarios
# ============================================================


def test_empty_series():
    """Test empty Series returns empty boolean Series."""
    ser = pd.Series([], dtype=object)
    result = ser.str.match("test")
    assert len(result) == 0
    assert result.dtype == bool


def test_empty_pattern():
    """Test empty pattern matches all strings at start (empty match)."""
    ser = pd.Series(["test", "hello", ""])
    result = ser.str.match("")
    assert list(result) == [True, True, True]


def test_na_fill_true():
    """Test custom na value (True) for object dtype."""
    ser = pd.Series([None, None, "test"])
    result = ser.str.match("test", na=True)
    assert result.iloc[0] == True
    assert result.iloc[1] == True
    assert result.iloc[2] == True


def test_na_none_default_object_dtype():
    """Test na=None returns False for object dtype."""
    ser = pd.Series([None, "hello"])
    result = ser.str.match("test", na=None)
    assert result.iloc[0] == False
    assert result.iloc[1] == False


def test_na_none_default_string_dtype():
    """Test StringDtype properly propagates NA with na=None."""
    ser = pd.Series([pd.NA, "hello"], dtype="string")
    result = ser.str.match("test", na=None)
    assert result.iloc[0] is pd.NA
    assert result.iloc[1] is pd.NA or result.iloc[1] == False


def test_special_regex_metacharacters():
    """Test regex metacharacters are interpreted as regex, not literal."""
    ser = pd.Series(["test", "t*st", "tXst"])
    result = ser.str.match("t.*t")
    assert list(result) == [True, True, True]


def test_regex_dollar_sign():
    """Test $ anchor matches end of string."""
    ser = pd.Series(["test", "testing", "best"])
    result = ser.str.match("test$")
    assert list(result) == [True, False, False]


def test_regex_pipe_alternation():
    """Test | (or) alternation in pattern."""
    ser = pd.Series(["cat", "dog", "bird"])
    result = ser.str.match("cat|dog")
    assert list(result) == [True, True, False]


def test_regex_character_class():
    """Test [] character class in pattern."""
    ser = pd.Series(["a1", "b2", "c3", "a"])
    result = ser.str.match("[abc][0-9]")
    assert list(result) == [True, True, True, False]


def test_regex_escape_special_chars():
    """Test escaped special characters match literally."""
    ser = pd.Series(["test", "t.st", "tXst"])
    result = ser.str.match(r"t\.st")
    assert list(result) == [False, True, False]


def test_flags_combined():
    """Test combining flags with case insensitive matching."""
    ser = pd.Series(["TEST", "Test", "test"])
    pattern = re.compile("^test", re.IGNORECASE)
    result = ser.str.match(pattern)
    assert list(result) == [True, True, True]


def test_case_true_explicit():
    """Test explicit case=True (case sensitive)."""
    ser = pd.Series(["Test", "TEST", "test"])
    result = ser.str.match("test", case=True)
    assert list(result) == [False, False, True]


def test_case_false_overrides_flags():
    """Test case=False overrides case in flags."""
    ser = pd.Series(["Test", "TEST"])
    result = ser.str.match("test", case=False, flags=re.IGNORECASE)
    assert list(result) == [True, True]


def test_empty_string_in_series():
    """Test empty string in series matches appropriately."""
    ser = pd.Series(["", "test", "testing"])
    result = ser.str.match("test")
    assert list(result) == [False, True, True]


def test_only_none_values():
    """Test Series with only None values."""
    ser = pd.Series([None, None], dtype=object)
    result = ser.str.match("test")
    assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])


def test_compiled_pattern_with_flags():
    """Test compiled regex with additional flags."""
    ser = pd.Series(["Test", "TEST", "test"])
    pattern = re.compile("^test", re.IGNORECASE)
    result = ser.str.match(pattern)
    assert list(result) == [True, True, True]


def test_pattern_with_groups():
    """Test pattern with capturing groups (groups don't affect match result)."""
    ser = pd.Series(["test123", "hello"])
    result = ser.str.match(r"^test(\d+)")
    assert list(result) == [True, False]


def test_index_input():
    """Test with Index input (returns array of booleans)."""
    idx = pd.Index(["test", "hello", "test123"])
    result = idx.str.match("test")
    assert hasattr(result, '__iter__')
    assert list(result) == [True, False, True]


def test_mixed_case_with_na():
    """Test NaN handling with different na fill values."""
    ser = pd.Series(["test", None, "hello"])
    result_default = ser.str.match("test")
    result_na_false = ser.str.match("test", na=False)
    result_na_true = ser.str.match("test", na=True)
    assert result_default.iloc[1] == False
    assert result_na_false.iloc[1] == False
    assert result_na_true.iloc[1] == True


def test_string_dtype_na_behavior():
    """Test StringDtype properly propagates NA."""
    ser = pd.Series(["test", None, "hello"], dtype="string")
    result = ser.str.match("test")
    assert result.iloc[0] == True
    assert result.iloc[1] is pd.NA
    assert result.iloc[2] == False
