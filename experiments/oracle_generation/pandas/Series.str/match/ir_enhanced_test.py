import hypothesis
import pandas as pd
import pytest
import re
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ==================== BASELINE TESTS ====================


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


# ==================== NEW TESTS (Inspired by IR) ====================
# These test high-stakes edge cases beyond the baseline happy path tests


@given(data=st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=50))
@settings(max_examples=100)
def test_empty_string_handling(data):
    """[NEW - IR] Test that empty strings are handled correctly."""
    ser = pd.Series(data)
    result = ser.str.match("")
    for i in range(len(ser)):
        expected = bool(re.match("", ser.iloc[i]))
        assert result.iloc[i] == expected


def test_all_nan_series():
    """[NEW - IR] Test behavior when all values are NaN."""
    ser = pd.Series([None, None, None], dtype="object")
    result = ser.str.match("test")
    assert all(pd.isna(result))


def test_all_empty_strings():
    """[NEW - IR] Test behavior when all strings are empty."""
    ser = pd.Series(["", "", ""])
    result = ser.str.match("test")
    assert all(not x for x in result)


def test_regex_matches_nothing():
    """[NEW - IR] Test regex pattern that matches nothing returns all False."""
    ser = pd.Series(["hello", "world", "test"])
    result = ser.str.match("^z$")
    assert all(not x for x in result)


def test_regex_matches_everything():
    """[NEW - IR] Test regex pattern that matches everything at start."""
    ser = pd.Series(["hello", "world", "test"])
    result = ser.str.match(".*")
    assert all(x for x in result)


def test_special_regex_characters():
    """[NEW - IR] Test strings containing special regex characters are escaped properly."""
    ser = pd.Series(["[test]", "(test)", "test.", "test*", "test+", "test?"])
    result = ser.str.match("\\[")
    assert result.iloc[0] == True
    assert all(not x for x in result.iloc[1:])


def test_unicode_characters():
    """[NEW - IR] Test Unicode characters in strings."""
    ser = pd.Series(["hello", "world", "世界", "🎉test"])
    result = ser.str.match("h")
    assert result.iloc[0] == True
    assert all(not x for x in result.iloc[1:])


@given(
    data=st.lists(st.text(min_size=0), min_size=1, max_size=20),
    pat=st.sampled_from(["^[a-z]", "^[A-Z]", "^[0-9]", "^[a-zA-Z]", "^\\w", "^\\d"]),
)
@settings(max_examples=50)
def test_common_regex_patterns(data, pat):
    """[NEW - IR] Test common regex patterns from IR."""
    ser = pd.Series(data)
    result = ser.str.match(pat)
    for i in range(len(ser)):
        expected = bool(re.match(pat, ser.iloc[i]))
        assert result.iloc[i] == expected


def test_na_fill_custom_value_true():
    """[NEW - IR] Test custom na value (True) for object dtype."""
    ser = pd.Series([None, "hello", None], dtype="object")
    result = ser.str.match("test", na=True)
    assert result.iloc[0] == True
    assert result.iloc[1] == False
    assert result.iloc[2] == True


@given(data=st.lists(st.text(min_size=0), min_size=1, max_size=20))
@settings(max_examples=50)
def test_multidigit_numbers_in_pattern(data):
    """[NEW - IR] Test pattern with multi-digit numbers."""
    ser = pd.Series(data)
    result = ser.str.match("123")
    for i in range(len(ser)):
        expected = bool(re.match("123", ser.iloc[i]))
        assert result.iloc[i] == expected


def test_anchor_at_start_only():
    """[NEW - IR] Verify match is anchored at start (not contains)."""
    ser = pd.Series(["abc", "xabc", "abcx", "ab"])
    result = ser.str.match("abc")
    assert result.iloc[0] == True
    assert result.iloc[1] == False
    assert result.iloc[2] == True
    assert result.iloc[3] == False


def test_case_and_flags_conflict():
    """[NEW - IR] Test case=True with flags=re.IGNORECASE raises ValueError."""
    ser = pd.Series(["Test", "test", "TEST"])
    with pytest.raises(ValueError, match="Cannot both specify"):
        ser.str.match("test", case=True, flags=re.IGNORECASE)


@given(data=st.lists(st.text(min_size=0), min_size=1, max_size=20))
@settings(max_examples=50)
def test_word_boundary_patterns(data):
    """[NEW - IR] Test word boundary regex patterns."""
    ser = pd.Series(data)
    result = ser.str.match("\\btest")
    for i in range(len(ser)):
        expected = bool(re.match("\\btest", ser.iloc[i]))
        assert result.iloc[i] == expected


def test_whitespace_patterns():
    """[NEW - IR] Test whitespace regex patterns."""
    ser = pd.Series([" test", "test ", " test ", "test"])
    result = ser.str.match("\\s")
    assert result.iloc[0] == True
    assert result.iloc[1] == False
    assert result.iloc[2] == True
    assert result.iloc[3] == False


def test_digit_patterns():
    """[NEW - IR] Test digit matching patterns."""
    ser = pd.Series(["1abc", "abc1", "1", "abc"])
    result = ser.str.match("\\d")
    assert result.iloc[0] == True
    assert result.iloc[1] == False
    assert result.iloc[2] == True
    assert result.iloc[3] == False
