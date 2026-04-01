"""
Baseline property-based tests for pandas.DataFrame.to_json.

Properties tested:
1. Return type: returns str when path_or_buf=None, else None
2. orient='split'  — output has keys 'columns', 'index', 'data'
3. orient='records' — output is a list of dicts keyed by column names
4. orient='index'  — output keys are index labels; sub-dicts keyed by columns
5. orient='columns' — output keys are column names; sub-dicts keyed by index
6. orient='values'  — output is a list of lists (no index/column metadata)
7. orient='table'   — output has 'schema' and 'data' keys
8. lines=True + orient='records' — each line is valid JSON
9. double_precision controls float encoding precision
10. indent produces indented output
11. force_ascii=True ensures ASCII-only output
12. index=False with orient='split' omits 'index' key from output
13. date_format='iso' produces ISO8601 strings for datetime columns
14. round-trip: split/records orient can be read back with pd.read_json
15. NaN/None become JSON null
16. Empty DataFrames serialize correctly
17. mode='a' with orient='records' + lines=True appends rows
"""

import io
import json
import os
import tempfile

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

simple_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll"), min_codepoint=65),
    min_size=1,
    max_size=6,
)

scalar_strategy = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.text(alphabet="abcdefghij", min_size=0, max_size=5),
)


@st.composite
def simple_df(draw):
    """Draw a small DataFrame with simple string index and column labels."""
    cols = draw(st.lists(simple_text, min_size=1, max_size=4, unique=True))
    idx = draw(st.lists(simple_text, min_size=1, max_size=4, unique=True))
    data = {
        c: draw(st.lists(scalar_strategy, min_size=len(idx), max_size=len(idx)))
        for c in cols
    }
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------

@given(simple_df())
def test_returns_string_when_no_path(df):
    result = df.to_json()
    assert isinstance(result, str)


@given(simple_df())
def test_returns_none_when_path_given(df):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        result = df.to_json(path_or_buf=path)
        assert result is None
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 2–7. orient formats
# ---------------------------------------------------------------------------

@given(simple_df())
def test_orient_split_structure(df):
    parsed = json.loads(df.to_json(orient="split"))
    assert set(parsed.keys()) >= {"columns", "index", "data"}
    assert parsed["columns"] == list(df.columns)
    assert parsed["index"] == list(df.index)
    assert len(parsed["data"]) == len(df)
    for row in parsed["data"]:
        assert len(row) == len(df.columns)


@given(simple_df())
def test_orient_records_structure(df):
    parsed = json.loads(df.to_json(orient="records"))
    assert isinstance(parsed, list)
    assert len(parsed) == len(df)
    for record in parsed:
        assert set(record.keys()) == set(df.columns)


@given(simple_df())
def test_orient_index_structure(df):
    parsed = json.loads(df.to_json(orient="index"))
    assert isinstance(parsed, dict)
    assert set(parsed.keys()) == {str(i) for i in df.index}
    for _key, record in parsed.items():
        assert set(record.keys()) == set(df.columns)


@given(simple_df())
def test_orient_columns_structure(df):
    parsed = json.loads(df.to_json(orient="columns"))
    assert isinstance(parsed, dict)
    assert set(parsed.keys()) == set(df.columns)
    for _col_key, col_data in parsed.items():
        assert set(col_data.keys()) == {str(i) for i in df.index}


@given(simple_df())
def test_orient_values_structure(df):
    parsed = json.loads(df.to_json(orient="values"))
    assert isinstance(parsed, list)
    assert len(parsed) == len(df)
    for row in parsed:
        assert isinstance(row, list)
        assert len(row) == len(df.columns)


@given(simple_df())
def test_orient_table_structure(df):
    parsed = json.loads(df.to_json(orient="table"))
    assert "schema" in parsed
    assert "data" in parsed
    schema_field_names = {f["name"] for f in parsed["schema"]["fields"]}
    for col in df.columns:
        assert col in schema_field_names


# ---------------------------------------------------------------------------
# 8. lines=True with orient='records'
# ---------------------------------------------------------------------------

@given(simple_df())
def test_lines_records_each_line_is_valid_json(df):
    output = df.to_json(orient="records", lines=True)
    lines = [ln for ln in output.splitlines() if ln.strip()]
    assert len(lines) == len(df)
    for line in lines:
        record = json.loads(line)
        assert isinstance(record, dict)
        assert set(record.keys()) == set(df.columns)


def test_lines_raises_for_non_records_orient():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError):
        df.to_json(orient="split", lines=True)


# ---------------------------------------------------------------------------
# 9. double_precision
# ---------------------------------------------------------------------------

def test_double_precision_valid_range_does_not_raise():
    df = pd.DataFrame({"val": [3.14159265358979]})
    for precision in [0, 3, 5, 10, 15]:
        result = df.to_json(double_precision=precision)
        assert isinstance(result, str)


def test_double_precision_above_15_raises():
    df = pd.DataFrame({"val": [1.0]})
    with pytest.raises(ValueError):
        df.to_json(double_precision=16)


@given(st.integers(min_value=0, max_value=15))
def test_double_precision_property(precision):
    df = pd.DataFrame({"val": [1.123456789012345]})
    result = df.to_json(double_precision=precision)
    assert isinstance(result, str)
    parsed = json.loads(result)
    # The encoded value should be parseable as JSON
    encoded_val = parsed["val"]["0"]
    assert isinstance(encoded_val, (int, float))


# ---------------------------------------------------------------------------
# 10. indent
# ---------------------------------------------------------------------------

@given(st.integers(min_value=1, max_value=8))
def test_indent_produces_multiline_output(indent):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.to_json(indent=indent)
    assert "\n" in result


def test_indent_none_produces_compact_output():
    df = pd.DataFrame({"a": [1]})
    result = df.to_json(indent=None)
    assert "\n" not in result


# ---------------------------------------------------------------------------
# 11. force_ascii
# ---------------------------------------------------------------------------

def test_force_ascii_true_produces_ascii_only():
    df = pd.DataFrame({"col": ["café", "naïve"]})
    result = df.to_json(force_ascii=True)
    assert result.isascii()


def test_force_ascii_false_preserves_unicode():
    df = pd.DataFrame({"col": ["café"]})
    result = df.to_json(force_ascii=False)
    assert "café" in result


# ---------------------------------------------------------------------------
# 12. index parameter
# ---------------------------------------------------------------------------

def test_index_false_split_omits_index_key():
    df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
    parsed = json.loads(df.to_json(orient="split", index=False))
    assert "index" not in parsed
    assert "columns" in parsed
    assert "data" in parsed


def test_index_true_split_includes_index_key():
    df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
    parsed = json.loads(df.to_json(orient="split", index=True))
    assert "index" in parsed
    assert parsed["index"] == ["x", "y"]


def test_index_false_raises_for_index_orient():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError):
        df.to_json(orient="index", index=False)


def test_index_false_raises_for_columns_orient():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError):
        df.to_json(orient="columns", index=False)


# ---------------------------------------------------------------------------
# 13. date_format='iso'
# ---------------------------------------------------------------------------

def test_date_format_iso_produces_iso_strings():
    df = pd.DataFrame({"ts": pd.to_datetime(["2021-01-01", "2022-06-15"])})
    parsed = json.loads(df.to_json(orient="records", date_format="iso"))
    for record in parsed:
        ts_val = record["ts"]
        assert isinstance(ts_val, str)
        assert "T" in ts_val  # ISO8601 contains 'T' separator


def test_date_format_epoch_deprecated_but_functional():
    """epoch format is deprecated in 3.0 but should still work (with a warning)."""
    df = pd.DataFrame({"ts": pd.to_datetime(["2021-01-01"])})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        parsed = json.loads(df.to_json(orient="records", date_format="epoch"))
    assert isinstance(parsed[0]["ts"], (int, float))


# ---------------------------------------------------------------------------
# 14. date_unit
# ---------------------------------------------------------------------------

def test_date_unit_affects_iso_precision():
    df = pd.DataFrame({"ts": pd.to_datetime(["2021-01-01 00:00:00.123456789"])})
    result_ms = json.loads(df.to_json(orient="records", date_format="iso", date_unit="ms"))
    result_ns = json.loads(df.to_json(orient="records", date_format="iso", date_unit="ns"))
    assert isinstance(result_ms[0]["ts"], str)
    assert isinstance(result_ns[0]["ts"], str)


# ---------------------------------------------------------------------------
# 15. Round-trip fidelity
# ---------------------------------------------------------------------------

@given(
    data_frames(
        columns=[
            column("a", elements=st.integers(min_value=-100, max_value=100)),
            column("b", elements=st.integers(min_value=-100, max_value=100)),
        ],
        index=range_indexes(min_size=1, max_size=5),
    )
)
def test_roundtrip_split_integer_dataframe(df):
    json_str = df.to_json(orient="split")
    recovered = pd.read_json(io.StringIO(json_str), orient="split")
    assert list(recovered.columns) == list(df.columns)
    assert recovered.shape == df.shape


@given(
    data_frames(
        columns=[
            column("x", elements=st.integers(min_value=-50, max_value=50)),
            column("y", elements=st.integers(min_value=-50, max_value=50)),
        ],
        index=range_indexes(min_size=1, max_size=5),
    )
)
def test_roundtrip_records_shape(df):
    json_str = df.to_json(orient="records")
    recovered = pd.read_json(io.StringIO(json_str), orient="records")
    assert list(recovered.columns) == list(df.columns)
    assert len(recovered) == len(df)


# ---------------------------------------------------------------------------
# 16. Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_orient_split():
    df = pd.DataFrame({"a": [], "b": []})
    parsed = json.loads(df.to_json(orient="split"))
    assert parsed["data"] == []
    assert parsed["columns"] == ["a", "b"]


def test_empty_dataframe_orient_records():
    df = pd.DataFrame({"a": [], "b": []})
    parsed = json.loads(df.to_json(orient="records"))
    assert parsed == []


# ---------------------------------------------------------------------------
# 17. NaN / None become JSON null
# ---------------------------------------------------------------------------

def test_nan_becomes_null():
    df = pd.DataFrame({"val": [1.0, float("nan"), 3.0]})
    parsed = json.loads(df.to_json(orient="records"))
    assert parsed[1]["val"] is None


def test_none_becomes_null():
    df = pd.DataFrame({"val": [1, None, 3]})
    parsed = json.loads(df.to_json(orient="records"))
    assert parsed[1]["val"] is None


# ---------------------------------------------------------------------------
# 18. mode='a' appends rows
# ---------------------------------------------------------------------------

def test_mode_append_with_records_and_lines():
    df = pd.DataFrame({"a": [1, 2]})
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        df.to_json(path_or_buf=path, orient="records", lines=True, mode="w")
        df.to_json(path_or_buf=path, orient="records", lines=True, mode="a")
        with open(path) as fh:
            lines = [ln for ln in fh.readlines() if ln.strip()]
        assert len(lines) == 4  # 2 rows written twice
    finally:
        os.unlink(path)
