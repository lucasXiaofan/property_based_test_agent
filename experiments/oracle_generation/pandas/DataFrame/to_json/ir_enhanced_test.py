import json

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st


# Baseline test cases copied from baseline_test.py.


@given(
    values=st.lists(
        st.integers(min_value=-1000, max_value=1000),
        min_size=1,
        max_size=6,
    )
)
def test_to_json_returns_a_json_string_when_no_path_is_provided(values):
    df = pd.DataFrame({"value": values})

    result = df.to_json()

    assert isinstance(result, str)
    assert json.loads(result) == json.loads(df.to_json(orient="columns"))


@given(
    rows=st.lists(
        st.tuples(
            st.text(min_size=0, max_size=5),
            st.text(min_size=0, max_size=5),
        ),
        min_size=1,
        max_size=5,
    )
)
def test_split_orient_contains_index_columns_and_data(rows):
    df = pd.DataFrame(rows, columns=["left", "right"])

    result = json.loads(df.to_json(orient="split"))

    assert set(result) == {"index", "columns", "data"}
    assert result["columns"] == ["left", "right"]
    assert result["index"] == list(range(len(df)))
    assert len(result["data"]) == len(df)
    assert all(len(row) == len(df.columns) for row in result["data"])


@given(
    rows=st.lists(
        st.tuples(
            st.integers(min_value=-20, max_value=20),
            st.text(min_size=0, max_size=5),
        ),
        min_size=1,
        max_size=5,
    )
)
def test_records_orient_returns_list_of_row_mappings(rows):
    df = pd.DataFrame(rows, columns=["number", "text"])

    result = json.loads(df.to_json(orient="records"))

    assert isinstance(result, list)
    assert len(result) == len(df)
    assert all(set(record) == {"number", "text"} for record in result)


def test_table_orient_contains_schema_data_and_pandas_version():
    df = pd.DataFrame(
        [["a", "b"], ["c", "d"]],
        index=["row 1", "row 2"],
        columns=["col 1", "col 2"],
    )

    result = json.loads(df.to_json(orient="table"))

    assert set(result).issuperset({"schema", "data"})
    assert "pandas_version" in result["schema"]
    assert isinstance(result["data"], list)
    assert all(isinstance(row, dict) for row in result["data"])


def test_nan_and_none_are_serialized_as_null():
    df = pd.DataFrame({"a": [1.0, float("nan"), 3.0], "b": [None, "x", "y"]})

    result = df.to_json()

    assert "null" in result


def test_lines_requires_records_orient():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    with pytest.raises(ValueError):
        df.to_json(orient="split", lines=True)


def test_records_lines_output_is_line_delimited_json():
    df = pd.DataFrame([["a", "b"], ["c", "d"]], columns=["col 1", "col 2"])

    result = df.to_json(orient="records", lines=True)

    lines = result.strip().splitlines()
    assert len(lines) == len(df)
    assert all(isinstance(json.loads(line), dict) for line in lines)


@given(
    value=st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=-1e6,
        max_value=1e6,
    ),
    precision=st.integers(min_value=16, max_value=30),
)
def test_double_precision_above_15_raises_value_error(value, precision):
    df = pd.DataFrame({"a": [value]})

    with pytest.raises(ValueError):
        df.to_json(double_precision=precision)


def test_indent_none_and_zero_are_equivalent():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    assert df.to_json(indent=None) == df.to_json(indent=0)


# New test cases inspired by ir_v2.json.


def test_index_false_is_rejected_for_index_orient():
    df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])

    with pytest.raises(ValueError):
        df.to_json(orient="index", index=False)


def test_index_false_is_rejected_for_columns_orient():
    df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])

    with pytest.raises(ValueError):
        df.to_json(orient="columns", index=False)


def test_table_orient_uses_iso_dates_by_default():
    df = pd.DataFrame({"dt": pd.to_datetime(["2021-01-01", "2021-06-15"])})

    default_result = df.to_json(orient="table")
    explicit_iso_result = df.to_json(orient="table", date_format="iso")

    assert default_result == explicit_iso_result


@given(date_unit=st.sampled_from(["s", "ms", "us", "ns"]))
def test_iso_date_precision_tracks_date_unit(date_unit):
    df = pd.DataFrame({"dt": pd.to_datetime(["2021-01-01 00:00:00.123456789"])})

    result = json.loads(df.to_json(orient="table", date_format="iso", date_unit=date_unit))
    rendered = result["data"][0]["dt"]

    expected_fractional_lengths = {"s": 0, "ms": 3, "us": 6, "ns": 9}
    if expected_fractional_lengths[date_unit] == 0:
        assert "." not in rendered
    else:
        fraction = rendered.split(".", 1)[1].rstrip("Z")
        assert len(fraction) == expected_fractional_lengths[date_unit]


def test_mode_append_requires_lines_true_even_for_records(tmp_path):
    path = tmp_path / "records.jsonl"
    df = pd.DataFrame({"a": [1], "b": [2]})

    with pytest.raises(ValueError):
        df.to_json(path, orient="records", mode="a", lines=False)


def test_mode_append_requires_records_orient_when_lines_true(tmp_path):
    path = tmp_path / "not-records.jsonl"
    df = pd.DataFrame({"a": [1], "b": [2]})

    with pytest.raises(ValueError):
        df.to_json(path, orient="split", mode="a", lines=True)


def test_force_ascii_false_preserves_unicode_characters():
    df = pd.DataFrame({"greeting": ["你好", "cafe"]})

    result = df.to_json(force_ascii=False)

    assert "你好" in result


def test_force_ascii_true_escapes_unicode_characters():
    df = pd.DataFrame({"greeting": ["你好", "cafe"]})

    result = df.to_json(force_ascii=True)

    assert "你好" not in result
    assert "\\u4f60\\u597d" in result
