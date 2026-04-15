import json

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st


# Baseline test cases derived from the function doc markdown only.


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

