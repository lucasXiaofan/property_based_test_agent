import json
import os
import tempfile

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


# =============================================================================
# NEW TESTS - Added based on documentation analysis
# =============================================================================


def test_index_orient_returns_dict_indexed_by_index():
    # Documentation: 'index' : dict like {index -> {column -> value}}
    df = pd.DataFrame(
        [["a", "b"], ["c", "d"]],
        index=["row 1", "row 2"],
        columns=["col 1", "col 2"],
    )

    result = json.loads(df.to_json(orient="index"))

    assert "row 1" in result
    assert "row 2" in result
    assert result["row 1"] == {"col 1": "a", "col 2": "b"}
    assert result["row 2"] == {"col 1": "c", "col 2": "d"}


def test_values_orient_returns_just_values_array():
    # Documentation: 'values' : just the values array
    df = pd.DataFrame(
        [["a", "b"], ["c", "d"]],
        columns=["col 1", "col 2"],
    )

    result = json.loads(df.to_json(orient="values"))

    assert isinstance(result, list)
    assert result == [["a", "b"], ["c", "d"]]


def test_invalid_orient_raises_value_error():
    df = pd.DataFrame({"a": [1, 2]})

    with pytest.raises(ValueError):
        df.to_json(orient="invalid_orient")


def test_date_format_epoch_converts_to_epoch_milliseconds():
    # Documentation: 'epoch' = epoch milliseconds
    df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01", "2020-01-02"])})

    result = df.to_json(orient="records", date_format="epoch")

    parsed = json.loads(result)
    assert isinstance(parsed[0]["dt"], (int, float))


def test_date_format_iso_converts_to_iso8601():
    # Documentation: 'iso' = ISO8601
    df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01", "2020-01-02"])})

    result = df.to_json(orient="records", date_format="iso")

    parsed = json.loads(result)
    assert "2020-01-01" in parsed[0]["dt"]


def test_table_orient_default_date_format_is_iso():
    # Documentation: For orient='table', the default is 'iso'
    df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01"])})

    result = df.to_json(orient="table")
    parsed = json.loads(result)

    assert "2020-01-01" in str(parsed["data"][0].get("dt", ""))


def test_force_ascii_false_preserves_non_ascii_characters():
    # Documentation: force_asciibool, default True - Force encoded string to be ASCII
    df = pd.DataFrame({"text": ["hello", "こんにちは", "café"]})

    result = df.to_json(force_ascii=False)

    assert "こんにちは" in result
    assert "café" in result


def test_force_ascii_true_encodes_to_ascii():
    df = pd.DataFrame({"text": ["hello", "こんにちは"]})

    result = df.to_json(force_ascii=True)

    assert "こんにちは" not in result
    assert r"\u" in result


def test_date_unit_seconds():
    # Documentation: One of 's', 'ms', 'us', 'ns' for second, millisecond, microsecond, and nanosecond
    df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01T00:00:00", "2020-01-02T00:00:00"])})

    result = df.to_json(date_unit="s")
    parsed = json.loads(result)

    dt_val = parsed["dt"]["0"]
    assert isinstance(dt_val, int)
    assert dt_val == 1577836800


def test_date_unit_nanoseconds():
    df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01T00:00:00"])})

    result = df.to_json(date_unit="ns")
    parsed = json.loads(result)

    dt_val = parsed["dt"]["0"]
    assert isinstance(dt_val, int)
    assert dt_val == 1577836800000000000


def test_index_parameter_false_excludes_index_from_output():
    # Documentation: indexbool or None, default None - The index is only used when orient is 'split', 'index', 'column', or 'table'
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["x", "y"])

    result = df.to_json(orient="split", index=False)
    parsed = json.loads(result)

    assert "index" not in parsed


def test_index_parameter_true_includes_index_in_output():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["x", "y"])

    result = df.to_json(orient="split", index=True)
    parsed = json.loads(result)

    assert parsed["index"] == ["x", "y"]


def test_index_column_name_with_index_name_raises_for_table_orient():
    # Documentation: The string 'index' as a column name with empty Index or if it is 'index' will raise a ValueError
    # This raises when there's overlapping names between the index and columns for table orient
    df = pd.DataFrame({"index": [1, 2]}, index=[1, 2])
    df.index.name = "index"

    with pytest.raises(ValueError, match="Overlapping names"):
        df.to_json(orient="table")


def test_default_handler_for_non_serializable_object():
    # Documentation: Handler to call if object cannot otherwise be converted to a suitable format for JSON
    class CustomObject:
        def __init__(self, value):
            self.value = value

    obj = CustomObject(42)
    df = pd.DataFrame({"a": [obj]})

    def handler(x):
        return {"custom": x.value}

    result = df.to_json(default_handler=handler)
    parsed = json.loads(result)

    assert parsed["a"]["0"]["custom"] == 42


def test_write_to_file_path():
    # Documentation: If None, the result is returned as a string. Otherwise returns None.
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        result = df.to_json(path_or_buf=temp_path)
        assert result is None

        with open(temp_path, "r") as f:
            content = f.read()
        parsed = json.loads(content)
        assert parsed == {"a": {"0": 1, "1": 2}, "b": {"0": 3, "1": 4}}
    finally:
        os.unlink(temp_path)


def test_compression_gzip():
    # Documentation: compressionstr or dict, default 'infer'
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
        temp_path = f.name

    try:
        df.to_json(path_or_buf=temp_path, compression="gzip")
        assert os.path.getsize(temp_path) > 0
    finally:
        os.unlink(temp_path)


def test_compression_none():
    df = pd.DataFrame({"a": [1, 2]})

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        df.to_json(path_or_buf=temp_path, compression=None)

        with open(temp_path, "r") as f:
            content = f.read()
        assert "a" in content
    finally:
        os.unlink(temp_path)


def test_mode_append_with_records_and_lines():
    # Documentation: mode='a' is only supported when lines is True and orient is 'records'
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [3, 4]})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        df1.to_json(path_or_buf=temp_path, orient="records", lines=True, mode="w")
        df2.to_json(path_or_buf=temp_path, orient="records", lines=True, mode="a")

        with open(temp_path, "r") as f:
            content = f.read()

        lines = content.strip().splitlines()
        assert len(lines) == 4
    finally:
        os.unlink(temp_path)


def test_mode_append_without_lines_raises_error():
    df = pd.DataFrame({"a": [1, 2]})

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        with pytest.raises(ValueError):
            df.to_json(path_or_buf=temp_path, mode="a")
    finally:
        os.unlink(temp_path)


def test_double_precision_max_value_15():
    # Documentation: The possible maximal value is 15
    df = pd.DataFrame({"a": [1.12345678901234567890]})

    result = df.to_json(double_precision=15)

    assert json.loads(result)["a"][0] == 1.123456789012345


def test_double_precision_10_is_default():
    df = pd.DataFrame({"a": [1.123456789]})

    result_default = df.to_json()
    result_explicit = df.to_json(double_precision=10)

    assert result_default == result_explicit


def test_indent_with_positive_value():
    df = pd.DataFrame({"a": [1, 2]})

    result = df.to_json(indent=4)

    assert "    " in result


def test_nan_handling_with_force_ascii():
    df = pd.DataFrame({"a": [float("nan"), float("inf")]})

    result = df.to_json()

    assert "null" in result


def test_table_orient_schema_contains_fields():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    result = json.loads(df.to_json(orient="table"))

    fields = result["schema"]["fields"]
    field_names = [f["name"] for f in fields]
    assert "index" in field_names
    assert "a" in field_names
    assert "b" in field_names


def test_table_orient_schema_contains_primary_key():
    df = pd.DataFrame({"a": [1, 2]}, index=["row1", "row2"])

    result = json.loads(df.to_json(orient="table"))

    assert "primaryKey" in result["schema"]
    assert "index" in result["schema"]["primaryKey"]


def test_split_orient_with_custom_index():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["x", "y"])

    result = json.loads(df.to_json(orient="split"))

    assert result["index"] == ["x", "y"]


def test_split_orient_with_index_false():
    # Documentation: index=False is only valid when 'orient' is 'split', 'table', 'records', or 'values'
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["x", "y"])

    result = df.to_json(orient="split", index=False)
    parsed = json.loads(result)

    assert "index" not in parsed
    assert parsed["columns"] == ["a", "b"]
    assert parsed["data"] == [[1, 2], [3, 4]]


def test_empty_dataframe():
    df = pd.DataFrame()

    result = df.to_json()

    assert result == "{}"


def test_single_value_dataframe():
    df = pd.DataFrame({"a": [42]})

    result = json.loads(df.to_json())

    assert result["a"][0] == 42


def test_boolean_values():
    df = pd.DataFrame({"a": [True, False, True]})

    result = df.to_json()

    assert "true" in result
    assert "false" in result


def test_orient_values_with_numeric_index():
    df = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])

    result = json.loads(df.to_json(orient="values"))

    assert result == [[1, 2], [3, 4]]
