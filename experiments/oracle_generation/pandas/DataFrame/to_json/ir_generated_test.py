import json
import tempfile

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st


@st.composite
def jsonable_dataframe(draw):
    size = draw(st.integers(min_value=0, max_value=5))
    ints = draw(st.lists(st.integers(min_value=-10, max_value=10), min_size=size, max_size=size))
    texts = draw(
        st.lists(
            st.one_of(st.none(), st.text(alphabet="abcXYZ", min_size=0, max_size=4)),
            min_size=size,
            max_size=size,
        )
    )
    return pd.DataFrame({"a": ints, "b": texts})


@st.composite
def float_dataframe(draw):
    size = draw(st.integers(min_value=1, max_value=5))
    values = draw(
        st.lists(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-1e6,
                max_value=1e6,
            ),
            min_size=size,
            max_size=size,
        )
    )
    return pd.DataFrame({"a": values})


@st.composite
def second_aligned_datetime_dataframe(draw):
    size = draw(st.integers(min_value=1, max_value=4))
    seconds = draw(
        st.lists(
            st.integers(min_value=-10_000, max_value=10_000),
            min_size=size,
            max_size=size,
        )
    )
    return pd.DataFrame({"dt": pd.to_datetime(seconds, unit="s")}), seconds


@given(jsonable_dataframe())
def test_default_orient_matches_columns_orient(df):
    assert json.loads(df.to_json()) == json.loads(df.to_json(orient="columns"))


@given(jsonable_dataframe())
def test_indent_none_matches_indent_zero(df):
    assert df.to_json(indent=None) == df.to_json(indent=0)


def test_force_ascii_controls_unicode_escaping():
    df = pd.DataFrame({"text": ["café", "東京"]})

    ascii_json = df.to_json(force_ascii=True)
    unicode_json = df.to_json(force_ascii=False)

    assert "\\u00e9" in ascii_json
    assert "café" in unicode_json
    assert "東京" in unicode_json


def test_nan_and_none_are_serialized_as_json_null():
    df = pd.DataFrame({"a": [1.0, float("nan")], "b": [None, "x"]})

    payload = json.loads(df.to_json(orient="records"))

    assert payload[0]["b"] is None
    assert payload[1]["a"] is None


@given(float_dataframe(), st.integers(min_value=16, max_value=100))
def test_double_precision_above_fifteen_raises_value_error(df, precision):
    with pytest.raises(ValueError):
        df.to_json(double_precision=precision)


@given(jsonable_dataframe(), st.sampled_from(["split", "index", "columns", "values", "table"]))
def test_lines_true_rejects_non_records_orient(df, orient):
    with pytest.raises(ValueError):
        df.to_json(orient=orient, lines=True)


@given(jsonable_dataframe(), st.sampled_from(["index", "columns"]))
def test_index_false_rejects_index_and_columns_orients(df, orient):
    with pytest.raises(ValueError):
        df.to_json(orient=orient, index=False)


def test_mode_append_requires_lines_and_records(tmp_path):
    path = tmp_path / "out.jsonl"
    df = pd.DataFrame({"a": [1], "b": ["x"]})

    with pytest.raises(ValueError):
        df.to_json(path, orient="records", mode="a")

    with pytest.raises(ValueError):
        df.to_json(path, orient="split", lines=True, mode="a")


@given(jsonable_dataframe())
def test_records_orient_is_insensitive_to_index_labels(df):
    renamed = df.copy()
    renamed.index = [f"row_{i}" for i in range(len(df))]

    assert json.loads(df.to_json(orient="records")) == json.loads(renamed.to_json(orient="records"))


@pytest.mark.filterwarnings("ignore:'epoch' date format is deprecated.*:pandas.errors.Pandas4Warning")
@given(second_aligned_datetime_dataframe())
def test_date_unit_scales_epoch_output_for_second_aligned_datetimes(df_and_seconds):
    df, seconds = df_and_seconds

    records_s = json.loads(df.to_json(orient="records", date_format="epoch", date_unit="s"))
    records_ms = json.loads(df.to_json(orient="records", date_format="epoch", date_unit="ms"))
    records_us = json.loads(df.to_json(orient="records", date_format="epoch", date_unit="us"))
    records_ns = json.loads(df.to_json(orient="records", date_format="epoch", date_unit="ns"))

    assert [row["dt"] for row in records_s] == seconds
    assert [row["dt"] for row in records_ms] == [value * 1_000 for value in seconds]
    assert [row["dt"] for row in records_us] == [value * 1_000_000 for value in seconds]
    assert [row["dt"] for row in records_ns] == [value * 1_000_000_000 for value in seconds]


def test_table_orient_defaults_to_iso_for_datetimes():
    df = pd.DataFrame({"dt": pd.to_datetime(["2021-01-01", "2021-06-15"])})

    assert df.to_json(orient="table") == df.to_json(orient="table", date_format="iso")


@given(jsonable_dataframe())
def test_append_mode_with_records_and_lines_appends_one_json_object_per_row(df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/append.jsonl"

        df.to_json(path, orient="records", lines=True, mode="w")
        df.to_json(path, orient="records", lines=True, mode="a")

        with open(path, encoding="utf-8") as handle:
            content = handle.read()

    expected_lines = 2 * len(df)
    non_empty_lines = [line for line in content.splitlines() if line.strip()]

    if expected_lines == 0:
        assert not non_empty_lines
    else:
        assert len(non_empty_lines) == expected_lines
        assert all(isinstance(json.loads(line), dict) for line in non_empty_lines)


def test_empty_dataframe_split_orient_has_empty_structural_fields():
    payload = json.loads(pd.DataFrame().to_json(orient="split"))

    assert payload == {"columns": [], "index": [], "data": []}
