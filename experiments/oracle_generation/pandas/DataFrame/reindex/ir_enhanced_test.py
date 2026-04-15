import numpy as np
import pandas as pd
from hypothesis import given, settings, assume, HealthCheck
from hypothesis.strategies import (
    integers,
    lists,
    floats,
    sampled_from,
    text,
)


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    new_index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
def test_ir_returns_dataframe(index, new_index):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(new_index)

    assert isinstance(result, pd.DataFrame)


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    new_index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_ir_result_index_equals_specified_index(index, new_index):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(index=new_index)

    assert list(result.index) == list(new_index)


@given(
    data=lists(floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=5),
    original_columns=lists(integers(), min_size=1, max_size=5, unique=True),
    new_columns=lists(integers(), min_size=1, max_size=5, unique=True),
)
@settings(max_examples=100)
def test_ir_result_columns_equals_specified_columns(
    data, original_columns, new_columns
):
    assume(len(data) == len(original_columns))
    df = pd.DataFrame([data], columns=original_columns)

    result = df.reindex(columns=new_columns)

    assert list(result.columns) == list(new_columns)


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    new_index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_ir_new_labels_filled_with_nan_by_default(index, new_index):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(new_index)

    new_labels = [lbl for lbl in result.index if lbl not in df.index]
    if len(new_labels) > 0:
        assert result.loc[new_labels].isna().all().all()


@given(
    index=lists(
        integers(min_value=0, max_value=1000), min_size=1, max_size=10, unique=True
    ),
    new_index=lists(
        integers(min_value=0, max_value=1000), min_size=1, max_size=10, unique=True
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
def test_ir_existing_labels_values_preserved(index, new_index):
    assume(len(index) >= 1)
    assume(len(set(index) & set(new_index)) > 0)
    assume(set(index) == set(new_index))
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(new_index)

    common_index = result.index.intersection(df.index)
    assert len(common_index) > 0
    assert result.loc[common_index].equals(df.loc[common_index])


@given(
    index=lists(
        integers(min_value=0, max_value=100), min_size=1, max_size=10, unique=True
    ),
    new_index=lists(
        integers(min_value=0, max_value=100), min_size=1, max_size=10, unique=True
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
def test_ir_new_labels_filled_with_fill_value(index, new_index):
    assume(len(index) >= 1)
    assume(len(set(index) & set(new_index)) > 0)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(new_index, fill_value=0)

    new_labels = [lbl for lbl in result.index if lbl not in df.index]
    if len(new_labels) > 0:
        for lbl in new_labels:
            assert result.loc[lbl, "value"] == 0


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    new_index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_ir_fill_value_string_replaces_missing(index, new_index):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(new_index, fill_value="missing")

    new_labels = [lbl for lbl in result.index if lbl not in df.index]
    if len(new_labels) > 0:
        assert (result.loc[new_labels] == "missing").all().all()


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    new_index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_ir_result_is_new_object(index, new_index):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(new_index)

    assert result is not df


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_ir_identical_index_content_unchanged(index):
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(index=index)

    assert result.equals(df)


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_ir_empty_index_returns_empty_dataframe(index):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(index=[])

    assert len(result) == 0 and list(result.columns) == list(df.columns)


@given(
    data=lists(floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=5),
    original_columns=lists(integers(), min_size=1, max_size=5, unique=True),
    new_columns=lists(integers(), min_size=1, max_size=5, unique=True),
)
@settings(max_examples=100)
def test_ir_columns_reindex_preserves_row_index(data, original_columns, new_columns):
    assume(len(data) == len(original_columns))
    df = pd.DataFrame([data], columns=original_columns)

    result = df.reindex(columns=new_columns)

    assert list(result.index) == list(df.index)


@given(
    data=lists(floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=5),
    original_columns=lists(
        integers(min_value=0, max_value=100), min_size=1, max_size=5, unique=True
    ),
    new_columns=lists(
        integers(min_value=0, max_value=100), min_size=1, max_size=5, unique=True
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
def test_ir_new_columns_not_in_original_filled_with_nan(
    data, original_columns, new_columns
):
    assume(len(data) == len(original_columns))
    assume(len(set(original_columns) & set(new_columns)) > 0)
    df = pd.DataFrame([data], columns=original_columns)

    result = df.reindex(columns=new_columns)

    new_cols = [col for col in result.columns if col not in df.columns]
    if len(new_cols) > 0:
        assert result[new_cols].isna().all().all()


@given(
    n=integers(min_value=3, max_value=6),
)
@settings(max_examples=100)
def test_ir_bfill_fills_index_gaps_from_next_valid(n):
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    data = [100.0, 101.0, np.nan, 100.0, 89.0, 88.0][:n]
    df = pd.DataFrame({"prices": data}, index=dates)

    new_dates = pd.date_range("2009-12-29", periods=n + 3, freq="D")

    result = df.reindex(new_dates, method="bfill")

    assert result.notna().sum().sum() >= df.notna().sum().sum()


@given(
    n=integers(min_value=3, max_value=6),
)
@settings(max_examples=100)
def test_ir_ffill_fills_index_gaps_from_last_valid(n):
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    data = [100.0, 101.0, np.nan, 100.0, 89.0, 88.0][:n]
    df = pd.DataFrame({"prices": data}, index=dates)

    new_dates = pd.date_range("2009-12-29", periods=n + 3, freq="D")

    result = df.reindex(new_dates, method="ffill")

    assert result.notna().sum().sum() >= df.notna().sum().sum()


@given(
    n=integers(min_value=3, max_value=4),
)
@settings(max_examples=100)
def test_ir_method_does_not_fill_original_nan_values(n):
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    data = [100.0, 101.0, np.nan, 100.0][:n]
    df = pd.DataFrame({"prices": data}, index=dates)

    new_dates = pd.date_range("2009-12-31", periods=n + 2, freq="D")

    result = df.reindex(new_dates, method="bfill")

    common_index = result.index.intersection(df.index)
    assert result.loc[common_index].isna().equals(df.loc[common_index].isna())


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    labels=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_ir_axis_style_equivalent_to_kwarg_style_for_index(index, labels):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result_axis = df.reindex(labels, axis="index")
    result_kwarg = df.reindex(index=labels)

    assert result_axis.equals(result_kwarg)


@given(
    data=lists(floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=5),
    original_columns=lists(integers(), min_size=1, max_size=5, unique=True),
    labels=lists(integers(), min_size=1, max_size=5, unique=True),
)
@settings(max_examples=100)
def test_ir_axis_style_equivalent_to_kwarg_style_for_columns(
    data, original_columns, labels
):
    assume(len(data) == len(original_columns))
    df = pd.DataFrame([data], columns=original_columns)

    result_axis = df.reindex(labels, axis="columns")
    result_kwarg = df.reindex(columns=labels)

    assert result_axis.equals(result_kwarg)


@given(
    n=integers(min_value=3, max_value=6),
    limit=integers(min_value=1, max_value=3),
)
@settings(max_examples=100)
def test_ir_limit_constrains_consecutive_fills(n, limit):
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    data = [100.0, 101.0, np.nan, 100.0, 89.0, 88.0][:n]
    df = pd.DataFrame({"prices": data}, index=dates)

    new_dates = pd.date_range("2009-12-29", periods=n + 3, freq="D")

    result = df.reindex(new_dates, method="ffill", limit=limit)
    result_no_limit = df.reindex(new_dates, method="ffill")

    assert result.notna().sum().sum() <= result_no_limit.notna().sum().sum()


@given(
    index=lists(integers(), min_size=1, max_size=8, unique=True),
)
@settings(max_examples=100)
def test_ir_superset_index_adds_nan_rows_for_new_labels(index):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    new_index = list(index) + [max(index) + 1, max(index) + 2]

    result = df.reindex(new_index)

    assert len(result) == len(new_index)
    new_labels = [lbl for lbl in new_index if lbl not in df.index]
    if len(new_labels) > 0:
        assert result.loc[new_labels].isna().all().all()


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_ir_empty_df_reindex(index):
    df = pd.DataFrame(columns=["http_status", "response_time"])

    result = df.reindex(index=index)

    assert len(result) == len(index)
    assert list(result.columns) == list(df.columns)


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    new_index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_ir_all_new_labels_get_fill_value(index, new_index):
    assume(len(index) >= 1)
    assume(len(set(index) & set(new_index)) == 0)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(new_index, fill_value=-1)

    assert (result["value"] == -1).all()


@given(
    original_index=lists(
        integers(min_value=0, max_value=100), min_size=3, max_size=8, unique=True
    ).map(sorted),
    new_index=lists(
        integers(min_value=0, max_value=100), min_size=5, max_size=12, unique=True
    ).map(sorted),
)
@settings(max_examples=100)
def test_ir_method_nearest_fills_gaps(original_index, new_index):
    data = list(range(len(original_index)))
    df = pd.DataFrame({"value": data}, index=original_index)

    result = df.reindex(new_index, method="nearest")

    for idx in new_index:
        if idx not in original_index:
            assert not pd.isna(result.loc[idx, "value"])


@given(
    original_index=lists(
        integers(min_value=0, max_value=100), min_size=3, max_size=8, unique=True
    ).map(sorted),
    new_index=lists(
        integers(min_value=0, max_value=100), min_size=5, max_size=12, unique=True
    ).map(sorted),
    tolerance=floats(
        min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100)
def test_ir_tolerance_works_with_method(original_index, new_index, tolerance):
    data = list(range(len(original_index)))
    df = pd.DataFrame({"value": data}, index=original_index)

    result = df.reindex(new_index, method="nearest", tolerance=tolerance)

    assert isinstance(result, pd.DataFrame)


@given(
    labels=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_ir_axis_zero_equals_index(labels):
    df = pd.DataFrame({"value": [1, 2, 3]}, index=[1, 2, 3])

    result = df.reindex(labels, axis=0)

    assert len(result) == len(labels)
    assert list(result.index) == list(labels)


@given(
    data=lists(floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=5),
    original_columns=lists(integers(), min_size=1, max_size=5, unique=True),
    labels=lists(integers(), min_size=1, max_size=5, unique=True),
)
@settings(max_examples=100)
def test_ir_axis_one_equals_columns(data, original_columns, labels):
    assume(len(data) == len(original_columns))
    df = pd.DataFrame([data], columns=original_columns)

    result = df.reindex(labels, axis=1)

    assert len(result.columns) == len(labels)
    assert list(result.columns) == list(labels)


@given(
    n=integers(min_value=4, max_value=6),
)
@settings(max_examples=100)
def test_ir_original_nan_not_filled_by_method(n):
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    data = [100.0, np.nan, np.nan, 100.0, 89.0, 88.0][:n]
    df = pd.DataFrame({"prices": data}, index=dates)

    new_dates = pd.date_range("2009-12-31", periods=n + 2, freq="D")

    result = df.reindex(new_dates, method="ffill")

    assert pd.isna(result.loc[dates[1], "prices"]) if len(dates) > 1 else True
    assert pd.isna(result.loc[dates[2], "prices"]) if len(dates) > 2 else True


@given(
    index=lists(integers(), min_size=2, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_ir_reindex_preserves_dtypes(index):
    assume(len(index) >= 2)
    df = pd.DataFrame({"int_col": [1, 2], "float_col": [1.0, 2.0]}, index=index[:2])

    result = df.reindex(index[:2])

    assert result["int_col"].dtype == np.int64 or result["int_col"].dtype == np.int32
    assert result["float_col"].dtype == np.float64
