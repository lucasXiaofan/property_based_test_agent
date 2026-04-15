import numpy as np
import pandas as pd
from hypothesis import given, settings, assume, HealthCheck
from hypothesis.strategies import integers, lists, floats, sampled_from, none, tuples


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    new_index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
def test_reindex_preserves_existing_values(index, new_index):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)
    common_index = list(set(index) & set(new_index))
    assume(len(common_index) > 0)

    result = df.reindex(new_index)

    for idx in common_index:
        assert result.loc[idx, "value"] == df.loc[idx, "value"]


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    new_index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_reindex_new_positions_have_nan(index, new_index):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(new_index)

    new_positions = set(new_index) - set(index)
    for idx in new_positions:
        assert pd.isna(result.loc[idx, "value"])


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    new_index=lists(integers(), min_size=1, max_size=10, unique=True),
    fill_value=floats(allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_reindex_fill_value(index, new_index, fill_value):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(new_index, fill_value=fill_value)

    new_positions = set(new_index) - set(index)
    for idx in new_positions:
        assert result.loc[idx, "value"] == fill_value


@given(
    original_index=lists(
        integers(min_value=0, max_value=1000), min_size=3, max_size=10, unique=True
    ).map(sorted),
    new_index=lists(
        integers(min_value=0, max_value=1000), min_size=5, max_size=15, unique=True
    ).map(sorted),
)
@settings(max_examples=100)
def test_reindex_method_ffill(original_index, new_index):
    assume(new_index[0] < original_index[0])
    data = list(range(len(original_index)))
    df = pd.DataFrame({"value": data}, index=original_index)

    result = df.reindex(new_index, method="ffill")

    for idx in new_index:
        if idx < original_index[0]:
            assert pd.isna(result.loc[idx, "value"])


@given(
    original_index=lists(
        integers(min_value=0, max_value=1000), min_size=3, max_size=10, unique=True
    ).map(sorted),
    new_index=lists(
        integers(min_value=0, max_value=1000), min_size=5, max_size=15, unique=True
    ).map(sorted),
)
@settings(max_examples=100)
def test_reindex_method_bfill(original_index, new_index):
    assume(new_index[-1] > original_index[-1])
    data = list(range(len(original_index)))
    df = pd.DataFrame({"value": data}, index=original_index)

    result = df.reindex(new_index, method="bfill")

    for idx in new_index:
        if idx > original_index[-1]:
            assert pd.isna(result.loc[idx, "value"])


@given(
    data=lists(floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10),
    original_columns=lists(integers(), min_size=1, max_size=10, unique=True),
    new_columns=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_reindex_columns_preserves_existing(data, original_columns, new_columns):
    assume(len(data) == len(original_columns))
    df = pd.DataFrame([data], columns=original_columns)
    common_columns = list(set(original_columns) & set(new_columns))
    assume(len(common_columns) > 0)

    result = df.reindex(columns=new_columns)

    for col in common_columns:
        assert result[col].iloc[0] == df[col].iloc[0]


@given(
    data=lists(floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10),
    original_columns=lists(integers(), min_size=1, max_size=10, unique=True),
    new_columns=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_reindex_new_columns_have_nan(data, original_columns, new_columns):
    assume(len(data) == len(original_columns))
    df = pd.DataFrame([data], columns=original_columns)

    result = df.reindex(columns=new_columns)

    new_cols = set(new_columns) - set(original_columns)
    for col in new_cols:
        assert pd.isna(result[col].iloc[0])


@given(
    index=lists(
        integers(min_value=0, max_value=100), min_size=2, max_size=10, unique=True
    ).map(sorted),
    new_index=lists(
        integers(min_value=0, max_value=100), min_size=3, max_size=15, unique=True
    ).map(sorted),
    limit=integers(min_value=1, max_value=5),
)
@settings(max_examples=100)
def test_reindex_limit(index, new_index, limit):
    assume(len(index) >= 2)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(new_index, method="ffill", limit=limit)

    new_positions = sorted(set(new_index) - set(index))
    for i, idx in enumerate(new_positions):
        if i < limit and idx > index[0]:
            assert not pd.isna(result.loc[idx, "value"])


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    new_index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_reindex_preserves_shape_with_same_length(index, new_index):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(new_index)

    assert len(result) == len(new_index)


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    new_index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_reindex_returns_dataframe(index, new_index):
    assume(len(index) >= 1)
    data = list(range(len(index)))
    df = pd.DataFrame({"value": data}, index=index)

    result = df.reindex(new_index)

    assert isinstance(result, pd.DataFrame)


@given(
    n=integers(min_value=1, max_value=10),
    columns=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100)
def test_reindex_same_index_returns_equivalent(n, columns):
    assume(len(columns) >= 1)
    index = list(range(n))
    data = list(range(len(columns)))
    df = pd.DataFrame([data], index=index, columns=columns)

    result = df.reindex(index=index, columns=columns)

    assert result.equals(df)
