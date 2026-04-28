import numpy as np
import pandas as pd
from hypothesis import given, settings, assume, HealthCheck
from hypothesis.strategies import integers, lists, floats, sampled_from, none, tuples


@given(
    index=lists(integers(), min_size=1, max_size=10, unique=True),
    new_index=lists(integers(), min_size=1, max_size=10, unique=True),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
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


# NEW TESTS BELOW - Added to cover documentation-specified behavior


@given(
    original_index=lists(
        integers(min_value=0, max_value=100), min_size=3, max_size=10, unique=True
    ).map(sorted),
    new_index=lists(
        integers(min_value=0, max_value=100), min_size=5, max_size=15, unique=True
    ).map(sorted),
)
@settings(max_examples=100)
def test_reindex_method_nearest(original_index, new_index):
    """Test method='nearest' fills gaps using nearest valid observation."""
    assume(len(original_index) >= 2)
    data = list(range(len(original_index)))
    df = pd.DataFrame({"value": data}, index=original_index)

    result = df.reindex(new_index, method="nearest")

    for idx in new_index:
        if idx in original_index:
            continue
        distances = [abs(idx - i) for i in original_index]
        min_dist = min(distances)
        closest_original_indices = [original_index[i] for i, d in enumerate(distances) if d == min_dist]
        assert result.loc[idx, "value"] in [df.loc[oi, "value"] for oi in closest_original_indices]


@given(
    original_index=lists(
        integers(min_value=0, max_value=100), min_size=3, max_size=10, unique=True
    ).map(sorted),
    new_index=lists(
        integers(min_value=0, max_value=100), min_size=5, max_size=15, unique=True
    ).map(sorted),
    tolerance=integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_reindex_tolerance_scalar(original_index, new_index, tolerance):
    """Test tolerance parameter limits how far inexact matches can be made."""
    assume(len(original_index) >= 2)
    data = list(range(len(original_index)))
    df = pd.DataFrame({"value": data}, index=original_index)

    result = df.reindex(new_index, method="nearest", tolerance=tolerance)

    for idx in new_index:
        if idx in original_index:
            continue
        distances = [abs(idx - i) for i in original_index]
        min_dist = min(distances)
        if min_dist <= tolerance:
            closest_original_indices = [original_index[i] for i, d in enumerate(distances) if d == min_dist]
            assert result.loc[idx, "value"] in [df.loc[oi, "value"] for oi in closest_original_indices]
        else:
            assert pd.isna(result.loc[idx, "value"])


@given(
    original_index=lists(
        integers(min_value=0, max_value=100), min_size=3, max_size=8, unique=True
    ).map(sorted),
    new_index=lists(
        integers(min_value=0, max_value=100), min_size=3, max_size=8, unique=True
    ).map(sorted),
    tolerance=lists(integers(min_value=1, max_value=20), min_size=3, max_size=8),
)
@settings(max_examples=100)
def test_reindex_tolerance_list(original_index, new_index, tolerance):
    """Test tolerance parameter with list-like variable tolerance per element."""
    assume(len(original_index) >= 3)
    assume(len(tolerance) == len(new_index))
    data = list(range(len(original_index)))
    df = pd.DataFrame({"value": data}, index=original_index)

    result = df.reindex(new_index, method="nearest", tolerance=tolerance)

    for i, idx in enumerate(new_index):
        if idx in original_index:
            continue
        distances = [abs(idx - j) for j in original_index]
        min_dist = min(distances)
        if min_dist <= tolerance[i]:
            closest_original_indices = [original_index[j] for j, d in enumerate(distances) if d == min_dist]
            assert result.loc[idx, "value"] in [df.loc[oi, "value"] for oi in closest_original_indices]
        else:
            assert pd.isna(result.loc[idx, "value"])


def test_reindex_original_nan_not_filled_by_method():
    """Doc-specified: NaN values in original DataFrame are NOT filled by method."""
    date_index = pd.date_range("1/1/2010", periods=6, freq="D")
    df = pd.DataFrame(
        {"prices": [100, 101, np.nan, 100, 89, 88]}, index=date_index
    )
    date_index2 = pd.date_range("12/29/2009", periods=10, freq="D")

    result = df.reindex(date_index2, method="bfill")

    assert pd.isna(result.loc["2010-01-03", "prices"])


def test_reindex_method_pad_alias():
    """Test that 'pad' is an alias for 'ffill'."""
    df = pd.DataFrame({"value": [1, 2, 3]}, index=[1, 2, 3])
    result_pad = df.reindex([0, 1, 2, 3], method="pad")
    result_ffill = df.reindex([0, 1, 2, 3], method="ffill")

    assert result_pad.equals(result_ffill)


def test_reindex_method_backfill_alias():
    """Test that 'backfill' is an alias for 'bfill'."""
    df = pd.DataFrame({"value": [1, 2, 3]}, index=[1, 2, 3])
    result_backfill = df.reindex([1, 2, 3, 4], method="backfill")
    result_bfill = df.reindex([1, 2, 3, 4], method="bfill")

    assert result_backfill.equals(result_bfill)


def test_reindex_fill_value_string():
    """Test fill_value with non-numeric compatible value (string)."""
    df = pd.DataFrame({"col": [1, 2, 3]}, index=[1, 2, 3])
    result = df.reindex([1, 2, 3, 4], fill_value="missing")

    assert result.loc[4, "col"] == "missing"


def test_reindex_fill_value_integer():
    """Test fill_value with integer fill value."""
    df = pd.DataFrame({"col": [1.5, 2.5, 3.5]}, index=[1, 2, 3])
    result = df.reindex([1, 2, 3, 4], fill_value=0)

    assert result.loc[4, "col"] == 0


def test_reindex_axis_parameter():
    """Test axis parameter for axis-style keyword arguments."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[0, 1])

    result = df.reindex([0, 1, 2], axis="index")
    assert len(result) == 3
    assert pd.isna(result.loc[2, "a"])
    assert pd.isna(result.loc[2, "b"])

    result_cols = df.reindex(["a", "b", "c"], axis="columns")
    assert "c" in result_cols.columns
    assert pd.isna(result_cols.loc[0, "c"])


def test_reindex_labels_parameter():
    """Test labels parameter for axis-style keyword arguments."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[0, 1])

    result = df.reindex([0, 1, 2], axis="index")

    assert len(result) == 3
    assert pd.isna(result.loc[2, "a"])


def test_reindex_empty_dataframe():
    """Test reindex on empty DataFrame returns empty DataFrame with new index."""
    df = pd.DataFrame({"col": []})
    new_index = [1, 2, 3]

    result = df.reindex(new_index)

    assert len(result) == len(new_index)
    assert list(result.index) == new_index


def test_reindex_preserve_dtype():
    """Test that reindex preserves the original dtypes where possible."""
    df = pd.DataFrame({"int_col": [1, 2], "float_col": [1.0, 2.0]}, index=[0, 1])

    result = df.reindex([0, 1])

    assert result["int_col"].dtype == np.int64
    assert result["float_col"].dtype == np.float64

    result_with_nan = df.reindex([0, 1, 2])
    assert result_with_nan["float_col"].dtype == np.float64


def test_reindex_multiindex_level():
    """Test reindex with MultiIndex using level parameter."""
    index = pd.MultiIndex.from_tuples(
        [(1, "a"), (1, "b"), (2, "a"), (2, "b")], names=["num", "letter"]
    )
    df = pd.DataFrame({"value": [1, 2, 3, 4]}, index=index)

    result = df.reindex(level="num", labels=[1, 2])

    assert set(result.index.get_level_values("num")) == {1, 2}
    assert len(result) == 4


def test_reindex_method_with_existing_index():
    """Test reindex method when new index is same as old index."""
    df = pd.DataFrame({"value": [1, 2, 3]}, index=[1, 2, 3])

    result = df.reindex([1, 2, 3], method="ffill")

    assert result.equals(df)


def test_reindex_multiple_columns_with_fill_value():
    """Test reindex with multiple columns and fill_value."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[0, 1])
    result = df.reindex([0, 1, 2], fill_value=-1)

    assert result.loc[2, "a"] == -1
    assert result.loc[2, "b"] == -1


def test_reindex_both_index_and_columns():
    """Test reindex with both index and columns specified."""
    df = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])

    result = df.reindex(index=["a", "b", "c"], columns=["x", "y", "z"])

    assert result.loc["a", "x"] == 1
    assert result.loc["a", "z"] is np.nan or pd.isna(result.loc["a", "z"])
    assert result.loc["c", "x"] is np.nan or pd.isna(result.loc["c", "x"])
