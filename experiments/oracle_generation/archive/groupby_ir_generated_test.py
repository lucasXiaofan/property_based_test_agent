import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st


@st.composite
def key_val_frames(draw, *, allow_none=False):
    size = draw(st.integers(min_value=2, max_value=8))
    key_strategy = st.sampled_from(["a", "b", "c"])
    if allow_none:
        key_strategy = st.one_of(st.sampled_from(["a", "b"]), st.none())
    keys = draw(st.lists(key_strategy, min_size=size, max_size=size))
    if allow_none and all(key is not None for key in keys):
        idx = draw(st.integers(min_value=0, max_value=size - 1))
        keys[idx] = None
    vals = draw(st.lists(st.integers(min_value=-10, max_value=10), min_size=size, max_size=size))
    return pd.DataFrame({"key": keys, "val": vals})


@st.composite
def categorical_frames(draw):
    size = draw(st.integers(min_value=1, max_value=6))
    keys = draw(st.lists(st.sampled_from(["a", "b"]), min_size=size, max_size=size))
    cat = pd.Categorical(keys, categories=["a", "b", "c"])
    vals = draw(st.lists(st.integers(min_value=-10, max_value=10), min_size=size, max_size=size))
    return pd.DataFrame({"key": cat, "val": vals})


def multiindex_frame():
    return pd.DataFrame(
        {"val": [1.0, 2.0, 3.0, 4.0]},
        index=pd.MultiIndex.from_arrays(
            [["a", "a", "b", "b"], ["x", "y", "x", "y"]],
            names=["level0", "level1"],
        ),
    )


@given(key_val_frames())
def test_groupby_sum_equivalent_for_as_index_toggle(df):
    left = df.groupby("key", as_index=True).sum(numeric_only=True).sort_index()
    right = df.groupby("key", as_index=False).sum(numeric_only=True).set_index("key").sort_index()
    pd.testing.assert_frame_equal(left, right)


@given(key_val_frames())
def test_sort_flag_controls_group_key_order(df):
    first_seen = list(dict.fromkeys(df["key"]))
    sorted_keys = sorted(first_seen)

    unsorted = list(df.groupby("key", sort=False).groups)
    ordered = list(df.groupby("key", sort=True).groups)

    assert unsorted == first_seen
    assert ordered == sorted_keys


@given(key_val_frames())
def test_row_order_is_preserved_within_each_group(df):
    grouped = df.groupby("key", sort=False)
    for key, group in grouped:
        expected = df.index[df["key"] == key].tolist()
        assert group.index.tolist() == expected


@given(key_val_frames(allow_none=True))
def test_dropna_false_matches_fillna_sentinel_grouping(df):
    # This catches NA-key handling bugs by comparing with an explicit non-null surrogate.
    sentinel = "__missing__"
    with_na = df.groupby("key", dropna=False).sum(numeric_only=True)
    explicit = df.assign(key=df["key"].fillna(sentinel)).groupby("key").sum(numeric_only=True)
    normalized = {
        (sentinel if pd.isna(key) else key): value
        for key, value in with_na["val"].items()
    }
    assert normalized == explicit["val"].to_dict()


@given(key_val_frames(allow_none=True))
def test_dropna_interacts_with_row_coverage_as_documented(df):
    dropped_rows = sum(len(group) for _, group in df.groupby("key", dropna=True))
    kept_rows = sum(len(group) for _, group in df.groupby("key", dropna=False))

    assert kept_rows == len(df)
    assert dropped_rows == int(df["key"].notna().sum())


@given(categorical_frames())
def test_observed_flag_controls_unused_categorical_levels(df):
    observed_true = df.groupby("key", observed=True).size()
    observed_false = df.groupby("key", observed=False).size()

    assert set(observed_true.index) == set(df["key"].dropna().unique())
    assert list(observed_false.index) == list(df["key"].cat.categories)
    assert observed_false.loc["c"] == 0


@given(categorical_frames(), st.booleans())
def test_observed_false_and_as_index_toggle_preserve_same_totals(df, sort):
    indexed = df.groupby("key", observed=False, as_index=True, sort=sort).sum(numeric_only=True)
    sql_style = (
        df.groupby("key", observed=False, as_index=False, sort=sort)
        .sum(numeric_only=True)
        .set_index("key")
    )
    pd.testing.assert_frame_equal(indexed, sql_style)


@given(key_val_frames())
def test_group_keys_flag_changes_like_indexed_apply_shape(df):
    like_indexed = lambda part: part[["val"]]

    with_keys = df.groupby("key", group_keys=True)[["val"]].apply(like_indexed)
    without_keys = df.groupby("key", group_keys=False)[["val"]].apply(like_indexed)

    assert isinstance(with_keys.index, pd.MultiIndex)
    assert not isinstance(without_keys.index, pd.MultiIndex)
    pd.testing.assert_frame_equal(without_keys, df[["val"]])


@given(key_val_frames())
def test_function_on_index_matches_explicit_parity_array(df):
    by_function = df.groupby(lambda idx: idx % 2).sum(numeric_only=True).sort_index()
    explicit = df.groupby(df.index.to_series().mod(2).to_numpy()).sum(numeric_only=True).sort_index()
    pd.testing.assert_frame_equal(by_function, explicit)


def test_level_name_and_integer_produce_same_groups():
    df = multiindex_frame()
    by_int = df.groupby(level=0).sum(numeric_only=True).sort_index()
    by_name = df.groupby(level="level0").sum(numeric_only=True).sort_index()
    pd.testing.assert_frame_equal(by_int, by_name)


def test_by_and_level_cannot_be_supplied_together():
    df = multiindex_frame()
    with pytest.raises(TypeError):
        df.groupby(by="val", level=0)
