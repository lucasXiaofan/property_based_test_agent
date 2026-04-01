import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings, strategies as st


@st.composite
def string_index_frames(draw, min_size=1, max_size=5):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    labels = draw(
        st.lists(
            st.text(
                min_size=1,
                max_size=6,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
            ),
            min_size=size,
            max_size=size,
            unique=True,
        )
    )
    left = draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, width=32),
            min_size=size,
            max_size=size,
        )
    )
    right = draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, width=32),
            min_size=size,
            max_size=size,
        )
    )
    return pd.DataFrame(
        {"http_status": left, "response_time": right},
        index=pd.Index(labels, dtype=object),
    )


def _novel_labels(existing, count):
    base = "__new__"
    out = []
    i = 0
    existing = set(existing)
    while len(out) < count:
        candidate = f"{base}{i}"
        if candidate not in existing:
            out.append(candidate)
        i += 1
    return out


@given(df=string_index_frames(), extra_count=st.integers(min_value=1, max_value=3))
@settings(max_examples=40)
def test_reindex_index_preserves_overlap_order_and_marks_new_rows_missing(df, extra_count):
    target = _novel_labels(df.index, extra_count) + list(reversed(df.index))

    result = df.reindex(target)

    assert list(result.index) == target
    pd.testing.assert_frame_equal(
        result.loc[list(df.index[::-1])],
        df.loc[list(df.index[::-1])],
        check_index_type=False,
    )
    assert result.loc[target[:extra_count]].isna().all().all()


@given(df=string_index_frames())
@settings(max_examples=40)
def test_reindex_columns_preserves_row_index_and_sets_new_columns_missing(df):
    target_columns = ["response_time", "user_agent", "http_status", "cache_status"]

    result = df.reindex(columns=target_columns)

    assert list(result.columns) == target_columns
    assert list(result.index) == list(df.index)
    assert result[["user_agent", "cache_status"]].isna().all().all()
    pd.testing.assert_series_equal(result["response_time"], df["response_time"], check_names=False)
    pd.testing.assert_series_equal(result["http_status"], df["http_status"], check_names=False)


@given(df=string_index_frames(), extra_count=st.integers(min_value=1, max_value=2))
@settings(max_examples=30)
def test_fill_value_string_only_affects_new_rows(df, extra_count):
    new_labels = _novel_labels(df.index, extra_count)
    target = list(df.index) + new_labels

    result = df.reindex(target, fill_value="missing")

    assert np.array_equal(result.loc[list(df.index)].to_numpy(), df.to_numpy())
    assert (result.loc[new_labels] == "missing").all().all()


def test_method_requires_truly_non_monotonic_index():
    df = pd.DataFrame(
        {"http_status": [200.0, 201.0, 202.0], "response_time": [1.0, 2.0, 3.0]},
        index=pd.Index([1, 3, 2]),
    )

    with pytest.raises(ValueError):
        df.reindex([1, 2, 3, 4], method="ffill")


@given(df=string_index_frames())
@settings(max_examples=40)
def test_identical_index_with_copy_false_can_return_same_object(df):
    with pytest.warns(Warning):
        result = df.reindex(df.index, copy=False)

    pd.testing.assert_frame_equal(result, df)
    assert result is not df


@given(df=string_index_frames())
@settings(max_examples=40)
def test_axis_style_for_index_matches_keyword_style(df):
    target = _novel_labels(df.index, 2) + list(df.index[:2])

    result = df.reindex(target, axis="index")
    expected = df.reindex(index=target)

    pd.testing.assert_frame_equal(result, expected)


@given(df=string_index_frames())
@settings(max_examples=40)
def test_axis_style_for_columns_matches_keyword_style(df):
    labels = ["http_status", "user_agent", "response_time"]

    result = df.reindex(labels, axis=1)
    expected = df.reindex(columns=labels)

    pd.testing.assert_frame_equal(result, expected)


def test_ffill_limit_applies_only_to_inserted_gaps_and_not_original_nans():
    df = pd.DataFrame(
        {"prices": [100.0, np.nan, 80.0]},
        index=pd.Index([0, 2, 4]),
    )

    result = df.reindex([0, 1, 2, 3, 4], method="ffill", limit=1)

    assert result.loc[1, "prices"] == 100.0
    assert pd.isna(result.loc[3, "prices"])
    assert pd.isna(result.loc[2, "prices"])
    assert result.loc[4, "prices"] == 80.0


def test_nearest_with_tolerance_blocks_far_matches():
    df = pd.DataFrame({"prices": [10.0, 20.0]}, index=pd.Index([0.0, 2.0]))

    blocked = df.reindex([1.6], method="nearest", tolerance=0.3)
    matched = df.reindex([1.6], method="nearest", tolerance=0.5)

    assert blocked.isna().all().all()
    assert matched.iloc[0, 0] == 20.0


@given(df=string_index_frames(), mid_extra=st.integers(min_value=1, max_value=2), final_extra=st.integers(min_value=1, max_value=2))
@settings(max_examples=30)
def test_reindex_is_compositional_without_fill_methods(df, mid_extra, final_extra):
    mid = list(df.index) + _novel_labels(df.index, mid_extra)
    final = _novel_labels(df.index, final_extra) + list(df.index[::-1])

    via_mid = df.reindex(mid).reindex(final)
    direct = df.reindex(final)

    pd.testing.assert_frame_equal(via_mid, direct)
