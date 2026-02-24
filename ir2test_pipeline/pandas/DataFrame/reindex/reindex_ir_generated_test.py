import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st


@st.composite
def unique_string_dataframe(draw, min_rows=1, max_rows=6, min_cols=1, max_cols=4):
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    index = draw(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
                min_size=1,
                max_size=6,
            ),
            min_size=n_rows,
            max_size=n_rows,
            unique=True,
        )
    )
    columns = draw(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
                min_size=1,
                max_size=6,
            ),
            min_size=n_cols,
            max_size=n_cols,
            unique=True,
        )
    )

    data = {}
    for col in columns:
        data[col] = draw(
            st.lists(
                st.floats(
                    allow_nan=False,
                    allow_infinity=False,
                    min_value=-1e5,
                    max_value=1e5,
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )

    return pd.DataFrame(data, index=index)


@st.composite
def monotonic_int_index_dataframe(draw, min_rows=3, max_rows=8, min_cols=1, max_cols=3):
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    index_vals = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=50),
                min_size=n_rows,
                max_size=n_rows,
                unique=True,
            )
        )
    )

    data = {}
    for i in range(n_cols):
        data[f"c{i}"] = draw(
            st.lists(
                st.integers(min_value=-1000, max_value=1000),
                min_size=n_rows,
                max_size=n_rows,
            )
        )

    return pd.DataFrame(data, index=index_vals)


@st.composite
def non_monotonic_int_index_dataframe(draw, min_rows=3, max_rows=8, min_cols=1, max_cols=3):
    df = draw(monotonic_int_index_dataframe(min_rows=min_rows, max_rows=max_rows, min_cols=min_cols, max_cols=max_cols))
    shuffled = draw(st.permutations(list(df.index)))
    assume(list(shuffled) != sorted(shuffled))
    return df.reindex(list(shuffled))


class TestValidContracts:
    @given(df=unique_string_dataframe(min_rows=1), axis_is_name=st.booleans())
    @settings(max_examples=60)
    def test_p1_identity_labels_preserve_data(self, df, axis_is_name):
        labels = list(df.index)
        axis = "index" if axis_is_name else 0
        result = df.reindex(labels=labels, axis=axis)
        assert list(result.index) == labels
        pd.testing.assert_frame_equal(result, df)

    @given(df=unique_string_dataframe(min_rows=1))
    @settings(max_examples=40)
    def test_p2_empty_target_rows(self, df):
        result = df.reindex(labels=[], axis="index")
        assert len(result.index) == 0
        assert list(result.columns) == list(df.columns)

    @given(df=unique_string_dataframe(min_rows=1))
    @settings(max_examples=40)
    def test_p3_disjoint_rows_default_to_nan(self, df):
        disjoint = [f"__new_row_{i}__" for i in range(3)]
        assume(all(lbl not in df.index for lbl in disjoint))
        result = df.reindex(index=disjoint)
        assert list(result.index) == disjoint
        assert result.isna().all().all()

    @given(df=unique_string_dataframe(min_rows=2))
    @settings(max_examples=50)
    def test_p4_partial_overlap_rows_preserve_matches_and_nan_for_new(self, df):
        kept = list(df.index[:1])
        new_labels = kept + ["__missing_a__", "__missing_b__"]
        assume(all(lbl not in df.index for lbl in new_labels[1:]))

        result = df.reindex(index=new_labels)
        pd.testing.assert_series_equal(result.loc[kept[0]], df.loc[kept[0]], check_names=False)
        assert result.loc["__missing_a__"].isna().all()
        assert result.loc["__missing_b__"].isna().all()

    @given(df=unique_string_dataframe(min_rows=1, min_cols=2))
    @settings(max_examples=50)
    def test_p5_partial_overlap_columns(self, df):
        keep = [df.columns[0]]
        new_columns = keep + ["__new_col_a__", "__new_col_b__"]
        assume(all(c not in df.columns for c in new_columns[1:]))

        result = df.reindex(columns=new_columns)
        assert list(result.columns) == new_columns
        pd.testing.assert_series_equal(result[keep[0]], df[keep[0]], check_names=False)
        assert result["__new_col_a__"].isna().all()
        assert result["__new_col_b__"].isna().all()

    @given(df=unique_string_dataframe(min_rows=1, min_cols=1))
    @settings(max_examples=50)
    def test_p6_keyword_identity_index_and_columns(self, df):
        result = df.reindex(index=df.index, columns=df.columns)
        pd.testing.assert_frame_equal(result, df)

    @given(df=unique_string_dataframe(min_rows=1))
    @settings(max_examples=40)
    def test_p7_empty_index_keeps_columns(self, df):
        result = df.reindex(index=[])
        assert result.shape[0] == 0
        assert list(result.columns) == list(df.columns)

    @given(df=unique_string_dataframe(min_rows=1, min_cols=1))
    @settings(max_examples=40)
    def test_p8_empty_columns_keeps_index(self, df):
        result = df.reindex(columns=[])
        assert result.shape[1] == 0
        assert list(result.index) == list(df.index)

    @given(df=unique_string_dataframe(min_rows=1, min_cols=1))
    @settings(max_examples=50)
    def test_p9_disjoint_columns_default_to_nan(self, df):
        new_columns = ["__x__", "__y__", "__z__"]
        assume(all(c not in df.columns for c in new_columns))
        result = df.reindex(columns=new_columns)
        assert list(result.columns) == new_columns
        assert result.isna().all().all()

    @given(df=unique_string_dataframe(min_rows=1, min_cols=2))
    @settings(max_examples=50)
    def test_p10_labels_with_axis_columns_targets_columns(self, df):
        labels = [df.columns[0], "__new_col__"]
        assume(labels[1] not in df.columns)
        result = df.reindex(labels=labels, axis="columns")
        assert list(result.columns) == labels
        assert list(result.index) == list(df.index)

    @given(df=monotonic_int_index_dataframe(min_rows=3))
    @settings(max_examples=60)
    def test_p11_ffill_on_monotonic_index(self, df):
        idx = list(df.index)
        low, high = idx[0], idx[1]
        assume(high - low >= 2)
        between = low + 1
        target = sorted(idx + [between])

        result = df.reindex(index=target, method="ffill")
        pd.testing.assert_series_equal(result.loc[between], df.loc[low], check_names=False)

    @given(df=monotonic_int_index_dataframe(min_rows=3))
    @settings(max_examples=60)
    def test_p12_bfill_on_monotonic_index(self, df):
        idx = list(df.index)
        low, high = idx[0], idx[1]
        assume(high - low >= 2)
        between = low + 1
        target = sorted(idx + [between])

        result = df.reindex(index=target, method="bfill")
        pd.testing.assert_series_equal(result.loc[between], df.loc[high], check_names=False)

    def test_p13_nearest_respects_scalar_tolerance(self):
        df = pd.DataFrame({"v": [10.0, 20.0]}, index=[0.0, 1.0])
        result = df.reindex(index=[0.1, 0.9], method="nearest", tolerance=0.15)
        assert result.loc[0.1, "v"] == 10.0
        assert result.loc[0.9, "v"] == 20.0

        result_tight = df.reindex(index=[0.1, 0.9], method="nearest", tolerance=0.05)
        assert pd.isna(result_tight.loc[0.1, "v"])
        assert pd.isna(result_tight.loc[0.9, "v"])

    @given(df=unique_string_dataframe(min_rows=1))
    @settings(max_examples=50)
    def test_p14_copy_true_false_same_logical_result(self, df):
        new_index = list(df.index) + ["__new_copy_row__"]
        assume("__new_copy_row__" not in df.index)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_true = df.reindex(index=new_index, copy=True)
            r_false = df.reindex(index=new_index, copy=False)

        pd.testing.assert_frame_equal(r_true, r_false)

    def test_p15_multiindex_level_reindex_preserves_structure(self):
        mi = pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 1)], names=["outer", "inner"]
        )
        df = pd.DataFrame({"v": [1, 2, 3]}, index=mi)

        result = df.reindex(index=["a", "b", "c"], level=0)
        assert result.index.nlevels == df.index.nlevels
        assert set(result.index.get_level_values(0)).issubset({"a", "b", "c"})
        assert set(result.index.get_level_values(0)).issubset(set(df.index.get_level_values(0)))

    @given(df=unique_string_dataframe(min_rows=1))
    @settings(max_examples=50)
    def test_p16_numeric_fill_value_applied_to_new_rows(self, df):
        new_index = list(df.index) + ["__n1__", "__n2__"]
        assume(all(lbl not in df.index for lbl in ["__n1__", "__n2__"]))

        result = df.reindex(index=new_index, fill_value=0)
        assert (result.loc["__n1__"] == 0).all()
        assert (result.loc["__n2__"] == 0).all()

    @given(df=unique_string_dataframe(min_rows=1))
    @settings(max_examples=50)
    def test_p17_string_fill_value_applied_to_new_rows(self, df):
        new_index = list(df.index) + ["__s1__"]
        assume("__s1__" not in df.index)

        result = df.reindex(index=new_index, fill_value="missing")
        assert (result.loc["__s1__"] == "missing").all()

    def test_p18_ffill_with_limit_caps_consecutive_fills(self):
        df = pd.DataFrame({"v": [1.0, 2.0]}, index=[10, 20])
        target = [10, 11, 12, 13, 20]

        result = df.reindex(index=target, method="ffill", limit=1)
        assert result.loc[11, "v"] == 1.0
        assert pd.isna(result.loc[12, "v"])
        assert pd.isna(result.loc[13, "v"])

    def test_p19_nearest_listlike_tolerance_is_elementwise(self):
        df = pd.DataFrame({"v": [100.0, 200.0]}, index=[0.0, 1.0])
        target = [0.1, 0.9]
        tolerance = [0.2, 0.01]

        result = df.reindex(index=target, method="nearest", tolerance=tolerance)
        assert result.loc[0.1, "v"] == 100.0
        assert pd.isna(result.loc[0.9, "v"])

    @given(df=unique_string_dataframe(min_rows=1))
    @settings(max_examples=50)
    def test_p20_disjoint_labels_axis_index_all_nan(self, df):
        labels = ["__d1__", "__d2__"]
        assume(all(lbl not in df.index for lbl in labels))
        result = df.reindex(labels=labels, axis="index")
        assert list(result.index) == labels
        assert result.isna().all().all()


class TestInvalidContracts:
    @given(df=unique_string_dataframe(min_rows=1))
    @settings(max_examples=30)
    def test_e1_labels_positional_and_keyword_raises(self, df):
        with pytest.raises(TypeError):
            df.reindex([0, 1], labels=[0, 1])

    def test_e2_duplicate_source_index_raises(self):
        df = pd.DataFrame({"v": [1, 2, 3]}, index=[1, 2, 1])
        with pytest.raises(ValueError, match="duplicate labels"):
            df.reindex(index=[0, 1, 2])

    def test_e3_duplicate_source_columns_raises(self):
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
        with pytest.raises(ValueError, match="duplicate labels"):
            df.reindex(columns=["a"])

    @given(df=unique_string_dataframe(min_rows=1, min_cols=1))
    @settings(max_examples=30)
    def test_e4_axis_with_columns_style_raises(self, df):
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex([0, 1], columns=[df.columns[0]], axis=1)

    @given(df=unique_string_dataframe(min_rows=1, min_cols=1))
    @settings(max_examples=30)
    def test_e5_too_many_positional_args_raises(self, df):
        with pytest.raises(TypeError):
            df.reindex([0, 1], ["A", "B"])

    @given(df=monotonic_int_index_dataframe(min_rows=3))
    @settings(max_examples=30)
    def test_e6_invalid_method_name_raises(self, df):
        with pytest.raises(ValueError, match="Invalid fill method"):
            df.reindex([1, 0, 2], method="asfreq")

    @given(df=non_monotonic_int_index_dataframe(min_rows=3))
    @settings(max_examples=40)
    def test_e7_method_with_non_monotonic_index_raises(self, df):
        target = list(df.index) + [999]
        with pytest.raises(ValueError):
            df.reindex(target, method="ffill")

    def test_e8_method_with_level_raises(self):
        mi = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
        df = pd.DataFrame({"v": [1, 2, 3]}, index=mi)
        with pytest.raises(TypeError, match="Fill method not supported if level passed"):
            df.reindex(["a", "b"], level=0, method="ffill")

    def test_e9_list_tolerance_wrong_size_raises(self):
        df = pd.DataFrame({"v": [10.0, 20.0, 30.0]}, index=[0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="tolerance size"):
            df.reindex([0.1, 0.9, 1.9], method="nearest", tolerance=[0.2, 0.2])

    @given(df=monotonic_int_index_dataframe(min_rows=3))
    @settings(max_examples=30)
    def test_e10_tolerance_without_method_raises(self, df):
        target = list(df.index) + [100]
        with pytest.raises(ValueError):
            df.reindex(target, method=None, tolerance=1)
