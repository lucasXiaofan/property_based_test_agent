import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, indexes

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_float_df = data_frames(
    columns=[column("a", dtype=float), column("b", dtype=float)],
    index=indexes(dtype=int, min_size=2, max_size=6, unique=True),
)


# ---------------------------------------------------------------------------
# TestValidContracts
# ---------------------------------------------------------------------------


class TestValidContracts:

    @given(_float_df)
    @settings(max_examples=50)
    def test_p1_new_labels_get_nan_without_fill_value(self, df):
        """P1: labels not present in original index receive NaN when fill_value is not specified."""
        assume(len(df) >= 1)
        orig_idx = df.index.tolist()
        extra = [max(orig_idx) + 1, max(orig_idx) + 2]
        new_index = orig_idx + extra
        result = df.reindex(new_index)
        new_labels = set(new_index) - set(orig_idx)
        for lbl in new_labels:
            assert result.loc[lbl].isna().all(), (
                f"Expected all-NaN row at new label {lbl}"
            )

    @given(_float_df)
    @settings(max_examples=50)
    def test_p2_returns_distinct_object(self, df):
        """P2: reindex always returns a new DataFrame object distinct from the original."""
        result = df.reindex(df.index.tolist())
        assert result is not df

    def test_p3_copy_true_raises_deprecation_warning(self):
        """P3: explicitly passing copy=True raises a deprecation warning.
        IR claims FutureWarning; pandas 3.0.0 emits Pandas4Warning (DeprecationWarning subclass).
        """
        df = pd.DataFrame({"x": [1.0, 2.0]}, index=[0, 1])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = df.reindex([0, 1, 2], copy=True)
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_p3_copy_false_raises_deprecation_warning(self):
        """P3: explicitly passing copy=False raises a deprecation warning.
        IR claims FutureWarning; pandas 3.0.0 emits Pandas4Warning (DeprecationWarning subclass).
        """
        df = pd.DataFrame({"x": [1.0, 2.0]}, index=[0, 1])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = df.reindex([0, 1, 2], copy=False)
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_p4_ffill_uses_preceding_observation(self):
        """P4: method='ffill' fills each new label with last original observation preceding it."""
        df = pd.DataFrame({"v": [10.0, 20.0, 30.0]}, index=[1, 3, 5])
        result = df.reindex([1, 2, 3, 4, 5], method="ffill")
        assert result.loc[2, "v"] == 10.0  # preceding: row 1
        assert result.loc[4, "v"] == 20.0  # preceding: row 3

    def test_p5_bfill_uses_following_observation(self):
        """P5: method='bfill' fills each new label with next original observation following it."""
        df = pd.DataFrame({"v": [10.0, 20.0, 30.0]}, index=[1, 3, 5])
        result = df.reindex([1, 2, 3, 4, 5], method="bfill")
        assert result.loc[2, "v"] == 20.0  # following: row 3
        assert result.loc[4, "v"] == 30.0  # following: row 5

    def test_p6_pad_alias_for_ffill(self):
        """P6: 'pad' is an exact alias for 'ffill' and produces bit-for-bit identical results."""
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=[0, 2, 4])
        new_idx = [0, 1, 2, 3, 4]
        result_ffill = df.reindex(new_idx, method="ffill")
        result_pad   = df.reindex(new_idx, method="pad")
        assert result_ffill.equals(result_pad)

    def test_p7_backfill_alias_for_bfill(self):
        """P7: 'backfill' is an exact alias for 'bfill' and produces bit-for-bit identical results."""
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=[0, 2, 4])
        new_idx = [0, 1, 2, 3, 4]
        result_bfill    = df.reindex(new_idx, method="bfill")
        result_backfill = df.reindex(new_idx, method="backfill")
        assert result_bfill.equals(result_backfill)

    def test_p8_limit_caps_consecutive_fill(self):
        """P8: limit=N caps forward fill to at most N consecutive new positions; beyond N get NaN."""
        df = pd.DataFrame({"v": [10.0, 50.0]}, index=[0, 10])
        # labels 1, 2, 3 are consecutive new rows after row 0
        result = df.reindex([0, 1, 2, 3, 10], method="ffill", limit=1)
        assert result.loc[1, "v"] == 10.0    # first consecutive: filled
        assert np.isnan(result.loc[2, "v"])  # second consecutive: NaN
        assert np.isnan(result.loc[3, "v"])  # third consecutive: NaN

    def test_p9_fill_value_for_new_labels_only(self):
        """P9: fill_value replaces NaN for new labels while leaving existing label values unchanged."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=[0, 1])
        result = df.reindex([0, 1, 2, 3], fill_value=0)
        assert (result.loc[2] == 0).all()
        assert (result.loc[3] == 0).all()
        assert result.loc[0, "a"] == 1.0
        assert result.loc[1, "b"] == 4.0

    def test_p10_nearest_scalar_tolerance_far_label_gets_nan(self):
        """P10: when method='nearest' and scalar tolerance, labels farther than tolerance get NaN."""
        df = pd.DataFrame({"v": [1.0, 2.0]}, index=[0.0, 10.0])
        # label 4.5: nearest original is 0.0 (dist=4.5), tolerance=3.0 → 4.5 > 3.0 → NaN
        result = df.reindex([0.0, 4.5, 10.0], method="nearest", tolerance=3.0)
        assert np.isnan(result.loc[4.5, "v"])
        assert result.loc[0.0, "v"] == 1.0   # exact match → kept
        assert result.loc[10.0, "v"] == 2.0  # exact match → kept

    def test_p11_nearest_list_tolerance_per_label(self):
        """P11: list-like tolerance applies independently to each new label position."""
        df = pd.DataFrame({"v": [1.0, 2.0]}, index=[0.0, 10.0])
        # tolerances [0.5, 6.0, 0.5] for new index [0.0, 4.5, 10.0]
        # label 4.5: nearest=0.0 (dist=4.5), tol=6.0 → 4.5 < 6.0 → filled with 1.0
        result = df.reindex([0.0, 4.5, 10.0], method="nearest", tolerance=[0.5, 6.0, 0.5])
        assert result.loc[0.0, "v"] == 1.0
        assert result.loc[4.5, "v"] == 1.0  # within tolerance of nearest original (0.0)
        assert result.loc[10.0, "v"] == 2.0

    def test_p12_nearest_picks_smallest_distance(self):
        """P12: method='nearest' selects the original label with smallest absolute distance."""
        df = pd.DataFrame({"v": [1.0, 3.0, 5.0]}, index=[0, 4, 10])
        # label 3: dist to 0=3, dist to 4=1, dist to 10=7 → nearest is 4 → value 3.0
        result = df.reindex([3], method="nearest")
        assert result.loc[3, "v"] == 3.0

    def test_p13_fill_method_preserves_preexisting_nan_in_retained_rows(self):
        """P13: fill methods do not fill pre-existing NaN in retained original rows."""
        # row 1 has NaN in the original and is retained in the new index
        df = pd.DataFrame({"v": [1.0, np.nan, 3.0]}, index=[0, 1, 4])
        result = df.reindex([0, 1, 2, 4], method="ffill")
        assert np.isnan(result.loc[1, "v"])

    @given(_float_df)
    @settings(max_examples=50)
    def test_p14_retained_labels_preserve_original_values(self, df):
        """P14: labels present in both original and new index have exact original row values."""
        assume(len(df) >= 1)
        orig_idx = df.index.tolist()
        extra = [max(orig_idx) + 100]
        new_index = orig_idx + extra
        result = df.reindex(new_index)
        for lbl in orig_idx:
            assert result.loc[lbl].equals(df.loc[lbl])

    @given(_float_df)
    @settings(max_examples=50)
    def test_p15_result_index_matches_requested_order(self, df):
        """P15: the result index is exactly the requested new index in the same order."""
        assume(len(df) >= 2)
        orig_idx = df.index.tolist()
        # Reverse to test ordering is respected
        new_index = list(reversed(orig_idx)) + [max(orig_idx) + 1]
        result = df.reindex(new_index)
        assert result.index.tolist() == new_index

    @given(_float_df)
    @settings(max_examples=50)
    def test_p16_columns_unchanged_on_row_reindex(self, df):
        """P16: column labels are unchanged when only the row index is reindexed."""
        assume(len(df) >= 1)
        orig_idx = df.index.tolist()
        result = df.reindex(orig_idx + [max(orig_idx) + 1])
        assert result.columns.tolist() == df.columns.tolist()

    def test_p17_identity_reindex_equals_original(self):
        """P17: reindexing with the identical index in the same order produces a DataFrame equal to the original."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[10, 20, 30])
        result = df.reindex(df.index.tolist())
        assert result.equals(df)
        assert result is not df

    def test_p18_int_columns_promoted_to_float64_on_nan_introduction(self):
        """P18: integer-dtype columns are promoted to float64 when new labels introduce NaN."""
        df = pd.DataFrame({"col": np.array([1, 2, 3], dtype=np.int64)}, index=[0, 1, 2])
        result = df.reindex([0, 1, 2, 3])
        assert result["col"].dtype == np.float64

    @given(_float_df)
    @settings(max_examples=50)
    def test_p19_row_count_equals_new_index_length(self, df):
        """P19: the number of rows in the result equals the length of the requested new index."""
        assume(len(df) >= 1)
        orig_idx = df.index.tolist()
        extra = [max(orig_idx) + 1, max(orig_idx) + 2]
        new_index = orig_idx + extra
        result = df.reindex(new_index)
        assert len(result) == len(new_index)

    @given(_float_df)
    @settings(max_examples=50)
    def test_p20_return_type_is_dataframe(self, df):
        """P20: the return value is always an instance of pd.DataFrame regardless of parameters."""
        assume(len(df) >= 1)
        orig_idx = df.index.tolist()
        extra = [max(orig_idx) + 1]
        assert isinstance(df.reindex(orig_idx), pd.DataFrame)
        assert isinstance(df.reindex(orig_idx + extra), pd.DataFrame)
        assert isinstance(df.reindex([]), pd.DataFrame)

    def test_p21_column_reindex_new_cols_nan_retained_preserved(self):
        """P21: column reindexing fills new column labels with NaN and preserves retained column values."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = df.reindex(columns=["a", "b", "c", "d"])
        assert result["c"].isna().all()
        assert result["d"].isna().all()
        assert result["a"].equals(df["a"])
        assert result["b"].equals(df["b"])

    @given(_float_df)
    @settings(max_examples=50)
    def test_p22_original_not_mutated(self, df):
        """P22: the original DataFrame is not mutated by any call to reindex."""
        assume(len(df) >= 1)
        original_copy = df.copy(deep=True)
        orig_idx = df.index.tolist()
        _ = df.reindex(orig_idx + [max(orig_idx) + 1], fill_value=0)
        assert df.equals(original_copy)

    def test_p23_both_axes_reindexed_simultaneously(self):
        """P23: when both index and columns are provided, the result is reindexed on both axes."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=[0, 1])
        new_index = [0, 1, 2]
        new_columns = ["a", "b", "c"]
        result = df.reindex(index=new_index, columns=new_columns)
        assert result.index.tolist() == new_index
        assert result.columns.tolist() == new_columns


# ---------------------------------------------------------------------------
# TestInvalidContracts
# ---------------------------------------------------------------------------


class TestInvalidContracts:

    def test_e1_non_monotonic_index_with_method_raises_value_error(self):
        """E1: using method on a DataFrame with non-monotonic index raises ValueError."""
        df = pd.DataFrame({"v": [1.0, 3.0, 2.0, 4.0]}, index=[0, 2, 1, 3])
        with pytest.raises(ValueError):
            df.reindex([0, 1, 2, 3], method="ffill")

    def test_e2_unrecognised_method_raises_value_error(self):
        """E2: passing an unrecognised string as method raises ValueError."""
        df = pd.DataFrame({"v": [1.0, 2.0]}, index=[0, 1])
        with pytest.raises(ValueError):
            df.reindex([0, 1, 2], method="linear")

    @pytest.mark.xfail(
        reason="IR (E3) claims wrong-length list tolerance raises ValueError; "
               "pandas 3.0.0 does not enforce this and raises no exception.",
        strict=True,
    )
    def test_e3_list_tolerance_wrong_length_raises_value_error(self):
        """E3: passing a list-like tolerance whose length differs from the new index length raises ValueError."""
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=[1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            df.reindex([1.0, 2.0, 3.0], method="nearest", tolerance=[0.1, 0.5])

    def test_e4_limit_without_method_raises_value_error(self):
        """E4: setting limit without a fill method raises ValueError."""
        df = pd.DataFrame({"v": [1.0, 2.0]}, index=[0, 1])
        with pytest.raises(ValueError):
            df.reindex([0, 1, 2], limit=1)

    def test_e5_invalid_axis_raises_value_error(self):
        """E5: passing an invalid axis value raises ValueError."""
        df = pd.DataFrame({"v": [1.0, 2.0]}, index=["a", "b"])
        with pytest.raises(ValueError):
            df.reindex(["a", "b"], axis=2)
