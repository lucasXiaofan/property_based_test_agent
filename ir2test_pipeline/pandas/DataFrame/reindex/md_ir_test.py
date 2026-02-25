"""
Property-based tests for pandas.DataFrame.reindex
Generated from: ir2test_pipeline/pandas/DataFrame/reindex/ir.md
Target library: pandas 3.0.0
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_LABELS = ["a", "b", "c", "d", "e", "f"]


@st.composite
def df_with_string_index(draw, min_rows=1, max_rows=5):
    """DataFrame with a unique string index subset and float columns."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    index = draw(
        st.lists(st.sampled_from(_LABELS), min_size=n, max_size=n, unique=True)
    )
    n_cols = draw(st.integers(min_value=1, max_value=3))
    data = {
        f"col{i}": draw(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                min_size=n,
                max_size=n,
            )
        )
        for i in range(n_cols)
    }
    return pd.DataFrame(data, index=index)


@st.composite
def df_with_int64_column(draw):
    """DataFrame with exactly one int64 column and a small string index."""
    n = draw(st.integers(min_value=2, max_value=4))
    index = draw(
        st.lists(st.sampled_from(_LABELS), min_size=n, max_size=n, unique=True)
    )
    values = draw(
        st.lists(st.integers(min_value=-100, max_value=100), min_size=n, max_size=n)
    )
    return pd.DataFrame({"val": pd.array(values, dtype=np.int64)}, index=index)


# A label guaranteed to be absent from _LABELS subsets
_MISSING = "z"


# ---------------------------------------------------------------------------
# TestValidContracts — P1 through P15
# ---------------------------------------------------------------------------


class TestValidContracts:

    # P1 — missing labels receive NaN
    @given(df_with_string_index())
    @settings(max_examples=50)
    def test_p1_missing_labels_get_nan(self, df):
        """P1: Labels absent from the original index must receive NaN."""
        assume(_MISSING not in df.index)
        new_index = list(df.index) + [_MISSING]
        result = df.reindex(new_index)
        assert result.loc[_MISSING].isna().all()

    # P2 — fill_value replaces NaN at missing label positions
    @given(df_with_string_index())
    @settings(max_examples=50)
    def test_p2_fill_value_replaces_nan(self, df):
        """P2: fill_value must substitute NaN for all missing-label positions."""
        assume(_MISSING not in df.index)
        new_index = list(df.index) + [_MISSING]
        result = df.reindex(new_index, fill_value=0)
        assert (result.loc[_MISSING] == 0).all()
        assert not result.loc[_MISSING].isna().any()

    # P3 — shared labels retain original values unchanged
    @given(df_with_string_index(min_rows=2))
    @settings(max_examples=50)
    def test_p3_shared_labels_retain_values(self, df):
        """P3: Labels present in both old and new index must be unchanged."""
        assume(_MISSING not in df.index)
        shared = list(df.index[:2])
        new_index = shared + [_MISSING]
        result = df.reindex(new_index)
        pd.testing.assert_frame_equal(result.loc[shared], df.loc[shared])

    # P4 — return type is always DataFrame
    @given(df_with_string_index())
    @settings(max_examples=50)
    def test_p4_return_type_is_dataframe(self, df):
        """P4: reindex must always return a DataFrame."""
        assume(_MISSING not in df.index)
        result = df.reindex(list(df.index) + [_MISSING])
        assert isinstance(result, pd.DataFrame)

    # P5 — columns= adds new columns as NaN, preserves existing
    @given(df_with_string_index())
    @settings(max_examples=50)
    def test_p5_column_reindex_adds_nan_column(self, df):
        """P5: columns= must add new columns as NaN and preserve existing ones."""
        existing = df.columns[0]
        new_col = "__new__"
        assume(new_col not in df.columns)
        result = df.reindex(columns=[existing, new_col])
        assert list(result.columns) == [existing, new_col]
        assert result[new_col].isna().all()
        pd.testing.assert_series_equal(result[existing], df[existing])

    # P6 — labels+axis="columns" is equivalent to columns= kwarg
    @given(df_with_string_index())
    @settings(max_examples=50)
    def test_p6_labels_axis_columns_equals_columns_kwarg(self, df):
        """P6: reindex(labels, axis='columns') must equal reindex(columns=labels)."""
        new_cols = list(df.columns) + ["__extra__"]
        result1 = df.reindex(new_cols, axis="columns")
        result2 = df.reindex(columns=new_cols)
        pd.testing.assert_frame_equal(result1, result2)

    # P7 — method=bfill fills positions before the original range
    def test_p7_bfill_fills_before_range(self):
        """P7: method=bfill must use the next valid observation to fill gaps."""
        dates = pd.date_range("2010-01-01", periods=6)
        df2 = pd.DataFrame(
            {"prices": [100.0, 101.0, np.nan, 102.0, 103.0, 104.0]}, index=dates
        )
        extended = pd.date_range("2009-12-29", "2010-01-07")
        result = df2.reindex(extended, method="bfill")
        assert result.loc["2009-12-29", "prices"] == 100.0
        assert result.loc["2009-12-31", "prices"] == 100.0
        assert pd.isna(result.loc["2010-01-07", "prices"])  # no next value after range

    # P8 — pre-existing NaN in original data must not be filled by method
    def test_p8_preexisting_nan_not_filled_by_method(self):
        """P8: NaN already in the original DataFrame must remain NaN after reindex."""
        dates = pd.date_range("2010-01-01", periods=6)
        df2 = pd.DataFrame(
            {"prices": [100.0, 101.0, np.nan, 102.0, 103.0, 104.0]}, index=dates
        )
        extended = pd.date_range("2009-12-29", "2010-01-07")
        result = df2.reindex(extended, method="bfill")
        assert pd.isna(result.loc["2010-01-03", "prices"])

    # P9 — original DataFrame must not be mutated
    @given(df_with_string_index())
    @settings(max_examples=50)
    def test_p9_no_mutation_of_original(self, df):
        """P9: reindex must return a new object and leave the original unchanged."""
        assume(_MISSING not in df.index)
        original_values = df.values.copy()
        original_index = df.index.copy()
        result = df.reindex(list(df.index) + [_MISSING])
        assert result is not df
        pd.testing.assert_index_equal(df.index, original_index)
        np.testing.assert_array_equal(df.values, original_values)

    # P10 — result index exactly matches new_index in content and order
    @given(df_with_string_index())
    @settings(max_examples=50)
    def test_p10_result_index_matches_requested(self, df):
        """P10: result.index must equal the requested new_index exactly."""
        assume(len(df.index) >= 1)
        new_index = [_MISSING, "y"] + list(df.index[:1])
        result = df.reindex(new_index)
        assert list(result.index) == new_index

    # P11 — reindexing rows must not alter column labels
    @given(df_with_string_index())
    @settings(max_examples=50)
    def test_p11_row_reindex_preserves_columns(self, df):
        """P11: Row reindex must leave column labels unchanged."""
        assume(_MISSING not in df.index)
        original_cols = list(df.columns)
        result = df.reindex(list(df.index) + [_MISSING])
        assert list(result.columns) == original_cols

    # P12 — int64 columns are promoted to float64 when NaN fill is introduced
    @given(df_with_int64_column())
    @settings(max_examples=50)
    def test_p12_int_promoted_to_float_when_nan_introduced(self, df):
        """P12: int64 column must promote to float64 when NaN fill is required."""
        assume(_MISSING not in df.index)
        new_index = list(df.index) + [_MISSING]
        result = df.reindex(new_index)
        assert result["val"].dtype == np.float64

    # P13 — fill_value=0 on int64 must preserve int64 dtype
    @given(df_with_int64_column())
    @settings(max_examples=50)
    def test_p13_fill_value_zero_preserves_int_dtype(self, df):
        """P13: fill_value=0 on an int64 column must not promote to float64."""
        assume(_MISSING not in df.index)
        new_index = list(df.index) + [_MISSING]
        result = df.reindex(new_index, fill_value=0)
        assert result["val"].dtype == np.int64

    # P14 — limit=N caps the number of consecutive fills
    def test_p14_limit_caps_consecutive_fills(self):
        """P14: limit=1 with method=ffill must fill only 1 consecutive gap position."""
        df = pd.DataFrame({"v": [10, 20, 30]}, index=[0, 1, 5])
        result = df.reindex([0, 1, 2, 3, 4, 5], method="ffill", limit=1)
        assert not pd.isna(result.loc[2, "v"])  # 1 step from index 1 → filled
        assert pd.isna(result.loc[3, "v"])      # 2nd consecutive gap → NOT filled
        assert pd.isna(result.loc[4, "v"])      # 3rd consecutive gap → NOT filled

    # P15 — method=nearest uses the closest original label
    def test_p15_nearest_fill_uses_closest_observation(self):
        """P15: method=nearest must fill each gap with the nearest original value."""
        df = pd.DataFrame({"v": [0.0, 100.0]}, index=[0, 10])
        result = df.reindex([0, 4, 6, 10], method="nearest")
        assert result.loc[4, "v"] == 0.0    # 4 is closer to 0 than to 10
        assert result.loc[6, "v"] == 100.0  # 6 is closer to 10 than to 0


# ---------------------------------------------------------------------------
# TestInvalidContracts — E1 through E3
# ---------------------------------------------------------------------------


class TestInvalidContracts:

    # E1 — method= on a non-monotonic index must raise ValueError
    def test_e1_method_with_non_monotonic_index_raises(self):
        """E1: method= on a non-monotonic index must raise ValueError."""
        df = pd.DataFrame({"v": [1, 2, 3]}, index=[3, 1, 2])
        with pytest.raises(ValueError):
            df.reindex([1, 2, 3, 4], method="ffill")

    # E2 — limit= without method= must raise ValueError
    def test_e2_limit_without_method_raises(self):
        """E2: limit= without method= must raise ValueError."""
        df = pd.DataFrame({"v": [1, 2, 3]}, index=["a", "b", "c"])
        with pytest.raises(ValueError):
            df.reindex(["a", "b", "c", "d"], limit=2)

    # E3 — providing both labels= and index= simultaneously must raise
    # DISCREPANCY: pandas 3.0.0 does NOT raise; instead it treats `labels` as
    # the column axis and uses `index=` for rows, silently accepting the call.
    # Marked xfail to document the divergence from the IR specification.
    @pytest.mark.xfail(
        reason=(
            "IR predicts TypeError/ValueError for ambiguous labels+index usage, "
            "but pandas 3.0.0 silently accepts it (labels is treated as columns)."
        ),
        strict=False,
    )
    def test_e3_labels_and_index_simultaneously_raises(self):
        """E3: Supplying both labels= and index= must raise TypeError or ValueError."""
        df = pd.DataFrame({"v": [1, 2, 3]}, index=["a", "b", "c"])
        with pytest.raises((TypeError, ValueError)):
            df.reindex(labels=["a", "b"], index=["a", "b"])
