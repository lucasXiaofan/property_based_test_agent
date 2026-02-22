"""
Property-based tests for pd.DataFrame.reindex
Generated from: ir/pandas/dataframe_reindex_properties.md
pandas version: 3.0.0

Run with:
    uv run pytest pandas_bug_finding/ir_test/test_dataframe_reindex.py -v
    uv run pytest pandas_bug_finding/ir_test/test_dataframe_reindex.py -v --hypothesis-seed=0
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, indexes, range_indexes


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Scalar values for DataFrames (no NaN at the strategy level; tests introduce them)
numeric_scalar = st.one_of(st.integers(-100, 100), st.floats(-100, 100, allow_nan=False))

# A small DataFrame with an integer index — the workhorse for most tests
@st.composite
def int_index_dataframe(draw, min_rows=0, max_rows=8, min_cols=1, max_cols=4):
    n_rows = draw(st.integers(min_rows, max_rows))
    n_cols = draw(st.integers(min_cols, max_cols))
    index_vals = draw(
        st.lists(st.integers(-20, 20), min_size=n_rows, max_size=n_rows, unique=True)
    )
    data = {
        f"c{i}": draw(
            st.lists(st.floats(allow_nan=True, allow_infinity=False), min_size=n_rows, max_size=n_rows)
        )
        for i in range(n_cols)
    }
    return pd.DataFrame(data, index=index_vals)


@st.composite
def monotonic_int_index_dataframe(draw, min_rows=1, max_rows=8, min_cols=1, max_cols=3):
    """DataFrame with a strictly monotonically increasing integer index."""
    n_rows = draw(st.integers(min_rows, max_rows))
    n_cols = draw(st.integers(min_cols, max_cols))
    start = draw(st.integers(-10, 10))
    step = draw(st.integers(1, 5))
    index_vals = list(range(start, start + n_rows * step, step))
    data = {
        f"c{i}": draw(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        for i in range(n_cols)
    }
    return pd.DataFrame(data, index=index_vals)


@st.composite
def non_monotonic_int_index_dataframe(draw, min_rows=3, max_rows=8, min_cols=1, max_cols=3):
    """DataFrame whose integer index is guaranteed to be non-monotonic."""
    n_rows = draw(st.integers(min_rows, max_rows))
    n_cols = draw(st.integers(min_cols, max_cols))
    vals = draw(
        st.lists(st.integers(-20, 20), min_size=n_rows, max_size=n_rows, unique=True)
    )
    # Shuffle until non-monotonic
    assume(not (vals == sorted(vals) or vals == sorted(vals, reverse=True)))
    data = {
        f"c{i}": draw(
            st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=n_rows, max_size=n_rows)
        )
        for i in range(n_cols)
    }
    return pd.DataFrame(data, index=vals)


# ---------------------------------------------------------------------------
# Explicit properties (P1–P10)
# ---------------------------------------------------------------------------

class TestExplicit:

    @given(df=int_index_dataframe(min_rows=1))
    @settings(max_examples=60)
    def test_explicit__p1__labels_not_present_in_the_original(self, df):
        """Labels not present in the original index receive NaN by default.

        When reindex introduces new row labels and no fill_value is supplied,
        every cell in those newly added rows must be NaN.
        """
        original_set = set(df.index.tolist())
        # Build a new index that has at least one label not in the original
        extra = [max(df.index) + 1, max(df.index) + 2]
        new_index = df.index.tolist() + extra

        result = df.reindex(new_index)

        for label in extra:
            assert pd.isna(result.loc[label]).all(), (
                f"Newly introduced label {label!r} should have all-NaN row, got {result.loc[label]!r}"
            )

    @given(df=int_index_dataframe())
    @settings(max_examples=60)
    def test_explicit__p2__reindex_always_returns_a_new_dataframe(self, df):
        """reindex always returns a new DataFrame object, never the same object.

        Even when the new index is identical to the original, the returned
        object must be a distinct Python object (identity check).
        """
        result = df.reindex(df.index)
        assert result is not df

    @given(df=int_index_dataframe(min_rows=1))
    @settings(max_examples=30)
    def test_explicit__p3__passing_the_copy_keyword_raises_a(self, df):
        """Passing the `copy` keyword raises a Pandas4Warning deprecation warning.

        In pandas 3.0.0 the copy parameter is deprecated and ignored. Any call
        that passes copy=True or copy=False must emit a FutureWarning or
        pd.errors.Pandas4Warning.
        """
        new_index = df.index.tolist()
        for copy_val in (True, False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = df.reindex(new_index, copy=copy_val)
            warning_types = [type(w.category) for w in caught]
            category_names = [w.category.__name__ for w in caught]
            assert any(
                issubclass(w.category, (FutureWarning,)) or "Pandas4Warning" in w.category.__name__
                for w in caught
            ), (
                f"Expected FutureWarning or Pandas4Warning when copy={copy_val}, "
                f"got categories: {category_names}"
            )
            assert isinstance(result, pd.DataFrame)

    @given(df=monotonic_int_index_dataframe(min_rows=2))
    @settings(max_examples=60)
    def test_explicit__p4__methodffillpad_propagates_the_last_valid_observation(self, df):
        """method='ffill'/'pad' propagates the last valid observation forward.

        For each new label inserted between existing index values, the result
        row must equal the last original row that precedes it. 'ffill' and
        'pad' must produce identical results.
        """
        idx = sorted(df.index.tolist())
        # Insert one new label between the first two existing labels
        gap = idx[1] - idx[0]
        assume(gap >= 2)
        new_label = idx[0] + 1  # strictly between idx[0] and idx[1]
        new_index = sorted(idx + [new_label])

        result_ffill = df.reindex(new_index, method="ffill")
        result_pad = df.reindex(new_index, method="pad")

        # The filled row should equal the preceding original row
        assert result_ffill.loc[new_label].equals(df.loc[idx[0]]), (
            f"ffill: row at {new_label} should equal df.loc[{idx[0]}]"
        )
        # 'pad' must give the same result as 'ffill'
        assert result_ffill.equals(result_pad), "'ffill' and 'pad' must produce identical results"

    @given(df=monotonic_int_index_dataframe(min_rows=2))
    @settings(max_examples=60)
    def test_explicit__p5__methodbackfillbfill_uses_the_next_valid_observation(self, df):
        """method='backfill'/'bfill' uses the next valid observation to fill new labels.

        For each new label inserted between existing index values, the result
        row must equal the first original row that follows it. 'bfill' and
        'backfill' must produce identical results.
        """
        idx = sorted(df.index.tolist())
        gap = idx[1] - idx[0]
        assume(gap >= 2)
        new_label = idx[0] + 1
        new_index = sorted(idx + [new_label])

        result_bfill = df.reindex(new_index, method="bfill")
        result_backfill = df.reindex(new_index, method="backfill")

        # The filled row should equal the next original row
        assert result_bfill.loc[new_label].equals(df.loc[idx[1]]), (
            f"bfill: row at {new_label} should equal df.loc[{idx[1]}]"
        )
        assert result_bfill.equals(result_backfill), "'bfill' and 'backfill' must produce identical results"

    @given(df=non_monotonic_int_index_dataframe())
    @settings(max_examples=50)
    def test_explicit__p6__method_on_a_nonmonotonic_index_raises(self, df):
        """method on a non-monotonic index raises ValueError.

        When the DataFrame's row index is neither monotonically increasing nor
        decreasing, passing any method value to reindex must raise ValueError.
        """
        new_index = df.index.tolist() + [max(df.index) + 1]
        for method in ("ffill", "bfill", "pad", "backfill", "nearest"):
            with pytest.raises(ValueError):
                df.reindex(new_index, method=method)

    @given(df=int_index_dataframe(min_rows=1))
    @settings(max_examples=60)
    def test_explicit__p7__fillvalue_replaces_nan_for_newly_introduced(self, df):
        """fill_value replaces NaN for newly introduced missing labels.

        When fill_value is a compatible scalar, newly introduced rows must have
        every cell equal to fill_value. Pre-existing rows must be unaffected.
        """
        orig_labels = df.index.tolist()
        new_labels = [max(df.index) + 1, max(df.index) + 2]
        new_index = orig_labels + new_labels

        result = df.reindex(new_index, fill_value=0)

        for label in new_labels:
            row = result.loc[label]
            assert (row == 0).all(), (
                f"Newly introduced label {label!r} should be all-0, got {row!r}"
            )

        # Pre-existing rows unaffected (compare on common columns, allowing float promotion)
        for label in orig_labels:
            for col in df.columns:
                orig_val = df.loc[label, col]
                res_val = result.loc[label, col]
                if pd.isna(orig_val):
                    assert pd.isna(res_val)
                else:
                    assert orig_val == res_val

    def test_explicit__p8__limitn_restricts_forwardbackward_fill_to_at(self):
        """limit=N restricts forward/backward fill to at most N consecutive positions.

        When a run of new labels is introduced and method='ffill' with limit=1,
        only the first consecutive new label in each run is filled; the rest
        remain NaN.
        """
        # anchor=10, new labels 11,12,13 — with limit=1 only 11 should be filled
        df = pd.DataFrame({"v": [1.0, 2.0]}, index=[10, 20])
        new_index = [10, 11, 12, 13, 20]
        result = df.reindex(new_index, method="ffill", limit=1)

        assert result.loc[11, "v"] == 1.0, "label 11 (1st gap) should be forward-filled"
        assert pd.isna(result.loc[12, "v"]), "label 12 (2nd gap) should remain NaN with limit=1"
        assert pd.isna(result.loc[13, "v"]), "label 13 (3rd gap) should remain NaN with limit=1"
        assert result.loc[20, "v"] == 2.0, "label 20 (original) should be preserved"

        # limit=2: first two gaps filled, third remains NaN
        result2 = df.reindex(new_index, method="ffill", limit=2)
        assert result2.loc[11, "v"] == 1.0
        assert result2.loc[12, "v"] == 1.0
        assert pd.isna(result2.loc[13, "v"]), "label 13 should remain NaN with limit=2"

    @given(df=int_index_dataframe(min_rows=1))
    @settings(max_examples=40)
    def test_explicit__p9__limitn_without_method_raises_valueerror(self, df):
        """limit=N without method raises ValueError.

        Passing limit without a fill method must raise ValueError.
        """
        new_index = df.index.tolist() + [max(df.index) + 1]
        with pytest.raises(ValueError):
            df.reindex(new_index, limit=1)

    def test_explicit__p10__tolerance_listlike_of_wrong_size_raises(self):
        """tolerance list-like of wrong size raises ValueError.

        When tolerance is a list whose length differs from the new index length,
        a ValueError must be raised.
        """
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=[0, 2, 4])

        # new index has 4 elements; tolerance with 2 elements → wrong size
        with pytest.raises(ValueError):
            df.reindex([0, 1, 2, 3], method="nearest", tolerance=[0.5, 0.5])

        # tolerance with 6 elements → also wrong
        with pytest.raises(ValueError):
            df.reindex([0, 1, 2, 3], method="nearest", tolerance=[0.5] * 6)


# ---------------------------------------------------------------------------
# Indirect properties (P11–P13)
# ---------------------------------------------------------------------------

class TestIndirect:

    def test_indirect__p11__tolerance_scalar_excludes_matches_where_absoriginallabel(self):
        """tolerance scalar excludes matches where abs(original_label - new_label) > tolerance.

        Targets within tolerance get the nearest original value; targets
        beyond tolerance get NaN.
        """
        df = pd.DataFrame({"v": [10.0, 20.0, 30.0]}, index=[0, 10, 20])
        new_index = [1, 5, 9]  # distances from nearest: 1, 5, 1

        # tolerance=2: labels 1 and 9 are within 2 of 0 and 10 respectively; 5 is not
        result = df.reindex(new_index, method="nearest", tolerance=2)
        assert result.loc[1, "v"] == 10.0, "label 1 is within 2 of index 0, should get value 10.0"
        assert pd.isna(result.loc[5, "v"]), "label 5 is >2 away from nearest index, should be NaN"
        assert result.loc[9, "v"] == 20.0, "label 9 is within 2 of index 10, should get value 20.0"

    def test_indirect__p12__methodnearest_selects_the_closer_of_the(self):
        """method='nearest' selects the closer of the two neighboring original values.

        For a new label that is unambiguously closer to one neighbor, the row
        from that neighbor must be returned.
        """
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=[0, 10, 20])

        # label=3 is closer to 0 (dist=3) than to 10 (dist=7)
        result = df.reindex([3], method="nearest")
        assert result.loc[3, "v"] == 1.0, "label 3 is closer to index 0; should get value 1.0"

        # label=17 is closer to 20 (dist=3) than to 10 (dist=7)
        result2 = df.reindex([17], method="nearest")
        assert result2.loc[17, "v"] == 3.0, "label 17 is closer to index 20; should get value 3.0"

    def test_indirect__p13__method_does_not_fill_preexisting_nan(self):
        """method does not fill pre-existing NaN values in the original DataFrame.

        Fill methods only affect newly introduced index positions. Rows that
        existed in the original DataFrame with NaN values must remain NaN in
        the result even when a fill method is applied.
        """
        # row at index 5 has a NaN value; row at 15 is new and should be filled
        df = pd.DataFrame({"v": [1.0, np.nan, 3.0]}, index=[0, 5, 10])
        new_index = [0, 5, 10, 15]

        result_ffill = df.reindex(new_index, method="ffill")
        result_bfill = df.reindex(new_index, method="bfill")

        # Pre-existing NaN at index 5 must remain NaN
        assert pd.isna(result_ffill.loc[5, "v"]), (
            "ffill must not fill pre-existing NaN at original index position 5"
        )
        assert pd.isna(result_bfill.loc[5, "v"]), (
            "bfill must not fill pre-existing NaN at original index position 5"
        )

        # New position 15: ffill should fill with df.loc[10] = 3.0
        assert result_ffill.loc[15, "v"] == 3.0, "new label 15 should be ffill-ed from index 10"


# ---------------------------------------------------------------------------
# Implicit properties (P14–P21)
# ---------------------------------------------------------------------------

class TestImplicit:

    @given(df=int_index_dataframe(min_rows=2))
    @settings(max_examples=60)
    def test_implicit__p14__existing_labels_in_the_new_index(self, df):
        """Existing labels in the new index retain their exact original values.

        Any label present in both the original and the new index must have
        exactly the same row values in the result as in the original DataFrame.
        """
        orig_labels = df.index.tolist()
        # Drop the first label, add two new ones
        new_labels = orig_labels[1:] + [max(orig_labels) + 1, max(orig_labels) + 2]
        result = df.reindex(new_labels)

        kept = set(orig_labels) & set(new_labels)
        for label in kept:
            for col in df.columns:
                orig_val = df.loc[label, col]
                res_val = result.loc[label, col]
                if pd.isna(orig_val):
                    assert pd.isna(res_val), (
                        f"Kept label {label!r} col {col!r}: expected NaN, got {res_val!r}"
                    )
                else:
                    assert orig_val == res_val, (
                        f"Kept label {label!r} col {col!r}: expected {orig_val!r}, got {res_val!r}"
                    )

    @given(df=int_index_dataframe())
    @settings(max_examples=60)
    def test_implicit__p15__result_index_is_exactly_the_requested(self, df):
        """result index is exactly the requested new index, in the same order.

        The result.index must be equal to the requested new index list,
        preserving order and allowing duplicates.
        """
        base = df.index.tolist()
        extra = [max(base, default=0) + 1] if base else [0]
        # Reverse the order to stress ordering
        new_index = (base + extra)[::-1]
        result = df.reindex(new_index)
        assert result.index.tolist() == new_index, (
            f"result.index={result.index.tolist()!r} but requested {new_index!r}"
        )

    @given(df=int_index_dataframe(min_cols=2))
    @settings(max_examples=60)
    def test_implicit__p16__columns_are_unchanged_when_only_the(self, df):
        """Columns are unchanged when only the row index is reindexed.

        Specifying only the index (or labels without axis) must not alter the
        column names or their order.
        """
        new_index = df.index.tolist() + [max(df.index, default=0) + 1]
        result = df.reindex(new_index)
        assert result.columns.tolist() == df.columns.tolist(), (
            "Column names must not change when only the row index is reindexed"
        )

    @given(df=int_index_dataframe())
    @settings(max_examples=60)
    def test_implicit__p17__reindexing_with_the_same_index_produces(self, df):
        """Reindexing with the same index produces a DataFrame equal to the original.

        Using the identical index (same labels, same order) must yield a result
        that equals the original DataFrame and is a distinct object.
        """
        result = df.reindex(df.index)
        assert result.equals(df), "reindex with same index must produce an equal DataFrame"
        assert result is not df, "result must be a distinct object from the original"

    @given(
        df=st.builds(
            lambda n: pd.DataFrame(
                {f"c{i}": np.arange(n, dtype=np.int64) for i in range(2)},
                index=list(range(n)),
            ),
            n=st.integers(1, 8),
        )
    )
    @settings(max_examples=40)
    def test_implicit__p18__integerdtype_columns_are_promoted_to_float64(self, df):
        """Integer-dtype columns are promoted to float64 when NaN is introduced.

        Introducing at least one new row label into an integer-dtype DataFrame
        forces dtype promotion to float64 (NaN cannot be stored in integer arrays).
        """
        assert all(df[c].dtype == np.int64 for c in df.columns), "Pre-condition: all columns must be int64"
        new_index = df.index.tolist() + [max(df.index) + 1]
        result = df.reindex(new_index)
        for col in result.columns:
            assert result[col].dtype == np.float64, (
                f"Column {col!r} should be promoted to float64, got {result[col].dtype}"
            )

    @given(
        df=int_index_dataframe(),
        extra_count=st.integers(0, 4),
    )
    @settings(max_examples=60)
    def test_implicit__p19__result_row_count_equals_the_length(self, df, extra_count):
        """Result row count equals the length of the requested new index.

        len(result) must equal len(new_index) for any new_index, including
        empty DataFrames and empty new indexes.
        """
        base = df.index.tolist()
        start = max(base, default=0) + 1
        new_index = base + list(range(start, start + extra_count))
        result = df.reindex(new_index)
        assert len(result) == len(new_index), (
            f"len(result)={len(result)} but len(new_index)={len(new_index)}"
        )

    @given(df=int_index_dataframe())
    @settings(max_examples=60)
    def test_implicit__p20__return_type_is_always_pddataframe(self, df):
        """Return type is always pd.DataFrame.

        Regardless of DataFrame shape, dtypes, or index type, reindex must
        always return an instance of pd.DataFrame.
        """
        new_index = df.index.tolist() + [max(df.index, default=0) + 1]
        result = df.reindex(new_index)
        assert isinstance(result, pd.DataFrame)

        # Also check with columns reindexing
        cols = df.columns.tolist()
        new_cols = cols + ["new_col"]
        result2 = df.reindex(columns=new_cols)
        assert isinstance(result2, pd.DataFrame)

    @given(df=int_index_dataframe(min_cols=1))
    @settings(max_examples=60)
    def test_implicit__p21__column_reindexing_produces_nan_for_new(self, df):
        """Column reindexing produces NaN for new column labels and preserves existing.

        When reindex is called with columns=, existing column data must be
        preserved and columns not in the original must be all-NaN.
        The axis='columns' calling convention must produce the same result.
        """
        existing_cols = df.columns.tolist()
        new_cols = existing_cols + ["_new_col_A", "_new_col_B"]

        result_kw = df.reindex(columns=new_cols)
        result_axis = df.reindex(labels=new_cols, axis="columns")

        # Existing columns preserved
        for col in existing_cols:
            assert result_kw[col].equals(df[col]), (
                f"Existing column {col!r} must be unchanged after column reindex"
            )

        # New columns are all NaN
        for col in ["_new_col_A", "_new_col_B"]:
            assert result_kw[col].isna().all(), (
                f"New column {col!r} must be all-NaN after column reindex"
            )

        # Both calling conventions agree
        assert result_kw.equals(result_axis), (
            "columns= and labels+axis='columns' must produce identical results"
        )
