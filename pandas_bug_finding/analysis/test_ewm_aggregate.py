"""
Property-based tests for pd.DataFrame.ewm().aggregate / .agg
Generated from: ir/pandas/ewm_aggregate_properties.md
pandas version: 3.0.0

Run with:
    uv run pytest pandas_bug_finding/ir_test/test_ewm_aggregate.py -v
    uv run pytest pandas_bug_finding/ir_test/test_ewm_aggregate.py -v --hypothesis-seed=0

Documentation errors noted in the IR (treat test failures with caution):
  E1 [SEVERE] - docstring claims Series.agg(single_func) returns a scalar; empirically returns Series
  E2 [SEVERE] - docstring implies callables are accepted; empirically raises AttributeError
  E3 [SEVERE] - docstring claims DataFrame.agg(single_func) returns a Series; empirically returns DataFrame
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Valid EWM string function names (only these work in pandas 3.0.0)
# ---------------------------------------------------------------------------
VALID_FUNCS = ["mean", "std", "var", "sum"]

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

float_val = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)


@st.composite
def float_dataframe(draw, min_rows=1, max_rows=8, min_cols=1, max_cols=4, allow_nan=False):
    """DataFrame with float64 columns and a RangeIndex."""
    n_rows = draw(st.integers(min_rows, max_rows))
    n_cols = draw(st.integers(min_cols, max_cols))
    scalar = float_val if not allow_nan else st.one_of(float_val, st.just(np.nan))
    data = {
        f"c{i}": draw(st.lists(scalar, min_size=n_rows, max_size=n_rows))
        for i in range(n_cols)
    }
    return pd.DataFrame(data, dtype=float)


@st.composite
def float_series(draw, min_len=1, max_len=8):
    """Series with float64 values."""
    n = draw(st.integers(min_len, max_len))
    vals = draw(st.lists(float_val, min_size=n, max_size=n))
    return pd.Series(vals, dtype=float)


@st.composite
def valid_alpha(draw):
    """alpha in (0, 1] — the simplest single decay parameter."""
    return draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))


@st.composite
def valid_com(draw):
    """com >= 0 for decay parameter equivalence tests."""
    return draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))


# ---------------------------------------------------------------------------
# Explicit properties (P1, P2, P3, P6, P7, P8)
# ---------------------------------------------------------------------------

class TestExplicit:

    @given(df=float_dataframe(), func=st.sampled_from(VALID_FUNCS), alpha=valid_alpha())
    @settings(max_examples=60)
    def test_explicit__p1__agg_is_an_alias_for_aggregate(self, df, func, alpha):
        """`agg` is an alias for `aggregate` and produces identical results.

        For every valid string func, df.ewm(alpha=a).agg(func) must equal
        df.ewm(alpha=a).aggregate(func) element-by-element.
        """
        r_agg = df.ewm(alpha=alpha).agg(func)
        r_aggregate = df.ewm(alpha=alpha).aggregate(func)
        assert r_agg.equals(r_aggregate), (
            f"agg({func!r}) and aggregate({func!r}) differ with alpha={alpha}"
        )

    @given(df=float_dataframe())
    @settings(max_examples=30)
    def test_explicit__p2__funcnone_raises_typeerror(self, df):
        """func=None raises TypeError.

        aggregate(None) must raise TypeError with a message mentioning 'func'.
        """
        with pytest.raises(TypeError, match="(?i)func"):
            df.ewm(alpha=0.5).aggregate(None)

    @given(df=float_dataframe(min_rows=1, min_cols=1))
    @settings(max_examples=50)
    def test_explicit__p3__list_of_string_func_names_returns(self, df):
        """List of string func names returns a DataFrame for DataFrame input.

        When func is a list of 2+ valid string names, the result must be a
        DataFrame. For DataFrame input the columns should be a MultiIndex of
        (col_name, func_name) tuples.
        """
        func_list = ["mean", "std"]
        result = df.ewm(alpha=0.5).aggregate(func_list)
        assert isinstance(result, pd.DataFrame), (
            f"Expected DataFrame, got {type(result)}"
        )
        assert isinstance(result.columns, pd.MultiIndex), (
            "List func on DataFrame should produce MultiIndex columns"
        )
        # Each original column should appear with each func name
        top_level = set(result.columns.get_level_values(0))
        assert top_level == set(df.columns), (
            f"Top-level columns {top_level} should match df.columns {set(df.columns)}"
        )
        second_level = set(result.columns.get_level_values(1))
        assert second_level == set(func_list), (
            f"Second-level columns {second_level} should be func names {set(func_list)}"
        )

    @given(df=float_dataframe(min_cols=2))
    @settings(max_examples=50)
    def test_explicit__p6__dict_func_applies_different_functions_to(self, df):
        """dict func applies different functions to different columns.

        When func is a dict mapping column names to single string func names,
        the result columns must be exactly the dict keys (no extras from
        unkeyed columns).
        """
        cols = df.columns.tolist()
        # Map first column to 'mean', second to 'std'
        func_dict = {cols[0]: "mean", cols[1]: "std"}
        result = df.ewm(alpha=0.5).aggregate(func_dict)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == set(func_dict.keys()), (
            f"result.columns={set(result.columns)!r} should equal dict keys {set(func_dict.keys())!r}"
        )

    def test_explicit__p7__dict_with_column_mapped_to_a(self):
        """dict with a column mapped to a list of funcs produces MultiIndex columns.

        When any dict value is a list, the result columns must be a MultiIndex.
        The columns for the multi-func entry should be (col, func) pairs; the
        single-func entry gets (col, func) too.
        """
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        result = df.ewm(alpha=0.5).aggregate({"A": ["mean", "std"], "B": "mean"})

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.columns, pd.MultiIndex), (
            "dict with a list value should produce MultiIndex columns"
        )
        col_tuples = set(result.columns.tolist())
        assert ("A", "mean") in col_tuples, "('A','mean') should be in result columns"
        assert ("A", "std") in col_tuples, "('A','std') should be in result columns"
        assert ("B", "mean") in col_tuples, "('B','mean') should be in result columns"
        # 'B' should only have 'mean', not 'std'
        assert ("B", "std") not in col_tuples, "('B','std') should NOT be in result columns"

    @given(df=float_dataframe(min_cols=1))
    @settings(max_examples=30)
    def test_explicit__p8__dict_with_a_nonexistent_column_key(self, df):
        """dict with a non-existent column key raises KeyError.

        Passing a dict key that is not in df.columns must raise KeyError.
        """
        bad_key = "NONEXISTENT_COL_XYZ"
        assume(bad_key not in df.columns)
        with pytest.raises(KeyError):
            df.ewm(alpha=0.5).aggregate({bad_key: "mean"})


# ---------------------------------------------------------------------------
# Indirect properties (P4, P5)
# ---------------------------------------------------------------------------

class TestIndirect:

    @given(df=float_dataframe(min_rows=1, min_cols=1), func=st.sampled_from(VALID_FUNCS))
    @settings(max_examples=60)
    def test_indirect__p4__dataframe_with_single_string_func_returns(self, df, func):
        """DataFrame + single string func returns a DataFrame of the same shape.

        The docstring incorrectly states the result is a Series; empirically
        ewm.aggregate preserves the DataFrame shape. This test pins the actual
        behavior: result must be a DataFrame with result.shape == df.shape and
        identical column names.
        """
        result = df.ewm(alpha=0.5).aggregate(func)
        assert isinstance(result, pd.DataFrame), (
            f"Expected DataFrame, got {type(result)} (docstring error E3 — actual behavior)"
        )
        assert result.shape == df.shape, (
            f"Shape mismatch: result.shape={result.shape}, df.shape={df.shape}"
        )
        assert result.columns.tolist() == df.columns.tolist(), (
            "Columns must be identical to input DataFrame columns"
        )

    @given(s=float_series(min_len=1), func=st.sampled_from(VALID_FUNCS))
    @settings(max_examples=60)
    def test_indirect__p5__series_with_single_string_func_returns(self, s, func):
        """Series + single string func returns a Series of the same length.

        The docstring incorrectly states the result is a scalar; empirically
        ewm.aggregate on a Series returns a Series with the same length. This
        test pins the actual (correct) behavior.
        """
        result = s.ewm(alpha=0.5).aggregate(func)
        assert isinstance(result, pd.Series), (
            f"Expected Series, got {type(result)} (docstring error E1 — actual behavior)"
        )
        assert len(result) == len(s), (
            f"Length mismatch: len(result)={len(result)}, len(s)={len(s)}"
        )


# ---------------------------------------------------------------------------
# Implicit properties (P9–P17)
# ---------------------------------------------------------------------------

class TestImplicit:

    @given(df=float_dataframe(min_rows=1))
    @settings(max_examples=60)
    def test_implicit__p9__the_original_dataframe_is_not_mutated(self, df):
        """The original DataFrame is not mutated by aggregate.

        After calling .ewm().aggregate(), the original DataFrame must be
        identical to its pre-call snapshot.
        """
        snapshot = df.copy()
        df.ewm(alpha=0.5).aggregate("mean")
        assert df.equals(snapshot), "aggregate must not mutate the original DataFrame"

        df.ewm(alpha=0.5).aggregate(["mean", "std"])
        assert df.equals(snapshot), "aggregate with list must not mutate the original DataFrame"

        if len(df.columns) >= 1:
            col = df.columns[0]
            df.ewm(alpha=0.5).aggregate({col: "mean"})
            assert df.equals(snapshot), "aggregate with dict must not mutate the original DataFrame"

    @given(df=float_dataframe(min_rows=1), alpha=valid_alpha())
    @settings(max_examples=60)
    def test_implicit__p10__aggregate_is_deterministic_identical_calls_return(self, df, alpha):
        """aggregate is deterministic: identical calls return equal results.

        Two consecutive calls with the same DataFrame and EWM config must
        produce results that are equal via .equals().
        """
        r1 = df.ewm(alpha=alpha).aggregate("mean")
        r2 = df.ewm(alpha=alpha).aggregate("mean")
        assert r1.equals(r2), (
            f"Two calls to aggregate('mean') with alpha={alpha} returned different results"
        )

    def test_implicit__p11__any_callable_func_raises_attributeerror_ewm(self):
        """Any callable func raises AttributeError (EWM does not support apply).

        The docstring implies callables are supported, but every callable raises
        AttributeError: 'ExponentialMovingWindow' object has no attribute 'apply'.
        This test pins the actual broken behavior (docstring error E2).
        """
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        callables = [
            lambda x: x.mean(),
            np.mean,
            np.sum,
            pd.Series.mean,
        ]
        for func in callables:
            with pytest.raises(AttributeError, match="apply"):
                df.ewm(alpha=0.5).aggregate(func)

    @given(df=float_dataframe(min_rows=1))
    @settings(max_examples=60)
    def test_implicit__p12__result_index_is_identical_to_the(self, df):
        """Result index is identical to the original DataFrame index.

        After aggregation with any valid string func, result.index must equal
        df.index for all index types.
        """
        result = df.ewm(alpha=0.5).aggregate("mean")
        assert result.index.equals(df.index), (
            f"result.index={result.index.tolist()!r} differs from df.index={df.index.tolist()!r}"
        )

    @given(df=float_dataframe(min_rows=1))
    @settings(max_examples=40)
    def test_implicit__p13__result_is_always_a_new_object(self, df):
        """Result is always a new object, not the original DataFrame.

        aggregate must return a distinct Python object (identity check).
        """
        result = df.ewm(alpha=0.5).aggregate("mean")
        assert result is not df, "aggregate must return a new object, not the original DataFrame"

    def test_implicit__p14__empty_dataframe_input_returns_an_empty(self):
        """Empty DataFrame input returns an empty DataFrame with the same columns.

        A DataFrame with 0 rows but 1+ columns must produce an empty result
        DataFrame with identical column names.
        """
        for n_cols in range(1, 5):
            cols = [f"c{i}" for i in range(n_cols)]
            df = pd.DataFrame({c: pd.Series([], dtype=float) for c in cols})
            for func in ("mean", "std", "var"):
                result = df.ewm(alpha=0.5).aggregate(func)
                assert isinstance(result, pd.DataFrame), (
                    f"Empty DataFrame + func={func!r} should return DataFrame, got {type(result)}"
                )
                assert result.shape[0] == 0, (
                    f"Empty DataFrame should produce 0 rows, got {result.shape[0]}"
                )
                assert result.columns.tolist() == cols, (
                    f"Columns {result.columns.tolist()!r} must match {cols!r}"
                )

    @given(n=st.integers(1, 8))
    @settings(max_examples=30)
    def test_implicit__p15__nanonly_column_produces_allnan_output_column(self, n):
        """NaN-only column produces all-NaN output column.

        A column that is entirely NaN must produce an entirely NaN result
        column. No ValueError should be raised and the column must not be
        silently dropped.
        """
        df = pd.DataFrame({"nan_col": [np.nan] * n, "ok_col": [1.0] * n})
        result = df.ewm(alpha=0.5).aggregate("mean")
        assert "nan_col" in result.columns, "NaN-only column must not be dropped from result"
        assert result["nan_col"].isna().all(), (
            "NaN-only column should produce all-NaN output"
        )

    @given(
        df=float_dataframe(min_rows=3, max_rows=8, allow_nan=False),
        min_periods=st.integers(2, 4),
    )
    @settings(max_examples=40)
    def test_implicit__p16__minperiodsn_causes_the_first_n1_output(self, df, min_periods):
        """min_periods=N causes the first N-1 output positions to be NaN.

        With min_periods >= 2, the first (min_periods - 1) rows of the result
        must be NaN regardless of EWM parameters.
        """
        assume(len(df) >= min_periods)
        result = df.ewm(alpha=0.5, min_periods=min_periods).aggregate("mean")

        for i in range(min_periods - 1):
            assert result.iloc[i].isna().all(), (
                f"Row {i} should be NaN with min_periods={min_periods}, got {result.iloc[i].tolist()}"
            )

        # Row at min_periods-1 (0-indexed) should have non-NaN values
        if len(df) >= min_periods:
            row = result.iloc[min_periods - 1]
            assert row.notna().all(), (
                f"Row {min_periods - 1} should be non-NaN with min_periods={min_periods}"
            )

    def test_implicit__p17__std_aggregation_of_a_single_observation(self):
        """std aggregation of a single observation returns NaN.

        With exactly 1 row, ewm.aggregate('std') must return NaN for every
        cell — standard deviation requires at least 2 observations. This is
        inherent to std and is not overridden by min_periods=0.
        """
        df = pd.DataFrame({"A": [5.0], "B": [-3.0]})
        result = df.ewm(alpha=0.5).aggregate("std")
        assert isinstance(result, pd.DataFrame)
        assert result.isna().all().all(), (
            f"std of single-row DataFrame must be all-NaN, got {result}"
        )

        # min_periods=0 should not change this — NaN is inherent to std
        result_mp0 = df.ewm(alpha=0.5, min_periods=0).aggregate("std")
        assert result_mp0.isna().all().all(), (
            "std of single-row DataFrame must be all-NaN even with min_periods=0"
        )


# ---------------------------------------------------------------------------
# Convention properties (P18)
# ---------------------------------------------------------------------------

class TestConvention:

    @given(
        df=float_dataframe(min_rows=2, allow_nan=False),
        com=valid_com(),
    )
    @settings(max_examples=50)
    def test_convention__p18__ewm_parameters_com_span_halflife_alpha(self, df, com):
        """EWM parameters com, span, and alpha produce mathematically equivalent results.

        The relationships defined in the ewm docstring:
          alpha = 1 / (1 + com)   for com >= 0
          alpha = 2 / (span + 1)  for span >= 1

        must produce numerically equivalent results when used with agg('mean').
        Floating-point tolerance is applied via np.allclose.
        """
        # com ↔ alpha equivalence
        alpha_from_com = 1.0 / (1.0 + com)
        r_com = df.ewm(com=com).agg("mean")
        r_alpha = df.ewm(alpha=alpha_from_com).agg("mean")

        assert np.allclose(r_com.values, r_alpha.values, rtol=1e-9, atol=1e-12, equal_nan=True), (
            f"com={com} and alpha={alpha_from_com} (1/(1+com)) should produce identical results"
        )

        # span ↔ alpha equivalence (span must be >= 1)
        span = draw_span_from_com(com)
        if span >= 1.0:
            alpha_from_span = 2.0 / (span + 1.0)
            r_span = df.ewm(span=span).agg("mean")
            r_alpha2 = df.ewm(alpha=alpha_from_span).agg("mean")
            assert np.allclose(
                r_span.values, r_alpha2.values, rtol=1e-9, atol=1e-12, equal_nan=True
            ), (
                f"span={span} and alpha={alpha_from_span} (2/(span+1)) should produce identical results"
            )


def draw_span_from_com(com: float) -> float:
    """Derive a valid span >= 1 from a com value.  span = (2 - alpha) / alpha = 2*(1+com) - 1."""
    alpha = 1.0 / (1.0 + com)
    # span = 2/alpha - 1
    return 2.0 / alpha - 1.0
