"""
Property-based tests for pd.DataFrame.ewm().aggregate()
Generated from: ir/pandas/ewm_aggregate_properties.md
Library: pandas 3.0.0
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes, series


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

VALID_FUNCS = ["mean", "std", "var", "sum"]


def numeric_dataframe(min_rows=2, max_rows=20, min_cols=1, max_cols=4):
    """DataFrame with float columns, controllable row/col count."""
    return data_frames(
        columns=[
            column(name=f"col_{i}", dtype=float)
            for i in range(max_cols)
        ],
        index=range_indexes(min_size=min_rows, max_size=max_rows),
    ).filter(lambda df: len(df.columns) >= min_cols)


def two_col_dataframe():
    """DataFrame with exactly columns 'A' and 'B', float dtype, 2+ rows."""
    return data_frames(
        columns=[column("A", dtype=float), column("B", dtype=float)],
        index=range_indexes(min_size=2, max_size=20),
    )


def float_series(min_size=1, max_size=20):
    """Series of floats."""
    return series(dtype=float, index=range_indexes(min_size=min_size, max_size=max_size))


valid_alpha = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)
valid_func_str = st.sampled_from(VALID_FUNCS)


# ---------------------------------------------------------------------------
# Explicit properties
# ---------------------------------------------------------------------------


class TestExplicit:

    @given(df=numeric_dataframe(), func=valid_func_str, alpha=valid_alpha)
    @settings(max_examples=50)
    def test_explicit__p1__agg_is_an_alias_for_aggregate(self, df, func, alpha):
        """
        agg is an alias for aggregate and produces identical results.
        Source: docstring Notes: 'agg is an alias for aggregate. Use the alias.'
        """
        ewm_agg = df.ewm(alpha=alpha).agg(func)
        ewm_aggregate = df.ewm(alpha=alpha).aggregate(func)
        assert ewm_agg.equals(ewm_aggregate)

    @given(df=numeric_dataframe(), alpha=valid_alpha)
    @settings(max_examples=20)
    def test_explicit__p1__agg_alias_with_list_func(self, df, alpha):
        """
        agg alias equivalence also holds for list func input.
        """
        func = ["mean", "std"]
        ewm_agg = df.ewm(alpha=alpha).agg(func)
        ewm_aggregate = df.ewm(alpha=alpha).aggregate(func)
        assert ewm_agg.equals(ewm_aggregate)

    @given(df=numeric_dataframe(min_rows=1), alpha=valid_alpha)
    @settings(max_examples=20)
    def test_explicit__p2__funcnone_raises_typeerror(self, df, alpha):
        """
        func=None raises TypeError: 'Must provide func or tuples of (column, aggfunc)'.
        Source: empirical observation.
        """
        with pytest.raises(TypeError):
            df.ewm(alpha=alpha).aggregate(None)

    @given(df=numeric_dataframe(), alpha=valid_alpha)
    @settings(max_examples=30)
    def test_explicit__p3__list_of_string_func_names_returns(self, df, alpha):
        """
        List of string func names returns a DataFrame for DataFrame input.
        Source: docstring Returns: 'DataFrame: when DataFrame.agg is called with several functions'.
        """
        result = df.ewm(alpha=alpha).aggregate(["mean", "std"])
        assert isinstance(result, pd.DataFrame)
        # MultiIndex columns: (col_name, func_name)
        assert isinstance(result.columns, pd.MultiIndex)
        assert result.shape[0] == df.shape[0]

    @given(s=float_series(min_size=2), alpha=valid_alpha)
    @settings(max_examples=30)
    def test_explicit__p3__list_func_returns_dataframe_for_series(self, s, alpha):
        """
        List of string func names returns a DataFrame for Series input.
        Source: docstring Returns.
        """
        result = s.ewm(alpha=alpha).aggregate(["mean", "std"])
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(s)
        # func names are columns
        assert set(result.columns) == {"mean", "std"}

    @given(df=two_col_dataframe(), alpha=valid_alpha)
    @settings(max_examples=30)
    def test_explicit__p6__dict_func_applies_different_functions_to(self, df, alpha):
        """
        dict func applies different functions to different columns;
        result contains only the dict-specified columns.
        Source: docstring: 'dict of axis labels -> functions, function names or list of such'.
        """
        func_dict = {"A": "mean", "B": "std"}
        result = df.ewm(alpha=alpha).aggregate(func_dict)
        assert set(result.columns) == {"A", "B"}

    @given(df=two_col_dataframe(), alpha=valid_alpha)
    @settings(max_examples=30)
    def test_explicit__p7__dict_with_a_column_mapped_to(self, df, alpha):
        """
        dict with a column mapped to a list of funcs produces MultiIndex columns for that column.
        Source: docstring: 'dict of axis labels -> functions, function names or list of such'.
        """
        func_dict = {"A": ["mean", "std"], "B": "mean"}
        result = df.ewm(alpha=alpha).aggregate(func_dict)
        assert isinstance(result.columns, pd.MultiIndex)
        col_tuples = set(result.columns.tolist())
        assert ("A", "mean") in col_tuples
        assert ("A", "std") in col_tuples
        assert ("B", "mean") in col_tuples

    @given(df=numeric_dataframe(), alpha=valid_alpha)
    @settings(max_examples=20)
    def test_explicit__p8__dict_with_a_nonexistent_column_key(self, df, alpha):
        """
        dict with a non-existent column key raises KeyError.
        Source: empirical: raises KeyError for missing column labels.
        """
        with pytest.raises(KeyError):
            df.ewm(alpha=alpha).aggregate({"NONEXISTENT_COL": "mean"})


# ---------------------------------------------------------------------------
# Indirect properties
# ---------------------------------------------------------------------------


class TestIndirect:

    @given(df=numeric_dataframe(), alpha=valid_alpha, func=valid_func_str)
    @settings(max_examples=50)
    def test_indirect__p4__dataframe_with_single_string_func_returns(self, df, alpha, func):
        """
        DataFrame with single string func returns a DataFrame with the same shape and columns
        as the input — NOT a Series as the docstring incorrectly states.
        Source: E3: docstring incorrectly says 'Series'; empirically returns DataFrame of same shape.
        """
        result = df.ewm(alpha=alpha).aggregate(func)
        assert isinstance(result, pd.DataFrame), (
            f"Expected DataFrame, got {type(result)}"
        )
        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)

    @given(s=float_series(min_size=1), alpha=valid_alpha, func=valid_func_str)
    @settings(max_examples=50)
    def test_indirect__p5__series_with_single_string_func_returns(self, s, alpha, func):
        """
        Series with single string func returns a Series of the same length —
        NOT a scalar as the docstring incorrectly states.
        Source: E1: docstring incorrectly says 'scalar'; empirically returns Series of same length.
        """
        result = s.ewm(alpha=alpha).aggregate(func)
        assert isinstance(result, pd.Series), (
            f"Expected Series, got {type(result)}"
        )
        assert len(result) == len(s)


# ---------------------------------------------------------------------------
# Implicit properties
# ---------------------------------------------------------------------------


class TestImplicit:

    @given(df=numeric_dataframe(), alpha=valid_alpha)
    @settings(max_examples=50)
    def test_implicit__p9__the_original_dataframe_is_not_mutated(self, df, alpha):
        """
        The original DataFrame is not mutated by aggregate.
        """
        snapshot = df.copy()
        df.ewm(alpha=alpha).aggregate("mean")
        assert df.equals(snapshot)

    @given(df=numeric_dataframe(), alpha=valid_alpha, func=valid_func_str)
    @settings(max_examples=30)
    def test_implicit__p9__not_mutated_with_list_func(self, df, alpha, func):
        """
        The original DataFrame is not mutated when using list func.
        """
        snapshot = df.copy()
        df.ewm(alpha=alpha).aggregate(["mean", "std"])
        assert df.equals(snapshot)

    @given(df=numeric_dataframe(), alpha=valid_alpha, func=valid_func_str)
    @settings(max_examples=50)
    def test_implicit__p10__aggregate_is_deterministic_identical_calls_return(self, df, alpha, func):
        """
        aggregate is deterministic: identical calls return equal results.
        """
        r1 = df.ewm(alpha=alpha).aggregate(func)
        r2 = df.ewm(alpha=alpha).aggregate(func)
        assert r1.equals(r2)

    def test_implicit__p11__any_callable_func_raises_attributeerror_ewm(self):
        """
        Any callable func raises AttributeError — EWM does not support apply-based aggregation.
        Source: E2: AttributeError: 'ExponentialMovingWindow' object has no attribute 'apply'.
        """
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})

        def custom_func(x):
            return x.sum()

        callables = [
            lambda x: x.mean(),
            np.mean,
            np.sum,
            pd.Series.mean,
            custom_func,
        ]
        for func in callables:
            with pytest.raises(AttributeError):
                df.ewm(alpha=0.5).aggregate(func)

    @given(df=numeric_dataframe(), alpha=valid_alpha, func=valid_func_str)
    @settings(max_examples=30)
    def test_implicit__p12__result_index_is_identical_to_the(self, df, alpha, func):
        """
        Result index is identical to the original DataFrame/Series index.
        Source: notes/pandas.md: Index preservation — most operations preserve the original index.
        """
        result = df.ewm(alpha=alpha).aggregate(func)
        assert result.index.equals(df.index)

    def test_implicit__p12__result_index_preserved_for_string_index(self):
        """Result index is identical when input has a string index."""
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
        result = df.ewm(alpha=0.5).aggregate("mean")
        assert result.index.equals(df.index)

    def test_implicit__p12__result_index_preserved_for_datetime_index(self):
        """Result index is identical when input has a DatetimeIndex."""
        idx = pd.date_range("2021-01-01", periods=5, freq="D")
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        result = df.ewm(alpha=0.5).aggregate("mean")
        assert result.index.equals(df.index)

    @given(df=numeric_dataframe(), alpha=valid_alpha, func=valid_func_str)
    @settings(max_examples=30)
    def test_implicit__p13__result_is_always_a_new_object(self, df, alpha, func):
        """
        Result is always a new object, not the original DataFrame.
        Source: notes/pandas.md: Identity — pandas operations consistently return a new object.
        """
        result = df.ewm(alpha=alpha).aggregate(func)
        assert result is not df

    def test_implicit__p14__empty_dataframe_input_returns_an_empty(self):
        """
        Empty DataFrame input returns an empty DataFrame with the same columns.
        """
        for n_cols in range(1, 5):
            cols = {f"col_{i}": pd.Series([], dtype=float) for i in range(n_cols)}
            df = pd.DataFrame(cols)
            for func in ["mean", "std", "var"]:
                result = df.ewm(alpha=0.5).aggregate(func)
                assert isinstance(result, pd.DataFrame)
                assert result.shape == (0, n_cols)
                assert list(result.columns) == list(df.columns)

    @given(n=st.integers(min_value=2, max_value=10), alpha=valid_alpha)
    @settings(max_examples=30)
    def test_implicit__p15__nanonly_column_produces_allnan_output_column(self, n, alpha):
        """
        NaN-only column produces all-NaN output column — NaN is not silently replaced or dropped.
        """
        df = pd.DataFrame({
            "nan_col": [np.nan] * n,
            "normal_col": list(range(n)),
        }, dtype=float)
        result = df.ewm(alpha=alpha).aggregate("mean")
        assert result["nan_col"].isna().all(), (
            "Expected all-NaN output for NaN-only input column"
        )

    @given(
        data=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=3, max_size=15),
        alpha=valid_alpha,
        min_periods=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=40)
    def test_implicit__p16__minperiodsn_causes_the_first_n1_output(self, data, alpha, min_periods):
        """
        min_periods=N causes the first N-1 output positions to be NaN
        regardless of EWM decay parameters.
        Source: ewm() docstring: 'min_periods: Minimum number of observations in window required.'
        """
        assume(len(data) >= min_periods)
        df = pd.DataFrame({"A": data})
        result = df.ewm(alpha=alpha, min_periods=min_periods).aggregate("mean")
        # First min_periods-1 rows must be NaN
        if min_periods >= 2:
            assert result.iloc[: min_periods - 1].isna().all().all(), (
                f"Expected first {min_periods - 1} rows to be NaN"
            )
        # Row at min_periods-1 (0-indexed) should be non-NaN (enough obs)
        assert not result.iloc[min_periods - 1].isna().any()

    @given(alpha=valid_alpha)
    @settings(max_examples=20)
    def test_implicit__p16__min_periods_exceeds_rows_all_nan(self, alpha):
        """
        When min_periods > n_rows, all result rows are NaN.
        """
        df = pd.DataFrame({"A": [1.0, 2.0]})
        result = df.ewm(alpha=alpha, min_periods=5).aggregate("mean")
        assert result.isna().all().all()

    @given(alpha=valid_alpha, value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False))
    @settings(max_examples=30)
    def test_implicit__p17__std_aggregation_of_a_single_observation(self, alpha, value):
        """
        std aggregation of a single observation returns NaN — insufficient data for std.
        Source: notes/pandas.md: std of single obs is NaN — consistent with N-1 degrees of freedom.
        """
        df = pd.DataFrame({"A": [value]})
        result = df.ewm(alpha=alpha).aggregate("std")
        assert result.isna().all().all(), (
            f"Expected NaN for std of single row, got {result}"
        )

    def test_implicit__p19__invalid_string_func_name_nonewm_method(self):
        """
        Invalid string func name (non-EWM method) raises AttributeError.
        Source: notes/pandas.md: Valid string names (EWM-native only): mean, std, var, sum.
        """
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        invalid_names = ["min", "max", "median", "count", "kurt", "skew", "sem", "first", "last", "foobar"]
        for name in invalid_names:
            with pytest.raises(AttributeError):
                df.ewm(alpha=0.5).aggregate(name)

    @given(
        df=data_frames(
            columns=[column("A", dtype=float), column("B", dtype=float)],
            index=range_indexes(min_size=3, max_size=20),
        ).filter(lambda d: not d.isnull().any().any()),
        alpha=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
    )
    @settings(max_examples=40)
    def test_implicit__p20__adjustfalse_produces_numerically_different_results_from(self, df, alpha):
        """
        adjust=False produces numerically different results from adjust=True for the same input.
        Source: ewm() docstring: adjust=True uses weighted sum formula; adjust=False uses recursive.
        """
        result_true = df.ewm(alpha=alpha, adjust=True).agg("mean")
        result_false = df.ewm(alpha=alpha, adjust=False).agg("mean")
        # They may agree on the first row but should differ overall for n >= 2
        # Check at least one cell differs (using values to handle NaN-safe comparison)
        assert not np.allclose(result_true.values, result_false.values, equal_nan=True), (
            "adjust=True and adjust=False produced identical results — unexpected"
        )


# ---------------------------------------------------------------------------
# Convention properties
# ---------------------------------------------------------------------------


class TestConvention:

    @given(
        df=data_frames(
            columns=[column("A", dtype=float), column("B", dtype=float)],
            index=range_indexes(min_size=2, max_size=20),
        ).filter(lambda d: not d.isnull().any().any()),
        c=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=40)
    def test_convention__p18__ewm_parameters_com_span_halflife_alpha(self, df, c):
        """
        EWM parameters com and alpha are mathematically equivalent:
        alpha = 1 / (1 + com).
        Source: notes/pandas.md: Decay parameter equivalence — com=c ↔ alpha=1/(1+c).
        """
        alpha = 1.0 / (1.0 + c)
        result_com = df.ewm(com=c).agg("mean")
        result_alpha = df.ewm(alpha=alpha).agg("mean")
        assert np.allclose(result_com.values, result_alpha.values, equal_nan=True), (
            f"com={c} and alpha={alpha:.6f} produced different results"
        )

    @given(
        df=data_frames(
            columns=[column("A", dtype=float)],
            index=range_indexes(min_size=2, max_size=20),
        ).filter(lambda d: not d.isnull().any().any()),
        span=st.floats(min_value=2.0, max_value=20.0, allow_nan=False),
    )
    @settings(max_examples=40)
    def test_convention__p18__span_alpha_equivalence(self, df, span):
        """
        EWM parameters span and alpha are mathematically equivalent:
        alpha = 2 / (span + 1).
        Source: notes/pandas.md: Decay parameter equivalence — span=s ↔ alpha=2/(s+1).
        """
        alpha = 2.0 / (span + 1.0)
        result_span = df.ewm(span=span).agg("mean")
        result_alpha = df.ewm(alpha=alpha).agg("mean")
        assert np.allclose(result_span.values, result_alpha.values, equal_nan=True), (
            f"span={span} and alpha={alpha:.6f} produced different results"
        )
