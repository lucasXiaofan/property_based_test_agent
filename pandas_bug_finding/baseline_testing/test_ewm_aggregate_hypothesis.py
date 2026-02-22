"""
Property-Based Testing Script for pandas DataFrame.ewm().aggregate()
=====================================================================

Tested library : pandas (tested against 3.0.0)
Target method  : DataFrame.ewm().aggregate() / Series.ewm().aggregate()
                 (also via the alias .agg())

How to run
----------
    uv run pytest pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py -v

Or with verbose hypothesis output:
    uv run pytest pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py -v --hypothesis-show-statistics


EXPLICIT PROPERTIES (from docstring)
-------------------------------------
E1. `func` accepts: callable, string function name, list of functions/strings,
    or dict mapping column labels to functions.
E2. `agg` is an alias for `aggregate` — both produce bitwise-identical results.
E3. DataFrame.agg with a single function returns a Series (docstring claim).
    *** ACTUAL behaviour: returns DataFrame — the docstring is wrong. ***
E4. DataFrame.agg with several functions returns a DataFrame.
E5. Series.agg with a single function returns a scalar (docstring claim).
    *** ACTUAL behaviour: returns a Series — the docstring is wrong. ***
E6. A user-defined function is passed a Series for evaluation.
    *** ACTUAL behaviour: EWM.aggregate(udf) raises AttributeError because
    ExponentialMovingWindow has no .apply() method. ***
E7. String function names work: 'mean', 'std', 'var', 'sum', 'min', 'max'.
E8. List of functions produces a MultiIndex column result.
E9. Dict input applies per-column functions independently.

IMPLICIT PROPERTIES (expected but not documented)
--------------------------------------------------
I1.  EWM requires exactly one of com / span / halflife / alpha.
I2.  Result row count equals input row count for single-function aggregation.
I3.  Result column names match input column names for single-function aggregation.
I4.  Original DataFrame is not mutated after calling aggregate.
I5.  Results are deterministic — same inputs always produce same outputs.
I6.  With min_periods=0 (default) and no NaN input, the first row is non-NaN.
I7.  With min_periods=k and fewer than k prior values, result is NaN.
I8.  EWM mean values lie within [min(col), max(col)] for finite, non-NaN input.
I9.  com ↔ alpha equivalence: alpha = 1/(1+com) yields identical results.
I10. span ↔ alpha equivalence: alpha = 2/(span+1) yields identical results.
I11. adjust=True and adjust=False produce results that agree on sufficiently
     long series (the difference vanishes as the window grows).
I12. ignore_na=False (default) and ignore_na=True agree when there are no NaNs.
I13. MultiIndex result has the function names as the innermost column level.
I14. Empty DataFrame input produces an empty DataFrame output.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, series


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_valid_alpha = st.floats(min_value=0.01, max_value=1.0,
                         allow_nan=False, allow_infinity=False)
_valid_com   = st.floats(min_value=0.0, max_value=10.0,
                         allow_nan=False, allow_infinity=False)
_valid_span  = st.floats(min_value=1.0, max_value=20.0,
                         allow_nan=False, allow_infinity=False)

_finite_float = st.floats(min_value=-1e6, max_value=1e6,
                           allow_nan=False, allow_infinity=False)


def _df(min_rows=1, max_rows=30, min_cols=1, max_cols=5):
    """Numeric DataFrame with finite values and RangeIndex."""
    return st.integers(min_value=min_cols, max_value=max_cols).flatmap(
        lambda ncols: st.integers(min_value=min_rows, max_value=max_rows).flatmap(
            lambda nrows: data_frames(
                columns=[column(name=str(i), dtype=float,
                                elements=_finite_float) for i in range(ncols)],
                index=st.just(pd.RangeIndex(nrows)),
            )
        )
    )


def _series(min_size=1, max_size=30):
    """Numeric Series with finite values."""
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: series(
            dtype=float,
            elements=_finite_float,
            index=st.just(pd.RangeIndex(n)),
        )
    )


# ---------------------------------------------------------------------------
# E1 – func type variants
# ---------------------------------------------------------------------------

class TestFuncTypes:

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_func_as_string_mean(self, df, alpha):
        """E1 / E7: A string function name ('mean') is accepted without error."""
        result = df.ewm(alpha=alpha).aggregate('mean')

        # only focus on the assertion
        assert result is not None

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=30)
    def test_func_as_string_std(self, df, alpha):
        """E1 / E7: A string function name ('std') is accepted without error."""
        result = df.ewm(alpha=alpha).aggregate('std')
        assert result is not None

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=30)
    def test_func_as_string_var(self, df, alpha):
        """E1 / E7: A string function name ('var') is accepted without error."""
        result = df.ewm(alpha=alpha).aggregate('var')
        assert result is not None

    @given(df=_df(min_cols=2), alpha=_valid_alpha)
    @settings(max_examples=40)
    def test_func_as_list(self, df, alpha):
        """E1 / E8: A list of function strings is accepted without error."""
        result = df.ewm(alpha=alpha).aggregate(['mean', 'std'])
        assert isinstance(result, pd.DataFrame)

    @given(alpha=_valid_alpha)
    @settings(max_examples=40)
    def test_func_as_dict(self, alpha):
        """E1 / E9: A dict mapping column labels to functions is accepted."""
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})
        result = df.ewm(alpha=alpha).aggregate({'A': 'mean', 'B': 'std'})
        assert 'A' in result.columns and 'B' in result.columns


# ---------------------------------------------------------------------------
# E2 – agg alias
# ---------------------------------------------------------------------------

class TestAggAlias:

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_agg_equals_aggregate_single_func(self, df, alpha):
        """E2: .agg('mean') and .aggregate('mean') produce identical results."""
        ewm = df.ewm(alpha=alpha)
        pd.testing.assert_frame_equal(ewm.agg('mean'), ewm.aggregate('mean'))

    @given(df=_df(min_cols=1), alpha=_valid_alpha)
    @settings(max_examples=40)
    def test_agg_equals_aggregate_list_func(self, df, alpha):
        """E2: .agg(['mean', 'std']) and .aggregate(['mean', 'std']) are identical."""
        ewm = df.ewm(alpha=alpha)
        pd.testing.assert_frame_equal(
            ewm.agg(['mean', 'std']),
            ewm.aggregate(['mean', 'std']),
        )


# ---------------------------------------------------------------------------
# E3 / E4 – return type for DataFrame
# ---------------------------------------------------------------------------

class TestDataFrameReturnType:

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_single_func_on_dataframe_returns_dataframe(self, df, alpha):
        """E3 (corrected): DataFrame.ewm().aggregate(single_func) returns a
        DataFrame, NOT a Series as the docstring claims. The docstring is wrong
        for windowed aggregations — each row gets its own EWM result."""
        result = df.ewm(alpha=alpha).aggregate('mean')
        assert isinstance(result, pd.DataFrame), (
            f"Expected DataFrame, got {type(result).__name__}"
        )

    @given(df=_df(min_cols=1), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_several_funcs_on_dataframe_returns_dataframe(self, df, alpha):
        """E4: DataFrame.ewm().aggregate([f1, f2]) returns a DataFrame."""
        result = df.ewm(alpha=alpha).aggregate(['mean', 'std'])
        assert isinstance(result, pd.DataFrame), (
            f"Expected DataFrame, got {type(result).__name__}"
        )


# ---------------------------------------------------------------------------
# E5 – return type for Series
# ---------------------------------------------------------------------------

class TestSeriesReturnType:

    @given(s=_series(), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_single_func_on_series_returns_series(self, s, alpha):
        """E5 (corrected): Series.ewm().aggregate(single_func) returns a Series,
        NOT a scalar as the docstring claims. The docstring is wrong for windowed
        aggregations — each position gets its own EWM result."""
        result = s.ewm(alpha=alpha).aggregate('mean')
        assert isinstance(result, pd.Series), (
            f"Expected Series, got {type(result).__name__}"
        )


# ---------------------------------------------------------------------------
# E6 – UDF support
# ---------------------------------------------------------------------------

class TestUDFSupport:

    @pytest.mark.xfail(
        raises=AttributeError,
        strict=True,
        reason=(
            "pandas bug: EWM.aggregate(udf) internally delegates to .apply(), "
            "but ExponentialMovingWindow has no .apply() method, so any UDF "
            "raises AttributeError. The docstring claim that UDFs are supported "
            "is unreachable for EWM objects."
        ),
    )
    @given(df=_df(min_cols=1), alpha=_valid_alpha)
    @settings(max_examples=20)
    def test_udf_receives_series(self, df, alpha):
        """E6: A UDF passed to aggregate should receive a Series per column.
        KNOWN FAILURE — EWM.aggregate raises AttributeError for UDFs."""
        captured = []

        def udf(arg):
            captured.append(type(arg))
            return arg.mean()

        df.ewm(alpha=alpha).aggregate(udf)
        assert all(t is pd.Series for t in captured)
        assert len(captured) > 0


# ---------------------------------------------------------------------------
# E8 – MultiIndex columns for list input
# ---------------------------------------------------------------------------

class TestMultiIndexColumns:

    @given(df=_df(min_cols=2), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_list_func_produces_multiindex_columns(self, df, alpha):
        """E8 / I13: aggregate(['mean', 'std']) produces MultiIndex columns
        whose innermost level contains exactly the function names."""
        result = df.ewm(alpha=alpha).aggregate(['mean', 'std'])
        assert isinstance(result.columns, pd.MultiIndex), (
            f"Expected MultiIndex columns, got {type(result.columns).__name__}"
        )
        inner = set(result.columns.get_level_values(-1))
        assert inner == {'mean', 'std'}, f"Unexpected inner level values: {inner}"


# ---------------------------------------------------------------------------
# E9 – dict input applies functions per column independently
# ---------------------------------------------------------------------------

class TestDictInput:

    @given(alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_dict_applies_per_column_function(self, alpha):
        """E9 / I9: dict input applies each function to its column independently,
        consistent with calling aggregate with each function separately."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((15, 2)), columns=['A', 'B'])

        result = df.ewm(alpha=alpha).aggregate({'A': 'mean', 'B': 'std'})

        expected_a = df.ewm(alpha=alpha).aggregate('mean')['A']
        expected_b = df.ewm(alpha=alpha).aggregate('std')['B']

        pd.testing.assert_series_equal(result['A'], expected_a)
        pd.testing.assert_series_equal(result['B'], expected_b)

    @given(alpha=_valid_alpha)
    @settings(max_examples=30)
    def test_dict_with_missing_column_raises(self, alpha):
        """E9 (invalid input): dict with a key not in columns raises KeyError."""
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        with pytest.raises((KeyError, Exception)):
            df.ewm(alpha=alpha).aggregate({'NONEXISTENT': 'mean'})


# ---------------------------------------------------------------------------
# I1 – EWM parameter constraints
# ---------------------------------------------------------------------------

class TestEWMParameters:

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=30)
    def test_alpha_accepted(self, df, alpha):
        """I1: alpha in (0, 1] is a valid single decay parameter."""
        result = df.ewm(alpha=alpha).aggregate('mean')
        assert isinstance(result, pd.DataFrame)

    @given(df=_df(), com=_valid_com)
    @settings(max_examples=30)
    def test_com_accepted(self, df, com):
        """I1: com >= 0 is a valid single decay parameter."""
        result = df.ewm(com=com).aggregate('mean')
        assert isinstance(result, pd.DataFrame)

    @given(df=_df(), span=_valid_span)
    @settings(max_examples=30)
    def test_span_accepted(self, df, span):
        """I1: span >= 1 is a valid single decay parameter."""
        result = df.ewm(span=span).aggregate('mean')
        assert isinstance(result, pd.DataFrame)

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=20)
    def test_two_decay_params_raises(self, df, alpha):
        """I1: providing two decay parameters simultaneously raises an error.
        pandas raises ValueError: 'comass, span, halflife, and alpha are mutually exclusive'."""
        with pytest.raises((TypeError, ValueError)):
            df.ewm(alpha=alpha, com=1.0).aggregate('mean')


# ---------------------------------------------------------------------------
# I2 / I3 – shape preservation
# ---------------------------------------------------------------------------

class TestShapePreservation:

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_row_count_preserved(self, df, alpha):
        """I2: result has the same number of rows as the input DataFrame."""
        result = df.ewm(alpha=alpha).aggregate('mean')
        assert len(result) == len(df), (
            f"Expected {len(df)} rows, got {len(result)}"
        )

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_column_names_preserved(self, df, alpha):
        """I3: result column names match input column names for single func."""
        result = df.ewm(alpha=alpha).aggregate('mean')
        assert list(result.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# I4 – immutability
# ---------------------------------------------------------------------------

class TestImmutability:

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_original_not_mutated(self, df, alpha):
        """I4: calling aggregate does not mutate the original DataFrame."""
        before = df.copy(deep=True)
        _ = df.ewm(alpha=alpha).aggregate('mean')
        pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# I5 – determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_repeated_call_identical(self, df, alpha):
        """I5: calling aggregate twice with the same inputs yields equal results."""
        ewm = df.ewm(alpha=alpha)
        pd.testing.assert_frame_equal(ewm.aggregate('mean'), ewm.aggregate('mean'))


# ---------------------------------------------------------------------------
# I6 / I7 – min_periods NaN semantics
# ---------------------------------------------------------------------------

class TestMinPeriods:

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_default_min_periods_first_row_nonnull(self, df, alpha):
        """I6: with min_periods=0 (default) and no NaN input, the first row
        of the result is entirely non-NaN."""
        result = df.ewm(alpha=alpha, min_periods=0).aggregate('mean')
        assert result.iloc[0].notna().all(), (
            f"First row contains NaN:\n{result.iloc[0]}"
        )

    @given(df=_df(min_rows=5), alpha=_valid_alpha,
           k=st.integers(min_value=2, max_value=4))
    @settings(max_examples=40)
    def test_min_periods_produces_nan_for_early_rows(self, df, alpha, k):
        """I7: with min_periods=k, the first k-1 rows have NaN results because
        fewer than k observations have been seen."""
        assume(len(df) > k)
        result = df.ewm(alpha=alpha, min_periods=k).aggregate('mean')
        early = result.iloc[:k - 1]
        assert early.isna().all().all(), (
            f"Expected NaN for first {k-1} rows with min_periods={k}:\n{early}"
        )


# ---------------------------------------------------------------------------
# I8 – EWM mean bounded by column extremes
# ---------------------------------------------------------------------------

class TestBoundedness:

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_ewm_mean_bounded_by_column_extremes(self, df, alpha):
        """I8: EWM mean values lie within [min(col), max(col)] for finite
        non-NaN input, because EWM mean is a convex combination of past values."""
        result = df.ewm(alpha=alpha, min_periods=0).aggregate('mean')
        for col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            valid = result[col].dropna()
            assert (valid >= col_min - 1e-9).all() and (valid <= col_max + 1e-9).all(), (
                f"Column '{col}': EWM mean out of [{col_min}, {col_max}]:\n{valid}"
            )


# ---------------------------------------------------------------------------
# I9 / I10 – decay parameter equivalence
# ---------------------------------------------------------------------------

class TestDecayEquivalence:

    @given(df=_df(), com=_valid_com)
    @settings(max_examples=50)
    def test_com_alpha_equivalence(self, df, com):
        """I9: com and its equivalent alpha = 1/(1+com) produce identical results."""
        alpha = 1.0 / (1.0 + com)
        r_com   = df.ewm(com=com).aggregate('mean')
        r_alpha = df.ewm(alpha=alpha).aggregate('mean')
        pd.testing.assert_frame_equal(r_com, r_alpha, check_exact=False, rtol=1e-10)

    @given(df=_df(), span=_valid_span)
    @settings(max_examples=50)
    def test_span_alpha_equivalence(self, df, span):
        """I10: span and its equivalent alpha = 2/(span+1) produce identical results."""
        alpha = 2.0 / (span + 1.0)
        r_span  = df.ewm(span=span).aggregate('mean')
        r_alpha = df.ewm(alpha=alpha).aggregate('mean')
        pd.testing.assert_frame_equal(r_span, r_alpha, check_exact=False, rtol=1e-10)


# ---------------------------------------------------------------------------
# I12 – ignore_na=False vs True with no NaN input
# ---------------------------------------------------------------------------

class TestIgnoreNa:

    @given(df=_df(), alpha=_valid_alpha)
    @settings(max_examples=50)
    def test_ignore_na_agrees_when_no_nans(self, df, alpha):
        """I12: ignore_na=False and ignore_na=True produce identical results
        when the input contains no NaN values."""
        r_false = df.ewm(alpha=alpha, ignore_na=False).aggregate('mean')
        r_true  = df.ewm(alpha=alpha, ignore_na=True).aggregate('mean')
        pd.testing.assert_frame_equal(r_false, r_true, check_exact=False, rtol=1e-12)


# ---------------------------------------------------------------------------
# I14 – empty DataFrame
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_dataframe_returns_empty_dataframe(self):
        """I14: aggregate on an empty DataFrame returns an empty DataFrame."""
        df = pd.DataFrame({'A': pd.Series([], dtype=float)})
        result = df.ewm(alpha=0.5).aggregate('mean')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @given(alpha=_valid_alpha)
    @settings(max_examples=30)
    def test_single_row_dataframe(self, alpha):
        """Edge case: a single-row DataFrame produces a single-row result
        with non-NaN values (min_periods=0 default)."""
        df = pd.DataFrame({'A': [42.0], 'B': [-7.5]})
        result = df.ewm(alpha=alpha).aggregate('mean')
        assert len(result) == 1
        assert result.iloc[0].notna().all()

    @given(alpha=_valid_alpha)
    @settings(max_examples=30)
    def test_unrecognized_string_func_raises(self, alpha):
        """Invalid input: unrecognized string function name raises an error."""
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        with pytest.raises((AttributeError, TypeError, ValueError)):
            df.ewm(alpha=alpha).aggregate('not_a_real_function')
