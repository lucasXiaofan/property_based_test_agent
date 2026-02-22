"""
Property-based tests for pd.DataFrame.ewm.aggregate
Generated from: ir/pandas/ewm_aggregate_properties.md
Library: pandas 3.0.0
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, series


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

valid_alpha = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_com = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)

numeric_dtype = st.sampled_from([np.float64, np.float32, np.int64, np.int32])


def numeric_dataframe(min_rows=1, max_rows=20, min_cols=1, max_cols=4):
    """Generate a numeric DataFrame with finite values."""
    return st.integers(min_value=min_cols, max_value=max_cols).flatmap(
        lambda ncols: data_frames(
            columns=[
                column(name=str(i), dtype=float,
                       elements=st.floats(min_value=-1e6, max_value=1e6,
                                          allow_nan=False, allow_infinity=False))
                for i in range(ncols)
            ],
            index=st.integers(min_value=min_rows, max_value=max_rows).flatmap(
                lambda n: st.just(pd.RangeIndex(n))
            ),
        )
    )


def numeric_series_strategy(min_size=1, max_size=20):
    """Generate a numeric Series with finite values."""
    return series(
        dtype=float,
        elements=st.floats(min_value=-1e6, max_value=1e6,
                           allow_nan=False, allow_infinity=False),
        index=st.integers(min_value=min_size, max_value=max_size).flatmap(
            lambda n: st.just(pd.RangeIndex(n))
        ),
    )


# ---------------------------------------------------------------------------
# Explicit properties
# ---------------------------------------------------------------------------

class TestExplicit:

    @given(df=numeric_dataframe(), alpha=valid_alpha)
    @settings(max_examples=50)
    def test_explicit__p1__agg_is_an_alias_for_aggregate(self, df, alpha):
        """agg is an alias for aggregate and produces identical results for all valid inputs.

        Source: docstring Notes section — 'agg is an alias for aggregate. Use the alias.'
        """
        assume(len(df) > 0 and len(df.columns) > 0)
        ewm = df.ewm(alpha=alpha)
        pd.testing.assert_frame_equal(ewm.agg('mean'), ewm.aggregate('mean'))

        func_list = ['mean', 'std']
        result_agg = ewm.agg(func_list)
        result_aggregate = ewm.aggregate(func_list)
        pd.testing.assert_frame_equal(result_agg, result_aggregate)

    @given(df=numeric_dataframe(min_cols=1), alpha=valid_alpha)
    @settings(max_examples=50)
    def test_explicit__p2__dataframeaggregate_with_several_functions_returns_a(self, df, alpha):
        """DataFrame.aggregate with several functions returns a DataFrame.

        Source: docstring Returns section — 'DataFrame: when DataFrame.agg is called with several functions'
        """
        assume(len(df) > 0 and len(df.columns) > 0)
        result = df.ewm(alpha=alpha).aggregate(['mean', 'std'])
        assert isinstance(result, pd.DataFrame), (
            f"Expected pd.DataFrame, got {type(result)}"
        )

    @pytest.mark.xfail(
        raises=AttributeError,
        strict=True,
        reason=(
            "pandas bug: EWM.aggregate() falls back to .apply() for UDFs, "
            "but ExponentialMovingWindow has no .apply() method. "
            "The docstring claim that UDFs 'will be passed a Series' is unreachable."
        ),
    )
    @given(df=numeric_dataframe(min_cols=2), alpha=valid_alpha)
    @settings(max_examples=30)
    def test_explicit__p3__a_userdefined_function_passed_to_aggregate(self, df, alpha):
        """A user-defined function passed to aggregate receives a Series for evaluation.

        Source: docstring Notes section — 'A passed user-defined-function will be passed a Series for evaluation.'

        KNOWN FAILURE: ExponentialMovingWindow.aggregate() internally calls
        .apply() for UDFs, but EWM does not implement .apply(), so this raises
        AttributeError. The docstring's UDF claim is incorrect for EWM objects.
        """
        assume(len(df) > 0 and len(df.columns) >= 2)
        captured_types = []

        def udf(arg):
            captured_types.append(type(arg))
            return arg.mean()

        df.ewm(alpha=alpha).aggregate(udf)
        assert all(t is pd.Series for t in captured_types), (
            f"Expected all args to be pd.Series, got: {set(captured_types)}"
        )
        assert len(captured_types) > 0, "UDF was never called"


# ---------------------------------------------------------------------------
# Indirect properties
# ---------------------------------------------------------------------------

class TestIndirect:

    @given(df=numeric_dataframe(min_cols=2), alpha=valid_alpha)
    @settings(max_examples=50)
    def test_indirect__p11__when_func_is_a_list_of(self, df, alpha):
        """When func is a list of functions, the result has a MultiIndex column
        where one level contains the function names.

        Source: docstring Parameters — 'list of functions and/or function names, e.g. [np.sum, "mean"]'
        """
        assume(len(df) > 0 and len(df.columns) >= 2)
        result = df.ewm(alpha=alpha).aggregate(['mean', 'std'])
        assert isinstance(result.columns, pd.MultiIndex), (
            f"Expected MultiIndex columns, got {type(result.columns)}"
        )
        innermost = set(result.columns.get_level_values(-1))
        assert innermost == {'mean', 'std'}, (
            f"Expected {{'mean', 'std'}}, got {innermost}"
        )

    @given(alpha=valid_alpha)
    @settings(max_examples=50)
    def test_indirect__p12__when_func_is_a_dict_mapping(self, alpha):
        """When func is a dict mapping column labels to functions, each column is
        aggregated by its assigned function independently.

        Source: docstring Parameters — 'dict of axis labels -> functions, function names or list of such'
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 2))
        df = pd.DataFrame(data, columns=['A', 'B'])

        result = df.ewm(alpha=alpha).aggregate({'A': 'mean', 'B': 'std'})

        expected_a = df.ewm(alpha=alpha).aggregate('mean')['A']
        expected_b = df.ewm(alpha=alpha).aggregate('std')['B']

        pd.testing.assert_series_equal(result['A'], expected_a)
        pd.testing.assert_series_equal(result['B'], expected_b)

    @given(com=valid_com, df=numeric_dataframe())
    @settings(max_examples=50)
    def test_indirect__p13__mathematically_equivalent_decay_parameterizations_com_vs(self, com, df):
        """Mathematically equivalent decay parameterizations (com vs alpha) produce
        identical aggregate results.

        Source: online documentation — 'com: α = 1 / (1 + com), for com ≥ 0'
        """
        assume(len(df) > 0 and len(df.columns) > 0)
        alpha = 1.0 / (1.0 + com)
        result_com = df.ewm(com=com).aggregate('mean')
        result_alpha = df.ewm(alpha=alpha).aggregate('mean')
        pd.testing.assert_frame_equal(result_com, result_alpha, check_exact=False, rtol=1e-10)


# ---------------------------------------------------------------------------
# Implicit properties
# ---------------------------------------------------------------------------

class TestImplicit:

    @given(df=numeric_dataframe(), alpha=valid_alpha)
    @settings(max_examples=50)
    def test_implicit__p4__calling_aggregate_with_a_single_function(self, df, alpha):
        """Calling aggregate with a single function on an EWM DataFrame returns a
        DataFrame (not a Series as the docstring incorrectly states).

        Note: The docstring erroneously states the return type is Series; actual
        behaviour for window operations is to return a DataFrame of the same shape.
        """
        assume(len(df) > 0 and len(df.columns) > 0)
        result = df.ewm(alpha=alpha).aggregate('mean')
        assert isinstance(result, pd.DataFrame), (
            f"Expected pd.DataFrame for single-function aggregate on DataFrame, got {type(result)}"
        )

    @given(s=numeric_series_strategy(), alpha=valid_alpha)
    @settings(max_examples=50)
    def test_implicit__p5__calling_aggregate_with_a_single_function(self, s, alpha):
        """Calling aggregate with a single function on an EWM Series returns a
        Series (not a scalar as the docstring incorrectly states).

        Note: The docstring erroneously states the return type is scalar; actual
        behaviour for window operations is to return a Series of the same length.
        """
        assume(len(s) > 0)
        result = s.ewm(alpha=alpha).aggregate('mean')
        assert isinstance(result, pd.Series), (
            f"Expected pd.Series for single-function aggregate on Series, got {type(result)}"
        )

    @given(df=numeric_dataframe(), alpha=valid_alpha)
    @settings(max_examples=50)
    def test_implicit__p6__aggregate_does_not_mutate_the_original(self, df, alpha):
        """aggregate does not mutate the original DataFrame."""
        assume(len(df) > 0 and len(df.columns) > 0)
        df_before = df.copy(deep=True)
        _ = df.ewm(alpha=alpha).aggregate('mean')
        pd.testing.assert_frame_equal(df, df_before)

    @given(df=numeric_dataframe(), alpha=valid_alpha)
    @settings(max_examples=50)
    def test_implicit__p7__aggregate_is_deterministic_identical_inputs_and(self, df, alpha):
        """aggregate is deterministic — identical inputs and EWM configuration
        always produce identical outputs."""
        assume(len(df) > 0 and len(df.columns) > 0)
        result1 = df.ewm(alpha=alpha).aggregate('mean')
        result2 = df.ewm(alpha=alpha).aggregate('mean')
        pd.testing.assert_frame_equal(result1, result2)

    @given(df=numeric_dataframe(), alpha=valid_alpha)
    @settings(max_examples=50)
    def test_implicit__p8__result_row_count_equals_input_dataframe(self, df, alpha):
        """Result row count equals input DataFrame row count when applying a single function."""
        assume(len(df) > 0 and len(df.columns) > 0)
        result = df.ewm(alpha=alpha).aggregate('mean')
        assert len(result) == len(df), (
            f"Expected {len(df)} rows, got {len(result)}"
        )

    @given(df=numeric_dataframe(), alpha=valid_alpha)
    @settings(max_examples=50)
    def test_implicit__p9__result_column_names_match_input_dataframe(self, df, alpha):
        """Result column names match input DataFrame column names when applying a single function."""
        assume(len(df) > 0 and len(df.columns) > 0)
        result = df.ewm(alpha=alpha).aggregate('mean')
        assert list(result.columns) == list(df.columns), (
            f"Expected columns {list(df.columns)}, got {list(result.columns)}"
        )

    @given(df=numeric_dataframe(), alpha=valid_alpha)
    @settings(max_examples=50)
    def test_implicit__p10__with_minperiods0_default_aggregate_produces_nonnan(self, df, alpha):
        """With min_periods=0 (default), aggregate produces non-NaN output for the
        first row when input has no NaN values.

        Source: online documentation — Parameter min_periods: 'Minimum number of
        observations in window required to have a value; otherwise result is np.nan'
        """
        assume(len(df) > 0 and len(df.columns) > 0)
        # Ensure no NaN in input
        df_clean = df.dropna()
        assume(len(df_clean) > 0)

        result = df_clean.ewm(alpha=alpha).aggregate('mean')
        first_row = result.iloc[0]
        assert first_row.notna().all(), (
            f"Expected no NaN in first row with min_periods=0 and no-NaN input, "
            f"got:\n{first_row}"
        )
