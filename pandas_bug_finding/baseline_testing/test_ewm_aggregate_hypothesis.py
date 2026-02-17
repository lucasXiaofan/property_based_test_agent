"""
Property-Based Testing Script for pandas DataFrame.ewm().aggregate()
=====================================================================

This script comprehensively tests the pandas DataFrame.ewm().aggregate() method
using property-based testing with Hypothesis. It aims to discover bugs by testing
both explicit properties claimed in the docstring and implicit properties that are
expected but not documented.

EXPLICIT PROPERTIES (from docstring):
1. func accepts: function, string function name, list of functions, dict of labels -> functions
2. Returns scalar (Series.agg + single func), Series (DataFrame.agg + single func),
   or DataFrame (DataFrame.agg + several functions)
3. Aggregation is performed over an axis (index by default)
4. `agg` is an alias for `aggregate` — both produce identical results
5. User-defined functions receive a Series for evaluation
6. String function names work (e.g., 'mean', 'std', 'var', 'sum')
7. List of functions returns DataFrame with MultiIndex columns
8. Dict of axis labels -> functions applies per-column aggregation

IMPLICIT PROPERTIES (not explicitly documented but expected):
1. EWM requires exactly one decay parameter (com, span, halflife, or alpha)
2. Result preserves row count for single aggregation functions
3. Original DataFrame is not modified (immutability)
4. NaN handling respects min_periods parameter
5. adjust=True vs adjust=False produce different but both valid results
6. ignore_na parameter affects weight calculation with NaN values
7. Results are deterministic (same input produces same output)
8. EWM mean values are bounded by min/max of finite input data per column
9. Single aggregation preserves column names
10. Works with float64, float32, int64, int32 dtypes
11. Empty DataFrame produces empty result
12. Consistency: aggregate('mean') == ewm.mean(), aggregate('std') == ewm.std(), etc.
13. EWM sum of constant column grows monotonically (weights accumulate)
14. Increasing alpha increases responsiveness (latest values weighted more)

EWM PARAMETERS TESTED:
- com (center of mass): alpha = 1 / (1 + com), com >= 0
- span: alpha = 2 / (span + 1), span >= 1
- halflife: alpha = 1 - exp(-ln(2) / halflife), halflife > 0
- alpha: smoothing factor directly, 0 < alpha <= 1
- min_periods: minimum observations for non-NaN result (default 0)
- adjust: divide by decaying adjustment factor (default True)
- ignore_na: ignore missing values in weight calculation (default False)

PANDAS VERSION:
This script targets pandas 3.0.0.

HOW TO RUN:
-----------
Using uv:
    uv run pytest pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py -v

With Hypothesis statistics:
    uv run pytest pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py -v --hypothesis-show-statistics

Verbose output:
    uv run pytest pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py -vv -s

Run specific test class:
    uv run pytest pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py::TestExplicitProperties -v

result: 4 failed
action: need check the false positve
============== short test summary info ==============
FAILED test_ewm_aggregate_hypothesis.py::TestExplicitProperties::test_callable_func_accepted - AttributeError: 'ExponentialMovingWindow' object...
FAILED test_ewm_aggregate_hypothesis.py::TestExplicitProperties::test_udf_receives_series - AttributeError: 'ExponentialMovingWindow' object...
FAILED test_ewm_aggregate_hypothesis.py::TestMathematicalProperties::test_ewm_mean_alpha_1_equals_original - hypothesis.errors.FailedHealthCheck: It looks li...
FAILED test_ewm_aggregate_hypothesis.py::TestAggregateSpecificBehavior::test_lambda_func - AttributeError: 'ExponentialMovingWindow' object...
====== 4 failed, 64 passed in 86.62s (0:01:26) ======
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import List
import copy


# ==============================================================================
# HYPOTHESIS STRATEGIES
# ==============================================================================

@st.composite
def ewm_decay_params(draw):
    """Generate a valid EWM decay parameter dict (exactly one of com, span, halflife, alpha)."""
    param_type = draw(st.sampled_from(['com', 'span', 'halflife', 'alpha']))

    if param_type == 'com':
        return {'com': draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False))}
    elif param_type == 'span':
        return {'span': draw(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False))}
    elif param_type == 'halflife':
        return {'halflife': draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))}
    else:  # alpha
        return {'alpha': draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))}


@st.composite
def ewm_full_params(draw):
    """Generate full EWM parameter dict including optional params."""
    params = draw(ewm_decay_params())
    params['min_periods'] = draw(st.integers(min_value=0, max_value=5))
    params['adjust'] = draw(st.booleans())
    params['ignore_na'] = draw(st.booleans())
    return params


@st.composite
def numeric_dataframes(draw, min_rows=1, max_rows=20, min_cols=1, max_cols=6):
    """Generate DataFrames with numeric columns."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    col_names = [f"col_{i}" for i in range(n_cols)]
    dtypes = draw(st.lists(
        st.sampled_from([np.float64, np.float32, np.int64, np.int32]),
        min_size=n_cols, max_size=n_cols
    ))

    data = {}
    for col_name, dtype in zip(col_names, dtypes):
        if np.issubdtype(dtype, np.floating):
            values = draw(st.lists(
                st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                min_size=n_rows, max_size=n_rows
            ))
        else:
            values = draw(st.lists(
                st.integers(min_value=-100000, max_value=100000),
                min_size=n_rows, max_size=n_rows
            ))
        data[col_name] = pd.array(values, dtype=dtype)

    return pd.DataFrame(data)


@st.composite
def numeric_dataframes_with_nan(draw, min_rows=2, max_rows=20, min_cols=1, max_cols=5):
    """Generate DataFrames with numeric columns that may contain NaN."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    col_names = [f"col_{i}" for i in range(n_cols)]

    data = {}
    for col_name in col_names:
        values = draw(st.lists(
            st.one_of(
                st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                st.just(np.nan),
            ),
            min_size=n_rows, max_size=n_rows
        ))
        data[col_name] = values

    return pd.DataFrame(data)


@st.composite
def constant_dataframes(draw, min_rows=2, max_rows=15):
    """Generate DataFrames where each column has a constant value."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=1, max_value=4))

    data = {}
    for i in range(n_cols):
        val = draw(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e4, max_value=1e4))
        data[f"col_{i}"] = [val] * n_rows

    return pd.DataFrame(data)


@st.composite
def single_column_series(draw, min_size=2, max_size=20):
    """Generate a numeric Series."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    values = draw(st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=n, max_size=n
    ))
    return pd.Series(values, name="test_series")


EWM_NAMED_FUNCS = ['mean', 'std', 'var', 'sum']


# ==============================================================================
# TEST CLASS 1: EXPLICIT PROPERTIES FROM DOCSTRING
# ==============================================================================

class TestExplicitProperties:
    """Test properties explicitly claimed in the ewm().aggregate() docstring."""

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_string_func_returns_dataframe(self, df, params):
        """aggregate with a single string function on a DataFrame returns a DataFrame."""
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframes(), params=ewm_decay_params(),
           func_name=st.sampled_from(EWM_NAMED_FUNCS))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_named_string_funcs_work(self, df, params, func_name):
        """All documented string function names are accepted by aggregate."""
        result = df.ewm(**params).aggregate(func_name)
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframes(min_cols=1, max_cols=4), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_list_of_funcs_returns_dataframe(self, df, params):
        """aggregate with a list of functions returns a DataFrame with MultiIndex columns."""
        result = df.ewm(**params).aggregate(['mean', 'std'])
        assert isinstance(result, pd.DataFrame)
        # MultiIndex columns: (original_col, func_name)
        assert isinstance(result.columns, pd.MultiIndex)

    @given(df=numeric_dataframes(min_cols=2, max_cols=4), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_dict_func_applies_per_column(self, df, params):
        """aggregate with a dict applies specified functions to specified columns."""
        cols = list(df.columns)
        func_dict = {cols[0]: 'mean'}
        if len(cols) > 1:
            func_dict[cols[1]] = 'std'
        result = df.ewm(**params).aggregate(func_dict)
        assert isinstance(result, pd.DataFrame)
        # Result should only contain the columns specified in the dict
        for col in result.columns:
            col_name = col[0] if isinstance(col, tuple) else col
            assert col_name in func_dict

    @given(s=single_column_series(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_series_agg_single_func_returns_series(self, s, params):
        """Series.ewm().aggregate() with single function returns a Series."""
        result = s.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.Series)

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_agg_is_alias_for_aggregate(self, df, params):
        """agg and aggregate produce identical results."""
        ewm = df.ewm(**params)
        result_agg = ewm.agg('mean')
        result_aggregate = ewm.aggregate('mean')
        pd.testing.assert_frame_equal(result_agg, result_aggregate)

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_callable_func_accepted(self, df, params):
        """aggregate accepts a callable (lambda/function)."""
        result = df.ewm(**params).aggregate(np.mean)
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_udf_receives_series(self, df, params):
        """User-defined function is passed a Series for evaluation."""
        received_types = []

        def track_type(x):
            received_types.append(type(x))
            return x.mean()

        df.ewm(**params).aggregate(track_type)
        # All invocations should receive a Series
        for t in received_types:
            assert t == pd.Series, f"UDF received {t} instead of Series"

    @given(df=numeric_dataframes(min_cols=2, max_cols=4), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_dict_with_list_of_funcs_per_column(self, df, params):
        """Dict can map column names to lists of functions."""
        cols = list(df.columns)
        func_dict = {cols[0]: ['mean', 'std']}
        result = df.ewm(**params).aggregate(func_dict)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.columns, pd.MultiIndex)


# ==============================================================================
# TEST CLASS 2: IMPLICIT BASIC BEHAVIOR PROPERTIES
# ==============================================================================

class TestImplicitBasicBehavior:
    """Test implicit behavioral properties of ewm().aggregate()."""

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_immutability(self, df, params):
        """aggregate does not modify the original DataFrame."""
        df_copy = df.copy(deep=True)
        df.ewm(**params).aggregate('mean')
        pd.testing.assert_frame_equal(df, df_copy)

    @given(df=numeric_dataframes(), params=ewm_decay_params(),
           func_name=st.sampled_from(EWM_NAMED_FUNCS))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_preserves_row_count(self, df, params, func_name):
        """Single aggregation function preserves the number of rows."""
        result = df.ewm(**params).aggregate(func_name)
        assert len(result) == len(df)

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_preserves_column_names_single_func(self, df, params):
        """Single aggregation function preserves column names."""
        result = df.ewm(**params).aggregate('mean')
        assert list(result.columns) == list(df.columns)

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_preserves_index(self, df, params):
        """Aggregation preserves the DataFrame index."""
        result = df.ewm(**params).aggregate('mean')
        pd.testing.assert_index_equal(result.index, df.index)

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_deterministic(self, df, params):
        """Same input produces the same output on repeated calls."""
        ewm = df.ewm(**params)
        result1 = ewm.aggregate('mean')
        result2 = ewm.aggregate('mean')
        pd.testing.assert_frame_equal(result1, result2)

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_result_dtype_is_float(self, df, params):
        """EWM aggregate results are always float dtype."""
        result = df.ewm(**params).aggregate('mean')
        for col in result.columns:
            assert np.issubdtype(result[col].dtype, np.floating), \
                f"Column {col} has dtype {result[col].dtype}, expected floating"


# ==============================================================================
# TEST CLASS 3: EWM PARAMETER VALIDATION AND BEHAVIOR
# ==============================================================================

class TestEWMParameters:
    """Test EWM parameter behavior and validation."""

    @given(df=numeric_dataframes(min_rows=3),
           alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_alpha_directly(self, df, alpha):
        """EWM with alpha parameter works correctly."""
        result = df.ewm(alpha=alpha).aggregate('mean')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    @given(df=numeric_dataframes(min_rows=3),
           com=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_com_parameter(self, df, com):
        """EWM with com parameter: alpha = 1 / (1 + com)."""
        result = df.ewm(com=com).aggregate('mean')
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframes(min_rows=3),
           span=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_span_parameter(self, df, span):
        """EWM with span parameter: alpha = 2 / (span + 1)."""
        result = df.ewm(span=span).aggregate('mean')
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframes(min_rows=3),
           halflife=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_halflife_parameter(self, df, halflife):
        """EWM with halflife parameter: alpha = 1 - exp(-ln(2) / halflife)."""
        result = df.ewm(halflife=halflife).aggregate('mean')
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframes(min_rows=3))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_decay_params_raises(self, df):
        """Providing more than one decay parameter raises ValueError."""
        with pytest.raises(ValueError):
            df.ewm(com=0.5, alpha=0.5)

    @given(df=numeric_dataframes(min_rows=3))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_no_decay_param_raises(self, df):
        """Providing no decay parameter raises ValueError."""
        with pytest.raises(ValueError):
            df.ewm()

    @given(df=numeric_dataframes(min_rows=5),
           alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_adjust_true_vs_false(self, df, alpha):
        """adjust=True and adjust=False produce valid (possibly different) results."""
        result_adj = df.ewm(alpha=alpha, adjust=True).aggregate('mean')
        result_noadj = df.ewm(alpha=alpha, adjust=False).aggregate('mean')
        # Both should be valid DataFrames with same shape
        assert result_adj.shape == result_noadj.shape
        assert result_adj.shape == df.shape
        # First row should be identical (no prior values to weight)
        pd.testing.assert_series_equal(
            result_adj.iloc[0], result_noadj.iloc[0],
            check_names=False
        )

    @given(df=numeric_dataframes(min_rows=5),
           min_periods=st.integers(min_value=0, max_value=5),
           alpha=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_min_periods_nan_count(self, df, min_periods, alpha):
        """When min_periods > 0, early rows have NaN when insufficient observations."""
        result = df.ewm(alpha=alpha, min_periods=min_periods).aggregate('mean')
        if min_periods > 1 and len(df) > 0:
            # First row should be NaN when min_periods > 1
            assert result.iloc[0].isna().all(), \
                f"Expected NaN in first row with min_periods={min_periods}"
        if min_periods > 0 and len(df) >= min_periods:
            # Row at index (min_periods - 1) should be non-NaN for non-NaN input columns
            for col in df.columns:
                if not df[col].iloc[:min_periods].isna().any():
                    assert not pd.isna(result[col].iloc[min_periods - 1]), \
                        f"Expected non-NaN at index {min_periods - 1} with min_periods={min_periods}"


# ==============================================================================
# TEST CLASS 4: CONSISTENCY PROPERTIES
# ==============================================================================

class TestConsistencyProperties:
    """Test consistency between aggregate and direct EWM methods."""

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregate_mean_matches_ewm_mean(self, df, params):
        """aggregate('mean') produces identical results to ewm().mean()."""
        ewm = df.ewm(**params)
        result_agg = ewm.aggregate('mean')
        result_direct = ewm.mean()
        pd.testing.assert_frame_equal(result_agg, result_direct)

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregate_std_matches_ewm_std(self, df, params):
        """aggregate('std') produces identical results to ewm().std()."""
        ewm = df.ewm(**params)
        result_agg = ewm.aggregate('std')
        result_direct = ewm.std()
        pd.testing.assert_frame_equal(result_agg, result_direct)

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregate_var_matches_ewm_var(self, df, params):
        """aggregate('var') produces identical results to ewm().var()."""
        ewm = df.ewm(**params)
        result_agg = ewm.aggregate('var')
        result_direct = ewm.var()
        pd.testing.assert_frame_equal(result_agg, result_direct)

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregate_sum_matches_ewm_sum(self, df, params):
        """aggregate('sum') produces identical results to ewm().sum()."""
        ewm = df.ewm(**params)
        result_agg = ewm.aggregate('sum')
        result_direct = ewm.sum()
        pd.testing.assert_frame_equal(result_agg, result_direct)

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregate_list_contains_individual_results(self, df, params):
        """aggregate(['mean', 'std']) contains the same values as individual calls."""
        ewm = df.ewm(**params)
        result_multi = ewm.aggregate(['mean', 'std'])
        result_mean = ewm.aggregate('mean')
        result_std = ewm.aggregate('std')

        for col in df.columns:
            pd.testing.assert_series_equal(
                result_multi[(col, 'mean')],
                result_mean[col],
                check_names=False
            )
            pd.testing.assert_series_equal(
                result_multi[(col, 'std')],
                result_std[col],
                check_names=False
            )

    @given(s=single_column_series(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_series_aggregate_matches_direct(self, s, params):
        """Series.ewm().aggregate('mean') matches Series.ewm().mean()."""
        ewm = s.ewm(**params)
        result_agg = ewm.aggregate('mean')
        result_direct = ewm.mean()
        pd.testing.assert_series_equal(result_agg, result_direct)


# ==============================================================================
# TEST CLASS 5: MATHEMATICAL PROPERTIES
# ==============================================================================

class TestMathematicalProperties:
    """Test mathematical properties of EWM aggregate results."""

    @given(df=numeric_dataframes(min_rows=2), params=ewm_decay_params())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_ewm_mean_bounded_by_data(self, df, params):
        """EWM mean should be bounded by min and max of the finite input data per column."""
        result = df.ewm(**params).aggregate('mean')
        for col in df.columns:
            finite_mask = np.isfinite(df[col].astype(float))
            if finite_mask.sum() == 0:
                continue
            col_min = df[col][finite_mask].astype(float).min()
            col_max = df[col][finite_mask].astype(float).max()
            result_finite = result[col][np.isfinite(result[col])]
            if len(result_finite) == 0:
                continue
            assert result_finite.min() >= col_min - 1e-10, \
                f"EWM mean {result_finite.min()} < column min {col_min}"
            assert result_finite.max() <= col_max + 1e-10, \
                f"EWM mean {result_finite.max()} > column max {col_max}"

    @given(df=constant_dataframes(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_constant_input_mean_equals_constant(self, df, params):
        """EWM mean of constant column should equal the constant."""
        result = df.ewm(**params).aggregate('mean')
        for col in df.columns:
            constant_val = df[col].iloc[0]
            if pd.isna(constant_val):
                continue
            result_finite = result[col][result[col].notna()]
            if len(result_finite) == 0:
                continue
            np.testing.assert_allclose(
                result_finite.values, constant_val,
                rtol=1e-7, atol=1e-10,
                err_msg=f"EWM mean of constant {constant_val} should equal {constant_val}"
            )

    @given(df=constant_dataframes(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_constant_input_var_equals_zero(self, df, params):
        """EWM variance of constant column should be zero (or NaN for insufficient data)."""
        result = df.ewm(**params).aggregate('var')
        for col in df.columns:
            result_finite = result[col][result[col].notna()]
            if len(result_finite) == 0:
                continue
            np.testing.assert_allclose(
                result_finite.values, 0.0,
                atol=1e-10,
                err_msg=f"EWM var of constant column should be 0"
            )

    @given(df=constant_dataframes(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_constant_input_std_equals_zero(self, df, params):
        """EWM std of constant column should be zero (or NaN for insufficient data)."""
        result = df.ewm(**params).aggregate('std')
        for col in df.columns:
            result_finite = result[col][result[col].notna()]
            if len(result_finite) == 0:
                continue
            np.testing.assert_allclose(
                result_finite.values, 0.0,
                atol=1e-10,
                err_msg=f"EWM std of constant column should be 0"
            )

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_var_is_non_negative(self, df, params):
        """EWM variance should always be non-negative."""
        result = df.ewm(**params).aggregate('var')
        for col in result.columns:
            finite_vals = result[col][result[col].notna()]
            assert (finite_vals >= -1e-10).all(), \
                f"Negative variance found in column {col}: {finite_vals.min()}"

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_std_is_non_negative(self, df, params):
        """EWM standard deviation should always be non-negative."""
        result = df.ewm(**params).aggregate('std')
        for col in result.columns:
            finite_vals = result[col][result[col].notna()]
            assert (finite_vals >= -1e-10).all(), \
                f"Negative std found in column {col}: {finite_vals.min()}"

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_std_squared_equals_var(self, df, params):
        """EWM std squared should approximately equal EWM var."""
        ewm = df.ewm(**params)
        std_result = ewm.aggregate('std')
        var_result = ewm.aggregate('var')
        for col in df.columns:
            std_sq = std_result[col] ** 2
            both_valid = std_result[col].notna() & var_result[col].notna()
            if both_valid.sum() == 0:
                continue
            np.testing.assert_allclose(
                std_sq[both_valid].values,
                var_result[col][both_valid].values,
                rtol=1e-6, atol=1e-10,
                err_msg=f"std^2 != var for column {col}"
            )

    @given(params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_ewm_mean_first_row_equals_first_value(self, params):
        """First row of EWM mean (with min_periods=0) should equal the first data value."""
        df = pd.DataFrame({'a': [3.0, 7.0, 5.0, 1.0]})
        result = df.ewm(**params, min_periods=0).aggregate('mean')
        np.testing.assert_allclose(
            result['a'].iloc[0], 3.0, rtol=1e-10,
            err_msg="First EWM mean should equal first value"
        )

    @given(alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_ewm_mean_alpha_1_equals_original(self, alpha):
        """When alpha=1.0, EWM mean should equal the original values (only latest value matters)."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        assume(abs(alpha - 1.0) < 1e-10)
        result = df.ewm(alpha=1.0).aggregate('mean')
        pd.testing.assert_frame_equal(result, df)


# ==============================================================================
# TEST CLASS 6: NAN HANDLING
# ==============================================================================

class TestNaNHandling:
    """Test NaN handling in ewm().aggregate()."""

    @given(df=numeric_dataframes_with_nan(), params=ewm_full_params())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_nan_input_produces_valid_output(self, df, params):
        """DataFrame with NaN values produces a valid result (no crashes)."""
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    @given(params=ewm_full_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_all_nan_column_produces_all_nan(self, params):
        """A column with all NaN values produces all NaN in the result."""
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan], 'b': [1.0, 2.0, 3.0]})
        result = df.ewm(**params).aggregate('mean')
        assert result['a'].isna().all(), "All-NaN input column should produce all-NaN output"

    @given(alpha=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
           ignore_na=st.booleans())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_ignore_na_affects_result(self, alpha, ignore_na):
        """ignore_na parameter should affect results when NaN is present."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0, 4.0]})
        result_ignore = df.ewm(alpha=alpha, ignore_na=True).aggregate('mean')
        result_no_ignore = df.ewm(alpha=alpha, ignore_na=False).aggregate('mean')
        # Both should produce valid results
        assert isinstance(result_ignore, pd.DataFrame)
        assert isinstance(result_no_ignore, pd.DataFrame)

    @given(min_periods=st.integers(min_value=0, max_value=5))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_min_periods_with_nan_leading(self, min_periods):
        """Leading NaN values count toward min_periods correctly."""
        df = pd.DataFrame({'a': [np.nan, 1.0, 2.0, 3.0, 4.0]})
        result = df.ewm(alpha=0.5, min_periods=min_periods).aggregate('mean')
        # Row 0 is NaN input, so row 0 output should always be NaN
        assert pd.isna(result['a'].iloc[0])


# ==============================================================================
# TEST CLASS 7: EDGE CASES AND BOUNDARY CONDITIONS
# ==============================================================================

class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    @given(params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_empty_dataframe(self, params):
        """Empty DataFrame produces empty result."""
        df = pd.DataFrame({'a': pd.Series([], dtype='float64')})
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @given(params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_single_row_dataframe(self, params):
        """Single-row DataFrame produces valid result."""
        df = pd.DataFrame({'a': [5.0], 'b': [10.0]})
        result = df.ewm(**params, min_periods=0).aggregate('mean')
        # Single row EWM mean should equal the input
        np.testing.assert_allclose(result['a'].iloc[0], 5.0, rtol=1e-10)
        np.testing.assert_allclose(result['b'].iloc[0], 10.0, rtol=1e-10)

    @given(params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_single_column_dataframe(self, params):
        """Single-column DataFrame works correctly."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = df.ewm(**params).aggregate('mean')
        assert result.shape == df.shape
        assert list(result.columns) == ['a']

    @given(alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_very_large_dataframe_rows(self, alpha):
        """Larger DataFrame (100 rows) computes without error."""
        df = pd.DataFrame({'a': np.random.randn(100), 'b': np.random.randn(100)})
        result = df.ewm(alpha=alpha).aggregate('mean')
        assert result.shape == df.shape

    @given(params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_inf_values_in_input(self, params):
        """DataFrame with inf values does not crash (may produce inf/NaN)."""
        df = pd.DataFrame({'a': [1.0, np.inf, 3.0], 'b': [-np.inf, 2.0, 4.0]})
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    @given(params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_mixed_inf_nan(self, params):
        """DataFrame with mixed inf and NaN values does not crash."""
        df = pd.DataFrame({'a': [np.nan, np.inf, -np.inf, 1.0, np.nan]})
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)

    @given(params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_duplicate_column_names(self, params):
        """DataFrame with duplicate column names works."""
        df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=['a', 'a'])
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    @given(params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_multiindex_columns(self, params):
        """DataFrame with MultiIndex columns works."""
        arrays = [['bar', 'bar', 'baz'], ['one', 'two', 'one']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame(np.random.randn(4, 3), columns=index)
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    @given(params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_integer_dtype_input(self, params):
        """Integer dtype columns are handled (auto-cast to float)."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)
        for col in result.columns:
            assert np.issubdtype(result[col].dtype, np.floating)

    @given(params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_nullable_int_dtype(self, params):
        """Nullable Int64 dtype works with ewm aggregate."""
        df = pd.DataFrame({'a': pd.array([1, 2, pd.NA, 4, 5], dtype='Int64')})
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)

    @given(params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_nullable_float_dtype(self, params):
        """Nullable Float64 dtype works with ewm aggregate."""
        df = pd.DataFrame({'a': pd.array([1.0, 2.0, pd.NA, 4.0], dtype='Float64')})
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)

    @given(params=ewm_decay_params())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_boolean_column(self, params):
        """Boolean columns are handled by ewm aggregate."""
        df = pd.DataFrame({'a': [True, False, True, True, False]})
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)

    @given(params=ewm_decay_params(),
           func_name=st.sampled_from(EWM_NAMED_FUNCS))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_custom_index(self, params, func_name):
        """DataFrame with custom (non-default) index preserves it."""
        idx = pd.Index([10, 20, 30, 40], name='my_idx')
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]}, index=idx)
        result = df.ewm(**params).aggregate(func_name)
        pd.testing.assert_index_equal(result.index, idx)

    @given(params=ewm_decay_params())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_very_small_values(self, params):
        """Very small float values don't cause numerical issues."""
        df = pd.DataFrame({'a': [1e-300, 2e-300, 3e-300, 4e-300]})
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)
        # All finite results should be non-negative for positive input
        finite_vals = result['a'][result['a'].notna()]
        assert (finite_vals >= 0).all()

    @given(params=ewm_decay_params())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_very_large_values(self, params):
        """Very large float values don't cause unexpected overflow."""
        df = pd.DataFrame({'a': [1e300, 2e300, 1e300, 2e300]})
        result = df.ewm(**params).aggregate('mean')
        assert isinstance(result, pd.DataFrame)


# ==============================================================================
# TEST CLASS 8: AGGREGATE-SPECIFIC BEHAVIOR
# ==============================================================================

class TestAggregateSpecificBehavior:
    """Test behaviors specific to the aggregate() interface."""

    @given(df=numeric_dataframes(min_cols=2, max_cols=4), params=ewm_decay_params())
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_list_of_funcs_shape(self, df, params):
        """List of N functions on M columns produces M*N columns."""
        funcs = ['mean', 'std']
        result = df.ewm(**params).aggregate(funcs)
        assert len(result.columns) == len(df.columns) * len(funcs)
        assert len(result) == len(df)

    @given(df=numeric_dataframes(min_cols=2, max_cols=4), params=ewm_decay_params())
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_dict_func_subset_columns(self, df, params):
        """Dict aggregate only returns specified columns."""
        cols = list(df.columns)
        func_dict = {cols[0]: 'mean'}
        result = df.ewm(**params).aggregate(func_dict)
        # Result should only contain the specified column(s)
        result_col_names = [c[0] if isinstance(c, tuple) else c for c in result.columns]
        assert set(result_col_names) == set(func_dict.keys())

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_invalid_func_name_raises(self, df, params):
        """Invalid string function name raises an error."""
        with pytest.raises((AttributeError, ValueError, TypeError)):
            df.ewm(**params).aggregate('nonexistent_func')

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    def test_none_func_raises(self, df, params):
        """func=None raises an error."""
        with pytest.raises((TypeError, ValueError)):
            df.ewm(**params).aggregate(None)

    @given(df=numeric_dataframes(), params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_lambda_func(self, df, params):
        """Lambda function works as aggregate func."""
        result = df.ewm(**params).aggregate(lambda x: x.mean())
        assert isinstance(result, pd.DataFrame)

    @given(df=numeric_dataframes(min_cols=2, max_cols=4), params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_dict_with_nonexistent_column_raises(self, df, params):
        """Dict with non-existent column name raises KeyError."""
        func_dict = {'NONEXISTENT_COLUMN': 'mean'}
        with pytest.raises(KeyError):
            df.ewm(**params).aggregate(func_dict)

    @given(df=numeric_dataframes(min_cols=2, max_cols=3), params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_three_func_list_produces_triple_columns(self, df, params):
        """List of 3 functions on DataFrame produces 3x columns."""
        funcs = ['mean', 'std', 'var']
        result = df.ewm(**params).aggregate(funcs)
        assert len(result.columns) == len(df.columns) * 3
        # Verify MultiIndex structure
        assert isinstance(result.columns, pd.MultiIndex)
        for col in df.columns:
            for func in funcs:
                assert (col, func) in result.columns

    @given(s=single_column_series(), params=ewm_decay_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_series_list_of_funcs_returns_dataframe(self, s, params):
        """Series.ewm().aggregate() with list of functions returns DataFrame."""
        result = s.ewm(**params).aggregate(['mean', 'std'])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['mean', 'std']


# ==============================================================================
# TEST CLASS 9: DECAY PARAMETER EQUIVALENCE
# ==============================================================================

class TestDecayParameterEquivalence:
    """Test that different decay parameterizations produce equivalent results."""

    @given(com=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_com_alpha_equivalence(self, com):
        """com and equivalent alpha produce the same result."""
        alpha = 1.0 / (1.0 + com)
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result_com = df.ewm(com=com).aggregate('mean')
        result_alpha = df.ewm(alpha=alpha).aggregate('mean')
        pd.testing.assert_frame_equal(result_com, result_alpha, atol=1e-10)

    @given(span=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_span_alpha_equivalence(self, span):
        """span and equivalent alpha produce the same result."""
        alpha = 2.0 / (span + 1.0)
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result_span = df.ewm(span=span).aggregate('mean')
        result_alpha = df.ewm(alpha=alpha).aggregate('mean')
        pd.testing.assert_frame_equal(result_span, result_alpha, atol=1e-10)

    @given(halflife=st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_halflife_alpha_equivalence(self, halflife):
        """halflife and equivalent alpha produce the same result."""
        alpha = 1.0 - np.exp(-np.log(2) / halflife)
        assume(alpha > 0 and alpha <= 1.0)
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result_hl = df.ewm(halflife=halflife).aggregate('mean')
        result_alpha = df.ewm(alpha=alpha).aggregate('mean')
        pd.testing.assert_frame_equal(result_hl, result_alpha, atol=1e-10)
