# Properties to Test: `pd.DataFrame.ewm.aggregate`

**Library**: pandas  
**Version**: 3.0.0  
**Signature**: `ExponentialMovingWindow.aggregate(func=None, *args, **kwargs)`


## Logical Errors in Documentation

> ⛔ **Severe errors found.** The properties below are extracted from the parts of the spec that are still sound, but treat test results with caution.

- **[⛔ SEVERE]** The docstring states 'scalar: when Series.agg is called with single function'. This is incorrect for EWM. Because EWM is a window operation, calling aggregate with a single function on an EWM Series returns a Series of the same length as the input (each element is the EWM result up to that point), never a scalar.
- **[⛔ SEVERE]** The docstring states 'Series: when DataFrame.agg is called with a single function'. This is incorrect for EWM. Calling aggregate with a single function on an EWM DataFrame returns a DataFrame of the same shape as the input, not a Series. The return type description was likely copied verbatim from the non-windowed DataFrame.agg docstring without adapting it for the window semantics.
- **[⚠️ minor]** The example in the aggregate docstring shows df.ewm(alpha=0.5).mean(), which calls the .mean() method directly, not .aggregate() or its alias .agg(). This example does not demonstrate any usage of the aggregate interface.


## Valid Input Domain

Use these to construct `@given` strategies and `assume()` calls.

- **`self`**: ExponentialMovingWindow object, constructed via DataFrame.ewm() or Series.ewm()
- **`ewm_configuration`**: {'decay_param': 'exactly one of com (>= 0), span (>= 1), halflife (> 0), or alpha (0 < alpha <= 1) must be provided when times is not used', 'min_periods': 'int >= 0, default 0; minimum observations required to produce a non-NaN result', 'adjust': 'bool, default True; when True applies decaying adjustment factor for early-period imbalance', 'ignore_na': 'bool, default False; when True ignores NaN values in weight calculations'}
- **`func`**: one of: callable, string function name, list of callables and/or strings, or dict mapping column labels to functions/strings/lists
- **`args`**: positional arguments forwarded to func (only applicable when func is callable)
- **`kwargs`**: keyword arguments forwarded to func (only applicable when func is callable)

**Cross-parameter constraints:**
- func must not be None (behavior is undefined per ambiguity A1)
- func as dict — keys must be column labels present in the calling DataFrame/Series
- mutating functions are explicitly not supported and are excluded from the input domain

**Invalid inputs (should raise):**
- func as an unrecognized string name (e.g. 'invalid_func') — raises AttributeError or similar
- func as dict with keys not present in the DataFrame columns — raises KeyError or similar
- mutating UDFs — undefined behavior per docstring warning


## Properties to Test

Each property maps to one test. Use `claim` as the docstring,
`expression` as the assertion skeleton, `when` as the `assume()` guard,
and `strategy` to decide what inputs to generate.

**Test naming convention** — name every test function exactly as shown in each
property's **Test name** field below:

```
test_{type}__{id}__{claim_slug}
```

- `type` is the property classification: `explicit`, `indirect`, `implicit`,
  `convention`, or `ambiguity`.
- `id` is the property ID in lowercase (e.g. `p1`, `p3`).
- `claim_slug` is the first six meaningful words of the claim, snake_cased.

This makes failure triage instant: a failing `test_explicit__*` test means a
documented guarantee is broken; a failing `test_implicit__*` is an inferred
invariant; a failing `test_indirect__*` needs interpretation before filing a bug.

### 📄 Explicit

#### P1: agg is an alias for aggregate and produces identical results for all valid inputs — source: user-provided docstring — Notes section: 'agg is an alias for aggregate. Use the alias.'

**Test name**: `test_explicit__p1__agg_is_an_alias_for_aggregate`

```python
# assert:   pd.testing.assert_frame_equal(df.ewm(alpha=a).agg(func), df.ewm(alpha=a).aggregate(func))
# when:     any valid func (string, callable, list, dict), any valid ewm configuration
```

**Strategy**: generate numeric DataFrames of various shapes and dtypes, construct EWM with randomly drawn valid alpha or com values, call both .agg(func) and .aggregate(func) with the same func argument (string 'mean', callable, and list ['mean', 'std']), assert results are equal using pd.testing.assert_frame_equal

#### P2: DataFrame.aggregate with several functions returns a DataFrame — source: user-provided docstring — Returns section: 'DataFrame: when DataFrame.agg is called with several functions'

**Test name**: `test_explicit__p2__dataframeaggregate_with_several_functions_returns_a`

```python
# assert:   isinstance(df.ewm(alpha=a).aggregate([f1, f2]), pd.DataFrame)
# when:     func is a list of 2 or more valid EWM-compatible function strings, called on a DataFrame EWM object
```

**Strategy**: generate numeric DataFrames with at least one column, construct EWM with valid decay parameters, call .aggregate(['mean', 'std']), assert the result is an instance of pd.DataFrame

#### P3: A user-defined function passed to aggregate receives a Series for evaluation — source: user-provided docstring — Notes section: 'A passed user-defined-function will be passed a Series for evaluation.'

**Test name**: `test_explicit__p3__a_userdefined_function_passed_to_aggregate`

```python
# assert:   all(isinstance(arg, pd.Series) for arg in captured_args)
# when:     func is a callable (UDF), called on a DataFrame EWM object
```

**Strategy**: define a UDF that appends type(arg) to an external list before returning arg.mean(), call df.ewm(alpha=0.5).aggregate(udf) on a multi-column DataFrame, assert every captured argument type is pd.Series

### 🔍 Indirect

#### P11: When func is a list of functions, the result has a MultiIndex column where one level contains the function names — source: user-provided docstring — Parameters: 'list of functions and/or function names, e.g. [np.sum, "mean"]'

**Test name**: `test_indirect__p11__when_func_is_a_list_of`

```python
# assert:   isinstance(result.columns, pd.MultiIndex) and set(result.columns.get_level_values(-1)) == {'mean', 'std'}
# when:     func=['mean', 'std'], called on a multi-column DataFrame EWM object
```

**Strategy**: generate a multi-column numeric DataFrame, call .ewm(alpha=a).aggregate(['mean', 'std']), assert result.columns is a MultiIndex and the innermost level contains exactly the strings 'mean' and 'std'

#### P12: When func is a dict mapping column labels to functions, each column is aggregated by its assigned function independently — source: user-provided docstring — Parameters: 'dict of axis labels -> functions, function names or list of such'

**Test name**: `test_indirect__p12__when_func_is_a_dict_mapping`

```python
# assert:   pd.testing.assert_series_equal(result['A'], df.ewm(alpha=a).aggregate('mean')['A']) and pd.testing.assert_series_equal(result['B'], df.ewm(alpha=a).aggregate('std')['B'])
# when:     func={'A': 'mean', 'B': 'std'}, DataFrame has columns 'A' and 'B', called on a DataFrame EWM object
```

**Strategy**: generate a two-column numeric DataFrame with columns named 'A' and 'B', call .ewm(alpha=a).aggregate({'A': 'mean', 'B': 'std'}), compare result column 'A' against .ewm(alpha=a).aggregate('mean')['A'] and result column 'B' against .ewm(alpha=a).aggregate('std')['B'] using pd.testing.assert_series_equal

#### P13: Mathematically equivalent decay parameterizations (com vs alpha) produce identical aggregate results — source: fetched online documentation — Parameters: 'com: α = 1 / (1 + com), for com ≥ 0' and 'alpha: Specify smoothing factor α directly, 0 < α ≤ 1'

**Test name**: `test_indirect__p13__mathematically_equivalent_decay_parameterizations_com_vs`

```python
# assert:   pd.testing.assert_frame_equal(df.ewm(com=c).aggregate('mean'), df.ewm(alpha=1/(1+c)).aggregate('mean'), check_exact=False, rtol=1e-10)
# when:     func='mean', com >= 0, using alpha = 1/(1+com) as the equivalent parameterization
```

**Strategy**: draw com values in [0, 10], compute equivalent alpha = 1.0 / (1.0 + com), generate a numeric DataFrame, call .ewm(com=com).aggregate('mean') and .ewm(alpha=alpha).aggregate('mean'), assert both results are numerically equal using pd.testing.assert_frame_equal with check_exact=False and rtol=1e-10

### 💡 Implicit

#### P4: Calling aggregate with a single function on an EWM DataFrame returns a DataFrame (not a Series as the docstring incorrectly states)

**Test name**: `test_implicit__p4__calling_aggregate_with_a_single_function`

```python
# assert:   isinstance(df.ewm(alpha=a).aggregate('mean'), pd.DataFrame)
# when:     func is a single string or callable, called on a DataFrame EWM object
```

**Strategy**: generate numeric DataFrames of various shapes, construct EWM with valid alpha/com, call .aggregate('mean') and .aggregate(np.mean-compatible callable), assert each result is an instance of pd.DataFrame

#### P5: Calling aggregate with a single function on an EWM Series returns a Series (not a scalar as the docstring incorrectly states)

**Test name**: `test_implicit__p5__calling_aggregate_with_a_single_function`

```python
# assert:   isinstance(series.ewm(alpha=a).aggregate('mean'), pd.Series)
# when:     func is a single string or callable, called on a Series EWM object
```

**Strategy**: generate numeric Series of various lengths, construct EWM with valid alpha/com, call .aggregate('mean'), assert the result is an instance of pd.Series

#### P6: aggregate does not mutate the original DataFrame

**Test name**: `test_implicit__p6__aggregate_does_not_mutate_the_original`

```python
# assert:   df_before.equals(df_after) where df_after is checked after calling df.ewm(alpha=a).aggregate(func)
# when:     any valid non-mutating func, any valid ewm configuration
```

**Strategy**: generate a numeric DataFrame, capture a deep copy before calling .ewm(alpha=a).aggregate('mean'), compare the original to the copy using pd.testing.assert_frame_equal after the call completes

#### P7: aggregate is deterministic — identical inputs and EWM configuration always produce identical outputs

**Test name**: `test_implicit__p7__aggregate_is_deterministic_identical_inputs_and`

```python
# assert:   pd.testing.assert_frame_equal(result1, result2)
# when:     any valid func, any valid ewm configuration, no random state in func
```

**Strategy**: generate a numeric DataFrame, call .ewm(alpha=a).aggregate('mean') twice with exactly the same EWM parameters, assert both results are equal using pd.testing.assert_frame_equal

#### P8: Result row count equals input DataFrame row count when applying a single function

**Test name**: `test_implicit__p8__result_row_count_equals_input_dataframe`

```python
# assert:   len(df.ewm(alpha=a).aggregate('mean')) == len(df)
# when:     func is a single string or callable, called on a DataFrame EWM object
```

**Strategy**: generate DataFrames of various lengths including length 1 and lengths > 10, call .ewm(alpha=a).aggregate('mean'), assert len(result) == len(df)

#### P9: Result column names match input DataFrame column names when applying a single function

**Test name**: `test_implicit__p9__result_column_names_match_input_dataframe`

```python
# assert:   list(df.ewm(alpha=a).aggregate('mean').columns) == list(df.columns)
# when:     func is a single string or callable, called on a DataFrame EWM object
```

**Strategy**: generate DataFrames with various column name configurations (string names, integer names, mixed types), call .ewm(alpha=a).aggregate('mean'), assert result.columns equals df.columns

#### P10: With min_periods=0 (default), aggregate produces non-NaN output for the first row when input has no NaN values — source: fetched online documentation — Parameter min_periods: 'Minimum number of observations in window required to have a value; otherwise result is np.nan'

**Test name**: `test_implicit__p10__with_minperiods0_default_aggregate_produces_nonnan`

```python
# assert:   result.iloc[0].notna().all()
# when:     func='mean', min_periods=0 (default), input DataFrame has no NaN values, at least one row
```

**Strategy**: generate numeric DataFrames with no NaN values and at least one row, call .ewm(alpha=a).aggregate('mean') with default min_periods=0, assert the first row of the result has no NaN values



## Helpful Resources


- **Basic tutorial**: https://hypothesis.readthedocs.io/en/latest/quickstart.html
- **Strategies reference**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
- **NumPy strategies**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#numpy
- **Pandas strategies**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas    - 
You only need this md, and generate test cases in pandas_bug_finding/ir_test
    
