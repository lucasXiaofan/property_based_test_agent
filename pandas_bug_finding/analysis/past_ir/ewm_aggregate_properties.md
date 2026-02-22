# Properties to Test: `pd.DataFrame.ewm().aggregate`

**Library**: pandas  
**Version**: 3.0.0  
**Signature**: `ExponentialMovingWindow.aggregate(func=None, *args, **kwargs)`


## Logical Errors in Documentation

> ⛔ **Severe errors found.** The properties below are extracted from the parts of the spec that are still sound, but treat test results with caution.

- **[⛔ SEVERE]** The docstring states 'scalar: when Series.agg is called with single function'. Empirically, Series.ewm().agg('mean') always returns a Series of length n_rows, never a scalar. EWM operations are time-series window ops that produce one value per row — reduction to a single scalar is impossible by definition.  *(affects: P5)*
- **[⛔ SEVERE]** The docstring lists 'function' and 'list of functions and/or function names' as accepted func inputs, implying arbitrary callables work. Empirically, every callable tested (np.mean, np.sum, np.std, pd.Series.mean, lambda x: x.iloc[-1]) raises AttributeError: 'ExponentialMovingWindow' object has no attribute 'apply'. Callables are NOT supported by ewm.aggregate in pandas 3.0.0. This is a copy-paste error from the generic .agg() docstring that does not apply to the EWM window context.  *(affects: P11)*
- **[⛔ SEVERE]** The docstring states 'Series: when DataFrame.agg is called with a single function'. Empirically, DataFrame.ewm().agg('mean') returns a DataFrame with shape (n_rows, n_cols) — the same shape as the input. A Series is never returned for DataFrame input. Again this appears to be copied from the standard non-windowed .agg() docstring where the operation reduces along the time axis.  *(affects: P4)*


## Valid Input Domain

Use these to construct `@given` strategies and `assume()` calls.

- **`df_or_series`**: pandas DataFrame or Series with numeric dtype columns/values; may be empty, contain NaN, have any Index type (RangeIndex, string, DatetimeIndex)
- **`ewm_params`**: ExponentialMovingWindow created from DataFrame.ewm() or Series.ewm(); requires exactly one of com/span/halflife/alpha (or halflife+times with adjust=True); min_periods defaults to 0; adjust and ignore_na are booleans
- **`func_string`**: one of the supported string names: 'mean', 'std', 'var', 'sum'. Note: 'min', 'max', 'median', 'count', 'kurt', 'skew', 'sem', 'first', 'last' raise AttributeError (not valid for ExponentialMovingWindow)
- **`func_list`**: list of supported string names, e.g. ['mean', 'std']; all elements must be valid string function names
- **`func_dict`**: dict mapping column label -> string name or list of string names; all keys must be present in the DataFrame columns; for Series input this form is not applicable
- **`args_kwargs`**: positional and keyword arguments forwarded to func; only relevant for callables, which are NOT supported (see E2)

**Cross-parameter constraints:**
- func=None raises TypeError: 'Must provide func or tuples of (column, aggfunc)'
- Callable func values (functions, lambdas, np.ufuncs) always raise AttributeError — only string names are valid (see E2)
- dict keys must be existing column labels; missing keys raise KeyError
- String func names are restricted to EWM-native methods: 'mean', 'std', 'var', 'sum' (and 'cov', 'corr' with non-standard MultiIndex output)
- EWM parent must have exactly one decay parameter unless times is provided

**Invalid inputs (should raise):**
- func=None → TypeError: 'Must provide func'
- func=<any callable> → AttributeError: 'ExponentialMovingWindow' object has no attribute 'apply'
- func='min' or 'max' or 'median' → AttributeError: not a valid function for ExponentialMovingWindow
- func={'Z': 'mean'} where 'Z' not in columns → KeyError
- ewm() with no decay parameter → TypeError
- ewm() with more than one decay parameter → TypeError


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

#### P1: agg is an alias for aggregate and produces identical results — source: docstring Notes: 'agg is an alias for aggregate. Use the alias.'

**Test name**: `test_explicit__p1__agg_is_an_alias_for_aggregate`

```python
# assert:   df.ewm(alpha=0.5).agg(func).equals(df.ewm(alpha=0.5).aggregate(func))
# when:     any valid func input
```

**Strategy**: generate DataFrames with float columns. Call both .agg() and .aggregate() with the same valid string func ('mean', 'std', 'var', 'sum'). Assert the results are equal via .equals().

#### P2: func=None raises TypeError — source: empirical: raises TypeError: 'Must provide func or tuples of (column, aggfunc)'

**Test name**: `test_explicit__p2__funcnone_raises_typeerror`

```python
# assert:   pytest.raises(TypeError): df.ewm(alpha=0.5).aggregate(None)
```

**Strategy**: Generate any non-empty DataFrame with numeric columns. Call .aggregate(None). Assert TypeError is raised with message containing 'func'. Do not parametrize further — the error is deterministic regardless of DataFrame shape.

#### P3: list of string func names returns a DataFrame for both DataFrame and Series input — source: docstring Returns: 'DataFrame: when DataFrame.agg is called with several functions'

**Test name**: `test_explicit__p3__list_of_string_func_names_returns`

```python
# assert:   isinstance(result, pd.DataFrame)
# when:     func is a list of 2+ valid string names
```

**Strategy**: Generate DataFrames and Series with float values. Call .aggregate(['mean', 'std']). Assert isinstance(result, pd.DataFrame) for both. For DataFrame input also verify shape is (n_rows, n_cols * n_funcs) with MultiIndex columns of (col_name, func_name) tuples. For Series input verify shape is (n_rows, n_funcs) with func names as columns.

#### P6: dict func applies different functions to different columns; result has only the dict-specified columns — source: docstring: 'dict of axis labels -> functions, function names or list of such'

**Test name**: `test_explicit__p6__dict_func_applies_different_functions_to`

```python
# assert:   set(result.columns) == set(func_dict.keys()) (when each value is a single func)
# when:     func is a dict mapping column names to single string func names
```

**Strategy**: Generate DataFrames with at least 2 numeric columns. Build a dict assigning 'mean' to one column and 'std' to another. Call .aggregate(dict). Assert result.columns contains exactly the dict keys. Assert no extra columns appear from un-keyed columns.

#### P7: dict with column mapped to a list of funcs produces MultiIndex columns for that column — source: docstring: 'dict of axis labels -> functions, function names or list of such'

**Test name**: `test_explicit__p7__dict_with_column_mapped_to_a`

```python
# assert:   isinstance(result.columns, pd.MultiIndex) when any dict value is a list
# when:     func is a dict where at least one value is a list of string names
```

**Strategy**: Generate DataFrame with 2 numeric columns A and B. Call .aggregate({'A': ['mean', 'std'], 'B': 'mean'}). Assert isinstance(result.columns, pd.MultiIndex). Assert result has columns ('A','mean'), ('A','std'), ('B','mean') — not column 'B' paired with ('B','std').

#### P8: dict with a non-existent column key raises KeyError — source: empirical: raises KeyError: "Label(s) ['Z'] do not exist"

**Test name**: `test_explicit__p8__dict_with_a_nonexistent_column_key`

```python
# assert:   pytest.raises(KeyError): df.ewm(alpha=0.5).aggregate({'Z': 'mean'})
# when:     'Z' is not in df.columns
```

**Strategy**: Generate DataFrames and pick a column label guaranteed not in the frame (e.g., 'NONEXISTENT_COL'). Pass as sole dict key. Assert KeyError is raised with the missing label mentioned.

### 🔍 Indirect

#### P4: DataFrame with single string func returns a DataFrame with the same shape and columns as the input (NOT a Series as the docstring incorrectly states) — source: E3: docstring incorrectly says 'Series'; empirically returns DataFrame of same shape

**Test name**: `test_indirect__p4__dataframe_with_single_string_func_returns`

```python
# assert:   isinstance(result, pd.DataFrame) and result.shape == df.shape and list(result.columns) == list(df.columns)
# when:     input is a DataFrame, func is a single valid string name
```

**Strategy**: Generate DataFrames with 1+ numeric columns and 1+ rows. Call .aggregate('mean'). Assert result is a DataFrame (not Series), result.shape == df.shape, and result.columns matches df.columns. This directly tests against the incorrect docstring claim of 'Series' — the actual behavior is shape-preserving DataFrame output.

#### P5: Series with single string func returns a Series (NOT a scalar as the docstring incorrectly states) — source: E1: docstring incorrectly says 'scalar'; empirically returns Series of same length

**Test name**: `test_indirect__p5__series_with_single_string_func_returns`

```python
# assert:   isinstance(result, pd.Series) and len(result) == len(s)
# when:     input is a Series, func is a single valid string name
```

**Strategy**: Generate Series with float values and length >= 1. Call .aggregate('mean'). Assert isinstance(result, pd.Series) is True (not a scalar, not a DataFrame). Assert len(result) == len(s). This directly tests against the incorrect docstring claim of 'scalar'.

### 💡 Implicit

#### P9: the original DataFrame is not mutated by aggregate

**Test name**: `test_implicit__p9__the_original_dataframe_is_not_mutated`

```python
# assert:   df_before.equals(df_after) where df_after is df after calling .aggregate()
```

**Strategy**: Generate a DataFrame. Snapshot it with df.copy(). Call .ewm(alpha=0.5).aggregate('mean'). Assert the original DataFrame still equals the snapshot. Test with multiple func types (string, list, dict).

#### P10: aggregate is deterministic: identical calls return equal results

**Test name**: `test_implicit__p10__aggregate_is_deterministic_identical_calls_return`

```python
# assert:   r1.equals(r2) where r1 and r2 are results of two identical calls
```

**Strategy**: Generate a DataFrame and valid EWM config. Call .aggregate(func) twice with the same func. Assert r1.equals(r2). Use hypothesis to vary the DataFrame contents and ewm decay parameters.

#### P11: any callable func raises AttributeError (EWM does not support apply-based aggregation) — source: E2: AttributeError: 'ExponentialMovingWindow' object has no attribute 'apply'

**Test name**: `test_implicit__p11__any_callable_func_raises_attributeerror_ewm`

```python
# assert:   pytest.raises(AttributeError): df.ewm(alpha=0.5).aggregate(callable_func)
# when:     func is any Python callable (lambda, np.ufunc, function reference)
```

**Strategy**: Pass several callable types: a lambda, np.mean, np.sum, pd.Series.mean, and a custom def function. Assert each raises AttributeError with message mentioning 'apply'. This verifies the undocumented restriction (see E2 — the docstring incorrectly implies callables are supported).

#### P12: result index is identical to the original DataFrame/Series index

**Test name**: `test_implicit__p12__result_index_is_identical_to_the`

```python
# assert:   result.index.equals(df.index)
```

**Strategy**: Generate DataFrames with varied index types: RangeIndex, string index (['a','b','c']), DatetimeIndex, and arbitrary integer index. Call .aggregate('mean'). Assert result.index equals the input index for all cases.

#### P13: result is always a new object, not the original DataFrame

**Test name**: `test_implicit__p13__result_is_always_a_new_object`

```python
# assert:   result is not df
```

**Strategy**: Create a DataFrame. Call .aggregate('mean'). Assert result is not df (identity check). Also confirm result.values is not df.values. Use hypothesis to vary input shape.

#### P14: empty DataFrame input returns an empty DataFrame with the same columns

**Test name**: `test_implicit__p14__empty_dataframe_input_returns_an_empty`

```python
# assert:   result.shape == (0, n_cols) and list(result.columns) == list(df.columns)
# when:     df has 0 rows but 1+ columns
```

**Strategy**: Construct DataFrames with 0 rows and 1-4 numeric columns. Call .aggregate('mean'). Assert result is a DataFrame with the same column names and 0 rows. Test also with 'std' and 'var'.

#### P15: NaN-only column produces all-NaN output column (NaN is not silently replaced or dropped)

**Test name**: `test_implicit__p15__nanonly_column_produces_allnan_output_column`

```python
# assert:   result['A'].isna().all() when df['A'].isna().all()
# when:     a column contains only NaN values
```

**Strategy**: Generate DataFrames where at least one column is entirely NaN (pd.Series([np.nan]*n)). Call .aggregate('mean'). Assert the corresponding result column is entirely NaN. Verify no ValueError or silent dropping occurs.

#### P16: min_periods=N causes the first N-1 output positions to be NaN regardless of EWM parameters

**Test name**: `test_implicit__p16__minperiodsn_causes_the_first_n1_output`

```python
# assert:   result.iloc[:min_periods-1].isna().all().all() when min_periods >= 2
# when:     min_periods >= 2 and the DataFrame has at least min_periods rows without NaN
```

**Strategy**: Generate DataFrames with non-NaN float data, n_rows >= 3. Set min_periods=2. Call .aggregate('mean'). Assert result row 0 is NaN and rows 1+ are non-NaN. Also test min_periods > n_rows: all result rows should be NaN.

#### P17: std aggregation of a single observation returns NaN (insufficient data for standard deviation)

**Test name**: `test_implicit__p17__std_aggregation_of_a_single_observation`

```python
# assert:   result['A'].isna().all() when df has 1 row
# when:     func='std' and the DataFrame has exactly 1 row
```

**Strategy**: Create a single-row DataFrame. Call .ewm(alpha=0.5).aggregate('std'). Assert the result is NaN (not 0.0 or any finite value). This reflects the statistical requirement of at least 2 observations for std. Also confirm min_periods=0 does not override this — the NaN here is inherent to std, not min_periods.

### 📚 Convention

#### P18: EWM parameters com, span, halflife, alpha produce mathematically equivalent results when using the corresponding parameterizations (com=0.5 ↔ alpha=2/3) — source: ewm docstring: alpha = 1/(1+com) for com>=0; alpha = 2/(span+1) for span>=1

**Test name**: `test_convention__p18__ewm_parameters_com_span_halflife_alpha`

```python
# assert:   df.ewm(com=c).agg('mean').equals(df.ewm(alpha=1/(1+c)).agg('mean'))
# when:     non-NaN numeric DataFrame, adjust=True (default)
```

**Strategy**: Generate a non-NaN float DataFrame and a positive float c in (0, 10). Compute alpha = 1/(1+c). Assert df.ewm(com=c).agg('mean') equals df.ewm(alpha=alpha).agg('mean') using np.allclose for floating point tolerance. Also test span: alpha = 2/(span+1), so span=2 ↔ alpha=2/3.



## Helpful Resources


- **Basic tutorial**: https://hypothesis.readthedocs.io/en/latest/quickstart.html
- **Strategies reference**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
- **NumPy strategies**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#numpy
- **Pandas strategies**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas    - 
    
