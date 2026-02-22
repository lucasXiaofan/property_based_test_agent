# Properties to Test: `pd.DataFrame.reindex`

**Library**: pandas  
**Version**: 3.0.0  
**Signature**: `DataFrame.reindex(labels=None, *, index=None, columns=None, axis=None, method=None, copy=<no_default>, level=None, fill_value=nan, limit=None, tolerance=None)`


## Logical Errors in Documentation

- **[⚠️ minor]** The `copy` parameter description says 'default False' but the signature shows `copy=<no_default>`. Since the parameter is deprecated and ignored in 3.0.0, this inconsistency is cosmetic but may confuse readers inspecting the signature vs the docstring.  *(affects: P3)*


## Valid Input Domain

Use these to construct `@given` strategies and `assume()` calls.

- **`df`**: Any pandas DataFrame; may be empty, have 0 or more rows/columns, various dtypes (numeric, object, nullable), any Index type (RangeIndex, Int64Index, DatetimeIndex, MultiIndex, string Index)
- **`labels`**: array-like or None; when provided alone (axis=None), defaults to axis=0 (row index); when provided with an explicit `axis`, targets that axis
- **`index`**: array-like or None; new row labels for the DataFrame
- **`columns`**: array-like or None; new column labels for the DataFrame
- **`axis`**: 0 / 'index' or 1 / 'columns' or None; selects which axis `labels` applies to; only relevant when `labels` is used
- **`method`**: None (default) | 'pad' | 'ffill' | 'backfill' | 'bfill' | 'nearest'; fill method for newly introduced labels; ONLY valid when the index being reindexed is monotonically increasing or decreasing
- **`copy`**: bool (deprecated in 3.0.0, ignored; raises Pandas4Warning when passed)
- **`level`**: int or level name; for MultiIndex DataFrames, broadcast reindexing across one level only
- **`fill_value`**: scalar (default np.nan); value used for newly introduced labels that have no match in the original index; if a 'compatible' scalar is used it fills as-is; 'incompatible' scalars (e.g. string fill into float column) silently promote column dtype to object
- **`limit`**: int or None (default); max number of consecutive positions to fill; raises ValueError when method=None
- **`tolerance`**: scalar, or list-like of same size as the new index, or None; max distance between original and new labels for inexact matching; list-like must match size of target index exactly

**Cross-parameter constraints:**
- `method` requires the axis being reindexed to be monotonically increasing or decreasing; raises ValueError otherwise
- `limit` is only valid when method is 'pad'/'ffill', 'backfill'/'bfill', or 'nearest'; raises ValueError when method=None
- `tolerance` list-like must be the same size as the target index; raises ValueError if sizes differ
- `labels`+`axis` and `index`/`columns` are two distinct calling conventions; when both `labels` and `index` (or `columns`) are supplied simultaneously, behavior is undocumented — see unresolved_ambiguities A1

**Invalid inputs (should raise):**
- method not in {None, 'pad', 'ffill', 'backfill', 'bfill', 'nearest'} → raises ValueError
- axis not in {0, 1, 'index', 'columns'} → raises ValueError
- limit is not None and method is None → raises ValueError
- tolerance is list-like but len(tolerance) != len(new_index) → raises ValueError
- method specified on a non-monotonic index → raises ValueError: 'index must be monotonic increasing or decreasing'


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

#### P1: labels not present in the original index receive NaN by default

**Test name**: `test_explicit__p1__labels_not_present_in_the_original`

```python
# assert:   pd.isna(result.loc[new_label]).all() for new_label in (new_index - original_index)
# when:     fill_value is not specified (defaults to np.nan)
```

**Strategy**: Generate a DataFrame with a fixed index. Generate a new index that is a superset including at least one label not in the original. Call reindex without fill_value. Assert all newly introduced rows are NaN across all columns.

#### P2: reindex always returns a new DataFrame object, never the same object

**Test name**: `test_explicit__p2__reindex_always_returns_a_new_dataframe`

```python
# assert:   result is not df
```

**Strategy**: Generate any DataFrame and call reindex with any valid new index (including the exact same index as the original). Assert the returned object is not the same Python object (identity check: `result is not df`).

#### P3: passing the `copy` keyword raises a Pandas4Warning deprecation warning

**Test name**: `test_explicit__p3__passing_the_copy_keyword_raises_a`

```python
# assert:   pytest.warns(FutureWarning | pd.errors.Pandas4Warning): df.reindex(new_index, copy=True)
```

**Strategy**: Generate any DataFrame and call reindex with copy=True and separately with copy=False. Use warnings.catch_warnings or pytest.warns to assert a Pandas4Warning (or FutureWarning) is raised. Assert the result is still a correct DataFrame regardless of the copy value.

#### P4: method='ffill'/'pad' propagates the last valid observation forward to fill new labels

**Test name**: `test_explicit__p4__methodffillpad_propagates_the_last_valid_observation`

```python
# assert:   result.loc[new_label] == df.loc[last_valid_before_new_label]
# when:     index is monotonically increasing, method='ffill' or 'pad', new_label has a preceding valid row in the original
```

**Strategy**: Generate a DataFrame with a monotonically increasing integer or datetime index. Add new index values that fall between or after existing ones. Call reindex with method='ffill'. Assert that each filled row equals the last original row preceding it. Also verify 'pad' produces identical results to 'ffill'.

#### P5: method='backfill'/'bfill' uses the next valid observation to fill new labels

**Test name**: `test_explicit__p5__methodbackfillbfill_uses_the_next_valid_observation`

```python
# assert:   result.loc[new_label] == df.loc[next_valid_after_new_label]
# when:     index is monotonically increasing, method='bfill' or 'backfill', new_label has a following valid row in the original
```

**Strategy**: Generate a DataFrame with a monotonically increasing index. Include new index values that precede existing ones. Call reindex with method='bfill'. Assert each filled row equals the first original row after it. Verify 'backfill' produces identical results to 'bfill'.

#### P6: method on a non-monotonic index raises ValueError

**Test name**: `test_explicit__p6__method_on_a_nonmonotonic_index_raises`

```python
# assert:   pytest.raises(ValueError): df.reindex(new_index, method='ffill')
# when:     the DataFrame's index is not monotonically increasing or decreasing
```

**Strategy**: Generate DataFrames with non-monotonic indexes (shuffled integers, shuffled strings, unsorted dates). Call reindex with each of the four method values ('ffill','bfill','pad','backfill','nearest'). Assert ValueError is raised in all cases.

#### P7: fill_value replaces NaN for newly introduced missing labels

**Test name**: `test_explicit__p7__fillvalue_replaces_nan_for_newly_introduced`

```python
# assert:   (result.loc[new_labels] == fill_value).all().all()
# when:     fill_value is a scalar compatible with the column dtype
```

**Strategy**: Generate a DataFrame and a new index that includes labels not in the original. Call reindex with fill_value=0 and with fill_value='missing'. Assert all newly-introduced cells equal the specified fill_value. Confirm that pre-existing rows are unaffected.

#### P8: limit=N restricts forward/backward fill to at most N consecutive positions

**Test name**: `test_explicit__p8__limitn_restricts_forwardbackward_fill_to_at`

```python
# assert:   count of consecutive filled cells <= limit
# when:     method in {'ffill','bfill','pad','backfill','nearest'} and limit is a positive integer
```

**Strategy**: Generate a monotonic DatetimeIndex DataFrame. Expand the index to create runs of 3+ consecutive new labels. Call reindex with method='ffill' and limit=1 (then limit=2). Assert that only the first 1 (or 2) new consecutive labels are filled; the rest remain NaN.

#### P9: limit=N without method raises ValueError

**Test name**: `test_explicit__p9__limitn_without_method_raises_valueerror`

```python
# assert:   pytest.raises(ValueError): df.reindex(new_index, limit=1)
# when:     method is None (default)
```

**Strategy**: Generate any DataFrame. Call reindex with limit=1 and no method argument. Assert ValueError is raised with a message about limit requiring a fill method.

#### P10: tolerance list-like of wrong size raises ValueError

**Test name**: `test_explicit__p10__tolerance_listlike_of_wrong_size_raises`

```python
# assert:   pytest.raises(ValueError): df.reindex(new_index, method='nearest', tolerance=[...])
# when:     len(tolerance) != len(new_index)
```

**Strategy**: Generate a DataFrame with a numeric index. Call reindex with method='nearest' and tolerance as a list of wrong length (both shorter and longer than new_index). Assert ValueError is raised with a message about size mismatch.

### 🔍 Indirect

#### P11: tolerance scalar excludes matches where abs(original_label - new_label) > tolerance, leaving them as NaN

**Test name**: `test_indirect__p11__tolerance_scalar_excludes_matches_where_absoriginallabel`

```python
# assert:   pd.isna(result.loc[new_label]) when abs(nearest_original - new_label) > tolerance
# when:     method='nearest' and tolerance is a scalar
```

**Strategy**: Generate a DataFrame with numeric index values. Construct a new index where some target values are within tolerance of an original label and some are not. Assert that targets within tolerance get the nearest value and targets beyond tolerance get NaN.

#### P12: method='nearest' selects the closer of the two neighboring original values

**Test name**: `test_indirect__p12__methodnearest_selects_the_closer_of_the`

```python
# assert:   result.loc[t] == df.loc[min(original_labels, key=lambda x: abs(x - t))]
# when:     method='nearest', index is monotonic numeric or datetime
```

**Strategy**: Generate a DataFrame with a monotonically increasing integer index. Include new index values that are equidistant from two original labels (to probe tie-breaking) and values clearly closer to one neighbor. Assert that non-tie cases select the geometrically nearest original value.

#### P13: method does not fill pre-existing NaN values in the original DataFrame — it only fills newly introduced index positions

**Test name**: `test_indirect__p13__method_does_not_fill_preexisting_nan`

```python
# assert:   pd.isna(result.loc[original_nan_label]).all()
# when:     original DataFrame contains NaN at some index position, that position is included in the new index, method is not None
```

**Strategy**: Generate a monotonic-index DataFrame that contains NaN values at known rows. Expand the index with new labels. Apply reindex with method='bfill' or 'ffill'. Assert that the original NaN rows remain NaN in the result (the fill only works on new positions).

### 💡 Implicit

#### P14: existing labels in the new index retain their exact original values

**Test name**: `test_implicit__p14__existing_labels_in_the_new_index`

```python
# assert:   result.loc[kept_label].equals(df.loc[kept_label])
# when:     kept_label is present in both the original and the new index
```

**Strategy**: Generate a DataFrame with mixed numeric and NaN values. Construct a new index that is a superset of the original (some labels dropped, some added). For every label present in both indexes, assert the row values in the result match the original row exactly.

#### P15: result index is exactly the requested new index, in the same order

**Test name**: `test_implicit__p15__result_index_is_exactly_the_requested`

```python
# assert:   result.index.tolist() == list(new_index)
# when:     reindexing rows (index parameter or labels without axis)
```

**Strategy**: Generate a DataFrame and a new index in arbitrary order (including repeated labels, reversed order). Call reindex. Assert result.index.tolist() equals the list representation of the requested new index.

#### P16: columns are unchanged when only the row index is reindexed

**Test name**: `test_implicit__p16__columns_are_unchanged_when_only_the`

```python
# assert:   result.columns.tolist() == df.columns.tolist()
# when:     only `index` or `labels` (axis=0) is specified, not `columns`
```

**Strategy**: Generate DataFrames with various numbers of columns and column names. Call reindex using only the index parameter. Assert result.columns matches df.columns exactly in name and order.

#### P17: reindexing with the same index produces a DataFrame equal to the original

**Test name**: `test_implicit__p17__reindexing_with_the_same_index_produces`

```python
# assert:   result.equals(df)
# when:     new_index contains exactly the same labels in the same order as df.index
```

**Strategy**: Generate any DataFrame (including with NaN values, various dtypes). Call reindex with the identical index. Assert result.equals(df) is True and result is not df (distinct object).

#### P18: integer-dtype columns are promoted to float64 when NaN is introduced

**Test name**: `test_implicit__p18__integerdtype_columns_are_promoted_to_float64`

```python
# assert:   result[col].dtype == np.float64
# when:     df[col].dtype is int64 or int32 and at least one new row label is introduced
```

**Strategy**: Generate DataFrames with int64 columns (no NaN). Call reindex with a new index that includes at least one new label. Assert the column dtype is float64 in the result. Verify that float columns and object columns retain their dtype.

#### P19: result row count equals the length of the requested new index

**Test name**: `test_implicit__p19__result_row_count_equals_the_length`

```python
# assert:   len(result) == len(new_index)
# when:     reindexing rows (axis=0)
```

**Strategy**: Generate DataFrames and new indexes of varying lengths (0, 1, many). Assert len(result) equals len(new_index) in all cases, including empty DataFrames and empty new indexes.

#### P20: return type is always pd.DataFrame

**Test name**: `test_implicit__p20__return_type_is_always_pddataframe`

```python
# assert:   isinstance(result, pd.DataFrame)
```

**Strategy**: Generate DataFrames with varied shapes, dtypes, and index types. Call reindex under multiple valid parameter combinations. Assert the return value is always an instance of pd.DataFrame in every case.

#### P21: column reindexing produces NaN for new column labels and preserves existing column values

**Test name**: `test_implicit__p21__column_reindexing_produces_nan_for_new`

```python
# assert:   result[existing_col].equals(df[existing_col]) and pd.isna(result[new_col]).all()
# when:     using columns= or labels+axis='columns'
```

**Strategy**: Generate a DataFrame and a new column list that is a superset (adds new columns, drops others). Call reindex(columns=new_cols). Assert existing column data is preserved and new columns are all NaN. Verify via axis='columns' calling convention produces the same result.



## Helpful Resources


- **Basic tutorial**: https://hypothesis.readthedocs.io/en/latest/quickstart.html
- **Strategies reference**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
- **NumPy strategies**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#numpy
- **Pandas strategies**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas    - 
    
