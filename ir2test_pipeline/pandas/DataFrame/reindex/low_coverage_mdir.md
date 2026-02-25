# Test IR: pandas.DataFrame.reindex
- library: pandas
- version: 3.0.0
- function: pandas.DataFrame.reindex
- doc: https://pandas.pydata.org/pandas-docs/version/3.0.0/reference/api/pandas.DataFrame.reindex.html

---

## Post Conditions

### P1
- track: valid
- type: explicit
- reason: Core contract — labels absent from the original index must receive NaN. Catches bugs where reindex silently drops rows or fills with wrong default.
- input: df with string index ["a","b","c"], new_index=["a","b","c","d"] where "d" is missing
- assertion:
```python
result = df.reindex(["a", "b", "c", "d"])
assert result.loc["d"].isna().all()
```
- source: "Places NA/NaN in locations having no value in the previous index."

### P2
- track: valid
- type: explicit
- reason: fill_value substitutes for NaN at missing label positions. Catches bugs where fill_value is ignored and NaN appears despite a valid fill value being provided.
- input: df with numeric columns, new_index with one missing label, fill_value=0
- assertion:
```python
result = df.reindex(new_index, fill_value=0)
assert (result.loc[missing_labels] == 0).all().all()
assert not result.loc[missing_labels].isna().any().any()
```
- source: "Value to use for missing values. Defaults to NaN, but can be any 'compatible' value."

### P3
- track: valid
- type: explicit
- reason: Labels present in both old and new index must retain their original values unchanged. Catches bugs where reindex corrupts shared rows.
- input: df with index ["a","b","c"], new_index=["b","c","d"] (b and c are shared)
- assertion:
```python
result = df.reindex(["b", "c", "d"])
pd.testing.assert_frame_equal(result.loc[["b", "c"]], df.loc[["b", "c"]])
```
- source: "Conform DataFrame to new index with optional filling logic."

### P4
- track: valid
- type: explicit
- reason: Return type must always be DataFrame regardless of parameters. Catches bugs where reindex returns a wrong type.
- input: any valid df.reindex(new_index)
- assertion:
```python
result = df.reindex(new_index)
assert isinstance(result, pd.DataFrame)
```
- source: "Returns: DataFrame — DataFrame with changed index."

### P5
- track: valid
- type: explicit
- reason: columns= parameter reindexes the column axis, adding new columns as NaN and preserving existing ones. Catches bugs where column reindexing is broken or mixes axes.
- input: df with columns=["a","b"], reindex with columns=["a","c"] where "c" is new
- assertion:
```python
result = df.reindex(columns=["a", "c"])
assert list(result.columns) == ["a", "c"]
assert result["c"].isna().all()
pd.testing.assert_series_equal(result["a"], df["a"])
```
- source: "New labels for the columns." and column reindexing example in docs.

### P6
- track: valid
- type: indirect
- reason: labels+axis="columns" must be exactly equivalent to columns= kwarg. Catches bugs where axis routing fails or produces different results for equivalent call forms.
- input: df, new_cols (array-like of column names)
- assertion:
```python
result1 = df.reindex(new_cols, axis="columns")
result2 = df.reindex(columns=new_cols)
pd.testing.assert_frame_equal(result1, result2)
```
- source: Docs list two calling conventions as alternatives: `(labels, axis=...)` and `(index=..., columns=...)`. Equivalence is implied by both routing to the same axis.

### P7
- track: valid
- type: explicit
- reason: method=bfill fills new index positions before the original range using the next valid observation. Catches bugs in backward-fill logic, especially for index extension.
- input: df2 with monotonic date index [2010-01-01..2010-01-06], new_index extending to [2009-12-29..2010-01-07], method="bfill"
- assertion:
```python
result = df2.reindex(date_index2, method="bfill")
assert result.loc["2009-12-29", "prices"] == 100.0
assert result.loc["2009-12-31", "prices"] == 100.0
assert pd.isna(result.loc["2010-01-07", "prices"])  # no next valid after range
```
- source: "backfill / bfill: Use next valid observation to fill gap."

### P8
- track: valid
- type: explicit
- reason: Pre-existing NaN values in the original DataFrame must NOT be filled by method propagation. Catches bugs where fill propagates into already-NaN positions that belong to the original data.
- input: df2 with NaN at 2010-01-03 in original data, new_index including 2010-01-03, method="bfill"
- assertion:
```python
result = df2.reindex(date_index2, method="bfill")
assert pd.isna(result.loc["2010-01-03", "prices"])
```
- source: "NaN values already present in the original DataFrame will not be filled by propagation schemes; use fillna() if you need to fill those."

### P9
- track: valid
- type: implicit
- reason: Reindex must not mutate the original DataFrame. Catches bugs where the operation modifies the caller in-place.
- input: any df.reindex(new_index)
- assertion:
```python
original_values = df.values.copy()
original_index = df.index.copy()
result = df.reindex(new_index)
assert result is not df
pd.testing.assert_index_equal(df.index, original_index)
np.testing.assert_array_equal(df.values, original_values)
```
- source: Standard pandas no-mutation convention; operations on DataFrames return new objects.

### P10
- track: valid
- type: implicit
- reason: The result index must exactly equal the requested new_index in content and order. Catches bugs where result index diverges from what was requested.
- input: df with arbitrary index, new_index with shuffled and new labels
- assertion:
```python
result = df.reindex(new_index)
assert list(result.index) == list(new_index)
```
- source: "Conform DataFrame to new index" implies the result index equals the target exactly.

### P11
- track: valid
- type: implicit
- reason: Reindexing only the row index must leave column labels unchanged. Catches bugs where a row reindex accidentally alters the column axis.
- input: df with columns=["x","y"], df.reindex(new_row_index) without columns param
- assertion:
```python
result = df.reindex(new_index)
assert list(result.columns) == list(df.columns)
```
- source: Convention: reindex(index=...) targets only the row axis; columns are orthogonal and untouched.

### P12
- track: valid
- type: indirect
- reason: Integer column dtype must be promoted to float64 when NaN fill is required (default fill_value). Catches bugs where dtype promotion is skipped, producing incorrect integer overflow or silent data corruption.
- input: df with int64 column, new_index with missing labels (default fill_value=NaN)
- assertion:
```python
result = df.reindex(new_index)
assert result[int_col].dtype == np.float64
```
- source: "Places NA/NaN in locations having no value." NaN cannot be stored in int64, so float64 promotion is the only valid behavior.

### P13
- track: valid
- type: indirect
- reason: fill_value=0 on an integer column must preserve int64 dtype without float promotion. Catches bugs where dtype promotion happens even when fill_value is dtype-compatible.
- input: df with int64 column, new_index with missing labels, fill_value=0
- assertion:
```python
result = df.reindex(new_index, fill_value=0)
assert result[int_col].dtype == np.int64
```
- source: Docs example shows integer columns remain integer with fill_value=0: "Safari 404 0.07" (int) vs "Safari 404.0 0.07" (float) when NaN fill is used.

### P14
- track: valid
- type: explicit
- reason: limit=N with a fill method caps consecutive filled positions to N. Catches bugs where limit is accepted but silently ignored.
- input: df with monotonic integer index [0,1,5], new_index=[0,1,2,3,4,5], method="ffill", limit=1
- assertion:
```python
result = df.reindex([0,1,2,3,4,5], method="ffill", limit=1)
assert not pd.isna(result.iloc[2])  # index 2: 1 step from 1, filled
assert pd.isna(result.iloc[3])      # index 3: 2nd consecutive, NOT filled
assert pd.isna(result.iloc[4])      # index 4: 3rd consecutive, NOT filled
```
- source: "Maximum number of consecutive elements to forward or backward fill."

### P15
- track: valid
- type: explicit
- reason: method=nearest fills each gap with the value from the closest original label. Catches bugs where nearest incorrectly defaults to ffill or bfill semantics.
- input: df with monotonic integer index [0, 10] with different values, new_index=[0, 4, 6, 10]
- assertion:
```python
result = df.reindex([0, 4, 6, 10], method="nearest")
assert result.iloc[1].equals(df.iloc[0])  # 4 is closer to 0 than 10
assert result.iloc[2].equals(df.iloc[1])  # 6 is closer to 10 than 0
```
- source: "nearest: Use nearest valid observations to fill gap."

### E1
- track: invalid
- type: explicit
- reason: Providing method with a non-monotonic index must raise. Catches bugs where fill is silently applied to a non-monotonic index, producing undefined or wrong results.
- input: df with non-monotonic index [3,1,2], new_index=[1,2,3,4], method="ffill"
- assertion:
```python
with pytest.raises(ValueError):
    df.reindex([1, 2, 3, 4], method="ffill")
```
- source: "Method to use for filling holes in reindexed DataFrame. Only applicable to DataFrames/Series with a monotonically increasing/decreasing index."

### E2
- track: invalid
- type: indirect
- reason: limit without method must raise ValueError. Catches bugs where limit is silently ignored instead of flagging an invalid parameter combination.
- input: df, new_index, limit=2, method=None
- assertion:
```python
with pytest.raises(ValueError):
    df.reindex(new_index, limit=2)
```
- source: "Maximum number of consecutive elements to forward or backward fill" — limit is only defined in the context of a fill method; specifying it without method is user error implied by the parameter's description.

### E3
- track: invalid
- type: indirect
- reason: Providing both labels and index simultaneously must raise to prevent ambiguous axis routing. Catches bugs where one silently wins and the other is discarded.
- input: df.reindex(labels=["a","b"], index=["c","d"])
- assertion:
```python
with pytest.raises((TypeError, ValueError)):
    df.reindex(labels=["a", "b"], index=["a", "b"])
```
- source: The two calling conventions — `(labels, axis=...)` and `(index=..., columns=...)` — are documented as mutually exclusive alternatives. Combining them is inherently ambiguous.
