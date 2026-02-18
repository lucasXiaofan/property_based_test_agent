# pandas Module Notes

Conventions and patterns discovered across property-based testing of pandas functions.
Loaded automatically at Step 1 of the pbt-w-ir skill.

---

## Index Behavior

- **Identity**: pandas operations consistently return a new object (`result is not df`), even when the result is logically identical to the input. This is true for reindex with the same index, round with 0 decimals, etc.
- **Index preservation**: Index-altering operations (reindex, set_index, reset_index) are the exception; most operations (round, ewm, agg) preserve the original index unchanged.
- **RangeIndex**: Most non-reshaping ops preserve the input index type. merge() resets to RangeIndex; reindex() sets the index to exactly the requested new index.

## NaN / Dtype Handling

- **int → float promotion**: Any operation that may introduce NaN into an integer column silently promotes that column to float64. This applies to reindex (new labels), merge (outer join), etc.
- **NaN in fill methods**: Fill methods (ffill, bfill, nearest) applied during reindex only fill *newly introduced* index positions. Pre-existing NaN values in the original DataFrame are NOT filled — use `fillna()` for that.
- **fill_value dtype coercion**: Passing an incompatible scalar fill_value (e.g. string into float column) silently promotes the column to object dtype. Not documented; behavior is silent.

## Deprecation Warnings

- **copy parameter (3.0.0+)**: Any method that still accepts `copy=` raises `Pandas4Warning` (subclass of FutureWarning) in pandas 3.0.0. The parameter is ignored — Copy-on-Write is always active.

## method / Filling

- **Monotonicity requirement**: All fill methods ('ffill', 'bfill', 'pad', 'backfill', 'nearest') require the axis being operated on to be monotonically increasing or decreasing. Raises `ValueError: index must be monotonic increasing or decreasing` otherwise.
- **limit without method**: Passing `limit=N` with `method=None` raises `ValueError: limit argument only valid if doing pad, backfill or nearest reindexing`.

## Hypothesis Testing

- Use `hypothesis.extra.pandas` (`st.data_frames`, `st.indexes`) for typed DataFrame generation, not raw `st.lists()`.
- For monotonic index generation, use `st.integers()` with `unique=True` then sort, or `pd.date_range` via `st.datetimes()`.
- For invalid-input tests, use `st.text()` filtered to exclude valid enum values.

## Calling Conventions

- **Two-convention functions**: Some pandas functions (reindex, rename) support two calling conventions: `(labels, axis=...)` and `(index=..., columns=...)`. Mixing conventions simultaneously is undocumented and produces surprising behavior — avoid testing mixed invocations.

## EWM / Window Aggregation (ewm.aggregate)

- **Callables not supported**: `ExponentialMovingWindow.aggregate()` does NOT support callable func arguments (lambdas, np.ufuncs, function refs). All callables raise `AttributeError: 'ExponentialMovingWindow' object has no attribute 'apply'`. Only string function names are valid: `'mean'`, `'std'`, `'var'`, `'sum'`. The docstring is wrong on this point.
- **Return type is shape-preserving**: Unlike the standard DataFrame.agg() which reduces along the time axis, ewm.aggregate() always returns time-series output with the same row count as the input. A single string func on a DataFrame returns a DataFrame (not a Series); on a Series returns a Series (not a scalar).
- **Valid string names (EWM-native only)**: `mean`, `std`, `var`, `sum`. Other common names (`min`, `max`, `median`, `count`, `kurt`, `skew`, `sem`, `first`, `last`) raise `AttributeError: not a valid function for ExponentialMovingWindow`.
- **cov/corr shape anomaly**: `ewm.agg('cov')` and `ewm.agg('corr')` succeed but return a DataFrame with MultiIndex rows of shape `(n_rows * n_cols, n_cols)` — qualitatively different from other aggregators. Treat as unresolved ambiguity.
- **Decay parameter equivalence**: `com=c` is mathematically equivalent to `alpha=1/(1+c)`; `span=s` is equivalent to `alpha=2/(s+1)`. Useful for cross-parameterization property tests.
- **std of single obs is NaN**: EWM std of a single-row DataFrame returns NaN (not 0.0), consistent with N-1 degrees of freedom for std.
