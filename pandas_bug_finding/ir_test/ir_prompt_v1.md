# IR Specification: `pandas.DataFrame.reindex`

**Version**: pandas 3.0.0
**Source URL**: https://pandas.pydata.org/pandas-docs/version/3.0/reference/api/pandas.DataFrame.reindex.html
**Source code**: https://github.com/pandas-dev/pandas/blob/v3.0.0/pandas/core/frame.py#L5843-L6089

---

## 1. Function Metadata

| Field | Value |
|---|---|
| **Fully Qualified Name** | `pandas.DataFrame.reindex` |
| **Function Category** | **Transformer** |
| **Category Rationale** | Takes a DataFrame, returns a new DataFrame with conforming index/columns. Does not reduce dimensionality, does not mutate in place. |

---

## 2. Typed Preconditions

| Param | Type Constraint | Valid Range / Enum | Nullable | Default | Constraint Strength | Provenance | Notes |
|---|---|---|---|---|---|---|---|
| `labels` | array-like | any hashable elements | Yes (omit) | `None` | may | doc | Positional shortcut; mutually exclusive usage pattern with `index`/`columns` when `axis` is specified |
| `index` | array-like (preferably `pd.Index`) | any hashable elements valid as index | Yes (omit) | `None` | may | doc | New row labels. Preferably Index object to avoid duplication |
| `columns` | array-like (preferably `pd.Index`) | any hashable elements valid as columns | Yes (omit) | `None` | may | doc | New column labels. Preferably Index object to avoid duplication |
| `axis` | `int` or `str` | `{0, 1, 'index', 'columns'}` | Yes (omit) | `None` | must | doc | Only meaningful when `labels` is provided |
| `method` | `str` or `None` | `{None, 'backfill', 'bfill', 'pad', 'ffill', 'nearest'}` | Yes | `None` | must | doc | **Requires monotonically increasing/decreasing index** to work |
| `copy` | `bool` | `{True, False}` | No | `False` | may | doc | **DEPRECATED in 3.0** — ignored, always returns new object via CoW. Will be removed in 4.0 |
| `level` | `int` or name | valid level of a MultiIndex | Yes (omit) | `None` | may | doc | Only meaningful when index is a `MultiIndex` |
| `fill_value` | scalar | any value "compatible" with the dtype | No | `np.nan` | may | doc | Used for missing values introduced by reindexing, NOT for pre-existing NaN |
| `limit` | `int` or `None` | positive integer | Yes | `None` | may | doc | Max consecutive elements to forward/backward fill. Only meaningful when `method` is not None |
| `tolerance` | scalar or list-like | numeric; must match index dtype | Yes (omit) | `None` | may | doc | Max distance for inexact matches. List-like must be same size as new index and match index dtype exactly |

---

## 3. Typed Postconditions

| Property | Constraint | Constraint Strength | Provenance | Notes |
|---|---|---|---|---|
| **Return type** | `pandas.DataFrame` | must | doc | Always returns a DataFrame, never Series |
| **Output index** | Equals the requested new index (or original if not reindexed) | must | doc | `result.index` equals `index` param if provided |
| **Output columns** | Equals the requested new columns (or original if not reindexed) | must | doc | `result.columns` equals `columns` param if provided |
| **Values preserved** | For labels present in both old and new index, values are identical | must | doc | Core contract of reindex |
| **New label fill** | Labels in new index but NOT in old index are filled with `fill_value` (default NaN) | must | doc | Unless `method` is specified |
| **dtype preservation** | Columns retain original dtype when possible; may upcast to accommodate `fill_value`/NaN | should | inferred | e.g., int column → float if NaN introduced. Needs source_code/test verification for 3.0 nullable dtypes |
| **Row count** | `len(result) == len(new_index)` | must | doc | |
| **Column count** | `len(result.columns) == len(new_columns)` | must | doc | |
| **New object** | Always returns a new object (CoW lazy copy) | must | doc | Even if new index == old index, since pandas 3.0 |
| **method fill** | When `method` is specified, fills only NaN introduced by reindexing, NOT pre-existing NaN in data | must | doc | Explicitly documented: "filling while reindexing does not look at DataFrame values, but only compares the original and desired indexes" |
| **Order preserved** | Output row order matches the order of the new index | must | inferred | |

---

## 4. Edge Case Tags

| Edge Case | Applicable? | Constraint Strength | Provenance | Expected Behavior | Notes |
|---|---|---|---|---|---|
| **Empty DataFrame input** | Yes | must | inferred | Returns empty DataFrame with new index/columns | Need to verify: dtypes preserved? Shape = `(len(new_index), 0)` or `(len(new_index), len(new_columns))`? |
| **Empty new index** | Yes | must | inferred | Returns DataFrame with 0 rows | |
| **Single element** | Yes | should | inferred | Works normally | |
| **Duplicates in NEW index** | Yes | must | inferred | _NEED VERIFICATION_: Does it allow duplicate labels in the target? Does it raise? Does it produce duplicate rows? |
| **Duplicates in ORIGINAL index** | Yes | must | inferred | _NEED VERIFICATION_: Likely raises or produces ambiguous results. Check if `InvalidIndexError` is raised |
| **NaN/None in new index** | Yes | should | inferred | _NEED VERIFICATION_: NaN as an index label — does it match existing NaN labels? |
| **All labels new (no overlap)** | Yes | must | doc | All values become `fill_value` | |
| **All labels same (identity reindex)** | Yes | must | inferred | Values identical to original; new object returned (CoW) |
| **Mixed types in index** | Yes | should | inferred | _NEED VERIFICATION_: e.g., mixing int and str labels |
| **MultiIndex** | Yes | may | doc | `level` param enables broadcast matching | |
| **Very large index** | Yes | may | inferred | Performance concern, not correctness | |
| **method with non-monotonic index** | Yes | must | doc | _NEED VERIFICATION_: Should raise `ValueError`? Doc says "only applicable to monotonically increasing/decreasing index" but doesn't specify error |
| **tolerance with non-numeric index** | Yes | should | inferred | _NEED VERIFICATION_: What happens with string index + tolerance? |
| **fill_value type mismatch** | Yes | should | inferred | _NEED VERIFICATION_: e.g., `fill_value="missing"` on int column — does it upcast? The doc example shows this working |
| **limit without method** | Yes | should | inferred | _NEED VERIFICATION_: Is `limit` silently ignored or raises? |
| **Both labels and index provided** | Yes | must | inferred | _NEED VERIFICATION_: Should raise `TypeError`? Mutually exclusive calling conventions |

---

## 5. Error Specification

| Input Violation | Expected Exception | Message Pattern | Constraint Strength | Provenance | Notes |
|---|---|---|---|---|---|
| `method` used with non-monotonic index | `ValueError` (expected) | _NEED VERIFICATION_ | must | inferred | Doc warns it's "only applicable" but doesn't specify error type |
| `axis` not in `{0, 1, 'index', 'columns'}` | `ValueError` (expected) | _NEED VERIFICATION_ | must | inferred | Standard pandas axis validation |
| `labels` + `index` both provided simultaneously | `TypeError` (expected) | _NEED VERIFICATION_ | must | inferred | Conflicting specification — check actual behavior |
| `tolerance` dtype doesn't match index dtype | `ValueError` or `TypeError` (expected) | _NEED VERIFICATION_ | should | inferred from doc: "dtype must exactly match the index's type" |
| `tolerance` list-like wrong size | `ValueError` (expected) | _NEED VERIFICATION_ | should | inferred from doc: "must be the same size as the index" |
| `limit` is negative | `ValueError` (expected) | _NEED VERIFICATION_ | should | inferred | |
| `method` not in valid enum | `ValueError` (expected) | _NEED VERIFICATION_ | must | inferred | |
| `level` on non-MultiIndex | `TypeError` (expected) | _NEED VERIFICATION_ | should | inferred | |

> ⚠️ **Gap**: The documentation does **not** specify any `Raises` section. All error specifications above are **inferred**. To fill these in, we need either `source_code` inspection or empirical testing.

---

## 6. Side Effect Declaration

| Property | Value | Constraint Strength | Provenance | Notes |
|---|---|---|---|---|
| **Mutates input** | **No** | must | doc | "A new object is produced unless the new index is equivalent to the current one and copy=False" — but in 3.0 with CoW, always new object |
| **Modifies global state** | No | must | inferred | |
| **Copy vs View** | **Always copy (CoW lazy)** in pandas 3.0 | must | doc | `copy` param is deprecated and ignored. Uses Copy-on-Write mechanism |
| **Input DataFrame unchanged** | Yes — original df index, columns, and values must be unchanged after call | must | doc + inferred | Critical assertion for Transformer category |

---

## 7. Behavioral Invariants

| Invariant | Description | Constraint Strength | Provenance | Notes |
|---|---|---|---|---|
| **Idempotence** | `df.reindex(idx).reindex(idx)` equals `df.reindex(idx)` | must | inferred | Core Transformer invariant |
| **Identity** | `df.reindex(df.index)` preserves all values (`.equals(df)` is True) | must | inferred | Though a new object is returned |
| **Subset-superset round-trip** | If `new_idx ⊃ old_idx`, then `df.reindex(new_idx).reindex(old_idx)` recovers original values | must | inferred | Values at original index positions should be identical |
| **Column reindex symmetry** | `df.reindex(columns=cols)` == `df.reindex(cols, axis='columns')` == `df.reindex(cols, axis=1)` | must | doc | Two calling conventions must produce identical results |
| **Label-axis equivalence** | `df.reindex(labels=X, axis='index')` == `df.reindex(index=X)` | must | doc | |
| **fill_value consistency** | For any label `L` not in original index: `result.loc[L] == fill_value` for all columns (when method=None) | must | doc | |
| **method=None is default** | `df.reindex(idx)` == `df.reindex(idx, method=None)` | must | doc | |
| **method does not fill pre-existing NaN** | If `df` has NaN at existing index positions, `reindex` with `method` does NOT fill those | must | doc | Explicitly documented invariant |
| **Output shape determinism** | `result.shape == (len(new_index), len(new_columns))` where new_index/new_columns default to original if not specified | must | inferred | |
| **Commutativity of index+columns reindex** | `df.reindex(index=I, columns=C)` == `df.reindex(index=I).reindex(columns=C)` | should | inferred | _NEED VERIFICATION_: Should hold but need to confirm no interaction effects |

### Hidden Expectations for Transformer Category

| Invariant | Applies? | Constraint Strength | Provenance | Notes |
|---|---|---|---|---|
| **dtype preservation** (no unnecessary upcast) | Partially | should | inferred | Int columns may upcast to float when NaN introduced. With nullable Int64 dtype, may preserve. _NEEDS testing for pandas 3.0 behavior_ |
| **Index name preservation** | Yes | should | inferred | `result.index.name` should equal new index's `.name`, or original if not reindexed. _NEED VERIFICATION_ |
| **Column name preservation** | Yes | should | inferred | Same as above for columns axis |
| **attrs preservation** | Unknown | may | inferred | _NEED VERIFICATION_: Does `df.attrs` carry over? |

---

utilize above information to write a test script that utilize general test strategies and hypothesis to test the properties above. 

116 A more complete reference is available in the *Hypothesis Quick Reference* section below.
For a comprehensive reference:
268
269 - **Basic tutorial**: https://hypothesis.readthedocs.io/en/latest/quickstart.html
270 - **Strategies reference**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
271 - **NumPy strategies**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#numpy
272 - **Pandas strategies**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas

write your test script in pandas_bug_finding/ir_test/tests/pandas_round_ir_testing_script.py, and generate natural language docstring on the top of script to summarize the properties your tested, and how to run the script with uv