# pandas `DataFrame.reindex` analysis: 2.3 -> 3.0

## Scope

- API page of interest: `https://pandas.pydata.org/pandas-docs/version/3.0/reference/api/pandas.DataFrame.reindex.html`
- Code/doc comparison baseline: `v2.3.3` vs `v3.0.0` tags in local pandas repo
- Bug mining window: last 1 year from today (`2025-02-13` to `2026-02-13`, UTC), using `gh`

---

## 1) What changed from 2.3 to 3.0 for `reindex`

### A. `copy` semantics changed (major)

In 2.3 (`pandas/core/generic.py` in `v2.3.3`), `reindex(..., copy=...)` still influenced behavior (`copy` passed through internals).

In 3.0 (`pandas/core/generic.py`), `copy` is explicitly deprecated/ignored:

- Signature changed to `copy: bool | lib.NoDefault = lib.no_default`
- `self._check_copy_deprecation(copy)` is called in `reindex`
- fast-path identical-axis branch returns `self.copy(deep=False)` regardless of `copy=True/False`
- `_reindex_axes` and `_reindex_multi` paths no longer take/use `copy`

Relevant code locations:
- `pandas/core/generic.py:5194` (`def reindex`)
- `pandas/core/generic.py:5431` (`self._check_copy_deprecation(copy)`)
- `pandas/core/generic.py:5464` to `pandas/core/generic.py:5478`
- `pandas/core/generic.py:4426` (`def _check_copy_deprecation`)

Release note confirmation:
- `doc/source/whatsnew/v3.0.0.rst:902`
- `doc/source/whatsnew/v3.0.0.rst:912`

### B. New automatic level matching for `Index -> MultiIndex` reindexing

3.0 added a behavior guard before reindexing:

- if `level is None`
- and target `index` is `MultiIndex`
- and source index is not `MultiIndex`
- and `self.index.name` exists in target level names

then pandas auto-sets `level = self.index.name`.

This avoids an all-NaN result in a common `Index` to `MultiIndex` expansion case.

Relevant code location:
- `pandas/core/generic.py:5428`

Release note bug-fix entry:
- `doc/source/whatsnew/v3.0.0.rst:1324` (issue `#60923`)

### C. Public API text now explicitly says `copy` has no effect in 3.0 docs

The 3.0 API docs (from tagged source/docstrings) include deprecation text that `copy` is ignored and will be removed in pandas 4.0.

---

## 2) Why these changes were introduced

### A. CoW consistency and API simplification

Primary reason: Copy-on-Write default in 3.0. pandas standardized method behavior so `copy` no longer tries to force eager copies in methods like `reindex`.

Evidence:
- `doc/source/whatsnew/v3.0.0.rst:890` onward (copy keyword deprecations)
- `doc/source/whatsnew/v3.0.0.rst:912` to `:914` (copy keyword no effect from 3.0)
- Related tracking issue in release notes: `#57347`

### B. Correctness fix for MultiIndex reindexing

The automatic level match logic was introduced to fix incorrect all-NaN outcomes when reindexing from `Index` into `MultiIndex` where a name match existed.

Evidence:
- bug issue `#60923`
- fixing PR `#61969`
- code guard at `pandas/core/generic.py:5428`

---

## 3) Potential risks introduced by these changes

1. Code relying on `copy=True` as a hard copy trigger no longer gets that guarantee.
2. Memory aliasing assumptions can break tests/user code that depended on eager copying side-effects.
3. Name-sensitive auto-level matching (`self.index.name in target.names`) can be surprising:
   - matching names now change behavior (values propagated)
   - non-matching names still produce all-NaN in equivalent shape expansion
4. `fill_value` behavior is still dtype/internals sensitive; docs may imply broader compatibility than actual behavior in edge cases.
5. Reindex behavior through internal block paths (PeriodDtype / mixed blocks / multi-column fill) can regress even when API contract appears unchanged.

---

## 4) Last 1 year: GH bugs relevant to `reindex`

Window used: `2025-02-13` to `2026-02-13`.

| Issue | Created (UTC) | State | Relevance to `reindex` | Fixed by PR |
|---|---:|---|---|---|
| [#60923](https://github.com/pandas-dev/pandas/issues/60923) | 2025-02-13 | Closed | `Series.reindex` from `Index` to `MultiIndex` produced all `NaN` unexpectedly. | [#61969](https://github.com/pandas-dev/pandas/pull/61969) |
| [#60980](https://github.com/pandas-dev/pandas/issues/60980) | 2025-02-21 | Closed | Incorrect column result after `unstack(...).reindex(..., axis=1)` with `Period` data. | [#61114](https://github.com/pandas-dev/pandas/pull/61114) |
| [#61055](https://github.com/pandas-dev/pandas/issues/61055) | 2025-03-04 | Closed | Duplicate report of `#60980`. | N/A (duplicate) |
| [#61291](https://github.com/pandas-dev/pandas/issues/61291) | 2025-04-15 | Open | `reindex(..., fill_value=None)` not respected as users expect in several dtype paths. | Not fixed |
| [#63993](https://github.com/pandas-dev/pandas/issues/63993) | 2026-02-02 | Closed | `DataFrame.reindex(columns=[...], fill_value="missing")` raised `AssertionError` in 3.0.0. | [#63994](https://github.com/pandas-dev/pandas/pull/63994) |
| [#64047](https://github.com/pandas-dev/pandas/issues/64047) | 2026-02-05 | Closed | Duplicate of the string `fill_value` reindex bug. | N/A (duplicate of `#63993`) |

---

## 5) Hidden expectations (not clearly spelled out in docs)

| Hidden expectation | Why it matters | Evidence |
|---|---|---|
| `Index -> MultiIndex` reindexing behavior depends on **index name matching target level name** when `level=None`. | Same values/shape can yield fully propagated values or all-NaN solely by naming. | code: `pandas/core/generic.py:5428`; tests below from PR `#61969` |
| `fill_value` is not universally honored as given (especially `None`) across all dtype/internals paths. | Users expect literal fill semantics; internals may normalize to NA/NaN. | issue [#61291](https://github.com/pandas-dev/pandas/issues/61291) |
| String `fill_value` for multi-column reindex in 3.0.0 was broken (assertion), so practical compatibility differed from docs. | Docs imply generic scalar compatibility; 3.0.0 had a real runtime failure. | issues [#63993](https://github.com/pandas-dev/pandas/issues/63993), [#64047](https://github.com/pandas-dev/pandas/issues/64047), fix PR `#63994` |
| Reindex correctness can depend on internal block layout (e.g., `PeriodDtype` after reshape operations). | Bugs may appear only through specific reshape/reindex execution paths. | issue [#60980](https://github.com/pandas-dev/pandas/issues/60980), PR `#61114` tests |

---

## 6) Fixed items: where new regression tests were added

### Fix for `#60923` via PR `#61969`

- `pandas/tests/series/methods/test_reindex.py:447`
  - `test_reindex_multiindex_automatic_level`
- `pandas/tests/frame/methods/test_reindex.py:1270`
  - `test_reindex_index_name_matches_multiindex_level`
- `pandas/tests/frame/methods/test_reindex.py:1287`
  - `test_reindex_index_name_no_match_multiindex_level`

### Fix for `#60980` via PR `#61114`

- `pandas/tests/indexing/test_indexing.py:766`
  - `test_period_column_slicing`
- `pandas/tests/internals/test_internals.py:1322`
  - `test_period_reindex_axis`

### Fix for `#63993` via PR `#63994`

- `pandas/tests/frame/methods/test_reindex.py:822`
  - `test_reindex_with_string_fill_value`

---

## 7) Practical recommendation for users migrating 2.3 -> 3.0

1. Remove `copy=` usage from `reindex` calls; do explicit `.copy()` when eager copy is required.
2. Add tests for `Index -> MultiIndex` reindexing with both matching and non-matching index names.
3. Add explicit tests for `fill_value` with your actual dtypes (especially `None`, `string`, extension dtypes).
4. If you depend on `unstack + reindex` with `Period`/extension dtypes, pin versions with regression tests around those paths.
