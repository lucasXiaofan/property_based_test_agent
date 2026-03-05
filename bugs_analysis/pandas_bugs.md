# Pandas — Recent Closed Logical Bug Analysis

> **Source:** GitHub repository [`pandas-dev/pandas`](https://github.com/pandas-dev/pandas)
> **Scope:** Logical bugs in the library itself (excluding hardware, OS, and dependency-version issues)
> **Date Range:** Jan – Mar 2026

---

## Table 1 — Developer-Confirmed Bugs

Bugs confirmed by pandas core maintainers via labels (`Regression`, category labels), explicit comments
("Thanks for the report", "agreed", "confirmed this") or merged fix PRs. Key reviewers:
`rhshadrach`, `jorisvandenbossche`, `mroeschke`, `eicchen`.

| # | Issue | Title | Category | Closed | Documentation URL |
|---|-------|-------|----------|--------|-------------------|
| 1 | [#64267](https://github.com/pandas-dev/pandas/issues/64267) | `pd.col()` with `**` power operator produces wrong error message in new expression syntax | Expressions / API | 2026-02-23 | [pandas.col](https://pandas.pydata.org/docs/reference/api/pandas.col.html) |
| 2 | [#64044](https://github.com/pandas-dev/pandas/issues/64044) | `pd.to_timedelta` unit parameter handled inconsistently — regression from 2.x | Timedelta / Regression | 2026-02-11 | [pandas.to_timedelta](https://pandas.pydata.org/docs/reference/api/pandas.to_timedelta.html) |
| 3 | [#63993](https://github.com/pandas-dev/pandas/issues/63993) | `DataFrame.reindex` with multiple `columns` and a string `fill_value` raises `AssertionError` in 3.0 | Indexing / Regression | 2026-02-11 | [DataFrame.reindex](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reindex.html) |
| 4 | [#63920](https://github.com/pandas-dev/pandas/issues/63920) | `groupby(observed=False)` assigns rows with `NaN` categorical keys to the **wrong** groups | Groupby / Categorical / Regression | 2026-01-31 | [DataFrame.groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) |
| 5 | [#63889](https://github.com/pandas-dev/pandas/issues/63889) | `DataFrame.from_records()` with a `None` column name now produces all-`NaN` values instead of the actual data | IO / Constructors / Regression | 2026-02-10 | [DataFrame.from_records](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_records.html) |
| 6 | [#63899](https://github.com/pandas-dev/pandas/issues/63899) | `DataFrame` created from an `Index` can silently mutate the original `Index` (CoW violation) | Copy / View Semantics | 2026-01-28 | [Copy-on-Write guide](https://pandas.pydata.org/docs/user_guide/copy_on_write.html) |
| 7 | [#63879](https://github.com/pandas-dev/pandas/issues/63879) | Converting a NumPy masked array to a pandas array via `pd.array()` does not preserve `NA` values | Constructors / Missing Data | 2026-02-16 | [pandas.array](https://pandas.pydata.org/docs/reference/api/pandas.array.html) |

### Confirmation Details

| Issue | Confirming Developer | Key Quote |
|-------|---------------------|-----------|
| #64267 | `rhshadrach` | Triaged with `expressions` label; fix merged |
| #64044 | `rhshadrach` | Labeled `Regression`; acknowledged and fixed |
| #63993 | `rhshadrach` | Labeled `Regression`; "Thanks for the report" + fix |
| #63920 | `eicchen` | Labeled `Regression`; confirmed and fixed in 3.0 |
| #63889 | `rhshadrach` | "Thanks for the report! you can restore 2.3 behavior via …" and labeled `Regression` |
| #63899 | `jorisvandenbossche` | "I can confirm this. For Series we correctly track this, but apparently not for DataFrames." |
| #63879 | `rhshadrach` | "This was fixed for Series/DataFrame in #24581, while pandas has little support …" — acknowledged as a known gap |

---

## Table 2 — Reported Bugs NOT Yet Confirmed by Developers

These issues were filed as bugs but either (a) received no maintainer response, (b) the root cause
is disputed, or (c) the issue may lie outside pandas (e.g., in a dependency).

| # | Issue | Title | Category | Closed | Note | Documentation URL |
|---|-------|-------|----------|--------|------|-------------------|
| 1 | [#63935](https://github.com/pandas-dev/pandas/issues/63935) | 3.0.0 regression — cannot shuffle `Series[string]` with `numpy.random.shuffle` | Strings | 2026-02-09 | Maintainer `mroeschke` asked for clarification (PyArrow-backed vs Python string?); no explicit bug confirmation; may be a NumPy interop issue | [StringDtype](https://pandas.pydata.org/docs/reference/api/pandas.StringDtype.html) |
| 2 | [#63903](https://github.com/pandas-dev/pandas/issues/63903) | `apply()` silently converts nullable `Int64` to `float64`, causing precision loss | Dtype Conversions | 2026-02-16 | Only community contributor `kjmin622` commented. No pandas maintainer confirmed or triaged. | [DataFrame.apply](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html) |
| 3 | [#64060](https://github.com/pandas-dev/pandas/issues/64060) | `pd.merge` left join with 3 keys incorrectly matches a `NaN` key with a non-`NaN` value when using PyArrow 21.0 | Merge / Arrow | 2026-02-17 | Labeled `Arrow`; `jorisvandenbossche` investigated but could not reproduce on all PyArrow builds — cause likely inside PyArrow, not pandas itself | [DataFrame.merge](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html) |
| 4 | [#64055](https://github.com/pandas-dev/pandas/issues/64055) | `pd.read_sas()` raises an error and no longer works | IO / SAS | 2026-02-06 | Labeled `Needs Triage`; no maintainer commented; possibly a format or file-encoding issue | [pandas.read_sas](https://pandas.pydata.org/docs/reference/api/pandas.read_sas.html) |

---

## Notes

- pandas labels `Bug` + `Regression` reliably signal developer confirmation — regressions always get fixed.
- Issues labeled only `Needs Triage` have not been reviewed by a maintainer yet.
- Issues labeled `Arrow` often turn out to be bugs in the `pyarrow` backend rather than in pandas core.
- The `Copy / view semantics` category (#63899) is particularly important post-CoW (pandas 3.0) because
  the semantics changed substantially and edge cases are still being discovered.
