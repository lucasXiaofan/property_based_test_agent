# Pandas Documentation-Related Logic Issues (Post-2024)

This report was compiled using `gh` CLI queries against `pandas-dev/pandas`, focusing on items created after 2024 that indicate a logic/behavior mismatch, ambiguity, or stale behavior description in documentation.

## Scope and filtering

- Repository: `pandas-dev/pandas`
- Time filter: `created: > 2024-12-31`
- Item types: GitHub Issues and Pull Requests
- Relevance criterion: Documentation statement is incorrect, stale, underspecified, or inconsistent with runtime behavior

## Relevant issues / PRs

| ID | Type | Created (UTC) | State | Logic issue tied to docs | Link |
|---|---|---:|---|---|---|
| #64094 | Issue | 2026-02-09 | Open | `float_precision` options are listed but behavior semantics are unclear (e.g., “round_trip”), leading to ambiguous expected parsing behavior. | https://github.com/pandas-dev/pandas/issues/64094 |
| #64095 | PR | 2026-02-09 | Open | Follow-up to #64094; proposes behavior-level clarification of `float_precision` in `read_csv` / `read_table` docs. | https://github.com/pandas-dev/pandas/pull/64095 |
| #64066 | PR | 2026-02-07 | Open | Clarifies NA normalization behavior for `dtype="str"` vs object dtype; docs previously left behavior differences underexplained. | https://github.com/pandas-dev/pandas/pull/64066 |
| #64071 | PR | 2026-02-08 | Merged | Removes stale categorical side-effects section that no longer holds under pandas 3.0 Copy-on-Write semantics. | https://github.com/pandas-dev/pandas/pull/64071 |
| #63818 | Issue | 2026-01-22 | Open | `DataFrame.stack()` docs still describe pre-3.0 semantics/parameters, while behavior changed in 3.0. | https://github.com/pandas-dev/pandas/issues/63818 |
| #63973 | PR | 2026-02-01 | Open | Clarifies deprecation/effect of `dropna`, `sort`, and `future_stack` parameters in `DataFrame.stack` after 3.0 behavior changes. | https://github.com/pandas-dev/pandas/pull/63973 |
| #64061 | PR | 2026-02-07 | Open | Documents `.str` behavior on mixed-type Series (non-strings yielding `NaN`) to align user expectation with actual element-wise behavior. | https://github.com/pandas-dev/pandas/pull/64061 |
| #64089 | PR | 2026-02-09 | Open | Updates `NDFrame.resample` docs to reflect behavior changes from prior implementation PR (#61985). | https://github.com/pandas-dev/pandas/pull/64089 |
| #63191 | Issue | 2025-11-24 | Open | `Index.to_frame(name=...)` docs claim a default of `index.name`, but implementation default is `lib.no_default` (logic-default mismatch). | https://github.com/pandas-dev/pandas/issues/63191 |
| #63144 | Issue | 2025-11-18 | Open | `astype(copy=...)` docs/type hints describe `bool`, while source supports `None`; this changes behavioral interpretation of defaulting/copy semantics. | https://github.com/pandas-dev/pandas/issues/63144 |
| #63363 | PR | 2025-12-14 | Open | Clarifies deterministic tie behavior in sorting operations so docs match observed runtime outcomes. | https://github.com/pandas-dev/pandas/pull/63363 |
| #63497 | PR | 2025-12-28 | Open | Adds note that `Index.union` with duplicates behaves like a multiset union (max multiplicity), clarifying non-obvious logic. | https://github.com/pandas-dev/pandas/pull/63497 |
| #61781 | Issue | 2025-07-04 | Open | `Series.mask` docs describe alignment against `other`, but behavior aligns `self` and `cond`; this is a direct logic-description error. | https://github.com/pandas-dev/pandas/issues/61781 |
| #62035 | PR | 2025-08-03 | Open | Fixes behavior (`str.rsplit` with regex should raise) and explicitly updates docs, addressing prior behavior/doc inconsistency. | https://github.com/pandas-dev/pandas/pull/62035 |

## Notes

- All listed items are from 2025 or 2026 (strictly after 2024).
- I prioritized entries where documentation affects interpretation of runtime logic, defaults, coercion/alignment semantics, or deprecation behavior.
