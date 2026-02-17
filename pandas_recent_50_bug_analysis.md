# Pandas Recent 50 Bugs Analysis

- Data source: `pandas_bugs_2023_now.csv`
- Selection rule: 50 most recent rows by `createdAt` descending
- Time window covered: 2025-12-31 to 2026-02-07 (UTC)
- Generated file date: 2026-02-12

## Quick Takeaways

- The recent bug stream is concentrated in core data operations (Indexing, IO, Dtypes/Arrays, GroupBy, Time series) and Arrow-backend compatibility.
- Multiple regressions in 3.0.0 appear in indexing/reindex/groupby/dtype paths, which are good targets for property-based tests.
- IO-related bugs are frequent but often depend on external formats/backends, making them noisier for automated bug-finding pipelines.

## Module Distribution (Recent 50)

| Module | Count |
|---|---:|
| IO | 12 |
| Dtypes/Arrays | 6 |
| Time series | 5 |
| Strings | 5 |
| Indexing | 3 |
| Build/Packaging | 3 |
| GroupBy | 3 |
| Arrow backend | 2 |
| Stability/Crash | 2 |
| Memory semantics | 2 |
| Needs triage/Unknown | 2 |
| Other/Core | 1 |
| Expressions | 1 |
| Testing/Assertions | 1 |
| Sorting | 1 |
| Concurrency | 1 |

## Frequent Labels (Recent 50)

| Label | Count |
|---|---:|
| Bug | 50 |
| Needs Triage | 13 |
| Arrow | 8 |
| Strings | 5 |
| Regression | 4 |
| Upstream issue | 3 |
| Indexing | 3 |
| Groupby | 3 |
| IO Data | 3 |
| Dtype Conversions | 2 |
| Copy / view semantics | 2 |
| Missing-data | 2 |
| Apply | 2 |
| IO CSV | 2 |
| Duplicate Report | 1 |

## Bug-by-Bug Summary (Recent 50)

| # | Created | Module | Bug Summary | Labels |
|---:|---|---|---|---|
| 64060 | 2026-02-07 | Arrow backend | pandas 3.0.0 and pyarrow 21.0.0, pd.merge left join with 3 keys incorrectly matches a NaN key with a non-Na... | Bug;Arrow |
| 64055 | 2026-02-06 | IO | pandas read_sas not working anymore | Bug;Needs Triage |
| 64047 | 2026-02-05 | Other/Core | Reindexing with fill_value | Bug;Duplicate Report |
| 64044 | 2026-02-05 | Time series | pd.to_timedelta unit inconsistently handled | Bug;Regression;Timedelta |
| 64023 | 2026-02-04 | Stability/Crash | segfault in test when using pandas with freezegun | Bug;Needs Triage |
| 64012 | 2026-02-03 | Time series | to_datetime does not recast units when passing a DatetimeIndex | Bug;Needs Triage |
| 64007 | 2026-02-03 | IO | Index is incorrectly de-serialised by fastparquet after mulitple parquets written to different io.BytesIO s... | Bug;IO Parquet;Upstream issue |
| 64006 | 2026-02-03 | IO | pd.read_hdf` unable to retrieve a pd.Series with dtype as "datetime64" since v3.0.0 | Bug;IO HDF5;Non-Nano |
| 63993 | 2026-02-02 | Indexing | DataFrame.reindex` with multiple `columns` and a string `fill_value` raises `AssertionError` with `3.0.0 | Bug;Indexing;Regression |
| 63939 | 2026-01-29 | Expressions | unary operators not supported with `pd.col | Bug;expressions |
| 63938 | 2026-01-29 | Dtypes/Arrays | Odd behavior of `Index.searchsorted | Bug;Dtype Conversions |
| 63936 | 2026-01-29 | Memory semantics | refs not set when creating DataFrame from Series/Index with StringDtype | Bug;Copy / view semantics |
| 63935 | 2026-01-29 | Strings | 3.0.0 Regression - can no longer shuffle `Series[string]` with numpy | Bug;Strings |
| 63921 | 2026-01-28 | Build/Packaging | import pandas ImportError: DLL load failed while importing base | Bug;Needs Triage |
| 63920 | 2026-01-28 | GroupBy | groupby with observed=False assigns rows with NaN categorical keys to wrong groups | Bug;Groupby;Missing-data;Regression;Categorical |
| 63913 | 2026-01-28 | GroupBy | Cannot access groups when using `.groupby(dropna=False) | Bug;Groupby;Missing-data |
| 63904 | 2026-01-27 | Testing/Assertions | assert_frame_equal changes behavior in 3.0 for nested arrays without documentation | Bug;Needs Triage |
| 63903 | 2026-01-27 | Dtypes/Arrays | Pandas converts nullable int to float, even when this loses data | Bug;Dtype Conversions;Apply |
| 63899 | 2026-01-27 | Memory semantics | DataFrame derived from Index can mutate the Index | Bug;Copy / view semantics |
| 63889 | 2026-01-26 | IO | Handling of `None` column name in `from_records()` is now all-NaN instead of values | Bug;IO Data;Regression |
| 63879 | 2026-01-26 | Dtypes/Arrays | Conversion from numpy masked arrays to pandas array does not preserve missing values | Bug;Constructors |
| 63876 | 2026-01-25 | Strings | AttributeError: 'ArrowStringArray' object has no attribute 'item' | Bug;Strings;ExtensionArray |
| 63855 | 2026-01-24 | Time series | ExponentialMovingWindow.agg/aggregate fails when passing a callable | Bug;Window |
| 63842 | 2026-01-24 | Strings | DataFrame[StringDtype].where(DataFrame[bool], list[str])` returns object type instead of StringDtype. | Bug;Strings |
| 63832 | 2026-01-23 | Strings | bug in pandas 3.0 with posixpath inside Series | Bug;Strings;Arrow |
| 63830 | 2026-01-23 | IO | read_excel with pyarrow backend does not infer correct data type with mixed values inside column | Bug;IO Excel;Arrow |
| 63791 | 2026-01-21 | Time series | inferred_freq is infers the wrong freq 2BQS-OCT instead of 2BQS-JAN | Bug;Frequency |
| 63790 | 2026-01-21 | GroupBy | sort_values() does not retain Name property of Dataframe | Bug;Groupby;Apply |
| 63787 | 2026-01-21 | IO | Inconsistent BOM handling in pd.read_csv with encoding='utf-8' | Bug;IO CSV |
| 63774 | 2026-01-20 | IO | DataFrame.from_records` ignores exclude argument when `nrows=0 | Bug;IO Data |
| 63769 | 2026-01-20 | Time series | rolling on time-based index wrongly | Bug;Needs Triage |
| 63757 | 2026-01-19 | Needs triage/Unknown | (empty issue title in dataset) | Bug;Needs Triage |
| 63745 | 2026-01-18 | Arrow backend | better error message when doing `convert_dtypes(dtype_backend="pyarrow")` without pyarrow installed | Bug;Error Reporting;Arrow |
| 63739 | 2026-01-18 | Strings | str methods fail with regex backreferences on PyArrow dtype | Bug;Strings;Arrow |
| 63735 | 2026-01-18 | Sorting | sort_values fails on columns with more than 2^18 entries | Bug;Upstream issue;Sorting |
| 63732 | 2026-01-18 | Dtypes/Arrays | Inconsistent NaN detection based on Series dtype, to_numeric dtype_backend, None vs np.nan, strings in orig... | Bug;Needs Triage |
| 63708 | 2026-01-16 | IO | pd.read_spss()` generates `attrs` that cause `DataFrame.to_parquet` to crash when generating JSON metadata. | Bug;IO Data;Upstream issue;Arrow |
| 63686 | 2026-01-14 | Build/Packaging | mesonbuild versioning in environment.yml | Bug;Needs Triage |
| 63685 | 2026-01-14 | Concurrency | data races when simultaneously reading data frames in multiple threads | Bug;Free Threading |
| 63681 | 2026-01-14 | IO | ExcelWriter can't use ISFORMULA in conditional formatting | Bug;Needs Triage |
| 63650 | 2026-01-12 | Stability/Crash | pd.to_numeric Segmentation fault | Bug;Needs Discussion |
| 63640 | 2026-01-11 | Build/Packaging | Missing explicit libm dependency in meson.build causes undefined symbol errors | Bug;Build |
| 63630 | 2026-01-10 | IO | Reading from CSV files with mixed numbers and letters leads to Segmentation Fault | Bug;IO CSV |
| 63604 | 2026-01-07 | Needs triage/Unknown | DataFrame from an array of dictionaries, has missing columns | Bug;Needs Triage |
| 63581 | 2026-01-06 | Dtypes/Arrays | iloc[0] raises ValueError with SparseArray column and numpy array column | Bug;Sparse;Closing Candidate |
| 63574 | 2026-01-05 | IO | pd.read_sql returns fewer rows than manual cursor execution for the same query and connection | Bug;IO SQL;Needs Triage |
| 63572 | 2026-01-05 | IO | pd.read_json fails with "Value is too big!" on large integers, while json.load + DataFrame works | Bug;IO JSON |
| 63567 | 2026-01-04 | Dtypes/Arrays | pd.array([float("nan")], str)` does not convert to `pd.NA | Bug;Needs Triage |
| 63527 | 2025-12-31 | Indexing | PyArrow-backed datetime index missing numpy-like attributes | Bug;Indexing;Arrow |
| 63526 | 2025-12-31 | Indexing | .loc` slicing fails on PyArrow-backed date index | Bug;Indexing;Arrow |

## Recommendation For Your Research Pipeline

### Which module to start with

Start with **Indexing** (then GroupBy as phase 2).

### Why Indexing first

- High bug relevance: recent issues include `.loc` slicing, `reindex(fill_value)`, and Arrow-backed index edge cases.
- Good fit for NL -> IR -> test: indexing APIs have clear preconditions and expected invariants (ordering, boundary behavior, NA handling).
- Lower false positives than IO modules: most indexing tests are pure in-memory and do not require external files/backends.
- Strong property templates: slice monotonicity, alignment consistency, idempotence for repeated reindexing, and NA-key handling are property-friendly.

### Suggested intermediate representation (IR) fields for Indexing

- `api`: e.g., `DataFrame.loc`, `DataFrame.reindex`, `Index.searchsorted`
- `input_schema`: index type, dtype, monotonicity, uniqueness, NA distribution
- `operation`: slice/reindex/lookup with exact parameters
- `oracle_type`: equality/order/exception/non-crash/invariant
- `metamorphic_relations`: repeated reindex stability, equivalent key forms, sorting-preserving relations
- `backend`: numpy vs pyarrow when relevant

### Phase plan

1. Indexing core (no IO, no optional deps).
2. GroupBy with missing-data and categorical keys.
3. Dtypes/Arrays conversions.
4. IO modules after stabilizing environment controls.