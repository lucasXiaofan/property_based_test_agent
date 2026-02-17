# Recent Validated Non-Regression Bug Analysis (Pandas)

This report was built using shell + `gh` on `pandas-dev/pandas` and filtered to recent bug issues that are **not** labeled `Regression`.

## Commands used

```sh
# 1) recent bug candidates
gh search issues --repo pandas-dev/pandas --limit 40 \
  --json number,title,url,createdAt,state,labels "label:Bug"

# 2) inspect selected issues in detail
gh issue view <id> --repo pandas-dev/pandas \
  --json number,title,url,createdAt,state,labels,body
```

## Selection criteria

- Has `Bug` label
- Does **not** have `Regression` label
- Recent (2026-01 to 2026-02)
- Has reproducible example and clear behavior mismatch in the issue body

## 10 issues

| Issue | Created (UTC) | URL | Description |
|---|---|---|---|
| #64123 | 2026-02-12 | https://github.com/pandas-dev/pandas/issues/64123 | `Series.str.find` gives different index results by dtype (`object` vs `str`), appears byte-count vs char-count mismatch for multibyte chars. |
| #64060 | 2026-02-07 | https://github.com/pandas-dev/pandas/issues/64060 | `pd.merge` left join with 3 keys can wrongly match `NaN` key to non-`NaN` value (reported with `pyarrow==21.0.0`). |
| #64007 | 2026-02-03 | https://github.com/pandas-dev/pandas/issues/64007 | `read_parquet` (fastparquet path) can deserialize index incorrectly after writing multiple DataFrames to separate `BytesIO` streams. |
| #64006 | 2026-02-03 | https://github.com/pandas-dev/pandas/issues/64006 | `pd.read_hdf` in pandas 3.0 fails reading older `datetime64` Series payloads due to unit handling (`generic` vs expected time units). |
| #63938 | 2026-01-29 | https://github.com/pandas-dev/pandas/issues/63938 | `Index.searchsorted` behaves oddly with large `int64` values plus nullable `Int64` input (`[0,4]` unexpected). |
| #63936 | 2026-01-29 | https://github.com/pandas-dev/pandas/issues/63936 | CoW ref tracking bug: DataFrame from `Series/Index` with `dtype="str"` shares memory but refs are not marked correctly. |
| #63903 | 2026-01-27 | https://github.com/pandas-dev/pandas/issues/63903 | `.apply()` with nullable ints and missing values can coerce to float and lose precision for large integers. |
| #63879 | 2026-01-26 | https://github.com/pandas-dev/pandas/issues/63879 | Converting NumPy masked arrays via `pd.array` ignores mask; `pd.Series` handles mask but may alter dtype unexpectedly. |
| #63855 | 2026-01-24 | https://github.com/pandas-dev/pandas/issues/63855 | `ExponentialMovingWindow.aggregate` fails for callable (`sum`, `np.sum`) while string form works (`"sum"`). |
| #63787 | 2026-01-21 | https://github.com/pandas-dev/pandas/issues/63787 | `read_csv(..., encoding="utf-8")` BOM handling is inconsistent/underdocumented depending on parsing path. |

## Recommendation: where to start for LLM-generated bug-finding code

Start with **#63855**: https://github.com/pandas-dev/pandas/issues/63855

Why this is the best starter case:

- Minimal reproducible example (few lines)
- No external files or heavy dependencies
- Clear pass/fail oracle:
  - `df.ewm(1).aggregate("sum")` works
  - `df.ewm(1).aggregate(np.sum)` fails
- Fast to run repeatedly for prompt/test iteration
- Good target for evaluating whether an LLM can generate precise failing tests and isolate dispatch-path bugs
