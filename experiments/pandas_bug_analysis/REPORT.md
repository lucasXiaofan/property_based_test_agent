# Pandas 3.0 Bug Analysis

## Scope

- Source of truth for issue discovery: `gh search issues` and `gh issue view`.
- Local execution environment: `pandas==3.0.0`, Python 3.11.14.
- Counted set: 20 pandas library bugs with a maintainer confirmation signal.
- Excluded from the counted set: performance issues, dependency/upstream issues, docs/build/test-only bugs.
- Extra local check: one supplemental issue (`#58190`) that reproduces on `3.0.0` but did not have a strong enough confirmation signal in the fetched thread to count toward the 20.

## Headline Result

- Counted valid bugs analyzed: 20
- Counted bugs still reproducible on local `pandas 3.0.0`: 3
- Counted bugs not reproducible on local `pandas 3.0.0`: 17
- Supplemental reproduced-but-uncounted bug: 1 (`#58190`)

The main pattern is that many bugs were reported during the 3.0 development or release-candidate cycle, but the final `3.0.0` wheel already contains fixes for most of them. The three counted issues that still reproduce locally are `#62888`, `#63879`, and `#63993`.

## Still Reproducible On Local 3.0.0

| Issue | Local status | Input pattern | Hypothesis sketch |
| --- | --- | --- | --- |
| `#62888` | reproduced | `object` Series mixes `0`, `1`, `True`, `False`; `factorize()` collapses them into two uniques | `st.permutations([0, 1, True, False]).map(lambda xs: pd.Series(list(xs)))` |
| `#63879` | reproduced | `pd.array(np.ma.array(...))` ignores the mask and keeps masked integers as concrete values | `st.lists(st.integers(), min_size=2, max_size=6).flatmap(lambda xs: st.tuples(st.just(xs), st.lists(st.booleans(), min_size=len(xs), max_size=len(xs))))` |
| `#63993` | reproduced | `DataFrame.reindex(columns=[...], fill_value="missing")` crashes when more than one new column is added | `st.lists(st.text(min_size=1), min_size=2, max_size=5, unique=True).map(lambda extra: (pd.DataFrame({'a': [0]}), ['a'] + extra, 'missing'))` |

## Supplemental Reproduction

| Issue | Why it is not counted | Local status | Hypothesis sketch |
| --- | --- | --- | --- |
| `#58190` | the fetched thread lacked an explicit maintainer confirmation comment | reproduced | `st.just((pd.DataFrame([[0.0, 0.5, 0.0], [0.1, 0.0, 0.2], [0.2, 0.0, 0.0]]), pd.Series([1.0, 1.0, np.nan])))` |

## Counted Issue Catalog

| Issue | Confirmation signal | Local 3.0.0 result | Input summary | Hypothesis sketch |
| --- | --- | --- | --- | --- |
| `#58471` | maintainer comment by `rhshadrach` during triage | not reproduced | concat non-overlapping `Series` with non-`ns` `DatetimeIndex` values | `st.integers(min_value=2, max_value=4).map(lambda days: [pd.Series(range(288), index=pd.date_range('2024-01-01', periods=288, freq='5min', unit='us') + pd.Timedelta(days=i)) for i in range(days)])` |
| `#59965` | maintainer comments by `rhshadrach` and `jorisvandenbossche` | not reproduced | nullable floating arrays that contain `NaN`, then `mean(skipna=True)` | `st.lists(st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.integers()), min_size=4, max_size=4).map(lambda xs: pd.Series(xs).convert_dtypes())` |
| `#60922` | maintainer comment by `rhshadrach` | not reproduced | axis-1 concat where the right frame is a prefix on a non-`ns` datetime index | `st.integers(min_value=3, max_value=8).map(lambda n: (pd.date_range('2025-01-29 01:36', periods=n, freq='1min', unit='us'), n - 1))` |
| `#61099` | maintainer comment by `rhshadrach` | not reproduced | compare identical labels when one index is `object` and the other is pandas `string` | `st.lists(st.text(min_size=1), min_size=1, max_size=5, unique=True).map(lambda labels: (pd.Series(range(len(labels)), index=labels), pd.Index(labels).astype('string')))` |
| `#61175` | maintainer comment by `snitish`: "Confirmed on main." | not reproduced | `pd.eval("(x + y).dropna()")` on misaligned `Series` | `st.tuples(st.lists(st.integers(), min_size=2, max_size=6), st.lists(st.integers(), min_size=1, max_size=5)).map(lambda pair: (pd.Series(pair[0]), pd.Series(pair[1])))` |
| `#61356` | maintainer comment by `rhshadrach`: "Confirmed on main. PR to fix is up." | not reproduced | groupby on `Categorical` data containing `NaN` with `dropna=False` | `st.just(pd.DataFrame({'cat': pd.Categorical(['a', np.nan, 'a'], categories=['a', 'b', 'd']), 'vals': [1, 2, 3]}))` |
| `#61509` | maintainer comment by `rhshadrach`: "Confirmed on main..." | not reproduced | `pivot_table(..., margins=True)` where one grouping key contains `None`/`NaN` | `st.just(pd.DataFrame({'i': [1, 2, 3], 'g1': ['a', 'b', 'b'], 'g2': ['x', None, None]}))` |
| `#61621` | maintainer comment by `arthurlw`: "Confirmed on main!" | not reproduced | `infer_dtype` on object floats with trailing `pd.NA` vs trailing `np.nan` | `st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=5).map(lambda xs: (pd.Series(xs + [pd.NA], dtype=object), pd.Series(xs + [np.nan], dtype=object)))` |
| `#62094` | maintainer comment by `jbrockmendel` identifies the missing freq propagation | not reproduced | computed `TimedeltaIndex` with no stored `freq`, then `shift(1)` | `st.dates(min_value=pd.Timestamp('2000-01-01').date(), max_value=pd.Timestamp('2030-01-01').date()).map(lambda d: pd.date_range(d, periods=5) - pd.Timestamp('2019-01-03'))` |
| `#62240` | maintainer discussion by `rhshadrach` and `jorisvandenbossche`, plus linked PR | not reproduced | compiled regex with flags used in `str.match` and `str.contains` | `st.just((re.compile('foo', flags=re.IGNORECASE), pd.Series(['Foo', 'foo', 'Bar', '_Foo_', '_foo_'])))` |
| `#62595` | maintainer discussion by `jbrockmendel` | not reproduced | compare bool-multiplication behavior between `string[python]` and generic `string` | `st.just(pd.Series(['a', 'b', 'c']))` |
| `#62778` | maintainer-authored issue by `mroeschke` | not reproduced | pass a truthy non-bool to `numeric_only` in `GroupBy.mean` | `st.sampled_from([['B'], [1], 'yes']).map(lambda bad: (pd.DataFrame({'A': range(5), 'B': range(5)}), bad))` |
| `#62829` | maintainer comment by `rhshadrach` on error handling for mixed dict/`NaN` inputs | not reproduced | `json_normalize(..., max_level=0)` on a list mixing dicts and `NaN` | `st.lists(st.one_of(st.dictionaries(st.text(min_size=1), st.integers()), st.just(np.nan)), min_size=2, max_size=5)` |
| `#62888` | maintainer discussion by `rhshadrach` and `jbrockmendel` | reproduced | `factorize()` on object values mixing ints and bools | `st.permutations([0, 1, True, False]).map(lambda xs: pd.Series(list(xs)))` |
| `#63236` | maintainer comments by `WillAyd`, `jbrockmendel`, and `jorisvandenbossche` with a fix PR | not reproduced | `to_json()` on a frame whose columns are a non-`ns` `TimedeltaIndex` | `st.sampled_from(['us', 'ms', 's']).map(lambda unit: pd.DataFrame([[1]], columns=[pd.Timedelta('1D').as_unit(unit)]))` |
| `#63262` | maintainer comments by `jorisvandenbossche` and `jbrockmendel` | not reproduced | `Series.loc[start:stop]` where `start` and `stop` use different timestamp units | `st.just((pd.Series(1, index=pd.date_range('2000-01-01', periods=8, freq='h')), pd.Timestamp('2000-01-01 01:00:00')))` |
| `#63306` | maintainer comments by `mroeschke` and `jorisvandenbossche` | not reproduced | CoW write into a `Series` built from read-only categorical backing data | `st.just((pd.Index([0, 1, 2, 3], dtype='int8').to_numpy(), pd.Index(['a', 'b', 'c', 'd'])))` |
| `#63581` | maintainer comment by `rhshadrach` acknowledging the bug for pandas 3.x | not reproduced | `iloc[0]` on a row that mixes ndarray-valued cells and a `SparseArray` column | `st.just(pd.DataFrame({'id': ['A', 'B'], 'arr': [np.array([1.0, 2.0]), np.array([3.0, 4.0])]}))` |
| `#63879` | maintainer comment by `rhshadrach` | reproduced | `pd.array()` on a masked ndarray ignores the mask | `st.lists(st.integers(), min_size=2, max_size=6).flatmap(lambda xs: st.tuples(st.just(xs), st.lists(st.booleans(), min_size=len(xs), max_size=len(xs))))` |
| `#63993` | maintainer comment by `rhshadrach`: "confirmed on main." | reproduced | `reindex` with a string fill value and at least two missing output columns | `st.lists(st.text(min_size=1), min_size=2, max_size=5, unique=True).map(lambda extra: (pd.DataFrame({'a': [0]}), ['a'] + extra, 'missing'))` |

## Artifacts

- Harness: `experiments/pandas_bug_analysis/run_reproductions.py`
- Materialized results: `experiments/pandas_bug_analysis/results.json`

