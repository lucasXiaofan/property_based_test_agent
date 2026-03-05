# Kill Report — Traditional Mutants: pandas.DataFrame.reindex

- **pytest file**: `/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/ir2test_pipeline/pandas/DataFrame/reindex/baseline_test.py`
- **baseline passed**: False
- **mutation score**: 100.0%  (15 killed / 15 total)

## Results by Operator

| Operator | Killed | Survived |
|----------|--------|----------|
| AOR | 1 | 0 |
| COR | 1 | 0 |
| ROR | 4 | 0 |
| SDL | 8 | 0 |
| SVR | 1 | 0 |

## Per-Mutant Results

| Mutant ID | Operator | Status | Behavior Broken |
|-----------|----------|--------|----------------|
| `SDL_index` | SDL | ✅ killed | Row reindexing via index= keyword is silently ignored; DataFrame keeps original  |
| `SDL_columns` | SDL | ✅ killed | Column reindexing via columns= keyword is silently ignored; DataFrame keeps orig |
| `SDL_method` | SDL | ✅ killed | Fill method (ffill/bfill/nearest) is dropped; all gaps remain NaN instead of bei |
| `SDL_fill_value` | SDL | ✅ killed | Custom fill_value is never forwarded; missing positions always receive NaN. |
| `SDL_limit` | SDL | ✅ killed | limit= is never forwarded; fill continues without restriction past the requested |
| `SDL_tolerance` | SDL | ✅ killed | tolerance= is never forwarded; inexact matches succeed regardless of distance. |
| `SDL_level` | SDL | ✅ killed | level= is never forwarded; MultiIndex level broadcast is lost. |
| `SDL_labels_block` | SDL | ✅ killed | Positional labels are silently dropped; reindex(labels, axis='columns') is a no- |
| `ROR_axis_eq` | ROR | ✅ killed | Axis routing is reversed: labels intended for rows go to columns and vice versa. |
| `ROR_labels_none` | ROR | ✅ killed | Labels routing fires only when labels IS None; actual labels passed are silently |
| `ROR_index_none` | ROR | ✅ killed | Index is forwarded only when it IS None (no-op); caller-supplied index values ar |
| `ROR_columns_none` | ROR | ✅ killed | Columns are forwarded only when they IS None; caller-supplied columns are ignore |
| `COR_fill_sentinel` | COR | ✅ killed | fill_value forwarded only when it IS NaN (sentinel); custom values dropped. The  |
| `AOR_limit_plus1` | AOR | ✅ killed | limit is inflated by 1 (limit + 1); one extra consecutive fill is performed beyo |
| `SVR_axis_default` | SVR | ✅ killed | Default axis when axis=None is changed from 0 (rows) to 1 (columns); positional  |

## Why Traditional Mutants Test Different Behaviors

Each mutant removes or inverts exactly one decision in the Python wrapper
around `pd.DataFrame.reindex`.  A test that exercises the affected parameter
will detect the mutation; tests that never exercise that parameter will not.

| Operator | Code-level change | Behavioral impact |
|----------|-------------------|-------------------|
| SDL | Delete a statement | Silently drops one kwarg | Param never forwarded |
| ROR | Flip == / != | Wrong branch taken | Axis/routing inverted |
| COR | Invert 'not' | Guard condition reversed | Sentinel check flipped |
| AOR | ±1 on constant | Off-by-one | Limit too permissive |
| SVR | Change literal value | Wrong default | Default axis wrong |

Survival means tests don't cover that parameter path; killing means
they do.
