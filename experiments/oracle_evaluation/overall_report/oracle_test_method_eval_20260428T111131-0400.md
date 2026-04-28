# Oracle Test Method Evaluation

- Generated at: 2026-04-28T11:11:31.436167-04:00
- Target directory: `/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/experiments/oracle_generation/pandas`
- Cases evaluated: 10

## Overall Ranking

| Rank | Method | Score | Line | Branch | Mutant kill | Diff from best |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `improved_baseline` | 83.87 | 85.4 | 77.32 | 87.5 | 0.0 |
| 2 | `ir_enhanced` | 76.91 | 74.25 | 75.53 | 83.33 | 6.96 |
| 3 | `baseline` | 67.08 | 70.63 | 70.6 | 60.0 | 16.79 |

## Case Ranking

- `DataFrame/groupby` winner `baseline`: baseline 83.33 (-0.0), ir_enhanced 83.33 (-0.0), improved_baseline n/a (-n/a)
- `DataFrame/reindex` winner `improved_baseline`: improved_baseline 59.76 (-0.0), ir_enhanced 59.6 (-0.16), baseline 56.4 (-3.36)
- `DataFrame/to_json` winner `improved_baseline`: improved_baseline 100.0 (-0.0), ir_enhanced 86.54 (-13.46), baseline 42.31 (-57.69)
- `Index/astype` winner `improved_baseline`: improved_baseline 89.41 (-0.0), ir_enhanced 82.16 (-7.25), baseline 58.24 (-31.17)
- `Index/shift` winner `improved_baseline`: improved_baseline 100.0 (-0.0), baseline 66.67 (-33.33), ir_enhanced 66.67 (-33.33)
- `Series/factorize` winner `baseline`: baseline 62.39 (-0.0), improved_baseline 62.39 (-0.0), ir_enhanced 62.39 (-0.0)
- `Series/mean` winner `baseline`: baseline 100.0 (-0.0), improved_baseline 100.0 (-0.0), ir_enhanced 100.0 (-0.0)
- `Series/mul` winner `baseline`: baseline 100.0 (-0.0), improved_baseline 100.0 (-0.0), ir_enhanced 100.0 (-0.0)
- `Series.str/contains` winner `improved_baseline`: improved_baseline 58.33 (-0.0), ir_enhanced 58.33 (-0.0), baseline 41.67 (-16.66)
- `Series.str/match` winner `improved_baseline`: improved_baseline 84.92 (-0.0), ir_enhanced 70.11 (-14.81), baseline 59.79 (-25.13)
