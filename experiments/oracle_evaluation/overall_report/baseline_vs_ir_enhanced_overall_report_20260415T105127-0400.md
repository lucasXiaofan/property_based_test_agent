# Baseline vs IR Enhanced Oracle Evaluation

- Generated at: 2026-04-15T10:51:27.822038-04:00
- Scan root: `/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/experiments/oracle_generation/pandas`
- Cases compared: 10

## Line and Branch Coverage

- Average line coverage winner: `ir_enhanced`
- Baseline average line coverage: 70.63%
- IR enhanced average line coverage: 74.25%
- Average branch coverage winner: `ir_enhanced`
- Baseline average branch coverage: 70.6%
- IR enhanced average branch coverage: 75.53%

## Mutant Kill

- Kill rate winner: `ir_enhanced`
- Baseline overall kill rate: 60.0%
- IR enhanced overall kill rate: 77.78%
- Total mutants tested: 20

## Case Winners

- `DataFrame/groupby`: coverage overall `tie`, line `tie`, branch `tie`, mutant kill `tie`
- `DataFrame/reindex`: coverage overall `tie`, line `tie`, branch `tie`, mutant kill `baseline`
- `DataFrame/to_json`: coverage overall `ir_enhanced`, line `ir_enhanced`, branch `ir_enhanced`, mutant kill `ir_enhanced`
- `Index/astype`: coverage overall `ir_enhanced`, line `ir_enhanced`, branch `ir_enhanced`, mutant kill `ir_enhanced`
- `Index/shift`: coverage overall `tie`, line `tie`, branch `tie`, mutant kill `tie`
- `Series/factorize`: coverage overall `tie`, line `tie`, branch `tie`, mutant kill `tie`
- `Series/mean`: coverage overall `tie`, line `tie`, branch `tie`, mutant kill `tie`
- `Series/mul`: coverage overall `tie`, line `tie`, branch `tie`, mutant kill `tie`
- `Series.str/contains`: coverage overall `tie`, line `tie`, branch `tie`, mutant kill `tie`
- `Series.str/match`: coverage overall `ir_enhanced`, line `ir_enhanced`, branch `ir_enhanced`, mutant kill `tie`
