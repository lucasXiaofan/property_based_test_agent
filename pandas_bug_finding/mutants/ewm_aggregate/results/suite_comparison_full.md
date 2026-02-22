# EWM Aggregate: Suite Comparison (Full Files)

- Generated (UTC): `2026-02-18T13:33:21.407275+00:00`
- Target: `pandas.core.window.ewm.ExponentialMovingWindow.aggregate`
- Winner by unique kills: `tie`

## Suite Comparison

| Suite | Baseline Status | #Unique Killed | Killed Mutant IDs |
|---|---|---:|---|
| `baseline_testing` | FAIL (all targets) | 0 | - |
| `ir_test` | FAIL (all targets) | 0 | - |

## Cross-Suite IDs

- Killed by both: -
- Killed only by baseline_testing: -
- Killed only by ir_test: -
- Invalid in both suites: `M1_SUM_TO_MEAN`, `M2_TRUNCATE_LIST_FUNCS`, `M3_CALLABLE_TYPEERROR`

## Interpretation

- Both suites failed baseline; full-file mutant verdicts are not trustworthy.
