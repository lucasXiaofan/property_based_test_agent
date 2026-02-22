# EWM Aggregate: Suite Comparison (Targeted)

- Generated (UTC): `2026-02-18T13:42:06.791389+00:00`
- Target: `pandas.core.window.ewm.ExponentialMovingWindow.aggregate`
- Winner by unique kills: `ir_test`

## Suite Comparison

| Suite | Baseline Status | #Unique Killed | Killed Mutant IDs |
|---|---|---:|---|
| `baseline_testing` | PASS (all targets) | 1 | `M1_SUM_TO_MEAN` |
| `ir_test` | PASS (all targets) | 2 | `M2_TRUNCATE_LIST_FUNCS`, `M3_CALLABLE_TYPEERROR` |

## Cross-Suite IDs

- Killed by both: -
- Killed only by baseline_testing: `M1_SUM_TO_MEAN`
- Killed only by ir_test: `M2_TRUNCATE_LIST_FUNCS`, `M3_CALLABLE_TYPEERROR`
- Invalid in both suites: -

## Interpretation

- At least one suite has passing baselines; killed IDs are meaningful for those targets.
