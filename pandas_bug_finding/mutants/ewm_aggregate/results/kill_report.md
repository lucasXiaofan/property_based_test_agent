# EWM Aggregate Mutant Kill Report (Full Files)

- Generated (UTC): `2026-02-18T13:33:21.407275+00:00`
- Target: `pandas.core.window.ewm.ExponentialMovingWindow.aggregate`

## Mutants

| Mutant ID | Description |
|---|---|
| `M1_SUM_TO_MEAN` | If func == 'sum', force func='mean' before delegation. |
| `M2_TRUNCATE_LIST_FUNCS` | If func is a list, keep only the first function. |
| `M3_CALLABLE_TYPEERROR` | Raise TypeError for callable funcs instead of pandas behavior. |

## Outcomes

| Test Target | Baseline | Mutant | Outcome | Return Code |
|---|---|---|---|---:|
| `pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py` | FAIL | `M1_SUM_TO_MEAN` | **INVALID** | - |
| `pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py` | FAIL | `M2_TRUNCATE_LIST_FUNCS` | **INVALID** | - |
| `pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py` | FAIL | `M3_CALLABLE_TYPEERROR` | **INVALID** | - |
| `pandas_bug_finding/ir_test/test_ewm_aggregate.py` | FAIL | `M1_SUM_TO_MEAN` | **INVALID** | - |
| `pandas_bug_finding/ir_test/test_ewm_aggregate.py` | FAIL | `M2_TRUNCATE_LIST_FUNCS` | **INVALID** | - |
| `pandas_bug_finding/ir_test/test_ewm_aggregate.py` | FAIL | `M3_CALLABLE_TYPEERROR` | **INVALID** | - |

## Notes

- `INVALID` means baseline failed for that test target, so mutant classification is not trusted.
- Use node-level targets when full file baselines are unstable.
