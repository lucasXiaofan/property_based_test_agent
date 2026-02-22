# EWM Aggregate Mutant Kill Report (Targeted)

- Generated (UTC): `2026-02-18T13:42:06.791389+00:00`
- Target: `pandas.core.window.ewm.ExponentialMovingWindow.aggregate`

## Mutants

| Mutant ID | Description |
|---|---|
| `M1_SUM_TO_MEAN` | If func == 'sum', force func='mean' before delegation. |
| `M2_TRUNCATE_LIST_FUNCS` | If func is a multi-function list, raise ValueError. |
| `M3_CALLABLE_TYPEERROR` | Raise TypeError for callable funcs instead of pandas behavior. |

## Outcomes

| Test Target | Baseline | Mutant | Outcome | Return Code |
|---|---|---|---|---:|
| `pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py::TestConsistencyProperties::test_aggregate_sum_matches_ewm_sum` | PASS | `M1_SUM_TO_MEAN` | **KILLED** | 1 |
| `pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py::TestConsistencyProperties::test_aggregate_sum_matches_ewm_sum` | PASS | `M2_TRUNCATE_LIST_FUNCS` | **SURVIVED** | 0 |
| `pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py::TestConsistencyProperties::test_aggregate_sum_matches_ewm_sum` | PASS | `M3_CALLABLE_TYPEERROR` | **SURVIVED** | 0 |
| `pandas_bug_finding/ir_test/test_ewm_aggregate.py::TestExplicit::test_explicit__p3__list_of_string_func_names_returns` | PASS | `M1_SUM_TO_MEAN` | **SURVIVED** | 0 |
| `pandas_bug_finding/ir_test/test_ewm_aggregate.py::TestExplicit::test_explicit__p3__list_of_string_func_names_returns` | PASS | `M2_TRUNCATE_LIST_FUNCS` | **KILLED** | 1 |
| `pandas_bug_finding/ir_test/test_ewm_aggregate.py::TestExplicit::test_explicit__p3__list_of_string_func_names_returns` | PASS | `M3_CALLABLE_TYPEERROR` | **SURVIVED** | 0 |
| `pandas_bug_finding/ir_test/test_ewm_aggregate.py::TestImplicit::test_implicit__p11__any_callable_func_raises_attributeerror_ewm` | PASS | `M1_SUM_TO_MEAN` | **SURVIVED** | 0 |
| `pandas_bug_finding/ir_test/test_ewm_aggregate.py::TestImplicit::test_implicit__p11__any_callable_func_raises_attributeerror_ewm` | PASS | `M2_TRUNCATE_LIST_FUNCS` | **SURVIVED** | 0 |
| `pandas_bug_finding/ir_test/test_ewm_aggregate.py::TestImplicit::test_implicit__p11__any_callable_func_raises_attributeerror_ewm` | PASS | `M3_CALLABLE_TYPEERROR` | **KILLED** | 1 |

## Notes

- `INVALID` means baseline failed for that test target, so mutant classification is not trusted.
- Use node-level targets when full file baselines are unstable.
