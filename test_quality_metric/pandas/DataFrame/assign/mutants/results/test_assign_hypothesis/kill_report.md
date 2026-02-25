# Kill Report: pandas.DataFrame.assign

- pytest_file: `pandas_bug_finding/baseline_testing/test_assign_hypothesis.py`
- baseline_passed: True
- killed: 7
- survived: 0
- invalid_baseline: 0
- allow_failing_baseline: True
- mutation_score: 1.000

## Mutants
- `M_C1_no_new_columns` (C1): killed
- `M_C2_skip_overwrite` (C2): killed
- `M_C3_return_self_modified` (C3): killed
- `M_C4_dont_call_callables` (C4): killed
- `M_C5_ignore_noncallables` (C5): killed
- `M_C6_reversed_order` (C6): killed
- `M_C7_all_from_original` (C7): killed
