# Kill Report: pandas.DataFrame.loc

- pytest_file: `pandas_bug_finding/baseline_testing/test_loc_hypothesis.py`
- baseline_passed: True
- killed: 8
- survived: 0
- invalid_baseline: 0
- allow_failing_baseline: True
- mutation_score: 1.000

## Mutants
- `M_C1_use_iloc` (C1): killed
- `M_C2_int_as_position` (C2): killed
- `M_C3_slice_exclusive_stop` (C3): killed
- `M_C4_always_dataframe` (C4): killed
- `M_C5_list_squeeze` (C5): killed
- `M_C6_missing_returns_nan` (C6): killed
- `M_C7_bool_invert` (C7): killed
- `M_C8_callable_no_call` (C8): killed
