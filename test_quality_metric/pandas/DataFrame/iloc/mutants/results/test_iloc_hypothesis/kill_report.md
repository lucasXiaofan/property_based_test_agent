# Kill Report: pandas.DataFrame.iloc

- pytest_file: `pandas_bug_finding/baseline_testing/test_iloc_hypothesis.py`
- baseline_passed: True
- killed: 10
- survived: 0
- invalid_baseline: 0
- allow_failing_baseline: True
- mutation_score: 1.000

## Mutants
- `M_C1_use_loc` (C1): killed
- `M_C2_single_int_always_dataframe` (C2): killed
- `M_C3_list_squeeze_to_series` (C3): killed
- `M_C4_slice_drop_last` (C4): killed
- `M_C5_bool_invert` (C5): killed
- `M_C6_callable_no_call` (C6): killed
- `M_C7_tuple_ignore_cols` (C7): killed
- `M_C8_no_indexerror` (C8): killed
- `M_C9_slice_raises_indexerror` (C9): killed
- `M_C10_scalar_returns_series` (C10): killed
