# Kill Report: pandas.DataFrame.reindex

- pytest_file: `/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/ir2test_pipeline/pandas/DataFrame/reindex/reindex_ir_generated_test.py`
- baseline_passed: False
- killed: 16
- survived: 4
- invalid_baseline: 0
- allow_failing_baseline: True
- mutation_score: 0.800

## Mutants
- `M_C1_not_conformed_return` (C1): killed
- `M_C2_missing_labels_not_nan` (C2): killed
- `M_C3_ignore_fill_value` (C3): killed
- `M_C4_fill_value_numeric_only` (C4): killed
- `M_C5_ffill_backward` (C5): killed
- `M_C6_bfill_forward` (C6): killed
- `M_C7_nearest_as_ffill` (C7): killed
- `M_C8_method_without_monotonicity` (C8): killed
- `M_C9_fill_existing_nan` (C9): killed
- `M_C10_ignore_limit` (C10): killed
- `M_C11_break_index_columns_convention` (C11): survived
- `M_C12_break_labels_axis_convention` (C12): survived
- `M_C13_axis_int_rejected` (C13): killed
- `M_C14_columns_missing_not_nan` (C14): killed
- `M_C15_scalar_tolerance_not_uniform` (C15): killed
- `M_C16_list_tolerance_len_dtype_ignored` (C16): killed
- `M_C17_tolerance_inequality_relaxed` (C17): killed
- `M_C18_ignore_level_param` (C18): killed
- `M_C19_copy_changes_behavior` (C19): survived
- `M_C20_force_materialize_non_index` (C20): survived
