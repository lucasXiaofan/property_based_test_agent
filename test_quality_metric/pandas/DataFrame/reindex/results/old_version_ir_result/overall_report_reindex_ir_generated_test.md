# Overall Quality Report: pandas.DataFrame.reindex

- Generated at: 2026-02-25T02:56:09.552841+00:00
- Target dir: `test_quality_metric/pandas/DataFrame/reindex`
- Test file: `/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/ir2test_pipeline/pandas/DataFrame/reindex/reindex_ir_generated_test.py`
- Mutant eval exit code: 2
- Coverage exit code: 1
- LLM trace exit code: 0
- Properties coverage exit code: 0

## Mutant Kill Summary
- Killed: 16
- Survived: 4
- Invalid baseline: 0
- Mutation score: 0.800

## Coverage Summary
- Method count: 2
- Average method line coverage: 86.37%

### Method Coverage
- `pandas.DataFrame.reindex`: 1/1 lines (100.00%)
- `pandas.core.generic.NDFrame.reindex`: 16/22 lines (72.73%)

### Branch Coverage (overall)
- Branch coverage: 85/1960 (4.34%)

## Properties Coverage (LLM)
- Total clauses: 20
- Satisfied clauses: 17
- Unsatisfied clauses: 3
- Properties coverage: 0.850
- Avg confidence (satisfied): 1.000
- Avg confidence (unsatisfied): 1.000

### Clause Trace
- `C1`: satisfied by ['TestValidContracts::test_p1_identity_labels_preserve_data', 'TestValidContracts::test_p6_keyword_identity_index_and_columns'] (confidence=1.00)
- `C2`: satisfied by ['TestValidContracts::test_p3_disjoint_rows_default_to_nan', 'TestValidContracts::test_p4_partial_overlap_rows_preserve_matches_and_nan_for_new', 'TestValidContracts::test_p5_partial_overlap_columns', 'TestValidContracts::test_p9_disjoint_columns_default_to_nan', 'TestValidContracts::test_p20_disjoint_labels_axis_index_all_nan'] (confidence=1.00)
- `C3`: satisfied by ['TestValidContracts::test_p16_numeric_fill_value_applied_to_new_rows'] (confidence=1.00)
- `C4`: satisfied by ['TestValidContracts::test_p17_string_fill_value_applied_to_new_rows'] (confidence=1.00)
- `C5`: satisfied by ['TestValidContracts::test_p11_ffill_on_monotonic_index'] (confidence=1.00)
- `C6`: satisfied by ['TestValidContracts::test_p12_bfill_on_monotonic_index'] (confidence=1.00)
- `C7`: satisfied by ['TestValidContracts::test_p13_nearest_respects_scalar_tolerance', 'TestValidContracts::test_p19_nearest_listlike_tolerance_is_elementwise'] (confidence=1.00)
- `C8`: satisfied by ['TestInvalidContracts::test_e7_method_with_non_monotonic_index_raises'] (confidence=1.00)
- `C9`: unsatisfied (confidence=1.00)
- `C10`: satisfied by ['TestValidContracts::test_p18_ffill_with_limit_caps_consecutive_fills'] (confidence=1.00)
- `C11`: satisfied by ['TestValidContracts::test_p6_keyword_identity_index_and_columns'] (confidence=1.00)
- `C12`: satisfied by ['TestValidContracts::test_p1_identity_labels_preserve_data', 'TestValidContracts::test_p10_labels_with_axis_columns_targets_columns', 'TestValidContracts::test_p20_disjoint_labels_axis_index_all_nan'] (confidence=1.00)
- `C13`: satisfied by ['TestValidContracts::test_p1_identity_labels_preserve_data'] (confidence=1.00)
- `C14`: satisfied by ['TestValidContracts::test_p5_partial_overlap_columns', 'TestValidContracts::test_p9_disjoint_columns_default_to_nan', 'TestValidContracts::test_p10_labels_with_axis_columns_targets_columns'] (confidence=1.00)
- `C15`: satisfied by ['TestValidContracts::test_p13_nearest_respects_scalar_tolerance'] (confidence=1.00)
- `C16`: satisfied by ['TestInvalidContracts::test_e9_list_tolerance_wrong_size_raises'] (confidence=1.00)
- `C17`: unsatisfied (confidence=1.00)
- `C18`: satisfied by ['TestValidContracts::test_p15_multiindex_level_reindex_preserves_structure'] (confidence=1.00)
- `C19`: satisfied by ['TestValidContracts::test_p14_copy_true_false_same_logical_result'] (confidence=1.00)
- `C20`: unsatisfied (confidence=1.00)
