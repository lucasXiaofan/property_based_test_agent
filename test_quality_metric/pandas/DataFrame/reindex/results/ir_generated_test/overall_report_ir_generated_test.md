# Overall Quality Report: pandas.DataFrame.reindex

- Generated at: 2026-02-25T03:50:32.278158+00:00
- Target dir: `test_quality_metric/pandas/DataFrame/reindex`
- Test file: `/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/ir2test_pipeline/pandas/DataFrame/reindex/ir_generated_test.py`
- Mutant eval exit code: 0
- Coverage exit code: 0
- LLM trace exit code: 0
- Properties coverage exit code: 0

## Baseline Test Status
- All tests passed (no failures or xfails)

## Mutant Kill Summary
- Killed: 15
- Survived: 5
- Invalid baseline: 0
- Mutation score: 0.750

## Coverage Summary
- Method count: 2
- Average method line coverage: 81.82%

### Method Coverage
- `pandas.DataFrame.reindex`: 1/1 lines (100.00%)
- `pandas.core.generic.NDFrame.reindex`: 14/22 lines (63.64%)

### Branch Coverage (overall)
- Branch coverage: 67/1960 (3.42%)

## Properties Coverage (LLM)
- Total clauses: 20
- Satisfied clauses: 14
- Unsatisfied clauses: 6
- Properties coverage: 0.700
- Avg confidence (satisfied): 0.900
- Avg confidence (unsatisfied): 0.900

### Clause Trace
- `C1`: satisfied by ['TestValidContracts::test_p17_identity_reindex_equals_original', 'TestValidContracts::test_p20_return_type_is_dataframe'] (confidence=0.90)
- `C2`: satisfied by ['TestValidContracts::test_p1_new_labels_get_nan_without_fill_value'] (confidence=0.90)
- `C3`: satisfied by ['TestValidContracts::test_p9_fill_value_for_new_labels_only'] (confidence=0.90)
- `C4`: unsatisfied (confidence=0.90)
- `C5`: satisfied by ['TestValidContracts::test_p4_ffill_uses_preceding_observation', 'TestValidContracts::test_p6_pad_alias_for_ffill'] (confidence=0.90)
- `C6`: satisfied by ['TestValidContracts::test_p5_bfill_uses_following_observation', 'TestValidContracts::test_p7_backfill_alias_for_bfill'] (confidence=0.90)
- `C7`: satisfied by ['TestValidContracts::test_p10_nearest_scalar_tolerance_far_label_gets_nan', 'TestValidContracts::test_p11_nearest_list_tolerance_per_label', 'TestValidContracts::test_p12_nearest_picks_smallest_distance'] (confidence=0.90)
- `C8`: satisfied by ['TestInvalidContracts::test_e1_non_monotonic_index_with_method_raises_value_error'] (confidence=0.90)
- `C9`: satisfied by ['TestValidContracts::test_p13_fill_method_preserves_preexisting_nan_in_retained_rows'] (confidence=0.90)
- `C10`: satisfied by ['TestValidContracts::test_p8_limit_caps_consecutive_fill'] (confidence=0.90)
- `C11`: satisfied by ['TestValidContracts::test_p23_both_axes_reindexed_simultaneously'] (confidence=0.90)
- `C12`: unsatisfied (confidence=0.90)
- `C13`: unsatisfied (confidence=0.90)
- `C14`: satisfied by ['TestValidContracts::test_p21_column_reindex_new_cols_nan_retained_preserved'] (confidence=0.90)
- `C15`: satisfied by ['TestValidContracts::test_p10_nearest_scalar_tolerance_far_label_gets_nan'] (confidence=0.90)
- `C16`: unsatisfied (confidence=0.90)
- `C17`: satisfied by ['TestValidContracts::test_p10_nearest_scalar_tolerance_far_label_gets_nan', 'TestValidContracts::test_p11_nearest_list_tolerance_per_label', 'TestValidContracts::test_p12_nearest_picks_smallest_distance'] (confidence=0.90)
- `C18`: unsatisfied (confidence=0.90)
- `C19`: satisfied by ['TestValidContracts::test_p3_copy_true_raises_deprecation_warning', 'TestValidContracts::test_p3_copy_false_raises_deprecation_warning'] (confidence=0.90)
- `C20`: unsatisfied (confidence=0.90)
