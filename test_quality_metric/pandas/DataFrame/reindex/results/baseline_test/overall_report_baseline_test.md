# Overall Quality Report: pandas.DataFrame.reindex

- Generated at: 2026-02-25T03:43:21.788853+00:00
- Target dir: `test_quality_metric/pandas/DataFrame/reindex`
- Test file: `/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/ir2test_pipeline/pandas/DataFrame/reindex/baseline_test.py`
- Mutant eval exit code: 0
- Coverage exit code: 1
- LLM trace exit code: 0
- Properties coverage exit code: 0

## Baseline Test Status
- **Failed** (4):
  - `ir2test_pipeline/pandas/DataFrame/reindex/baseline_test.py::test_fill_value_used_for_missing_row_labels`
  - `ir2test_pipeline/pandas/DataFrame/reindex/baseline_test.py::test_limit_caps_consecutive_fills`
  - `ir2test_pipeline/pandas/DataFrame/reindex/baseline_test.py::test_missing_row_labels_get_nan`
  - `ir2test_pipeline/pandas/DataFrame/reindex/baseline_test.py::test_multiindex_level_reindex_adds_nan_for_new_level_value`

## Mutant Kill Summary
- Killed: 12
- Survived: 8
- Invalid baseline: 0
- Mutation score: 0.600

## Coverage Summary
- Method count: 2
- Average method line coverage: 86.37%

### Method Coverage
- `pandas.DataFrame.reindex`: 1/1 lines (100.00%)
- `pandas.core.generic.NDFrame.reindex`: 16/22 lines (72.73%)

### Branch Coverage (overall)
- Branch coverage: 103/1960 (5.26%)

## Properties Coverage (LLM)
- Total clauses: 20
- Satisfied clauses: 13
- Unsatisfied clauses: 7
- Properties coverage: 0.650
- Avg confidence (satisfied): 1.000
- Avg confidence (unsatisfied): 1.000

### Clause Trace
- `C1`: satisfied by ['test_reindex_same_index_reproduces_values', 'test_result_shape_matches_new_index_and_columns'] (confidence=1.00)
- `C2`: satisfied by ['test_missing_row_labels_get_nan', 'test_new_column_label_gets_nan'] (confidence=1.00)
- `C3`: satisfied by ['test_fill_value_used_for_missing_row_labels'] (confidence=1.00)
- `C4`: unsatisfied (confidence=1.00)
- `C5`: satisfied by ['test_ffill_propagates_last_valid_observation', 'test_pad_is_alias_for_ffill'] (confidence=1.00)
- `C6`: satisfied by ['test_bfill_uses_next_valid_observation', 'test_backfill_is_alias_for_bfill'] (confidence=1.00)
- `C7`: satisfied by ['test_nearest_fill_selects_closer_neighbor'] (confidence=1.00)
- `C8`: satisfied by ['test_method_raises_on_non_monotonic_index'] (confidence=1.00)
- `C9`: satisfied by ['test_method_does_not_fill_preexisting_nan'] (confidence=1.00)
- `C10`: satisfied by ['test_limit_caps_consecutive_fills'] (confidence=1.00)
- `C11`: satisfied by ['test_labels_axis_0_equals_index_kwarg'] (confidence=1.00)
- `C12`: satisfied by ['test_labels_axis_columns_equals_columns_kwarg'] (confidence=1.00)
- `C13`: unsatisfied (confidence=1.00)
- `C14`: satisfied by ['test_new_column_label_gets_nan'] (confidence=1.00)
- `C15`: unsatisfied (confidence=1.00)
- `C16`: unsatisfied (confidence=1.00)
- `C17`: unsatisfied (confidence=1.00)
- `C18`: satisfied by ['test_multiindex_level_reindex_preserves_matching_rows'] (confidence=1.00)
- `C19`: unsatisfied (confidence=1.00)
- `C20`: unsatisfied (confidence=1.00)
