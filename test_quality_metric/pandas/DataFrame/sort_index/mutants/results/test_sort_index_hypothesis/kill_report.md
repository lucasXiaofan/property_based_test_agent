# Kill Report: pandas.DataFrame.sort_index

- pytest_file: `/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/pandas_bug_finding/baseline_testing/test_sort_index_hypothesis.py`
- baseline_passed: True
- killed: 11
- survived: 1
- invalid_baseline: 0
- allow_failing_baseline: True
- mutation_score: 0.917

## Mutants
- `M_C1_no_sort` (C1): killed
- `M_C2_inplace_returns_self` (C2): killed
- `M_C3_ascending_reversed` (C3): killed
- `M_C4_axis1_sorts_rows` (C4): killed
- `M_C5_descending_becomes_ascending` (C5): killed
- `M_C6_na_last_becomes_first` (C6): killed
- `M_C7_na_first_becomes_last` (C7): killed
- `M_C8_ignore_index_no_relabel` (C8): killed
- `M_C9_key_not_applied` (C9): killed
- `M_C10_stable_becomes_quicksort` (C10): survived
- `M_C11_sort_remaining_ignored` (C11): killed
- `M_C12_ignore_level` (C12): killed
