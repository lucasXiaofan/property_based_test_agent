# DataFrame Function Selection (Max 5)

Source index page:
- https://pandas.pydata.org/pandas-docs/version/3.0.0/reference/frame.html

This folder defines a scalable starting set of DataFrame APIs for mutation-based test quality evaluation.

## Valuable Function Types
1. Label alignment and missing-label behavior
2. Relational join behavior
3. Split-apply-combine behavior
4. Time-series resampling behavior
5. Dtype conversion behavior

## Selected Functions (one per type)
1. `pandas.DataFrame.reindex`
2. `pandas.DataFrame.merge`
3. `pandas.DataFrame.groupby`
4. `pandas.DataFrame.resample`
5. `pandas.DataFrame.astype`

## Why these five first
- They have rich semantic contracts with many edge cases.
- They are common in real test suites.
- They are high-impact when behavior regresses.

## Procedure Compliance
Each selected function has this minimal structure (from `CODING_AGENT_PROCEDURE.md`):
- `metadata.json`
- `clauses.json`
- `mutants/<function>_mutants.py`
- `mutants/mapping.json`
- `mutants/run_mutant_eval.py`
- `mutants/results/`

## Baseline Test Command (current reindex test)
```bash
uv run pytest /Users/xiaofanlu/Documents/github_repos/property_based_test_agent/pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py -q
```

For non-`reindex` functions, replace `--pytest-file` with the corresponding target test file when available.
