# Test Quality Metric: Simple Procedure (v2)

## Goal
For one documented function (example: `pandas.DataFrame.reindex`), create:
1. Contract clauses from docs.
2. Executable clause-mapped mutants.
3. Mutation results against an existing pytest file.

Root folder:
`/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/test_quality_metric`

## Minimal Folder Layout
```text
test_quality_metric/
  run_mutant_eval.py                # shared runner for all functions
  <library>/<module_or_class>/<function>/
    metadata.json
    clauses.json
    mutants/
      <function>_mutants.py         # executable mutants, not docstring-only
      mapping.json
      results/
```

Example:
```text
test_quality_metric/pandas/DataFrame/reindex/
```

## Phase 1: Build Clauses
1. Input doc URL.
2. Read docs and confirm this is a callable with input/output behavior.
3. Create function folder.
4. Write `metadata.json` with basic info.
5. Write `clauses.json` with quote-backed clauses.

### `metadata.json` (minimum)
```json
{
  "library": "pandas",
  "version": "3.0.0",
  "qualified_name": "pandas.DataFrame.reindex",
  "doc_url": "https://pandas.pydata.org/pandas-docs/version/3.0.0/reference/api/pandas.DataFrame.reindex.html"
}
```

### `clauses.json` item format
```json
{
  "id": "C1",
  "description": "Short, testable clause",
  "quote": "Exact quote from docs",
  "category": "Return Contract"
}
```

Rules:
- Keep clauses testable.
- Keep quote exact.
- One clause per behavior.

## Phase 2: Build Executable Mutants
1. Create `mutants/<function>_mutants.py`.
2. Add runtime mutant implementations (actual code).
3. Add mutant registry + `apply_mutant()` + `reset_mutant()` + `list_mutants()`.
4. Write `mutants/mapping.json`.
5. Run shared evaluator and save results.

### Required mutant module API
Your mutant file must expose:
- `apply_mutant(mutant_id: str) -> dict[str, str]`
- `reset_mutant() -> None`
- `list_mutants() -> list[dict[str, str]]`

Docstring-only mutant stubs are not acceptable.

### `mapping.json` format
```json
{
  "function": "pandas.DataFrame.reindex",
  "mutants": [
    {"mutant_id": "M_C1_example", "clause_id": "C1"}
  ]
}
```

## Run Commands
Baseline test:
```bash
uv run pytest /Users/xiaofanlu/Documents/github_repos/property_based_test_agent/pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py -q
```

Shared mutant eval (single script for all functions):
```bash
uv run python /Users/xiaofanlu/Documents/github_repos/property_based_test_agent/test_quality_metric/run_mutant_eval.py \
  --pytest-file /Users/xiaofanlu/Documents/github_repos/property_based_test_agent/pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py \
  --mapping-file /Users/xiaofanlu/Documents/github_repos/property_based_test_agent/test_quality_metric/pandas/DataFrame/reindex/mutants/mapping.json \
  --mutants-file /Users/xiaofanlu/Documents/github_repos/property_based_test_agent/test_quality_metric/pandas/DataFrame/reindex/mutants/reindex_mutants.py \
  --output-dir /Users/xiaofanlu/Documents/github_repos/property_based_test_agent/test_quality_metric/pandas/DataFrame/reindex/mutants/results
```

## Success Criteria (v2)
- `metadata.json` exists.
- `clauses.json` exists and quotes docs.
- Mutants contain executable runtime code.
- One shared `run_mutant_eval.py` is used (no duplicated per-function runners).
- Mutation report exists in `mutants/results/`.
