# Mutant Testing Guide

This folder is the mutation-testing workspace for `pandas_bug_finding`.

Goal:
- generate mutants for a target pandas function
- run target tests from `baseline_testing` and/or `ir_test`
- produce a straightforward report:
  - which suite kills more mutants
  - mutant IDs killed by each suite
  - whether both suites failed baseline (invalid verdicts)

## Folder Layout

- `ewm_aggregate/ewm_aggregate_mutants.py`: mutant definitions for `ExponentialMovingWindow.aggregate`
- `ewm_aggregate/scripts/run_mutant_eval.py`: baseline + mutant executor, writes kill JSON
- `ewm_aggregate/scripts/render_kill_report_md.py`: converts kill JSON to Markdown table
- `scripts/compare_mutant_suites.py`: compares `baseline_testing` vs `ir_test` kill power
- `ewm_aggregate/results/`: generated JSON/MD reports

## Agent Workflow (Recommended)

1. Pick a target function and locate exact symbol in pandas source.
2. Create 3-5 single-change mutants (one behavior change per mutant).
3. Ensure mutant injection is env-driven (e.g. `MUTANT_ID`) so tests do not need edits.
4. Run baseline first, then each mutant with the same test target(s).
5. Save machine-readable JSON.
6. Render Markdown tables.
7. Compare suite-level kill strength and store the comparison report.

## Current EWM Aggregate Mutants

- `M1_SUM_TO_MEAN`
- `M2_TRUNCATE_LIST_FUNCS`
- `M3_CALLABLE_TYPEERROR`

## Run Mutation Evaluation

Full files:

```bash
uv run python pandas_bug_finding/mutants/ewm_aggregate/scripts/run_mutant_eval.py \
  --output pandas_bug_finding/mutants/ewm_aggregate/results/kill_report.json
```

Targeted nodes (recommended when full files have baseline failures):

```bash
uv run python pandas_bug_finding/mutants/ewm_aggregate/scripts/run_mutant_eval.py \
  --test-file pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py::TestConsistencyProperties::test_aggregate_sum_matches_ewm_sum \
  --test-file pandas_bug_finding/ir_test/test_ewm_aggregate.py::TestExplicit::test_explicit__p3__list_of_string_func_names_returns \
  --test-file pandas_bug_finding/ir_test/test_ewm_aggregate.py::TestImplicit::test_implicit__p11__any_callable_func_raises_attributeerror_ewm \
  --output pandas_bug_finding/mutants/ewm_aggregate/results/kill_report_targeted.json
```

## Render Human-Readable Kill Tables

```bash
python3 pandas_bug_finding/mutants/ewm_aggregate/scripts/render_kill_report_md.py \
  --input pandas_bug_finding/mutants/ewm_aggregate/results/kill_report.json \
  --output pandas_bug_finding/mutants/ewm_aggregate/results/kill_report.md \
  --title "EWM Aggregate Mutant Kill Report (Full Files)"

python3 pandas_bug_finding/mutants/ewm_aggregate/scripts/render_kill_report_md.py \
  --input pandas_bug_finding/mutants/ewm_aggregate/results/kill_report_targeted.json \
  --output pandas_bug_finding/mutants/ewm_aggregate/results/kill_report_targeted.md \
  --title "EWM Aggregate Mutant Kill Report (Targeted)"
```

## Straightforward Suite Comparison Report

This answers:
- which suite kills more mutants (`ir_test` or `baseline_testing`)
- exact mutant IDs each suite killed
- whether both suites failed baseline

Generate comparison from full-file report:

```bash
python3 pandas_bug_finding/mutants/scripts/compare_mutant_suites.py \
  --input pandas_bug_finding/mutants/ewm_aggregate/results/kill_report.json \
  --output-json pandas_bug_finding/mutants/ewm_aggregate/results/suite_comparison_full.json \
  --output-md pandas_bug_finding/mutants/ewm_aggregate/results/suite_comparison_full.md \
  --title "EWM Aggregate: Suite Comparison (Full Files)"
```

Generate comparison from targeted report:

```bash
python3 pandas_bug_finding/mutants/scripts/compare_mutant_suites.py \
  --input pandas_bug_finding/mutants/ewm_aggregate/results/kill_report_targeted.json \
  --output-json pandas_bug_finding/mutants/ewm_aggregate/results/suite_comparison_targeted.json \
  --output-md pandas_bug_finding/mutants/ewm_aggregate/results/suite_comparison_targeted.md \
  --title "EWM Aggregate: Suite Comparison (Targeted)"
```

## Reading Outcomes

- `killed`: baseline passed, mutant run failed
- `survived`: baseline passed, mutant run passed
- `invalid`: baseline failed, so mutant verdict is not reliable

If both suites fail baseline on full files, use node-level targeted runs for meaningful kill comparison.

