# Mutation Runner Bug Report and Fix

Date: 2026-02-24  
Scope: `test_quality_metric/run_mutant_eval.py`

## Summary
Mutation results were initially incorrect (showing all mutants killed) for some test scripts. The issue was not primarily test quality; it was runner-level execution artifacts.

## Symptoms
1. All/most mutants reported as `killed` even for very small test suites.
2. Output included non-semantic errors like:
   - Hypothesis `FailedHealthCheck(differing_executors)`
   - dataclass import/load error:
     - `AttributeError: 'NoneType' object has no attribute '__dict__'`
3. In baseline-failing scenarios, results were either misleading or hard to interpret.

## Root Causes

### RC1: In-process pytest reuse across mutants
- Original runner used repeated `pytest.main(...)` in the same Python process for baseline + each mutant.
- Hypothesis detected execution-environment inconsistency and raised `FailedHealthCheck(differing_executors)`.
- These health-check failures were counted as mutant kills.

### RC2: Dynamic module load bug for dataclass-based mutants
- Mutant module loaded via `importlib` without guaranteed registration in `sys.modules` in all code paths.
- `dataclass` internals expect module presence in `sys.modules`.
- Missing registration caused loader exceptions, which were incorrectly interpreted as mutant kills.

### RC3: Failure-node parser too strict
- Parser initially matched only `FAILED <node> - <error>` lines.
- Some pytest outputs are `FAILED <node>` (no ` - ...`).
- Baseline/new-failure sets were parsed incorrectly in differential mode.

## Why this looked like "one test kills all mutants"
Even with only one active test function, Hypothesis runs many generated examples per test. But the extreme all-kill pattern here was inflated by runner artifacts (RC1/RC2), not pure semantic mutant detection.

## Fixes Applied

### Fix A: Subprocess-isolated execution per mutant
- Replaced in-process `pytest.main(...)` loop with subprocess execution.
- Each baseline/mutant run starts in a fresh interpreter process.
- Removes Hypothesis cross-run contamination.

### Fix B: Correct module registration before execution
- Registered dynamically loaded modules in `sys.modules` before `exec_module(...)`.
- Applied both in main loader path and subprocess launcher path.

### Fix C: Robust failed-node parsing
- Updated regex to match both:
  - `FAILED <node> - <error>`
  - `FAILED <node>`

### Fix D: Differential mode for failing baselines
- Added `--allow-failing-baseline`.
- In this mode, mutant is `killed` only if it introduces new failing nodeids beyond baseline failures.

### Fix E: Per-test-script result folders
- Results now written to:
  - `mutants/results/<test_script_name>/kill_report.json`
  - `mutants/results/<test_script_name>/kill_report.md`
- Prevents overwrite across different test scripts.

## Validation After Fix
For:
- target: `test_quality_metric/pandas/DataFrame/reindex`
- pytest file: `pandas_bug_finding/analysis/past_ir/test_dataframe_reindex.py`

Observed after fixes:
- `killed=3`, `survived=17`, `score=0.150`
- This replaced the earlier false all-kill behavior.

## Current Correct Interpretation
- Mutation results now reflect semantic impact much more reliably.
- If baseline is red and `--allow-failing-baseline` is used, scoring is differential against baseline failures.
- If baseline is green, standard kill/survive semantics apply.

## Recommended Usage
```bash
uv run python /Users/xiaofanlu/Documents/github_repos/property_based_test_agent/test_quality_metric/run_mutant_eval.py \
  --target-dir /Users/xiaofanlu/Documents/github_repos/property_based_test_agent/test_quality_metric/pandas/DataFrame/reindex \
  --pytest-file /Users/xiaofanlu/Documents/github_repos/property_based_test_agent/pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py \
  --allow-failing-baseline
```

## File Changed
- `/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/test_quality_metric/run_mutant_eval.py`
