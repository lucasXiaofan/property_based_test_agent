# Standard for Creating Pandas Mutant Wrappers

## Purpose

Use mutant wrappers to compare how well `baseline_test.py` and `ir_generated_test.py` detect behavior changes in pandas APIs.

This standard is intentionally workflow-oriented:

1. Keep a canonical API document in `experiments/python_library_bug_analysis/downloaded_docs/`
2. Copy that document into the case directory in `experiments/oracle_generation/pandas/...`
3. Create a `mutant_wrapper.py` whose mutants are justified by that document
4. Evaluate both test suites against every mutant and write a timestamped JSON report in `experiments/oracle_evaluation/mutant_kill/`

## Canonical Layout

Each case directory should look like this:

```text
experiments/oracle_generation/pandas/<type>/<function>/
├── baseline_test.py
├── ir_generated_test.py
├── ir_v2.json
├── pandas.<qualified.api>.md
├── mutant_wrapper.py
└── conftest.py
```

The markdown file in the case directory should match the canonical copy in `downloaded_docs/`.

## Doc Sync Rule

Treat `experiments/python_library_bug_analysis/downloaded_docs/` as the canonical source when it exists.

If a case directory already has the only available copy of the doc, backfill `downloaded_docs/` first and then resync the case copy from that canonical path. This keeps future wrapper generation reproducible.

The current helper script for this is:

```bash
uv run python experiments/oracle_evaluation/mutant_kill/pandas_mutant_workflow.py sync-docs
```

## Mutant Design Rules

Every mutant must satisfy all of the following:

1. Change one documented behavior only
2. Be traceable to a doc anchor, default, or explicit contract
3. Preserve the function signature and return shape as much as possible
4. Stay small enough that a failing test points to one semantic gap

Prefer these mutant classes:

- Ignore a documented argument effect, for example treating `sort=False` as `sort=True`
- Ignore a caller override, for example forcing ASCII even with `force_ascii=False`
- Swap to a nearby but wrong documented behavior, for example search semantics instead of anchored match semantics
- Bias a scalar result in a minimal way when the doc contract is numeric

Avoid these:

- Multi-change mutants
- Mutants that only rename variables or reorder code
- Mutants that break import-time behavior
- Mutants that introduce syntax-level noise without a semantic difference

## Wrapper Structure

Each `mutant_wrapper.py` should provide:

```python
def get_mutant_id():
    ...

def mutant_<api>_M1(...):
    ...

def mutant_<api>_M2(...):
    ...

def install_mutants():
    ...

def uninstall_mutants():
    ...

MUTANT_INFO = {
    "M1": {
        "name": "...",
        "description": "...",
        "doc_anchor": "...",
        "expected_kill": "...",
    },
}
```

Requirements:

- Read the active mutant from `MUTANT_ID`
- Patch only the target pandas API for the current case
- Restore the original implementation in `uninstall_mutants()`
- Keep `MUTANT_INFO` complete because the evaluator reads it for reports

## Case-Specific Patch Guidance

Patch the public API entry point that the tests call.

Examples in the current pandas set:

- `pd.DataFrame.groupby`
- `pd.DataFrame.reindex`
- `pd.DataFrame.to_json`
- `pd.Index.astype`
- `pd.Series.mean`
- `pandas.core.strings.accessor.StringMethods.contains`

For datetime-like `Index.shift`, patch every relevant concrete class that exposes the behavior under test:

- `pd.DatetimeIndex.shift`
- `pd.TimedeltaIndex.shift`
- `pd.PeriodIndex.shift`

## Conftest Requirement

The case `conftest.py` should install and uninstall mutants automatically when `mutant_wrapper.py` exists. The current pandas cases already use a shared helper and an autouse fixture for this.

## Workflow Scripts

Current supported workflow:

```bash
uv run python experiments/oracle_evaluation/mutant_kill/pandas_mutant_workflow.py materialize
uv run python experiments/oracle_evaluation/mutant_kill/compare_pandas_test_methods.py
```

`pandas_mutant_workflow.py`:

- syncs docs into every case directory
- backfills missing canonical docs in `downloaded_docs/` when needed
- writes `mutant_wrapper.py` for every pandas case in scope

`compare_pandas_test_methods.py`:

- recursively finds all case directories under `experiments/oracle_generation/pandas`
- requires `baseline_test.py`, `ir_generated_test.py`, and `mutant_wrapper.py`
- runs each suite once without a mutant to validate the baseline
- runs each passing suite against every mutant in isolation
- writes a timestamped JSON report in `experiments/oracle_evaluation/mutant_kill/`

## Report Contract

The report must contain:

1. A timestamp
2. An overall winner comparing `baseline` vs `ir_generated`
3. Overall kill-rate totals for both methods
4. Per-function winner and per-method kill rate
5. Per-mutant detail showing killed vs survived for each method
6. Baseline validity for each method so broken suites are not counted as wins

Expected shape:

```json
{
  "generated_at": "...",
  "summary": {
    "methods": {
      "baseline": {
        "valid_mutants": 0,
        "killed_mutants": 0,
        "overall_kill_rate": 0.0
      },
      "ir_generated": {
        "valid_mutants": 0,
        "killed_mutants": 0,
        "overall_kill_rate": 0.0
      }
    },
    "overall_winner": "baseline"
  },
  "functions": [
    {
      "case_dir": "Series/mean",
      "winner": "ir_generated",
      "method_summaries": {
        "baseline": {
          "baseline_passed": true,
          "kill_rate": 0.5,
          "killed_mutant_ids": ["M1"],
          "survived_mutant_ids": ["M2"]
        },
        "ir_generated": {
          "baseline_passed": true,
          "kill_rate": 1.0,
          "killed_mutant_ids": ["M1", "M2"],
          "survived_mutant_ids": []
        }
      },
      "mutants": [
        {
          "mutant_id": "M1",
          "methods": {
            "baseline": {"status": "killed"},
            "ir_generated": {"status": "killed"}
          }
        }
      ]
    }
  ]
}
```

## Validation Checklist

- [ ] Canonical doc exists in `downloaded_docs/`
- [ ] Case doc exists in the function directory
- [ ] Wrapper mutants map to documented behavior
- [ ] Each mutant changes one semantic behavior only
- [ ] `MUTANT_INFO` includes `name`, `description`, `doc_anchor`, and `expected_kill`
- [ ] `install_mutants()` and `uninstall_mutants()` are reversible
- [ ] Baseline and mutant runs execute in separate subprocesses
- [ ] JSON output compares `baseline` and `ir_generated` at overall and per-function levels
