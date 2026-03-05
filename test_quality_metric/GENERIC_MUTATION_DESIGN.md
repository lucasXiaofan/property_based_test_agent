# Generic Mutant Generation — Design & Findings

## 1  Key Finding: Most Pandas Functions Are Pure Python

**The wrapper is not needed for the majority of pandas DataFrame methods.**

`inspect.getfile` reveals where each method's logic actually lives:

| Method | Python source file | Directly mutatable? |
|--------|--------------------|---------------------|
| `reindex` | `pandas/core/generic.py` (logic) + `frame.py` (1-line `super()` call) | ✅ yes |
| `head`, `tail` | `pandas/core/generic.py` | ✅ yes |
| `sort_index`, `merge`, `apply` | `pandas/core/frame.py` | ✅ yes |
| `loc`, `iloc` | `pandas/core/indexing.py` | ✅ yes (Python router; low-level in `_libs/indexing.so`) |
| `groupby` | `pandas/core/frame.py` via `@doc` decorator | ✅ yes |
| **Low-level ops** (`at`, `iat`, arithmetic) | `pandas/_libs/*.so` (Cython) | ❌ wrapper required |

The 45 compiled `.so` files in `pandas/_libs/` handle low-level array operations.
Every high-level DataFrame/Series API method delegates to them through a Python entry
point, and **that Python entry point is what we mutate**.

---

## 2  The Problem With the Current Approach

`reindex_wrapper.py` was created under the incorrect assumption that `reindex` was
a C extension. It is actually a pure Python method in `generic.py`.

The wrapper exists for two reasons that no longer apply:

1. **"C extension, can't mutate directly"** → False. `generic.py` is pure Python.
2. **"Need `install()` to patch pandas"** → Unnecessary if we mutate the actual source
   file and import it fresh per subprocess.

What the wrapper *did* achieve (and what we must preserve):

> **Scoping**: `generic.py` has 13,769 lines and 2,531 mutation sites.
> Only ~30–40 of those are inside `reindex`. Running all 2,531 mutants is wasteful.

---

## 3  The Solution: `--function-name` Flag

Add a `--function-name` argument to `run_traditional_mutation.py`.
`MutationCollector` tracks whether it is currently inside the named function's
body, and only records sites when it is.

```
--source-file  pandas/core/generic.py    (installed .venv copy)
--function-name reindex
--test-file    baseline_test.py
```

Result: ~30–40 targeted mutations inside `reindex` only.
No wrapper. No `install()`. No hand-crafted mutant functions.

### How it works internally

```python
class MutationCollector(cst.CSTVisitor):
    def __init__(self, function_name=None):
        self._target_fn = function_name
        self._depth = 0          # nesting depth inside target function
        self.sites = []

    def visit_FunctionDef(self, node):
        if self._target_fn and node.name.value == self._target_fn:
            self._depth += 1

    def leave_FunctionDef(self, node):
        if self._target_fn and node.name.value == self._target_fn:
            self._depth -= 1

    def _in_scope(self):
        return self._target_fn is None or self._depth > 0

    def visit_Integer(self, node):
        if self._in_scope():
            ...  # record mutation site
```

When `--function-name` is omitted, all sites in the file are collected (current behavior).

---

## 4  Where to Point `--source-file`

**Use the installed `.venv` copy, not the local pandas source repo.**

| Location | Purpose | Run tests against it? |
|----------|---------|----------------------|
| `.venv/lib/python3.11/site-packages/pandas/core/generic.py` | Installed pure Python, has compiled `_libs/` alongside it | ✅ yes — this is what `import pandas` loads |
| `pandas_bug_finding/pandas/pandas/core/generic.py` | Dev source, no compiled extensions | ❌ no — `import pandas` would fail without a build step |

### Why the installed copy

When a subprocess runs `import pandas`, Python loads from `.venv/lib/python3.11/site-packages/pandas/`.
If we overwrite `.../generic.py` with a mutated version before the subprocess starts,
the subprocess picks up the mutation automatically (Python recompiles `.pyc` when `.py`
is newer). After the subprocess exits, we restore the original.

The local `pandas_bug_finding/pandas/` is useful for:
- Reading the source to understand decision points
- Cross-checking which lines contain the logic you want to test
- Reference when writing test assertions

But tests should always `import pandas` (the installed version), never import from the
dev source directly.

---

## 5  Concrete Workflow (No Wrapper)

```bash
# reindex — logic is in generic.py
uv run python test_quality_metric/run_traditional_mutation.py \
    --source-file   .venv/lib/python3.11/site-packages/pandas/core/generic.py \
    --function-name reindex \
    --test-file     ir2test_pipeline/pandas/DataFrame/reindex/baseline_test.py \
    --output-dir    ir2test_pipeline/pandas/DataFrame/reindex/trad_mutant_results/

# sort_index — logic is in frame.py
uv run python test_quality_metric/run_traditional_mutation.py \
    --source-file   .venv/lib/python3.11/site-packages/pandas/core/frame.py \
    --function-name sort_index \
    --test-file     pandas_bug_finding/baseline_testing/test_sort_index_hypothesis.py \
    --output-dir    results/sort_index/

# head — logic is in generic.py
uv run python test_quality_metric/run_traditional_mutation.py \
    --source-file   .venv/lib/python3.11/site-packages/pandas/core/generic.py \
    --function-name head \
    --test-file     pandas_bug_finding/baseline_testing/test_head_hypothesis.py \
    --output-dir    results/head/
```

Tests are written normally — just `import pandas as pd` — no path gymnastics.

---

## 6  Source File Map for Common Functions

| Function | File to pass as `--source-file` |
|----------|---------------------------------|
| `reindex`, `head`, `tail`, `rename`, `fillna`, `dropna`, `ffill`, `bfill` | `.venv/.../pandas/core/generic.py` |
| `sort_index`, `sort_values`, `merge`, `join`, `apply`, `groupby`, `pivot_table`, `melt`, `explode` | `.venv/.../pandas/core/frame.py` |
| `loc`, `iloc`, `at`, `iat` | `.venv/.../pandas/core/indexing.py` |
| `Series.map`, `Series.apply` | `.venv/.../pandas/core/series.py` |
| `read_csv`, `read_json` | `.venv/.../pandas/io/parsers/readers.py`, `pandas/io/json/_json.py` |

To find any function's file:
```python
import inspect, pandas as pd
print(inspect.getfile(pd.DataFrame.your_function))
```

---

## 7  When a Wrapper Is Still Needed

| Case | Why | Example |
|------|-----|---------|
| Function body is entirely in a `.so` Cython file | No Python AST | `pd.DataFrame.at.__set__` |
| A method is a property backed by a C class | Can't overwrite `.py` to change it | `pd.DataFrame.values` |
| Third-party library with no Python source | No `.py` file at all | `numpy.linalg.norm` (wraps BLAS) |
| You want mutations that test *internal C logic* | C logic unreachable by AST | Sort stability in `_libs/algos.so` |

For these cases, write a wrapper that re-expresses the Python-level routing in a
separate `.py` file, and use the existing wrapper approach.

---

## 8  Implementation Plan for `--function-name`

The only file to modify is `test_quality_metric/run_traditional_mutation.py`:

1. Add `--function-name` CLI argument (optional, default `None`).
2. Pass it to `MutationCollector.__init__`.
3. Add `visit_FunctionDef` / `leave_FunctionDef` pair and `_depth` counter (shown in §3).
4. Guard every `_add()` call with `if self._in_scope()`.
5. Update `collect_mutation_sites` signature to accept `function_name=None`.

No other files change. No new files. The wrapper files remain as-is for backward
compatibility but are no longer needed for new functions.

---

## 9  Summary

| Before | After (`--function-name` added) |
|--------|----------------------------------|
| Write `reindex_wrapper.py` manually | No wrapper for pure-Python functions |
| Write `traditional_mutants.py` with one function per mutant | Fully automatic from AST |
| 15 hand-crafted mutants | ~30–50 AST-generated mutants per function |
| Works only for `reindex` | Works for any function in any pure-Python file |
| Tests import installed pandas | Tests import installed pandas (unchanged) |
| Need `install()` to monkey-patch | Script overwrites source file directly, restored after each mutant |

The only manual step remaining: write the test file and know which source file
contains the function. That's it.
