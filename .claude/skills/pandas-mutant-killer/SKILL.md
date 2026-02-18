---
name: pandas-mutant-killer
description: Generate and validate mutants for a target pandas function, then run a specified pytest file to determine whether each mutant is killed or survives. Use when the user asks to mutate pandas behavior and evaluate test effectiveness.
---

# Pandas Mutant Killer

Create mutants for a specific pandas function and evaluate whether a target test file kills each mutant.

Default target layout in this repo:
- pandas source: `pandas_bug_finding/pandas/pandas/...`
- tests: `pandas_bug_finding/ir_test/...`

## Inputs To Collect

- Target function symbol and file path.
- One or more test files that are supposed to detect the bug.
- Number of mutants to generate (default: 3-5).
- Whether to use direct source patching or runtime monkeypatch (default: runtime monkeypatch).

If the user gives only a function snippet, resolve its class/module in the local pandas source tree first.

## Output Contract

For each mutant, always produce:
- `mutant_id`
- target symbol (module/class/function)
- exact code change summary
- test command used
- outcome: `killed`, `survived`, or `invalid`
- failing test node(s) when killed

If requested, also write a machine-readable report:
- `pandas_bug_finding/analysis/mutant_results/<target_name>.json`

## Workflow

### 1) Locate the real target

1. Find the function in `pandas_bug_finding/pandas/pandas`.
2. Verify there is no similarly named method in another class that would be a mistaken target.
3. Copy the original body before mutation.

For the example snippet:
- file: `pandas_bug_finding/pandas/pandas/core/window/ewm.py`
- symbol: `pandas.core.window.ewm.ExponentialMovingWindow.aggregate`

### 2) Define mutation operators

Prefer semantic mutations over syntax breaks. Use one operator per mutant.

Good operators for wrapper-style methods (like `return super().aggregate(func, *args, **kwargs)`):
- Change argument forwarding (`func` replaced, dropped, or transformed).
- Remove `*args` or `**kwargs` forwarding.
- Swap implementation path (`super().aggregate` -> `super().agg` or another method when meaningful).
- Inject conditional behavior on callable/string/list/dict `func`.
- Change exception behavior (`None` handling, unsupported type handling).

Avoid:
- mutations that only rename locals without behavior change
- syntax-invalid mutants
- multi-change mutants that make root cause unclear

### 3) Install mutant for test execution

Use one of these methods.

#### Preferred: runtime monkeypatch mutant

Use a pytest fixture (usually in `pandas_bug_finding/ir_test/conftest.py`) that:
1. Imports target symbol.
2. Saves original callable.
3. Replaces it with mutant implementation when `MUTANT_ID` matches.
4. Restores original after test session.

This avoids rebuilding pandas and keeps baseline vs mutant runs comparable.

#### Alternative: direct source patch mutant

Patch the pandas source file directly, run tests, then restore the original code before next mutant.

Only use this when runtime monkeypatch cannot reach the target path.

### 4) Run baseline then mutant

Always run baseline first (no mutant enabled):

```bash
uv run pytest pandas_bug_finding/ir_test/test_dataframe_reindex.py -q
```

Then run with a specific mutant:

```bash
MUTANT_ID=M1 uv run pytest pandas_bug_finding/ir_test/test_dataframe_reindex.py -q
```

Interpretation:
- baseline fails: stop and mark run `invalid` (tests are not a reliable oracle yet).
- mutant run fails but baseline passes: mutant is `killed`.
- mutant run passes and baseline passes: mutant `survived`.

### 5) Report kills clearly

For killed mutants, capture:
- first failing test node id(s)
- assertion/error type
- short explanation of why this failure is expected from the mutation

For survived mutants, suggest new property/test ideas targeting the missed behavior.

## Example mutant (for ExponentialMovingWindow.aggregate)

Target:
`pandas_bug_finding/pandas/pandas/core/window/ewm.py`

Original:
```python
def aggregate(self, func=None, *args, **kwargs):
    return super().aggregate(func, *args, **kwargs)
```

Possible mutant M1:
```python
def aggregate(self, func=None, *args, **kwargs):
    if callable(func):
        func = func.__name__
    return super().aggregate(func, *args, **kwargs)
```

This mutant is useful because callable-vs-string behavior has known edge cases in EWM aggregation.

## Quality Bar

- Baseline test command must pass before claiming any kill.
- One mutant per run for clean attribution.
- Keep deterministic settings when needed (`--hypothesis-seed=<n>`).
- Do not silently leave patched code behind.

## Quick checklist

1. Confirm target function path and symbol.
2. Generate 3-5 single-change mutants.
3. Run baseline test file(s).
4. Run each mutant with identical test command.
5. Classify `killed/survived/invalid`.
6. Return concise mutant table and next tests to add for survivors.
