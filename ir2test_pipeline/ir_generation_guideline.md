# IR Generation Guideline

## Folder Structure Rule (Added)
To align with `test_quality_metric`, every IR target must use:

```text
ir2test_pipeline/
  <library>/
    <module_or_class>/
      <function>/
        ir.json
```

Example:

```text
ir2test_pipeline/pandas/DataFrame/reindex/ir.json
```

## Output Artifact Rule (Added)
IR generation step writes only one file in the function folder:
- `ir.json`

Do not generate extra files (`*.md`, notes, debug dumps) in this step.

---

# Task: Generate a Test IR from API Documentation
You are generating a Test IR (Intermediate Representation) from API documentation.
The goal is to identify reasonable inputs and expected behaviors that catch real bugs —
not just the happy path shown in the docs. Output structured markdown exactly matching
the format below.

## Input
- doc_url: {URL}
- function: {FULLY_QUALIFIED_NAME}

## Output Format
```
# Test IR: {function}
- library: {library}
- version: {version}
- function: {function}
- doc: {url}

---

## Post Conditions

### {P1 or E1}
- track: valid | invalid
- type: explicit | indirect | implicit
- reason: {why this input combination is worth testing, what bug it catches,
           why it is unique from other post-conditions}
- input: {specific values with types and ranges, 1-3 parameter combination}
- assertion:
```
  {pseudo-Python, close to runnable}
```
- source: {for explicit: direct quote from doc.
           for indirect: quote + one sentence why it implies this behavior.
           for implicit: explain the convention and why it is expected.}
```

## Rules
- P-ids for valid behavior checks, E-ids for exception checks.
- explicit: doc directly states the behavior.
- indirect: doc implies it but does not state it outright.
- implicit: not in doc but is a standard convention (e.g. no mutation, column preservation).
- Each post-condition must catch a unique failure mode — if two post-conditions would
  catch the same bug, merge them or narrow the input.
- Prefer input combinations that sit at partition boundaries or interact across 2-3
  parameters — most real bugs live there, not in simple single-parameter cases.
- For invalid track: input must be a concrete bad value, assertion must use pytest.raises.
- source field is required for all three types — never leave it empty.

---

## Example

Given: pd.DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None)

### P1
- track: valid
- type: explicit
- reason: Tests that scalar fill_value replaces all NaN cells. The basic contract of
  fillna — if this is wrong every downstream use is broken. Unique because other
  post-conditions test method-based fill or limit; this isolates the scalar path.
- input: df with NaN in multiple columns, value=0 (int scalar), method=None
- assertion:
```
  assert result.isna().sum().sum() == 0
  assert (result[original.isna()] == 0).all().all()
```
- source: "Fill NA/NaN values using the specified method. value: Value to use to fill
  holes."

### P2
- track: valid
- type: explicit
- reason: Tests ffill across a gap of 2 NaNs with limit=1 — only the first NaN in
  each run should be filled, the second must stay NaN. Classic off-by-one bug site.
  Unique from P1 because this tests propagation logic and the stopping condition, not
  scalar replacement.
- input: df with consecutive NaNs e.g. [1, NaN, NaN, 2], method='ffill', limit=1
- assertion:
```
  assert result.iloc[1] == 1     # first NaN filled
  assert result.iloc[2].isna()   # second NaN exceeds limit, stays NaN
```
- source: "limit: If method is specified, this is the maximum number of consecutive
  NaN values to forward/backward fill."

### P3
- track: valid
- type: implicit
- reason: fillna with inplace=False must not modify the original DataFrame. Any code
  that reuses the original after fillna would silently operate on wrong data if mutation
  occurs. Unique because no other post-condition checks the input side.
- input: df with NaN values, value=0, inplace=False
- assertion:
```
  snapshot = original.copy(deep=True)
  _ = original.fillna(value=0)
  assert original.equals(snapshot)
```
- source: Convention — pandas operations with inplace=False return a new object and
  leave the caller unchanged. Consistent across all DataFrame methods.

### E1
- track: invalid
- type: explicit
- reason: Providing both value and method is undefined — the two filling strategies
  are mutually exclusive. Without this check a typo or copy-paste error would silently
  use one and ignore the other with no warning.
- input: value=0, method='ffill' (both set simultaneously)
- assertion:
```
  with pytest.raises(ValueError):
      original.fillna(value=0, method='ffill')
```
- source: "TypeError: Cannot specify both 'value' and 'method'."

---

Now generate the same for:
doc_url: {URL}
function: {FULLY_QUALIFIED_NAME}
