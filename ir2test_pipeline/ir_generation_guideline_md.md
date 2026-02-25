# IR Generation Guideline

## Folder Structure Rule (Added)
To align with `test_quality_metric`, every IR target must use:

```text
ir2test_pipeline/
  <library>/
    <module_or_class>/
      <function>/
        ir.md
```

Example:

```text
ir2test_pipeline/pandas/DataFrame/reindex/ir.md
```

## Output Artifact Rule (Added)
IR generation step writes only one file in the function folder:
- `ir.md`

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
- Be exhaustive on high-value properties: cover the most bug-prone and semantically central
  contracts first, not just easy examples from the docs.
- Prioritize top-value properties using this order:
  1) core semantic contract (what must always be true),
  2) parameter interaction boundaries (2-3 parameters together),
  3) shape/type/index alignment invariants,
  4) mutation/copy semantics,
  5) documented error paths and validation constraints.
- Include all three evidence types in each IR when possible (`explicit`, `indirect`,
  `implicit`). Do not produce IRs that only contain explicit properties unless the API
  truly lacks reasonable indirect/implicit behaviors.
- Prefer fewer high-signal properties over many low-value restatements, but ensure the
  selected set is broad enough to catch distinct failure modes across happy path, edge
  path, and invalid path.
- Each post-condition must catch a unique failure mode — if two post-conditions would
  catch the same bug, merge them or narrow the input.
- Prefer input combinations that sit at partition boundaries or interact across 2-3
  parameters — most real bugs live there, not in simple single-parameter cases.
- For invalid track: input must be a concrete bad value, assertion must use pytest.raises.
- source field is required for all three types — never leave it empty.

---



Now generate the same for:
doc_url: {URL}
function: {FULLY_QUALIFIED_NAME}
