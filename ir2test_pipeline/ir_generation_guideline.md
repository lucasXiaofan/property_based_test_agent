# Prompt: Generate Test IR as JSON

You are generating a Test IR (Intermediate Representation) from API documentation.
The goal is to identify inputs and expected behaviors that catch real bugs — not just
the happy path. Output valid JSON only. No prose outside the JSON.
## Folder Structure Rule (Added)
To align with `test_quality_metric`, every IR target must use:

```text
ir2test_pipeline/
  <library>/
    <module_or_class>/
      <function>/
        ir.json
```

## Inputs
- doc_url: {URL}
- function: {FULLY_QUALIFIED_NAME}

## Output JSON Structure
```json
{
  "metadata": {
    "library": "string",
    "version": "string",
    "function": "string — fully qualified e.g. pd.DataFrame.reindex",
    "references": [
      { "id": "R1", "type": "api_doc | docstring | user_guide", "url": "string" }
    ]
  },
  "pre_conditions": {
    "{param_name}": {
      "type": "accepted Python types",
      "partitions": {
        "{ID e.g. L0, M1}": {
          "desc": "what makes this region semantically distinct",
          "example": "concrete value or expression"
        }
      },
      "interaction_hints": ["other param names whose behavior changes with this one"],
      "invalid_cases": [
        {
          "desc": "why this input is invalid",
          "example": "concrete invalid input",
          "expected_exception": "ValueError | TypeError | KeyError etc.",
          "note": "null or clarification if doc is ambiguous",
          "source_ref": "R1"
        }
      ]
    }
  },
  "post_conditions": [
    {
      "id": "P1",
      "track": "valid | invalid",
      "evidence": "explicit | indirect | implicit",
      "source_ref": "R1 — null only when evidence=implicit",
      "why": "(1) what real usage scenario this represents, (2) what specific bug this catches that other post-conditions would not, (3) why this trigger scope is right — not too broad, not too narrow",
      "claim": "one NL sentence of expected behavior",
      "formal": [
        "pseudo-Python assertion lines using symbols:",
        "  original = input df before the call",
        "  result   = returned value",
        "  params.* = parameter values e.g. params.method",
        "valid track:  assert result.something == expected",
        "invalid track: with pytest.raises(XError): original.fn(bad_input)"
      ],
      "trigger": {
        "{param_name}": ["partition IDs for valid track, or invalid_case desc for invalid track"]
      }
    }
  ]
}
```

## Evidence Types

- **explicit**: doc directly states this behavior. `source_ref` must point to a
  reference that contains a direct quote supporting the claim.
- **indirect**: doc implies the behavior through examples or parameter descriptions
  but does not state it outright. Quote the relevant text and explain the implication.
- **implicit**: standard convention not mentioned in the doc at all — e.g. no mutation,
  column preservation, return type. `source_ref` is str. Justify why this is expected.

## Rules for pre_conditions

- Partitions must be **mutually exclusive** and **collectively cover** the full valid
  input space for that parameter.
- A partition boundary is where observable behavior changes — not just where the value
  changes. Two inputs that produce identical behavior belong in the same partition.
- `interaction_hints` drives covering array priority — if A lists B, at least one
  post-condition must have both in its trigger.
- `invalid_cases` are tested 1-way independently — never cross two invalid cases.

## Rules for post_conditions

**Be exhaustive. For every function, ensure you cover:**
- Core output shape/structure (index, columns, length)
- Value preservation for matched inputs
- Fill/default behavior for unmatched inputs
- Each fill method variant if applicable (ffill, bfill, nearest etc.)
- Boundary partitions: empty input, identical input, fully disjoint input
- Limit/threshold stopping conditions and their off-by-one boundary
- Interaction between 2–3 parameters where behavior is non-obvious
- No mutation of the original input
- Return type and object identity
- One E-id per invalid_case in pre_conditions — no more, no less

**Post-condition quality checks:**
- P-ids for valid behavior, E-ids for exception checks
- Each post-condition must catch a **unique** failure mode — if two would catch the
  same bug, merge them or narrow the trigger
- Prefer triggers that combine 2–3 parameters — most real bugs live in interactions,
  not in single-parameter cases
- Every partition ID must appear in at least one valid-track trigger
- Every invalid_case must map to exactly one E-id
- No two post-conditions should share identical formal assertions