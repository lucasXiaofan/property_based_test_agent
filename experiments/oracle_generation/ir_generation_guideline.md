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
        ir_v2.json
```

## Inputs
- doc_url: {URL}
- target function name (optional)

# Function IR Generation Prompt

## Task

Read the documentation for a Python library function and produce a structured IR with metadata, pre-conditions, and post-conditions for property-based test generation.

## Hypothesis Reference

- Quickstart: https://hypothesis.readthedocs.io/en/latest/quickstart.html
- Strategies: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
- NumPy strategies: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#numpy
- Pandas strategies: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas

## Rules

**Pre-conditions**
- Derive partitions only from what the doc explicitly describes or demonstrates.
- `constraints` is a single Hypothesis strategy string, `eval()`-able with the library's standard imports plus `hypothesis.strategies as st`.


**Post-conditions**
- One post-condition per distinct observable behavior.
- `expected_behavior` is a single Python assertion using `result` for the return value.
- `confidence`: `"explicit"` if the doc shows it directly, `"implicit"` if it follows logically.

## Input

```
URL: <documentation URL>
```

## Output

A single valid JSON object. 

## JSON Schema

{
  metadata: {
    library, version, function, signature,
    reference_urls: [url, ...]
  },

  pre_condition: {
    // one key per parameter; if the function is a instance method, include "self" as a key
    <param_name>: [
      {
        id,           // snake_case Natural language identifier, start with self or input parameter name
        description,  // concisely quote from the doc, if no reference use ""
        constraints,  // single Hypothesis strategy string
      },
      ...
    ],
    ...
  },

  post_condition: [
    {
      id,                  // snake_case identifier
      source: "",          // concisely quote source description from the doc, if no reference use "", and confidence should be implicit, and explain why it is followed logically from docs
      pre_condition: {     // only use the partition_id
        <partition_id>, <partition_id>, ...
      },
      expected_behavior,   // single Python assertion string using `result`
      confidence           // "explicit" | "implicit"
    },
    ...
  ]
}

