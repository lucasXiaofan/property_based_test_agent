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
- target function name (optional)

Given a documentation URL for a library function, generate a formal specification JSON with the following structure:

```
{
  "metadata": {
    "library": string,
    "version": string,
    "function": string,
    "references": [url strings]
  },

  "preconditions": {
    "data_state": {
      "required": bool,  // true if the function's behavior depends on the state of an object (e.g. DataFrame), false for pure functions like max(x)
      "rationale": string,  // only if required=true, explain why data state matters
      "self": {  // only if required=true
        "type": string,
        "partitions": {
          "partition_name": "description with one concrete example"
        }
      }
    },
    "parameters": {
      "<param_name>": {
        "type": string,
        "partitions": {  // only define partitions that trigger distinct behavior; omit trivial restatements of type
          "partition_name": "description with one concrete example"
        },
        "invalid_cases": [
          { "desc": "what goes wrong and concrete example" }
        ]
      }
    }
  },

  "postconditions": [
    {
      "id": "PC-XX",
      "claim": "natural language: given what input state/params, what is the expected behavior",
      "trigger": {
        // structured map of parameter/data_state → partition name(s) that activate this postcondition
        // only include parameters that are relevant to this postcondition
        // reference partition names defined in preconditions; do not invent new terms
        "<param_or_self>": "partition_name | partition_name | ..."
      },
      "expected_behavior": "assertion or error in concise formal style, e.g. result.loc[x].isna().all() == True or raises ValueError",
      "confidence": "explicit | implicit"  // explicit = doc directly states this; implicit = logically follows from doc
    }
  ]
}
```

**Rules:**

1. **Partitions** — only define a partition when it produces a behaviorally distinct outcome. Do not create partitions that merely restate the type or the default value.
2. **data_state** — include only if the object's internal state (e.g. index monotonicity, existing NaN cells, gap structure) affects behavior or validity of parameters. For pure functions, set `required: false` and omit `self`.
3. **Triggers** — every key in a trigger must reference a partition name defined in preconditions. Never introduce terms (like `gap_size`) that are not formally defined somewhere in preconditions.
4. **confidence** — `explicit` if the doc directly states the behavior; `implicit` if it logically follows but is not stated outright.
5. **invalid_cases** — cover type errors, value errors, and constraint violations with a concrete example each.
6. **expected_behavior** — express as a checkable assertion or a raised exception, not prose.

