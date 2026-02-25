# IR -> Test Guideline (v1)

## Goal
Given `ir2test_pipeline/<library>/<module_or_class>/<function>/ir.json`, generate pytest + Hypothesis tests.

## Test Output Position (Required)
Write tests to:

```text
pandas_bug_finding/ir_test/<library>/<module_or_class>/<function>/ir_generated_test.py
```

Example:

```text
ir2test_pipeline/pandas/DataFrame/reindex/ir_generated_test.py
```

Create missing directories if needed.

## Required Tools
- `pytest`
- `hypothesis`

References:
- Basic tutorial: https://hypothesis.readthedocs.io/en/latest/quickstart.html
- Strategies reference: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
- NumPy strategies: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#numpy
- Pandas strategies: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas

## Generation Rules
1. Convert each valid post-condition (`P*`) into at least one property test.
2. Convert each invalid post-condition (`E*`) into exception tests (`pytest.raises`).
3. Use Hypothesis strategies that match pre-condition partitions.
4. Prioritize interaction hints for combined-parameter tests.
5. Keep tests deterministic enough for CI (`@settings(...)` as needed).

## Minimal Test File Structure
- imports (`pytest`, `hypothesis`, target library)
- shared strategies
- `TestValidContracts` for `P*`
- `TestInvalidContracts` for `E*`

## Run Command
```bash
uv run pytest <generated_test_file> -q
```

## Quality Checklist
- Every `P*` covered.
- Every `E*` covered.
- No fabricated behavior beyond IR.
- File is in required output position.
