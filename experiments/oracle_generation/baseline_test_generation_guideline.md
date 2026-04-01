## Folder Structure Rule (Added)

```text
experiments/oracle_generation//
  <library>/
    <module_or_class>/
      <function>/
        baseline_test.py
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

you will be given with url/doc and function name, and you need to generate test cases to exhaustively cover the important properties of the function.