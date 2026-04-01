You are generating property-based tests from an IR derived from library documentation.

Goal:
- Find potential implementation bugs, not just confirm documented examples.
- Prefer small, adversarial, high-signal cases over broad generic randomness.

Inputs:
- IR JSON for one target API
- Allowed files/directories
- Existing test file path to write

output location 
```text
experiments/oracle_generation//
  <library>/
    <module_or_class>/
      <function>/
        ir_generated_test.py
```

Required workflow:
1. Read the IR pre_condition partitions, and post_condition entries.
2. Expand each documented partition into boundary, degenerate, and adversarial variants when that can be done without contradicting the docs.
3. Generate tests in two groups:
   - focused single-property tests
   - interaction tests that combine high-risk parameters
4. Prefer tests that can fail on subtle implementation bugs:
   - null handling
   - dtype coercion
   - ordering stability
   - copy/view behavior
   - regex/literal confusion
   - monotonicity preconditions
   - category/observed/dropna interactions
6. For negative behaviors, use `pytest.raises(...)` or warning assertions, not `assert isinstance(result, Exception)`.
7. For each test, keep the generator minimal and explain the intended bug surface in a short comment only when the purpose is non-obvious.
8. Do not generate only happy-path tests. At least half of the tests must target edge cases, invalid combinations, or multi-parameter interactions.
9. When a direct expected result is hard to compute, use a stronger oracle:
   - differential comparison
   - metamorphic relation
   - invariant check
10. Before finishing, run the generated tests if possible and fix obvious issues.

Output requirements:
- Write pytest + hypothesis tests only.
- Reuse IR partitions where useful, but refine generators for bug-finding.
- Keep each test narrow and interpretable.

Helpful resources:
References:
- Basic tutorial: https://hypothesis.readthedocs.io/en/latest/quickstart.html
- Strategies reference: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
- NumPy strategies: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#numpy
- Pandas strategies: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas