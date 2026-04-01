# IR v2 to Bug-Finding Test Strategy

## Scope

This report is based on the `ir_v2.json` files under `experiments/oracle_generation/pandas`, including:

- `DataFrame.reindex`
- `DataFrame.groupby`
- `Index.astype`
- `Series.str.contains`

The current IR is useful as a documentation-grounded seed, but by itself it is still biased toward example coverage and happy-path assertions. A code agent can generate much stronger tests from it if the prompt and toolchain push it to synthesize adversarial cases, combine partitions intentionally, and validate behaviors with stronger oracle types than single direct assertions.

## What The Current IR Already Gives You

The existing IR has three strong properties:

- It anchors generation to a specific API, version, and doc URL.
- It already encodes parameter partitions as executable Hypothesis strategy strings.
- It captures many directly documented behaviors as testable post-conditions.

That makes it a good starting point for test synthesis. The agent does not need to infer the public API surface from scratch.

## Where The Current IR Falls Short

The main limitation is that the IR mostly describes documented valid cases. Real bug-finding needs more than that.

### 1. It under-specifies invalid and boundary inputs

Examples from the current IR:

- `Index.astype` models an incompatible dtype, but the post-condition encodes the exception as `assert isinstance(result, TypeError)`. For generation, this should instead become an exception oracle using `pytest.raises`.
- `DataFrame.reindex` includes partitions such as monotonic index and fill methods, but does not explicitly model incompatible combinations like `method` with a non-monotonic index, shape-sensitive `tolerance`, or conflicting axis argument patterns.

If the IR does not explicitly mark negative cases, an agent will overproduce valid-input tests.

### 2. It treats partitions independently instead of as interaction space

Most pandas bugs live in parameter interactions:

- `reindex(method=..., tolerance=..., limit=...)`
- `str.contains(pat, regex, case, flags, na)`
- `groupby(by, level, observed, dropna, sort, as_index)`

The IR lists these partitions, but does not rank or annotate high-risk combinations. A naive agent will sample one partition per parameter without prioritizing the interactions most likely to break.

### 3. It overuses direct-output assertions and underuses stronger oracles

Current post-conditions are mostly of the form:

- type checks
- shape checks
- direct equality against a simple expression

Those are necessary, but weak. High-value bug-finding tests usually need:

- metamorphic relations
- differential checks against an equivalent implementation
- round-trip or idempotence properties
- exception consistency properties
- invariants on labels, ordering, null propagation, and dtype stability

### 4. It lacks explicit oracle category metadata

The agent currently has to infer whether a post-condition should become:

- a direct assertion
- a `pytest.raises` check
- a metamorphic property
- a differential oracle
- a state/non-mutation invariant

That inference is possible, but expensive and inconsistent across runs.

### 5. Some post-conditions are not immediately executable as written

Examples:

- `DataFrame.reindex` uses names like `new_index` or `new_columns` inside `expected_behavior`, while the IR schema only identifies partitions, not local variable binding names.
- Several assertions depend on setup logic or equivalent-reference computation that is not represented in the IR.

This increases the chance that the agent writes brittle or syntactically incorrect tests.

## Strategy To Generate High-Quality Tests From IR v2

The best strategy is: use the current IR as a seed, then force the agent through a second synthesis layer that expands each documented property into bug-oriented test families.

### Layer 1: Convert IR partitions into risk-oriented input families

For each parameter partition, the agent should derive:

- nominal cases
- boundary cases
- degenerate cases
- adversarial cases
- interaction-triggering cases

Examples:

- For `Series.str.contains`, a literal pattern should expand into empty strings, mixed case, regex metacharacters used literally, null-containing series, very short and repeated strings, and strings where only one character changes the answer.
- For `DataFrame.reindex`, a valid index partition should expand into empty index, duplicated labels if allowed upstream, disjoint labels, partial overlap, monotonic and non-monotonic variants, and fill-value values that stress dtype coercion.

The key rule: do not sample uniformly from all partitions. Bias toward boundaries and interactions.

### Layer 2: Turn each documented post-condition into one or more oracle patterns

Every post-condition should be classified into one or more of:

- direct oracle: exact expected result can be computed cheaply
- exception oracle: behavior should raise a specific exception or warning
- metamorphic oracle: result should satisfy a relation after a controlled transformation
- differential oracle: compare against another implementation path
- invariant oracle: output preserves shape, labels, ordering, missingness, dtype, or non-mutation properties

Examples:

- `Index.astype`:
  - direct oracle: values match elementwise conversion where conversion is valid
  - exception oracle: incompatible casts raise
  - metamorphic oracle: casting to the same normalized dtype twice is idempotent
  - invariant oracle: length preserved

- `Series.str.contains`:
  - direct oracle: compare against `re.search` or substring membership
  - metamorphic oracle: toggling `case=False` should never flip a `True` match to `False` solely because of case differences
  - differential oracle: `regex=False` should agree with plain Python substring logic
  - invariant oracle: null positions respect `na=...`

- `DataFrame.groupby`:
  - differential oracle: compare `sort=False` ordering against first-occurrence order in the input
  - invariant oracle: partitioning rows by groups preserves row count after aggregation or regrouped inspection
  - metamorphic oracle: permuting rows should not change aggregate values, though it may change key order under `sort=False`

### Layer 3: Synthesize interaction-focused tests explicitly

The agent should generate tests in two buckets:

- single-property tests: one core property, minimized setup
- interaction tests: 2-4 parameters intentionally combined because the combination is risky

Recommended interaction heuristics:

- combine every optional parameter with the one most likely to constrain it
- combine dtype-sensitive inputs with fill/default/null behavior
- combine ordering-related options with duplicates, empties, and partial overlap
- combine regex options with literal metacharacters and flags

This matters because many library bugs only appear when individually valid options are used together.

### Layer 4: Prefer minimal reproducible generators over broad random generators

Broad generators create noise. Bug-finding works better when generators are shaped around failure modes.

Prefer:

- small container sizes
- high probability of empty or singleton cases
- high probability of repeated values
- high probability of nulls when null semantics matter
- explicit inclusion of edge tokens such as `""`, `"."`, `"|"`, `"0"`, `None`, `np.nan`

The agent should use targeted `st.one_of(...)`, weighted small domains, and conditional composite strategies instead of only generic `st.text()` or `st.lists(...)`.

### Layer 5: Add mutation and aliasing checks

Pandas bugs often involve accidental mutation, view/copy confusion, or unstable metadata propagation.

For every generated test where feasible, the agent should also consider:

- input object unchanged unless mutation is documented
- index and column metadata preserved where expected
- object identity expectations for `copy=True` and `copy=False`
- dtype and missingness stability

These are cheap and often catch real regressions.

## Recommended Extensions To The IR Schema

The current schema can still be used, but the following additions would make generation much more reliable:

### Add an `oracle_kind` field to each post-condition

Allowed values:

- `direct`
- `exception`
- `metamorphic`
- `differential`
- `invariant`

This tells the agent what sort of test to write.

### Add an `expected_exception` field for negative cases

Instead of encoding exceptions as returned values, represent them structurally:

```json
{
  "id": "incompatible_dtype_raises_type_error",
  "oracle_kind": "exception",
  "expected_exception": "TypeError"
}
```

### Add a `risk` field to partitions and post-conditions

Allowed values:

- `low`
- `medium`
- `high`

This lets the agent spend its budget on the most failure-prone combinations first.

### Add `interaction_with` hints

Example:

```json
{
  "id": "method_ffill",
  "interaction_with": ["self_monotonic_index_df", "limit_positive_int", "tolerance_scalar"]
}
```

This tells the agent which combinations deserve dedicated tests.

### Add `binding_name` or argument mapping metadata

If a post-condition references `new_index`, `new_columns`, or other local names, the IR should expose how those names map to actual function arguments.

## Prompting Strategy For The LLM Code Agent

The prompt should not just say "generate tests from this IR." That will usually produce shallow tests. The prompt should instruct the agent to transform documentation-derived properties into bug-finding test families.

## Recommended Prompt Template

```md
You are generating property-based tests from an IR derived from library documentation.

Goal:
- Find potential implementation bugs, not just confirm documented examples.
- Prefer small, adversarial, high-signal cases over broad generic randomness.

Inputs:
- IR JSON for one target API
- Allowed files/directories
- Existing test file path to write

Required workflow:
1. Read the IR metadata, pre_condition partitions, and post_condition entries.
2. Classify each post_condition into one of: direct, exception, metamorphic, differential, invariant.
3. Expand each documented partition into boundary, degenerate, and adversarial variants when that can be done without contradicting the docs.
4. Generate tests in two groups:
   - focused single-property tests
   - interaction tests that combine high-risk parameters
5. Prefer tests that can fail on subtle implementation bugs:
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
```

## Prompt Addendum For Stronger Results

These extra instructions are worth adding:

- "If the IR contains only documented valid cases, infer nearby invalid or boundary cases from the function signature and semantics, then test them."
- "Prefer one strong oracle over many weak assertions."
- "When parameter interactions are likely to be buggy, write dedicated tests for those combinations even if the docs present them separately."
- "If the IR post-condition is underspecified, compute an equivalent reference implementation or metamorphic relation instead of skipping the case."

## Tools The LLM Code Agent Needs

To turn this IR into high-quality tests reliably, the code agent needs more than file editing.

### Required runtime tools

- `pytest`
- `hypothesis`
- `pandas`
- `numpy`
- `re`

These are the execution dependencies for the produced tests.

### Required code-agent capabilities

- file search and read
- JSON parsing
- code editing
- test execution
- failure output inspection
- iterative repair after test failures

Without execution and repair, the agent will often leave behind syntactically valid but semantically broken tests.

### High-value helper tools

- a small IR validator that checks schema consistency
- a strategy evaluator that can `eval()` partition constraints in a safe harness
- a test skeleton generator that converts IR entries into pytest function stubs
- a minimization loop that reruns failing Hypothesis examples and preserves compact repros

### Strongly recommended derived tools

#### 1. IR linter

Checks for:

- missing partition references
- undefined variable names in `expected_behavior`
- exception behaviors encoded as normal return assertions
- contradictory partitions

This would catch issues already visible in the current IR examples.

#### 2. Oracle synthesizer

A helper that maps each post-condition to a preferred test style:

- direct assert
- `pytest.raises`
- invariant helper
- metamorphic helper
- reference-comparison helper

#### 3. Interaction planner

A helper that ranks parameter combinations by bug-finding potential, for example:

- optional parameters with documented constraints
- null-sensitive plus dtype-sensitive parameters
- ordering plus grouping parameters
- regex plus escaping plus case/flags parameters

This prevents the agent from spending all of its budget on low-risk combinations.

#### 4. Execution harness

A wrapper that:

- runs only the newly generated test file
- captures assertion and exception output
- surfaces minimal failing examples
- lets the agent patch the tests when generation mistakes are obvious

## Recommended End-To-End Workflow

1. Load one `ir_v2.json`.
2. Lint the IR.
3. Classify post-conditions by oracle type.
4. Expand partitions into bug-oriented generators.
5. Rank high-risk parameter interactions.
6. Generate a small set of focused tests first.
7. Run them.
8. Repair generation errors.
9. Add interaction-heavy tests.
10. Run again and keep the set that is stable, interpretable, and high-signal.

This is better than generating a large one-shot test file because it creates feedback and catches prompt-induced mistakes early.

## Concrete Recommendations For This Repository

For this repository, the most useful next step is not replacing `ir_v2.json`. It is adding a second-stage prompt and helper tooling on top of it.

Recommended changes:

- keep `ir_v2.json` as the documentation-grounded seed
- add an IR lint step before test generation
- add oracle classification and interaction planning as explicit generation steps
- update the generation prompt so at least half of generated tests are edge-case or interaction focused
- represent exceptions structurally instead of as returned `result` objects

## Bottom Line

`ir_v2.json` is a good documentation-to-testing bridge, but it is not yet a bug-finding specification.

To generate high-quality tests from it, the code agent needs:

- a prompt that explicitly prioritizes adversarial and interaction-heavy cases
- stronger oracle selection than simple direct assertions
- tools to lint IR, plan combinations, execute tests, and repair mistakes

If those pieces are added, the current IR can support much stronger pandas test generation without requiring a full schema redesign.
