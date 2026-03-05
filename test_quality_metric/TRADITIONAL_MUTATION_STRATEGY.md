# Traditional Mutation Testing Strategy

## 1  Two Approaches to Traditional Mutation — Comparison

Both approaches use the same classical operator taxonomy but differ fundamentally
in *where* they operate and *how* mutations are generated.

---

### 1.1  The mutmut / libcst approach (reference code)

```
SOURCE FILE (Python AST)
        │
        ▼
  libcst visitor walks every node
        │
        ├─ BaseNumber    → operator_number:    1  →  2
        ├─ BaseString    → operator_string:    "foo" → "XXfooXX" / .lower() / .upper()
        ├─ Assign        → operator_assignment: a = b  →  a = None
        ├─ AugAssign     → operator_augmented_assignment: a += b  →  a = b
        ├─ UnaryOperation→ operator_remove_unary_ops: not x  →  x
        ├─ Call          → operator_arg_removal: f(a, b)  →  f(b), f(a)
        ├─ Call          → operator_dict_arguments: dict(k=v) → dict(kXX=v)
        ├─ BooleanOp/Cmp → operator_swap_op: ==→!= +→- and→or < →<=
        ├─ Name          → operator_name: True→False deepcopy→copy
        ├─ Lambda        → operator_lambda: lambda: None → lambda: 0
        └─ Match         → operator_match: drop case arms
```

**Characteristics**

| Property | Value |
|----------|-------|
| Input | Any Python source file (`.py`) |
| Mutation site | Every literal / operator / call / keyword in the AST |
| Generation | Fully automatic — no human knowledge of the code required |
| Granularity | Many hundreds of mutants per file (every node gets a turn) |
| Requires source | Yes — cannot handle C/Cython extensions (.pyd / .so) |
| Equivalent-mutant risk | High — syntactic changes may leave behavior identical |

---

### 1.2  The wrapper-based approach (this project, `traditional_mutants.py`)

```
LIBRARY FUNCTION (C/Cython — unreadable by AST tools)
        │
        ▼
  reindex_wrapper.py  ← hand-written Python shim with
  (explicit decision    one conditional per parameter
   points)
        │
        ▼
  traditional_mutants.py  ← one function per mutation
  (each function implements one operator applied to
   one decision point)
        │
        ▼
  run_traditional_mutant_eval.py  → monkey-patches
  DataFrame.reindex at runtime and runs pytest
```

**Characteristics**

| Property | Value |
|----------|-------|
| Input | A hand-crafted Python wrapper around the target function |
| Mutation site | Exactly the decision points the wrapper exposes |
| Generation | Semi-manual — wrapper is written once per function, mutations are systematic |
| Granularity | O(parameters × operators) — 15 mutants for reindex |
| Requires source | No — works on any callable, including C extensions |
| Equivalent-mutant risk | Low — every mutation changes what kwargs reach pandas |

---

### 1.3  Operator correspondence

| Classical operator | mutmut / libcst | Wrapper approach |
|--------------------|----------------|-----------------|
| **AOR** Arithmetic | `operator_number`: `n → n+1` | `AOR_limit_plus1`: `limit → limit+1` |
| **ROR** Relational | `operator_swap_op`: `== → !=` `< → <=` | `ROR_axis_eq`: `axis_int == 0 → != 0` |
| **COR** Conditional | `operator_swap_op`: `and → or`; `operator_remove_unary_ops`: `not x → x` | `COR_fill_sentinel`: `not _is_nan() → _is_nan()` |
| **SDL** Statement delete | `operator_arg_removal`: `f(a,b) → f(b)` (closest equivalent) | `SDL_limit`: delete `kwargs['limit']=limit` |
| **SVR** Scalar/var replace | `operator_name`: `True→False`; `operator_assignment`: `a=b → a=None` | `SVR_axis_default`: default return `0 → 1` |
| **MOR** Method / operator | `operator_symmetric_string_methods_swap`: `.lower()→.upper()` | not implemented |
| String mutations | `operator_string`: case / prefix variations | not implemented (irrelevant for numeric API) |
| Dict-arg mutations | `operator_dict_arguments` | not implemented |
| Lambda mutations | `operator_lambda` | not implemented |
| Match-case mutations | `operator_match` | not implemented |

---

### 1.4  Shared foundations

Both approaches share the same theoretical basis:

1. **Mutation ≡ a small, local, syntactically valid change** to the program under test.
2. **Kill condition**: a test detects the mutation iff it observes a behaviorally
   different output.
3. **Mutation score** = killed / (killed + survived) measures test suite
   effectiveness.
4. **Why different mutants test different behaviors**: each mutant perturbs exactly
   one code path.  Only tests whose execution passes through that path can kill it.

---

### 1.5  Key differences

| Dimension | mutmut / libcst | Wrapper approach |
|-----------|----------------|-----------------|
| **Scope** | Any Python file | Functions with C/Cython internals |
| **Scalability** | Fully automated, scales to thousands of mutants | Requires one wrapper per function |
| **Precision** | Many equivalent / trivial mutants (string case, +1 on irrelevant constant) | Mutations map 1-to-1 to observable behavioral contracts |
| **Coupling to tests** | Tests exercise the *implementation source code* | Tests exercise the *documented API surface* |
| **Surviving mutants** | Surviving mutants indicate missing tests OR equivalent mutants | Surviving mutants reliably indicate uncovered behavior |
| **Integration** | `mutmut run` / `cosmic-ray` CLI | `run_mutant_eval.py --mutants-file traditional_mutants.py` |
| **Tooling needed** | `libcst`, `mutmut` | Only standard library + `pytest` |

The core trade-off:
> **mutmut maximises automation at the cost of mutation relevance.**
> **The wrapper approach maximises relevance at the cost of per-function effort.**

A hybrid (§ 3.3 below) captures the benefits of both.

---

## 2  What Makes Mutants "Handle Different Functions"

When applying mutation testing to a function like `pd.DataFrame.reindex`, a
mutant only exercises the behavior it directly perturbs:

```
Function parameter  →  Decision point in wrapper  →  Mutant that kills it  →  Tests that kill it
─────────────────────────────────────────────────────────────────────────────────────────────────
index=              →  kwargs['index'] = index     →  SDL_index             →  test_missing_row_labels_get_nan
columns=            →  kwargs['columns'] = columns →  SDL_columns           →  test_new_column_label_gets_nan
method=             →  kwargs['method'] = method   →  SDL_method            →  test_ffill_propagates_…
fill_value=         →  if not _is_nan_sentinel(…)  →  SDL_fill_value / COR  →  test_fill_value_used_for_…
limit=              →  kwargs['limit'] = limit     →  SDL_limit / AOR       →  test_limit_caps_…
tolerance=          →  kwargs['tolerance'] = …     →  SDL_tolerance         →  test_tolerance_within_…
level=              →  kwargs['level'] = level     →  SDL_level             →  test_multiindex_level_…
axis routing        →  if axis_int == 0            →  ROR_axis_eq / SVR     →  test_labels_axis_columns_…
```

This map is the "mutation ↔ behavior ↔ test" traceability graph.  It explains
why SDL_limit cannot be killed by a test that only calls `df.reindex(new_index)`
without a `limit=` argument — the deleted statement is never reached.

---

## 3  Generalising to Other Functions and Libraries

### 3.1  Decision: wrapper vs. direct AST mutation

```
Is the function implemented in pure Python?
├── YES → use mutmut / libcst directly on the source file.
│         Produces many fine-grained mutants automatically.
└── NO (C / Cython / Rust extension) ─────────────────────────────────────────┐
    │                                                                           │
    Can you fully re-implement the function's parameter-routing logic           │
    in a thin Python shim?                                                      │
    ├── YES → write wrapper.py + traditional_mutants.py (this project's        │
    │         pattern).  Apply mutmut to the wrapper for additional coverage.  │
    └── NO  → use semantics-derived LLM mutants only.                          │
              (behaviour cannot be systematically re-expressed in Python)       │
              ────────────────────────────────────────────────────────────────────┘
```

### 3.2  Wrapper recipe for a new function

For any function `lib.Module.method(param1, param2, ..., paramN)`:

1. **Create `<method>_wrapper.py`**
   - Capture `_ORIGINAL = lib.Module.method` at import time.
   - Write `def method_shim(self_or_first_arg, *, param1=default1, ...):`.
   - For each parameter, write a guarded assignment:
     ```python
     if param_i is not <sentinel>:      # ROR + SDL target
         kwargs['param_i'] = param_i    # SDL target
     ```
   - Call `_ORIGINAL(self_or_first_arg, **kwargs)` at the end.
   - Add a `_resolve_axis`-style helper for any parameter whose value gets
     interpreted/routed (axis names, enum strings, etc.).  That helper is
     the ROR/SVR target.

2. **Create `<method>_traditional_mutants.py`** by systematically applying:

   | Operator | What to write |
   |----------|--------------|
   | SDL per param | Copy the shim; delete one `kwargs[p] = p` assignment |
   | SDL labels block | Delete the entire positional-argument routing block |
   | ROR per guard | Copy the shim; flip `is not None` to `is None` for one guard |
   | ROR axis/routing | Invert the routing condition (e.g. `== 0` → `!= 0`) |
   | COR sentinel | Invert `not _is_sentinel(v)` to `_is_sentinel(v)` |
   | AOR numeric | `limit → limit + 1` or `tolerance → tolerance * 2` |
   | SVR constant | Change the default return in the resolver (e.g. `0 → 1`) |

3. **Create `<method>_traditional_mapping.json`** listing all mutant IDs with
   their operator, target line, and behavior broken.

4. **Run** via the shared evaluator:
   ```bash
   python test_quality_metric/run_mutant_eval.py \
     --pytest-file path/to/test_<method>.py \
     --mutants-file path/to/<method>_traditional_mutants.py \
     --mapping-file path/to/<method>_traditional_mapping.json \
     --output-dir path/to/results/
   ```

### 3.3  Hybrid: run mutmut on the wrapper itself

Because the wrapper is pure Python, `mutmut` can further mutate it automatically,
generating many additional fine-grained mutations beyond the hand-crafted set:

```bash
# install mutmut
uv add mutmut

# run mutmut on the wrapper, using the existing test file as oracle
mutmut run \
  --paths-to-mutate ir2test_pipeline/pandas/DataFrame/reindex/reindex_wrapper.py \
  --tests-dir     ir2test_pipeline/pandas/DataFrame/reindex/ \
  --runner        "pytest baseline_test.py -q"

mutmut results   # view surviving mutants → gaps in test coverage
```

This combines automated exhaustiveness (mutmut covers every AST node in the
wrapper) with semantic relevance (the wrapper only exposes decision points that
affect observable API behavior).

### 3.4  Scaling across multiple pandas functions

The wrapper pattern is mechanical enough to automate:

```
function2test.csv  (list of functions)
        │
        ▼
  generate_wrapper.py  ─── reads function signature via inspect.signature()
        │                   writes <method>_wrapper.py boilerplate
        │
        ▼
  generate_traditional_mutants.py  ─── reads wrapper.py AST
        │                               writes one SDL/ROR/COR per decision point
        │
        ▼
  run_mutant_eval.py  ─── existing infrastructure, unchanged
```

Key automation step: extract all guarded `kwargs[k] = v` assignments from the
wrapper's AST using `libcst`, then generate one SDL mutant per assignment and
one ROR mutant per guard automatically — without any per-function human authoring.

### 3.5  Scaling to other libraries

The approach is library-agnostic.  The only library-specific knowledge required
is the sentinel value (what counts as "not provided"):

| Library | Common sentinel pattern | Wrapper note |
|---------|------------------------|--------------|
| pandas | `np.nan`, `None`, `lib.no_default` | check `pd.api.extensions.no_default` |
| NumPy | `None`, integer defaults like `-1` | guard on `v is not None` or `v != -1` |
| scikit-learn | `None` | `if param is not None: kwargs[p] = param` |
| PyTorch | `None`, `torch.default_generator` | standard `None` guard works |
| stdlib | varies by function | read signature default from `inspect.Parameter.default` |

For libraries implemented in pure Python (e.g. most of stdlib, many smaller
libraries), skip the wrapper entirely and run `mutmut` directly on the source.

### 3.6  Recommended decision flowchart

```
New function to evaluate
        │
        ▼
Is it pure Python?  ──YES──► mutmut directly on source + run_mutant_eval.py
        │
       NO (C extension)
        │
        ▼
Does it have ≥ 3 keyword parameters with sentinel defaults?
        │
       YES ──► write wrapper.py (30–60 min) + traditional_mutants.py (automated
        │      from wrapper AST) + optionally run mutmut on the wrapper
        │
       NO ──► SDL / ROR not productive (too few decision points);
              use LLM semantic mutants instead (existing approach)
```

---

## 4  Open Gaps and Future Work

| Gap | Impact | Possible fix |
|-----|--------|--------------|
| No string / enum mutation in wrapper approach | Misses bugs like wrong method name string `"ffill"` vs `"pad"` | Add `SVR_method_name` mutant: `"ffill" → "pad"` in routing helpers |
| No augmented-assignment mutations | Wrappers rarely use `+=`, so low priority | Covered by mutmut hybrid |
| Equivalent mutants not detected | SDL on an already-None kwarg is a no-op | Add pre-check: only generate SDL mutant for param if at least one test call passes that param |
| Wrapper authoring still manual | Bottleneck at scale | Auto-generate wrapper boilerplate from `inspect.signature()` |
| mutmut integration not wired into pipeline | Hybrid approach is manual | Extend `run_quality_report.sh` with a mutmut step on wrapper files |
| No inter-parameter interaction mutants | Bugs where two params interact (e.g. method+tolerance) are missed | Add combinatorial SDL: delete two params at once |
