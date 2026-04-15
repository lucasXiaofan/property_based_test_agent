# Overall Analysis of Confirmed Bugs Across Pandas 3.0.0, NumPy 2.4.4, and Django 6.0.3

## Scope and method

This report aggregates the counted confirmed bugs from:

- `experiments/python_library_bug_analysis/pandas_3_0_0_confirmed_bug_report.md`
- `experiments/python_library_bug_analysis/numpy_2_4_4_confirmed_bug_report.md`
- `experiments/python_library_bug_analysis/django_6_0_3_confirmed_bug_report.md`

Only the counted confirmed cases are included. The supplemental uncounted pandas case `#58190` is excluded from all percentages below.

Total counted confirmed bugs across the three reports: `40`

- Pandas: `27` (`67.5%`)
- NumPy: `5` (`12.5%`)
- Django: `8` (`20.0%`)

## Executive summary

The combined result is clear:

- `100%` of the counted confirmed cases are library-logic issues.
- `0%` of the counted confirmed cases are merely developer expectation mismatches.

That conclusion follows directly from the inclusion rules used by all three source reports, which explicitly kept library-logic bugs and excluded pure confusion, deprecation-finalization expectations, environment issues, docs-only items, and other non-library-defect cases.

Documentation is still relevant to most confirmed bugs, but usually not enough by itself:

- docs highly relevant: `23 / 40` = `57.5%`
- docs partially relevant: `14 / 40` = `35.0%`
- docs weak or low relevance: `3 / 40` = `7.5%`

So, if "relevant to documentation" means at least partially supported by documented API behavior, then:

- doc-relevant confirmed bugs: `37 / 40` = `92.5%`
- not meaningfully doc-driven: `3 / 40` = `7.5%`

## Library logic vs. expectation mismatch

### Counted confirmed set

- Library logic issue: `40 / 40` = `100%`
- Developer expected-behavior mismatch: `0 / 40` = `0%`

### Interpretation

The confirmed inventories were already filtered to remove non-bugs. That means this dataset is useful for studying real library defects, but it is not a balanced sample of all user-reported problems. In particular, it under-represents "user expected X, library correctly does Y" cases because those were intentionally excluded.

NumPy makes this especially explicit by excluding:

- indexing or precedence confusion cases
- acceptable deprecation-finalization behavior

So the right conclusion is not "expectation mismatch never happens"; it is "expectation mismatch was filtered out before confirmation counting."

## How relevant documentation is

### Main result

Across the confirmed cases, documentation helps in most bugs, but usually as a contract signal rather than a full bug oracle.

| Docs relevance | Count | Percentage |
| --- | ---: | ---: |
| High | 23 | 57.5% |
| Partial | 14 | 35.0% |
| Low | 3 | 7.5% |

### What this means

1. Documentation is useful for finding the target behavior in most cases.
2. Documentation alone is rarely enough to generate the failing input.
3. The missing ingredient is usually constrained input generation around edge conditions such as invalid grammar, missing values, dtype/unit combinations, aliasing, async dispatch, and shape/alignment corner cases.

In practice, the strongest workflow is:

- use docs to define the contract
- use structured edge-case generation to reach the bad execution path
- use either an exceptional oracle or a semantic comparison oracle to detect failure

## Dominant issue-reason categories

The categories below assign one dominant reason to each confirmed issue so the percentages sum to `100%`.

| Dominant reason category | Count | Percentage |
| --- | ---: | ---: |
| Validation, parser, or error-path defect | 18 | 45.0% |
| Wrong semantic result or contract output | 10 | 25.0% |
| Type, dtype, unit, or conversion defect | 6 | 15.0% |
| Hidden state mutation, aliasing, or dispatch defect | 4 | 10.0% |
| Boundary arithmetic or normalization defect | 2 | 5.0% |

### 1. Validation, parser, or error-path defect: `18 / 40` = `45.0%`

This is the single largest category.

Typical pattern:

- valid input raises unexpectedly
- invalid input does not raise the required error
- parser or validation code takes the wrong branch
- the library surfaces the wrong exception or crashes in an unhandled way

Examples from the source reports:

- pandas parameter-validation and wrong-error-path cases
- NumPy `weekmask` raising on valid input
- Django template parsing, header parsing, redirect-follow handling, and invalid joined update errors

This category matters because it is a strong fit for Exceptional-oracle testing.

### 2. Wrong semantic result or contract output: `10 / 40` = `25.0%`

These bugs do not mainly fail by crashing. They return the wrong value, wrong grouping, wrong alignment, wrong container behavior, or otherwise violate the intended API result.

Typical pattern:

- wrong grouped output
- wrong alignment or reshape result
- output argument or return-container contract not honored
- silent semantic corruption rather than an exception

Examples from the source reports:

- pandas wrong `reindex`, grouping, alignment, and constructor-result cases
- NumPy ignoring `out=` or violating return-container expectations

This category is the clearest argument that differential or relational oracles are necessary, not just exception checking.

### 3. Type, dtype, unit, or conversion defect: `6 / 40` = `15.0%`

These bugs concentrate around coercion and representation boundaries.

Typical pattern:

- non-`ns` temporal-unit handling breaks
- mixed sentinel or mixed dtype inputs take the wrong conversion path
- conversion APIs behave inconsistently for equivalent logical inputs

Examples from the source reports:

- pandas non-`ns` datetime/timedelta issues
- pandas `pd.NaT` assignment and `to_timedelta` inconsistencies

This category shows that type-partitioned generation is not optional for numerical and dataframe libraries.

### 4. Hidden state mutation, aliasing, or dispatch defect: `4 / 40` = `10.0%`

These bugs are driven by side effects rather than a local input/output mismatch.

Typical pattern:

- an object shares state unexpectedly
- import-time or constructor-time behavior mutates external state
- the library dispatches to the wrong sync/async path

Examples from the source reports:

- pandas aliasing and CoW-sensitive behavior
- NumPy import-path mutation of interpreter-global regex cache state
- Django `acreate()` dispatching to `save()` instead of `asave()`

These cases are less directly documented and often require stateful tests.

### 5. Boundary arithmetic or normalization defect: `2 / 40` = `5.0%`

This is the smallest category, but it is still important because boundary logic often survives ordinary tests.

Typical pattern:

- leap-year boundary calculations
- scheme or normalization handling at canonicalization boundaries

Examples from the source reports:

- Django `timesince` leap-year arithmetic
- Django `URLField.to_python()` scheme handling

## Cross-library conclusions

### 1. Documentation is necessary but not sufficient

`92.5%` of confirmed bugs are at least partially connected to documented behavior, but only `57.5%` are strongly doc-driven. The gap between those numbers is the space where edge-case input generation matters most.

### 2. Real confirmed bugs are mostly not expectation mismatches

Within this filtered dataset, the confirmed issues are genuine library defects, not user misunderstandings. That makes the dataset useful for evaluating bug-finding methods, but it also means it should not be used to estimate how often raw bug reports are actually user confusion.

### 3. Error-path bugs are the largest single family

At `45.0%`, validation/parser/error-path defects are the biggest category. This supports investing in exception-focused property tests, especially for parser-heavy and validation-heavy APIs.

### 4. Silent semantic bugs are still the majority when combined

If we combine:

- wrong semantic result or contract output (`25.0%`)
- type/dtype/unit/conversion defects (`15.0%`)
- hidden state or aliasing/dispatch defects (`10.0%`)
- boundary arithmetic or normalization defects (`5.0%`)

then non-trivial non-parser semantic behavior accounts for `55.0%` of confirmed bugs.

That means exception checking alone is not enough for these libraries.

## Practical implication for bug finding

The best overall strategy suggested by this dataset is:

1. Start from documented contracts when they exist.
2. Generate inputs around known high-risk partitions:
   - invalid syntax and malformed parser inputs
   - missing values and categorical/groupby combinations
   - non-default dtype and temporal-unit combinations
   - aliasing-sensitive and stateful object construction
   - async-dispatch and request-flow state transitions
3. Use two oracle families together:
   - exceptional oracles for validation/parser/error-path bugs
   - semantic or differential oracles for silent wrong-result bugs

## Bottom line

For these three reports, confirmed bugs are overwhelmingly real library-logic defects rather than expectation mismatches, and most are at least partially grounded in documented behavior. However, the dominant failure mechanism is not just "docs were wrong" or "users misunderstood the API." The actual pattern is that the libraries usually violate their contract only under narrow edge-case conditions, which means documentation review must be combined with targeted input partitioning and stronger semantic oracles.
