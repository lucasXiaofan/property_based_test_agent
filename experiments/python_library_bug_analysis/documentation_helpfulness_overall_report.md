# Documentation Helpfulness for Confirmed Bug Reproduction

## Scope

This report summarizes how useful the upstream documentation is for reproducing the confirmed bugs listed in:

- `experiments/python_library_bug_analysis/pandas_3_0_0_confirmed_bug_report.md`
- `experiments/python_library_bug_analysis/numpy_2_4_4_confirmed_bug_report.md`
- `experiments/python_library_bug_analysis/django_6_0_3_confirmed_bug_report.md`

The analysis uses the case matrices in the corresponding JSON inventories together with targeted upstream documentation inspection through `gh`.

## Evaluation categories

- `high`
  The documentation states a user-facing contract clearly enough that it can directly suggest a reproducer or an expected result.
- `partial`
  The documentation is relevant to the public API, but the actual bug depends on an edge case, hidden execution path, or under-emphasized corner condition.
- `low`
  The documentation is not very helpful for reproduction because the bug is mostly internal, side-effect-driven, or depends on semantics that are barely documented.

## Overall result

Across all three libraries, there are `41` total cases in the case matrices.

| Docs help category | Count | Percentage |
| --- | ---: | ---: |
| `high` | 24 | 58.5% |
| `partial` | 14 | 34.1% |
| `low` | 3 | 7.3% |

If `high` and `partial` are both counted as "documentation helped at least somewhat", then docs helped in `38 / 41` cases, or `92.7%`.

For counted cases only, excluding the supplemental pandas case, there are `40` cases:

| Docs help category | Count | Percentage |
| --- | ---: | ---: |
| `high` | 23 | 57.5% |
| `partial` | 14 | 35.0% |
| `low` | 3 | 7.5% |

## Per-library result

### Pandas 3.0.0

Total cases in matrix: `28`

| Docs help category | Count | Percentage |
| --- | ---: | ---: |
| `high` | 15 | 53.6% |
| `partial` | 11 | 39.3% |
| `low` | 2 | 7.1% |

Docs helped at least somewhat in `26 / 28` cases, or `92.9%`.

Interpretation:

- Pandas documentation is often useful when the bug violates explicit API behavior such as `pivot_table`, `groupby`, `factorize`, `json_normalize`, or parameter validation contracts.
- Pandas documentation is only partially useful when the failure depends on non-`ns` temporal units, internal dtype/block behavior, aliasing, or copy-on-write edge cases.
- Pandas documentation is weakest for obscure corner semantics such as `None` as a real `from_records` column label or mixed sparse/object extraction behavior.

### NumPy 2.4.4

Total cases in matrix: `5`

| Docs help category | Count | Percentage |
| --- | ---: | ---: |
| `high` | 3 | 60.0% |
| `partial` | 1 | 20.0% |
| `low` | 1 | 20.0% |

Docs helped at least somewhat in `4 / 5` cases, or `80.0%`.

Interpretation:

- NumPy documentation is strongest when the bug is a direct API contract violation, such as honoring `out=`, accepting valid `weekmask` inputs, or preserving a documented return container.
- NumPy documentation is only partially useful when the docs expose the API but do not strongly specify the edge behavior, such as empty object reductions in `vecdot`.
- NumPy documentation is weakest when the bug is an internal import-time side effect rather than a public API contract problem.

### Django 6.0.3

Total cases in matrix: `8`

| Docs help category | Count | Percentage |
| --- | ---: | ---: |
| `high` | 6 | 75.0% |
| `partial` | 2 | 25.0% |

Docs helped at least somewhat in `8 / 8` cases, or `100.0%`.

Interpretation:

- Django documentation is the most directly useful of the three libraries because it often specifies public behavior precisely enough to imply both valid usage and failure expectations.
- The strongest examples are the test client API, template-tag syntax, ORM update restrictions, and user-facing URL parsing behavior.
- Django documentation becomes only partially useful when the public contract is broad but the bug depends on a subtle implementation boundary, such as leap-year arithmetic or `acreate()` dispatching to `save()` instead of `asave()`.

## Reason categories for why docs help or do not help

### 1. Explicit contract or grammar documentation

These are cases where documentation is highly relevant because it gives a direct contract that can be turned into a reproducer.

Common patterns:

- parameter validation rules
- accepted input forms
- redirect-follow behavior
- template-tag grammar
- return type or output container guarantees
- aggregation and labeling semantics

Examples:

- pandas `pivot_table(..., margins=True)`
- pandas `factorize`
- numpy FFT functions with `out=`
- numpy `meshgrid` return-type contract
- django test client `query_params` plus `follow=True`
- django `firstof` syntax

### 2. Public API documented, edge condition under-specified

These are cases where documentation helps only partially because it gets you to the right function, but not to the exact failing state.

Common patterns:

- leap-year boundaries
- malformed but parser-adjacent inputs
- async override dispatch
- missing values combined with categorical/groupby behavior
- non-`ns` datetime or timedelta units
- aliasing or copy-on-write sensitive objects

In these cases, documentation narrows the search space, but reproducing the bug still depends on specialized input construction.

### 3. Internal-path or underdocumented-corner bugs

These are cases where documentation is weak or not very helpful.

Common patterns:

- import-time global side effects
- internal block-manager or storage-path quirks
- obscure constructor corner cases
- behavior that is technically user-visible but not meaningfully documented as a contract

Examples:

- NumPy `f2py.crackfortran` mutating `re._MAXCACHE`
- pandas `from_records` with a real `None` column label

## Upstream documentation evidence checked with `gh`

Targeted `gh` inspection confirmed that several high-help cases are backed by explicit docs:

- Django testing docs explicitly document `Client(..., query_params=...)`, `Client.get(..., follow=False, ..., query_params=None, ...)`, and `redirect_chain`, which makes them highly relevant to the redirect-follow bug.
- Django template docs explicitly document `firstof` usage and `as value` syntax, which makes them highly relevant to malformed `firstof` grammar bugs.
- Pandas reshaping docs explicitly document that `pivot_table(..., margins=True)` adds an `All` row and column, which is highly relevant to the missing-total bug.
- Pandas reshaping docs explicitly document that `factorize` encodes missing values as `-1`, which is highly relevant to factorization-semantics bugs.
- Pandas `from_records` docs identify the constructor and supported input shape, but they do not document the `None`-label edge clearly, which is why that case remains partial or low.
- NumPy repo docs clearly expose `meshgrid` and `vecdot` as public APIs, but the directly visible repo documentation is weaker on the exact corner-case expectations than Django's and some of pandas' docs.

## Bottom line

Documentation is useful for confirmed bug reproduction most of the time, but not all documentation help is equal.

- In `58.5%` of all cases, docs are highly relevant and can directly suggest a reproducer or expected behavior.
- In `34.1%` of cases, docs help only partially because the bug depends on a narrow edge condition.
- In `7.3%` of cases, docs are not very helpful because the bug is mostly internal or underdocumented.

The practical conclusion is:

- use documentation aggressively for API-contract, parser, validation, and return-type bugs
- treat documentation as only a starting point for edge-condition bugs
- do not expect docs alone to surface internal-state, aliasing, or hidden-path failures

## Sources

- Local inventories:
  - `experiments/python_library_bug_analysis/pandas_3_0_0_confirmed_bug_inventory.json`
  - `experiments/python_library_bug_analysis/numpy_2_4_4_confirmed_bug_inventory.json`
  - `experiments/python_library_bug_analysis/django_6_0_3_confirmed_bug_inventory.json`
- Upstream docs inspected with `gh`:
  - <https://github.com/django/django/blob/main/docs/topics/testing/tools.txt>
  - <https://github.com/django/django/blob/main/docs/ref/templates/builtins.txt>
  - <https://github.com/pandas-dev/pandas/blob/main/doc/source/user_guide/reshaping.rst>
  - <https://github.com/pandas-dev/pandas/blob/main/doc/source/user_guide/dsintro.rst>
  - <https://github.com/numpy/numpy/blob/main/doc/source/reference/routines.array-creation.rst>
  - <https://github.com/numpy/numpy/blob/main/doc/source/reference/routines.linalg.rst>
