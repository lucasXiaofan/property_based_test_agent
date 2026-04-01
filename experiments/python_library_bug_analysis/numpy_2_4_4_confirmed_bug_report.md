# NumPy 2.4.4 confirmed bug inventory

## Scope

- Local target version: `numpy==2.4.4`
- Latest stable version confirmed from PyPI on `2026-03-31`; release timestamp for `2.4.4` is `2026-03-29`
- Discovery sources:
  - `gh search issues --repo numpy/numpy --state closed --label '00 - Bug'`
  - `gh api graphql` review of issue comments and closing PR links
- Inclusion rule: library-logic bugs only; no pure performance, build, install, hardware-specific, or environment-rooted cases
- Confirmation rule: maintainer-authored confirmation, maintainer-authored fix path, or issue closure by a merged fix PR

## Headline result

- Counted confirmed cases: `5`
- Cases reproduced on local `numpy 2.4.4`: `3`

The counted set is `#31104`, `#31019`, `#30909`, `#30883`, and `#30641`.

Reproduced locally on `2.4.4`:

- `#31104`
- `#31019`
- `#30909`

Confirmed but not reproduced on local `2.4.4`:

- `#30883`
- `#30641`

Important exclusions from the broader `gh` search:

- `#31081` and `#30732`: closed as indexing/precedence confusion, not library bugs
- `#30591`: maintainer discussion concluded the behavior was an acceptable deprecation finalization, not a counted bug for this inventory
- free-threading, build, and platform-specific crash reports were excluded when the failure was primarily environment- or runtime-specific rather than ordinary library logic

## Exceptional oracles and input constraints

This NumPy set is dominated by silent semantic bugs rather than crash bugs.

Clear Exceptional-oracle fit:

- `#30883`, where a valid boolean `weekmask` unexpectedly raised instead of constructing a calendar

Semantic-result or state-mutation bugs:

- `#31104` mutates interpreter-global regex cache state
- `#31019` returns `None` for an empty object reduction
- `#30909` silently ignores `out=`
- `#30641` violated the documented return-container contract

Input constraints are still essential:

- object dtype plus empty inputs: `#31019`
- APIs with writable `out=` buffers: `#30909`
- valid boolean-array weekmasks: `#30883`
- the `copy=` flag split: `#30641`
- import-path side effects: `#31104`

The practical conclusion is that Exceptional oracles alone are weak for NumPy 2.4.4. The higher-yield strategy is semantic comparison against API contracts:

- `out=` should be honored
- empty reductions should return an identity-like result, not `None`
- documented container types should not depend on a flag that is unrelated to API shape
- importing a submodule should not silently mutate unrelated interpreter-global state

## Whether documentation helps

Documentation is highly useful for:

- `#30909`, because `out=` semantics are explicit
- `#30883`, because `weekmask` accepts array-like weekday masks
- `#30641`, because the return type contract is documented

Documentation helps only partially for:

- `#31019`, where the surprising behavior is easy to spot but the exact empty-object reduction contract is less explicit

Documentation is weak for:

- `#31104`, because the bug is an import-time side effect outside the public API contract

So for NumPy 2.4.4, docs are most valuable when the bug is an API contract violation, and much less valuable when the problem is hidden internal state mutation.

## Full case matrix

| Issue | Local 2.4.4 | Exceptional oracle? | Input constraints | Docs help? |
| --- | --- | --- | --- | --- |
| `#31104` | reproduced | no | moderate | low |
| `#31019` | reproduced | no | essential | partial |
| `#30909` | reproduced | no | essential | high |
| `#30883` | not reproduced | strong | essential | high |
| `#30641` | not reproduced | no | moderate | high |

## Artifact

- Machine-readable inventory: `experiments/python_library_bug_analysis/numpy_2_4_4_confirmed_bug_inventory.json`
