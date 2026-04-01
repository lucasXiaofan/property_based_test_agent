# Pandas 3.0.0 confirmed bug inventory

## Scope

- Local target version: `pandas==3.0.0`
- Discovery sources:
  - the prior local snapshot in `experiments/python_library_bug_analysis/pandas_bug_finding_results.json`
  - additional GitHub issue review with `gh issue view ... --comments`
- Inclusion rule: library-logic bugs only; no pure performance, install, environment, or dependency-rooted issues
- Confirmation rule: only issues with maintainer-authored confirmation, maintainer-authored bug filing, or a clearly maintainer-owned fix path

## Headline result

- Counted confirmed cases: `27`
- Supplemental reproduced-but-uncounted case: `#58190`
- New confirmed cases found beyond the existing JSON: `7`
- Counted cases still reproducible on local `pandas 3.0.0`: `8`
- Supplemental reproduced-but-uncounted cases on local `3.0.0`: `1`

The new `gh`-derived additions are `#60980`, `#63420`, `#63889`, `#63899`, `#63920`, `#64044`, and `#64267`. Of those, `#63889`, `#63899`, `#63920`, `#64044`, and `#64267` still reproduce locally on `3.0.0`; `#60980` and `#63420` appear already fixed in the final `3.0.0` wheel.

## New additions beyond the old JSON

| Issue | Local 3.0.0 | Confirmation basis | Why it belongs |
| --- | --- | --- | --- |
| `#60980` | not reproduced | `rhshadrach`: `Confirmed on main.` | Wrong `reindex` result after `unstack` with `Period` data; library logic only |
| `#63420` | not reproduced | maintainer discussion by `loicdiridollou` and `jbrockmendel` isolates the regression | Assigning list of `pd.NaT` into `datetime64[us]` column raised `TypeError` |
| `#63889` | reproduced | maintainer discussion between `rhshadrach` and `jorisvandenbossche` ends in agreement that the constructor should populate the column | `from_records` loses values when the real column label is `None` |
| `#63899` | reproduced | `jorisvandenbossche`: `I can confirm this.` | `DataFrame(Index)` aliasing lets a frame mutation mutate the source `Index` |
| `#63920` | reproduced | maintainer-owned fix path; `rhshadrach` says `I believe I have a fix.` | `groupby(observed=False)` assigns NaN categorical rows to the wrong group |
| `#64044` | reproduced | `rhshadrach`: `Confirmed on main.` | `to_timedelta` handles a shared unit inconsistently for mixed fractional/integer inputs |
| `#64267` | reproduced | `rhshadrach` milestones it for `3.0.2` because the issue is straightforward | `pd.col()` expression syntax is missing `**`, surfacing the wrong user-facing error |

## Exceptional oracles and input constraints

The inventory splits cleanly into two families.

1. Strong Exceptional-oracle fits.
   These are bugs where a generated test can catch the failure just by checking for an unexpected exception or an obviously wrong error path. Clear examples are `#61099`, `#61175`, `#61356`, `#62094`, `#62778`, `#62829`, `#63262`, `#63306`, `#63581`, `#63993`, `#63420`, `#64267`, and supplemental `#58190`.
2. Semantic-result bugs that need relational or differential checking.
   These do not crash; they silently return the wrong values, wrong grouping, wrong alignment, or wrong dtype semantics. Examples are `#58471`, `#59965`, `#60922`, `#61509`, `#61621`, `#62240`, `#62888`, `#63236`, `#63879`, `#60980`, `#63889`, `#63899`, `#63920`, and `#64044`.

Across almost the entire set, input constraints are essential rather than optional. The recurring high-value constraints are:

- non-`ns` temporal units: `#58471`, `#60922`, `#63236`, `#63262`, `#63420`
- missing values combined with categorical/groupby/pivot semantics: `#61356`, `#61509`, `#63879`, `#63920`
- mixed dtype or mixed sentinel values: `#59965`, `#61621`, `#62829`, `#62888`, `#63889`
- shape/alignment edge cases: `#61175`, `#60980`, `#63993`, `#58190`
- CoW or aliasing-sensitive inputs: `#63306`, `#63899`

The practical takeaway is that Exceptional oracles alone are not enough to cover pandas 3.0.0 bug-finding. They are valuable for roughly a dozen cases, but many of the confirmed bugs require constrained input generation plus semantic comparison against an invariant, a simpler equivalent formulation, or prior-version behavior.

## Whether documentation helps

Documentation is most useful for bugs where the API contract is explicit:

- parameter validation or invalid-input behavior: `#62778`, `#62829`, `#64267`
- method semantics with clear user-facing expectations: `#61175`, `#61356`, `#61509`, `#62240`, `#63236`, `#63920`, `#64044`
- constructor or conversion promises: `#63879`, `#63993`

Documentation helps only partially for internal-path bugs where the public contract is broad but the failure depends on a hidden execution path:

- internal dtype/block-state bugs: `#60980`, `#63306`, `#63420`
- non-`ns` datetime/timedelta implementation bugs: `#58471`, `#60922`, `#62094`, `#63262`
- aliasing and CoW invariants: `#63899`

Documentation is least helpful when the issue depends on underdocumented corner semantics rather than a clearly stated contract:

- `#63581` mixed `SparseArray` plus `ndarray` row extraction
- `#63889` `None` as a real column label in `from_records`

So documentation review is useful, but it is not sufficient. For pandas 3.0.0, the best bug-finding yield comes from combining:

- doc-derived constraints for public parameters and advertised behavior
- targeted input partitions for missing values, extension dtypes, non-`ns` temporal units, and aliasing-sensitive objects
- Exceptional oracles for crash paths
- semantic equivalence checks for silent wrong-result bugs

## Full case matrix

| Issue | Source | Local 3.0.0 | Exceptional oracle? | Input constraints | Docs help? |
| --- | --- | --- | --- | --- | --- |
| `#58471` | existing | not reproduced | no | essential | partial |
| `#59965` | existing | not reproduced | no | essential | high |
| `#60922` | existing | not reproduced | no | essential | partial |
| `#61099` | existing | not reproduced | strong | essential | partial |
| `#61175` | existing | not reproduced | strong | essential | high |
| `#61356` | existing | not reproduced | strong | essential | high |
| `#61509` | existing | not reproduced | no | essential | high |
| `#61621` | existing | not reproduced | no | essential | partial |
| `#62094` | existing | not reproduced | strong | essential | partial |
| `#62240` | existing | not reproduced | no | essential | high |
| `#62595` | existing | not reproduced | no | moderate | partial |
| `#62778` | existing | not reproduced | strong | essential | high |
| `#62829` | existing | not reproduced | strong | essential | high |
| `#62888` | existing | reproduced | no | essential | high |
| `#63236` | existing | not reproduced | no | essential | high |
| `#63262` | existing | not reproduced | strong | essential | partial |
| `#63306` | existing | not reproduced | strong | essential | partial |
| `#63581` | existing | not reproduced | strong | essential | low |
| `#63879` | existing | reproduced | no | essential | high |
| `#63993` | existing | reproduced | strong | essential | high |
| `#58190` | existing supplemental | reproduced | strong | essential | high |
| `#60980` | gh new | not reproduced | no | essential | partial |
| `#63420` | gh new | not reproduced | strong | essential | partial |
| `#63889` | gh new | reproduced | no | essential | low |
| `#63899` | gh new | reproduced | no | essential | partial |
| `#63920` | gh new | reproduced | no | essential | high |
| `#64044` | gh new | reproduced | no | essential | high |
| `#64267` | gh new | reproduced | strong | moderate | high |

## Artifact

- Machine-readable inventory: `experiments/python_library_bug_analysis/pandas_3_0_0_confirmed_bug_inventory.json`
