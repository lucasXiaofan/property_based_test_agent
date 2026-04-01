# Django 6.0.3 confirmed bug inventory

## Scope

- Local target version: `Django==6.0.3`
- Local reproduction runtime: Python `3.12`, because Django `6.0.3` requires Python `>=3.12`
- Latest stable version confirmed from PyPI on `2026-03-31`; release timestamp for `6.0.3` is `2026-03-03`
- Discovery sources:
  - `gh search issues --repo django/django --state closed --include-prs '"Fixed #" ...'`
  - targeted PR inspection with `gh pr view`
- Inclusion rule: library-logic bugs only; no docs-only, UI/CSS-only, performance-only, install, or environment-rooted issues
- Confirmation rule: Django uses Trac for ticketing, so closed GitHub fix PRs discovered through `gh` were treated as the GitHub-visible developer-confirmation path

## Headline result

- Counted confirmed cases: `8`
- Cases reproduced on local `Django 6.0.3`: `7`

The counted set is:

- `ticket #20868 / PR #20870`
- `ticket #36966 / PR #20831`
- `ticket #36991 / PR #20934`
- `ticket #36888 / PR #20896`
- `ticket #36998 / PR #20978`
- `ticket #36998 / PR #20976`
- `ticket #33091 / PR #20887`
- `ticket #35758 / PR #20819`

Reproduced locally on `6.0.3`:

- `#20868`, `#36966`, `#36991`, `#36888`, `#36998` before-`as`, `#36998` after-`as`, `#33091`

Confirmed but not reproduced on local `6.0.3`:

- `#35758`

Excluded from the broader March 2026 PR search:

- docs clarifications and release-note-only PRs
- admin/UI layout fixes
- backend- or infrastructure-specific cases that were not ordinary library-logic bugs for a default local setup
- test-only or logging-only fixes without user-facing library semantics

## Exceptional oracles and input constraints

Django 6.0.3 shows a much stronger Exceptional-oracle profile than the NumPy set.

Strong Exceptional-oracle fits:

- `#36966` test client redirect-follow path crashes
- `#36991` malformed header parsing raises an unhandled `LookupError`
- `#36998` both malformed `firstof` grammar variants silently avoid the required `TemplateSyntaxError`
- `#33091` the ORM raises the wrong `FieldError` for an invalid joined update

Non-exceptional or semantic-dispatch bugs:

- `#20868` leap-year arithmetic in `timesince`
- `#36888` `acreate()` dispatches to `save()` instead of `asave()`
- `#35758` scheme handling in `URLField.to_python()`

Input constraints are still critical:

- leap-year boundary dates: `#20868`
- redirecting views plus `follow=True` plus `query_params`: `#36966`
- malformed RFC 2231 encoded parameters: `#36991`
- async model hooks: `#36888`
- invalid template-tag grammar: `#36998`
- multi-table inheritance plus `F()` references across tables: `#33091`

The practical conclusion is that Exceptional oracles are very useful for Django 6.0.3, but they still need tightly-constrained input generation. Most of these bugs only appear under fairly specific parser, routing, async, or ORM-shape conditions.

## Whether documentation helps

Documentation is highly useful for:

- `#36966`, because the test client contract strongly implies redirect following should work with normal request parameters
- `#36991`, because malformed header parsing should not crash request processing
- `#36998`, because template-tag grammar should reject malformed syntax
- `#33091`, because joined-field restrictions in ORM update queries are part of the documented query model
- `#35758`, because URL parsing behavior is user-facing

Documentation helps only partially for:

- `#20868`, where the public contract is broad but the leap-year edge depends on implementation details
- `#36888`, where docs describe async ORM entry points but do not usually foreground the interaction with custom `asave()` overrides

So for Django 6.0.3, documentation review is often useful, but still insufficient on its own. The highest-yield bug-finding strategy is:

- parser-invalid inputs for template and header handling
- stateful request flows for the test client
- async model hooks
- relational ORM shapes such as MTI plus `F()` expressions

## Full case matrix

| Ticket / PR | Local 6.0.3 | Exceptional oracle? | Input constraints | Docs help? |
| --- | --- | --- | --- | --- |
| `#20868 / #20870` | reproduced | no | essential | partial |
| `#36966 / #20831` | reproduced | strong | essential | high |
| `#36991 / #20934` | reproduced | strong | essential | high |
| `#36888 / #20896` | reproduced | no | essential | partial |
| `#36998 / #20978` | reproduced | strong | essential | high |
| `#36998 / #20976` | reproduced | strong | essential | high |
| `#33091 / #20887` | reproduced | strong | essential | high |
| `#35758 / #20819` | not reproduced | no | moderate | high |

## Artifact

- Machine-readable inventory: `experiments/python_library_bug_analysis/django_6_0_3_confirmed_bug_inventory.json`
