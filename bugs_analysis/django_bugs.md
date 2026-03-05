# Django — Recent Closed Logical Bug Analysis

> **Source:** GitHub repository [`django/django`](https://github.com/django/django)
> **Bug Tracker:** [Trac](https://code.djangoproject.com/query) (primary), GitHub PRs (fix delivery)
> **Scope:** Logical bugs in the library itself (excluding documentation, CSS/UI tweaks, and CI-only changes)
> **Date Range:** Jan – Mar 2026
>
> **Django workflow note:** Django does not use GitHub Issues for bug reports. Bugs are filed on
> the [Trac tracker](https://code.djangoproject.com); PRs that carry a "Fixed #XXXXX" title and are
> **merged** by a core developer are the definitive sign of developer confirmation.

---

## Table 1 — Developer-Confirmed Bugs

All entries below correspond to merged GitHub PRs. A merge by a Django core committer constitutes
explicit developer confirmation and acceptance of the bug report.

| # | Trac Ticket | GitHub PR | Title | Component | Merged | Documentation URL |
|---|-------------|-----------|-------|-----------|--------|-------------------|
| 1 | [#36961](https://code.djangoproject.com/ticket/36961) | [PR 20789](https://github.com/django/django/pull/20789) | `TypeError` raised in deprecation warning machinery when Django is imported via a namespace package (`__file__ is None`) | Core / Internals | 2026-03-02 | [Deprecation timeline](https://docs.djangoproject.com/en/stable/internals/deprecation/) |
| 2 | [#36951](https://code.djangoproject.com/ticket/36951) | [PR 20722](https://github.com/django/django/pull/20722) | `log_task_finished` signal handler logs `None Type: None` when no exception occurred — spurious noise in task logs | Async tasks / Signals | 2026-02-25 | [Background tasks](https://docs.djangoproject.com/en/stable/topics/async/#background-tasks) |
| 3 | [#36931](https://code.djangoproject.com/ticket/36931) | [PR 20714](https://github.com/django/django/pull/20714) | Multipart parser crashes with unhandled `LookupError` for file uploads with an invalid RFC 2231–encoded `Content-Disposition` header | Request / Multipart | 2026-02-24 | [HttpRequest.FILES](https://docs.djangoproject.com/en/stable/ref/request-response/#django.http.HttpRequest.FILES) |
| 4 | [#36935](https://code.djangoproject.com/ticket/36935) | [PR 20727](https://github.com/django/django/pull/20727) | `ContentType.app_labeled_name` falls back to only `self.model` when `model_class()` returns `None`, causing ambiguity when multiple apps share a model name | ORM / ContentTypes | 2026-02-21 | [ContentType.app_labeled_name](https://docs.djangoproject.com/en/stable/ref/contrib/contenttypes/#django.contrib.contenttypes.models.ContentType) |
| 5 | [#36921](https://code.djangoproject.com/ticket/36921) | [PR 20679](https://github.com/django/django/pull/20679) | `KeyError` raised when saving an inline model instance whose model is not registered with the admin site | Admin / Inline | 2026-02-11 | [InlineModelAdmin](https://docs.djangoproject.com/en/stable/ref/contrib/admin/#inlinemodeladmin-objects) |
| 6 | [#36903](https://code.djangoproject.com/ticket/36903) | [PR 20646](https://github.com/django/django/pull/20646) | `NameError` when using `inspect.signature()` on Django functions under Python 3.14 due to deferred (PEP 649) annotations | Core / Python 3.14 compat | 2026-02-10 | [import_string](https://docs.djangoproject.com/en/stable/ref/utils/#django.utils.module_loading.import_string) |
| 7 | [#36890](https://code.djangoproject.com/ticket/36890) | [PR 20628](https://github.com/django/django/pull/20628) | `StringAgg(distinct=True)` raises an error on SQLite when using the default delimiter — works correctly on PostgreSQL | ORM / Aggregation | 2026-02-10 | [StringAgg](https://docs.djangoproject.com/en/stable/ref/contrib/postgres/aggregates/#stringagg) |

### Confirmation Details

| Ticket | Fix Author | Merge Committer | Notes |
|--------|-----------|-----------------|-------|
| #36961 | `mariocesar` | Django core dev | Bug introduced while testing a related PR; Trac ticket created from discovery |
| #36951 | `neonmik` | Django core dev | Incorrect `exc_info` format when `exception=None` — log noise in all non-failing tasks |
| #36931 | `Aviah` | Django core dev | RFC 2231 malformed headers previously only triggered `ValueError`; `LookupError` was unhandled |
| #36935 | `Rounin` | Django core dev | Bug introduced by a previous PR; needed fallback to include `app_label` for disambiguation |
| #36921 | `Alasdair` | Django core dev | Regression from PR #18934; added guard to avoid `KeyError` on unregistered models |
| #36903 | `Tim Graham` | Django core dev | PEP 649 deferred annotations in Python 3.14 break `inspect.signature()` across Django |
| #36890 | `Claude Paroz` | Django core dev | SQLite's `GROUP_CONCAT` requires different handling for `distinct=True` with non-default delimiter |

---

## Table 2 — Reported / Proposed Fixes NOT Yet Confirmed by Core Developers

These GitHub PRs reference Trac tickets but were **closed without merging**. The underlying bug may
be real, but the approach or the bug report itself has not been accepted by the core team.

| # | Trac Ticket | GitHub PR | Title | Component | PR Closed | Note | Documentation URL |
|---|-------------|-----------|-------|-----------|-----------|------|-------------------|
| 1 | [#36855](https://code.djangoproject.com/ticket/36855) | [PR 20520](https://github.com/django/django/pull/20520) | `pre_save` field cache not persisted when saving a model instance with an explicit primary key | ORM / Model Save | 2026-01-xx | PR closed without merge; fix approach disputed — core team may disagree on correct semantics | [Model.save()](https://docs.djangoproject.com/en/stable/ref/models/instances/#django.db.models.Model.save) |
| 2 | [#36837](https://code.djangoproject.com/ticket/36837) | [PR 20485](https://github.com/django/django/pull/20485) | `Client.force_login()` iterates over `AUTHENTICATION_BACKENDS` but does not verify that the selected backend has a `get_user` method, potentially using the wrong backend | Testing / Auth | 2026-01-xx | PR closed; root behavior may be intentional per current auth design | [Client.force_login](https://docs.djangoproject.com/en/stable/topics/testing/tools/#django.test.Client.force_login) |
| 3 | [#36845](https://code.djangoproject.com/ticket/36845) | [PR 20504](https://github.com/django/django/pull/20504) | `Left()` database function does not support negative index values for PostgreSQL (unlike Python's `str[:n]`) | ORM / DB Functions | 2026-01-xx | PR closed; negative indexing in `Left()` may be outside the intended contract | [Left()](https://docs.djangoproject.com/en/stable/ref/models/database-functions/#left) |

---

## Notes

- Django's Trac uses a formal **"Accepted"** status to signal developer confirmation. Only tickets
  that reach `Accepted` and result in a merged PR are reliably confirmed.
- A PR with "Fixed #XXXXX" that is **closed without merging** means either (a) the fix approach was
  rejected, (b) a different PR was chosen, or (c) the bug itself wasn't confirmed.
- Issue #36903 (Python 3.14 + deferred annotations) is a forward-compatibility bug — highly relevant
  because Python 3.14 is close to stable release and affects all Django projects using type annotations.
- Issue #36890 (SQLite + `StringAgg`) reveals that ORM aggregation cross-database parity is still
  incomplete — a systematic gap that property-based testing with multiple DB backends could expose.
