# NumPy — Recent Closed Logical Bug Analysis

> **Source:** GitHub repository [`numpy/numpy`](https://github.com/numpy/numpy)
> **Scope:** Logical bugs in the library itself (excluding CI, build-system, and hardware-specific issues)
> **Date Range:** Jan – Mar 2026

---

## Table 1 — Developer-Confirmed Bugs

Bugs confirmed by NumPy core maintainers via explicit comments or fix PRs. Key reviewers:
`WarrenWeckesser`, `ngoldbaum`, `seberg`, `ev-br`, `kumaraditya303`.

| # | Issue | Title | Category | Closed | Documentation URL |
|---|-------|-------|----------|--------|-------------------|
| 1 | [#30909](https://github.com/numpy/numpy/issues/30909) | `np.fft.hfft`, `np.fft.ifft2`, and `np.fft.irfft2` silently ignore the `out=` parameter | FFT | 2026-03-02 | [numpy.fft.hfft](https://numpy.org/doc/stable/reference/generated/numpy.fft.hfft.html) |
| 2 | [#30883](https://github.com/numpy/numpy/issues/30883) | `numpy.busdaycalendar` `weekmask` parameter rejects a boolean array input that should be valid | Core / datetime | 2026-02-27 | [numpy.busdaycalendar](https://numpy.org/doc/stable/reference/generated/numpy.busdaycalendar.html) |
| 3 | [#30641](https://github.com/numpy/numpy/issues/30641) | `np.meshgrid` return type differs depending on the value of `copy=` — contradicts documented behavior | Core / Indexing | 2026-01-24 | [numpy.meshgrid](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html) |
| 4 | [#30732](https://github.com/numpy/numpy/issues/30732) | In-place update via chained advanced indexing (`arr[mesh][bool_mask] = val`) does not modify the original array | Advanced Indexing | 2026-02-02 | [Advanced Indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing) |
| 5 | [#30658](https://github.com/numpy/numpy/issues/30658) | Data race / memory safety bug in internal `npy_hashtable.c` under free-threading | Free-threading / C-API | 2026-01-16 | [NumPy C-API](https://numpy.org/doc/stable/reference/c-api/array.html) |
| 6 | [#30591](https://github.com/numpy/numpy/issues/30591) | `numpy.ma.masked_equal()` raises `ValueError` when `fill_value` is a scalar float and `missing_value` is an ndarray — regression introduced in 2.4.0 | Masked Arrays | 2026-01-07 | [numpy.ma.masked_equal](https://numpy.org/doc/stable/reference/generated/numpy.ma.masked_equal.html) |
| 7 | [#30648](https://github.com/numpy/numpy/issues/30648) | Data races in `_buffer_get_info` when multiple threads concurrently access the buffer protocol of the same `ndarray` | Free-threading / Buffer Protocol | 2026-01-27 | [Array Interface](https://numpy.org/doc/stable/reference/arrays.interface.html) |

### Confirmation Details

| Issue | Confirming Developer | Key Quote |
|-------|---------------------|-----------|
| #30909 | `WarrenWeckesser` | "It looks like there are three functions that have this problem: `hfft`, `ifft2`, and `irfft2`" — confirmed and expanded scope |
| #30883 | Core maintainer (labeled `00 - Bug` + component label) | Labeled and triaged by team; fix submitted |
| #30641 | `ngoldbaum` | "Maybe we should just change this to match the docs?" — `seberg`: "yes … I think just fix it." |
| #30732 | `ngoldbaum` | Asked for a smaller reproducer — actively engaged and confirmed reproducibility |
| #30658 | `kumaraditya303` | Opened draft fix PR #30662; labeled `free-threading` |
| #30591 | `ngoldbaum` | "Thanks for tracking down the PR that caused this. I'll try to take a look." — confirmed as regression |
| #30648 | Core team | Labeled `00 - Bug` + `free-threading`; fixed as part of free-threading safety work |

---

## Table 2 — Reported Bugs NOT Yet Confirmed by Developers

These issues were closed without a fix: either dismissed as expected behavior, due to incomplete
reproduction, or labeled "Close?" indicating no developer endorsement.

| # | Issue | Title | Category | Closed | Note | Documentation URL |
|---|-------|-------|----------|--------|------|-------------------|
| 1 | [#30830](https://github.com/numpy/numpy/issues/30830) | `np.asarray(x, dtype=object)` raises `ValueError` for certain input shapes | Core / dtype | 2026-02-16 | `ngoldbaum` explicitly closed: *"I don't think we're going to change this behavior and there are workarounds."* — closed as expected behavior, not a bug | [numpy.asarray](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html) |
| 2 | [#30603](https://github.com/numpy/numpy/issues/30603) | `np.full` and `np.full_like` produce different results for the same fill | Core / Creation | 2026-01-07 | `rkern` noted the reproduction code was incomplete — issue was closed due to reporter error, not confirmed as a library bug | [numpy.full](https://numpy.org/doc/stable/reference/generated/numpy.full.html) |
| 3 | [#30874](https://github.com/numpy/numpy/issues/30874) | `numpy.linalg.norm` consumes 100% CPU and appears to hang | Linear Algebra | 2026-02-25 | Labeled `57 - Close?`; behavior depends on BLAS threading and system load — not confirmed as a NumPy logic error | [numpy.linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) |
| 4 | [#30706](https://github.com/numpy/numpy/issues/30706) | `np.unique` uses unexpectedly large memory when called on a float array | Core / Memory | 2026-01-22 | No developer engagement; labeled `00 - Bug` but no confirmation; may be algorithmic complexity, not a correctness bug | [numpy.unique](https://numpy.org/doc/stable/reference/generated/numpy.unique.html) |

---

## Notes

- The `57 - Close?` label is NumPy's "needs further review before closing" marker — issues with this
  label are unlikely to be confirmed bugs.
- The `00 - Bug` label alone is not sufficient for confirmation; maintainer engagement (comments or fix PR) is required.
- The two free-threading bugs (#30658, #30648) are part of a broader ongoing effort to make NumPy
  safe under Python 3.13's free-threaded mode (`PYTHON_GIL_DISABLED`).
- `np.meshgrid` bug (#30641) is a classic contract violation: documented behavior vs. implementation diverge
  based on a flag value, making it an ideal candidate for property-based testing.
