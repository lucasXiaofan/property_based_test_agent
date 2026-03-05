# PyTorch — Recent Closed Logical Bug Analysis

> **Source:** GitHub repository [`pytorch/pytorch`](https://github.com/pytorch/pytorch)
> **Scope:** Logical / correctness bugs in the library itself (excluding hardware driver, CUDA build, and OS-specific install issues)
> **Date Range:** Jan – Mar 2026

---

## Table 1 — Developer-Confirmed Bugs

Bugs confirmed by PyTorch core team via the `triaged` label (requires maintainer review), `high priority`
escalation, `module: correctness (silent)` label (wrong answer without error), or explicit fix PR
links. Key reviewers: `eellison`, `malfet`, `janeyx99`, `laithsakka`, `bdhirsh`, `bornaehsani`.

| # | Issue | Title | Category | Closed | Documentation URL |
|---|-------|-------|----------|--------|-------------------|
| 1 | [#173383](https://github.com/pytorch/pytorch/issues/173383) | `torch._foreach_copy_` produces **wrong numerical results** when destination tensors have mixed dtypes | foreach / Correctness | 2026-01-27 | [foreach ops](https://pytorch.org/docs/stable/generated/torch.foreach_copy_.html) |
| 2 | [#174069](https://github.com/pytorch/pytorch/issues/174069) | Inductor backend: `argmax`/`max` returns **incorrect indices** for boolean tensors on CUDA | Inductor / CUDA | 2026-02-19 | [torch.argmax](https://pytorch.org/docs/stable/generated/torch.argmax.html) |
| 3 | [#174339](https://github.com/pytorch/pytorch/issues/174339) | MPS backend: `F.grid_sample` returns **incorrect output** — NHWC kernel correctness bug + missing memory-format in kernel cache | MPS / nn.functional | 2026-02-11 | [F.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) |
| 4 | [#172732](https://github.com/pytorch/pytorch/issues/172732) | `functools.wraps` inadvertently **inherits `_torchdynamo_orig_callable`**, causing `@torch.compile` to silently bypass the `@disable` decorator | TorchDynamo / Compile | 2026-02-13 | [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html) |
| 5 | [#175902](https://github.com/pytorch/pytorch/issues/175902) | MPS backend: `torch.mm` and `torch.addmm` produce **wrong results** for zero-dimensional tensor edge cases that CPU handles correctly | MPS / Linear Algebra | 2026-02-28 | [torch.mm](https://pytorch.org/docs/stable/generated/torch.mm.html) |
| 6 | [#175683](https://github.com/pytorch/pytorch/issues/175683) | MPS backend: `F.grid_sample` **crashes** with a zero-size grid tensor (`RuntimeError: [srcBuf length] > 0`) | MPS / Error Checking | 2026-02-25 | [F.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) |
| 7 | [#174379](https://github.com/pytorch/pytorch/issues/174379) | Inductor: `native_layer_norm_backward` crashes with dynamic shapes — generated kernel computes an incorrect workspace slice | Inductor / Dynamic Shapes | 2026-03-02 | [torch.nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) |

### Confirmation Details

| Issue | Confirming Developer | Key Quote / Evidence |
|-------|---------------------|----------------------|
| #173383 | `eellison` | Labeled `high priority` + `module: correctness (silent)` + `actionable`; CC'd `janeyx99` for fix |
| #174069 | PyTorch oncall (`pt2`) | Labeled `high priority` + `triaged`; assigned to inductor oncall |
| #174339 | `malfet` | "Considering that grid-sampler is almost an elementwise op …" — actively drove the investigation + `module: correctness (silent)` label |
| #172732 | PT2 team | Labeled `high priority`; fix tracked by `WeakKeyDictionary` hotfix and full fix |
| #175902 | `bornaehsani` | "PR to fix: #175905" — immediately submitted fix PR confirming the bug |
| #175683 | `malfet` | "This is already fixed by #174343 and will be available in the upcoming 2.11 release." |
| #174379 | `laithsakka` | "This is shared with inductor I guess exposed by enabling dynamic shapes in `native_layer_norm_backward`" |

---

## Table 2 — Reported Bugs NOT Yet Confirmed by Developers

These issues were triaged (bot-triaged or awaiting developer review) but no core developer explicitly
confirmed the bug or linked a fix PR.

| # | Issue | Title | Category | Closed | Note | Documentation URL |
|---|-------|-------|----------|--------|------|-------------------|
| 1 | [#176230](https://github.com/pytorch/pytorch/issues/176230) | `gaussian_nll_loss` possible incorrect behavior with broadcastable variance shapes having multiple unit dimensions | nn / Loss Functions | 2026-03-02 | Closed by reporter to refile with the proper bug-report template; no developer confirmed the behavior is wrong | [F.gaussian_nll_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.gaussian_nll_loss.html) |
| 2 | [#175545](https://github.com/pytorch/pytorch/issues/175545) | `FlexAttention` does not allow a captured tensor with `ndim > 0` | nn / Attention | 2026-02-25 | Labeled `triaged` by bot; no developer confirmed whether this is a bug vs. a current limitation by design | [FlexAttention](https://pytorch.org/docs/stable/nn.attention.flex_attention.html) |
| 3 | [#174775](https://github.com/pytorch/pytorch/issues/174775) | FP16 `nn.functional.mish` has a large numerical discrepancy vs. NumPy 2.3 reference | nn / Precision | 2026-02-11 | Labeled `module: dependency bug` — root cause attributed to a NumPy 2.3 behavioral change, not a PyTorch logic error | [F.mish](https://pytorch.org/docs/stable/generated/torch.nn.functional.mish.html) |

---

## Notes

- **`module: correctness (silent)` label** is the clearest developer signal — it means the op returns
  a wrong answer without raising any error, which is the most dangerous class of bug.
- The MPS (Apple Silicon GPU) backend has a cluster of correctness issues (#174339, #175902, #175683)
  because its kernel implementations are not always bit-for-bit equivalent to the reference CPU path.
- Issue #172732 (`functools.wraps` + `_torchdynamo_orig_callable`) is a pure Python-level logic bug
  with no hardware dependency — a good candidate for property-based testing of decorator composition.
- PyTorch's `triaged` label requires a human team member to apply it, so it indicates human review
  even when added via the `bot-triaged` workflow.
