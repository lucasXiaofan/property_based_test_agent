# Line and Branch Coverage for pandas APIs (including `super()` implementations)

This folder gives you a repeatable way to measure coverage for a target pandas API like `pandas.DataFrame.reindex` against your own test file.

## Why wrapper-only coverage is misleading

`DataFrame.reindex` in `pandas/core/frame.py` is mostly a thin wrapper that calls `super().reindex(...)`.

If you only check that wrapper lines, you may report high coverage while missing the real logic in `NDFrame.reindex` (`pandas/core/generic.py`) and downstream helpers.

The script here solves that by:
1. Resolving the target method from a dotted API path.
2. Following its MRO/super chain (`DataFrame.reindex` -> `NDFrame.reindex` -> ...).
3. Running your pytest file under `coverage.py` with `branch=True`.
4. Reporting:
- method-level **line** coverage for each resolved method in the chain,
- file-level **line + branch** coverage for all included source files.

## Files

- `run_api_coverage.py`: coverage runner.
- `run_reindex_coverage.sh`: convenience command for your current reindex hypothesis test.

## Prerequisites

From repo root (`/Users/xiaofanlu/Documents/github_repos/property_based_test_agent`):

```bash
uv add coverage
```

You already have `pytest` and `pandas` in this project.

Important:
- Default mode uses installed pandas (`--runtime installed`).
- Your local checkout at `pandas_bug_finding/pandas` is not importable unless pandas C extensions are built, so do not use `--runtime local` unless you already built it.

## Usage

### Generic form

```bash
uv run python pandas_bug_finding/line_branch_coverage/run_api_coverage.py \
  --runtime installed \
  --repo-root pandas_bug_finding/pandas \
  --api pandas.DataFrame.reindex \
  --test pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py \
  -- -q
```

### Save a machine-readable report

```bash
uv run python pandas_bug_finding/line_branch_coverage/run_api_coverage.py \
  --runtime installed \
  --repo-root pandas_bug_finding/pandas \
  --api pandas.DataFrame.reindex \
  --test pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py \
  --json-out pandas_bug_finding/line_branch_coverage/reindex_coverage.json \
  -- -q
```

### Wrapper-only mode (not recommended for pandas wrappers)

```bash
uv run python pandas_bug_finding/line_branch_coverage/run_api_coverage.py \
  --runtime installed \
  --repo-root pandas_bug_finding/pandas \
  --api pandas.DataFrame.reindex \
  --test pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py \
  --no-follow-super \
  -- -q
```

### If you really want runtime from local checkout

Only do this if local pandas is built and importable.

```bash
uv run python pandas_bug_finding/line_branch_coverage/run_api_coverage.py \
  --runtime local \
  --repo-root pandas_bug_finding/pandas \
  --api pandas.DataFrame.reindex \
  --test pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py \
  -- -q
```

## Recommended interpretation workflow

1. Run default mode (with super-following).
2. Check method-level line coverage for:
- `pandas.core.frame.DataFrame.reindex`
- `pandas.core.generic.NDFrame.reindex`
3. Check file-level branch coverage in `generic.py`.
4. If branch coverage is low, add targeted tests for:
- argument validation branches (`labels/index/columns/axis` interactions),
- monotonic vs non-monotonic index behavior for `method=...`,
- `limit`, `tolerance`, `level`, and MultiIndex paths,
- error/exception branches.

## Thin-wrapper best practice

For pandas-style APIs, treat coverage target as:
- public wrapper method **plus**
- delegated implementation methods reached via `super()` / MRO.

That is the correct way to avoid false confidence from wrapper-only line coverage.
