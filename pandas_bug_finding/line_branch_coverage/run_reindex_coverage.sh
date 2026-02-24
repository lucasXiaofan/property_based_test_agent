#!/usr/bin/env bash
set -euo pipefail

uv run python pandas_bug_finding/line_branch_coverage/run_api_coverage.py \
  --runtime installed \
  --repo-root pandas_bug_finding/pandas \
  --api pandas.DataFrame.reindex \
  --test pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py \
  --json-out pandas_bug_finding/line_branch_coverage/reindex_coverage.json \
  -- -q
