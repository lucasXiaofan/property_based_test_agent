#!/usr/bin/env bash
# run_quality_report.sh — end-to-end test-quality pipeline for a single API target
#
# USAGE
#   bash test_quality_metric/run_quality_report.sh <target_dir> [test_file] [--use_baseline] [--workers N] [...]
#
# POSITIONAL ARGUMENTS
#   target_dir   Path to the quality-metric directory for the API under test.
#                Must contain:
#                  metadata.json       — at least {"qualified_name": "pandas.DataFrame.reindex"}
#                  mutants/mapping.json
#                  mutants/*_mutants.py
#                Example: test_quality_metric/pandas/DataFrame/reindex
#
#   test_file    (Optional) Explicit path to a pytest file to run against mutants.
#                If omitted, the script resolves it automatically from ir2test_pipeline/
#                (see "AUTO-RESOLUTION" below).
#
# OPTIONS
#   --use_baseline   Pick the baseline test file instead of the IR-generated one
#                    during auto-resolution (see below).
#   --workers N      Pass -n N to the mutant evaluation step (parallel workers).
#   Any other flags  Forwarded verbatim to run_mutant_eval.py.
#
# AUTO-RESOLUTION (when test_file is omitted)
#   The script mirrors the target_dir path under ir2test_pipeline/ and searches
#   for a test file there:
#
#     Default (IR-generated):
#       ir2test_pipeline/<rel_path>/*ir_generated_test.py
#       e.g. ir2test_pipeline/pandas/DataFrame/reindex/reindex_ir_generated_test.py
#
#     With --use_baseline:
#       ir2test_pipeline/<rel_path>/*baseline_test.py
#       e.g. ir2test_pipeline/pandas/DataFrame/reindex/baseline_test.py
#
# BASELINE TEST FILE FORMAT
#   A baseline test must be a standard pytest file whose name ends with
#   "baseline_test.py" (e.g. baseline_test.py or reindex_baseline_test.py).
#   It should import pytest and hypothesis and define test_ functions.
#   Example skeleton:
#
#     # ir2test_pipeline/pandas/DataFrame/reindex/baseline_test.py
#     import pandas as pd
#     import pytest
#     from hypothesis import given, settings
#     from hypothesis import strategies as st
#
#     @given(...)
#     def test_reindex_returns_dataframe(...):
#         ...
#
#   The file is located alongside the IR JSON and the IR-generated test inside
#   ir2test_pipeline/<library>/<Class>/<method>/.
#
# OUTPUTS (written under <target_dir>/results/<test_stem>/)
#   kill_report.json / kill_report.md     — mutant kill results
#   coverage.json                         — line/branch coverage
#   properties_llm_trace.json             — LLM property-clause trace
#   properties_coverage.json              — aggregated properties coverage
#   overall_report_<test_stem>.md         — combined quality report
#
# EXAMPLES
#   # Default: run the IR-generated test
#   bash test_quality_metric/run_quality_report.sh \
#       test_quality_metric/pandas/DataFrame/reindex
#
#   # Baseline test (auto-resolved from ir2test_pipeline/)
#   bash test_quality_metric/run_quality_report.sh \
#       test_quality_metric/pandas/DataFrame/reindex --use_baseline
#
#   # Explicit test file path
#   bash test_quality_metric/run_quality_report.sh \
#       test_quality_metric/pandas/DataFrame/reindex \
#       ir2test_pipeline/pandas/DataFrame/reindex/baseline_test.py
#
#   # Parallel mutant evaluation with 4 workers
#   bash test_quality_metric/run_quality_report.sh \
#       test_quality_metric/pandas/DataFrame/reindex --workers 4

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <target_dir> [test_file] [--use_baseline] [--workers N] [other run_mutant_eval args]"
  echo "Example (default ir-generated test):"
  echo "  $0 test_quality_metric/pandas/DataFrame/reindex"
  echo "Example (baseline test from ir2test_pipeline folder):"
  echo "  $0 test_quality_metric/pandas/DataFrame/reindex --use_baseline"
  exit 2
fi

TARGET_DIR="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Auto-load repo .env if present (e.g., DEEPSEEK_API_KEY)
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

TEST_FILE=""
USE_BASELINE=0
EXTRA_MUTANT_ARGS=()

# Optional 2nd positional test file if it does not start with --
if [[ $# -gt 0 && "${1}" != --* ]]; then
  TEST_FILE="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --use_baseline)
      USE_BASELINE=1
      shift
      ;;
    *)
      EXTRA_MUTANT_ARGS+=("$1")
      shift
      ;;
  esac
done

METADATA_JSON="$TARGET_DIR/metadata.json"
if [[ ! -f "$METADATA_JSON" ]]; then
  echo "Error: metadata.json not found at $METADATA_JSON"
  exit 2
fi

MAPPING_FILE="$TARGET_DIR/mutants/mapping.json"
if [[ ! -f "$MAPPING_FILE" ]]; then
  echo "Error: mapping.json not found at $MAPPING_FILE"
  exit 2
fi

MUTANTS_FILE="$(find "$TARGET_DIR/mutants" -maxdepth 1 -name '*_mutants.py' | head -n 1)"
if [[ -z "$MUTANTS_FILE" ]]; then
  echo "Error: no '*_mutants.py' found under $TARGET_DIR/mutants"
  exit 2
fi

if [[ -z "$TEST_FILE" ]]; then
  REL_PATH="$(python3 - <<'PY' "$TARGET_DIR" "$REPO_ROOT/test_quality_metric"
import pathlib, sys
p = pathlib.Path(sys.argv[1]).resolve()
base = pathlib.Path(sys.argv[2]).resolve()
print(p.relative_to(base).as_posix())
PY
)"

  IR_DIR="$REPO_ROOT/ir2test_pipeline/$REL_PATH"
  if [[ ! -d "$IR_DIR" ]]; then
    echo "Error: corresponding IR folder not found: $IR_DIR"
    exit 2
  fi

  if [[ "$USE_BASELINE" -eq 1 ]]; then
    TEST_FILE="$(find "$IR_DIR" -maxdepth 1 -name '*baseline_test.py' | head -n 1)"
  else
    TEST_FILE="$(find "$IR_DIR" -maxdepth 1 -name '*ir_generated_test.py' | head -n 1)"
  fi

  if [[ -z "$TEST_FILE" ]]; then
    if [[ "$USE_BASELINE" -eq 1 ]]; then
      echo "Error: no '*baseline_test.py' found under $IR_DIR"
    else
      echo "Error: no '*ir_generated_test.py' found under $IR_DIR"
    fi
    exit 2
  fi
fi

if [[ ! -f "$TEST_FILE" ]]; then
  echo "Error: test file not found: $TEST_FILE"
  exit 2
fi

TEST_STEM="$(basename "$TEST_FILE")"
TEST_STEM="${TEST_STEM%.*}"
RESULT_DIR="$TARGET_DIR/results/$TEST_STEM"
mkdir -p "$RESULT_DIR"

KILL_JSON="$RESULT_DIR/kill_report.json"
KILL_MD="$RESULT_DIR/kill_report.md"
COVERAGE_JSON="$RESULT_DIR/coverage.json"
LLM_TRACE_JSON="$RESULT_DIR/properties_llm_trace.json"
PROPS_COVERAGE_JSON="$RESULT_DIR/properties_coverage.json"
OVERALL_MD="$RESULT_DIR/overall_report_${TEST_STEM}.md"

API_NAME="$(uv run python - <<'PY' "$METADATA_JSON"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data.get('qualified_name', '').strip())
PY
)"

if [[ -z "$API_NAME" ]]; then
  echo "Error: metadata.json missing 'qualified_name'"
  exit 2
fi

echo "[quality_report] target_dir: $TARGET_DIR"
echo "[quality_report] pytest_file: $TEST_FILE"
echo "[quality_report] api: $API_NAME"

echo "[quality_report] running mutant evaluation..."
set +e
_mutant_eval_cmd=(
  uv run python "$REPO_ROOT/test_quality_metric/run_mutant_eval.py"
  --mapping-file "$MAPPING_FILE"
  --mutants-file "$MUTANTS_FILE"
  --output-dir "$TARGET_DIR/results"
  --pytest-file "$TEST_FILE"
)
if [[ ${#EXTRA_MUTANT_ARGS[@]} -gt 0 ]]; then
  _mutant_eval_cmd+=("${EXTRA_MUTANT_ARGS[@]}")
fi
"${_mutant_eval_cmd[@]}"
MUT_EXIT=$?
set -e

echo "[quality_report] running line/branch coverage..."
set +e
uv run python "$REPO_ROOT/test_quality_metric/run_api_coverage.py" \
  --runtime installed \
  --repo-root "$REPO_ROOT/pandas_bug_finding/pandas" \
  --api "$API_NAME" \
  --test "$TEST_FILE" \
  --json-out "$COVERAGE_JSON" \
  -- -q
COV_EXIT=$?
set -e

echo "[quality_report] running llm properties coverage..."
set +e
uv run python "$REPO_ROOT/test_quality_metric/llm_clause_trace.py" \
  --target-dir "$TARGET_DIR" \
  --test-file "$TEST_FILE" \
  --out-json "$LLM_TRACE_JSON"
LLM_EXIT=$?
if [[ "$LLM_EXIT" -eq 0 ]]; then
  uv run python "$REPO_ROOT/test_quality_metric/calc_properties_coverage.py" \
    --input-json "$LLM_TRACE_JSON" \
    --output-json "$PROPS_COVERAGE_JSON"
  PROP_EXIT=$?
else
  PROP_EXIT=1
fi
set -e

uv run python - <<'PY' "$KILL_JSON" "$COVERAGE_JSON" "$LLM_TRACE_JSON" "$PROPS_COVERAGE_JSON" "$OVERALL_MD" "$TARGET_DIR" "$TEST_FILE" "$API_NAME" "$MUT_EXIT" "$COV_EXIT" "$LLM_EXIT" "$PROP_EXIT"
import json
import os
import sys
from datetime import datetime, timezone

(kill_json, coverage_json, llm_trace_json, props_json, overall_md,
 target_dir, test_file, api_name, mut_exit, cov_exit, llm_exit, prop_exit) = sys.argv[1:]
mut_exit = int(mut_exit)
cov_exit = int(cov_exit)
llm_exit = int(llm_exit)
prop_exit = int(prop_exit)

def load_json(p):
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

kill = load_json(kill_json)
coverage = load_json(coverage_json)
llm = load_json(llm_trace_json)
props = load_json(props_json)

lines = []
lines.append(f"# Overall Quality Report: {api_name}")
lines.append("")
lines.append(f"- Generated at: {datetime.now(timezone.utc).isoformat()}")
lines.append(f"- Target dir: `{target_dir}`")
lines.append(f"- Test file: `{test_file}`")
lines.append(f"- Mutant eval exit code: {mut_exit}")
lines.append(f"- Coverage exit code: {cov_exit}")
lines.append(f"- LLM trace exit code: {llm_exit}")
lines.append(f"- Properties coverage exit code: {prop_exit}")
lines.append("")

lines.append("## Baseline Test Status")
if kill is None:
    lines.append("- kill_report.json not found")
else:
    baseline_info = kill.get("baseline", {})
    failed_ids = baseline_info.get("failed_nodeids", [])
    xfail_ids = baseline_info.get("xfail_nodeids", [])
    if not failed_ids and not xfail_ids:
        lines.append("- All tests passed (no failures or xfails)")
    if failed_ids:
        lines.append(f"- **Failed** ({len(failed_ids)}):")
        for nid in failed_ids:
            lines.append(f"  - `{nid}`")
    if xfail_ids:
        lines.append(f"- **Xfailed** ({len(xfail_ids)}):")
        for nid in xfail_ids:
            lines.append(f"  - `{nid}`")
lines.append("")

lines.append("## Mutant Kill Summary")
if kill is None:
    lines.append("- kill_report.json not found")
else:
    summary = kill.get("summary", {})
    lines.append(f"- Killed: {summary.get('killed', 0)}")
    lines.append(f"- Survived: {summary.get('survived', 0)}")
    lines.append(f"- Invalid baseline: {summary.get('invalid_baseline', 0)}")
    lines.append(f"- Mutation score: {summary.get('mutation_score', 0):.3f}")

lines.append("")
lines.append("## Coverage Summary")
if coverage is None:
    lines.append("- coverage.json not found")
else:
    methods = coverage.get("methods", [])
    method_cov_values = [m.get("line_coverage_percent", 0) for m in methods]
    avg_method_cov = (sum(method_cov_values) / len(method_cov_values)) if method_cov_values else 0.0
    lines.append(f"- Method count: {len(methods)}")
    lines.append(f"- Average method line coverage: {avg_method_cov:.2f}%")
    lines.append("")
    lines.append("### Method Coverage")
    for m in methods:
        lines.append(
            f"- `{m.get('label')}`: {m.get('covered_lines', 0)}/{m.get('num_statements', 0)} "
            f"lines ({m.get('line_coverage_percent', 0):.2f}%)"
        )
    totals = coverage.get("totals", {})
    lines.append("")
    lines.append("### Branch Coverage (overall)")
    lines.append(
        f"- Branch coverage: {totals.get('covered_branches', 0)}/{totals.get('num_branches', 0)} "
        f"({totals.get('branch_coverage_percent', 0):.2f}%)"
    )

lines.append("")
lines.append("## Properties Coverage (LLM)")
if props is None:
    lines.append("- properties_coverage.json not found")
else:
    lines.append(f"- Total clauses: {props.get('total_clauses', 0)}")
    lines.append(f"- Satisfied clauses: {props.get('satisfied_clauses', 0)}")
    lines.append(f"- Unsatisfied clauses: {props.get('unsatisfied_clauses', 0)}")
    lines.append(f"- Properties coverage: {props.get('properties_coverage', 0):.3f}")
    lines.append(f"- Avg confidence (satisfied): {props.get('avg_confidence_satisfied', 0):.3f}")
    lines.append(f"- Avg confidence (unsatisfied): {props.get('avg_confidence_unsatisfied', 0):.3f}")

if llm is not None:
    lines.append("")
    lines.append("### Clause Trace")
    for row in llm.get("clauses", []):
        cid = row.get("clause_id")
        sat = row.get("satisfied")
        conf = row.get("confidence", 0)
        by = row.get("satisfied_by", [])
        if sat:
            lines.append(f"- `{cid}`: satisfied by {by} (confidence={conf:.2f})")
        else:
            lines.append(f"- `{cid}`: unsatisfied (confidence={conf:.2f})")

with open(overall_md, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines) + "\n")
print(f"[quality_report] wrote: {overall_md}")
PY

echo "[quality_report] done"
echo "[quality_report] kill report: $KILL_MD"
echo "[quality_report] coverage json: $COVERAGE_JSON"
echo "[quality_report] llm trace json: $LLM_TRACE_JSON"
echo "[quality_report] properties coverage json: $PROPS_COVERAGE_JSON"
echo "[quality_report] overall md: $OVERALL_MD"
