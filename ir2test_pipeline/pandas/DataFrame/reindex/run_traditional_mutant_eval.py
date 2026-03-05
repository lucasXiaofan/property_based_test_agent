"""
run_traditional_mutant_eval.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Self-contained evaluator for traditional (non-LLM) mutants of
pandas.DataFrame.reindex against baseline_test.py.

WHAT THIS SCRIPT DOES
─────────────────────
1. Runs baseline_test.py once to record the clean test result.
2. For each mutant in traditional_mutants.py:
   a. Patches pd.DataFrame.reindex with the mutant implementation via
      a tiny subprocess launcher (same technique as run_mutant_eval.py).
   b. Runs pytest; if any test that passed in the baseline now fails →
      the mutant is "killed".  If all tests still pass → "survived".
3. Writes a kill_report.json and a human-readable kill_report.md to
   trad_mutant_results/<test_stem>/.

WHY TRADITIONAL MUTANTS HANDLE DIFFERENT BEHAVIORS
───────────────────────────────────────────────────
Each mutant corresponds to one mutation operator applied to one decision
point in the Python wrapper (reindex_wrapper.py).  The operator determines
what kind of code change was made; the decision point determines which
aspect of reindex behavior is perturbed:

  SDL (Statement Deletion)  – removes a kwarg pass-through entirely.
    SDL_method      → no fill method → tests for ffill/bfill/nearest detect it
    SDL_fill_value  → always NaN     → tests with custom fill_value detect it
    SDL_limit       → no cap         → tests with limit= boundaries detect it
    SDL_tolerance   → no restriction → tests with tolerance= detect it
    SDL_level       → no MultiIndex  → tests with level= detect it
    SDL_index       → row labels     → tests with explicit index= detect it
    SDL_columns     → column labels  → tests with explicit columns= detect it
    SDL_labels_block→ positional arg → tests using df.reindex(labels) detect it

  ROR (Relational Operator Replace) – flips a condition.
    ROR_axis_eq     → rows↔columns swap → tests checking axis routing detect it
    ROR_labels_none → labels ignored → positional label tests detect it
    ROR_index_none  → index ignored  → index= tests detect it
    ROR_columns_none→ columns ignored→ columns= tests detect it

  COR (Conditional Operator Replace) – inverts a logical condition.
    COR_fill_sentinel→ custom fill dropped → fill_value tests detect it

  AOR (Arithmetic Operator Replace) – off-by-one on a numeric arg.
    AOR_limit_plus1 → +1 extra fill → tight-boundary limit tests detect it

  SVR (Scalar Variable Replace) – wrong literal constant.
    SVR_axis_default→ default axis wrong → axis=None tests detect it

USAGE
─────
  python run_traditional_mutant_eval.py [--workers N] [--verbose]

OPTIONS
  --workers N   Parallel worker count (default: cpu_count, max 8)
  --verbose     Show full pytest output for each mutant
  --output-dir  Output directory (default: trad_mutant_results/)

REQUIREMENTS
  pytest, hypothesis, pandas, numpy  (already installed in this project)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent.resolve()
MUTANTS_FILE = HERE / "traditional_mutants.py"
MAPPING_FILE = HERE / "traditional_mapping.json"
TEST_FILE = HERE / "baseline_test.py"
DEFAULT_OUTPUT = HERE / "trad_mutant_results"

FAILED_RE = re.compile(r"^FAILED\s+([^\s]+)", re.MULTILINE)

# ── subprocess launcher ───────────────────────────────────────────────────────

_LAUNCHER = r"""
import importlib.util, json, sys

mutants_file = sys.argv[1]
pytest_file  = sys.argv[2]
mutant_id    = sys.argv[3] if len(sys.argv) > 3 else ""
extra_args   = json.loads(sys.argv[4]) if len(sys.argv) > 4 else ["-q", "--tb=short"]

if mutants_file and mutant_id:
    spec = importlib.util.spec_from_file_location("_trad_mutants", mutants_file)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["_trad_mutants"] = mod
    spec.loader.exec_module(mod)
    mod.apply_mutant(mutant_id)

import pytest
rc = pytest.main([pytest_file] + extra_args)

if mutants_file and mutant_id:
    mod.reset_mutant()

raise SystemExit(rc)
"""


def _run(pytest_file: Path, mutants_file: Path | None, mutant_id: str | None,
         extra_args: list[str] | None = None) -> tuple[int, str, str]:
    cmd = [
        sys.executable, "-c", _LAUNCHER,
        str(mutants_file) if mutants_file else "",
        str(pytest_file),
        mutant_id or "",
    ]
    if extra_args is not None:
        cmd.append(json.dumps(extra_args))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def _failed_nodeids(stdout: str, stderr: str) -> set[str]:
    return set(FAILED_RE.findall(stdout + "\n" + stderr))


# ── per-mutant worker ─────────────────────────────────────────────────────────

def _eval_one(row: dict, baseline_failed: set[str], verbose: bool) -> dict:
    mid = row["mutant_id"]
    op  = row.get("operator", "?")

    rc, out, err = _run(TEST_FILE, MUTANTS_FILE, mid,
                        extra_args=["--tb=short", "-q"])
    mutant_failed = _failed_nodeids(out, err)
    new_failed = sorted(mutant_failed - baseline_failed)

    # A mutant is killed when at least one previously-passing test now fails.
    # When baseline itself has failures we only count *new* failures.
    if not baseline_failed:
        status = "survived" if rc == 0 else "killed"
    else:
        status = "killed" if new_failed else "survived"

    result = {
        "mutant_id": mid,
        "operator": op,
        "description": row.get("behavior_broken", ""),
        "status": status,
        "returncode": rc,
        "new_failed_nodeids": new_failed,
    }
    if verbose:
        result["stdout"] = out
        result["stderr"] = err

    icon = "✓" if status == "killed" else "✗"
    print(f"  [{icon}] {mid:<22} ({op})  → {status}")
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int,
                    default=min(8, os.cpu_count() or 1))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    out_root = Path(args.output_dir) / TEST_FILE.stem
    out_root.mkdir(parents=True, exist_ok=True)

    mapping = json.loads(MAPPING_FILE.read_text())

    print("=" * 60)
    print("Traditional Mutant Evaluation — pandas.DataFrame.reindex")
    print("=" * 60)
    print(f"  test file   : {TEST_FILE}")
    print(f"  mutants file: {MUTANTS_FILE}")
    print(f"  mutants     : {len(mapping['mutants'])}")
    print(f"  workers     : {args.workers}")
    print()

    # ── baseline ─────────────────────────────────────────────────────────────
    print("Running baseline …")
    b_rc, b_out, b_err = _run(TEST_FILE, None, None,
                               extra_args=["--tb=no", "-q"])
    baseline_failed = _failed_nodeids(b_out, b_err)
    b_ok = b_rc == 0
    print(f"  baseline passed: {b_ok}",
          f"(failing tests: {len(baseline_failed)})" if baseline_failed else "")
    print()

    # ── mutant runs ───────────────────────────────────────────────────────────
    print("Evaluating mutants …")
    rows = mapping["mutants"]
    ordered: list[dict | None] = [None] * len(rows)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_eval_one, row, baseline_failed, args.verbose): i
            for i, row in enumerate(rows)
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            ordered[i] = fut.result()

    runs: list[dict] = [r for r in ordered if r is not None]

    # ── summary ───────────────────────────────────────────────────────────────
    killed   = sum(1 for r in runs if r["status"] == "killed")
    survived = sum(1 for r in runs if r["status"] == "survived")
    score    = killed / (killed + survived) if (killed + survived) else 0.0

    # Group by operator
    by_op: dict[str, dict] = {}
    for r in runs:
        op = r["operator"]
        if op not in by_op:
            by_op[op] = {"killed": 0, "survived": 0}
        by_op[op][r["status"]] += 1

    print()
    print("=" * 60)
    print(f"  Mutation score : {score:.1%}  ({killed} killed / {killed + survived} total)")
    print()
    print("  By operator:")
    for op, counts in sorted(by_op.items()):
        k, s = counts["killed"], counts["survived"]
        print(f"    {op:<5}  killed={k}  survived={s}")
    print("=" * 60)

    # ── write kill_report.json ────────────────────────────────────────────────
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "function": mapping["function"],
        "mutant_source": "traditional",
        "pytest_file": str(TEST_FILE),
        "baseline": {
            "passed": b_ok,
            "failed_nodeids": sorted(baseline_failed),
        },
        "summary": {
            "killed": killed,
            "survived": survived,
            "mutation_score": round(score, 4),
            "by_operator": by_op,
        },
        "runs": runs,
    }
    json_path = out_root / "kill_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ── write kill_report.md ─────────────────────────────────────────────────
    md_lines = [
        f"# Kill Report — Traditional Mutants: {mapping['function']}",
        "",
        f"- **pytest file**: `{TEST_FILE}`",
        f"- **baseline passed**: {b_ok}",
        f"- **mutation score**: {score:.1%}  ({killed} killed / {killed + survived} total)",
        "",
        "## Results by Operator",
        "",
        "| Operator | Killed | Survived |",
        "|----------|--------|----------|",
    ]
    for op, counts in sorted(by_op.items()):
        md_lines.append(f"| {op} | {counts['killed']} | {counts['survived']} |")

    md_lines += [
        "",
        "## Per-Mutant Results",
        "",
        "| Mutant ID | Operator | Status | Behavior Broken |",
        "|-----------|----------|--------|----------------|",
    ]
    for r in runs:
        icon = "✅" if r["status"] == "killed" else "❌"
        desc = r["description"][:80]
        md_lines.append(f"| `{r['mutant_id']}` | {r['operator']} | {icon} {r['status']} | {desc} |")

    md_lines += [
        "",
        "## Why Traditional Mutants Test Different Behaviors",
        "",
        "Each mutant removes or inverts exactly one decision in the Python wrapper",
        "around `pd.DataFrame.reindex`.  A test that exercises the affected parameter",
        "will detect the mutation; tests that never exercise that parameter will not.",
        "",
        "| Operator | Code-level change | Behavioral impact |",
        "|----------|-------------------|-------------------|",
        "| SDL | Delete a statement | Silently drops one kwarg | Param never forwarded |",
        "| ROR | Flip == / != | Wrong branch taken | Axis/routing inverted |",
        "| COR | Invert 'not' | Guard condition reversed | Sentinel check flipped |",
        "| AOR | ±1 on constant | Off-by-one | Limit too permissive |",
        "| SVR | Change literal value | Wrong default | Default axis wrong |",
        "",
        "Survival means tests don't cover that parameter path; killing means",
        "they do.",
    ]

    md_path = out_root / "kill_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\n  results → {out_root}")
    print(f"  wrote   → {json_path.name}")
    print(f"  wrote   → {md_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
