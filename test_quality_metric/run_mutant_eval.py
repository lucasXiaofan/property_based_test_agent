"""Shared mutation evaluator for test_quality_metric.

Use one shared runner for all functions to avoid redundant per-function scripts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

FAILED_LINE_RE = re.compile(r"^FAILED\s+([^\s]+)(?:\s+-\s+.+)?$", re.MULTILINE)


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def _load_module_from_file(module_file: str) -> ModuleType:
    path = Path(module_file).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _run_pytest_subprocess(
    pytest_file: str,
    mutants_file: str | None,
    mutant_id: str | None,
) -> RunResult:
    launcher = r"""
import importlib.util
import sys

mutants_file = sys.argv[1]
pytest_file = sys.argv[2]
mutant_id = sys.argv[3] if len(sys.argv) > 3 else ""

if mutants_file and mutant_id:
    spec = importlib.util.spec_from_file_location("active_mutants_module", mutants_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load mutants module: {mutants_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["active_mutants_module"] = module
    spec.loader.exec_module(module)
    module.apply_mutant(mutant_id)
else:
    module = None

import pytest
rc = pytest.main([pytest_file, "-q"])

if module is not None:
    module.reset_mutant()

raise SystemExit(rc)
"""
    cmd = [sys.executable, "-c", launcher, mutants_file or "", pytest_file]
    if mutant_id:
        cmd.append(mutant_id)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return RunResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def _extract_failed_nodeids(stdout: str, stderr: str) -> set[str]:
    text = f"{stdout}\n{stderr}"
    return set(FAILED_LINE_RE.findall(text))


def _default_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(8, cpu))


def _run_one_mutant(
    idx: int,
    row: dict[str, str],
    pytest_file: str,
    mutants_file: str,
    baseline_passed: bool,
    allow_failing_baseline: bool,
    baseline_failed_nodeids: set[str],
) -> tuple[int, dict[str, object], str]:
    mutant_id = row["mutant_id"]
    clause_id = row.get("clause_id")
    if not baseline_passed and not allow_failing_baseline:
        result = {
            "mutant_id": mutant_id,
            "clause_id": clause_id,
            "status": "invalid_baseline",
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
        }
        log = f"[run_mutant_eval] {mutant_id} ({clause_id}): invalid_baseline"
        return idx, result, log

    run = _run_pytest_subprocess(
        pytest_file=pytest_file,
        mutants_file=mutants_file,
        mutant_id=mutant_id,
    )
    mutant_failed_nodeids = _extract_failed_nodeids(run.stdout, run.stderr)
    new_failed_nodeids = sorted(mutant_failed_nodeids - baseline_failed_nodeids)
    if baseline_passed:
        status = "survived" if run.passed else "killed"
        log = f"[run_mutant_eval] {mutant_id} ({clause_id}): {status}"
    else:
        status = "killed" if new_failed_nodeids else "survived"
        log = (
            f"[run_mutant_eval] {mutant_id} ({clause_id}): {status}"
            f" | new_failures={len(new_failed_nodeids)}"
        )

    result = {
        "mutant_id": mutant_id,
        "clause_id": clause_id,
        "status": status,
        "returncode": run.returncode,
        "failed_nodeids": sorted(mutant_failed_nodeids),
        "new_failed_nodeids": new_failed_nodeids,
        "stdout_tail": run.stdout[-2000:],
        "stderr_tail": run.stderr[-2000:],
    }
    return idx, result, log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytest-file", required=True)
    parser.add_argument(
        "--target-dir",
        help=(
            "Function folder, e.g. test_quality_metric/pandas/DataFrame/reindex. "
            "If provided, mapping/mutants/output paths are inferred."
        ),
    )
    parser.add_argument("--mapping-file")
    parser.add_argument("--mutants-file")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--allow-failing-baseline",
        action="store_true",
        help=(
            "If baseline is failing, run mutants anyway and mark killed only when "
            "new failing test nodeids appear beyond baseline."
        ),
        default=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help="Number of parallel workers for mutant execution. Default is auto (up to 8).",
    )
    args = parser.parse_args()

    if args.target_dir:
        target_dir = Path(args.target_dir).resolve()
        mapping_file = target_dir / "mutants" / "mapping.json"
        mutant_candidates = sorted((target_dir / "mutants").glob("*_mutants.py"))
        if len(mutant_candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one '*_mutants.py' under {target_dir / 'mutants'}, "
                f"found {len(mutant_candidates)}"
            )
        mutants_file = mutant_candidates[0]
        output_dir = target_dir / "mutants" / "results"
    else:
        if not (args.mapping_file and args.mutants_file and args.output_dir):
            raise RuntimeError(
                "Either provide --target-dir, or provide all of "
                "--mapping-file, --mutants-file, and --output-dir."
            )
        mapping_file = Path(args.mapping_file).resolve()
        mutants_file = Path(args.mutants_file).resolve()
        output_dir = Path(args.output_dir).resolve()

    mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    print(f"[run_mutant_eval] target function: {mapping.get('function')}")
    print(f"[run_mutant_eval] pytest file: {args.pytest_file}")
    print(f"[run_mutant_eval] mapping: {mapping_file}")
    print(f"[run_mutant_eval] mutants: {mutants_file}")
    print(f"[run_mutant_eval] workers: {args.workers}")

    mutants_mod = _load_module_from_file(str(mutants_file))
    if not hasattr(mutants_mod, "apply_mutant") or not hasattr(mutants_mod, "reset_mutant"):
        raise RuntimeError("Mutants module must expose apply_mutant(mutant_id) and reset_mutant()")

    results: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "function": mapping.get("function"),
        "pytest_file": args.pytest_file,
        "mutants": mapping.get("mutants", []),
        "runs": [],
    }

    baseline = _run_pytest_subprocess(
        pytest_file=args.pytest_file,
        mutants_file=None,
        mutant_id=None,
    )
    baseline_failed_nodeids = _extract_failed_nodeids(baseline.stdout, baseline.stderr)
    print(f"[run_mutant_eval] baseline passed: {baseline.passed}")
    if baseline_failed_nodeids:
        print(f"[run_mutant_eval] baseline failing nodeids: {len(baseline_failed_nodeids)}")
    results["baseline"] = {
        "passed": baseline.passed,
        "returncode": baseline.returncode,
        "failed_nodeids": sorted(baseline_failed_nodeids),
        "stdout_tail": baseline.stdout[-2000:],
        "stderr_tail": baseline.stderr[-2000:],
    }

    baseline_passed = bool(results["baseline"]["passed"])

    rows = mapping.get("mutants", [])
    ordered_results: list[dict[str, object] | None] = [None] * len(rows)
    workers = max(1, args.workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _run_one_mutant,
                i,
                row,
                args.pytest_file,
                str(mutants_file),
                baseline_passed,
                args.allow_failing_baseline,
                baseline_failed_nodeids,
            )
            for i, row in enumerate(rows)
        ]
        for fut in concurrent.futures.as_completed(futures):
            idx, run_row, log_line = fut.result()
            ordered_results[idx] = run_row
            print(log_line)

    results["runs"] = [r for r in ordered_results if r is not None]

    killed = sum(1 for x in results["runs"] if x["status"] == "killed")
    survived = sum(1 for x in results["runs"] if x["status"] == "survived")
    invalid_baseline = sum(1 for x in results["runs"] if x["status"] == "invalid_baseline")
    denom = killed + survived
    score = (killed / denom) if denom else 0.0
    results["summary"] = {
        "killed": killed,
        "survived": survived,
        "invalid_baseline": invalid_baseline,
        "mutation_score": score,
    }

    test_script_name = Path(args.pytest_file).stem
    out_dir = output_dir / test_script_name
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "kill_report.json"
    md_path = out_dir / "kill_report.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = [
        f"# Kill Report: {results['function']}",
        "",
        f"- pytest_file: `{results['pytest_file']}`",
        f"- baseline_passed: {results['baseline']['passed']}",
        f"- killed: {killed}",
        f"- survived: {survived}",
        f"- invalid_baseline: {invalid_baseline}",
        f"- allow_failing_baseline: {args.allow_failing_baseline}",
        f"- mutation_score: {score:.3f}",
        "",
        "## Mutants",
    ]
    for run in results["runs"]:
        md.append(f"- `{run['mutant_id']}` ({run.get('clause_id')}): {run['status']}")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        "[run_mutant_eval] summary: "
        f"killed={killed}, survived={survived}, invalid_baseline={invalid_baseline}, "
        f"score={score:.3f}"
    )
    print(f"[run_mutant_eval] results folder: {out_dir}")
    print(f"[run_mutant_eval] wrote: {json_path}")
    print(f"[run_mutant_eval] wrote: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
