#!/usr/bin/env python3
"""Compare baseline and IR-generated pandas tests by mutant kill rate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.oracle_evaluation.pandas_eval_common import API_CASES
from experiments.oracle_evaluation.mutant_kill.pandas_mutant_workflow import materialize

MUTANT_KILL_DIR = Path(__file__).resolve().parent
PANDAS_CASES_DIR = REPO_ROOT / "experiments" / "oracle_generation" / "pandas"


@dataclass
class TestRunResult:
    passed: bool
    returncode: int
    duration_seconds: float
    failing_tests: list[str]
    stdout_tail: str


def timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def discover_case_dirs() -> list[Path]:
    discovered = []
    for baseline_path in sorted(PANDAS_CASES_DIR.rglob("baseline_test.py")):
        case_dir = baseline_path.parent
        if (case_dir / "ir_generated_test.py").exists() and (case_dir / "mutant_wrapper.py").exists():
            discovered.append(case_dir)
    return discovered


def load_mutant_info(wrapper_path: Path) -> dict[str, dict[str, Any]]:
    module_name = f"_mutant_wrapper_{wrapper_path.parent.as_posix().replace('/', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, wrapper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import mutant wrapper: {wrapper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "MUTANT_INFO", {})


def run_pytest(test_file: Path, mutant_id: str | None, timeout_seconds: int) -> TestRunResult:
    cmd = ["uv", "run", "pytest", str(test_file), "-q", "--tb=short"]
    env = os.environ.copy()
    env.pop("MUTANT_ID", None)
    if mutant_id:
        env["MUTANT_ID"] = mutant_id

    started = datetime.now().timestamp()
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_seconds,
    )
    duration_seconds = datetime.now().timestamp() - started
    output = completed.stdout + completed.stderr
    failing_tests = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED "):
            failing_tests.append(stripped[len("FAILED ") :])
        elif "::" in stripped and stripped.endswith(" FAILED"):
            failing_tests.append(stripped[: -len(" FAILED")])
    return TestRunResult(
        passed=completed.returncode == 0,
        returncode=completed.returncode,
        duration_seconds=duration_seconds,
        failing_tests=sorted(set(failing_tests)),
        stdout_tail=output[-4000:],
    )


def method_report_name(test_file: Path) -> str:
    return "baseline" if test_file.name == "baseline_test.py" else "ir_generated"


def summarize_method_overall(function_reports: list[dict[str, Any]], method_name: str) -> dict[str, Any]:
    valid_mutants = 0
    killed_mutants = 0
    valid_functions = 0
    winning_functions = 0

    for function_report in function_reports:
        summary = function_report["method_summaries"][method_name]
        valid_mutants += summary["valid_mutants"]
        killed_mutants += summary["killed_mutants"]
        if summary["baseline_passed"]:
            valid_functions += 1
        if function_report["winner"] == method_name:
            winning_functions += 1

    kill_rate = killed_mutants / valid_mutants if valid_mutants else None
    return {
        "valid_mutants": valid_mutants,
        "killed_mutants": killed_mutants,
        "overall_kill_rate": kill_rate,
        "functions_with_passing_baseline": valid_functions,
        "functions_won": winning_functions,
    }


def choose_winner(baseline_rate: float | None, ir_rate: float | None) -> str:
    if baseline_rate is None and ir_rate is None:
        return "no_valid_baseline"
    if baseline_rate == ir_rate:
        return "tie"
    if ir_rate is None:
        return "baseline"
    if baseline_rate is None:
        return "ir_generated"
    return "baseline" if baseline_rate > ir_rate else "ir_generated"


def build_function_report(case_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    wrapper_path = case_dir / "mutant_wrapper.py"
    mutant_info = load_mutant_info(wrapper_path)
    baseline_test = case_dir / "baseline_test.py"
    ir_generated_test = case_dir / "ir_generated_test.py"

    method_inputs = {
        "baseline": baseline_test,
        "ir_generated": ir_generated_test,
    }

    baseline_runs: dict[str, TestRunResult] = {}
    method_details: dict[str, dict[str, Any]] = {}
    for method_name, test_file in method_inputs.items():
        baseline_run = run_pytest(test_file, mutant_id=None, timeout_seconds=timeout_seconds)
        baseline_runs[method_name] = baseline_run
        method_details[method_name] = {
            "test_file": str(test_file),
            "baseline_passed": baseline_run.passed,
            "baseline_run": asdict(baseline_run),
            "valid_mutants": 0,
            "killed_mutants": 0,
            "survived_mutants": 0,
            "invalid_mutants": len(mutant_info) if not baseline_run.passed else 0,
            "kill_rate": None,
            "killed_mutant_ids": [],
            "survived_mutant_ids": [],
            "invalid_mutant_ids": [],
        }

    mutants: list[dict[str, Any]] = []
    for mutant_id, info in mutant_info.items():
        mutant_entry = {
            "mutant_id": mutant_id,
            "mutant_name": info.get("name", mutant_id),
            "description": info.get("description"),
            "doc_anchor": info.get("doc_anchor"),
            "methods": {},
        }
        for method_name, test_file in method_inputs.items():
            baseline_run = baseline_runs[method_name]
            if not baseline_run.passed:
                mutant_result = {
                    "baseline_passed": False,
                    "mutant_run": None,
                    "killed": False,
                    "status": "invalid_baseline",
                }
                method_details[method_name]["invalid_mutant_ids"].append(mutant_id)
            else:
                mutant_run = run_pytest(test_file, mutant_id=mutant_id, timeout_seconds=timeout_seconds)
                killed = not mutant_run.passed
                mutant_result = {
                    "baseline_passed": True,
                    "mutant_run": asdict(mutant_run),
                    "killed": killed,
                    "status": "killed" if killed else "survived",
                }
                method_details[method_name]["valid_mutants"] += 1
                if killed:
                    method_details[method_name]["killed_mutants"] += 1
                    method_details[method_name]["killed_mutant_ids"].append(mutant_id)
                else:
                    method_details[method_name]["survived_mutants"] += 1
                    method_details[method_name]["survived_mutant_ids"].append(mutant_id)
            mutant_entry["methods"][method_name] = mutant_result
        mutants.append(mutant_entry)

    for method_name, summary in method_details.items():
        if summary["valid_mutants"]:
            summary["kill_rate"] = summary["killed_mutants"] / summary["valid_mutants"]

    baseline_rate = method_details["baseline"]["kill_rate"]
    ir_rate = method_details["ir_generated"]["kill_rate"]
    winner = choose_winner(baseline_rate, ir_rate)

    matching_case = next((case for case in API_CASES if case.directory == case_dir), None)
    function_name = matching_case.function if matching_case else str(case_dir.relative_to(PANDAS_CASES_DIR)).replace("/", ".")
    case_doc = next(iter(sorted(case_dir.glob("*.md"))), None)

    return {
        "case_dir": str(case_dir.relative_to(PANDAS_CASES_DIR)),
        "function": function_name,
        "case_doc": str(case_doc) if case_doc else None,
        "mutant_wrapper": str(wrapper_path),
        "total_mutants": len(mutant_info),
        "winner": winner,
        "method_summaries": method_details,
        "mutants": mutants,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output path. Defaults to mutant_kill/pandas_mutant_comparison_<timestamp>.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    materialize()

    case_dirs = discover_case_dirs()
    function_reports = [build_function_report(case_dir, args.timeout_seconds) for case_dir in case_dirs]
    baseline_overall = summarize_method_overall(function_reports, "baseline")
    ir_overall = summarize_method_overall(function_reports, "ir_generated")
    overall_winner = choose_winner(
        baseline_overall["overall_kill_rate"],
        ir_overall["overall_kill_rate"],
    )

    generated_at = datetime.now().astimezone().isoformat()
    report = {
        "generated_at": generated_at,
        "cases_root": str(PANDAS_CASES_DIR),
        "summary": {
            "total_functions": len(function_reports),
            "total_mutants": sum(report_item["total_mutants"] for report_item in function_reports),
            "methods": {
                "baseline": baseline_overall,
                "ir_generated": ir_overall,
            },
            "overall_winner": overall_winner,
        },
        "functions": function_reports,
    }

    output_path = args.output or MUTANT_KILL_DIR / f"pandas_mutant_comparison_{timestamp_slug()}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)

    if overall_winner == "ir_generated":
        return 0
    if overall_winner == "baseline":
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
