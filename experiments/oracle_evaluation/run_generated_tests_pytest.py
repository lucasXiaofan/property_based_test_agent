#!/usr/bin/env python3
"""Batch-run generated pytest suites and write a timestamped JSON report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "experiments" / "oracle_generation" / "pandas"
RESULTS_DIR = REPO_ROOT / "experiments" / "oracle_evaluation" / "results"
SUITE_FILES = {
    "baseline": "baseline_test.py",
    "ir": "ir_generated_test.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("baseline", "ir", "both"),
        default="both",
        help="Which generated test suites to run.",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=TEST_ROOT,
        help="Root directory containing generated pandas tests.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional explicit output file path.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to invoke pytest.",
    )
    parser.add_argument(
        "--timezone",
        default="America/New_York",
        help="IANA timezone to record in the report.",
    )
    parser.add_argument(
        "--max-output-lines",
        type=int,
        default=80,
        help="Maximum stdout/stderr lines to retain per pytest run.",
    )
    return parser.parse_args()


def select_suites(suite_arg: str) -> list[str]:
    if suite_arg == "both":
        return ["baseline", "ir"]
    return [suite_arg]


def discover_tests(test_root: Path, suite_names: list[str]) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    for ir_path in sorted(test_root.glob("**/ir_v2.json")):
        function_dir = ir_path.parent
        relative_dir = function_dir.relative_to(test_root)
        suite_paths: dict[str, str] = {}
        for suite_name in suite_names:
            test_file = function_dir / SUITE_FILES[suite_name]
            if test_file.exists():
                suite_paths[suite_name] = str(test_file)
        if suite_paths:
            discovered.append(
                {
                    "function_dir": str(function_dir),
                    "relative_dir": str(relative_dir),
                    "suites": suite_paths,
                }
            )
    return discovered


def trim_output(text: str, max_lines: int) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def run_pytest(python_exe: str, test_file: str, max_output_lines: int) -> dict[str, Any]:
    cmd = [python_exe, "-m", "pytest", test_file, "-q"]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "status": "passed" if proc.returncode == 0 else "failed",
        "stdout_tail": trim_output(proc.stdout, max_output_lines),
        "stderr_tail": trim_output(proc.stderr, max_output_lines),
    }


def summarize(report_functions: list[dict[str, Any]], suite_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for suite_name in suite_names:
        total = 0
        passed = 0
        failed_files: list[str] = []
        for item in report_functions:
            suite = item["runs"].get(suite_name)
            if suite is None:
                continue
            total += 1
            if suite["returncode"] == 0:
                passed += 1
            else:
                failed_files.append(item["relative_dir"])
        summary[suite_name] = {
            "total_files": total,
            "passed_files": passed,
            "failed_files": total - passed,
            "failed_targets": failed_files,
        }
    return summary


def default_output_path(timezone_name: str) -> Path:
    now = datetime.now(ZoneInfo(timezone_name))
    stamp = now.strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"generated_pytest_report_{stamp}.json"


def main() -> int:
    args = parse_args()
    test_root = args.test_root.resolve()
    suite_names = select_suites(args.suite)
    discovered = discover_tests(test_root, suite_names)
    if not discovered:
        raise SystemExit(f"no matching generated tests found under {test_root}")

    output_path = args.output_json.resolve() if args.output_json else default_output_path(args.timezone)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now_local = datetime.now(ZoneInfo(args.timezone))
    report: dict[str, Any] = {
        "generated_at": {
            "timezone": args.timezone,
            "local": now_local.isoformat(),
            "utc": datetime.utcnow().isoformat() + "Z",
        },
        "test_root": str(test_root),
        "suite_selection": suite_names,
        "functions": [],
    }

    overall_exit_code = 0
    for item in discovered:
        runs: dict[str, Any] = {}
        for suite_name, test_file in item["suites"].items():
            result = run_pytest(args.python, test_file, args.max_output_lines)
            runs[suite_name] = result
            if result["returncode"] != 0:
                overall_exit_code = 1

        report["functions"].append(
            {
                "function_dir": item["function_dir"],
                "relative_dir": item["relative_dir"],
                "runs": runs,
            }
        )

    report["summary"] = summarize(report["functions"], suite_names)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)
    return overall_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
