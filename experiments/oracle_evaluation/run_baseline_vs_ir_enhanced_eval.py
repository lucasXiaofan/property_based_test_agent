#!/usr/bin/env python3
"""Compare baseline_test.py and ir_enhanced_test.py across pandas oracle cases.

This script scans a pandas oracle-generation tree for case directories that contain
both `baseline_test.py` and `ir_enhanced_test.py`. For each case, it runs:

1. Line/branch coverage against the target pandas API.
2. Mutant-kill evaluation using the case-local `mutant_wrapper.py`.

It then writes a timestamped overall report to:
`experiments/oracle_evaluation/overall_report/`
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_GEN_ROOT = REPO_ROOT / "experiments" / "oracle_generation" / "pandas"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "oracle_evaluation" / "overall_report"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.oracle_evaluation.run_api_coverage import run_subprocess_case  # noqa: E402


SUITES: tuple[tuple[str, str], ...] = (
    ("baseline", "baseline_test.py"),
    ("ir_enhanced", "ir_enhanced_test.py"),
)


@dataclass
class TestRunResult:
    passed: bool
    returncode: int
    duration_seconds: float
    failing_tests: list[str]
    stdout_tail: str


def timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=ORACLE_GEN_ROOT,
        help="Root directory to scan for pandas oracle cases.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("local_pandas/pandas"),
        help="Local pandas checkout root used by line/branch coverage.",
    )
    parser.add_argument(
        "--runtime",
        choices=("installed", "local"),
        default="local",
        help="Import pandas from the installed environment or the local checkout.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Per-pytest timeout for mutant-kill runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the timestamped overall report files.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to pytest for line/branch coverage. Use '--' before them.",
    )
    return parser.parse_args()


def clean_pytest_args(pytest_args: list[str]) -> list[str]:
    if pytest_args and pytest_args[0] == "--":
        return pytest_args[1:]
    return pytest_args[:]


def discover_case_dirs(scan_root: Path) -> list[Path]:
    case_dirs: list[Path] = []
    for baseline_path in sorted(scan_root.rglob("baseline_test.py")):
        case_dir = baseline_path.parent
        if (case_dir / "ir_enhanced_test.py").exists():
            case_dirs.append(case_dir)
    return case_dirs


def case_dir_to_api(case_dir: Path, scan_root: Path) -> str:
    rel = case_dir.resolve().relative_to(scan_root.resolve())
    return f"pandas.{'.'.join(rel.parts)}"


def load_mutant_info(wrapper_path: Path) -> dict[str, dict[str, Any]]:
    module_name = f"_mutant_wrapper_{wrapper_path.parent.as_posix().replace('/', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, wrapper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import mutant wrapper: {wrapper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "MUTANT_INFO", {})


def choose_mutant_winner(baseline_rate: float | None, ir_rate: float | None) -> str:
    if baseline_rate is None and ir_rate is None:
        return "no_valid_baseline"
    if baseline_rate == ir_rate:
        return "tie"
    if ir_rate is None:
        return "baseline"
    if baseline_rate is None:
        return "ir_enhanced"
    return "baseline" if baseline_rate > ir_rate else "ir_enhanced"


def prepend_pythonpath(env: dict[str, str], *paths: Path) -> dict[str, str]:
    current = env.get("PYTHONPATH", "")
    parts = [str(path.resolve()) for path in paths]
    if current:
        parts.append(current)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_mutant_pytest(test_file: Path, mutant_id: str | None, timeout_seconds: int) -> TestRunResult:
    cmd = [sys.executable, "-m", "pytest", str(test_file), "-q", "--tb=short"]
    env = os.environ.copy()
    env.pop("MUTANT_ID", None)
    if mutant_id:
        env["MUTANT_ID"] = mutant_id
    prepend_pythonpath(env, REPO_ROOT / "local_pandas" / "pandas", REPO_ROOT)

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


def suite_winner_from_coverage(
    baseline_coverage: dict[str, Any], ir_enhanced_coverage: dict[str, Any]
) -> dict[str, str]:
    baseline_core = baseline_coverage["core_method_totals"]
    ir_core = ir_enhanced_coverage["core_method_totals"]

    baseline_line = float(baseline_core["line_coverage_percent"])
    ir_line = float(ir_core["line_coverage_percent"])
    baseline_branch = float(baseline_core["branch_coverage_percent"])
    ir_branch = float(ir_core["branch_coverage_percent"])

    line_winner = "tie"
    if baseline_line > ir_line:
        line_winner = "baseline"
    elif ir_line > baseline_line:
        line_winner = "ir_enhanced"

    branch_winner = "tie"
    if baseline_branch > ir_branch:
        branch_winner = "baseline"
    elif ir_branch > baseline_branch:
        branch_winner = "ir_enhanced"

    overall_winner = "tie"
    if branch_winner != "tie":
        overall_winner = branch_winner
    elif line_winner != "tie":
        overall_winner = line_winner

    return {
        "line": line_winner,
        "branch": branch_winner,
        "overall": overall_winner,
    }


def build_case_coverage_report(
    *,
    case_dir: Path,
    scan_root: Path,
    repo_root: Path,
    runtime: str,
    pytest_args: list[str],
) -> dict[str, Any]:
    api = case_dir_to_api(case_dir, scan_root)
    suites: dict[str, dict[str, Any]] = {}
    coverage_script_path = REPO_ROOT / "experiments" / "oracle_evaluation" / "run_api_coverage.py"
    for suite_name, file_name in SUITES:
        test_file = case_dir / file_name
        with tempfile.NamedTemporaryFile(
            "w+",
            suffix=f"__{case_dir.name}__{suite_name}.json",
            delete=False,
        ) as tmp:
            json_out = Path(tmp.name)

        suite_payload = run_subprocess_case(
            script_path=coverage_script_path,
            repo_root=repo_root,
            runtime=runtime,
            api=api,
            test_file=test_file,
            json_out=json_out,
            no_follow_super=False,
            no_require_core=False,
            pytest_args=pytest_args,
        )
        json_out.unlink(missing_ok=True)

        if suite_payload["coverage"] is None:
            if runtime != "local":
                raise RuntimeError(
                    f"Coverage failed for {test_file} with runtime={runtime}.\n"
                    f"stdout:\n{suite_payload['stdout']}\n"
                    f"stderr:\n{suite_payload['stderr']}"
                )
            with tempfile.NamedTemporaryFile(
                "w+",
                suffix=f"__{case_dir.name}__{suite_name}__installed.json",
                delete=False,
            ) as tmp:
                fallback_json_out = Path(tmp.name)
            fallback_payload = run_subprocess_case(
                script_path=coverage_script_path,
                repo_root=repo_root,
                runtime="installed",
                api=api,
                test_file=test_file,
                json_out=fallback_json_out,
                no_follow_super=False,
                no_require_core=False,
                pytest_args=pytest_args,
            )
            fallback_json_out.unlink(missing_ok=True)
            if fallback_payload["coverage"] is None:
                raise RuntimeError(
                    f"Coverage failed for {test_file} with runtime=local and runtime=installed.\n"
                    f"local stdout:\n{suite_payload['stdout']}\n"
                    f"local stderr:\n{suite_payload['stderr']}\n"
                    f"installed stdout:\n{fallback_payload['stdout']}\n"
                    f"installed stderr:\n{fallback_payload['stderr']}"
                )
            suite_payload = fallback_payload
            runtime_used = "installed"
            runtime_fallback_reason = "local runtime produced no coverage payload"
        else:
            runtime_used = runtime
            runtime_fallback_reason = None

        suites[suite_name] = {
            "command": suite_payload["command"],
            "returncode": suite_payload["returncode"],
            "stdout": suite_payload["stdout"],
            "stderr": suite_payload["stderr"],
            "coverage": {
                **suite_payload["coverage"],
                "runtime_requested": runtime,
                "runtime_used": runtime_used,
                "runtime_fallback_reason": runtime_fallback_reason,
            },
        }

    winners = suite_winner_from_coverage(
        suites["baseline"]["coverage"], suites["ir_enhanced"]["coverage"]
    )
    return {
        "case_id": str(case_dir.relative_to(scan_root)),
        "target_api": api,
        "case_dir": str(case_dir),
        "suites": suites,
        "winner": winners,
    }


def run_mutant_suite(
    *, test_file: Path, mutant_ids: list[str], timeout_seconds: int
) -> dict[str, Any]:
    baseline_run = run_mutant_pytest(test_file, mutant_id=None, timeout_seconds=timeout_seconds)
    summary = {
        "test_file": str(test_file),
        "baseline_passed": baseline_run.passed,
        "baseline_run": asdict(baseline_run),
        "valid_mutants": 0,
        "killed_mutants": 0,
        "survived_mutants": 0,
        "invalid_mutants": len(mutant_ids) if not baseline_run.passed else 0,
        "kill_rate": None,
        "killed_mutant_ids": [],
        "survived_mutant_ids": [],
        "invalid_mutant_ids": [],
    }

    mutant_runs: dict[str, dict[str, Any]] = {}
    for mutant_id in mutant_ids:
        if not baseline_run.passed:
            summary["invalid_mutant_ids"].append(mutant_id)
            mutant_runs[mutant_id] = {
                "baseline_passed": False,
                "mutant_run": None,
                "killed": False,
                "status": "invalid_baseline",
            }
            continue

        mutant_run = run_mutant_pytest(test_file, mutant_id=mutant_id, timeout_seconds=timeout_seconds)
        killed = not mutant_run.passed
        summary["valid_mutants"] += 1
        if killed:
            summary["killed_mutants"] += 1
            summary["killed_mutant_ids"].append(mutant_id)
        else:
            summary["survived_mutants"] += 1
            summary["survived_mutant_ids"].append(mutant_id)
        mutant_runs[mutant_id] = {
            "baseline_passed": True,
            "mutant_run": asdict(mutant_run),
            "killed": killed,
            "status": "killed" if killed else "survived",
        }

    if summary["valid_mutants"]:
        summary["kill_rate"] = summary["killed_mutants"] / summary["valid_mutants"]

    return {
        "summary": summary,
        "mutant_runs": mutant_runs,
    }


def build_case_mutant_report(case_dir: Path, scan_root: Path, timeout_seconds: int) -> dict[str, Any]:
    wrapper_path = case_dir / "mutant_wrapper.py"
    mutant_info = load_mutant_info(wrapper_path)
    mutant_ids = list(mutant_info.keys())

    suite_results: dict[str, dict[str, Any]] = {}
    for suite_name, file_name in SUITES:
        suite_results[suite_name] = run_mutant_suite(
            test_file=case_dir / file_name,
            mutant_ids=mutant_ids,
            timeout_seconds=timeout_seconds,
        )

    baseline_rate = suite_results["baseline"]["summary"]["kill_rate"]
    ir_rate = suite_results["ir_enhanced"]["summary"]["kill_rate"]
    winner = choose_mutant_winner(baseline_rate, ir_rate)

    mutants: list[dict[str, Any]] = []
    for mutant_id, info in mutant_info.items():
        mutants.append(
            {
                "mutant_id": mutant_id,
                "mutant_name": info.get("name", mutant_id),
                "description": info.get("description"),
                "doc_anchor": info.get("doc_anchor"),
                "methods": {
                    suite_name: suite_results[suite_name]["mutant_runs"][mutant_id]
                    for suite_name, _ in SUITES
                },
            }
        )

    return {
        "case_id": str(case_dir.relative_to(scan_root)),
        "target_api": case_dir_to_api(case_dir, scan_root),
        "case_dir": str(case_dir),
        "mutant_wrapper": str(wrapper_path),
        "mutant_count": len(mutant_ids),
        "winner": winner,
        "method_summaries": {
            suite_name: suite_results[suite_name]["summary"] for suite_name, _ in SUITES
        },
        "mutants": mutants,
    }


def summarize_coverage_reports(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_line_values: list[float] = []
    baseline_branch_values: list[float] = []
    ir_line_values: list[float] = []
    ir_branch_values: list[float] = []
    line_wins = {"baseline": 0, "ir_enhanced": 0, "tie": 0}
    branch_wins = {"baseline": 0, "ir_enhanced": 0, "tie": 0}
    overall_wins = {"baseline": 0, "ir_enhanced": 0, "tie": 0}

    for case in case_reports:
        baseline_core = case["suites"]["baseline"]["coverage"]["core_method_totals"]
        ir_core = case["suites"]["ir_enhanced"]["coverage"]["core_method_totals"]
        baseline_line_values.append(float(baseline_core["line_coverage_percent"]))
        baseline_branch_values.append(float(baseline_core["branch_coverage_percent"]))
        ir_line_values.append(float(ir_core["line_coverage_percent"]))
        ir_branch_values.append(float(ir_core["branch_coverage_percent"]))
        line_wins[case["winner"]["line"]] += 1
        branch_wins[case["winner"]["branch"]] += 1
        overall_wins[case["winner"]["overall"]] += 1

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    avg_baseline_line = avg(baseline_line_values)
    avg_ir_line = avg(ir_line_values)
    avg_baseline_branch = avg(baseline_branch_values)
    avg_ir_branch = avg(ir_branch_values)

    line_winner = "tie"
    if avg_baseline_line > avg_ir_line:
        line_winner = "baseline"
    elif avg_ir_line > avg_baseline_line:
        line_winner = "ir_enhanced"

    branch_winner = "tie"
    if avg_baseline_branch > avg_ir_branch:
        branch_winner = "baseline"
    elif avg_ir_branch > avg_baseline_branch:
        branch_winner = "ir_enhanced"

    overall_winner = "tie"
    if overall_wins["baseline"] > overall_wins["ir_enhanced"]:
        overall_winner = "baseline"
    elif overall_wins["ir_enhanced"] > overall_wins["baseline"]:
        overall_winner = "ir_enhanced"

    return {
        "cases_compared": len(case_reports),
        "line_coverage": {
            "winner": line_winner,
            "baseline_percent": avg_baseline_line,
            "ir_enhanced_percent": avg_ir_line,
            "margin_percentage_points": round(abs(avg_baseline_line - avg_ir_line), 2),
            "case_wins": line_wins,
        },
        "branch_coverage": {
            "winner": branch_winner,
            "baseline_percent": avg_baseline_branch,
            "ir_enhanced_percent": avg_ir_branch,
            "margin_percentage_points": round(abs(avg_baseline_branch - avg_ir_branch), 2),
            "case_wins": branch_wins,
        },
        "overall_case_winner": {
            "winner": overall_winner,
            "case_wins": overall_wins,
        },
    }


def summarize_mutant_reports(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    total_mutants = 0
    baseline_valid = 0
    baseline_killed = 0
    ir_valid = 0
    ir_killed = 0
    case_wins = {"baseline": 0, "ir_enhanced": 0, "tie": 0, "no_valid_baseline": 0}

    for case in case_reports:
        total_mutants += int(case["mutant_count"])
        baseline_summary = case["method_summaries"]["baseline"]
        ir_summary = case["method_summaries"]["ir_enhanced"]
        baseline_valid += int(baseline_summary["valid_mutants"])
        baseline_killed += int(baseline_summary["killed_mutants"])
        ir_valid += int(ir_summary["valid_mutants"])
        ir_killed += int(ir_summary["killed_mutants"])
        case_wins[case["winner"]] += 1

    baseline_rate = round((baseline_killed / baseline_valid) * 100, 2) if baseline_valid else None
    ir_rate = round((ir_killed / ir_valid) * 100, 2) if ir_valid else None

    overall_winner = "tie"
    if baseline_rate is None and ir_rate is None:
        overall_winner = "no_valid_baseline"
    elif baseline_rate is None:
        overall_winner = "ir_enhanced"
    elif ir_rate is None:
        overall_winner = "baseline"
    elif baseline_rate > ir_rate:
        overall_winner = "baseline"
    elif ir_rate > baseline_rate:
        overall_winner = "ir_enhanced"

    return {
        "cases_compared": len(case_reports),
        "mutants_tested": total_mutants,
        "kill_rate": {
            "winner": overall_winner,
            "baseline_percent": baseline_rate,
            "ir_enhanced_percent": ir_rate,
            "margin_percentage_points": (
                round(abs(baseline_rate - ir_rate), 2)
                if baseline_rate is not None and ir_rate is not None
                else None
            ),
        },
        "case_wins": case_wins,
    }


def build_top_level_summary(
    coverage_summary: dict[str, Any], mutant_summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "line_branch_coverage": {
            "cases_compared": coverage_summary["cases_compared"],
            "line_winner": coverage_summary["line_coverage"]["winner"],
            "line_margin_percentage_points": coverage_summary["line_coverage"]["margin_percentage_points"],
            "branch_winner": coverage_summary["branch_coverage"]["winner"],
            "branch_margin_percentage_points": coverage_summary["branch_coverage"]["margin_percentage_points"],
            "overall_case_winner": coverage_summary["overall_case_winner"]["winner"],
        },
        "mutant_kill": {
            "cases_compared": mutant_summary["cases_compared"],
            "mutants_tested": mutant_summary["mutants_tested"],
            "winner": mutant_summary["kill_rate"]["winner"],
            "margin_percentage_points": mutant_summary["kill_rate"]["margin_percentage_points"],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report["dimension_summaries"]["line_branch_coverage"]
    mutant = report["dimension_summaries"]["mutant_kill"]

    lines = [
        "# Baseline vs IR Enhanced Oracle Evaluation",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Scan root: `{report['scan_root']}`",
        f"- Cases compared: {report['total_cases']}",
        "",
        "## Line and Branch Coverage",
        "",
        f"- Average line coverage winner: `{coverage['line_coverage']['winner']}`",
        f"- Baseline average line coverage: {coverage['line_coverage']['baseline_percent']}%",
        f"- IR enhanced average line coverage: {coverage['line_coverage']['ir_enhanced_percent']}%",
        f"- Average branch coverage winner: `{coverage['branch_coverage']['winner']}`",
        f"- Baseline average branch coverage: {coverage['branch_coverage']['baseline_percent']}%",
        f"- IR enhanced average branch coverage: {coverage['branch_coverage']['ir_enhanced_percent']}%",
        "",
        "## Mutant Kill",
        "",
        f"- Kill rate winner: `{mutant['kill_rate']['winner']}`",
        f"- Baseline overall kill rate: {mutant['kill_rate']['baseline_percent']}%",
        f"- IR enhanced overall kill rate: {mutant['kill_rate']['ir_enhanced_percent']}%",
        f"- Total mutants tested: {mutant['mutants_tested']}",
        "",
        "## Case Winners",
        "",
    ]

    for case in report["case_reports"]:
        case_id = case["case_id"]
        coverage_case = case["line_branch_coverage"]["winner"]
        mutant_case = case["mutant_kill"]["winner"]
        lines.append(
            f"- `{case_id}`: coverage overall `{coverage_case['overall']}`, "
            f"line `{coverage_case['line']}`, branch `{coverage_case['branch']}`, "
            f"mutant kill `{mutant_case}`"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    scan_root = args.scan_root.resolve()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pytest_args = clean_pytest_args(args.pytest_args)

    case_dirs = discover_case_dirs(scan_root)
    if not case_dirs:
        raise SystemExit(f"No cases found with both baseline_test.py and ir_enhanced_test.py under {scan_root}")

    coverage_case_reports: list[dict[str, Any]] = []
    mutant_case_reports: list[dict[str, Any]] = []
    combined_case_reports: list[dict[str, Any]] = []

    for case_dir in case_dirs:
        coverage_report = build_case_coverage_report(
            case_dir=case_dir,
            scan_root=scan_root,
            repo_root=repo_root,
            runtime=args.runtime,
            pytest_args=pytest_args,
        )
        coverage_case_reports.append(coverage_report)

        mutant_report = build_case_mutant_report(
            case_dir=case_dir,
            scan_root=scan_root,
            timeout_seconds=args.timeout_seconds,
        )
        mutant_case_reports.append(mutant_report)

        combined_case_reports.append(
            {
                "case_id": coverage_report["case_id"],
                "target_api": coverage_report["target_api"],
                "line_branch_coverage": coverage_report,
                "mutant_kill": mutant_report,
            }
        )

    coverage_summary = summarize_coverage_reports(coverage_case_reports)
    mutant_summary = summarize_mutant_reports(mutant_case_reports)
    report = {
        "report_type": "baseline_vs_ir_enhanced_oracle_evaluation",
        "generated_at": datetime.now().astimezone().isoformat(),
        "scan_root": str(scan_root),
        "repo_root": str(repo_root),
        "runtime": args.runtime,
        "pytest_args": pytest_args,
        "total_cases": len(case_dirs),
        "dimension_summaries": {
            "line_branch_coverage": coverage_summary,
            "mutant_kill": mutant_summary,
        },
        "top_level_summary": build_top_level_summary(coverage_summary, mutant_summary),
        "case_reports": combined_case_reports,
    }

    timestamp = timestamp_slug()
    json_path = output_dir / f"baseline_vs_ir_enhanced_overall_report_{timestamp}.json"
    md_path = output_dir / f"baseline_vs_ir_enhanced_overall_report_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
