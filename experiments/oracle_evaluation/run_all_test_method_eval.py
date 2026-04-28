#!/usr/bin/env python3
"""Rank available oracle test methods by coverage and mutant-kill score."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN_ROOT = REPO_ROOT / "experiments" / "oracle_generation" / "pandas"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "oracle_evaluation" / "overall_report"
METHOD_FILES = {
    "baseline": "baseline_test.py",
    "ir_enhanced": "ir_enhanced_test.py",
    "improved_baseline": "improved_baseline_test.py",
}
PROGRESS_WIDTH = 24

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.oracle_evaluation.run_api_coverage import run_subprocess_case  # noqa: E402


@dataclass
class TestRunResult:
    passed: bool
    returncode: int
    duration_seconds: float
    failing_tests: list[str]
    stdout_tail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scan_root",
        nargs="?",
        type=Path,
        default=DEFAULT_SCAN_ROOT,
        help="Target directory containing task folders.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("local_pandas/pandas"))
    parser.add_argument("--runtime", choices=("installed", "local"), default="local")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to coverage pytest. Use '--' before them.",
    )
    return parser.parse_args()


def timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def clean_pytest_args(args: list[str] | None) -> list[str]:
    if not args:
        return []
    return args[1:] if args[0] == "--" else args[:]


def progress_bar(done: int, total: int) -> str:
    if total <= 0:
        return "[" + "." * PROGRESS_WIDTH + "]"
    filled = round(PROGRESS_WIDTH * done / total)
    return "[" + "#" * filled + "." * (PROGRESS_WIDTH - filled) + "]"


def log(message: str, quiet: bool = False, end: str = "\n") -> None:
    if not quiet:
        print(message, file=sys.stderr, end=end, flush=True)


def discover_case_dirs(scan_root: Path) -> list[Path]:
    roots = {path.parent for name in METHOD_FILES.values() for path in scan_root.rglob(name)}
    return sorted(path for path in roots if (path / "mutant_wrapper.py").exists())


def available_methods(case_dir: Path) -> dict[str, Path]:
    return {
        method: case_dir / file_name
        for method, file_name in METHOD_FILES.items()
        if (case_dir / file_name).exists()
    }


def case_dir_to_api(case_dir: Path, scan_root: Path) -> str:
    return f"pandas.{'.'.join(case_dir.resolve().relative_to(scan_root.resolve()).parts)}"


def load_mutant_info(wrapper_path: Path) -> dict[str, dict[str, Any]]:
    module_name = f"_mutant_wrapper_{abs(hash(wrapper_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, wrapper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import mutant wrapper: {wrapper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "MUTANT_INFO", {})


def prepend_pythonpath(env: dict[str, str], *paths: Path) -> None:
    env["PYTHONPATH"] = os.pathsep.join(
        [str(path.resolve()) for path in paths] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )


def run_mutant_pytest(test_file: Path, mutant_id: str | None, timeout: int) -> TestRunResult:
    env = os.environ.copy()
    env.pop("MUTANT_ID", None)
    if mutant_id:
        env["MUTANT_ID"] = mutant_id
    prepend_pythonpath(env, REPO_ROOT / "local_pandas" / "pandas", REPO_ROOT)

    started = datetime.now().timestamp()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-q", "--tb=short"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        output = completed.stdout + completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        output = stdout + stderr + f"\nTimed out after {timeout}s"
        returncode = 124
    failing_tests = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED "):
            failing_tests.append(stripped[len("FAILED ") :])
        elif "::" in stripped and stripped.endswith(" FAILED"):
            failing_tests.append(stripped[: -len(" FAILED")])
    return TestRunResult(
        passed=returncode == 0,
        returncode=returncode,
        duration_seconds=datetime.now().timestamp() - started,
        failing_tests=sorted(set(failing_tests)),
        stdout_tail=output[-4000:],
    )


def run_coverage(
    test_file: Path,
    case_dir: Path,
    scan_root: Path,
    repo_root: Path,
    runtime: str,
    pytest_args: list[str],
    quiet: bool,
) -> dict[str, Any]:
    script_path = REPO_ROOT / "experiments" / "oracle_evaluation" / "run_api_coverage.py"
    for runtime_try in ([runtime, "installed"] if runtime == "local" else [runtime]):
        log(f"    coverage: runtime={runtime_try} running...", quiet)
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
            json_out = Path(tmp.name)
        try:
            payload = run_subprocess_case(
                script_path=script_path,
                repo_root=repo_root,
                runtime=runtime_try,
                api=case_dir_to_api(case_dir, scan_root),
                test_file=test_file,
                json_out=json_out,
                no_follow_super=False,
                no_require_core=False,
                pytest_args=pytest_args,
            )
        except json.JSONDecodeError as exc:
            payload = {"returncode": 1, "coverage": None, "stderr": f"Invalid coverage JSON: {exc}"}
        json_out.unlink(missing_ok=True)
        if payload["coverage"] is not None:
            core = payload["coverage"]["core_method_totals"]
            log(
                "    coverage: "
                f"line={float(core['line_coverage_percent']):.2f}% "
                f"branch={float(core['branch_coverage_percent']):.2f}%",
                quiet,
            )
            return {
                "returncode": payload["returncode"],
                "runtime_used": runtime_try,
                "line_percent": float(core["line_coverage_percent"]),
                "branch_percent": float(core["branch_coverage_percent"]),
                "payload": payload["coverage"],
            }
        log(f"    coverage: runtime={runtime_try} failed", quiet)
    return {"returncode": payload["returncode"], "runtime_used": None, "error": payload["stderr"][-4000:]}


def run_mutants(test_file: Path, mutant_ids: list[str], timeout: int, quiet: bool) -> dict[str, Any]:
    log("    mutants: baseline run...", quiet)
    baseline_run = run_mutant_pytest(test_file, None, timeout)
    summary: dict[str, Any] = {
        "baseline_passed": baseline_run.passed,
        "baseline_run": asdict(baseline_run),
        "valid_mutants": 0,
        "killed_mutants": 0,
        "survived_mutants": 0,
        "invalid_mutants": 0 if baseline_run.passed else len(mutant_ids),
        "kill_percent": None,
        "killed_mutant_ids": [],
        "survived_mutant_ids": [],
    }
    if not baseline_run.passed:
        log(f"    mutants: baseline failed; skipped {len(mutant_ids)} mutants", quiet)
        return summary
    if not mutant_ids:
        log("    mutants: no mutants found", quiet)
        return summary

    for index, mutant_id in enumerate(mutant_ids, 1):
        log(
            f"\r    mutants: {index}/{len(mutant_ids)} {progress_bar(index, len(mutant_ids))} "
            f"running {mutant_id}...",
            quiet,
            end="",
        )
        mutant_run = run_mutant_pytest(test_file, mutant_id, timeout)
        summary["valid_mutants"] += 1
        if mutant_run.passed:
            summary["survived_mutants"] += 1
            summary["survived_mutant_ids"].append(mutant_id)
        else:
            summary["killed_mutants"] += 1
            summary["killed_mutant_ids"].append(mutant_id)
    if summary["valid_mutants"]:
        summary["kill_percent"] = round(summary["killed_mutants"] / summary["valid_mutants"] * 100, 2)
    log(
        f"\r    mutants: {len(mutant_ids)}/{len(mutant_ids)} {progress_bar(len(mutant_ids), len(mutant_ids))} "
        f"killed={summary['killed_mutants']} survived={summary['survived_mutants']} "
        f"kill={fmt(summary['kill_percent'])}%      ",
        quiet,
    )
    return summary


def score_method(metrics: dict[str, Any]) -> float | None:
    values = [
        metrics.get("line_percent"),
        metrics.get("branch_percent"),
        metrics.get("kill_percent"),
    ]
    values = [float(value) for value in values if value is not None]
    return round(sum(values) / len(values), 2) if values else None


def rank_methods(methods: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "method": method,
            "score": data["score"],
            "line_percent": data.get("line_percent"),
            "branch_percent": data.get("branch_percent"),
            "kill_percent": data.get("kill_percent"),
        }
        for method, data in methods.items()
    ]
    rows.sort(key=lambda row: (row["score"] is None, -(row["score"] or 0), row["method"]))
    best = rows[0]["score"] if rows else None
    for index, row in enumerate(rows, 1):
        row["rank"] = index
        row["diff_from_best"] = None if best is None or row["score"] is None else round(best - row["score"], 2)
    return rows


def build_case_report(
    case_dir: Path,
    scan_root: Path,
    repo_root: Path,
    runtime: str,
    timeout: int,
    pytest_args: list[str],
    case_index: int,
    case_total: int,
    quiet: bool,
) -> dict[str, Any]:
    mutant_ids = list(load_mutant_info(case_dir / "mutant_wrapper.py").keys())
    methods = {}
    method_files = available_methods(case_dir)
    case_id = str(case_dir.relative_to(scan_root))
    log(f"[case {case_index}/{case_total}] {case_id} ({len(method_files)} methods, {len(mutant_ids)} mutants)", quiet)
    for method_index, (method, test_file) in enumerate(method_files.items(), 1):
        log(f"  [method {method_index}/{len(method_files)}] {method} -> {test_file.name}", quiet)
        coverage = run_coverage(test_file, case_dir, scan_root, repo_root, runtime, pytest_args, quiet)
        mutants = run_mutants(test_file, mutant_ids, timeout, quiet)
        metrics = {
            "test_file": str(test_file),
            "coverage": coverage,
            "mutant_kill": mutants,
            "line_percent": coverage.get("line_percent"),
            "branch_percent": coverage.get("branch_percent"),
            "kill_percent": mutants.get("kill_percent"),
        }
        metrics["score"] = score_method(metrics)
        methods[method] = metrics
        log(f"  done: {method} score={fmt(metrics['score'])}", quiet)
    case_report = {
        "case_id": case_id,
        "target_api": case_dir_to_api(case_dir, scan_root),
        "mutant_count": len(mutant_ids),
        "methods": methods,
        "ranking": rank_methods(methods),
    }
    winner = case_report["ranking"][0]["method"] if case_report["ranking"] else "none"
    log(f"[case {case_index}/{case_total}] done: winner={winner}", quiet)
    return case_report


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        method: {"cases": 0, "line": [], "branch": [], "kill": [], "score": []}
        for method in METHOD_FILES
    }
    case_wins = {method: 0 for method in METHOD_FILES} | {"tie": 0}
    for case in cases:
        top_score = case["ranking"][0]["score"] if case["ranking"] else None
        winners = [row["method"] for row in case["ranking"] if row["score"] == top_score]
        case_wins[winners[0] if len(winners) == 1 else "tie"] += 1
        for method, data in case["methods"].items():
            totals[method]["cases"] += 1
            for key, metric in [("line", "line_percent"), ("branch", "branch_percent"), ("kill", "kill_percent")]:
                if data.get(metric) is not None:
                    totals[method][key].append(data[metric])
            if data["score"] is not None:
                totals[method]["score"].append(data["score"])

    methods = {}
    for method, data in totals.items():
        methods[method] = {"cases": data["cases"]}
        for key in ["line", "branch", "kill", "score"]:
            values = data[key]
            methods[method][f"avg_{key}_percent" if key != "score" else "avg_score"] = (
                round(sum(values) / len(values), 2) if values else None
            )

    ranking = rank_methods(
        {
            method: {
                "score": data["avg_score"],
                "line_percent": data["avg_line_percent"],
                "branch_percent": data["avg_branch_percent"],
                "kill_percent": data["avg_kill_percent"],
            }
            for method, data in methods.items()
            if data["cases"]
        }
    )
    return {"case_wins": case_wins, "methods": methods, "ranking": ranking}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Oracle Test Method Evaluation",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Target directory: `{report['scan_root']}`",
        f"- Cases evaluated: {report['total_cases']}",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Method | Score | Line | Branch | Mutant kill | Diff from best |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]["ranking"]:
        lines.append(
            f"| {row['rank']} | `{row['method']}` | {fmt(row['score'])} | {fmt(row['line_percent'])} | "
            f"{fmt(row['branch_percent'])} | {fmt(row['kill_percent'])} | {fmt(row['diff_from_best'])} |"
        )
    lines += ["", "## Case Ranking", ""]
    for case in report["case_reports"]:
        winner = case["ranking"][0]["method"] if case["ranking"] else "none"
        parts = [f"{row['method']} {fmt(row['score'])} (-{fmt(row['diff_from_best'])})" for row in case["ranking"]]
        lines.append(f"- `{case['case_id']}` winner `{winner}`: " + ", ".join(parts))
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)


def main() -> int:
    args = parse_args()
    scan_root = args.scan_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = discover_case_dirs(scan_root)
    if not case_dirs:
        raise SystemExit(f"No task folders with known test files and mutant_wrapper.py under {scan_root}")

    pytest_args = clean_pytest_args(args.pytest_args)
    log(f"Found {len(case_dirs)} task folders under {scan_root}", args.quiet)
    cases = []
    for index, case_dir in enumerate(case_dirs, 1):
        cases.append(
            build_case_report(
                case_dir=case_dir,
                scan_root=scan_root,
                repo_root=args.repo_root.resolve(),
                runtime=args.runtime,
                timeout=args.timeout_seconds,
                pytest_args=pytest_args,
                case_index=index,
                case_total=len(case_dirs),
                quiet=args.quiet,
            )
        )
    report = {
        "report_type": "oracle_test_method_evaluation",
        "generated_at": datetime.now().astimezone().isoformat(),
        "scan_root": str(scan_root),
        "methods": METHOD_FILES,
        "score": "mean of available line, branch, and mutant-kill percentages",
        "total_cases": len(cases),
        "summary": summarize(cases),
        "case_reports": cases,
    }

    stamp = timestamp_slug()
    json_path = output_dir / f"oracle_test_method_eval_{stamp}.json"
    md_path = output_dir / f"oracle_test_method_eval_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
