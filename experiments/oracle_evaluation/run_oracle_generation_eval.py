#!/usr/bin/env python3
"""Run API/property coverage on oracle-generation tests and write a comparison report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_ROOT = REPO_ROOT / "experiments" / "oracle_generation" / "pandas"
REPORT_ROOT = REPO_ROOT / "experiments" / "oracle_evaluation" / "report"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=ORACLE_ROOT,
        help="Root directory containing experiments/oracle_generation/pandas test suites.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to experiments/oracle_evaluation/report/<timestamp>/.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to invoke the evaluator scripts.",
    )
    parser.add_argument(
        "--timezone",
        default="America/New_York",
        help="IANA timezone stored in the report timestamp.",
    )
    parser.add_argument(
        "--skip-property-coverage",
        action="store_true",
        help="Skip the LLM-backed property coverage step.",
    )
    parser.add_argument(
        "--property-model",
        default="deepseek-chat",
        help="Model name forwarded to test_quality_metric/llm_clause_trace.py.",
    )
    return parser.parse_args()


def iter_function_dirs(oracle_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for ir_path in sorted(oracle_root.glob("**/ir_v2.json")):
        function_dir = ir_path.parent
        if (function_dir / "baseline_test.py").exists() and (function_dir / "ir_generated_test.py").exists():
            dirs.append(function_dir)
    return dirs


def load_ir_metadata(function_dir: Path) -> dict[str, Any]:
    ir_path = function_dir / "ir_v2.json"
    try:
        ir_data = json.loads(ir_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {ir_path}: {exc}") from exc
    metadata = ir_data.get("metadata", {})
    function_name = metadata.get("function")
    if not function_name:
        raise ValueError(f"missing metadata.function in {ir_path}")
    return {
        "qualified_name": f"{metadata.get('library', 'pandas')}.{function_name}",
        "signature": metadata.get("signature"),
        "reference_urls": metadata.get("reference_urls", []),
        "post_condition": ir_data.get("post_condition", []),
    }


def build_property_target_dir(function_dir: Path, tmp_root: Path, oracle_root: Path) -> Path:
    meta = load_ir_metadata(function_dir)
    target_dir = tmp_root / function_dir.relative_to(oracle_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "qualified_name": meta["qualified_name"],
        "signature": meta["signature"],
        "reference_urls": meta["reference_urls"],
        "source": str(function_dir / "ir_v2.json"),
    }
    clauses = []
    for item in meta["post_condition"]:
        clause_id = str(item.get("id", "")).strip()
        if not clause_id:
            continue
        description_parts = [
            str(item.get("source", "")).strip(),
            str(item.get("expected_behavior", "")).strip(),
        ]
        clauses.append(
            {
                "id": clause_id,
                "description": " | ".join(part for part in description_parts if part),
            }
        )
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (target_dir / "clauses.json").write_text(json.dumps(clauses, indent=2), encoding="utf-8")
    return target_dir


def run_command(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_process_log(result: subprocess.CompletedProcess[str], *, keep_lines: int = 50) -> dict[str, Any]:
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    stderr_lines = [line for line in result.stderr.splitlines() if line.strip()]
    payload: dict[str, Any] = {"returncode": result.returncode}
    if stdout_lines:
        payload["stdout_tail"] = stdout_lines[-keep_lines:]
    if stderr_lines:
        payload["stderr_tail"] = stderr_lines[-keep_lines:]
    return payload


def collect_tail_lines(process_blob: dict[str, Any] | None) -> list[str]:
    if not process_blob:
        return []
    return list(process_blob.get("stdout_tail", [])) + list(process_blob.get("stderr_tail", []))


def diagnose_api_coverage(api_run: dict[str, Any]) -> str | None:
    if api_run.get("returncode") == 0 and api_run.get("summary") is not None:
        return None

    text = "\n".join(collect_tail_lines(api_run))
    if "Missing dependency" in text or "ModuleNotFoundError" in text:
        return "coverage runner dependency/import failure"
    if "FAILED " in text or "AssertionError" in text or "Hypothesis" in text:
        return "pytest failed while coverage was running"
    if api_run.get("returncode") == 0 and api_run.get("summary") is None:
        return "coverage runner exited 0 but did not write coverage JSON"
    return f"coverage runner failed with exit code {api_run.get('returncode')}"


def diagnose_property_coverage(property_run: dict[str, Any] | None) -> str | None:
    if property_run is None:
        return None
    llm_process = property_run.get("llm_process") or {}
    calc_process = property_run.get("calc_process")
    summary = property_run.get("summary")
    text = "\n".join(collect_tail_lines(llm_process) + collect_tail_lines(calc_process))
    if summary is not None and llm_process.get("returncode") == 0 and (
        calc_process is None or calc_process.get("returncode") == 0
    ):
        return None
    if "missing API key env var" in text:
        return "missing DEEPSEEK_API_KEY for property coverage"
    if "HTTP " in text:
        return "LLM request failed during property coverage"
    if llm_process.get("returncode") not in (None, 0):
        return f"LLM clause trace failed with exit code {llm_process['returncode']}"
    if calc_process and calc_process.get("returncode") not in (None, 0):
        return f"property coverage aggregation failed with exit code {calc_process['returncode']}"
    if summary is None:
        return "property coverage summary was not produced"
    return "unknown property coverage failure"


def run_api_coverage(python_exe: str, qualified_name: str, test_file: Path, tmp_dir: Path) -> dict[str, Any]:
    output_path = tmp_dir / f"{test_file.stem}_api_coverage.json"
    cmd = [
        python_exe,
        str(REPO_ROOT / "experiments" / "oracle_evaluation" / "run_api_coverage.py"),
        "--api",
        qualified_name,
        "--test",
        str(test_file),
        "--json-out",
        str(output_path),
    ]
    process = run_command(cmd)
    summary = load_json(output_path) if output_path.exists() else None
    result = {
        "command": cmd,
        **compact_process_log(process),
        "summary": summary,
    }
    result["diagnosis"] = diagnose_api_coverage(result)
    return result


def run_property_coverage(
    python_exe: str,
    property_target_dir: Path,
    test_file: Path,
    tmp_dir: Path,
    model: str,
) -> dict[str, Any]:
    trace_path = tmp_dir / f"{test_file.stem}_properties_llm_trace.json"
    coverage_path = tmp_dir / f"{test_file.stem}_properties_coverage.json"
    env = os.environ.copy()

    llm_cmd = [
        python_exe,
        str(REPO_ROOT / "test_quality_metric" / "llm_clause_trace.py"),
        "--target-dir",
        str(property_target_dir),
        "--test-file",
        str(test_file),
        "--out-json",
        str(trace_path),
        "--model",
        model,
    ]
    llm_process = run_command(llm_cmd, env=env)

    calc_cmd = [
        python_exe,
        str(REPO_ROOT / "experiments" / "oracle_evaluation" / "calc_properties_coverage.py"),
        "--input-json",
        str(trace_path),
        "--output-json",
        str(coverage_path),
    ]
    calc_process = run_command(calc_cmd, env=env) if llm_process.returncode == 0 else None
    summary = load_json(coverage_path) if coverage_path.exists() else None
    trace_summary = load_json(trace_path) if trace_path.exists() else None

    result = {
        "llm_command": llm_cmd,
        "llm_process": compact_process_log(llm_process),
        "calc_command": calc_cmd,
        "calc_process": None if calc_process is None else compact_process_log(calc_process),
        "trace_summary": trace_summary,
        "summary": summary,
    }
    result["diagnosis"] = diagnose_property_coverage(result)
    return result


def compare_numbers(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "unknown"
    if a > b:
        return "baseline_test"
    if b > a:
        return "ir_generated_test"
    return "tie"


def extract_api_score(api_run: dict[str, Any]) -> dict[str, float | None]:
    summary = api_run.get("summary") or {}
    totals = summary.get("totals") or {}
    return {
        "line_coverage_percent": totals.get("line_coverage_percent"),
        "branch_coverage_percent": totals.get("branch_coverage_percent"),
    }


def extract_property_score(property_run: dict[str, Any] | None) -> float | None:
    if not property_run:
        return None
    summary = property_run.get("summary") or {}
    return summary.get("properties_coverage")


def build_suite_summary(entries: list[dict[str, Any]], suite_name: str) -> dict[str, Any]:
    line_scores: list[float] = []
    branch_scores: list[float] = []
    property_scores: list[float] = []
    api_successes = 0
    property_successes = 0

    for entry in entries:
        suite = entry.get("suites", {}).get(suite_name)
        if suite is None:
            continue
        api_score = extract_api_score(suite["api_coverage"])
        if suite["api_coverage"]["diagnosis"] is None:
            api_successes += 1
        if api_score["line_coverage_percent"] is not None:
            line_scores.append(float(api_score["line_coverage_percent"]))
        if api_score["branch_coverage_percent"] is not None:
            branch_scores.append(float(api_score["branch_coverage_percent"]))
        property_score = extract_property_score(suite.get("property_coverage"))
        if suite.get("property_coverage") and suite["property_coverage"]["diagnosis"] is None:
            property_successes += 1
        if property_score is not None:
            property_scores.append(float(property_score))

    def avg(values: list[float]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    functions_seen = sum(1 for entry in entries if suite_name in entry.get("suites", {}))
    return {
        "functions_seen": functions_seen,
        "api_coverage_successes": api_successes,
        "property_coverage_successes": property_successes,
        "avg_api_line_coverage_percent": avg(line_scores),
        "avg_api_branch_coverage_percent": avg(branch_scores),
        "avg_property_coverage": avg(property_scores),
    }


def resolve_report_dir(explicit_dir: Path | None, timezone_name: str) -> Path:
    if explicit_dir is not None:
        return explicit_dir.resolve()
    stamp = datetime.now(ZoneInfo(timezone_name)).strftime("%Y%m%d_%H%M%S")
    return (REPORT_ROOT / stamp).resolve()


def render_summary_md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Oracle Evaluation Report")
    lines.append("")
    lines.append(f"Generated at: {report['generated_at']['local']}")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    for suite_name in ("baseline_test", "ir_generated_test"):
        suite = report["aggregate"][suite_name]
        lines.append(
            f"- `{suite_name}`: functions={suite['functions_seen']}, "
            f"api successes={suite['api_coverage_successes']}, "
            f"property successes={suite['property_coverage_successes']}, "
            f"avg api line={suite['avg_api_line_coverage_percent']}, "
            f"avg api branch={suite['avg_api_branch_coverage_percent']}, "
            f"avg property={suite['avg_property_coverage']}"
        )
    lines.append(
        f"- Winners: property=`{report['aggregate']['comparison']['property_coverage_winner']}`, "
        f"api line=`{report['aggregate']['comparison']['api_line_coverage_winner']}`, "
        f"api branch=`{report['aggregate']['comparison']['api_branch_coverage_winner']}`"
    )
    lines.append("")
    lines.append("## Details")
    lines.append("")
    for item in report["functions"]:
        lines.append(f"### `{item.get('qualified_name') or item['function_dir']}`")
        if item.get("setup_error"):
            lines.append(f"- setup error: {item['setup_error']}")
            lines.append("")
            continue
        lines.append(
            f"- winners: property=`{item['comparison']['property_coverage_winner']}`, "
            f"api line=`{item['comparison']['api_line_coverage_winner']}`, "
            f"api branch=`{item['comparison']['api_branch_coverage_winner']}`"
        )
        for suite_name in ("baseline_test", "ir_generated_test"):
            suite = item["suites"][suite_name]
            api_score = extract_api_score(suite["api_coverage"])
            property_score = extract_property_score(suite["property_coverage"])
            lines.append(
                f"- `{suite_name}`: api line={api_score['line_coverage_percent']}, "
                f"api branch={api_score['branch_coverage_percent']}, "
                f"property={property_score}, "
                f"api diagnosis={suite['api_coverage']['diagnosis']}, "
                f"property diagnosis={None if suite['property_coverage'] is None else suite['property_coverage']['diagnosis']}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    oracle_root = args.oracle_root.resolve()
    function_dirs = iter_function_dirs(oracle_root)
    if not function_dirs:
        raise SystemExit(f"no test-pair directories found under {oracle_root}")

    report_dir = resolve_report_dir(args.report_dir, args.timezone)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / "report.json"
    report_md = report_dir / "summary.md"

    now_local = datetime.now(ZoneInfo(args.timezone))
    report: dict[str, Any] = {
        "generated_at": {
            "timezone": args.timezone,
            "local": now_local.isoformat(),
            "utc": datetime.utcnow().isoformat() + "Z",
        },
        "oracle_root": str(oracle_root),
        "property_coverage_enabled": not args.skip_property_coverage,
        "property_coverage_requires": {
            "env_var": "DEEPSEEK_API_KEY",
            "script": str(REPO_ROOT / "test_quality_metric" / "llm_clause_trace.py"),
        },
        "functions": [],
    }

    with tempfile.TemporaryDirectory(prefix="oracle_eval_") as tmp:
        tmp_root = Path(tmp)
        for function_dir in function_dirs:
            try:
                meta = load_ir_metadata(function_dir)
                property_target_dir = build_property_target_dir(function_dir, tmp_root / "property_targets", oracle_root)
            except Exception as exc:
                report["functions"].append(
                    {
                        "function_dir": str(function_dir),
                        "qualified_name": None,
                        "setup_error": str(exc),
                        "comparison": {
                            "property_coverage_winner": "unknown",
                            "api_line_coverage_winner": "unknown",
                            "api_branch_coverage_winner": "unknown",
                        },
                        "suites": {},
                    }
                )
                continue

            suites: dict[str, Any] = {}
            for suite_name in ("baseline_test", "ir_generated_test"):
                test_file = function_dir / f"{suite_name}.py"
                suite_tmp_dir = tmp_root / "runs" / function_dir.relative_to(oracle_root) / suite_name
                suite_tmp_dir.mkdir(parents=True, exist_ok=True)
                api_result = run_api_coverage(args.python, meta["qualified_name"], test_file, suite_tmp_dir)
                property_result = None
                if not args.skip_property_coverage:
                    property_result = run_property_coverage(
                        args.python,
                        property_target_dir,
                        test_file,
                        suite_tmp_dir,
                        args.property_model,
                    )
                suites[suite_name] = {
                    "test_file": str(test_file),
                    "api_coverage": api_result,
                    "property_coverage": property_result,
                }

            baseline_api = extract_api_score(suites["baseline_test"]["api_coverage"])
            ir_api = extract_api_score(suites["ir_generated_test"]["api_coverage"])
            baseline_property = extract_property_score(suites["baseline_test"]["property_coverage"])
            ir_property = extract_property_score(suites["ir_generated_test"]["property_coverage"])

            report["functions"].append(
                {
                    "function_dir": str(function_dir),
                    "qualified_name": meta["qualified_name"],
                    "comparison": {
                        "property_coverage_winner": compare_numbers(baseline_property, ir_property),
                        "api_line_coverage_winner": compare_numbers(
                            baseline_api["line_coverage_percent"],
                            ir_api["line_coverage_percent"],
                        ),
                        "api_branch_coverage_winner": compare_numbers(
                            baseline_api["branch_coverage_percent"],
                            ir_api["branch_coverage_percent"],
                        ),
                    },
                    "suites": suites,
                }
            )

    report["aggregate"] = {
        "baseline_test": build_suite_summary(report["functions"], "baseline_test"),
        "ir_generated_test": build_suite_summary(report["functions"], "ir_generated_test"),
    }
    report["aggregate"]["comparison"] = {
        "property_coverage_winner": compare_numbers(
            report["aggregate"]["baseline_test"]["avg_property_coverage"],
            report["aggregate"]["ir_generated_test"]["avg_property_coverage"],
        ),
        "api_line_coverage_winner": compare_numbers(
            report["aggregate"]["baseline_test"]["avg_api_line_coverage_percent"],
            report["aggregate"]["ir_generated_test"]["avg_api_line_coverage_percent"],
        ),
        "api_branch_coverage_winner": compare_numbers(
            report["aggregate"]["baseline_test"]["avg_api_branch_coverage_percent"],
            report["aggregate"]["ir_generated_test"]["avg_api_branch_coverage_percent"],
        ),
    }

    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md.write_text(render_summary_md(report), encoding="utf-8")
    print(report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
