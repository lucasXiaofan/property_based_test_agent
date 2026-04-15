#!/usr/bin/env python3
"""Generate a combined summary from the latest oracle evaluation JSON reports."""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = REPO_ROOT / "experiments" / "oracle_evaluation"
SOURCE_DIRS = {
    "line_branch_coverage": EVAL_ROOT / "line_branch_coverage",
    "mutant_kill": EVAL_ROOT / "mutant_kill",
    "property_count": EVAL_ROOT / "property_count",
}
DEFAULT_OUTPUT_DIR = EVAL_ROOT / "overall_report"
SUITE_METHODS = ("baseline", "ir_generated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output path. Defaults to overall_report/oracle_overall_report_<timestamp>.json",
    )
    return parser.parse_args()


def timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def latest_json_file(directory: Path) -> Path:
    candidates = [path for path in directory.glob("*.json") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no JSON files found in {directory}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_tests_in_file(test_file: Path) -> int:
    tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
    count = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            count += 1
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    count += 1
    return count


def safe_pct_change(winner_value: float, loser_value: float) -> float | None:
    if loser_value == 0:
        return None
    return round(((winner_value - loser_value) / loser_value) * 100, 2)


def compute_tests_tested(
    *,
    line_branch_report: dict[str, Any],
    mutant_kill_report: dict[str, Any],
    property_count_report: dict[str, Any],
) -> dict[str, Any]:
    case_map: dict[str, dict[str, Any]] = {}

    for case in line_branch_report.get("case_details", []):
        case_key = str(case.get("case_id"))
        entry = case_map.setdefault(case_key, {"case_id": case_key, "methods": {}})
        for method_name in SUITE_METHODS:
            suite = case.get("suites", {}).get(method_name, {})
            test_target = suite.get("coverage", {}).get("test_target")
            if test_target:
                entry["methods"].setdefault(method_name, str(test_target))

    for function in mutant_kill_report.get("functions", []):
        case_key = str(function.get("case_dir"))
        entry = case_map.setdefault(case_key, {"case_id": case_key, "methods": {}})
        for method_name in SUITE_METHODS:
            test_file = function.get("method_summaries", {}).get(method_name, {}).get("test_file")
            if test_file:
                entry["methods"].setdefault(method_name, str(test_file))

    for function in property_count_report.get("functions", []):
        case_key = str(function.get("function_key"))
        entry = case_map.setdefault(case_key, {"case_id": case_key, "methods": {}})
        for method_name in SUITE_METHODS:
            test_file = function.get("suites", {}).get(method_name, {}).get("test_file")
            if test_file:
                entry["methods"].setdefault(method_name, str(test_file))

    totals = {method_name: 0 for method_name in SUITE_METHODS}
    cases: list[dict[str, Any]] = []

    for case_key in sorted(case_map):
        method_counts: dict[str, int | None] = {}
        for method_name in SUITE_METHODS:
            test_file_str = case_map[case_key]["methods"].get(method_name)
            if not test_file_str:
                method_counts[method_name] = None
                continue
            test_count = count_tests_in_file(Path(test_file_str))
            method_counts[method_name] = test_count
            totals[method_name] += test_count
        cases.append({"case_id": case_key, "test_counts": method_counts})

    return {
        "total_cases": len(cases),
        "total_tests_by_method": totals,
        "cases": cases,
    }


def build_line_branch_summary(report: dict[str, Any], tests_tested: dict[str, int]) -> dict[str, Any]:
    totals = report["totals"]
    baseline_line = float(totals["average_baseline_core_line_coverage"])
    ir_line = float(totals["average_ir_generated_core_line_coverage"])
    baseline_branch = float(totals["average_baseline_core_branch_coverage"])
    ir_branch = float(totals["average_ir_generated_core_branch_coverage"])

    line_winner = "baseline" if baseline_line > ir_line else "ir_generated" if ir_line > baseline_line else "tie"
    branch_winner = (
        "baseline" if baseline_branch > ir_branch else "ir_generated" if ir_branch > baseline_branch else "tie"
    )

    return {
        "cases_compared": totals["num_cases"],
        "tests_tested_by_method": tests_tested,
        "line_coverage": {
            "winner": line_winner,
            "baseline_percent": baseline_line,
            "ir_generated_percent": ir_line,
            "margin_percent": round(abs(baseline_line - ir_line), 2),
            "margin_percentage_points": round(abs(baseline_line - ir_line), 2),
        },
        "branch_coverage": {
            "winner": branch_winner,
            "baseline_percent": baseline_branch,
            "ir_generated_percent": ir_branch,
            "margin_percent": round(abs(baseline_branch - ir_branch), 2),
            "margin_percentage_points": round(abs(baseline_branch - ir_branch), 2),
        },
    }


def build_mutant_kill_summary(report: dict[str, Any], tests_tested: dict[str, int]) -> dict[str, Any]:
    summary = report["summary"]
    baseline = summary["methods"]["baseline"]
    ir_generated = summary["methods"]["ir_generated"]
    baseline_rate = float(baseline["overall_kill_rate"]) * 100
    ir_rate = float(ir_generated["overall_kill_rate"]) * 100
    winner = summary["overall_winner"]
    winner_value = baseline_rate if winner == "baseline" else ir_rate
    loser_value = ir_rate if winner == "baseline" else baseline_rate

    return {
        "functions_compared": summary["total_functions"],
        "mutants_tested": summary["total_mutants"],
        "tests_tested_by_method": tests_tested,
        "kill_rate": {
            "winner": winner,
            "baseline_percent": round(baseline_rate, 2),
            "ir_generated_percent": round(ir_rate, 2),
            "margin_percent": round(abs(baseline_rate - ir_rate), 2),
            "margin_percentage_points": round(abs(baseline_rate - ir_rate), 2),
            "winner_relative_advantage_percent": safe_pct_change(winner_value, loser_value),
        },
        "functions_won": {
            "baseline": baseline["functions_won"],
            "ir_generated": ir_generated["functions_won"],
            "ties": sum(1 for item in report.get("functions", []) if item.get("winner") == "tie"),
        },
    }


def build_property_count_summary(report: dict[str, Any], tests_tested: dict[str, int]) -> dict[str, Any]:
    overall = report["overall_summary"]
    baseline_count = int(overall["baseline_total_unique_property_count"])
    ir_count = int(overall["ir_generated_total_unique_property_count"])
    winner = overall["overall_winner"]
    winner_value = baseline_count if winner == "baseline" else ir_count
    loser_value = ir_count if winner == "baseline" else baseline_count

    return {
        "functions_compared": overall["functions_evaluated"],
        "tests_tested_by_method": tests_tested,
        "unique_property_count": {
            "winner": winner,
            "baseline_count": baseline_count,
            "ir_generated_count": ir_count,
            "margin_count": abs(baseline_count - ir_count),
            "winner_relative_advantage_percent": safe_pct_change(float(winner_value), float(loser_value)),
        },
        "functions_won": {
            "baseline": overall["baseline_function_wins"],
            "ir_generated": overall["ir_generated_function_wins"],
            "ties": overall["tied_functions"],
        },
    }


def build_top_level_overview(dimension_summaries: dict[str, Any]) -> dict[str, Any]:
    line_branch = dimension_summaries["line_branch_coverage"]
    mutant_kill = dimension_summaries["mutant_kill"]
    property_count = dimension_summaries["property_count"]

    return {
        "line_branch_coverage": {
            "total_cases_compared": line_branch["cases_compared"],
            "line_coverage_winner": line_branch["line_coverage"]["winner"],
            "line_coverage_margin_percentage_points": line_branch["line_coverage"]["margin_percentage_points"],
            "branch_coverage_winner": line_branch["branch_coverage"]["winner"],
            "branch_coverage_margin_percentage_points": line_branch["branch_coverage"]["margin_percentage_points"],
        },
        "mutant_kill": {
            "total_cases_compared": mutant_kill["functions_compared"],
            "winner": mutant_kill["kill_rate"]["winner"],
            "margin_percentage_points": mutant_kill["kill_rate"]["margin_percentage_points"],
            "winner_relative_advantage_percent": mutant_kill["kill_rate"]["winner_relative_advantage_percent"],
        },
        "property_count": {
            "total_cases_compared": property_count["functions_compared"],
            "winner": property_count["unique_property_count"]["winner"],
            "margin_count": property_count["unique_property_count"]["margin_count"],
            "winner_relative_advantage_percent": property_count["unique_property_count"][
                "winner_relative_advantage_percent"
            ],
        },
    }


def winner_scorecard(dimension_summaries: dict[str, Any]) -> dict[str, int]:
    scorecard = {"baseline": 0, "ir_generated": 0, "tie": 0}
    for summary in dimension_summaries.values():
        for metric_summary in summary.values():
            if isinstance(metric_summary, dict) and "winner" in metric_summary:
                winner = metric_summary["winner"]
                scorecard[winner] = scorecard.get(winner, 0) + 1
    return scorecard


def main() -> int:
    args = parse_args()

    latest_reports = {name: latest_json_file(path) for name, path in SOURCE_DIRS.items()}
    loaded_reports = {name: load_json(path) for name, path in latest_reports.items()}

    test_inventory = compute_tests_tested(
        line_branch_report=loaded_reports["line_branch_coverage"],
        mutant_kill_report=loaded_reports["mutant_kill"],
        property_count_report=loaded_reports["property_count"],
    )
    total_tests_by_method = test_inventory["total_tests_by_method"]

    dimension_summaries = {
        "line_branch_coverage": build_line_branch_summary(
            loaded_reports["line_branch_coverage"],
            total_tests_by_method,
        ),
        "mutant_kill": build_mutant_kill_summary(
            loaded_reports["mutant_kill"],
            total_tests_by_method,
        ),
        "property_count": build_property_count_summary(
            loaded_reports["property_count"],
            total_tests_by_method,
        ),
    }

    report = {
        "report_type": "oracle_evaluation_overall_report",
        "generated_at": datetime.now().astimezone().isoformat(),
        "overall_dimension_summary": build_top_level_overview(dimension_summaries),
        "inputs": {name: str(path) for name, path in latest_reports.items()},
        "test_inventory": test_inventory,
        "dimension_summaries": dimension_summaries,
        "winner_scorecard": winner_scorecard(dimension_summaries),
    }

    output_path = args.output or (args.output_dir / f"oracle_overall_report_{timestamp_slug()}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
