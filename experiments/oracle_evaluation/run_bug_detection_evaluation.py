#!/usr/bin/env python3
"""Evaluate pandas test suites by whether they capture reported bug triggers."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.oracle_evaluation.pandas_eval_common import (
    API_CASES,
    BUG_INVENTORY_PATH,
    get_issue_entries_for_case,
    load_bug_inventory,
    winner_label,
)


STATUS_SCORES = {"yes": 1.0, "no": 0.0}


TRIGGER_ASSESSMENTS: dict[tuple[str, int], dict[str, Any]] = {
    ("Series/mean", 59965): {
        "trigger_summary": "Nullable FloatingArray / convert_dtypes input mixed with missing values, then reduction semantics with skipna handling.",
        "baseline": {
            "status": "no",
            "reason": "The baseline suite checks `skipna=True` and `skipna=False` with NaNs, but it does not directly use nullable FloatingArray / `convert_dtypes()` inputs, so it does not match the reported trigger.",
        },
        "ir_generated": {
            "status": "no",
            "reason": "The IR suite stresses skipna behavior, but it still uses plain float/bool Series rather than nullable FloatingArray inputs, so it does not match the reported trigger.",
        },
    },
    ("Index/astype", 61099): {
        "trigger_summary": "Build a Series with an object index, convert a sibling Index to nullable string dtype with `astype('string')`, then compare the Series objects.",
        "baseline": {
            "status": "no",
            "reason": "The baseline astype suite validates dtype conversion, copy behavior, and impossible casts, but it never converts to nullable string dtype and never performs downstream Series comparison.",
        },
        "ir_generated": {
            "status": "no",
            "reason": "The IR astype suite focuses on numeric/object conversions and copy semantics only; it never creates a string-dtype Index and compares Series indexed by object vs string indexes.",
        },
    },
    ("DataFrame/groupby", 61356): {
        "trigger_summary": "Categorical grouper containing NaN, grouped with `dropna=False`, then inspect `.groups` / NA-bucket handling.",
        "baseline": {
            "status": "no",
            "reason": "The baseline suite does not directly combine categorical keys, NaN, `dropna=False`, and `.groups` inspection in one test, so it does not match the reported trigger.",
        },
        "ir_generated": {
            "status": "no",
            "reason": "The IR suite covers related pieces, but not the exact `.groups` failure path with categorical NaNs under `dropna=False`, so it is not a direct trigger match.",
        },
    },
    ("Index/shift", 62094): {
        "trigger_summary": "Create a TimedeltaIndex by subtracting a Timestamp from a date range, producing a computed freq-less index, then call `shift(1)`.",
        "baseline": {
            "status": "no",
            "reason": "The baseline shift suite uses `date_range` and `timedelta_range` constructors with explicit frequencies; it never constructs the computed freq-less TimedeltaIndex from timestamp subtraction.",
        },
        "ir_generated": {
            "status": "no",
            "reason": "The IR suite tests freqless TimedeltaIndex behavior, but not the reported computed index produced by datetime arithmetic, so it does not directly match the trigger.",
        },
    },
    ("Series.str/contains", 62240): {
        "trigger_summary": "Pass a compiled regex object, especially one carrying `re.IGNORECASE`, through `Series.str.contains` and compare to expected regex semantics.",
        "baseline": {
            "status": "no",
            "reason": "The baseline contains suite never passes a compiled `re.Pattern` object as `pat`, so it does not directly match the reported trigger.",
        },
        "ir_generated": {
            "status": "no",
            "reason": "The IR contains suite still omits compiled regex objects, so it does not directly match the reported trigger.",
        },
    },
    ("Series.str/match", 62240): {
        "trigger_summary": "Pass a compiled regex object, including one with embedded `re.IGNORECASE`, through `Series.str.match` and verify behavior matches Python regex semantics.",
        "baseline": {
            "status": "no",
            "reason": "The baseline match suite compares compiled regex vs string patterns, but not with embedded `re.IGNORECASE`, so it does not directly match the reported trigger.",
        },
        "ir_generated": {
            "status": "yes",
            "reason": "The IR match suite directly tests both compiled regex parity and a compiled `re.IGNORECASE` pattern, which captures the reported buggy condition.",
        },
    },
    ("Series/mul", 62595): {
        "trigger_summary": "Multiply a string Series by boolean values and compare behavior across string backends, especially arrow-backed strings.",
        "baseline": {
            "status": "no",
            "reason": "The baseline mul suite is entirely numeric and never exercises string Series, bool operands, or backend-specific string semantics.",
        },
        "ir_generated": {
            "status": "no",
            "reason": "The IR mul suite is also numeric-only, so it misses the string-backend and bool-multiplication trigger entirely.",
        },
    },
    ("Series/factorize", 62888): {
        "trigger_summary": "Object-dtype Series mixing `0`, `1`, `False`, and `True`, then check whether factorization preserves four distinct values.",
        "baseline": {
            "status": "no",
            "reason": "The baseline factorize suite never mixes ints and bools in the same object Series, so it does not hit the hash/equality collision that drives the bug.",
        },
        "ir_generated": {
            "status": "no",
            "reason": "The IR factorize suite focuses on strings, missing values, and categoricals, not mixed int/bool object inputs.",
        },
    },
    ("DataFrame/to_json", 63236): {
        "trigger_summary": "Serialize a DataFrame whose column labels are non-nanosecond `TimedeltaIndex` values and verify unit-preserving JSON output.",
        "baseline": {
            "status": "no",
            "reason": "The baseline to_json suite checks orient structure, precision, JSON nulls, and datetime formatting, but not TimedeltaIndex column-label serialization.",
        },
        "ir_generated": {
            "status": "no",
            "reason": "The IR to_json suite checks epoch scaling for datetime values, not non-nanosecond TimedeltaIndex column labels.",
        },
    },
    ("DataFrame/reindex", 63993): {
        "trigger_summary": "Reindex columns so that at least two new columns are introduced while using a string `fill_value`, which previously crashed.",
        "baseline": {
            "status": "no",
            "reason": "The baseline reindex suite uses numeric fill values and tests column insertion separately, so it never combines multi-column column reindexing with a string `fill_value`.",
        },
        "ir_generated": {
            "status": "no",
            "reason": "The IR suite uses `fill_value='missing'` only for row reindexing and does not combine it with multi-column column reindexing, so it does not directly match the reported trigger.",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the JSON evaluation output. Defaults to a timestamped file under experiments/oracle_evaluation.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=REPO_ROOT / "experiments" / "oracle_evaluation" / "agent_evaluation_report.md",
        help="Path for the Markdown report.",
    )
    return parser.parse_args()


def suite_score(status: str) -> float:
    return STATUS_SCORES[status]


def format_rate(value: float) -> str:
    return f"{value:.3f}"


def evaluate(output_json: Path, report_md: Path) -> dict[str, Any]:
    generated_at = datetime.now().astimezone().isoformat()
    inventory = load_bug_inventory()

    api_results: list[dict[str, Any]] = []
    baseline_score_total = 0.0
    ir_score_total = 0.0
    issue_links_total = 0
    baseline_yes = 0
    ir_yes = 0
    for case in API_CASES:
        issues = get_issue_entries_for_case(case)
        issue_rows = []
        api_baseline_score = 0.0
        api_ir_score = 0.0
        api_baseline_yes = 0
        api_ir_yes = 0
        for entry in issues:
            issue_id = int(entry["issue"])
            assessment = TRIGGER_ASSESSMENTS[(case.case_dir, issue_id)]
            local_case = inventory.get(issue_id, {})

            baseline_status = assessment["baseline"]["status"]
            ir_status = assessment["ir_generated"]["status"]
            baseline_score = suite_score(baseline_status)
            ir_score = suite_score(ir_status)

            issue_links_total += 1
            baseline_score_total += baseline_score
            ir_score_total += ir_score
            api_baseline_score += baseline_score
            api_ir_score += ir_score

            baseline_yes += int(baseline_status == "yes")
            ir_yes += int(ir_status == "yes")
            api_baseline_yes += int(baseline_status == "yes")
            api_ir_yes += int(ir_status == "yes")

            issue_rows.append(
                {
                    "issue": issue_id,
                    "issue_url": entry["issue_url"],
                    "issue_title": entry["issue_title"],
                    "hypothesis_sketch": entry["hypothesis_sketch"],
                    "local_status": local_case.get("local_status", "unknown"),
                    "trigger_summary": assessment["trigger_summary"],
                    "baseline": {
                        "status": baseline_status,
                        "score": baseline_score,
                        "reason": assessment["baseline"]["reason"],
                    },
                    "ir_generated": {
                        "status": ir_status,
                        "score": ir_score,
                        "reason": assessment["ir_generated"]["reason"],
                    },
                }
            )

        api_results.append(
            {
                "case_dir": case.case_dir,
                "function": case.function,
                "doc_path": str(case.doc_path),
                "issues": issue_rows,
                "summary": {
                    "issue_links": len(issue_rows),
                    "baseline_yes": api_baseline_yes,
                    "baseline_score": api_baseline_score,
                    "baseline_trigger_coverage_rate": (api_baseline_score / len(issue_rows)) if issue_rows else 0.0,
                    "ir_generated_yes": api_ir_yes,
                    "ir_generated_score": api_ir_score,
                    "ir_generated_trigger_coverage_rate": (api_ir_score / len(issue_rows)) if issue_rows else 0.0,
                    "higher_quality": winner_label(
                        (api_baseline_score / len(issue_rows)) if issue_rows else 0.0,
                        (api_ir_score / len(issue_rows)) if issue_rows else 0.0,
                    ),
                },
            }
        )

    result = {
        "generated_at": generated_at,
        "pandas_version": pd.__version__,
        "bug_inventory": str(BUG_INVENTORY_PATH),
        "scoring": {
            "yes": 1.0,
            "no": 0.0,
        },
        "apis": api_results,
        "overall": {
            "issue_links_evaluated": issue_links_total,
            "baseline_yes": baseline_yes,
            "baseline_score": baseline_score_total,
            "baseline_trigger_coverage_rate": baseline_score_total / issue_links_total if issue_links_total else 0.0,
            "ir_generated_yes": ir_yes,
            "ir_generated_score": ir_score_total,
            "ir_generated_trigger_coverage_rate": ir_score_total / issue_links_total if issue_links_total else 0.0,
            "higher_quality": winner_label(
                baseline_score_total / issue_links_total if issue_links_total else 0.0,
                ir_score_total / issue_links_total if issue_links_total else 0.0,
            ),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(render_report(result, output_json), encoding="utf-8")
    return result


def render_report(result: dict[str, Any], output_json: Path) -> str:
    overall = result["overall"]
    lines = [
        "# Agent Evaluation Report",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Pandas version: `{result['pandas_version']}`",
        f"- JSON details: `{output_json}`",
        f"- Scoring: `yes=1.0`, `no=0.0`",
        "",
        "## Overall",
        "",
        f"- Baseline trigger coverage: `{format_rate(overall['baseline_trigger_coverage_rate'])}` ({overall['baseline_score']} over {overall['issue_links_evaluated']} issue-links; yes={overall['baseline_yes']})",
        f"- IR-generated trigger coverage: `{format_rate(overall['ir_generated_trigger_coverage_rate'])}` ({overall['ir_generated_score']} over {overall['issue_links_evaluated']} issue-links; yes={overall['ir_generated_yes']})",
        f"- Higher-quality suite by bug-trigger coverage: `{overall['higher_quality']}`",
        "",
        "This report does not depend on whether the local pandas wheel still reproduces the bug. It checks whether each suite actually exercises the reported buggy input or condition from the counted issue inventory.",
        "",
        "## Per API",
        "",
    ]

    for api in result["apis"]:
        summary = api["summary"]
        lines.extend(
            [
                f"### {api['function']}",
                "",
                f"- Baseline: `{format_rate(summary['baseline_trigger_coverage_rate'])}` (score={summary['baseline_score']}, yes={summary['baseline_yes']})",
                f"- IR-generated: `{format_rate(summary['ir_generated_trigger_coverage_rate'])}` (score={summary['ir_generated_score']}, yes={summary['ir_generated_yes']})",
                f"- Higher quality: `{summary['higher_quality']}`",
                "",
            ]
        )
        for issue in api["issues"]:
            lines.extend(
                [
                    f"- Issue `#{issue['issue']}`: {issue['issue_title']}",
                    f"  Trigger: {issue['trigger_summary']}",
                    f"  Baseline: `{issue['baseline']['status']}`. {issue['baseline']['reason']}",
                    f"  IR-generated: `{issue['ir_generated']['status']}`. {issue['ir_generated']['reason']}",
                    "",
                ]
            )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    output_json = args.output_json or (
        REPO_ROOT
        / "experiments"
        / "oracle_evaluation"
        / f"bug_trigger_evaluation_{timestamp}.json"
    )
    result = evaluate(output_json, args.report_md)
    print(json.dumps(result["overall"], indent=2))
    print(f"wrote: {output_json}")
    print(f"wrote: {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
