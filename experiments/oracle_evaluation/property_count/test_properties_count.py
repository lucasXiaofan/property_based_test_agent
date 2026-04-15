#!/usr/bin/env python3
"""Count unique documentation-based properties exercised by pandas test suites.

For each function directory under `experiments/oracle_generation/pandas`, this script:
  - loads the function documentation `.md`
  - loads `baseline_test.py`
  - loads `ir_generated_test.py`

Each test file is evaluated independently by the model. The model decides, for every
test method in that file, whether the test checks a documentation-grounded property and
whether that property is unique within the same file.

The final JSON report contains:
  - timestamp
  - per-function baseline vs ir_generated unique property counts
  - per-function winner and margin
  - overall winner and margin across all functions
  - detailed per-test analysis for both suites
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE_ROOT = REPO_ROOT / "experiments" / "oracle_generation" / "pandas"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "oracle_evaluation" / "property_count"
DEFAULT_TIMEZONE = "America/New_York"
SUITE_FILES = {
    "baseline": "baseline_test.py",
    "ir_generated": "ir_generated_test.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def collect_test_names(test_path: Path) -> list[str]:
    tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    names.append(f"{node.name}::{item.name}")
    return names


def iter_function_dirs(oracle_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for doc_path in sorted(oracle_root.glob("**/*.md")):
        function_dir = doc_path.parent
        if all((function_dir / suite_file).exists() for suite_file in SUITE_FILES.values()):
            dirs.append(function_dir)
    return dirs


def find_doc_file(function_dir: Path) -> Path:
    doc_files = sorted(function_dir.glob("*.md"))
    if not doc_files:
        raise FileNotFoundError(f"no documentation .md found in {function_dir}")
    if len(doc_files) > 1:
        raise ValueError(f"expected exactly one .md in {function_dir}, found {len(doc_files)}")
    return doc_files[0]


def deepseek_json_call(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: openai. Install it in the active environment before running this script."
        ) from exc

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError("model returned empty content")
    return json.loads(content)


SYSTEM_PROMPT = (
    "You are a strict software-testing auditor.\n"
    "Given one Python test file and one documentation file for the same API, determine for each test "
    "whether it checks a documentation-grounded property and whether that property is unique within "
    "that same test file.\n\n"
    "Definitions:\n"
    "1. documentation_grounded: the tested behavior is explicitly stated or clearly implied in the docs.\n"
    "2. unique_doc_property: the test checks a documentation-grounded property that is not already "
    "semantically covered by another test in the same file.\n"
    "3. If two tests check the same documented behavior, only mark one of them unique_doc_property=true.\n"
    "4. If a test is not documentation-grounded, unique_doc_property must be false.\n"
    "5. Return valid JSON only."
)


def build_user_prompt(
    *,
    function_name: str,
    suite_name: str,
    test_names: list[str],
    test_code: str,
    doc_text: str,
) -> str:
    schema_example = {
        "function_name": function_name,
        "suite_name": suite_name,
        "tests": [
            {
                "test_name": "test_example",
                "documentation_grounded": True,
                "unique_doc_property": True,
                "property_description": "Missing labels are filled with NaN by default.",
                "reason": "The docs explicitly state this default behavior, and no other test in the file checks the same documented property.",
            }
        ],
    }
    payload = {
        "task": (
            "For each allowed test name, decide whether it checks a documentation-grounded property, "
            "and whether that documented property is unique within this test file."
        ),
        "rules": [
            "Return exactly one row for every allowed test name.",
            "Use only names from allowed_test_names.",
            "unique_doc_property can be true only when documentation_grounded is true.",
            "property_description should be the concise documented property checked by the test. Leave it empty if the test is not documentation-grounded.",
            "reason should briefly justify both judgments.",
        ],
        "required_output_schema": schema_example,
        "function_name": function_name,
        "suite_name": suite_name,
        "allowed_test_names": test_names,
        "documentation": doc_text,
        "test_code": test_code,
    }
    return json.dumps(payload, ensure_ascii=False)


def normalize_suite_results(
    *,
    raw: dict[str, Any],
    valid_test_names: list[str],
) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for rec in raw.get("tests", []):
        test_name = str(rec.get("test_name", "")).strip()
        if test_name not in valid_test_names or test_name in by_name:
            continue

        documentation_grounded = bool(rec.get("documentation_grounded", False))
        unique_doc_property = bool(rec.get("unique_doc_property", False)) if documentation_grounded else False
        property_description = str(rec.get("property_description", "")).strip()
        reason = str(rec.get("reason", "")).strip()

        by_name[test_name] = {
            "test_name": test_name,
            "documentation_grounded": documentation_grounded,
            "unique_doc_property": unique_doc_property,
            "property_description": property_description if documentation_grounded else "",
            "reason": reason,
        }

    rows: list[dict[str, Any]] = []
    for test_name in valid_test_names:
        row = by_name.get(
            test_name,
            {
                "test_name": test_name,
                "documentation_grounded": False,
                "unique_doc_property": False,
                "property_description": "",
                "reason": "",
            },
        )
        rows.append(row)
    return rows


def build_suite_summary(
    *,
    suite_name: str,
    test_file: Path,
    tests: list[dict[str, Any]],
) -> dict[str, Any]:
    unique_rows = [test for test in tests if test["unique_doc_property"]]
    grounded_rows = [test for test in tests if test["documentation_grounded"]]
    return {
        "suite_name": suite_name,
        "test_file": str(test_file),
        "test_count": len(tests),
        "documentation_grounded_test_count": len(grounded_rows),
        "unique_property_count": len(unique_rows),
        "unique_properties": [
            {
                "test_name": test["test_name"],
                "property_description": test["property_description"],
            }
            for test in unique_rows
        ],
        "tests": tests,
    }


def compare_counts(baseline_count: int, ir_count: int) -> tuple[str, int]:
    if baseline_count > ir_count:
        return "baseline", baseline_count - ir_count
    if ir_count > baseline_count:
        return "ir_generated", ir_count - baseline_count
    return "tie", 0


def evaluate_suite(
    *,
    args: argparse.Namespace,
    api_key: str,
    function_name: str,
    suite_name: str,
    test_file: Path,
    doc_text: str,
) -> dict[str, Any]:
    test_names = collect_test_names(test_file)
    raw = deepseek_json_call(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(
            function_name=function_name,
            suite_name=suite_name,
            test_names=test_names,
            test_code=test_file.read_text(encoding="utf-8"),
            doc_text=doc_text,
        ),
        max_tokens=args.max_tokens,
    )
    tests = normalize_suite_results(raw=raw, valid_test_names=test_names)
    return build_suite_summary(suite_name=suite_name, test_file=test_file, tests=tests)


def evaluate_function_dir(args: argparse.Namespace, function_dir: Path, api_key: str) -> dict[str, Any]:
    doc_file = find_doc_file(function_dir)
    doc_text = doc_file.read_text(encoding="utf-8")
    function_name = doc_file.stem

    suites: dict[str, Any] = {}
    for suite_name, suite_filename in SUITE_FILES.items():
        suites[suite_name] = evaluate_suite(
            args=args,
            api_key=api_key,
            function_name=function_name,
            suite_name=suite_name,
            test_file=function_dir / suite_filename,
            doc_text=doc_text,
        )

    baseline_count = suites["baseline"]["unique_property_count"]
    ir_count = suites["ir_generated"]["unique_property_count"]
    winner, margin = compare_counts(baseline_count, ir_count)

    return {
        "function_dir": str(function_dir),
        "function_key": str(function_dir.relative_to(args.oracle_root)),
        "function_name": function_name,
        "doc_file": str(doc_file),
        "comparison": {
            "baseline_unique_property_count": baseline_count,
            "ir_generated_unique_property_count": ir_count,
            "winner": winner,
            "margin": margin,
        },
        "suites": suites,
    }


def evaluate_function_dir_job(
    index: int,
    total: int,
    args: argparse.Namespace,
    function_dir: Path,
    api_key: str,
    print_lock: threading.Lock,
) -> dict[str, Any]:
    relative_name = str(function_dir.relative_to(args.oracle_root))
    with print_lock:
        print(f"[{index}/{total}] Starting {relative_name}")

    result = evaluate_function_dir(args, function_dir, api_key)
    baseline_count = result["comparison"]["baseline_unique_property_count"]
    ir_count = result["comparison"]["ir_generated_unique_property_count"]
    winner = result["comparison"]["winner"]
    margin = result["comparison"]["margin"]

    with print_lock:
        print(
            f"[{index}/{total}] Finished {relative_name} "
            f"(baseline={baseline_count}, ir_generated={ir_count}, winner={winner}, margin={margin})"
        )
    return result


def build_overall_summary(function_results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_total = sum(item["comparison"]["baseline_unique_property_count"] for item in function_results)
    ir_total = sum(item["comparison"]["ir_generated_unique_property_count"] for item in function_results)
    overall_winner, overall_margin = compare_counts(baseline_total, ir_total)
    return {
        "functions_evaluated": len(function_results),
        "baseline_total_unique_property_count": baseline_total,
        "ir_generated_total_unique_property_count": ir_total,
        "overall_winner": overall_winner,
        "overall_margin": overall_margin,
        "baseline_function_wins": sum(1 for item in function_results if item["comparison"]["winner"] == "baseline"),
        "ir_generated_function_wins": sum(
            1 for item in function_results if item["comparison"]["winner"] == "ir_generated"
        ),
        "tied_functions": sum(1 for item in function_results if item["comparison"]["winner"] == "tie"),
    }


def build_report_path(output_dir: Path, tz_name: str) -> Path:
    local_now = datetime.now(ZoneInfo(tz_name))
    stamp = local_now.strftime("%Y%m%dT%H%M%S%z")
    return output_dir / f"pandas_unique_property_count_report_{stamp}.json"


def main() -> int:
    args = parse_args()
    args.oracle_root = args.oracle_root.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.oracle_root.exists():
        raise SystemExit(f"missing oracle root: {args.oracle_root}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key env var: {args.api_key_env}")

    function_dirs = iter_function_dirs(args.oracle_root)
    if not function_dirs:
        raise SystemExit(f"no function directories found under: {args.oracle_root}")

    total = len(function_dirs)
    worker_count = min(args.workers, total)
    print(f"Evaluating {total} functions with {worker_count} worker(s)")

    function_results: list[dict[str, Any]] = []
    print_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_dir = {
            executor.submit(
                evaluate_function_dir_job,
                idx,
                total,
                args,
                function_dir,
                api_key,
                print_lock,
            ): function_dir
            for idx, function_dir in enumerate(function_dirs, start=1)
        }
        for future in concurrent.futures.as_completed(future_to_dir):
            function_dir = future_to_dir[future]
            try:
                function_results.append(future.result())
            except Exception as exc:
                relative_name = function_dir.relative_to(args.oracle_root)
                raise RuntimeError(f"failed while evaluating {relative_name}: {exc}") from exc

    function_results.sort(key=lambda item: item["function_key"])
    overall_summary = build_overall_summary(function_results)

    utc_now = datetime.now(timezone.utc)
    local_now = utc_now.astimezone(ZoneInfo(args.timezone))
    report = {
        "report_type": "pandas_unique_documented_property_count",
        "generated_at_utc": utc_now.isoformat(),
        "generated_at_local": local_now.isoformat(),
        "timezone": args.timezone,
        "oracle_root": str(args.oracle_root),
        "model": args.model,
        "overall_summary": overall_summary,
        "functions": function_results,
    }

    report_path = build_report_path(args.output_dir, args.timezone)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote: {report_path}")
    print(
        "Overall: "
        f"baseline={overall_summary['baseline_total_unique_property_count']} "
        f"ir_generated={overall_summary['ir_generated_total_unique_property_count']} "
        f"winner={overall_summary['overall_winner']} "
        f"margin={overall_summary['overall_margin']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
