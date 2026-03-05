#!/usr/bin/env python3
"""Evaluate each test function for unique and valid pre/post conditions via LLM.

For each test the LLM determines:
  1. unique – the (pre-condition, post-condition) pair is not duplicated by another test.
  2. valid  – both conditions are explicitly grounded in the target documentation.

A test counts as "unique_and_valid" only when both flags are True.

Usage:
  python llm_test_quality_trace.py \\
      --test-file  path/to/test_foo.py \\
      --doc-file   path/to/foo.md \\
      --out-json   path/to/output.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-file", required=True, help="Path to the pytest test file")
    p.add_argument("--doc-file", required=True, help="Path to the function documentation (.md)")
    p.add_argument("--out-json", required=True, help="Destination for the output JSON report")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--max-tokens", type=int, default=8000)
    return p.parse_args()


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


def deepseek_json_call(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> dict:
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
        raise RuntimeError("DeepSeek returned empty content")
    return json.loads(content)


def normalize_result(raw: dict, valid_test_names: set[str]) -> list[dict]:
    """Validate and normalise the LLM response into a list of per-test records."""
    rows: list[dict] = []
    for rec in raw.get("tests", []):
        name = rec.get("test_name", "")
        if name not in valid_test_names:
            continue  # skip hallucinated names

        is_unique = bool(rec.get("is_unique", False))
        is_valid = bool(rec.get("is_valid", False))

        if is_unique and is_valid:
            overall = "unique_and_valid"
        elif not is_unique:
            overall = "not_unique"
        else:
            overall = "not_valid"

        rows.append(
            {
                "test_name": name,
                "pre_condition": rec.get("pre_condition", ""),
                "post_condition": rec.get("post_condition", ""),
                "is_unique": is_unique,
                "unique_reason": rec.get("unique_reason", ""),
                "is_valid": is_valid,
                "valid_reason": rec.get("valid_reason", ""),
                "overall": overall,
            }
        )

    # Preserve input ordering
    order = {n: i for i, n in enumerate(valid_test_names)}
    rows.sort(key=lambda r: order.get(r["test_name"], 9999))
    return rows


def build_summary(rows: list[dict]) -> dict:
    counts: dict[str, int] = {"unique_and_valid": 0, "not_unique": 0, "not_valid": 0}
    for r in rows:
        counts[r["overall"]] = counts.get(r["overall"], 0) + 1
    total = len(rows)
    return {
        "total_tests": total,
        "unique_and_valid_count": counts["unique_and_valid"],
        "not_unique_count": counts["not_unique"],
        "not_valid_count": counts["not_valid"],
        "unique_and_valid_rate": (counts["unique_and_valid"] / total) if total else 0.0,
    }


SYSTEM_PROMPT = (
    "You are a strict software-testing quality auditor. "
    "Your job is to analyse a Python test file alongside the official API documentation "
    "and evaluate each test function on two independent criteria.\n\n"
    "Criteria definitions:\n"
    "  1. UNIQUE – The combination of (pre_condition, post_condition) described by this test "
    "is NOT semantically duplicated by another test in the same file. "
    "Two tests are duplicates if they exercise the same setup and assert the same behaviour, "
    "even if they use different variable names or data.\n"
   "  2. VALID – All of the following must hold:\n"
"     a. DOC-GROUNDED: Both pre_condition and post_condition are explicitly stated "
"        or clearly implied by the official documentation. "
"        Internal implementation details not mentioned in the docs do not qualify.\n"
"     b. FALSIFIABLE: The post_condition would fail for at least one plausible "
"        incorrect implementation of the function. "
"        Assertions that are always true regardless of implementation are invalid "
"        (e.g. 'result is not None' when None is never a valid return, "
)


def build_user_prompt(test_names: list[str], test_code: str, doc_text: str) -> str:
    schema_example = {
        "tests": [
            {
                "test_name": "test_example",
                "is_unique": True,
                "unique_reason": "No other test covers this specific combination",
                "is_valid": True,
                "valid_reason": "Documentation states missing labels receive NaN by default",
            }
        ]
    }
    payload = {
        "task": (
            "For every test function listed in allowed_test_names, produce one entry in the "
            "'tests' array. Evaluate each test on the two criteria (is_unique, is_valid) "
            "and populate all fields including the *_reason explanations."
        ),
        "rules": [
            "Output must be a valid JSON object with a top-level 'tests' array.",
            "Use only test names from allowed_test_names.",
            "is_unique: compare the (pre_condition, post_condition) pair across ALL tests in the file. Mark False only if another test has the same semantic pair.",
            "is_valid: both conditions must be traceable to the documentation text. Cite the relevant documentation sentence in valid_reason.",
            "overall quality is unique_and_valid only when both flags are True.",
            "Provide concise but precise *_reason strings (1-2 sentences each).",
        ],
        "required_output_schema": schema_example,
        "allowed_test_names": test_names,
        "documentation": doc_text,
        "test_code": test_code,
    }
    return json.dumps(payload, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    test_file = Path(args.test_file).resolve()
    doc_file = Path(args.doc_file).resolve()
    out_json = Path(args.out_json).resolve()

    if not test_file.exists():
        raise SystemExit(f"missing test file: {test_file}")
    if not doc_file.exists():
        raise SystemExit(f"missing doc file: {doc_file}")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key env var: {args.api_key_env}")

    test_code = test_file.read_text(encoding="utf-8")
    doc_text = doc_file.read_text(encoding="utf-8")
    test_names = collect_test_names(test_file)

    if not test_names:
        raise SystemExit(f"no test functions found in: {test_file}")

    print(f"Evaluating {len(test_names)} tests against {doc_file.name} …")

    user_prompt = build_user_prompt(test_names, test_code, doc_text)

    llm_raw = deepseek_json_call(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=args.max_tokens,
    )

    rows = normalize_result(llm_raw, set(test_names))

    # Warn about any tests the LLM skipped
    returned = {r["test_name"] for r in rows}
    missing = [n for n in test_names if n not in returned]
    if missing:
        print(f"WARNING: LLM did not return entries for {len(missing)} tests: {missing}")

    summary = build_summary(rows)

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_file": str(test_file),
        "doc_file": str(doc_file),
        "model": args.model,
        "tests": rows,
        "summary": summary,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote: {out_json}")

    # Print human-readable summary
    s = summary
    print(
        f"\nSummary ({s['total_tests']} tests):\n"
        f"  unique_and_valid : {s['unique_and_valid_count']}  "
        f"({s['unique_and_valid_rate']:.0%})\n"
        f"  not_unique       : {s['not_unique_count']}\n"
        f"  not_valid        : {s['not_valid_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
