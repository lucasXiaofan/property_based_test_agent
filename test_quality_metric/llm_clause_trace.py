#!/usr/bin/env python3
"""Generate clause-to-test satisfaction mapping JSON via DeepSeek JSON mode."""

from __future__ import annotations

import argparse
import ast
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-dir", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--max-tokens", type=int, default=4000)
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
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e

    data = json.loads(raw)
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if not content:
        raise RuntimeError("DeepSeek returned empty content")
    return json.loads(content)


def normalize_result(raw: dict, clauses: list[dict], valid_test_names: set[str]) -> list[dict]:
    by_id = {str(item.get("clause_id")): item for item in raw.get("clauses", []) if item.get("clause_id")}
    rows: list[dict] = []
    for clause in clauses:
        cid = clause["id"]
        rec = by_id.get(cid, {})
        satisfied_by = [t for t in rec.get("satisfied_by", []) if t in valid_test_names]
        satisfied = bool(rec.get("satisfied", False) and satisfied_by)
        confidence = rec.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        rows.append(
            {
                "clause_id": cid,
                "description": clause.get("description", ""),
                "satisfied": satisfied,
                "satisfied_by": satisfied_by,
                "unsatisfied_reason": rec.get("unsatisfied_reason", "") if not satisfied else "",
                "confidence": confidence,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    target_dir = Path(args.target_dir).resolve()
    test_file = Path(args.test_file).resolve()
    out_json = Path(args.out_json).resolve()

    clauses_path = target_dir / "clauses.json"
    metadata_path = target_dir / "metadata.json"
    if not clauses_path.exists():
        raise SystemExit(f"missing clauses.json: {clauses_path}")
    if not metadata_path.exists():
        raise SystemExit(f"missing metadata.json: {metadata_path}")
    if not test_file.exists():
        raise SystemExit(f"missing test file: {test_file}")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key env var: {args.api_key_env}")

    clauses = json.loads(clauses_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    test_code = test_file.read_text(encoding="utf-8")
    test_names = collect_test_names(test_file)

    system_prompt = (
        "You are a strict software testing reviewer. "
        "Return json only. The word json is intentional. "
        "Only mark a clause satisfied when one or more test names explicitly verify it. "
        "Use only provided test names in satisfied_by."
    )

    user_prompt = json.dumps(
        {
            "task": "Map each clause to tests that satisfy it.",
            "required_output_json_schema": {
                "clauses": [
                    {
                        "clause_id": "C1",
                        "satisfied": True,
                        "satisfied_by": ["TestClass::test_name"],
                        "unsatisfied_reason": "",
                        "confidence": 0.0,
                    }
                ]
            },
            "rules": [
                "Output must be valid json object.",
                "Use only test names from allowed_test_names.",
                "If not satisfied, satisfied_by must be empty and give unsatisfied_reason.",
                "confidence must be between 0 and 1.",
            ],
            "allowed_test_names": test_names,
            "clauses": clauses,
            "test_file": str(test_file),
            "test_code": test_code,
        },
        ensure_ascii=False,
    )

    llm_raw = deepseek_json_call(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=args.max_tokens,
    )

    normalized = normalize_result(llm_raw, clauses, set(test_names))
    satisfied = sum(1 for r in normalized if r["satisfied"])
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "function": metadata.get("qualified_name"),
        "model": args.model,
        "test_file": str(test_file),
        "clauses": normalized,
        "summary": {
            "total_clauses": len(normalized),
            "satisfied_clauses": satisfied,
            "unsatisfied_clauses": len(normalized) - satisfied,
            "clause_coverage": (satisfied / len(normalized)) if normalized else 0.0,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
