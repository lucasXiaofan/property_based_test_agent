#!/usr/bin/env python3
"""Calculate properties coverage summary from LLM clause trace JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = Path(args.input_json).resolve()
    out = Path(args.output_json).resolve()
    data = json.loads(inp.read_text(encoding="utf-8"))
    clauses = data.get("clauses", [])

    total = len(clauses)
    satisfied_rows = [c for c in clauses if c.get("satisfied")]
    unsatisfied_rows = [c for c in clauses if not c.get("satisfied")]

    def avg_conf(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get("confidence", 0.0)))
            except Exception:
                vals.append(0.0)
        return sum(vals) / len(vals)

    result = {
        "function": data.get("function"),
        "test_file": data.get("test_file"),
        "total_clauses": total,
        "satisfied_clauses": len(satisfied_rows),
        "unsatisfied_clauses": len(unsatisfied_rows),
        "properties_coverage": (len(satisfied_rows) / total) if total else 0.0,
        "avg_confidence_satisfied": avg_conf(satisfied_rows),
        "avg_confidence_unsatisfied": avg_conf(unsatisfied_rows),
        "unsatisfied_clause_ids": [r.get("clause_id") for r in unsatisfied_rows],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
