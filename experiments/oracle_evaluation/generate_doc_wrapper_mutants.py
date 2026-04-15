#!/usr/bin/env python3
"""Generate a catalog of documentation-derived wrapper mutants for pandas cases."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.oracle_evaluation.pandas_eval_common import API_CASES, DOC_MUTANT_TEMPLATES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/oracle_generation/pandas/doc_wrapper_mutants.json"),
        help="Where to write the mutant catalog JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated_at = datetime.now().astimezone().isoformat()

    payload = {
        "generated_at": generated_at,
        "generator": Path(__file__).name,
        "apis": [],
    }

    for case in API_CASES:
        doc_text = case.doc_path.read_text(encoding="utf-8")
        api_entry = {
            "case_dir": case.case_dir,
            "function": case.function,
            "doc_label": case.doc_label,
            "doc_path": str(case.doc_path),
            "ir_json": str(case.ir_json),
            "doc_excerpt": doc_text[:400],
            "mutants": DOC_MUTANT_TEMPLATES[case.function],
        }
        payload["apis"].append(api_entry)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
