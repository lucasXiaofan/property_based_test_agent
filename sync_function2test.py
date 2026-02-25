#!/usr/bin/env python3
"""
Scan ir2test_pipeline/ and test_quality_metric/ to update function2test.csv.

For each function row in the CSV, derives the folder path from the doc_url:
  pandas.DataFrame.reindex.html -> pandas/DataFrame/reindex

Checks:
  ir2test_pipeline/{lib}/{cls}/{func}/baseline_test.py   -> has_baseline_test
  ir2test_pipeline/{lib}/{cls}/{func}/ir_generated_test.py -> has_ir2test
  test_quality_metric/{lib}/{cls}/{func}/mutants/mapping.json -> has_quality_metrics

Updates function2test.csv in-place and prints a summary.
"""

import csv
from pathlib import Path

BASE = Path(__file__).parent
IR2TEST = BASE / "ir2test_pipeline"
QUALITY = BASE / "test_quality_metric"
CSV_PATH = BASE / "function2test.csv"


def parse_doc_url(doc_url: str) -> tuple[str, str, str] | None:
    """
    Extract (lib, cls, func) from a pandas API doc URL.

    Example:
      https://.../api/pandas.DataFrame.reindex.html
      -> ("pandas", "DataFrame", "reindex")
    """
    try:
        filename = doc_url.rstrip("/").split("/")[-1]  # pandas.DataFrame.reindex.html
        stem = filename.replace(".html", "")            # pandas.DataFrame.reindex
        parts = stem.split(".")                         # ["pandas", "DataFrame", "reindex"]
        if len(parts) >= 3:
            lib = parts[0]
            cls = parts[1]
            func = ".".join(parts[2:])  # handle dotted names like ewm.aggregate
            return lib, cls, func
    except Exception:
        pass
    return None


def check_artifacts(lib: str, cls: str, func: str) -> dict[str, bool]:
    ir_dir = IR2TEST / lib / cls / func
    q_dir = QUALITY / lib / cls / func
    return {
        "has_baseline_test": (ir_dir / "baseline_test.py").exists(),
        "has_ir2test": (ir_dir / "ir_generated_test.py").exists(),
        "has_quality_metrics": (q_dir / "mutants" / "mapping.json").exists(),
    }


def main() -> None:
    rows: list[dict] = []
    fieldnames: list[str] = []

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            parsed = parse_doc_url(row.get("doc_url", ""))
            if parsed:
                lib, cls, func = parsed
                artifacts = check_artifacts(lib, cls, func)
                row["has_baseline_test"] = str(artifacts["has_baseline_test"])
                row["has_ir2test"] = str(artifacts["has_ir2test"])
                row["has_quality_metrics"] = str(artifacts["has_quality_metrics"])
            rows.append(row)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"Updated: {CSV_PATH}")
    print(f"Total functions: {len(rows)}")
    has_any = [
        r for r in rows
        if r.get("has_baseline_test") == "True"
        or r.get("has_ir2test") == "True"
        or r.get("has_quality_metrics") == "True"
    ]
    print(f"Functions with at least one artifact: {len(has_any)}")
    if has_any:
        print()
        header = f"  {'function':<20} {'baseline':<10} {'ir2test':<10} {'quality'}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in has_any:
            print(
                f"  {r['function_name']:<20} "
                f"{r['has_baseline_test']:<10} "
                f"{r['has_ir2test']:<10} "
                f"{r['has_quality_metrics']}"
            )


if __name__ == "__main__":
    main()
