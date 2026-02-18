"""Run baseline and mutant test evaluations and write JSON kill reports."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Ensure imports work whether run from repo root or nested directories.
PBF_ROOT = Path(__file__).resolve().parents[3]
if str(PBF_ROOT) not in sys.path:
    sys.path.insert(0, str(PBF_ROOT))

from mutants.ewm_aggregate.ewm_aggregate_mutants import list_mutants


FAILED_LINE_RE = re.compile(r"^FAILED\s+(.+?)\s+-\s+(.+)$", re.MULTILINE)


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str
    cmd: list[str]

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def run_pytest(test_file: str, mutant_id: str | None, hypothesis_seed: int | None) -> CmdResult:
    env = os.environ.copy()
    if mutant_id:
        env["MUTANT_ID"] = mutant_id
    else:
        env.pop("MUTANT_ID", None)

    cmd = [sys.executable, "-m", "pytest", test_file, "-q"]
    if hypothesis_seed is not None:
        cmd.append(f"--hypothesis-seed={hypothesis_seed}")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return CmdResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        cmd=cmd,
    )


def parse_failures(output_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for node, err in FAILED_LINE_RE.findall(output_text):
        rows.append({"nodeid": node.strip(), "error": err.strip()})
    return rows


def short_summary(text: str, max_chars: int = 1200) -> str:
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars] + "... [truncated]"


def evaluate_test_file(test_file: str, mutant_ids: list[str], hypothesis_seed: int | None) -> dict:
    baseline = run_pytest(test_file=test_file, mutant_id=None, hypothesis_seed=hypothesis_seed)
    baseline_failures = parse_failures(baseline.stdout + "\n" + baseline.stderr)

    record = {
        "test_file": test_file,
        "baseline": {
            "passed": baseline.passed,
            "returncode": baseline.returncode,
            "command": " ".join(baseline.cmd),
            "failures": baseline_failures,
            "stdout_tail": short_summary(baseline.stdout),
            "stderr_tail": short_summary(baseline.stderr),
        },
        "mutants": [],
    }

    for mutant_id in mutant_ids:
        if not baseline.passed:
            record["mutants"].append(
                {
                    "mutant_id": mutant_id,
                    "outcome": "invalid",
                    "reason": "baseline_failed",
                    "command": None,
                    "returncode": None,
                    "failures": [],
                }
            )
            continue

        mutant_run = run_pytest(
            test_file=test_file, mutant_id=mutant_id, hypothesis_seed=hypothesis_seed
        )
        mutant_failures = parse_failures(mutant_run.stdout + "\n" + mutant_run.stderr)
        outcome = "killed" if not mutant_run.passed else "survived"

        record["mutants"].append(
            {
                "mutant_id": mutant_id,
                "outcome": outcome,
                "command": " ".join(mutant_run.cmd),
                "returncode": mutant_run.returncode,
                "failures": mutant_failures,
                "stdout_tail": short_summary(mutant_run.stdout),
                "stderr_tail": short_summary(mutant_run.stderr),
            }
        )

    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-file",
        action="append",
        dest="test_files",
        default=[],
        help="Pytest file path. Repeatable. Defaults to the two ewm aggregate files.",
    )
    parser.add_argument(
        "--mutant-id",
        action="append",
        dest="mutant_ids",
        default=[],
        help="Specific mutant id(s). Repeatable. Defaults to all registered mutants.",
    )
    parser.add_argument(
        "--hypothesis-seed",
        type=int,
        default=0,
        help="Seed for deterministic Hypothesis runs. Use -1 to omit the seed flag.",
    )
    parser.add_argument(
        "--output",
        default="pandas_bug_finding/mutants/ewm_aggregate/results/kill_report.json",
        help="Path for JSON report output.",
    )
    args = parser.parse_args()

    test_files = args.test_files or [
        "pandas_bug_finding/baseline_testing/test_ewm_aggregate_hypothesis.py",
        "pandas_bug_finding/ir_test/test_ewm_aggregate.py",
    ]
    mutants = args.mutant_ids or [m["mutant_id"] for m in list_mutants()]
    hypothesis_seed = None if args.hypothesis_seed == -1 else args.hypothesis_seed

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_symbol": "pandas.core.window.ewm.ExponentialMovingWindow.aggregate",
        "mutants": list_mutants(),
        "evaluations": [],
    }

    for test_file in test_files:
        report["evaluations"].append(
            evaluate_test_file(
                test_file=test_file,
                mutant_ids=mutants,
                hypothesis_seed=hypothesis_seed,
            )
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {out}")

    # Non-zero exit only if baseline passes and at least one mutant survives all evaluated files.
    survivors = []
    for ev in report["evaluations"]:
        if not ev["baseline"]["passed"]:
            continue
        for mr in ev["mutants"]:
            if mr["outcome"] == "survived":
                survivors.append((ev["test_file"], mr["mutant_id"]))
    if survivors:
        print("Survivors detected:")
        for test_file, mutant_id in survivors:
            print(f"  - {mutant_id} survived under {test_file}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
