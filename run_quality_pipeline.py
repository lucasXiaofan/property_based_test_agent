#!/usr/bin/env python3
"""
Automation script that:
1. Finds the next function in function2test.csv without has_quality_metrics
2. Launches Claude Code in the project dir with the CODING_AGENT_PROCEDURE instruction
3. Streams and prints stdout
"""

import csv
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
CSV_PATH = PROJECT_DIR / "function2test.csv"
PROCEDURE_PATH = PROJECT_DIR / "test_quality_metric" / "CODING_AGENT_PROCEDURE.md"


def find_next_function(csv_path: Path) -> tuple[str, str] | None:
    """Return (function_name, doc_url) for the first row missing quality metrics."""
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["has_quality_metrics"].strip() == "False":
                return row["function_name"].strip(), row["doc_url"].strip()
    return None


def build_prompt(function_name: str, url: str, procedure_path: Path) -> str:
    procedure_text = procedure_path.read_text()
    return (
        f"Follow the instruction below exactly. Do not check existing code in this "
        f"project before starting — just follow the procedure and complete the task.\n\n"
        f"Given function name: {function_name}\n"
        f"Given url: {url}\n\n"
        f"--- PROCEDURE ---\n{procedure_text}"
    )


def run_claude(prompt: str, cwd: Path) -> None:
    """Run claude -p <prompt> in cwd, writing directly to the terminal."""
    import os
    cmd = ["claude", "-p", prompt, "--verbose", "--dangerously-skip-permissions"]
    print(f"[run_quality_pipeline] Running: claude -p '...' in {cwd}\n", flush=True)
    print("=" * 60, flush=True)

    # Unset CLAUDECODE so nested session check is bypassed
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    # Inherit stdin/stdout/stderr so Claude writes directly to the terminal
    process = subprocess.run(cmd, cwd=str(cwd), env=env)

    print("\n" + "=" * 60, flush=True)
    print(f"[run_quality_pipeline] Claude exited with code {process.returncode}", flush=True)


SYNC_SCRIPT = PROJECT_DIR / "sync_function2test.py"


def run_sync() -> None:
    print(f"\n[run_quality_pipeline] Running sync_function2test.py ...", flush=True)
    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT)],
        cwd=str(PROJECT_DIR),
        text=True,
    )
    print(f"[run_quality_pipeline] sync exited with code {result.returncode}", flush=True)


def main() -> None:
    result = find_next_function(CSV_PATH)
    if result is None:
        print("All functions already have quality metrics. Nothing to do.")
        sys.exit(0)

    function_name, url = result
    print(f"[run_quality_pipeline] Next function: {function_name}")
    print(f"[run_quality_pipeline] Doc URL: {url}\n")

    prompt = build_prompt(function_name, url, PROCEDURE_PATH)
    run_claude(prompt, PROJECT_DIR)
    run_sync()


if __name__ == "__main__":
    main()
