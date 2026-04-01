#!/usr/bin/env python3
"""
Run tasks from a JSONL file, one independent CLI session per task, up to 5 in parallel.

Each line in the JSONL file is a separate task run in its own agent session.
Tasks run concurrently up to --max-parallel (default 5).

Task file format (JSONL — one JSON object per line):
    {"id": "gen-1", "prompt": "Generate property-based tests for pandas DataFrame.reindex"}
    {"id": "gen-2", "prompt": "Generate property-based tests for pandas DataFrame.merge"}
    {"id": "gen-3", "prompt": "Generate property-based tests for pandas Series.apply"}

A plain JSON array is also accepted:
    [
        {"id": "gen-1", "prompt": "..."},
        {"id": "gen-2", "prompt": "..."}
    ]

If the object has no "prompt" key, all key-value pairs are concatenated as "key: value" lines.

Usage examples:

    # Run with Claude Code (default), 5 parallel sessions, cwd = task file's directory
    python run_task.py tasks.jsonl

    # Run with Codex CLI
    python run_task.py tasks.jsonl --runner codex

    # Limit to 3 concurrent sessions
    python run_task.py tasks.jsonl --max-parallel 3

    # Set a specific working directory for every session
    python run_task.py tasks.jsonl --cwd /path/to/repo

    # Forward extra flags to the underlying CLI
    python run_task.py tasks.jsonl -- --output-format json
"""

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

MAX_PARALLEL = 5


def load_tasks(task_file: str) -> list[dict]:
    """Support both JSONL (one JSON object per line) and a JSON array."""
    text = Path(task_file).read_text()
    stripped = text.strip()
    if stripped.startswith("["):
        return json.loads(stripped)
    tasks = []
    for line in stripped.splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks


def build_prompt(task: dict) -> str:
    if "prompt" in task:
        return task["prompt"]
    parts = []
    for key, value in task.items():
        parts.append(f"{key}: {value if isinstance(value, str) else json.dumps(value)}")
    return "\n".join(parts)


def run_one(idx: int, task: dict, runner: str, cwd: str, extra_args: list[str],
            results: list, lock: threading.Lock) -> None:
    prompt = build_prompt(task)
    label = task.get("id") or task.get("task") or f"task-{idx}"

    if runner == "claude":
        cmd = ["claude","--dangerously-skip-permissions", "-p", prompt] + extra_args
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    else:
        cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
               "--cd", cwd, prompt] + extra_args
        proc = subprocess.run(cmd, capture_output=True, text=True)

    with lock:
        status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
        print(f"[{label}] {status}")
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.returncode != 0 and proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        results[idx] = proc.returncode


def main():
    raw_argv = sys.argv[1:]
    if "--" in raw_argv:
        sep_idx = raw_argv.index("--")
        argv = raw_argv[:sep_idx]
        extra = raw_argv[sep_idx + 1:]
    else:
        argv = raw_argv
        extra = []

    parser = argparse.ArgumentParser(
        description="Run each task in a JSONL file as an independent CLI session (max 5 parallel)"
    )
    parser.add_argument("task_file", help="JSONL file — one JSON object per line, each is one task")
    parser.add_argument("--runner", choices=["claude", "codex"], default="claude")
    parser.add_argument("--cwd", default=None,
                        help="Working directory for every session (default: task file's directory)")
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL,
                        help=f"Max concurrent sessions (default: {MAX_PARALLEL})")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.task_file)
    cwd = args.cwd or str(Path(args.task_file).parent.resolve())

    print(f"[run_task] runner={args.runner}  cwd={cwd}  tasks={len(tasks)}  "
          f"max_parallel={args.max_parallel}\n")

    results = [None] * len(tasks)
    lock = threading.Lock()
    semaphore = threading.Semaphore(args.max_parallel)

    def worker(idx, task):
        with semaphore:
            run_one(idx, task, args.runner, cwd, extra, results, lock)

    threads = [threading.Thread(target=worker, args=(i, t), daemon=True)
               for i, t in enumerate(tasks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = sum(1 for rc in results if rc != 0)
    print(f"\n[run_task] done — {len(tasks) - failed}/{len(tasks)} succeeded")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
