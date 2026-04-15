#!/usr/bin/env python3
"""Run line/branch coverage for pandas experiment tests.

Single-case example:
    ./.venv/bin/python experiments/oracle_evaluation/run_api_coverage.py \
      --api pandas.DataFrame.reindex \
      --test experiments/oracle_generation/pandas/DataFrame/reindex/baseline_test.py \
      --json-out /tmp/reindex_baseline.json \
      -- -q

Batch example:
    ./.venv/bin/python experiments/oracle_evaluation/run_api_coverage.py \
      --scan-root experiments/oracle_generation/pandas \
      --output-dir experiments/oracle_evaluation/line_branch_coverage \
      -- -q
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType, MethodType
from typing import Any


@dataclass(frozen=True)
class ResolvedMethod:
    owner: type
    method_name: str
    func: Any
    file: Path
    start_line: int
    end_line: int
    delegates_to: str | None = None
    implementation_role: str = "core"

    @property
    def label(self) -> str:
        return f"{self.owner.__module__}.{self.owner.__name__}.{self.method_name}"


@dataclass(frozen=True)
class FileCoverageSummary:
    path: Path
    covered_lines: int
    num_statements: int
    covered_branches: int
    num_branches: int


@dataclass(frozen=True)
class MethodCoverageSummary:
    method: ResolvedMethod
    covered_lines: int
    num_statements: int
    missing_lines: list[int]
    covered_branches: int
    num_branches: int
    missing_branches: dict[int, list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("local_pandas/pandas"),
        help="Local pandas checkout root. Defaults to local_pandas/pandas.",
    )
    parser.add_argument(
        "--runtime",
        choices=("installed", "local"),
        default="local",
        help="Import pandas from the installed environment or the local checkout.",
    )
    parser.add_argument("--api", help="Single-case dotted API path, e.g. pandas.DataFrame.reindex")
    parser.add_argument("--test", type=Path, help="Single-case pytest file or nodeid.")
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=None,
        help="Batch mode root. Scans case directories under experiments/oracle_generation/pandas.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/oracle_evaluation/line_branch_coverage"),
        help="Directory for batch JSON outputs.",
    )
    parser.add_argument(
        "--no-follow-super",
        action="store_true",
        help="Only measure the exact method, do not include MRO super implementations.",
    )
    parser.add_argument(
        "--no-require-core",
        action="store_true",
        help="Allow wrapper-only coverage summaries.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path for a single-case machine-readable JSON summary.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to pytest. Use '--' before them.",
    )
    args = parser.parse_args()

    if args.scan_root is None and (args.api is None or args.test is None):
        parser.error("single-case mode requires both --api and --test")
    if args.scan_root is not None and (args.api is not None or args.test is not None):
        parser.error("use either single-case mode (--api/--test) or batch mode (--scan-root)")
    return args


def _clean_pytest_args(pytest_args: list[str]) -> list[str]:
    if pytest_args and pytest_args[0] == "--":
        return pytest_args[1:]
    return pytest_args[:]


def _ensure_venv_bin_on_path() -> None:
    bin_dir = Path(sys.executable).resolve().parent
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if str(bin_dir) not in parts:
        os.environ["PATH"] = os.pathsep.join([str(bin_dir), *parts]) if parts else str(bin_dir)


def import_local_pandas(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if not (repo_root / "pandas").exists():
        raise SystemExit(f"--repo-root does not look like a pandas checkout: {repo_root}")
    _ensure_venv_bin_on_path()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    importlib.invalidate_caches()


def resolve_accessor_api(api_path: str) -> tuple[Any, type, str] | None:
    import pandas as pd

    tokens = api_path.split(".")
    if tokens[:3] == ["pandas", "Series", "str"] and len(tokens) == 4:
        accessor = pd.Series(["x"]).str
        owner = accessor.__class__
        method_name = tokens[-1]
        return getattr(owner, method_name), owner, method_name
    return None


def resolve_api(api_path: str) -> tuple[Any, type | None, str | None]:
    accessor_result = resolve_accessor_api(api_path)
    if accessor_result is not None:
        return accessor_result

    if not api_path.startswith("pandas."):
        raise SystemExit("--api must start with 'pandas.'")

    root = importlib.import_module("pandas")
    obj: Any = root
    owner_cls: type | None = None
    method_name: str | None = None

    for token in api_path.split(".")[1:]:
        prev = obj
        try:
            obj = getattr(obj, token)
        except AttributeError as exc:
            raise SystemExit(f"Could not resolve token '{token}' in '{api_path}'") from exc
        if inspect.isclass(prev):
            owner_cls = prev
            method_name = token

    return obj, owner_cls, method_name


def unwrap_callable(func: Any) -> Any:
    if isinstance(func, (FunctionType, MethodType)):
        return getattr(func, "__func__", func)
    return func


def detect_simple_delegation(func: Any) -> str | None:
    source_text = "".join(inspect.getsourcelines(func)[0])
    module_ast = ast.parse(textwrap.dedent(source_text))
    fn_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not fn_nodes:
        return None

    body = list(fn_nodes[0].body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1:
        return None

    stmt = body[0]
    value = stmt.value if isinstance(stmt, (ast.Return, ast.Expr)) else None
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
        return None

    receiver = value.func.value
    if isinstance(receiver, ast.Name) and receiver.id == "self":
        return f"self.{value.func.attr}"
    if (
        isinstance(receiver, ast.Call)
        and isinstance(receiver.func, ast.Name)
        and receiver.func.id == "super"
    ):
        return f"super().{value.func.attr}"
    return None


def resolve_method(owner: type, method_name: str) -> ResolvedMethod:
    func = unwrap_callable(getattr(owner, method_name))
    file_name = inspect.getsourcefile(func)
    if file_name is None:
        raise SystemExit(f"No source file for {owner.__name__}.{method_name}")
    source_lines, start_line = inspect.getsourcelines(func)
    end_line = start_line + len(source_lines) - 1
    return ResolvedMethod(
        owner=owner,
        method_name=method_name,
        func=func,
        file=Path(file_name).resolve(),
        start_line=start_line,
        end_line=end_line,
        delegates_to=detect_simple_delegation(func),
    )


def collect_method_chain(owner: type, method_name: str, follow_super: bool) -> list[ResolvedMethod]:
    methods: list[ResolvedMethod] = []
    seen: set[tuple[Path, int, str]] = set()

    def add_method(cls: type) -> None:
        if not hasattr(cls, method_name):
            return
        resolved = resolve_method(cls, method_name)
        key = (resolved.file, resolved.start_line, resolved.label)
        if key not in seen:
            seen.add(key)
            methods.append(resolved)

    add_method(owner)
    if follow_super:
        for base in owner.__mro__[1:]:
            if hasattr(base, method_name):
                add_method(base)

    if not methods:
        raise SystemExit(f"Could not resolve method chain for {owner.__name__}.{method_name}")

    tagged: list[ResolvedMethod] = []
    saw_core = False
    for method in methods:
        role = "wrapper" if not saw_core and method.delegates_to is not None else "core"
        if role == "core":
            saw_core = True
        tagged.append(
            ResolvedMethod(
                owner=method.owner,
                method_name=method.method_name,
                func=method.func,
                file=method.file,
                start_line=method.start_line,
                end_line=method.end_line,
                delegates_to=method.delegates_to,
                implementation_role=role,
            )
        )
    if tagged:
        last = tagged[-1]
        tagged[-1] = ResolvedMethod(
            owner=last.owner,
            method_name=last.method_name,
            func=last.func,
            file=last.file,
            start_line=last.start_line,
            end_line=last.end_line,
            delegates_to=last.delegates_to,
            implementation_role="core",
        )
    return tagged


def run_coverage(
    include_files: list[Path],
    test_target: Path,
    pytest_args: list[str],
) -> tuple[int, Any, dict[str, Any]]:
    try:
        import coverage
        import pytest
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency. Install coverage and pytest in the active environment."
        ) from exc

    cov = coverage.Coverage(branch=True, include=[str(path) for path in include_files])
    cov.start()
    exit_code = pytest.main([str(test_target), *_clean_pytest_args(pytest_args)])
    cov.stop()
    cov.save()

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cov.json_report(outfile=str(tmp_path), include=[str(path) for path in include_files])
        payload = json.loads(tmp_path.read_text())
    finally:
        tmp_path.unlink(missing_ok=True)

    return exit_code, cov, payload


def per_method_line_coverage(cov: Any, method: ResolvedMethod) -> tuple[int, int, list[int]]:
    _, statements, _, missing, _ = cov.analysis2(str(method.file))
    stmt_set = {line for line in statements if method.start_line <= line <= method.end_line}
    missing_set = {line for line in missing if method.start_line <= line <= method.end_line}

    source_text = "".join(inspect.getsourcelines(method.func)[0])
    module_ast = ast.parse(textwrap.dedent(source_text))
    fn_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if fn_nodes and fn_nodes[0].body:
        fn = fn_nodes[0]
        first_body_line = method.start_line + fn.body[0].lineno - 1
        for line in range(method.start_line, first_body_line):
            stmt_set.discard(line)
            missing_set.discard(line)
        if (
            isinstance(fn.body[0], ast.Expr)
            and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
            and isinstance(fn.body[0].value.value, str)
        ):
            doc_start = method.start_line + fn.body[0].lineno - 1
            doc_end = method.start_line + fn.body[0].end_lineno - 1
            for line in range(doc_start, doc_end + 1):
                stmt_set.discard(line)
                missing_set.discard(line)

    return len(stmt_set - missing_set), len(stmt_set), sorted(missing_set)


def per_method_branch_coverage(
    cov: Any, method: ResolvedMethod
) -> tuple[int, int, dict[int, list[int]]]:
    analysis = cov._analyze(str(method.file))
    executed = analysis.executed_branch_arcs()
    missing = analysis.missing_branch_arcs()

    covered_branches = 0
    total_branches = 0
    missing_branches: dict[int, list[int]] = {}

    for line in sorted(set(executed) | set(missing)):
        if not (method.start_line <= line <= method.end_line):
            continue
        covered_targets = sorted(executed.get(line, []))
        missing_targets = sorted(missing.get(line, []))
        covered_branches += len(covered_targets)
        total_branches += len(covered_targets) + len(missing_targets)
        if missing_targets:
            missing_branches[line] = missing_targets

    return covered_branches, total_branches, missing_branches


def build_method_summary(cov: Any, method: ResolvedMethod) -> MethodCoverageSummary:
    covered_lines, num_statements, missing_lines = per_method_line_coverage(cov, method)
    covered_branches, num_branches, missing_branches = per_method_branch_coverage(
        cov, method
    )
    return MethodCoverageSummary(
        method=method,
        covered_lines=covered_lines,
        num_statements=num_statements,
        missing_lines=missing_lines,
        covered_branches=covered_branches,
        num_branches=num_branches,
        missing_branches=missing_branches,
    )


def summarize_method_totals(
    summaries: list[MethodCoverageSummary], role: str | None = None
) -> dict[str, Any]:
    selected = [
        summary
        for summary in summaries
        if role is None or summary.method.implementation_role == role
    ]
    covered_lines = sum(item.covered_lines for item in selected)
    num_statements = sum(item.num_statements for item in selected)
    covered_branches = sum(item.covered_branches for item in selected)
    num_branches = sum(item.num_branches for item in selected)
    return {
        "covered_lines": covered_lines,
        "num_statements": num_statements,
        "line_coverage_percent": round(pct(covered_lines, num_statements), 2),
        "covered_branches": covered_branches,
        "num_branches": num_branches,
        "branch_coverage_percent": round(pct(covered_branches, num_branches), 2),
    }


def validate_local_resolution(methods: list[ResolvedMethod], repo_root: Path) -> None:
    bad_methods = [method for method in methods if not method.file.is_relative_to(repo_root.resolve())]
    if bad_methods:
        details = "\n".join(f"  - {method.label}: {method.file}" for method in bad_methods)
        raise SystemExit(
            "Resolved methods are not coming from the local pandas checkout.\n"
            f"Expected under: {repo_root.resolve()}\n{details}"
        )


def summarize_file_coverage(
    json_payload: dict[str, Any], include_files: list[Path]
) -> list[FileCoverageSummary]:
    files_blob = json_payload.get("files", {})
    summaries: list[FileCoverageSummary] = []
    cwd = Path.cwd().resolve()

    for file_path in include_files:
        abs_key = str(file_path).replace("\\", "/")
        try:
            rel_key = str(file_path.resolve().relative_to(cwd)).replace("\\", "/")
        except ValueError:
            rel_key = abs_key
        blob = files_blob.get(abs_key) or files_blob.get(rel_key)
        if blob is None:
            for key, candidate in files_blob.items():
                if str(key).replace("\\", "/").endswith(rel_key):
                    blob = candidate
                    break
        if blob is None:
            continue
        summary = blob["summary"]
        summaries.append(
            FileCoverageSummary(
                path=file_path,
                covered_lines=int(summary.get("covered_lines", 0)),
                num_statements=int(summary.get("num_statements", 0)),
                covered_branches=int(summary.get("covered_branches", 0)),
                num_branches=int(summary.get("num_branches", 0)),
            )
        )
    return summaries


def pct(a: int, b: int) -> float:
    return 100.0 if b == 0 else (a / b) * 100.0


def case_dir_to_api(case_dir: Path, scan_root: Path) -> str:
    rel = case_dir.resolve().relative_to(scan_root.resolve())
    return f"pandas.{'.'.join(rel.parts)}"


def single_case_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.runtime == "local":
        import_local_pandas(args.repo_root)

    _, owner_cls, method_name = resolve_api(args.api)
    if owner_cls is None or method_name is None:
        raise SystemExit(f"Target is not a supported method API: {args.api}")

    methods = collect_method_chain(
        owner=owner_cls,
        method_name=method_name,
        follow_super=not args.no_follow_super,
    )
    if args.runtime == "local":
        validate_local_resolution(methods, args.repo_root)
    if not args.no_require_core and not any(
        method.implementation_role == "core" for method in methods
    ):
        raise SystemExit("Could not identify a non-wrapper core implementation.")

    include_files = sorted({method.file for method in methods})

    print(f"Target API: {args.api}")
    print("Resolved method chain:")
    for method in methods:
        suffix = f", delegates_to={method.delegates_to}" if method.delegates_to else ""
        print(
            f"  - {method.label} [{method.implementation_role}] "
            f"({method.file}:{method.start_line}-{method.end_line}{suffix})"
        )

    exit_code, cov, json_payload = run_coverage(
        include_files=include_files,
        test_target=args.test,
        pytest_args=args.pytest_args,
    )

    print("\nMethod coverage:")
    method_summaries = [build_method_summary(cov, method) for method in methods]
    method_results: list[dict[str, Any]] = []
    for summary in method_summaries:
        method = summary.method
        print(
            f"  - {method.label} [{method.implementation_role}]: "
            f"lines {summary.covered_lines}/{summary.num_statements} "
            f"({pct(summary.covered_lines, summary.num_statements):.1f}%), "
            f"branches {summary.covered_branches}/{summary.num_branches} "
            f"({pct(summary.covered_branches, summary.num_branches):.1f}%), "
            f"missing_lines={summary.missing_lines}, "
            f"missing_branches={summary.missing_branches}"
        )
        method_results.append(
            {
                "label": method.label,
                "implementation_role": method.implementation_role,
                "delegates_to": method.delegates_to,
                "file": str(method.file),
                "start_line": method.start_line,
                "end_line": method.end_line,
                "covered_lines": summary.covered_lines,
                "num_statements": summary.num_statements,
                "line_coverage_percent": round(
                    pct(summary.covered_lines, summary.num_statements), 2
                ),
                "missing_lines": summary.missing_lines,
                "covered_branches": summary.covered_branches,
                "num_branches": summary.num_branches,
                "branch_coverage_percent": round(
                    pct(summary.covered_branches, summary.num_branches), 2
                ),
                "missing_branches": summary.missing_branches,
            }
        )

    print("\nFile line/branch coverage:")
    file_results: list[dict[str, Any]] = []
    file_summaries = summarize_file_coverage(json_payload, include_files)
    for summary in file_summaries:
        print(
            f"  - {summary.path}: "
            f"lines {summary.covered_lines}/{summary.num_statements} "
            f"({pct(summary.covered_lines, summary.num_statements):.1f}%), "
            f"branches {summary.covered_branches}/{summary.num_branches} "
            f"({pct(summary.covered_branches, summary.num_branches):.1f}%)"
        )
        file_results.append(
            {
                "file": str(summary.path),
                "covered_lines": summary.covered_lines,
                "num_statements": summary.num_statements,
                "line_coverage_percent": round(
                    pct(summary.covered_lines, summary.num_statements), 2
                ),
                "covered_branches": summary.covered_branches,
                "num_branches": summary.num_branches,
                "branch_coverage_percent": round(
                    pct(summary.covered_branches, summary.num_branches), 2
                ),
            }
        )

    total_covered_lines = sum(item["covered_lines"] for item in file_results)
    total_num_statements = sum(item["num_statements"] for item in file_results)
    total_covered_branches = sum(item["covered_branches"] for item in file_results)
    total_num_branches = sum(item["num_branches"] for item in file_results)

    payload = {
        "target_api": args.api,
        "test_target": str(args.test),
        "pytest_exit_code": exit_code,
        "methods": method_results,
        "files": file_results,
        "method_totals": summarize_method_totals(method_summaries),
        "core_method_totals": summarize_method_totals(method_summaries, role="core"),
        "totals": {
            "covered_lines": total_covered_lines,
            "num_statements": total_num_statements,
            "line_coverage_percent": round(
                pct(total_covered_lines, total_num_statements), 2
            ),
            "covered_branches": total_covered_branches,
            "num_branches": total_num_branches,
            "branch_coverage_percent": round(
                pct(total_covered_branches, total_num_branches), 2
            ),
        },
    }
    return payload


def sanitize_case_id(case_id: str) -> str:
    return case_id.replace("/", "__").replace(".", "_")


def discover_case_dirs(scan_root: Path) -> list[Path]:
    candidates = {
        path.parent
        for name in ("baseline_test.py", "ir_generated_test.py")
        for path in scan_root.rglob(name)
    }
    return sorted(candidates)


def compare_suite_metrics(
    baseline: dict[str, Any] | None, generated: dict[str, Any] | None
) -> dict[str, Any]:
    if baseline is None or generated is None:
        return {
            "has_both": False,
            "baseline_present": baseline is not None,
            "ir_generated_present": generated is not None,
            "better_test_method": (
                "baseline"
                if baseline is not None
                else "ir_generated"
                if generated is not None
                else "unknown"
            ),
        }

    baseline_core = baseline.get("core_method_totals", {})
    generated_core = generated.get("core_method_totals", {})
    baseline_total = baseline.get("totals", {})
    generated_total = generated.get("totals", {})
    core_line_delta = round(
        generated_core.get("line_coverage_percent", 0.0)
        - baseline_core.get("line_coverage_percent", 0.0),
        2,
    )
    core_branch_delta = round(
        generated_core.get("branch_coverage_percent", 0.0)
        - baseline_core.get("branch_coverage_percent", 0.0),
        2,
    )
    file_line_delta = round(
        generated_total.get("line_coverage_percent", 0.0)
        - baseline_total.get("line_coverage_percent", 0.0),
        2,
    )
    file_branch_delta = round(
        generated_total.get("branch_coverage_percent", 0.0)
        - baseline_total.get("branch_coverage_percent", 0.0),
        2,
    )

    if core_branch_delta > 0 or (core_branch_delta == 0 and core_line_delta > 0):
        better_test_method = "ir_generated"
    elif core_branch_delta < 0 or (core_branch_delta == 0 and core_line_delta < 0):
        better_test_method = "baseline"
    else:
        better_test_method = "tie"

    return {
        "has_both": True,
        "baseline_present": True,
        "ir_generated_present": True,
        "core_line_coverage_delta": core_line_delta,
        "core_branch_coverage_delta": core_branch_delta,
        "file_line_coverage_delta": file_line_delta,
        "file_branch_coverage_delta": file_branch_delta,
        "better_test_method": better_test_method,
    }


def aggregate_batch_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_core_lines: list[float] = []
    baseline_core_branches: list[float] = []
    generated_core_lines: list[float] = []
    generated_core_branches: list[float] = []
    better_core_line = 0
    better_core_branch = 0
    baseline_better = 0
    generated_better = 0
    ties = 0
    passing_suites = 0
    total_suites = 0

    for case in case_results:
        for suite_name in ("baseline", "ir_generated"):
            suite = case["suites"].get(suite_name)
            if suite is None:
                continue
            total_suites += 1
            if suite.get("returncode") == 0:
                passing_suites += 1

        baseline = case["suites"].get("baseline", {}).get("coverage")
        generated = case["suites"].get("ir_generated", {}).get("coverage")
        if baseline is not None:
            baseline_core_lines.append(
                baseline["core_method_totals"]["line_coverage_percent"]
            )
            baseline_core_branches.append(
                baseline["core_method_totals"]["branch_coverage_percent"]
            )
        if generated is not None:
            generated_core_lines.append(
                generated["core_method_totals"]["line_coverage_percent"]
            )
            generated_core_branches.append(
                generated["core_method_totals"]["branch_coverage_percent"]
            )
        comparison = case["comparison"]
        if comparison.get("has_both"):
            if comparison["core_line_coverage_delta"] > 0:
                better_core_line += 1
            if comparison["core_branch_coverage_delta"] > 0:
                better_core_branch += 1
        winner = comparison.get("better_test_method")
        if winner == "baseline":
            baseline_better += 1
        elif winner == "ir_generated":
            generated_better += 1
        elif winner == "tie":
            ties += 1

    overall_winner = "tie"
    if generated_better > baseline_better:
        overall_winner = "ir_generated"
    elif baseline_better > generated_better:
        overall_winner = "baseline"

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    return {
        "num_cases": len(case_results),
        "num_suites": total_suites,
        "passing_suites": passing_suites,
        "failing_suites": total_suites - passing_suites,
        "average_baseline_core_line_coverage": avg(baseline_core_lines),
        "average_baseline_core_branch_coverage": avg(baseline_core_branches),
        "average_ir_generated_core_line_coverage": avg(generated_core_lines),
        "average_ir_generated_core_branch_coverage": avg(generated_core_branches),
        "cases_where_ir_improves_core_line_coverage": better_core_line,
        "cases_where_ir_improves_core_branch_coverage": better_core_branch,
        "cases_where_baseline_is_better": baseline_better,
        "cases_where_ir_generated_is_better": generated_better,
        "cases_that_tie": ties,
        "better_test_method_overall": overall_winner,
    }


def run_subprocess_case(
    script_path: Path,
    repo_root: Path,
    runtime: str,
    api: str,
    test_file: Path,
    json_out: Path,
    no_follow_super: bool,
    no_require_core: bool,
    pytest_args: list[str],
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(script_path),
        "--repo-root",
        str(repo_root),
        "--runtime",
        runtime,
        "--api",
        api,
        "--test",
        str(test_file),
        "--json-out",
        str(json_out),
    ]
    if no_follow_super:
        cmd.append("--no-follow-super")
    if no_require_core:
        cmd.append("--no-require-core")
    cleaned_pytest_args = _clean_pytest_args(pytest_args)
    if cleaned_pytest_args:
        cmd.append("--")
        cmd.extend(cleaned_pytest_args)

    completed = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    coverage_json = None
    if json_out.exists():
        coverage_json = json.loads(json_out.read_text())

    return {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "coverage": coverage_json,
    }


def run_batch(args: argparse.Namespace) -> int:
    scan_root = args.scan_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")

    case_results: list[dict[str, Any]] = []
    script_path = Path(__file__).resolve()
    for case_dir in discover_case_dirs(scan_root):
        case_id = str(case_dir.relative_to(scan_root))
        api = case_dir_to_api(case_dir, scan_root)
        suites: dict[str, Any] = {}
        for suite_name, file_name in (
            ("baseline", "baseline_test.py"),
            ("ir_generated", "ir_generated_test.py"),
        ):
            test_file = case_dir / file_name
            if not test_file.exists():
                suites[suite_name] = None
                continue
            with tempfile.NamedTemporaryFile(
                "w+",
                suffix=f"__{sanitize_case_id(case_id)}__{suite_name}.json",
                delete=False,
            ) as tmp:
                json_out = Path(tmp.name)
            suites[suite_name] = run_subprocess_case(
                script_path=script_path,
                repo_root=args.repo_root.resolve(),
                runtime=args.runtime,
                api=api,
                test_file=test_file,
                json_out=json_out,
                no_follow_super=args.no_follow_super,
                no_require_core=args.no_require_core,
                pytest_args=args.pytest_args,
            )
            json_out.unlink(missing_ok=True)
            print(
                f"{case_id} {suite_name}: returncode={suites[suite_name]['returncode']} "
                f"winner_data={'yes' if suites[suite_name]['coverage'] is not None else 'no'}"
            )

        baseline_coverage = (
            suites["baseline"]["coverage"] if suites.get("baseline") else None
        )
        generated_coverage = (
            suites["ir_generated"]["coverage"] if suites.get("ir_generated") else None
        )
        case_results.append(
            {
                "case_id": case_id,
                "target_api": api,
                "case_dir": str(case_dir),
                "suites": suites,
                "comparison": compare_suite_metrics(
                    baseline_coverage,
                    generated_coverage,
                ),
            }
        )

    summary = {
        "report_type": "pandas_line_branch_coverage_batch",
        "generated_at": datetime.now().astimezone().isoformat(),
        "scan_root": str(scan_root),
        "repo_root": str(args.repo_root.resolve()),
        "runtime": args.runtime,
        "pytest_args": _clean_pytest_args(args.pytest_args),
        "totals": aggregate_batch_summary(case_results),
        "case_details": case_results,
    }
    report_path = output_dir / f"pandas_line_branch_coverage_report_{timestamp}.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote batch report to {report_path}")

    return 0 if summary["totals"]["failing_suites"] == 0 else 1


def main() -> int:
    args = parse_args()
    if args.scan_root is not None:
        return run_batch(args)

    payload = single_case_payload(args)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON summary to {args.json_out}")
    print(f"\npytest exit code: {payload['pytest_exit_code']}")
    return int(payload["pytest_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
