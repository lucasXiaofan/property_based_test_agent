#!/usr/bin/env python3
"""Run line/branch coverage for a target pandas API against a pytest test file.

Example:
    uv run python pandas_bug_finding/line_branch_coverage/run_api_coverage.py \
      --repo-root pandas_bug_finding/pandas \
      --api pandas.DataFrame.reindex \
      --test pandas_bug_finding/baseline_testing/test_reindex_hypothesis.py \
      -- -q
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
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

    @property
    def label(self) -> str:
        return f"{self.owner.__module__}.{self.owner.__name__}.{self.method_name}"


@dataclass
class FileCoverageSummary:
    path: Path
    covered_lines: int
    num_statements: int
    covered_branches: int
    num_branches: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Optional local pandas checkout path (e.g. pandas_bug_finding/pandas). "
            "Used for reference only; runtime imports use installed pandas by default."
        ),
    )
    parser.add_argument(
        "--runtime",
        choices=("installed", "local"),
        default="installed",
        help=(
            "Where to import pandas from. 'installed' is recommended unless your "
            "local checkout is built with compiled extensions."
        ),
    )
    parser.add_argument(
        "--api",
        required=True,
        help="Dotted API path, e.g. pandas.DataFrame.reindex",
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help="Pytest file or nodeid to execute.",
    )
    parser.add_argument(
        "--no-follow-super",
        action="store_true",
        help="Only measure the exact method, do not include MRO super implementations.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a machine-readable JSON summary.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to pytest. Use '--' before them.",
    )
    return parser.parse_args()


def import_local_pandas(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if not (repo_root / "pandas").exists():
        raise SystemExit(f"--repo-root does not look like a pandas checkout: {repo_root}")

    sys.path.insert(0, str(repo_root))
    importlib.invalidate_caches()


def resolve_api(api_path: str) -> tuple[Any, type | None, str | None]:
    if not api_path.startswith("pandas"):
        raise SystemExit("--api must start with 'pandas'")

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


def resolve_method(owner: type, method_name: str) -> ResolvedMethod:
    func = unwrap_callable(owner.__dict__[method_name])
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
    )


def collect_method_chain(owner: type, method_name: str, follow_super: bool) -> list[ResolvedMethod]:
    methods: list[ResolvedMethod] = []
    seen: set[tuple[Path, int, str]] = set()

    def add_method(cls: type) -> None:
        if method_name not in cls.__dict__:
            return
        resolved = resolve_method(cls, method_name)
        key = (resolved.file, resolved.start_line, resolved.label)
        if key not in seen:
            seen.add(key)
            methods.append(resolved)

    add_method(owner)
    if follow_super:
        for base in owner.__mro__[1:]:
            add_method(base)

    if not methods:
        raise SystemExit(f"Could not resolve method chain for {owner.__name__}.{method_name}")
    return methods


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
            "Missing dependency. Install with: uv add coverage\n"
            "Then rerun this script."
        ) from exc

    cov = coverage.Coverage(branch=True, include=[str(p) for p in include_files])
    cov.start()

    args = [str(test_target)]
    cleaned_extra = pytest_args[:]
    if cleaned_extra and cleaned_extra[0] == "--":
        cleaned_extra = cleaned_extra[1:]
    args.extend(cleaned_extra)

    exit_code = pytest.main(args)

    cov.stop()
    cov.save()

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cov.json_report(outfile=str(tmp_path), include=[str(p) for p in include_files])
        payload = json.loads(tmp_path.read_text())
    finally:
        tmp_path.unlink(missing_ok=True)

    return exit_code, cov, payload


def per_method_line_coverage(cov: Any, method: ResolvedMethod) -> tuple[int, int, list[int]]:
    _, statements, _, missing, _ = cov.analysis2(str(method.file))
    stmt_set = {ln for ln in statements if method.start_line <= ln <= method.end_line}
    missing_set = {ln for ln in missing if method.start_line <= ln <= method.end_line}

    # Focus on executable method body lines:
    # exclude decorators/signature lines and the method docstring block.
    source_text = "".join(inspect.getsourcelines(method.func)[0])
    module_ast = ast.parse(textwrap.dedent(source_text))
    fn_nodes = [
        n
        for n in module_ast.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if fn_nodes:
        fn = fn_nodes[0]
        if fn.body:
            first_body_line = method.start_line + fn.body[0].lineno - 1
            def_line_end = first_body_line - 1
            for ln in range(method.start_line, def_line_end + 1):
                stmt_set.discard(ln)
                missing_set.discard(ln)

            if (
                isinstance(fn.body[0], ast.Expr)
                and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
                and isinstance(fn.body[0].value.value, str)
            ):
                doc_start = method.start_line + fn.body[0].lineno - 1
                doc_end = method.start_line + fn.body[0].end_lineno - 1
                for ln in range(doc_start, doc_end + 1):
                    stmt_set.discard(ln)
                    missing_set.discard(ln)

    covered = len(stmt_set - missing_set)
    total = len(stmt_set)
    return covered, total, sorted(missing_set)


def summarize_file_coverage(json_payload: dict[str, Any], include_files: list[Path]) -> list[FileCoverageSummary]:
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
            # coverage may key files as relative paths; use suffix match as fallback
            suffix = rel_key
            for key, candidate in files_blob.items():
                norm_key = str(key).replace("\\", "/")
                if norm_key.endswith(suffix):
                    blob = candidate
                    break
        if blob is None:
            continue
        s = blob["summary"]
        summaries.append(
            FileCoverageSummary(
                path=file_path,
                covered_lines=int(s.get("covered_lines", 0)),
                num_statements=int(s.get("num_statements", 0)),
                covered_branches=int(s.get("covered_branches", 0)),
                num_branches=int(s.get("num_branches", 0)),
            )
        )

    return summaries


def pct(a: int, b: int) -> float:
    if b == 0:
        return 100.0
    return (a / b) * 100.0


def main() -> int:
    args = parse_args()
    if args.runtime == "local":
        if args.repo_root is None:
            raise SystemExit("--runtime=local requires --repo-root")
        import_local_pandas(args.repo_root)

    resolved_obj, owner_cls, method_name = resolve_api(args.api)
    if owner_cls is None or method_name is None:
        raise SystemExit(
            "Target is not a class method. Pass something like pandas.DataFrame.reindex"
        )

    _ = resolved_obj  # resolved for validation; method chain is taken from class MRO.
    methods = collect_method_chain(
        owner=owner_cls,
        method_name=method_name,
        follow_super=not args.no_follow_super,
    )

    local_methods = methods
    if args.repo_root is not None:
        repo_root = args.repo_root.resolve()
        in_repo = [m for m in methods if str(m.file).startswith(str(repo_root))]
        if not in_repo:
            print(
                "Warning: resolved methods are not from --repo-root. "
                "This is expected when using --runtime=installed."
            )

    include_files = sorted({m.file for m in local_methods})

    print(f"Target API: {args.api}")
    print("Resolved method chain:")
    for m in local_methods:
        print(f"  - {m.label} ({m.file}:{m.start_line}-{m.end_line})")

    exit_code, cov, json_payload = run_coverage(
        include_files=include_files,
        test_target=args.test,
        pytest_args=args.pytest_args,
    )

    print("\nMethod line coverage:")
    method_results: list[dict[str, Any]] = []
    for m in local_methods:
        covered, total, missing = per_method_line_coverage(cov, m)
        print(
            f"  - {m.label}: {covered}/{total} lines "
            f"({pct(covered, total):.1f}%), missing={missing}"
        )
        method_results.append(
            {
                "label": m.label,
                "file": str(m.file),
                "start_line": m.start_line,
                "end_line": m.end_line,
                "covered_lines": covered,
                "num_statements": total,
                "line_coverage_percent": round(pct(covered, total), 2),
                "missing_lines": missing,
            }
        )

    file_summaries = summarize_file_coverage(json_payload, include_files)
    print("\nFile line/branch coverage:")
    file_results: list[dict[str, Any]] = []
    for s in file_summaries:
        print(
            f"  - {s.path}: "
            f"lines {s.covered_lines}/{s.num_statements} ({pct(s.covered_lines, s.num_statements):.1f}%), "
            f"branches {s.covered_branches}/{s.num_branches} ({pct(s.covered_branches, s.num_branches):.1f}%)"
        )
        file_results.append(
            {
                "file": str(s.path),
                "covered_lines": s.covered_lines,
                "num_statements": s.num_statements,
                "line_coverage_percent": round(
                    pct(s.covered_lines, s.num_statements), 2
                ),
                "covered_branches": s.covered_branches,
                "num_branches": s.num_branches,
                "branch_coverage_percent": round(
                    pct(s.covered_branches, s.num_branches), 2
                ),
            }
        )

    if args.json_out is not None:
        total_covered_lines = sum(item["covered_lines"] for item in file_results)
        total_num_statements = sum(item["num_statements"] for item in file_results)
        total_covered_branches = sum(item["covered_branches"] for item in file_results)
        total_num_branches = sum(item["num_branches"] for item in file_results)
        output = {
            "target_api": args.api,
            "pytest_exit_code": exit_code,
            "methods": method_results,
            "files": file_results,
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
        args.json_out.write_text(json.dumps(output, indent=2))
        print(f"\nWrote JSON summary to {args.json_out}")

    print(f"\npytest exit code: {exit_code}")
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
