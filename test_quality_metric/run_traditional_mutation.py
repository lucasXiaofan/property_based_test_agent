"""
run_traditional_mutation.py — Generic AST-based traditional mutation testing.

Generates mutations from any Python source file using classical mutation
operators (similar to mutmut), then runs a pytest test file against each
mutation to determine kill / survive status.

No LLM, no manual work — mutations are enumerated purely from the AST.

USAGE
-----
# Any pure-Python function — no wrapper needed:
python test_quality_metric/run_traditional_mutation.py \\
    --source-file   .venv/lib/python3.11/site-packages/pandas/core/generic.py \\
    --function-name reindex \\
    --test-file     ir2test_pipeline/pandas/DataFrame/reindex/baseline_test.py \\
    --output-dir    ir2test_pipeline/pandas/DataFrame/reindex/trad_mutant_results/ \\
    [--max-mutants 100]   # optional cap (default: unlimited)
    [--timeout 120]       # per-mutant wall-clock limit in seconds (default: 120)
    [-q / --quiet]        # suppress per-mutant subprocess output

# Wrapper approach (still works, for backward compatibility or C-extension functions):
python test_quality_metric/run_traditional_mutation.py \\
    --source-file  ir2test_pipeline/pandas/DataFrame/reindex/reindex_wrapper.py \\
    --test-file    ir2test_pipeline/pandas/DataFrame/reindex/baseline_test.py \\
    --output-dir   ir2test_pipeline/pandas/DataFrame/reindex/trad_mutant_results/

MECHANISM
---------
1. Parse source file with libcst
2. Walk the CST with MutationCollector to enumerate mutation sites
   - If --function-name is given, only sites inside that function are collected
3. For each site, apply SingleMutationApplier to get mutated source string
4. Write mutated source to disk (source file is overwritten; original held in memory)
5. Launch subprocess that imports the mutated file, calls install() if present,
   then runs pytest — tests use `import pandas` as normal
6. Restore original source from memory backup
7. Compare subprocess result to baseline → killed / survived
8. Write kill_report.json + kill_report.md to --output-dir

WRAPPER SUPPORT
---------------
For C-extension functions that have no pure-Python source (rare — most pandas
high-level API is pure Python), use a Python wrapper with an install() function
that monkey-patches the library. This script auto-detects install() and calls it.
For pure-Python pandas functions, point --source-file at the installed .py file
and use --function-name to scope mutations to just the target function.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import libcst as cst

# ─────────────────────────────────────────────────────────────────────────────
# Mutation site descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MutationSite:
    index: int                     # sequential id within this source file
    operator: str                  # e.g. "operator_swap_op"
    description: str               # human-readable change
    original_node: cst.CSTNode
    mutated_node: cst.CSTNode
    # identity key for SingleMutationApplier: match by object id
    node_id: int = field(init=False)

    def __post_init__(self) -> None:
        self.node_id = id(self.original_node)

    @property
    def mutant_id(self) -> str:
        return f"M{self.index:04d}"


# ─────────────────────────────────────────────────────────────────────────────
# Mutation collector — walks CST and records (original, mutated) pairs
# ─────────────────────────────────────────────────────────────────────────────

class MutationCollector(cst.CSTVisitor):
    """Enumerate every mutation site in the source tree.

    If function_name is given, only collect sites inside that function's body.
    Nested helper functions defined inside the target function are included.
    """

    def __init__(self, function_name: str | None = None) -> None:
        self._target_fn = function_name
        self._depth = 0   # nesting depth inside the target function
        self.sites: list[MutationSite] = []

    # ── scope tracking ────────────────────────────────────────────────────────
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        if self._target_fn and node.name.value == self._target_fn:
            self._depth += 1

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        if self._target_fn and node.name.value == self._target_fn:
            self._depth -= 1

    def _in_scope(self) -> bool:
        """True when we should record a mutation site."""
        return self._target_fn is None or self._depth > 0

    def _add(self, operator: str, description: str, orig: cst.CSTNode, mut: cst.CSTNode) -> None:
        if not self._in_scope():
            return
        idx = len(self.sites)
        self.sites.append(MutationSite(idx, operator, description, orig, mut))

    # ── operator_number: integer/float literal +1 ─────────────────────────────
    def visit_Integer(self, node: cst.Integer) -> None:
        try:
            val = int(node.value, 0)
        except ValueError:
            return
        self._add(
            "operator_number",
            f"int {node.value} → {val + 1}",
            node,
            node.with_changes(value=str(val + 1)),
        )

    def visit_Float(self, node: cst.Float) -> None:
        try:
            val = float(node.value)
        except ValueError:
            return
        self._add(
            "operator_number",
            f"float {node.value} → {val + 1.0}",
            node,
            node.with_changes(value=repr(val + 1.0)),
        )

    # ── operator_string: string literal mutation ───────────────────────────────
    def visit_SimpleString(self, node: cst.SimpleString) -> None:
        # Skip docstrings that are the first statement of a module/class/func
        raw = node.value
        # Only mutate short non-docstring strings (heuristic: no newlines)
        try:
            inner = _extract_string_value(raw)
        except ValueError:
            return
        if not inner or "\n" in inner:
            return
        mutated_inner = f"XX{inner}XX"
        new_raw = _replace_string_value(raw, mutated_inner)
        self._add(
            "operator_string",
            f"str {raw!r} → 'XX...XX' variant",
            node,
            node.with_changes(value=new_raw),
        )

    # ── operator_name: True↔False, deepcopy→copy ──────────────────────────────
    def visit_Name(self, node: cst.Name) -> None:
        swaps = {"True": "False", "False": "True"}
        if node.value in swaps:
            self._add(
                "operator_name",
                f"{node.value} → {swaps[node.value]}",
                node,
                node.with_changes(value=swaps[node.value]),
            )

    # ── operator_assignment: a = b → a = None ─────────────────────────────────
    def visit_Assign(self, node: cst.Assign) -> None:
        # Only mutate simple RHS that is not None/True/False
        if isinstance(node.value, (cst.Name,)) and node.value.value in ("None", "True", "False"):
            return
        if isinstance(node.value, cst.Name):
            self._add(
                "operator_assignment",
                f"a = {node.value.value} → a = None",
                node,
                node.with_changes(value=cst.Name("None")),
            )

    # ── operator_swap_op: relational / arithmetic / logical operator swap ──────
    _COMPARISON_SWAPS: dict[str, type[cst.BaseCompOp]] = {
        "==": cst.NotEqual,
        "!=": cst.Equal,
        "<":  cst.GreaterThan,
        ">":  cst.LessThan,
        "<=": cst.GreaterThanEqual,
        ">=": cst.LessThanEqual,
    }
    _ARITH_SWAPS: dict[type, type] = {
        cst.Add:      cst.Subtract,
        cst.Subtract: cst.Add,
        cst.Multiply: cst.Divide,
        cst.Divide:   cst.Multiply,
    }

    def visit_Comparison(self, node: cst.Comparison) -> None:
        for comp in node.comparisons:
            op = comp.operator
            op_str = _comp_op_to_str(op)
            if op_str in self._COMPARISON_SWAPS:
                new_op_cls = self._COMPARISON_SWAPS[op_str]
                new_op = new_op_cls(
                    whitespace_before=op.whitespace_before,
                    whitespace_after=op.whitespace_after,
                )
                new_comp = comp.with_changes(operator=new_op)
                new_comparisons = tuple(
                    new_comp if c is comp else c for c in node.comparisons
                )
                self._add(
                    "operator_swap_op",
                    f"comparison {op_str} → {_comp_op_to_str(new_op)}",
                    node,
                    node.with_changes(comparisons=new_comparisons),
                )

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> None:
        op = node.operator
        if type(op) in self._ARITH_SWAPS:
            new_op_cls = self._ARITH_SWAPS[type(op)]
            new_op = new_op_cls(
                whitespace_before=op.whitespace_before,
                whitespace_after=op.whitespace_after,
            )
            self._add(
                "operator_swap_op",
                f"arith {type(op).__name__} → {new_op_cls.__name__}",
                node,
                node.with_changes(operator=new_op),
            )

    def visit_BooleanOperation(self, node: cst.BooleanOperation) -> None:
        op = node.operator
        if isinstance(op, cst.And):
            new_op = cst.Or(
                whitespace_before=op.whitespace_before,
                whitespace_after=op.whitespace_after,
            )
            self._add(
                "operator_swap_op",
                "and → or",
                node,
                node.with_changes(operator=new_op),
            )
        elif isinstance(op, cst.Or):
            new_op = cst.And(
                whitespace_before=op.whitespace_before,
                whitespace_after=op.whitespace_after,
            )
            self._add(
                "operator_swap_op",
                "or → and",
                node,
                node.with_changes(operator=new_op),
            )

    # ── operator_remove_unary_ops: not x → x ──────────────────────────────────
    def visit_UnaryOperation(self, node: cst.UnaryOperation) -> None:
        if isinstance(node.operator, cst.Not):
            self._add(
                "operator_remove_unary_ops",
                "remove 'not'",
                node,
                node.expression,  # replace with unwrapped expression
            )

    # ── operator_keywords: is/is not, in/not in, break/continue ──────────────
    def visit_ComparisonTarget(self, node: cst.ComparisonTarget) -> None:
        op = node.operator
        if isinstance(op, cst.Is):
            new_op = cst.IsNot(
                whitespace_before=op.whitespace_before,
                whitespace_after=op.whitespace_after,
            )
            self._add(
                "operator_keywords",
                "is → is not",
                node,
                node.with_changes(operator=new_op),
            )
        elif isinstance(op, cst.IsNot):
            new_op = cst.Is(
                whitespace_before=op.whitespace_before,
                whitespace_after=op.whitespace_after,
            )
            self._add(
                "operator_keywords",
                "is not → is",
                node,
                node.with_changes(operator=new_op),
            )
        elif isinstance(op, cst.In):
            new_op = cst.NotIn(
                whitespace_before=op.whitespace_before,
                whitespace_after=op.whitespace_after,
            )
            self._add(
                "operator_keywords",
                "in → not in",
                node,
                node.with_changes(operator=new_op),
            )
        elif isinstance(op, cst.NotIn):
            new_op = cst.In(
                whitespace_before=op.whitespace_before,
                whitespace_after=op.whitespace_after,
            )
            self._add(
                "operator_keywords",
                "not in → in",
                node,
                node.with_changes(operator=new_op),
            )

    # ── operator_augmented_assignment: a += b → a = b ─────────────────────────
    def visit_AugAssign(self, node: cst.AugAssign) -> None:
        # Convert a += b  →  a = b
        # AugAssign target is an Attribute or Name; value is the RHS
        self._add(
            "operator_augmented_assignment",
            f"augmented assignment → simple assignment",
            node,
            cst.Assign(
                targets=[cst.AssignTarget(target=node.target)],
                value=node.value,
                semicolon=cst.MaybeSentinel.DEFAULT,
            ),
        )

    # ── operator_arg_removal: f(a,b) → f(None,b), f(b), etc. ─────────────────
    def visit_Call(self, node: cst.Call) -> None:
        args = node.args
        if len(args) < 2:
            return
        # Replace each positional arg with None (one mutation per arg)
        for i, arg in enumerate(args):
            if arg.keyword is not None:
                continue  # skip keyword args — too noisy
            new_args = list(args)
            new_args[i] = arg.with_changes(value=cst.Name("None"))
            self._add(
                "operator_arg_removal",
                f"arg[{i}] → None",
                node,
                node.with_changes(args=tuple(new_args)),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Single mutation applier — replaces exactly one node by object identity
# ─────────────────────────────────────────────────────────────────────────────

class SingleMutationApplier(cst.CSTTransformer):
    """Replace exactly one target node in the tree by object identity."""

    def __init__(self, site: MutationSite) -> None:
        super().__init__()
        self._target_id = site.node_id
        self._mutated_node = site.mutated_node
        self._applied = False

    def on_leave(self, original_node: cst.CSTNode, updated_node: cst.CSTNode) -> cst.CSTNode:
        if id(original_node) == self._target_id and not self._applied:
            self._applied = True
            return self._mutated_node
        return updated_node


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _comp_op_to_str(op: cst.BaseCompOp) -> str:
    mapping = {
        cst.Equal: "==",
        cst.NotEqual: "!=",
        cst.LessThan: "<",
        cst.GreaterThan: ">",
        cst.LessThanEqual: "<=",
        cst.GreaterThanEqual: ">=",
        cst.Is: "is",
        cst.IsNot: "is not",
        cst.In: "in",
        cst.NotIn: "not in",
    }
    return mapping.get(type(op), type(op).__name__)


def _extract_string_value(raw: str) -> str:
    """Extract the string content from a SimpleString literal token."""
    # Handle common quote styles: 'x', "x", '''x''', """x"""
    for quote in ('"""', "'''", '"', "'"):
        if raw.startswith(quote) and raw.endswith(quote) and len(raw) > 2 * len(quote):
            return raw[len(quote):-len(quote)]
    raise ValueError(f"Cannot extract string value from {raw!r}")


def _replace_string_value(raw: str, new_inner: str) -> str:
    """Replace the inner content of a SimpleString literal token."""
    for quote in ('"""', "'''", '"', "'"):
        if raw.startswith(quote) and raw.endswith(quote) and len(raw) > 2 * len(quote):
            return f"{quote}{new_inner}{quote}"
    raise ValueError(f"Cannot replace string value in {raw!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Core: collect sites and generate mutated source strings
# ─────────────────────────────────────────────────────────────────────────────

def collect_mutation_sites(
    source: str,
    function_name: str | None = None,
) -> tuple[cst.Module, list[MutationSite]]:
    """Parse source and return (tree, list of mutation sites).

    If function_name is given, only sites inside that function are returned.
    """
    tree = cst.parse_module(source)
    collector = MutationCollector(function_name=function_name)
    tree.visit(collector)
    return tree, collector.sites


def apply_mutation(tree: cst.Module, site: MutationSite) -> str:
    """Apply a single mutation to the tree and return the mutated source string."""
    applier = SingleMutationApplier(site)
    mutated_tree = tree.visit(applier)
    return mutated_tree.code


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess launcher (inline string — avoids writing temp files)
# ─────────────────────────────────────────────────────────────────────────────

_LAUNCHER = r"""
import importlib.util
import json as _json
import sys

source_file = sys.argv[1]
test_file   = sys.argv[2]
extra_args  = _json.loads(sys.argv[3]) if len(sys.argv) > 3 else ["-q"]

spec = importlib.util.spec_from_file_location("_mutated_src", source_file)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load mutated source: {source_file}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if hasattr(mod, "install"):
    mod.install()

import pytest
raise SystemExit(pytest.main([test_file] + extra_args))
"""

FAILED_LINE_RE = re.compile(r"^FAILED\s+([^\s]+)(?:\s+-\s+.+)?$", re.MULTILINE)


def _run_test(
    source_file: str,
    test_file: str,
    extra_args: list[str],
    timeout: int,
) -> tuple[int, str, str]:
    """Run pytest against source_file via the inline launcher.

    Returns (returncode, stdout, stderr).
    """
    cmd = [
        sys.executable, "-c", _LAUNCHER,
        source_file,
        test_file,
        json.dumps(extra_args),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"


def _extract_failed_nodeids(stdout: str, stderr: str) -> set[str]:
    text = f"{stdout}\n{stderr}"
    return set(FAILED_LINE_RE.findall(text))


# ─────────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────────

def run(
    source_file: Path,
    test_file: Path,
    output_dir: Path,
    max_mutants: int | None,
    timeout: int,
    quiet: bool,
    function_name: str | None = None,
) -> int:
    # ── read original source ──────────────────────────────────────────────────
    original_source = source_file.read_text(encoding="utf-8")

    # ── collect mutation sites ────────────────────────────────────────────────
    print(f"[run_traditional_mutation] source: {source_file}")
    if function_name:
        print(f"[run_traditional_mutation] function: {function_name}")
    print(f"[run_traditional_mutation] test:   {test_file}")
    try:
        tree, sites = collect_mutation_sites(original_source, function_name=function_name)
    except cst.ParserSyntaxError as exc:
        print(f"[run_traditional_mutation] ERROR: cannot parse source: {exc}", file=sys.stderr)
        return 1

    if max_mutants is not None and len(sites) > max_mutants:
        print(
            f"[run_traditional_mutation] collected {len(sites)} sites, "
            f"capping at {max_mutants}"
        )
        sites = sites[:max_mutants]
    else:
        print(f"[run_traditional_mutation] collected {len(sites)} mutation sites")

    extra_args = ["-q", "--tb=no"] if quiet else ["-v", "--tb=short"]

    # ── baseline run ──────────────────────────────────────────────────────────
    print("[run_traditional_mutation] running baseline …")
    base_rc, base_out, base_err = _run_test(
        str(source_file), str(test_file),
        ["--tb=no", "-rxf"],
        timeout,
    )
    baseline_passed = base_rc == 0
    baseline_failed = _extract_failed_nodeids(base_out, base_err)
    print(
        f"[run_traditional_mutation] baseline: {'PASS' if baseline_passed else 'FAIL'}"
        + (f" ({len(baseline_failed)} failing)" if baseline_failed else "")
    )

    results: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_file),
        "test_file": str(test_file),
        "mutants": [
            {
                "mutant_id": s.mutant_id,
                "operator": s.operator,
                "description": s.description,
            }
            for s in sites
        ],
        "baseline": {
            "passed": baseline_passed,
            "returncode": base_rc,
            "failed_nodeids": sorted(baseline_failed),
            "stdout_tail": base_out[-2000:],
            "stderr_tail": base_err[-2000:],
        },
        "runs": [],
    }

    # ── mutant loop ───────────────────────────────────────────────────────────
    killed = survived = timed_out = 0

    for site in sites:
        # generate mutated source
        try:
            mutated_source = apply_mutation(tree, site)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[run_traditional_mutation] {site.mutant_id} ({site.operator}): "
                f"ERROR generating mutation — {exc}"
            )
            results["runs"].append({
                "mutant_id": site.mutant_id,
                "operator": site.operator,
                "description": site.description,
                "status": "error",
                "error": str(exc),
            })
            continue

        # overwrite source file with mutation
        source_file.write_text(mutated_source, encoding="utf-8")

        try:
            rc, stdout, stderr = _run_test(
                str(source_file), str(test_file), extra_args, timeout
            )
        finally:
            # always restore original
            source_file.write_text(original_source, encoding="utf-8")

        if rc == -1:
            # timeout
            status = "timeout"
            timed_out += 1
        elif baseline_passed:
            status = "killed" if rc != 0 else "survived"
        else:
            # baseline already failing: check for NEW failures
            mutant_failed = _extract_failed_nodeids(stdout, stderr)
            new_failed = mutant_failed - baseline_failed
            status = "killed" if new_failed else "survived"

        if status == "killed":
            killed += 1
        elif status == "survived":
            survived += 1

        print(
            f"[run_traditional_mutation] {site.mutant_id} "
            f"({site.operator}): {status} | {site.description[:60]}"
        )
        if not quiet:
            if stdout.strip():
                print(textwrap.indent(stdout[-500:], "    "))

        results["runs"].append({
            "mutant_id": site.mutant_id,
            "operator": site.operator,
            "description": site.description,
            "status": status,
            "returncode": rc,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        })

    # ── summary ───────────────────────────────────────────────────────────────
    denom = killed + survived
    score = (killed / denom) if denom else 0.0
    results["summary"] = {
        "total_sites": len(sites),
        "killed": killed,
        "survived": survived,
        "timed_out": timed_out,
        "mutation_score": score,
    }

    # ── write outputs ─────────────────────────────────────────────────────────
    test_name = test_file.stem
    out_dir = output_dir / test_name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "kill_report.json"
    md_path   = out_dir / "kill_report.md"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # markdown report
    md_lines = [
        f"# Traditional Mutation Kill Report",
        "",
        f"- **source_file**: `{source_file}`",
        f"- **test_file**: `{test_file}`",
        f"- **generated_at**: {results['generated_at_utc']}",
        f"- **baseline_passed**: {baseline_passed}",
        f"- **total_sites**: {len(sites)}",
        f"- **killed**: {killed}",
        f"- **survived**: {survived}",
        f"- **timed_out**: {timed_out}",
        f"- **mutation_score**: {score:.3f}",
        "",
        "## Mutant Results",
        "",
        "| Mutant ID | Operator | Status | Description |",
        "|-----------|----------|--------|-------------|",
    ]
    for run in results["runs"]:
        status_icon = {"killed": "✅", "survived": "❌", "timeout": "⏱️", "error": "💥"}.get(
            run["status"], "?"
        )
        desc = run["description"][:80]
        md_lines.append(
            f"| `{run['mutant_id']}` | {run['operator']} | {status_icon} {run['status']} | {desc} |"
        )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(
        f"\n[run_traditional_mutation] summary: "
        f"killed={killed}, survived={survived}, timeout={timed_out}, "
        f"score={score:.3f}"
    )
    print(f"[run_traditional_mutation] wrote: {json_path}")
    print(f"[run_traditional_mutation] wrote: {md_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-file", required=True,
        help="Python source file to mutate (e.g. reindex_wrapper.py)",
    )
    parser.add_argument(
        "--test-file", required=True,
        help="pytest test file to run against each mutant",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write kill_report.json and kill_report.md",
    )
    parser.add_argument(
        "--max-mutants", type=int, default=None,
        help="Cap on the number of mutants to evaluate (default: unlimited)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Per-mutant wall-clock time limit in seconds (default: 120)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress per-mutant subprocess output",
    )
    parser.add_argument(
        "--function-name", default=None,
        help=(
            "Only mutate inside this function (or method) name. "
            "Use this to scope mutations when --source-file is a large library file "
            "(e.g. pandas/core/generic.py). "
            "Example: --function-name reindex"
        ),
    )
    args = parser.parse_args()

    source_file = Path(args.source_file).resolve()
    test_file   = Path(args.test_file).resolve()
    output_dir  = Path(args.output_dir).resolve()

    if not source_file.exists():
        print(f"ERROR: source-file not found: {source_file}", file=sys.stderr)
        return 1
    if not test_file.exists():
        print(f"ERROR: test-file not found: {test_file}", file=sys.stderr)
        return 1

    return run(
        source_file=source_file,
        test_file=test_file,
        output_dir=output_dir,
        max_mutants=args.max_mutants,
        timeout=args.timeout,
        quiet=args.quiet,
        function_name=args.function_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
