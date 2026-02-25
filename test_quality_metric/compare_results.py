#!/usr/bin/env python3
"""
compare_results.py  —  Side-by-side quality comparison between two test suites.

Reads per-suite artefacts from:
    <target-dir>/results/<suite>/kill_report.json
    <target-dir>/results/<suite>/coverage.json
    <target-dir>/results/<suite>/properties_coverage.json

Reads shared clause metadata from:
    <target-dir>/clauses.json

Writes:
    <target-dir>/results/comparison_report.md

Usage
-----
    # Compare ALL eligible function dirs under test_quality_metric/ (default)
    uv run python test_quality_metric/compare_results.py

    # Single function
    uv run python test_quality_metric/compare_results.py \\
        --target-dir test_quality_metric/pandas/DataFrame/reindex

    # Override which two suites to compare
    uv run python test_quality_metric/compare_results.py \\
        --target-dir test_quality_metric/pandas/DataFrame/reindex \\
        --suite-a baseline_test --suite-b ir_generated_test

A function directory is "eligible" when its results/ subfolder contains at least
two non-'old*' suite directories each having kill_report.json, coverage.json,
and properties_coverage.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | list:
    with open(path) as fh:
        return json.load(fh)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def delta_str(a: float, b: float, *, as_pct: bool = False) -> str:
    """Return a signed delta string, e.g. '+0.15' or '-3.2%'."""
    d = b - a
    if as_pct:
        return f"{d * 100:+.1f}pp"
    return f"{d:+.3f}"


def winner_arrow(a: float, b: float, *, higher_is_better: bool = True) -> str:
    if abs(a - b) < 1e-9:
        return "—"
    better = b > a if higher_is_better else b < a
    return "**B**" if better else "**A**"


def short_name(nodeid: str) -> str:
    """Strip the file path prefix from a pytest node id."""
    return nodeid.split("::")[-1]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_suite(results_dir: Path, suite: str) -> dict:
    suite_dir = results_dir / suite
    kill = load_json(suite_dir / "kill_report.json")
    cov = load_json(suite_dir / "coverage.json")
    props = load_json(suite_dir / "properties_coverage.json")
    return {"kill": kill, "coverage": cov, "props": props, "name": suite}


def load_clauses(target_dir: Path) -> dict[str, dict]:
    path = target_dir / "clauses.json"
    if not path.exists():
        return {}
    clauses = load_json(path)
    return {c["id"]: c for c in clauses}


def discover_suites(results_dir: Path) -> list[str]:
    """Return result sub-directory names, excluding 'old*' prefixed ones."""
    return sorted(
        d.name
        for d in results_dir.iterdir()
        if d.is_dir() and not d.name.startswith("old")
    )


_REQUIRED_FILES = ("kill_report.json", "coverage.json", "properties_coverage.json")


def suite_is_complete(suite_dir: Path) -> bool:
    return all((suite_dir / f).exists() for f in _REQUIRED_FILES)


def discover_target_dirs(root: Path) -> list[Path]:
    """Walk *root* and return every function directory that is eligible for comparison.

    A directory is eligible when:
    - it has a results/ subdirectory
    - that results/ contains >= 2 non-'old*' suite dirs
    - each of those suite dirs is complete (has all three required JSON files)
    """
    eligible = []
    for results_dir in sorted(root.rglob("results")):
        if not results_dir.is_dir():
            continue
        target_dir = results_dir.parent
        suites = [
            d for d in results_dir.iterdir()
            if d.is_dir() and not d.name.startswith("old") and suite_is_complete(d)
        ]
        if len(suites) >= 2:
            eligible.append(target_dir)
    return eligible


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_header(target_dir: Path, suite_a: str, suite_b: str) -> str:
    fn_parts = target_dir.parts  # e.g. [..., 'pandas', 'DataFrame', 'reindex']
    fn_label = ".".join(fn_parts[-3:]) if len(fn_parts) >= 3 else target_dir.name
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"# Test Suite Comparison: `{fn_label}`\n\n"
        f"- Generated at: {now}\n"
        f"- Suite A: `{suite_a}`\n"
        f"- Suite B: `{suite_b}`\n"
        f"- Target dir: `{target_dir}`\n"
    )


def section_test_status(a: dict, b: dict) -> str:
    def _status_block(suite: dict) -> str:
        bl = suite["kill"]["baseline"]
        passed = bl.get("passed", False)
        failed = bl.get("failed_nodeids", [])
        xfail = bl.get("xfail_nodeids", [])
        status = "PASS" if passed else "FAIL"
        lines = [f"- Status: **{status}**"]
        if failed:
            lines.append(f"- Failed ({len(failed)}):")
            for nid in failed:
                lines.append(f"  - `{short_name(nid)}`")
        if xfail:
            lines.append(f"- xfail ({len(xfail)}):")
            for nid in xfail:
                lines.append(f"  - `{short_name(nid)}`")
        if passed and not xfail:
            lines.append("- All tests passed.")
        return "\n".join(lines)

    return (
        "## 1. Baseline Test Status\n\n"
        f"### Suite A — `{a['name']}`\n"
        f"{_status_block(a)}\n\n"
        f"### Suite B — `{b['name']}`\n"
        f"{_status_block(b)}\n"
    )


def section_mutant_kill(a: dict, b: dict, clauses: dict[str, dict]) -> str:
    runs_a = {r["mutant_id"]: r for r in a["kill"].get("runs", [])}
    runs_b = {r["mutant_id"]: r for r in b["kill"].get("runs", [])}
    summary_a = a["kill"].get("summary", {})
    summary_b = b["kill"].get("summary", {})

    all_ids = sorted(set(runs_a) | set(runs_b))

    def icon(status: str | None) -> str:
        if status == "killed":
            return "✅ killed"
        if status == "survived":
            return "❌ survived"
        if status == "invalid_baseline":
            return "⚠️ invalid"
        return "—"

    def change(sa: str | None, sb: str | None) -> str:
        if sa == sb:
            return ""
        if sa == "survived" and sb == "killed":
            return "🆕 B kills"
        if sa == "killed" and sb == "survived":
            return "🔻 B loses"
        return "changed"

    rows = ["| Mutant ID | Clause | Description | Suite A | Suite B | Change |",
            "|-----------|--------|-------------|---------|---------|--------|"]
    for mid in all_ids:
        r_a = runs_a.get(mid, {})
        r_b = runs_b.get(mid, {})
        cid = r_a.get("clause_id") or r_b.get("clause_id", "")
        desc = clauses.get(cid, {}).get("description", "")[:55]
        sa = r_a.get("status")
        sb = r_b.get("status")
        rows.append(f"| `{mid}` | {cid} | {desc} | {icon(sa)} | {icon(sb)} | {change(sa, sb)} |")

    score_a = summary_a.get("mutation_score", 0)
    score_b = summary_b.get("mutation_score", 0)
    killed_a = summary_a.get("killed", 0)
    survived_a = summary_a.get("survived", 0)
    killed_b = summary_b.get("killed", 0)
    survived_b = summary_b.get("survived", 0)
    total = killed_a + survived_a  # same total for both suites

    summary_rows = [
        "\n### Mutation Score Summary\n",
        "| Metric | Suite A | Suite B | Delta |",
        "|--------|---------|---------|-------|",
        f"| Killed | {killed_a}/{total} | {killed_b}/{total} | {killed_b - killed_a:+d} |",
        f"| Survived | {survived_a}/{total} | {survived_b}/{total} | {survived_b - survived_a:+d} |",
        f"| **Mutation score** | **{score_a:.3f}** | **{score_b:.3f}** | **{delta_str(score_a, score_b)}** |",
    ]

    return (
        "## 2. Mutant Kill Comparison\n\n"
        + "\n".join(rows)
        + "\n"
        + "\n".join(summary_rows)
        + "\n"
    )


def section_coverage(a: dict, b: dict) -> str:
    cov_a = a["coverage"]
    cov_b = b["coverage"]

    # Method-level
    methods_a = {m["label"]: m for m in cov_a.get("methods", [])}
    methods_b = {m["label"]: m for m in cov_b.get("methods", [])}
    all_methods = sorted(set(methods_a) | set(methods_b))

    def method_avg_pct(cov: dict) -> float:
        methods = cov.get("methods", [])
        if not methods:
            return 0.0
        return sum(m["line_coverage_percent"] for m in methods) / len(methods)

    avg_a = method_avg_pct(cov_a)
    avg_b = method_avg_pct(cov_b)

    rows = ["| Method | Suite A lines | Suite B lines | Suite A % | Suite B % | Delta |",
            "|--------|--------------|--------------|-----------|-----------|-------|"]
    for lbl in all_methods:
        ma = methods_a.get(lbl, {})
        mb = methods_b.get(lbl, {})
        la = f"{ma.get('covered_lines', '?')}/{ma.get('num_statements', '?')}" if ma else "—"
        lb = f"{mb.get('covered_lines', '?')}/{mb.get('num_statements', '?')}" if mb else "—"
        pa = ma.get("line_coverage_percent", 0) if ma else 0.0
        pb = mb.get("line_coverage_percent", 0) if mb else 0.0
        short = lbl.split(".")[-1]
        rows.append(f"| `{short}` | {la} | {lb} | {pa:.1f}% | {pb:.1f}% | {pb - pa:+.1f}pp |")

    rows.append(f"| **Average** | | | **{avg_a:.2f}%** | **{avg_b:.2f}%** | **{avg_b - avg_a:+.2f}pp** |")

    # Branch coverage
    tot_a = cov_a.get("totals", {})
    tot_b = cov_b.get("totals", {})
    br_a_cov = tot_a.get("covered_branches", 0)
    br_a_tot = tot_a.get("num_branches", 1)
    br_b_cov = tot_b.get("covered_branches", 0)
    br_b_tot = tot_b.get("num_branches", 1)
    br_pct_a = br_a_cov / br_a_tot * 100
    br_pct_b = br_b_cov / br_b_tot * 100

    branch_rows = [
        "\n### Branch Coverage\n",
        "| Metric | Suite A | Suite B | Delta |",
        "|--------|---------|---------|-------|",
        f"| Covered branches | {br_a_cov}/{br_a_tot} | {br_b_cov}/{br_b_tot} | {br_b_cov - br_a_cov:+d} |",
        f"| **Branch coverage** | **{br_pct_a:.2f}%** | **{br_pct_b:.2f}%** | **{br_pct_b - br_pct_a:+.2f}pp** |",
    ]

    return (
        "## 3. Coverage Comparison\n\n"
        "### Method Line Coverage\n\n"
        + "\n".join(rows)
        + "\n"
        + "\n".join(branch_rows)
        + "\n"
    )


def section_properties(a: dict, b: dict, clauses: dict[str, dict]) -> str:
    props_a = a["props"]
    props_b = b["props"]

    # Build clause satisfaction sets
    unsat_a = set(props_a.get("unsatisfied_clause_ids", []))
    unsat_b = set(props_b.get("unsatisfied_clause_ids", []))
    all_ids = sorted(clauses.keys(), key=lambda x: int(x[1:]))

    def icon(cid: str, unsat: set) -> str:
        return "❌" if cid in unsat else "✅"

    def change(cid: str) -> str:
        a_ok = cid not in unsat_a
        b_ok = cid not in unsat_b
        if a_ok == b_ok:
            return ""
        return "🆕 B covers" if b_ok else "🔻 B loses"

    rows = [
        "| Clause | Category | Description | Suite A | Suite B | Change |",
        "|--------|----------|-------------|---------|---------|--------|",
    ]
    for cid in all_ids:
        cl = clauses.get(cid, {})
        cat = cl.get("category", "")
        desc = cl.get("description", "")[:60]
        rows.append(
            f"| {cid} | {cat} | {desc} | {icon(cid, unsat_a)} | {icon(cid, unsat_b)} | {change(cid)} |"
        )

    cov_a = props_a.get("properties_coverage", 0)
    cov_b = props_b.get("properties_coverage", 0)
    sat_a = props_a.get("satisfied_clauses", 0)
    sat_b = props_b.get("satisfied_clauses", 0)
    total = props_a.get("total_clauses", 0)

    summary_rows = [
        "\n### Properties Coverage Summary\n",
        "| Metric | Suite A | Suite B | Delta |",
        "|--------|---------|---------|-------|",
        f"| Satisfied | {sat_a}/{total} | {sat_b}/{total} | {sat_b - sat_a:+d} |",
        f"| **Coverage** | **{pct(cov_a)}** | **{pct(cov_b)}** | **{delta_str(cov_a, cov_b, as_pct=True)}** |",
    ]

    return (
        "## 4. Properties Coverage Comparison\n\n"
        + "\n".join(rows)
        + "\n"
        + "\n".join(summary_rows)
        + "\n"
    )


def section_summary(a: dict, b: dict) -> str:
    # Mutation score
    ms_a = a["kill"].get("summary", {}).get("mutation_score", 0)
    ms_b = b["kill"].get("summary", {}).get("mutation_score", 0)

    # Properties coverage
    pc_a = a["props"].get("properties_coverage", 0)
    pc_b = b["props"].get("properties_coverage", 0)

    # Method avg line coverage
    def avg_method_cov(cov: dict) -> float:
        methods = cov.get("methods", [])
        if not methods:
            return 0.0
        return sum(m["line_coverage_percent"] for m in methods) / len(methods) / 100

    mc_a = avg_method_cov(a["coverage"])
    mc_b = avg_method_cov(b["coverage"])

    # Branch coverage
    tot_a = a["coverage"].get("totals", {})
    tot_b = b["coverage"].get("totals", {})
    br_a = tot_a.get("covered_branches", 0) / max(tot_a.get("num_branches", 1), 1)
    br_b = tot_b.get("covered_branches", 0) / max(tot_b.get("num_branches", 1), 1)

    # Baseline pass
    bl_a = "✅ PASS" if a["kill"]["baseline"].get("passed") else f"❌ FAIL ({len(a['kill']['baseline'].get('failed_nodeids', []))} tests)"
    bl_b = "✅ PASS" if b["kill"]["baseline"].get("passed") else f"❌ FAIL ({len(b['kill']['baseline'].get('failed_nodeids', []))} tests)"

    rows = [
        "| Metric | Suite A | Suite B | Delta | Better |",
        "|--------|---------|---------|-------|--------|",
        f"| Baseline passes | {bl_a} | {bl_b} | — | — |",
        f"| Mutation score | {ms_a:.3f} | {ms_b:.3f} | {delta_str(ms_a, ms_b)} | {winner_arrow(ms_a, ms_b)} |",
        f"| Properties coverage | {pct(pc_a)} | {pct(pc_b)} | {delta_str(pc_a, pc_b, as_pct=True)} | {winner_arrow(pc_a, pc_b)} |",
        f"| Avg method line cov | {pct(mc_a)} | {pct(mc_b)} | {delta_str(mc_a, mc_b, as_pct=True)} | {winner_arrow(mc_a, mc_b)} |",
        f"| Branch coverage | {pct(br_a)} | {pct(br_b)} | {delta_str(br_a, br_b, as_pct=True)} | {winner_arrow(br_a, br_b)} |",
    ]

    return "## 5. Summary\n\n" + "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_report(target_dir: Path, suite_a: str, suite_b: str) -> str:
    results_dir = target_dir / "results"
    a = load_suite(results_dir, suite_a)
    b = load_suite(results_dir, suite_b)
    clauses = load_clauses(target_dir)

    parts = [
        section_header(target_dir, suite_a, suite_b),
        section_test_status(a, b),
        section_mutant_kill(a, b, clauses),
        section_coverage(a, b),
        section_properties(a, b, clauses),
        section_summary(a, b),
    ]
    return "\n---\n\n".join(parts)


def run_one(target_dir: Path, suite_a: str | None, suite_b: str | None,
            out_path: Path | None) -> bool:
    """Run comparison for a single target_dir. Returns True on success."""
    results_dir = target_dir / "results"
    suites = [
        d.name for d in sorted(results_dir.iterdir())
        if d.is_dir() and not d.name.startswith("old") and suite_is_complete(d)
    ]
    if len(suites) < 2:
        print(f"  SKIP {target_dir}: fewer than 2 complete suites (found: {suites})")
        return False

    sa = suite_a or suites[0]
    sb = suite_b or suites[1]
    dest = out_path or (results_dir / "comparison_report.md")

    try:
        report = build_report(target_dir, sa, sb)
        dest.write_text(report)
        print(f"  OK   {dest}")
        return True
    except Exception as exc:
        print(f"  ERR  {target_dir}: {exc}", file=sys.stderr)
        return False


def main() -> None:
    # The script lives inside the test_quality_metric/ directory; use that as
    # the default root for auto-discovery.
    script_root = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description=(
            "Generate side-by-side comparison reports for all eligible function "
            "directories, or for a single --target-dir."
        )
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help=(
            "Path to one function target directory "
            "(e.g. test_quality_metric/pandas/DataFrame/reindex). "
            "Omit to process ALL eligible directories under test_quality_metric/."
        ),
    )
    parser.add_argument(
        "--suite-a",
        default=None,
        help="Name of the first suite sub-directory (auto-detected if omitted).",
    )
    parser.add_argument(
        "--suite-b",
        default=None,
        help="Name of the second suite sub-directory (auto-detected if omitted).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output markdown file path. "
            "Only valid together with --target-dir. "
            "Default: <target-dir>/results/comparison_report.md."
        ),
    )
    args = parser.parse_args()

    if args.output and not args.target_dir:
        parser.error("--output requires --target-dir")

    if args.target_dir:
        # Single-function mode
        target_dir = Path(args.target_dir)
        if not (target_dir / "results").exists():
            print(f"ERROR: results directory not found under {target_dir}", file=sys.stderr)
            sys.exit(1)
        out_path = Path(args.output) if args.output else None
        ok = run_one(target_dir, args.suite_a, args.suite_b, out_path)
        sys.exit(0 if ok else 1)
    else:
        # Batch mode — scan all eligible dirs under test_quality_metric/
        targets = discover_target_dirs(script_root)
        if not targets:
            print(f"No eligible function directories found under {script_root}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(targets)} eligible function director{'y' if len(targets) == 1 else 'ies'}:")
        ok_count = 0
        for td in targets:
            ok_count += run_one(td, args.suite_a, args.suite_b, None)
        print(f"\n{ok_count}/{len(targets)} comparison reports written.")
        sys.exit(0 if ok_count == len(targets) else 1)


if __name__ == "__main__":
    main()
