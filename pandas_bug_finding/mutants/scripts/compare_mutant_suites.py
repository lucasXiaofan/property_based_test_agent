"""Compare mutant outcomes between baseline_testing and ir_test suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def suite_from_test_target(test_target: str) -> str:
    if "/baseline_testing/" in test_target:
        return "baseline_testing"
    if "/ir_test/" in test_target:
        return "ir_test"
    return "other"


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(report: dict) -> dict:
    mutant_ids = [m["mutant_id"] for m in report.get("mutants", [])]
    suites = {
        "baseline_testing": {
            "test_targets": [],
            "baseline_pass_count": 0,
            "baseline_fail_count": 0,
            "killed": set(),
            "survived": set(),
            "invalid": set(),
        },
        "ir_test": {
            "test_targets": [],
            "baseline_pass_count": 0,
            "baseline_fail_count": 0,
            "killed": set(),
            "survived": set(),
            "invalid": set(),
        },
    }

    for ev in report.get("evaluations", []):
        test_target = ev.get("test_file", "")
        suite = suite_from_test_target(test_target)
        if suite not in suites:
            continue

        suites[suite]["test_targets"].append(test_target)
        baseline_passed = bool(ev.get("baseline", {}).get("passed"))
        if baseline_passed:
            suites[suite]["baseline_pass_count"] += 1
        else:
            suites[suite]["baseline_fail_count"] += 1

        for mr in ev.get("mutants", []):
            mid = mr.get("mutant_id")
            outcome = mr.get("outcome")
            if mid is None:
                continue
            if outcome == "killed":
                suites[suite]["killed"].add(mid)
            elif outcome == "survived":
                suites[suite]["survived"].add(mid)
            else:
                suites[suite]["invalid"].add(mid)

    baseline_killed = suites["baseline_testing"]["killed"]
    ir_killed = suites["ir_test"]["killed"]
    both_killed = baseline_killed & ir_killed
    only_baseline = baseline_killed - ir_killed
    only_ir = ir_killed - baseline_killed

    both_suites_baseline_failed = (
        suites["baseline_testing"]["baseline_pass_count"] == 0
        and suites["ir_test"]["baseline_pass_count"] == 0
    )

    invalid_in_baseline = suites["baseline_testing"]["invalid"]
    invalid_in_ir = suites["ir_test"]["invalid"]
    both_invalid_mutants = invalid_in_baseline & invalid_in_ir

    winner = "tie"
    if len(baseline_killed) > len(ir_killed):
        winner = "baseline_testing"
    elif len(ir_killed) > len(baseline_killed):
        winner = "ir_test"

    return {
        "generated_at_utc": report.get("generated_at_utc"),
        "target_symbol": report.get("target_symbol"),
        "mutant_ids": mutant_ids,
        "winner_by_unique_kills": winner,
        "both_suites_baseline_failed": both_suites_baseline_failed,
        "suite_stats": {
            "baseline_testing": {
                "test_targets": suites["baseline_testing"]["test_targets"],
                "baseline_pass_count": suites["baseline_testing"]["baseline_pass_count"],
                "baseline_fail_count": suites["baseline_testing"]["baseline_fail_count"],
                "killed_ids": sorted(suites["baseline_testing"]["killed"]),
                "survived_ids": sorted(suites["baseline_testing"]["survived"]),
                "invalid_ids": sorted(suites["baseline_testing"]["invalid"]),
            },
            "ir_test": {
                "test_targets": suites["ir_test"]["test_targets"],
                "baseline_pass_count": suites["ir_test"]["baseline_pass_count"],
                "baseline_fail_count": suites["ir_test"]["baseline_fail_count"],
                "killed_ids": sorted(suites["ir_test"]["killed"]),
                "survived_ids": sorted(suites["ir_test"]["survived"]),
                "invalid_ids": sorted(suites["ir_test"]["invalid"]),
            },
        },
        "cross_suite": {
            "killed_by_both_ids": sorted(both_killed),
            "killed_only_by_baseline_testing_ids": sorted(only_baseline),
            "killed_only_by_ir_test_ids": sorted(only_ir),
            "invalid_in_both_ids": sorted(both_invalid_mutants),
        },
    }


def render_md(summary: dict, title: str) -> str:
    b = summary["suite_stats"]["baseline_testing"]
    i = summary["suite_stats"]["ir_test"]

    def baseline_state(s: dict) -> str:
        if s["baseline_pass_count"] == 0 and s["baseline_fail_count"] > 0:
            return "FAIL (all targets)"
        if s["baseline_fail_count"] == 0 and s["baseline_pass_count"] > 0:
            return "PASS (all targets)"
        return f"MIXED ({s['baseline_pass_count']} pass / {s['baseline_fail_count']} fail)"

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Generated (UTC): `{summary.get('generated_at_utc')}`")
    lines.append(f"- Target: `{summary.get('target_symbol')}`")
    lines.append(f"- Winner by unique kills: `{summary.get('winner_by_unique_kills')}`")
    lines.append("")

    lines.append("## Suite Comparison")
    lines.append("")
    lines.append("| Suite | Baseline Status | #Unique Killed | Killed Mutant IDs |")
    lines.append("|---|---|---:|---|")
    lines.append(
        f"| `baseline_testing` | {baseline_state(b)} | {len(b['killed_ids'])} | "
        + (", ".join(f"`{x}`" for x in b["killed_ids"]) if b["killed_ids"] else "-")
        + " |"
    )
    lines.append(
        f"| `ir_test` | {baseline_state(i)} | {len(i['killed_ids'])} | "
        + (", ".join(f"`{x}`" for x in i["killed_ids"]) if i["killed_ids"] else "-")
        + " |"
    )
    lines.append("")

    cs = summary["cross_suite"]
    lines.append("## Cross-Suite IDs")
    lines.append("")
    lines.append(f"- Killed by both: {', '.join(f'`{x}`' for x in cs['killed_by_both_ids']) or '-'}")
    lines.append(
        "- Killed only by baseline_testing: "
        + (", ".join(f"`{x}`" for x in cs["killed_only_by_baseline_testing_ids"]) or "-")
    )
    lines.append(
        "- Killed only by ir_test: "
        + (", ".join(f"`{x}`" for x in cs["killed_only_by_ir_test_ids"]) or "-")
    )
    lines.append(
        "- Invalid in both suites: "
        + (", ".join(f"`{x}`" for x in cs["invalid_in_both_ids"]) or "-")
    )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    if summary["both_suites_baseline_failed"]:
        lines.append("- Both suites failed baseline; full-file mutant verdicts are not trustworthy.")
    else:
        lines.append("- At least one suite has passing baselines; killed IDs are meaningful for those targets.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Mutant kill report JSON")
    parser.add_argument("--output-json", required=True, help="Comparison summary JSON")
    parser.add_argument("--output-md", required=True, help="Comparison summary Markdown")
    parser.add_argument("--title", default="Mutant Suite Comparison", help="Markdown title")
    args = parser.parse_args()

    report = load_report(Path(args.input))
    summary = summarize(report)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(render_md(summary, args.title), encoding="utf-8")

    print(f"Wrote: {output_json}")
    print(f"Wrote: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

