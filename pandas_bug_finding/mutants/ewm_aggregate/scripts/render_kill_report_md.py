from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def badge(outcome: str) -> str:
    if outcome == "killed":
        return "KILLED"
    if outcome == "survived":
        return "SURVIVED"
    if outcome == "invalid":
        return "INVALID"
    return outcome.upper()


def make_md(report: dict, title: str) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report.get('generated_at_utc', 'unknown')}`")
    lines.append(f"- Target: `{report.get('target_symbol', 'unknown')}`")
    lines.append("")

    lines.append("## Mutants")
    lines.append("")
    lines.append("| Mutant ID | Description |")
    lines.append("|---|---|")
    for m in report.get("mutants", []):
        lines.append(f"| `{m.get('mutant_id','')}` | {m.get('description','')} |")
    lines.append("")

    lines.append("## Outcomes")
    lines.append("")
    lines.append("| Test Target | Baseline | Mutant | Outcome | Return Code |")
    lines.append("|---|---|---|---|---:|")

    for ev in report.get("evaluations", []):
        test_file = ev.get("test_file", "")
        baseline = ev.get("baseline", {})
        baseline_state = "PASS" if baseline.get("passed") else "FAIL"
        for mr in ev.get("mutants", []):
            lines.append(
                "| "
                f"`{test_file}` | "
                f"{baseline_state} | "
                f"`{mr.get('mutant_id','')}` | "
                f"**{badge(mr.get('outcome',''))}** | "
                f"{mr.get('returncode', '') if mr.get('returncode', None) is not None else '-'} |"
            )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `INVALID` means baseline failed for that test target, so mutant classification is not trusted.")
    lines.append("- Use node-level targets when full file baselines are unstable.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render mutant kill report JSON to Markdown")
    parser.add_argument("--input", required=True, help="Path to kill report JSON")
    parser.add_argument("--output", required=True, help="Path to output Markdown")
    parser.add_argument("--title", default="Mutant Kill Report", help="Markdown title")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    report = load(inp)
    md = make_md(report, args.title)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote markdown report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
