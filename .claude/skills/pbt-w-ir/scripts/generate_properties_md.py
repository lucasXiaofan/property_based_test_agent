#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
generate_properties_md.py

Reads a PBT IR JSON and produces a clean Markdown file
that a test-generation agent reads to write Hypothesis tests.

Usage:
    uv run generate_properties_md.py ir/pandas/merge.json
    # outputs: ir/pandas/merge_properties.md
"""

import json
import re
import sys
from pathlib import Path

PROPERTY_TYPE_LABELS = {
    "explicit":   "📄 Explicit",
    "indirect":   "🔍 Indirect",
    "implicit":   "💡 Implicit",
    "convention": "📚 Convention",
    "ambiguity":  "⚠️ Resolved Ambiguity",
}


def slugify(text: str, max_words: int = 6) -> str:
    """Convert a claim string to a snake_case identifier fragment."""
    words = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()
    return "_".join(words[:max_words])


def test_name(ptype: str, pid: str, claim: str) -> str:
    return f"test_{ptype}__{pid.lower()}__{slugify(claim)}"


def load_ir(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def section(title: str, lines: list[str]) -> list[str]:
    return ["", f"## {title}", ""] + lines + [""]


def generate_md(ir: dict) -> str:
    meta = ir.get("metadata", {})
    out = []

    # Header
    out += [
        f"# Properties to Test: `{meta.get('function', 'unknown')}`",
        "",
        f"**Library**: {meta.get('library', '?')}  ",
        f"**Version**: {meta.get('version', '?')}  ",
        f"**Signature**: `{meta.get('signature', '?')}`",
        "",
    ]

    # Logical errors — always shown, never blocks output
    errors = ir.get("logical_errors", [])
    if errors:
        severe = [e for e in errors if e.get("severity") == "severe"]
        minor  = [e for e in errors if e.get("severity") != "severe"]

        lines = []
        if severe:
            lines.append("> ⛔ **Severe errors found.** The properties below are extracted from the parts of the spec that are still sound, but treat test results with caution.")
            lines.append("")
        for e in errors:
            badge = "⛔ SEVERE" if e.get("severity") == "severe" else "⚠️ minor"
            affected = f"  *(affects: {', '.join(e['affects_properties'])})*" if e.get("affects_properties") else ""
            lines.append(f"- **[{badge}]** {e['description']}{affected}")

        out += section("Logical Errors in Documentation", lines)

    # Input domain
    domain = ir.get("input_domain", {})
    if domain:
        domain = dict(domain)  # copy so we can pop
        constraints = domain.pop("constraints", [])
        invalid = domain.pop("invalid_inputs", [])

        lines = ["Use these to construct `@given` strategies and `assume()` calls.", ""]
        for param, desc in domain.items():
            lines.append(f"- **`{param}`**: {desc}")
        if constraints:
            lines += ["", "**Cross-parameter constraints:**"]
            lines += [f"- {c}" for c in constraints]
        if invalid:
            lines += ["", "**Invalid inputs (should raise):**"]
            lines += [f"- {inv}" for inv in invalid]

        out += section("Valid Input Domain", lines)

    # Properties — grouped by type
    properties = ir.get("properties", [])
    if properties:
        type_order = ["explicit", "indirect", "implicit", "convention", "ambiguity"]
        grouped: dict[str, list] = {t: [] for t in type_order}
        for p in properties:
            grouped.setdefault(p.get("type", "implicit"), []).append(p)

        lines = [
            "Each property maps to one test. Use `claim` as the docstring,",
            "`expression` as the assertion skeleton, `when` as the `assume()` guard,",
            "and `strategy` to decide what inputs to generate.",
            "",
            "**Test naming convention** — name every test function exactly as shown in each",
            "property's **Test name** field below:",
            "",
            "```",
            "test_{type}__{id}__{claim_slug}",
            "```",
            "",
            "- `type` is the property classification: `explicit`, `indirect`, `implicit`,",
            "  `convention`, or `ambiguity`.",
            "- `id` is the property ID in lowercase (e.g. `p1`, `p3`).",
            "- `claim_slug` is the first six meaningful words of the claim, snake_cased.",
            "",
            "This makes failure triage instant: a failing `test_explicit__*` test means a",
            "documented guarantee is broken; a failing `test_implicit__*` is an inferred",
            "invariant; a failing `test_indirect__*` needs interpretation before filing a bug.",
            "",
        ]

        for ptype in type_order:
            group = grouped.get(ptype, [])
            if not group:
                continue
            label = PROPERTY_TYPE_LABELS.get(ptype, ptype)
            lines.append(f"### {label}")
            lines.append("")
            for p in group:
                source_note = f" — source: {p['source']}" if p.get("source") else ""
                lines.append(f"#### {p['id']}: {p['claim']}{source_note}")
                lines.append("")
                lines.append(f"**Test name**: `{test_name(ptype, p['id'], p['claim'])}`")
                lines.append("")
                lines.append("```python")
                lines.append(f"# assert:   {p['expression']}")
                if p.get("when"):
                    lines.append(f"# when:     {p['when']}")
                lines.append("```")
                lines.append("")
                lines.append(f"**Strategy**: {p['strategy']}")
                lines.append("")

        out += section("Properties to Test", lines)
    
    ## helpful resources 
    resources = """
- **Basic tutorial**: https://hypothesis.readthedocs.io/en/latest/quickstart.html
- **Strategies reference**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
- **NumPy strategies**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#numpy
- **Pandas strategies**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas    - 
You only need this md, and generate test cases in pandas_bug_finding/ir_test
    """
    out += section("Helpful Resources", resources.splitlines())

    # Unresolved ambiguities
    # unresolved = ir.get("unresolved_ambiguities", [])
    # if unresolved:
    #     lines = ["Do not write tests for these. The behavior is undefined or undocumented.", ""]
    #     for a in unresolved:
    #         lines.append(f"#### {a['id']}: {a['description']}")
    #         if a.get("maintainer_note"):
    #             lines.append("")
    #             lines.append(f"> **Maintainer note**: {a['maintainer_note']}")
    #         lines.append("")

    #     out += section("Unresolved Ambiguities — Do Not Test", lines)

    return "\n".join(out)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run generate_properties_md.py <ir_json_path>")
        sys.exit(1)

    ir_path = Path(sys.argv[1])
    if not ir_path.exists():
        print(f"File not found: {ir_path}")
        sys.exit(1)

    ir = load_ir(ir_path)
    md = generate_md(ir)

    out_path = ir_path.with_name(ir_path.stem + "_properties.md")
    out_path.write_text(md)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()