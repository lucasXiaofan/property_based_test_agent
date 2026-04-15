from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PANDAS_ORACLE_DIR = BASE_DIR / "pandas"
DEFAULT_DOCS_DIR = (
    BASE_DIR.parent / "python_library_bug_analysis" / "downloaded_docs"
)
DEFAULT_OUTPUT_PATH = BASE_DIR / "tasks.json"
DEFAULT_LIMIT = 3

CONSTRAINT_PATH = BASE_DIR / "code_agent_constraint.md"
IR_GUIDELINE_PATH = BASE_DIR / "ir_generation_guideline.md"
BASELINE_GUIDELINE_PATH = BASE_DIR / "baseline_test_generation_guideline.md"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def find_function_folders_without_baseline(
    root_dir: Path = PANDAS_ORACLE_DIR,
) -> list[Path]:
    candidates = sorted(path.parent for path in root_dir.rglob("ir_v2.json"))
    missing = [folder for folder in candidates if not (folder / "baseline_test.py").exists()]
    if not missing:
        raise ValueError(f"no function folder under {root_dir} is missing baseline_test.py")
    return missing


def build_target_folder_task(target_folder: Path) -> list[dict[str, str]]:
    target_folder_str = str(target_folder.resolve())
    prompt = (
        f"Given `{BASELINE_GUIDELINE_PATH}`, read the function doc markdown from the "
        "target function folder and make the baseline test without referencing other "
        "existing files. "
        f"Target function folder: {target_folder_str}\n\n"
        "Run all code with `uv`.\n\n"
        "After finishing the baseline test and making sure it runs, utilize the "
        "`ir_v2.json` in that folder to make a copy of the baseline test named "
        "`ir_enhanced_test.py`. Make sure you use the IR to generate unique tests "
        "beyond the baseline — not just happy path tests, but high-stakes edge cases.\n\n"
        "In `ir_enhanced_test.py`, add comments indicating which test cases are new "
        "(inspired by the IR) and which are from the baseline."
    )
    return [
        {
            "task": f"baseline-and-ir-enhanced-{slugify(target_folder_str)}",
            "prompt": prompt,
        }
    ]


def write_target_folder_task(output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    task_items: list[dict[str, str]] = []
    for folder in find_function_folders_without_baseline():
        task_items.extend(build_target_folder_task(folder))
    output_path.write_text(
        json.dumps(task_items, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def derive_target_parts(doc_path: Path) -> tuple[str, str, str]:
    stem = doc_path.stem
    parts = stem.split(".")
    if len(parts) >= 3 and parts[0] == "pandas":
        library = parts[0]
        module_or_class = ".".join(parts[1:-1])
        function = parts[-1]
        return library, module_or_class, function
    if len(parts) == 2 and parts[0] == "pandas":
        return "pandas", "top_level", parts[1]
    return "pandas", "misc", stem


def parse_source_url(doc_text: str) -> str:
    match = re.search(r"^- Source URL:\s*(\S+)\s*$", doc_text, flags=re.MULTILINE)
    return match.group(1) if match else ""


def build_output_path(
    task_type: str,
    library: str,
    module_or_class: str,
    function: str,
) -> str:
    filename = "baseline_test.py" if task_type == "baseline" else "ir_v2.json"
    return str(BASE_DIR / library / module_or_class / function / filename)


def build_prompt(
    task_type: str,
    constraint_text: str,
    guideline_text: str,
    doc_path: Path,
    doc_text: str,
    library: str,
    module_or_class: str,
    function: str,
) -> str:
    source_url = parse_source_url(doc_text)
    output_path = build_output_path(task_type, library, module_or_class, function)
    task_label = (
        "baseline test generation"
        if task_type == "baseline"
        else "Itermediate representation (IR) generation"
    )
    return (
        f"You are performing {task_label} for pandas documentation.\n\n"
        f"## Code Agent Constraint\n"
        f"{constraint_text.strip()}\n\n"
        f"## Task Guideline\n"
        f"{guideline_text.strip()}\n\n"
        f"## Target Metadata\n"
        f"- documentation_markdown_path: {doc_path}\n"
        f"- doc_url: {source_url or 'unknown'}\n"
        f"- library: {library}\n"
        f"- module_or_class: {module_or_class}\n"
        f"- function_name: {function}\n"
        f"- expected_output_path: {output_path}\n\n"
        f"Use the markdown documentation below as the documentation input for the task.\n"
        f"Do not fetch other documentation unless the prompt explicitly tells you to.\n\n"
        f"## Documentation Markdown\n"
        f"```md\n{doc_text.rstrip()}\n```\n"
    )


def extract_doc_paths_from_task_json(task_json_path: Path) -> set[str]:
    task_items = json.loads(task_json_path.read_text(encoding="utf-8"))
    doc_paths: set[str] = set()
    for item in task_items:
        prompt = item.get("prompt", "")
        for line in prompt.splitlines():
            if line.startswith("- documentation_markdown_path: "):
                doc_paths.add(line.split(": ", 1)[1].strip())
                break
    return doc_paths


def should_exclude_doc(doc_path: Path, exclude_doc_paths: set[str]) -> bool:
    doc_path_str = str(doc_path)
    doc_name = doc_path.name
    return any(
        excluded == doc_path_str or Path(excluded).name == doc_name
        for excluded in exclude_doc_paths
    )


def build_tasks(
    docs_dir: Path,
    limit: int,
    exclude_doc_paths: set[str] | None = None,
) -> list[dict[str, str]]:
    constraint_text = CONSTRAINT_PATH.read_text(encoding="utf-8")
    ir_guideline_text = IR_GUIDELINE_PATH.read_text(encoding="utf-8")
    baseline_guideline_text = BASELINE_GUIDELINE_PATH.read_text(encoding="utf-8")

    excluded = exclude_doc_paths or set()
    doc_paths = [
        path for path in sorted(docs_dir.glob("*.md")) if not should_exclude_doc(path, excluded)
    ][:limit]
    if len(doc_paths) < limit:
        raise ValueError(
            f"expected at least {limit} markdown docs in {docs_dir} after exclusions, "
            f"found {len(doc_paths)}"
        )

    tasks: list[dict[str, str]] = []
    for doc_path in doc_paths:
        doc_text = doc_path.read_text(encoding="utf-8")
        library, module_or_class, function = derive_target_parts(doc_path)
        base_slug = slugify(f"{library}-{module_or_class}-{function}")

        for task_type, guideline_text in (
            ("baseline", baseline_guideline_text),
            ("ir_generation", ir_guideline_text),
        ):
            tasks.append(
                {
                    "task": f"{task_type}-{base_slug}",
                    "prompt": build_prompt(
                        task_type=task_type,
                        constraint_text=constraint_text,
                        guideline_text=guideline_text,
                        doc_path=doc_path,
                        doc_text=doc_text,
                        library=library,
                        module_or_class=module_or_class,
                        function=function,
                    ),
                }
            )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build oracle-generation task JSON from guideline files and downloaded "
            "documentation markdown."
        )
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help=f"Directory containing downloaded markdown docs. Default: {DEFAULT_DOCS_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for the generated task JSON. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of docs to process. Default: {DEFAULT_LIMIT}",
    )
    parser.add_argument(
        "--exclude-task-json",
        type=Path,
        action="append",
        default=[],
        help="Existing task JSON to exclude already-covered documentation paths from.",
    )
    parser.add_argument(
        "--target-folder-task",
        action="store_true",
        help=(
            "Write task.json for all function folders under "
            "experiments/oracle_generation/pandas that has ir_v2.json but no baseline_test.py."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_folder_task:
        output_path = write_target_folder_task(args.output)
        print(f"wrote tasks to {output_path}")
        return 0

    excluded_doc_paths: set[str] = set()
    for task_json_path in args.exclude_task_json:
        excluded_doc_paths.update(extract_doc_paths_from_task_json(task_json_path))

    tasks = build_tasks(
        args.docs_dir,
        args.limit,
        exclude_doc_paths=excluded_doc_paths,
    )
    args.output.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(tasks)} tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
