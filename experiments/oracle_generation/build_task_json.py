from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_FOLDER = BASE_DIR / "pandas"
DEFAULT_OUTPUT_PATH = BASE_DIR / "tasks.json"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def is_leaf_function_folder(folder: Path) -> bool:
    return folder.is_dir() and any(folder.glob("*.md"))


def find_leaf_function_folders(target_folder: Path) -> list[Path]:
    target_folder = target_folder.resolve()
    if is_leaf_function_folder(target_folder):
        return [target_folder]

    folders = {
        path.parent.resolve()
        for path in target_folder.rglob("*.md")
        if ".hypothesis" not in path.parts and "__pycache__" not in path.parts
    }
    return sorted(folders)


def build_task(recipe_text: str, recipe_path: Path, target_folder: Path, created_at: str) -> dict:
    target_folder = target_folder.resolve()
    prompt = recipe_text.format(target_folder=str(target_folder))
    try:
        label = str(target_folder.relative_to(BASE_DIR.resolve()))
    except ValueError:
        label = str(target_folder)

    return {
        "task": f"{recipe_path.stem}-{slugify(label)}",
        "prompt": prompt,
        "created_at": created_at,
        "completed": False,
        "completed_at": None,
        "target_folder": str(target_folder),
        "recipe_path": str(recipe_path.resolve()),
    }


def build_tasks(recipe_path: Path, target_folders: list[Path]) -> list[dict]:
    recipe_text = recipe_path.read_text(encoding="utf-8")
    created_at = now_iso()
    leaf_folders = sorted(
        {
            folder
            for target_folder in target_folders
            for folder in find_leaf_function_folders(target_folder)
        }
    )
    if not leaf_folders:
        roots = ", ".join(str(folder) for folder in target_folders)
        raise ValueError(f"no leaf function folders with markdown docs under {roots}")

    return [
        build_task(recipe_text, recipe_path, target_folder, created_at)
        for target_folder in leaf_folders
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build task JSON by rendering a markdown recipe for target folders."
    )
    parser.add_argument(
        "recipe",
        type=Path,
        help="Markdown recipe using f-string-style {target_folder} placeholders.",
    )
    parser.add_argument(
        "--target-folder",
        type=Path,
        action="append",
        help=f"Target leaf folder or root to scan. Default: {DEFAULT_TARGET_FOLDER}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for generated task JSON. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_folders = args.target_folder or [DEFAULT_TARGET_FOLDER]
    tasks = build_tasks(args.recipe, target_folders)
    args.output.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(tasks)} tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
