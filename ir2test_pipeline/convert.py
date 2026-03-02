"""Convert a pandas documentation URL to markdown and save it to the appropriate directory.

Usage:
    python convert.py <url>
    python convert.py https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reindex.html
"""

import sys
import os
import requests
from bs4 import BeautifulSoup
import markdownify
from pathlib import Path


BASE_DIR = Path(__file__).parent

def url_to_output_path(url: str) -> Path:
    """Derive output path from pandas doc URL.

    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reindex.html
    -> pandas/DataFrame/reindex/pandas.DataFrame.reindex.md
    """
    # Extract filename without extension, e.g. "pandas.DataFrame.reindex"
    basename = url.rstrip("/").split("/")[-1]
    if basename.endswith(".html"):
        basename = basename[:-5]

    parts = basename.split(".")  # ['pandas', 'DataFrame', 'reindex']
    if len(parts) < 2:
        raise ValueError(f"Cannot derive output path from URL basename: {basename!r}")

    # Skip the leading "pandas" module prefix for directory structure
    dir_parts = parts[1:]  # ['DataFrame', 'reindex']
    rel_dir = Path(*dir_parts) if len(dir_parts) > 1 else Path(dir_parts[0])
    output_dir = BASE_DIR / "pandas" / rel_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / f"{basename}.md"


def extract_main_content(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    main_content = (
        soup.find("main", id="main-content")
        or soup.find("article")
        or soup
    )

    return markdownify.markdownify(str(main_content), heading_style="ATX")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <url>")
        print("Example: python convert.py https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reindex.html")
        sys.exit(1)

    url = sys.argv[1]

    try:
        output_file = url_to_output_path(url)
        md_text = extract_main_content(url)
        output_file.write_text(md_text, encoding="utf-8")
        print(f"Saved: {output_file}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
