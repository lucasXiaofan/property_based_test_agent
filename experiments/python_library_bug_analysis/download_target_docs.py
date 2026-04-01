from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "results.json"
DEFAULT_DOCS_DIR = BASE_DIR / "downloaded_docs"
DEFAULT_OUTPUT_PATH = BASE_DIR / "counted_case_docs.json"
USER_AGENT = (
    "property-based-test-agent/1.0 "
    "(https://github.com/pandas-dev/pandas docs downloader)"
)


@dataclass(frozen=True)
class DocTarget:
    label: str
    relative_url: str
    filename: str


DOC_TARGETS_BY_ISSUE: dict[int, tuple[DocTarget, ...]] = {
    58471: (
        DocTarget(
            "pandas.concat",
            "reference/api/pandas.concat.html",
            "pandas.concat.md",
        ),
    ),
    59965: (
        DocTarget(
            "pandas.Series.mean",
            "reference/api/pandas.Series.mean.html",
            "pandas.Series.mean.md",
        ),
    ),
    60922: (
        DocTarget(
            "pandas.concat",
            "reference/api/pandas.concat.html",
            "pandas.concat.md",
        ),
    ),
    61099: (
        DocTarget(
            "pandas.Index.astype",
            "reference/api/pandas.Index.astype.html",
            "pandas.Index.astype.md",
        ),
    ),
    61175: (
        DocTarget(
            "pandas.eval",
            "reference/api/pandas.eval.html",
            "pandas.eval.md",
        ),
    ),
    61356: (
        DocTarget(
            "pandas.DataFrame.groupby",
            "reference/api/pandas.DataFrame.groupby.html",
            "pandas.DataFrame.groupby.md",
        ),
    ),
    61509: (
        DocTarget(
            "pandas.pivot_table",
            "reference/api/pandas.pivot_table.html",
            "pandas.pivot_table.md",
        ),
    ),
    61621: (
        DocTarget(
            "pandas.api.types.infer_dtype",
            "reference/api/pandas.api.types.infer_dtype.html",
            "pandas.api.types.infer_dtype.md",
        ),
    ),
    62094: (
        DocTarget(
            "pandas.Index.shift",
            "reference/api/pandas.Index.shift.html",
            "pandas.Index.shift.md",
        ),
    ),
    62240: (
        DocTarget(
            "pandas.Series.str.match",
            "reference/api/pandas.Series.str.match.html",
            "pandas.Series.str.match.md",
        ),
        DocTarget(
            "pandas.Series.str.contains",
            "reference/api/pandas.Series.str.contains.html",
            "pandas.Series.str.contains.md",
        ),
    ),
    62595: (
        DocTarget(
            "pandas.Series.mul",
            "reference/api/pandas.Series.mul.html",
            "pandas.Series.mul.md",
        ),
    ),
    62778: (
        DocTarget(
            "pandas.api.typing.DataFrameGroupBy.mean",
            "reference/api/pandas.api.typing.DataFrameGroupBy.mean.html",
            "pandas.api.typing.DataFrameGroupBy.mean.md",
        ),
    ),
    62829: (
        DocTarget(
            "pandas.json_normalize",
            "reference/api/pandas.json_normalize.html",
            "pandas.json_normalize.md",
        ),
    ),
    62888: (
        DocTarget(
            "pandas.Series.factorize",
            "reference/api/pandas.Series.factorize.html",
            "pandas.Series.factorize.md",
        ),
    ),
    63236: (
        DocTarget(
            "pandas.DataFrame.to_json",
            "reference/api/pandas.DataFrame.to_json.html",
            "pandas.DataFrame.to_json.md",
        ),
    ),
    63262: (
        DocTarget(
            "Indexing and selecting data",
            "user_guide/indexing.html",
            "user_guide.indexing.md",
        ),
    ),
    63306: (
        DocTarget(
            "Copy-on-Write",
            "user_guide/copy_on_write.html",
            "user_guide.copy_on_write.md",
        ),
    ),
    63581: (
        DocTarget(
            "Indexing and selecting data",
            "user_guide/indexing.html",
            "user_guide.indexing.md",
        ),
    ),
    63879: (
        DocTarget(
            "pandas.array",
            "reference/api/pandas.array.html",
            "pandas.array.md",
        ),
    ),
    63993: (
        DocTarget(
            "pandas.DataFrame.reindex",
            "reference/api/pandas.DataFrame.reindex.html",
            "pandas.DataFrame.reindex.md",
        ),
    ),
}


class ArticleMarkdownParser(HTMLParser):
    """Extract the main docs article into a simple markdown representation."""

    BLOCK_TAGS = {
        "article",
        "section",
        "div",
        "p",
        "ul",
        "ol",
        "table",
        "tbody",
        "thead",
        "tr",
        "dl",
    }
    SKIP_TAGS = {"script", "style", "noscript"}
    HEADING_LEVELS = {
        "h1": 1,
        "h2": 2,
        "h3": 3,
        "h4": 4,
        "h5": 5,
        "h6": 6,
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.article_depth = 0
        self.skip_depth = 0
        self.pre_depth = 0
        self.code_depth = 0
        self.heading_level: int | None = None
        self.list_stack: list[str] = []
        self.pending_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "") or ""
        is_article = tag == "article" or "bd-article" in classes.split()
        if is_article:
            self.article_depth += 1
            self._ensure_gap(2)
            return
        if self.article_depth == 0:
            return
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self.HEADING_LEVELS:
            self.heading_level = self.HEADING_LEVELS[tag]
            self._ensure_gap(2)
            self.parts.append("#" * self.heading_level + " ")
            return
        if tag in {"ul", "ol"}:
            self.list_stack.append(tag)
            self._ensure_gap(1)
            return
        if tag == "li":
            self._ensure_gap(1)
            self.parts.append("- ")
            return
        if tag == "br":
            self.parts.append("\n")
            return
        if tag == "pre":
            self.pre_depth += 1
            self._ensure_gap(2)
            self.parts.append("```text\n")
            return
        if tag == "code":
            if self.pre_depth == 0:
                self.parts.append("`")
            self.code_depth += 1
            return
        if tag == "a":
            self.pending_href = attrs_dict.get("href")
            return
        if tag in self.BLOCK_TAGS:
            self._ensure_gap(1)

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self.article_depth:
            self.article_depth -= 1
            self._ensure_gap(2)
            return
        if self.article_depth == 0:
            return
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in self.HEADING_LEVELS:
            self.heading_level = None
            self._ensure_gap(2)
            return
        if tag in {"ul", "ol"} and self.list_stack:
            self.list_stack.pop()
            self._ensure_gap(1)
            return
        if tag == "pre" and self.pre_depth:
            self.pre_depth -= 1
            if self.pre_depth == 0:
                self.parts.append("\n```\n")
            return
        if tag == "code" and self.code_depth:
            self.code_depth -= 1
            if self.pre_depth == 0:
                self.parts.append("`")
            return
        if tag == "a":
            self.pending_href = None
            return
        if tag in self.BLOCK_TAGS:
            self._ensure_gap(1)

    def handle_data(self, data: str) -> None:
        if self.article_depth == 0 or self.skip_depth:
            return
        text = unescape(data)
        if not text.strip():
            if self.pre_depth:
                self.parts.append(text)
            return
        if self.pre_depth:
            self.parts.append(text)
            return
        normalized = re.sub(r"\s+", " ", text)
        self.parts.append(normalized)

    def get_markdown(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n"

    def _ensure_gap(self, size: int) -> None:
        if not self.parts:
            return
        suffix = "".join(self.parts[-3:])
        newline_count = len(suffix) - len(suffix.rstrip("\n"))
        if newline_count < size:
            self.parts.append("\n" * (size - newline_count))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def fetch_text(url: str, timeout_seconds: float) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout_seconds) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def download_doc_markdown(
    docs_base_url: str,
    target: DocTarget,
    destination: Path,
    timeout_seconds: float,
) -> dict[str, str]:
    doc_url = docs_base_url.rstrip("/") + "/" + target.relative_url.lstrip("/")
    try:
        html = fetch_text(doc_url, timeout_seconds=timeout_seconds)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"{exc} while fetching {doc_url}") from exc
    parser = ArticleMarkdownParser()
    parser.feed(html)
    markdown = parser.get_markdown()
    if not markdown.strip():
        raise ValueError(f"failed to extract article content from {doc_url}")
    content = (
        f"# {target.label}\n\n"
        f"- Source URL: {doc_url}\n\n"
        f"{markdown}"
    )
    destination.write_text(content, encoding="utf-8")
    return {
        "label": target.label,
        "source_url": doc_url,
        "markdown_file": str(destination),
    }


def iter_counted_cases(results_path: Path) -> Iterable[dict[str, object]]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"{results_path} does not contain a top-level 'cases' list")
    for case in cases:
        if isinstance(case, dict) and case.get("counted_in_valid_set") is True:
            yield case


def build_manifest(
    results_path: Path,
    docs_dir: Path,
    output_path: Path,
    docs_base_url: str,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    docs_dir.mkdir(parents=True, exist_ok=True)
    for case in iter_counted_cases(results_path):
        issue = int(case["issue"])
        targets = DOC_TARGETS_BY_ISSUE.get(issue)
        if not targets:
            raise KeyError(f"no documentation mapping configured for issue #{issue}")

        downloaded_docs = []
        for target in targets:
            destination = docs_dir / target.filename
            downloaded_docs.append(
                download_doc_markdown(
                    docs_base_url=docs_base_url,
                    target=target,
                    destination=destination,
                    timeout_seconds=timeout_seconds,
                )
            )

        manifest.append(
            {
                "issue": issue,
                "issue_url": case["url"],
                "issue_title": case["title"],
                "hypothesis_sketch": case["hypothesis_sketch"],
                "documentation_md_dir": str(docs_dir),
                "documentation_files": downloaded_docs,
            }
        )

    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download pandas docs for counted bug-analysis cases and materialize "
            "a JSON manifest with local markdown directories."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help=f"Path to the bug analysis results JSON. Default: {RESULTS_PATH}",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help=f"Directory where markdown docs will be stored. Default: {DEFAULT_DOCS_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for the emitted JSON manifest. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--docs-base-url",
        default="https://pandas.pydata.org/pandas-docs/version/3.0",
        help=(
            "Base URL for pandas docs. Default: "
            "https://pandas.pydata.org/pandas-docs/version/3.0"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for each docs page request. Default: 30",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest(
            results_path=args.results,
            docs_dir=args.docs_dir,
            output_path=args.output,
            docs_base_url=args.docs_base_url,
            timeout_seconds=args.timeout_seconds,
        )
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {len(manifest)} cases to {args.output} "
        f"and markdown docs under {args.docs_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
