#!/usr/bin/env python3
"""Shared helpers for pandas oracle evaluation scripts."""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PANDAS_CASES_DIR = REPO_ROOT / "experiments" / "oracle_generation" / "pandas"
BUG_ANALYSIS_DIR = REPO_ROOT / "experiments" / "python_library_bug_analysis"
COUNTED_CASE_DOCS_PATH = BUG_ANALYSIS_DIR / "counted_case_docs.json"
BUG_INVENTORY_PATH = BUG_ANALYSIS_DIR / "pandas_3_0_0_confirmed_bug_inventory.json"
DOWNLOADED_DOCS_DIR = BUG_ANALYSIS_DIR / "downloaded_docs"


@dataclass(frozen=True)
class ApiCase:
    case_dir: str
    function: str
    doc_label: str
    doc_path: Path
    patch_targets: tuple[str, ...]

    @property
    def directory(self) -> Path:
        return PANDAS_CASES_DIR / self.case_dir

    @property
    def baseline_test(self) -> Path:
        return self.directory / "baseline_test.py"

    @property
    def ir_generated_test(self) -> Path:
        return self.directory / "ir_generated_test.py"

    @property
    def ir_json(self) -> Path:
        return self.directory / "ir_v2.json"


API_CASES: tuple[ApiCase, ...] = (
    ApiCase(
        case_dir="DataFrame/groupby",
        function="pandas.DataFrame.groupby",
        doc_label="pandas.DataFrame.groupby",
        doc_path=DOWNLOADED_DOCS_DIR / "pandas.DataFrame.groupby.md",
        patch_targets=("pandas.core.frame.DataFrame.groupby",),
    ),
    ApiCase(
        case_dir="DataFrame/reindex",
        function="pandas.DataFrame.reindex",
        doc_label="pandas.DataFrame.reindex",
        doc_path=DOWNLOADED_DOCS_DIR / "pandas.DataFrame.reindex.md",
        patch_targets=("pandas.core.frame.DataFrame.reindex",),
    ),
    ApiCase(
        case_dir="DataFrame/to_json",
        function="pandas.DataFrame.to_json",
        doc_label="pandas.DataFrame.to_json",
        doc_path=DOWNLOADED_DOCS_DIR / "pandas.DataFrame.to_json.md",
        patch_targets=("pandas.core.generic.NDFrame.to_json",),
    ),
    ApiCase(
        case_dir="Index/astype",
        function="pandas.Index.astype",
        doc_label="pandas.Index.astype",
        doc_path=DOWNLOADED_DOCS_DIR / "pandas.Index.astype.md",
        patch_targets=("pandas.core.indexes.base.Index.astype",),
    ),
    ApiCase(
        case_dir="Index/shift",
        function="pandas.Index.shift",
        doc_label="pandas.Index.shift",
        doc_path=DOWNLOADED_DOCS_DIR / "pandas.Index.shift.md",
        patch_targets=(
            "pandas.core.indexes.datetimes.DatetimeIndex.shift",
            "pandas.core.indexes.timedeltas.TimedeltaIndex.shift",
            "pandas.core.indexes.period.PeriodIndex.shift",
        ),
    ),
    ApiCase(
        case_dir="Series/factorize",
        function="pandas.Series.factorize",
        doc_label="pandas.Series.factorize",
        doc_path=DOWNLOADED_DOCS_DIR / "pandas.Series.factorize.md",
        patch_targets=("pandas.core.base.IndexOpsMixin.factorize",),
    ),
    ApiCase(
        case_dir="Series/mean",
        function="pandas.Series.mean",
        doc_label="pandas.Series.mean",
        doc_path=DOWNLOADED_DOCS_DIR / "pandas.Series.mean.md",
        patch_targets=("pandas.core.series.Series.mean",),
    ),
    ApiCase(
        case_dir="Series/mul",
        function="pandas.Series.mul",
        doc_label="pandas.Series.mul",
        doc_path=DOWNLOADED_DOCS_DIR / "pandas.Series.mul.md",
        patch_targets=("pandas.core.series.Series.mul",),
    ),
    ApiCase(
        case_dir="Series.str/contains",
        function="pandas.Series.str.contains",
        doc_label="pandas.Series.str.contains",
        doc_path=DOWNLOADED_DOCS_DIR / "pandas.Series.str.contains.md",
        patch_targets=("pandas.core.strings.accessor.StringMethods.contains",),
    ),
    ApiCase(
        case_dir="Series.str/match",
        function="pandas.Series.str.match",
        doc_label="pandas.Series.str.match",
        doc_path=DOWNLOADED_DOCS_DIR / "pandas.Series.str.match.md",
        patch_targets=("pandas.core.strings.accessor.StringMethods.match",),
    ),
)


API_BY_FUNCTION = {case.function: case for case in API_CASES}


DOC_MUTANT_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "pandas.DataFrame.groupby": [
        {
            "id": "force_sort_true",
            "title": "Ignore sort=False",
            "doc_anchor": "sortbool, default True",
            "description": "Force group keys to be sorted even when the caller requests sort=False.",
        },
        {
            "id": "force_dropna_true",
            "title": "Ignore dropna=False",
            "doc_anchor": "dropnabool, default True",
            "description": "Drop NA groups even when the caller requests dropna=False.",
        },
    ],
    "pandas.DataFrame.reindex": [
        {
            "id": "ignore_fill_value",
            "title": "Ignore fill_value",
            "doc_anchor": "fill_value scalar, default np.nan",
            "description": "Remove caller-specified fill_value and fall back to NaN-filled entries.",
        },
        {
            "id": "raise_multi_column_string_fill",
            "title": "Crash on multi-column string fill",
            "doc_anchor": "Conform DataFrame to new index with optional filling logic",
            "description": "Raise a TypeError when reindexing columns with a string fill_value and two or more missing output columns.",
        },
    ],
    "pandas.DataFrame.to_json": [
        {
            "id": "force_ascii_true",
            "title": "Ignore force_ascii=False",
            "doc_anchor": "force_ascii bool, default True",
            "description": "Serialize with ASCII escaping even when the caller disables it.",
        },
        {
            "id": "corrupt_epoch_date_unit",
            "title": "Break epoch unit scaling",
            "doc_anchor": "date_unit controls timestamp unit",
            "description": "Corrupt epoch-based datetime units by scaling numeric timestamps incorrectly.",
        },
    ],
    "pandas.Index.astype": [
        {
            "id": "ignore_dtype_change",
            "title": "Ignore requested dtype",
            "doc_anchor": "dtype str or dtype",
            "description": "Return the original index unchanged when the caller asks for a different dtype.",
        },
        {
            "id": "copy_false_still_copies",
            "title": "Ignore copy=False identity",
            "doc_anchor": "copy bool, default True",
            "description": "Return a distinct object even when astype is a no-op with copy=False.",
        },
    ],
    "pandas.Index.shift": [
        {
            "id": "negate_periods",
            "title": "Reverse shift direction",
            "doc_anchor": "periods int, can be positive or negative",
            "description": "Apply the opposite shift direction from the caller-provided periods.",
        },
        {
            "id": "ignore_explicit_freq",
            "title": "Ignore explicit freq",
            "doc_anchor": "freq DateOffset, timedelta, or str, optional",
            "description": "Discard an explicit freq argument and reuse implicit index frequency behavior instead.",
        },
    ],
    "pandas.Series.factorize": [
        {
            "id": "force_sort_false",
            "title": "Ignore sort=True",
            "doc_anchor": "sort bool, default False",
            "description": "Force first-occurrence order even when the caller requests sort=True.",
        },
        {
            "id": "force_use_na_sentinel_false",
            "title": "Ignore use_na_sentinel=True",
            "doc_anchor": "use_na_sentinel bool, default True",
            "description": "Encode missing values as non-negative codes and include them in uniques even when the caller requests sentinel -1 behavior.",
        },
    ],
    "pandas.Series.mean": [
        {
            "id": "force_skipna_false",
            "title": "Ignore skipna=True",
            "doc_anchor": "skipna bool, default True",
            "description": "Treat the operation as skipna=False, so missing values poison the result.",
        },
        {
            "id": "bias_numeric_result",
            "title": "Bias numeric result",
            "doc_anchor": "Return the mean of the values",
            "description": "Return a numerically biased mean while keeping the return type otherwise plausible.",
        },
    ],
    "pandas.Series.mul": [
        {
            "id": "swap_mul_for_add",
            "title": "Use addition instead of multiplication",
            "doc_anchor": "Return multiplication of series and other",
            "description": "Delegate to addition semantics instead of multiplication semantics.",
        },
        {
            "id": "ignore_fill_value",
            "title": "Ignore fill_value",
            "doc_anchor": "fill_value float or None",
            "description": "Drop fill_value handling so one-sided missing values remain missing.",
        },
    ],
    "pandas.Series.str.contains": [
        {
            "id": "regex_false_acts_like_regex_true",
            "title": "Ignore regex=False",
            "doc_anchor": "regex bool, default True",
            "description": "Treat literal matching requests as regex matching.",
        },
        {
            "id": "case_false_acts_like_case_true",
            "title": "Ignore case=False",
            "doc_anchor": "case bool, default True",
            "description": "Make case-insensitive searches behave as case-sensitive.",
        },
    ],
    "pandas.Series.str.match": [
        {
            "id": "match_acts_like_contains",
            "title": "Use search semantics instead of match semantics",
            "doc_anchor": "Determine if each string starts with a match",
            "description": "Delegate to contains-style search behavior rather than start-anchored match behavior.",
        },
        {
            "id": "case_false_acts_like_case_true",
            "title": "Ignore case=False",
            "doc_anchor": "case bool, default True",
            "description": "Make case-insensitive match requests behave as case-sensitive.",
        },
    ],
}


ISSUE_TRIGGER_HINTS: dict[int, dict[str, str]] = {
    59965: {
        "summary": "nullable/floating mean with missing values and skipna handling",
        "baseline": "The baseline suite exercises skipna but does not target nullable FloatingArray conversions that triggered the report.",
        "ir_generated": "The IR suite exercises skipna and order invariants, but it also stays on ordinary float/bool Series instead of nullable FloatingArray inputs.",
    },
    61099: {
        "summary": "object-index Series compared against string-dtype Index",
        "baseline": "The baseline astype suite checks dtype conversion and copy behavior, not downstream Series comparison across object vs string indexes.",
        "ir_generated": "The IR astype suite checks dtype conversion and identity semantics, not Series comparison behavior after astype('string').",
    },
    61356: {
        "summary": "categorical groupby with NaN and dropna=False",
        "baseline": "The baseline groupby suite does not specifically combine categorical keys, NaN values, dropna=False, and .groups inspection.",
        "ir_generated": "The IR groupby suite covers dropna=False and categorical grouping, but it validates aggregations rather than the reported .groups failure path.",
    },
    62094: {
        "summary": "computed freq-less TimedeltaIndex shifted by a nonzero period",
        "baseline": "The baseline shift suite mostly exercises ordinary frequency-aware indexes and generic shift properties.",
        "ir_generated": "The IR shift suite checks freqless indexes, but it expects NullFrequencyError for nonzero freqless TimedeltaIndex shifts rather than the reported arithmetic regression setup.",
    },
    62240: {
        "summary": "compiled regex with flags on str.match/str.contains",
        "baseline": "The baseline string-method suite focuses on plain strings and basic regex/literal behavior, not compiled patterns with flags.",
        "ir_generated": "The IR string-method suite stresses regex behavior, but it does not construct compiled regex objects with embedded flags.",
    },
    62595: {
        "summary": "string Series multiplied by booleans on arrow-backed strings",
        "baseline": "The baseline mul suite is numeric, so it never exercises string backends or bool-string multiplication semantics.",
        "ir_generated": "The IR mul suite is also numeric-only and does not probe string backends or bool operands.",
    },
    62888: {
        "summary": "object Series mixing 0/1 with False/True",
        "baseline": "The baseline factorize suite never mixes ints and bools in the same object Series, so the hash/equality collision bug is outside its input space.",
        "ir_generated": "The IR factorize suite focuses on strings, missing values, and categoricals; it never generates the mixed int/bool object values that reproduce the bug.",
    },
    63236: {
        "summary": "non-nanosecond TimedeltaIndex column labels serialized to JSON",
        "baseline": "The baseline to_json suite checks general orient and JSON shape behavior, not non-ns TimedeltaIndex unit preservation.",
        "ir_generated": "The IR to_json suite checks epoch scaling for datetime values, but not TimedeltaIndex column-label unit serialization.",
    },
    63993: {
        "summary": "column reindex with string fill_value and multiple new output columns",
        "baseline": "The baseline reindex suite uses numeric fill_value and separate column-structure checks, so it misses the multi-column string-fill crash path.",
        "ir_generated": "The IR reindex suite uses a string fill_value for row insertion and column reindexing separately, but not the reported column reindex plus string fill_value combination with multiple new columns.",
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_counted_case_docs() -> list[dict[str, Any]]:
    return load_json(COUNTED_CASE_DOCS_PATH)


def load_bug_inventory() -> dict[int, dict[str, Any]]:
    inventory = load_json(BUG_INVENTORY_PATH)
    return {int(case["issue"]): case for case in inventory["cases"]}


def get_api_case(case_dir: str) -> ApiCase:
    for case in API_CASES:
        if case.case_dir == case_dir:
            return case
    raise KeyError(case_dir)


def get_issue_entries_for_case(case: ApiCase) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in load_counted_case_docs():
        labels = {doc["label"] for doc in entry.get("documentation_files", [])}
        if case.doc_label in labels:
            entries.append(entry)
    return entries


def import_attr(target: str) -> tuple[Any, str]:
    module_name, attr_path = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return module, attr_path


def resolve_patch_target(target: str) -> tuple[Any, str]:
    parts = target.split(".")
    module = importlib.import_module(".".join(parts[:-2]))
    owner = getattr(module, parts[-2])
    return owner, parts[-1]


def baseline_suite_name(path: Path) -> str:
    return "baseline" if path.name == "baseline_test.py" else "ir_generated"


def run_pytest_file(test_file: Path) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "pytest", str(test_file), "-q", "--maxfail=1"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def run_mutant_harness(
    harness_script: Path,
    *,
    function: str,
    mutant_id: str,
    test_file: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(harness_script),
        "--run-single-mutant",
        "--function",
        function,
        "--mutant-id",
        mutant_id,
        "--test-file",
        str(test_file),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def winner_label(left: float, right: float) -> str:
    if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        return "tie"
    return "baseline" if left > right else "ir_generated"


def issue_hint(issue: int, suite_name: str) -> str:
    hint = ISSUE_TRIGGER_HINTS.get(issue, {})
    return hint.get(suite_name, "No issue-specific trigger heuristic is available for this suite.")

