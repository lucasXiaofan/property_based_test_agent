from __future__ import annotations

import json
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "results.json"


@dataclass(frozen=True)
class CaseMetadata:
    issue: int
    title: str
    url: str
    counted_in_valid_set: bool
    confirmation_signal: str
    input_summary: str
    hypothesis_sketch: str


@dataclass(frozen=True)
class CaseResult:
    status: str
    detail: str


def bug(detail: str) -> CaseResult:
    return CaseResult("bug_reproduced", detail)


def not_bug(detail: str) -> CaseResult:
    return CaseResult("not_reproduced", detail)


def run_case_58190() -> CaseResult:
    matrix = pd.DataFrame(
        [[0.0, 0.5, 0.0], [0.1, 0.0, 0.2], [0.2, 0.0, 0.0]]
    )
    mask_source = pd.Series([1.0, 1.0, np.nan])
    try:
        result = matrix.where(mask_source.notna(), axis=1)
    except Exception as exc:  # pragma: no cover - this is the bug path
        return bug(f"{type(exc).__name__}: {exc}")
    return not_bug(f"returned columns={result.columns.tolist()}")


def run_case_58471() -> CaseResult:
    indexes = [
        pd.date_range("2024-01-01", periods=24 * 12, freq="5min", unit="us")
        + pd.Timedelta(days=offset)
        for offset in range(3)
    ]
    series_list = [pd.Series(range(len(idx)), index=idx) for idx in indexes]
    result = pd.concat(series_list, axis=1)
    if len(result.index) != 24 * 12 * 3:
        return bug(f"len={len(result.index)} expected=864")
    return not_bug(
        f"len={len(result.index)} tail={list(map(str, result.index[-3:]))}"
    )


def run_case_59965() -> CaseResult:
    left = pd.Series({"a": 0.0, "b": 1, "c": 1, "d": 0})
    right = pd.Series({"a": 0.0, "b": 2, "c": 2, "d": 2})
    result = left.convert_dtypes() / right.convert_dtypes()
    mean_value = result.mean(skipna=True)
    if pd.isna(mean_value):
        return bug(f"mean={mean_value!r} dtype={result.dtype}")
    return not_bug(f"mean={mean_value!r} dtype={result.dtype}")


def run_case_60922() -> CaseResult:
    idx = pd.date_range("2025-01-29 01:36", periods=4, freq="1min", unit="us")
    left = pd.DataFrame(index=idx, data={"a": [1, 2, 3, 4], "b": [2, 2, 2, 2]})
    right = pd.DataFrame(index=idx[:3], data={"c": [9, 8, 7], "d": [6, 6, 6]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pd.concat([left, right], axis="columns")
    if result.shape[0] != 4:
        return bug(f"shape={result.shape}")
    return not_bug(f"shape={result.shape} index={list(map(str, result.index))}")


def run_case_61099() -> CaseResult:
    left = pd.Series([1, 2, 3], index=["a", "b", "c"])
    right = pd.Series([4, 5, 6], index=["a", "b", "c"])
    right.index = right.index.astype("string")
    try:
        _ = left < right
    except Exception as exc:  # pragma: no cover - this is the bug path
        return bug(f"{type(exc).__name__}: {exc}")
    return not_bug("comparison succeeded")


def run_case_61175() -> CaseResult:
    x = pd.Series([1, 2, 3, 5])
    y = pd.Series([2, 3, 4])
    try:
        result = pd.eval("(x + y).dropna()")
    except Exception as exc:  # pragma: no cover - this is the bug path
        return bug(f"{type(exc).__name__}: {exc}")
    return not_bug(f"result={result.tolist()}")


def run_case_61356() -> CaseResult:
    frame = pd.DataFrame(
        {
            "cat": pd.Categorical(["a", np.nan, "a"], categories=["a", "b", "d"]),
            "vals": [1, 2, 3],
        }
    )
    grouped = frame.groupby("cat", observed=True, dropna=False)
    try:
        groups = grouped.groups
    except Exception as exc:  # pragma: no cover - this is the bug path
        return bug(f"{type(exc).__name__}: {exc}")
    return not_bug(f"groups={groups}")


def run_case_61509() -> CaseResult:
    frame = pd.DataFrame(
        {"i": [1, 2, 3], "g1": ["a", "b", "b"], "g2": ["x", None, None]}
    )
    table = frame.pivot_table(
        index="g1",
        columns="g2",
        values="i",
        aggfunc="count",
        dropna=False,
        margins=True,
    )
    nan_columns = [column for column in table.columns if pd.isna(column)]
    if nan_columns and pd.isna(table.loc["All", nan_columns[0]]):
        return bug(f"all_row={table.loc['All'].to_dict()}")
    return not_bug(f"all_row={table.loc['All'].to_dict()}")


def run_case_61621() -> CaseResult:
    with_pd_na = infer_dtype(pd.Series([1.0, 2.0, 0.3, pd.NA], dtype=object))
    with_nan = infer_dtype(pd.Series([1.0, 2.0, 0.3, np.nan], dtype=object))
    if with_pd_na != with_nan:
        return bug(f"with_pdNA={with_pd_na} with_nan={with_nan}")
    return not_bug(f"with_pdNA={with_pd_na} with_nan={with_nan}")


def run_case_62094() -> CaseResult:
    index = pd.date_range("1/1/2021", "1/5/2021") - pd.Timestamp("1/3/2019")
    try:
        shifted = index.shift(1)
    except Exception as exc:  # pragma: no cover - this is the bug path
        return bug(f"{type(exc).__name__}: {exc}")
    return not_bug(f"shifted_head={list(map(str, shifted[:3]))}")


def run_case_62240() -> CaseResult:
    regex = re.compile("foo", flags=re.IGNORECASE)
    values = ["Foo", "foo", "Bar", "_Foo_", "_foo_"]
    series = pd.Series(values, index=values)
    python_match = [bool(regex.match(value)) for value in values]
    python_search = [bool(regex.search(value)) for value in values]
    pandas_match = series.str.match(regex).tolist()
    pandas_contains = series.str.contains(regex).tolist()
    if pandas_match != python_match or pandas_contains != python_search:
        return bug(
            f"match={pandas_match} expected_match={python_match} "
            f"contains={pandas_contains} expected_contains={python_search}"
        )
    return not_bug(
        f"match={pandas_match} contains={pandas_contains} "
        f"matches_python=True"
    )


def run_case_62595() -> CaseResult:
    try:
        python_backend = pd.Series(["a", "b", "c"], dtype="string[python]") * True
    except Exception as exc:
        return not_bug(f"python backend also raises {type(exc).__name__}: {exc}")

    try:
        generic_backend = pd.Series(["a", "b", "c"], dtype="string") * True
    except Exception as exc:  # pragma: no cover - this is the bug path
        return bug(
            f"generic backend raises {type(exc).__name__}: {exc}; "
            f"python backend={python_backend.tolist()}"
        )
    return not_bug(f"generic backend={generic_backend.tolist()}")


def run_case_62778() -> CaseResult:
    frame = pd.DataFrame({"A": range(5), "B": range(5)})
    try:
        result = frame.groupby(["A"]).mean(["B"])
    except Exception as exc:
        return not_bug(f"now raises {type(exc).__name__}: {exc}")
    return bug(f"accepted non-bool numeric_only, shape={result.shape}")


def run_case_62829() -> CaseResult:
    data = {
        1: {"id": 10, "status": "AVAL"},
        2: {"id": 30, "status": "AVAL", "items": {"id": 12, "size": 20}},
        3: {"id": 50, "status": "AVAL", "items": {"id": 13, "size": 30}},
    }
    frame = pd.DataFrame.from_dict(data, orient="index")
    try:
        result = pd.json_normalize(frame["items"].tolist(), max_level=0)
    except Exception as exc:
        if isinstance(exc, AttributeError):
            return bug(f"{type(exc).__name__}: {exc}")
        return not_bug(f"now raises {type(exc).__name__}: {exc}")
    return not_bug(f"columns={list(result.columns)}")


def run_case_62888() -> CaseResult:
    uniques = pd.Series([0, 1, True, False]).factorize()[1]
    if len(uniques) == 2:
        return bug(f"uniques={list(uniques)}")
    return not_bug(f"uniques={list(uniques)}")


def run_case_63236() -> CaseResult:
    frame = pd.DataFrame([[1]], columns=[pd.Timedelta("1D").as_unit("us")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso = frame.to_json(date_format="iso")
        epoch = frame.to_json(date_format="epoch")
    expected_iso = '{"P1DT0H0M0S":{"0":1}}'
    expected_epoch = '{"86400000":{"0":1}}'
    if iso != expected_iso or epoch != expected_epoch:
        return bug(f"iso={iso} epoch={epoch}")
    return not_bug(f"iso={iso} epoch={epoch}")


def run_case_63262() -> CaseResult:
    series = pd.Series(1, index=pd.date_range(start="2000-01-01", freq="h", periods=8))
    start = pd.Timestamp("2000-01-01 01:00:00")
    stop = start + pd.Timedelta(1)
    try:
        result = series.loc[start:stop]
    except Exception as exc:  # pragma: no cover - this is the bug path
        return bug(f"{type(exc).__name__}: {exc}")
    return not_bug(f"len={len(result)} index={list(map(str, result.index))}")


def run_case_63306() -> CaseResult:
    codes = pd.Index([0, 1, 2, 3], dtype="int8").to_numpy()
    categories = pd.Index(["a", "b", "c", "d"])
    data = pd.Categorical.from_codes(codes, categories)
    series = pd.Series(pd.Index(data))
    try:
        series[[False, False, True, True]] = categories[2:4]
    except Exception as exc:  # pragma: no cover - this is the bug path
        return bug(f"{type(exc).__name__}: {exc}")
    return not_bug(f"series={series.tolist()}")


def run_case_63581() -> CaseResult:
    frame = pd.DataFrame(
        {
            "id": ["A", "B"],
            "arr": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        }
    )
    frame["sparse"] = pd.arrays.SparseArray([1, 1], fill_value=0)
    try:
        row = frame[frame["id"] == "A"].iloc[0]
    except Exception as exc:  # pragma: no cover - this is the bug path
        return bug(f"{type(exc).__name__}: {exc}")
    return not_bug(f"row_keys={list(row.index)}")


def run_case_63879() -> CaseResult:
    masked = np.ma.array([1, 2, 3, 4], mask=[False, True, False, True])
    result = pd.array(masked)
    if all(value is not pd.NA for value in result):
        return bug(f"array={result.tolist()} dtype={result.dtype}")
    return not_bug(f"array={result.tolist()} dtype={result.dtype}")


def run_case_63993() -> CaseResult:
    try:
        result = pd.DataFrame({"a": [0]}).reindex(
            columns=["a", "b", "c"], fill_value="missing"
        )
    except Exception as exc:  # pragma: no cover - this is the bug path
        return bug(f"{type(exc).__name__}: {exc}")
    return not_bug(f"values={result.to_dict(orient='list')}")


VALID_CASES: list[tuple[CaseMetadata, Callable[[], CaseResult]]] = [
    (
        CaseMetadata(
            issue=58471,
            title="concat on non-ns DatetimeIndex drops rows",
            url="https://github.com/pandas-dev/pandas/issues/58471",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comment by rhshadrach while triaging the failing example."
            ),
            input_summary=(
                "Concatenate several non-overlapping Series whose DatetimeIndex uses a "
                "non-nanosecond unit such as us."
            ),
            hypothesis_sketch=(
                "st.integers(min_value=2, max_value=4).map("
                "lambda days: [pd.Series(range(288), index=pd.date_range("
                "'2024-01-01', periods=288, freq='5min', unit='us') + "
                "pd.Timedelta(days=i)) for i in range(days)])"
            ),
        ),
        run_case_58471,
    ),
    (
        CaseMetadata(
            issue=59965,
            title="FloatingArray reductions do not skip NaN correctly",
            url="https://github.com/pandas-dev/pandas/issues/59965",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comments by rhshadrach and jorisvandenbossche discuss the "
                "buggy FloatingArray reduce path."
            ),
            input_summary=(
                "Create nullable floating results that mix ordinary floats, division by "
                "zero, and NaN, then call a reduction with skipna=True."
            ),
            hypothesis_sketch=(
                "st.lists(st.one_of(st.floats(allow_nan=False, allow_infinity=False), "
                "st.integers()), min_size=4, max_size=4).map("
                "lambda xs: pd.Series(xs).convert_dtypes())"
            ),
        ),
        run_case_59965,
    ),
    (
        CaseMetadata(
            issue=60922,
            title="concat misaligns non-ns DatetimeIndex columns",
            url="https://github.com/pandas-dev/pandas/issues/60922",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comment by rhshadrach notes the bug could be reproduced on "
                "main while investigating."
            ),
            input_summary=(
                "Concatenate frames on axis=1 where both sides share a non-ns "
                "DatetimeIndex and one side is a strict prefix of the other."
            ),
            hypothesis_sketch=(
                "st.integers(min_value=3, max_value=8).map("
                "lambda n: (pd.date_range('2025-01-29 01:36', periods=n, freq='1min', "
                "unit='us'), n - 1))"
            ),
        ),
        run_case_60922,
    ),
    (
        CaseMetadata(
            issue=61099,
            title="Series comparison fails for object-index vs string-index",
            url="https://github.com/pandas-dev/pandas/issues/61099",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comment by rhshadrach discusses the failing infer-string case."
            ),
            input_summary=(
                "Compare two otherwise identical Series after converting one index to the "
                "nullable string dtype."
            ),
            hypothesis_sketch=(
                "st.lists(st.text(min_size=1), min_size=1, max_size=5, unique=True).map("
                "lambda labels: (pd.Series(range(len(labels)), index=labels), "
                "pd.Index(labels).astype('string')))"
            ),
        ),
        run_case_61099,
    ),
    (
        CaseMetadata(
            issue=61175,
            title="pd.eval crashes on method call after binary op",
            url="https://github.com/pandas-dev/pandas/issues/61175",
            counted_in_valid_set=True,
            confirmation_signal="Maintainer comment by snitish: 'Confirmed on main.'",
            input_summary=(
                "Build two misaligned Series, combine them in pd.eval, and then call a "
                "Series method like dropna() on the BinOp result."
            ),
            hypothesis_sketch=(
                "st.tuples(st.lists(st.integers(), min_size=2, max_size=6), "
                "st.lists(st.integers(), min_size=1, max_size=5)).map("
                "lambda pair: (pd.Series(pair[0]), pd.Series(pair[1])))"
            ),
        ),
        run_case_61175,
    ),
    (
        CaseMetadata(
            issue=61356,
            title="groupby.groups fails with categorical + NaN + dropna=False",
            url="https://github.com/pandas-dev/pandas/issues/61356",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comment by rhshadrach: 'Confirmed on main. PR to fix is up.'"
            ),
            input_summary=(
                "Group by a categorical column whose values include NaN and ask for "
                "groups with observed=True, dropna=False."
            ),
            hypothesis_sketch=(
                "st.just(pd.DataFrame({'cat': pd.Categorical(['a', np.nan, 'a'], "
                "categories=['a', 'b', 'd']), 'vals': [1, 2, 3]}))"
            ),
        ),
        run_case_61356,
    ),
    (
        CaseMetadata(
            issue=61509,
            title="pivot_table margins omit NaN bucket totals",
            url="https://github.com/pandas-dev/pandas/issues/61509",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comment by rhshadrach: 'Confirmed on main...'"
            ),
            input_summary=(
                "Count with pivot_table using margins=True when one grouping key has "
                "missing values."
            ),
            hypothesis_sketch=(
                "st.just(pd.DataFrame({'i': [1, 2, 3], 'g1': ['a', 'b', 'b'], "
                "'g2': ['x', None, None]}))"
            ),
        ),
        run_case_61509,
    ),
    (
        CaseMetadata(
            issue=61621,
            title="infer_dtype diverges on pd.NA vs np.nan in object arrays",
            url="https://github.com/pandas-dev/pandas/issues/61621",
            counted_in_valid_set=True,
            confirmation_signal="Maintainer comment by arthurlw: 'Confirmed on main!'",
            input_summary=(
                "Call infer_dtype on an object Series containing floats plus pd.NA, then "
                "compare it to the same data with np.nan."
            ),
            hypothesis_sketch=(
                "st.lists(st.floats(allow_nan=False, allow_infinity=False), "
                "min_size=1, max_size=5).map("
                "lambda xs: (pd.Series(xs + [pd.NA], dtype=object), "
                "pd.Series(xs + [np.nan], dtype=object)))"
            ),
        ),
        run_case_61621,
    ),
    (
        CaseMetadata(
            issue=62094,
            title="TimedeltaIndex.shift regressed on computed freq-less indexes",
            url="https://github.com/pandas-dev/pandas/issues/62094",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comment by jbrockmendel identifies the missing arithmetic "
                "frequency propagation."
            ),
            input_summary=(
                "Construct a TimedeltaIndex by subtracting a Timestamp from a date range, "
                "then call shift(1)."
            ),
            hypothesis_sketch=(
                "st.dates(min_value=pd.Timestamp('2000-01-01').date(), "
                "max_value=pd.Timestamp('2030-01-01').date()).map("
                "lambda d: pd.date_range(d, periods=5) - pd.Timestamp('2019-01-03'))"
            ),
        ),
        run_case_62094,
    ),
    (
        CaseMetadata(
            issue=62240,
            title="Compiled regex handling in str.match/str.contains is inconsistent",
            url="https://github.com/pandas-dev/pandas/issues/62240",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer discussion by rhshadrach and jorisvandenbossche, including a "
                "linked PR for the regression."
            ),
            input_summary=(
                "Use a compiled regex with flags against Series.str.match and "
                "Series.str.contains."
            ),
            hypothesis_sketch=(
                "st.just((re.compile('foo', flags=re.IGNORECASE), "
                "pd.Series(['Foo', 'foo', 'Bar', '_Foo_', '_foo_'])))"
            ),
        ),
        run_case_62240,
    ),
    (
        CaseMetadata(
            issue=62595,
            title="Arrow-backed string Series multiply behaves differently from python strings",
            url="https://github.com/pandas-dev/pandas/issues/62595",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer discussion by jbrockmendel on expected behavior for bool * "
                "string arrays."
            ),
            input_summary=(
                "Multiply a string Series by True and compare python-backed strings "
                "against the generic string dtype."
            ),
            hypothesis_sketch=(
                "st.just(pd.Series(['a', 'b', 'c']))"
            ),
        ),
        run_case_62595,
    ),
    (
        CaseMetadata(
            issue=62778,
            title="groupby reductions accept non-bool numeric_only",
            url="https://github.com/pandas-dev/pandas/issues/62778",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer-authored issue by mroeschke."
            ),
            input_summary=(
                "Pass a truthy non-bool object to a GroupBy reduction's numeric_only "
                "parameter."
            ),
            hypothesis_sketch=(
                "st.sampled_from([['B'], [1], 'yes']).map("
                "lambda bad: (pd.DataFrame({'A': range(5), 'B': range(5)}), bad))"
            ),
        ),
        run_case_62778,
    ),
    (
        CaseMetadata(
            issue=62829,
            title="json_normalize with max_level mishandles NaN entries",
            url="https://github.com/pandas-dev/pandas/issues/62829",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comment by rhshadrach discusses the invalid mixed dict/NaN "
                "input and expected error handling."
            ),
            input_summary=(
                "Call json_normalize(max_level=0) on a list that mixes nested dicts and "
                "NaN values."
            ),
            hypothesis_sketch=(
                "st.lists(st.one_of(st.dictionaries(st.text(min_size=1), st.integers()), "
                "st.just(np.nan)), min_size=2, max_size=5)"
            ),
        ),
        run_case_62829,
    ),
    (
        CaseMetadata(
            issue=62888,
            title="factorize collapses 0/False and 1/True in object dtype",
            url="https://github.com/pandas-dev/pandas/issues/62888",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer discussion by rhshadrach and jbrockmendel on the incorrect "
                "object factorization semantics."
            ),
            input_summary=(
                "Factorize object data that mixes ints and bools with equal hashes and "
                "equal comparisons."
            ),
            hypothesis_sketch=(
                "st.permutations([0, 1, True, False]).map(lambda xs: pd.Series(list(xs)))"
            ),
        ),
        run_case_62888,
    ),
    (
        CaseMetadata(
            issue=63236,
            title="to_json stringifies non-ns TimedeltaIndex with wrong units",
            url="https://github.com/pandas-dev/pandas/issues/63236",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comments by WillAyd, jbrockmendel, and jorisvandenbossche, "
                "with a linked fix PR."
            ),
            input_summary=(
                "Serialize a frame whose column index is a non-nanosecond Timedelta."
            ),
            hypothesis_sketch=(
                "st.sampled_from(['us', 'ms', 's']).map("
                "lambda unit: pd.DataFrame([[1]], columns=[pd.Timedelta('1D').as_unit(unit)]))"
            ),
        ),
        run_case_63236,
    ),
    (
        CaseMetadata(
            issue=63262,
            title="Datetime slicing fails with mixed timestamp units",
            url="https://github.com/pandas-dev/pandas/issues/63262",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comments by jorisvandenbossche and jbrockmendel on the slice "
                "bound casting bug."
            ),
            input_summary=(
                "Slice a DatetimeIndex-backed Series with start/stop Timestamps that have "
                "different internal units."
            ),
            hypothesis_sketch=(
                "st.just((pd.Series(1, index=pd.date_range('2000-01-01', periods=8, "
                "freq='h')), pd.Timestamp('2000-01-01 01:00:00')))"
            ),
        ),
        run_case_63262,
    ),
    (
        CaseMetadata(
            issue=63306,
            title="CoW write path fails on read-only categorical backing data",
            url="https://github.com/pandas-dev/pandas/issues/63306",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comments by mroeschke and jorisvandenbossche generalize the "
                "read-only array problem."
            ),
            input_summary=(
                "Construct a Series from an Index backed by read-only categorical codes "
                "and then assign through a boolean mask in CoW mode."
            ),
            hypothesis_sketch=(
                "st.just((pd.Index([0, 1, 2, 3], dtype='int8').to_numpy(), "
                "pd.Index(['a', 'b', 'c', 'd'])))"
            ),
        ),
        run_case_63306,
    ),
    (
        CaseMetadata(
            issue=63581,
            title="iloc[0] fails on rows mixing SparseArray and ndarray objects",
            url="https://github.com/pandas-dev/pandas/issues/63581",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comment by rhshadrach acknowledges the bug and leaves it open "
                "for pandas 3.x."
            ),
            input_summary=(
                "Take a row from a filtered DataFrame that contains both ndarray-valued "
                "cells and a SparseArray column."
            ),
            hypothesis_sketch=(
                "st.just(pd.DataFrame({'id': ['A', 'B'], 'arr': [np.array([1.0, 2.0]), "
                "np.array([3.0, 4.0])]}))"
            ),
        ),
        run_case_63581,
    ),
    (
        CaseMetadata(
            issue=63879,
            title="pd.array ignores masks on numpy masked arrays",
            url="https://github.com/pandas-dev/pandas/issues/63879",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comment by rhshadrach explains that Series/DataFrame handled "
                "the mask but pd.array did not."
            ),
            input_summary=(
                "Pass a numpy masked array to pd.array and check whether masked elements "
                "become missing values."
            ),
            hypothesis_sketch=(
                "st.lists(st.integers(), min_size=2, max_size=6).flatmap("
                "lambda xs: st.tuples(st.just(xs), st.lists(st.booleans(), "
                "min_size=len(xs), max_size=len(xs))))"
            ),
        ),
        run_case_63879,
    ),
    (
        CaseMetadata(
            issue=63993,
            title="DataFrame.reindex crashes with multi-column string fill_value",
            url="https://github.com/pandas-dev/pandas/issues/63993",
            counted_in_valid_set=True,
            confirmation_signal=(
                "Maintainer comment by rhshadrach: 'confirmed on main.'"
            ),
            input_summary=(
                "Reindex a one-column DataFrame to multiple columns using a string "
                "fill_value."
            ),
            hypothesis_sketch=(
                "st.lists(st.text(min_size=1), min_size=2, max_size=5, unique=True).map("
                "lambda extra: (pd.DataFrame({'a': [0]}), ['a'] + extra, 'missing'))"
            ),
        ),
        run_case_63993,
    ),
]


SUPPLEMENTAL_CASES: list[tuple[CaseMetadata, Callable[[], CaseResult]]] = [
    (
        CaseMetadata(
            issue=58190,
            title="DataFrame.where(..., axis=1) crashes on Series mask",
            url="https://github.com/pandas-dev/pandas/issues/58190",
            counted_in_valid_set=False,
            confirmation_signal=(
                "Bug-labeled issue with a clear reproducer, but the fetched thread did "
                "not include an explicit maintainer confirmation comment."
            ),
            input_summary=(
                "Apply DataFrame.where with axis=1 using a 1D boolean mask derived from a "
                "Series that contains NaN."
            ),
            hypothesis_sketch=(
                "st.just((pd.DataFrame([[0.0, 0.5, 0.0], [0.1, 0.0, 0.2], "
                "[0.2, 0.0, 0.0]]), pd.Series([1.0, 1.0, np.nan])))"
            ),
        ),
        run_case_58190,
    ),
]


def execute_case(case: CaseMetadata, runner: Callable[[], CaseResult]) -> dict[str, object]:
    try:
        result = runner()
    except Exception as exc:  # pragma: no cover - harness guard
        result = CaseResult("harness_error", f"{type(exc).__name__}: {exc}")
    return {
        "issue": case.issue,
        "title": case.title,
        "url": case.url,
        "counted_in_valid_set": case.counted_in_valid_set,
        "confirmation_signal": case.confirmation_signal,
        "input_summary": case.input_summary,
        "hypothesis_sketch": case.hypothesis_sketch,
        "status": result.status,
        "detail": result.detail,
    }


def main() -> None:
    rows = [execute_case(case, runner) for case, runner in VALID_CASES + SUPPLEMENTAL_CASES]
    summary = {
        "pandas_version": pd.__version__,
        "valid_cases": sum(1 for row in rows if row["counted_in_valid_set"]),
        "valid_reproduced": sum(
            1
            for row in rows
            if row["counted_in_valid_set"] and row["status"] == "bug_reproduced"
        ),
        "supplemental_cases": sum(1 for row in rows if not row["counted_in_valid_set"]),
        "supplemental_reproduced": sum(
            1
            for row in rows
            if not row["counted_in_valid_set"] and row["status"] == "bug_reproduced"
        ),
    }
    payload = {"summary": summary, "cases": rows}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
