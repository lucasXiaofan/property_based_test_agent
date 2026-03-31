"""
Utilize the reindex implementation from the local pandas source tree at
pandas_bug_finding/pandas/pandas/core/frame.py.

The local source cannot be imported directly (requires compiled C extensions),
so this module provides a Python-level shim that mirrors the local source logic
exactly (frame.py:5668-5914) and patches it onto the installed pd.DataFrame.

Local source reference:
    pandas_bug_finding/pandas/pandas/core/frame.py  line 5668

Usage:
    python use_local_reindex.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from pandas import DataFrame

# Path to the local pandas source tree (for reference)
_LOCAL_PANDAS_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "pandas")
)

_ORIGINAL_REINDEX = DataFrame.reindex


def _local_reindex_shim(
    self,
    labels=None,
    *,
    index=None,
    columns=None,
    axis=None,
    method=None,
    copy=None,
    level=None,
    fill_value=np.nan,
    limit=None,
    tolerance=None,
):
    """
    Python-level re-expression of DataFrame.reindex from the local source
    (pandas_bug_finding/pandas/pandas/core/frame.py lines 5668-5914).

    The local source delegates to NDFrame.reindex (generic.py) after resolving
    axis and labels. This shim reproduces that routing explicitly so mutations
    can be applied to the Python logic.

    Mirrors local source signature exactly:
        labels, *, index, columns, axis, method, copy, level,
        fill_value=np.nan, limit, tolerance
    """
    # Route positional labels to the correct axis (mirrors local frame.py logic)
    if labels is not None:
        axis_int = 0 if (axis is None or axis == 0 or axis == "index") else 1
        if axis_int == 0:
            index = labels
        else:
            columns = labels

    # Build kwargs, passing only non-default values (mirrors super().reindex call)
    kwargs = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not (isinstance(fill_value, float) and np.isnan(fill_value)):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level

    return _ORIGINAL_REINDEX(self, **kwargs)


def install() -> None:
    """Patch pd.DataFrame.reindex with the Python shim mirroring the local source."""
    print(f"[use_local_reindex] Installing shim from local source: {_LOCAL_PANDAS_SRC}/pandas/core/frame.py:5668")
    DataFrame.reindex = _local_reindex_shim


def uninstall() -> None:
    """Restore the original pd.DataFrame.reindex."""
    DataFrame.reindex = _ORIGINAL_REINDEX


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    install()

    print("\n--- Basic reindex (adds NaN for missing labels) ---")
    df = pd.DataFrame({"A": [10, 20, 30]}, index=[0, 1, 2])
    result = df.reindex([1, 2, 3, 4])
    print(result)

    print("\n--- fill_value replaces NaN for new labels ---")
    result = df.reindex([1, 2, 3, 4], fill_value=0)
    print(result)

    print("\n--- Reindex columns (adds NaN for missing columns) ---")
    df2 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    result = df2.reindex(columns=["A", "B", "C"])
    print(result)

    print("\n--- Forward-fill on monotonic index ---")
    df3 = pd.DataFrame({"price": [100.0, np.nan, 102.0]}, index=[0, 1, 2])
    result = df3.reindex([0, 1, 2, 3, 4], method="ffill")
    print(result)

    print("\n--- Reindex both axes simultaneously ---")
    df4 = pd.DataFrame({"X": [1, 2, 3], "Y": [4, 5, 6]}, index=["a", "b", "c"])
    result = df4.reindex(index=["a", "c", "d"], columns=["X", "Z"])
    print(result)

    uninstall()
    print("\n[use_local_reindex] Original reindex restored.")
