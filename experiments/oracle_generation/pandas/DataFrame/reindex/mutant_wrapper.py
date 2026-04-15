"""Mutant wrappers for pandas.DataFrame.reindex."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ORIGINAL_REINDEX = None


def get_mutant_id():
    return os.environ.get("MUTANT_ID")


def mutant_reindex_M1(
    self,
    labels=None,
    *,
    index=None,
    columns=None,
    axis=None,
    method=None,
    copy=pd.api.extensions.no_default,
    level=None,
    fill_value=np.nan,
    limit=None,
    tolerance=None,
):
    """M1: ignore a caller-specified fill_value."""
    if fill_value is not None and not pd.isna(fill_value):
        fill_value = np.nan
    return ORIGINAL_REINDEX(
        self,
        labels=labels,
        index=index,
        columns=columns,
        axis=axis,
        method=method,
        copy=copy,
        level=level,
        fill_value=fill_value,
        limit=limit,
        tolerance=tolerance,
    )


def mutant_reindex_M2(
    self,
    labels=None,
    *,
    index=None,
    columns=None,
    axis=None,
    method=None,
    copy=pd.api.extensions.no_default,
    level=None,
    fill_value=np.nan,
    limit=None,
    tolerance=None,
):
    """M2: raise on multi-column reindex with a string fill_value."""
    target_columns = columns
    if target_columns is None and axis in (1, "columns"):
        target_columns = labels
    if (
        target_columns is not None
        and isinstance(fill_value, str)
        and len([col for col in target_columns if col not in self.columns]) >= 2
    ):
        raise TypeError("mutant: string fill_value unsupported for multiple missing columns")
    return ORIGINAL_REINDEX(
        self,
        labels=labels,
        index=index,
        columns=columns,
        axis=axis,
        method=method,
        copy=copy,
        level=level,
        fill_value=fill_value,
        limit=limit,
        tolerance=tolerance,
    )


def install_mutants():
    global ORIGINAL_REINDEX
    if ORIGINAL_REINDEX is not None:
        return
    ORIGINAL_REINDEX = pd.DataFrame.reindex
    mutant_id = get_mutant_id()
    if mutant_id == "M1":
        pd.DataFrame.reindex = mutant_reindex_M1
    elif mutant_id == "M2":
        pd.DataFrame.reindex = mutant_reindex_M2


def uninstall_mutants():
    global ORIGINAL_REINDEX
    if ORIGINAL_REINDEX is None:
        return
    pd.DataFrame.reindex = ORIGINAL_REINDEX
    ORIGINAL_REINDEX = None


MUTANT_INFO = {
    "M1": {
        "name": "ignore_fill_value",
        "description": "Remove caller-specified fill_value handling and fall back to NaN.",
        "doc_anchor": "fill_value scalar, default np.nan",
        "expected_kill": "Tests that assert explicit fill_value is reflected in newly created labels.",
    },
    "M2": {
        "name": "raise_multi_column_string_fill",
        "description": "Raise a TypeError when reindexing columns with a string fill_value and multiple missing columns.",
        "doc_anchor": "Conform DataFrame to new index with optional filling logic",
        "expected_kill": "Tests that cover successful multi-column column-reindex with string fill_value.",
    },
}
