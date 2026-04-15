"""Mutant wrappers for pandas.Series.mean."""

from __future__ import annotations

import os

import pandas as pd

ORIGINAL_MEAN = None


def get_mutant_id():
    return os.environ.get("MUTANT_ID")


def mutant_mean_M1(self, *, axis=0, skipna=True, numeric_only=False, **kwargs):
    """M1: ignore skipna=True and poison the result with NA propagation."""
    if skipna is True:
        skipna = False
    return ORIGINAL_MEAN(
        self,
        axis=axis,
        skipna=skipna,
        numeric_only=numeric_only,
        **kwargs,
    )


def mutant_mean_M2(self, *, axis=0, skipna=True, numeric_only=False, **kwargs):
    """M2: bias numeric results while keeping the API shape intact."""
    result = ORIGINAL_MEAN(
        self,
        axis=axis,
        skipna=skipna,
        numeric_only=numeric_only,
        **kwargs,
    )
    if pd.notna(result) and isinstance(result, (int, float)):
        return result + 1.0
    return result


def install_mutants():
    global ORIGINAL_MEAN
    if ORIGINAL_MEAN is not None:
        return
    ORIGINAL_MEAN = pd.Series.mean
    mutant_id = get_mutant_id()
    if mutant_id == "M1":
        pd.Series.mean = mutant_mean_M1
    elif mutant_id == "M2":
        pd.Series.mean = mutant_mean_M2


def uninstall_mutants():
    global ORIGINAL_MEAN
    if ORIGINAL_MEAN is None:
        return
    pd.Series.mean = ORIGINAL_MEAN
    ORIGINAL_MEAN = None


MUTANT_INFO = {
    "M1": {
        "name": "force_skipna_false",
        "description": "Treat the operation as skipna=False, so missing values poison the result.",
        "doc_anchor": "skipna bool, default True",
        "expected_kill": "Tests that assert NaN values are ignored when skipna=True.",
    },
    "M2": {
        "name": "bias_numeric_result",
        "description": "Return a numerically biased mean while keeping the return type otherwise plausible.",
        "doc_anchor": "Return the mean of the values",
        "expected_kill": "Tests that assert exact or approximate mean values.",
    },
}
