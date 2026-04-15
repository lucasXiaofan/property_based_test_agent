"""Mutant wrappers for pandas.Series.factorize."""

from __future__ import annotations

import os

import pandas as pd

ORIGINAL_FACTORIZE = None


def get_mutant_id():
    return os.environ.get("MUTANT_ID")


def mutant_factorize_M1(self, sort=False, use_na_sentinel=True):
    """M1: ignore sort=True and keep first-occurrence order."""
    if sort is True:
        sort = False
    return ORIGINAL_FACTORIZE(self, sort=sort, use_na_sentinel=use_na_sentinel)


def mutant_factorize_M2(self, sort=False, use_na_sentinel=True):
    """M2: ignore use_na_sentinel=True and emit non-negative NA codes."""
    if use_na_sentinel is True:
        use_na_sentinel = False
    return ORIGINAL_FACTORIZE(self, sort=sort, use_na_sentinel=use_na_sentinel)


def install_mutants():
    global ORIGINAL_FACTORIZE
    if ORIGINAL_FACTORIZE is not None:
        return
    ORIGINAL_FACTORIZE = pd.Series.factorize
    mutant_id = get_mutant_id()
    if mutant_id == "M1":
        pd.Series.factorize = mutant_factorize_M1
    elif mutant_id == "M2":
        pd.Series.factorize = mutant_factorize_M2


def uninstall_mutants():
    global ORIGINAL_FACTORIZE
    if ORIGINAL_FACTORIZE is None:
        return
    pd.Series.factorize = ORIGINAL_FACTORIZE
    ORIGINAL_FACTORIZE = None


MUTANT_INFO = {
    "M1": {
        "name": "force_sort_false",
        "description": "Force first-occurrence order even when the caller requests sort=True.",
        "doc_anchor": "sort bool, default False",
        "expected_kill": "Tests that assert sort=True changes uniques order and codes accordingly.",
    },
    "M2": {
        "name": "force_use_na_sentinel_false",
        "description": "Encode missing values as ordinary categories even when the caller requests sentinel -1 behavior.",
        "doc_anchor": "use_na_sentinel bool, default True",
        "expected_kill": "Tests that assert NA handling for codes and uniques with use_na_sentinel=True.",
    },
}
