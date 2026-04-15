"""Mutant wrappers for pandas.DataFrame.groupby."""

from __future__ import annotations

import os

import pandas as pd

ORIGINAL_GROUPBY = None


def get_mutant_id():
    return os.environ.get("MUTANT_ID")


def mutant_groupby_M1(
    self,
    by=None,
    level=None,
    *,
    as_index=True,
    sort=True,
    group_keys=True,
    observed=True,
    dropna=True,
):
    """M1: ignore sort=False and always sort group keys."""
    if sort is False:
        sort = True
    return ORIGINAL_GROUPBY(
        self,
        by=by,
        level=level,
        as_index=as_index,
        sort=sort,
        group_keys=group_keys,
        observed=observed,
        dropna=dropna,
    )


def mutant_groupby_M2(
    self,
    by=None,
    level=None,
    *,
    as_index=True,
    sort=True,
    group_keys=True,
    observed=True,
    dropna=True,
):
    """M2: ignore dropna=False and always drop NA groups."""
    if dropna is False:
        dropna = True
    return ORIGINAL_GROUPBY(
        self,
        by=by,
        level=level,
        as_index=as_index,
        sort=sort,
        group_keys=group_keys,
        observed=observed,
        dropna=dropna,
    )


def install_mutants():
    global ORIGINAL_GROUPBY
    if ORIGINAL_GROUPBY is not None:
        return
    ORIGINAL_GROUPBY = pd.DataFrame.groupby
    mutant_id = get_mutant_id()
    if mutant_id == "M1":
        pd.DataFrame.groupby = mutant_groupby_M1
    elif mutant_id == "M2":
        pd.DataFrame.groupby = mutant_groupby_M2


def uninstall_mutants():
    global ORIGINAL_GROUPBY
    if ORIGINAL_GROUPBY is None:
        return
    pd.DataFrame.groupby = ORIGINAL_GROUPBY
    ORIGINAL_GROUPBY = None


MUTANT_INFO = {
    "M1": {
        "name": "force_sort_true",
        "description": "Force group keys to be sorted even when the caller requests sort=False.",
        "doc_anchor": "sort bool, default True",
        "expected_kill": "Tests that assert group order changes when sort=False.",
    },
    "M2": {
        "name": "force_dropna_true",
        "description": "Drop NA groups even when the caller requests dropna=False.",
        "doc_anchor": "dropna bool, default True",
        "expected_kill": "Tests that assert NA-key groups are retained with dropna=False.",
    },
}
