"""Mutant wrappers for pandas.Index.astype."""

from __future__ import annotations

import os

import pandas as pd

ORIGINAL_ASTYPE = None


def get_mutant_id():
    return os.environ.get("MUTANT_ID")


def mutant_astype_M1(self, dtype, copy=True):
    """M1: ignore requested dtype changes."""
    requested = pd.api.types.pandas_dtype(dtype)
    if requested != self.dtype:
        return self.copy(deep=True) if copy else self
    return ORIGINAL_ASTYPE(self, dtype=dtype, copy=copy)


def mutant_astype_M2(self, dtype, copy=True):
    """M2: force copying on no-op astype(copy=False)."""
    requested = pd.api.types.pandas_dtype(dtype)
    if requested == self.dtype and copy is False:
        return ORIGINAL_ASTYPE(self, dtype=dtype, copy=True)
    return ORIGINAL_ASTYPE(self, dtype=dtype, copy=copy)


def install_mutants():
    global ORIGINAL_ASTYPE
    if ORIGINAL_ASTYPE is not None:
        return
    ORIGINAL_ASTYPE = pd.Index.astype
    mutant_id = get_mutant_id()
    if mutant_id == "M1":
        pd.Index.astype = mutant_astype_M1
    elif mutant_id == "M2":
        pd.Index.astype = mutant_astype_M2


def uninstall_mutants():
    global ORIGINAL_ASTYPE
    if ORIGINAL_ASTYPE is None:
        return
    pd.Index.astype = ORIGINAL_ASTYPE
    ORIGINAL_ASTYPE = None


MUTANT_INFO = {
    "M1": {
        "name": "ignore_dtype_change",
        "description": "Return the original index unchanged when the caller asks for a different dtype.",
        "doc_anchor": "dtype str or dtype",
        "expected_kill": "Tests that assert dtype conversion and converted values.",
    },
    "M2": {
        "name": "copy_false_still_copies",
        "description": "Return a distinct object even when astype is a no-op with copy=False.",
        "doc_anchor": "copy bool, default True",
        "expected_kill": "Tests that assert identity semantics for no-op astype(copy=False).",
    },
}
