"""Mutant wrappers for pandas.Series.mul."""

from __future__ import annotations

import os

import pandas as pd

ORIGINAL_MUL = None


def get_mutant_id():
    return os.environ.get("MUTANT_ID")


def mutant_mul_M1(self, other, level=None, fill_value=None, axis=0):
    """M1: perform addition instead of multiplication."""
    return self.add(other, level=level, fill_value=fill_value, axis=axis)


def mutant_mul_M2(self, other, level=None, fill_value=None, axis=0):
    """M2: ignore an explicit fill_value."""
    if fill_value is not None:
        fill_value = None
    return ORIGINAL_MUL(self, other, level=level, fill_value=fill_value, axis=axis)


def install_mutants():
    global ORIGINAL_MUL
    if ORIGINAL_MUL is not None:
        return
    ORIGINAL_MUL = pd.Series.mul
    mutant_id = get_mutant_id()
    if mutant_id == "M1":
        pd.Series.mul = mutant_mul_M1
    elif mutant_id == "M2":
        pd.Series.mul = mutant_mul_M2


def uninstall_mutants():
    global ORIGINAL_MUL
    if ORIGINAL_MUL is None:
        return
    pd.Series.mul = ORIGINAL_MUL
    ORIGINAL_MUL = None


MUTANT_INFO = {
    "M1": {
        "name": "swap_mul_for_add",
        "description": "Delegate to addition semantics instead of multiplication semantics.",
        "doc_anchor": "Return multiplication of series and other",
        "expected_kill": "Tests that assert element-wise multiplication values.",
    },
    "M2": {
        "name": "ignore_fill_value",
        "description": "Drop fill_value handling so one-sided missing values remain missing.",
        "doc_anchor": "fill_value float or None",
        "expected_kill": "Tests that assert fill_value influences alignment with missing data.",
    },
}
