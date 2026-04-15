"""Mutant wrappers for pandas.Index.shift."""

from __future__ import annotations

import os

import pandas as pd

ORIGINAL_SHIFT = {}
PATCH_TYPES = (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)


def get_mutant_id():
    return os.environ.get("MUTANT_ID")


def mutant_shift_M1(self, periods=1, freq=None):
    """M1: reverse the requested shift direction."""
    return ORIGINAL_SHIFT[type(self)](self, periods=-periods, freq=freq)


def mutant_shift_M2(self, periods=1, freq=None):
    """M2: ignore an explicit freq argument."""
    if freq is not None:
        freq = None
    return ORIGINAL_SHIFT[type(self)](self, periods=periods, freq=freq)


def install_mutants():
    if ORIGINAL_SHIFT:
        return
    mutant_id = get_mutant_id()
    for cls in PATCH_TYPES:
        ORIGINAL_SHIFT[cls] = cls.shift
        if mutant_id == "M1":
            cls.shift = mutant_shift_M1
        elif mutant_id == "M2":
            cls.shift = mutant_shift_M2


def uninstall_mutants():
    if not ORIGINAL_SHIFT:
        return
    for cls, original in ORIGINAL_SHIFT.items():
        cls.shift = original
    ORIGINAL_SHIFT.clear()


MUTANT_INFO = {
    "M1": {
        "name": "negate_periods",
        "description": "Apply the opposite shift direction from the caller-provided periods.",
        "doc_anchor": "periods int, can be positive or negative",
        "expected_kill": "Tests that assert positive and negative shifts move values in the expected direction.",
    },
    "M2": {
        "name": "ignore_explicit_freq",
        "description": "Discard an explicit freq argument and reuse implicit index frequency behavior instead.",
        "doc_anchor": "freq DateOffset, timedelta, or str, optional",
        "expected_kill": "Tests that assert explicit freq changes the shift increment.",
    },
}
