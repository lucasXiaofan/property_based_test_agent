"""Mutant wrappers for pandas.Series.str.contains."""

from __future__ import annotations

import os

from pandas._libs import lib
from pandas.core.strings.accessor import StringMethods

ORIGINAL_CONTAINS = None


def get_mutant_id():
    return os.environ.get("MUTANT_ID")


def mutant_contains_M1(self, pat, case=True, flags=0, na=lib.no_default, regex=True):
    """M1: treat regex=False requests as regex=True."""
    if regex is False:
        regex = True
    return ORIGINAL_CONTAINS(self, pat, case=case, flags=flags, na=na, regex=regex)


def mutant_contains_M2(self, pat, case=True, flags=0, na=lib.no_default, regex=True):
    """M2: ignore case=False and use case-sensitive matching."""
    if case is False:
        case = True
    return ORIGINAL_CONTAINS(self, pat, case=case, flags=flags, na=na, regex=regex)


def install_mutants():
    global ORIGINAL_CONTAINS
    if ORIGINAL_CONTAINS is not None:
        return
    ORIGINAL_CONTAINS = StringMethods.contains
    mutant_id = get_mutant_id()
    if mutant_id == "M1":
        StringMethods.contains = mutant_contains_M1
    elif mutant_id == "M2":
        StringMethods.contains = mutant_contains_M2


def uninstall_mutants():
    global ORIGINAL_CONTAINS
    if ORIGINAL_CONTAINS is None:
        return
    StringMethods.contains = ORIGINAL_CONTAINS
    ORIGINAL_CONTAINS = None


MUTANT_INFO = {
    "M1": {
        "name": "regex_false_acts_like_regex_true",
        "description": "Treat literal matching requests as regex matching.",
        "doc_anchor": "regex bool, default True",
        "expected_kill": "Tests that distinguish literal and regex interpretation.",
    },
    "M2": {
        "name": "case_false_acts_like_case_true",
        "description": "Make case-insensitive searches behave as case-sensitive.",
        "doc_anchor": "case bool, default True",
        "expected_kill": "Tests that assert case=False performs case-insensitive matching.",
    },
}
