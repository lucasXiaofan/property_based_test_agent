"""Mutant wrappers for pandas.Series.str.match."""

from __future__ import annotations

import os

from pandas._libs import lib
from pandas.core.strings.accessor import StringMethods

ORIGINAL_MATCH = None


def get_mutant_id():
    return os.environ.get("MUTANT_ID")


def mutant_match_M1(self, pat, case=lib.no_default, flags=lib.no_default, na=lib.no_default):
    """M1: use contains-style search semantics instead of anchored match semantics."""
    contains_kwargs = {"pat": pat, "na": na, "regex": True}
    if case is not lib.no_default:
        contains_kwargs["case"] = case
    if flags is not lib.no_default:
        contains_kwargs["flags"] = flags
    return self.contains(**contains_kwargs)


def mutant_match_M2(self, pat, case=lib.no_default, flags=lib.no_default, na=lib.no_default):
    """M2: ignore case=False and use case-sensitive matching."""
    if case is False:
        case = True
    return ORIGINAL_MATCH(self, pat, case=case, flags=flags, na=na)


def install_mutants():
    global ORIGINAL_MATCH
    if ORIGINAL_MATCH is not None:
        return
    ORIGINAL_MATCH = StringMethods.match
    mutant_id = get_mutant_id()
    if mutant_id == "M1":
        StringMethods.match = mutant_match_M1
    elif mutant_id == "M2":
        StringMethods.match = mutant_match_M2


def uninstall_mutants():
    global ORIGINAL_MATCH
    if ORIGINAL_MATCH is None:
        return
    StringMethods.match = ORIGINAL_MATCH
    ORIGINAL_MATCH = None


MUTANT_INFO = {
    "M1": {
        "name": "match_acts_like_contains",
        "description": "Delegate to contains-style search behavior rather than start-anchored match behavior.",
        "doc_anchor": "Determine if each string starts with a match",
        "expected_kill": "Tests that distinguish search semantics from anchored match semantics.",
    },
    "M2": {
        "name": "case_false_acts_like_case_true",
        "description": "Make case-insensitive match requests behave as case-sensitive.",
        "doc_anchor": "case bool, default True",
        "expected_kill": "Tests that assert case=False performs case-insensitive matching.",
    },
}
