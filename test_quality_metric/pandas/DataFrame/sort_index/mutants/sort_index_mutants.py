"""Runtime mutants for pandas.DataFrame.sort_index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from pandas import DataFrame

MutantFunc = Callable[..., object]
_ORIGINAL_SORT_INDEX = DataFrame.sort_index
_ACTIVE_MUTANT_ID: str | None = None


@dataclass(frozen=True)
class MutantSpec:
    mutant_id: str
    description: str
    impl: MutantFunc


# ---------------------------------------------------------------------------
# C1: Returns a new DataFrame sorted by label when inplace=False
# Mutant: return unsorted copy — result is not sorted
# ---------------------------------------------------------------------------

def _mutant_c1_no_sort(self, *args, **kwargs):
    if kwargs.get("inplace", False):
        return None
    return self.copy()


# ---------------------------------------------------------------------------
# C2: Returns None when inplace=True
# Mutant: return self (the DataFrame) instead of None
# ---------------------------------------------------------------------------

def _mutant_c2_inplace_returns_self(self, *args, **kwargs):
    _ORIGINAL_SORT_INDEX(self, *args, **kwargs)
    if kwargs.get("inplace", False):
        return self  # should be None
    return _ORIGINAL_SORT_INDEX(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# C3: Default ascending=True produces ascending order
# Mutant: flip ascending boolean — True becomes False
# ---------------------------------------------------------------------------

def _mutant_c3_ascending_reversed(self, *args, **kwargs):
    kwargs = dict(kwargs)
    asc = kwargs.get("ascending", True)
    if isinstance(asc, bool):
        kwargs["ascending"] = not asc
    return _ORIGINAL_SORT_INDEX(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# C4: axis=1 sorts column labels
# Mutant: replace axis=1/'columns' with axis=0 — sorts rows instead
# ---------------------------------------------------------------------------

def _mutant_c4_axis1_sorts_rows(self, *args, **kwargs):
    kwargs = dict(kwargs)
    axis = kwargs.get("axis", 0)
    if axis in (1, "columns"):
        kwargs["axis"] = 0
    return _ORIGINAL_SORT_INDEX(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# C5: ascending=False produces descending order
# Mutant: ignore ascending=False, always sort ascending
# ---------------------------------------------------------------------------

def _mutant_c5_descending_becomes_ascending(self, *args, **kwargs):
    kwargs = dict(kwargs)
    asc = kwargs.get("ascending", True)
    if isinstance(asc, bool) and not asc:
        kwargs["ascending"] = True
    return _ORIGINAL_SORT_INDEX(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# C6: na_position='last' puts NaNs at the end
# Mutant: replace 'last' with 'first' — NaNs move to start
# ---------------------------------------------------------------------------

def _mutant_c6_na_last_becomes_first(self, *args, **kwargs):
    kwargs = dict(kwargs)
    if kwargs.get("na_position", "last") == "last":
        kwargs["na_position"] = "first"
    return _ORIGINAL_SORT_INDEX(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# C7: na_position='first' puts NaNs at the beginning
# Mutant: replace 'first' with 'last' — NaNs move to end
# ---------------------------------------------------------------------------

def _mutant_c7_na_first_becomes_last(self, *args, **kwargs):
    kwargs = dict(kwargs)
    if kwargs.get("na_position") == "first":
        kwargs["na_position"] = "last"
    return _ORIGINAL_SORT_INDEX(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# C8: ignore_index=True relabels axis as 0..n-1
# Mutant: always set ignore_index=False — index is never relabeled
# ---------------------------------------------------------------------------

def _mutant_c8_ignore_index_no_relabel(self, *args, **kwargs):
    kwargs = dict(kwargs)
    kwargs["ignore_index"] = False
    return _ORIGINAL_SORT_INDEX(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# C9: key function is applied to index values before sorting
# Mutant: strip the key function — sort without transformation
# ---------------------------------------------------------------------------

def _mutant_c9_key_not_applied(self, *args, **kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("key", None)
    return _ORIGINAL_SORT_INDEX(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# C10: mergesort and stable are the only stable algorithms
# Mutant: replace 'mergesort'/'stable' with 'quicksort'
# ---------------------------------------------------------------------------

def _mutant_c10_stable_becomes_quicksort(self, *args, **kwargs):
    kwargs = dict(kwargs)
    if kwargs.get("kind") in ("mergesort", "stable"):
        kwargs["kind"] = "quicksort"
    return _ORIGINAL_SORT_INDEX(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# C11: sort_remaining=True sorts remaining MultiIndex levels
# Mutant: always force sort_remaining=False
# ---------------------------------------------------------------------------

def _mutant_c11_sort_remaining_ignored(self, *args, **kwargs):
    kwargs = dict(kwargs)
    kwargs["sort_remaining"] = False
    return _ORIGINAL_SORT_INDEX(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# C12: level parameter sorts on specified index level(s)
# Mutant: strip level parameter — sort by all levels together
# ---------------------------------------------------------------------------

def _mutant_c12_ignore_level(self, *args, **kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("level", None)
    return _ORIGINAL_SORT_INDEX(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MUTANTS: dict[str, MutantSpec] = {
    "M_C1_no_sort": MutantSpec(
        "M_C1_no_sort", "C1: return unsorted copy instead of sorted DataFrame", _mutant_c1_no_sort
    ),
    "M_C2_inplace_returns_self": MutantSpec(
        "M_C2_inplace_returns_self", "C2: return self instead of None on inplace=True", _mutant_c2_inplace_returns_self
    ),
    "M_C3_ascending_reversed": MutantSpec(
        "M_C3_ascending_reversed", "C3: flip ascending bool — True becomes False", _mutant_c3_ascending_reversed
    ),
    "M_C4_axis1_sorts_rows": MutantSpec(
        "M_C4_axis1_sorts_rows", "C4: axis=1 sorts rows instead of columns", _mutant_c4_axis1_sorts_rows
    ),
    "M_C5_descending_becomes_ascending": MutantSpec(
        "M_C5_descending_becomes_ascending", "C5: ignore ascending=False, always sort ascending", _mutant_c5_descending_becomes_ascending
    ),
    "M_C6_na_last_becomes_first": MutantSpec(
        "M_C6_na_last_becomes_first", "C6: na_position='last' replaced with 'first'", _mutant_c6_na_last_becomes_first
    ),
    "M_C7_na_first_becomes_last": MutantSpec(
        "M_C7_na_first_becomes_last", "C7: na_position='first' replaced with 'last'", _mutant_c7_na_first_becomes_last
    ),
    "M_C8_ignore_index_no_relabel": MutantSpec(
        "M_C8_ignore_index_no_relabel", "C8: always set ignore_index=False", _mutant_c8_ignore_index_no_relabel
    ),
    "M_C9_key_not_applied": MutantSpec(
        "M_C9_key_not_applied", "C9: strip key function before sort", _mutant_c9_key_not_applied
    ),
    "M_C10_stable_becomes_quicksort": MutantSpec(
        "M_C10_stable_becomes_quicksort", "C10: replace mergesort/stable with quicksort", _mutant_c10_stable_becomes_quicksort
    ),
    "M_C11_sort_remaining_ignored": MutantSpec(
        "M_C11_sort_remaining_ignored", "C11: force sort_remaining=False always", _mutant_c11_sort_remaining_ignored
    ),
    "M_C12_ignore_level": MutantSpec(
        "M_C12_ignore_level", "C12: strip level parameter", _mutant_c12_ignore_level
    ),
}


def list_mutants() -> list[dict[str, str]]:
    return [{"mutant_id": s.mutant_id, "description": s.description} for s in _MUTANTS.values()]


def apply_mutant(mutant_id: str) -> dict[str, str]:
    global _ACTIVE_MUTANT_ID
    if mutant_id not in _MUTANTS:
        valid = ", ".join(sorted(_MUTANTS.keys()))
        raise ValueError(f"Unknown mutant_id={mutant_id!r}. Valid: {valid}")
    spec = _MUTANTS[mutant_id]
    DataFrame.sort_index = spec.impl
    _ACTIVE_MUTANT_ID = mutant_id
    return {"mutant_id": spec.mutant_id, "description": spec.description}


def reset_mutant() -> None:
    global _ACTIVE_MUTANT_ID
    DataFrame.sort_index = _ORIGINAL_SORT_INDEX
    _ACTIVE_MUTANT_ID = None


def get_active_mutant() -> str | None:
    return _ACTIVE_MUTANT_ID
