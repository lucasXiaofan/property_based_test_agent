"""Runtime mutants for pandas.DataFrame.head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from pandas import DataFrame

MutantFunc = Callable[..., object]
_ORIGINAL_HEAD = DataFrame.head
_ACTIVE_MUTANT_ID: str | None = None


@dataclass(frozen=True)
class MutantSpec:
    mutant_id: str
    description: str
    impl: MutantFunc


# C1: head() returns the first n rows.
def _mutant_c1_returns_last_n(self, n=5):
    """Return last n rows instead of first n rows."""
    return self.tail(n)


# C2: Default value of n is 5.
def _mutant_c2_default_n_is_10(self, n=10):
    """Use default n=10 instead of n=5."""
    return _ORIGINAL_HEAD(self, n)


# C3: head() returns the same type as the caller (DataFrame).
def _mutant_c3_returns_values(self, n=5):
    """Return underlying numpy array instead of DataFrame."""
    return _ORIGINAL_HEAD(self, n).values


# C4: head() behavior mirrors df[:n].
def _mutant_c4_off_by_one(self, n=5):
    """Return df[:n+1] (off-by-one) instead of df[:n]."""
    return self.iloc[: n + 1]


# C5: n=0 returns an empty object.
def _mutant_c5_zero_returns_all(self, n=5):
    """Return all rows when n=0 instead of empty DataFrame."""
    if n == 0:
        return self.copy()
    return _ORIGINAL_HEAD(self, n)


# C6: Negative n returns all rows except the last |n| rows.
def _mutant_c6_negative_n_returns_empty(self, n=5):
    """Return empty DataFrame for any negative n instead of df[:-n]."""
    if n < 0:
        return self.iloc[0:0]
    return _ORIGINAL_HEAD(self, n)


# C7: When n > len(df), all rows are returned (no error).
def _mutant_c7_overflow_raises(self, n=5):
    """Raise IndexError when n > len(df) instead of returning all rows."""
    if n > len(self):
        raise IndexError(f"Mutant C7: n={n} exceeds DataFrame length {len(self)}")
    return _ORIGINAL_HEAD(self, n)


_MUTANTS: dict[str, MutantSpec] = {
    "M_C1_returns_last_n": MutantSpec(
        "M_C1_returns_last_n",
        "Returns last n rows instead of first n rows",
        _mutant_c1_returns_last_n,
    ),
    "M_C2_default_n_is_10": MutantSpec(
        "M_C2_default_n_is_10",
        "Default n=10 instead of n=5",
        _mutant_c2_default_n_is_10,
    ),
    "M_C3_returns_values": MutantSpec(
        "M_C3_returns_values",
        "Returns numpy array instead of DataFrame",
        _mutant_c3_returns_values,
    ),
    "M_C4_off_by_one": MutantSpec(
        "M_C4_off_by_one",
        "Returns df[:n+1] (off-by-one) instead of df[:n]",
        _mutant_c4_off_by_one,
    ),
    "M_C5_zero_returns_all": MutantSpec(
        "M_C5_zero_returns_all",
        "n=0 returns all rows instead of empty DataFrame",
        _mutant_c5_zero_returns_all,
    ),
    "M_C6_negative_n_returns_empty": MutantSpec(
        "M_C6_negative_n_returns_empty",
        "Negative n returns empty DataFrame instead of df[:-n]",
        _mutant_c6_negative_n_returns_empty,
    ),
    "M_C7_overflow_raises": MutantSpec(
        "M_C7_overflow_raises",
        "Raises IndexError when n > len(df) instead of returning all rows",
        _mutant_c7_overflow_raises,
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
    DataFrame.head = spec.impl
    _ACTIVE_MUTANT_ID = mutant_id
    return {"mutant_id": spec.mutant_id, "description": spec.description}


def reset_mutant() -> None:
    global _ACTIVE_MUTANT_ID
    DataFrame.head = _ORIGINAL_HEAD
    _ACTIVE_MUTANT_ID = None


def get_active_mutant() -> str | None:
    return _ACTIVE_MUTANT_ID
