"""Runtime mutants for pandas.DataFrame.assign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from pandas import DataFrame

MutantFunc = Callable[..., object]
_ORIGINAL_ASSIGN = DataFrame.assign
_ACTIVE_MUTANT_ID: str | None = None


@dataclass(frozen=True)
class MutantSpec:
    mutant_id: str
    description: str
    impl: MutantFunc


# ---------------------------------------------------------------------------
# C1: Returns new object with all original + new columns
# Mutant: return a copy of self without adding new columns
# ---------------------------------------------------------------------------

def _mutant_c1_no_new_columns(self, **kwargs):
    """C1 mutant: returns copy of self, ignoring all new column assignments."""
    return self.copy()


# ---------------------------------------------------------------------------
# C2: Existing columns are overwritten
# Mutant: skip kwargs whose key already exists as a column
# ---------------------------------------------------------------------------

def _mutant_c2_skip_overwrite(self, **kwargs):
    """C2 mutant: silently drops kwargs that would overwrite an existing column."""
    filtered = {k: v for k, v in kwargs.items() if k not in self.columns}
    return _ORIGINAL_ASSIGN(self, **filtered)


# ---------------------------------------------------------------------------
# C3: Original DataFrame not modified
# Mutant: modify self in-place and return self (not a new object)
# ---------------------------------------------------------------------------

def _mutant_c3_return_self_modified(self, **kwargs):
    """C3 mutant: assigns directly into self and returns self, modifying the original."""
    for k, v in kwargs.items():
        if callable(v):
            self[k] = v(self)
        else:
            self[k] = v
    return self


# ---------------------------------------------------------------------------
# C4: Callable values evaluated on DataFrame
# Mutant: treat callable as a non-callable, assign None instead
# ---------------------------------------------------------------------------

def _mutant_c4_dont_call_callables(self, **kwargs):
    """C4 mutant: does not invoke callables; assigns None for any callable value."""
    new_kwargs = {}
    for k, v in kwargs.items():
        if callable(v):
            new_kwargs[k] = None
        else:
            new_kwargs[k] = v
    return _ORIGINAL_ASSIGN(self, **new_kwargs)


# ---------------------------------------------------------------------------
# C5: Non-callable values directly assigned
# Mutant: skip non-callable values (only process callables)
# ---------------------------------------------------------------------------

def _mutant_c5_ignore_noncallables(self, **kwargs):
    """C5 mutant: ignores non-callable values; only callable kwargs are applied."""
    new_kwargs = {k: v for k, v in kwargs.items() if callable(v)}
    return _ORIGINAL_ASSIGN(self, **new_kwargs)


# ---------------------------------------------------------------------------
# C6: Items computed and assigned in order
# Mutant: process kwargs in reversed order
# ---------------------------------------------------------------------------

def _mutant_c6_reversed_order(self, **kwargs):
    """C6 mutant: processes kwargs in reversed order, violating in-order computation."""
    result = self.copy()
    for k, v in reversed(list(kwargs.items())):
        if callable(v):
            result[k] = v(result)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# C7: Later items may refer to newly created or modified columns
# Mutant: each callable receives a snapshot of the original df, not the updated one
# ---------------------------------------------------------------------------

def _mutant_c7_all_from_original(self, **kwargs):
    """C7 mutant: all callables receive the original df snapshot, ignoring earlier assignments."""
    snapshot = self.copy()
    result = self.copy()
    for k, v in kwargs.items():
        if callable(v):
            result[k] = v(snapshot)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MUTANTS: dict[str, MutantSpec] = {
    "M_C1_no_new_columns": MutantSpec(
        "M_C1_no_new_columns", "C1", _mutant_c1_no_new_columns
    ),
    "M_C2_skip_overwrite": MutantSpec(
        "M_C2_skip_overwrite", "C2", _mutant_c2_skip_overwrite
    ),
    "M_C3_return_self_modified": MutantSpec(
        "M_C3_return_self_modified", "C3", _mutant_c3_return_self_modified
    ),
    "M_C4_dont_call_callables": MutantSpec(
        "M_C4_dont_call_callables", "C4", _mutant_c4_dont_call_callables
    ),
    "M_C5_ignore_noncallables": MutantSpec(
        "M_C5_ignore_noncallables", "C5", _mutant_c5_ignore_noncallables
    ),
    "M_C6_reversed_order": MutantSpec(
        "M_C6_reversed_order", "C6", _mutant_c6_reversed_order
    ),
    "M_C7_all_from_original": MutantSpec(
        "M_C7_all_from_original", "C7", _mutant_c7_all_from_original
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
    DataFrame.assign = spec.impl
    _ACTIVE_MUTANT_ID = mutant_id
    return {"mutant_id": spec.mutant_id, "description": spec.description}


def reset_mutant() -> None:
    global _ACTIVE_MUTANT_ID
    DataFrame.assign = _ORIGINAL_ASSIGN
    _ACTIVE_MUTANT_ID = None


def get_active_mutant() -> str | None:
    return _ACTIVE_MUTANT_ID
