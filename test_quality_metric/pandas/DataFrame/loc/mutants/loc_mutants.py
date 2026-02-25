"""
Executable mutants for pandas.DataFrame.loc.

Each mutant monkey-patches pandas.core.indexing._LocIndexer.__getitem__
to violate one specific clause from clauses.json.

API
---
apply_mutant(mutant_id)  -> dict[str, str]   — activates a mutant
reset_mutant()           -> None              — restores original behaviour
list_mutants()           -> list[dict]        — enumerate all mutants
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.core.indexing import _LocIndexer

# ---------------------------------------------------------------------------
# Preserve the original __getitem__ before any patching
# ---------------------------------------------------------------------------

_original_loc_getitem = _LocIndexer.__getitem__

_ACTIVE_MUTANT: str | None = None

# ---------------------------------------------------------------------------
# Mutant registry
# ---------------------------------------------------------------------------

_MUTANT_REGISTRY: list[dict[str, str]] = [
    {
        "mutant_id": "M_C1_use_iloc",
        "clause_id": "C1",
        "description": (
            "Use iloc (position-based) for scalar keys instead of loc (label-based). "
            "String labels cause TypeError; integer labels that differ from their "
            "position raise IndexError."
        ),
    },
    {
        "mutant_id": "M_C2_int_as_position",
        "clause_id": "C2",
        "description": (
            "Treat integer scalar keys as positional indices (iloc semantics). "
            "Fails when the integer label does not coincide with its ordinal position."
        ),
    },
    {
        "mutant_id": "M_C3_slice_exclusive_stop",
        "clause_id": "C3",
        "description": (
            "Make label-slice stop exclusive (regular Python slice semantics), "
            "dropping the last row that loc normally includes."
        ),
    },
    {
        "mutant_id": "M_C4_always_dataframe",
        "clause_id": "C4",
        "description": (
            "Wrap every Series result in a one-row DataFrame so that single-label "
            "row access never returns a Series."
        ),
    },
    {
        "mutant_id": "M_C5_list_squeeze",
        "clause_id": "C5",
        "description": (
            "Call .squeeze() on the result of list-key access so that a "
            "single-row selection collapses to a Series instead of a DataFrame."
        ),
    },
    {
        "mutant_id": "M_C6_missing_returns_nan",
        "clause_id": "C6",
        "description": (
            "Catch KeyError for missing labels and return a NaN-filled Series "
            "instead of re-raising, suppressing the documented exception."
        ),
    },
    {
        "mutant_id": "M_C7_bool_invert",
        "clause_id": "C7",
        "description": (
            "Invert boolean array/Series masks so that rows where True are "
            "excluded and rows where False are included."
        ),
    },
    {
        "mutant_id": "M_C8_callable_no_call",
        "clause_id": "C8",
        "description": (
            "When the key is callable, skip calling it and return an empty "
            "DataFrame, ignoring the callable's filtering logic entirely."
        ),
    },
]


# ---------------------------------------------------------------------------
# Patched __getitem__ dispatcher
# ---------------------------------------------------------------------------

def _patched_loc_getitem(self: _LocIndexer, key):  # type: ignore[override]
    global _ACTIVE_MUTANT

    if _ACTIVE_MUTANT is None:
        return _original_loc_getitem(self, key)

    # ---- M_C1: use iloc (positional) for scalar keys ----------------------
    if _ACTIVE_MUTANT == "M_C1_use_iloc":
        if not isinstance(key, (tuple, list, slice, np.ndarray, pd.Series)):
            # Pass the key directly to iloc; string keys → TypeError, out-of-range int → IndexError
            return self.obj.iloc[key]
        return _original_loc_getitem(self, key)

    # ---- M_C2: treat integer scalars as positions -------------------------
    if _ACTIVE_MUTANT == "M_C2_int_as_position":
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            return self.obj.iloc[int(key)]
        return _original_loc_getitem(self, key)

    # ---- M_C3: make slice stop exclusive ----------------------------------
    if _ACTIVE_MUTANT == "M_C3_slice_exclusive_stop":
        if isinstance(key, slice) and key.stop is not None:
            result = _original_loc_getitem(self, key)
            if isinstance(result, pd.DataFrame) and len(result) > 0:
                return result.iloc[:-1]  # drop the last (inclusive) stop row
            return result
        return _original_loc_getitem(self, key)

    # ---- M_C4: always return DataFrame even for single label --------------
    if _ACTIVE_MUTANT == "M_C4_always_dataframe":
        result = _original_loc_getitem(self, key)
        if isinstance(result, pd.Series):
            return result.to_frame().T
        return result

    # ---- M_C5: squeeze list-key result so single row becomes Series -------
    if _ACTIVE_MUTANT == "M_C5_list_squeeze":
        result = _original_loc_getitem(self, key)
        if isinstance(key, list) and isinstance(result, pd.DataFrame):
            return result.squeeze(axis=0)
        return result

    # ---- M_C6: swallow KeyError, return NaN row ---------------------------
    if _ACTIVE_MUTANT == "M_C6_missing_returns_nan":
        try:
            return _original_loc_getitem(self, key)
        except KeyError:
            cols = self.obj.columns
            return pd.Series(
                [np.nan] * len(cols),
                index=cols,
                name=key,
            )

    # ---- M_C7: invert boolean masks ---------------------------------------
    if _ACTIVE_MUTANT == "M_C7_bool_invert":
        # Normalise key to an inverted form when it is boolean
        mutated_key = key
        if isinstance(key, pd.Series) and key.dtype == bool:
            mutated_key = ~key
        elif isinstance(key, np.ndarray) and key.dtype == bool:
            mutated_key = ~key
        elif isinstance(key, list) and key and isinstance(key[0], (bool, np.bool_)):
            mutated_key = [not k for k in key]
        return _original_loc_getitem(self, mutated_key)

    # ---- M_C8: skip calling the callable, return empty DataFrame ----------
    if _ACTIVE_MUTANT == "M_C8_callable_no_call":
        if callable(key):
            return self.obj.iloc[[]]  # empty DataFrame, same columns
        return _original_loc_getitem(self, key)

    # Fallback: identity (should not happen for registered mutants)
    return _original_loc_getitem(self, key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_mutant(mutant_id: str) -> dict[str, str]:
    """Activate *mutant_id* by monkey-patching _LocIndexer.__getitem__."""
    global _ACTIVE_MUTANT
    ids = {m["mutant_id"] for m in _MUTANT_REGISTRY}
    if mutant_id not in ids:
        raise ValueError(f"Unknown mutant_id: {mutant_id!r}. Known: {sorted(ids)}")
    _ACTIVE_MUTANT = mutant_id
    _LocIndexer.__getitem__ = _patched_loc_getitem  # type: ignore[method-assign]
    info = next(m for m in _MUTANT_REGISTRY if m["mutant_id"] == mutant_id)
    return dict(info)


def reset_mutant() -> None:
    """Restore the original _LocIndexer.__getitem__."""
    global _ACTIVE_MUTANT
    _ACTIVE_MUTANT = None
    _LocIndexer.__getitem__ = _original_loc_getitem  # type: ignore[method-assign]


def list_mutants() -> list[dict[str, str]]:
    """Return a copy of the mutant registry."""
    return [dict(m) for m in _MUTANT_REGISTRY]
