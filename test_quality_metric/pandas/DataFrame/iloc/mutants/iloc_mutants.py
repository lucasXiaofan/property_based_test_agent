"""
Executable mutants for pandas.DataFrame.iloc.

Each mutant monkey-patches pandas.core.indexing._iLocIndexer.__getitem__
to violate one specific clause from clauses.json.

API
---
apply_mutant(mutant_id)  -> dict[str, str]   -- activates a mutant
reset_mutant()           -> None             -- restores original behaviour
list_mutants()           -> list[dict]       -- enumerate all mutants
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.core.indexing import _iLocIndexer

# ---------------------------------------------------------------------------
# Preserve the original __getitem__ before any patching
# ---------------------------------------------------------------------------

_original_iloc_getitem = _iLocIndexer.__getitem__

_ACTIVE_MUTANT: str | None = None

# ---------------------------------------------------------------------------
# Mutant registry
# ---------------------------------------------------------------------------

_MUTANT_REGISTRY: list[dict[str, str]] = [
    {
        "mutant_id": "M_C1_use_loc",
        "clause_id": "C1",
        "description": (
            "Use loc (label-based) for scalar integer keys instead of iloc (position-based). "
            "Fails when integer labels don't match positions (e.g., index=[10,20,30])."
        ),
    },
    {
        "mutant_id": "M_C2_single_int_always_dataframe",
        "clause_id": "C2",
        "description": (
            "Wrap every Series result in a one-row DataFrame so that single-integer "
            "row access never returns a Series."
        ),
    },
    {
        "mutant_id": "M_C3_list_squeeze_to_series",
        "clause_id": "C3",
        "description": (
            "Call .squeeze(axis=0) on single-row DataFrame results from list access, "
            "collapsing them to a Series instead of a DataFrame."
        ),
    },
    {
        "mutant_id": "M_C4_slice_drop_last",
        "clause_id": "C4",
        "description": (
            "Drop the last row from slice results, simulating an off-by-one error "
            "where the slice stop is treated as inclusive."
        ),
    },
    {
        "mutant_id": "M_C5_bool_invert",
        "clause_id": "C5",
        "description": (
            "Invert boolean array/list masks so that rows where True are excluded "
            "and rows where False are included."
        ),
    },
    {
        "mutant_id": "M_C6_callable_no_call",
        "clause_id": "C6",
        "description": (
            "When the key is callable, skip calling it and return an empty DataFrame, "
            "ignoring the callable's filtering logic entirely."
        ),
    },
    {
        "mutant_id": "M_C7_tuple_ignore_cols",
        "clause_id": "C7",
        "description": (
            "When a tuple key is provided, ignore the column indexer and return "
            "the full row selection only (no column filtering)."
        ),
    },
    {
        "mutant_id": "M_C8_no_indexerror",
        "clause_id": "C8",
        "description": (
            "Suppress IndexError for out-of-bounds integer access, returning an "
            "empty Series instead of raising."
        ),
    },
    {
        "mutant_id": "M_C9_slice_raises_indexerror",
        "clause_id": "C9",
        "description": (
            "Raise IndexError for slice indexers when stop exceeds DataFrame length, "
            "violating the documented out-of-bounds slice tolerance."
        ),
    },
    {
        "mutant_id": "M_C10_scalar_returns_series",
        "clause_id": "C10",
        "description": (
            "For scalar (single-row + single-col) tuple access, return the full row "
            "as a Series instead of the individual scalar value."
        ),
    },
]


# ---------------------------------------------------------------------------
# Patched __getitem__ dispatcher
# ---------------------------------------------------------------------------

def _patched_iloc_getitem(self: _iLocIndexer, key):  # type: ignore[override]
    global _ACTIVE_MUTANT

    if _ACTIVE_MUTANT is None:
        return _original_iloc_getitem(self, key)

    # ---- M_C1: use loc (label-based) for scalar integer keys ----------------
    if _ACTIVE_MUTANT == "M_C1_use_loc":
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            return self.obj.loc[int(key)]
        return _original_iloc_getitem(self, key)

    # ---- M_C2: single integer always returns DataFrame ----------------------
    if _ACTIVE_MUTANT == "M_C2_single_int_always_dataframe":
        result = _original_iloc_getitem(self, key)
        if isinstance(result, pd.Series):
            return result.to_frame().T
        return result

    # ---- M_C3: list-of-ints, squeeze single row to Series ------------------
    if _ACTIVE_MUTANT == "M_C3_list_squeeze_to_series":
        result = _original_iloc_getitem(self, key)
        if isinstance(key, list) and isinstance(result, pd.DataFrame) and len(result) == 1:
            return result.squeeze(axis=0)
        return result

    # ---- M_C4: slice – drop the last row (off-by-one) ----------------------
    if _ACTIVE_MUTANT == "M_C4_slice_drop_last":
        if isinstance(key, slice):
            result = _original_iloc_getitem(self, key)
            if isinstance(result, pd.DataFrame) and len(result) > 0:
                return result.iloc[:-1]
            return result
        return _original_iloc_getitem(self, key)

    # ---- M_C5: invert boolean masks ----------------------------------------
    if _ACTIVE_MUTANT == "M_C5_bool_invert":
        mutated_key = key
        if isinstance(key, pd.Series) and key.dtype == bool:
            mutated_key = ~key
        elif isinstance(key, np.ndarray) and key.dtype == bool:
            mutated_key = ~key
        elif isinstance(key, list) and key and isinstance(key[0], (bool, np.bool_)):
            mutated_key = [not k for k in key]
        return _original_iloc_getitem(self, mutated_key)

    # ---- M_C6: callable not called, return empty DataFrame -----------------
    if _ACTIVE_MUTANT == "M_C6_callable_no_call":
        if callable(key):
            return self.obj.iloc[[]]
        return _original_iloc_getitem(self, key)

    # ---- M_C7: tuple ignores column indexer --------------------------------
    if _ACTIVE_MUTANT == "M_C7_tuple_ignore_cols":
        if isinstance(key, tuple) and len(key) >= 2:
            return _original_iloc_getitem(self, key[0])
        return _original_iloc_getitem(self, key)

    # ---- M_C8: suppress IndexError for out-of-bounds integer ---------------
    if _ACTIVE_MUTANT == "M_C8_no_indexerror":
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            n = len(self.obj)
            k = int(key)
            if k >= n or k < -n:
                return pd.Series(dtype=float)
        return _original_iloc_getitem(self, key)

    # ---- M_C9: slice raises IndexError for out-of-bounds stop --------------
    if _ACTIVE_MUTANT == "M_C9_slice_raises_indexerror":
        if isinstance(key, slice):
            n = len(self.obj)
            stop = key.stop
            start = key.start if key.start is not None else 0
            if (stop is not None and stop > n) or start < -n:
                raise IndexError(
                    f"Mutant C9: slice indexer {key} is out of bounds for axis with {n} elements"
                )
        return _original_iloc_getitem(self, key)

    # ---- M_C10: scalar row+col access returns full row as Series ------------
    if _ACTIVE_MUTANT == "M_C10_scalar_returns_series":
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if (
                isinstance(row_key, (int, np.integer))
                and not isinstance(row_key, bool)
                and isinstance(col_key, (int, np.integer))
                and not isinstance(col_key, bool)
            ):
                return _original_iloc_getitem(self, row_key)
        return _original_iloc_getitem(self, key)

    # Fallback: identity (should not happen for registered mutants)
    return _original_iloc_getitem(self, key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_mutant(mutant_id: str) -> dict[str, str]:
    """Activate *mutant_id* by monkey-patching _iLocIndexer.__getitem__."""
    global _ACTIVE_MUTANT
    ids = {m["mutant_id"] for m in _MUTANT_REGISTRY}
    if mutant_id not in ids:
        raise ValueError(f"Unknown mutant_id: {mutant_id!r}. Known: {sorted(ids)}")
    _ACTIVE_MUTANT = mutant_id
    _iLocIndexer.__getitem__ = _patched_iloc_getitem  # type: ignore[method-assign]
    info = next(m for m in _MUTANT_REGISTRY if m["mutant_id"] == mutant_id)
    return dict(info)


def reset_mutant() -> None:
    """Restore the original _iLocIndexer.__getitem__."""
    global _ACTIVE_MUTANT
    _ACTIVE_MUTANT = None
    _iLocIndexer.__getitem__ = _original_iloc_getitem  # type: ignore[method-assign]


def list_mutants() -> list[dict[str, str]]:
    """Return a copy of the mutant registry."""
    return [dict(m) for m in _MUTANT_REGISTRY]
