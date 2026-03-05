"""
Traditional mutations for pandas.DataFrame.reindex.

DESIGN PHILOSOPHY
─────────────────
Unlike LLM-generated mutants (which are crafted by reasoning about the
*semantic contract* of the function), these mutants are derived from the
*syntactic structure* of reindex_wrapper.py using standard mutation operators:

  SDL  – Statement Deletion Lifting  : remove a statement entirely
  ROR  – Relational Operator Replace : swap == / != / < / >
  COR  – Conditional Operator Replace: flip 'not', swap 'and'/'or'
  AOR  – Arithmetic Operator Replace : ±1 off-by-one on numeric argument
  SVR  – Scalar Variable Replace     : replace a literal constant

WHY DIFFERENT MUTANTS TEST DIFFERENT BEHAVIORS
──────────────────────────────────────────────
Each decision point in reindex_wrapper.py controls exactly one parameter
being forwarded to the real pandas implementation.  A mutant that *removes*
or *inverts* that decision silently drops or mis-routes the parameter, which
only tests catch if they exercise that parameter.

  SDL_method     → kills tests that use method= (ffill/bfill/nearest)
  SDL_fill_value → kills tests that verify custom fill_value
  SDL_limit      → kills tests that check limit= capping
  SDL_tolerance  → kills tests that verify tolerance= matching
  SDL_level      → kills tests that use level= with MultiIndex
  SDL_index      → kills tests for row reindexing via index=
  SDL_columns    → kills tests for column reindexing via columns=
  SDL_labels     → kills tests that pass positional labels
  ROR_axis       → kills tests that check axis routing direction
  ROR_labels     → kills tests where labels IS given
  ROR_index      → kills tests where index IS given
  ROR_columns    → kills tests where columns IS given
  COR_fill_sent  → kills tests that call reindex() with fill_value=NaN
                   (now the NaN fill_value IS forwarded → no change) OR
                   tests that pass non-NaN fill_value (now skipped)
  AOR_limit      → kills tests whose limit= boundary is exact (off-by-one)
  SVR_axis_def   → kills tests that rely on default axis=0 routing

INTERFACE
─────────
Same as all other *_mutants.py files in this project:
  apply_mutant(mutant_id)  – patch pd.DataFrame.reindex
  reset_mutant()           – restore original
  list_mutants()           – return list of {mutant_id, description}
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame

MutantFunc = Callable[..., Any]

_ORIGINAL = DataFrame.reindex
_ACTIVE_MUTANT_ID: str | None = None


# ── helpers (copied from reindex_wrapper so mutations can alter them) ────────

def _resolve_axis(axis: Any) -> int:
    if axis is None or axis == 0 or axis == "index":
        return 0
    if axis == 1 or axis == "columns":
        return 1
    raise ValueError(f"No axis named {axis!r}")


def _is_nan_sentinel(value: Any) -> bool:
    if value is None:
        return False
    try:
        return isinstance(value, float) and math.isnan(value)
    except (TypeError, ValueError):
        return False


# ── base (correct) shim ──────────────────────────────────────────────────────

def _correct_shim(self, labels=None, *, index=None, columns=None,
                  axis=None, method=None, copy=None, level=None,
                  fill_value=np.nan, limit=None, tolerance=None):
    """Correct pass-through — functionally equivalent to the original."""
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# SDL — Statement Deletion Lifting mutants
#   Each drops one parameter from the kwargs dict that is forwarded to pandas.
# ══════════════════════════════════════════════════════════════════════════════

def _SDL_index(self, labels=None, *, index=None, columns=None,
               axis=None, method=None, copy=None, level=None,
               fill_value=np.nan, limit=None, tolerance=None):
    """SDL: delete 'kwargs[\"index\"] = index' — row reindexing is ignored."""
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    # SDL: omit index entirely
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


def _SDL_columns(self, labels=None, *, index=None, columns=None,
                 axis=None, method=None, copy=None, level=None,
                 fill_value=np.nan, limit=None, tolerance=None):
    """SDL: delete 'kwargs[\"columns\"] = columns' — column reindexing is ignored."""
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    # SDL: omit columns entirely
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


def _SDL_method(self, labels=None, *, index=None, columns=None,
                axis=None, method=None, copy=None, level=None,
                fill_value=np.nan, limit=None, tolerance=None):
    """SDL: delete 'kwargs[\"method\"] = method' — fill methods are ignored."""
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    # SDL: omit method entirely
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


def _SDL_fill_value(self, labels=None, *, index=None, columns=None,
                    axis=None, method=None, copy=None, level=None,
                    fill_value=np.nan, limit=None, tolerance=None):
    """SDL: delete 'kwargs[\"fill_value\"] = fill_value' — custom fill ignored."""
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    # SDL: omit fill_value entirely — always uses NaN
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


def _SDL_limit(self, labels=None, *, index=None, columns=None,
               axis=None, method=None, copy=None, level=None,
               fill_value=np.nan, limit=None, tolerance=None):
    """SDL: delete 'kwargs[\"limit\"] = limit' — limit capping is ignored."""
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    # SDL: omit limit entirely — unlimited filling
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


def _SDL_tolerance(self, labels=None, *, index=None, columns=None,
                   axis=None, method=None, copy=None, level=None,
                   fill_value=np.nan, limit=None, tolerance=None):
    """SDL: delete 'kwargs[\"tolerance\"] = tolerance' — tolerance ignored."""
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    # SDL: omit tolerance — match without distance restriction
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


def _SDL_level(self, labels=None, *, index=None, columns=None,
               axis=None, method=None, copy=None, level=None,
               fill_value=np.nan, limit=None, tolerance=None):
    """SDL: delete 'kwargs[\"level\"] = level' — MultiIndex level ignored."""
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    # SDL: omit level — MultiIndex broadcast level is lost
    return _ORIGINAL(self, **kwargs)


def _SDL_labels_block(self, labels=None, *, index=None, columns=None,
                      axis=None, method=None, copy=None, level=None,
                      fill_value=np.nan, limit=None, tolerance=None):
    """SDL: delete the entire 'if labels is not None' routing block.

    Positional labels are silently dropped, so reindex(labels, axis='columns')
    degenerates to a no-op reindex.
    """
    # SDL: entire labels routing block deleted
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# ROR — Relational Operator Replacement mutants
#   Flip an equality/inequality check so the opposite branch executes.
# ══════════════════════════════════════════════════════════════════════════════

def _ROR_axis_eq(self, labels=None, *, index=None, columns=None,
                 axis=None, method=None, copy=None, level=None,
                 fill_value=np.nan, limit=None, tolerance=None):
    """ROR: 'if axis_int == 0' → 'if axis_int != 0'.

    Labels that should go to the row index are routed to columns and vice
    versa — axis direction is reversed.
    """
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int != 0:           # ROR: == → !=
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


def _ROR_labels_none(self, labels=None, *, index=None, columns=None,
                     axis=None, method=None, copy=None, level=None,
                     fill_value=np.nan, limit=None, tolerance=None):
    """ROR: 'if labels is not None' → 'if labels is None'.

    The labels routing block fires only when labels IS None (always a no-op
    because labels is None means index/columns already direct).  When actual
    labels are passed, the block is skipped and they are silently ignored.
    """
    if labels is None:              # ROR: 'is not' → 'is'
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


def _ROR_index_none(self, labels=None, *, index=None, columns=None,
                    axis=None, method=None, copy=None, level=None,
                    fill_value=np.nan, limit=None, tolerance=None):
    """ROR: 'if index is not None' → 'if index is None'.

    The index kwarg is forwarded only when index IS None — i.e., only the
    implicit None is forwarded, never the actual caller-supplied index.
    """
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is None:               # ROR: 'is not' → 'is'
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


def _ROR_columns_none(self, labels=None, *, index=None, columns=None,
                      axis=None, method=None, copy=None, level=None,
                      fill_value=np.nan, limit=None, tolerance=None):
    """ROR: 'if columns is not None' → 'if columns is None'.

    Columns kwarg is forwarded only when columns IS None — column reindexing
    requests are silently ignored.
    """
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is None:             # ROR: 'is not' → 'is'
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# COR — Conditional Operator Replacement mutants
#   Invert a boolean condition or swap logical connectives.
# ══════════════════════════════════════════════════════════════════════════════

def _COR_fill_sentinel(self, labels=None, *, index=None, columns=None,
                       axis=None, method=None, copy=None, level=None,
                       fill_value=np.nan, limit=None, tolerance=None):
    """COR: 'if not _is_nan_sentinel(fill_value)' → 'if _is_nan_sentinel(fill_value)'.

    The fill_value is forwarded only when it IS NaN (the sentinel) — so custom
    fill values are silently dropped while the default NaN is forwarded (no-op
    since NaN is the default anyway).
    """
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if _is_nan_sentinel(fill_value):        # COR: removed 'not'
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# AOR — Arithmetic Operator Replacement mutants
#   Off-by-one on a numeric argument forwarded to pandas.
# ══════════════════════════════════════════════════════════════════════════════

def _AOR_limit_plus1(self, labels=None, *, index=None, columns=None,
                     axis=None, method=None, copy=None, level=None,
                     fill_value=np.nan, limit=None, tolerance=None):
    """AOR: 'kwargs[\"limit\"] = limit' → 'kwargs[\"limit\"] = limit + 1'.

    The limit is inflated by 1, allowing one more consecutive fill than the
    caller requested.  Tests with tight limit boundaries (e.g. limit=1 where
    position 2 must be NaN) detect this.
    """
    if labels is not None:
        axis_int = _resolve_axis(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit + 1     # AOR: + 1 off-by-one
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# SVR — Scalar Variable Replacement mutants
#   Replace a literal constant with a wrong value.
# ══════════════════════════════════════════════════════════════════════════════

def _SVR_axis_default(self, labels=None, *, index=None, columns=None,
                      axis=None, method=None, copy=None, level=None,
                      fill_value=np.nan, limit=None, tolerance=None):
    """SVR: default axis return value 0 → 1 inside _resolve_axis.

    When axis=None the function now defaults to axis=1 (columns) instead of
    axis=0 (rows), mis-routing positional labels.
    """
    def _resolve_axis_svr(a: Any) -> int:
        if a is None:
            return 1                    # SVR: 0 → 1
        if a == 0 or a == "index":
            return 0
        if a == 1 or a == "columns":
            return 1
        raise ValueError(f"No axis named {a!r}")

    if labels is not None:
        axis_int = _resolve_axis_svr(axis)
        if axis_int == 0:
            index = labels
        else:
            columns = labels
    kwargs: dict[str, Any] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    if method is not None:
        kwargs["method"] = method
    if not _is_nan_sentinel(fill_value):
        kwargs["fill_value"] = fill_value
    if limit is not None:
        kwargs["limit"] = limit
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    if level is not None:
        kwargs["level"] = level
    return _ORIGINAL(self, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Registry + public API (same interface as existing *_mutants.py files)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _MutantSpec:
    mutant_id: str
    operator: str      # SDL / ROR / COR / AOR / SVR
    description: str
    impl: MutantFunc


_MUTANTS: dict[str, _MutantSpec] = {
    # SDL
    "SDL_index":        _MutantSpec("SDL_index",        "SDL", "Delete kwargs['index'] = index", _SDL_index),
    "SDL_columns":      _MutantSpec("SDL_columns",      "SDL", "Delete kwargs['columns'] = columns", _SDL_columns),
    "SDL_method":       _MutantSpec("SDL_method",       "SDL", "Delete kwargs['method'] = method", _SDL_method),
    "SDL_fill_value":   _MutantSpec("SDL_fill_value",   "SDL", "Delete kwargs['fill_value'] = fill_value", _SDL_fill_value),
    "SDL_limit":        _MutantSpec("SDL_limit",        "SDL", "Delete kwargs['limit'] = limit", _SDL_limit),
    "SDL_tolerance":    _MutantSpec("SDL_tolerance",    "SDL", "Delete kwargs['tolerance'] = tolerance", _SDL_tolerance),
    "SDL_level":        _MutantSpec("SDL_level",        "SDL", "Delete kwargs['level'] = level", _SDL_level),
    "SDL_labels_block": _MutantSpec("SDL_labels_block", "SDL", "Delete entire labels routing block", _SDL_labels_block),
    # ROR
    "ROR_axis_eq":      _MutantSpec("ROR_axis_eq",      "ROR", "axis_int == 0 → axis_int != 0", _ROR_axis_eq),
    "ROR_labels_none":  _MutantSpec("ROR_labels_none",  "ROR", "labels is not None → labels is None", _ROR_labels_none),
    "ROR_index_none":   _MutantSpec("ROR_index_none",   "ROR", "index is not None → index is None", _ROR_index_none),
    "ROR_columns_none": _MutantSpec("ROR_columns_none", "ROR", "columns is not None → columns is None", _ROR_columns_none),
    # COR
    "COR_fill_sentinel": _MutantSpec("COR_fill_sentinel", "COR", "if not _is_nan_sentinel → if _is_nan_sentinel", _COR_fill_sentinel),
    # AOR
    "AOR_limit_plus1":  _MutantSpec("AOR_limit_plus1",  "AOR", "limit → limit + 1 (off-by-one)", _AOR_limit_plus1),
    # SVR
    "SVR_axis_default": _MutantSpec("SVR_axis_default", "SVR", "default axis 0 → 1 (default to columns)", _SVR_axis_default),
}


def list_mutants() -> list[dict[str, str]]:
    return [
        {"mutant_id": s.mutant_id, "operator": s.operator, "description": s.description}
        for s in _MUTANTS.values()
    ]


def apply_mutant(mutant_id: str) -> dict[str, str]:
    global _ACTIVE_MUTANT_ID
    if mutant_id not in _MUTANTS:
        valid = ", ".join(sorted(_MUTANTS.keys()))
        raise ValueError(f"Unknown mutant_id={mutant_id!r}. Valid: {valid}")
    spec = _MUTANTS[mutant_id]
    DataFrame.reindex = spec.impl
    _ACTIVE_MUTANT_ID = mutant_id
    return {"mutant_id": spec.mutant_id, "operator": spec.operator, "description": spec.description}


def reset_mutant() -> None:
    global _ACTIVE_MUTANT_ID
    DataFrame.reindex = _ORIGINAL
    _ACTIVE_MUTANT_ID = None


def get_active_mutant() -> str | None:
    return _ACTIVE_MUTANT_ID
