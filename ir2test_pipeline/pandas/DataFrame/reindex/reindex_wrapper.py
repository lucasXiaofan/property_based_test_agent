"""
Python-level shim around pd.DataFrame.reindex for traditional mutation testing.

WHY A WRAPPER?
  pd.DataFrame.reindex is implemented in C/Cython, so standard mutation tools
  (mutmut, cosmic-ray) cannot produce AST mutations of its internal logic.
  This wrapper re-expresses every *decision point* as explicit Python code so
  that traditional mutation operators can be applied to it.

DECISION POINTS (annotations show which operator targets each line):
  _resolve_axis()           – ROR, SVR targets
  _route_labels()           – ROR, SDL targets
  reindex_shim()            – SDL, COR targets for each kwarg pass-through

USAGE (monkey-patch pandas once per process):
  from reindex_wrapper import install, uninstall
  install()   # pd.DataFrame.reindex → reindex_shim
  ...
  uninstall() # restore original
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

_ORIGINAL_REINDEX = DataFrame.reindex


# ── helpers ─────────────────────────────────────────────────────────────────

def _resolve_axis(axis: Any) -> int:
    """
    Map axis argument to integer 0 (rows) or 1 (columns).

    Traditional mutation targets:
      ROR: return 0  →  return 1        (SVR on the constant)
      ROR: axis in (...) checks         (condition flip)
    """
    if axis is None or axis == 0 or axis == "index":   # ROR: change == to !=
        return 0                                        # SVR: change 0 to 1
    if axis == 1 or axis == "columns":                 # ROR: change == to !=
        return 1                                        # SVR: change 1 to 0
    raise ValueError(f"No axis named {axis!r}")


def _is_nan_sentinel(value: Any) -> bool:
    """
    Return True when value is the NaN fill sentinel (the default fill_value).

    Traditional mutation targets:
      COR: 'or'  →  'and'              (tightens the condition)
      ROR: isinstance(...) checks
    """
    if value is None:
        return False
    try:
        return isinstance(value, float) and math.isnan(value)  # COR: 'and' → 'or'
    except (TypeError, ValueError):
        return False


# ── shim ────────────────────────────────────────────────────────────────────

def reindex_shim(
    self,
    labels=None,
    *,
    index=None,
    columns=None,
    axis=None,
    method=None,
    copy=None,
    level=None,
    fill_value: float = np.nan,
    limit=None,
    tolerance=None,
):
    """
    Thin Python shim.  Delegates to _ORIGINAL_REINDEX after resolving arguments.

    Each `if` guard is an SDL (Statement Deletion Lifting) target — deleting
    the guarded block corresponds to "ignoring" that parameter entirely.

    The axis routing block is additionally an ROR target.
    """
    # ── [SDL target] Resolve positional labels to the right axis ─────────────
    if labels is not None:                              # ROR: 'is not' → 'is'
        axis_int = _resolve_axis(axis)
        if axis_int == 0:                               # ROR: == → !=
            index = labels
        else:
            columns = labels

    # ── Build kwargs — each assignment is an SDL target ──────────────────────
    kwargs: dict[str, Any] = {}

    if index is not None:                               # ROR: 'is not' → 'is'
        kwargs["index"] = index                         # SDL: delete this line

    if columns is not None:                             # ROR: 'is not' → 'is'
        kwargs["columns"] = columns                     # SDL: delete this line

    if method is not None:                              # ROR: 'is not' → 'is'
        kwargs["method"] = method                       # SDL: delete this line

    if not _is_nan_sentinel(fill_value):               # COR: remove 'not'
        kwargs["fill_value"] = fill_value               # SDL: delete this line

    if limit is not None:                               # ROR: 'is not' → 'is'
        kwargs["limit"] = limit                         # SDL: delete this line
        # AOR target: kwargs["limit"] = limit + 1 (off-by-one)

    if tolerance is not None:                           # ROR: 'is not' → 'is'
        kwargs["tolerance"] = tolerance                 # SDL: delete this line

    if level is not None:                               # ROR: 'is not' → 'is'
        kwargs["level"] = level                         # SDL: delete this line

    return _ORIGINAL_REINDEX(self, **kwargs)


# ── install / uninstall ──────────────────────────────────────────────────────

def install() -> None:
    """Patch pd.DataFrame.reindex to use reindex_shim."""
    DataFrame.reindex = reindex_shim


def uninstall() -> None:
    """Restore the original pd.DataFrame.reindex."""
    DataFrame.reindex = _ORIGINAL_REINDEX
