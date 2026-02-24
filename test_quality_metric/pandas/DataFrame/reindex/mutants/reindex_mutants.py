"""Runtime mutants for pandas.DataFrame.reindex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from pandas import DataFrame

MutantFunc = Callable[..., object]
_ORIGINAL_REINDEX = DataFrame.reindex
_ACTIVE_MUTANT_ID: str | None = None


@dataclass(frozen=True)
class MutantSpec:
    mutant_id: str
    description: str
    impl: MutantFunc


def _mutant_c1_not_conformed_return(self, *args, **kwargs):
    return self.copy(deep=True)


def _mutant_c2_missing_labels_not_nan(self, *args, **kwargs):
    return _ORIGINAL_REINDEX(self, *args, **kwargs).fillna(0)


def _mutant_c3_ignore_fill_value(self, *args, **kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("fill_value", None)
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c4_fill_value_numeric_only(self, *args, **kwargs):
    fv = kwargs.get("fill_value", None)
    if fv is not None and not isinstance(fv, (int, float, complex, bool)):
        raise TypeError("Mutant C4 rejects non-numeric fill_value")
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c5_ffill_backward(self, *args, **kwargs):
    kwargs = dict(kwargs)
    if kwargs.get("method") in {"ffill", "pad"}:
        kwargs["method"] = "bfill"
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c6_bfill_forward(self, *args, **kwargs):
    kwargs = dict(kwargs)
    if kwargs.get("method") in {"bfill", "backfill"}:
        kwargs["method"] = "ffill"
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c7_nearest_as_ffill(self, *args, **kwargs):
    kwargs = dict(kwargs)
    if kwargs.get("method") == "nearest":
        kwargs["method"] = "ffill"
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c8_method_without_monotonicity(self, *args, **kwargs):
    kwargs = dict(kwargs)
    if kwargs.get("method") is not None:
        kwargs["method"] = None
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c9_fill_existing_nan(self, *args, **kwargs):
    return _ORIGINAL_REINDEX(self, *args, **kwargs).fillna(0)


def _mutant_c10_ignore_limit(self, *args, **kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("limit", None)
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c11_break_index_columns_convention(self, *args, **kwargs):
    kwargs = dict(kwargs)
    if "index" in kwargs and "columns" in kwargs:
        kwargs.pop("columns", None)
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c12_break_labels_axis_convention(self, *args, **kwargs):
    kwargs = dict(kwargs)
    if args:
        kwargs["axis"] = 0
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c13_axis_int_rejected(self, *args, **kwargs):
    axis = kwargs.get("axis", None)
    if isinstance(axis, int):
        raise ValueError("Mutant C13 rejects integer axis")
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c14_columns_missing_not_nan(self, *args, **kwargs):
    kwargs = dict(kwargs)
    if "columns" in kwargs and "fill_value" not in kwargs:
        kwargs["fill_value"] = 0
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c15_scalar_tolerance_not_uniform(self, *args, **kwargs):
    kwargs = dict(kwargs)
    tol = kwargs.get("tolerance", None)
    if isinstance(tol, (int, float)):
        kwargs["tolerance"] = tol / 10
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c16_list_tolerance_len_dtype_ignored(self, *args, **kwargs):
    kwargs = dict(kwargs)
    tol = kwargs.get("tolerance", None)
    if isinstance(tol, (list, tuple, pd.Series, pd.Index)) and len(tol) > 0:
        kwargs["tolerance"] = tol[0]
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c17_tolerance_inequality_relaxed(self, *args, **kwargs):
    kwargs = dict(kwargs)
    tol = kwargs.get("tolerance", None)
    if isinstance(tol, (int, float)):
        kwargs["tolerance"] = tol * 10
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c18_ignore_level_param(self, *args, **kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("level", None)
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c19_copy_changes_behavior(self, *args, **kwargs):
    kwargs = dict(kwargs)
    copy_flag = kwargs.get("copy", None)
    if copy_flag is True:
        return self
    if copy_flag is False:
        return self.copy(deep=True)
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


def _mutant_c20_force_materialize_non_index(self, *args, **kwargs):
    kwargs = dict(kwargs)
    for key in ("labels", "index", "columns"):
        value = kwargs.get(key)
        if isinstance(value, pd.Index):
            kwargs[key] = list(value)
    return _ORIGINAL_REINDEX(self, *args, **kwargs)


_MUTANTS: dict[str, MutantSpec] = {
    "M_C1_not_conformed_return": MutantSpec("M_C1_not_conformed_return", "C1", _mutant_c1_not_conformed_return),
    "M_C2_missing_labels_not_nan": MutantSpec("M_C2_missing_labels_not_nan", "C2", _mutant_c2_missing_labels_not_nan),
    "M_C3_ignore_fill_value": MutantSpec("M_C3_ignore_fill_value", "C3", _mutant_c3_ignore_fill_value),
    "M_C4_fill_value_numeric_only": MutantSpec("M_C4_fill_value_numeric_only", "C4", _mutant_c4_fill_value_numeric_only),
    "M_C5_ffill_backward": MutantSpec("M_C5_ffill_backward", "C5", _mutant_c5_ffill_backward),
    "M_C6_bfill_forward": MutantSpec("M_C6_bfill_forward", "C6", _mutant_c6_bfill_forward),
    "M_C7_nearest_as_ffill": MutantSpec("M_C7_nearest_as_ffill", "C7", _mutant_c7_nearest_as_ffill),
    "M_C8_method_without_monotonicity": MutantSpec("M_C8_method_without_monotonicity", "C8", _mutant_c8_method_without_monotonicity),
    "M_C9_fill_existing_nan": MutantSpec("M_C9_fill_existing_nan", "C9", _mutant_c9_fill_existing_nan),
    "M_C10_ignore_limit": MutantSpec("M_C10_ignore_limit", "C10", _mutant_c10_ignore_limit),
    "M_C11_break_index_columns_convention": MutantSpec("M_C11_break_index_columns_convention", "C11", _mutant_c11_break_index_columns_convention),
    "M_C12_break_labels_axis_convention": MutantSpec("M_C12_break_labels_axis_convention", "C12", _mutant_c12_break_labels_axis_convention),
    "M_C13_axis_int_rejected": MutantSpec("M_C13_axis_int_rejected", "C13", _mutant_c13_axis_int_rejected),
    "M_C14_columns_missing_not_nan": MutantSpec("M_C14_columns_missing_not_nan", "C14", _mutant_c14_columns_missing_not_nan),
    "M_C15_scalar_tolerance_not_uniform": MutantSpec("M_C15_scalar_tolerance_not_uniform", "C15", _mutant_c15_scalar_tolerance_not_uniform),
    "M_C16_list_tolerance_len_dtype_ignored": MutantSpec("M_C16_list_tolerance_len_dtype_ignored", "C16", _mutant_c16_list_tolerance_len_dtype_ignored),
    "M_C17_tolerance_inequality_relaxed": MutantSpec("M_C17_tolerance_inequality_relaxed", "C17", _mutant_c17_tolerance_inequality_relaxed),
    "M_C18_ignore_level_param": MutantSpec("M_C18_ignore_level_param", "C18", _mutant_c18_ignore_level_param),
    "M_C19_copy_changes_behavior": MutantSpec("M_C19_copy_changes_behavior", "C19", _mutant_c19_copy_changes_behavior),
    "M_C20_force_materialize_non_index": MutantSpec("M_C20_force_materialize_non_index", "C20", _mutant_c20_force_materialize_non_index)
}


def list_mutants() -> list[dict[str, str]]:
    return [{"mutant_id": s.mutant_id, "description": s.description} for s in _MUTANTS.values()]


def apply_mutant(mutant_id: str) -> dict[str, str]:
    global _ACTIVE_MUTANT_ID
    if mutant_id not in _MUTANTS:
        valid = ", ".join(sorted(_MUTANTS.keys()))
        raise ValueError(f"Unknown mutant_id={mutant_id!r}. Valid: {valid}")
    spec = _MUTANTS[mutant_id]
    DataFrame.reindex = spec.impl
    _ACTIVE_MUTANT_ID = mutant_id
    return {"mutant_id": spec.mutant_id, "description": spec.description}


def reset_mutant() -> None:
    global _ACTIVE_MUTANT_ID
    DataFrame.reindex = _ORIGINAL_REINDEX
    _ACTIVE_MUTANT_ID = None


def get_active_mutant() -> str | None:
    return _ACTIVE_MUTANT_ID
