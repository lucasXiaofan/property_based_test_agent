"""Runtime mutants for pandas.core.window.ewm.ExponentialMovingWindow.aggregate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pandas.core.window.ewm import ExponentialMovingWindow


MutantFunc = Callable[..., object]
_ORIGINAL_AGGREGATE = ExponentialMovingWindow.aggregate
_ACTIVE_MUTANT_ID: str | None = None


@dataclass(frozen=True)
class MutantSpec:
    mutant_id: str
    description: str
    impl: MutantFunc


def _mutant_m1(self, func=None, *args, **kwargs):
    """Semantic bug: misroutes 'sum' requests to 'mean'."""
    if func == "sum":
        func = "mean"
    return _ORIGINAL_AGGREGATE(self, func, *args, **kwargs)


def _mutant_m2(self, func=None, *args, **kwargs):
    """Semantic bug: rejects multi-function list aggregations."""
    if isinstance(func, list) and len(func) > 1:
        raise ValueError("Mutant M2 rejects multi-function list aggregations")
    return _ORIGINAL_AGGREGATE(self, func, *args, **kwargs)


def _mutant_m3(self, func=None, *args, **kwargs):
    """Semantic bug: callable func path raises wrong exception type."""
    if callable(func):
        raise TypeError("Mutant M3 rejects callable funcs")
    return _ORIGINAL_AGGREGATE(self, func, *args, **kwargs)


_MUTANTS: dict[str, MutantSpec] = {
    "M1_SUM_TO_MEAN": MutantSpec(
        mutant_id="M1_SUM_TO_MEAN",
        description="If func == 'sum', force func='mean' before delegation.",
        impl=_mutant_m1,
    ),
    "M2_TRUNCATE_LIST_FUNCS": MutantSpec(
        mutant_id="M2_TRUNCATE_LIST_FUNCS",
        description="If func is a multi-function list, raise ValueError.",
        impl=_mutant_m2,
    ),
    "M3_CALLABLE_TYPEERROR": MutantSpec(
        mutant_id="M3_CALLABLE_TYPEERROR",
        description="Raise TypeError for callable funcs instead of pandas behavior.",
        impl=_mutant_m3,
    ),
}


def list_mutants() -> list[dict[str, str]]:
    """Return stable mutant metadata for scripts/reporting."""
    return [
        {"mutant_id": spec.mutant_id, "description": spec.description}
        for spec in _MUTANTS.values()
    ]


def apply_mutant(mutant_id: str) -> dict[str, str]:
    """Patch ExponentialMovingWindow.aggregate in-process."""
    global _ACTIVE_MUTANT_ID

    if mutant_id not in _MUTANTS:
        valid = ", ".join(sorted(_MUTANTS.keys()))
        raise ValueError(f"Unknown mutant_id={mutant_id!r}. Valid: {valid}")

    spec = _MUTANTS[mutant_id]
    ExponentialMovingWindow.aggregate = spec.impl
    ExponentialMovingWindow.agg = spec.impl
    _ACTIVE_MUTANT_ID = mutant_id
    return {"mutant_id": spec.mutant_id, "description": spec.description}


def reset_mutant() -> None:
    """Restore original pandas implementation."""
    global _ACTIVE_MUTANT_ID
    ExponentialMovingWindow.aggregate = _ORIGINAL_AGGREGATE
    ExponentialMovingWindow.agg = _ORIGINAL_AGGREGATE
    _ACTIVE_MUTANT_ID = None


def get_active_mutant() -> str | None:
    return _ACTIVE_MUTANT_ID
