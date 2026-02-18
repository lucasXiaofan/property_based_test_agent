"""Shared pytest hooks for pandas_bug_finding tests."""

from __future__ import annotations

import os

import pytest

from mutants.ewm_aggregate.ewm_aggregate_mutants import apply_mutant, reset_mutant


@pytest.fixture(scope="session", autouse=True)
def _apply_requested_mutant():
    """Apply runtime mutant when MUTANT_ID is set; otherwise no-op."""
    mutant_id = os.environ.get("MUTANT_ID")
    if not mutant_id:
        yield
        return

    info = apply_mutant(mutant_id)
    print(
        f"[mutant] active={info['mutant_id']} target="
        "pandas.core.window.ewm.ExponentialMovingWindow.aggregate"
    )
    try:
        yield
    finally:
        reset_mutant()
        print("[mutant] reset to original aggregate")
