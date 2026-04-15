from __future__ import annotations

import sys
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in (CASE_DIR, *CASE_DIR.parents) if (parent / "pyproject.toml").exists())
HELPER_DIR = REPO_ROOT / "experiments" / "oracle_evaluation" / "line_branch_coverage"
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from local_pandas_conftest_helper import configure_local_pandas_case

CASE_CONFIG = configure_local_pandas_case(__file__)


if CASE_CONFIG["has_mutant_wrapper"]:
    @pytest.fixture(scope="session", autouse=True)
    def install_case_mutants():
        from mutant_wrapper import install_mutants, uninstall_mutants

        install_mutants()
        yield
        uninstall_mutants()
