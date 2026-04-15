# Local Coverage `conftest.py` Guide

Use this pattern for any new pandas case directory under `experiments/oracle_generation/pandas/...`.

## Goal

Make pytest import the local pandas `3.0.0` checkout from `local_pandas/pandas`, not the site-packages wheel, and fail fast if the target API resolves outside the local checkout.

## Steps

1. Add a `conftest.py` in the target function folder.
2. Insert the case folder and `experiments/oracle_evaluation/line_branch_coverage` onto `sys.path`.
3. Import `configure_local_pandas_case` from `local_pandas_conftest_helper.py`.
4. Call `configure_local_pandas_case(__file__)` at import time.
5. If the folder has a `mutant_wrapper.py`, keep the autouse fixture that installs and uninstalls mutants.

## Template

```python
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
```

## Notes

- The helper derives the target API from the folder path, so keep the case directory structure aligned with the pandas API path.
- The helper currently supports the regular method cases in this repo plus `Series.str.*` accessors.
- If a new accessor family is added later, update `local_pandas_conftest_helper.py` so coverage resolution can find the correct owner class.
