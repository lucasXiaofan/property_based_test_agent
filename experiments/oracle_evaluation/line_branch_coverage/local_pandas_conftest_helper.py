from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path
from types import FunctionType, MethodType
from typing import Any


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {start}")


def _prepend_once(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _ensure_path_prefix(path: Path) -> None:
    value = str(path.resolve())
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if value not in parts:
        os.environ["PATH"] = os.pathsep.join([value, *parts]) if parts else value


def _unwrap_callable(obj: Any) -> Any:
    if isinstance(obj, (FunctionType, MethodType)):
        return getattr(obj, "__func__", obj)
    return obj


def _resolve_accessor_method(api_path: str) -> tuple[Any, type, str] | None:
    import pandas as pd

    tokens = api_path.split(".")
    if tokens[:3] == ["pandas", "Series", "str"] and len(tokens) == 4:
        accessor = pd.Series(["x"]).str
        owner = accessor.__class__
        method_name = tokens[-1]
        return getattr(owner, method_name), owner, method_name
    return None


def resolve_api(api_path: str) -> tuple[Any, type | None, str | None]:
    accessor_result = _resolve_accessor_method(api_path)
    if accessor_result is not None:
        return accessor_result

    if not api_path.startswith("pandas."):
        raise RuntimeError(f"Unsupported API path: {api_path}")

    root = importlib.import_module("pandas")
    obj: Any = root
    owner_cls: type | None = None
    method_name: str | None = None

    for token in api_path.split(".")[1:]:
        prev = obj
        try:
            obj = getattr(obj, token)
        except AttributeError as exc:
            raise RuntimeError(
                f"Could not resolve token {token!r} in {api_path!r}"
            ) from exc
        if inspect.isclass(prev):
            owner_cls = prev
            method_name = token

    return obj, owner_cls, method_name


def case_dir_to_api(case_dir: Path, cases_root: Path) -> str:
    rel = case_dir.resolve().relative_to(cases_root.resolve())
    return f"pandas.{'.'.join(rel.parts)}"


def configure_local_pandas_case(conftest_file: str) -> dict[str, Any]:
    case_dir = Path(conftest_file).resolve().parent
    repo_root = _find_repo_root(case_dir)
    cases_root = repo_root / "experiments" / "oracle_generation" / "pandas"
    local_pandas_root = repo_root / "local_pandas" / "pandas"
    venv_bin = repo_root / ".venv" / "bin"
    helper_dir = repo_root / "experiments" / "oracle_evaluation" / "line_branch_coverage"

    _prepend_once(case_dir)
    _prepend_once(helper_dir)
    _prepend_once(local_pandas_root)
    _ensure_path_prefix(venv_bin)
    importlib.invalidate_caches()

    import pandas as pd

    if pd.__version__ != "3.0.0":
        raise RuntimeError(f"Expected local pandas 3.0.0, got {pd.__version__}")

    pandas_module = Path(pd.__file__).resolve()
    if not pandas_module.is_relative_to(local_pandas_root.resolve()):
        raise RuntimeError(
            f"Tests must import pandas from {local_pandas_root.resolve()}, got {pandas_module}"
        )

    target_api = case_dir_to_api(case_dir, cases_root)
    target_obj, owner_cls, method_name = resolve_api(target_api)
    if owner_cls is None or method_name is None:
        raise RuntimeError(f"Could not resolve method owner for {target_api}")

    target_callable = _unwrap_callable(getattr(owner_cls, method_name))
    source_file = Path(inspect.getsourcefile(target_callable) or "").resolve()
    if not source_file.is_relative_to(local_pandas_root.resolve()):
        raise RuntimeError(
            f"Target API {target_api} is not resolving from the local pandas checkout: {source_file}"
        )

    return {
        "case_dir": case_dir,
        "repo_root": repo_root,
        "target_api": target_api,
        "target_source": source_file,
        "has_mutant_wrapper": (case_dir / "mutant_wrapper.py").exists(),
    }
