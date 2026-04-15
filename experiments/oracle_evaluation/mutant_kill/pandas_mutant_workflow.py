#!/usr/bin/env python3
"""Sync pandas case docs and generate doc-grounded mutant wrappers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.oracle_evaluation.pandas_eval_common import API_CASES, DOWNLOADED_DOCS_DIR


WRAPPER_BODIES: dict[str, str] = {
    "DataFrame/groupby": dedent(
        '''
        """Mutant wrappers for pandas.DataFrame.groupby."""

        from __future__ import annotations

        import os

        import pandas as pd

        ORIGINAL_GROUPBY = None


        def get_mutant_id():
            return os.environ.get("MUTANT_ID")


        def mutant_groupby_M1(
            self,
            by=None,
            level=None,
            *,
            as_index=True,
            sort=True,
            group_keys=True,
            observed=True,
            dropna=True,
        ):
            """M1: ignore sort=False and always sort group keys."""
            if sort is False:
                sort = True
            return ORIGINAL_GROUPBY(
                self,
                by=by,
                level=level,
                as_index=as_index,
                sort=sort,
                group_keys=group_keys,
                observed=observed,
                dropna=dropna,
            )


        def mutant_groupby_M2(
            self,
            by=None,
            level=None,
            *,
            as_index=True,
            sort=True,
            group_keys=True,
            observed=True,
            dropna=True,
        ):
            """M2: ignore dropna=False and always drop NA groups."""
            if dropna is False:
                dropna = True
            return ORIGINAL_GROUPBY(
                self,
                by=by,
                level=level,
                as_index=as_index,
                sort=sort,
                group_keys=group_keys,
                observed=observed,
                dropna=dropna,
            )


        def install_mutants():
            global ORIGINAL_GROUPBY
            if ORIGINAL_GROUPBY is not None:
                return
            ORIGINAL_GROUPBY = pd.DataFrame.groupby
            mutant_id = get_mutant_id()
            if mutant_id == "M1":
                pd.DataFrame.groupby = mutant_groupby_M1
            elif mutant_id == "M2":
                pd.DataFrame.groupby = mutant_groupby_M2


        def uninstall_mutants():
            global ORIGINAL_GROUPBY
            if ORIGINAL_GROUPBY is None:
                return
            pd.DataFrame.groupby = ORIGINAL_GROUPBY
            ORIGINAL_GROUPBY = None


        MUTANT_INFO = {
            "M1": {
                "name": "force_sort_true",
                "description": "Force group keys to be sorted even when the caller requests sort=False.",
                "doc_anchor": "sort bool, default True",
                "expected_kill": "Tests that assert group order changes when sort=False.",
            },
            "M2": {
                "name": "force_dropna_true",
                "description": "Drop NA groups even when the caller requests dropna=False.",
                "doc_anchor": "dropna bool, default True",
                "expected_kill": "Tests that assert NA-key groups are retained with dropna=False.",
            },
        }
        '''
    ).strip()
    + "\n",
    "DataFrame/reindex": dedent(
        '''
        """Mutant wrappers for pandas.DataFrame.reindex."""

        from __future__ import annotations

        import os

        import numpy as np
        import pandas as pd

        ORIGINAL_REINDEX = None


        def get_mutant_id():
            return os.environ.get("MUTANT_ID")


        def mutant_reindex_M1(
            self,
            labels=None,
            *,
            index=None,
            columns=None,
            axis=None,
            method=None,
            copy=pd.api.extensions.no_default,
            level=None,
            fill_value=np.nan,
            limit=None,
            tolerance=None,
        ):
            """M1: ignore a caller-specified fill_value."""
            if fill_value is not None and not pd.isna(fill_value):
                fill_value = np.nan
            return ORIGINAL_REINDEX(
                self,
                labels=labels,
                index=index,
                columns=columns,
                axis=axis,
                method=method,
                copy=copy,
                level=level,
                fill_value=fill_value,
                limit=limit,
                tolerance=tolerance,
            )


        def mutant_reindex_M2(
            self,
            labels=None,
            *,
            index=None,
            columns=None,
            axis=None,
            method=None,
            copy=pd.api.extensions.no_default,
            level=None,
            fill_value=np.nan,
            limit=None,
            tolerance=None,
        ):
            """M2: raise on multi-column reindex with a string fill_value."""
            target_columns = columns
            if target_columns is None and axis in (1, "columns"):
                target_columns = labels
            if (
                target_columns is not None
                and isinstance(fill_value, str)
                and len([col for col in target_columns if col not in self.columns]) >= 2
            ):
                raise TypeError("mutant: string fill_value unsupported for multiple missing columns")
            return ORIGINAL_REINDEX(
                self,
                labels=labels,
                index=index,
                columns=columns,
                axis=axis,
                method=method,
                copy=copy,
                level=level,
                fill_value=fill_value,
                limit=limit,
                tolerance=tolerance,
            )


        def install_mutants():
            global ORIGINAL_REINDEX
            if ORIGINAL_REINDEX is not None:
                return
            ORIGINAL_REINDEX = pd.DataFrame.reindex
            mutant_id = get_mutant_id()
            if mutant_id == "M1":
                pd.DataFrame.reindex = mutant_reindex_M1
            elif mutant_id == "M2":
                pd.DataFrame.reindex = mutant_reindex_M2


        def uninstall_mutants():
            global ORIGINAL_REINDEX
            if ORIGINAL_REINDEX is None:
                return
            pd.DataFrame.reindex = ORIGINAL_REINDEX
            ORIGINAL_REINDEX = None


        MUTANT_INFO = {
            "M1": {
                "name": "ignore_fill_value",
                "description": "Remove caller-specified fill_value handling and fall back to NaN.",
                "doc_anchor": "fill_value scalar, default np.nan",
                "expected_kill": "Tests that assert explicit fill_value is reflected in newly created labels.",
            },
            "M2": {
                "name": "raise_multi_column_string_fill",
                "description": "Raise a TypeError when reindexing columns with a string fill_value and multiple missing columns.",
                "doc_anchor": "Conform DataFrame to new index with optional filling logic",
                "expected_kill": "Tests that cover successful multi-column column-reindex with string fill_value.",
            },
        }
        '''
    ).strip()
    + "\n",
    "DataFrame/to_json": dedent(
        '''
        """Mutant wrappers for pandas.DataFrame.to_json."""

        from __future__ import annotations

        import os

        import pandas as pd

        ORIGINAL_TO_JSON = None


        def get_mutant_id():
            return os.environ.get("MUTANT_ID")


        def mutant_to_json_M1(
            self,
            path_or_buf=None,
            *,
            orient=None,
            date_format=None,
            double_precision=10,
            force_ascii=True,
            date_unit="ms",
            default_handler=None,
            lines=False,
            compression="infer",
            index=None,
            indent=None,
            storage_options=None,
            mode="w",
        ):
            """M1: ignore force_ascii=False and always escape as ASCII."""
            if force_ascii is False:
                force_ascii = True
            return ORIGINAL_TO_JSON(
                self,
                path_or_buf=path_or_buf,
                orient=orient,
                date_format=date_format,
                double_precision=double_precision,
                force_ascii=force_ascii,
                date_unit=date_unit,
                default_handler=default_handler,
                lines=lines,
                compression=compression,
                index=index,
                indent=indent,
                storage_options=storage_options,
                mode=mode,
            )


        def mutant_to_json_M2(
            self,
            path_or_buf=None,
            *,
            orient=None,
            date_format=None,
            double_precision=10,
            force_ascii=True,
            date_unit="ms",
            default_handler=None,
            lines=False,
            compression="infer",
            index=None,
            indent=None,
            storage_options=None,
            mode="w",
        ):
            """M2: corrupt epoch scaling by rewriting date_unit."""
            mutated_date_unit = date_unit
            if date_format in (None, "epoch"):
                if date_unit == "ms":
                    mutated_date_unit = "us"
                elif date_unit == "s":
                    mutated_date_unit = "ms"
            return ORIGINAL_TO_JSON(
                self,
                path_or_buf=path_or_buf,
                orient=orient,
                date_format=date_format,
                double_precision=double_precision,
                force_ascii=force_ascii,
                date_unit=mutated_date_unit,
                default_handler=default_handler,
                lines=lines,
                compression=compression,
                index=index,
                indent=indent,
                storage_options=storage_options,
                mode=mode,
            )


        def install_mutants():
            global ORIGINAL_TO_JSON
            if ORIGINAL_TO_JSON is not None:
                return
            ORIGINAL_TO_JSON = pd.DataFrame.to_json
            mutant_id = get_mutant_id()
            if mutant_id == "M1":
                pd.DataFrame.to_json = mutant_to_json_M1
            elif mutant_id == "M2":
                pd.DataFrame.to_json = mutant_to_json_M2


        def uninstall_mutants():
            global ORIGINAL_TO_JSON
            if ORIGINAL_TO_JSON is None:
                return
            pd.DataFrame.to_json = ORIGINAL_TO_JSON
            ORIGINAL_TO_JSON = None


        MUTANT_INFO = {
            "M1": {
                "name": "force_ascii_true",
                "description": "Serialize with ASCII escaping even when the caller disables it.",
                "doc_anchor": "force_ascii bool, default True",
                "expected_kill": "Tests that assert Unicode characters survive when force_ascii=False.",
            },
            "M2": {
                "name": "corrupt_epoch_date_unit",
                "description": "Corrupt epoch-based datetime units by rewriting date_unit before serialization.",
                "doc_anchor": "date_unit controls timestamp unit",
                "expected_kill": "Tests that assert epoch timestamp scale or date_unit behavior.",
            },
        }
        '''
    ).strip()
    + "\n",
    "Index/astype": dedent(
        '''
        """Mutant wrappers for pandas.Index.astype."""

        from __future__ import annotations

        import os

        import pandas as pd

        ORIGINAL_ASTYPE = None


        def get_mutant_id():
            return os.environ.get("MUTANT_ID")


        def mutant_astype_M1(self, dtype, copy=True):
            """M1: ignore requested dtype changes."""
            requested = pd.api.types.pandas_dtype(dtype)
            if requested != self.dtype:
                return self.copy(deep=True) if copy else self
            return ORIGINAL_ASTYPE(self, dtype=dtype, copy=copy)


        def mutant_astype_M2(self, dtype, copy=True):
            """M2: force copying on no-op astype(copy=False)."""
            requested = pd.api.types.pandas_dtype(dtype)
            if requested == self.dtype and copy is False:
                return ORIGINAL_ASTYPE(self, dtype=dtype, copy=True)
            return ORIGINAL_ASTYPE(self, dtype=dtype, copy=copy)


        def install_mutants():
            global ORIGINAL_ASTYPE
            if ORIGINAL_ASTYPE is not None:
                return
            ORIGINAL_ASTYPE = pd.Index.astype
            mutant_id = get_mutant_id()
            if mutant_id == "M1":
                pd.Index.astype = mutant_astype_M1
            elif mutant_id == "M2":
                pd.Index.astype = mutant_astype_M2


        def uninstall_mutants():
            global ORIGINAL_ASTYPE
            if ORIGINAL_ASTYPE is None:
                return
            pd.Index.astype = ORIGINAL_ASTYPE
            ORIGINAL_ASTYPE = None


        MUTANT_INFO = {
            "M1": {
                "name": "ignore_dtype_change",
                "description": "Return the original index unchanged when the caller asks for a different dtype.",
                "doc_anchor": "dtype str or dtype",
                "expected_kill": "Tests that assert dtype conversion and converted values.",
            },
            "M2": {
                "name": "copy_false_still_copies",
                "description": "Return a distinct object even when astype is a no-op with copy=False.",
                "doc_anchor": "copy bool, default True",
                "expected_kill": "Tests that assert identity semantics for no-op astype(copy=False).",
            },
        }
        '''
    ).strip()
    + "\n",
    "Index/shift": dedent(
        '''
        """Mutant wrappers for pandas.Index.shift."""

        from __future__ import annotations

        import os

        import pandas as pd

        ORIGINAL_SHIFT = {}
        PATCH_TYPES = (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)


        def get_mutant_id():
            return os.environ.get("MUTANT_ID")


        def mutant_shift_M1(self, periods=1, freq=None):
            """M1: reverse the requested shift direction."""
            return ORIGINAL_SHIFT[type(self)](self, periods=-periods, freq=freq)


        def mutant_shift_M2(self, periods=1, freq=None):
            """M2: ignore an explicit freq argument."""
            if freq is not None:
                freq = None
            return ORIGINAL_SHIFT[type(self)](self, periods=periods, freq=freq)


        def install_mutants():
            if ORIGINAL_SHIFT:
                return
            mutant_id = get_mutant_id()
            for cls in PATCH_TYPES:
                ORIGINAL_SHIFT[cls] = cls.shift
                if mutant_id == "M1":
                    cls.shift = mutant_shift_M1
                elif mutant_id == "M2":
                    cls.shift = mutant_shift_M2


        def uninstall_mutants():
            if not ORIGINAL_SHIFT:
                return
            for cls, original in ORIGINAL_SHIFT.items():
                cls.shift = original
            ORIGINAL_SHIFT.clear()


        MUTANT_INFO = {
            "M1": {
                "name": "negate_periods",
                "description": "Apply the opposite shift direction from the caller-provided periods.",
                "doc_anchor": "periods int, can be positive or negative",
                "expected_kill": "Tests that assert positive and negative shifts move values in the expected direction.",
            },
            "M2": {
                "name": "ignore_explicit_freq",
                "description": "Discard an explicit freq argument and reuse implicit index frequency behavior instead.",
                "doc_anchor": "freq DateOffset, timedelta, or str, optional",
                "expected_kill": "Tests that assert explicit freq changes the shift increment.",
            },
        }
        '''
    ).strip()
    + "\n",
    "Series/factorize": dedent(
        '''
        """Mutant wrappers for pandas.Series.factorize."""

        from __future__ import annotations

        import os

        import pandas as pd

        ORIGINAL_FACTORIZE = None


        def get_mutant_id():
            return os.environ.get("MUTANT_ID")


        def mutant_factorize_M1(self, sort=False, use_na_sentinel=True):
            """M1: ignore sort=True and keep first-occurrence order."""
            if sort is True:
                sort = False
            return ORIGINAL_FACTORIZE(self, sort=sort, use_na_sentinel=use_na_sentinel)


        def mutant_factorize_M2(self, sort=False, use_na_sentinel=True):
            """M2: ignore use_na_sentinel=True and emit non-negative NA codes."""
            if use_na_sentinel is True:
                use_na_sentinel = False
            return ORIGINAL_FACTORIZE(self, sort=sort, use_na_sentinel=use_na_sentinel)


        def install_mutants():
            global ORIGINAL_FACTORIZE
            if ORIGINAL_FACTORIZE is not None:
                return
            ORIGINAL_FACTORIZE = pd.Series.factorize
            mutant_id = get_mutant_id()
            if mutant_id == "M1":
                pd.Series.factorize = mutant_factorize_M1
            elif mutant_id == "M2":
                pd.Series.factorize = mutant_factorize_M2


        def uninstall_mutants():
            global ORIGINAL_FACTORIZE
            if ORIGINAL_FACTORIZE is None:
                return
            pd.Series.factorize = ORIGINAL_FACTORIZE
            ORIGINAL_FACTORIZE = None


        MUTANT_INFO = {
            "M1": {
                "name": "force_sort_false",
                "description": "Force first-occurrence order even when the caller requests sort=True.",
                "doc_anchor": "sort bool, default False",
                "expected_kill": "Tests that assert sort=True changes uniques order and codes accordingly.",
            },
            "M2": {
                "name": "force_use_na_sentinel_false",
                "description": "Encode missing values as ordinary categories even when the caller requests sentinel -1 behavior.",
                "doc_anchor": "use_na_sentinel bool, default True",
                "expected_kill": "Tests that assert NA handling for codes and uniques with use_na_sentinel=True.",
            },
        }
        '''
    ).strip()
    + "\n",
    "Series/mean": dedent(
        '''
        """Mutant wrappers for pandas.Series.mean."""

        from __future__ import annotations

        import os

        import pandas as pd

        ORIGINAL_MEAN = None


        def get_mutant_id():
            return os.environ.get("MUTANT_ID")


        def mutant_mean_M1(self, *, axis=0, skipna=True, numeric_only=False, **kwargs):
            """M1: ignore skipna=True and poison the result with NA propagation."""
            if skipna is True:
                skipna = False
            return ORIGINAL_MEAN(
                self,
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
                **kwargs,
            )


        def mutant_mean_M2(self, *, axis=0, skipna=True, numeric_only=False, **kwargs):
            """M2: bias numeric results while keeping the API shape intact."""
            result = ORIGINAL_MEAN(
                self,
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
                **kwargs,
            )
            if pd.notna(result) and isinstance(result, (int, float)):
                return result + 1.0
            return result


        def install_mutants():
            global ORIGINAL_MEAN
            if ORIGINAL_MEAN is not None:
                return
            ORIGINAL_MEAN = pd.Series.mean
            mutant_id = get_mutant_id()
            if mutant_id == "M1":
                pd.Series.mean = mutant_mean_M1
            elif mutant_id == "M2":
                pd.Series.mean = mutant_mean_M2


        def uninstall_mutants():
            global ORIGINAL_MEAN
            if ORIGINAL_MEAN is None:
                return
            pd.Series.mean = ORIGINAL_MEAN
            ORIGINAL_MEAN = None


        MUTANT_INFO = {
            "M1": {
                "name": "force_skipna_false",
                "description": "Treat the operation as skipna=False, so missing values poison the result.",
                "doc_anchor": "skipna bool, default True",
                "expected_kill": "Tests that assert NaN values are ignored when skipna=True.",
            },
            "M2": {
                "name": "bias_numeric_result",
                "description": "Return a numerically biased mean while keeping the return type otherwise plausible.",
                "doc_anchor": "Return the mean of the values",
                "expected_kill": "Tests that assert exact or approximate mean values.",
            },
        }
        '''
    ).strip()
    + "\n",
    "Series/mul": dedent(
        '''
        """Mutant wrappers for pandas.Series.mul."""

        from __future__ import annotations

        import os

        import pandas as pd

        ORIGINAL_MUL = None


        def get_mutant_id():
            return os.environ.get("MUTANT_ID")


        def mutant_mul_M1(self, other, level=None, fill_value=None, axis=0):
            """M1: perform addition instead of multiplication."""
            return self.add(other, level=level, fill_value=fill_value, axis=axis)


        def mutant_mul_M2(self, other, level=None, fill_value=None, axis=0):
            """M2: ignore an explicit fill_value."""
            if fill_value is not None:
                fill_value = None
            return ORIGINAL_MUL(self, other, level=level, fill_value=fill_value, axis=axis)


        def install_mutants():
            global ORIGINAL_MUL
            if ORIGINAL_MUL is not None:
                return
            ORIGINAL_MUL = pd.Series.mul
            mutant_id = get_mutant_id()
            if mutant_id == "M1":
                pd.Series.mul = mutant_mul_M1
            elif mutant_id == "M2":
                pd.Series.mul = mutant_mul_M2


        def uninstall_mutants():
            global ORIGINAL_MUL
            if ORIGINAL_MUL is None:
                return
            pd.Series.mul = ORIGINAL_MUL
            ORIGINAL_MUL = None


        MUTANT_INFO = {
            "M1": {
                "name": "swap_mul_for_add",
                "description": "Delegate to addition semantics instead of multiplication semantics.",
                "doc_anchor": "Return multiplication of series and other",
                "expected_kill": "Tests that assert element-wise multiplication values.",
            },
            "M2": {
                "name": "ignore_fill_value",
                "description": "Drop fill_value handling so one-sided missing values remain missing.",
                "doc_anchor": "fill_value float or None",
                "expected_kill": "Tests that assert fill_value influences alignment with missing data.",
            },
        }
        '''
    ).strip()
    + "\n",
    "Series.str/contains": dedent(
        '''
        """Mutant wrappers for pandas.Series.str.contains."""

        from __future__ import annotations

        import os

        from pandas._libs import lib
        from pandas.core.strings.accessor import StringMethods

        ORIGINAL_CONTAINS = None


        def get_mutant_id():
            return os.environ.get("MUTANT_ID")


        def mutant_contains_M1(self, pat, case=True, flags=0, na=lib.no_default, regex=True):
            """M1: treat regex=False requests as regex=True."""
            if regex is False:
                regex = True
            return ORIGINAL_CONTAINS(self, pat, case=case, flags=flags, na=na, regex=regex)


        def mutant_contains_M2(self, pat, case=True, flags=0, na=lib.no_default, regex=True):
            """M2: ignore case=False and use case-sensitive matching."""
            if case is False:
                case = True
            return ORIGINAL_CONTAINS(self, pat, case=case, flags=flags, na=na, regex=regex)


        def install_mutants():
            global ORIGINAL_CONTAINS
            if ORIGINAL_CONTAINS is not None:
                return
            ORIGINAL_CONTAINS = StringMethods.contains
            mutant_id = get_mutant_id()
            if mutant_id == "M1":
                StringMethods.contains = mutant_contains_M1
            elif mutant_id == "M2":
                StringMethods.contains = mutant_contains_M2


        def uninstall_mutants():
            global ORIGINAL_CONTAINS
            if ORIGINAL_CONTAINS is None:
                return
            StringMethods.contains = ORIGINAL_CONTAINS
            ORIGINAL_CONTAINS = None


        MUTANT_INFO = {
            "M1": {
                "name": "regex_false_acts_like_regex_true",
                "description": "Treat literal matching requests as regex matching.",
                "doc_anchor": "regex bool, default True",
                "expected_kill": "Tests that distinguish literal and regex interpretation.",
            },
            "M2": {
                "name": "case_false_acts_like_case_true",
                "description": "Make case-insensitive searches behave as case-sensitive.",
                "doc_anchor": "case bool, default True",
                "expected_kill": "Tests that assert case=False performs case-insensitive matching.",
            },
        }
        '''
    ).strip()
    + "\n",
    "Series.str/match": dedent(
        '''
        """Mutant wrappers for pandas.Series.str.match."""

        from __future__ import annotations

        import os

        from pandas._libs import lib
        from pandas.core.strings.accessor import StringMethods

        ORIGINAL_MATCH = None


        def get_mutant_id():
            return os.environ.get("MUTANT_ID")


        def mutant_match_M1(self, pat, case=lib.no_default, flags=lib.no_default, na=lib.no_default):
            """M1: use contains-style search semantics instead of anchored match semantics."""
            contains_kwargs = {"pat": pat, "na": na, "regex": True}
            if case is not lib.no_default:
                contains_kwargs["case"] = case
            if flags is not lib.no_default:
                contains_kwargs["flags"] = flags
            return self.contains(**contains_kwargs)


        def mutant_match_M2(self, pat, case=lib.no_default, flags=lib.no_default, na=lib.no_default):
            """M2: ignore case=False and use case-sensitive matching."""
            if case is False:
                case = True
            return ORIGINAL_MATCH(self, pat, case=case, flags=flags, na=na)


        def install_mutants():
            global ORIGINAL_MATCH
            if ORIGINAL_MATCH is not None:
                return
            ORIGINAL_MATCH = StringMethods.match
            mutant_id = get_mutant_id()
            if mutant_id == "M1":
                StringMethods.match = mutant_match_M1
            elif mutant_id == "M2":
                StringMethods.match = mutant_match_M2


        def uninstall_mutants():
            global ORIGINAL_MATCH
            if ORIGINAL_MATCH is None:
                return
            StringMethods.match = ORIGINAL_MATCH
            ORIGINAL_MATCH = None


        MUTANT_INFO = {
            "M1": {
                "name": "match_acts_like_contains",
                "description": "Delegate to contains-style search behavior rather than start-anchored match behavior.",
                "doc_anchor": "Determine if each string starts with a match",
                "expected_kill": "Tests that distinguish search semantics from anchored match semantics.",
            },
            "M2": {
                "name": "case_false_acts_like_case_true",
                "description": "Make case-insensitive match requests behave as case-sensitive.",
                "doc_anchor": "case bool, default True",
                "expected_kill": "Tests that assert case=False performs case-insensitive matching.",
            },
        }
        '''
    ).strip()
    + "\n",
}


def canonical_case_doc_path(case_dir: Path, function_name: str) -> Path:
    return case_dir / f"{function_name}.md"


def find_best_doc_source(case) -> Path | None:
    if case.doc_path.exists():
        return case.doc_path
    canonical_local = canonical_case_doc_path(case.directory, case.function)
    if canonical_local.exists():
        return canonical_local
    markdown_files = sorted(case.directory.glob("*.md"))
    return markdown_files[0] if markdown_files else None


def sync_docs() -> list[dict[str, str]]:
    DOWNLOADED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    changes: list[dict[str, str]] = []

    for case in API_CASES:
        source = find_best_doc_source(case)
        if source is None:
            changes.append({"case_dir": case.case_dir, "status": "missing_doc"})
            continue

        source_text = source.read_text(encoding="utf-8")
        if not case.doc_path.exists() or case.doc_path.read_text(encoding="utf-8") != source_text:
            case.doc_path.write_text(source_text, encoding="utf-8")
            changes.append(
                {
                    "case_dir": case.case_dir,
                    "status": "backfilled_downloaded_doc" if source != case.doc_path else "refreshed_downloaded_doc",
                }
            )

        target = canonical_case_doc_path(case.directory, case.function)
        if not target.exists() or target.read_text(encoding="utf-8") != source_text:
            target.write_text(source_text, encoding="utf-8")
            changes.append({"case_dir": case.case_dir, "status": "synced_case_doc"})

    return changes


def generate_wrappers() -> list[Path]:
    generated: list[Path] = []
    for case in API_CASES:
        wrapper_body = WRAPPER_BODIES[case.case_dir]
        wrapper_path = case.directory / "mutant_wrapper.py"
        if not wrapper_path.exists() or wrapper_path.read_text(encoding="utf-8") != wrapper_body:
            wrapper_path.write_text(wrapper_body, encoding="utf-8")
            generated.append(wrapper_path)
    return generated


def materialize() -> tuple[list[dict[str, str]], list[Path]]:
    return sync_docs(), generate_wrappers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("sync-docs", "generate-wrappers", "materialize"),
        nargs="?",
        default="materialize",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "sync-docs":
        changes = sync_docs()
        print(f"synced docs for {len({entry['case_dir'] for entry in changes})} cases")
        for entry in changes:
            print(f"{entry['case_dir']}: {entry['status']}")
        return 0
    if args.command == "generate-wrappers":
        generated = generate_wrappers()
        print(f"generated {len(generated)} wrapper files")
        for path in generated:
            print(path)
        return 0

    changes, generated = materialize()
    print(f"doc updates: {len(changes)}")
    print(f"wrapper updates: {len(generated)}")
    for entry in changes:
        print(f"{entry['case_dir']}: {entry['status']}")
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
