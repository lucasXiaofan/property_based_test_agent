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
