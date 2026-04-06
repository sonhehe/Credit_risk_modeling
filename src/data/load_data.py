"""Data loading utilities for Lending Club accepted loans."""

from __future__ import annotations

import pandas as pd

from src.data.schema import REQUIRED_COLUMNS


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase and stripped formatting."""
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def get_load_columns() -> list[str]:
    """Return the minimal set of columns to load from raw file."""
    return REQUIRED_COLUMNS.copy()


def _resolve_usecols(path: str, load_columns: list[str]) -> list[str] | None:
    """Resolve case-insensitive usecols against the CSV header.

    Returns None when exact projection is not possible.
    """
    header = pd.read_csv(path, nrows=0)
    standardized_to_original = {
        str(col).strip().lower(): str(col) for col in header.columns
    }

    resolved: list[str] = []
    missing: list[str] = []
    for col in load_columns:
        original = standardized_to_original.get(col)
        if original is None:
            missing.append(col)
        else:
            resolved.append(original)

    if missing:
        return None
    return resolved


def load_raw_data(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load Lending Club CSV efficiently and standardize columns."""
    load_columns = get_load_columns()
    resolved_usecols = _resolve_usecols(path, load_columns)

    df = pd.read_csv(
        path,
        usecols=resolved_usecols,
        low_memory=False,
        nrows=nrows,
    )
    return standardize_column_names(df)
