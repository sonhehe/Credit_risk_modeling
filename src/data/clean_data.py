"""Data cleaning logic for Lending Club accepted loans."""

from __future__ import annotations

import pandas as pd

from src.data.schema import CATEGORICAL_COLUMNS, DATE_COLUMNS, NUMERIC_COLUMNS


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse configured date columns to pandas datetime."""
    out = df.copy()
    for col in DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace(",", "", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    return numeric


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce core numeric columns and impute missing values with median."""
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = _clean_numeric_series(out[col])
        median_val = out[col].median(skipna=True)
        if pd.notna(median_val):
            out[col] = out[col].fillna(median_val)
    return out


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing categorical values with 'unknown'."""
    out = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = (
            out[col]
            .astype("string")
            .str.strip()
            .fillna("unknown")
            .replace("", "unknown")
        )
    return out


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run end-to-end data cleaning in interpretable steps."""
    cleaned = parse_dates(df)
    cleaned = clean_numeric_columns(cleaned)
    cleaned = clean_categorical_columns(cleaned)
    return cleaned
