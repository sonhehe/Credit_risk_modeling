"""Preprocessing helpers for retail credit-risk feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


NUMERIC_CANDIDATES: tuple[str, ...] = (
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "revol_util",
    "delinq_2yrs",
    "loan_amnt",
)

CATEGORICAL_CANDIDATES: tuple[str, ...] = (
    "emp_length",
    "home_ownership",
    "term",
    "purpose",
    "grade",
    "sub_grade",
)


@dataclass(frozen=True)
class FeatureSpec:
    """Selected feature names after checking data availability."""

    numeric_features: list[str]
    categorical_features: list[str]


def _clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_term_months(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _parse_emp_length_years(series: pd.Series) -> pd.Series:
    txt = series.astype("string").str.lower().str.strip()
    txt = txt.replace({"< 1 year": "0", "10+ years": "10", "n/a": pd.NA})
    extracted = txt.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def select_feature_spec(df: pd.DataFrame) -> FeatureSpec:
    """Select available retail credit-risk features from preferred candidates."""
    numeric_features = [col for col in NUMERIC_CANDIDATES if col in df.columns]
    categorical_features = [col for col in CATEGORICAL_CANDIDATES if col in df.columns]

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        numeric_features = [col for col in numeric_features if col not in {"fico_range_low", "fico_range_high"}]
        numeric_features.append("fico_avg")

    return FeatureSpec(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def preprocess_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Build cleaned base features before missing-value handling and encoding."""
    out = pd.DataFrame(index=df.index)

    for col in spec.numeric_features:
        if col == "fico_avg":
            fico_low = _clean_numeric(df["fico_range_low"])
            fico_high = _clean_numeric(df["fico_range_high"])
            out[col] = (fico_low + fico_high) / 2.0
            continue
        out[col] = _clean_numeric(df[col])

    for col in spec.categorical_features:
        raw = df[col].astype("string").str.strip()
        if col == "term":
            out["term_months"] = _parse_term_months(raw)
            continue
        if col == "emp_length":
            out["emp_length_years"] = _parse_emp_length_years(raw)
            continue
        out[col] = raw.str.lower()

    return out
