"""Data schema definitions and validation utilities for Lending Club loans."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS: list[str] = [
    "loan_status",
    "issue_d",
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "purpose",
    "dti",
    "delinq_2yrs",
    "revol_util",
    "fico_range_low",
    "fico_range_high",
]

DATE_COLUMNS: list[str] = ["issue_d"]

NUMERIC_COLUMNS: list[str] = [
    "loan_amnt",
    "funded_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "revol_util",
    "fico_range_low",
    "fico_range_high",
]

CATEGORICAL_COLUMNS: list[str] = [
    "loan_status",
    "term",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "purpose",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that all required columns are present in the DataFrame.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )
