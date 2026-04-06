"""Target labeling utilities for default prediction."""

from __future__ import annotations

import pandas as pd

DEFAULT_STATUSES: set[str] = {
    "charged off",
    "default",
    "late (31-120 days)",
    "late (16-30 days)",
    "does not meet the credit policy. status:charged off",
}

NON_DEFAULT_STATUSES: set[str] = {
    "fully paid",
    "does not meet the credit policy. status:fully paid",
}

AMBIGUOUS_STATUSES: set[str] = {
    "current",
    "issued",
    "in grace period",
    "policy status:current",
}


def _normalize_status(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def create_default_target(
    df: pd.DataFrame,
    target_col: str = "default_target",
) -> pd.DataFrame:
    """Create binary default target and drop ambiguous statuses."""
    if "loan_status" not in df.columns:
        raise ValueError("Missing required column 'loan_status' for target labeling.")

    out = df.copy()
    status_norm = _normalize_status(out["loan_status"])

    keep_mask = ~status_norm.isin(AMBIGUOUS_STATUSES)
    out = out.loc[keep_mask].copy()
    status_norm = status_norm.loc[keep_mask]

    status_to_target = {s: 1 for s in DEFAULT_STATUSES}
    status_to_target.update({s: 0 for s in NON_DEFAULT_STATUSES})

    out[target_col] = status_norm.map(status_to_target)
    out = out.loc[out[target_col].isin([0, 1])].copy()
    out[target_col] = out[target_col].astype("int8")
    return out
