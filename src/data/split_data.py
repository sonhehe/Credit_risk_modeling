"""Train/test split helpers for time-based modeling."""

from __future__ import annotations

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    date_col: str = "issue_d",
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train and test partitions."""
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe.")

    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].isna().all():
        raise ValueError(
            f"Date column '{date_col}' could not be parsed; all values are invalid."
        )

    out = out.sort_values(date_col, ascending=True)
    split_idx = int(len(out) * train_fraction)

    train_df = out.iloc[:split_idx].reset_index(drop=True)
    test_df = out.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df
