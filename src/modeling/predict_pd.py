"""PD model inference helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression


def predict_pd_probability(
    model: LogisticRegression,
    df: pd.DataFrame,
    feature_columns: list[str],
    output_col: str = "pd_proba",
) -> pd.DataFrame:
    """Score input dataframe with PD probabilities using fixed feature order."""
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features for inference: {missing_features}")

    X = df[feature_columns].copy()
    scored_df = df.copy()
    scored_df[output_col] = model.predict_proba(X)[:, 1]
    return scored_df
