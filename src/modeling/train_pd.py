"""PD model training utilities using logistic regression baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.linear_model import LogisticRegression


DEFAULT_TARGET_CANDIDATES: tuple[str, ...] = ("default", "target_default", "is_default", "y")


@dataclass(frozen=True)
class PDTrainingArtifacts:
    """Container for fitted PD model and the associated feature order."""

    model: LogisticRegression
    feature_columns: list[str]


def infer_target_column(df: pd.DataFrame, candidates: Iterable[str] = DEFAULT_TARGET_CANDIDATES) -> str:
    """Infer the target column from a list of common names."""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Unable to infer target column. Expected one of: "
        f"{', '.join(candidates)}. Available columns: {', '.join(df.columns)}"
    )


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Create X/y matrices and enforce stable feature ordering."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    if feature_columns is None:
        candidate_cols = [c for c in df.columns if c != target_col]
    else:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        candidate_cols = list(feature_columns)

    if not candidate_cols:
        raise ValueError("No feature columns provided for model training.")

    X = df[candidate_cols].copy()
    y = df[target_col].astype(int)
    return X, y, candidate_cols


def train_pd_model(
    train_df: pd.DataFrame,
    target_col: str | None = None,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
    max_iter: int = 1000,
) -> PDTrainingArtifacts:
    """Train logistic regression PD model."""
    resolved_target = target_col or infer_target_column(train_df)
    X_train, y_train, feature_order = build_feature_matrix(
        train_df,
        target_col=resolved_target,
        feature_columns=feature_columns,
    )

    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    return PDTrainingArtifacts(model=model, feature_columns=feature_order)
