"""Runnable training pipeline for baseline PD logistic regression model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.modeling.predict_pd import predict_pd_probability
from src.modeling.save_model import save_pd_model
from src.modeling.train_pd import infer_target_column, train_pd_model


DEFAULT_TRAIN_PATH = "data/processed/train_dataset.parquet"
DEFAULT_TEST_PATH = "data/processed/test_dataset.parquet"
DEFAULT_MODEL_PATH = "artifacts/models/pd_model.pkl"
DEFAULT_SCORED_TEST_PATH = "data/processed/scored_loans.parquet"


def _read_table(path: str) -> pd.DataFrame:
    """Read parquet/csv input based on extension."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    return pd.read_parquet(file_path)


def _save_with_fallback(df: pd.DataFrame, output_path: str) -> str:
    """Save parquet and fallback to CSV when parquet engine is unavailable."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(path, index=False)
        return str(path)
    except Exception:
        fallback_path = path.with_suffix(".csv")
        df.to_csv(fallback_path, index=False)
        return str(fallback_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PD logistic regression training pipeline.")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH, help="Path to processed train dataset")
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH, help="Path to processed test dataset")
    parser.add_argument("--target-col", default=None, help="Target column name (auto-detected if omitted)")
    parser.add_argument("--model-output", default=DEFAULT_MODEL_PATH, help="Path to save model artifact")
    parser.add_argument(
        "--scored-test-output",
        default=DEFAULT_SCORED_TEST_PATH,
        help="Path to save scored test dataset",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    print("[1/6] Loading processed train/test data...")
    train_df = _read_table(args.train_path)
    test_df = _read_table(args.test_path)

    target_col = args.target_col or infer_target_column(train_df)
    print(f"[2/6] Training logistic regression model using target='{target_col}'...")
    training_artifacts = train_pd_model(train_df=train_df, target_col=target_col)

    print("[3/6] Scoring train dataset...")
    scored_train_df = predict_pd_probability(
        model=training_artifacts.model,
        df=train_df,
        feature_columns=training_artifacts.feature_columns,
        output_col="pd_proba",
    )

    print("[4/6] Scoring test dataset...")
    scored_test_df = predict_pd_probability(
        model=training_artifacts.model,
        df=test_df,
        feature_columns=training_artifacts.feature_columns,
        output_col="pd_proba",
    )

    print("[5/6] Saving model artifact...")
    model_path = save_pd_model(
        model=training_artifacts.model,
        feature_columns=training_artifacts.feature_columns,
        output_path=args.model_output,
    )

    print("[6/6] Saving scored outputs...")
    scored_test_path = _save_with_fallback(scored_test_df, args.scored_test_output)

    print("PD training flow complete.")
    print(f"Model artifact: {model_path}")
    print(f"Scored test data: {scored_test_path}")
    print("Sample train probabilities:")
    print(scored_train_df[[target_col, "pd_proba"]].head(5).to_string(index=False))
    print("Sample test probabilities:")
    print(scored_test_df[[target_col, "pd_proba"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
