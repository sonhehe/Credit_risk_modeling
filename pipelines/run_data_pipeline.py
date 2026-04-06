"""Runnable data pipeline for Lending Club credit risk modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.clean_data import clean_data
from src.data.label_target import create_default_target
from src.data.load_data import load_raw_data
from src.data.schema import validate_required_columns
from src.data.split_data import time_based_split


DEFAULT_INPUT = "accepted_2007_to_2018Q4.csv"
DEFAULT_INTERIM_OUTPUT = "data/interim/cleaned_loans.parquet"
DEFAULT_TRAIN_OUTPUT = "data/processed/train_dataset.parquet"
DEFAULT_TEST_OUTPUT = "data/processed/test_dataset.parquet"


def _save_with_fallback(df: pd.DataFrame, output_path: str) -> str:
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
    parser = argparse.ArgumentParser(description="Run Lending Club data pipeline.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to raw accepted loans CSV")
    parser.add_argument(
        "--interim-output",
        default=DEFAULT_INTERIM_OUTPUT,
        help="Output path for cleaned dataset",
    )
    parser.add_argument(
        "--train-output",
        default=DEFAULT_TRAIN_OUTPUT,
        help="Output path for train dataset",
    )
    parser.add_argument(
        "--test-output",
        default=DEFAULT_TEST_OUTPUT,
        help="Output path for test dataset",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit for faster iteration",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Chronological train split fraction",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    print("[1/7] Loading raw data...")
    df = load_raw_data(path=args.input, nrows=args.nrows)
    print(f"Loaded {len(df):,} rows.")

    print("[2/7] Validating required schema...")
    validate_required_columns(df)

    print("[3/7] Cleaning data...")
    cleaned_df = clean_data(df)

    print("[4/7] Creating default target...")
    labeled_df = create_default_target(cleaned_df)

    print("[5/7] Performing time-based split...")
    train_df, test_df = time_based_split(
        labeled_df,
        date_col="issue_d",
        train_fraction=args.train_fraction,
    )

    print("[6/7] Saving interim cleaned data...")
    interim_path = _save_with_fallback(labeled_df, args.interim_output)
    print(f"Interim dataset saved: {interim_path}")

    print("[7/7] Saving train/test data...")
    train_path = _save_with_fallback(train_df, args.train_output)
    test_path = _save_with_fallback(test_df, args.test_output)
    print(f"Train dataset saved: {train_path}")
    print(f"Test dataset saved: {test_path}")


if __name__ == "__main__":
    main()
