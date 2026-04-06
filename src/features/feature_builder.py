"""End-to-end, leakage-safe feature builder for PD modeling."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.features.encode import OneHotEncoderSimple
from src.features.missing import MissingValueHandler
from src.features.preprocess import FeatureSpec, preprocess_features, select_feature_spec


LEAKAGE_COLUMNS: set[str] = {
    "loan_status",
    "default_flag",
    "charged_off",
    "recoveries",
    "collection_recovery_fee",
    "total_rec_prncp",
    "total_rec_int",
    "last_pymnt_d",
    "last_credit_pull_d",
}


@dataclass
class FeatureBuilder:
    """Train/test-consistent feature pipeline with preserved feature names."""

    feature_spec_: FeatureSpec | None = None
    missing_handler_: MissingValueHandler = field(default_factory=MissingValueHandler)
    encoder_: OneHotEncoderSimple = field(default_factory=OneHotEncoderSimple)
    feature_names_: list[str] = field(default_factory=list)

    def _drop_leakage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        removable = [col for col in LEAKAGE_COLUMNS if col in df.columns]
        return df.drop(columns=removable, errors="ignore")

    def _split_by_dtype(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        numeric = df.select_dtypes(include=["number"]).copy()
        categorical = df.select_dtypes(exclude=["number"]).copy()
        return numeric, categorical

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        safe = self._drop_leakage_columns(df)
        self.feature_spec_ = select_feature_spec(safe)

        base = preprocess_features(safe, self.feature_spec_)
        filled = self.missing_handler_.fit_transform(base)

        num_df, cat_df = self._split_by_dtype(filled)
        encoded_cat = self.encoder_.fit_transform(cat_df)

        model_df = pd.concat([num_df, encoded_cat], axis=1)
        self.feature_names_ = model_df.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_spec_ is None:
            raise ValueError("FeatureBuilder must be fitted before calling transform().")

        safe = self._drop_leakage_columns(df)
        base = preprocess_features(safe, self.feature_spec_)
        filled = self.missing_handler_.transform(base)

        num_df, cat_df = self._split_by_dtype(filled)
        encoded_cat = self.encoder_.transform(cat_df)

        model_df = pd.concat([num_df, encoded_cat], axis=1)
        return model_df.reindex(columns=self.feature_names_, fill_value=0.0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
