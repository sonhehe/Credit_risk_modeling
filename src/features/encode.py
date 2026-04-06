"""Categorical encoding utilities for model-ready feature matrices."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class OneHotEncoderSimple:
    """Deterministic one-hot encoding with fixed categories from training data."""

    categories_: dict[str, list[str]] = field(default_factory=dict)
    output_columns_: list[str] = field(default_factory=list)
    unknown_token: str = "__unknown__"

    def fit(self, df: pd.DataFrame) -> "OneHotEncoderSimple":
        self.categories_.clear()
        self.output_columns_.clear()

        for col in df.columns:
            series = df[col].astype("string").fillna(self.unknown_token)
            unique = sorted({str(value) for value in series.tolist()})
            if self.unknown_token not in unique:
                unique.append(self.unknown_token)
            self.categories_[col] = unique
            self.output_columns_.extend([f"{col}__{category}" for category in unique])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded_blocks: list[pd.DataFrame] = []

        for col, categories in self.categories_.items():
            raw = df[col].astype("string").fillna(self.unknown_token)
            raw = raw.where(raw.isin(categories), self.unknown_token)

            dummies = pd.get_dummies(raw, prefix=col, prefix_sep="__", dtype=float)
            expected_columns = [f"{col}__{category}" for category in categories]
            dummies = dummies.reindex(columns=expected_columns, fill_value=0.0)
            encoded_blocks.append(dummies)

        if not encoded_blocks:
            return pd.DataFrame(index=df.index)

        encoded = pd.concat(encoded_blocks, axis=1)
        return encoded.reindex(columns=self.output_columns_, fill_value=0.0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
