"""Simple and explainable missing-value handling."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MissingValueHandler:
    """Median for numeric and mode/'unknown' for categorical columns."""

    numeric_fill_values: dict[str, float] = field(default_factory=dict)
    categorical_fill_values: dict[str, str] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                median = series.median(skipna=True)
                self.numeric_fill_values[col] = float(median) if pd.notna(median) else 0.0
            else:
                non_missing = series.dropna()
                if non_missing.empty:
                    self.categorical_fill_values[col] = "unknown"
                else:
                    mode = non_missing.mode(dropna=True)
                    self.categorical_fill_values[col] = str(mode.iloc[0]) if not mode.empty else "unknown"
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, fill_value in self.numeric_fill_values.items():
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fill_value)

        for col, fill_value in self.categorical_fill_values.items():
            if col in out.columns:
                out[col] = (
                    out[col]
                    .astype("string")
                    .fillna(fill_value)
                    .replace("", fill_value)
                )

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
