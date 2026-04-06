"""Weight of Evidence (WOE) and Information Value (IV) utilities."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_SMOOTHING = 0.5
DEFAULT_UNKNOWN_BIN = "Unknown"



def compute_woe_iv_table(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    *,
    event: int = 1,
    smoothing: float = DEFAULT_SMOOTHING,
) -> pd.DataFrame:
    """Compute per-bin WOE and IV contributions for one binned feature."""
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")
    if feature_col not in df.columns or target_col not in df.columns:
        raise KeyError("feature_col and target_col must exist in df")

    tmp = df[[feature_col, target_col]].copy()
    tmp[feature_col] = tmp[feature_col].astype("string").fillna(DEFAULT_UNKNOWN_BIN)
    tmp["is_event"] = (tmp[target_col] == event).astype(int)

    grouped = tmp.groupby(feature_col, dropna=False, as_index=False).agg(
        total=(target_col, "size"),
        events=("is_event", "sum"),
    )
    grouped["non_events"] = grouped["total"] - grouped["events"]

    n_bins = len(grouped)
    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()

    grouped["dist_events"] = (grouped["events"] + smoothing) / (
        total_events + smoothing * n_bins
    )
    grouped["dist_non_events"] = (grouped["non_events"] + smoothing) / (
        total_non_events + smoothing * n_bins
    )

    grouped["woe"] = np.log(grouped["dist_non_events"] / grouped["dist_events"])
    grouped["iv_component"] = (
        grouped["dist_non_events"] - grouped["dist_events"]
    ) * grouped["woe"]
    grouped["iv_total"] = grouped["iv_component"].sum()

    grouped = grouped.rename(columns={feature_col: "bin_label"})
    grouped.insert(0, "feature", feature_col)
    return grouped.sort_values("bin_label").reset_index(drop=True)



def woe_mapping_from_table(woe_table: pd.DataFrame) -> dict[str, float]:
    """Create a bin_label -> woe mapping dictionary from a WOE table."""
    required = {"bin_label", "woe"}
    missing = required - set(woe_table.columns)
    if missing:
        raise ValueError(f"woe_table missing columns: {sorted(missing)}")

    mapping = (
        woe_table[["bin_label", "woe"]]
        .drop_duplicates(subset=["bin_label"])
        .set_index("bin_label")["woe"]
        .to_dict()
    )
    return {str(k): float(v) for k, v in mapping.items()}



def apply_woe_mapping(
    binned_series: pd.Series,
    mapping: dict[str, float],
    *,
    default_woe: float = 0.0,
) -> pd.Series:
    """Map bin labels to numeric WOE values."""
    labels = binned_series.astype("string").fillna(DEFAULT_UNKNOWN_BIN)
    out = labels.map(mapping).fillna(default_woe)
    return pd.to_numeric(out, errors="coerce").fillna(default_woe)



def fit_woe_tables(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    *,
    event: int = 1,
    smoothing: float = DEFAULT_SMOOTHING,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fit WOE/IV tables for multiple binned features."""
    tables: dict[str, pd.DataFrame] = {}
    iv_rows: list[dict[str, float]] = []

    for feature in feature_cols:
        if feature not in df.columns:
            continue
        table = compute_woe_iv_table(
            df,
            feature_col=feature,
            target_col=target_col,
            event=event,
            smoothing=smoothing,
        )
        tables[feature] = table
        iv_rows.append({"feature": feature, "iv": float(table["iv_total"].iloc[0])})

    iv_summary = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(
        drop=True
    )
    return tables, iv_summary



def transform_to_woe(
    df: pd.DataFrame,
    woe_tables: dict[str, pd.DataFrame],
    *,
    suffix: str = "_woe",
    default_woe: float = 0.0,
) -> pd.DataFrame:
    """Apply fitted WOE tables to a dataframe of binned features."""
    out = df.copy()
    for feature, table in woe_tables.items():
        if feature not in out.columns:
            continue
        mapping = woe_mapping_from_table(table)
        out[f"{feature}{suffix}"] = apply_woe_mapping(
            out[feature],
            mapping,
            default_woe=default_woe,
        )
    return out



def export_scorecard_woe_table(woe_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Export WOE tables in one scorecard-friendly long format."""
    if not woe_tables:
        return pd.DataFrame(
            columns=[
                "feature",
                "bin_label",
                "total",
                "events",
                "non_events",
                "dist_events",
                "dist_non_events",
                "woe",
                "iv_component",
                "iv_total",
            ]
        )

    parts = []
    for feature, table in woe_tables.items():
        part = table.copy()
        part["feature"] = feature
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["feature", "bin_label"]).reset_index(drop=True)
    return out
