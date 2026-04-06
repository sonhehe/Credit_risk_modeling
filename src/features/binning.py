"""Utilities for scorecard-oriented binning of numerical variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_MISSING_LABEL = "Missing"


@dataclass(frozen=True)
class BinDefinition:
    """Definition of a numerical binning strategy for one feature."""

    method: str
    edges: tuple[float, ...]
    labels: tuple[str, ...]
    right: bool = True
    include_lowest: bool = True
    missing_label: str = DEFAULT_MISSING_LABEL



def _interval_labels_from_edges(edges: Sequence[float], precision: int = 6) -> list[str]:
    labels: list[str] = []
    for i in range(len(edges) - 1):
        left = round(float(edges[i]), precision)
        right = round(float(edges[i + 1]), precision)
        labels.append(f"[{left}, {right}]")
    return labels



def _clean_edges(edges: Sequence[float]) -> list[float]:
    unique = sorted({float(x) for x in edges if pd.notna(x)})
    if len(unique) < 2:
        raise ValueError("At least two unique numeric edges are required for binning.")
    return unique



def make_quantile_bin_definition(
    series: pd.Series,
    n_bins: int = 5,
    *,
    precision: int = 6,
    missing_label: str = DEFAULT_MISSING_LABEL,
) -> BinDefinition:
    """Create quantile-based bins for a numerical series."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError("Cannot build quantile bins from an all-missing series.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = numeric.quantile(quantiles).tolist()
    clean_edges = _clean_edges(edges)

    if len(clean_edges) < 2:
        min_val = float(numeric.min())
        max_val = float(numeric.max())
        if min_val == max_val:
            clean_edges = [min_val - 1e-9, max_val + 1e-9]
        else:
            clean_edges = [min_val, max_val]

    labels = _interval_labels_from_edges(clean_edges, precision=precision)
    return BinDefinition(
        method="quantile",
        edges=tuple(clean_edges),
        labels=tuple(labels),
        missing_label=missing_label,
    )



def make_rule_bin_definition(
    edges: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    right: bool = True,
    include_lowest: bool = True,
    precision: int = 6,
    missing_label: str = DEFAULT_MISSING_LABEL,
) -> BinDefinition:
    """Create rule-based bins from explicit edge definitions."""
    clean_edges = _clean_edges(edges)

    if labels is None:
        resolved_labels = _interval_labels_from_edges(clean_edges, precision=precision)
    else:
        resolved_labels = list(labels)
        if len(resolved_labels) != len(clean_edges) - 1:
            raise ValueError("labels length must equal len(edges) - 1")

    return BinDefinition(
        method="rule",
        edges=tuple(clean_edges),
        labels=tuple(str(x) for x in resolved_labels),
        right=right,
        include_lowest=include_lowest,
        missing_label=missing_label,
    )



def apply_bin_definition(series: pd.Series, bin_def: BinDefinition) -> pd.Series:
    """Apply a bin definition and return a string-labeled binned series."""
    numeric = pd.to_numeric(series, errors="coerce")
    binned = pd.cut(
        numeric,
        bins=list(bin_def.edges),
        labels=list(bin_def.labels),
        right=bin_def.right,
        include_lowest=bin_def.include_lowest,
    )
    out = binned.astype("string").fillna(bin_def.missing_label)
    return out



def fit_numerical_binning(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    method: str = "quantile",
    n_bins: int = 5,
    rule_edges: dict[str, Sequence[float]] | None = None,
    rule_labels: dict[str, Sequence[str]] | None = None,
    missing_label: str = DEFAULT_MISSING_LABEL,
) -> dict[str, BinDefinition]:
    """Fit bin definitions for a set of numerical columns."""
    result: dict[str, BinDefinition] = {}
    rule_edges = rule_edges or {}
    rule_labels = rule_labels or {}

    for col in columns:
        if col not in df.columns:
            continue

        if method == "quantile":
            result[col] = make_quantile_bin_definition(
                df[col],
                n_bins=n_bins,
                missing_label=missing_label,
            )
        elif method == "rule":
            if col not in rule_edges:
                raise ValueError(f"Missing rule edges for column '{col}'.")
            result[col] = make_rule_bin_definition(
                rule_edges[col],
                labels=rule_labels.get(col),
                missing_label=missing_label,
            )
        else:
            raise ValueError("method must be either 'quantile' or 'rule'")

    return result



def transform_with_binning(
    df: pd.DataFrame,
    binning_map: dict[str, BinDefinition],
    *,
    suffix: str = "_bin",
) -> pd.DataFrame:
    """Transform numerical columns into scorecard-friendly bin labels."""
    out = df.copy()
    for col, bin_def in binning_map.items():
        if col in out.columns:
            out[f"{col}{suffix}"] = apply_bin_definition(out[col], bin_def)
    return out



def export_binning_table(feature: str, bin_def: BinDefinition) -> pd.DataFrame:
    """Export an interpretable table describing one fitted binning definition."""
    rows = []
    for idx, label in enumerate(bin_def.labels):
        rows.append(
            {
                "feature": feature,
                "bin_index": idx,
                "bin_label": label,
                "left_edge": bin_def.edges[idx],
                "right_edge": bin_def.edges[idx + 1],
                "is_missing_bin": False,
            }
        )

    rows.append(
        {
            "feature": feature,
            "bin_index": len(bin_def.labels),
            "bin_label": bin_def.missing_label,
            "left_edge": np.nan,
            "right_edge": np.nan,
            "is_missing_bin": True,
        }
    )
    return pd.DataFrame(rows)
