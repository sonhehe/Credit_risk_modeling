"""Persist and restore feature-pipeline metadata for reproducible inference."""

from __future__ import annotations

import json
from pathlib import Path

from src.features.feature_builder import FeatureBuilder
from src.features.preprocess import FeatureSpec


def save_feature_builder(builder: FeatureBuilder, path: str | Path) -> None:
    """Serialize fitted FeatureBuilder metadata to JSON."""
    if builder.feature_spec_ is None:
        raise ValueError("Cannot save an unfitted FeatureBuilder.")

    payload = {
        "feature_spec": {
            "numeric_features": builder.feature_spec_.numeric_features,
            "categorical_features": builder.feature_spec_.categorical_features,
        },
        "numeric_fill_values": builder.missing_handler_.numeric_fill_values,
        "categorical_fill_values": builder.missing_handler_.categorical_fill_values,
        "categories": builder.encoder_.categories_,
        "output_columns": builder.encoder_.output_columns_,
        "unknown_token": builder.encoder_.unknown_token,
        "feature_names": builder.feature_names_,
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_feature_builder(path: str | Path) -> FeatureBuilder:
    """Load FeatureBuilder metadata from JSON."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))

    builder = FeatureBuilder()
    builder.feature_spec_ = FeatureSpec(
        numeric_features=payload["feature_spec"]["numeric_features"],
        categorical_features=payload["feature_spec"]["categorical_features"],
    )
    builder.missing_handler_.numeric_fill_values = {
        str(k): float(v) for k, v in payload["numeric_fill_values"].items()
    }
    builder.missing_handler_.categorical_fill_values = {
        str(k): str(v) for k, v in payload["categorical_fill_values"].items()
    }
    builder.encoder_.categories_ = {
        str(k): [str(item) for item in values]
        for k, values in payload["categories"].items()
    }
    builder.encoder_.output_columns_ = [str(col) for col in payload["output_columns"]]
    builder.encoder_.unknown_token = str(payload["unknown_token"])
    builder.feature_names_ = [str(col) for col in payload["feature_names"]]

    return builder
