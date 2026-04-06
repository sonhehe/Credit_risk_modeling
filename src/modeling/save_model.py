"""Model persistence helpers for PD model artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_pd_model(model: Any, feature_columns: list[str], output_path: str) -> Path:
    """Save model and feature order to a pickle artifact."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "feature_columns": list(feature_columns),
    }
    with path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)

    return path
