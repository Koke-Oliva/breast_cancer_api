from __future__ import annotations

import math
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))

_bundle: dict[str, Any] | None = None


class InputValidationError(ValueError):
    """Raised when a prediction payload violates the public API contract."""


def get_bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
        loaded = joblib.load(MODEL_PATH)
        if not isinstance(loaded, dict) or "model" not in loaded:
            raise RuntimeError(
                "Unsupported model artifact. Re-run src.api.training to generate the model bundle."
            )
        _bundle = loaded
    return _bundle


def reset_model_cache() -> None:
    global _bundle
    _bundle = None


def get_contract() -> dict[str, Any]:
    bundle = get_bundle()
    return {
        "model_version": bundle["model_version"],
        "max_batch_size": MAX_BATCH_SIZE,
        "features": [
            {
                "name": name,
                "type": "number",
                "training_min": bundle["feature_ranges"][name]["min"],
                "training_max": bundle["feature_ranges"][name]["max"],
            }
            for name in bundle["feature_names"]
        ],
        "labels": bundle["label_mapping"],
        "positive_class": "malignant",
        "range_policy": (
            "Inputs outside the min/max observed in training are rejected. "
            "These ranges describe the model training domain, not clinical reference limits."
        ),
    }


def validate_instances(rows: Any) -> list[dict[str, float]]:
    bundle = get_bundle()
    feature_names = list(bundle["feature_names"])
    expected = set(feature_names)
    ranges = bundle["feature_ranges"]

    if not isinstance(rows, list):
        raise InputValidationError("'instances' must be a JSON array.")
    if not rows:
        raise InputValidationError("'instances' must contain at least one item.")
    if len(rows) > MAX_BATCH_SIZE:
        raise InputValidationError(
            f"Batch too large: maximum {MAX_BATCH_SIZE} instances per request."
        )

    validated: list[dict[str, float]] = []

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise InputValidationError(f"instances[{idx}] must be a JSON object.")

        keys = set(row)
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)

        if missing:
            raise InputValidationError(
                f"instances[{idx}] is missing {len(missing)} feature(s): {missing}."
            )
        if extra:
            raise InputValidationError(
                f"instances[{idx}] contains unexpected feature(s): {extra}."
            )

        clean_row: dict[str, float] = {}
        for feature in feature_names:
            value = row[feature]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InputValidationError(
                    f"instances[{idx}].{feature!r} must be numeric."
                )

            number = float(value)
            if not math.isfinite(number):
                raise InputValidationError(
                    f"instances[{idx}].{feature!r} must be finite."
                )

            low = float(ranges[feature]["min"])
            high = float(ranges[feature]["max"])
            if number < low or number > high:
                raise InputValidationError(
                    f"instances[{idx}].{feature!r}={number} is outside the "
                    f"training range [{low}, {high}]."
                )
            clean_row[feature] = number

        validated.append(clean_row)

    return validated


def predict_batch(rows: Any) -> tuple[list[dict[str, Any]], str]:
    bundle = get_bundle()
    clean_rows = validate_instances(rows)
    feature_names = list(bundle["feature_names"])

    X = pd.DataFrame(clean_rows, columns=feature_names)
    model = bundle["model"]

    preds = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    malignant_idx = classes.index(1)
    benign_idx = classes.index(0)

    results = []
    for pred, proba in zip(preds, probabilities):
        pred_int = int(pred)
        results.append({
            "prediction": pred_int,
            "label": bundle["label_mapping"][pred_int],
            "malignant_probability": float(proba[malignant_idx]),
            "probabilities": {
                "benign": float(proba[benign_idx]),
                "malignant": float(proba[malignant_idx]),
            },
        })

    return results, str(bundle["model_version"])
