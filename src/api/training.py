from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_summary(cv_result: dict, key: str) -> dict[str, float]:
    values = np.asarray(cv_result[f"test_{key}"], dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}


def train_and_save_model(
    artifact_path: str = "artifacts/model.pkl",
    model_card_path: str = "artifacts/model_card.json",
    calibration_plot_path: str = "artifacts/calibration_curve.png",
    model_version: str | None = None,
) -> dict:
    model_version = model_version or os.getenv("MODEL_VERSION", "1.0.0")

    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()

    # scikit-learn codifica 0=malignant, 1=benign.
    # Para la API se define explícitamente la clase positiva como malignant=1.
    y = (data.target == 0).astype(int)
    y.name = "malignant"

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 8],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        n_jobs=-1,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "brier": float(brier_score_loss(y_test, y_prob)),
    }

    feature_ranges = {
        feature: {
            "min": float(X_train[feature].min()),
            "max": float(X_train[feature].max()),
        }
        for feature in X.columns
    }

    bundle = {
        "model": model,
        "model_version": model_version,
        "feature_names": list(X.columns),
        "feature_ranges": feature_ranges,
        "label_mapping": {0: "benign", 1: "malignant"},
        "positive_class": 1,
    }

    artifact = Path(artifact_path)
    card_path = Path(model_card_path)
    calibration_path = Path(calibration_plot_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, artifact)

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=6, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Random Forest")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Calibración perfecta")
    ax.set_xlabel("Probabilidad predicha de malignidad")
    ax.set_ylabel("Fracción observada de malignidad")
    ax.set_title("Curva de calibración - test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(calibration_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    model_card = {
        "model_name": "Breast Cancer Random Forest API",
        "model_version": model_version,
        "artifact": artifact.as_posix(),
        "artifact_sha256": _sha256(artifact),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.getenv("GITHUB_SHA"),
        "dataset": {
            "name": "Breast Cancer Wisconsin Diagnostic",
            "source": "sklearn.datasets.load_breast_cancer",
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "class_definition": {
                "0": "benign",
                "1": "malignant",
            },
            "positive_class": "malignant",
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "model": {
            "type": "RandomForestClassifier",
            "best_params": grid.best_params_,
            "selection_metric": "roc_auc",
            "best_cv_roc_auc": float(grid.best_score_),
        },
        "cross_validation": {
            "strategy": "StratifiedKFold",
            "folds": 5,
            "metrics": {
                name: _metric_summary(cv_scores, name)
                for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]
            },
        },
        "test_metrics": test_metrics,
        "calibration": {
            "metric": "brier_score_loss",
            "brier": test_metrics["brier"],
            "reliability_plot": calibration_path.as_posix(),
        },
        "features": [
            {
                "name": feature,
                "type": "number",
                "training_min": feature_ranges[feature]["min"],
                "training_max": feature_ranges[feature]["max"],
            }
            for feature in X.columns
        ],
        "limitations": [
            "Evaluación interna sobre un único split estratificado; no sustituye validación externa.",
            "El dataset es pequeño y proviene de mediciones diagnósticas estructuradas, no de población clínica general.",
            "Las probabilidades se evalúan con Brier y curva de calibración, pero el modelo no se recalibra.",
            "Los rangos de entrada corresponden al dominio observado en entrenamiento y no constituyen límites clínicos.",
            "El servicio es demostrativo y no debe utilizarse para diagnóstico ni decisión clínica.",
        ],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
        },
    }

    with card_path.open("w", encoding="utf-8") as fh:
        json.dump(model_card, fh, ensure_ascii=False, indent=2)

    print(json.dumps({
        "model_version": model_version,
        "best_params": grid.best_params_,
        "test_metrics": test_metrics,
        "artifact": artifact.as_posix(),
        "model_card": card_path.as_posix(),
    }, indent=2))

    return model_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and serialize the Breast Cancer model.")
    parser.add_argument("--artifact", default="artifacts/model.pkl")
    parser.add_argument("--model-card", default="artifacts/model_card.json")
    parser.add_argument("--calibration-plot", default="artifacts/calibration_curve.png")
    parser.add_argument("--model-version", default=os.getenv("MODEL_VERSION", "1.0.0"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save_model(
        artifact_path=args.artifact,
        model_card_path=args.model_card,
        calibration_plot_path=args.calibration_plot,
        model_version=args.model_version,
    )
