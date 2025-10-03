import os
import joblib
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")
_model = None
_feature_order = None  # cache de features


def get_model():
    global _model, _feature_order
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"⚠️ No se encontró el modelo en {MODEL_PATH}. Ejecuta training.py primero."
            )
        _model = joblib.load(MODEL_PATH)

        # si el modelo guarda los nombres de features (scikit-learn >=1.0)
        if hasattr(_model, "feature_names_in_"):
            _feature_order = list(_model.feature_names_in_)
        else:
            # fallback manual: orden oficial de las 30 features del dataset
            _feature_order = [
                "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
                "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
                "radius error", "texture error", "perimeter error", "area error", "smoothness error",
                "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
                "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
                "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
            ]
    return _model


def predict_batch(rows):
    model = get_model()
    global _feature_order

    if _feature_order is None:
        raise RuntimeError("No se pudo determinar el orden de las features.")

    # Convertir JSON a matriz numpy en el orden correcto
    X = np.array([[float(r[k]) for k in _feature_order] for r in rows], dtype=float)

    preds = model.predict(X).tolist()
    proba = (
        model.predict_proba(X)[:, -1].tolist()
        if hasattr(model, "predict_proba")
        else [None] * len(preds)
    )

    return [
        {"prediction": int(p), "proba": float(proba[i])}
        for i, p in enumerate(preds)
    ]
