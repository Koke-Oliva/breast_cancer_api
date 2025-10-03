# tests/test_model.py
import os
import pytest
import joblib
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")

@pytest.mark.parametrize("features", [
    # Un ejemplo de instancia con 30 valores (dataset Breast Cancer)
    np.random.rand(30).tolist()
])
def test_model_prediction(features):
    assert os.path.exists(MODEL_PATH), f"Modelo no encontrado en {MODEL_PATH}"

    # Cargar modelo
    model = joblib.load(MODEL_PATH)

    # Convertir a numpy array y hacer predicción
    X = np.array([features], dtype=float)
    y_pred = model.predict(X)

    # Verificamos que la predicción sea 0 o 1
    assert y_pred[0] in [0, 1]
