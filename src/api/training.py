# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_and_save_model():
    # Cargar dataset Breast Cancer
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Entrenar modelo RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Crear carpeta artifacts si no existe
    os.makedirs("artifacts", exist_ok=True)

    # Guardar modelo en artifacts/model.pkl
    joblib.dump(model, "artifacts/model.pkl")
    print("✅ Modelo entrenado y guardado en artifacts/model.pkl")

if __name__ == "__main__":
    train_and_save_model()
