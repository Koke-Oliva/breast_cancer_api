# Model artifacts

Esta carpeta contiene los artefactos reproducibles generados por src.api.training:

- model.pkl: bundle del modelo usado en inferencia.
- model_card.json: procedencia, versión, contrato de features, métricas, calibración y limitaciones.
- calibration_curve.png: reliability diagram del conjunto de test.

La trazabilidad no depende solo del nombre del archivo. model_card.json registra model_version, commit de entrenamiento, SHA-256 del artefacto, versiones del entorno, split/semilla, métricas y rangos de features.

CI/CD también etiqueta la imagen Docker con la versión del modelo y el SHA del commit.
