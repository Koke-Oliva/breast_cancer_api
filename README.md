# 🧠 Breast Cancer API – MLOps en la Nube

Este proyecto implementa un **flujo completo de MLOps** que integra un modelo de Machine Learning (clasificación de cáncer de mama con Random Forest), lo expone como **API REST con Flask**, y automatiza pruebas y despliegues usando **Docker y GitHub Actions**.  
El ciclo de vida incluye: **entrenamiento del modelo, empaquetado, pruebas, CI/CD y despliegue en GHCR**.

---

## 🚀 Ciclo de vida del proyecto

### 1️⃣ Entrenamiento del modelo
- Dataset: **Breast Cancer Wisconsin** de `sklearn.datasets.load_breast_cancer`.
- Modelo: `RandomForestClassifier`.
- El modelo entrenado se guarda en `artifacts/model.pkl`.

### 2️⃣ API con Flask
- Ruta de verificación: `GET /api/health`  
- Ruta de predicción: `POST /api/predict` con entrada JSON.  
- Respuesta: JSON con predicción (`0` o `1`) y probabilidad.

### 3️⃣ Contenerización con Docker
- Dockerfile define la instalación de dependencias y la ejecución de la API.
- La API expone el puerto **8080**.
- Variables de entorno permiten configurar la ruta del modelo (`MODEL_PATH`).

📷 **Evidencia: Docker instalado**  
![Docker instalado](./img/Docker%20instalado.png)

---

### 4️⃣ Pruebas de la API
Se realizaron pruebas locales y en Postman:

- **GET /api/health** → responde `{"status": "ok"}`  
📷 ![Evidencia GET](./img/Evidencia%20-%20GET.png)

- **POST /api/predict** → recibe JSON con 30 features y devuelve predicción con probabilidad  
📷 ![Evidencia POST](./img/Evidencia%20-%20POST.png)

---

### 5️⃣ CI/CD con GitHub Actions
Se configuró un workflow (`.github/workflows/ci.yml`) con dos etapas:

1. **test** → ejecuta pruebas con `pytest`.
2. **build-and-push** → si los tests pasan, construye la imagen Docker y la publica en GHCR.

📷 **Evidencia CI/CD**  
![Evidencia CI/CD](./img/Evidencia%20CI%20CD.png)

---

### 6️⃣ Publicación en GHCR
La imagen Docker se publica automáticamente en **GitHub Container Registry (GHCR)**:  
`ghcr.io/koke-oliva/breast_cancer_api:latest`

📷 **Evidencia: Imagen publicada en GHCR**  
![Evidencia Imagen GHCR](./img/Evidencia%20-%20Imagen%20Docker%20en%20el%20GHCR.png)

---

## 📂 Estructura del Proyecto

```bash
breast_cancer_api/
├── artifacts/              # Modelos entrenados (model.pkl)
├── img/                    # Evidencias en imágenes para README
├── src/
│   ├── api/                # Código API
│   │   ├── training.py     # Entrenamiento del modelo
│   │   ├── routes.py       # Endpoints API
│   │   └── model.py        # Carga del modelo y predicciones
│   ├── tests/              # Tests unitarios
│   └── main.py             # Punto de entrada Flask
├── .github/
│   └── workflows/
│       └── ci.yml          # Workflow CI/CD
├── requirements.txt        # Dependencias
├── Dockerfile              # Imagen Docker
└── README.md               # Documentación

```bash

## ✅ Conclusiones
Este proyecto demuestra un flujo **end-to-end en MLOps**:
- Entrenamiento y versionado de modelos.
- API accesible por REST.
- Contenerización reproducible con Docker.
- Pruebas automatizadas.
- Despliegue continuo con GitHub Actions y publicación en GHCR.


