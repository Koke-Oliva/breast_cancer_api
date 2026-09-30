import copy
import math


def test_root_describes_service(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_json()
    assert body["service"] == "breast-cancer-api"
    assert body["predict"] == "/api/predict"


def test_health_loads_model(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_schema_documents_30_numeric_features(client):
    response = client.get("/api/schema")
    assert response.status_code == 200
    body = response.get_json()
    assert len(body["features"]) == 30
    assert all(feature["type"] == "number" for feature in body["features"])
    assert body["positive_class"] == "malignant"


def test_predict_valid_payload(client, valid_payload):
    response = client.post("/api/predict", json=valid_payload)
    assert response.status_code == 200

    body = response.get_json()
    assert body["model_version"]
    assert len(body["results"]) == 1

    result = body["results"][0]
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("benign", "malignant")
    assert 0.0 <= result["malignant_probability"] <= 1.0
    assert math.isclose(
        result["probabilities"]["benign"] + result["probabilities"]["malignant"],
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def test_predict_supports_batch(client, valid_payload):
    payload = {"instances": valid_payload["instances"] * 2}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    assert len(response.get_json()["results"]) == 2


def test_predict_requires_json(client):
    response = client.post("/api/predict", data="not-json")
    assert response.status_code == 415


def test_predict_requires_instances(client):
    response = client.post("/api/predict", json={})
    assert response.status_code == 422
    assert response.get_json()["error"] == "validation_error"


def test_predict_rejects_empty_instances(client):
    response = client.post("/api/predict", json={"instances": []})
    assert response.status_code == 422


def test_predict_rejects_missing_feature(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["instances"][0].pop("mean radius")
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert "missing" in response.get_json()["message"].lower()


def test_predict_rejects_extra_feature(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["instances"][0]["unexpected"] = 1
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert "unexpected" in response.get_json()["message"].lower()


def test_predict_rejects_non_numeric_feature(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["instances"][0]["mean radius"] = "17.99"
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert "numeric" in response.get_json()["message"].lower()


def test_predict_rejects_out_of_training_range(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["instances"][0]["mean radius"] = 1_000_000.0
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert "outside the training range" in response.get_json()["message"]


def test_predict_rejects_oversized_batch(client, valid_payload):
    payload = {"instances": valid_payload["instances"] * 101}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert "maximum" in response.get_json()["message"].lower()
