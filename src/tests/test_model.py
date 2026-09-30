from pathlib import Path

from src.api import model as model_service


def test_model_bundle_is_versioned_and_complete():
    assert Path(model_service.MODEL_PATH).exists()

    bundle = model_service.get_bundle()

    assert bundle["model_version"]
    assert len(bundle["feature_names"]) == 30
    assert set(bundle["feature_names"]) == set(bundle["feature_ranges"])
    assert bundle["positive_class"] == 1
    assert bundle["label_mapping"][0] == "benign"
    assert bundle["label_mapping"][1] == "malignant"


def test_contract_contains_training_ranges():
    contract = model_service.get_contract()

    assert len(contract["features"]) == 30
    for feature in contract["features"]:
        assert feature["training_min"] <= feature["training_max"]
