import json
from pathlib import Path

import pytest

from src.api import model as model_service
from src.main import create_app


@pytest.fixture(autouse=True)
def reset_model_cache():
    model_service.reset_model_cache()
    yield
    model_service.reset_model_cache()


@pytest.fixture()
def app():
    app = create_app()
    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def valid_payload():
    path = Path("src/tests/fixtures/predict_valid.json")
    return json.loads(path.read_text(encoding="utf-8"))
