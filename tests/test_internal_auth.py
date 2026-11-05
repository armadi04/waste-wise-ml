from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import FakePredictor
from test_prediction import make_image
from waste_ml.api.main import create_app
from waste_ml.config import Settings


def _client(**settings) -> TestClient:
    return TestClient(create_app(predictor=FakePredictor(), settings=Settings(**settings)))


def _predict(client: TestClient, headers: dict | None = None):
    return client.post(
        "/api/v1/predict",
        files={"file": ("waste.jpg", make_image(), "image/jpeg")},
        headers=headers or {},
    )


def test_predict_requires_token_when_configured() -> None:
    with _client(internal_token="s3cret-token-value") as client:
        assert _predict(client).status_code == 401
        assert _predict(client, {"X-Internal-Token": "wrong"}).status_code == 401
        assert _predict(client, {"X-Internal-Token": "s3cret-token-value"}).status_code == 200
        assert client.get("/health").status_code == 200


def test_production_refuses_to_start_without_token() -> None:
    with pytest.raises(RuntimeError, match="INTERNAL_TOKEN"), _client(app_env="production"):
        pass


def test_docs_hidden_in_production() -> None:
    with _client(app_env="production", internal_token="s3cret-token-value") as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
