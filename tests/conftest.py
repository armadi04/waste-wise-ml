from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from waste_ml.api.main import create_app
from waste_ml.config import Settings
from waste_ml.preprocessing import load_image


class FakePredictor:
    model_version = "1.0.0"

    def __init__(self) -> None:
        self.classes = ["organic", "plastic", "paper"]

    def model_info(self) -> dict:
        return {
            "model_name": "waste-classifier",
            "model_version": self.model_version,
            "architecture": "MobileNetV2",
            "classes": self.classes,
            "input_size": [224, 224],
        }

    def predict_bytes(self, data: bytes) -> dict:
        load_image(data)
        return {
            "predicted_class": "plastic",
            "confidence": 0.94,
            "probabilities": {"organic": 0.02, "plastic": 0.94, "paper": 0.04},
            "model_version": self.model_version,
        }


class FailingPredictor(FakePredictor):
    def predict_bytes(self, data: bytes) -> dict:
        raise RuntimeError("boom")


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(
        predictor=FakePredictor(),
        settings=Settings(max_upload_size_mb=5),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def small_upload_client() -> Iterator[TestClient]:
    app = create_app(
        predictor=FakePredictor(),
        settings=Settings(max_upload_size_mb=1),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failing_client() -> Iterator[TestClient]:
    app = create_app(
        predictor=FailingPredictor(),
        settings=Settings(max_upload_size_mb=5),
    )
    with TestClient(app) as test_client:
        yield test_client
