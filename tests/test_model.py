from __future__ import annotations

import numpy as np
import pytest

from waste_ml.config import CLASSES, get_settings
from waste_ml.inference.predictor import WasteClassifier
from waste_ml.preprocessing import load_image


@pytest.fixture(scope="module")
def classifier() -> WasteClassifier:
    settings = get_settings()
    if not settings.resolved_model_path.is_file():
        pytest.skip("Trained model artifact is not available")
    if not settings.resolved_model_metadata_path.is_file():
        pytest.skip("Model metadata is not available")
    return WasteClassifier(
        settings.resolved_model_path,
        settings.resolved_model_metadata_path,
    )


def test_model_loads_and_returns_three_probabilities(classifier: WasteClassifier) -> None:
    image = load_image(_test_image_bytes())
    result = classifier.predict(image)

    assert len(result["probabilities"]) == len(CLASSES)
    assert set(result["probabilities"]) == set(CLASSES)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["predicted_class"] in CLASSES


def test_predicted_class_matches_highest_probability(classifier: WasteClassifier) -> None:
    result = classifier.predict(load_image(_test_image_bytes()))
    highest = max(result["probabilities"], key=result["probabilities"].get)
    assert result["predicted_class"] == highest
    assert np.isclose(result["confidence"], result["probabilities"][highest])


def _test_image_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (224, 224), color=(120, 80, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()
