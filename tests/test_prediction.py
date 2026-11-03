from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image


def make_image(fmt: str = "JPEG") -> bytes:
    image = Image.new("RGB", (256, 256), color=(30, 180, 60))
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_successful_prediction(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("waste.jpg", make_image(), "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "plastic"
    assert 0.0 <= payload["confidence"] <= 1.0
    assert set(payload["probabilities"]) == {"organic", "plastic", "paper"}
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-6
    assert payload["model_version"] == "1.0.0"


def test_invalid_image(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid image file"}


def test_unsupported_format(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("animation.gif", make_image("GIF"), "image/gif")},
    )
    assert response.status_code == 415
    assert response.json() == {"detail": "Unsupported image format"}


def test_oversized_upload(small_upload_client: TestClient) -> None:
    payload = b"\x00" * (1024 * 1024 + 1)
    response = small_upload_client.post(
        "/api/v1/predict",
        files={"file": ("big.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "exceeds maximum size" in response.json()["detail"]


def test_missing_file(client: TestClient) -> None:
    response = client.post("/api/v1/predict")
    assert response.status_code == 422
    assert "detail" in response.json()


def test_internal_inference_error(failing_client: TestClient) -> None:
    response = failing_client.post(
        "/api/v1/predict",
        files={"file": ("waste.png", make_image("PNG"), "image/png")},
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal inference error"}
