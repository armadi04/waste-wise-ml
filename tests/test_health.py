from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "wastewaste-ml"}


def test_versioned_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "wastewaste-ml"}


def test_model_info_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "waste-classifier"
    assert payload["model_version"] == "1.0.0"
    assert payload["architecture"] == "MobileNetV2"
    assert payload["classes"] == ["organic", "plastic", "paper"]
    assert payload["input_size"] == [224, 224]


def test_openapi_documentation_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/api/v1/predict" in schema["paths"]
    assert "/api/v1/model/info" in schema["paths"]
