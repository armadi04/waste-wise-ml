"""Pydantic response schemas documented in Swagger/OpenAPI."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["healthy"])
    service: str = Field(examples=["waste-ml"])


class ModelInfoResponse(BaseModel):
    model_name: str = Field(examples=["waste-classifier"])
    model_version: str = Field(examples=["1.0.0"])
    architecture: str = Field(examples=["MobileNetV2"])
    classes: list[str] = Field(examples=[["organic", "plastic", "paper"]])
    input_size: list[int] = Field(examples=[[224, 224]])


class PredictionResponse(BaseModel):
    predicted_class: str = Field(examples=["plastic"])
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model probability for the predicted class, not overall accuracy.",
        examples=[0.94],
    )
    probabilities: dict[str, float] = Field(
        examples=[{"organic": 0.02, "plastic": 0.94, "paper": 0.04}]
    )
    model_version: str = Field(examples=["1.0.0"])


class ErrorResponse(BaseModel):
    detail: str = Field(examples=["Unsupported image format"])
