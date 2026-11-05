"""HTTP routes for health checks, model metadata, and prediction."""

from __future__ import annotations

import hmac
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from waste_ml.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from waste_ml.preprocessing import InvalidImageError, UnsupportedImageFormatError

logger = logging.getLogger("waste_ml.api")

public_router = APIRouter(tags=["health"])


async def require_internal_token(
    request: Request,
    x_internal_token: Annotated[str | None, Header()] = None,
) -> None:
    """Reject callers that do not present the shared backend secret."""
    expected = request.app.state.settings.internal_token
    if not expected:
        return  # Unconfigured: allowed outside production (startup enforces it there).
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal token",
        )


api_router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_internal_token)])

PREDICTION_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid or unreadable image"},
    413: {"model": ErrorResponse, "description": "Upload exceeds the maximum size"},
    415: {"model": ErrorResponse, "description": "Unsupported image format"},
    422: {"model": ErrorResponse, "description": "Request validation error"},
    500: {"model": ErrorResponse, "description": "Internal inference error"},
}


def health_payload() -> HealthResponse:
    return HealthResponse(status="healthy", service="wastewaste-ml")


@public_router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health() -> HealthResponse:
    return health_payload()


@api_router.get("/health", response_model=HealthResponse, summary="Versioned health check")
async def api_health() -> HealthResponse:
    return health_payload()


@api_router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model information",
)
async def model_info(request: Request) -> ModelInfoResponse:
    info = request.app.state.predictor.model_info()
    return ModelInfoResponse(**info)


async def read_upload_limited(file: UploadFile, max_bytes: int, max_size_mb: int) -> bytes:
    """Read an upload without ever buffering more than the configured limit."""
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum size of {max_size_mb} MB",
        )

    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024
    while chunk := await file.read(chunk_size):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File exceeds maximum size of {max_size_mb} MB",
            )
        chunks.append(chunk)

    return b"".join(chunks)


@api_router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a waste image",
    description=(
        "Upload a JPEG, PNG, or WebP image (maximum 5 MB by default). "
        "The service returns the predicted class, confidence, per-class "
        "probabilities, and the model version."
    ),
    responses=PREDICTION_ERROR_RESPONSES,
)
async def predict(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="Waste image in JPEG, PNG, or WebP format (max 5 MB)."),
    ],
) -> PredictionResponse:
    settings = request.app.state.settings
    data = await read_upload_limited(
        file,
        max_bytes=settings.max_upload_size_bytes,
        max_size_mb=settings.max_upload_size_mb,
    )

    try:
        async with request.app.state.predict_semaphore:
            result = await run_in_threadpool(request.app.state.predictor.predict_bytes, data)
    except UnsupportedImageFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image format",
        ) from exc
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed for uploaded file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference error",
        ) from exc

    return PredictionResponse(**result)
