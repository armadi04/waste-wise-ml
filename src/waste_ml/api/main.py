"""FastAPI application factory and entry point."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from waste_ml.api.routes import api_router, public_router
from waste_ml.config import Settings, get_settings
from waste_ml.inference.predictor import WasteClassifier

logger = logging.getLogger("waste_ml.api")


def create_app(
    predictor: Any | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create the FastAPI app, optionally with an injected predictor for tests."""
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = app_settings
        app.state.predictor = predictor
        app.state.predict_semaphore = asyncio.Semaphore(
            max(1, app_settings.max_concurrent_predictions)
        )

        if app_settings.is_production and not app_settings.internal_token:
            raise RuntimeError("INTERNAL_TOKEN must be set when APP_ENV=production")

        if app.state.predictor is None:
            try:
                app.state.predictor = WasteClassifier(
                    model_path=app_settings.resolved_model_path,
                    metadata_path=app_settings.resolved_model_metadata_path,
                )
            except Exception as exc:
                logger.exception("Failed to load the waste classifier at startup")
                raise RuntimeError(
                    "Failed to load waste classifier from "
                    f"{app_settings.resolved_model_path}: {exc}"
                ) from exc

        logger.info(
            "WasteWaste ML ready with model version %s",
            getattr(app.state.predictor, "model_version", "unknown"),
        )
        yield

    app = FastAPI(
        title=app_settings.app_name,
        version="1.0.0",
        description=(
            "REST API for classifying waste images into organic, plastic, and paper. "
            "The model is loaded once at application startup."
        ),
        lifespan=lifespan,
        # Interactive docs and the schema are not served in production.
        docs_url=None if app_settings.is_production else "/docs",
        redoc_url=None if app_settings.is_production else "/redoc",
        openapi_url=None if app_settings.is_production else "/openapi.json",
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    app.include_router(public_router)
    app.include_router(api_router)
    return app


app = create_app()
