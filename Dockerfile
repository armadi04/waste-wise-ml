FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH=artifacts/waste_classifier.keras \
    MODEL_METADATA_PATH=artifacts/model_metadata.json \
    MAX_UPLOAD_SIZE_MB=5

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

COPY artifacts ./artifacts

EXPOSE 8000

CMD ["uvicorn", "waste_ml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
