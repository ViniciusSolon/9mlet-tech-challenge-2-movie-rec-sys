FROM python:3.11-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN uv sync --frozen --no-dev

FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH" \
    GIT_PYTHON_REFRESH=quiet

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY pyproject.toml uv.lock README.md ./
COPY dvc.yaml params.yaml ./
COPY .dvc ./.dvc

RUN mkdir -p /app/dvc-storage /app/data/raw /app/data/processed /app/models

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
