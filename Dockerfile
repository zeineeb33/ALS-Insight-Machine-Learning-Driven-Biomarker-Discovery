# syntax=docker/dockerfile:1

# ---- build stage: resolve the locked runtime dependencies into a virtualenv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
# Runtime only: no dev group and no "pipeline" extra (no pandas/sklearn/pydeseq2/xgboost).
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# ---- runtime stage: slim image, non-root, serves precomputed results ----
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY results ./results
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ALS_INSIGHT_RESULTS_DIR=/app/results \
    PORT=8000
USER app
EXPOSE 8000
# Render (and most PaaS) inject $PORT; a single worker fits the 512 MB free tier.
CMD ["sh", "-c", "exec uvicorn --factory als_insight.serving.app:create_app --host 0.0.0.0 --port \"$PORT\" --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
