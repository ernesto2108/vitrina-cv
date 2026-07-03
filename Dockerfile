# syntax=docker/dockerfile:1.7

# ─── Stage 1: dependency resolver ───────────────────────────────────────────
# Resolves and installs only production dependencies into an isolated prefix.
# Keeps the final image clean: no build tools, no cache.
FROM python:3.12-slim AS deps

WORKDIR /build

# uv is the fastest resolver/installer; pinned for reproducibility.
RUN pip install --no-cache-dir uv==0.4.29

COPY pyproject.toml uv.lock ./

# Export pinned deps from the lockfile, then install into an isolated venv.
# --no-emit-project: omit the package itself (source copied in stage 2).
RUN uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.txt && \
    python -m venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt


# ─── Stage 2: runtime image ─────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# OpenCV headless needs these shared libs; no X11 / GUI needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy resolved venv from deps stage (single layer — avoids cache churn).
COPY --from=deps /opt/venv /opt/venv

# Copy application source.
COPY src/ ./src/

# Make the venv the active Python environment.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ─── Runtime defaults (all overridable via --env / env_file) ────────────────
ENV CV_ENGINE="opencv" \
    CV_MODEL_PATH="" \
    CV_PREFLIGHT_MIN_RESOLUTION="800x600" \
    CV_PREFLIGHT_MIN_CONTRAST="0.35" \
    CV_PREFLIGHT_MIN_LINE_DENSITY="0.005" \
    PORT="8000"

# Unprivileged user — no root in production.
RUN useradd --no-create-home --shell /bin/false vitrina
USER vitrina

EXPOSE 8000

# ─── Healthcheck ────────────────────────────────────────────────────────────
# interval: wait between probes after the container is running.
# start_period: grace window for the warm-up before failures count.
# The /health endpoint returns 200 only after the engine is fully ready (ADR-010).
HEALTHCHECK \
    --interval=15s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD python -c "import urllib.request, sys; r=urllib.request.urlopen('http://localhost:${PORT}/health'); sys.exit(0 if r.status==200 else 1)"

# ─── Entrypoint ─────────────────────────────────────────────────────────────
# uvicorn logs to stdout (PYTHONUNBUFFERED=1 above).
# --host 0.0.0.0 required inside Docker; port from env.
CMD ["sh", "-c", "uvicorn vitrina_cv.main:app --host 0.0.0.0 --port ${PORT}"]
