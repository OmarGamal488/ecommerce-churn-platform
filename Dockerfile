# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy project files needed for install
COPY pyproject.toml README.md ./
COPY src/__init__.py ./src/__init__.py

# Install dependencies (with extended timeout for slow connections)
RUN pip install --no-cache-dir --prefix=/install \
    --timeout=300 --retries=5 .

# ---- Stage 2: Runtime ----
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies needed by LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY dashboards/ ./dashboards/
COPY data/ ./data/
COPY models/ ./models/
COPY notebooks/ ./notebooks/
COPY pyproject.toml ./

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV FASTAPI_URL=http://fastapi:8000

# Expose ports (Dash=8050, FastAPI=8000)
EXPOSE 8050 8000
