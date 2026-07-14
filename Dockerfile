# Slim multi-stage container configuration
# ==============================================================================
# STAGE 1: Dependency Compiler / Builder Layer
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Enforce clean Python bytecode compilation and disable caching to save space
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install systemic operating dependencies required to compile native C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install application dependencies into a isolated local user directory
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# ==============================================================================
# STAGE 2: Secure Production Deployment Layer
# ==============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

# Install minimal runtime libraries required by psycopg2 (PostgreSQL adapter)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged system user account to isolate runtime execution tasks
RUN useradd --create-home appuser
USER appuser

# Extract pre-compiled dependency structures from the builder workspace stage
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code workspace maps into the secure container path
COPY --chown=appuser:appuser setup.py .
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser src/ ./src/

# Install the application as a local editable module package layer
RUN pip install --user --no-deps .

# Expose standard FastAPI Uvicorn engine execution port boundary
EXPOSE 8000

# Expose structured application runtime health endpoints 
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Launch the FastAPI web server using explicit console script hooks
ENTRYPOINT ["telecom-api-start"]
