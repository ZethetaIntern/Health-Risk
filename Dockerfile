# ==============================================================================
# HealthRisk AI & HealthRisk Lab - Docker Container Definition
# Multi-Domain Healthcare Risk Analytics & Interactive Simulation Engine
# ==============================================================================

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install ".[dev,mlops]" || pip install torch torch-geometric transformers xgboost lightgbm scikit-learn lifelines shap streamlit plotly pyyaml pydantic python-dotenv httpx tenacity rich pytest pytest-cov

# Copy repository source code
COPY . /app/

# Expose Streamlit & API ports
EXPOSE 8501
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 0

# Default entrypoint starts the HealthRisk Lab simulation
CMD ["streamlit", "run", "simulation/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
