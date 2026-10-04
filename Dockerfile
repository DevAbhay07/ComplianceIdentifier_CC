# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# NitiMark — Product Compliance ML API
# Dockerfile for production deployment on Render / Cloud
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── Metadata ─────────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="NitiMark Product Compliance ML API"
LABEL org.opencontainers.image.description="FastAPI wrapper for the NitiMark Legal Metrology ML model"
LABEL org.opencontainers.image.source="https://github.com/DevAbhay07/ComplianceIdentifier_CC"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (security best practice) ───────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (before copying source for layer caching) ─────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY api/         ./api/
COPY src/         ./src/
COPY models/product_compliance/best_product_compliance_model.joblib \
     ./models/product_compliance/best_product_compliance_model.joblib

# ── Fix ownership ─────────────────────────────────────────────────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# ── Start FastAPI via Uvicorn ─────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
