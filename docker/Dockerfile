# ==============================================================================
# OpenLLM Ops Intelligence Backend Dockerfile
# CUDA-enabled Python container for LLM Infrastructure Monitoring & Troubleshooting
# ==============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including curl and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend /app

# Create persistent storage directories
RUN mkdir -p /app/data/uploads /app/data/logs /app/chroma_db

# Expose backend port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
