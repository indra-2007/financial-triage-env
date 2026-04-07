FROM python:3.10-slim

WORKDIR /app

# Ensure git and curl are available (curl for healthcheck, git for openenv-core install)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . /app/

# Set PYTHONPATH so absolute directory imports work properly
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose required port for Hugging Face Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://0.0.0.0:7860/health || exit 1

# Start the uvicorn server unconditionally
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
