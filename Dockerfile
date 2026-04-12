# Dockerfile for Danish Electricity Price Forecast
# Multi-stage build to keep the image small

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Create data and artifacts directories
RUN mkdir -p data artifacts

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501 5000

# Default command: run the pipeline then start servers
CMD ["bash", "-c", "\
    python src/run_pipeline.py && \
    uvicorn src.api:app --host 0.0.0.0 --port 8000 & \
    streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0 & \
    mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 0.0.0.0 --port 5000 & \
    wait \
"]
