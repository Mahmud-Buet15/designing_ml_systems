# TaxiETA app image: API, streaming consumer, replayer, MLflow server (Milestones 6, 9).
FROM python:3.11-slim

# LightGBM needs OpenMP at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[stream,tracking,serve]"

ENV TAXIETA_ROOT=/app \
    TAXIETA_DATA=/app/data \
    TAXIETA_REPORTS=/app/reports \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "taxieta.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
