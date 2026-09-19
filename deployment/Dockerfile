# ==============================================================================
# CyberShield-BigData Production Docker Container (torch track, no Spark/Java)
# =============================================================================
FROM python:3.10-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/feature_store models reports logs benchmark_results \
    && chmod -R 777 data models reports logs benchmark_results

EXPOSE 8000

CMD ["uvicorn", "serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
