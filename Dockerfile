# Finmodel — one container image that runs on Cloud Run, Koyeb, Render,
# Fly, Railway, an Oracle/any VM, or your own machine.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    MAX_HEAVY_JOBS=2

# libgomp1 = OpenMP runtime required by XGBoost on slim images
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY Finmodel_multiasset_ticker_analysis.py .

# Run as an unprivileged user; yfinance keeps its small cache in $HOME.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8080

# One worker: the result cache lives in process memory, and a single worker
# keeps RAM predictable on small instances. Hosts inject $PORT.
CMD exec uvicorn Finmodel_multiasset_ticker_analysis:app \
    --host 0.0.0.0 --port ${PORT:-8080} --workers 1 \
    --proxy-headers --forwarded-allow-ips="*" --timeout-keep-alive 30
