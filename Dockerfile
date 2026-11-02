FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000 \
    MAX_HEAVY_JOBS=1 \
    ML_THREADS=1 \
    IO_WORKERS=4
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY Finmodel_multiasset_ticker_analysis.py .
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn Finmodel_multiasset_ticker_analysis:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --proxy-headers --forwarded-allow-ips='*' --timeout-keep-alive 30"]
