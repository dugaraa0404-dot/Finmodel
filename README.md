# Finmodel — Render deployment

This folder packages the uploaded Finmodel 17.1.0 FastAPI application for Render Docker deployment. The application is a single Python file, including its embedded intelligence and dashboard modules.

## Deploy

1. Upload **the contents of this folder** to the root of a GitHub repository (not an extra nested folder).
2. In Render, choose **New > Blueprint**, connect that repository and apply `render.yaml`; or choose **New > Web Service**, select **Docker** and the repository.
3. If Render requests `FRED_API_KEY`, enter your key; optional FRED-backed features may be limited without it.
4. After deployment, visit `/health` on the service URL, then visit `/`.

Render may take time to install XGBoost dependencies. Free instances may sleep when idle. This bundle was syntax-checked, not deployed or integration-tested against live market-data providers.

To run locally (Python 3.12): `pip install -r requirements.txt` then `uvicorn Finmodel_multiasset_ticker_analysis:app --host 127.0.0.1 --port 8000`.
