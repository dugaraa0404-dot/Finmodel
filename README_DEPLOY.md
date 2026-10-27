# Finmodel — public free-tier deployment

This package contains the existing Finmodel FastAPI application and the deployment configuration for a free-tier Render Web Service.

## What is included

- `Finmodel_multiasset_ticker_analysis.py` — existing application (FastAPI app object: `app`)
- `requirements.txt` — Python runtime dependencies
- `render.yaml` — Render Blueprint configuration
- `Procfile` — compatible web start command
- `.gitignore` — excludes local environments, logs and secrets

## Deploy (no Python running on your own computer)

1. Create a GitHub repository and upload the files in this folder to its root.
2. Sign in to Render and connect that repository.
3. Choose **New + → Blueprint** if you want Render to read `render.yaml`, or create a **Web Service** and use:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn Finmodel_multiasset_ticker_analysis:app --host 0.0.0.0 --port $PORT`
   - Plan: Free
4. Wait for the build to finish, then open the assigned `https://<service-name>.onrender.com` URL.
5. Check `/health` and `/docs` on the deployed domain.

## Cost and free-tier limitations

- The configuration requests Render's Free plan. It does not intentionally configure paid resources.
- Render free web services sleep after 15 minutes without requests and wake when a visitor returns; the first request after sleep can be slow.
- Free instance hours, bandwidth, build minutes and resource limits apply. Check the provider dashboard and current plan terms.
- Free services do not have persistent local disk. This app's process-local cache is temporary and is cleared on restarts.
- This package does not add a database, background worker, paid market-data API, or paid service.
- Market data continues to use the providers already used by the app, principally Yahoo Finance via `yfinance`. Coverage, latency, ticker support and request throttling are outside this package's control.
- A free hosting plan is not an uptime guarantee. If usage exceeds free quotas or a provider changes its terms, zero cost or continuous availability cannot be guaranteed.
- A custom domain usually costs money. Use the free `onrender.com` subdomain to target a $0 monthly infrastructure budget.

## Before sharing publicly

1. Test key pages and API endpoints after deployment.
2. Check Render logs if the build fails. Large numerical packages such as XGBoost can make builds slower and the app may be memory-constrained on free instances.
3. Add request throttling and a shared cache before a large public launch. The existing in-memory cache is per-process and is not durable.
4. Review CORS, error messages, data-source terms and any user-submitted input before treating this as a production financial service.
5. Do not store passwords, API keys, or other secrets in the repository.

## Local smoke test (optional)

Install requirements in a virtual environment, then run:

```bash
uvicorn Finmodel_multiasset_ticker_analysis:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, `http://127.0.0.1:8000/health`, or `http://127.0.0.1:8000/docs`.
