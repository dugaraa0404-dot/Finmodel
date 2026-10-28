# Finmodel — running it publicly

One Docker image, four ways to host it. Pick by what you need:

| Host | Cost | Power | Sleeps? | Best when |
|---|---|---|---|---|
| **Google Cloud Run** (recommended) | Always-free monthly allowance; card required | 1 vCPU / 1 GiB per request | Scales to zero; ~10–20 s cold start | You want a public HTTPS URL with no server to manage |
| **Oracle Cloud Always Free VM** | Free; card required for verification | Arm VM, currently documented as up to 2 OCPU / 12 GB | Never | You want it always on and fast |
| Koyeb free | Free (1 service); card required | 512 MB / 0.1 vCPU | Scales to zero when idle | Small, light traffic |
| Render free | Free | 512 MB / 0.1 vCPU | After 15 min idle | Small, light traffic |

On the 0.1-vCPU free tiers a first analysis (5 years of data + XGBoost training + peers + news) can take 30–90 s. Cloud Run and Oracle give a full CPU, so the same request is several times faster.

Hugging Face Spaces is no longer a free option for this: Docker Spaces now require a paid Hugging Face plan.

---

## What was fixed for public use

These would have broken the app on any host:

1. **Every stock analysis failed** — `XGBRegressor` needs `scikit-learn`, which was missing from `requirements.txt`. (FX/crypto silently fell back to a simple drift model, hiding the problem.)
2. **yfinance was pinned below 1.0** — 0.2.x is unmaintained and breaks when Yahoo changes its site. Now `yfinance>=1.0,<2.0`.
3. **The dashboard never loaded** — the page script called `setDerivativeMode()`, which doesn't exist; the error stopped the script before the first analysis ran.
4. **Auto-refresh hijacked the page** — every 60 s it re-ran the analysis, jumped back to the Dashboard tab, and re-scanned 25 markets. Now it refreshes quietly every 5 minutes, only while the tab is visible, and stays on the current view.
5. **Financial Model download always failed** — the workbook writer passed an openpyxl sheet to pandas. Fixed.
6. **Server guards** — heavy requests queue for a limited number of slots (`MAX_HEAVY_JOBS`) so a few visitors can't run the instance out of memory; the cache is bounded; the instrument board, scanner and news are now cached (they previously hit Yahoo on every page view); tickers in URLs are validated; CORS no longer combines `*` with credentials.
7. Smaller deps: `xgboost-cpu` on Linux (≈6 MB instead of ≈300 MB of CUDA libraries); deprecated pandas/datetime calls replaced; `python file.py` now binds `0.0.0.0`.

---

## Step 0 — put the files on GitHub (all options use this)

1. Create a new GitHub repository (public or private).
2. Upload every file in this folder to the repository root, including `Dockerfile`, `.dockerignore` and `.gitignore`.

---

## Option A — Google Cloud Run (recommended)

Everything below runs in **Cloud Shell** in your browser — nothing to install on your computer.

1. Go to console.cloud.google.com, create a project, and attach a billing account (needed even for the free tier).
2. **Set a budget alert first:** Billing → Budgets & alerts → create a budget of e.g. $1 with email alerts.
3. Open Cloud Shell (the `>_` icon, top right) and run:

```bash
git clone https://github.com/<you>/<repo>.git finmodel && cd finmodel

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud run deploy finmodel \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi --cpu 1 \
  --concurrency 8 \
  --min-instances 0 --max-instances 1 \
  --timeout 300 \
  --set-env-vars MAX_HEAVY_JOBS=2
```

4. Answer **Y** if asked to create an Artifact Registry repository. When it finishes it prints `Service URL: https://finmodel-xxxx.a.run.app` — that is your public link.
5. To update later: push changes to GitHub, then in Cloud Shell `cd finmodel && git pull` and re-run the same `gcloud run deploy` command.

Why these flags: `us-central1` is a free-tier pricing region; `--max-instances 1` caps spending and keeps the in-memory cache in one place; `--min-instances 0` means you pay nothing while idle. The always-free allowance is 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests per month — roughly 50 hours of actual request processing at these settings. Old container images in Artifact Registry count toward a small free storage allowance; delete old ones occasionally.

---

## Option B — Oracle Cloud Always Free VM (always on)

1. Create an Oracle Cloud account and pick your home region (free resources live there).
2. **Compute → Instances → Create instance**: Image *Ubuntu 24.04*, Shape *Ampere VM.Standard.A1.Flex* (stay within the "Always Free eligible" OCPU/RAM shown), add your SSH key, create. If you get "Out of host capacity", retry later or choose 1 OCPU.
3. **Networking → your VCN → Security list → Add ingress rules**: TCP 80 and TCP 443 from `0.0.0.0/0`.
4. SSH in (`ssh ubuntu@<public-ip>`) and run:

```bash
# Oracle's Ubuntu images block ports with iptables even after step 3:
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

git clone https://github.com/<you>/<repo>.git finmodel && cd finmodel
# Free HTTPS hostname that maps to your IP, e.g. IP 129.146.10.20 -> 129-146-10-20.sslip.io
echo "DOMAIN=<ip-with-dashes>.sslip.io" > .env
docker compose up -d --build
```

5. Open `https://<ip-with-dashes>.sslip.io`. Caddy gets the HTTPS certificate automatically. If you own a domain, point an A record at the IP and put that domain in `.env` instead.
6. Update later: `git pull && docker compose up -d --build`. Logs: `docker compose logs -f app`.

---

## Option C — Koyeb or Render (512 MB free tiers)

- **Koyeb:** Create Web Service → GitHub → select the repo → Builder: **Dockerfile** → Instance: **Free** → Region Frankfurt or Washington → add env var `MAX_HEAVY_JOBS=1` → health check path `/health` → Deploy.
- **Render:** New + → **Blueprint** → select the repo. `render.yaml` now builds the Dockerfile on the free plan with `MAX_HEAVY_JOBS=1`.

Keep `MAX_HEAVY_JOBS=1` on 512 MB instances: one analysis plus the app uses roughly 250–350 MB.

---

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | 8080 | Port to listen on (hosts set this automatically) |
| `MAX_HEAVY_JOBS` | 2 | Analyses allowed to run at once; others queue |
| `HEAVY_JOB_WAIT_SECONDS` | 120 | Queue wait before returning "busy, retry" (503) |
| `CACHE_TTL_SECONDS` | 900 | How long results are reused |
| `CACHE_MAX_ENTRIES` | 150 | Cache size cap |
| `ALLOWED_ORIGINS` | `*` | Comma-separated origins allowed to call the API from other sites |

---

## Things to know before sharing the link

- **Yahoo Finance is the weak point.** Yahoo throttles traffic from cloud data centres. yfinance 1.x mimics a browser, and the cache plus job limit keep request volume low, but if a host's IP gets throttled you'll see "Could not fetch price" or empty news. Wait a while or redeploy (Cloud Run may get a different outbound IP). No free host can guarantee Yahoo access.
- **The economic calendar is hard-coded through 15 Dec 2026.** After that date the Macro Calendar is empty until 2027 dates are added to `build_economic_calendar()`.
- The cache is in memory: restarts and Cloud Run scale-to-zero clear it, so the first visitor after idle waits for a fresh analysis.
- The dashboard loads Tailwind from its CDN "play" script, which logs a "not for production" console warning. It works; switch to a compiled stylesheet if that matters to you.
- Add a disclaimer that outputs are research estimates, not investment advice, and review Yahoo's terms if the site gets real traffic.

## Run it on your own computer (optional)

```bash
pip install -r requirements.txt
python Finmodel_multiasset_ticker_analysis.py     # http://localhost:8000
# or with Docker:
docker build -t finmodel . && docker run -p 8080:8080 finmodel   # http://localhost:8080
```
