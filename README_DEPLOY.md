# Finmodel 17 — macro, news & geopolitical intelligence, public-ready

One Docker image, four ways to host it. Pick by what you need:

| Host | Cost | Power | Sleeps? | Best when |
|---|---|---|---|---|
| **Google Cloud Run** (recommended) | Always-free monthly allowance; card required | 1 vCPU / 1 GiB per request | Scales to zero; ~10–20 s cold start | You want a public HTTPS URL with no server to manage |
| **Oracle Cloud Always Free VM** | Free; card required for verification | Arm VM, currently documented as up to 2 OCPU / 12 GB | Never | You want it always on and fast |
| Koyeb free | Free (1 service); card required | 512 MB / 0.1 vCPU | Scales to zero when idle | Small, light traffic |
| Render free | Free | 512 MB / 0.1 vCPU | After 15 min idle | Small, light traffic |

Hugging Face Spaces is no longer a free option: Docker Spaces now require a paid Hugging Face plan.

---

## What's new in version 17 (intelligence layer)

Three new views in the sidebar, all built on the shared data layer so they add few provider calls:

**1. Macro & News Intelligence** (replaces "Macro Calendar & News")
- **A brief written by the app** at the top: dominant themes, what moved the asset last week (factor attribution: market, rates, dollar, oil, gold, volatility vs stock-specific), the next 7 days of high-impact events, the most relevant geopolitical scenario, and a bottom line.
- **Themes, not a pile of headlines.** Yahoo Finance and Google News (asset, macro and geopolitical searches) are merged; the same story from several outlets becomes one item with "also covered by" links. Stories are grouped into 15 themes (monetary policy, inflation, labor, growth, trade, sanctions, conflict, energy, politics, regulation, earnings, AI, supply chains, currencies, credit). Each theme gets a write-up: coverage and tone (finance word list, negation-aware, newer and more widely reported stories weigh more), the economic mechanism, the measured link to the asset (its beta to the relevant factor, how much that factor moved last week, and how much of the asset's return that explains), and the next calendar release that will test it.
- **Headline stream** with filters (about the asset / macro / geopolitical / by theme), tone chips and source links.
- **Macro exposures table:** the asset's 3-year weekly betas to the S&P 500, US 10Y yield, dollar index, WTI, gold and VIX, with t-statistics.

**2. Automatic economic calendar** (inside the same view)
- **Official US release dates from FRED** (St. Louis Fed) for CPI, PCE, PPI, jobs, JOLTS, claims, GDP, retail sales, industrial production, UMich, housing, durable goods, ECI and FOMC — needs a free `FRED_API_KEY` (step 0b below).
- **Consensus forecasts and previous values** for this and next week, including ECB, BoJ, BoE, China and other global events, from the Forex Factory weekly feed (no key).
- **Latest actual readings** from FRED data (no key): e.g. "Core CPI 3.0% y/y in Sep (prior 3.1%)".
- **Written analysis per event:** what it measures, why it matters, where things stand, consensus, what a hot or cold surprise usually does, **how this asset has reacted** (its beta to the channels, and its average move on the last ~40 release days vs normal days), and what to watch. Links to the official release page, FRED release calendar and data series.
- Times in **your local time zone** with countdowns; filters for horizon, importance, region and "only events that move this asset"; a "most relevant for this asset" pick.
- **Add to my calendar (.ics)** — download, or paste `https://<your-service-url>/api/calendar.ics` into Google/Apple/Outlook calendar "subscribe by URL" to keep it updated automatically.
- Fallbacks: without a key or if FRED is down, the Fed's published FOMC schedule (2024–2027), ISM conventions and the built-in BLS dates keep the calendar populated. The analysis pipeline never waits for the calendar.

**3. Geopolitical Shock Simulator** (new view)
- 12 scenarios with causal chains and historical analogues: Strait of Hormuz disruption, Russia–NATO escalation, Taiwan Strait blockade, US–China tariffs, sanctions on a commodity exporter, ceasefire, Fed hawkish/dovish, global recession scare, inflation re-acceleration, US fiscal/term-premium shock, major cyberattack — or a custom scenario.
- **How the "connections" are simulated:** a scenario shocks some factors (e.g. oil +22%, VIX +8). The other factors follow through their historical co-movement (conditional Gaussian, using either all weeks or only stress weeks, when correlations rise). 6,000 Monte Carlo paths with shock uncertainty and the asset's own residual risk give the outcome distribution, chance of loss, worst-5% average and the contribution of each channel.
- **Propagation network** diagram (trigger → shocked factors → spillovers → asset), outcome histogram, contribution chart, written analysis with model-fit caveats.
- **Stress-test matrix:** every scenario's expected effect on the asset, plus how strongly today's headlines match each scenario.

**4. Indicator Lab** (new view)
- Overlays: SMA/EMA (any periods), Bollinger Bands, Keltner Channels, Donchian Channel, Supertrend, Parabolic SAR, rolling VWAP, Ichimoku. Oscillators: RSI, MACD, Stochastic, ADX/DMI, CCI, Williams %R, MFI, OBV, ATR%. Daily, weekly or hourly; all parameters editable.
- **Signal summary** across 15 rules and a **reliability check on this asset:** average forward return, hit rate and "edge" vs the asset's normal drift after every past bullish and bearish signal.

New API endpoints: `/api/intel/{ticker}`, `/api/calendar`, `/api/calendar.ics`, `/api/factors/{ticker}`, `/api/geo/scenarios`, `/api/geo/simulate/{ticker}`, `/api/indicators/{ticker}` (see `/docs`). The new code lives in `finmodel_intel.py` (analysis) and `finmodel_intel_ui.py` (page); the main file only gained small hooks.

---

## What's new in version 16 (efficiency rebuild)

Measured with the same offline Yahoo stand-in for both versions (0.25 s per provider call):

| Scenario | v15 time | v16 time | Yahoo calls v15 → v16 | Data sent v15 → v16 |
|---|---|---|---|---|
| First analysis of a stock (AAPL) | 9.1 s | 1.1 s | 36 → 20 | 113 KB → 21 KB |
| Second stock (MSFT, shares benchmark/peers) | 9.1 s | 0.3 s | 36 → 6 | 113 KB → 22 KB |
| Gold futures (GC=F) | 2.5 s | 0.6 s | 10 → 5 | 106 KB → 20 KB |
| Market board, one category | 2.5 s | 0.3 s | 10 → 1 | 30 KB → 8 KB |
| Financial Model Studio | 9.3 s | 0.02 s | 37 → 0 | 11 KB → 3 KB |
| 5 visitors open the same new ticker at once | 9.1 s | 0.35 s | 72 → 6 | — |
| Dashboard page | — | — | — | 166 KB → 37 KB |
| Whole benchmark session | — | — | **204 → 39** | — |

In a browser test of the dashboard (0.15 s per call), the first numbers appeared after **0.4 s instead of 5.7 s**.

How:

1. **Shared market-data layer.** Every Yahoo dataset (prices, company info, statements, news) is downloaded once and reused across requests and visitors, with lifetimes that match the data: prices 15 min, quotes 2 min, company info 6 h, annual statements 24 h, headlines 30 min. The S&P 500 benchmark is fetched once for everyone. Shorter price histories are cut from the 5-year download instead of being fetched again. Failed or empty answers are retried after a minute instead of being hammered.
2. **Parallel downloads.** Price history, statements, news, the benchmark, the risk-free rate, the geopolitical feed and all 5 peers are fetched at the same time, so a cold analysis waits for the slowest call instead of the sum of all of them.
3. **Batched market board and scanner.** One provider request per board category (was one per symbol) and one for the whole scanner.
4. **gzip on every response.** Pages and JSON shrink by 65–80%.
5. **Request coalescing.** When several visitors ask for the same ticker at once, it is computed once and everyone gets that result.
6. **Stale-while-revalidate.** After 15 minutes a cached analysis is still served instantly (for up to an hour) while a fresh one is computed in the background.
7. **Progressive loading.** The page asks a new `/api/quick/{ticker}` endpoint for name, price, chart and technicals and paints them immediately, while the full analysis finishes in parallel on the same downloads.
8. **Indicators in the browser.** SMA lines are computed client-side and chart numbers are rounded, so less data travels.
9. **ML model trained once per day.** The XGBoost model is cached per ticker until a new daily bar arrives, so repeat visits, exports and the Model Studio cost no training time. It now predicts the **20-day return** from stationary features instead of the raw price level (tree models can't forecast above the highest price they were trained on). "Confidence" is now the model's out-of-sample directional hit rate on the most recent 20% of history.
10. **Live risk-free rate.** CAPM, Sharpe, the portfolio optimizer and exports use the latest US 10-year Treasury yield from FRED (no API key; cached 12 h), falling back to 4.25% if FRED is unreachable. Override with `RISK_FREE_RATE`.
11. **Calendar that doesn't expire.** The Fed's 2027 FOMC schedule is built in, and jobs-report and ISM dates beyond the hard-coded BLS list are generated from their standard release conventions and marked "(est.)".
12. **Pre-warming.** `/api/admin/warm` computes popular tickers, the market board and the scanner ahead of visitors (for a scheduler), and `WARM_INTERVAL_SECONDS` does the same continuously on an always-on VM.
13. **Optional disk cache.** With `CACHE_DIR` set, downloaded prices, info and statements survive restarts (restart test: 18 → 7 provider calls).
14. **Static-site export.** `/download/static-site?api_base=…` gives you the dashboard as a standalone `index.html` for free static hosting; it calls your API for data, so the page appears instantly even while the API wakes up.
15. **Stage timings.** Each analysis logs and returns `timings` (prefetch, risk, fundamentals, ml, peers, news, scoring, total), and every response carries a `Server-Timing` header visible in the browser's network tab.

Also fixed in this round: the instrument board's mini-charts were never destroyed (memory leak when switching categories), and clicking a board category un-highlighted the chart's timeframe tab and vice versa.

### Fixes from the first public-readiness round (still included)

- `scikit-learn` was missing, so every stock analysis failed; yfinance is now `>=1.0`.
- The dashboard script called a non-existent `setDerivativeMode()` and never loaded.
- Auto-refresh no longer jumps back to the Dashboard every minute (now every 5 min, only while the tab is visible).
- The Financial Model spreadsheet download works.
- Limited heavy-job slots, bounded caches, ticker validation, safer CORS, `xgboost-cpu` on Linux.

---

## Step 0 — put the files on GitHub (all options use this)

1. Create a new GitHub repository (public makes cloning from Cloud Shell simplest).
2. Upload every file in this folder to the repository root, including the hidden `.gitignore` and `.dockerignore`.

## Step 0b — get a free FRED API key (2 minutes, recommended)

1. Go to fredaccount.stlouisfed.org, create a free account and sign in.
2. Open **API Keys** → **Request API Key**, describe the use (e.g. "personal research dashboard") and submit.
3. Copy the 32-character key. It is free, read-only and only used by the server; you pass it as the `FRED_API_KEY` environment variable below (never put it in GitHub).

Without it the site still works; the calendar then relies on the weekly feed plus the built-in schedule and shows event-day statistics only for FOMC meetings.

---

## Option A — Google Cloud Run (recommended)

Everything below runs in **Cloud Shell** in your browser.

1. In console.cloud.google.com create a project and attach a billing account.
2. **Set a budget alert first:** Billing → Budgets & alerts → $1 with email alerts.
3. Open Cloud Shell (`>_`, top right) and run:

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
  --set-env-vars MAX_HEAVY_JOBS=2,FRED_API_KEY=<your-fred-key>
```

4. Answer **Y** to the prompts. The printed `Service URL` is your public link.
5. Update later: `git pull` in Cloud Shell and re-run the same deploy command.

The always-free allowance is 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests per month. `--max-instances 1` caps spending and keeps the in-memory cache in one place.

### Optional: keep popular tickers warm on Cloud Run

Pick a long random token, then:

```bash
gcloud run services update finmodel --region us-central1 --update-env-vars WARM_TOKEN=<long-random-token>

gcloud services enable cloudscheduler.googleapis.com
gcloud scheduler jobs create http finmodel-warm \
  --location us-central1 \
  --schedule "*/15 13-21 * * 1-5" --time-zone "UTC" \
  --uri "https://<your-service-url>/api/admin/warm?token=<long-random-token>" \
  --http-method GET --attempt-deadline 300s
```

That runs every 15 minutes during US market hours on weekdays, so visitors rarely hit a cold start or an uncached ticker. Each run uses roughly 30–60 CPU-seconds, which keeps a month of warming well inside the free allowance at these settings. Choose the tickers with `WARM_TICKERS`. On Cloud Run, CPU is only allocated while a request is running, so stale-while-revalidate refreshes finish during later requests or the next warm run.

---

## Option B — Oracle Cloud Always Free VM (always on, fastest)

1. Create an Oracle Cloud account (free resources live in your home region).
2. **Compute → Instances → Create instance**: *Ubuntu 24.04*, shape *Ampere VM.Standard.A1.Flex* (stay within "Always Free eligible"), add your SSH key. If you get "Out of host capacity", retry later or pick 1 OCPU.
3. **Networking → your VCN → Security list → Add ingress rules**: TCP 80 and 443 from `0.0.0.0/0`.
4. SSH in (`ssh ubuntu@<public-ip>`) and run:

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

git clone https://github.com/<you>/<repo>.git finmodel && cd finmodel
echo "DOMAIN=<ip-with-dashes>.sslip.io" > .env      # e.g. 129-146-10-20.sslip.io
echo "FRED_API_KEY=<your-free-fred-key>" >> .env    # optional, recommended
docker compose up -d --build
```

5. Open `https://<ip-with-dashes>.sslip.io` (Caddy gets the HTTPS certificate automatically).
6. `docker-compose.yml` already enables the disk cache (`CACHE_DIR`) and a 15-minute background warmer (`WARM_INTERVAL_SECONDS=900`). Update with `git pull && docker compose up -d --build`; logs with `docker compose logs -f app`.

---

## Option C — Koyeb or Render (512 MB free tiers)

- **Koyeb:** Create Web Service → GitHub → your repo → Builder **Dockerfile** → Instance **Free** → env vars `MAX_HEAVY_JOBS=1` and `FRED_API_KEY=<your key>` → health check `/health` → Deploy.
- **Render:** New + → **Blueprint** → your repo (`render.yaml` builds the Dockerfile on the free plan with `MAX_HEAVY_JOBS=1`); Render asks for `FRED_API_KEY` during setup.

---

## Optional — serve the page from free static hosting

The page then appears instantly while the API wakes from scale-to-zero.

1. Open `https://<your-api-url>/download/static-site?api_base=https://<your-api-url>` — it downloads `index.html` pointed at your API.
2. Upload that `index.html` to Cloudflare Pages, Netlify or a GitHub Pages repository.
3. Optional: restrict which sites may call your API with `ALLOWED_ORIGINS=https://<your-pages-domain>`.

---

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | 8080 | Port to listen on (hosts set this automatically) |
| `MAX_HEAVY_JOBS` | 2 | Fresh analyses allowed at once (cache hits never wait) |
| `HEAVY_JOB_WAIT_SECONDS` | 120 | Wait for a slot before answering "busy, retry" (503) |
| `CACHE_TTL_SECONDS` | 900 | How long an analysis counts as fresh |
| `CACHE_STALE_SECONDS` | 3600 | How long after that it is still served while refreshing |
| `CACHE_MAX_ENTRIES` | 150 | Max cached analyses/boards/portfolios |
| `PRICE_TTL_SECONDS` / `INFO_TTL_SECONDS` / `STATEMENT_TTL_SECONDS` | 900 / 21600 / 86400 | Data-layer lifetimes |
| `CACHE_DIR` | (off) | Folder to persist downloaded data across restarts |
| `WARM_TOKEN` | (off) | Enables `/api/admin/warm?token=…` |
| `WARM_TICKERS` | AAPL,MSFT,NVDA,SPY,QQQ,GC=F,CL=F,BTC-USD,EURUSD=X | What warming precomputes |
| `WARM_INTERVAL_SECONDS` | 0 (off) | Background warming loop for always-on hosts |
| `IO_WORKERS` | 8 | Parallel download threads |
| `ML_THREADS` | 2 | CPU threads for XGBoost training |
| `RISK_FREE_RATE` | (live FRED) | Fixed risk-free rate, e.g. `0.0425` |
| `ALLOWED_ORIGINS` | `*` | Sites allowed to call the API from a browser |
| `FRED_API_KEY` | (off) | Free St. Louis Fed key: official release dates and event-day statistics |

`/health` shows cache sizes, the current risk-free rate and counters such as `provider_calls`, `data_hits`, `coalesced_requests` and `cache_stale_served`, so you can watch how much Yahoo traffic the caches are saving.

---

## Things to know before sharing the link

- **Yahoo Finance is still the weak point.** Yahoo throttles cloud data-centre traffic. The data layer cuts request volume by roughly 80%, which helps a lot, but no free host can guarantee Yahoo access. If you see "Could not fetch price", wait or redeploy.
- **The intelligence layer is research, not advice.** Tone scores describe headline wording; scenario shock sizes are illustrative and loosely based on past episodes; betas and event statistics describe the past 3–5 years and can break in a crisis. The pages say so next to each result.
- **The weekly consensus feed (Forex Factory) is unofficial** and covers about two weeks; official dates come from FRED. Google News links go through news.google.com redirects.
- **Estimated calendar dates** after mid-December 2026 are labelled "(est.)" — check bls.gov and ismworld.org before relying on them. FOMC dates come from the Fed's published 2027 schedule.
- **The ML numbers changed meaning:** the forecast is a 20-day return model, and "confidence" is a historical directional hit rate (around 50% is coin-flip level). Treat it as a research signal, not a prediction.
- The dashboard loads Tailwind's CDN "play" script, which logs a "not for production" console warning; it works.
- Add a disclaimer that outputs are research estimates, not investment advice.

## Run it on your own computer (optional)

```bash
pip install -r requirements.txt
python Finmodel_multiasset_ticker_analysis.py     # http://localhost:8000
# or with Docker:
docker build -t finmodel . && docker run -p 8080:8080 finmodel   # http://localhost:8080
```
