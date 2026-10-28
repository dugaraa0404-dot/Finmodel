"""finmodel_intel -- macro, news and geopolitical intelligence for Finmodel.

Adds four research layers on top of the main app's shared data layer:

1. AUTOMATIC ECONOMIC CALENDAR
   Official US release dates from FRED (St. Louis Fed; free API key in
   FRED_API_KEY), this/next week's consensus forecasts and previous values for
   global events from the Forex Factory weekly JSON feed, latest actual
   readings from FRED series (no key needed), and the Fed's FOMC schedule as a
   permanent fallback. Every event carries a written analysis, official links
   and -- for the selected asset -- its measured event-day sensitivity.

2. NEWS INTELLIGENCE
   Asset, macro and geopolitical headlines from several feeds are
   de-duplicated, grouped into themes, scored for tone with a finance lexicon
   (recency-weighted, negation-aware) and turned into coherent write-ups that
   connect each theme to the asset through its measured factor exposures and
   to the upcoming calendar.

3. GEOPOLITICAL SHOCK SIMULATOR
   A weekly factor model (S&P 500, US 10Y yield, US dollar, WTI, gold, VIX)
   estimates the asset's exposures. Scenarios shock some factors; the others
   follow through their historical co-movement (conditional Gaussian, normal or
   stress-regime covariance); Monte Carlo gives the asset's outcome
   distribution, factor contributions and a propagation network.

4. INDICATOR LAB
   ~20 selectable technical indicators with a signal summary and a per-asset
   reliability check (average forward return after each signal historically).

Nothing here is investment advice: scenario sizes are illustrative and all
statistics describe history.
"""
from __future__ import annotations

import math
import os
import re
import time
import threading
import html as _html
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote as _quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

core = None                      # main app module, set by install()
router = APIRouter()
ET_TZ = ZoneInfo("America/New_York")
UTC = timezone.utc
_UA = "Mozilla/5.0 (compatible; finmodel/17; +https://github.com)"
_bg_started: set[str] = set()
_bg_lock = threading.Lock()


def install(app, core_module) -> None:
    """Wire the module into the main app (called once at import time)."""
    global core
    core = core_module
    core._DATA_TTL.update({
        "cal_fred": 21600,        # FRED release schedule: 6 h
        "cal_ff": 10800,          # weekly consensus feed: 3 h
        "fred_hist": 86400,       # past release dates: 24 h
        "fred_obs": 21600,        # latest readings: 6 h
        "rss": 1800,              # headline feeds: 30 min
    })
    app.include_router(router)


# =============================================================================
# Small helpers
# =============================================================================
def _now() -> datetime:
    return datetime.now(UTC)


def _http_get(url: str, params: dict | None = None, timeout: float = 8.0) -> requests.Response:
    resp = requests.get(url, params=params, timeout=timeout,
                        headers={"User-Agent": _UA, "Accept": "*/*"})
    resp.raise_for_status()
    return resp


def _fnum(x, nd=4):
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd)


def _series_list(s: pd.Series, nd=4) -> list:
    return [None if (v is None or not np.isfinite(v)) else round(float(v), nd) for v in s.to_numpy(dtype=float)]


def _naive_daily_close(frame: pd.DataFrame) -> pd.Series:
    """Close prices on a tz-naive, de-duplicated daily date index."""
    if frame is None or frame.empty or "Close" not in frame:
        return pd.Series(dtype=float)
    s = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    s.index = idx.normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _background_once(key: str, fn) -> None:
    """Run fn in the background at most once at a time per key."""
    with _bg_lock:
        if key in _bg_started:
            return
        _bg_started.add(key)

    def job():
        try:
            fn()
        except Exception as exc:
            core.log.info("background %s failed: %s", key, exc)
        finally:
            with _bg_lock:
                _bg_started.discard(key)
    core._BG_POOL.submit(job)


def _significance(t: float | None) -> str:
    if t is None:
        return "untested"
    a = abs(t)
    if a >= 2.5:
        return "strong evidence"
    if a >= 2.0:
        return "statistically meaningful"
    if a >= 1.0:
        return "weak evidence"
    return "not distinguishable from zero"


# =============================================================================
# 1. Economic calendar
# =============================================================================
_IMPORTANCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "VERY HIGH": 3}

# Factor keys used across the module (Yahoo symbols).
F_SPX, F_TNX, F_USD, F_OIL, F_GOLD, F_VIX = "^GSPC", "^TNX", "DX-Y.NYB", "CL=F", "GC=F", "^VIX"

EVENT_TYPES: list[dict] = [
    {"key": "cpi", "name": "US Consumer Price Index (CPI)", "fred": [r"consumer price index"],
     "ff": r"^(Core )?CPI\b", "time": "08:30", "importance": "VERY HIGH", "category": "Inflation",
     "measures": "The monthly change in prices paid by urban consumers for a basket of goods and services. Core CPI excludes food and energy.",
     "why": "CPI is the most-watched inflation print. It feeds directly into the Fed's rate path, and therefore into Treasury yields, the dollar and the discount rate applied to every asset.",
     "hot": "A hotter print usually lifts yields and the dollar as markets price fewer cuts; long-duration equities, gold and bonds tend to suffer.",
     "cold": "A cooler print usually lowers yields and the dollar, supporting growth stocks, bonds and gold.",
     "channels": {F_TNX: 1, F_USD: 1, F_SPX: -1, F_GOLD: -1},
     "link": "https://www.bls.gov/cpi/",
     "series": [("CPIAUCSL", "Headline CPI", "yoy"), ("CPILFESL", "Core CPI", "yoy")],
     "themes": ["inflation", "monetary_policy"]},
    {"key": "pce", "name": "US PCE Inflation & Personal Spending", "fred": [r"personal income and outlays"],
     "ff": r"^(Core )?PCE\b|^Personal (Spending|Income)", "time": "08:30", "importance": "VERY HIGH", "category": "Inflation",
     "measures": "Personal income, consumer spending and the PCE price index -- the inflation gauge the Fed formally targets at 2%.",
     "why": "Core PCE is the Fed's preferred inflation measure, so it moves rate expectations even when CPI is already known; spending shows the health of the consumer (about two-thirds of GDP).",
     "hot": "Firm core PCE pushes rate-cut expectations out, lifting yields and the dollar.",
     "cold": "Soft core PCE supports the disinflation narrative and lower yields.",
     "channels": {F_TNX: 1, F_USD: 1, F_SPX: -1, F_GOLD: -1},
     "link": "https://www.bea.gov/data/personal-consumption-expenditures-price-index",
     "series": [("PCEPI", "Headline PCE", "yoy"), ("PCEPILFE", "Core PCE", "yoy")],
     "themes": ["inflation", "monetary_policy"]},
    {"key": "ppi", "name": "US Producer Price Index (PPI)", "fred": [r"producer price index"],
     "ff": r"^(Core )?PPI\b", "time": "08:30", "importance": "HIGH", "category": "Inflation",
     "measures": "Prices received by domestic producers -- a pipeline indicator for consumer inflation and for corporate margins.",
     "why": "Several PPI components feed directly into the PCE calculation, so economists revise their PCE forecasts on it; it also signals margin pressure for companies.",
     "hot": "Hot PPI can raise expected PCE and yields and hints at margin pressure for firms that cannot pass on costs.",
     "cold": "Cool PPI eases pipeline inflation concerns.",
     "channels": {F_TNX: 1, F_USD: 1, F_SPX: -1},
     "link": "https://www.bls.gov/ppi/",
     "series": [("PPIFIS", "PPI final demand", "yoy")],
     "themes": ["inflation"]},
    {"key": "nfp", "name": "US Employment Situation (Jobs Report)", "fred": [r"employment situation"],
     "ff": r"^(Non-Farm Employment Change|Unemployment Rate|Average Hourly Earnings)", "time": "08:30", "importance": "VERY HIGH", "category": "Labor",
     "measures": "Nonfarm payroll job growth, the unemployment rate and average hourly earnings for the prior month.",
     "why": "Payrolls are the single biggest scheduled market mover: they test both halves of the Fed's mandate (employment and, via wages, inflation).",
     "hot": "Strong payrolls and wages push yields and the dollar up as cuts get priced out; equities can rise on growth or fall on rates depending on which effect dominates.",
     "cold": "Weak payrolls pull yields and the dollar down and raise recession odds; defensive assets and gold tend to benefit.",
     "channels": {F_TNX: 1, F_USD: 1},
     "link": "https://www.bls.gov/news.release/empsit.toc.htm",
     "series": [("PAYEMS", "Payroll change", "diff_k"), ("UNRATE", "Unemployment rate", "level_pct")],
     "themes": ["labor", "monetary_policy"]},
    {"key": "jolts", "name": "US JOLTS Job Openings", "fred": [r"job openings and labor turnover( survey)?"],
     "ff": r"^JOLTS", "time": "10:00", "importance": "HIGH", "category": "Labor",
     "measures": "Job openings, hires, quits and layoffs -- labor demand rather than employment.",
     "why": "Openings relative to unemployed workers show how tight the labor market is, which drives wage inflation.",
     "hot": "More openings than expected signal a tight labor market and sticky wages (yields up).",
     "cold": "Falling openings and quits point to cooling wage pressure (yields down).",
     "channels": {F_TNX: 1, F_USD: 1},
     "link": "https://www.bls.gov/jlt/",
     "series": [("JTSJOL", "Job openings", "level_k")],
     "themes": ["labor"]},
    {"key": "claims", "name": "US Weekly Jobless Claims", "fred": [r"unemployment insurance weekly claims( report)?"],
     "ff": r"^Unemployment Claims", "time": "08:30", "importance": "MEDIUM", "category": "Labor",
     "measures": "New filings for unemployment insurance in the latest week -- the most timely labor-market signal.",
     "why": "A sustained rise in claims is one of the earliest recession warnings.",
     "hot": "Higher claims (weaker labor market) usually pull yields lower.",
     "cold": "Low claims confirm a resilient labor market.",
     "channels": {F_TNX: -1, F_USD: -1},
     "link": "https://oui.doleta.gov/unemploy/claims.asp",
     "series": [("ICSA", "Initial claims", "level_k_raw")],
     "themes": ["labor"]},
    {"key": "gdp", "name": "US GDP", "fred": [r"gross domestic product"],
     "ff": r"\bGDP\b", "time": "08:30", "importance": "HIGH", "category": "Growth",
     "measures": "Inflation-adjusted output of the US economy, quarterly at an annualised rate (advance, second and third estimates).",
     "why": "GDP confirms the growth backdrop behind earnings; the advance estimate moves markets most.",
     "hot": "Strong growth supports cyclical earnings but can lift yields.",
     "cold": "Weak growth raises recession odds, lowering yields and hurting cyclicals.",
     "channels": {F_SPX: 1, F_TNX: 1, F_USD: 1},
     "link": "https://www.bea.gov/data/gdp/gross-domestic-product",
     "series": [("A191RL1Q225SBEA", "Real GDP growth (SAAR)", "level_pct")],
     "themes": ["growth"]},
    {"key": "retail", "name": "US Retail Sales", "fred": [r"advance monthly sales for retail and food services"],
     "ff": r"^(Core )?Retail Sales", "time": "08:30", "importance": "HIGH", "category": "Consumer",
     "measures": "Monthly sales at retailers and restaurants (not adjusted for inflation).",
     "why": "The timeliest read on consumer demand, which drives most of US growth.",
     "hot": "Strong spending supports consumer-facing earnings and can lift yields.",
     "cold": "Weak spending raises slowdown fears.",
     "channels": {F_SPX: 1, F_TNX: 1},
     "link": "https://www.census.gov/retail/index.html",
     "series": [("RSAFS", "Retail sales", "mom")],
     "themes": ["growth", "consumer"]},
    {"key": "ip", "name": "US Industrial Production", "fred": [r"(g\.17 )?industrial production and capacity utilization"],
     "ff": r"^(Industrial Production|Capacity Utilization)", "time": "09:15", "importance": "MEDIUM", "category": "Growth",
     "measures": "Output of factories, mines and utilities, plus capacity utilization.",
     "why": "Tracks the goods cycle that matters for industrials, materials and energy demand.",
     "hot": "Rising output supports cyclicals and commodity demand.",
     "cold": "Falling output points to a manufacturing slowdown.",
     "channels": {F_SPX: 1, F_OIL: 1},
     "link": "https://www.federalreserve.gov/releases/g17/current/default.htm",
     "series": [("INDPRO", "Industrial production", "mom")],
     "themes": ["growth"]},
    {"key": "umich", "name": "UMich Consumer Sentiment & Inflation Expectations", "fred": [r"surveys of consumers"],
     "ff": r"UoM|Michigan", "time": "10:00", "importance": "MEDIUM", "category": "Consumer",
     "measures": "Household confidence and 1-year / 5-year inflation expectations from the University of Michigan survey.",
     "why": "The Fed watches long-run inflation expectations closely; a de-anchoring would force tighter policy.",
     "hot": "Rising inflation expectations can lift yields; rising confidence supports spending.",
     "cold": "Falling confidence warns of weaker spending.",
     "channels": {F_TNX: 1},
     "link": "https://www.sca.isr.umich.edu/",
     "series": [("UMCSENT", "Consumer sentiment", "level")],
     "themes": ["consumer", "inflation"]},
    {"key": "housing", "name": "US Housing Starts & Building Permits", "fred": [r"new residential construction"],
     "ff": r"^(Housing Starts|Building Permits)", "time": "08:30", "importance": "MEDIUM", "category": "Housing",
     "measures": "New homes started and permits issued -- the most rate-sensitive part of the economy.",
     "why": "Housing reacts first to mortgage rates and leads construction, materials and durable-goods demand.",
     "hot": "Strong housing points to resilience despite high rates.",
     "cold": "Weak housing shows rates are biting.",
     "channels": {F_SPX: 1, F_TNX: 1},
     "link": "https://www.census.gov/construction/nrc/index.html",
     "series": [("HOUST", "Housing starts (ann. rate)", "level_k")],
     "themes": ["growth"]},
    {"key": "durable", "name": "US Durable Goods Orders", "fred": [r"advance (report on )?durable goods.*"],
     "ff": r"^(Core )?Durable Goods", "time": "08:30", "importance": "MEDIUM", "category": "Growth",
     "measures": "New orders for long-lasting manufactured goods; core orders proxy business investment.",
     "why": "Core capital-goods orders are a leading indicator of business capex.",
     "hot": "Strong orders support industrials and capex-linked names.",
     "cold": "Weak orders signal cautious corporate spending.",
     "channels": {F_SPX: 1},
     "link": "https://www.census.gov/manufacturing/m3/index.html",
     "series": [("DGORDER", "Durable goods orders", "mom")],
     "themes": ["growth"]},
    {"key": "eci", "name": "US Employment Cost Index", "fred": [r"employment cost index"],
     "ff": r"^Employment Cost Index", "time": "08:30", "importance": "MEDIUM", "category": "Labor",
     "measures": "Quarterly growth in wages and benefits, adjusted for job mix -- the Fed's favourite wage gauge.",
     "why": "Wage growth above ~3.5% is hard to square with 2% inflation, so ECI shapes the services-inflation outlook.",
     "hot": "Faster labor-cost growth feeds sticky-services-inflation concerns (yields up).",
     "cold": "Slower wage growth supports disinflation.",
     "channels": {F_TNX: 1, F_USD: 1},
     "link": "https://www.bls.gov/eci/",
     "series": [("ECIALLCIV", "Employment cost index", "yoy4")],
     "themes": ["labor", "inflation"]},
    {"key": "fomc", "name": "Federal Reserve FOMC Decision", "fred": [r"fomc press release"],
     "ff": r"^(Federal Funds Rate|FOMC Statement|FOMC Press Conference|FOMC Economic Projections)", "time": "14:00", "importance": "VERY HIGH", "category": "Central bank",
     "measures": "The Fed's policy-rate decision and statement (press conference at 14:30 ET; projections and dot plot in March, June, September and December).",
     "why": "The policy rate anchors every other interest rate, the dollar and equity valuations; guidance on future meetings often matters more than the decision itself.",
     "hot": "A hawkish surprise (fewer cuts, higher dots) lifts yields and the dollar and pressures equities and gold.",
     "cold": "A dovish surprise lowers yields and the dollar and supports equities, bonds and gold.",
     "channels": {F_TNX: 1, F_USD: 1, F_SPX: -1, F_GOLD: -1},
     "link": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
     "series": [("DFEDTARU", "Fed funds target (upper)", "level_pct")],
     "themes": ["monetary_policy"]},
    {"key": "ism_mfg", "name": "ISM Manufacturing PMI", "fred": [],
     "ff": r"^ISM Manufacturing", "time": "10:00", "importance": "MEDIUM", "category": "Growth",
     "measures": "Purchasing managers' survey of US factory activity; above 50 = expansion. The prices-paid sub-index tracks input inflation.",
     "why": "A timely, forward-looking read on the goods cycle and pipeline price pressure.",
     "hot": "A rise above 50 supports cyclicals and commodities.",
     "cold": "A sub-50 print renews growth-slowdown concerns.",
     "channels": {F_SPX: 1, F_OIL: 1, F_TNX: 1},
     "link": "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/",
     "series": [], "themes": ["growth"]},
    {"key": "ism_svc", "name": "ISM Services PMI", "fred": [],
     "ff": r"^ISM (Services|Non-Manufacturing)", "time": "10:00", "importance": "MEDIUM", "category": "Growth",
     "measures": "Purchasing managers' survey of the US services sector (most of the economy); above 50 = expansion.",
     "why": "Services drive employment and sticky inflation; the prices sub-index is watched for services inflation.",
     "hot": "A strong print supports the soft-landing narrative but can lift yields.",
     "cold": "Weakness raises recession concerns.",
     "channels": {F_SPX: 1, F_TNX: 1},
     "link": "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/",
     "series": [], "themes": ["growth"]},
]
_TYPE_BY_KEY = {t["key"]: t for t in EVENT_TYPES}

# Generic write-ups for non-US events from the weekly feed.
_GENERIC_TYPES = [
    (r"Rate Statement|Policy Rate|Cash Rate|Bank Rate|Refinancing Rate|Interest Rate Decision|Monetary Policy|Press Conference|Rate Decision",
     "Central bank", "A central-bank policy decision or communication.",
     "Policy rates set the price of money in that currency: a hawkish surprise usually strengthens the currency and raises local yields; a dovish one does the opposite. Large central banks (ECB, BoJ, BoE, PBoC) also spill over into US yields and the dollar.",
     {F_USD: -1}),
    (r"\bCPI\b|Inflation|\bHICP\b|\bPPI\b", "Inflation", "Consumer or producer price growth.",
     "Inflation data shifts expectations for that central bank's next moves and therefore the currency and local bond yields.", {}),
    (r"\bGDP\b", "Growth", "Quarterly economic output.",
     "Growth surprises change earnings expectations and the policy outlook for that economy.", {}),
    (r"\bPMI\b", "Growth", "Purchasing managers' survey; above 50 = expansion.",
     "PMIs are the earliest monthly read on activity; China's PMIs in particular move commodities and Asian equities.", {F_OIL: 1}),
    (r"Employment|Unemployment|Payroll|Jobless|Claimant", "Labor", "Labor-market data.",
     "Labor tightness drives wages, inflation and policy expectations.", {}),
    (r"Retail|Consumer|Spending", "Consumer", "Consumer demand data.",
     "Household demand is the main growth engine in most large economies.", {}),
    (r"Trade Balance|Current Account|Exports|Imports", "Trade", "External trade data.",
     "Trade figures matter for currencies and, for China, for global demand and commodities.", {}),
]

_COUNTRY_REGION = {"USD": "United States", "EUR": "Euro area", "GBP": "United Kingdom", "JPY": "Japan",
                   "CNY": "China", "CAD": "Canada", "AUD": "Australia", "NZD": "New Zealand",
                   "CHF": "Switzerland", "ALL": "Global"}

# Historical FOMC decision days (second day of each meeting) for event-day
# statistics when no FRED key is configured.
_FOMC_DECISION_DAYS = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]
_SEP_MONTHS = {3, 6, 9, 12}


def _fred_key() -> str:
    return os.environ.get("FRED_API_KEY", "").strip()


def _match_type_fred(release_name: str) -> dict | None:
    name = re.sub(r"\s+", " ", release_name.lower()).strip()
    for t in EVENT_TYPES:
        if any(re.fullmatch(p, name) for p in t["fred"]):
            return t
    return None


def _match_type_ff(title: str, country: str) -> dict | None:
    if country != "USD":
        return None
    for t in EVENT_TYPES:
        if re.search(t["ff"], title, re.I):
            return t
    return None


def _et_datetime(d: date, hhmm: str) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime(d.year, d.month, d.day, h, m, tzinfo=ET_TZ)


# --- sources -------------------------------------------------------------------
def _load_fred_schedule() -> list[dict]:
    key = _fred_key()
    if not key:
        return []
    today = date.today()
    try:
        resp = _http_get("https://api.stlouisfed.org/fred/releases/dates", params={
            "api_key": key, "file_type": "json", "include_release_dates_with_no_data": "true",
            "realtime_start": (today - timedelta(days=3)).isoformat(),
            "realtime_end": (today + timedelta(days=100)).isoformat(),
            "sort_order": "asc", "order_by": "release_date", "limit": 1000}, timeout=10)
        rows = resp.json().get("release_dates", [])
    except Exception as exc:
        core.log.info("FRED schedule unavailable: %s", exc)
        return []
    out = []
    for r in rows:
        t = _match_type_fred(str(r.get("release_name", "")))
        if t is None:
            continue
        try:
            d = date.fromisoformat(str(r["date"]))
        except Exception:
            continue
        out.append({"type": t["key"], "date": d.isoformat(), "release_id": r.get("release_id"),
                    "release_name": r.get("release_name")})
    return out


def _load_ff_week(which: str) -> list[dict]:
    try:
        resp = _http_get(f"https://nfs.faireconomy.media/ff_calendar_{which}.json", timeout=8)
        rows = resp.json()
    except Exception as exc:
        core.log.info("weekly calendar feed (%s) unavailable: %s", which, exc)
        return []
    out = []
    for r in rows if isinstance(rows, list) else []:
        impact = str(r.get("impact", "")).strip().title()
        if impact not in {"High", "Medium"}:
            continue
        try:
            dt = datetime.fromisoformat(str(r.get("date")).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ET_TZ)
        except Exception:
            continue
        out.append({"title": str(r.get("title", "")).strip(),
                    "country": str(r.get("country") or r.get("currency") or "").upper().strip(),
                    "datetime": dt.astimezone(UTC), "impact": impact,
                    "forecast": str(r.get("forecast") or "").strip(),
                    "previous": str(r.get("previous") or "").strip(),
                    "actual": str(r.get("actual") or "").strip()})
    return out


def _fred_obs(series_id: str) -> pd.Series:
    """Full history of a FRED series via the keyless CSV endpoint."""
    def load():
        try:
            resp = _http_get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id}, timeout=8)
        except Exception as exc:
            core.log.info("FRED series %s unavailable: %s", series_id, exc)
            return None
        rows = []
        for line in resp.text.strip().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                rows.append((pd.Timestamp(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if not rows:
            return None
        return pd.Series([v for _, v in rows], index=[d for d, _ in rows])
    value = core._data_get("fred_obs", (series_id,), load)
    return value if isinstance(value, pd.Series) else pd.Series(dtype=float)


def _reading(series_id: str, label: str, transform: str) -> dict | None:
    s = _fred_obs(series_id).dropna()
    if len(s) < 3:
        return None
    last_date = s.index[-1]
    period = last_date.strftime("%b %Y") if transform != "level_k_raw" else last_date.strftime("%d %b %Y")
    try:
        if transform in ("yoy", "yoy4"):
            lag = 12 if transform == "yoy" else 4
            if len(s) <= lag + 1:
                return None
            cur = (s.iloc[-1] / s.iloc[-1 - lag] - 1) * 100
            prev = (s.iloc[-2] / s.iloc[-2 - lag] - 1) * 100
            text = f"{label}: {cur:.1f}% y/y in {period} (prior {prev:.1f}%)"
            unit = "% y/y"
        elif transform == "mom":
            cur = (s.iloc[-1] / s.iloc[-2] - 1) * 100
            prev = (s.iloc[-2] / s.iloc[-3] - 1) * 100
            text = f"{label}: {cur:+.1f}% m/m in {period} (prior {prev:+.1f}%)"
            unit = "% m/m"
        elif transform == "diff_k":
            cur = s.iloc[-1] - s.iloc[-2]
            prev = s.iloc[-2] - s.iloc[-3]
            avg3 = (s.iloc[-1] - s.iloc[-4]) / 3 if len(s) > 4 else None
            text = f"{label}: {cur:+,.0f}k in {period} (prior {prev:+,.0f}k" + (f", 3-mo avg {avg3:+,.0f}k)" if avg3 is not None else ")")
            unit = "k jobs"
        elif transform == "level_pct":
            cur, prev = s.iloc[-1], s.iloc[-2]
            text = f"{label}: {cur:.2f}% ({period}; prior {prev:.2f}%)"
            unit = "%"
        elif transform == "level_k":
            cur, prev = s.iloc[-1], s.iloc[-2]
            text = f"{label}: {cur:,.0f}k ({period}; prior {prev:,.0f}k)"
            unit = "k"
        elif transform == "level_k_raw":
            cur, prev = s.iloc[-1] / 1000, s.iloc[-2] / 1000
            avg4 = s.iloc[-4:].mean() / 1000 if len(s) >= 4 else None
            text = f"{label}: {cur:,.0f}k week of {period} (prior {prev:,.0f}k" + (f", 4-wk avg {avg4:,.0f}k)" if avg4 else ")")
            unit = "k"
        else:
            cur, prev = s.iloc[-1], s.iloc[-2]
            text = f"{label}: {cur:,.1f} ({period}; prior {prev:,.1f})"
            unit = ""
    except Exception:
        return None
    return {"label": label, "value": _fnum(cur, 3), "prior": _fnum(prev, 3), "unit": unit,
            "date": last_date.date().isoformat(), "text": text,
            "url": f"https://fred.stlouisfed.org/series/{series_id}"}


def _static_fallback_events() -> list[dict]:
    """The main app's built-in schedule, used when no live source answers."""
    out = []
    try:
        rows = core._static_economic_calendar(limit=60)
    except Exception:
        rows = []
    for r in rows:
        try:
            d = date.fromisoformat(r["date"])
        except Exception:
            continue
        name = str(r.get("event", ""))
        t = None
        for key, pattern in _STATIC_PATTERNS:
            if re.search(pattern, name, re.I):
                t = _TYPE_BY_KEY[key]
                break
        time_et = r.get("time_et") if re.match(r"^\d\d:\d\d$", str(r.get("time_et"))) else (t["time"] if t else "08:30")
        if r.get("time_et") == "FOMC":                      # decision is on the meeting's second day
            d = d + timedelta(days=1)
            time_et = "14:00"
        out.append({"type": t["key"] if t else None, "title": name, "date": d,
                    "time": time_et, "importance": r.get("importance", "HIGH"),
                    "country": "USD", "sources": [r.get("source", "Built-in schedule")], "estimated": "(est.)" in str(r.get("source", ""))})
    return out


_STATIC_PATTERNS = [("fomc", r"\bFOMC\b"), ("cpi", r"Consumer Price Index|\bCPI\b"), ("ppi", r"Producer Price|\bPPI\b"),
                    ("nfp", r"Employment Situation|Jobs Report"), ("jolts", r"\bJOLTS\b"), ("eci", r"Employment Cost"),
                    ("ism_mfg", r"ISM Manufacturing"), ("ism_svc", r"ISM Services"), ("pce", r"\bPCE\b|Personal Income"),
                    ("gdp", r"\bGDP\b|Gross Domestic"), ("retail", r"Retail Sales")]


def _fomc_events_ahead() -> list[dict]:
    today = date.today()
    out = []
    for iso in _FOMC_DECISION_DAYS:
        d = date.fromisoformat(iso)
        if today - timedelta(days=3) <= d <= today + timedelta(days=120):
            out.append({"type": "fomc", "title": "Federal Reserve FOMC Decision" + (" + Projections (dot plot)" if d.month in _SEP_MONTHS else ""),
                        "date": d, "time": "14:00", "importance": "VERY HIGH", "country": "USD",
                        "sources": ["Federal Reserve published schedule"]})
    return out


def _ism_estimates() -> list[dict]:
    out = []
    today = date.today()
    y, m = today.year, today.month
    for _ in range(3):
        for key, n in (("ism_mfg", 1), ("ism_svc", 3)):
            d = core._nth_business_day(y, m, n)
            if today - timedelta(days=3) <= d <= today + timedelta(days=100):
                out.append({"type": key, "title": _TYPE_BY_KEY[key]["name"], "date": d, "time": "10:00",
                            "importance": "MEDIUM", "country": "USD", "sources": ["ISM release convention (est.)"],
                            "estimated": True})
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _build_calendar() -> dict:
    """Merge every source into one de-duplicated, sorted event list."""
    fred_rows = core._data_get("cal_fred", ("schedule",), _load_fred_schedule) or []
    ff_rows = []
    for which in ("thisweek", "nextweek"):
        ff_rows += core._data_get("cal_ff", (which,), lambda w=which: _load_ff_week(w)) or []

    events: dict[tuple, dict] = {}

    def add(ev: dict):
        k = (ev.get("type") or ev["title"].lower(), ev["date"].isoformat(), ev.get("country", "USD"))
        if k in events:
            existing = events[k]
            for s in ev.get("sources", []):
                if s not in existing["sources"]:
                    existing["sources"].append(s)
            if not ev.get("estimated"):
                existing["estimated"] = False
            return existing
        events[k] = ev
        return ev

    for r in fred_rows:
        t = _TYPE_BY_KEY[r["type"]]
        d = date.fromisoformat(r["date"])
        title = t["name"]
        if t["key"] == "fomc" and d.month in _SEP_MONTHS:
            title += " + Projections (dot plot)"
        add({"type": t["key"], "title": title, "date": d, "time": t["time"], "importance": t["importance"],
             "country": "USD", "sources": ["FRED release schedule"], "release_id": r.get("release_id")})

    for ev in _fomc_events_ahead():
        if not any(e["type"] == "fomc" and abs((e["date"] - ev["date"]).days) <= 1 for e in events.values()):
            add(ev)
    for ev in _ism_estimates():
        add(ev)
    # The built-in schedule fills whatever FRED did not return (all of it when there is no key).
    fred_types = {r["type"] for r in fred_rows}
    for ev in _static_fallback_events():
        if ev["type"] in {"fomc", "ism_mfg", "ism_svc"} or (ev["type"] in fred_types):
            continue
        add(ev)

    # Weekly consensus feed: attach to matching US events, or add as new (global) events.
    for r in ff_rows:
        dt_et = r["datetime"].astimezone(ET_TZ)
        t = _match_type_ff(r["title"], r["country"])
        line = {"label": r["title"], "forecast": r["forecast"], "previous": r["previous"], "actual": r["actual"]}
        target = None
        if t is not None:
            for e in events.values():
                if e["type"] == t["key"] and abs((e["date"] - dt_et.date()).days) <= 1:
                    target = e
                    break
        if target is None:
            target = add({"type": t["key"] if t else None,
                          "title": t["name"] if t else f"{_COUNTRY_REGION.get(r['country'], r['country'])}: {r['title']}",
                          "date": dt_et.date(), "time": dt_et.strftime("%H:%M"),
                          "importance": (t["importance"] if t else ("HIGH" if r["impact"] == "High" else "MEDIUM")),
                          "country": r["country"] or "ALL", "sources": [], "ff_title": r["title"]})
        target.setdefault("consensus", [])
        if line not in target["consensus"]:
            target["consensus"].append(line)
        target["time"] = dt_et.strftime("%H:%M")       # exact time from the feed
        target["date"] = dt_et.date()
        if "Forex Factory weekly feed" not in target["sources"]:
            target["sources"].append("Forex Factory weekly feed")
        target["estimated"] = False

    out = []
    for e in events.values():
        try:
            dt = _et_datetime(e["date"], e["time"])
        except Exception:
            dt = _et_datetime(e["date"], "08:30")
        e["datetime_utc"] = dt.astimezone(UTC).isoformat()
        e["date_et"] = e["date"].isoformat()
        out.append(e)
    out.sort(key=lambda e: e["datetime_utc"])
    status = {
        "fred_schedule": "live" if fred_rows else ("no FRED_API_KEY set" if not _fred_key() else "unavailable"),
        "weekly_feed": "live" if ff_rows else "unavailable",
        "fallback_used": not fred_rows,
    }
    return {"events": out, "status": status, "built_at": _now().isoformat()}


def calendar_snapshot(block: bool = True) -> dict | None:
    """Cached calendar. With block=False never waits on the network."""
    key = "intel::calendar"
    hit = core._cache_peek(key)
    if hit is not None and hit[0] <= 6 * 3600:
        return hit[1]
    if not block:
        _background_once("calendar", lambda: core._cached_call(key, _build_calendar, heavy=False, ttl=6 * 3600))
        return hit[1] if hit is not None else None
    return core._cached_call(key, _build_calendar, heavy=False, ttl=6 * 3600,
                             cache_if=lambda v: bool(v.get("events")))


def calendar_rows_for_pipeline(limit: int = 12) -> list[dict] | None:
    """Rows in the main app's legacy calendar format, without blocking an analysis."""
    snap = calendar_snapshot(block=False)
    if not snap or not snap.get("events"):
        return None
    today = date.today().isoformat()
    rows = []
    for e in snap["events"]:
        if e["date_et"] < today or _IMPORTANCE_RANK.get(e["importance"], 1) < 2:
            continue
        t = _TYPE_BY_KEY.get(e.get("type") or "")
        rows.append({"date": e["date_et"], "time_et": e["time"], "event": e["title"],
                     "importance": e["importance"], "source": ", ".join(e["sources"]) or "Calendar",
                     "possible_result": (f"{t['hot']} {t['cold']}" if t else _generic_for(e)[3])})
        if len(rows) >= limit:
            break
    return rows


def _generic_for(e: dict):
    title = e.get("ff_title") or e.get("title", "")
    for pattern, cat, measures, why, channels in _GENERIC_TYPES:
        if re.search(pattern, title, re.I):
            return cat, measures, channels, why
    return "Other", "A scheduled release or event.", {}, "Scheduled events can move the local currency and rates when they surprise expectations."


# --- event-day sensitivity -------------------------------------------------------
def _release_history(event_type: str, release_id) -> list[date]:
    if event_type == "fomc" and not (_fred_key() and release_id):
        return [date.fromisoformat(d) for d in _FOMC_DECISION_DAYS if date.fromisoformat(d) < date.today()]
    if not (_fred_key() and release_id):
        return []

    def load():
        try:
            resp = _http_get("https://api.stlouisfed.org/fred/release/dates", params={
                "api_key": _fred_key(), "file_type": "json", "release_id": release_id,
                "realtime_start": (date.today() - timedelta(days=5 * 365)).isoformat(),
                "realtime_end": (date.today() - timedelta(days=1)).isoformat(),
                "sort_order": "desc", "limit": 400}, timeout=10)
            return [r["date"] for r in resp.json().get("release_dates", [])]
        except Exception as exc:
            core.log.info("FRED release history %s unavailable: %s", release_id, exc)
            return None
    rows = core._data_get("fred_hist", (str(release_id),), load) or []
    out = []
    for d in rows:
        try:
            out.append(date.fromisoformat(d))
        except Exception:
            continue
    return out


def event_sensitivity(close: pd.Series, dates: list[date], max_events: int = 40) -> dict | None:
    """Asset behaviour on past release days vs all other days (close-to-close)."""
    if close is None or len(close) < 260 or not dates:
        return None
    ret = close.pct_change().dropna() * 100
    idx = set(ret.index.date)
    ev_days = sorted({d for d in dates if d in idx}, reverse=True)[:max_events]
    if len(ev_days) < 6:
        return None
    mask = np.isin(ret.index.date, ev_days)
    ev, normal = ret[mask], ret[~mask]
    ratio = float(ev.abs().mean() / normal.abs().mean()) if normal.abs().mean() > 0 else None
    biggest = ev.abs().idxmax()
    return {"n": int(len(ev)), "avg_abs_move_event": _fnum(ev.abs().mean(), 2),
            "avg_abs_move_normal": _fnum(normal.abs().mean(), 2), "ratio": _fnum(ratio, 2),
            "avg_signed_move": _fnum(ev.mean(), 2), "up_share": _fnum((ev > 0).mean() * 100, 0),
            "largest": {"date": biggest.date().isoformat(), "move": _fnum(ev.loc[biggest], 2)},
            "since": min(ev_days).isoformat()}


def _event_analysis(e: dict, fm: dict | None, ticker: str | None, sens: dict | None) -> dict:
    t = _TYPE_BY_KEY.get(e.get("type") or "")
    if t:
        category, measures, why, channels = t["category"], t["measures"], t["why"], t["channels"]
        hot, cold = t["hot"], t["cold"]
        links = [{"label": "Official release page", "url": t["link"]}]
        if e.get("release_id"):
            links.append({"label": "FRED release calendar", "url": f"https://fred.stlouisfed.org/releases/{e['release_id']}"})
    else:
        category, measures, channels, why = _generic_for(e)
        hot = "An upside surprise usually strengthens the local currency and lifts local yields."
        cold = "A downside surprise usually weakens the currency and lowers local yields."
        links = []
    links.append({"label": "Forex Factory calendar", "url": "https://www.forexfactory.com/calendar"})

    paragraphs = [f"What it is: {measures}", f"Why it matters: {why}"]
    readings = []
    if t:
        for sid, label, tr in t.get("series", []):
            r = _reading(sid, label, tr)
            if r:
                readings.append(r)
    if readings:
        paragraphs.append("Where things stand: " + "; ".join(r["text"] for r in readings) + ".")
    if e.get("consensus"):
        bits = []
        for c in e["consensus"][:4]:
            part = c["label"]
            if c.get("forecast"):
                part += f" forecast {c['forecast']}"
            if c.get("previous"):
                part += f" (previous {c['previous']})"
            if c.get("actual"):
                part += f", actual {c['actual']}"
            bits.append(part)
        paragraphs.append("Consensus: " + "; ".join(bits) + ".")
    paragraphs.append(f"If it surprises high: {hot}")
    paragraphs.append(f"If it surprises low: {cold}")

    lens = []
    if fm and fm.get("ok") and channels:
        for f, sign in channels.items():
            b = fm["betas"].get(f)
            if not b or b.get("beta") is None:
                continue
            unit_move = {"bp": 10, "pts": 1}.get(b["unit"], 1)
            implied = b["beta"] * unit_move * sign
            lens.append(f"a {'+' if sign > 0 else '-'}{unit_move:g}{'bp' if b['unit'] == 'bp' else (' pt' if b['unit'] == 'pts' else '%')} "
                        f"move in {b['label']} has coincided with {implied:+.2f}% in {ticker} "
                        f"({_significance(b.get('t'))})")
    asset_text = ""
    if lens:
        asset_text = (f"For {ticker}: on a high surprise the usual channels are " +
                      ", ".join(f"{_FACTOR_LABEL.get(f, f)} {'up' if s > 0 else 'down'}" for f, s in channels.items()) +
                      ". Historically (weekly data), " + "; ".join(lens) + ".")
    if sens:
        asset_text += (f" On the last {sens['n']} release days {ticker} moved {sens['avg_abs_move_event']}% on average "
                       f"vs {sens['avg_abs_move_normal']}% on other days ({sens['ratio']}x), closing higher "
                       f"{sens['up_share']:.0f}% of the time; largest move {sens['largest']['move']:+.2f}% on {sens['largest']['date']}.")
    if asset_text:
        paragraphs.append(asset_text.strip())

    watch = "Watch the core/ex-food-and-energy measures and revisions to prior months as much as the headline."
    if t and t["key"] == "fomc":
        watch = "Watch the statement's guidance language, the press conference and (in Mar/Jun/Sep/Dec) the median dot for next year."
    elif t and t["key"] == "nfp":
        watch = "Watch revisions to the prior two months, average hourly earnings and the unemployment rate, not just the payroll headline."
    paragraphs.append(f"What to watch: {watch}")
    return {"category": category, "paragraphs": paragraphs, "readings": readings, "links": links,
            "channels": {(_FACTOR_LABEL.get(k, k)): v for k, v in channels.items()}}


def calendar_payload(ticker: str | None = None, days: int = 21, min_importance: str = "MEDIUM",
                     region: str = "all") -> dict:
    snap = calendar_snapshot(block=True) or {"events": [], "status": {}}
    min_rank = _IMPORTANCE_RANK.get(min_importance.upper(), 1)
    now = _now()
    lo = (now - timedelta(days=2)).isoformat()
    hi = (now + timedelta(days=days)).isoformat()
    events = [e for e in snap["events"] if lo <= e["datetime_utc"] <= hi
              and _IMPORTANCE_RANK.get(e["importance"], 1) >= min_rank
              and (region == "all" or (region == "us") == (e.get("country") == "USD"))]

    fm, close = None, None
    if ticker:
        try:
            fm = factor_model(ticker)
        except Exception as exc:
            core.log.info("factor model failed for %s: %s", ticker, exc)
        try:
            close = _naive_daily_close(core._ticker(ticker).history(period="5y"))
        except Exception:
            close = None

    # Event-day statistics: one per event type, computed in parallel.
    sens_by_type: dict[str, dict | None] = {}
    if ticker and close is not None and len(close):
        types = {}
        for e in events:
            if e.get("type") and e["type"] not in types:
                types[e["type"]] = e.get("release_id")
        futures = {k: core._IO_POOL.submit(_release_history, k, rid) for k, rid in types.items()}
        for k, fut in futures.items():
            try:
                sens_by_type[k] = event_sensitivity(close, fut.result(timeout=20))
            except Exception:
                sens_by_type[k] = None

    # Readings: prefetch FRED series in parallel (each cached 6 h).
    sids = {sid for e in events for sid, _, _ in _TYPE_BY_KEY.get(e.get("type") or "", {}).get("series", [])}
    core._prefetch([lambda s=s: _fred_obs(s) for s in sids])

    out = []
    for e in events:
        sens = sens_by_type.get(e.get("type") or "")
        item = {k: e.get(k) for k in ("type", "title", "date_et", "time", "datetime_utc", "importance",
                                     "country", "sources", "consensus", "estimated", "release_id")}
        item["region"] = _COUNTRY_REGION.get(e.get("country") or "", e.get("country"))
        item["released"] = e["datetime_utc"] < now.isoformat()
        item["analysis"] = _event_analysis(e, fm, ticker, sens)
        item["sensitivity"] = sens
        out.append(item)

    # Which upcoming event matters most for this asset?
    focus = None
    if ticker:
        best = None
        for item in out:
            if item["released"]:
                continue
            s = item.get("sensitivity") or {}
            score = (s.get("ratio") or 1.0) * (1 + _IMPORTANCE_RANK.get(item["importance"], 1))
            if best is None or score > best[0]:
                best = (score, item)
        if best:
            it = best[1]
            s = it.get("sensitivity") or {}
            focus = {"title": it["title"], "datetime_utc": it["datetime_utc"],
                     "why": (f"{ticker} has moved {s['ratio']}x its normal daily range on past release days."
                             if s.get("ratio") else "Highest-importance upcoming event.")}
    return {"ticker": ticker, "events": out, "status": snap.get("status", {}), "focus": focus,
            "fred_key_configured": bool(_fred_key()), "built_at": snap.get("built_at")}


def calendar_ics(days: int = 60, min_importance: str = "HIGH") -> str:
    snap = calendar_snapshot(block=True) or {"events": []}
    min_rank = _IMPORTANCE_RANK.get(min_importance.upper(), 2)
    now = _now()
    hi = (now + timedelta(days=days)).isoformat()
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//finmodel//macro calendar//EN",
             "CALSCALE:GREGORIAN", "X-WR-CALNAME:Finmodel macro calendar"]
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    for e in snap["events"]:
        if not (now.isoformat() <= e["datetime_utc"] <= hi) or _IMPORTANCE_RANK.get(e["importance"], 1) < min_rank:
            continue
        start = datetime.fromisoformat(e["datetime_utc"]).astimezone(UTC)
        end = start + timedelta(minutes=30)
        t = _TYPE_BY_KEY.get(e.get("type") or "")
        desc = (t["why"] if t else _generic_for(e)[3]).replace("\n", " ")
        if e.get("consensus"):
            desc += " Consensus: " + "; ".join(f"{c['label']} f {c.get('forecast') or '-'} p {c.get('previous') or '-'}"
                                               for c in e["consensus"][:3])
        uid = re.sub(r"[^A-Za-z0-9]", "", f"{e.get('type') or e['title']}{e['date_et']}{e.get('country')}")[:60]
        summary = f"[{e['importance']}] {e['title']}"

        def esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
        lines += ["BEGIN:VEVENT", f"UID:{uid}@finmodel", f"DTSTAMP:{stamp}",
                  f"DTSTART:{start.strftime('%Y%m%dT%H%M%SZ')}", f"DTEND:{end.strftime('%Y%m%dT%H%M%SZ')}",
                  f"SUMMARY:{esc(summary)}", f"DESCRIPTION:{esc(desc)[:900]}"]
        if t:
            lines.append(f"URL:{t['link']}")
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


# =============================================================================
# 2. Factor model (shared by news, calendar and simulator)
# =============================================================================
FACTORS = [
    (F_SPX, "S&P 500", "pct"),
    (F_TNX, "US 10Y yield", "bp"),
    (F_USD, "US dollar index", "pct"),
    (F_OIL, "WTI crude oil", "pct"),
    (F_GOLD, "Gold", "pct"),
    (F_VIX, "VIX volatility index", "pts"),
]
_FACTOR_LABEL = {f: l for f, l, _ in FACTORS}
_FACTOR_UNIT = {f: u for f, _, u in FACTORS}


def _weekly_changes(frame: pd.DataFrame, unit: str) -> pd.Series:
    s = _naive_daily_close(frame)
    if s.empty:
        return s
    if unit == "bp" and s.median() > 20:              # some feeds quote yield x10
        s = s / 10.0
    w = s.resample("W-FRI").last().dropna()
    if unit == "pct":
        return w.pct_change() * 100
    if unit == "bp":
        return w.diff() * 100
    return w.diff()


def _factor_model_uncached(ticker: str) -> dict:
    symbols = [f for f, _, _ in FACTORS]
    hist = core._batch_history(symbols + ([ticker] if ticker not in symbols else []), period="3y")
    cols = {}
    for f, _, unit in FACTORS:
        if f in hist:
            ch = _weekly_changes(hist[f], unit)
            if len(ch) > 30:
                cols[f] = ch
    asset_frame = hist.get(ticker)
    if asset_frame is None or asset_frame.empty:
        asset_frame = core._ticker(ticker).history(period="3y")
    y = _weekly_changes(asset_frame, "pct")
    if y.empty or len(cols) < 3:
        return {"ok": False, "reason": "Not enough price history for the factor model.", "ticker": ticker}
    data = pd.concat({**cols, "__asset__": y}, axis=1).dropna()
    data = data[np.isfinite(data).all(axis=1)]
    if len(data) < 52:
        return {"ok": False, "reason": "Less than one year of overlapping weekly data.", "ticker": ticker}

    factor_cols = [c for c in cols if c != ticker]
    asset_is_factor = ticker in cols
    X = data[factor_cols].to_numpy(float)
    yv = data["__asset__"].to_numpy(float)
    n, k = X.shape
    mu_x, sd_x = X.mean(0), X.std(0, ddof=1)
    sd_x[sd_x == 0] = 1.0
    Z = (X - mu_x) / sd_x
    yc = yv - yv.mean()
    lam = 2.0                                           # mild ridge: S&P and VIX are collinear
    A = Z.T @ Z + lam * np.eye(k)
    A_inv = np.linalg.inv(A)
    bz = A_inv @ Z.T @ yc
    resid = yc - Z @ bz
    dof = max(n - k - 1, 1)
    sigma2 = float(resid @ resid / dof)
    cov_b = sigma2 * A_inv @ (Z.T @ Z) @ A_inv
    se_z = np.sqrt(np.clip(np.diag(cov_b), 1e-18, None))
    r2 = float(1 - (resid @ resid) / (yc @ yc)) if yc @ yc > 0 else 0.0
    betas = {}
    for j, f in enumerate(factor_cols):
        b = bz[j] / sd_x[j]
        betas[f] = {"label": _FACTOR_LABEL[f], "unit": _FACTOR_UNIT[f], "beta": _fnum(b, 5),
                    "t": _fnum(bz[j] / se_z[j], 2), "factor_weekly_sd": _fnum(sd_x[j], 4),
                    "per_1sd_move": _fnum(b * sd_x[j], 3)}
    if asset_is_factor:
        betas[ticker] = {"label": _FACTOR_LABEL[ticker], "unit": _FACTOR_UNIT[ticker], "beta": 1.0, "t": None,
                         "factor_weekly_sd": _fnum(data[ticker].std(), 4), "per_1sd_move": _fnum(data[ticker].std(), 3),
                         "self": True}

    # Factor covariance: all weeks and a stress regime (top quartile of equity or volatility moves).
    all_f = [f for f, _, _ in FACTORS if f in cols]
    F = data[all_f]
    stress_mask = (F[F_SPX].abs() >= F[F_SPX].abs().quantile(0.75)) if F_SPX in F else pd.Series(False, index=F.index)
    if F_VIX in F:
        stress_mask = stress_mask | (F[F_VIX] >= F[F_VIX].quantile(0.75))
    cov_normal = F.cov().to_numpy()
    cov_stress = F[stress_mask].cov().to_numpy() if stress_mask.sum() >= 20 else cov_normal
    corr = F.corr().round(2)

    # Attribution of the latest week and the latest 4 weeks (weekly bins end on Friday;
    # the last bin can be a partial week, so dates shown are real trading days).
    asof = _naive_daily_close(asset_frame).index[-1].date()

    def attribution(rows: pd.DataFrame) -> dict:
        tot = rows.sum()
        contrib = {}
        for f in factor_cols:
            contrib[f] = float(betas[f]["beta"] * tot[f])
        if asset_is_factor:
            contrib = {ticker: float(tot["__asset__"])}
        explained = sum(contrib.values())
        actual = float(((1 + rows["__asset__"] / 100).prod() - 1) * 100)
        return {"actual": _fnum(actual, 2), "explained": _fnum(explained, 2), "stock_specific": _fnum(actual - explained, 2),
                "factor_moves": {f: _fnum(tot[f], 2) for f in all_f},
                "contributions": {f: _fnum(v, 2) for f, v in contrib.items()},
                "start": (rows.index[0] - pd.Timedelta(days=4)).date().isoformat(),
                "end": min(rows.index[-1].date(), asof).isoformat(),
                "partial": rows.index[-1].date() > asof}

    return {"ok": True, "ticker": ticker, "asset_is_factor": asset_is_factor, "weeks": int(n),
            "start": data.index[0].date().isoformat(), "end": data.index[-1].date().isoformat(),
            "r2": _fnum(r2, 3), "resid_weekly_sd": _fnum(math.sqrt(sigma2), 3) if not asset_is_factor else 0.0,
            "betas": betas, "factor_order": all_f,
            "cov_normal": cov_normal.tolist(), "cov_stress": cov_stress.tolist(),
            "stress_weeks": int(stress_mask.sum()),
            "correlation": {r: {c: _fnum(corr.loc[r, c], 2) for c in corr.columns} for r in corr.index},
            "last_week": attribution(data.tail(1)), "last_4_weeks": attribution(data.tail(4)),
            "labels": _FACTOR_LABEL, "units": _FACTOR_UNIT}


def factor_model(ticker: str) -> dict:
    t = ticker.strip().upper()
    return core._cached_call(f"intel::factors::{t}", lambda: _factor_model_uncached(t), heavy=False,
                             ttl=6 * 3600, cache_if=lambda v: bool(v.get("ok")))


# =============================================================================
# 3. News intelligence
# =============================================================================
THEMES: dict[str, dict] = {
    "monetary_policy": {"label": "Monetary policy & rates",
                        "kw": [r"\bfed\b", "federal reserve", "fomc", "powell", "rate cut", "rate hike", "interest rate", "central bank",
                               r"\becb\b", "lagarde", r"\bboj\b", "bank of japan", "bank of england", "treasury yield", "bond yield", "dot plot"],
                        "channel": "Policy expectations set the discount rate: higher expected rates lift Treasury yields and the dollar and compress valuation multiples, especially for long-duration growth assets.",
                        "factors": {F_TNX: 1, F_USD: 1}, "events": ["fomc", "cpi", "pce", "nfp"]},
    "inflation": {"label": "Inflation",
                  "kw": ["inflation", r"\bcpi\b", r"\bpce\b", r"\bppi\b", "prices rose", "price pressures", "disinflation", "deflation", "cost of living"],
                  "channel": "Inflation surprises move rate expectations: hotter prices mean fewer cuts, higher yields and pressure on bonds, gold and richly valued equities.",
                  "factors": {F_TNX: 1, F_USD: 1}, "events": ["cpi", "pce", "ppi", "umich"]},
    "labor": {"label": "Labor market",
              "kw": ["jobs report", "payroll", "unemployment", "jobless", "layoff", "hiring", "labor market", "labour market", "job openings", "wages"],
              "channel": "Labor strength sustains spending but keeps wage-driven inflation alive; weakness raises recession odds and pulls yields lower.",
              "factors": {F_TNX: 1, F_SPX: 1}, "events": ["nfp", "jolts", "claims", "eci"]},
    "growth": {"label": "Growth & recession risk",
               "kw": [r"\bgdp\b", "recession", "slowdown", "economic growth", r"\bpmi\b", "manufacturing", "soft landing", "contraction", "expansion"],
               "channel": "Growth news shifts earnings expectations and risk appetite: cyclicals, oil and yields rise with growth optimism and fall with recession fears.",
               "factors": {F_SPX: 1, F_OIL: 1, F_TNX: 1}, "events": ["gdp", "ism_mfg", "ism_svc", "ip", "retail"]},
    "trade": {"label": "Trade policy & tariffs",
              "kw": ["tariff", "trade war", "trade deal", "export control", "import duty", "trade talks", "customs", r"\bwto\b", "decoupling"],
              "channel": "Tariffs raise input costs and invite retaliation, hurting global supply chains, exporters and risk sentiment; trade deals reverse this.",
              "factors": {F_SPX: -1, F_VIX: 1, F_USD: 1}, "events": []},
    "sanctions": {"label": "Sanctions",
                  "kw": ["sanction", "embargo", "blacklist", "asset freeze", "price cap", "entity list"],
                  "channel": "Sanctions restrict trade and financial flows; on commodity exporters they tighten supply (oil, gas, metals, grains) and lift prices.",
                  "factors": {F_OIL: 1, F_GOLD: 1}, "events": []},
    "conflict": {"label": "War & military conflict",
                 "kw": [r"\bwar\b", "missile", "airstrike", "air strike", "invasion", "military", "troops", "drone attack", "attack on",
                        "ceasefire", "escalat", "hostilities", "naval", "blockade"],
                 "channel": "Conflict raises uncertainty and safe-haven demand (gold, dollar, Treasuries); if supply routes or producers are involved, energy and freight spike.",
                 "factors": {F_VIX: 1, F_GOLD: 1, F_OIL: 1, F_SPX: -1}, "events": []},
    "energy": {"label": "Energy & commodities",
               "kw": ["oil price", "crude", r"\bopec\b", "brent", "natural gas", r"\blng\b", "refinery", "pipeline", "strait of hormuz", "red sea", "barrel", "energy price"],
               "channel": "Energy prices feed headline inflation and consumer purchasing power; producers benefit, energy-intensive industries and importers suffer.",
               "factors": {F_OIL: 1}, "events": ["cpi"]},
    "politics": {"label": "Elections, fiscal & domestic politics",
                 "kw": ["election", "congress", "senate", "shutdown", "debt ceiling", "budget", "fiscal", "tax bill", "white house", "parliament", "coalition"],
                 "channel": "Fiscal and political uncertainty moves term premia (long-dated yields), the dollar and sector winners/losers from policy.",
                 "factors": {F_TNX: 1, F_VIX: 1}, "events": []},
    "regulation": {"label": "Regulation & legal",
                   "kw": ["antitrust", "lawsuit", "regulator", r"\bsec\b", "probe", "investigation", r"fined?\b", "ruling", r"courts?\b", r"bann?(ed|s)?\b"],
                   "channel": "Legal and regulatory actions are mostly company-specific: they hit margins, business models or capital returns rather than macro factors.",
                   "factors": {}, "events": []},
    "earnings": {"label": "Earnings & company fundamentals",
                 "kw": ["earnings", "revenue", "guidance", "profit", "quarterly results", "outlook", r"\beps\b", "margin", "forecast", "buyback", "dividend"],
                 "channel": "Results and guidance reset earnings expectations directly -- the most important driver of an individual stock over months.",
                 "factors": {}, "events": []},
    "tech_ai": {"label": "Technology & AI",
                "kw": [r"\bai\b", "artificial intelligence", "chip", "semiconductor", r"\bgpu\b", "data center", "cloud", "software"],
                "channel": "AI and semiconductor news drives capex expectations across the tech supply chain and moves the largest index weights.",
                "factors": {F_SPX: 1}, "events": []},
    "supply_chain": {"label": "Supply chains & shipping",
                     "kw": ["supply chain", "shipping", "freight", "shortage", r"ports?\b", "container", "logistics", "rare earth"],
                     "channel": "Supply disruptions raise costs and delay revenue; they also add to goods inflation.",
                     "factors": {F_OIL: 1}, "events": []},
    "currency": {"label": "Currencies",
                 "kw": ["dollar", "yuan", "renminbi", r"\byen\b", r"\beuro\b", "currency", r"\bforex\b", "exchange rate", "devaluation"],
                 "channel": "Currency moves change the translated value of foreign revenue and the competitiveness of exporters; a strong dollar tightens global financial conditions.",
                 "factors": {F_USD: 1}, "events": []},
    "credit": {"label": "Banking & credit",
               "kw": ["bank failure", "credit", "default", "bond market", "spread", "liquidity", "downgrade", "debt crisis", "private credit"],
               "channel": "Credit stress tightens financing conditions and triggers risk-off moves (higher VIX, safe-haven flows).",
               "factors": {F_VIX: 1, F_SPX: -1}, "events": []},
}
_THEME_RE = {k: [re.compile(p if p.startswith("\\b") else r"\b" + p, re.I) for p in v["kw"]] for k, v in THEMES.items()}

_POS = set("""beat beats beating surge surges surged soar soars soared rally rallies rallied gain gains gained jump jumps jumped
rise rises rising rose record strong stronger strength upgrade upgraded upgrades outperform growth grows grew expand expands expansion
rebound rebounds rebounded recover recovery optimism optimistic bullish boost boosts boosted improve improves improved improvement
easing eases eased cool cools cooled cooling deal agreement truce ceasefire breakthrough approve approves approved approval win wins
profit profitable raise raises raised resilient robust beat-and-raise accelerate accelerates upbeat""".split())
_NEG = set("""miss misses missed fall falls fell falling plunge plunges plunged slump slumps slumped drop drops dropped decline declines
declined sink sinks sank tumble tumbles tumbled crash crashes crashed weak weaker weakness downgrade downgraded downgrades cut cuts
loss losses lose loses warn warns warning fear fears worry worries concern concerns risk risks threat threatens threatened war attack
attacks sanction sanctions tariff tariffs lawsuit sued probe investigation recession slowdown layoffs layoff default crisis turmoil
escalate escalates escalation escalated conflict strike strikes missile shortage disruption disrupted bearish volatile volatility
selloff sell-off slowing contraction inflation hawkish delay delays delayed halt halts halted ban bans banned shutdown""".split())
_NEGATORS = {"not", "no", "never", "without", "despite", "fails", "failed", "less", "eases", "ease", "avoid", "avoids", "avoided"}
_STOP = set("the a an and or of to in on for with at by from as is are was were be been it its this that after before over under into amid "
            "says said new will would could may might up down out than more most stock stocks shares market markets news report".split())


def _tone(text: str) -> float:
    words = re.findall(r"[a-z][a-z\-]+", text.lower())
    score = 0.0
    hits = 0
    for i, w in enumerate(words):
        val = 1.0 if w in _POS else (-1.0 if w in _NEG else 0.0)
        if val == 0.0:
            continue
        if any(p in _NEGATORS for p in words[max(0, i - 2):i]):
            val = -val
        score += val
        hits += 1
    return 0.0 if hits == 0 else max(-1.0, min(1.0, score / (hits + 1.0)))


def _tokens(title: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]{3,}", title.lower()) if w not in _STOP}


def _parse_time(value) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except Exception:
            return None
    s = str(value).strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except Exception:
        pass
    try:
        dt = parsedate_to_datetime(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except Exception:
        return None


def _google_news(query: str, limit: int = 25) -> list[dict]:
    def load():
        url = "https://news.google.com/rss/search?q=" + _quote(query) + "&hl=en-US&gl=US&ceid=US:en"
        try:
            root = ET.fromstring(_http_get(url, timeout=8).text)
        except Exception as exc:
            core.log.info("news feed unavailable (%s): %s", query[:40], exc)
            return None
        out = []
        for item in root.findall(".//item")[:limit]:
            title = _html.unescape((item.findtext("title") or "").strip())
            src_el = item.find("source")
            source = (src_el.text or "").strip() if src_el is not None else ""
            if source and title.endswith(" - " + source):
                title = title[: -len(source) - 3].strip()
            out.append({"title": title, "source": source or "Google News",
                        "source_url": src_el.get("url") if src_el is not None else None,
                        "url": (item.findtext("link") or "").strip(),
                        "published": (item.findtext("pubDate") or "").strip()})
        return out
    return core._data_get("rss", (query,), load) or []


_MACRO_QUERY = ('("Federal Reserve" OR inflation OR "Treasury yields" OR "jobs report" OR recession OR '
                '"central bank" OR tariffs OR "oil prices") when:3d')
_GEO_QUERY = ('(sanctions OR war OR missile OR ceasefire OR "export controls" OR "trade war" OR OPEC OR '
              '"Strait of Hormuz" OR Taiwan OR "Red Sea" OR escalation OR NATO) when:3d')


def _collect_news(ticker: str, company: str, asset_class: str) -> list[dict]:
    jobs = {
        "asset_rss": lambda: _google_news(f'"{company}" when:7d' if company and company.upper() != ticker else f"{ticker} when:7d"),
        "macro": lambda: _google_news(_MACRO_QUERY),
        "geo": lambda: _google_news(_GEO_QUERY),
        "yahoo": lambda: core.fetch_asset_news(ticker, "Technology"),
    }
    futures = {k: core._IO_POOL.submit(fn) for k, fn in jobs.items()}
    raw = []
    for origin, fut in futures.items():
        try:
            rows = fut.result(timeout=25) or []
        except Exception:
            rows = []
        for r in rows:
            title = str(r.get("title") or "").strip()
            url = str(r.get("url") or "")
            if not title or not url.startswith(("http://", "https://")):
                continue
            raw.append({"title": title, "source": r.get("source") or ("Yahoo Finance" if origin == "yahoo" else "News"),
                        "url": url, "published_dt": _parse_time(r.get("published")),
                        "origin": "asset" if origin in ("asset_rss", "yahoo") else origin,
                        "summary": str(r.get("summary") or "")[:300]})
    return raw


def _dedupe(items: list[dict]) -> list[dict]:
    clusters: list[dict] = []
    for it in sorted(items, key=lambda x: x["published_dt"] or datetime(1970, 1, 1, tzinfo=UTC), reverse=True):
        toks = _tokens(it["title"])
        if not toks:
            continue
        for c in clusters:
            inter = len(toks & c["_tok"])
            if inter and inter / len(toks | c["_tok"]) >= 0.5:
                if it["source"] not in c["outlets"]:
                    c["outlets"].append(it["source"])
                    c["also"].append({"source": it["source"], "url": it["url"]})
                if it["origin"] == "asset":
                    c["origin"] = "asset"
                break
        else:
            clusters.append({**it, "_tok": toks, "outlets": [it["source"]], "also": []})
    return clusters


def _classify(story: dict) -> list[str]:
    text = story["title"] + " " + story.get("summary", "")
    scores = []
    for k, pats in _THEME_RE.items():
        n = sum(1 for p in pats if p.search(text))
        if n:
            scores.append((n, k))
    scores.sort(reverse=True)
    return [k for _, k in scores[:2]]


def _tone_word(x: float) -> str:
    if x >= 0.25:
        return "clearly positive"
    if x >= 0.08:
        return "mildly positive"
    if x > -0.08:
        return "balanced"
    if x > -0.25:
        return "mildly negative"
    return "clearly negative"


def _fmt_move(f: str, v) -> str:
    if v is None:
        return "n/a"
    u = _FACTOR_UNIT.get(f, "pct")
    return f"{v:+.0f}bp" if u == "bp" else (f"{v:+.1f} pts" if u == "pts" else f"{v:+.1f}%")


def news_intelligence(ticker: str) -> dict:
    t = ticker.strip().upper()
    try:
        info = core._ticker(t).info or {}
    except Exception:
        info = {}
    asset_class, display, _ = core.classify_ticker_asset(t, info)
    company = info.get("shortName") or info.get("longName") or display or t
    stories = _dedupe(_collect_news(t, company, asset_class))
    now = _now()
    name_re = re.compile(re.escape(company.split(",")[0].split(" Inc")[0][:24]), re.I) if company else None

    for s in stories:
        s["age_h"] = (now - s["published_dt"]).total_seconds() / 3600 if s["published_dt"] else 72.0
        s["tone"] = round(_tone(s["title"] + " " + s.get("summary", "")), 3)
        s["themes"] = _classify(s)
        s["weight"] = math.exp(-max(s["age_h"], 0) / 36.0) * (1 + 0.5 * math.log1p(len(s["outlets"]) - 1))
        s["about_asset"] = s["origin"] == "asset" or bool(name_re and name_re.search(s["title"])) or t in s["title"].upper().split()

    fm = None
    try:
        fm = factor_model(t)
    except Exception:
        fm = None
    cal = None
    try:
        cal = calendar_snapshot(block=False)
    except Exception:
        cal = None
    upcoming = []
    if cal:
        nowi = now.isoformat()
        upcoming = [e for e in cal.get("events", []) if e["datetime_utc"] >= nowi][:60]

    # Theme aggregation and write-ups.
    themes = []
    for key, meta in THEMES.items():
        members = [s for s in stories if key in s["themes"]]
        if not members:
            continue
        w = np.array([m["weight"] for m in members])
        tones = np.array([m["tone"] for m in members])
        tone = float((w * tones).sum() / w.sum()) if w.sum() > 0 else 0.0
        recent = [m for m in members if m["age_h"] <= 24]
        older = [m for m in members if m["age_h"] > 24]
        trend = None
        if recent and older:
            trend = float(np.mean([m["tone"] for m in recent]) - np.mean([m["tone"] for m in older]))
        outlets = sorted({o for m in members for o in m["outlets"]})
        intensity = float(w.sum())
        lead = sorted(members, key=lambda m: (m["weight"], len(m["outlets"])), reverse=True)[:4]

        paras = []
        span_h = max((m["age_h"] for m in members), default=0)
        trend_txt = ""
        if trend is not None and abs(trend) >= 0.1:
            trend_txt = f", and the last 24 hours have been {'more positive' if trend > 0 else 'more negative'} than before"
        paras.append(f"{len(members)} {'story' if len(members) == 1 else 'stories'} from {len(outlets)} "
                     f"{'outlet' if len(outlets) == 1 else 'outlets'} over the last {max(1, round(span_h / 24))} "
                     f"day(s); tone is {_tone_word(tone)} ({tone:+.2f}){trend_txt}.")
        paras.append(f"Mechanism: {meta['channel']}")
        lens_bits = []
        if fm and fm.get("ok"):
            lw = fm.get("last_week", {})
            for f, sign in meta["factors"].items():
                b = fm["betas"].get(f)
                if not b or b.get("beta") is None:
                    continue
                move = (lw.get("factor_moves") or {}).get(f)
                contrib = (lw.get("contributions") or {}).get(f)
                bit = (f"{b['label']} moved {_fmt_move(f, move)} last week" if move is not None else b["label"])
                if b.get("self"):
                    bit += f" -- this is {t} itself"
                else:
                    per = "10bp" if b["unit"] == "bp" else ("1 pt" if b["unit"] == "pts" else "1%")
                    mult = 10 if b["unit"] == "bp" else 1
                    bit += (f"; {t} has historically moved {b['beta'] * mult:+.2f}% per {per} ({_significance(b.get('t'))})")
                    if contrib is not None:
                        bit += f", worth about {contrib:+.2f}% of last week's return"
                lens_bits.append(bit)
        if lens_bits:
            paras.append(f"Link to {t}: " + "; ".join(lens_bits) + ".")
        elif meta["factors"] == {}:
            paras.append(f"Link to {t}: this theme works through company-specific fundamentals rather than macro factors"
                         + (" -- several stories mention the asset directly." if any(m["about_asset"] for m in members) else "."))
        watch = []
        for e in upcoming:
            if e.get("type") in meta["events"]:
                watch.append({"title": e["title"], "datetime_utc": e["datetime_utc"], "importance": e["importance"]})
            if len(watch) >= 3:
                break
        if watch:
            paras.append("Next data that will test this theme: " + "; ".join(
                f"{w['title']} ({w['datetime_utc'][:10]})" for w in watch) + ".")
        themes.append({"key": key, "label": meta["label"], "count": len(members), "outlets": len(outlets),
                       "tone": round(tone, 3), "tone_word": _tone_word(tone), "trend": _fnum(trend, 3),
                       "intensity": round(intensity, 3), "paragraphs": paras, "watch": watch,
                       "stories": [{"title": m["title"], "url": m["url"], "source": m["source"], "tone": m["tone"],
                                    "age_h": round(m["age_h"], 1), "outlets": len(m["outlets"]), "also": m["also"][:4]}
                                   for m in lead]})
    themes.sort(key=lambda x: x["intensity"], reverse=True)
    max_int = max((th["intensity"] for th in themes), default=1.0) or 1.0
    for th in themes:
        th["intensity_pct"] = round(th["intensity"] / max_int * 100)

    asset_stories = [s for s in stories if s["about_asset"]]
    aw = sum(s["weight"] for s in asset_stories)
    asset_tone = sum(s["weight"] * s["tone"] for s in asset_stories) / aw if aw else 0.0
    macro_stories = [s for s in stories if not s["about_asset"]]
    mw = sum(s["weight"] for s in macro_stories)
    macro_tone = sum(s["weight"] * s["tone"] for s in macro_stories) / mw if mw else 0.0

    # Scenario relevance from headlines (feeds the simulator).
    relevance = scenario_relevance(stories)

    # The brief: a short, coherent narrative.
    brief = []
    if themes:
        top = [th["label"].lower() for th in themes[:3]]
        brief.append(f"Big picture: the dominant themes in the news flow are {', '.join(top[:-1]) + ' and ' + top[-1] if len(top) > 1 else top[0]}. "
                     f"Macro and geopolitical headlines read {_tone_word(macro_tone)} ({macro_tone:+.2f}); "
                     f"coverage of {company} reads {_tone_word(asset_tone)} ({asset_tone:+.2f}) across {len(asset_stories)} stories.")
    if fm and fm.get("ok"):
        lw = fm["last_week"]
        parts = sorted(((f, v) for f, v in (lw.get("contributions") or {}).items() if v is not None),
                       key=lambda x: abs(x[1]), reverse=True)[:2]
        if lw.get("actual") is not None:
            txt = (f"What moved {t} in the latest {'week so far' if lw.get('partial') else 'week'} ({lw['start']} to {lw['end']}): it returned {lw['actual']:+.2f}%. ")
            if fm.get("asset_is_factor"):
                txt += "It is itself one of the macro factors, so its move is the factor move."
            else:
                txt += (f"Macro factors explain {lw['explained']:+.2f}% (largest: " +
                        ", ".join(f"{_FACTOR_LABEL.get(f, f)} {v:+.2f}%" for f, v in parts) +
                        f"); the remaining {lw['stock_specific']:+.2f}% is specific to {t} -- "
                        f"{'company news is the more likely driver' if abs(lw['stock_specific'] or 0) > abs(lw['explained'] or 0) else 'the macro backdrop did most of the work'}. "
                        f"Over three years these factors explain {round((fm['r2'] or 0) * 100)}% of {t}'s weekly moves.")
            brief.append(txt)
    if upcoming:
        hi = [e for e in upcoming if _IMPORTANCE_RANK.get(e["importance"], 1) >= 2
              and e["datetime_utc"] <= (now + timedelta(days=7)).isoformat()]
        if hi:
            brief.append("Coming up in the next 7 days: " + "; ".join(
                f"{e['title']} ({e['date_et']} {e['time']} ET)" for e in hi[:5]) + ". Open the calendar for each event's analysis and "
                f"{t}'s historical reaction.")
    if relevance and relevance[0]["score"] > 0:
        top_s = relevance[0]
        brief.append(f"Geopolitical radar: current headlines match the \"{top_s['name']}\" scenario most closely "
                     f"({top_s['hits']} matching stories). The Geopolitical Simulator shows how that shock would "
                     f"propagate to {t} through oil, rates, the dollar and risk appetite.")
    risk_themes = [th for th in themes[:6] if th["tone"] < -0.08]
    support_themes = [th for th in themes[:6] if th["tone"] > 0.08]
    bottom = f"Bottom line: news tone for {t} is {_tone_word(asset_tone)}"
    if risk_themes:
        bottom += f"; the main headline risk is {risk_themes[0]['label'].lower()}"
    if support_themes:
        bottom += f"; the main support is {support_themes[0]['label'].lower()}"
    bottom += ". Tone scores measure the wording of headlines, not their eventual market impact -- confirm with price action."
    brief.append(bottom)

    headlines = [{"title": s["title"], "url": s["url"], "source": s["source"], "tone": s["tone"],
                  "themes": [THEMES[k]["label"] for k in s["themes"]], "age_h": round(s["age_h"], 1),
                  "about_asset": s["about_asset"], "origin": s["origin"], "outlets": len(s["outlets"]),
                  "also": s["also"][:4]}
                 for s in sorted(stories, key=lambda x: x["age_h"])[:60]]
    return {"ticker": t, "company": company, "asset_class": asset_class, "brief": brief, "themes": themes[:8],
            "headlines": headlines, "asset_tone": round(asset_tone, 3), "macro_tone": round(macro_tone, 3),
            "story_count": len(stories), "scenario_relevance": relevance[:5],
            "factor_model": _public_factor_model(fm), "built_at": now.isoformat(),
            "method": ("Headlines from Yahoo Finance and Google News (asset, macro and geopolitical searches) are merged when "
                       "their titles overlap by at least half, grouped into themes by keyword, and scored with a finance word "
                       "list (negation-aware, newer stories weigh more: 36-hour decay; stories carried by several outlets weigh more). "
                       "Links to the asset come from a 3-year weekly factor regression.")}


def _public_factor_model(fm: dict | None) -> dict | None:
    if not fm or not fm.get("ok"):
        return fm
    return {k: fm[k] for k in ("ok", "ticker", "weeks", "start", "end", "r2", "resid_weekly_sd", "betas",
                               "correlation", "last_week", "last_4_weeks", "labels", "units", "asset_is_factor", "stress_weeks")}


# =============================================================================
# 4. Geopolitical shock simulator
# =============================================================================
SCENARIOS: list[dict] = [
    {"key": "hormuz", "name": "Middle East escalation / Strait of Hormuz disruption", "region": "Middle East",
     "shocks": {F_OIL: 22.0, F_GOLD: 4.0, F_VIX: 8.0},
     "chain": ["Attacks or a blockade threaten tanker traffic through the Strait (a large share of seaborne oil).",
               "Crude spikes and war-risk insurance and freight rates jump.",
               "Higher energy prices lift headline inflation expectations, complicating rate cuts.",
               "Safe-haven demand lifts gold; equity risk premia widen as volatility rises."],
     "kw": ["iran", "hormuz", "israel", "houthi", "red sea", "tanker", "gulf", "middle east", "saudi"],
     "analogues": "1990 Iraq-Kuwait oil spike; 2019 Abqaiq attack; 2024 Red Sea shipping attacks."},
    {"key": "russia_nato", "name": "Russia-NATO escalation", "region": "Europe",
     "shocks": {F_SPX: -6.0, F_VIX: 10.0, F_GOLD: 5.0, F_OIL: 8.0},
     "chain": ["Direct confrontation or a strike on NATO territory raises tail risk in Europe.",
               "Energy and grain supply fears return; European gas and oil rise.",
               "Global risk-off: equities fall, volatility jumps, gold and the dollar are bid."],
     "kw": ["russia", "nato", "ukraine", "kremlin", "putin", "baltic", "poland"],
     "analogues": "February-March 2022 invasion of Ukraine."},
    {"key": "taiwan", "name": "Taiwan Strait blockade", "region": "Asia",
     "shocks": {F_SPX: -12.0, F_VIX: 18.0, F_GOLD: 6.0, F_USD: 2.0},
     "chain": ["A blockade or quarantine disrupts shipping around Taiwan.",
               "Advanced-chip supply (largely made in Taiwan) is threatened; tech supply chains seize.",
               "Sanctions and counter-sanctions between the US and China follow.",
               "Severe global risk-off: equities and Asian currencies fall, dollar and gold rise."],
     "kw": ["taiwan", "strait", "pla navy", "beijing", "tsmc", "south china sea", "blockade"],
     "analogues": "No direct precedent; 1995-96 Taiwan Strait crisis and 2022 PLA drills were milder versions."},
    {"key": "tariffs", "name": "US-China tariff escalation", "region": "Global trade",
     "shocks": {F_SPX: -7.0, F_VIX: 9.0, F_USD: 1.5},
     "chain": ["New broad tariffs and retaliation on US-China trade.",
               "Import costs rise and supply chains reroute; exporters and multinationals lose margin.",
               "Growth expectations fall and volatility rises; the dollar firms on safe-haven and trade-balance effects."],
     "kw": ["tariff", "trade war", "export control", "china", "retaliat", "duties", "chips act", "entity list"],
     "analogues": "2018-2019 trade war; April 2025 'Liberation Day' tariff announcement."},
    {"key": "sanctions_exporter", "name": "Sanctions on a major commodity exporter", "region": "Global",
     "shocks": {F_OIL: 12.0, F_GOLD: 2.0},
     "chain": ["New sanctions or price caps remove exports from a major oil, gas or metals producer.",
               "Supply tightens and commodity prices rise; shipping reroutes.",
               "Importers face higher inflation; producers elsewhere benefit."],
     "kw": ["sanction", "embargo", "price cap", "export ban", "venezuela", "iran oil", "russian oil"],
     "analogues": "2022 EU/G7 Russian oil embargo and price cap; 2018 re-imposed Iran sanctions."},
    {"key": "ceasefire", "name": "Ceasefire / de-escalation", "region": "Global",
     "shocks": {F_OIL: -10.0, F_VIX: -4.0, F_SPX: 3.0},
     "chain": ["A credible ceasefire or peace deal removes a source of risk.",
               "Energy risk premium unwinds; oil falls.",
               "Volatility drops and risk assets rally; safe havens give back gains."],
     "kw": ["ceasefire", "truce", "peace deal", "peace talks", "de-escalat", "agreement"],
     "analogues": "Oil's reversal after the 1991 Gulf War air campaign began; June 2025 Israel-Iran ceasefire."},
    {"key": "fed_hawk", "name": "Fed hawkish surprise", "region": "United States",
     "shocks": {F_TNX: 40.0, F_USD: 2.0, F_SPX: -4.0},
     "chain": ["Inflation re-accelerates or the Fed signals fewer cuts / more hikes.",
               "Front-end and 10Y yields jump; the dollar strengthens.",
               "Higher discount rates compress equity multiples, hitting long-duration growth names hardest."],
     "kw": ["hawkish", "rate hike", "higher for longer", "fewer cuts", "sticky inflation"],
     "analogues": "2013 taper tantrum; 2022 tightening cycle."},
    {"key": "fed_dove", "name": "Fed dovish pivot", "region": "United States",
     "shocks": {F_TNX: -40.0, F_USD: -2.0, F_SPX: 4.0},
     "chain": ["Weak data or financial stress lead the Fed to signal faster cuts.",
               "Yields and the dollar fall.",
               "Lower discount rates lift equities, gold and rate-sensitive sectors."],
     "kw": ["dovish", "rate cut", "pivot", "easing", "emergency cut"],
     "analogues": "2019 'insurance cuts'; late-2023 pivot."},
    {"key": "recession", "name": "Global recession scare", "region": "Global",
     "shocks": {F_SPX: -10.0, F_TNX: -50.0, F_OIL: -18.0, F_VIX: 12.0},
     "chain": ["Data roll over and earnings forecasts get cut.",
               "Oil and industrial commodities fall on demand fears.",
               "Bonds rally (yields down) as markets price aggressive easing; equities and cyclicals fall."],
     "kw": ["recession", "slowdown", "contraction", "layoffs", "hard landing", "unemployment rises"],
     "analogues": "Q4 2018 growth scare; 2008 and 2020 in more extreme form."},
    {"key": "inflation_shock", "name": "Inflation re-acceleration", "region": "United States",
     "shocks": {F_TNX: 35.0, F_OIL: 8.0, F_GOLD: 3.0},
     "chain": ["Prices re-accelerate (energy, wages, tariffs).",
               "Markets push out rate cuts; yields rise.",
               "Real assets (commodities, gold) hold up better than long-duration equities and bonds."],
     "kw": ["inflation", "hot cpi", "price pressures", "wages surge", "re-accelerat"],
     "analogues": "2021-2022 inflation surge; early-2024 hot CPI prints."},
    {"key": "fiscal", "name": "US fiscal / term-premium shock", "region": "United States",
     "shocks": {F_TNX: 50.0, F_USD: -2.0, F_GOLD: 5.0, F_SPX: -5.0},
     "chain": ["Deficit worries, a weak Treasury auction or political standoff lift the term premium.",
               "Long-dated yields rise even without Fed action; the dollar can weaken on credibility concerns.",
               "Gold benefits as a non-sovereign store of value; equities de-rate."],
     "kw": ["deficit", "debt ceiling", "treasury auction", "downgrade", "shutdown", "term premium", "fiscal"],
     "analogues": "2023 US downgrade and bond sell-off; 2022 UK gilt crisis."},
    {"key": "cyber", "name": "Major cyberattack on critical infrastructure", "region": "Global",
     "shocks": {F_VIX: 10.0, F_SPX: -5.0, F_GOLD: 3.0},
     "chain": ["A large attack disrupts payments, energy grids or communications.",
               "Uncertainty spikes; volatility rises and equities fall.",
               "Cybersecurity and safe-haven assets are bid."],
     "kw": ["cyberattack", "cyber attack", "ransomware", r"hack(ed|ers?|ing)?\b", "outage", "grid attack"],
     "analogues": "2021 Colonial Pipeline ransomware; 2017 NotPetya."},
]
_SCEN_BY_KEY = {s["key"]: s for s in SCENARIOS}


def scenario_relevance(stories: list[dict]) -> list[dict]:
    out = []
    for sc in SCENARIOS:
        hits, score = 0, 0.0
        pats = [re.compile(r"\b" + re.escape(k), re.I) for k in sc["kw"]]
        for st in stories:
            title = st["title"]
            n = sum(1 for p in pats if p.search(title))
            if n:
                hits += 1
                score += st.get("weight", 1.0) * n
        out.append({"key": sc["key"], "name": sc["name"], "hits": hits, "score": round(score, 3)})
    out.sort(key=lambda x: x["score"], reverse=True)
    mx = out[0]["score"] if out and out[0]["score"] > 0 else 1.0
    for o in out:
        o["relevance_pct"] = round(o["score"] / mx * 100)
    return out


def _conditional(cov: np.ndarray, order: list[str], shocks: dict[str, float]):
    idx_s = [order.index(f) for f in shocks if f in order]
    idx_o = [i for i in range(len(order)) if i not in idx_s]
    if not idx_s:
        return idx_s, idx_o, np.zeros((len(idx_o), 0)), cov[np.ix_(idx_o, idx_o)]
    S = cov[np.ix_(idx_s, idx_s)] + np.eye(len(idx_s)) * 1e-9
    B = cov[np.ix_(idx_o, idx_s)] @ np.linalg.inv(S)
    cond = cov[np.ix_(idx_o, idx_o)] - B @ cov[np.ix_(idx_s, idx_o)]
    cond = (cond + cond.T) / 2
    w, v = np.linalg.eigh(cond)
    cond = v @ np.diag(np.clip(w, 0, None)) @ v.T
    return idx_s, idx_o, B, cond


def simulate(ticker: str, scenario: str = "hormuz", shocks: dict[str, float] | None = None,
             horizon_weeks: int = 4, regime: str = "stress", n: int = 6000, uncertainty: float = 0.35) -> dict:
    t = ticker.strip().upper()
    fm = factor_model(t)
    if not fm.get("ok"):
        raise ValueError(fm.get("reason", "Factor model unavailable for this asset."))
    order = fm["factor_order"]
    cov = np.array(fm["cov_stress"] if regime == "stress" else fm["cov_normal"], dtype=float)
    sc = _SCEN_BY_KEY.get(scenario)
    if scenario == "custom" or sc is None:
        sc = {"key": "custom", "name": "Custom scenario", "region": "User-defined", "shocks": {}, "chain": [],
              "analogues": "", "kw": []}
    primary = {f: float(v) for f, v in (shocks if shocks is not None else sc["shocks"]).items() if f in order}
    h = max(1, min(int(horizon_weeks), 26))
    idx_s, idx_o, B, cond = _conditional(cov, order, primary)

    rng = np.random.default_rng(7)
    n = max(1000, min(int(n), 20000))
    k = len(order)
    draws = np.zeros((n, k))
    m_s = np.array([primary[order[i]] for i in idx_s], dtype=float)
    if idx_s:
        sd_s = np.sqrt(np.clip(np.diag(cov)[idx_s], 1e-12, None))
        noise_s = rng.standard_normal((n, len(idx_s))) * np.maximum(uncertainty * np.abs(m_s), 0.25 * sd_s)
        s_draw = m_s + noise_s
        draws[:, idx_s] = s_draw
        base_o = s_draw @ B.T
    else:
        base_o = np.zeros((n, len(idx_o)))
    if idx_o:
        w, v = np.linalg.eigh(cond * h)
        L = v @ np.diag(np.sqrt(np.clip(w, 0, None)))
        draws[:, idx_o] = base_o + rng.standard_normal((n, len(idx_o))) @ L.T

    betas = fm["betas"]
    if fm.get("asset_is_factor"):
        j = order.index(t)
        asset = draws[:, j].copy()
        if _FACTOR_UNIT[t] != "pct":
            raise ValueError(f"{t} is a level/yield factor; its scenario move is shown in the factor table.")
        beta_vec = np.array([1.0 if f == t else 0.0 for f in order])
    else:
        beta_vec = np.array([betas.get(f, {}).get("beta") or 0.0 for f in order], dtype=float)
        resid_sd = float(fm.get("resid_weekly_sd") or 0.0) * math.sqrt(h)
        asset = draws @ beta_vec + rng.standard_normal(n) * resid_sd

    exp_f = draws.mean(0)
    contrib = {order[i]: _fnum(beta_vec[i] * exp_f[i], 3) for i in range(k)}
    pct = np.percentile(asset, [5, 25, 50, 75, 95])
    es5 = float(asset[asset <= pct[0]].mean()) if np.any(asset <= pct[0]) else float(pct[0])
    counts, edges = np.histogram(asset, bins=36)

    # Propagation network: trigger -> primary factors -> secondary factors -> asset.
    sds_h = np.sqrt(np.clip(np.diag(cov) * h, 1e-12, None))
    nodes = [{"id": "trigger", "label": sc["name"], "kind": "trigger"}]
    edges_out = []
    for i in idx_s:
        f = order[i]
        nodes.append({"id": f, "label": _FACTOR_LABEL[f], "kind": "primary",
                      "move": _fnum(exp_f[i], 2), "move_text": _fmt_move(f, exp_f[i]), "z": _fnum(exp_f[i] / sds_h[i], 2)})
        edges_out.append({"from": "trigger", "to": f, "value": _fnum(primary[f], 2), "label": _fmt_move(f, primary[f])})
    for jj, i in enumerate(idx_o):
        f = order[i]
        nodes.append({"id": f, "label": _FACTOR_LABEL[f], "kind": "secondary",
                      "move": _fnum(exp_f[i], 2), "move_text": _fmt_move(f, exp_f[i]), "z": _fnum(exp_f[i] / sds_h[i], 2)})
        for ii, si in enumerate(idx_s):
            v = float(B[jj, ii] * m_s[ii])
            if abs(v) / sds_h[i] >= 0.05:
                edges_out.append({"from": order[si], "to": f, "value": _fnum(v, 2), "label": _fmt_move(f, v)})
    nodes.append({"id": "asset", "label": t, "kind": "asset", "move": _fnum(asset.mean(), 2),
                  "move_text": f"{asset.mean():+.2f}%"})
    for i in range(k):
        c = beta_vec[i] * exp_f[i]
        if abs(c) >= 0.02:
            edges_out.append({"from": order[i], "to": "asset", "value": _fnum(c, 3), "label": f"{c:+.2f}%"})

    # Narrative.
    prim_txt = ", ".join(f"{_FACTOR_LABEL[f]} {_fmt_move(f, v)}" for f, v in primary.items()) or "no factor fixed (pure historical risk)"
    sec = sorted(((order[i], exp_f[i] / sds_h[i], exp_f[i]) for i in idx_o), key=lambda x: abs(x[1]), reverse=True)[:3]
    sec_txt = ", ".join(f"{_FACTOR_LABEL[f]} {_fmt_move(f, v)}" for f, _, v in sec if abs(_) >= 0.15)
    top_c = sorted(((f, v) for f, v in contrib.items() if v), key=lambda x: abs(x[1]), reverse=True)[:3]
    p_loss = float((asset < 0).mean() * 100)
    narrative = [f"Assumptions: {prim_txt} (each shock drawn with \u00b1{int(uncertainty * 100)}% uncertainty), "
                 f"over a {h}-week horizon, using the {'stress-regime' if regime == 'stress' else 'full-sample'} "
                 f"co-movement of the six factors ({fm['stress_weeks'] if regime == 'stress' else fm['weeks']} weeks of data)."]
    if sec_txt:
        narrative.append(f"Spillovers: given those moves, history implies {sec_txt} on average for the factors not fixed by the scenario.")
    narrative.append(f"Result for {t}: expected {asset.mean():+.2f}% (median {pct[2]:+.2f}%), 90% range {pct[0]:+.2f}% to "
                     f"{pct[4]:+.2f}%, a {p_loss:.0f}% chance of a loss and an average loss of {es5:.2f}% in the worst 5% of paths.")
    if top_c:
        narrative.append("Main channels: " + ", ".join(f"{_FACTOR_LABEL.get(f, f)} {v:+.2f}%" for f, v in top_c) + ".")
    if not fm.get("asset_is_factor"):
        r2 = fm.get("r2") or 0
        if r2 < 0.2:
            narrative.append(f"Caution: these factors explain only {round(r2 * 100)}% of {t}'s weekly moves, so stock-specific "
                             "news dominates and the range above is mostly idiosyncratic noise around a small macro effect.")
        else:
            narrative.append(f"Model fit: the factors explain {round(r2 * 100)}% of {t}'s weekly moves over {fm['weeks']} weeks.")
    narrative.append("Shock sizes are illustrative, loosely informed by past episodes; real crises can break historical "
                     "correlations. This is a stress test, not a forecast.")

    factor_table = []
    for i, f in enumerate(order):
        b = betas.get(f, {})
        factor_table.append({"factor": f, "label": _FACTOR_LABEL[f], "unit": _FACTOR_UNIT[f],
                             "fixed": f in primary, "expected_move": _fnum(exp_f[i], 2),
                             "move_text": _fmt_move(f, exp_f[i]), "beta": b.get("beta"), "t": b.get("t"),
                             "contribution": contrib[f]})
    return {"ticker": t, "scenario": {k2: sc.get(k2) for k2 in ("key", "name", "region", "chain", "analogues")},
            "regime": regime, "horizon_weeks": h, "n": n,
            "stats": {"mean": _fnum(asset.mean(), 3), "p5": _fnum(pct[0], 3), "p25": _fnum(pct[1], 3), "median": _fnum(pct[2], 3),
                      "p75": _fnum(pct[3], 3), "p95": _fnum(pct[4], 3), "prob_loss": _fnum(p_loss, 1), "es5": _fnum(es5, 3),
                      "sd": _fnum(asset.std(), 3)},
            "histogram": {"counts": counts.tolist(), "edges": [round(float(e), 3) for e in edges]},
            "factor_table": factor_table, "contributions": contrib,
            "network": {"nodes": nodes, "edges": edges_out}, "narrative": narrative,
            "model": {"r2": fm.get("r2"), "weeks": fm.get("weeks"), "start": fm.get("start"), "end": fm.get("end"),
                      "asset_is_factor": fm.get("asset_is_factor")}}


def stress_matrix(ticker: str, regime: str = "stress") -> list[dict]:
    """Expected move of the asset in every scenario (analytic, fast)."""
    fm = factor_model(ticker)
    if not fm.get("ok"):
        return []
    order = fm["factor_order"]
    cov = np.array(fm["cov_stress"] if regime == "stress" else fm["cov_normal"], dtype=float)
    t = ticker.strip().upper()
    if fm.get("asset_is_factor"):
        beta_vec = np.array([1.0 if f == t else 0.0 for f in order])
    else:
        beta_vec = np.array([fm["betas"].get(f, {}).get("beta") or 0.0 for f in order], dtype=float)
    out = []
    for sc in SCENARIOS:
        primary = {f: v for f, v in sc["shocks"].items() if f in order}
        idx_s, idx_o, B, _ = _conditional(cov, order, primary)
        mean = np.zeros(len(order))
        m_s = np.array([primary[order[i]] for i in idx_s])
        mean[idx_s] = m_s
        if idx_s and idx_o:
            mean[idx_o] = B @ m_s
        out.append({"key": sc["key"], "name": sc["name"], "region": sc["region"],
                    "expected_move": _fnum(float(beta_vec @ mean), 2),
                    "factor_moves": {order[i]: _fmt_move(order[i], mean[i]) for i in range(len(order))}})
    return out


# =============================================================================
# 5. Indicator Lab
# =============================================================================
def _sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def _ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _atr(h, l, c, n):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return _wilder(tr, n)


def _rsi(c, n):
    d = c.diff()
    up, dn = _wilder(d.clip(lower=0), n), _wilder(-d.clip(upper=0), n)
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).where(dn != 0, 100.0)


def _supertrend(h, l, c, n, mult):
    atr = _atr(h, l, c, n).to_numpy()
    hl2 = ((h + l) / 2).to_numpy()
    up_b, lo_b = hl2 + mult * atr, hl2 - mult * atr
    cl = c.to_numpy()
    st = np.full(len(cl), np.nan)
    dirn = np.zeros(len(cl))
    fu, fl = up_b.copy(), lo_b.copy()
    for i in range(1, len(cl)):
        if np.isnan(atr[i]):
            continue
        fu[i] = up_b[i] if (np.isnan(fu[i - 1]) or up_b[i] < fu[i - 1] or cl[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lo_b[i] if (np.isnan(fl[i - 1]) or lo_b[i] > fl[i - 1] or cl[i - 1] < fl[i - 1]) else fl[i - 1]
        prev = dirn[i - 1] if dirn[i - 1] != 0 else 1
        if prev == 1 and cl[i] < fl[i]:
            dirn[i] = -1
        elif prev == -1 and cl[i] > fu[i]:
            dirn[i] = 1
        else:
            dirn[i] = prev
        st[i] = fl[i] if dirn[i] == 1 else fu[i]
    return pd.Series(st, index=c.index), pd.Series(dirn, index=c.index)


def _psar(h, l, step, mx):
    hi, lo = h.to_numpy(), l.to_numpy()
    n = len(hi)
    sar = np.full(n, np.nan)
    if n < 3:
        return pd.Series(sar, index=h.index)
    bull, af, ep = True, step, hi[0]
    sar[0] = lo[0]
    for i in range(1, n):
        prev = sar[i - 1]
        cur = prev + af * (ep - prev)
        if bull:
            cur = min(cur, lo[i - 1], lo[i - 2] if i > 1 else lo[i - 1])
            if lo[i] < cur:
                bull, cur, ep, af = False, ep, lo[i], step
            elif hi[i] > ep:
                ep, af = hi[i], min(af + step, mx)
        else:
            cur = max(cur, hi[i - 1], hi[i - 2] if i > 1 else hi[i - 1])
            if hi[i] > cur:
                bull, cur, ep, af = True, ep, hi[i], step
            elif lo[i] < ep:
                ep, af = lo[i], min(af + step, mx)
        sar[i] = cur
    return pd.Series(sar, index=h.index)


def _adx(h, l, c, n):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    atr = _atr(h, l, c, n)
    pdi = 100 * _wilder(pdm, n) / atr
    mdi = 100 * _wilder(mdm, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _wilder(dx, n), pdi, mdi


INDICATOR_DEFAULTS = {"sma": "50,200", "ema": "20", "bb_n": 20, "bb_k": 2.0, "kc_n": 20, "kc_mult": 2.0,
                      "dc_n": 20, "st_n": 10, "st_mult": 3.0, "psar_step": 0.02, "psar_max": 0.2, "vwap_n": 20,
                      "rsi_n": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "stoch_k": 14, "stoch_d": 3,
                      "adx_n": 14, "cci_n": 20, "wr_n": 14, "mfi_n": 14, "atr_n": 14, "fwd": 10}


def _parse_periods(s: str, lo=2, hi=400) -> list[int]:
    out = []
    for p in str(s or "").split(","):
        try:
            v = int(float(p))
        except ValueError:
            continue
        if lo <= v <= hi and v not in out:
            out.append(v)
    return out[:4]


def compute_indicators(df: pd.DataFrame, p: dict) -> tuple[dict, dict, pd.DataFrame]:
    c, h, l = df["Close"], df["High"], df["Low"]
    v = df["Volume"].fillna(0) if "Volume" in df else pd.Series(0.0, index=df.index)
    ov, osc = {}, {}
    for n in _parse_periods(p["sma"]):
        ov[f"SMA {n}"] = _sma(c, n)
    for n in _parse_periods(p["ema"]):
        ov[f"EMA {n}"] = _ema(c, n)
    mid = _sma(c, p["bb_n"])
    sd = c.rolling(p["bb_n"], min_periods=p["bb_n"]).std(ddof=0)
    ov["BB upper"], ov["BB middle"], ov["BB lower"] = mid + p["bb_k"] * sd, mid, mid - p["bb_k"] * sd
    kmid = _ema(c, p["kc_n"])
    katr = _atr(h, l, c, p["kc_n"])
    ov["Keltner upper"], ov["Keltner lower"] = kmid + p["kc_mult"] * katr, kmid - p["kc_mult"] * katr
    ov["Donchian upper"] = h.rolling(p["dc_n"], min_periods=p["dc_n"]).max()
    ov["Donchian lower"] = l.rolling(p["dc_n"], min_periods=p["dc_n"]).min()
    st, st_dir = _supertrend(h, l, c, p["st_n"], p["st_mult"])
    ov["Supertrend"] = st
    ov["Parabolic SAR"] = _psar(h, l, p["psar_step"], p["psar_max"])
    tp = (h + l + c) / 3
    ov["VWAP (rolling)"] = (tp * v).rolling(p["vwap_n"]).sum() / v.rolling(p["vwap_n"]).sum().replace(0, np.nan)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    ov["Ichimoku tenkan"], ov["Ichimoku kijun"] = tenkan, kijun
    ov["Ichimoku span A"], ov["Ichimoku span B"] = span_a, span_b

    osc["RSI"] = _rsi(c, p["rsi_n"])
    macd = _ema(c, p["macd_fast"]) - _ema(c, p["macd_slow"])
    sig = macd.ewm(span=p["macd_signal"], adjust=False).mean()
    osc["MACD"], osc["MACD signal"], osc["MACD histogram"] = macd, sig, macd - sig
    ll, hh = l.rolling(p["stoch_k"]).min(), h.rolling(p["stoch_k"]).max()
    k_raw = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    k = k_raw.rolling(3).mean()
    osc["Stochastic %K"], osc["Stochastic %D"] = k, k.rolling(p["stoch_d"]).mean()
    adx, pdi, mdi = _adx(h, l, c, p["adx_n"])
    osc["ADX"], osc["+DI"], osc["-DI"] = adx, pdi, mdi
    tp_sma = tp.rolling(p["cci_n"]).mean()
    mad = tp.rolling(p["cci_n"]).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    osc["CCI"] = (tp - tp_sma) / (0.015 * mad.replace(0, np.nan))
    whh, wll = h.rolling(p["wr_n"]).max(), l.rolling(p["wr_n"]).min()
    osc["Williams %R"] = -100 * (whh - c) / (whh - wll).replace(0, np.nan)
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(p["mfi_n"]).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(p["mfi_n"]).sum()
    osc["MFI"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    osc["OBV"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    osc["ATR"] = _atr(h, l, c, p["atr_n"])
    osc["ATR %"] = osc["ATR"] / c * 100

    # Signal series (+1 bullish, -1 bearish, 0 neutral) for every bar.
    sma_long = ov.get(f"SMA {max(_parse_periods(p['sma']) or [50])}", _sma(c, 50))
    sigs = pd.DataFrame(index=c.index)
    sigs["Price vs long SMA"] = np.sign(c - sma_long).fillna(0)
    r = osc["RSI"]
    sigs["RSI"] = np.where((r < 30) & (r > r.shift()), 1, np.where((r > 70) & (r < r.shift()), -1, 0))
    sigs["MACD"] = np.sign(macd - sig).fillna(0)
    kk, dd = osc["Stochastic %K"], osc["Stochastic %D"]
    sigs["Stochastic"] = np.where((kk < 20) & (kk > dd), 1, np.where((kk > 80) & (kk < dd), -1, 0))
    sigs["ADX / DMI"] = np.where(adx > 25, np.sign(pdi - mdi), 0)
    sigs["CCI"] = np.where(osc["CCI"] < -100, 1, np.where(osc["CCI"] > 100, -1, 0))
    sigs["Williams %R"] = np.where(osc["Williams %R"] < -80, 1, np.where(osc["Williams %R"] > -20, -1, 0))
    sigs["MFI"] = np.where(osc["MFI"] < 20, 1, np.where(osc["MFI"] > 80, -1, 0))
    sigs["Bollinger Bands"] = np.where(c < ov["BB lower"], 1, np.where(c > ov["BB upper"], -1, 0))
    sigs["Supertrend"] = st_dir
    sigs["Parabolic SAR"] = np.sign(c - ov["Parabolic SAR"]).fillna(0)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    sigs["Ichimoku"] = np.where((c > cloud_top) & (tenkan > kijun), 1, np.where((c < cloud_bot) & (tenkan < kijun), -1, 0))
    obv = osc["OBV"]
    sigs["OBV trend"] = np.sign(obv - obv.shift(20)).fillna(0)
    sigs["Donchian breakout"] = np.where(c >= ov["Donchian upper"].shift(), 1, np.where(c <= ov["Donchian lower"].shift(), -1, 0))
    sigs["EMA vs SMA cross"] = np.sign(_ema(c, 20) - _sma(c, 50)).fillna(0)
    return ov, osc, sigs.fillna(0)


_SIGNAL_REASON = {
    "Price vs long SMA": ("Price above its long moving average (uptrend)", "Price below its long moving average (downtrend)", ""),
    "RSI": ("RSI below 30 and turning up (oversold bounce)", "RSI above 70 and turning down (overbought)", "RSI between 30 and 70"),
    "MACD": ("MACD above its signal line", "MACD below its signal line", ""),
    "Stochastic": ("%K below 20 and crossing above %D", "%K above 80 and crossing below %D", "No extreme crossover"),
    "ADX / DMI": ("Strong trend (ADX>25) with +DI above -DI", "Strong trend (ADX>25) with -DI above +DI", "ADX below 25: no strong trend"),
    "CCI": ("CCI below -100 (oversold)", "CCI above +100 (overbought)", "CCI within \u00b1100"),
    "Williams %R": ("%R below -80 (oversold)", "%R above -20 (overbought)", "%R in mid-range"),
    "MFI": ("Money flow below 20 (oversold)", "Money flow above 80 (overbought)", "Money flow in mid-range"),
    "Bollinger Bands": ("Close below the lower band (stretched down)", "Close above the upper band (stretched up)", "Inside the bands"),
    "Supertrend": ("Supertrend in up mode", "Supertrend in down mode", ""),
    "Parabolic SAR": ("SAR dots below price", "SAR dots above price", ""),
    "Ichimoku": ("Above the cloud with tenkan over kijun", "Below the cloud with tenkan under kijun", "Inside the cloud or mixed lines"),
    "OBV trend": ("On-balance volume rising over 20 bars (accumulation)", "On-balance volume falling (distribution)", "Flat"),
    "Donchian breakout": ("Close at a new 20-bar high (breakout)", "Close at a new 20-bar low (breakdown)", "Inside the channel"),
    "EMA vs SMA cross": ("EMA 20 above SMA 50", "EMA 20 below SMA 50", ""),
}


def indicator_lab(ticker: str, period: str = "1y", interval: str = "1d", params: dict | None = None) -> dict:
    t = ticker.strip().upper()
    p = dict(INDICATOR_DEFAULTS)
    for k2, v in (params or {}).items():
        if v is None or k2 not in p:
            continue
        if isinstance(p[k2], str):
            p[k2] = str(v)
        elif isinstance(p[k2], int):
            p[k2] = int(max(2, min(int(float(v)), 400)))
        else:
            p[k2] = float(max(0.001, min(float(v), 10.0)))
    if interval == "1d":
        base = core._ticker(t).history(period="5y")
    elif interval == "1wk":
        base = core._ticker(t).history(period="10y", interval="1wk")
    elif interval == "1h":
        base = core._ticker(t).history(period="730d", interval="1h")
    else:
        raise ValueError("interval must be 1d, 1wk or 1h")
    if base is None or base.empty or len(base) < 60:
        raise ValueError(f"Not enough price history for {t}.")
    df = base[["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in base else [])].astype(float).dropna(subset=["Close"])
    ov, osc, sigs = compute_indicators(df, p)

    # Reliability of each signal on THIS asset: forward return after bullish / bearish readings.
    fwd_n = int(p["fwd"])
    fwd = (df["Close"].shift(-fwd_n) / df["Close"] - 1) * 100
    valid = fwd.notna()
    base_avg = float(fwd[valid].mean())
    base_hit = float((fwd[valid] > 0).mean() * 100)
    signals = []
    for name in sigs.columns:
        cur = int(sigs[name].iloc[-1])
        rel = {}
        for lab, val in (("bullish", 1), ("bearish", -1)):
            mask = valid & (sigs[name] == val)
            nobs = int(mask.sum())
            if nobs >= 15:
                fr = fwd[mask]
                rel[lab] = {"n": nobs, "avg_fwd": _fnum(fr.mean(), 2),
                            "hit": _fnum(((fr > 0) if val == 1 else (fr < 0)).mean() * 100, 0),
                            "edge": _fnum((fr.mean() - base_avg) * val, 2)}
        reasons = _SIGNAL_REASON.get(name, ("Bullish", "Bearish", "Neutral"))
        signals.append({"indicator": name, "signal": "BUY" if cur > 0 else ("SELL" if cur < 0 else "NEUTRAL"),
                        "reason": reasons[0] if cur > 0 else (reasons[1] if cur < 0 else (reasons[2] or "Neutral")),
                        "reliability": rel})
    buy = sum(1 for s in signals if s["signal"] == "BUY")
    sell = sum(1 for s in signals if s["signal"] == "SELL")
    neutral = len(signals) - buy - sell
    score = (buy - sell) / max(len(signals), 1)
    label = ("Strong bullish" if score >= 0.5 else "Bullish" if score >= 0.15 else
             "Strong bearish" if score <= -0.5 else "Bearish" if score <= -0.15 else "Neutral")

    window = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260, "max": 100000}.get(period, 252)
    if interval == "1wk":
        window = {"1y": 52, "2y": 104, "5y": 260, "max": 100000}.get(period, 260)
    if interval == "1h":
        window = {"1mo": 140, "3mo": 420, "6mo": 840, "1y": 1700}.get(period, 420)
    sl = slice(-window, None)
    idx = df.index[sl]
    fmt = "%Y-%m-%d %H:%M" if interval == "1h" else "%Y-%m-%d"
    nd = 2 if df["Close"].iloc[-1] >= 10 else 5
    return {
        "ticker": t, "interval": interval, "period": period, "params": p,
        "dates": pd.to_datetime(idx).strftime(fmt).tolist(),
        "ohlc": {col.lower(): _series_list(df[col].iloc[sl], nd) for col in df.columns},
        "overlays": {k2: _series_list(s.iloc[sl], nd) for k2, s in ov.items()},
        "oscillators": {k2: _series_list(s.iloc[sl], 3) for k2, s in osc.items()},
        "signals": signals,
        "summary": {"buy": buy, "sell": sell, "neutral": neutral, "score": round(score, 3), "label": label},
        "baseline": {"forward_bars": fwd_n, "avg_fwd": _fnum(base_avg, 2), "hit_up": _fnum(base_hit, 0),
                     "bars": int(valid.sum())},
        "note": ("Reliability compares the average forward return after each signal with the asset's unconditional "
                 f"average over the same {fwd_n} bars (\"edge\"). Overlapping windows inflate sample sizes, and past "
                 "behaviour need not repeat."),
    }


# =============================================================================
# API routes
# =============================================================================
def _guard(fn):
    try:
        return fn()
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        core.log.exception("intel endpoint failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@router.get("/api/calendar")
def api_calendar(ticker: str | None = Query(default=None), days: int = Query(21, ge=1, le=120),
                 importance: str = Query("MEDIUM"), region: str = Query("all")):
    t = ticker.strip().upper() if ticker else None
    key = f"intel::calendar_view::{t}::{days}::{importance.upper()}::{region}"
    return _guard(lambda: core._cached_call(key, lambda: calendar_payload(t, days, importance, region),
                                            heavy=False, ttl=1800, cache_if=lambda v: bool(v.get("events"))))


@router.get("/api/calendar.ics")
def api_calendar_ics(days: int = Query(60, ge=1, le=180), importance: str = Query("HIGH")):
    body = _guard(lambda: calendar_ics(days, importance))
    return Response(body, media_type="text/calendar",
                    headers={"Content-Disposition": 'attachment; filename="finmodel-macro-calendar.ics"'})


@router.get("/api/intel/{ticker}")
def api_intel(ticker: str):
    t = ticker.strip().upper()
    return _guard(lambda: core._cached_call(f"intel::news::{t}", lambda: news_intelligence(t), heavy=False,
                                            ttl=900, cache_if=lambda v: v.get("story_count", 0) > 0))


@router.get("/api/factors/{ticker}")
def api_factors(ticker: str):
    return _guard(lambda: _public_factor_model(factor_model(ticker)))


@router.get("/api/geo/scenarios")
def api_geo_scenarios(ticker: str | None = Query(default=None), regime: str = Query("stress")):
    def build():
        t = ticker.strip().upper() if ticker else None
        rel = []
        try:
            stories = []
            for q in (_GEO_QUERY, _MACRO_QUERY):
                for r in _google_news(q):
                    stories.append({"title": r["title"], "weight": 1.0})
            rel = scenario_relevance(stories)
        except Exception:
            rel = []
        rel_map = {r["key"]: r for r in rel}
        matrix = {m["key"]: m for m in (stress_matrix(t, regime) if t else [])}
        return {"ticker": t, "regime": regime, "scenarios": [
            {"key": s["key"], "name": s["name"], "region": s["region"], "chain": s["chain"], "analogues": s["analogues"],
             "shocks": {f: _fmt_move(f, v) for f, v in s["shocks"].items()},
             "shock_values": s["shocks"],
             "relevance_pct": rel_map.get(s["key"], {}).get("relevance_pct", 0),
             "headline_hits": rel_map.get(s["key"], {}).get("hits", 0),
             "expected_move": matrix.get(s["key"], {}).get("expected_move")} for s in SCENARIOS],
            "factors": [{"symbol": f, "label": l, "unit": u} for f, l, u in FACTORS]}
    key = f"intel::geo::{ticker}::{regime}"
    return _guard(lambda: core._cached_call(key, build, heavy=False, ttl=1800))


@router.get("/api/geo/simulate/{ticker}")
def api_geo_simulate(ticker: str, scenario: str = Query("hormuz"), horizon_weeks: int = Query(4, ge=1, le=26),
                     regime: str = Query("stress"), shocks: str | None = Query(default=None),
                     uncertainty: float = Query(0.35, ge=0.0, le=1.0)):
    parsed = None
    if shocks:
        parsed = {}
        for part in shocks.split(","):
            if ":" not in part:
                continue
            f, v = part.rsplit(":", 1)
            f = f.strip().upper()
            try:
                val = float(v)
            except ValueError:
                continue
            if f in _FACTOR_LABEL and abs(val) <= 500:
                parsed[f] = val
    reg = "stress" if regime == "stress" else "normal"
    return _guard(lambda: simulate(ticker, scenario=scenario, shocks=parsed, horizon_weeks=horizon_weeks,
                                   regime=reg, uncertainty=uncertainty))


@router.get("/api/indicators/{ticker}")
def api_indicators(ticker: str, period: str = Query("1y"), interval: str = Query("1d"),
                   sma: str | None = None, ema: str | None = None, bb_n: int | None = None, bb_k: float | None = None,
                   kc_n: int | None = None, kc_mult: float | None = None, dc_n: int | None = None,
                   st_n: int | None = None, st_mult: float | None = None, psar_step: float | None = None,
                   psar_max: float | None = None, vwap_n: int | None = None, rsi_n: int | None = None,
                   macd_fast: int | None = None, macd_slow: int | None = None, macd_signal: int | None = None,
                   stoch_k: int | None = None, stoch_d: int | None = None, adx_n: int | None = None,
                   cci_n: int | None = None, wr_n: int | None = None, mfi_n: int | None = None,
                   atr_n: int | None = None, fwd: int | None = None):
    params = {k2: v for k2, v in locals().items() if k2 in INDICATOR_DEFAULTS}
    key = "intel::ind::" + ticker.upper() + "::" + period + "::" + interval + "::" + repr(sorted((k2, v) for k2, v in params.items() if v is not None))
    return _guard(lambda: core._cached_call(key, lambda: indicator_lab(ticker, period, interval, params),
                                            heavy=False, ttl=900))
