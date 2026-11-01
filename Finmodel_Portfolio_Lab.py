import os
import time
import threading
import logging
import hashlib
import hmac
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, Future, wait
from contextlib import contextmanager
from functools import wraps
import io
import json
import zipfile
import shutil
import html as _html
import re
from pathlib import Path as _Path
from urllib.parse import quote as _url_quote
import requests
from datetime import date, datetime as _dt, timezone, timedelta
import uvicorn
import yfinance as yf
import pandas as pd
import numpy as np
try:
    from scipy.optimize import minimize as _scipy_minimize
except Exception:
    _scipy_minimize = None
from xgboost import XGBRegressor
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

# ------------------------------------------------------------------------------
# 1. APPLICATION SETUP & SERVICE METADATA
# ------------------------------------------------------------------------------
app = FastAPI(
    title="finmodel",
    description="High-Performance Quantitative Financial Intelligence & Risk Terminal",
    version="17.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# 1a. PERFORMANCE & PUBLIC-HOSTING INFRASTRUCTURE
# ------------------------------------------------------------------------------
# Everything that makes the site fast and safe to share lives here:
#   * gzip for every response (JSON payloads shrink ~65%)
#   * a shared MARKET-DATA LAYER: each Yahoo dataset (price history, company
#     info, statements, news) is fetched once, de-duplicated across requests
#     and users, and kept for a TTL that matches how often it changes
#   * parallel prefetch of independent network calls
#   * RESULT CACHE with request coalescing ("single flight") and
#     stale-while-revalidate, so popular tickers answer instantly
#   * a cap on simultaneous heavy computations (protects small instances)
#   * per-stage timings in logs, responses and the Server-Timing header
#   * optional disk persistence (CACHE_DIR) and periodic pre-warming
log = logging.getLogger("finmodel")
if not logging.getLogger().handlers:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app.add_middleware(GZipMiddleware, minimum_size=1000)

_STATS: Counter = Counter()
_TL = threading.local()

_IO_POOL = ThreadPoolExecutor(max_workers=max(2, int(os.environ.get("IO_WORKERS", "8"))),
                              thread_name_prefix="fm-io")
_BG_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fm-bg")

# --- request validation + Server-Timing ---------------------------------------
_SAFE_PATH = re.compile(r"^[A-Za-z0-9/._\-=^%]*$")


@app.middleware("http")
async def _public_guards(request: Request, call_next):
    path = request.url.path
    # Tickers arrive as path segments; reject anything that is not ticker-like.
    if path.startswith("/api/") and (not _SAFE_PATH.match(path) or len(path) > 200):
        return JSONResponse({"detail": "Invalid ticker or path."}, status_code=400)
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["Server-Timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.0f}"
    return response


# --- heavy-job slots -----------------------------------------------------------
# Every analysis downloads years of data and fits a model. On a small instance a
# few simultaneous fresh analyses could exhaust RAM, so at most MAX_HEAVY_JOBS
# run at once. Cache hits never wait for a slot.
_MAX_HEAVY_JOBS = max(1, int(os.environ.get("MAX_HEAVY_JOBS", "2")))
_HEAVY_JOB_WAIT_SECONDS = float(os.environ.get("HEAVY_JOB_WAIT_SECONDS", "120"))
_HEAVY_SLOTS = threading.BoundedSemaphore(_MAX_HEAVY_JOBS)


class BusyError(HTTPException):
    def __init__(self):
        super().__init__(status_code=503,
                         detail="Server is busy with other analyses. Please retry in a minute.",
                         headers={"Retry-After": "30"})


def _run_heavy(fn):
    if getattr(_TL, "holding_slot", False):          # re-entrant (nested heavy calls)
        return fn()
    if not _HEAVY_SLOTS.acquire(timeout=_HEAVY_JOB_WAIT_SECONDS):
        _STATS["busy_rejections"] += 1
        raise BusyError()
    _TL.holding_slot = True
    try:
        return fn()
    finally:
        _TL.holding_slot = False
        _HEAVY_SLOTS.release()


# --- per-stage timings -----------------------------------------------------------
@contextmanager
def _stage(name: str):
    rec = getattr(_TL, "timings", None)
    started = time.perf_counter()
    try:
        yield
    finally:
        if rec is not None:
            rec[name] = rec.get(name, 0) + round((time.perf_counter() - started) * 1000)


def _timed(name: str):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with _stage(name):
                return fn(*args, **kwargs)
        return wrapper
    return deco


@contextmanager
def _timing_session():
    previous = getattr(_TL, "timings", None)
    _TL.timings = {}
    started = time.perf_counter()
    try:
        yield _TL.timings
    finally:
        _TL.timings["total"] = round((time.perf_counter() - started) * 1000)
        _TL.timings = previous


def _prefetch(callables) -> None:
    """Run independent network loads in parallel; results land in the data layer."""
    futures = [_IO_POOL.submit(fn) for fn in callables]
    wait(futures, timeout=float(os.environ.get("PREFETCH_TIMEOUT_SECONDS", "45")))
    for f in futures:
        if f.done() and f.exception() is not None:
            log.debug("prefetch failed: %s", f.exception())


# --- shared market-data layer -------------------------------------------------------
_DATA_TTL = {
    "history": int(os.environ.get("PRICE_TTL_SECONDS", "900")),       # prices: 15 min
    "quote": int(os.environ.get("QUOTE_TTL_SECONDS", "120")),          # last price: 2 min
    "info": int(os.environ.get("INFO_TTL_SECONDS", "21600")),          # company info: 6 h
    "statements": int(os.environ.get("STATEMENT_TTL_SECONDS", "86400")),  # annual statements: 24 h
    "news": 1800, "geo": 1800,                                          # headlines: 30 min
    "rf": 43200,                                                        # risk-free rate: 12 h
}
_EMPTY_RETRY_SECONDS = 60          # empty/failed answers are retried after a minute
_DATA_MAX_ENTRIES = int(os.environ.get("DATA_CACHE_MAX_ENTRIES", "800"))
_PERSISTED_KINDS = {"history", "info", "statements"}
_CACHE_DIR = os.environ.get("CACHE_DIR", "").strip()
_DATA: dict[tuple, tuple[float, object]] = {}
_DATA_GUARD = threading.Lock()
_DATA_LOCKS = [threading.Lock() for _ in range(64)]   # striped per-key locks


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, pd.DataFrame):
        return value.empty
    if isinstance(value, (dict, list, tuple)):
        return len(value) == 0
    return False


def _copy(value):
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _disk_path(key: tuple) -> str:
    return os.path.join(_CACHE_DIR, hashlib.sha1(repr(key).encode()).hexdigest() + ".pkl")


def _disk_load(key: tuple, ttl: int):
    if not _CACHE_DIR or key[0] not in _PERSISTED_KINDS:
        return None
    path = _disk_path(key)
    try:
        if time.time() - os.path.getmtime(path) > ttl:
            return None
        with open(path, "rb") as fh:            # files written only by this app
            return pickle.load(fh)
    except Exception:
        return None


def _disk_save(key: tuple, value) -> None:
    if not _CACHE_DIR or key[0] not in _PERSISTED_KINDS or _is_empty(value):
        return
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = _disk_path(key)
        tmp = f"{path}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception as exc:
        log.debug("disk cache write failed: %s", exc)


def _data_store(key: tuple, value, fetched_at: float | None = None) -> None:
    ttl = _DATA_TTL[key[0]]
    stamp = time.time() if fetched_at is None else fetched_at
    if _is_empty(value):
        stamp = time.time() - ttl + _EMPTY_RETRY_SECONDS
    with _DATA_GUARD:
        _DATA[key] = (stamp, value)
        if len(_DATA) > _DATA_MAX_ENTRIES:
            now = time.time()
            for k in [k for k, (t, _) in _DATA.items() if now - t > _DATA_TTL[k[0]]]:
                _DATA.pop(k, None)
            while len(_DATA) > _DATA_MAX_ENTRIES:
                _DATA.pop(min(_DATA, key=lambda k: _DATA[k][0]), None)


_MISS = object()


def _data_lookup(key: tuple):
    """Fresh cached value, or _MISS. (None is a valid cached answer, e.g. no quote.)"""
    entry = _DATA.get(key)
    if entry is not None and time.time() - entry[0] <= _DATA_TTL[key[0]]:
        return entry[1]
    return _MISS


def _data_peek(key: tuple):
    value = _data_lookup(key)
    return None if value is _MISS else value


def _data_get(kind: str, key_parts: tuple, loader):
    key = (kind,) + tuple(key_parts)
    value = _data_lookup(key)
    if value is not _MISS:
        _STATS["data_hits"] += 1
        return value
    with _DATA_LOCKS[hash(key) % len(_DATA_LOCKS)]:   # one fetch per key, others wait
        value = _data_lookup(key)
        if value is not _MISS:
            _STATS["data_hits"] += 1
            return value
        value = _disk_load(key, _DATA_TTL[kind])
        if value is not None:
            _STATS["disk_hits"] += 1
            _data_store(key, value)
            return value
        _STATS["provider_calls"] += 1
        _STATS[f"provider_calls_{kind}"] += 1
        value = loader()
        _data_store(key, value)
        _disk_save(key, value)
        return value


_SLICE_OFFSETS = {"1d": 1, "5d": 5, "1mo": pd.DateOffset(months=1), "3mo": pd.DateOffset(months=3),
                  "6mo": pd.DateOffset(months=6), "1y": pd.DateOffset(years=1),
                  "2y": pd.DateOffset(years=2), "3y": pd.DateOffset(years=3)}


def _slice_history(base: pd.DataFrame, period: str) -> pd.DataFrame:
    offset = _SLICE_OFFSETS[period]
    if isinstance(offset, int):
        return base.tail(offset).copy()
    return base[base.index > base.index[-1] - offset].copy()


def _history_key(symbol: str, period: str, interval: str = "1d", auto_adjust: bool = True, extra: tuple = ()) -> tuple:
    return ("history", symbol, period, interval, bool(auto_adjust), extra)


class _CachedTicker:
    """Drop-in replacement for yf.Ticker backed by the shared data layer.

    Two analyses (or two visitors) asking for the same symbol share one download.
    Shorter daily histories are sliced from a fresh 5-year download instead of
    being fetched again.
    """

    def __init__(self, symbol, *args, **kwargs):
        self.ticker = str(symbol).strip().upper()
        self._raw_obj = None

    def _raw(self):
        if self._raw_obj is None:
            self._raw_obj = yf.Ticker(self.ticker)
        return self._raw_obj

    def history(self, period: str = "1mo", interval: str = "1d", auto_adjust: bool = True, **kwargs):
        period, interval = str(period), str(interval)
        extra = tuple(sorted(kwargs.items()))
        if interval == "1d" and not extra and period in _SLICE_OFFSETS:
            base = _data_peek(_history_key(self.ticker, "5y", "1d", auto_adjust))
            if base is not None and not base.empty:
                _STATS["data_hits"] += 1
                return _slice_history(base, period)
        loader = lambda: self._raw().history(period=period, interval=interval, auto_adjust=auto_adjust, **kwargs)
        return _copy(_data_get("history", _history_key(self.ticker, period, interval, auto_adjust, extra)[1:], loader))

    @property
    def info(self) -> dict:
        return _copy(_data_get("info", (self.ticker,), lambda: self._raw().info or {}))

    @property
    def fast_info(self) -> dict:
        def load():
            try:
                return self._raw().fast_info.get("last_price")
            except Exception:
                return None
        return {"last_price": _data_get("quote", (self.ticker,), load)}

    def _statement(self, name: str):
        return _copy(_data_get("statements", (self.ticker, name), lambda: getattr(self._raw(), name)))

    @property
    def financials(self):
        return self._statement("financials")

    @property
    def balance_sheet(self):
        return self._statement("balance_sheet")

    @property
    def cashflow(self):
        return self._statement("cashflow")

    @property
    def news(self) -> list:
        return _copy(_data_get("news", (self.ticker,), lambda: self._raw().news or []))


_ticker = _CachedTicker


def _batch_history(symbols: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
    """Daily history for many symbols with ONE provider call for whatever is not cached."""
    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for sym in symbols:
        base = _data_peek(_history_key(sym, "5y"))
        cached = _slice_history(base, period) if (base is not None and not base.empty and period in _SLICE_OFFSETS) \
            else _data_peek(_history_key(sym, period))
        if cached is not None and not cached.empty:
            _STATS["data_hits"] += 1
            out[sym] = cached.copy()
        else:
            missing.append(sym)
    if missing:
        try:
            _STATS["provider_calls"] += 1
            _STATS["provider_calls_batch"] += 1
            data = yf.download(missing, period=period, interval="1d", auto_adjust=True,
                               group_by="ticker", threads=True, progress=False)
            for sym in missing:
                try:
                    frame = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
                except Exception:
                    continue
                frame = frame.dropna(how="all")
                if not frame.empty:
                    _data_store(_history_key(sym, period), frame)
                    out[sym] = frame.copy()
        except Exception as exc:
            log.warning("batch download failed (%s); falling back to single requests", exc)
        for sym in [s for s in missing if s not in out]:
            try:
                frame = _ticker(sym).history(period=period, interval="1d", auto_adjust=True)
                if frame is not None and not frame.empty:
                    out[sym] = frame
            except Exception:
                continue
    return out


# --- live risk-free rate ---------------------------------------------------------
_RF_FALLBACK = 0.0425


def _risk_free_rate() -> float:
    """US 10-year Treasury yield from FRED (no API key), cached 12 h; 4.25% fallback."""
    override = os.environ.get("RISK_FREE_RATE", "").strip()
    if override:
        try:
            return float(override)
        except ValueError:
            pass

    def load():
        try:
            resp = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
                                timeout=6, headers={"User-Agent": "finmodel/1.0"})
            resp.raise_for_status()
        except Exception as exc:
            log.info("FRED unavailable (%s); using fallback risk-free rate", exc)
            return None                      # cached as "empty" -> retried after a minute, not on every call
        for line in reversed(resp.text.strip().splitlines()[1:]):
            try:
                value = float(line.split(",")[-1])
            except ValueError:
                continue                     # FRED marks holidays with "."
            if 0.0 < value < 20.0:
                return value / 100.0
        return None

    try:
        value = _data_get("rf", ("DGS10",), load)
    except Exception:
        value = None
    return float(value) if value else _RF_FALLBACK


# --- result cache: single flight + stale-while-revalidate ----------------------------
_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "900"))        # fresh for 15 min
_CACHE_STALE_SECONDS = int(os.environ.get("CACHE_STALE_SECONDS", "3600"))   # then served stale up to 1 h while refreshing
_CACHE_MAX_ENTRIES = int(os.environ.get("CACHE_MAX_ENTRIES", "150"))
_CACHE_LOCK = threading.Lock()
_INFLIGHT: dict[str, Future] = {}
_REFRESHING: set[str] = set()


def _cache_peek(key: str):
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
    if entry is None:
        return None
    return time.time() - entry[0], entry[1]


def _cache_get(key: str):
    """Fresh value or None (kept for simple callers)."""
    hit = _cache_peek(key)
    if hit is None or hit[0] > _CACHE_TTL_SECONDS:
        return None
    return hit[1]


def _cache_set(key: str, value) -> None:
    with _CACHE_LOCK:
        now = time.time()
        _CACHE[key] = (now, value)
        if len(_CACHE) > _CACHE_MAX_ENTRIES:
            limit = _CACHE_TTL_SECONDS + _CACHE_STALE_SECONDS
            for k in [k for k, (t, _) in _CACHE.items() if now - t > limit]:
                _CACHE.pop(k, None)
            while len(_CACHE) > _CACHE_MAX_ENTRIES:
                _CACHE.pop(min(_CACHE, key=lambda k: _CACHE[k][0]), None)


def _single_flight(key: str, fn):
    """If the same key is already being computed, wait for that result instead of recomputing."""
    with _CACHE_LOCK:
        fut = _INFLIGHT.get(key)
        owner = fut is None
        if owner:
            fut = Future()
            _INFLIGHT[key] = fut
    if not owner:
        _STATS["coalesced_requests"] += 1
        return fut.result()
    try:
        value = fn()
        fut.set_result(value)
        return value
    except BaseException as exc:
        fut.set_exception(exc)
        raise
    finally:
        with _CACHE_LOCK:
            _INFLIGHT.pop(key, None)


def _cached_call(key: str, fn, *, heavy: bool = True, cache_if=None, ttl: int | None = None):
    ttl = _CACHE_TTL_SECONDS if ttl is None else ttl

    def compute():
        value = _run_heavy(fn) if heavy else fn()
        if cache_if is None or cache_if(value):
            _cache_set(key, value)
        return value

    hit = _cache_peek(key)
    if hit is not None:
        age, value = hit
        if age <= ttl:
            _STATS["cache_fresh"] += 1
            return value
        if age <= ttl + _CACHE_STALE_SECONDS:
            _STATS["cache_stale_served"] += 1
            _refresh_in_background(key, compute)
            return value
    _STATS["cache_miss"] += 1
    return _single_flight(key, compute)


def _refresh_in_background(key: str, compute) -> None:
    with _CACHE_LOCK:
        if key in _REFRESHING or key in _INFLIGHT:
            return
        _REFRESHING.add(key)

    def job():
        try:
            _single_flight(key, compute)
        except Exception as exc:
            log.info("background refresh of %s failed: %s", key, exc)
        finally:
            with _CACHE_LOCK:
                _REFRESHING.discard(key)

    _BG_POOL.submit(job)


def _with_timings(name: str, fn):
    """Run fn inside a timing session, attach the stage timings and log them."""
    with _timing_session() as timings:
        result = fn()
    if isinstance(result, dict):
        result = {**result, "timings": dict(timings)}
    log.info("%s %s", name, " ".join(f"{k}={v}ms" for k, v in timings.items()))
    return result


# ------------------------------------------------------------------------------
# 2. DATA STRUCTURE SCHEMAS
# ------------------------------------------------------------------------------
class ValuationInputs(BaseModel):
    discount_rate: float
    perpetual_growth: float
    projected_growth_rate: float
    latest_free_cash_flow: float


class MLInsights(BaseModel):
    predicted_target_price: float
    ml_implied_upside: float
    model_confidence_score: float
    feature_metrics: dict


class ChartData(BaseModel):
    dates: list[str]
    historical_close: list[float]
    sma_20: list[float]
    sma_50: list[float]
    ml_forecast_dates: list[str]
    ml_forecast_prices: list[float]


class TechnicalData(BaseModel):
    rsi14: float
    macd: float
    macd_signal: float
    atr14: float
    sma20: float
    sma50: float
    volume_ratio: float
    distance_from_52w_high: float
    trend: str
    rsi_series: list[float]
    macd_series: list[float]
    macd_signal_series: list[float]


class RiskPortfolioMetrics(BaseModel):
    risk_free_rate: float
    market_return_5y: float
    beta: float
    capm_expected_return: float
    variance_5y: float
    std_dev_5y: float
    sharpe_ratio: float
    m_squared: float
    treynor_ratio: float
    information_ratio: float = 0.0
    jensen_alpha: float = 0.0
    tracking_error: float = 0.0
    active_return_annualized: float = 0.0


class FinancialRatios(BaseModel):
    roic: float
    current_ratio: float
    quick_ratio: float
    debt_to_equity: float
    interest_coverage: float
    fcf_yield: float
    # FIX: extended ratio set -- returns, margins, efficiency/turnover, liquidity.
    roa: float = 0.0
    roe: float = 0.0
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    net_margin: float = 0.0
    asset_turnover: float = 0.0
    inventory_turnover: float = 0.0
    receivables_turnover: float = 0.0
    cash_ratio: float = 0.0


class PeerItem(BaseModel):
    ticker: str
    name: str
    price: float
    forward_pe: float
    trailing_pe: float
    revenue_growth: float
    profit_margin: float
    beta: float


class ValuationResponse(BaseModel):
    ticker: str
    company_name: str
    sector: str
    asset_class: str = "Equity"
    analysis_mode: str = "ISSUER"
    quote_unit: str = "USD"
    current_price: float
    fair_value_per_share: float
    dcf_implied_upside_percent: float
    currency: str
    inputs: ValuationInputs
    ml_forecast: MLInsights
    chart_data: ChartData
    technicals: TechnicalData
    risk_metrics: RiskPortfolioMetrics
    financial_ratios: FinancialRatios
    altman: dict = {}
    peers: list[PeerItem]
    peer_target_price: float
    peer_multiple: float
    peer_note: str
    analyst_target_price: float
    analyst_count: int
    composite_target_price: float
    composite_implied_upside_percent: float
    methodology: str
    risk_charts: dict
    news: list[dict]
    economic_calendar: list[dict]
    geopolitical_news: list[dict] = []
    cfd_scenario: dict = {}
    trade_setup: dict = {}
    ratio_history: list[dict] = []
    financial_statements: dict = {}
    directional_decision: dict = {}
    fundamental_analysis: dict = {}
    timings: dict = {}


# ------------------------------------------------------------------------------
# 3. QUANTITATIVE, RISK & FINANCIAL ANALYSIS PIPELINE
# ------------------------------------------------------------------------------
def _safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def get_sector_peers(sector: str, target_ticker: str) -> list[str]:
    """Dynamically map sector to direct industry competitors to ensure strict peer alignment."""
    sector_map = {
        "Technology": ["MSFT", "AAPL", "GOOGL", "NVDA", "META", "AMD", "AVGO", "ORCL", "CRM"],
        "Financial Services": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK"],
        "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO"],
        "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW"],
        "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS"],
        "Industrials": ["CAT", "GE", "UNP", "HON", "UPS", "BA", "LMT"],
        "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MDLZ"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
        "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE"],
        "Real Estate": ["PLD", "AMT", "EQIX", "CCI", "PSA", "O"],
        "Basic Materials": ["LIN", "APD", "FCX", "NEM", "ECL", "SHW"]
    }

    default_pool = ["MSFT", "AAPL", "GOOGL", "AMZN", "NVDA", "META"]
    pool = sector_map.get(sector, default_pool)

    # Exclude target ticker from its own peer group and return up to 5 strict peers
    filtered = [t for t in pool if t.upper() != target_ticker.upper()]
    return filtered[:5]


@_timed('risk')
def calculate_risk_and_portfolio_metrics(history_5y: pd.DataFrame, beta: float = 1.0, current_price: float = 0.0) -> dict:
    """CAPM + Sharpe/Treynor + Information Ratio + Jensen alpha vs S&P 500."""
    risk_free_rate = _risk_free_rate()
    market = _ticker("^GSPC").history(period="5y", auto_adjust=True)
    if history_5y is None or history_5y.empty or market is None or market.empty:
        return {"risk_free_rate":risk_free_rate,"market_return_5y":0.0,"beta":beta,"capm_expected_return":risk_free_rate,
                "variance_5y":0.0,"std_dev_5y":0.0,"sharpe_ratio":0.0,"m_squared":0.0,"treynor_ratio":0.0,
                "information_ratio":0.0,"jensen_alpha":0.0,"tracking_error":0.0,"active_return_annualized":0.0}
    joined = pd.concat([history_5y["Close"].rename("asset"), market["Close"].rename("market")], axis=1).dropna()
    if len(joined) < 252:
        return {"risk_free_rate":risk_free_rate,"market_return_5y":0.0,"beta":beta,"capm_expected_return":risk_free_rate,
                "variance_5y":0.0,"std_dev_5y":0.0,"sharpe_ratio":0.0,"m_squared":0.0,"treynor_ratio":0.0,
                "information_ratio":0.0,"jensen_alpha":0.0,"tracking_error":0.0,"active_return_annualized":0.0}
    r = np.log(joined/joined.shift(1)).dropna()
    a, m = r["asset"], r["market"]
    trading_days = 252
    asset_return = float(a.mean()*trading_days)
    market_return = float(m.mean()*trading_days)
    market_var = float(m.var(ddof=1))
    covariance = float(a.cov(m))
    beta_calc = covariance/market_var if market_var > 0 else beta
    std = float(a.std(ddof=1)*np.sqrt(trading_days))
    variance = std**2
    capm = risk_free_rate + beta_calc*(market_return-risk_free_rate)
    sharpe = (asset_return-risk_free_rate)/std if std > 0 else 0.0
    market_std = float(m.std(ddof=1)*np.sqrt(trading_days))
    m2 = risk_free_rate + sharpe*market_std
    treynor = (asset_return-risk_free_rate)/beta_calc if beta_calc != 0 else 0.0
    simple = joined.pct_change().dropna()
    active = simple["asset"]-simple["market"]
    active_return_ann = float(active.mean()*trading_days)
    tracking_error = float(active.std(ddof=1)*np.sqrt(trading_days)) if len(active)>1 else 0.0
    information_ratio = active_return_ann/tracking_error if tracking_error>0 else 0.0
    jensen_alpha = asset_return-capm
    five_year_total = float(joined["asset"].iloc[-1]/joined["asset"].iloc[0]-1)
    return {"risk_free_rate":round(risk_free_rate,4),"market_return_5y":round(market_return,4),"beta":round(beta_calc,3),
            "capm_expected_return":round(capm,4),"variance_5y":round(variance,6),"std_dev_5y":round(std,4),
            "sharpe_ratio":round(sharpe,3),"m_squared":round(m2,4),"treynor_ratio":round(treynor,4),
            "information_ratio":round(information_ratio,4),"jensen_alpha":round(jensen_alpha,4),
            "tracking_error":round(tracking_error,4),"active_return_annualized":round(active_return_ann,4),
            "asset_return_5y_annualized":round(asset_return,4),"five_year_total_return":round(five_year_total,4)}

def _statement_row(df: pd.DataFrame, aliases: list[str]) -> pd.Series | None:
    """Return the first matching annual statement row from a yfinance frame."""
    if df is None or df.empty:
        return None
    normalized = {str(idx).strip().lower(): idx for idx in df.index}
    for alias in aliases:
        key = alias.strip().lower()
        if key in normalized:
            return pd.to_numeric(df.loc[normalized[key]], errors="coerce")
    return None


def _annual_value(series: pd.Series | None, year) -> float | None:
    if series is None:
        return None
    try:
        for col in series.index:
            ts = pd.Timestamp(col)
            if ts.year == int(year):
                value = pd.to_numeric(series[col], errors="coerce")
                return None if pd.isna(value) else float(value)
    except Exception:
        return None
    return None


def _safe_div(num, den) -> float | None:
    if num is None or den is None:
        return None
    if abs(float(den)) < 1e-12:
        return None
    return float(num) / float(den)


@_timed('fundamentals')
def calculate_annual_financial_ratios(ticker_obj: yf.Ticker, current_price: float) -> list[dict]:
    """Calculate ratios for the five most recent fiscal years available from annual statements.

    Missing provider history is represented as null/N/A, never zero. This prevents
    missing data from being mistaken for a genuine 0% margin or 0x liquidity ratio.
    """
    income = ticker_obj.financials
    balance = ticker_obj.balance_sheet
    cashflow = ticker_obj.cashflow

    if income is None or income.empty or balance is None or balance.empty:
        return []

    income_cols = list(income.columns)
    balance_cols = list(balance.columns)
    all_years = sorted({pd.Timestamp(c).year for c in income_cols + balance_cols}, reverse=True)
    years = all_years[:5]

    revenue = _statement_row(income, ['Total Revenue', 'Operating Revenue'])
    gross_profit = _statement_row(income, ['Gross Profit'])
    cost_revenue = _statement_row(income, ['Cost Of Revenue', 'Cost of Revenue'])
    operating_income = _statement_row(income, ['Operating Income', 'EBIT'])
    ebit = _statement_row(income, ['EBIT', 'Operating Income'])
    net_income = _statement_row(income, ['Net Income', 'Net Income Common Stockholders', 'Net Income Including Noncontrolling Interests'])
    tax_provision = _statement_row(income, ['Tax Provision'])
    pretax = _statement_row(income, ['Pretax Income'])
    interest_expense = _statement_row(income, ['Interest Expense', 'Interest Expense Non Operating'])

    total_assets = _statement_row(balance, ['Total Assets'])
    equity = _statement_row(balance, ['Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest'])
    total_debt = _statement_row(balance, ['Total Debt'])
    cash = _statement_row(balance, ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments'])
    current_assets = _statement_row(balance, ['Current Assets'])
    current_liabilities = _statement_row(balance, ['Current Liabilities', 'Total Current Liabilities'])
    inventory = _statement_row(balance, ['Inventory'])
    receivables = _statement_row(balance, ['Receivables', 'Accounts Receivable'])

    fcf_series = _statement_row(cashflow, ['Free Cash Flow']) if cashflow is not None and not cashflow.empty else None
    if fcf_series is None and cashflow is not None and not cashflow.empty:
        ocf = _statement_row(cashflow, ['Operating Cash Flow'])
        capex = _statement_row(cashflow, ['Capital Expenditure', 'Capital Expenditures'])
        if ocf is not None and capex is not None:
            fcf_series = ocf.add(capex, fill_value=0)

    # Current share price is only appropriate for a current FCF yield. For historical
    # rows, leave FCF yield as N/A rather than fabricating historical market caps.
    market_cap_now = _safe_float((ticker_obj.info or {}).get('marketCap'), 0.0)

    rows = []
    for i, year in enumerate(years):
        # Use the immediately older fiscal year for average-balance ratios where possible.
        older_year = years[i + 1] if i + 1 < len(years) else None

        rev = _annual_value(revenue, year)
        gp = _annual_value(gross_profit, year)
        if gp is None:
            cr = _annual_value(cost_revenue, year)
            gp = rev - cr if rev is not None and cr is not None else None
        op = _annual_value(operating_income, year)
        ebit_v = _annual_value(ebit, year)
        ni = _annual_value(net_income, year)
        tax_v = _annual_value(tax_provision, year)
        pretax_v = _annual_value(pretax, year)
        int_exp = _annual_value(interest_expense, year)

        assets = _annual_value(total_assets, year)
        assets_old = _annual_value(total_assets, older_year) if older_year else None
        eq = _annual_value(equity, year)
        eq_old = _annual_value(equity, older_year) if older_year else None
        debt = _annual_value(total_debt, year)
        cash_v = _annual_value(cash, year)
        ca = _annual_value(current_assets, year)
        cl = _annual_value(current_liabilities, year)
        inv = _annual_value(inventory, year)
        rec = _annual_value(receivables, year)
        fcf = _annual_value(fcf_series, year) if fcf_series is not None else None

        avg_assets = ((assets + assets_old) / 2) if assets is not None and assets_old is not None else assets
        avg_eq = ((eq + eq_old) / 2) if eq is not None and eq_old is not None else eq

        tax_rate = None
        if pretax_v is not None and pretax_v > 0 and tax_v is not None:
            tax_rate = max(0.0, min(float(tax_v / pretax_v), 0.35))
        elif pretax_v is not None and pretax_v > 0:
            tax_rate = 0.21

        nopat = None if ebit_v is None or tax_rate is None else ebit_v * (1 - tax_rate)
        invested_capital = None
        if debt is not None and eq is not None:
            invested_capital = debt + eq - (cash_v or 0.0)

        if rev is not None and market_cap_now > 0:
            # This is a valuation snapshot yield, not a historical market-cap yield.
            fcf_yield = _safe_div(fcf, market_cap_now) if fcf is not None else None
        else:
            fcf_yield = None

        rows.append({
            'year': year,
            'roic': _safe_div(nopat, invested_capital),
            'roa': _safe_div(ni, avg_assets),
            'roe': _safe_div(ni, avg_eq),
            'gross_margin': _safe_div(gp, rev),
            'operating_margin': _safe_div(op, rev),
            'net_margin': _safe_div(ni, rev),
            'current_ratio': _safe_div(ca, cl),
            'quick_ratio': _safe_div((cash_v or 0.0) + (rec or 0.0), cl),
            'cash_ratio': _safe_div(cash_v, cl),
            'debt_to_equity': _safe_div(debt, eq),
            'interest_coverage': _safe_div(ebit_v, abs(int_exp) if int_exp is not None else None),
            'asset_turnover': _safe_div(rev, avg_assets),
            'inventory_turnover': _safe_div(abs(_annual_value(cost_revenue, year)) if _annual_value(cost_revenue, year) is not None else None, inv),
            'receivables_turnover': _safe_div(rev, rec),
            'fcf_yield': fcf_yield,
            'revenue': rev,
            'net_income': ni,
            'free_cash_flow': fcf,
        })

    return rows


@_timed('fundamentals')
def calculate_altman_z_score(ticker_obj: yf.Ticker, sector: str = "") -> dict:
    """Classic public-company Altman Z-score from latest annual statements."""
    sector_norm=(sector or "").lower()
    if any(k in sector_norm for k in ["financial", "bank", "insurance"]):
        return {"score":None,"zone":"N/A","status":"Not applicable to financial institutions","model":"Altman Z-Score (classic public-company model)","year":None,"components":{}}
    income=ticker_obj.financials; balance=ticker_obj.balance_sheet; info=ticker_obj.info or {}
    if income is None or income.empty or balance is None or balance.empty:
        return {"score":None,"zone":"N/A","status":"Insufficient annual financial statements","model":"Altman Z-Score (classic public-company model)","year":None,"components":{}}
    years=sorted({pd.Timestamp(c).year for c in list(income.columns)+list(balance.columns)}, reverse=True)
    year=years[0] if years else None
    if year is None:
        return {"score":None,"zone":"N/A","status":"No fiscal year available","model":"Altman Z-Score (classic public-company model)","year":None,"components":{}}
    revenue=_annual_value(_statement_row(income,["Total Revenue","Operating Revenue"]),year)
    ebit=_annual_value(_statement_row(income,["EBIT","Operating Income"]),year)
    assets=_annual_value(_statement_row(balance,["Total Assets"]),year)
    ca=_annual_value(_statement_row(balance,["Current Assets"]),year)
    cl=_annual_value(_statement_row(balance,["Current Liabilities","Total Current Liabilities"]),year)
    re=_annual_value(_statement_row(balance,["Retained Earnings"]),year)
    eq=_annual_value(_statement_row(balance,["Stockholders Equity","Common Stock Equity","Total Equity Gross Minority Interest"]),year)
    liab=_annual_value(_statement_row(balance,["Total Liabilities Net Minority Interest","Total Liabilities"]),year)
    if liab is None and assets is not None and eq is not None: liab=assets-eq
    market_cap=_safe_float(info.get("marketCap"),0.0)
    if any(v is None for v in [revenue,ebit,assets,ca,cl,re,liab]) or assets==0 or liab==0 or market_cap<=0:
        return {"score":None,"zone":"N/A","status":"Required Altman inputs are incomplete","model":"Altman Z-Score (classic public-company model)","year":year,"components":{}}
    x1=(ca-cl)/assets; x2=re/assets; x3=ebit/assets; x4=market_cap/liab; x5=revenue/assets
    score=1.2*x1+1.4*x2+3.3*x3+0.6*x4+1.0*x5
    zone="Safe" if score>2.99 else "Grey" if score>=1.81 else "Distress"
    return {"score":round(float(score),3),"zone":zone,"status":"Calculated from latest annual statements","model":"Altman Z-Score (classic public-company model)","year":year,
            "components":{"X1_working_capital_to_assets":round(float(x1),4),"X2_retained_earnings_to_assets":round(float(x2),4),"X3_EBIT_to_assets":round(float(x3),4),"X4_market_equity_to_liabilities":round(float(x4),4),"X5_sales_to_assets":round(float(x5),4)}}


def _statement_values_by_year(df: pd.DataFrame, labels: list[str], years: list[int]) -> dict[int, float | None]:
    series = _statement_row(df, labels)
    return {year: _annual_value(series, year) for year in years}


@_timed('fundamentals')
def calculate_financial_statement_analysis(ticker_obj: yf.Ticker) -> dict:
    """Load and analyze annual income statement, balance sheet and cash flow.

    The provider's available fiscal years are preserved. Missing lines are None/N/A,
    never fabricated as zeros. The returned analysis is descriptive and comparative.
    """
    income = ticker_obj.financials
    balance = ticker_obj.balance_sheet
    cashflow = ticker_obj.cashflow

    frames = [df for df in (income, balance, cashflow) if df is not None and not df.empty]
    if not frames:
        return {"years": [], "income_statement": [], "balance_sheet": [], "cash_flow": [], "analysis": {"status": "No annual statements returned by provider."}}

    years = sorted({pd.Timestamp(c).year for df in frames for c in df.columns}, reverse=True)[:5]

    income_specs = [
        ("Revenue", ["Total Revenue", "Operating Revenue"]),
        ("Gross Profit", ["Gross Profit"]),
        ("Operating Income", ["Operating Income", "EBIT"]),
        ("EBITDA", ["EBITDA"]),
        ("Pretax Income", ["Pretax Income"]),
        ("Net Income", ["Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"]),
        ("Diluted EPS", ["Diluted EPS"]),
    ]
    balance_specs = [
        ("Cash & Equivalents", ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
        ("Total Assets", ["Total Assets"]),
        ("Current Assets", ["Current Assets"]),
        ("Current Liabilities", ["Current Liabilities", "Total Current Liabilities"]),
        ("Total Debt", ["Total Debt"]),
        ("Stockholders Equity", ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
        ("Inventory", ["Inventory"]),
        ("Receivables", ["Receivables", "Accounts Receivable"]),
    ]
    cashflow_specs = [
        ("Operating Cash Flow", ["Operating Cash Flow"]),
        ("Capital Expenditure", ["Capital Expenditure", "Capital Expenditures"]),
        ("Free Cash Flow", ["Free Cash Flow"]),
        ("Investing Cash Flow", ["Investing Cash Flow"]),
        ("Financing Cash Flow", ["Financing Cash Flow"]),
        ("Net Change in Cash", ["Changes In Cash", "Net Change In Cash"]),
    ]

    def build_table(df, specs):
        return [
            {"metric": label, "values": [_annual_value(_statement_row(df, aliases), y) for y in years]}
            for label, aliases in specs
        ]

    income_rows = build_table(income, income_specs)
    balance_rows = build_table(balance, balance_specs)
    cashflow_rows = build_table(cashflow, cashflow_specs)

    def values_map(rows):
        return {r["metric"]: r["values"] for r in rows}

    im, bm, cm = values_map(income_rows), values_map(balance_rows), values_map(cashflow_rows)

    def growth(values):
        vals=[v for v in values if v is not None]
        if len(vals) < 2 or values[0] is None or values[1] in (None, 0):
            return None
        return float(values[0] / values[1] - 1)

    revenue_growth = growth(im.get("Revenue", []))
    net_income_growth = growth(im.get("Net Income", []))
    assets_growth = growth(bm.get("Total Assets", []))
    debt_growth = growth(bm.get("Total Debt", []))
    fcf_values = cm.get("Free Cash Flow", [])
    latest_fcf = fcf_values[0] if fcf_values else None

    analysis = {
        "status": "Annual financial statements loaded.",
        "years_loaded": len(years),
        "revenue_growth_yoy": revenue_growth,
        "net_income_growth_yoy": net_income_growth,
        "assets_growth_yoy": assets_growth,
        "debt_growth_yoy": debt_growth,
        "latest_free_cash_flow": latest_fcf,
        "interpretation": []
    }
    if revenue_growth is not None:
        analysis["interpretation"].append("Revenue is growing year-over-year." if revenue_growth > 0 else "Revenue is declining year-over-year.")
    if net_income_growth is not None:
        analysis["interpretation"].append("Net income is improving year-over-year." if net_income_growth > 0 else "Net income is declining year-over-year.")
    if debt_growth is not None:
        analysis["interpretation"].append("Debt increased year-over-year." if debt_growth > 0 else "Debt decreased year-over-year.")
    if latest_fcf is not None:
        analysis["interpretation"].append("Free cash flow is available for the latest fiscal year.")

    return {
        "years": years,
        "income_statement": income_rows,
        "balance_sheet": balance_rows,
        "cash_flow": cashflow_rows,
        "analysis": analysis,
    }


def extract_financial_ratios(ticker_obj: yf.Ticker, current_price: float) -> dict:
    info = ticker_obj.info or {}
    annual_history = calculate_annual_financial_ratios(ticker_obj, current_price)

    if annual_history:
        latest = annual_history[0]
        return {
            'roic': round(latest['roic'], 4) if latest['roic'] is not None else 0.0,
            'current_ratio': round(latest['current_ratio'], 2) if latest['current_ratio'] is not None else 0.0,
            'quick_ratio': round(latest['quick_ratio'], 2) if latest['quick_ratio'] is not None else 0.0,
            'debt_to_equity': round(latest['debt_to_equity'], 2) if latest['debt_to_equity'] is not None else 0.0,
            'interest_coverage': round(latest['interest_coverage'], 2) if latest['interest_coverage'] is not None else 0.0,
            'fcf_yield': round(latest['fcf_yield'], 4) if latest['fcf_yield'] is not None else 0.0,
            'roa': round(latest['roa'], 4) if latest['roa'] is not None else 0.0,
            'roe': round(latest['roe'], 4) if latest['roe'] is not None else 0.0,
            'gross_margin': round(latest['gross_margin'], 4) if latest['gross_margin'] is not None else 0.0,
            'operating_margin': round(latest['operating_margin'], 4) if latest['operating_margin'] is not None else 0.0,
            'net_margin': round(latest['net_margin'], 4) if latest['net_margin'] is not None else 0.0,
            'asset_turnover': round(latest['asset_turnover'], 2) if latest['asset_turnover'] is not None else 0.0,
            'inventory_turnover': round(latest['inventory_turnover'], 2) if latest['inventory_turnover'] is not None else 0.0,
            'receivables_turnover': round(latest['receivables_turnover'], 2) if latest['receivables_turnover'] is not None else 0.0,
            'cash_ratio': round(latest['cash_ratio'], 2) if latest['cash_ratio'] is not None else 0.0,
        }

    return {
        'roic':0.0,'current_ratio':0.0,'quick_ratio':0.0,'debt_to_equity':0.0,
        'interest_coverage':0.0,'fcf_yield':0.0,'roa':0.0,'roe':0.0,
        'gross_margin':0.0,'operating_margin':0.0,'net_margin':0.0,
        'asset_turnover':0.0,'inventory_turnover':0.0,'receivables_turnover':0.0,'cash_ratio':0.0
    }


# XGBoost forecast, redesigned for speed and soundness:
#   * predicts the 20-trading-day forward RETURN from stationary features
#     (returns, distance from moving averages, volatility, volume ratio).
#     Tree models cannot extrapolate, so predicting raw price LEVELS capped every
#     forecast at the historical maximum; returns don't have that problem.
#   * the model is trained once per symbol per new daily bar and then reused,
#     so repeat visits and the export/model endpoints cost no training time.
#   * confidence = out-of-sample directional hit rate on the most recent 20% of
#     history, with a 20-day gap so test labels never overlap training labels.
_ML_FEATURES = ["ret_1", "ret_5", "ret_20", "dist_sma20", "dist_sma50", "vol_20", "vol_ratio"]
_ML_HORIZON = 20
_ML_THREADS = max(1, int(os.environ.get("ML_THREADS", "2")))
_ML_CACHE: dict[tuple, tuple] = {}
_ML_LOCK = threading.Lock()


def _ml_frame(history: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(history["Close"], errors="coerce").astype(float).ffill()
    if "Volume" in history:
        volume = pd.to_numeric(history["Volume"], errors="coerce").astype(float)
    else:
        volume = pd.Series(np.nan, index=close.index)
    df = pd.DataFrame(index=history.index)
    df["Close"] = close
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    ret = close.pct_change()
    df["ret_1"] = ret
    df["ret_5"] = close.pct_change(5)
    df["ret_20"] = close.pct_change(20)
    df["dist_sma20"] = close / df["SMA_20"] - 1.0
    df["dist_sma50"] = close / df["SMA_50"] - 1.0
    df["vol_20"] = ret.rolling(20).std()
    vol_ratio = volume / volume.rolling(20).mean()
    df["vol_ratio"] = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)   # FX/indices have no volume
    df["Target"] = close.shift(-_ML_HORIZON) / close - 1.0
    return df


def _price_decimals(price: float) -> int:
    return 2 if abs(price) >= 10 else 6


@_timed("ml")
def train_and_predict_ml_target(ticker_obj, current_price: float) -> tuple[dict, dict]:
    history = ticker_obj.history(period="5y")
    if history is None or len(history) < 80:
        raise ValueError("Insufficient historical pricing data for ML model training.")
    info = ticker_obj.info or {}
    df = _ml_frame(history)
    train = df.dropna(subset=_ML_FEATURES + ["Target"])
    if len(train) < 60:
        raise ValueError("Insufficient data points available after rolling feature generation.")

    symbol = str(getattr(ticker_obj, "ticker", "")).upper()
    key = (symbol, str(df.index[-1])[:10], len(train))
    with _ML_LOCK:
        cached = _ML_CACHE.get(key)
    if cached is None:
        X, y = train[_ML_FEATURES], train["Target"]
        split = int(len(train) * 0.8)
        test_start = min(split + _ML_HORIZON, len(train))
        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, n_jobs=_ML_THREADS)
        model.fit(X.iloc[:split], y.iloc[:split])
        if test_start < len(train):
            predicted = model.predict(X.iloc[test_start:])
            actual = y.iloc[test_start:].to_numpy()
            hit_rate = float(np.mean(np.sign(predicted) == np.sign(actual)))
        else:
            hit_rate = 0.5
        cached = (model, hit_rate)
        with _ML_LOCK:
            _ML_CACHE[key] = cached
            for old in [k for k in _ML_CACHE if k[0] == symbol and k != key]:
                _ML_CACHE.pop(old, None)          # keep only the newest model per symbol
            while len(_ML_CACHE) > 300:
                _ML_CACHE.pop(next(iter(_ML_CACHE)), None)
        _STATS["ml_models_trained"] += 1
    else:
        _STATS["ml_models_reused"] += 1
    model, hit_rate = cached

    predicted_return = float(np.clip(model.predict(df[_ML_FEATURES].iloc[[-1]])[0], -0.5, 0.5))
    predicted_price = current_price * (1.0 + predicted_return)
    confidence = round(min(99.9, max(0.0, hit_rate * 100.0)), 2)
    ml_upside = predicted_return * 100.0

    dec = _price_decimals(current_price)
    dates = df.index.strftime("%Y-%m-%d").tolist()
    last_date = pd.to_datetime(dates[-1])
    forecast_dates = [dates[-1]] + [(last_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 21)]
    forecast_prices = np.linspace(current_price, predicted_price, num=21).round(dec).tolist()

    chart_payload = {
        # SMA lines are computed in the browser from historical_close (smaller payload).
        "dates": dates, "historical_close": df["Close"].round(dec).tolist(),
        "sma_20": [], "sma_50": [],
        "ml_forecast_dates": forecast_dates, "ml_forecast_prices": forecast_prices,
    }
    ml_metrics = {
        "predicted_target_price": round(predicted_price, dec), "ml_implied_upside": round(ml_upside, 2),
        "model_confidence_score": confidence, "feature_metrics": {
            "pe_ratio": round(info.get("forwardPE", 0) or 0, 2),
            "profit_margins": round(info.get("profitMargins", 0) or 0, 4),
            "return_on_equity": round(info.get("returnOnEquity", 0) or 0, 4),
            "beta": round(info.get("beta", 1.0) or 1.0, 2),
        },
    }
    return ml_metrics, chart_payload


def calculate_technicals(history: pd.DataFrame) -> dict:
    close = history['Close'].dropna().astype(float)
    volume = history['Volume'].fillna(0).astype(float) if 'Volume' in history else pd.Series(index=close.index,
                                                                                             data=0.0)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    tr = pd.concat([
        history['High'] - history['Low'],
        (history['High'] - history['Close'].shift()).abs(),
        (history['Low'] - history['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    latest = close.iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    avg_volume20 = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()

    return {
        'rsi14': round(_safe_float(rsi.iloc[-1], 50), 2), 'macd': round(_safe_float(macd.iloc[-1]), 4),
        'macd_signal': round(_safe_float(signal.iloc[-1]), 4), 'atr14': round(_safe_float(atr14.iloc[-1]), 4),
        'sma20': round(_safe_float(sma20), 2), 'sma50': round(_safe_float(sma50), 2),
        'volume_ratio': round(_safe_float(volume.iloc[-1] / avg_volume20 if avg_volume20 else 1.0, 1.0), 2),
        'distance_from_52w_high': round((_safe_float(latest / close.tail(252).max() - 1) * 100), 2) if len(
            close) else 0.0,
        'trend': 'bullish' if latest > sma20 > sma50 else ('bearish' if latest < sma20 < sma50 else 'mixed'),
        'rsi_series': rsi.tail(252).round(2).tolist(), 'macd_series': macd.tail(252).round(4).tolist(),
        'macd_signal_series': signal.tail(252).round(4).tolist(),
    }


@_timed('peers')
def peer_snapshot(tickers: list[str], target_ticker: str, target_name: str, target_price: float, target_info: dict) -> \
list[dict]:
    out = [{
        'ticker': target_ticker.upper(),
        'name': target_name,
        'price': round(target_price, 2),
        'forward_pe': round(_safe_float(target_info.get('forwardPE')), 2),
        'trailing_pe': round(_safe_float(target_info.get('trailingPE')), 2),
        'revenue_growth': round(_safe_float(target_info.get('revenueGrowth')) * 100, 2),
        'profit_margin': round(_safe_float(target_info.get('profitMargins')) * 100, 2),
        'beta': round(_safe_float(target_info.get('beta'), 1), 2),
    }]

    _prefetch([lambda s=sym: _ticker(s).info for sym in tickers] +
              [lambda s=sym: _ticker(s).fast_info for sym in tickers])
    for symbol in tickers:
        try:
            t = _ticker(symbol)
            info = t.info or {}
            price = None
            try:
                price = t.fast_info.get('last_price')
            except Exception:
                pass
            if not price:
                price = info.get('currentPrice') or info.get('regularMarketPrice')
            out.append({
                'ticker': symbol.upper(),
                'name': info.get('shortName', symbol.upper()),
                'price': round(_safe_float(price), 2),
                'forward_pe': round(_safe_float(info.get('forwardPE')), 2),
                'trailing_pe': round(_safe_float(info.get('trailingPE')), 2),
                'revenue_growth': round(_safe_float(info.get('revenueGrowth')) * 100, 2),
                'profit_margin': round(_safe_float(info.get('profitMargins')) * 100, 2),
                'beta': round(_safe_float(info.get('beta'), 1), 2),
            })
        except Exception:
            continue
    return out



# Federal Reserve tentative 2027 FOMC schedule (press release, 5 Sep 2025).
# March, June, September and December meetings include the Summary of
# Economic Projections ("dot plot").
_FOMC_2027 = [
    {"date": start, "end_date": end, "time_et": "FOMC",
     "event": "Federal Reserve FOMC Meeting" + (" + Dot Plot" if sep else ""),
     "importance": "VERY HIGH", "source": "Federal Reserve",
     "possible_result": ("A more hawkish policy path/dot plot can pressure growth and real-estate multiples; "
                         "a dovish surprise can support them through lower discount rates.") if sep else
                        ("A more hawkish tone can pressure growth/real-estate multiples; a dovish surprise can "
                         "support them through lower discount rates.")}
    for start, end, sep in [
        ("2027-01-26", "2027-01-27", False), ("2027-03-16", "2027-03-17", True),
        ("2027-04-27", "2027-04-28", False), ("2027-06-08", "2027-06-09", True),
        ("2027-07-27", "2027-07-28", False), ("2027-09-14", "2027-09-15", True),
        ("2027-10-26", "2027-10-27", False), ("2027-12-07", "2027-12-08", True),
        ("2028-01-25", "2028-01-26", False),
    ]
]


def _us_holidays(year: int) -> set:
    fixed = [date(year, 1, 1), date(year, 6, 19), date(year, 7, 4), date(year, 11, 11), date(year, 12, 25)]
    days = set(fixed)
    for d in fixed:                      # observed on Friday/Monday when on a weekend
        if d.weekday() == 5:
            days.add(d - timedelta(days=1))
        elif d.weekday() == 6:
            days.add(d + timedelta(days=1))
    memorial = date(year, 5, 31)
    while memorial.weekday() != 0:
        memorial -= timedelta(days=1)
    labor = date(year, 9, 1)
    while labor.weekday() != 0:
        labor += timedelta(days=1)
    thanksgiving = date(year, 11, 1)
    while thanksgiving.weekday() != 3:
        thanksgiving += timedelta(days=1)
    days.update({memorial, labor, thanksgiving + timedelta(days=21)})
    return days


def _nth_business_day(year: int, month: int, n: int) -> date:
    holidays, d, count = _us_holidays(year), date(year, month, 1), 0
    while True:
        if d.weekday() < 5 and d not in holidays:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)


def _first_friday(year: int, month: int) -> date:
    holidays, d = _us_holidays(year), date(year, month, 1)
    while d.weekday() != 4 or d in holidays:
        d += timedelta(days=1)
    return d


def _estimated_recurring_events(known: list[dict]) -> list[dict]:
    """Keep the calendar populated after the hard-coded BLS/ISM dates run out.

    Uses the long-standing conventions (jobs report on the first Friday, ISM
    Manufacturing on the 1st business day, ISM Services on the 3rd). Every
    generated entry is labelled "(est.)" -- confirm against bls.gov / ismworld.org.
    """
    data_dates = [_dt.strptime(e["end_date"], "%Y-%m-%d").date() for e in known
                  if str(e.get("source", "")).startswith(("BLS", "ISM"))]
    coverage_end = max(data_dates) if data_dates else date.today()
    horizon = date.today() + timedelta(days=150)
    out = []
    y, m = coverage_end.year, coverage_end.month
    while True:
        m += 1
        if m > 12:
            y, m = y + 1, 1
        if date(y, m, 1) > horizon:
            break
        ref = date(y, m, 1) - timedelta(days=1)          # these releases cover the prior month
        label = ref.strftime("%B")
        for d, t, name, imp, src, result in [
            (_nth_business_day(y, m, 1), "10:00", f"ISM Manufacturing PMI - {label}", "MEDIUM", "ISM (est.)",
             "A reading above 50 signals expansion and can support cyclical sectors; a sub-50 print can renew growth-slowdown concerns."),
            (_nth_business_day(y, m, 3), "10:00", f"ISM Services PMI - {label}", "MEDIUM", "ISM (est.)",
             "Services make up the bulk of the economy, so a strong print can reinforce a soft-landing narrative; weakness can raise recession concerns."),
            (_first_friday(y, m), "08:30", f"US Employment Situation (Jobs Report) - {label}", "VERY HIGH", "BLS (est.)",
             "A strong payrolls/low unemployment print can push out rate-cut expectations; a weak print can accelerate them and pressure the dollar."),
        ]:
            if d > coverage_end:
                iso = d.isoformat()
                out.append({"date": iso, "end_date": iso, "time_et": t, "event": name, "importance": imp,
                            "source": src, "possible_result": result})
    return out


def _static_economic_calendar(limit: int = 12) -> list[dict]:
    """Near-term high-impact US macro calendar used by the dashboard.

    FIX (analyst judgment call): the old version was five static, fixed-date
    stub entries that would silently go stale the moment the calendar dates
    passed. This rebuilds it as a real, time-aware 2026 calendar:
      - FOMC dates are the Federal Reserve's official 2026 schedule (all
        eight meetings; the three marked with a Summary of Economic
        Projections/"dot plot" are flagged VERY HIGH).
      - CPI / PPI / Employment Situation (jobs report) / JOLTS / Employment
        Cost Index dates for Sep-Dec 2026 are pulled from the BLS's own
        published 2026 release schedule (bls.gov/schedule/2026).
      - ISM Manufacturing/Services PMI dates are estimated using the
        standard 1st-business-day / 3rd-business-day convention ISM has
        used for years -- flagged as an estimate via the source field,
        since ISM's own calendar wasn't consulted directly.
    Entries are filtered to only what's still upcoming (multi-day entries,
    like FOMC, stay listed through their last day) and sorted chronologically,
    so the calendar stays relevant as real time passes without needing a
    manual rewrite every quarter -- rather than reproducing the old
    static/fake-feeling list.
    """
    events = [
        {"date":"2026-09-10","end_date":"2026-09-10","time_et":"08:30","event":"US Producer Price Index (PPI) - August",
         "importance":"HIGH","source":"BLS",
         "possible_result":"Hotter-than-expected PPI can lift inflation and rate expectations; cooler PPI can support a softer rate outlook."},
        {"date":"2026-09-11","end_date":"2026-09-11","time_et":"08:30","event":"US Consumer Price Index (CPI) - August",
         "importance":"VERY HIGH","source":"BLS",
         "possible_result":"Hotter CPI can push yields higher and pressure rate-sensitive valuation multiples; cooler CPI can support lower yields and risk assets."},
        {"date":"2026-09-15","end_date":"2026-09-16","time_et":"FOMC","event":"Federal Reserve FOMC Meeting + Dot Plot",
         "importance":"VERY HIGH","source":"Federal Reserve",
         "possible_result":"A more hawkish policy path/dot plot can pressure growth and real-estate multiples; a dovish surprise can support them through lower discount rates."},
        {"date":"2026-09-29","end_date":"2026-09-29","time_et":"10:00","event":"US JOLTS Job Openings - August",
         "importance":"HIGH","source":"BLS",
         "possible_result":"Stronger labor demand can reinforce higher-for-longer rate expectations; weaker demand can support an easier-policy view."},
        {"date":"2026-10-01","end_date":"2026-10-01","time_et":"10:00","event":"ISM Manufacturing PMI - September",
         "importance":"MEDIUM","source":"ISM (est.)",
         "possible_result":"A reading above 50 signals expansion and can support cyclical sectors; a sub-50 print can renew growth-slowdown concerns."},
        {"date":"2026-10-02","end_date":"2026-10-02","time_et":"08:30","event":"US Employment Situation (Jobs Report) - September",
         "importance":"VERY HIGH","source":"BLS",
         "possible_result":"A strong payrolls/low unemployment print can push out rate-cut expectations; a weak print can accelerate them and pressure the dollar."},
        {"date":"2026-10-05","end_date":"2026-10-05","time_et":"10:00","event":"ISM Services PMI - September",
         "importance":"MEDIUM","source":"ISM (est.)",
         "possible_result":"Services make up the bulk of the economy, so a strong print can reinforce a soft-landing narrative; weakness can raise recession concerns."},
        {"date":"2026-10-14","end_date":"2026-10-14","time_et":"08:30","event":"US Consumer Price Index (CPI) - September",
         "importance":"VERY HIGH","source":"BLS",
         "possible_result":"Hotter CPI can push yields higher and pressure rate-sensitive valuation multiples; cooler CPI can support lower yields and risk assets."},
        {"date":"2026-10-15","end_date":"2026-10-15","time_et":"08:30","event":"US Producer Price Index (PPI) - September",
         "importance":"HIGH","source":"BLS",
         "possible_result":"Hotter-than-expected PPI can lift inflation and rate expectations; cooler PPI can support a softer rate outlook."},
        {"date":"2026-10-27","end_date":"2026-10-28","time_et":"FOMC","event":"Federal Reserve FOMC Meeting",
         "importance":"VERY HIGH","source":"Federal Reserve",
         "possible_result":"A more hawkish tone can pressure growth/real-estate multiples; a dovish surprise can support them through lower discount rates."},
        {"date":"2026-10-30","end_date":"2026-10-30","time_et":"08:30","event":"US Employment Cost Index - Q3",
         "importance":"MEDIUM","source":"BLS",
         "possible_result":"Faster wage-cost growth can feed sticky-services-inflation concerns; a slowdown can support the disinflation narrative."},
        {"date":"2026-12-01","end_date":"2026-12-01","time_et":"10:00","event":"ISM Manufacturing PMI - November",
         "importance":"MEDIUM","source":"ISM (est.)",
         "possible_result":"A reading above 50 signals expansion and can support cyclical sectors; a sub-50 print can renew growth-slowdown concerns."},
        {"date":"2026-12-03","end_date":"2026-12-03","time_et":"10:00","event":"ISM Services PMI - November",
         "importance":"MEDIUM","source":"ISM (est.)",
         "possible_result":"Services make up the bulk of the economy, so a strong print can reinforce a soft-landing narrative; weakness can raise recession concerns."},
        {"date":"2026-12-04","end_date":"2026-12-04","time_et":"08:30","event":"US Employment Situation (Jobs Report) - November",
         "importance":"VERY HIGH","source":"BLS",
         "possible_result":"A strong payrolls/low unemployment print can push out rate-cut expectations; a weak print can accelerate them and pressure the dollar."},
        {"date":"2026-12-08","end_date":"2026-12-09","time_et":"FOMC","event":"Federal Reserve FOMC Meeting + Dot Plot",
         "importance":"VERY HIGH","source":"Federal Reserve",
         "possible_result":"The year's final dot plot can reset rate-path expectations heading into the new year, moving both yields and equity multiples."},
        {"date":"2026-12-10","end_date":"2026-12-10","time_et":"08:30","event":"US Consumer Price Index (CPI) - November",
         "importance":"VERY HIGH","source":"BLS",
         "possible_result":"Hotter CPI can push yields higher and pressure rate-sensitive valuation multiples; cooler CPI can support lower yields and risk assets."},
        {"date":"2026-12-15","end_date":"2026-12-15","time_et":"08:30","event":"US Producer Price Index (PPI) - November",
         "importance":"HIGH","source":"BLS",
         "possible_result":"Hotter-than-expected PPI can lift inflation and rate expectations; cooler PPI can support a softer rate outlook."},
    ]

    events = events + _FOMC_2027 + _estimated_recurring_events(events)
    today = date.today()

    def _end_date(e: dict) -> date:
        return _dt.strptime(e["end_date"], "%Y-%m-%d").date()

    upcoming = [e for e in events if _end_date(e) >= today]
    upcoming.sort(key=_end_date)

    # Trim the internal end_date helper field before returning -- the
    # front-end only expects date/time_et/event/importance/source/possible_result.
    return [{k: v for k, v in e.items() if k != "end_date"} for e in upcoming[:limit]]


def build_economic_calendar() -> list[dict]:
    """Upcoming high-impact events for the analysis pipeline.

    Uses the automatic calendar (FRED schedule + weekly consensus feed, see
    finmodel_intel) when it is cached; never waits on the network -- a cold
    calendar is built in the background and the built-in schedule is used meanwhile.
    """
    try:
        rows = finmodel_intel.calendar_rows_for_pipeline()
        if rows:
            return rows
    except Exception as exc:
        log.info("automatic calendar unavailable: %s", exc)
    return _static_economic_calendar()


def classify_news_impact(title: str, summary: str, sector: str) -> tuple[str, str]:
    txt = (title + " " + summary).lower()
    if any(w in txt for w in ["inflation", "hotter", "higher rates", "rate hike", "hawkish", "yield surge", "strong jobs"]):
        if sector in {"Technology", "Communication Services", "Real Estate", "Utilities"}:
            return "NEGATIVE RISK", "Potential valuation pressure if yields and discount rates rise."
        if sector == "Financial Services":
            return "MIXED / POSITIVE", "Higher rates can support lending spreads, but credit/economic risks can offset the benefit."
        return "MIXED", "Potential discount-rate pressure; sector-specific effects may vary."
    if any(w in txt for w in ["disinflation", "cooling inflation", "rate cut", "dovish", "lower yields", "weak jobs"]):
        if sector in {"Technology", "Communication Services", "Real Estate", "Utilities"}:
            return "POSITIVE RISK", "Potential support if lower yields improve valuation multiples."
        if sector == "Financial Services":
            return "MIXED / NEGATIVE", "Lower rates can compress spreads, although credit demand may improve."
        return "MIXED", "Potentially supportive through lower discount rates."
    if any(w in txt for w in ["beat", "upgrade", "raises guidance", "record revenue", "strong demand"]):
        return "POSITIVE", "Potentially supportive if the headline improves earnings expectations."
    if any(w in txt for w in ["miss", "downgrade", "cuts guidance", "weak demand", "margin pressure", "lawsuit"]):
        return "NEGATIVE", "Potentially damaging to earnings expectations or sentiment."
    return "WATCH", "Direction is uncertain; confirm with price, volume and subsequent guidance."


def _news_date_label(value) -> str:
    if isinstance(value, (int, float)):
        try:
            return _dt.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return str(value)
    return str(value)


@_timed('news')
def fetch_geopolitical_news(ticker_symbol: str, company_name: str, sector: str) -> list[dict]:
    return _copy(_data_get('geo', (str(ticker_symbol).upper(),),
                           lambda: _fetch_geopolitical_news_uncached(ticker_symbol, company_name, sector)))


def _fetch_geopolitical_news_uncached(ticker_symbol: str, company_name: str, sector: str) -> list[dict]:
    """Fetch asset-related geopolitical headlines from Google News RSS.

    This is intentionally a research feed, not a trade signal. The query is
    built from the selected asset plus geopolitical terms so the dashboard can
    distinguish ordinary company headlines from country/geopolitical catalysts.
    """
    terms = (
        'geopolitical OR tariffs OR sanctions OR war OR conflict OR ' 
        'trade restrictions OR export controls OR energy shock OR election OR ' 
        'shipping disruption OR supply chain'
    )
    query = f'"{ticker_symbol}" OR "{company_name}" {terms}'
    url = 'https://news.google.com/rss/search?q=' + _url_quote(query) + '&hl=en-US&gl=US&ceid=US:en'
    out = []
    try:
        resp = requests.get(url, timeout=8, headers={'User-Agent':'finmodel/1.0'})
        resp.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        for item in root.findall('.//item')[:12]:
            title = _html.unescape((item.findtext('title') or '').strip())
            link = (item.findtext('link') or '').strip()
            pub = (item.findtext('pubDate') or '').strip()
            source = (item.findtext('source') or 'Google News').strip()
            desc = _html.unescape((item.findtext('description') or '').strip())
            # Strip HTML tags from the RSS description.
            import re
            desc = re.sub(r'<[^>]+>', ' ', desc)
            desc = re.sub(r'\s+', ' ', desc).strip()
            txt = (title + ' ' + desc).lower()
            tags = []
            if any(w in txt for w in ['sanction','tariff','export control','trade restriction']): tags.append('TRADE / POLICY')
            if any(w in txt for w in ['war','conflict','missile','invasion','ceasefire','military']): tags.append('CONFLICT')
            if any(w in txt for w in ['election','government','president','regulation','regulator']): tags.append('POLICY / POLITICS')
            if any(w in txt for w in ['shipping','strait','supply chain','pipeline','oil shock']): tags.append('SUPPLY / ENERGY')
            if not tags: tags.append('GEOPOLITICAL WATCH')
            out.append({
                'title': title or 'Untitled', 'summary': desc[:350], 'published': pub,
                'source': source, 'url': link, 'tags': tags, 'sector': sector
            })
    except Exception:
        return []
    return out


def build_cfd_scenario(
    ticker: str, company_name: str, sector: str, current_price: float,
    technicals: dict, risk_metrics: dict, trade_setup: dict,
    news: list[dict], economic_calendar: list[dict], geopolitical_news: list[dict]
) -> dict:
    """Build an educational CFD scenario dashboard from the existing research stack.

    No order is placed and no probability of profit is estimated. The output is
    a transparent scenario map: price distance, ATR, catalyst pressure and
    event exposure.
    """
    atr = max(_safe_float(technicals.get('atr14')), current_price * 0.01)
    beta = _safe_float(risk_metrics.get('beta'), 1.0)
    vol = _safe_float(risk_metrics.get('std_dev_5y'))
    rsi = _safe_float(technicals.get('rsi14'), 50.0)
    trend = str(technicals.get('trend') or 'UNKNOWN')

    high_events = [e for e in economic_calendar if str(e.get('importance','')).upper() in {'HIGH','VERY HIGH'}]
    event_risk = min(100.0, len(high_events) * 12.0)

    positive = negative = 0
    for n in (news or []):
        impact = str(n.get('impact','')).upper()
        if 'POSITIVE' in impact: positive += 1
        elif 'NEGATIVE' in impact: negative += 1
    geo_risk = min(100.0, len(geopolitical_news) * 8.0)

    headline_bias = max(-100.0, min(100.0, (positive-negative)*15.0))
    if trend.upper() == 'BULLISH': headline_bias += 10
    elif trend.upper() == 'BEARISH': headline_bias -= 10
    headline_bias = max(-100.0, min(100.0, headline_bias))

    catalyst_pressure = min(100.0, 0.45*event_risk + 0.30*geo_risk + 0.25*abs(headline_bias))
    risk_regime = 'ELEVATED' if catalyst_pressure >= 60 else 'WATCH' if catalyst_pressure >= 30 else 'NORMAL'

    long_entry = _safe_float(trade_setup.get('long_suitable_entry'), current_price)
    short_entry = _safe_float(trade_setup.get('short_suitable_entry'), current_price)

    # Educational price-shock scenarios around each fitted direction price.
    scenarios = []
    for label, entry, direction in [('LONG', long_entry, 'LONG'), ('SHORT', short_entry, 'SHORT')]:
        for mult in (-2.0, -1.0, 1.0, 2.0):
            move = mult * atr
            terminal = entry + move
            pnl_per_unit = (terminal-entry) if direction == 'LONG' else (entry-terminal)
            scenarios.append({
                'direction': direction, 'shock_atr': mult, 'entry': round(entry,2),
                'terminal_price': round(terminal,2), 'pnl_per_unit': round(pnl_per_unit,2),
                'move_percent': round((move/entry)*100,2) if entry else 0.0
            })

    event_focus=[]
    for e in economic_calendar[:8]:
        ev=str(e.get('event',''))
        relevance='HIGH' if sector in {'Technology','Energy','Industrials','Financial Services','Basic Materials'} else 'MEDIUM'
        event_focus.append({**e, 'asset_relevance': relevance})

    return {
        'ticker': ticker.upper(), 'company_name': company_name, 'sector': sector,
        'spot': round(current_price,2), 'atr14': round(atr,2), 'beta': round(beta,3),
        'volatility_5y': round(vol,4), 'rsi14': round(rsi,2), 'trend': trend,
        'long_fitted_entry': round(long_entry,2), 'short_fitted_entry': round(short_entry,2),
        'event_count': len(economic_calendar), 'high_impact_event_count': len(high_events),
        'geopolitical_headline_count': len(geopolitical_news),
        'event_risk_score': round(event_risk,1), 'geopolitical_risk_score': round(geo_risk,1),
        'headline_bias_score': round(headline_bias,1), 'catalyst_pressure_score': round(catalyst_pressure,1),
        'risk_regime': risk_regime,
        'scenario_table': scenarios, 'event_focus': event_focus,
        'methodology': 'Scenario map = existing fitted long/short price + ATR shocks + event count + geopolitical headline count + headline/trend context. It is not a return-probability model or an order signal.'
    }


@_timed('news')
def fetch_asset_news(ticker_symbol: str, sector: str) -> list[dict]:
    """Fetch recent ticker headlines exposed by yfinance."""
    out = []
    try:
        raw = _ticker(ticker_symbol).news or []
        for item in raw[:10]:
            content = item.get("content", item)
            title = content.get("title") or item.get("title") or "Untitled"
            summary = content.get("summary") or item.get("summary") or ""
            published = content.get("pubDate") or item.get("providerPublishTime") or ""
            canonical = content.get("canonicalUrl") or {}
            url = canonical.get("url", "") if isinstance(canonical, dict) else ""
            url = url or item.get("link", "")
            impact, implication = classify_news_impact(title, summary, sector)
            out.append({"title":title,"summary":summary[:350],"published":str(published),"url":url,
                        "impact":impact,"possible_result":implication})
    except Exception:
        pass
    return out


@_timed('risk')
def build_risk_charts(history_5y: pd.DataFrame, benchmark_symbol: str = "^GSPC") -> dict:
    """Build SML, CAL and asset-vs-market return scatter data from 5Y history."""
    market = _ticker(benchmark_symbol).history(period="5y", auto_adjust=True)
    if history_5y.empty or market.empty:
        return {}
    prices = pd.concat([history_5y["Close"].rename("asset"), market["Close"].rename("market")], axis=1).dropna()
    returns = prices.pct_change().dropna()
    if len(returns) < 100:
        return {}
    x = returns["market"].to_numpy()
    y = returns["asset"].to_numpy()
    beta = float(np.cov(y, x, ddof=1)[0,1] / np.var(x, ddof=1))
    alpha_daily = float(y.mean() - beta*x.mean())
    corr = float(np.corrcoef(x, y)[0,1])
    rf = _risk_free_rate()
    market_ann = float(x.mean()*252)
    asset_ann = float(y.mean()*252)
    asset_vol = float(y.std(ddof=1)*np.sqrt(252))
    beta_grid = np.linspace(-0.25, 2.75, 25)
    sml = rf + beta_grid*(market_ann-rf)
    sample = returns.tail(min(800, len(returns)))
    scatter = [{"market":round(float(a*100),3),"asset":round(float(b*100),3)} for a,b in zip(sample["market"], sample["asset"])]
    sharpe = (asset_ann-rf)/asset_vol if asset_vol > 0 else 0.0
    cal_vol = np.linspace(0, max(asset_vol*1.8, 0.30), 25)
    cal = rf + sharpe*cal_vol
    return {
        "beta":beta,"correlation":corr,"market_annual_return":market_ann,"asset_annual_return":asset_ann,
        "asset_annual_volatility":asset_vol,
        "scatter":scatter,
        "regression":{"x0":-5,"x1":5,"y0":(alpha_daily+beta*(-0.05))*100,"y1":(alpha_daily+beta*0.05)*100},
        "sml":{"beta":beta_grid.round(3).tolist(),"return":(sml*100).round(3).tolist()},
        "cal":{"volatility":(cal_vol*100).round(3).tolist(),"return":(cal*100).round(3).tolist()}
    }


def calculate_trade_setup(current_price: float, technicals: dict, equity: float = 10000.0,
                          used_margin: float = 2000.0, units: float = 100.0,
                          entry_price: float | None = None,
                          direction: str = "LONG",
                          composite_target_price: float | None = None,
                          fair_value_per_share: float | None = None,
                          peer_target_price: float | None = None,
                          ml_target_price: float | None = None,
                          analyst_target_price: float | None = None,
                          beta: float = 1.0) -> dict:
    """Volatility-based educational trade scenario calculator.

    FIX: leverage input replaced with a directly-chosen `used_margin` --
    the caller freely decides how much of their equity to commit as margin
    to this position. `free_margin` (equity - used_margin) is a derived
    OUTPUT, and `implied_leverage` (notional / used_margin) is reported as
    an informational output rather than being an input that drives margin.

    FIX: position sizing switched from a risk-percent model (which derived
    units from equity*risk% / stop-distance) to a direct unit-count model --
    the caller supplies `units` (the position size) and, optionally, a
    chosen `entry_price` to size that position at. Risk in dollars is now a
    derived OUTPUT (units * stop distance) rather than an input.

    FIX: the "most suitable entry price" now actually uses ALL of the app's
    analysis, not just price structure. It blends four independent lenses:
      - `technical_anchor`  (35%): price/SMA/ATR structure -- same logic
         used before, unchanged.
      - `valuation_anchor`  (35%): the app's own composite valuation target
         (DCF fair value + peer sector-multiple target + ML forecast +
         analyst consensus, i.e. every valuation method the app runs),
         discounted/premium-adjusted by a 5% margin of safety so you're not
         told to pay exactly fair value.
      - `momentum_anchor`   (15%): RSI(14) mean-reversion read -- overbought
         waits for a pullback to SMA20, oversold says the current price is
         already an acceptable entry.
      - `risk_anchor`       (15%): a beta/ATR-scaled pullback or premium off
         current price, so higher-beta (more volatile) names ask for a
         wider cushion before entry.
    The blend is clamped to +/-3 ATR of current price so a DCF target that's
    wildly detached from the market can't produce a nonsensical entry.
    If `composite_target_price` isn't supplied, it's rebuilt on the fly from
    whichever of the individual valuation targets (DCF/peer/ML/analyst) are
    available; if none are, the valuation lens falls back to current price
    (neutral) rather than skewing the blend.
    """
    atr = max(float(technicals.get("atr14",0)), current_price*0.01)
    sma20 = float(technicals.get("sma20",current_price))
    sma50 = float(technicals.get("sma50",current_price))
    rsi = float(technicals.get("rsi14", 50) or 50)
    direction = direction.upper()

    # --- Lens 1: technical price/SMA/ATR structure (unchanged from before) ---
    if direction == "LONG":
        technical_anchor = min(current_price, max(sma50, sma20 + 0.25*atr)) if current_price > sma20 else current_price
    else:
        technical_anchor = max(current_price, min(sma50, sma20 - 0.25*atr)) if current_price < sma20 else current_price

    # --- Lens 2: valuation consensus (DCF + peer + ML + analyst) ---
    if composite_target_price and composite_target_price > 0:
        valuation_target = composite_target_price
    else:
        weights = {"dcf": 0.30, "peer": 0.20, "ml": 0.25, "analyst": 0.25}
        values = {}
        if fair_value_per_share and fair_value_per_share > 0:
            values["dcf"] = fair_value_per_share
        if peer_target_price and peer_target_price > 0:
            values["peer"] = peer_target_price
        if ml_target_price and ml_target_price > 0:
            values["ml"] = ml_target_price
        if analyst_target_price and analyst_target_price > 0:
            values["analyst"] = analyst_target_price
        if values:
            weight_sum = sum(weights[k] for k in values)
            valuation_target = sum(values[k]*weights[k] for k in values) / weight_sum
        else:
            valuation_target = current_price

    margin_of_safety = 0.05
    if direction == "LONG":
        valuation_anchor = valuation_target * (1 - margin_of_safety)
    else:
        valuation_anchor = valuation_target * (1 + margin_of_safety)

    # --- Lens 3: RSI(14) momentum / mean-reversion read ---
    if direction == "LONG":
        if rsi >= 70:
            momentum_anchor = sma20            # overbought -- wait for a pullback
        elif rsi <= 30:
            momentum_anchor = current_price    # oversold -- current price is already fine
        else:
            momentum_anchor = (current_price + sma20) / 2
    else:
        if rsi <= 30:
            momentum_anchor = sma20            # oversold -- wait for a bounce before shorting
        elif rsi >= 70:
            momentum_anchor = current_price    # overbought -- current price is already fine
        else:
            momentum_anchor = (current_price + sma20) / 2

    # --- Lens 4: beta/ATR-scaled risk buffer off current price ---
    risk_buffer = beta * atr * 0.5
    risk_anchor = current_price - risk_buffer if direction == "LONG" else current_price + risk_buffer

    # Build BOTH direction-specific model prices.  Long uses the valuation
    # margin-of-safety below fair value; short uses the premium above fair
    # value.  Both are independently bounded by current price +/- 3 ATR.
    long_valuation_anchor = valuation_target * (1 - margin_of_safety)
    short_valuation_anchor = valuation_target * (1 + margin_of_safety)
    long_technical_anchor = min(current_price, max(sma50, sma20 + 0.25*atr)) if current_price > sma20 else current_price
    short_technical_anchor = max(current_price, min(sma50, sma20 - 0.25*atr)) if current_price < sma20 else current_price
    long_momentum_anchor = (
        sma20 if rsi >= 70 else current_price if rsi <= 30 else (current_price+sma20)/2
    )
    short_momentum_anchor = (
        sma20 if rsi <= 30 else current_price if rsi >= 70 else (current_price+sma20)/2
    )
    long_risk_anchor = current_price - risk_buffer
    short_risk_anchor = current_price + risk_buffer

    long_suitable_entry = (
        0.35*long_technical_anchor + 0.35*long_valuation_anchor +
        0.15*long_momentum_anchor + 0.15*long_risk_anchor
    )
    short_suitable_entry = (
        0.35*short_technical_anchor + 0.35*short_valuation_anchor +
        0.15*short_momentum_anchor + 0.15*short_risk_anchor
    )
    # Directional guardrails keep a long model from drifting above the spot
    # price and a short model from drifting below it, while still allowing
    # neutral/weak signals to fall back toward spot.
    long_suitable_entry = min(long_suitable_entry, current_price)
    short_suitable_entry = max(short_suitable_entry, current_price)
    lower_bound, upper_bound = current_price - 3*atr, current_price + 3*atr
    long_suitable_entry = min(max(long_suitable_entry, lower_bound), upper_bound)
    short_suitable_entry = min(max(short_suitable_entry, lower_bound), upper_bound)
    most_suitable_entry = long_suitable_entry if direction == "LONG" else short_suitable_entry

    entry = float(entry_price) if (entry_price is not None and entry_price > 0) else most_suitable_entry

    if direction == "LONG":
        stop = entry - atr
        target = entry + 2*atr
    else:
        stop = entry + atr
        target = entry - 2*atr

    distance = abs(entry-stop)
    units = max(units, 0)
    used_margin = max(used_margin, 0)
    notional = units*entry
    free_margin = equity - used_margin
    implied_leverage = (notional/used_margin) if used_margin > 0 else None
    risk_dollars = units*distance
    pnl_tp = units*abs(target-entry)
    pnl_sl = -units*abs(stop-entry)

    # Gap between the raw valuation consensus (pre margin-of-safety) and the
    # current market price -- how rich/cheap the market thinks this is.
    fundamental_gap_percent = ((valuation_target - current_price) / current_price) * 100 if current_price else 0.0

    return {
        "direction":direction,"entry":round(entry,2),"stop_loss":round(stop,2),"take_profit":round(target,2),
        "most_suitable_entry":round(most_suitable_entry,2),
        "long_suitable_entry":round(long_suitable_entry,2),
        "short_suitable_entry":round(short_suitable_entry,2),
        "long_technical_anchor":round(long_technical_anchor,2),
        "short_technical_anchor":round(short_technical_anchor,2),
        "long_valuation_anchor":round(long_valuation_anchor,2),
        "short_valuation_anchor":round(short_valuation_anchor,2),
        "long_momentum_anchor":round(long_momentum_anchor,2),
        "short_momentum_anchor":round(short_momentum_anchor,2),
        "long_risk_anchor":round(long_risk_anchor,2),
        "short_risk_anchor":round(short_risk_anchor,2),
        "technical_anchor":round(technical_anchor,2),
        "valuation_anchor":round(valuation_anchor,2),
        "valuation_target":round(valuation_target,2),
        "momentum_anchor":round(momentum_anchor,2),
        "risk_anchor":round(risk_anchor,2),
        "fundamental_gap_percent":round(fundamental_gap_percent,2),
        "atr14":round(atr,2),"risk_reward":2.0,"equity":round(equity,2),
        "used_margin":round(used_margin,2),"free_margin":round(free_margin,2),
        "implied_leverage": round(implied_leverage,2) if implied_leverage is not None else None,
        "units":round(units,4),"notional":round(notional,2),
        "risk_dollars":round(risk_dollars,2),"profit_at_target":round(pnl_tp,2),
        "equity_if_target":round(equity+pnl_tp,2),"equity_if_stop":round(equity+pnl_sl,2),
        "margin_exceeds_equity": used_margin > equity,
        "note":"Educational scenario only. Real CFD outcomes also depend on spread, slippage, financing, broker margin rules and liquidation rules."
    }

def make_directional_decision(
    current_price: float,
    fair_value: float,
    composite_target: float,
    ml_target: float,
    analyst_target: float,
    technicals: dict,
    risk_metrics: dict,
    financial_ratios: dict,
    cfd_scenario: dict | None = None,
    news: list[dict] | None = None,
    economic_calendar: list[dict] | None = None,
    geopolitical_news: list[dict] | None = None,
) -> dict:
    """Create an educational directional *research bias* from the platform's analyses.

    This is a transparent scoring model, not a prediction of profit and not an
    order-execution recommendation.  Positive scores represent evidence that
    is more consistent with a bullish/long scenario; negative scores represent
    evidence more consistent with a bearish/short scenario.
    """
    price = _safe_float(current_price, 0.0)
    if price <= 0:
        return {"decision":"NO CLEAR DIRECTION", "score":0.0, "confidence":"LOW", "components":[],
                "long_case":[], "short_case":[], "action":"WAIT / INSUFFICIENT DATA",
                "methodology":"No valid current price."}

    def clip(x, lo=-1.0, hi=1.0):
        return max(lo, min(hi, float(x)))

    def pct_gap(target):
        t = _safe_float(target, price)
        return (t - price) / price if price else 0.0

    score_parts = []
    long_case, short_case = [], []

    def add(name, value, weight, long_text, short_text, threshold=0.0):
        contribution = clip(value) * weight
        score_parts.append({"name":name, "raw":round(float(value),4), "weight":weight,
                            "contribution":round(contribution,2)})
        if contribution > threshold:
            long_case.append(long_text)
        elif contribution < -threshold:
            short_case.append(short_text)
        return contribution

    # Valuation: DCF + composite + ML + analyst consensus.
    valuation_raw = float(np.mean([pct_gap(fair_value), pct_gap(composite_target), pct_gap(ml_target), pct_gap(analyst_target)]))
    valuation_raw = clip(valuation_raw / 0.25)
    score = add("Valuation consensus", valuation_raw, 30, "Valuation targets are above spot.", "Valuation targets are below spot.", 0.75)

    # Technical trend, RSI, MACD and volume.
    trend = str(technicals.get("trend") or "").upper()
    trend_raw = 1 if trend == "BULLISH" else -1 if trend == "BEARISH" else 0
    score += add("Trend", trend_raw, 15, "Trend structure is bullish.", "Trend structure is bearish.", 1.0)

    rsi = _safe_float(technicals.get("rsi14"), 50)
    # Favor momentum direction while avoiding extreme RSI being treated as a blind signal.
    rsi_raw = clip((rsi - 50) / 25)
    score += add("RSI momentum", rsi_raw, 10, "RSI supports positive momentum.", "RSI supports negative momentum.", 0.4)

    macd = _safe_float(technicals.get("macd"), 0)
    macd_sig = _safe_float(technicals.get("macd_signal"), 0)
    macd_raw = clip((macd - macd_sig) / (abs(macd_sig) + abs(macd) + 1e-9) * 2)
    score += add("MACD", macd_raw, 10, "MACD is on the bullish side of its signal.", "MACD is on the bearish side of its signal.", 0.4)

    vol_ratio = _safe_float(technicals.get("volume_ratio"), 1)
    # Volume confirms trend only mildly; without direction it contributes little.
    vol_confirm = trend_raw * clip((vol_ratio - 1.0) / 1.5)
    score += add("Volume confirmation", vol_confirm, 5, "Volume is confirming the prevailing bullish trend.", "Volume is confirming the prevailing bearish trend.", 0.2)

    # CAPM / risk-adjusted context.
    capm = _safe_float(risk_metrics.get("capm_expected_return"), 0)
    asset_ann = _safe_float(risk_metrics.get("market_return_5y"), 0)
    capm_gap = clip((capm - _safe_float(risk_metrics.get('risk_free_rate'), _RF_FALLBACK)) / 0.20)
    score += add("CAPM return context", capm_gap, 5, "CAPM context is favorable relative to the risk-free rate.", "CAPM context is weak relative to the risk-free rate.", 0.25)

    sharpe = _safe_float(risk_metrics.get("sharpe_ratio"), 0)
    score += add("Sharpe context", clip(sharpe / 2.0), 5, "Historical risk-adjusted return context is constructive.", "Historical risk-adjusted return context is weak.", 0.25)

    # Fundamentals: latest margins/ROE/ROIC and leverage.
    roic = _safe_float(financial_ratios.get("roic"), 0)
    roe = _safe_float(financial_ratios.get("roe"), 0)
    dte = _safe_float(financial_ratios.get("debt_to_equity"), 0)
    fund_raw = clip((roic + roe) / 0.60) - clip(dte / 3.0) * 0.35
    score += add("Fundamentals", clip(fund_raw), 10, "Profitability fundamentals support the bullish case.", "Profitability/leverage fundamentals weigh on the bullish case.", 0.35)

    # News / macro / geopolitical context is intentionally a smaller part of total score.
    positives = negatives = 0
    for n in (news or []):
        impact = str(n.get("impact", "")).upper()
        if "POSITIVE" in impact: positives += 1
        elif "NEGATIVE" in impact: negatives += 1
    news_raw = clip((positives - negatives) / 4.0)
    score += add("Asset news", news_raw, 5, "Recent asset headlines lean positive.", "Recent asset headlines lean negative.", 0.25)

    macro_count = len([e for e in (economic_calendar or []) if str(e.get("importance", "")).upper() in {"HIGH", "VERY HIGH"}])
    geo_count = len(geopolitical_news or [])
    catalyst = _safe_float((cfd_scenario or {}).get("catalyst_pressure_score"), 0)
    # High catalyst pressure is uncertainty, so it mostly reduces confidence rather than direction.
    caution = clip(catalyst / 100.0)

    # Determine directional classification from score, with a neutral band.
    if score >= 20:
        decision = "LONG BIAS"
        action = "BULLISH SCENARIO"
    elif score <= -20:
        decision = "SHORT BIAS"
        action = "BEARISH SCENARIO"
    else:
        decision = "NO CLEAR DIRECTION"
        action = "WAIT / MIXED EVIDENCE"

    strength = abs(score)
    confidence = "HIGH" if strength >= 45 and caution < 0.65 else "MEDIUM" if strength >= 20 else "LOW"
    if caution >= 0.75:
        confidence = "LOW — HIGH CATALYST RISK"

    return {
        "decision": decision,
        "action": action,
        "score": round(score, 1),
        "confidence": confidence,
        "catalyst_pressure": round(catalyst, 1),
        "high_impact_events": macro_count,
        "geopolitical_headlines": geo_count,
        "components": score_parts,
        "long_case": long_case,
        "short_case": short_case,
        "valuation_gaps": {
            "dcf": round(pct_gap(fair_value)*100,2),
            "composite": round(pct_gap(composite_target)*100,2),
            "ml": round(pct_gap(ml_target)*100,2),
            "analyst": round(pct_gap(analyst_target)*100,2),
        },
        "methodology": "Research-bias score from valuation (30%), trend (15%), RSI (10%), MACD (10%), fundamentals (10%), CAPM (5%), Sharpe (5%), volume confirmation (5%) and asset news (5%). Macro/geopolitical catalyst pressure changes confidence rather than directly forcing long/short. Not a probability-of-profit model or execution signal."
    }



# ------------------------------------------------------------------------------
# PORTFOLIO OPTIMIZATION ENGINE
# ------------------------------------------------------------------------------
def _normalize_ticker_list(raw: str | list[str]) -> list[str]:
    if isinstance(raw, str):
        parts = raw.replace("\n", ",").replace(";", ",").split(",")
    else:
        parts = raw or []
    out = []
    seen = set()
    for item in parts:
        sym = str(item).strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _portfolio_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(weights @ cov @ weights)


def _portfolio_random_fallback(mu: np.ndarray, cov: np.ndarray, rf: float, method: str,
                               n_samples: int = 12000) -> np.ndarray:
    n = len(mu)
    if n == 1:
        return np.array([1.0])
    rng = np.random.default_rng(42)
    # Uniform long-only simplex samples; works even if scipy is unavailable.
    weights = rng.dirichlet(np.ones(n), size=max(1000, min(int(n_samples), 20000)))
    vols = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', weights, cov, weights), 1e-18))
    rets = weights @ mu
    if method == 'MIN_RISK':
        idx = int(np.argmin(vols))
    else:
        sharpes = (rets - rf) / vols
        idx = int(np.nanargmax(sharpes))
    return weights[idx]


def _solve_long_only_portfolio(mu: np.ndarray, cov: np.ndarray, rf: float, method: str) -> np.ndarray:
    n = len(mu)
    x0 = np.full(n, 1.0 / n)
    if n == 1:
        return x0

    if _scipy_minimize is not None:
        bounds = [(0.0, 1.0)] * n
        constraints = ({'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)},)
        if method == 'MIN_RISK':
            fun = lambda w: _portfolio_objective(w, cov)
        else:
            def fun(w):
                variance = _portfolio_objective(w, cov)
                vol = max(np.sqrt(variance), 1e-10)
                return -float((w @ mu - rf) / vol)
        try:
            result = _scipy_minimize(fun, x0, method='SLSQP', bounds=bounds,
                                     constraints=constraints,
                                     options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
            if result.success and np.all(np.isfinite(result.x)):
                w = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
                s = w.sum()
                if s > 0:
                    return w / s
        except Exception:
            pass
    return _portfolio_random_fallback(mu, cov, rf, method)


def _efficient_frontier_samples(mu: np.ndarray, cov: np.ndarray, rf: float,
                                base_weights: np.ndarray, n_samples: int = 12000) -> list[dict]:
    """Create a long-only efficient-frontier approximation from simplex portfolios.

    The exact optimization uses the selected minimum-risk/tangency method. The
    visible frontier is a dense long-only simulation envelope so it remains
    responsive for arbitrary asset counts. Points are filtered into the upper
    return envelope (non-dominated by lower volatility and higher/equal return).
    """
    n = len(mu)
    if n == 1:
        vol = float(np.sqrt(max(cov[0,0], 0)))
        return [{'volatility':vol, 'return':float(mu[0]), 'sharpe':(float(mu[0])-rf)/vol if vol else 0.0}]
    rng = np.random.default_rng(7)
    n_samples = max(2000, min(int(n_samples), 25000))
    sims = rng.dirichlet(np.ones(n), size=n_samples)
    base_arr = np.asarray(base_weights, dtype=float)
    if base_arr.ndim == 1:
        base_arr = base_arr.reshape(1, -1)
    # Include equal-weight plus exact optimizer solutions in the simulation cloud.
    sims = np.vstack([sims, np.full((1, n), 1/n), base_arr])
    p_ret = sims @ mu
    p_var = np.maximum(np.einsum('ij,jk,ik->i', sims, cov, sims), 0.0)
    p_vol = np.sqrt(p_var)
    p_sharpe = np.where(p_vol > 0, (p_ret - rf) / p_vol, np.nan)

    order = np.argsort(p_vol)
    frontier = []
    best_return = -np.inf
    for idx in order:
        r = float(p_ret[idx]); v = float(p_vol[idx])
        if r > best_return + 1e-5:
            frontier.append({'volatility':v, 'return':r,
                             'sharpe':float(p_sharpe[idx]) if np.isfinite(p_sharpe[idx]) else 0.0})
            best_return = r
    # Downsample for the browser while preserving endpoints.
    if len(frontier) > 240:
        keep_idx = np.linspace(0, len(frontier)-1, 240).astype(int)
        frontier = [frontier[i] for i in keep_idx]
    return frontier


@_timed('portfolio')
def optimize_portfolio(tickers: str | list[str], method: str = 'MIN_RISK',
                       lookback: str = '5y', benchmark_symbol: str = '^GSPC') -> dict:
    """Estimate long-only portfolio weights for an arbitrary user-selected basket.

    METHODS
    - MIN_RISK: minimum-variance portfolio, sum(weights)=1, weights>=0.
    - TANGENCY: max Sharpe portfolio using the same constraints.
    - COMPARE: calculates both and returns the selected table for comparison UI.

    Expected returns are annualized arithmetic daily returns (mean * 252),
    covariance is annualized daily covariance, and the risk-free rate matches
    the platform CAPM assumption. This is a model/simulation, not a trade order.
    """
    symbols = _normalize_ticker_list(tickers)
    if not symbols:
        raise ValueError('Enter at least one asset ticker.')
    method = str(method or 'MIN_RISK').upper().replace('-', '_').replace(' ', '_')
    if method not in {'MIN_RISK','TANGENCY','COMPARE'}:
        raise ValueError('method must be MIN_RISK, TANGENCY, or COMPARE')

    # No hard-coded asset-count ceiling: every distinct ticker supplied is
    # processed. Practical limits are determined by provider latency/memory.
    close_map = {}
    names = {}
    failed = []
    _prefetch([lambda s=sym: _ticker(s).history(period=lookback, auto_adjust=True) for sym in symbols] +
              [lambda s=sym: _ticker(s).info for sym in symbols] +
              [lambda: _ticker(benchmark_symbol).history(period=lookback, auto_adjust=True)])
    for sym in symbols:
        try:
            obj = _ticker(sym)
            hist = obj.history(period=lookback, auto_adjust=True)
            if hist is None or hist.empty or 'Close' not in hist:
                failed.append({'ticker':sym,'reason':'No price history returned'})
                continue
            s = pd.to_numeric(hist['Close'], errors='coerce').dropna()
            if len(s) < 252:
                failed.append({'ticker':sym,'reason':'Less than 252 observations'})
                continue
            close_map[sym] = s
            try:
                info = obj.info or {}
                names[sym] = info.get('shortName') or info.get('longName') or sym
            except Exception:
                names[sym] = sym
        except Exception as exc:
            failed.append({'ticker':sym,'reason':str(exc)[:140]})

    if not close_map:
        raise ValueError('No selected assets returned sufficient price history.')

    prices = pd.concat(close_map, axis=1, join='inner').dropna()
    if len(prices) < 252:
        raise ValueError('The selected assets do not have enough common history for portfolio analysis.')

    returns = prices.pct_change().dropna()
    # Remove columns that collapsed after alignment.
    good = [c for c in returns.columns if returns[c].count() >= 252 and returns[c].std(ddof=1) > 0]
    returns = returns[good]
    if returns.empty:
        raise ValueError('No assets have usable non-zero return variance.')

    symbols = list(returns.columns)
    try:
        benchmark = _ticker(benchmark_symbol).history(period=lookback, auto_adjust=True)['Close']
        benchmark = pd.to_numeric(benchmark, errors='coerce').dropna()
    except Exception:
        benchmark = pd.Series(dtype=float)
    if benchmark.empty:
        benchmark = pd.Series(index=returns.index, dtype=float)
    joined = pd.concat([returns, benchmark.rename('market_price')], axis=1).dropna()
    if 'market_price' in joined and len(joined) >= 252:
        market_returns = joined['market_price'].pct_change().dropna()
    else:
        market_returns = pd.Series(dtype=float)

    mu = returns.mean().to_numpy(dtype=float) * 252.0
    cov = returns.cov().to_numpy(dtype=float) * 252.0
    # Ridge regularization keeps covariance invertible/stable when the basket
    # contains highly correlated assets or more assets than independent factors.
    avg_var = float(np.trace(cov) / max(len(symbols), 1))
    ridge = max(avg_var * 1e-5, 1e-8)
    cov = np.asarray(cov, dtype=float) + np.eye(len(symbols)) * ridge

    if not market_returns.empty:
        market_ann = float(market_returns.mean() * 252.0)
        market_var = float(market_returns.var(ddof=1))
        common_idx = returns.index.intersection(market_returns.index)
        betas = []
        for sym in symbols:
            r = returns.loc[common_idx, sym]
            m = market_returns.loc[common_idx]
            mv = float(m.var(ddof=1))
            betas.append(float(r.cov(m) / mv) if mv > 0 else 0.0)
        betas = np.asarray(betas, dtype=float)
    else:
        market_ann = 0.0
        betas = np.zeros(len(symbols), dtype=float)

    rf = _risk_free_rate()
    min_w = _solve_long_only_portfolio(mu, cov, rf, 'MIN_RISK')
    tangency_w = _solve_long_only_portfolio(mu, cov, rf, 'TANGENCY')
    selected_w = min_w if method == 'MIN_RISK' else tangency_w

    def metrics(w):
        ret = float(w @ mu)
        var = float(w @ cov @ w)
        vol = float(np.sqrt(max(var, 0.0)))
        sharpe = float((ret-rf)/vol) if vol > 0 else 0.0
        beta = float(w @ betas)
        return {'expected_return':ret,'variance':var,'volatility':vol,'sharpe':sharpe,'beta':beta}

    min_m = metrics(min_w); tan_m = metrics(tangency_w); sel_m = metrics(selected_w)
    frontier = _efficient_frontier_samples(mu, cov, rf, np.vstack([selected_w, min_w, tangency_w]))

    asset_vol = returns.std(ddof=1).to_numpy(dtype=float) * np.sqrt(252)
    rows=[]
    for i,sym in enumerate(symbols):
        rows.append({
            'ticker':sym,
            'name':names.get(sym,sym),
            'weight':round(float(selected_w[i]),6),
            'min_risk_weight':round(float(min_w[i]),6),
            'tangency_weight':round(float(tangency_w[i]),6),
            'expected_return':round(float(mu[i]),6),
            'volatility':round(float(asset_vol[i]),6),
            'variance':round(float(cov[i,i]),8),
            'beta':round(float(betas[i]),4),
            'asset_sharpe':round(float((mu[i]-rf)/asset_vol[i]),4) if asset_vol[i] > 0 else 0.0,
            'return_contribution':round(float(selected_w[i]*mu[i]),6),
            'risk_contribution':round(float(selected_w[i]*(cov @ selected_w)[i] / max(sel_m['variance'],1e-12)),6),
        })
    # Keep all requested valid assets; table is sorted by selected weight for readability.
    rows.sort(key=lambda x: x['weight'], reverse=True)

    beta_axis = np.linspace(float(min(betas.min(), -0.25)), float(max(betas.max(), 2.0)), 80)
    sml_return = rf + beta_axis*(market_ann-rf)
    # The CAL is anchored to the tangency portfolio by definition; the selected
    # method is still shown separately on the chart/table.
    cal_vols = np.linspace(0.0, max(float(tan_m['volatility'])*1.8, float(sel_m['volatility'])*1.2, 0.30), 80)
    cal_returns = rf + tan_m['sharpe']*cal_vols

    scatter = [{
        'ticker':s['ticker'],'weight':s['weight'],
        'expected_return':s['expected_return'],'volatility':s['volatility'],'beta':s['beta']
    } for s in rows]

    return {
        'method':method,
        'lookback':lookback,
        'benchmark':benchmark_symbol,
        'risk_free_rate':rf,
        'assets_requested':len(_normalize_ticker_list(tickers)),
        'assets_used':len(symbols),
        'failed_assets':failed,
        'data_start':str(prices.index.min().date()),
        'data_end':str(prices.index.max().date()),
        'selected':{
            'expected_return':round(sel_m['expected_return'],6),
            'variance':round(sel_m['variance'],8),
            'volatility':round(sel_m['volatility'],6),
            'sharpe':round(sel_m['sharpe'],4),
            'beta':round(sel_m['beta'],4),
            'weight_sum':round(float(selected_w.sum()),6),
        },
        'minimum_risk':{**{k:round(v,6) for k,v in min_m.items()},'weight_sum':round(float(min_w.sum()),6)},
        'tangency':{**{k:round(v,6) for k,v in tan_m.items()},'weight_sum':round(float(tangency_w.sum()),6)},
        'market':{'annual_return':round(float(market_ann),6)},
        'model_inputs':{'tickers':symbols,'annual_returns':mu.tolist(),'annual_covariance':cov.tolist(),'betas':betas.tolist()},
        'rows':rows,
        'scatter':scatter,
        'sml':{'beta':beta_axis.tolist(),'return':sml_return.tolist()},
        'cal':{'volatility':cal_vols.tolist(),'return':cal_returns.tolist()},
        'efficient_frontier':frontier,
        'methodology':(
            'Long-only mean-variance portfolio simulation. MIN_RISK minimizes annualized portfolio variance. '
            'TANGENCY maximizes estimated Sharpe ratio against the platform risk-free assumption. '
            'Weights sum to 100% and no short weights are allowed in this portfolio optimizer. '
            'Expected returns use arithmetic daily returns annualized by 252; covariance uses daily returns annualized by 252. '
            'The efficient frontier shown is a long-only simulation envelope. Results are estimates, not guarantees or execution instructions.'
        )
    }


@_timed('scoring')
def build_fundamental_analysis(
    info: dict,
    current_price: float,
    financial_ratios: dict,
    ratio_history: list[dict],
    financial_statements: dict,
    fair_value: float,
    peer_target: float,
    peer_multiple: float,
    analyst_target: float,
    composite_target: float,
    ml_target: float,
    peers: list[dict],
    risk_metrics: dict,
    news: list[dict],
    economic_calendar: list[dict],
    geopolitical_news: list[dict],
) -> dict:
    """Integrated fundamental research view for a selected asset.

    Combines annual statements, 5Y ratios, valuation cross-checks, peer context,
    cash flow, balance-sheet health, growth, capital efficiency and selected risk/
    catalyst context. Scores are research diagnostics, not trade signals.
    """
    def finite(v):
        try:
            return v is not None and np.isfinite(float(v))
        except Exception:
            return False

    def clamp(v, lo=0.0, hi=100.0):
        return float(max(lo, min(hi, v)))

    def norm_centered(v, center=0.0, scale=0.20):
        if not finite(v):
            return 50.0
        return clamp(50.0 + float(v-center) / scale * 50.0)

    def latest(history, key):
        if not history:
            return None
        v = history[0].get(key)
        return float(v) if finite(v) else None

    # ------------------------------------------------------------------
    # Core latest ratios
    # ------------------------------------------------------------------
    roic = latest(ratio_history, 'roic')
    roa = latest(ratio_history, 'roa')
    roe = latest(ratio_history, 'roe')
    gross_margin = latest(ratio_history, 'gross_margin')
    op_margin = latest(ratio_history, 'operating_margin')
    net_margin = latest(ratio_history, 'net_margin')
    current_ratio = latest(ratio_history, 'current_ratio')
    quick_ratio = latest(ratio_history, 'quick_ratio')
    cash_ratio = latest(ratio_history, 'cash_ratio')
    debt_equity = latest(ratio_history, 'debt_to_equity')
    interest_coverage = latest(ratio_history, 'interest_coverage')
    asset_turnover = latest(ratio_history, 'asset_turnover')
    fcf_yield = latest(ratio_history, 'fcf_yield')

    # ------------------------------------------------------------------
    # Statement-derived data / growth
    # ------------------------------------------------------------------
    fs = financial_statements or {}
    years = list(fs.get('years') or [])
    income_rows = {r.get('metric'): r.get('values', []) for r in fs.get('income_statement', [])}
    balance_rows = {r.get('metric'): r.get('values', []) for r in fs.get('balance_sheet', [])}
    cash_rows = {r.get('metric'): r.get('values', []) for r in fs.get('cash_flow', [])}

    revenue_values = income_rows.get('Revenue', [])
    net_income_values = income_rows.get('Net Income', [])
    fcf_values = cash_rows.get('Free Cash Flow', [])
    debt_values = balance_rows.get('Total Debt', [])

    def yoy(values):
        vals = []
        for i in range(max(len(values)-1, 0)):
            a, b = values[i], values[i+1]
            if finite(a) and finite(b) and float(b) != 0:
                vals.append((float(a)/float(b))-1.0)
            else:
                vals.append(None)
        return vals

    def cagr(values):
        usable = [float(v) for v in values if finite(v)]
        if len(usable) < 2 or usable[-1] <= 0 or usable[0] <= 0:
            return None
        years_span = max(len(usable)-1, 1)
        return (usable[0] / usable[-1]) ** (1.0 / years_span) - 1.0

    revenue_growth_yoy = yoy(revenue_values)
    net_income_growth_yoy = yoy(net_income_values)
    fcf_growth_yoy = yoy(fcf_values)
    debt_growth_yoy = yoy(debt_values)

    # History is latest -> oldest, so CAGR uses latest / oldest.
    def cagr_latest_to_oldest(values):
        usable = [float(v) for v in values if finite(v)]
        if len(usable) < 2 or usable[0] <= 0 or usable[-1] <= 0:
            return None
        n = len(usable)-1
        return (usable[0] / usable[-1]) ** (1.0 / n) - 1.0

    revenue_cagr = cagr_latest_to_oldest(revenue_values)
    net_income_cagr = cagr_latest_to_oldest(net_income_values)
    fcf_positive_years = sum(1 for v in fcf_values if finite(v) and float(v) > 0)

    # ------------------------------------------------------------------
    # Valuation cross-check
    # ------------------------------------------------------------------
    valuation_targets = {
        'DCF': float(fair_value) if finite(fair_value) else None,
        'Peer target': float(peer_target) if finite(peer_target) else None,
        'ML forecast': float(ml_target) if finite(ml_target) else None,
        'Analyst target': float(analyst_target) if finite(analyst_target) else None,
        'Composite': float(composite_target) if finite(composite_target) else None,
    }
    valuation_gaps = {k: ((v-current_price)/current_price if finite(v) and current_price else None)
                      for k, v in valuation_targets.items()}
    gap_values = [v for v in valuation_gaps.values() if finite(v)]
    valuation_gap_avg = float(np.mean(gap_values)) if gap_values else 0.0
    valuation_gap_median = float(np.median(gap_values)) if gap_values else 0.0

    forward_pe = info.get('forwardPE')
    trailing_pe = info.get('trailingPE')
    price_to_book = info.get('priceToBook')
    ev_to_ebitda = info.get('enterpriseToEbitda')
    price_to_sales = info.get('priceToSalesTrailing12Months')
    peg_ratio = info.get('pegRatio')

    peer_forward_pe = [p.get('forward_pe') for p in (peers or []) if finite(p.get('forward_pe')) and float(p.get('forward_pe')) > 0]
    peer_forward_median = float(np.median(peer_forward_pe)) if peer_forward_pe else None
    multiple_relative_gap = None
    if finite(forward_pe) and peer_forward_median and peer_forward_median > 0:
        multiple_relative_gap = float(forward_pe)/peer_forward_median - 1.0

    # ------------------------------------------------------------------
    # Factor scores (0-100). Weights are visible in methodology.
    # ------------------------------------------------------------------
    profitability = np.mean([
        norm_centered(roic, center=0.10, scale=0.10),
        norm_centered(roe, center=0.15, scale=0.15),
        norm_centered(net_margin, center=0.10, scale=0.15),
    ]) if any(finite(x) for x in [roic, roe, net_margin]) else 50.0

    growth_inputs = [
        norm_centered(revenue_cagr, 0.08, 0.15),
        norm_centered(net_income_cagr, 0.08, 0.20),
        norm_centered(revenue_growth_yoy[0] if revenue_growth_yoy else None, 0.05, 0.15),
    ]
    growth = float(np.mean(growth_inputs))

    liquidity = norm_centered(current_ratio, 1.5, 1.5)
    quick = norm_centered(quick_ratio, 1.0, 1.0)
    cash_score = norm_centered(cash_ratio, 0.25, 0.50)
    leverage_score = 70.0 if not finite(debt_equity) else clamp(100.0 - float(debt_equity)/3.0*100.0)
    coverage_score = 50.0 if not finite(interest_coverage) else clamp(float(interest_coverage)/12.0*100.0)
    balance_sheet = float(np.mean([liquidity, quick, cash_score, leverage_score, coverage_score]))

    fcf_score = 50.0
    if fcf_values:
        positive_ratio = fcf_positive_years / max(len(fcf_values), 1)
        latest_fcf = next((float(v) for v in fcf_values if finite(v)), None)
        fcf_score = clamp(positive_ratio*70.0 + (30.0 if latest_fcf is not None and latest_fcf > 0 else 0.0))
    cashflow = fcf_score

    efficiency = float(np.mean([
        norm_centered(asset_turnover, 1.0, 0.8),
        norm_centered(roic, 0.10, 0.10),
        norm_centered(gross_margin, 0.35, 0.25),
    ])) if any(finite(x) for x in [asset_turnover, roic, gross_margin]) else 50.0

    valuation_score = clamp(50.0 + valuation_gap_avg * 200.0)
    if multiple_relative_gap is not None:
        # Relative multiple is a secondary cross-check: cheaper than peers lifts the score.
        valuation_score = clamp(0.75*valuation_score + 0.25*clamp(50.0 - multiple_relative_gap*100.0))

    factor_scores = {
        'Valuation': round(valuation_score, 1),
        'Profitability': round(float(profitability), 1),
        'Growth': round(growth, 1),
        'Balance Sheet': round(balance_sheet, 1),
        'Cash Flow': round(cashflow, 1),
        'Efficiency': round(efficiency, 1),
    }
    weights = {'Valuation':0.25,'Profitability':0.20,'Growth':0.15,'Balance Sheet':0.15,'Cash Flow':0.15,'Efficiency':0.10}
    fundamental_score = float(sum(factor_scores[k]*weights[k] for k in factor_scores))

    # ------------------------------------------------------------------
    # Context from the broader platform (not folded into the fundamental score)
    # ------------------------------------------------------------------
    beta = risk_metrics.get('beta')
    sharpe = risk_metrics.get('sharpe_ratio')
    volatility = risk_metrics.get('std_dev_5y')
    high_events = sum(1 for e in economic_calendar if str(e.get('importance','')).upper() in {'HIGH','VERY HIGH'})

    strengths = []
    risks = []
    watch = []
    if finite(roic) and roic > 0.12: strengths.append(f"ROIC is {roic*100:.1f}%.")
    elif finite(roic) and roic < 0: risks.append(f"ROIC is negative at {roic*100:.1f}%.")
    if finite(roe) and roe > 0.15: strengths.append(f"ROE is {roe*100:.1f}%.")
    if finite(debt_equity) and debt_equity < 1.0: strengths.append(f"Debt-to-equity is {debt_equity:.2f}x.")
    elif finite(debt_equity) and debt_equity > 2.0: risks.append(f"Debt-to-equity is elevated at {debt_equity:.2f}x.")
    if finite(interest_coverage) and interest_coverage > 5: strengths.append(f"Interest coverage is {interest_coverage:.1f}x.")
    elif finite(interest_coverage) and interest_coverage < 2: risks.append(f"Interest coverage is only {interest_coverage:.1f}x.")
    if revenue_cagr is not None and revenue_cagr > 0.08: strengths.append(f"Revenue CAGR over available annual history is {revenue_cagr*100:.1f}%.")
    elif revenue_cagr is not None and revenue_cagr < 0: risks.append(f"Revenue CAGR over available annual history is {revenue_cagr*100:.1f}%.")
    if fcf_positive_years >= max(2, len(fcf_values)-1): strengths.append(f"Positive FCF in {fcf_positive_years}/{len(fcf_values)} available fiscal years.")
    elif fcf_values and fcf_positive_years == 0: risks.append("No positive FCF in the available fiscal-year history.")
    if valuation_gap_median > 0.05: strengths.append(f"Median valuation cross-check is {valuation_gap_median*100:.1f}% above spot.")
    elif valuation_gap_median < -0.05: risks.append(f"Median valuation cross-check is {valuation_gap_median*100:.1f}% below spot.")
    if peer_forward_median and finite(forward_pe):
        watch.append(f"Forward P/E {float(forward_pe):.1f}x vs peer median {peer_forward_median:.1f}x.")
    if high_events: watch.append(f"{high_events} high-impact macro event(s) are on the platform calendar.")
    if news: watch.append(f"{len(news)} recent asset headlines are available for review.")
    if geopolitical_news: watch.append(f"{len(geopolitical_news)} geopolitical headlines are available for review.")
    if finite(beta) and float(beta) > 1.4: watch.append(f"5Y beta is elevated at {float(beta):.2f}.")
    if finite(volatility): watch.append(f"5Y annualized volatility is {float(volatility)*100:.1f}%.")

    # A compact quality/data coverage indicator.
    coverage_items = [
        bool(years), len(years) >= 3,
        bool(ratio_history),
        any(finite(x) for x in [fair_value, peer_target, analyst_target, composite_target]),
        bool(peers),
        bool(fcf_values),
    ]
    coverage = round(sum(1 for x in coverage_items if x)/len(coverage_items)*100, 0)

    return {
        'score': round(fundamental_score, 1),
        'rating_band': 'Strong fundamentals' if fundamental_score >= 75 else 'Constructive fundamentals' if fundamental_score >= 60 else 'Mixed fundamentals' if fundamental_score >= 45 else 'Weak fundamentals',
        'data_coverage': coverage,
        'factor_scores': factor_scores,
        'weights': weights,
        'valuation': {
            'current_price': round(current_price, 4),
            'dcf': round(fair_value, 4) if finite(fair_value) else None,
            'peer_target': round(peer_target, 4) if finite(peer_target) else None,
            'ml_target': round(ml_target, 4) if finite(ml_target) else None,
            'analyst_target': round(analyst_target, 4) if finite(analyst_target) else None,
            'composite_target': round(composite_target, 4) if finite(composite_target) else None,
            'gap_average': round(valuation_gap_avg, 4),
            'gap_median': round(valuation_gap_median, 4),
            'forward_pe': float(forward_pe) if finite(forward_pe) else None,
            'trailing_pe': float(trailing_pe) if finite(trailing_pe) else None,
            'price_to_book': float(price_to_book) if finite(price_to_book) else None,
            'enterprise_to_ebitda': float(ev_to_ebitda) if finite(ev_to_ebitda) else None,
            'price_to_sales': float(price_to_sales) if finite(price_to_sales) else None,
            'peg_ratio': float(peg_ratio) if finite(peg_ratio) else None,
            'peer_forward_pe_median': peer_forward_median,
        },
        'profitability': {
            'roic': roic, 'roa': roa, 'roe': roe,
            'gross_margin': gross_margin, 'operating_margin': op_margin, 'net_margin': net_margin,
        },
        'growth': {
            'revenue_cagr': revenue_cagr,
            'net_income_cagr': net_income_cagr,
            'revenue_yoy': revenue_growth_yoy[0] if revenue_growth_yoy else None,
            'net_income_yoy': net_income_growth_yoy[0] if net_income_growth_yoy else None,
            'fcf_yoy': fcf_growth_yoy[0] if fcf_growth_yoy else None,
            'debt_yoy': debt_growth_yoy[0] if debt_growth_yoy else None,
            'years': years,
        },
        'balance_sheet': {
            'current_ratio': current_ratio, 'quick_ratio': quick_ratio, 'cash_ratio': cash_ratio,
            'debt_to_equity': debt_equity, 'interest_coverage': interest_coverage,
        },
        'cash_flow': {
            'positive_fcf_years': fcf_positive_years,
            'available_fcf_years': len(fcf_values),
            'latest_fcf': next((float(v) for v in fcf_values if finite(v)), None),
            'fcf_yield': fcf_yield,
        },
        'efficiency': {'asset_turnover': asset_turnover},
        'risk_context': {'beta': beta, 'sharpe_ratio': sharpe, 'volatility_5y': volatility},
        'valuation_cross_check': [
            {'method': k, 'target': valuation_targets[k], 'upside_percent': (valuation_gaps[k]*100 if finite(valuation_gaps[k]) else None)}
            for k in valuation_targets
        ],
        'trends': {
            'years': list(reversed(years)),
            'revenue_yoy': list(reversed(revenue_growth_yoy)),
            'net_income_yoy': list(reversed(net_income_growth_yoy)),
            'fcf_yoy': list(reversed(fcf_growth_yoy)),
            'debt_yoy': list(reversed(debt_growth_yoy)),
        },
        'strengths': strengths[:8],
        'risks': risks[:8],
        'watch_items': watch[:8],
        'methodology': 'Integrated fundamental research score: Valuation 25%, Profitability 20%, Growth 15%, Balance Sheet 15%, Cash Flow 15%, Efficiency 10%. The score uses the selected company\'s annual financial statements, 5Y ratio history, valuation cross-checks, peer multiples and cash-flow evidence. News, macro, geopolitical and market-risk measures are shown as context and do not directly change the fundamental score. This is a research diagnostic, not a trade signal or guaranteed outcome.'
    }


# ------------------------------------------------------------------------------
# MULTI-ASSET TICKER ANALYSIS
# ------------------------------------------------------------------------------
# Yahoo Finance exposes equities, ETFs, FX pairs, futures, rates/indexes and
# crypto through the same ticker interface. Issuer financial statements and
# peer multiples are not meaningful for many of those instruments, so the
# ticker-analysis endpoint switches to an asset-class-aware market pipeline.

MULTI_ASSET_QUOTE_MAP = {
    # Currencies
    'EURUSD=X': ('Currency', 'EUR/USD', 'USD per EUR'),
    'GBPUSD=X': ('Currency', 'GBP/USD', 'USD per GBP'),
    'USDJPY=X': ('Currency', 'USD/JPY', 'JPY per USD'),
    'AUDUSD=X': ('Currency', 'AUD/USD', 'USD per AUD'),
    'USDCNY=X': ('Currency', 'USD/CNY', 'CNY per USD'),
    'USDCHF=X': ('Currency', 'USD/CHF', 'CHF per USD'),
    # Metals
    'GC=F': ('Metal', 'Gold Futures', 'USD per troy oz'),
    'SI=F': ('Metal', 'Silver Futures', 'USD per troy oz'),
    'PL=F': ('Metal', 'Platinum Futures', 'USD per troy oz'),
    'HG=F': ('Metal', 'Copper Futures', 'USD per lb'),
    # Energy
    'CL=F': ('Energy', 'WTI Crude Oil', 'USD per barrel'),
    'BZ=F': ('Energy', 'Brent Crude', 'USD per barrel'),
    'NG=F': ('Energy', 'Natural Gas', 'USD per MMBtu'),
    # Agriculture / softs
    'ZC=F': ('Commodity', 'Corn Futures', 'USD per bushel'),
    'ZW=F': ('Commodity', 'Wheat Futures', 'USD per bushel'),
    'ZS=F': ('Commodity', 'Soybean Futures', 'USD per bushel'),
    'KC=F': ('Commodity', 'Coffee Futures', 'USD per lb'),
    # Rates / index-like instruments
    '^TNX': ('Rate', 'US 10Y Treasury Yield', 'percentage points'),
    '^FVX': ('Rate', 'US 5Y Treasury Yield', 'percentage points'),
    '^TYX': ('Rate', 'US 30Y Treasury Yield', 'percentage points'),
    # Crypto
    'BTC-USD': ('Crypto', 'Bitcoin', 'USD'),
    'ETH-USD': ('Crypto', 'Ethereum', 'USD'),
}


def classify_ticker_asset(ticker_symbol: str, info: dict | None = None) -> tuple[str, str, str]:
    symbol = str(ticker_symbol or '').strip().upper()
    if symbol in MULTI_ASSET_QUOTE_MAP:
        return MULTI_ASSET_QUOTE_MAP[symbol]
    info = info or {}
    quote_type = str(info.get('quoteType') or '').upper()
    if quote_type in {'CRYPTOCURRENCY', 'CRYPTO'} or symbol.endswith('-USD'):
        name = info.get('shortName') or symbol
        return 'Crypto', name, 'USD'
    if symbol.endswith('=X'):
        name = info.get('shortName') or symbol.replace('=X', '')
        return 'Currency', name, 'FX quote'
    if symbol.endswith('=F'):
        name = info.get('shortName') or symbol
        return 'Futures / Commodity', name, info.get('currency') or 'quoted contract price'
    if symbol.startswith('^'):
        name = info.get('shortName') or symbol
        return 'Index / Rate', name, info.get('currency') or 'index points'
    if quote_type in {'ETF', 'MUTUALFUND'}:
        return 'ETF / Fund', info.get('shortName') or symbol, info.get('currency') or 'USD'
    return 'Equity', info.get('shortName') or symbol, info.get('financialCurrency') or info.get('currency') or 'USD'


def _latest_yf_price(ticker_obj: yf.Ticker, info: dict, history: pd.DataFrame | None = None) -> float | None:
    price = None
    try:
        price = ticker_obj.fast_info.get('last_price')
    except Exception:
        pass
    if not price:
        price = info.get('currentPrice') or info.get('regularMarketPrice')
    if not price and history is not None and not history.empty:
        try:
            price = float(history['Close'].dropna().iloc[-1])
        except Exception:
            price = None
    try:
        value = float(price)
        return value if np.isfinite(value) and value > 0 else None
    except Exception:
        return None


def _generic_ml_fallback(history: pd.DataFrame, current_price: float) -> tuple[dict, dict]:
    close = pd.to_numeric(history['Close'], errors='coerce').dropna().astype(float)
    if close.empty:
        raise ValueError('No usable close history for the selected asset.')
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    ret = close.pct_change()
    drift = float(ret.tail(60).mean()) if ret.tail(60).notna().sum() else 0.0
    predicted = float(current_price * max(0.01, (1.0 + drift) ** 20))
    confidence = float(max(0.0, min(60.0, 30.0 + ret.tail(120).std(ddof=1) * 100.0))) if len(ret) > 20 else 0.0
    dates = close.index.strftime('%Y-%m-%d').tolist()
    last_date = pd.to_datetime(dates[-1])
    forecast_dates = [dates[-1]] + [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 21)]
    forecast_prices = np.linspace(current_price, predicted, num=21).round(6).tolist()
    chart = {
        'dates': dates,
        'historical_close': close.round(6).tolist(),
        'sma_20': [],   # computed in the browser
        'sma_50': [],
        'ml_forecast_dates': forecast_dates,
        'ml_forecast_prices': forecast_prices,
    }
    ml = {
        'predicted_target_price': round(predicted, 6),
        'ml_implied_upside': round((predicted/current_price - 1.0) * 100.0, 2),
        'model_confidence_score': round(confidence, 2),
        'feature_metrics': {'pe_ratio': 0.0, 'profit_margins': 0.0, 'return_on_equity': 0.0, 'beta': 1.0},
    }
    return ml, chart


def _generic_market_fundamental_view(asset_class: str, technicals: dict, risk_metrics: dict,
                                     current_price: float, fair_value: float, ml_target: float,
                                     news: list[dict], economic_calendar: list[dict],
                                     geopolitical_news: list[dict], quote_unit: str) -> dict:
    rsi = _safe_float(technicals.get('rsi14'), 50.0)
    trend = str(technicals.get('trend') or 'mixed').lower()
    vol = _safe_float(risk_metrics.get('std_dev_5y'), 0.0)
    beta = _safe_float(risk_metrics.get('beta'), 1.0)
    score = float(np.clip(50 + (rsi-50)*0.55 + (8 if trend == 'bullish' else -8 if trend == 'bearish' else 0) - min(18, vol*20), 0, 100))
    gap = ((fair_value/current_price)-1.0) if current_price else 0.0
    factor_scores = {
        'Model level': round(float(np.clip(50 + gap*200, 0, 100)), 1),
        'Trend': round(65 if trend == 'bullish' else 35 if trend == 'bearish' else 50, 1),
        'Momentum': round(float(np.clip(rsi, 0, 100)), 1),
        'Volatility': round(float(np.clip(100 - vol*100, 0, 100)), 1),
        'Risk-adjusted': round(float(np.clip(50 + _safe_float(risk_metrics.get('sharpe_ratio'))*20, 0, 100)), 1),
        'Catalyst coverage': round(float(np.clip(50 + (len(news) + len(geopolitical_news))*2 - len([e for e in economic_calendar if str(e.get('importance','')).upper() in {'HIGH','VERY HIGH'}])*2, 0, 100)), 1),
    }
    strengths, risks, watch = [], [], []
    if trend == 'bullish': strengths.append(f'{asset_class} trend is above the 20/50-period moving-average structure.')
    elif trend == 'bearish': risks.append(f'{asset_class} trend is below the 20/50-period moving-average structure.')
    if rsi >= 60: strengths.append(f'RSI(14) is {rsi:.1f}, consistent with positive recent momentum.')
    elif rsi <= 40: risks.append(f'RSI(14) is {rsi:.1f}, indicating weak recent momentum.')
    if vol > 0.35: risks.append(f'5Y annualized volatility is elevated at {vol*100:.1f}%.')
    else: watch.append(f'5Y annualized volatility is {vol*100:.1f}%.')
    watch.append(f'Quoted in {quote_unit}.')
    if len(news): watch.append(f'{len(news)} recent asset headlines available.')
    if len(economic_calendar): watch.append(f'{len([e for e in economic_calendar if str(e.get("importance","")).upper() in {"HIGH","VERY HIGH"}])} high-impact macro event(s) on the calendar.')
    return {
        'score': round(score, 1),
        'rating_band': 'Market structure strong' if score >= 70 else 'Market structure constructive' if score >= 55 else 'Market structure mixed' if score >= 45 else 'Market structure weak',
        'data_coverage': 100,
        'asset_class': asset_class,
        'factor_scores': factor_scores,
        'weights': {'Model level':0.20,'Trend':0.20,'Momentum':0.20,'Volatility':0.15,'Risk-adjusted':0.15,'Catalyst coverage':0.10},
        'valuation': {
            'current_price': round(current_price, 6),
            'dcf': round(fair_value, 6),
            'peer_target': None,
            'ml_target': round(ml_target, 6),
            'analyst_target': None,
            'composite_target': round(fair_value, 6),
            'gap_average': round(gap, 6),
            'gap_median': round(gap, 6),
            'forward_pe': None, 'trailing_pe': None, 'price_to_book': None,
            'enterprise_to_ebitda': None, 'price_to_sales': None, 'peg_ratio': None,
            'peer_forward_pe_median': None,
        },
        'profitability': {'roic': None, 'roa': None, 'roe': None, 'gross_margin': None, 'operating_margin': None, 'net_margin': None},
        'growth': {'revenue_cagr': None, 'net_income_cagr': None, 'revenue_yoy': None, 'net_income_yoy': None, 'fcf_yoy': None, 'debt_yoy': None, 'years': []},
        'balance_sheet': {'current_ratio': None, 'quick_ratio': None, 'cash_ratio': None, 'debt_to_equity': None, 'interest_coverage': None},
        'cash_flow': {'positive_fcf_years': 0, 'available_fcf_years': 0, 'latest_fcf': None, 'fcf_yield': None},
        'efficiency': {'asset_turnover': None},
        'risk_context': {'beta': beta, 'sharpe_ratio': _safe_float(risk_metrics.get('sharpe_ratio')), 'volatility_5y': vol},
        'valuation_cross_check': [
            {'method': 'Model level', 'target': round(fair_value, 6), 'upside_percent': round(gap*100, 2)},
            {'method': 'ML forecast', 'target': round(ml_target, 6), 'upside_percent': round(((ml_target/current_price)-1)*100,2) if current_price else 0},
        ],
        'trends': {'years': [], 'revenue_yoy': [], 'net_income_yoy': [], 'fcf_yoy': [], 'debt_yoy': []},
        'strengths': strengths,
        'risks': risks,
        'watch_items': watch,
        'methodology': 'Asset-class-aware market analysis: moving-average structure, RSI, MACD, volatility, risk-adjusted history, statistical model level and catalyst/news context. Issuer financial statements and equity peer valuation are not applied to FX, commodities, rates or crypto.',
    }


def run_market_asset_pipeline(ticker_symbol: str, discount_rate: float, perpetual_growth: float) -> dict:
    if perpetual_growth >= discount_rate:
        raise ValueError('Discount rate (WACC) must be strictly greater than Perpetual Growth Rate.')
    symbol = ticker_symbol.strip().upper()
    ticker = _ticker(symbol)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    asset_class, display_name, quote_unit = classify_ticker_asset(symbol, info)
    with _stage("prefetch"):
        _prefetch([
            lambda: ticker.history(period='5y', auto_adjust=True), lambda: ticker.fast_info, lambda: ticker.news,
            lambda: _ticker('^GSPC').history(period='5y', auto_adjust=True), _risk_free_rate,
            lambda: fetch_geopolitical_news(symbol, display_name, info.get('sector') or asset_class),
        ])
    history = ticker.history(period='5y', auto_adjust=True)
    if history is None or history.empty or len(history) < 80:
        raise ValueError(f'Insufficient historical pricing data for {symbol}.')
    current_price = _latest_yf_price(ticker, info, history)
    if current_price is None:
        raise ValueError(f'Could not fetch price for {symbol}.')

    beta_hint = _safe_float(info.get('beta'), 1.0)
    risk_metrics = calculate_risk_and_portfolio_metrics(history, beta_hint, current_price)
    technicals = calculate_technicals(history)
    try:
        ml_results, chart_data = train_and_predict_ml_target(ticker, current_price)
    except Exception:
        ml_results, chart_data = _generic_ml_fallback(history, current_price)

    sma20 = _safe_float(technicals.get('sma20'), current_price)
    sma50 = _safe_float(technicals.get('sma50'), current_price)
    ml_target = _safe_float(ml_results.get('predicted_target_price'), current_price)
    # This is a statistical/reference price level, not intrinsic equity value.
    fair_value = float(np.average([sma20, sma50, ml_target], weights=[0.40, 0.25, 0.35]))
    fair_value = round(max(0.000001, fair_value), 6)
    composite_target = round(float(np.average([fair_value, ml_target], weights=[0.55, 0.45])), 6)
    model_gap = ((fair_value-current_price)/current_price)*100 if current_price else 0.0

    sector = info.get('sector') or asset_class
    news = fetch_asset_news(symbol, sector)
    economic_calendar = build_economic_calendar()
    geopolitical_news = fetch_geopolitical_news(symbol, display_name, sector)
    peer_row = {
        'ticker': symbol, 'name': display_name, 'price': round(current_price, 6),
        'forward_pe': 0.0, 'trailing_pe': 0.0, 'revenue_growth': 0.0,
        'profit_margin': 0.0, 'beta': round(_safe_float(risk_metrics.get('beta'), beta_hint), 3)
    }
    financial_ratios = {
        'roic': 0.0, 'current_ratio': 0.0, 'quick_ratio': 0.0, 'debt_to_equity': 0.0,
        'interest_coverage': 0.0, 'fcf_yield': 0.0, 'roa': 0.0, 'roe': 0.0,
        'gross_margin': 0.0, 'operating_margin': 0.0, 'net_margin': 0.0,
        'asset_turnover': 0.0, 'inventory_turnover': 0.0, 'receivables_turnover': 0.0, 'cash_ratio': 0.0,
    }
    trade_setup = calculate_trade_setup(
        current_price, technicals,
        fair_value_per_share=fair_value,
        peer_target_price=current_price,
        ml_target_price=ml_target,
        analyst_target_price=current_price,
        composite_target_price=composite_target,
        beta=_safe_float(risk_metrics.get('beta'), beta_hint)
    )
    cfd_scenario = build_cfd_scenario(
        symbol, display_name, sector, current_price, technicals,
        risk_metrics, trade_setup, news, economic_calendar, geopolitical_news
    )
    directional_decision = make_directional_decision(
        current_price, fair_value, composite_target, ml_target, current_price,
        technicals, risk_metrics, financial_ratios, cfd_scenario, news,
        economic_calendar, geopolitical_news
    )
    fundamental_analysis = _generic_market_fundamental_view(
        asset_class, technicals, risk_metrics, current_price, fair_value, ml_target,
        news, economic_calendar, geopolitical_news, quote_unit
    )
    return {
        'ticker': symbol,
        'company_name': display_name,
        'sector': sector,
        'asset_class': asset_class,
        'analysis_mode': 'MARKET_ASSET',
        'quote_unit': quote_unit,
        'current_price': round(current_price, 6),
        'fair_value_per_share': round(fair_value, 6),
        'dcf_implied_upside_percent': round(model_gap, 2),
        'currency': info.get('financialCurrency') or info.get('currency') or 'USD',
        'inputs': {
            'discount_rate': discount_rate,
            'perpetual_growth': perpetual_growth,
            'projected_growth_rate': 0.0,
            'latest_free_cash_flow': 0.0,
        },
        'ml_forecast': ml_results,
        'chart_data': chart_data,
        'technicals': technicals,
        'risk_metrics': risk_metrics,
        'financial_ratios': financial_ratios,
        'altman': {"score": None, "zone": "N/A", "status": "Not applicable to non-equity asset classes", "model": "Altman Z-Score (classic public-company model)", "year": None, "components": {}},
        'peers': [peer_row],
        'peer_target_price': round(current_price, 6),
        'peer_multiple': 0.0,
        'peer_note': 'Peer multiples are not applicable to this asset class; the peer row is retained only to keep the shared ticker-analysis interface consistent.',
        'analyst_target_price': round(current_price, 6),
        'analyst_count': 0,
        'composite_target_price': composite_target,
        'composite_implied_upside_percent': round(((composite_target-current_price)/current_price)*100, 2),
        'methodology': 'Multi-asset market analysis for non-equity instruments. Uses historical price structure, SMA20/SMA50, RSI, MACD, ATR, volatility, market-risk context, statistical reference level, ML forecast and macro/news context. No issuer DCF, financial-statement ratios or equity peer multiples are assumed.',
        'risk_charts': build_risk_charts(history),
        'news': news,
        'economic_calendar': economic_calendar,
        'geopolitical_news': geopolitical_news,
        'cfd_scenario': cfd_scenario,
        'trade_setup': trade_setup,
        'ratio_history': [],
        'financial_statements': {},
        'directional_decision': directional_decision,
        'fundamental_analysis': fundamental_analysis,
    }


def run_quant_pipeline(ticker_symbol: str, discount_rate: float, perpetual_growth: float) -> dict:
    # Route market instruments (FX, futures, commodities, rates, crypto, etc.)
    # through the asset-class-aware pipeline. Equities/ETFs keep the full issuer
    # fundamentals + DCF/peer workflow below.
    symbol = str(ticker_symbol or '').strip().upper()
    pre_info = {}
    try:
        pre_info = _ticker(symbol).info or {}
    except Exception:
        pre_info = {}
    detected_class, _, _ = classify_ticker_asset(symbol, pre_info)
    if detected_class not in {'Equity', 'ETF / Fund'}:
        return run_market_asset_pipeline(symbol, discount_rate, perpetual_growth)

    if perpetual_growth >= discount_rate:
        raise ValueError("Discount rate (WACC) must be strictly greater than Perpetual Growth Rate.")

    ticker = _ticker(symbol)
    info = ticker.info or {}
    sector = info.get("sector", "Technology")
    peer_symbols = get_sector_peers(sector, ticker_symbol)

    # Everything below needs these downloads; fetch them all at once instead of one by one.
    _name_hint = info.get("shortName", symbol)
    with _stage("prefetch"):
        _prefetch([
            lambda: ticker.history(period="5y"), lambda: ticker.fast_info,
            lambda: ticker.financials, lambda: ticker.balance_sheet, lambda: ticker.cashflow,
            lambda: ticker.news, lambda: _ticker("^GSPC").history(period="5y", auto_adjust=True),
            _risk_free_rate, lambda: fetch_geopolitical_news(ticker_symbol, _name_hint, sector),
        ] + [lambda s=p: _ticker(s).info for p in peer_symbols]
          + [lambda s=p: _ticker(s).fast_info for p in peer_symbols])

    current_price = None
    try:
        current_price = ticker.fast_info.get("last_price")
    except Exception:
        pass
    if not current_price:
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not current_price:
        history = ticker.history(period="1d")
        if not history.empty:
            current_price = float(history["Close"].iloc[-1])
    if not current_price:
        raise ValueError(f"Could not fetch price for '{ticker_symbol}'. Invalid or blocked ticker.")

    history_5y = ticker.history(period="5y")
    beta = _safe_float(info.get("beta"), 1.0)

    risk_metrics = calculate_risk_and_portfolio_metrics(history_5y, beta, current_price)
    financial_ratios = extract_financial_ratios(ticker, current_price)
    ratio_history = calculate_annual_financial_ratios(ticker, current_price)
    financial_statements = calculate_financial_statement_analysis(ticker)
    altman = calculate_altman_z_score(ticker, sector)

    cashflow = ticker.cashflow
    fcf = 0.0
    if cashflow is not None and not cashflow.empty:
        if "Free Cash Flow" in cashflow.index:
            fcf = float(cashflow.loc["Free Cash Flow"].iloc[0])
        elif "Operating Cash Flow" in cashflow.index and "Capital Expenditure" in cashflow.index:
            fcf = float(cashflow.loc["Operating Cash Flow"].iloc[0]) + float(
                cashflow.loc["Capital Expenditure"].iloc[0])

    growth_rate = info.get("earningsGrowth", 0.05) or 0.05
    projection_years = 5
    projected_fcf = [fcf * ((1 + growth_rate) ** i) for i in range(1, projection_years + 1)]
    pv_fcf = sum([cf / ((1 + discount_rate) ** i) for i, cf in enumerate(projected_fcf, start=1)])

    terminal_value = (projected_fcf[-1] * (1 + perpetual_growth)) / (discount_rate - perpetual_growth)
    pv_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)

    enterprise_value = pv_fcf + pv_terminal_value
    equity_value = enterprise_value + (info.get("totalCash", 0) or 0) - (info.get("totalDebt", 0) or 0)
    shares_outstanding = info.get("sharesOutstanding", 1) or 1
    fair_value = equity_value / shares_outstanding
    dcf_upside = ((fair_value - current_price) / current_price) * 100

    ml_results, chart_data = train_and_predict_ml_target(ticker, current_price)
    technicals = calculate_technicals(history_5y)
    risk_charts = build_risk_charts(history_5y)

    company_name = info.get("shortName", ticker_symbol.upper())
    peers = peer_snapshot(peer_symbols, ticker_symbol, company_name, current_price, info)

    eps = _safe_float(info.get('forwardEps'))
    valid_pe = [p['forward_pe'] for p in peers if p['forward_pe'] > 0]
    peer_median = float(np.median(valid_pe)) if valid_pe else 20.0
    peer_target = eps * peer_median if eps > 0 else current_price

    analyst_target = _safe_float(info.get('targetMeanPrice'), current_price)
    analyst_count = int(_safe_float(info.get('numberOfAnalystOpinions'), 0))

    composite_target = round(
        0.30 * _safe_float(fair_value) +
        0.20 * _safe_float(peer_target) +
        0.25 * _safe_float(ml_results.get('predicted_target_price')) +
        0.25 * analyst_target, 2
    )
    composite_upside = round(((composite_target - current_price) / current_price) * 100, 2)
    news = fetch_asset_news(ticker_symbol, sector)
    economic_calendar = build_economic_calendar()
    geopolitical_news = fetch_geopolitical_news(ticker_symbol, company_name, sector)
    trade_setup = calculate_trade_setup(
        current_price, technicals,
        fair_value_per_share=fair_value,
        peer_target_price=peer_target,
        ml_target_price=ml_results.get('predicted_target_price'),
        analyst_target_price=analyst_target,
        composite_target_price=composite_target,
        beta=beta
    )

    cfd_scenario = build_cfd_scenario(
        ticker_symbol, company_name, sector, current_price, technicals,
        risk_metrics, trade_setup, news, economic_calendar, geopolitical_news
    )

    directional_decision = make_directional_decision(
        current_price, fair_value, composite_target,
        ml_results.get("predicted_target_price"), analyst_target,
        technicals, risk_metrics, financial_ratios,
        cfd_scenario, news, economic_calendar, geopolitical_news
    )

    fundamental_analysis = build_fundamental_analysis(
        info, current_price, financial_ratios, ratio_history, financial_statements,
        fair_value, peer_target, peer_median, analyst_target, composite_target,
        ml_results.get('predicted_target_price'), peers, risk_metrics, news,
        economic_calendar, geopolitical_news
    )

    return {
        "ticker": ticker_symbol.upper(),
        "company_name": company_name,
        "sector": sector,
        "asset_class": "Equity",
        "analysis_mode": "ISSUER",
        "quote_unit": info.get("financialCurrency", "USD") or "USD",
        "current_price": round(current_price, 2),
        "fair_value_per_share": round(fair_value, 2),
        "dcf_implied_upside_percent": round(dcf_upside, 2),
        "currency": info.get("financialCurrency", "USD"),
        "inputs": {
            "discount_rate": discount_rate, "perpetual_growth": perpetual_growth,
            "projected_growth_rate": round(growth_rate, 4), "latest_free_cash_flow": round(fcf, 2)
        },
        "ml_forecast": ml_results,
        "chart_data": chart_data,
        "technicals": technicals,
        "risk_metrics": risk_metrics,
        "financial_ratios": financial_ratios,
        "altman": altman,
        "peers": peers,
        "peer_target_price": round(peer_target, 2),
        "peer_multiple": round(peer_median, 2),
        "peer_note": f"Sector: {sector} (Median P/E {peer_median:.1f}x)",
        "analyst_target_price": round(analyst_target, 2),
        "analyst_count": analyst_count,
        "composite_target_price": composite_target,
        "composite_implied_upside_percent": composite_upside,
        "methodology": f"Sector Peers ({sector}) + 30% DCF + 25% XGBoost",
        "risk_charts": risk_charts,
        "news": news,
        "economic_calendar": economic_calendar,
        "geopolitical_news": geopolitical_news,
        "cfd_scenario": cfd_scenario,
        "trade_setup": trade_setup,
        "ratio_history": ratio_history,
        "financial_statements": financial_statements,
        "directional_decision": directional_decision,
        "fundamental_analysis": fundamental_analysis
    }


# ------------------------------------------------------------------------------
# 3b. EXPORT + POWER BI DATASET BUILDERS
# ------------------------------------------------------------------------------
EXPORT_RISK_FREE_ANNUAL = 0.0425


def _as_export_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a price DataFrame index for clean Excel/Power BI ingestion."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    idx = pd.to_datetime(out.index, errors='coerce')
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep='last')]
    return out.sort_index()


def _history_from_analysis_cache(analysis: dict) -> pd.DataFrame:
    """Rebuild a minimal historical-price frame from cached dashboard data.

    This is used only when Yahoo Finance cannot be reached during an export.
    The dashboard pipeline already fetched and cached the historical close series,
    so the export can still produce a useful close/return workbook instead of
    failing because of a transient DNS/network error.
    """
    chart = analysis.get('chart_data') or {}
    dates = chart.get('dates') or []
    closes = chart.get('historical_close') or []
    if not dates or not closes:
        return pd.DataFrame()
    n = min(len(dates), len(closes))
    idx = pd.to_datetime(dates[:n], errors='coerce')
    close = pd.to_numeric(pd.Series(closes[:n], index=idx), errors='coerce')
    frame = pd.DataFrame({'Close': close})
    frame.index.name = 'Date'
    return _as_export_date_index(frame.dropna(subset=['Close']))


def _safe_yf_history(ticker_obj, period='5y') -> tuple[pd.DataFrame, str | None]:
    """Fetch yfinance history with brief retries without crashing ASGI."""
    last_error = None
    for attempt in range(3):
        try:
            frame = _as_export_date_index(ticker_obj.history(period=period, auto_adjust=True))
            if not frame.empty:
                return frame, None
            last_error = "empty response"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame(), str(last_error) if last_error else "unknown provider error"


def build_export_bundle(ticker_symbol: str, discount_rate: float = 0.09, perpetual_growth: float = 0.025) -> dict:
    """Build normalized export tables and survive transient Yahoo/DNS outages."""
    symbol = ticker_symbol.upper().strip()
    ticker = _ticker(symbol)

    # Reuse the cached dashboard analysis when it exists so an Excel/Power BI
    # download does not retrain the model or refetch the full fundamentals stack.
    try:
        analysis = _analysis(symbol, discount_rate, perpetual_growth)
    except HTTPException:
        raise
    except Exception as exc:
        raise ValueError(
            f"Cannot build an export for '{symbol}' because live market data is unavailable. "
            f"Check your internet/DNS connection and try again. Provider error: {type(exc).__name__}: {exc}"
        ) from exc

    # IMPORTANT: do not call ticker.info here. run_quant_pipeline already loaded
    # the company metadata, and a second Yahoo quote request can fail independently
    # (as in query2.finance.yahoo.com DNS outages), even while the dashboard cache
    # is valid.
    company_name = analysis.get('company_name') or symbol
    sector = analysis.get('sector') or 'N/A'
    currency = analysis.get('currency') or 'USD'

    history, history_error = _safe_yf_history(ticker, period='5y')
    fallback_used = False
    if history.empty or 'Close' not in history.columns:
        history = _history_from_analysis_cache(analysis)
        fallback_used = True
    if history.empty or 'Close' not in history.columns:
        detail = f" Provider error: {history_error}." if history_error else ''
        raise ValueError(f"No five-year historical price data available for '{symbol}'.{detail}")

    market, market_error = _safe_yf_history(_ticker('^GSPC'), period='5y')
    close = pd.to_numeric(history['Close'], errors='coerce')
    market_close = pd.to_numeric(market['Close'], errors='coerce') if not market.empty and 'Close' in market.columns else pd.Series(dtype=float)

    prices = pd.DataFrame(index=history.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in history.columns:
            prices[col] = pd.to_numeric(history[col], errors='coerce')
    prices['Daily Return'] = close.pct_change()
    prices['Log Return'] = np.log(close / close.shift(1))
    prices['5D Return'] = close.pct_change(5)
    prices['20D Return'] = close.pct_change(20)
    prices = prices.reset_index().rename(columns={'index': 'Date', 'Close': 'Adjusted Close'})
    prices.insert(0, 'Ticker', symbol)
    prices.insert(1, 'Company', company_name)
    prices['Date'] = pd.to_datetime(prices['Date']).dt.tz_localize(None)

    joined = pd.concat([close.rename('Asset Close'), market_close.rename('S&P 500 Close')], axis=1, join='inner').dropna()
    returns = joined.pct_change().dropna()
    daily_rf = (1.0 + _risk_free_rate()) ** (1.0 / 252.0) - 1.0
    excess = pd.DataFrame(index=returns.index)
    excess['Asset Return'] = returns['Asset Close']
    excess['S&P 500 Return'] = returns['S&P 500 Close']
    excess['Daily Risk-Free Approx'] = daily_rf
    excess['Excess Return vs Risk-Free'] = excess['Asset Return'] - daily_rf
    excess['Active Return vs S&P 500'] = excess['Asset Return'] - excess['S&P 500 Return']
    excess['Cumulative Excess Return vs Risk-Free'] = (1.0 + excess['Excess Return vs Risk-Free']).cumprod() - 1.0
    excess['Cumulative Active Return vs S&P 500'] = (1.0 + excess['Active Return vs S&P 500']).cumprod() - 1.0
    excess['Rolling 20D Annualized Volatility'] = excess['Asset Return'].rolling(20).std(ddof=1) * np.sqrt(252)
    excess = excess.reset_index().rename(columns={'index': 'Date'})
    excess.insert(0, 'Ticker', symbol)
    excess.insert(1, 'Company', company_name)
    excess['Date'] = pd.to_datetime(excess['Date']).dt.tz_localize(None)

    ratio_history = analysis.get('ratio_history') or []
    ratio_df = pd.DataFrame(ratio_history)
    if ratio_df.empty:
        ratio_df = pd.DataFrame(columns=['year','roic','roa','roe','gross_margin','operating_margin','net_margin','current_ratio','quick_ratio','cash_ratio','debt_to_equity','interest_coverage','asset_turnover','inventory_turnover','receivables_turnover','fcf_yield','revenue','net_income','free_cash_flow'])
    ratio_df.insert(0, 'Ticker', symbol)
    ratio_df.insert(1, 'Company', company_name)
    ratio_df.insert(2, 'Sector', sector)

    statements = analysis.get('financial_statements') or {}
    stmt_rows = []
    for group_key, group_name in [('income_statement','Income Statement'),('balance_sheet','Balance Sheet'),('cash_flow','Cash Flow')]:
        for row in statements.get(group_key, []):
            vals = row.get('values', [])
            years = statements.get('years', [])
            for year, value in zip(years, vals):
                stmt_rows.append({'Ticker':symbol,'Company':company_name,'Statement':group_name,'Metric':row.get('metric'),'Fiscal Year':year,'Value':value})
    statements_df = pd.DataFrame(stmt_rows, columns=['Ticker','Company','Statement','Metric','Fiscal Year','Value'])

    val_summary = pd.DataFrame([{
        'Ticker': symbol,
        'Company': company_name,
        'Sector': sector,
        'Currency': currency,
        'Current Price': analysis.get('current_price'),
        'DCF Fair Value / Share': analysis.get('fair_value_per_share'),
        'DCF Implied Upside %': analysis.get('dcf_implied_upside_percent'),
        'XGBoost Target Price': analysis.get('ml_forecast', {}).get('predicted_target_price'),
        'Peer Target Price': analysis.get('peer_target_price'),
        'Analyst Target Price': analysis.get('analyst_target_price'),
        'Composite Target Price': analysis.get('composite_target_price'),
        'Beta': analysis.get('risk_metrics', {}).get('beta'),
        'CAPM Expected Return': analysis.get('risk_metrics', {}).get('capm_expected_return'),
        '5Y Annualized Volatility': analysis.get('risk_metrics', {}).get('std_dev_5y'),
        '5Y Variance': analysis.get('risk_metrics', {}).get('variance_5y'),
        'Sharpe Ratio': analysis.get('risk_metrics', {}).get('sharpe_ratio'),
        'Treynor Ratio': analysis.get('risk_metrics', {}).get('treynor_ratio'),
        'Information Ratio': analysis.get('risk_metrics', {}).get('information_ratio'),
        'Jensen Alpha': analysis.get('risk_metrics', {}).get('jensen_alpha'),
        'Tracking Error': analysis.get('risk_metrics', {}).get('tracking_error'),
        'Active Return Annualized': analysis.get('risk_metrics', {}).get('active_return_annualized'),
        'Altman Z-Score': analysis.get('altman', {}).get('score'),
        'Altman Zone': analysis.get('altman', {}).get('zone'),
    }])

    sources_rows = [
        ['Historical prices', 'Yahoo Finance via yfinance', f'https://finance.yahoo.com/quote/{symbol}/history'],
        ['Benchmark prices', 'Yahoo Finance via yfinance', 'https://finance.yahoo.com/quote/%5EGSPC/history'],
        ['Annual financial statements', 'Yahoo Finance via yfinance', f'https://finance.yahoo.com/quote/{symbol}/financials'],
        ['Annual balance sheet', 'Yahoo Finance via yfinance', f'https://finance.yahoo.com/quote/{symbol}/balance-sheet'],
        ['Annual cash flow', 'Yahoo Finance via yfinance', f'https://finance.yahoo.com/quote/{symbol}/cash-flow'],
        ['Risk-free rate', f'US 10Y Treasury yield, FRED DGS10 (fallback 4.25%) — {_risk_free_rate():.2%} annual', 'https://fred.stlouisfed.org/series/DGS10'],
    ]
    if fallback_used:
        sources_rows.append(['Export fallback', 'Cached dashboard historical closes used because live Yahoo history was unavailable', ''])
    if market.empty:
        sources_rows.append(['Benchmark fallback', 'S&P 500 daily history unavailable; benchmark-dependent export columns are N/A', ''])

    sources_df = pd.DataFrame(sources_rows,
    columns=['Dataset','Source','URL'])

    return {
        'symbol': symbol,
        'company_name': company_name,
        'sector': sector,
        'currency': currency,
        'historical_prices': prices,
        'excess_returns': excess,
        'financial_ratios': ratio_df,
        'financial_statements': statements_df,
        'valuation_summary': val_summary,
        'sources': sources_df,
        'analysis': analysis,
    }


def dataframe_from_statement_rows(statement_rows: list[dict], years: list[int]) -> pd.DataFrame:
    rows = []
    for row in statement_rows:
        for year, value in zip(years, row.get('values', [])):
            rows.append({'Metric': row.get('metric'), 'Fiscal Year': year, 'Value': value})
    return pd.DataFrame(rows, columns=['Metric','Fiscal Year','Value'])


def _safe_excel_sheet_name(name: str) -> str:
    banned = ['[',']',':','*','?','/','\\']
    out = str(name)
    for ch in banned:
        out = out.replace(ch, '-')
    return out[:31]


def build_excel_bytes(bundle: dict, mode: str = 'all') -> bytes:
    """Create an analyst-friendly .xlsx export using openpyxl (no xlsxwriter dependency)."""
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    from openpyxl.utils import get_column_letter

    output = io.BytesIO()

    # openpyxl is used deliberately so the Excel export works in environments
    # where xlsxwriter has not been installed in the active virtualenv.
    with pd.ExcelWriter(output, engine='openpyxl', datetime_format='yyyy-mm-dd') as writer:
        def write_df(name: str, df: pd.DataFrame, pct_cols=None, multiple_cols=None):
            sheet_name = _safe_excel_sheet_name(name)
            if df is None or df.empty:
                df = pd.DataFrame({'Message': ['No data available']})

            # Start at row 3 so row 1 can hold the export title.
            df.to_excel(writer, sheet_name=sheet_name, startrow=2, index=False)
            ws = writer.book[sheet_name]

            ws.freeze_panes = 'A4'
            ws.sheet_view.showGridLines = False
            ws['A1'] = f'finmodel export · {bundle["symbol"]} · {name}'
            ws['A1'].font = Font(bold=True, size=16, color='0F172A')
            ws['A1'].alignment = Alignment(vertical='center')
            ws.row_dimensions[1].height = 24

            # Header styling on Excel row 3.
            thin = Side(style='thin', color='CBD5E1')
            header_fill = PatternFill('solid', fgColor='0F172A')
            for cell in ws[3]:
                if cell.value is not None:
                    cell.font = Font(bold=True, color='FFFFFF')
                    cell.fill = header_fill
                    cell.border = Border(bottom=thin)
                    cell.alignment = Alignment(horizontal='center', vertical='center')

            last_col = max(ws.max_column, 1)
            last_row = max(ws.max_row, 3)
            ws.auto_filter.ref = f'A3:{get_column_letter(last_col)}{last_row}'

            pct_cols = set(pct_cols or [])
            multiple_cols = set(multiple_cols or [])
            # Apply number formats and sensible widths.
            for c in range(1, last_col + 1):
                header = ws.cell(3, c).value
                values = [ws.cell(r, c).value for r in range(4, min(last_row, 253) + 1)]
                max_len = max([len(str(header or ''))] + [len(str(v)) if v is not None else 0 for v in values])
                ws.column_dimensions[get_column_letter(c)].width = min(max(max_len + 2, 12), 28)

                if header in pct_cols:
                    for r in range(4, last_row + 1):
                        ws.cell(r, c).number_format = '0.00%;[Red](0.00%);-'
                    ws.column_dimensions[get_column_letter(c)].width = 16
                elif header in multiple_cols:
                    for r in range(4, last_row + 1):
                        ws.cell(r, c).number_format = '0.00x;[Red](0.00x);-'
                    ws.column_dimensions[get_column_letter(c)].width = 16
                elif header == 'Date':
                    for r in range(4, last_row + 1):
                        ws.cell(r, c).number_format = 'yyyy-mm-dd'
                    ws.column_dimensions[get_column_letter(c)].width = 13
                elif header == 'Volume':
                    for r in range(4, last_row + 1):
                        ws.cell(r, c).number_format = '#,##0;[Red](#,##0);-'
                    ws.column_dimensions[get_column_letter(c)].width = 15

            # Subtle alternating rows improve readability without requiring
            # xlsxwriter-specific formatting objects.
            alt_fill = PatternFill('solid', fgColor='F8FAFC')
            for r in range(4, last_row + 1, 2):
                for c in range(1, last_col + 1):
                    if ws.cell(r, c).fill.fill_type is None:
                        ws.cell(r, c).fill = alt_fill

        if mode in ('all', 'market'):
            write_df('Historical Prices', bundle['historical_prices'], pct_cols=['Daily Return','Log Return','5D Return','20D Return'])
            write_df('Excess Returns', bundle['excess_returns'], pct_cols=['Asset Return','S&P 500 Return','Daily Risk-Free Approx','Excess Return vs Risk-Free','Active Return vs S&P 500','Cumulative Excess Return vs Risk-Free','Cumulative Active Return vs S&P 500','Rolling 20D Annualized Volatility'])
        if mode in ('all', 'fundamentals'):
            write_df('Financial Ratios', bundle['financial_ratios'], pct_cols=['roic','roa','roe','gross_margin','operating_margin','net_margin','fcf_yield'], multiple_cols=['current_ratio','quick_ratio','cash_ratio','debt_to_equity','interest_coverage','asset_turnover','inventory_turnover','receivables_turnover'])
            write_df('Financial Statements', bundle['financial_statements'])
            write_df('Valuation Summary', bundle['valuation_summary'], pct_cols=['DCF Implied Upside %','CAPM Expected Return','5Y Annualized Volatility','Jensen Alpha','Tracking Error','Active Return Annualized'], multiple_cols=['Beta','Sharpe Ratio','Treynor Ratio','Information Ratio','Altman Z-Score'])
        write_df('Sources', bundle['sources'])

    output.seek(0)
    # Validate that a workbook was actually created before returning bytes.
    data = output.read()
    check = load_workbook(io.BytesIO(data), read_only=True)
    check.close()
    return data



def build_financial_model_excel(bundle: dict) -> bytes:
    """Create a multi-sheet financial-model workbook with a dashboard and native Excel charts."""
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.chart import LineChart, BarChart, Reference
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    default = wb.active
    wb.remove(default)
    dark = '0F172A'; cyan='22D3EE'; green='16A34A'; amber='F59E0B'; light='E2E8F0'; slate='64748B'
    header_fill = PatternFill('solid', fgColor=dark)
    thin = Side(style='thin', color='CBD5E1')

    def add_df(name, df, pct_cols=()):
        ws=wb.create_sheet(name[:31]); ws.sheet_view.showGridLines=False; ws.freeze_panes='A4'
        if df is None or df.empty:
            df=pd.DataFrame({'Message':['No data available']})
        # pandas cannot write into an openpyxl Worksheet object, so cells are written directly.
        for j, col in enumerate(df.columns, start=1):
            ws.cell(3, j, str(col))
        for i, row in enumerate(df.itertuples(index=False), start=4):
            for j, v in enumerate(row, start=1):
                if isinstance(v, np.integer):
                    v = int(v)
                elif isinstance(v, (float, np.floating)):
                    v = float(v) if np.isfinite(v) else None
                elif isinstance(v, pd.Timestamp):
                    v = v.tz_localize(None).to_pydatetime() if v.tzinfo else v.to_pydatetime()
                elif v is not None and not isinstance(v, (str, int, bool)):
                    v = str(v)
                ws.cell(i, j, v)
        ws['A1']=f"finmodel · {bundle['ticker']} · {name}"
        ws['A1'].font=Font(size=16,bold=True,color=dark)
        for cell in ws[3]:
            if cell.value is not None:
                cell.font=Font(bold=True,color='FFFFFF'); cell.fill=header_fill; cell.border=Border(bottom=thin); cell.alignment=Alignment(horizontal='center')
        for c in range(1,ws.max_column+1):
            vals=[ws.cell(r,c).value for r in range(4,min(ws.max_row,100)+1)]
            width=min(max([len(str(ws.cell(3,c).value or ''))]+[len(str(v)) for v in vals if v is not None])+2,28)
            ws.column_dimensions[get_column_letter(c)].width=max(width,12)
            if ws.cell(3,c).value in pct_cols:
                for r in range(4,ws.max_row+1): ws.cell(r,c).number_format='0.00%'
        ws.auto_filter.ref=f"A3:{get_column_letter(ws.max_column)}{ws.max_row}"
        return ws

    dash=wb.create_sheet('Model Dashboard'); dash.sheet_view.showGridLines=False
    dash['A1']=f"FINMODEL — FINANCIAL MODEL DASHBOARD — {bundle['company_name']} ({bundle['ticker']})"; dash['A1'].font=Font(size=20,bold=True,color='FFFFFF'); dash['A1'].fill=header_fill
    dash.merge_cells('A1:H1'); dash.row_dimensions[1].height=30
    metrics=bundle['dashboard']
    kpis=[('Current Price',metrics['Current Price']),('DCF / Share',metrics['DCF Price']),('WACC',metrics['WACC']),('Terminal Growth',metrics['Terminal Growth']),('Enterprise Value',metrics['Enterprise Value']),('Equity Value',metrics['Equity Value']),('Forecast CAGR',metrics['Forecast CAGR']),('Beta',metrics['Beta'])]
    for i,(label,val) in enumerate(kpis):
        col=1+(i%4)*2; row=3+(i//4)*3
        dash.cell(row,col,label); dash.cell(row,col).font=Font(bold=True,color=slate,size=10)
        dash.cell(row+1,col,val); dash.cell(row+1,col).font=Font(bold=True,color=green if i in [1,5] else cyan,size=16)
    dash['A9']='Model architecture'; dash['A9'].font=Font(bold=True,color=dark,size=13)
    arch=[('Historical','5Y statements + ratios + common-size + DuPont'),('Forecast','5Y revenue / EBIT / NOPAT / FCFF'),('Valuation','WACC + FCFF DCF + trading multiples'),('Scenarios','Bear / Base / Bull + WACC/g sensitivity'),('Statistics','Correlation matrix + growth / margin diagnostics')]
    for i,(a,b) in enumerate(arch,10): dash.cell(i,1,a).font=Font(bold=True,color=dark); dash.cell(i,2,b)
    dash.column_dimensions['A'].width=28; dash.column_dimensions['B'].width=54; dash.column_dimensions['C'].width=16; dash.column_dimensions['D'].width=16; dash.column_dimensions['E'].width=16; dash.column_dimensions['F'].width=16; dash.column_dimensions['G'].width=16; dash.column_dimensions['H'].width=16

    hist=bundle['historical']; forecast=bundle['forecast']; scenarios=bundle['scenarios']; sens=bundle['sensitivity']
    hist_ws=add_df('Historical Model',hist,pct_cols=['Revenue Growth','EBIT Growth','Net Income Growth','FCF Growth','EBIT Margin','Net Margin','FCF Margin','DuPont ROE','Common Size Revenue','Common Size EBIT','Common Size Net Income','Common Size FCF'])
    fc_ws=add_df('Forecast',forecast,pct_cols=['Revenue Growth','EBIT Margin','Tax Rate','FCFF Margin'])
    dcf_ws=add_df('DCF',bundle['dcf']); add_df('WACC',bundle['wacc'],pct_cols=['Value']); add_df('Multiples',bundle['multiples']); add_df('Scenarios',scenarios,pct_cols=['Revenue Growth','EBIT Margin','WACC','Upside']); add_df('Sensitivity',sens,pct_cols=[c for c in sens.columns if c!='WACC']); add_df('DuPont',bundle['dupont'],pct_cols=['Net Margin','DuPont ROE']); add_df('Correlations',bundle['correlation'])
    # Add native charts to dashboard.
    ch=LineChart(); ch.title='Revenue / EBIT / FCF Historical'; ch.y_axis.title='Amount'; ch.x_axis.title='Fiscal Year';
    if len(hist)>0:
        data=Reference(hist_ws,min_col=2,max_col=2+2,min_row=3,max_row=3+len(hist)); cats=Reference(hist_ws,min_col=1,min_row=4,max_row=3+len(hist)); ch.add_data(data,titles_from_data=True); ch.set_categories(cats); ch.height=7; ch.width=12; dash.add_chart(ch,'A13')
    sc=BarChart(); sc.title='Scenario Implied Price'; sc.y_axis.title='Price';
    scen_ws=wb['Scenarios']; scdata=Reference(scen_ws,min_col=7,max_col=7,min_row=3,max_row=6); sccats=Reference(scen_ws,min_col=1,min_row=4,max_row=6); sc.add_data(scdata,titles_from_data=True); sc.set_categories(sccats); sc.height=7; sc.width=10; dash.add_chart(sc,'G13')

    readme=wb.create_sheet('README'); readme['A1']='Financial Model Methodology'; readme['A1'].font=Font(size=16,bold=True,color=dark)
    notes=[
        'Historical: five most recent annual financial statement periods returned by the data provider.',
        'Forecast: five-year deterministic scenario derived from historical growth, operating margin and FCF margin.',
        'WACC: CAPM-style cost of equity + after-tax cost of debt using current balance-sheet weights.',
        'DCF: FCFF discounted at WACC with a terminal-growth perpetuity.',
        'Scenarios: Bear/Base/Bull vary growth, operating margin and WACC.',
        'Sensitivity: DCF value per share across WACC and terminal-growth combinations.',
        'Outputs are analytical estimates and should not be treated as guaranteed outcomes or trade instructions.'
    ]
    for i,n in enumerate(notes,3): readme.cell(i,1,n)
    readme.column_dimensions['A'].width=110

    buf=io.BytesIO(); wb.save(buf); buf.seek(0)
    test=load_workbook(io.BytesIO(buf.getvalue()),read_only=True); test.close()
    return buf.getvalue()


def build_powerbi_project_zip(bundle: dict) -> bytes:
    """Package a Power BI Project (PBIP) starter bound to the exported Excel file."""
    from uuid import uuid4
    root = 'Finmodel_PowerBI'
    report_name = 'Finmodel_Analytics.Report'
    model_name = 'Finmodel_Analytics.SemanticModel'
    data_file = 'Finmodel_Data.xlsx'

    data_bytes = build_excel_bytes(bundle, mode='all')
    project = {}
    project[f'{root}/Finmodel_Analytics.pbip'] = json.dumps({
        '$schema': 'https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json',
        'version': '1.0',
        'artifacts': [{'report': {'path': report_name}}],
        'settings': {'enableAutoRecovery': True}
    }, indent=2)
    project[f'{root}/{report_name}/definition.pbir'] = json.dumps({
        '$schema': 'https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json',
        'version': '4.0',
        'datasetReference': {'byPath': {'path': f'../{model_name}'}}
    }, indent=2)
    project[f'{root}/{report_name}/definition/version.json'] = json.dumps({
        '$schema': 'https://developer.microsoft.com/json-schemas/fabric/item/report/definition/versionMetadata/1.0.0/schema.json',
        'version': '2.0.0'
    }, indent=2)
    project[f'{root}/{report_name}/definition/report.json'] = json.dumps({
        '$schema': 'https://developer.microsoft.com/json-schemas/fabric/item/report/definition/report/3.2.0/schema.json',
        'themeCollection': {},
        'filterConfig': {'filters': []},
        'objects': {},
        'settings': {},
        'resourcePackages': [],
        'annotations': []
    }, indent=2)
    page_id = 'ReportSection' + uuid4().hex[:18]
    project[f'{root}/{report_name}/definition/pages/pages.json'] = json.dumps({
        '$schema': 'https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.1.0/schema.json',
        'pageOrder': [page_id],
        'activePageName': page_id
    }, indent=2)
    project[f'{root}/{report_name}/definition/pages/{page_id}/page.json'] = json.dumps({
        '$schema': 'https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/2.1.0/schema.json',
        'name': page_id,
        'displayName': f'Finmodel Export · {bundle["symbol"]}',
        'displayOption': 'FitToPage',
        'height': 720,
        'width': 1280
    }, indent=2)
    # Empty visuals folder is intentional; users can add charts/tables after opening the PBIP.
    project[f'{root}/{report_name}/definition/pages/{page_id}/visuals/.keep'] = ''
    project[f'{root}/{model_name}/definition.pbism'] = json.dumps({
        '$schema': 'https://developer.microsoft.com/json-schemas/fabric/item/semanticModel/definitionProperties/1.0.0/schema.json',
        'version': '4.2',
        'settings': {'qnaEnabled': True}
    }, indent=2)
    project[f'{root}/{model_name}/definition/database.tmdl'] = 'database\n\tcompatibilityLevel: 1702\n\tcompatibilityMode: powerBI\n\tlanguage: 1033\n'
    project[f'{root}/{model_name}/definition/model.tmdl'] = (
        'model Model\n\tculture: en-US\n\tdefaultPowerBIDataSourceVersion: powerBI_V3\n\tsourceQueryCulture: en-US\n\n'
        'ref table HistoricalPrices\nref table ExcessReturns\nref table FinancialRatios\nref table FinancialStatements\nref table ValuationSummary\n'
    )

    table_specs = {
        'HistoricalPrices': [
            ('Ticker','string','Ticker'),('Company','string','Company'),('Date','dateTime','Date'),('Open','double','Open'),('High','double','High'),('Low','double','Low'),('Adjusted Close','double','Adjusted Close'),('Volume','int64','Volume'),('Daily Return','double','Daily Return'),('Log Return','double','Log Return'),('5D Return','double','5D Return'),('20D Return','double','20D Return')],
        'ExcessReturns': [
            ('Ticker','string','Ticker'),('Company','string','Company'),('Date','dateTime','Date'),('Asset Return','double','Asset Return'),('S&P 500 Return','double','S&P 500 Return'),('Daily Risk-Free Approx','double','Daily Risk-Free Approx'),('Excess Return vs Risk-Free','double','Excess Return vs Risk-Free'),('Active Return vs S&P 500','double','Active Return vs S&P 500'),('Cumulative Excess Return vs Risk-Free','double','Cumulative Excess Return vs Risk-Free'),('Cumulative Active Return vs S&P 500','double','Cumulative Active Return vs S&P 500'),('Rolling 20D Annualized Volatility','double','Rolling 20D Annualized Volatility')],
        'FinancialRatios': [('Ticker','string','Ticker'),('Company','string','Company'),('Sector','string','Sector'),('year','int64','year'),('roic','double','roic'),('roa','double','roa'),('roe','double','roe'),('gross_margin','double','gross_margin'),('operating_margin','double','operating_margin'),('net_margin','double','net_margin'),('current_ratio','double','current_ratio'),('quick_ratio','double','quick_ratio'),('cash_ratio','double','cash_ratio'),('debt_to_equity','double','debt_to_equity'),('interest_coverage','double','interest_coverage'),('asset_turnover','double','asset_turnover'),('inventory_turnover','double','inventory_turnover'),('receivables_turnover','double','receivables_turnover'),('fcf_yield','double','fcf_yield'),('revenue','double','revenue'),('net_income','double','net_income'),('free_cash_flow','double','free_cash_flow')],
        'FinancialStatements': [('Ticker','string','Ticker'),('Company','string','Company'),('Statement','string','Statement'),('Metric','string','Metric'),('Fiscal Year','int64','Fiscal Year'),('Value','double','Value')],
        'ValuationSummary': [('Ticker','string','Ticker'),('Company','string','Company'),('Sector','string','Sector'),('Currency','string','Currency'),('Current Price','double','Current Price'),('DCF Fair Value / Share','double','DCF Fair Value / Share'),('DCF Implied Upside %','double','DCF Implied Upside %'),('XGBoost Target Price','double','XGBoost Target Price'),('Peer Target Price','double','Peer Target Price'),('Analyst Target Price','double','Analyst Target Price'),('Composite Target Price','double','Composite Target Price'),('Beta','double','Beta'),('CAPM Expected Return','double','CAPM Expected Return'),('5Y Annualized Volatility','double','5Y Annualized Volatility'),('5Y Variance','double','5Y Variance'),('Sharpe Ratio','double','Sharpe Ratio'),('Treynor Ratio','double','Treynor Ratio'),('Information Ratio','double','Information Ratio'),('Jensen Alpha','double','Jensen Alpha'),('Tracking Error','double','Tracking Error'),('Active Return Annualized','double','Active Return Annualized'),('Altman Z-Score','double','Altman Z-Score'),('Altman Zone','string','Altman Zone')]
    }
    def m_escape(s):
        return s.replace('"','""')
    for table, cols in table_specs.items():
        display = table
        qsheet = {'HistoricalPrices':'Historical Prices','ExcessReturns':'Excess Returns','FinancialRatios':'Financial Ratios','FinancialStatements':'Financial Statements','ValuationSummary':'Valuation Summary'}[table]
        lines=[f'table {display}']
        for name, dtype, source in cols:
            dt = {'string':'string','dateTime':'dateTime','double':'double','int64':'int64'}[dtype]
            lines.append(f'\tcolumn {name if name.isidentifier() else repr(name)}')
            lines.append(f'\t\tdataType: {dt}')
            safe_source = source if source.replace('_','').isalnum() else "'" + source.replace("'","''") + "'"
            lines.append(f'\t\tsourceColumn: {safe_source}')
        lines.append(f'\tpartition {table} = m')
        lines.append('\t\tmode: import')
        lines.append('\t\tsource =')
        lines.append('\t\t\tlet')
        lines.append(f'\t\t\t\tSource = Excel.Workbook(File.Contents("{m_escape(data_file)}"), null, true),')
        lines.append(f'\t\t\t\tData = Source{{[Item="{m_escape(qsheet)}",Kind="Sheet"]}}[Data],')
        lines.append('\t\t\t\tSkipped = Table.Skip(Data, 2),')
        lines.append('\t\t\t\tPromoted = Table.PromoteHeaders(Skipped, [PromoteAllScalars=true])')
        lines.append('\t\t\tin')
        lines.append('\t\t\t\tPromoted')
        project[f'{root}/{model_name}/definition/tables/{table}.tmdl'] = '\n'.join(lines) + '\n'

    project[f'{root}/Finmodel_Data.xlsx'] = data_bytes
    project[f'{root}/README.txt'] = (
        'Finmodel Power BI Project\n\n'
        f'Ticker: {bundle["symbol"]}\n\n'
        'Open Finmodel_Analytics.pbip with Power BI Desktop. The semantic model reads Finmodel_Data.xlsx from the project folder.\n'
        'If Power BI asks for a file path, point the source to the included Finmodel_Data.xlsx.\n'
        'The report starts with a blank canvas so users can build their own visuals from the imported tables.\n'
        'Data sheets: Historical Prices, Excess Returns, Financial Ratios, Financial Statements, Valuation Summary, Sources.\n'
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, content in project.items():
            if isinstance(content, str):
                zf.writestr(f'{root}/{name.split("/",1)[1]}', content)
            else:
                # content already includes root? handled above only for data file
                zf.writestr(name, content)
    buffer.seek(0)
    return buffer.read()


# ------------------------------------------------------------------------------
# 3c. STATISTICAL FINANCIAL MODEL + MULTI-ASSET BOARD
# ------------------------------------------------------------------------------

def _latest_statement_value(financial_statements: dict, section: str, metric: str, default=None):
    rows = {r.get('metric'): r.get('values', []) for r in (financial_statements or {}).get(section, [])}
    vals = rows.get(metric, [])
    if vals and vals[0] is not None:
        return _safe_float(vals[0], default if default is not None else 0.0)
    return default


def _historical_model_rows(financial_statements: dict, ratio_history: list[dict]) -> pd.DataFrame:
    years = list((financial_statements or {}).get('years') or [])
    inc = {r.get('metric'): r.get('values', []) for r in (financial_statements or {}).get('income_statement', [])}
    bal = {r.get('metric'): r.get('values', []) for r in (financial_statements or {}).get('balance_sheet', [])}
    cf = {r.get('metric'): r.get('values', []) for r in (financial_statements or {}).get('cash_flow', [])}
    rows = []
    for i, year in enumerate(years):
        revenue = inc.get('Revenue', [None] * len(years))[i] if i < len(inc.get('Revenue', [])) else None
        ebit = inc.get('Operating Income', [None] * len(years))[i] if i < len(inc.get('Operating Income', [])) else None
        net_income = inc.get('Net Income', [None] * len(years))[i] if i < len(inc.get('Net Income', [])) else None
        assets = bal.get('Total Assets', [None] * len(years))[i] if i < len(bal.get('Total Assets', [])) else None
        debt = bal.get('Total Debt', [None] * len(years))[i] if i < len(bal.get('Total Debt', [])) else None
        equity = bal.get('Stockholders Equity', [None] * len(years))[i] if i < len(bal.get('Stockholders Equity', [])) else None
        fcf = cf.get('Free Cash Flow', [None] * len(years))[i] if i < len(cf.get('Free Cash Flow', [])) else None
        rows.append({'Year': year, 'Revenue': revenue, 'EBIT': ebit, 'Net Income': net_income,
                     'Total Assets': assets, 'Total Debt': debt, 'Equity': equity, 'FCF': fcf,
                     'EBIT Margin': (ebit / revenue if revenue not in (None, 0) and ebit is not None else None),
                     'Net Margin': (net_income / revenue if revenue not in (None, 0) and net_income is not None else None),
                     'FCF Margin': (fcf / revenue if revenue not in (None, 0) and fcf is not None else None)})
    return pd.DataFrame(rows)


def _financial_model_assumptions(data: dict) -> dict:
    info = data.get('_info', {}) or {}
    rm = data.get('risk_metrics', {}) or {}
    fs = data.get('financial_statements', {}) or {}
    hist = _historical_model_rows(fs, data.get('ratio_history', []))
    last = hist.iloc[0].to_dict() if not hist.empty else {}
    revenue = _safe_float(last.get('Revenue'), 0.0)
    ebit = _safe_float(last.get('EBIT'), 0.0)
    fcf = _safe_float(last.get('FCF'), 0.0)
    revenue_growth = info.get('revenueGrowth')
    if revenue_growth is None or not np.isfinite(float(revenue_growth)):
        if len(hist) >= 2 and _safe_float(hist.iloc[-1].get('Revenue'), 0) > 0 and revenue > 0:
            revenue_growth = (revenue / _safe_float(hist.iloc[-1].get('Revenue'), revenue)) ** (1 / max(len(hist) - 1, 1)) - 1
        else:
            revenue_growth = 0.05
    ebit_margin = (ebit / revenue) if revenue else _safe_float(info.get('operatingMargins'), 0.15)
    fcf_margin = (fcf / revenue) if revenue else 0.05
    rf = _safe_float(rm.get('risk_free_rate'), 0.0425)
    beta = _safe_float(rm.get('beta'), _safe_float(info.get('beta'), 1.0))
    market_return = _safe_float(rm.get('market_return_5y'), 0.10)
    cost_of_equity = rf + beta * (market_return - rf)
    debt = _safe_float(last.get('Total Debt'), 0.0)
    equity_book = _safe_float(last.get('Equity'), 0.0)
    market_cap = _safe_float(info.get('marketCap'), 0.0)
    equity_value = market_cap if market_cap > 0 else max(equity_book, 1.0)
    interest_exp = abs(_latest_statement_value(fs, 'income_statement', 'Interest Expense', 0) or 0)
    cost_of_debt = interest_exp / debt if debt > 0 and interest_exp > 0 else 0.05
    pretax = _latest_statement_value(fs, 'income_statement', 'Pretax Income', 0) or 0
    tax = _latest_statement_value(fs, 'income_statement', 'Tax Provision', 0) or 0
    tax_rate = min(max(tax / pretax, 0), 0.35) if pretax > 0 and tax >= 0 else 0.21
    capital = max(equity_value + debt, 1.0)
    equity_weight = equity_value / capital
    debt_weight = debt / capital
    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)
    if wacc <= 0.01:
        wacc = max(rf + 0.03, 0.08)
    terminal_growth = min(max(wacc - 0.04, 0.015), 0.035)
    return {
        'base_revenue': revenue, 'base_ebit': ebit, 'base_fcf': fcf,
        'revenue_growth': float(revenue_growth), 'ebit_margin': float(ebit_margin), 'fcf_margin': float(fcf_margin),
        'risk_free_rate': rf, 'beta': beta, 'market_return': market_return,
        'cost_of_equity': cost_of_equity, 'cost_of_debt': cost_of_debt, 'tax_rate': tax_rate,
        'debt': debt, 'equity_value': equity_value, 'equity_weight': equity_weight, 'debt_weight': debt_weight,
        'wacc': wacc, 'terminal_growth': terminal_growth,
        'shares_outstanding': _safe_float(info.get('sharesOutstanding'), 1.0) or 1.0,
        'cash': _safe_float(info.get('totalCash'), 0.0),
    }


def build_financial_model_bundle(ticker_symbol: str) -> dict:
    """Build a statistical-program-style financial model from the same platform dataset.

    Includes historical/common-size analysis, five-year forecast, WACC, FCFF DCF,
    scenario/sensitivity analysis, trading multiples, DuPont decomposition and model
    outputs suitable for Excel/Power BI. This is a research model, not a prediction
    guarantee or execution instruction.
    """
    data = dict(_analysis(ticker_symbol.upper(), 0.09, 0.025))   # copy: never mutate the cached result
    # run_quant_pipeline does not expose raw info; load lightweight metadata once.
    ticker = _ticker(ticker_symbol.upper())
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    data['_info'] = info
    assumptions = _financial_model_assumptions(data)
    hist = _historical_model_rows(data.get('financial_statements', {}), data.get('ratio_history', []))
    years = list(data.get('financial_statements', {}).get('years') or [])

    # Common-size and historical analytics.
    if not hist.empty:
        hist['Revenue Growth'] = hist['Revenue'].pct_change(-1)
        hist['EBIT Growth'] = hist['EBIT'].pct_change(-1)
        hist['Net Income Growth'] = hist['Net Income'].pct_change(-1)
        hist['FCF Growth'] = hist['FCF'].pct_change(-1)
        hist['Debt / Equity'] = hist['Total Debt'] / hist['Equity'].replace(0, np.nan)
        hist['Asset Turnover'] = hist['Revenue'] / hist['Total Assets'].replace(0, np.nan)
        hist['Equity Multiplier'] = hist['Total Assets'] / hist['Equity'].replace(0, np.nan)
        hist['DuPont ROE'] = hist['Net Margin'] * hist['Asset Turnover'] * hist['Equity Multiplier']
        hist['Common Size Revenue'] = 1.0
        hist['Common Size EBIT'] = hist['EBIT Margin']
        hist['Common Size Net Income'] = hist['Net Margin']
        hist['Common Size FCF'] = hist['FCF Margin']

    # Forecast engine: configurable base case derived from historical data.
    base_rev = assumptions['base_revenue']
    g = assumptions['revenue_growth']
    margin = assumptions['ebit_margin']
    tax_rate = assumptions['tax_rate']
    fcf_margin = assumptions['fcf_margin']
    forecast_rows = []
    prev_rev = base_rev
    for i in range(1, 6):
        rev = prev_rev * (1 + g)
        ebit = rev * margin
        nopat = ebit * (1 - tax_rate)
        fcff = rev * fcf_margin
        forecast_rows.append({'Year': f'Y{i}', 'Revenue': rev, 'Revenue Growth': g, 'EBIT': ebit,
                              'EBIT Margin': margin, 'NOPAT': nopat, 'FCFF': fcff,
                              'Tax Rate': tax_rate, 'FCFF Margin': fcf_margin})
        prev_rev = rev
    forecast = pd.DataFrame(forecast_rows)

    # DCF / FCFF calculations.
    wacc = assumptions['wacc']
    tg = assumptions['terminal_growth']
    forecast['Discount Factor'] = [(1 + wacc) ** i for i in range(1, 6)]
    forecast['PV of FCFF'] = forecast['FCFF'] / forecast['Discount Factor']
    terminal_fcff = float(forecast.iloc[-1]['FCFF']) * (1 + tg)
    terminal_value = terminal_fcff / max(wacc - tg, 0.01)
    pv_terminal = terminal_value / forecast.iloc[-1]['Discount Factor']
    pv_forecast = float(forecast['PV of FCFF'].sum())
    enterprise_value = pv_forecast + pv_terminal
    equity_value_dcf = enterprise_value + assumptions['cash'] - assumptions['debt']
    dcf_per_share = equity_value_dcf / assumptions['shares_outstanding'] if assumptions['shares_outstanding'] else None

    # Trading multiples and implied values.
    current = _safe_float(data.get('current_price'), 0.0)
    latest_eps = _safe_float(info.get('trailingEps'), 0.0)
    latest_sales = _safe_float(info.get('totalRevenue'), 0.0)
    latest_ebitda = _safe_float(info.get('ebitda'), 0.0)
    latest_shares = assumptions['shares_outstanding']
    multiples = []
    if latest_eps > 0:
        multiples.append({'Method':'P/E','Multiple':_safe_float(info.get('trailingPE'), 0),'Metric':latest_eps,'Implied Price':_safe_float(info.get('trailingPE'),0)*latest_eps})
    if latest_sales > 0 and latest_shares > 0:
        ps = _safe_float(info.get('priceToSalesTrailing12Months'), 0)
        if ps > 0: multiples.append({'Method':'P/S','Multiple':ps,'Metric':latest_sales/latest_shares,'Implied Price':ps*(latest_sales/latest_shares)})
    if latest_ebitda > 0 and latest_shares > 0:
        ev_ebitda = _safe_float(info.get('enterpriseToEbitda'), 0)
        net_debt = assumptions['debt'] - assumptions['cash']
        if ev_ebitda > 0: multiples.append({'Method':'EV/EBITDA','Multiple':ev_ebitda,'Metric':latest_ebitda,'Implied Price':(ev_ebitda*latest_ebitda-net_debt)/latest_shares})
    multiples_df = pd.DataFrame(multiples)

    # Scenario analysis.
    scenario_specs = [('Bear', g-0.05, margin-0.03, wacc+0.02), ('Base', g, margin, wacc), ('Bull', g+0.05, margin+0.03, max(wacc-0.015, tg+0.02))]
    scenario_rows = []
    for label, sg, sm, sw in scenario_specs:
        prev = base_rev; pv = 0.0
        fcfs=[]
        for i in range(1,6):
            prev *= (1+sg); fcff=prev*max(sm,0.0)*(1-tax_rate); fcfs.append(fcff); pv += fcff/((1+sw)**i)
        tv=fcfs[-1]*(1+tg)/max(sw-tg,0.01)
        ev=pv+tv/((1+sw)**5)
        eq=ev+assumptions['cash']-assumptions['debt']
        px=eq/latest_shares if latest_shares else None
        scenario_rows.append({'Scenario':label,'Revenue Growth':sg,'EBIT Margin':sm,'WACC':sw,'Enterprise Value':ev,'Equity Value':eq,'Implied Price':px,'Upside':((px-current)/current if px and current else None)})
    scenarios = pd.DataFrame(scenario_rows)

    # WACC build.
    wacc_df = pd.DataFrame([
        {'Component':'Risk-free Rate','Value':assumptions['risk_free_rate']},
        {'Component':'Beta','Value':assumptions['beta']},
        {'Component':'Market Return','Value':assumptions['market_return']},
        {'Component':'Cost of Equity','Value':assumptions['cost_of_equity']},
        {'Component':'Cost of Debt','Value':assumptions['cost_of_debt']},
        {'Component':'Tax Rate','Value':assumptions['tax_rate']},
        {'Component':'Equity Weight','Value':assumptions['equity_weight']},
        {'Component':'Debt Weight','Value':assumptions['debt_weight']},
        {'Component':'WACC','Value':wacc},
        {'Component':'Terminal Growth','Value':tg},
    ])

    # Sensitivity table: WACC x terminal growth -> DCF price.
    wacc_grid = [round(max(wacc + x, tg + 0.01), 4) for x in (-0.02,-0.01,0,0.01,0.02)]
    tg_grid = [round(max(0.005, tg + x), 4) for x in (-0.01,-0.005,0,0.005,0.01)]
    sens = []
    for sw in wacc_grid:
        row={'WACC':sw}
        for sg in tg_grid:
            pv=sum(float(forecast.iloc[i]['FCFF'])/((1+sw)**(i+1)) for i in range(5))
            tv=float(forecast.iloc[-1]['FCFF'])*(1+sg)/max(sw-sg,0.01)
            eq=pv+tv/((1+sw)**5)+assumptions['cash']-assumptions['debt']
            row[f'g={sg:.2%}']=eq/latest_shares if latest_shares else None
        sens.append(row)
    sensitivity = pd.DataFrame(sens)

    # Statistical diagnostics: correlation of key historical model variables where enough data exists.
    corr_cols=['Revenue','EBIT','Net Income','FCF','Total Debt','Equity']
    corr = hist[corr_cols].corr() if not hist.empty else pd.DataFrame()
    corr_long = corr.reset_index().melt(id_vars='index', var_name='Variable', value_name='Correlation') if not corr.empty else pd.DataFrame(columns=['index','Variable','Correlation'])

    dashboard = {
        'Ticker': data['ticker'], 'Company': data['company_name'], 'Current Price': current,
        'DCF Price': dcf_per_share, 'WACC': wacc, 'Terminal Growth': tg,
        'Enterprise Value': enterprise_value, 'Equity Value': equity_value_dcf,
        'PV Forecast': pv_forecast, 'PV Terminal': pv_terminal,
        'Forecast CAGR': g, 'EBIT Margin': margin, 'FCFF Margin': fcf_margin,
        'Cost of Equity': assumptions['cost_of_equity'], 'Cost of Debt': assumptions['cost_of_debt'],
        'Beta': assumptions['beta'], 'Risk Free Rate': assumptions['risk_free_rate'],
    }
    return {
        'ticker': data['ticker'], 'company_name': data['company_name'], 'currency': data['currency'],
        'dashboard': dashboard, 'assumptions': pd.DataFrame([assumptions]), 'historical': hist,
        'forecast': forecast, 'dcf': pd.DataFrame([
            {'Metric':'PV of Forecast FCFF','Value':pv_forecast}, {'Metric':'Terminal Value','Value':terminal_value},
            {'Metric':'PV of Terminal Value','Value':pv_terminal}, {'Metric':'Enterprise Value','Value':enterprise_value},
            {'Metric':'Cash','Value':assumptions['cash']}, {'Metric':'Debt','Value':assumptions['debt']},
            {'Metric':'Equity Value','Value':equity_value_dcf}, {'Metric':'Shares Outstanding','Value':latest_shares},
            {'Metric':'DCF Value / Share','Value':dcf_per_share}
        ]),
        'wacc': wacc_df, 'multiples': multiples_df, 'scenarios': scenarios,
        'sensitivity': sensitivity, 'dupont': hist[['Year','Net Margin','Asset Turnover','Equity Multiplier','DuPont ROE']].copy() if not hist.empty else pd.DataFrame(),
        'correlation': corr_long,
        'source_analysis': data,
    }


# ------------------------------------------------------------------------------
# 4. API ROUTE HANDLERS
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# 4g. Fundamental Analysis Integration
# ------------------------------------------------------------------------------

MARKET_OPPORTUNITY_UNIVERSE = [
    ("SPY", "US Large Cap ETF"), ("QQQ", "Nasdaq 100 ETF"), ("DIA", "Dow Jones ETF"),
    ("IWM", "US Small Cap ETF"), ("NVDA", "NVIDIA"), ("AAPL", "Apple"), ("MSFT", "Microsoft"),
    ("AMZN", "Amazon"), ("META", "Meta Platforms"), ("GOOGL", "Alphabet"), ("TSLA", "Tesla"),
    ("AMD", "AMD"), ("AVGO", "Broadcom"), ("NFLX", "Netflix"), ("JPM", "JPMorgan"),
    ("XOM", "Exxon Mobil"), ("CVX", "Chevron"), ("GLD", "Gold ETF"), ("SLV", "Silver ETF"),
    ("TLT", "20+ Year Treasury ETF"), ("HYG", "High Yield Bond ETF"), ("EEM", "Emerging Markets ETF"),
    ("FXI", "China Large Cap ETF"), ("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"),
]

def _clip_pct(x: float, scale: float) -> float:
    return float(np.clip(x / scale if scale else 0.0, -1.0, 1.0))

def _scanner_row(symbol: str, name: str, hist: pd.DataFrame) -> dict | None:
    if hist is None or hist.empty or 'Close' not in hist:
        return None
    close = pd.to_numeric(hist['Close'], errors='coerce').dropna()
    if len(close) < 35:
        return None
    volume = pd.to_numeric(hist['Volume'], errors='coerce').reindex(close.index) if 'Volume' in hist else pd.Series(index=close.index, dtype=float)
    daily_ret = close.pct_change().dropna()
    price = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float(close.rolling(35).mean().iloc[-1])
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float((100 - 100/(1+rs)).fillna(50).iloc[-1])
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_hist = float((ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()).iloc[-1])
    ret1 = float(close.pct_change(1).iloc[-1])
    ret5 = float(close.pct_change(5).iloc[-1])
    ret20 = float(close.pct_change(20).iloc[-1])
    vol20 = float(daily_ret.std(ddof=1) * np.sqrt(252)) if len(daily_ret) > 10 else 0.0
    atr_pct = float(close.diff().abs().rolling(14).mean().iloc[-1] / price) if price else 0.0
    vol_ratio = 1.0
    if volume.notna().sum() >= 21:
        v20 = volume.rolling(20).mean().iloc[-1]
        if pd.notna(v20) and v20:
            vol_ratio = float(volume.iloc[-1] / v20)
    day_score = 100*(0.35*_clip_pct(ret1,.03) + 0.40*_clip_pct(ret5,.08) + 0.25*(1 if price>sma20 else -1))
    week_score = 100*(0.50*_clip_pct(ret5,.08) + 0.30*_clip_pct(ret20,.15) + 0.20*(1 if price>sma50 else -1))
    adj = (8 if rsi>55 else -8 if rsi<45 else 0) + (6 if macd_hist>0 else -6) + ((4 if ret5>=0 else -4) if vol_ratio>1.2 else 0)
    day_score=float(np.clip(day_score+adj,-100,100)); week_score=float(np.clip(week_score+adj*.5,-100,100))
    composite=float(np.clip(.45*day_score+.55*week_score,-100,100))
    direction='LONG CANDIDATE' if composite>=25 else ('SHORT CANDIDATE' if composite<=-25 else 'WATCH')
    aligned=(day_score>=15 and week_score>=15) or (day_score<=-15 and week_score<=-15)
    if direction=='LONG CANDIDATE':
        hypothesis='Day + week momentum aligned; continuation scenario is stronger.' if aligned else ('Weekly trend is stronger; pullback/continuation scenario.' if day_score<week_score else 'Daily strength leads; weekly confirmation is useful.')
    elif direction=='SHORT CANDIDATE':
        hypothesis='Day + week momentum aligned to the downside; continuation scenario is stronger.' if aligned else ('Weekly weakness dominates; rebounds can meet broader trend pressure.' if day_score>week_score else 'Daily weakness leads; weekly confirmation is useful.')
    else:
        hypothesis='Mixed timeframes; range/mean-reversion or wait-for-confirmation scenario.'
    risk_flag='ELEVATED' if vol20>.55 or atr_pct>.045 else ('WATCH' if vol20>.35 or atr_pct>.03 else 'NORMAL')
    scenario='Bullish breakout continuation' if composite>45 else ('Bearish breakdown continuation' if composite<-45 else 'Two-way / confirmation needed')
    return {'ticker':symbol,'name':name,'price':round(price,4),'day_score':round(day_score,1),'week_score':round(week_score,1),'composite_score':round(composite,1),'direction':direction,'ret_1d_pct':round(ret1*100,2),'ret_5d_pct':round(ret5*100,2),'ret_20d_pct':round(ret20*100,2),'rsi14':round(rsi,1),'macd_hist':round(macd_hist,5),'volume_ratio':round(vol_ratio,2),'volatility_annualized_pct':round(vol20*100,1),'atr_pct':round(atr_pct*100,2),'risk_flag':risk_flag,'hypothesis':hypothesis,'scenario':scenario,'timeframe_alignment':'ALIGNED' if aligned else 'MIXED'}

def scan_market_opportunities(symbols: list[str] | None = None) -> dict:
    name_map={s:n for s,n in MARKET_OPPORTUNITY_UNIVERSE}
    requested=[str(x).strip().upper() for x in (symbols or name_map.keys()) if str(x).strip()][:40]
    rows=[]
    frames=_batch_history(requested, '6mo')
    for sym in requested:
        h=frames.get(sym)
        if h is None:
            continue
        row=_scanner_row(sym,name_map.get(sym,sym),h)
        if row: rows.append(row)
    rows.sort(key=lambda r: abs(r['composite_score']), reverse=True)
    return {'updated_at':_dt.now(timezone.utc).isoformat().replace('+00:00', 'Z'),'universe_size':len(rows),'rows':rows,'long_candidates':[r for r in rows if r['direction']=='LONG CANDIDATE'][:8],'short_candidates':[r for r in rows if r['direction']=='SHORT CANDIDATE'][:8],'watchlist':[r for r in rows if r['direction']=='WATCH'][:8],'methodology':'Daily/weekly research-bias scanner using 1D/5D/20D returns, SMA20/50, RSI, MACD histogram, volume confirmation and volatility. Labels are scenario hypotheses, not probability-of-profit or execution signals.'}

def _opportunities_cached(tickers: str | None) -> dict:
    key = f"opportunities::{(tickers or '').strip().upper()}"
    symbols = [x for x in tickers.split(',') if x.strip()] if tickers else None
    return _cached_call(key, lambda: scan_market_opportunities(symbols), heavy=False,
                        cache_if=lambda r: bool(r.get('rows')))


@app.get("/api/market-opportunities")
def get_market_opportunities(tickers: str | None = Query(default=None)):
    return _opportunities_cached(tickers)


INSTRUMENT_BOARD = {
    'Equities': [('SPY','S&P 500 ETF'),('QQQ','Nasdaq 100 ETF'),('DIA','Dow Jones ETF'),('IWM','Russell 2000 ETF'),('NVDA','NVIDIA'),('AAPL','Apple'),('MSFT','Microsoft'),('AMZN','Amazon'),('META','Meta'),('TSLA','Tesla')],
    'Currencies': [('EURUSD=X','EUR/USD'),('GBPUSD=X','GBP/USD'),('USDJPY=X','USD/JPY'),('AUDUSD=X','AUD/USD'),('USDCNY=X','USD/CNY'),('USDCHF=X','USD/CHF')],
    'Metals': [('GC=F','Gold Futures'),('SI=F','Silver Futures'),('PL=F','Platinum Futures'),('HG=F','Copper Futures')],
    'Energy': [('CL=F','WTI Crude Oil'),('BZ=F','Brent Crude'),('NG=F','Natural Gas')],
    'Commodities': [('ZC=F','Corn'),('ZW=F','Wheat'),('ZS=F','Soybeans'),('KC=F','Coffee')],
    'Rates / Bonds': [('^TNX','US 10Y Yield'),('^FVX','US 5Y Yield'),('^TYX','US 30Y Yield')],
    'Crypto': [('BTC-USD','Bitcoin'),('ETH-USD','Ethereum')],
}


def _instrument_row(symbol: str, name: str, hist: pd.DataFrame | None = None) -> dict:
    try:
        if hist is None:
            hist = _ticker(symbol).history(period='6mo', interval='1d', auto_adjust=True)
        close = pd.to_numeric(hist.get('Close'), errors='coerce').dropna() if not hist.empty else pd.Series(dtype=float)
        if close.empty:
            raise ValueError('No quote data')
        last=float(close.iloc[-1]); prev=float(close.iloc[-2]) if len(close)>=2 else last
        p5=float(close.iloc[-6]) if len(close)>=6 else float(close.iloc[0])
        p20=float(close.iloc[-21]) if len(close)>=21 else float(close.iloc[0])
        return {'symbol':symbol,'name':name,'price':last,'change_1d':(last/prev-1) if prev else 0.0,
                'change_5d':(last/p5-1) if p5 else 0.0,'change_20d':(last/p20-1) if p20 else 0.0,
                'history_dates':close.tail(120).index.strftime('%Y-%m-%d').tolist(),
                'history_close':close.tail(120).round(6).tolist()}
    except Exception as exc:
        return {'symbol':symbol,'name':name,'price':None,'change_1d':None,'change_5d':None,'change_20d':None,
                'history_dates':[],'history_close':[],'error':str(exc)}


def _build_instrument_board(category: str | None, symbols: str | None) -> dict:
    name_lookup = {sym: name for items in INSTRUMENT_BOARD.values() for sym, name in items}
    if symbols:
        requested = [s.strip().upper() for s in symbols.split(',') if s.strip()][:40]
        frames = _batch_history(requested, '6mo')
        return {'category': 'Custom',
                'rows': [_instrument_row(sym, name_lookup.get(sym, sym), frames.get(sym, pd.DataFrame())) for sym in requested]}
    cats = [category] if category in INSTRUMENT_BOARD else list(INSTRUMENT_BOARD.keys())
    frames = _batch_history([sym for cat in cats for sym, _ in INSTRUMENT_BOARD[cat]], '6mo')
    return {'categories': {cat: [_instrument_row(sym, name, frames.get(sym, pd.DataFrame())) for sym, name in INSTRUMENT_BOARD[cat]]
                           for cat in cats}}


def _board_has_prices(result: dict) -> bool:
    groups = list((result.get('categories') or {}).values()) or [result.get('rows') or []]
    return any(row.get('price') is not None for rows in groups for row in rows)


def _instrument_board_cached(category: str | None, symbols: str | None) -> dict:
    key = f"board::{category or ''}::{(symbols or '').strip().upper()}"
    return _cached_call(key, lambda: _build_instrument_board(category, symbols), heavy=False, cache_if=_board_has_prices)


@app.get('/api/instrument-board')
def get_instrument_board(category: str | None = Query(default=None), symbols: str | None = Query(default=None)):
    return _instrument_board_cached(category, symbols)


def _analysis_key(symbol: str, discount_rate: float, perpetual_growth: float) -> str:
    return f"{symbol}|{float(discount_rate)}|{float(perpetual_growth)}"


def _analysis(symbol: str, discount_rate: float = 0.09, perpetual_growth: float = 0.025) -> dict:
    """Cached, coalesced, timed analysis shared by the dashboard, exports and model studio."""
    symbol = str(symbol or "").strip().upper()
    return _cached_call(
        _analysis_key(symbol, discount_rate, perpetual_growth),
        lambda: _with_timings(f"analysis {symbol}",
                              lambda: run_quant_pipeline(symbol, discount_rate, perpetual_growth)),
    )


def _frame_records(df):
    if df is None or df.empty:
        return []
    return df.replace({np.nan: None}).to_dict('records')


@app.get('/api/financial-model/{ticker}')
def get_financial_model(ticker: str):
    symbol = ticker.strip().upper()

    def build():
        bundle = build_financial_model_bundle(symbol)
        result = {k: bundle[k] for k in ['ticker', 'company_name', 'currency', 'dashboard']}
        result.update({k: _frame_records(bundle[k]) for k in
                       ['historical', 'forecast', 'dcf', 'wacc', 'multiples', 'scenarios', 'sensitivity', 'dupont', 'correlation']})
        return result

    try:
        return _cached_call(f"financial-model::{symbol}", build)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Financial model error: {type(exc).__name__}: {exc}")


@app.get('/api/download/financial-model/{ticker}')
def download_financial_model(ticker: str):
    try:
        bundle = build_financial_model_bundle(ticker)
        data = build_financial_model_excel(bundle)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Financial model export unavailable: {type(exc).__name__}: {exc}")
    filename = f"Finmodel_{ticker.upper()}_Financial_Model.xlsx"
    return StreamingResponse(io.BytesIO(data), media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             headers={'Content-Disposition': f'attachment; filename="{filename}"'})


def _excel_download(ticker: str, mode: str, suffix: str):
    try:
        bundle = build_export_bundle(ticker)
        data = build_excel_bytes(bundle, mode=mode)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Export service unavailable: {type(exc).__name__}: {exc}")
    filename = f"Finmodel_{ticker.upper()}_{suffix}.xlsx"
    return StreamingResponse(io.BytesIO(data), media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             headers={'Content-Disposition': f'attachment; filename="{filename}"'})


@app.get("/api/download/excel/{ticker}")
def download_all_excel(ticker: str):
    return _excel_download(ticker, 'all', 'Export')


@app.get("/api/download/excel/{ticker}/market")
def download_market_excel(ticker: str):
    return _excel_download(ticker, 'market', 'Historical_Excess_Returns')


@app.get("/api/download/excel/{ticker}/fundamentals")
def download_fundamentals_excel(ticker: str):
    return _excel_download(ticker, 'fundamentals', 'Ratios_Statements')


@app.get("/api/download/powerbi/{ticker}")
def download_powerbi_package(ticker: str):
    try:
        bundle = build_export_bundle(ticker)
        data = build_powerbi_project_zip(bundle)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Export service unavailable: {type(exc).__name__}: {exc}")
    filename = f"Finmodel_{ticker.upper()}_PowerBI_Project.zip"
    return StreamingResponse(io.BytesIO(data), media_type='application/zip',
                             headers={'Content-Disposition': f'attachment; filename="{filename}"'})


@app.get("/api/economic-calendar")
def get_economic_calendar():
    return {"calendar": build_economic_calendar()}


@app.get("/api/news/{ticker}")
def get_news(ticker: str):
    symbol = ticker.strip().upper()

    def build():
        try:
            sector = _ticker(symbol).info.get("sector", "Technology")
        except Exception:
            sector = "Technology"
        return {"ticker": symbol, "news": fetch_asset_news(symbol, sector), "economic_calendar": build_economic_calendar()}

    return _cached_call(f"news::{symbol}", build, heavy=False)


@app.get("/api/quick/{ticker}")
def get_quick_snapshot(ticker: str):
    """Fast first paint: name, price, chart and technicals in about a second.

    The browser calls this and the full analysis at the same time; both share the
    same underlying downloads through the data layer, so this costs nothing extra.
    """
    symbol = ticker.strip().upper()

    def build():
        t = _ticker(symbol)
        try:
            info = t.info or {}
        except Exception:
            info = {}
        hist = t.history(period='5y', auto_adjust=True)
        if hist is None or hist.empty or 'Close' not in hist:
            raise ValueError(f"No price history available for '{symbol}'.")
        price = _latest_yf_price(t, info, hist)
        if price is None:
            raise ValueError(f"Could not fetch price for '{symbol}'.")
        asset_class, name, quote_unit = classify_ticker_asset(symbol, info)
        dec = _price_decimals(price)
        close = pd.to_numeric(hist['Close'], errors='coerce').ffill()
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        return {
            'ticker': symbol, 'company_name': name, 'asset_class': asset_class,
            'analysis_mode': 'ISSUER' if asset_class in {'Equity', 'ETF / Fund'} else 'MARKET_ASSET',
            'quote_unit': quote_unit if asset_class not in {'Equity', 'ETF / Fund'} else (info.get('financialCurrency') or 'USD'),
            'currency': info.get('financialCurrency') or info.get('currency') or 'USD',
            'sector': info.get('sector') or asset_class,
            'current_price': round(float(price), dec),
            'chart_data': {'dates': dates, 'historical_close': close.round(dec).tolist(), 'sma_20': [], 'sma_50': [],
                           'ml_forecast_dates': [dates[-1]], 'ml_forecast_prices': [round(float(price), dec)]},
            'technicals': calculate_technicals(hist),
        }

    try:
        return _cached_call(f"quick::{symbol}", build, heavy=False, ttl=_DATA_TTL['history'])
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Quote provider error: {type(exc).__name__}: {exc}")


@app.get("/health")
def health_check():
    return {
        "status": "healthy", "engine": "finmodel", "version": "17.0.0",
        "cached_results": len(_CACHE), "cached_datasets": len(_DATA), "ml_models": len(_ML_CACHE),
        "heavy_job_slots": _MAX_HEAVY_JOBS, "risk_free_rate": round(_risk_free_rate(), 4),
        "stats": dict(_STATS),
        "export_endpoints": True, "powerbi_project_export": True,
        "fred_api_key": bool(os.environ.get("FRED_API_KEY", "").strip()),
    }


@app.get("/api/portfolio/optimize")
def api_portfolio_optimize(
    tickers: str = Query(..., description="Comma-separated arbitrary asset tickers"),
    method: str = Query('MIN_RISK'),
    lookback: str = Query('5y'),
    benchmark: str = Query('^GSPC')
):
    if lookback not in {'1y', '2y', '3y', '5y', '10y'}:
        raise HTTPException(status_code=400, detail="lookback must be one of 1y, 2y, 3y, 5y, 10y")
    cache_key = f"portfolio::{tickers.strip().upper()}::{method.upper()}::{lookback}::{benchmark.upper()}"
    try:
        return _cached_call(cache_key, lambda: optimize_portfolio(tickers, method=method, lookback=lookback,
                                                                  benchmark_symbol=benchmark))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/analysis/{ticker}", response_model=ValuationResponse)
def get_ticker_analysis(
        ticker: str,
        discount_rate: float = Query(0.09, ge=0.01, le=0.30),
        perpetual_growth: float = Query(0.025, ge=0.00, le=0.08)
):
    try:
        return _analysis(ticker, discount_rate, perpetual_growth)
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# ------------------------------------------------------------------------------
# 4h. PRE-WARMING + STATIC-SITE EXPORT
# ------------------------------------------------------------------------------
_WARM_TICKERS = [t for t in _normalize_ticker_list(os.environ.get(
    "WARM_TICKERS", "AAPL,MSFT,NVDA,SPY,QQQ,GC=F,CL=F,BTC-USD,EURUSD=X"))]
_WARM_INTERVAL_SECONDS = int(os.environ.get("WARM_INTERVAL_SECONDS", "0"))
_WARM_LOCK = threading.Lock()
_WARMER_STARTED = threading.Event()


def warm_cache() -> dict:
    """Compute the board, scanner and popular tickers so visitors get instant answers."""
    if not _WARM_LOCK.acquire(blocking=False):
        return {"status": "already running"}
    summary = {"ok": [], "failed": []}
    started = time.perf_counter()
    try:
        for category in INSTRUMENT_BOARD:
            try:
                _instrument_board_cached(category, None)
            except Exception as exc:
                summary["failed"].append(f"board:{category}: {exc}")
        try:
            _opportunities_cached(None)
        except Exception as exc:
            summary["failed"].append(f"scanner: {exc}")
        try:
            finmodel_intel.calendar_snapshot(block=True)
        except Exception as exc:
            summary["failed"].append(f"calendar: {exc}")
        for sym in _WARM_TICKERS:
            try:
                _analysis(sym)
                summary["ok"].append(sym)
            except Exception as exc:
                summary["failed"].append(f"{sym}: {exc}")
    finally:
        _WARM_LOCK.release()
    summary["seconds"] = round(time.perf_counter() - started, 1)
    log.info("cache warm finished: %s", summary)
    return summary


def _warm_loop():
    while True:
        try:
            warm_cache()
        except Exception as exc:
            log.warning("warm loop error: %s", exc)
        time.sleep(max(60, _WARM_INTERVAL_SECONDS))


@app.middleware("http")
async def _start_background_warmer(request: Request, call_next):
    # Always-on hosts (a VM) can keep popular tickers warm with WARM_INTERVAL_SECONDS.
    if _WARM_INTERVAL_SECONDS > 0 and not _WARMER_STARTED.is_set():
        _WARMER_STARTED.set()
        threading.Thread(target=_warm_loop, name="fm-warmer", daemon=True).start()
    return await call_next(request)


@app.get("/api/admin/warm")
def admin_warm(token: str = Query(default="")):
    """For a scheduler (e.g. Cloud Scheduler). Enabled only when WARM_TOKEN is set."""
    expected = os.environ.get("WARM_TOKEN", "")
    if not expected or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=404, detail="Not Found")
    return warm_cache()


_API_BASE_RE = re.compile(r"^https://[A-Za-z0-9.\-]+(:\d+)?/?$")


@app.get("/download/static-site")
def download_static_site(api_base: str = Query(..., description="Public URL of this API, e.g. https://finmodel-xyz.a.run.app")):
    """The dashboard as a standalone index.html for free static hosting
    (Cloudflare Pages, GitHub Pages, Netlify). It calls this API for data, so the
    page itself appears instantly even while the API wakes from scale-to-zero."""
    if not _API_BASE_RE.match(api_base):
        raise HTTPException(status_code=400, detail="api_base must look like https://your-api.example.com")
    page = render_dashboard().replace('<meta name="finmodel-api" content="">',
                                      f'<meta name="finmodel-api" content="{_html.escape(api_base.rstrip("/"))}">', 1)
    return HTMLResponse(page, headers={'Content-Disposition': 'attachment; filename="index.html"'})


# ------------------------------------------------------------------------------
# 5. FRONTEND DASHBOARD TEMPLATE WITH PEER COMPARISON CHARTS
# ------------------------------------------------------------------------------
import sys as _sys
import finmodel_intel
import finmodel_intel_ui
finmodel_intel.install(app, _sys.modules[__name__])


@app.get("/", response_class=HTMLResponse)
def render_dashboard():
    return finmodel_intel_ui.inject(_dashboard_base())


def _dashboard_base():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="finmodel-api" content="">
<title>finmodel | Advanced Quant Terminal</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --bg:#05080d;
    --panel:#0b1119;
    --line:#1b2734;
    --muted:#718096;
    --text:#e7edf5;
    --green:#19d58b;
    --cyan:#22d3ee;
    --red:#fb7185;
    --amber:#fbbf24;
}

* { box-sizing:border-box; }
body {
    margin:0;
    background:
        radial-gradient(circle at 80% -10%, rgba(34,211,238,.08), transparent 30%),
        radial-gradient(circle at 10% 0%, rgba(25,213,139,.06), transparent 25%),
        var(--bg);
    color:var(--text);
    font-family:Inter,system-ui,sans-serif;
}
.mono { font-family:"JetBrains Mono",monospace; }
.panel {
    background:linear-gradient(180deg,rgba(14,22,32,.96),rgba(8,14,21,.96));
    border:1px solid var(--line);
    border-radius:14px;
}
.green { color:var(--green); }
.cyan { color:var(--cyan); }
.red { color:var(--red); }
.amber { color:var(--amber); }
.muted { color:var(--muted); }

.nav {
    width:235px;
    min-width:180px;
    max-width:460px;
    height:calc(100vh - 32px);
    min-height:300px;
    max-height:calc(100vh - 32px);
    position:fixed;
    left:16px;
    top:16px;
    overflow-y:auto;
    overflow-x:hidden;
    border:1px solid var(--line);
    border-radius:16px;
    background:rgba(7,11,17,.97);
    box-shadow:0 18px 55px rgba(0,0,0,.42),0 0 0 1px rgba(34,211,238,.04) inset;
    z-index:50;
    will-change:left,top,width,height;
    resize:none;
}
.nav-resize-handle {
    position:absolute;
    z-index:60;
    touch-action:none;
    user-select:none;
}
.nav-resize-right, .nav-resize-left {
    top:12px;
    bottom:12px;
    width:10px;
    cursor:ew-resize;
}
.nav-resize-right { right:-5px; }
.nav-resize-left { left:-5px; }
.nav-resize-top, .nav-resize-bottom {
    left:12px;
    right:12px;
    height:10px;
    cursor:ns-resize;
}
.nav-resize-top { top:-5px; }
.nav-resize-bottom { bottom:-5px; }
.nav-resize-tl, .nav-resize-tr, .nav-resize-bl, .nav-resize-br {
    width:16px;
    height:16px;
}
.nav-resize-tl { left:-5px; top:-5px; cursor:nwse-resize; }
.nav-resize-tr { right:-5px; top:-5px; cursor:nesw-resize; }
.nav-resize-bl { left:-5px; bottom:-5px; cursor:nesw-resize; }
.nav-resize-br { right:-5px; bottom:-5px; cursor:nwse-resize; }
.nav-resize-right::after, .nav-resize-left::after, .nav-resize-top::after, .nav-resize-bottom::after {
    content:"";
    position:absolute;
    opacity:.15;
    background:rgba(34,211,238,.75);
}
.nav-resize-right::after, .nav-resize-left::after { top:8px; bottom:8px; width:2px; }
.nav-resize-right::after { right:3px; }
.nav-resize-left::after { left:3px; }
.nav-resize-top::after, .nav-resize-bottom::after { left:8px; right:8px; height:2px; }
.nav-resize-top::after { top:3px; }
.nav-resize-bottom::after { bottom:3px; }
.nav-resize-br::after, .nav-resize-bl::after, .nav-resize-tr::after, .nav-resize-tl::after {
    content:"";
    position:absolute;
    width:8px;
    height:8px;
    opacity:.75;
}
.nav-resize-br::after { right:3px; bottom:3px; border-right:2px solid rgba(34,211,238,.7); border-bottom:2px solid rgba(34,211,238,.7); }
.nav-resize-bl::after { left:3px; bottom:3px; border-left:2px solid rgba(34,211,238,.7); border-bottom:2px solid rgba(34,211,238,.7); }
.nav-resize-tr::after { right:3px; top:3px; border-right:2px solid rgba(34,211,238,.7); border-top:2px solid rgba(34,211,238,.7); }
.nav-resize-tl::after { left:3px; top:3px; border-left:2px solid rgba(34,211,238,.7); border-top:2px solid rgba(34,211,238,.7); }
.nav.resizing-sidebar {
    user-select:none;
    box-shadow:0 24px 70px rgba(0,0,0,.58),0 0 0 1px rgba(34,211,238,.4) inset;
}

.nav.dragging-sidebar {
    user-select:none;
    box-shadow:0 24px 70px rgba(0,0,0,.58),0 0 0 1px rgba(34,211,238,.35) inset;
}
.nav-drag-handle { cursor:move; user-select:none; touch-action:none; }
.nav-drag-handle button { cursor:pointer; touch-action:auto; }
.main-shell::before {
    content:"";
    position:fixed;
    top:0;
    bottom:0;
    left:-12px;
    width:1px;
    pointer-events:none;
    opacity:.22;
    background:linear-gradient(to bottom,transparent,var(--line) 12%,var(--line) 88%,transparent);
}
.main-shell.separate-right::before { left:auto; right:-12px; }
@media (max-width: 900px) { .main-shell::before { display:none; } }

.main-shell {
    min-width:0;
    min-height:100vh;
    margin-left:calc(235px + 40px);
    margin-right:24px;
    transition:margin .12s ease,width .12s ease;
    isolation:isolate;
}
.main {
    margin:0;
    min-height:100vh;
    width:100%;
    min-width:0;
}
.main-shell.separate-left { margin-left:var(--main-offset-left, 275px); margin-right:24px; }
.main-shell.separate-right { margin-left:24px; margin-right:var(--main-offset-right, 275px); }
@media (max-width: 900px) {
    .main-shell, .main-shell.separate-left, .main-shell.separate-right {
        margin:0 12px 0 12px;
    }
}
.workspace-nav {
    display:flex; flex-direction:column; gap:0; min-height:20px;
}
.workspace-drag-note {
    float:right; opacity:.55; font-size:8px; text-transform:none; letter-spacing:0;
}
.navlink {
    display:flex; align-items:center; gap:12px;
    padding:11px 14px; margin:4px 10px;
    border-radius:9px; color:#8492a3;
    cursor:pointer; font-size:13px;
    transition: all 0.15s ease;
}
.navlink:hover,.navlink.active {
    color:#fff; background:#111b27; border-left: 3px solid var(--green);
}
.navlink[draggable="true"] { cursor:grab; user-select:none; touch-action:none; }
.navlink[draggable="true"]:active { cursor:grabbing; }
.navlink.dragging { opacity:.45; outline:1px dashed var(--cyan); }
.navlink.drop-target { border-top:2px solid var(--cyan); }
.workspace-nav .navlink:active { cursor:grabbing; }
.nav.collapsed { width:72px; min-width:72px; }
.nav.collapsed .nav-brand-text, .nav.collapsed .navtitle, .nav.collapsed .nav-section-title, .nav.collapsed .nav-footer, .nav.collapsed .navlink span { display:none; }
.nav.collapsed .navlink { justify-content:center; padding:12px 8px; margin:5px 7px; }
.main-shell.nav-main-collapsed { margin-left:112px; }

/* Sidebar and main screen are two independent workspace views.
   The sidebar never occupies the main view's scroll/layout flow. */
body.workspace-separated {
    overflow:hidden;
}
.main-shell,
.main-shell.separate-left,
.main-shell.separate-right,
.main-shell.nav-main-collapsed {
    position:fixed !important;
    top:16px !important;
    bottom:16px !important;
    margin:0 !important;
    min-height:0 !important;
    border:1px solid var(--line);
    border-radius:16px;
    background:rgba(5,8,13,.72);
    box-shadow:0 18px 55px rgba(0,0,0,.30);
    overflow:hidden;
    isolation:isolate;
}
.main-shell > .main {
    width:100%;
    height:100%;
    min-height:0;
    overflow:auto;
    scrollbar-gutter:stable;
}
.main-shell.separate-left::before,
.main-shell.separate-right::before {
    opacity:.42;
}
@media (max-width:900px) {
    .main-shell,
    .main-shell.separate-left,
    .main-shell.separate-right,
    .main-shell.nav-main-collapsed {
        top:16px !important;
        bottom:16px !important;
        left:12px !important;
        right:12px !important;
        width:auto !important;
    }
    .main-shell > .main { height:100%; }
}
.badge {
    border:1px solid #174d3b;
    background:#09261d;
    color:var(--green);
    border-radius:999px;
    padding:4px 8px;
    font-size:10px;
}
.input {
    background:#070c12;
    border:1px solid #263444;
    color:#fff;
    border-radius:999px;
    outline:none;
}
.btn {
    border:1px solid #155d49;
    background:#0a2d23;
    color:var(--green);
    border-radius:999px;
    font-weight:700;
    cursor:pointer;
}
.btn:hover { background:#0d3a2c; }
.kpi { padding:17px; min-height:112px; }
.kpi .label { color:#718096; text-transform:uppercase; letter-spacing:.08em; font-size:10px; font-weight:700; }
.kpi .value { font-family:"JetBrains Mono",monospace; font-size:24px; font-weight:700; margin-top:9px; }

.tab { color:#728095; padding:6px 12px; border-radius:7px; cursor:pointer; font-size:12px; font-weight:600; }
.tab.active { color:#fff; background:#172230; }

.view-section { display: none; }
.view-section.active-view { display: block; }

.table-wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; }
th,td { padding:11px 9px; border-bottom:1px solid #17212d; }
th { color:#667487; font-size:10px; text-transform:uppercase; letter-spacing:.07em; text-align:right; }
th:first-child,td:first-child { text-align:left; }
td { font-family:"JetBrains Mono",monospace; font-size:12px; text-align:right; }
td:first-child { font-family:Inter,sans-serif; color:#c5cfdb; }

.metric-big { font-family:"JetBrains Mono",monospace; font-size:28px; font-weight:800; }
.trade-input { width:100%; padding:9px 11px; border-radius:9px; background:#070c12; border:1px solid #263444; color:#fff; outline:none; }
.news-row:hover { background:#0d1822; }
.badge.green { border-color:#14532d; color:#22c55e; }
.badge.red { border-color:#7f1d1d; color:#fb7185; }
.badge.amber { border-color:#78350f; color:#fbbf24; }
.status-dot { width:7px; height:7px; border-radius:50%; display:inline-block; background:var(--green); box-shadow:0 0 9px rgba(25,213,139,.7); }
</style>
</head>

<body class="workspace-separated">
<aside class="nav p-5">
    <div class="flex items-center gap-2 mb-9 px-2 nav-brand nav-drag-handle" title="Drag sidebar anywhere across the workspace"><button id="navCollapseBtn" class="btn px-2 py-1 text-xs" onclick="toggleNav(); event.stopPropagation()">⇆</button>
        <div class="w-9 h-9 rounded-xl flex items-center justify-center bg-emerald-950 border border-emerald-800 text-emerald-400 font-black mono">F</div>
        <div>
            <div class="text-lg font-black mono tracking-tight nav-brand-text">finmodel</div>
            <div class="text-[9px] muted uppercase tracking-[.2em]">quant terminal</div>
        </div>
    </div>

    <div class="text-[9px] muted uppercase tracking-[.2em] px-4 mb-2">Workspace <span class="workspace-drag-note">↕ reorder • ⤢ move sidebar</span></div>
    <div id="workspaceNav" class="workspace-nav" aria-label="Workspace navigation">
        <div class="navlink active" data-view="dashboard" onclick="switchView('dashboard', this)">▦ <span>Dashboard</span></div>
        <div class="navlink" data-view="portfolio" onclick="switchView('portfolio', this)">◫ <span>Portfolio Optimizer</span></div>
        <div class="navlink" data-view="risk" onclick="switchView('risk', this)">◰ <span>CAPM / CAL / SML</span></div>
        <div class="navlink" data-view="derivatives" onclick="switchView('derivatives', this)">◈ <span>Futures Simulation</span></div>
        <div class="navlink" data-view="cfd" onclick="switchView('cfd', this)">⌁ <span>CFD Scenario Lab</span></div>
        <div class="navlink" data-view="trade" onclick="switchView('trade', this)">⚡ <span>Trade Lab</span></div>
        <div class="navlink" data-view="news" onclick="switchView('news', this)">◉ <span>Macro Calendar & News</span></div>
        <div class="navlink" data-view="financials" onclick="switchView('financials', this)">▤ <span>Financial Ratios</span></div>
        <div class="navlink" data-view="exports" onclick="switchView('exports', this)">⇩ <span>Data Export</span></div>
        <div class="navlink" data-view="model" onclick="switchView('model', this)">∮ <span>Financial Model Studio</span></div>
        <div class="navlink" data-view="fundamentals" onclick="switchView('fundamentals', this)">◉ <span>Fundamental Analysis</span></div>
        <div class="navlink" data-view="markets" onclick="switchView('markets', this)">⌕ <span>Market Scanner</span></div>
        <div class="navlink" data-view="opportunities" onclick="switchView('opportunities', this)">◒ <span>Day / Week Opportunities</span></div>
        <div class="navlink" data-view="peers" onclick="switchView('peers', this)">⇄ <span>Peer Compare</span></div>
        <div class="navlink" data-view="valuation" onclick="switchView('valuation', this)">∑ <span>Valuation Lab</span></div>
        <div class="navlink" data-view="forecast" onclick="switchView('forecast', this)">⌁ <span>Technical Forecast</span></div>
    </div>
    <div class="nav-resize-handle nav-resize-left" data-resize="l" title="Drag left edge to resize"></div>
    <div class="nav-resize-handle nav-resize-right" data-resize="r" title="Drag right edge to resize"></div>
    <div class="nav-resize-handle nav-resize-top" data-resize="t" title="Drag top edge to resize"></div>
    <div class="nav-resize-handle nav-resize-bottom" data-resize="b" title="Drag bottom edge to resize"></div>
    <div class="nav-resize-handle nav-resize-tl" data-resize="tl" title="Drag top-left corner to resize"></div>
    <div class="nav-resize-handle nav-resize-tr" data-resize="tr" title="Drag top-right corner to resize"></div>
    <div class="nav-resize-handle nav-resize-bl" data-resize="bl" title="Drag bottom-left corner to resize"></div>
    <div class="nav-resize-handle nav-resize-br" data-resize="br" title="Drag bottom-right corner to resize"></div>
</aside>

<div class="main-shell" id="mainShell"><main class="main">
    <header class="sticky top-0 z-10 border-b border-slate-800/80 bg-[#05080d]/90 backdrop-blur-xl px-5 md:px-8 py-4">
        <div class="max-w-[1500px] mx-auto flex flex-col md:flex-row gap-3 justify-between md:items-center">
            <div>
                <div class="text-[10px] muted uppercase tracking-[.18em]">Quantitative Intelligence Terminal</div>
            </div>
            <div class="flex gap-2">
                <input id="tickerInput" value="AAPL" maxlength="12"
                    class="input mono uppercase px-4 py-2.5 w-32 text-center font-bold"
                    placeholder="TICKER"
                    onkeydown="if(event.key==='Enter') fetchValuation()">
                <button class="btn px-5 py-2.5" onclick="fetchValuation()">Analyze any asset →</button>
            </div>
        </div>
    </header>

    <div class="max-w-[1500px] mx-auto p-5 md:p-8 space-y-6">
        <section class="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-800/60 pb-6">
            <div>
                <div class="flex items-center gap-3">
                    <h1 id="companyName" class="text-2xl md:text-3xl font-extrabold tracking-tight">Loading...</h1>
                    <span id="tickerSymbol" class="badge mono">---</span>
                    <span id="assetClassBadge" class="badge mono">EQUITY · ISSUER</span>
                </div>
                <div id="currencyBadge" class="text-xs muted mt-2 mono">SECTOR BENCHMARKS</div>
            </div>
            <div class="text-left lg:text-right">
                <div class="text-[10px] muted uppercase tracking-widest">Current market price</div>
                <div id="currentPrice" class="text-3xl md:text-4xl font-bold mono mt-1">$--</div>
            </div>
        </section>

        <!-- KPI Grid -->
        <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3">
            <div class="panel kpi">
                <div class="label">DCF Fair Value</div>
                <div id="fairValue" class="value green">--</div>
                <div id="dcfUpside" class="text-xs mt-2">--</div>
            </div>
            <div class="panel kpi">
                <div class="label">XGBoost 20D Price</div>
                <div id="mlPrice" class="value cyan">--</div>
                <div class="text-xs muted mt-2">ML Predictive Forecast</div>
            </div>
            <div class="panel kpi">
                <div class="label">CAPM Expected Return</div>
                <div id="capmReturn" class="value cyan">--</div>
                <div class="text-xs muted mt-2">Annualized Discount Benchmark</div>
            </div>
            <div class="panel kpi">
                <div class="label">Sharpe Ratio (5Y)</div>
                <div id="sharpeRatio" class="value">--</div>
                <div class="text-xs muted mt-2">Excess Return / Std Dev</div>
            </div>
            <div class="panel kpi">
                <div class="label">Altman Z-Score</div>
                <div id="altmanKpi" class="value amber">--</div>
                <div id="altmanKpiNote" class="text-xs muted mt-2">Financial health diagnostic</div>
            </div>
            <div class="panel kpi">
                <div class="label">Quant Consensus Signal</div>
                <div id="signal" class="value green">--</div>
                <div id="signalNote" class="text-xs muted mt-2">Multi-Factor Engine</div>
            </div>
        </section>

        <!-- VIEW 1: DASHBOARD -->
        <div id="view-risk" class="view-section space-y-6">
            <section class="grid grid-cols-1 lg:grid-cols-3 gap-5">
                <div class="panel p-5"><div class="font-semibold">CAPM Expected Return</div><div class="text-[10px] muted mono mt-1">SECURITY MARKET LINE</div><div id="riskCapm" class="metric-big cyan mt-4">--</div><div class="text-xs muted mt-2">E(Ri) = Rf + β(Rm − Rf)</div></div>
                <div class="panel p-5"><div class="font-semibold">5Y Volatility / Variance</div><div class="grid grid-cols-2 gap-3 mt-4"><div><div class="text-[10px] muted">Std. Dev.</div><div id="riskStd" class="metric-big mt-1">--</div></div><div><div class="text-[10px] muted">Variance</div><div id="riskVar" class="metric-big mt-1">--</div></div></div></div>
                <div class="panel p-5"><div class="font-semibold">Risk-Adjusted Performance</div><div class="grid grid-cols-2 gap-3 mt-4"><div><div class="text-[10px] muted">Sharpe</div><div id="riskSharpe" class="metric-big green mt-1">--</div></div><div><div class="text-[10px] muted">Beta</div><div id="riskBeta2" class="metric-big cyan mt-1">--</div></div></div></div>
            </section>
            <section class="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div class="panel p-5"><div class="font-semibold">Information Ratio</div><div class="text-[10px] muted mono mt-1">ACTIVE RETURN / TRACKING ERROR</div><div id="riskInformation" class="metric-big green mt-4">--</div><div class="text-xs muted mt-2">Relative performance consistency vs S&amp;P 500</div></div>
                <div class="panel p-5"><div class="font-semibold">Jensen Alpha</div><div class="text-[10px] muted mono mt-1">REALIZED RETURN − CAPM REQUIRED RETURN</div><div id="riskJensen" class="metric-big cyan mt-4">--</div><div class="text-xs muted mt-2">Annualized alpha relative to CAPM</div></div>
                <div class="panel p-5"><div class="font-semibold">Tracking Error</div><div class="text-[10px] muted mono mt-1">ACTIVE RETURN VOLATILITY</div><div id="riskTrackingError" class="metric-big mt-4">--</div><div class="text-xs muted mt-2">Annualized standard deviation of active returns</div></div>
            </section>
            <section class="grid grid-cols-1 lg:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold">Security Market Line</div><div class="text-[10px] muted mono mt-1 mb-4">EXPECTED RETURN VS SYSTEMATIC RISK</div><div class="h-[300px]"><canvas id="smlChart"></canvas></div></div>
                <div class="panel p-5"><div class="font-semibold">Capital Allocation Line</div><div class="text-[10px] muted mono mt-1 mb-4">EXPECTED RETURN VS TOTAL RISK</div><div class="h-[300px]"><canvas id="calChart"></canvas></div></div>
            </section>
            <section class="panel p-5"><div class="font-semibold">Asset Return vs Market Return</div><div class="text-[10px] muted mono mt-1 mb-4">5Y DAILY RETURN SCATTER · REGRESSION SLOPE = BETA</div><div class="h-[360px]"><canvas id="scatterChart"></canvas></div><div id="scatterStats" class="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4 text-xs"></div></section>
        </div>

        <div id="view-derivatives" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div>
                        <div class="font-semibold text-lg">Futures Contract Simulator</div>
                        <div class="text-[10px] muted mono mt-1">STANDALONE FUTURES · COMMODITIES · ENERGY · METALS · AGRICULTURE · INDEX CONTRACTS</div>
                    </div>
                    <span class="badge">SIMULATION ONLY</span>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mt-5">
                    <div><div class="text-[10px] muted">Contract</div><select id="futureContract" class="trade-input" onchange="applyFuturePreset()">
                        <option value="CUSTOM">Custom</option><option value="CL">Crude Oil (WTI) / CL</option><option value="GC">Gold / GC</option><option value="SI">Silver / SI</option><option value="NG">Natural Gas / NG</option><option value="ZC">Corn / ZC</option><option value="ZW">Wheat / ZW</option><option value="ES">E-mini S&amp;P 500 / ES</option><option value="NQ">E-mini Nasdaq-100 / NQ</option>
                    </select></div>
                    <div><div class="text-[10px] muted">Current futures price</div><input id="futureSpot" class="trade-input mono" type="number" step="0.01" value="75"></div>
                    <div><div class="text-[10px] muted">Contract multiplier</div><input id="futureMultiplier" class="trade-input mono" type="number" step="0.01" value="1000"></div>
                    <div><div class="text-[10px] muted">Position</div><select id="futureDirection" class="trade-input"><option value="LONG">Long futures</option><option value="SHORT">Short futures</option></select></div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mt-3">
                    <div><div class="text-[10px] muted">Quantity (contracts)</div><input id="futureContracts" class="trade-input mono" type="number" min="1" step="1" value="1"></div>
                    <div><div class="text-[10px] muted">Annual financing / carry %</div><input id="futureRate" class="trade-input mono" type="number" step="0.01" value="4.25"></div>
                    <div><div class="text-[10px] muted">Storage / convenience yield %</div><input id="futureCarry" class="trade-input mono" type="number" step="0.01" value="0"></div>
                    <div><div class="text-[10px] muted">Months to expiry</div><input id="futureExpiry" class="trade-input mono" type="number" min="0.1" step="0.1" value="3"></div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mt-3">
                    <div><div class="text-[10px] muted">Scenario price change %</div><input id="futureShock" class="trade-input mono" type="number" step="0.1" value="5"></div>
                    <div class="md:col-span-3 text-xs muted flex items-end">This simulator is independent of the selected company. It models a generic futures contract using the contract's quoted price, multiplier, carry and expiry. It does not create an order or connect to a broker.</div>
                </div>
                <button class="btn px-5 py-2.5 mt-5" onclick="calculateFuturesSimulation()">Run Futures Simulation →</button>
            </section>

            <section class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
                <div class="panel kpi"><div class="label">Theoretical Futures</div><div id="futureTheo" class="value cyan">--</div></div>
                <div class="panel kpi"><div class="label">Notional Value</div><div id="futureNotional" class="value">--</div></div>
                <div class="panel kpi"><div class="label">Basis</div><div id="futureBasis" class="value">--</div></div>
                <div class="panel kpi"><div class="label">Break-even</div><div id="futureBreakeven" class="value">--</div></div>
                <div class="panel kpi"><div class="label">Scenario P/L</div><div id="futureScenarioPnl" class="value green">--</div></div>
            </section>

            <section class="grid grid-cols-1 lg:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold">Futures P/L at Expiry</div><div class="text-[10px] muted mono mt-1 mb-4">PRICE SCENARIO · P/L FOR SELECTED POSITION</div><div class="h-[320px]"><canvas id="futuresPayoffChart"></canvas></div></div>
                <div class="panel p-5"><div class="font-semibold">Theoretical Futures Curve</div><div class="text-[10px] muted mono mt-1 mb-4">COST-OF-CARRY THROUGH TIME</div><div class="h-[320px]"><canvas id="futuresCurveChart"></canvas></div></div>
            </section>

            <section class="panel p-5">
                <div class="font-semibold">Simulation Hypothesis</div>
                <div id="futureHypothesis" class="text-sm muted mt-3 leading-6">--</div>
            </section>
        </div>

        <div id="view-cfd" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                    <div><div class="font-semibold text-lg">CFD Scenario Lab</div><div class="text-[10px] muted mono mt-1">CATALYST RISK + FITTED PRICE + ATR SHOCK MAP</div></div>
                    <span id="cfdRiskRegime" class="badge">WAITING FOR ASSET</span>
                </div>
                <div class="text-xs muted mt-4 leading-6">Educational scenario analysis only. This panel maps how the selected asset could respond to volatility and catalysts; it does not place CFD orders, estimate a probability of profit, or determine a personalized trade.</div>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mt-5">
                    <div class="scenario"><div class="text-[10px] muted">Spot</div><strong id="cfdSpot" class="block mt-2 mono">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">ATR(14)</div><strong id="cfdAtr" class="block mt-2 mono">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">5Y Volatility</div><strong id="cfdVol" class="block mt-2 mono">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">Beta</div><strong id="cfdBeta" class="block mt-2 mono">--</strong></div>
                </div>
            </section>

            <section class="panel p-5">
                <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                    <div>
                        <div class="font-semibold text-lg">Directional Decision Engine</div>
                        <div class="text-[10px] muted mono mt-1">MULTI-FACTOR RESEARCH BIAS · NOT AN EXECUTION SIGNAL</div>
                    </div>
                    <div class="flex items-center gap-3">
                        <span id="decisionConfidence" class="badge">--</span>
                        <strong id="directionalDecision" class="text-2xl mono cyan">--</strong>
                    </div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mt-5">
                    <div class="scenario"><div class="text-[10px] muted">Composite score</div><strong id="decisionScore" class="block mt-2 mono">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">Scenario</div><strong id="decisionAction" class="block mt-2 mono">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">Catalyst pressure</div><strong id="decisionCatalyst" class="block mt-2 mono">--</strong></div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-5 mt-5">
                    <div class="scenario"><div class="text-[10px] muted mb-2">Evidence supporting bullish / long scenario</div><div id="decisionLongCase" class="text-xs leading-6">--</div></div>
                    <div class="scenario"><div class="text-[10px] muted mb-2">Evidence supporting bearish / short scenario</div><div id="decisionShortCase" class="text-xs leading-6">--</div></div>
                </div>
                <div class="text-[10px] muted mt-4">The classifier combines the platform's DCF/composite/ML/analyst valuation, technical trend, RSI, MACD, volume, fundamentals, CAPM, Sharpe and asset-news context. Macro/geopolitical risk mainly reduces confidence when catalyst pressure is high.</div>
            </section>

            <section class="grid grid-cols-1 md:grid-cols-4 gap-5">
                <div class="panel p-5"><div class="text-[10px] muted">Catalyst Pressure</div><div id="cfdCatalyst" class="metric-big cyan mt-3">--</div><div class="text-xs muted mt-2">0–100 scenario score</div></div>
                <div class="panel p-5"><div class="text-[10px] muted">Macro Event Risk</div><div id="cfdEventRisk" class="metric-big mt-3">--</div><div id="cfdEventCount" class="text-xs muted mt-2">--</div></div>
                <div class="panel p-5"><div class="text-[10px] muted">Geopolitical Risk</div><div id="cfdGeoRisk" class="metric-big mt-3">--</div><div id="cfdGeoCount" class="text-xs muted mt-2">--</div></div>
                <div class="panel p-5"><div class="text-[10px] muted">Headline / Trend Bias</div><div id="cfdBias" class="metric-big mt-3">--</div><div class="text-xs muted mt-2">Context only, not a signal</div></div>
            </section>

            <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold">CFD Price-Shock Scenario Graph</div><div class="text-[10px] muted mono mt-1 mb-4">P/L PER 1 UNIT AROUND FITTED LONG / SHORT ENTRY</div><div class="h-[330px]"><canvas id="cfdShockChart"></canvas></div></div>
                <div class="panel p-5"><div class="font-semibold">Catalyst Pressure Breakdown</div><div class="text-[10px] muted mono mt-1 mb-4">MACRO + GEOPOLITICS + HEADLINE CONTEXT</div><div class="h-[330px]"><canvas id="cfdCatalystChart"></canvas></div></div>
            </section>

            <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold mb-3">Fitted CFD Scenario Prices</div><div class="grid grid-cols-2 gap-3"><div class="scenario"><div class="text-[10px] muted">Long fitted price</div><strong id="cfdLongEntry" class="block mt-2 green mono">--</strong></div><div class="scenario"><div class="text-[10px] muted">Short fitted price</div><strong id="cfdShortEntry" class="block mt-2 red mono">--</strong></div></div><div class="text-xs muted mt-4">These reuse the Trade Lab's four-lens fitted prices: technical structure, valuation consensus, RSI momentum and beta/ATR risk buffer.</div></div>
                <div class="panel p-5"><div class="font-semibold mb-3">Upcoming Asset-Relevant Macro Events</div><div id="cfdEvents" class="space-y-2"></div></div>
            </section>

            <section class="panel p-5"><div class="flex justify-between items-center"><div><div class="font-semibold text-lg">Geopolitical News Related to Asset</div><div class="text-[10px] muted mono mt-1">TARIFFS · SANCTIONS · CONFLICT · POLICY · SUPPLY DISRUPTION</div></div><span id="cfdGeoStatus" class="badge">LIVE FEED</span></div><div id="cfdGeoNews" class="mt-5 space-y-2"></div></section>

            <section class="panel p-5"><div class="font-semibold">Scenario Interpretation</div><div id="cfdInterpretation" class="text-sm muted mt-3 leading-6">--</div></section>
        </div>

        <div id="view-trade" class="view-section space-y-6">
            <section class="grid grid-cols-1 lg:grid-cols-3 gap-5">
                <div class="panel p-5"><div class="font-semibold text-lg">Margin Trade Scenario</div><div class="text-[10px] muted mono mt-1">VOLATILITY / RISK-BASED CALCULATOR</div>
                    <div class="grid grid-cols-2 gap-3 mt-5">
                        <label class="text-xs muted">Equity<input id="tradeEquity" class="trade-input mt-1" type="number" value="10000" min="100"></label>
                        <label class="text-xs muted">Used Margin<input id="tradeUsedMargin" class="trade-input mt-1" type="number" value="2000" min="0" step="100"></label>
                        <label class="text-xs muted">Units<input id="tradeUnitsInput" class="trade-input mt-1" type="number" value="100" min="0.0001" step="0.0001"></label>
                        <label class="text-xs muted">Entry Price (optional)<input id="tradeEntryPrice" class="trade-input mt-1" type="number" placeholder="auto" min="0" step="0.01"></label>
                        <label class="text-xs muted col-span-2">Direction<select id="tradeDirection" class="trade-input mt-1" onchange="recalculateTrade()"><option>LONG</option><option>SHORT</option></select></label>
                    </div>
                    <div class="scenario mt-3">
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-3 items-center">
                            <div class="scenario"><div class="text-[10px] muted">Long fitted entry</div><strong id="tradeLongEntry" class="block mt-1 green mono">--</strong></div>
                            <div class="scenario"><div class="text-[10px] muted">Short fitted entry</div><strong id="tradeShortEntry" class="block mt-1 red mono">--</strong></div>
                            <div class="scenario"><div class="text-[10px] muted">Selected direction price</div><strong id="tradeSuitableEntry" class="block mt-1 cyan mono">--</strong></div>
                        </div>
                        <div class="flex justify-end mt-3">
                            <button type="button" class="btn px-3 py-1.5 text-xs" onclick="useSuitableEntry()">Use selected fitted price →</button>
                        </div>
                        <div class="grid grid-cols-2 gap-3 mt-3 text-xs">
                            <div><div class="text-[10px] muted">Technical anchor (35%)</div><strong id="tradeTechAnchor" class="block mt-1 mono">--</strong></div>
                            <div><div class="text-[10px] muted">Valuation anchor (35%)</div><strong id="tradeValAnchor" class="block mt-1 mono">--</strong></div>
                            <div><div class="text-[10px] muted">Momentum anchor (15%)</div><strong id="tradeMomentumAnchor" class="block mt-1 mono">--</strong></div>
                            <div><div class="text-[10px] muted">Risk/beta anchor (15%)</div><strong id="tradeRiskAnchor" class="block mt-1 mono">--</strong></div>
                        </div>
                    </div>
                    <button class="btn px-5 py-2.5 mt-5" onclick="recalculateTrade()">Calculate Scenario →</button>
                    <div class="text-[10px] muted mt-4">Scenario only: spreads, slippage, financing, broker margin rules and liquidation thresholds are not modeled. You choose how much of your equity to commit as margin; leverage shown is implied by that choice, not the other way around. The fitted long and short entry prices each blend SMA/ATR structure, the app's valuation consensus (DCF + peer multiples + XGBoost forecast + analyst targets), RSI(14) momentum, and beta/ATR risk buffering. Long is biased toward a lower/pullback entry; short is biased toward a higher/premium entry. Both are clamped to within 3 ATR of the current price.</div>
                </div>
                <div class="panel p-5 lg:col-span-2"><div class="font-semibold">Model Price Levels</div><div class="grid grid-cols-2 md:grid-cols-4 gap-3 mt-5">
                    <div class="scenario"><div class="text-[10px] muted">Entry</div><strong id="tradeEntry" class="block mt-2 cyan mono">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">Stop Loss</div><strong id="tradeSL" class="block mt-2 red mono">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">Take Profit</div><strong id="tradeTP" class="block mt-2 green mono">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">Risk / Reward</div><strong id="tradeRR" class="block mt-2 mono">--</strong></div>
                </div><div class="grid grid-cols-2 md:grid-cols-4 gap-3 mt-5">
                    <div class="scenario"><div class="text-[10px] muted">Risk $ (at stop)</div><strong id="tradeRiskDollars" class="block mt-2 red mono">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">Notional</div><strong id="tradeNotional" class="block mt-2 mono">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">Free Margin</div><strong id="tradeFreeMargin" class="block mt-2 mono">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">ATR(14)</div><strong id="tradeATR" class="block mt-2 mono">--</strong></div>
                </div><div class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-5">
                    <div class="scenario"><div class="text-[10px] muted">Used Margin</div><strong id="tradeUsedMarginOut" class="block mt-2 mono">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">Implied Leverage</div><strong id="tradeImpliedLeverage" class="block mt-2 mono">--</strong></div>
                </div></div>
            </section>
            <section class="panel p-5"><div class="font-semibold mb-4">Equity Change at Outcomes</div><div class="grid grid-cols-1 md:grid-cols-3 gap-3"><div class="scenario"><div class="text-[10px] muted">Starting equity</div><strong id="eqStart" class="block mt-2 mono">--</strong></div><div class="scenario"><div class="text-[10px] muted">Equity at TP</div><strong id="eqTP" class="block mt-2 green mono">--</strong></div><div class="scenario"><div class="text-[10px] muted">Equity at SL</div><strong id="eqSL" class="block mt-2 red mono">--</strong></div></div></section>
            <section class="panel p-5"><div class="font-semibold">Setup interpretation</div><div id="tradeNote" class="text-sm muted mt-3 leading-6">--</div></section>
        </div>

        <div id="view-news" class="view-section space-y-6">
            <section class="panel p-5"><div class="flex justify-between items-center"><div><div class="font-semibold text-lg">Economic Calendar</div><div class="text-[10px] muted mono mt-1">HIGH-IMPACT US MACRO EVENTS</div></div><span class="badge">SCHEDULE</span></div><div id="macroCalendar" class="mt-5 space-y-2"></div></section>
            <section class="panel p-5"><div class="font-semibold text-lg">Asset News & Possible Impact</div><div class="text-[10px] muted mono mt-1">RECENT TICKER HEADLINES + SCENARIO INTERPRETATION</div><div id="assetNews" class="mt-5 space-y-2"></div></section>
        </div>

        <div id="view-dashboard" class="view-section active-view space-y-6">
            <section class="panel p-4 md:p-5">
                <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
                    <div><div class="font-semibold">Multi-Asset Market Board</div><div class="text-[10px] muted mono mt-1">EQUITIES · FX · METALS · ENERGY · COMMODITIES · RATES · CRYPTO</div></div>
                    <div class="flex flex-wrap gap-2"><button class="tab active" onclick="loadInstrumentBoard('Equities',this)">Equities</button><button class="tab" onclick="loadInstrumentBoard('Currencies',this)">FX</button><button class="tab" onclick="loadInstrumentBoard('Metals',this)">Metals</button><button class="tab" onclick="loadInstrumentBoard('Energy',this)">Oil / Energy</button><button class="tab" onclick="loadInstrumentBoard('Commodities',this)">Commodities</button><button class="tab" onclick="loadInstrumentBoard('Rates / Bonds',this)">Rates</button><button class="tab" onclick="loadInstrumentBoard('Crypto',this)">Crypto</button></div>
                </div>
                <div id="instrumentBoard" class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3 mt-5"></div>
            </section>
            <section class="panel p-4 md:p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-4">
                    <div>
                        <div class="font-semibold">Price Action & Model Forecast</div>
                        <div class="text-[10px] muted mono mt-1">HISTORICAL CLOSE · SMA20 · SMA50 · XGBoost PROJECTION</div>
                    </div>
                    <div class="flex gap-1 text-xs bg-slate-950 p-1 rounded-lg border border-slate-800">
                        <div class="tab" onclick="setTimeframe('1M', this)">1M</div>
                        <div class="tab" onclick="setTimeframe('3M', this)">3M</div>
                        <div class="tab" onclick="setTimeframe('6M', this)">6M</div>
                        <div class="tab" onclick="setTimeframe('1Y', this)">1Y</div>
                        <div class="tab active" onclick="setTimeframe('5Y', this)">5Y</div>
                    </div>
                </div>
                <div class="h-[360px]">
                    <canvas id="lineChart"></canvas>
                </div>
            </section>

            <section class="grid grid-cols-1 xl:grid-cols-3 gap-5">
                <div class="panel p-5 xl:col-span-2">
                    <div class="font-semibold mb-2">Valuation Spread Comparison</div>
                    <div class="text-[10px] muted mono mb-4">MARKET VS DCF VS XGBOOST VS COMPOSITE TARGET</div>
                    <div class="h-[250px]">
                        <canvas id="barChart"></canvas>
                    </div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold">Fundamental Drivers</div>
                    <div class="text-[10px] muted mono mt-1 mb-5">KEY QUANT METRICS</div>
                    <div class="space-y-4">
                        <div class="flex justify-between text-xs mb-1"><span class="muted">Forward P/E</span><strong id="peRatio" class="mono">--</strong></div>
                        <div class="flex justify-between text-xs mb-1"><span class="muted">Profit Margin</span><strong id="margin" class="mono">--</strong></div>
                        <div class="flex justify-between text-xs mb-1"><span class="muted">Return on Equity</span><strong id="roe" class="mono">--</strong></div>
                        <div class="flex justify-between text-xs mb-1"><span class="muted">Beta</span><strong id="beta" class="mono">--</strong></div>
                    </div>
                </div>
            </section>
        </div>

        <!-- VIEW: PORTFOLIO OPTIMIZER -->
        <div id="view-portfolio" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="flex flex-col lg:flex-row lg:items-end gap-4 justify-between">
                    <div>
                        <div class="font-semibold text-lg">Portfolio Optimizer</div>
                        <div class="text-[10px] muted mono mt-1">UNLIMITED USER-SELECTED ASSETS · LONG-ONLY MEAN-VARIANCE SIMULATION</div>
                    </div>
                    <div class="flex items-center gap-2">
                        <span id="portfolioStatus" class="badge">READY</span>
                    </div>
                </div>
                <div class="grid grid-cols-1 lg:grid-cols-4 gap-3 mt-5">
                    <label class="text-xs muted lg:col-span-2">Chosen assets (comma / semicolon separated)
                        <input id="portfolioTickers" class="trade-input mt-1" type="text" value="AAPL,MSFT,NVDA,GOOGL,AMZN" placeholder="AAPL,MSFT,NVDA,...">
                    </label>
                    <label class="text-xs muted">Method
                        <select id="portfolioMethod" class="trade-input mt-1">
                            <option value="MIN_RISK">Minimum Risk</option>
                            <option value="TANGENCY">Tangency / Max Sharpe</option>
                            <option value="COMPARE">Compare Both</option>
                        </select>
                    </label>
                    <label class="text-xs muted">History
                        <select id="portfolioLookback" class="trade-input mt-1">
                            <option value="5y" selected>5 Years</option>
                            <option value="3y">3 Years</option>
                            <option value="10y">10 Years</option>
                        </select>
                    </label>
                </div>
                <div class="flex flex-wrap gap-2 mt-4">
                    <button type="button" class="btn px-5 py-2.5" onclick="fetchPortfolioOptimizer()">Calculate Portfolio →</button>
                    <button type="button" class="btn px-4 py-2.5" onclick="useCurrentTickerForPortfolio()">Use current company</button>
                </div>
                <div class="text-[10px] muted mt-4 leading-5">
                    Minimum Risk minimizes estimated variance. Tangency maximizes estimated Sharpe ratio. The optimizer uses all valid assets you enter, with weights constrained to 0–100% and total weight = 100%. This is a quantitative scenario model, not an order or personalized investment instruction.
                </div>
            </section>

            <section class="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div class="panel p-4"><div class="text-[10px] muted">Expected Return</div><div id="portReturn" class="metric-big cyan mt-2">--</div></div>
                <div class="panel p-4"><div class="text-[10px] muted">Volatility</div><div id="portVol" class="metric-big mt-2">--</div></div>
                <div class="panel p-4"><div class="text-[10px] muted">Variance</div><div id="portVar" class="metric-big mt-2">--</div></div>
                <div class="panel p-4"><div class="text-[10px] muted">Sharpe</div><div id="portSharpe" class="metric-big green mt-2">--</div></div>
                <div class="panel p-4"><div class="text-[10px] muted">Portfolio Beta</div><div id="portBeta" class="metric-big mt-2">--</div></div>
            </section>

            <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold">Asset Risk / Return Scatter</div><div class="text-[10px] muted mono mt-1 mb-4">EXPECTED RETURN VS ANNUALIZED VOLATILITY · BUBBLE = WEIGHT</div><div class="h-[340px]"><canvas id="portfolioScatterChart"></canvas></div></div>
                <div class="panel p-5"><div class="font-semibold">Security Market Line</div><div class="text-[10px] muted mono mt-1 mb-4">CAPM CONTEXT FOR SELECTED ASSETS</div><div class="h-[340px]"><canvas id="portfolioSmlChart"></canvas></div></div>
                <div class="panel p-5"><div class="font-semibold">Capital Allocation Line</div><div class="text-[10px] muted mono mt-1 mb-4">RISK-FREE RATE → SELECTED PORTFOLIO</div><div class="h-[340px]"><canvas id="portfolioCalChart"></canvas></div></div>
                <div class="panel p-5"><div class="font-semibold">Efficient Frontier</div><div class="text-[10px] muted mono mt-1 mb-4">LONG-ONLY SIMULATED EFFICIENT ENVELOPE</div><div class="h-[340px]"><canvas id="portfolioFrontierChart"></canvas></div></div>
            </section>

            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div>
                        <div class="font-semibold text-lg">Estimated Portfolio Weights</div>
                        <div id="portfolioMethodLabel" class="text-[10px] muted mono mt-1">--</div>
                    </div>
                    <div class="text-xs mono"><span id="portfolioAssetCount">0</span> assets · <span id="portfolioWeightSum">0.00%</span> total weight</div>
                </div>
                <div class="table-wrap mt-4">
                    <table id="portfolioTable"><thead><tr><th>Asset</th><th>Weight</th><th>Min Risk</th><th>Tangency</th><th>Exp Return</th><th>Volatility</th><th>Beta</th><th>Asset Sharpe</th><th>Return Contribution</th><th>Risk Contribution</th></tr></thead><tbody><tr><td colspan="10" class="muted">Run the optimizer to load assets.</td></tr></tbody></table>
                </div>
            </section>

            <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold">Method Comparison</div><div id="portfolioMethodComparison" class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4"></div></div>
                <div class="panel p-5"><div class="font-semibold">Model / Data Notes</div><div id="portfolioNotes" class="text-xs muted mt-4 leading-6">--</div></div>
            </section>
        </div>

        <!-- VIEW 2: RISK & CAPM PORTFOLIO LAB -->
        <div id="view-risk" class="view-section space-y-6">
            <section class="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div class="panel p-5">
                    <div class="font-semibold text-base mb-1">Capital Asset Pricing Model (CAPM)</div>
                    <div class="text-[10px] muted mono mb-4">E(R) = R_f + β * (R_m - R_f)</div>
                    <div class="space-y-3">
                        <div class="flex justify-between text-xs"><span class="muted">Risk-Free Rate (R_f)</span><span id="rfRate" class="mono font-bold">--</span></div>
                        <div class="flex justify-between text-xs"><span class="muted">Market Expected Return (R_m)</span><span id="rmReturn" class="mono font-bold">--</span></div>
                        <div class="flex justify-between text-xs"><span class="muted">Asset Beta (β)</span><span id="capmBeta" class="mono font-bold">--</span></div>
                        <div class="border-t border-slate-800 pt-3 flex justify-between text-sm"><span class="font-bold">CAPM Expected Return</span><span id="capmResult" class="mono font-bold cyan">--</span></div>
                    </div>
                </div>

                <div class="panel p-5">
                    <div class="font-semibold text-base mb-1">5-Year Volatility & Variance</div>
                    <div class="text-[10px] muted mono mb-4">LOG-NORMAL ANNUALIZED METRICS</div>
                    <div class="space-y-3">
                        <div class="flex justify-between text-xs"><span class="muted">Annualized Variance (σ²)</span><span id="var5y" class="mono font-bold">--</span></div>
                        <div class="flex justify-between text-xs"><span class="muted">Annualized Std Dev (σ)</span><span id="sd5y" class="mono font-bold">--</span></div>
                        <div class="flex justify-between text-xs"><span class="muted">52-Week High Dist.</span><span id="dist52w" class="mono font-bold">--</span></div>
                    </div>
                </div>

                <div class="panel p-5">
                    <div class="font-semibold text-base mb-1">Risk-Adjusted Ratios</div>
                    <div class="text-[10px] muted mono mb-4">PORTFOLIO PERFORMANCE MEASURES</div>
                    <div class="space-y-3">
                        <div class="flex justify-between text-xs"><span class="muted">Sharpe Ratio</span><span id="sharpeVal" class="mono font-bold green">--</span></div>
                        <div class="flex justify-between text-xs"><span class="muted">Modigliani–Modigliani (M²)</span><span id="mSquaredVal" class="mono font-bold cyan">--</span></div>
                        <div class="flex justify-between text-xs"><span class="muted">Treynor Ratio</span><span id="treynorVal" class="mono font-bold">--</span></div>
                    </div>
                </div>
            </section>
        </div>

        <!-- VIEW 3: FINANCIAL RATIOS LAB -->
        <div id="view-financials" class="view-section space-y-6">

            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-4">
                    <div><div class="font-semibold text-lg">Annual Financial Statements & Analysis</div><div class="text-[10px] muted mono mt-1">LOADED FOR SELECTED COMPANY · INCOME · BALANCE SHEET · CASH FLOW</div></div>
                    <span id="statementDataStatus" class="badge">WAITING</span>
                </div>
                <div id="statementAnalysisSummary" class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-5"></div>
                <div class="flex gap-2 mb-4">
                    <button id="stmtTabIncome" class="btn px-4 py-2 text-xs" onclick="showStatementTab('income')">Income Statement</button>
                    <button id="stmtTabBalance" class="btn px-4 py-2 text-xs" onclick="showStatementTab('balance')">Balance Sheet</button>
                    <button id="stmtTabCash" class="btn px-4 py-2 text-xs" onclick="showStatementTab('cash')">Cash Flow</button>
                </div>
                <div class="table-wrap"><table id="financialStatementTable"><thead></thead><tbody></tbody></table></div>
                <div id="statementInterpretation" class="scenario mt-4 text-xs leading-6 muted">--</div>
            </section>

            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-4">
                    <div>
                        <div class="font-semibold text-lg">5-Year Annual Financial Ratio History</div>
                        <div class="text-[10px] muted mono mt-1">CALCULATED FROM ANNUAL INCOME STATEMENT + BALANCE SHEET DATA</div>
                    </div>
                    <span id="ratioDataStatus" class="badge">LOADING</span>
                </div>
                <div class="table-wrap">
                    <table id="ratioHistoryTable">
                        <thead>
                            <tr>
                                <th>Ratio</th><th>FY-1</th><th>FY-2</th><th>FY-3</th><th>FY-4</th><th>FY-5</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div class="text-[10px] muted mt-3">N/A means the required annual statement line was not supplied by the data provider; missing data is never displayed as zero.</div>
            </section>
            <section class="grid grid-cols-1 lg:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold">Profitability Trend</div><div class="text-[10px] muted mono mt-1 mb-4">ROIC · ROA · ROE</div><div class="h-[300px]"><canvas id="ratioProfitChart"></canvas></div></div>
                <div class="panel p-5"><div class="font-semibold">Margin Trend</div><div class="text-[10px] muted mono mt-1 mb-4">GROSS · OPERATING · NET MARGIN</div><div class="h-[300px]"><canvas id="ratioMarginChart"></canvas></div></div>
            </section>
            <section class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">ROIC (Return on Invested Capital)</div>
                    <div class="text-2xl font-bold mono green mt-2" id="roicVal">--</div>
                    <div class="text-xs muted mt-1">NOPAT / Total Invested Capital</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Current Ratio</div>
                    <div class="text-2xl font-bold mono mt-2" id="currentRatioVal">--</div>
                    <div class="text-xs muted mt-1">Current Assets / Current Liabilities</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Quick Ratio</div>
                    <div class="text-2xl font-bold mono mt-2" id="quickRatioVal">--</div>
                    <div class="text-xs muted mt-1">(Cash + Rec) / Current Liabilities</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Debt-to-Equity</div>
                    <div class="text-2xl font-bold mono mt-2" id="debtEquityVal">--</div>
                    <div class="text-xs muted mt-1">Total Debt / Total Stockholders Equity</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Interest Coverage</div>
                    <div class="text-2xl font-bold mono mt-2" id="interestCoverageVal">--</div>
                    <div class="text-xs muted mt-1">EBIT / Interest Expense</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Altman Z-Score</div>
                    <div class="text-2xl font-bold mono amber mt-2" id="altmanZVal">--</div>
                    <div class="text-xs muted mt-1">Financial distress diagnostic · classic public-company model</div>
                    <div class="text-[10px] muted mt-2" id="altmanZoneVal">--</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Free Cash Flow Yield</div>
                    <div class="text-2xl font-bold mono cyan mt-2" id="fcfYieldVal">--</div>
                    <div class="text-xs muted mt-1">Latest FCF / Market Capitalization</div>
                </div>
            </section>
            <section class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Return on Assets (ROA)</div>
                    <div class="text-2xl font-bold mono green mt-2" id="roaVal">--</div>
                    <div class="text-xs muted mt-1">Net Income / Total Assets</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Return on Equity (ROE)</div>
                    <div class="text-2xl font-bold mono green mt-2" id="roeVal">--</div>
                    <div class="text-xs muted mt-1">Net Income / Shareholders Equity</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Gross Margin</div>
                    <div class="text-2xl font-bold mono mt-2" id="grossMarginVal">--</div>
                    <div class="text-xs muted mt-1">Gross Profit / Revenue</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Operating Margin</div>
                    <div class="text-2xl font-bold mono mt-2" id="operatingMarginVal">--</div>
                    <div class="text-xs muted mt-1">Operating Income / Revenue</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Net Margin</div>
                    <div class="text-2xl font-bold mono mt-2" id="netMarginVal">--</div>
                    <div class="text-xs muted mt-1">Net Income / Revenue</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Cash Ratio</div>
                    <div class="text-2xl font-bold mono mt-2" id="cashRatioVal">--</div>
                    <div class="text-xs muted mt-1">Cash & Equivalents / Current Liabilities</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Asset Turnover</div>
                    <div class="text-2xl font-bold mono cyan mt-2" id="assetTurnoverVal">--</div>
                    <div class="text-xs muted mt-1">Revenue / Total Assets</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Inventory Turnover</div>
                    <div class="text-2xl font-bold mono cyan mt-2" id="inventoryTurnoverVal">--</div>
                    <div class="text-xs muted mt-1">Cost of Revenue / Inventory</div>
                </div>
                <div class="panel p-5">
                    <div class="font-semibold text-sm mb-1">Receivables Turnover</div>
                    <div class="text-2xl font-bold mono cyan mt-2" id="receivablesTurnoverVal">--</div>
                    <div class="text-xs muted mt-1">Revenue / Accounts Receivable</div>
                </div>
            </section>
        </div>

        <!-- VIEW 4: MARKETS SCANNER -->

        <div id="view-fundamentals" class="view-section space-y-6">
    <section class="panel p-5">
        <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
            <div>
                <div class="font-semibold text-lg">Integrated Fundamental Analysis</div>
                <div class="text-[10px] muted mono mt-1">STATEMENTS · 5Y RATIOS · VALUATION · PEERS · CASH FLOW · RISK CONTEXT</div>
            </div>
            <span id="fundamentalBand" class="badge">WAITING FOR ASSET</span>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3 mt-5" id="fundamentalKpis"></div>
    </section>

    <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <div class="panel p-5">
            <div class="font-semibold">Fundamental Factor Scorecard</div>
            <div class="text-[10px] muted mono mt-1 mb-4">TRANSPARENT WEIGHTS · 0–100 DIAGNOSTIC</div>
            <div class="h-[330px]"><canvas id="fundamentalFactorChart"></canvas></div>
        </div>
        <div class="panel p-5">
            <div class="font-semibold">Valuation Cross-Check</div>
            <div class="text-[10px] muted mono mt-1 mb-4">CURRENT PRICE VS DCF · PEER · ML · ANALYST · COMPOSITE</div>
            <div class="h-[330px]"><canvas id="fundamentalValuationChart"></canvas></div>
        </div>
    </section>

    <section class="panel p-5">
        <div class="flex items-center justify-between mb-4">
            <div><div class="font-semibold">Fundamental Evidence Table</div><div class="text-[10px] muted mono mt-1">LATEST AVAILABLE DATA</div></div>
            <span id="fundamentalCoverage" class="badge">--</span>
        </div>
        <div class="table-wrap"><table id="fundamentalEvidenceTable"><thead><tr><th>Category</th><th>Metric</th><th>Value</th><th>Research interpretation</th></tr></thead><tbody></tbody></table></div>
    </section>

    <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <div class="panel p-5">
            <div class="font-semibold">5-Year Fundamental Trend</div>
            <div class="text-[10px] muted mono mt-1 mb-4">YOY REVENUE · NET INCOME · FCF · DEBT</div>
            <div class="h-[300px]"><canvas id="fundamentalTrendChart"></canvas></div>
        </div>
        <div class="panel p-5">
            <div class="font-semibold">Fundamental Research Notes</div>
            <div class="text-[10px] muted mono mt-1 mb-4">STRENGTHS · RISKS · WATCH ITEMS</div>
            <div id="fundamentalNotes" class="space-y-3 text-xs leading-6"></div>
        </div>
    </section>

    <section class="panel p-5">
        <div class="font-semibold">How the Analysis Connects to the Platform</div>
        <div class="text-xs muted leading-6 mt-3">
            The fundamental layer connects the selected asset’s annual income statement, balance sheet and cash flow, 5-year ratio history, DCF valuation, peer valuation, analyst target, machine-learning forecast, risk metrics and current news/event context. Market-risk, macro and geopolitical information is displayed as context rather than being silently mixed into the fundamental score.
        </div>
    </section>
</div>


<div id="view-model" class="view-section space-y-6">
    <section class="panel p-5">
        <div class="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
            <div><div class="font-semibold text-lg">Financial Model Studio</div><div class="text-[10px] muted mono mt-1">STATISTICAL FINANCIAL MODEL · FORECAST · WACC · DCF · MULTIPLES · SCENARIOS · SENSITIVITY</div></div>
            <button class="btn px-4 py-2" onclick="downloadFinancialModel()">Download Model Spreadsheet</button>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3 mt-5" id="modelKpis"></div>
    </section>
    <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <div class="panel p-5"><div class="font-semibold">5-Year Forecast</div><div class="text-[10px] muted mono mt-1 mb-4">REVENUE · EBIT · FCFF</div><div class="h-[320px]"><canvas id="modelForecastChart"></canvas></div></div>
        <div class="panel p-5"><div class="font-semibold">Scenario Valuation</div><div class="text-[10px] muted mono mt-1 mb-4">BEAR · BASE · BULL</div><div class="h-[320px]"><canvas id="modelScenarioChart"></canvas></div></div>
    </section>
    <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <div class="panel p-5"><div class="font-semibold">DCF Sensitivity</div><div class="text-[10px] muted mono mt-1 mb-4">WACC × TERMINAL GROWTH · IMPLIED VALUE / SHARE</div><div class="table-wrap"><table id="modelSensitivityTable"><tbody></tbody></table></div></div>
        <div class="panel p-5"><div class="font-semibold">WACC Build</div><div class="text-[10px] muted mono mt-1 mb-4">CAPM + AFTER-TAX DEBT</div><div class="table-wrap"><table id="modelWaccTable"><tbody></tbody></table></div></div>
    </section>
    <section class="panel p-5">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-3"><div><div class="font-semibold">Model Spreadsheet</div><div class="text-[10px] muted mono mt-1">HISTORICAL + FORECAST + DCF + MULTIPLES + DUPONT</div></div><span id="modelStatus" class="badge">WAITING FOR ASSET</span></div>
        <div class="table-wrap mt-4"><table id="modelSpreadsheet"><thead></thead><tbody></tbody></table></div>
    </section>
</div>

<div id="view-exports" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div>
                        <div class="font-semibold text-lg">Data Export Center</div>
                        <div class="text-[10px] muted mono mt-1">DOWNLOAD HISTORICAL PRICES · EXCESS RETURNS · RATIOS · STATEMENTS</div>
                    </div>
                    <span id="exportTickerBadge" class="badge">SELECT A TICKER</span>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 mt-6">
                    <button class="btn p-4 text-left" onclick="downloadExport('all')"><div class="font-semibold">Complete Excel</div><div class="text-[10px] muted mt-1">Prices + excess returns + ratios + statements + valuation</div></button>
                    <button class="btn p-4 text-left" onclick="downloadExport('market')"><div class="font-semibold">Prices & Excess Returns</div><div class="text-[10px] muted mt-1">5Y daily prices, asset/market returns and excess-return series</div></button>
                    <button class="btn p-4 text-left" onclick="downloadExport('fundamentals')"><div class="font-semibold">Ratios & Statements</div><div class="text-[10px] muted mt-1">5Y ratios plus annual income, balance sheet and cash flow</div></button>
                    <button class="btn p-4 text-left" onclick="downloadFinancialModel()"><div class="font-semibold">Financial Model</div><div class="text-[10px] muted mt-1">Forecast + WACC + DCF + scenarios + sensitivity + dashboard</div></button>
                    <button class="btn p-4 text-left" onclick="downloadPowerBI()"><div class="font-semibold">Power BI Project</div><div class="text-[10px] muted mt-1">PBIP starter + populated Excel data package</div></button>
                </div>
                <div class="scenario mt-5 text-xs leading-6 muted">
                    The export uses the same ticker data as the dashboard. Excess return is provided both versus the app's risk-free assumption and versus the S&P 500 benchmark. Missing annual statement data is exported as blank/N/A rather than zero.
                </div>
            </section>
            <section class="grid grid-cols-1 lg:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold">Excel contents</div><div class="text-[10px] muted mono mt-1 mb-4">STRUCTURED FOR EXCEL / POWER BI IMPORT</div><div class="space-y-2 text-xs muted"><div>Historical Prices — 5Y daily OHLCV + return series</div><div>Excess Returns — risk-free and S&P 500 relative returns</div><div>Financial Ratios — annual 5Y ratio history</div><div>Financial Statements — income, balance sheet, cash flow</div><div>Valuation Summary — core dashboard metrics</div><div>Sources — source links and methodology</div></div></div>
                <div class="panel p-5"><div class="font-semibold">Power BI workflow</div><div class="text-[10px] muted mono mt-1 mb-4">PBIP PROJECT FORMAT</div><div class="space-y-2 text-xs muted"><div>1. Open the downloaded <span class="mono">Finmodel_Analytics.pbip</span>.</div><div>2. The semantic model points to the included <span class="mono">Finmodel_Data.xlsx</span>.</div><div>3. Build or extend visuals from the loaded tables.</div><div>4. Re-download the package for another ticker when needed.</div></div></div>
            </section>
        </div>

        <div id="view-markets" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="font-semibold text-lg mb-2">Market Scanner & Technical Oscillators</div>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
                    <div class="panel p-4"><div class="text-[10px] muted">RSI 14</div><strong id="rsiVal" class="block text-xl font-mono mt-1">--</strong><div id="rsiState" class="text-xs muted mt-1">--</div></div>
                    <div class="panel p-4"><div class="text-[10px] muted">MACD Oscillator</div><strong id="macdVal" class="block text-xl font-mono mt-1">--</strong><div id="macdState" class="text-xs muted mt-1">--</div></div>
                    <div class="panel p-4"><div class="text-[10px] muted">Volume Ratio</div><strong id="volumeRatio" class="block text-xl font-mono mt-1">--</strong><div class="text-xs muted mt-1">vs 20D Average</div></div>
                </div>
                <div class="h-[250px]">
                    <canvas id="technicalChart"></canvas>
                </div>
            </section>
        </div>

        <!-- VIEW 5: PEER COMPARE (WITH CHARTS & SAME INDUSTRY SELECTION) -->
        <div id="view-opportunities" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
                    <div><div class="font-semibold text-lg">Market Opportunity Scanner</div><div class="text-[10px] muted mono mt-1">DAY + WEEK TREND · MULTI-ASSET HYPOTHESES · RESEARCH BIAS</div></div>
                    <div class="flex flex-col sm:flex-row gap-2"><input id="opportunityTickers" class="trade-input sm:w-[420px]" placeholder="Optional: NVDA,AAPL,GLD,SPY" /><button class="btn px-5 py-2.5" onclick="fetchOpportunities()">Scan market →</button></div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mt-5">
                    <div class="scenario"><div class="text-[10px] muted">Assets scanned</div><strong id="oppUniverse" class="block mt-2 metric-big">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">Long candidates</div><strong id="oppLongCount" class="block mt-2 metric-big green">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">Short candidates</div><strong id="oppShortCount" class="block mt-2 metric-big red">--</strong></div>
                    <div class="scenario"><div class="text-[10px] muted">Watch / mixed</div><strong id="oppWatchCount" class="block mt-2 metric-big cyan">--</strong></div>
                </div>
                <div class="text-xs muted mt-4">Daily and weekly trend signals are combined to form research candidates. They do not guarantee outcomes.</div>
            </section>
            <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold">Day vs Week Trend Map</div><div class="text-[10px] muted mono mt-1 mb-4">X = DAY SCORE · Y = WEEK SCORE</div><div class="h-[360px]"><canvas id="opportunityScatterChart"></canvas></div></div>
                <div class="panel p-5"><div class="font-semibold">Strongest Directional Candidates</div><div class="text-[10px] muted mono mt-1 mb-4">COMPOSITE SCORE</div><div class="h-[360px]"><canvas id="opportunityBarChart"></canvas></div></div>
            </section>
            <section class="grid grid-cols-1 xl:grid-cols-3 gap-5">
                <div class="panel p-5 xl:col-span-2"><div class="font-semibold">Opportunity Board</div><div class="table-wrap mt-4"><table id="opportunityTable"><thead><tr><th>Asset</th><th>Day</th><th>Week</th><th>Composite</th><th>1D</th><th>5D</th><th>20D</th><th>RSI</th><th>Risk</th><th>Bias</th></tr></thead><tbody></tbody></table></div></div>
                <div class="panel p-5"><div class="font-semibold">Hypothesis Engine</div><div id="opportunityHypotheses" class="space-y-2 mt-4"></div><div class="text-[10px] muted mt-4">Aligned trends support continuation hypotheses; mixed trends support confirmation/range hypotheses; elevated volatility increases uncertainty.</div></div>
            </section>
        </div>

        <div id="view-peers" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="flex justify-between items-center mb-4">
                    <div>
                        <div class="font-semibold text-lg">Same-Industry Peer Valuation Benchmarks</div>
                        <div id="peerSubtitle" class="text-[10px] muted mono mt-1 font-mono">SECTOR FILTERED COMPARISON</div>
                    </div>
                    <span id="peerMethod" class="badge">--</span>
                </div>
                <div class="table-wrap mb-6">
                    <table id="peerTable">
                        <thead><tr><th>Ticker</th><th>Price</th><th>Forward P/E</th><th>Revenue Growth</th><th>Margin</th><th>Beta</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
                    <div class="panel p-4">
                        <div class="text-xs font-semibold mb-2">Forward P/E Multiple Comparison</div>
                        <div class="h-[220px]"><canvas id="peerPeChart"></canvas></div>
                    </div>
                    <div class="panel p-4">
                        <div class="text-xs font-semibold mb-2">Profit Margin Comparison (%)</div>
                        <div class="h-[220px]"><canvas id="peerMarginChart"></canvas></div>
                    </div>
                </div>
            </section>
        </div>

        <!-- VIEW 6: VALUATION LAB -->
        <div id="view-valuation" class="view-section space-y-6">
            <section class="grid grid-cols-1 lg:grid-cols-2 gap-5">
                <div class="panel p-5">
                    <div class="font-semibold mb-2">DCF Interactive Controls</div>
                    <div class="text-[10px] muted mono mb-6">ADJUST DISCOUNT RATE AND PERPETUAL GROWTH</div>
                    <div class="space-y-6">
                        <div>
                            <div class="flex justify-between text-xs mb-2">
                                <span>Discount Rate (WACC)</span>
                                <strong id="waccVal" class="green mono">9.0%</strong>
                            </div>
                            <input id="waccSlider" type="range" min="0.05" max="0.20" step="0.005" value="0.09" class="w-full" oninput="updateSliders()">
                        </div>
                        <div>
                            <div class="flex justify-between text-xs mb-2">
                                <span>Perpetual Growth Rate</span>
                                <strong id="growthVal" class="green mono">2.50%</strong>
                            </div>
                            <input id="growthSlider" type="range" min="0.01" max="0.05" step="0.0025" value="0.025" class="w-full" oninput="updateSliders()">
                        </div>
                    </div>
                </div>

                <div class="panel p-5">
                    <div class="font-semibold mb-2">Sensitivity Matrix</div>
                    <div class="text-[10px] muted mono mb-4">DCF INTRINSIC VALUE SCENARIOS</div>
                    <div class="grid grid-cols-3 gap-3">
                        <div class="panel p-4 text-center">
                            <div class="text-[10px] muted uppercase">Bear Case</div>
                            <div id="bearVal" class="text-lg font-bold red mt-1 font-mono">--</div>
                            <div class="text-[10px] muted mt-1">WACC +1.5%</div>
                        </div>
                        <div class="panel p-4 text-center">
                            <div class="text-[10px] muted uppercase">Base Case</div>
                            <div id="baseVal" class="text-lg font-bold green mt-1 font-mono">--</div>
                            <div class="text-[10px] muted mt-1">Current Inputs</div>
                        </div>
                        <div class="panel p-4 text-center">
                            <div class="text-[10px] muted uppercase">Bull Case</div>
                            <div id="bullVal" class="text-lg font-bold cyan mt-1 font-mono">--</div>
                            <div class="text-[10px] muted mt-1">WACC -1.0%</div>
                        </div>
                    </div>
                </div>
            </section>
        </div>

        <!-- VIEW 7: TECHNICAL FORECAST LAB -->
        <div id="view-forecast" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="font-semibold text-lg mb-2">Research Composite Valuation Target</div>
                <div class="text-xs muted mb-4">Multi-factor valuation integrating DCF, peer multiples, machine learning, and consensus models.</div>
                <div class="text-4xl font-black mono cyan mb-2" id="compositeTarget">--</div>
                <div id="compositeUpside" class="text-sm green mb-6">--</div>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
                    <div class="panel p-3"><span class="muted block">Peer Target</span><span id="peerTarget" class="text-sm font-bold">--</span></div>
                    <div class="panel p-3"><span class="muted block">Analyst Target</span><span id="analystTarget" class="text-sm font-bold">--</span></div>
                    <div class="panel p-3"><span class="muted block">XGBoost Target</span><span id="mlForecastTarget" class="text-sm font-bold">--</span></div>
                    <div class="panel p-3"><span class="muted block">DCF Target</span><span id="dcfTargetCopy" class="text-sm font-bold">--</span></div>
                </div>
            </section>
        </div>

    </div>
</main></div>

<script>
// When this page is hosted on a static host, the meta tag points at the API.
const API_BASE = ((document.querySelector('meta[name="finmodel-api"]') || {}).content || '').replace(/\/$/, '');
let boardCharts = [];
let lineChartInstance = null;
let barChartInstance = null;
let techChartInstance = null;
let peerPeChartInstance = null;
let peerMarginChartInstance = null;
let currentData = null;
let selectedTimeframe = '5Y';
let smlChartInstance = null;
let calChartInstance = null;
let scatterChartInstance = null;
let derivPayoffChartInstance = null;
let derivFuturesChartInstance = null;
let derivativeMode = 'FUTURES';
let cfdShockChartInstance = null;
let cfdCatalystChartInstance = null;
let opportunityScatterChartInstance = null;
let opportunityBarChartInstance = null;
let portfolioScatterChartInstance = null;
let portfolioSmlChartInstance = null;
let portfolioCalChartInstance = null;
let portfolioFrontierChartInstance = null;
let portfolioData = null;


let modelChartInstance = null;
let modelScenarioChartInstance = null;
let modelData = null;

function formatBoardPrice(v){ return v==null ? 'N/A' : Number(v).toLocaleString(undefined,{maximumFractionDigits:4}); }
function pctBoard(v){ return v==null ? 'N/A' : ((Number(v)*100>=0?'+':'')+(Number(v)*100).toFixed(2)+'%'); }
async function analyzeBoardTicker(symbol){
    const input=document.getElementById('tickerInput');
    if(input){ input.value=symbol; input.focus(); }
    await fetchValuation();
    switchView('dashboard', document.querySelector('.navlink[data-view="dashboard"]'));
    window.scrollTo({top:0,behavior:'smooth'});
}

async function loadInstrumentBoard(category='Equities', btn=null){
    if(btn){btn.parentElement.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));btn.classList.add('active');}
    boardCharts.forEach(c=>{try{c.destroy();}catch(e){}}); boardCharts=[];
    const box=document.getElementById('instrumentBoard'); if(!box) return;
    box.innerHTML='<div class="muted text-sm">Loading '+category+'…</div>';
    try{ const res=await fetch(API_BASE+'/api/instrument-board?category='+encodeURIComponent(category)); const data=await res.json();
        const rows=(data.categories?.[category]||[]);
        box.innerHTML=rows.map(r=>`<div class="scenario">
            <div class="flex justify-between gap-2"><strong>${r.name}</strong><span class="mono muted">${r.symbol}</span></div>
            <div class="text-xl font-bold mono mt-2">${formatBoardPrice(r.price)}</div>
            <div class="grid grid-cols-3 gap-2 mt-3 text-[10px]"><div><span class="muted">1D</span><div class="mono ${Number(r.change_1d||0)>=0?'green':'red'}">${pctBoard(r.change_1d)}</div></div><div><span class="muted">5D</span><div class="mono ${Number(r.change_5d||0)>=0?'green':'red'}">${pctBoard(r.change_5d)}</div></div><div><span class="muted">20D</span><div class="mono ${Number(r.change_20d||0)>=0?'green':'red'}">${pctBoard(r.change_20d)}</div></div></div>
            <div class="h-[90px] mt-3"><canvas id="board-${r.symbol.replace(/[^A-Za-z0-9]/g,'-')}"></canvas></div>
            <button class="btn px-3 py-1.5 text-xs mt-3 w-full" onclick="analyzeBoardTicker('${r.symbol.replace(/'/g, "\'")}')">Analyze ${r.symbol} →</button>
        </div>`).join('') || '<div class="muted text-sm">No market data returned.</div>';
        rows.forEach(r=>{ const el=document.getElementById('board-'+r.symbol.replace(/[^A-Za-z0-9]/g,'-')); if(!el || !r.history_close?.length) return; boardCharts.push(new Chart(el,{type:'line',data:{labels:r.history_dates,datasets:[{data:r.history_close,borderColor:'#22d3ee',pointRadius:0,borderWidth:1.5}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{display:false},y:{display:false}}}})); });
    }catch(err){box.innerHTML='<div class="text-sm red">Instrument board unavailable: '+(err.message||'request failed')+'</div>';}
}

function renderFinancialModel(data){
    modelData=data;
    const d=data.dashboard||{}; const byId=id=>document.getElementById(id); if(!byId('modelKpis')) return;
    const k=[['Current Price',d['Current Price'],''],['DCF / Share',d['DCF Price'],''],['WACC',Number(d['WACC'])*100,'%'],['Terminal Growth',Number(d['Terminal Growth'])*100,'%'],['Enterprise Value',d['Enterprise Value'],''],['Equity Value',d['Equity Value'],''],['Forecast CAGR',Number(d['Forecast CAGR'])*100,'%'],['Beta',d['Beta'],'x']];
    byId('modelKpis').innerHTML=k.map(x=>`<div class="scenario"><div class="text-[10px] muted">${x[0]}</div><strong class="block mt-2 mono cyan">${x[1]==null?'N/A':Number(x[1]).toLocaleString(undefined,{maximumFractionDigits:2})}${x[2]}</strong></div>`).join('');
    byId('modelStatus').innerText=data.company_name+' · '+data.ticker+' · MODEL READY';
    const f=data.forecast||[];
    if(modelChartInstance) modelChartInstance.destroy();
    modelChartInstance=new Chart(byId('modelForecastChart'),{type:'line',data:{labels:f.map(x=>x.Year),datasets:[{label:'Revenue',data:f.map(x=>x.Revenue),borderColor:'#22d3ee',pointRadius:2},{label:'EBIT',data:f.map(x=>x.EBIT),borderColor:'#19d58b',pointRadius:2},{label:'FCFF',data:f.map(x=>x.FCFF),borderColor:'#fbbf24',pointRadius:2}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8492a3'}}},scales:{x:{ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}});
    const s=data.scenarios||[];
    if(modelScenarioChartInstance) modelScenarioChartInstance.destroy();
    modelScenarioChartInstance=new Chart(byId('modelScenarioChart'),{type:'bar',data:{labels:s.map(x=>x.Scenario),datasets:[{label:'Implied Price',data:s.map(x=>x['Implied Price']),borderWidth:1}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8492a3'}}},scales:{x:{ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}});
    const sens=data.sensitivity||[]; const st=byId('modelSensitivityTable'); st.innerHTML=sens.length?'<thead><tr><th>WACC</th>'+Object.keys(sens[0]).filter(k=>k!=='WACC').map(k=>'<th>'+k.replace('g=','')+'</th>').join('')+'</tr></thead><tbody>'+sens.map(r=>'<tr><td>'+Number(r.WACC*100).toFixed(2)+'%</td>'+Object.keys(r).filter(k=>k!=='WACC').map(k=>'<td>'+ (r[k]==null?'N/A':Number(r[k]).toFixed(2)) +'</td>').join('')+'</tr>').join('')+'</tbody>':'<tbody><tr><td>No sensitivity data</td></tr></tbody>';
    const w=data.wacc||[]; const wt=byId('modelWaccTable'); wt.innerHTML='<thead><tr><th>Component</th><th>Value</th></tr></thead><tbody>'+w.map(r=>'<tr><td>'+r.Component+'</td><td>'+((String(r.Component).includes('Rate')||String(r.Component).includes('Weight')||String(r.Component)==='WACC'||String(r.Component)==='Cost of Equity'||String(r.Component)==='Cost of Debt'||String(r.Component)==='Terminal Growth')?(Number(r.Value)*100).toFixed(2)+'%':Number(r.Value).toFixed(3))+'</td></tr>').join('')+'</tbody>';
    const h=data.historical||[]; const cols=h.length?['Year','Revenue','EBIT','Net Income','FCF','EBIT Margin','Net Margin','FCF Margin','Debt / Equity','DuPont ROE']:[]; const sp=byId('modelSpreadsheet'); sp.querySelector('thead').innerHTML='<tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr>'; sp.querySelector('tbody').innerHTML=h.map(r=>'<tr>'+cols.map(c=>{let v=r[c]; if(v==null)return '<td>N/A</td>'; if(String(c).includes('Margin')||String(c).includes('ROE'))return '<td>'+ (Number(v)*100).toFixed(2)+'%</td>'; return '<td>'+Number(v).toLocaleString(undefined,{maximumFractionDigits:2})+'</td>';}).join('')+'</tr>').join('');
}
async function loadFinancialModel(){
    const t=(document.getElementById('tickerInput')?.value||currentData?.ticker||'').trim(); if(!t) return;
    const byId=id=>document.getElementById(id); byId('modelStatus').innerText='CALCULATING…';
    try{ const res=await fetch(API_BASE+'/api/financial-model/'+encodeURIComponent(t)); const data=await res.json(); if(!res.ok) throw new Error(data.detail||'model failed'); renderFinancialModel(data); }
    catch(err){ byId('modelStatus').innerText='ERROR: '+(err.message||'model request failed'); }
}
function downloadFinancialModel(){const t=(document.getElementById('tickerInput')?.value||currentData?.ticker||'').trim(); if(t) window.location.href=API_BASE+'/api/download/financial-model/'+encodeURIComponent(t);}

function toggleNav(){
    const nav=document.querySelector('.nav'), shell=document.getElementById('mainShell'); if(!nav||!shell)return;
    nav.classList.toggle('collapsed');
    shell.classList.toggle('nav-main-collapsed', nav.classList.contains('collapsed'));
    localStorage.setItem('finmodel_nav_collapsed', nav.classList.contains('collapsed')?'1':'0');
    updateMainSeparation();
}
function updateMainSeparation(){
    const nav=document.querySelector('.nav');
    const shell=document.getElementById('mainShell');
    if(!nav || !shell) return;

    const rect=nav.getBoundingClientRect();
    const vw=window.innerWidth;
    const vh=window.innerHeight;
    const gap=20;
    const edge=16;
    const minMainWidth=340;
    const minMainHeight=260;

    shell.classList.remove('separate-left','separate-right');
    shell.style.paddingTop='0px';
    shell.style.paddingBottom='0px';

    /* Desktop: reserve a completely separate rectangular viewport beside the sidebar.
       The main screen never sits underneath or inside the sidebar. */
    if(vw > 900){
        const sidebarOnLeft = (rect.left + rect.width/2) <= vw/2;
        let left=edge, right=edge;
        if(sidebarOnLeft){
            left=Math.min(vw-edge-minMainWidth, Math.max(edge, rect.right+gap));
            shell.classList.add('separate-left');
        } else {
            right=Math.min(vw-edge-minMainWidth, Math.max(edge, vw-rect.left+gap));
            shell.classList.add('separate-right');
        }
        shell.style.left=left+'px';
        shell.style.right=right+'px';
        shell.style.top=edge+'px';
        shell.style.bottom=edge+'px';
        return;
    }

    /* Narrow screens: use a vertical split. Put the main view on the side of the
       viewport opposite the sidebar, while keeping the two views non-overlapping. */
    const sidebarNearTop=(rect.top + rect.height/2) <= vh/2;
    let top=edge, bottom=edge;
    if(sidebarNearTop){
        top=Math.min(vh-edge-minMainHeight, Math.max(edge, rect.bottom+gap));
    }else{
        bottom=Math.min(vh-edge-minMainHeight, Math.max(edge, vh-rect.top+gap));
    }
    shell.style.left=edge+'px';
    shell.style.right=edge+'px';
    shell.style.top=top+'px';
    shell.style.bottom=bottom+'px';
}

function setupMovableSidebar(){
    const nav=document.querySelector('.nav');
    const handle=document.querySelector('.nav-drag-handle');
    const resizeHandles=[...document.querySelectorAll('.nav-resize-handle')];
    const workspace=document.getElementById('workspaceNav');
    if(!nav || !handle) return;

    // Restore sidebar position and size.
    try {
        const savedPos=localStorage.getItem('finmodel_sidebar_position');
        if(savedPos){
            const pos=JSON.parse(savedPos);
            if(Number.isFinite(pos.left)) nav.style.left=Math.max(0,Math.min(pos.left,Math.max(0,window.innerWidth-nav.offsetWidth)))+'px';
            if(Number.isFinite(pos.top)) nav.style.top=Math.max(0,Math.min(pos.top,Math.max(0,window.innerHeight-nav.offsetHeight)))+'px';
        }
        const savedSize=localStorage.getItem('finmodel_sidebar_size');
        if(savedSize){
            const size=JSON.parse(savedSize);
            if(Number.isFinite(size.width) && size.width>=180) nav.style.width=Math.min(size.width,460)+'px';
            if(Number.isFinite(size.height) && size.height>=300) nav.style.height=Math.min(size.height,window.innerHeight-32)+'px';
        }
    } catch(e) {}
    try {
        if(localStorage.getItem('finmodel_nav_collapsed')==='1') nav.classList.add('collapsed');
    } catch(e) {}
    requestAnimationFrame(updateMainSeparation);

    // Sidebar drag: move the complete sidebar across the workspace.
    let dragState=null;
    const beginSidebarDrag=(e)=>{
        if(e.target.closest('button') || e.target.closest('.nav-resize-handle')) return;
        if(e.button!==undefined && e.button!==0) return;
        const r=nav.getBoundingClientRect();
        dragState={pointerId:e.pointerId,startX:e.clientX,startY:e.clientY,left:r.left,top:r.top};
        nav.classList.add('dragging-sidebar');
        handle.setPointerCapture?.(e.pointerId);
        e.preventDefault();
    };
    const moveSidebar=(e)=>{
        if(!dragState || e.pointerId!==dragState.pointerId) return;
        const left=Math.max(0,Math.min(dragState.left+(e.clientX-dragState.startX),window.innerWidth-nav.offsetWidth));
        const top=Math.max(0,Math.min(dragState.top+(e.clientY-dragState.startY),window.innerHeight-nav.offsetHeight));
        nav.style.left=left+'px';
        nav.style.top=top+'px';
        updateMainSeparation();
    };
    const endSidebarDrag=(e)=>{
        if(!dragState || (e.pointerId!==undefined && e.pointerId!==dragState.pointerId)) return;
        nav.classList.remove('dragging-sidebar');
        try { localStorage.setItem('finmodel_sidebar_position',JSON.stringify({left:nav.offsetLeft,top:nav.offsetTop})); } catch(err) {}
        dragState=null;
    };
    handle.addEventListener('pointerdown',beginSidebarDrag);
    handle.addEventListener('pointermove',moveSidebar);
    handle.addEventListener('pointerup',endSidebarDrag);
    handle.addEventListener('pointercancel',endSidebarDrag);

    // Sidebar resize: every edge and corner is independently draggable.
    // Horizontal resizing works from BOTH left and right, so the sidebar can grow/shrink in either direction.
    let resizeState=null;
    const beginResize=(e)=>{
        if(e.button!==undefined && e.button!==0) return;
        const handle=e.currentTarget;
        const dir=handle?.dataset?.resize || 'br';
        const r=nav.getBoundingClientRect();
        resizeState={pointerId:e.pointerId,startX:e.clientX,startY:e.clientY,width:r.width,height:r.height,left:r.left,top:r.top,right:r.right,bottom:r.bottom,dir};
        nav.classList.add('resizing-sidebar');
        handle.setPointerCapture?.(e.pointerId);
        e.preventDefault();
        e.stopPropagation();
    };
    const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
    const moveResize=(e)=>{
        if(!resizeState || e.pointerId!==resizeState.pointerId) return;
        const dx=e.clientX-resizeState.startX, dy=e.clientY-resizeState.startY;
        const minW=180, maxW=Math.min(460,window.innerWidth-16);
        const minH=300, maxH=Math.min(window.innerHeight-16,Math.max(300,window.innerHeight));
        let left=resizeState.left, top=resizeState.top;
        let width=resizeState.width, height=resizeState.height;
        const dir=resizeState.dir;
        if(dir.includes('l')){
            const newLeft=clamp(resizeState.left+dx,0,resizeState.right-minW);
            left=newLeft; width=clamp(resizeState.right-newLeft,minW,Math.min(maxW,resizeState.right));
            left=resizeState.right-width;
        }
        if(dir.includes('r')){
            width=clamp(resizeState.width+dx,minW,Math.min(maxW,window.innerWidth-resizeState.left));
        }
        if(dir.includes('t')){
            const newTop=clamp(resizeState.top+dy,0,resizeState.bottom-minH);
            top=newTop; height=clamp(resizeState.bottom-newTop,minH,Math.min(maxH,resizeState.bottom));
            top=resizeState.bottom-height;
        }
        if(dir.includes('b')){
            height=clamp(resizeState.height+dy,minH,Math.min(maxH,window.innerHeight-resizeState.top));
        }
        nav.style.left=left+'px';
        nav.style.top=top+'px';
        nav.style.width=width+'px';
        nav.style.height=height+'px';
        updateMainSeparation();
    };
    const endResize=(e)=>{
        if(!resizeState || (e.pointerId!==undefined && e.pointerId!==resizeState.pointerId)) return;
        nav.classList.remove('resizing-sidebar');
        try {
            localStorage.setItem('finmodel_sidebar_size',JSON.stringify({width:nav.offsetWidth,height:nav.offsetHeight}));
            localStorage.setItem('finmodel_sidebar_position',JSON.stringify({left:nav.offsetLeft,top:nav.offsetTop}));
        } catch(err) {}
        resizeState=null;
    };
    resizeHandles.forEach(handle=>{
        handle.addEventListener('pointerdown',beginResize);
        handle.addEventListener('pointermove',moveResize);
        handle.addEventListener('pointerup',endResize);
        handle.addEventListener('pointercancel',endResize);
    });
    const resetSidebarSize=()=>{
        nav.style.width='235px';
        nav.style.height='calc(100vh - 32px)';
        try { localStorage.setItem('finmodel_sidebar_size',JSON.stringify({width:235,height:Math.max(300,window.innerHeight-32)})); } catch(err) {}
        updateMainSeparation();
    };
    resizeHandles.forEach(handle=>handle.addEventListener('dblclick',resetSidebarSize));

    // Reorder workspace items vertically.
    if(workspace){
        let dragged=null;
        const savedOrder=localStorage.getItem('finmodel_nav_order');
        if(savedOrder){
            try{
                const ids=JSON.parse(savedOrder);
                ids.forEach(id=>{
                    const node=workspace.querySelector('[data-view="'+CSS.escape(id)+'"]');
                    if(node) workspace.appendChild(node);
                });
            }catch(e){}
        }
        [...workspace.querySelectorAll('.navlink')].forEach(item=>{
            item.setAttribute('draggable','true');
            item.addEventListener('dragstart',()=>{ dragged=item; item.classList.add('dragging'); });
            item.addEventListener('dragend',()=>{
                item.classList.remove('dragging');
                workspace.querySelectorAll('.drop-target').forEach(x=>x.classList.remove('drop-target'));
                dragged=null;
                const order=[...workspace.querySelectorAll('.navlink')].map(x=>x.dataset.view);
                localStorage.setItem('finmodel_nav_order',JSON.stringify(order));
            });
            item.addEventListener('dragover',(e)=>{
                e.preventDefault();
                if(!dragged || dragged===item) return;
                item.classList.add('drop-target');
                const rect=item.getBoundingClientRect();
                const before=e.clientY<rect.top+rect.height/2;
                workspace.insertBefore(dragged,before?item:item.nextSibling);
            });
            item.addEventListener('dragleave',()=>item.classList.remove('drop-target'));
            item.addEventListener('drop',(e)=>{e.preventDefault(); item.classList.remove('drop-target');});
        });
    }

    window.addEventListener('resize',()=>{
        const r=nav.getBoundingClientRect();
        const left=Math.max(0,Math.min(r.left,window.innerWidth-nav.offsetWidth));
        const top=Math.max(0,Math.min(r.top,window.innerHeight-nav.offsetHeight));
        nav.style.left=left+'px';
        nav.style.top=top+'px';
        if(window.innerWidth < nav.offsetWidth + 20) nav.style.width=Math.max(180,window.innerWidth-20)+'px';
        if(window.innerHeight < nav.offsetHeight + 16) nav.style.height=Math.max(300,window.innerHeight-16)+'px';
        updateMainSeparation();
    });

    nav.addEventListener('dblclick',(e)=>{
        if(e.target.closest('.navlink') || e.target.closest('button') || e.target.closest('.nav-resize-handle')) return;
        nav.style.left='16px'; nav.style.top='16px';
        try { localStorage.setItem('finmodel_sidebar_position',JSON.stringify({left:16,top:16})); } catch(err) {}
        updateMainSeparation();
    });
}

function saveNavOrder(){
    const workspace=document.getElementById('workspaceNav');
    if(!workspace)return;
    localStorage.setItem('finmodel_workspace_order',JSON.stringify([...workspace.querySelectorAll('.navlink')].map(x=>x.dataset.view)));
}

function switchView(viewName, element) {
    document.querySelectorAll('.navlink').forEach(el => el.classList.remove('active'));
    if (element) element.classList.add('active');

    document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active-view'));
    const activeSection = document.getElementById('view-' + viewName);
    if (activeSection) activeSection.classList.add('active-view');

    // Fundamental charts are initialized only after their panel becomes visible.
    // Chart.js can calculate a zero/incorrect canvas size when a chart is created
    // inside a display:none section during the initial dashboard render.
    if (viewName === 'fundamentals' && currentData) {
        requestAnimationFrame(() => requestAnimationFrame(() => renderFundamentalAnalysis(currentData, true)));
    }
    if (viewName === 'model') {
        requestAnimationFrame(() => loadFinancialModel());
    }
}

function setTimeframe(tf, element) {
    selectedTimeframe = tf;
    (element ? element.parentElement : document).querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    if (element) element.classList.add('active');
    if (currentData) renderMainLineChart(currentData);
}


function renderRiskAnalytics(data) {
    const rm = data.risk_metrics || {};
    const rc = data.risk_charts || {};
    document.getElementById('riskCapm').innerText = (rm.capm_expected_return * 100).toFixed(2) + '%';
    document.getElementById('riskStd').innerText = (rm.std_dev_5y * 100).toFixed(2) + '%';
    document.getElementById('riskVar').innerText = Number(rm.variance_5y).toFixed(6);
    document.getElementById('riskSharpe').innerText = Number(rm.sharpe_ratio).toFixed(2);
    document.getElementById('riskBeta2').innerText = Number(rm.beta).toFixed(3);
    const infoEl = document.getElementById('riskInformation');
    const jensenEl = document.getElementById('riskJensen');
    const trackingEl = document.getElementById('riskTrackingError');
    if (infoEl) infoEl.innerText = Number(rm.information_ratio ?? 0).toFixed(3);
    if (jensenEl) jensenEl.innerText = (Number(rm.jensen_alpha ?? 0) * 100).toFixed(2) + '%';
    if (trackingEl) trackingEl.innerText = (Number(rm.tracking_error ?? 0) * 100).toFixed(2) + '%';
    if (!rc.sml) return;

    if (smlChartInstance) smlChartInstance.destroy();
    smlChartInstance = new Chart(document.getElementById('smlChart'), {
        type:'line', data:{labels:rc.sml.beta,datasets:[{label:'SML',data:rc.sml.return,borderColor:'#22d3ee',borderWidth:2,pointRadius:0},{type:'scatter',label:data.ticker+' CAPM point',data:[{x:Number(rm.beta),y:Number(rm.capm_expected_return)*100}],backgroundColor:'#19d58b',pointRadius:6}]},
        options:{responsive:true,maintainAspectRatio:false,scales:{x:{title:{display:true,text:'Beta'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{title:{display:true,text:'Expected Return (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}
    });

    if (calChartInstance) calChartInstance.destroy();
    calChartInstance = new Chart(document.getElementById('calChart'), {
        type:'line', data:{labels:rc.cal.volatility,datasets:[{label:'CAL',data:rc.cal.return,borderColor:'#19d58b',borderWidth:2,pointRadius:0}]},
        options:{responsive:true,maintainAspectRatio:false,scales:{x:{title:{display:true,text:'Volatility (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{title:{display:true,text:'Expected Return (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}
    });

    if (scatterChartInstance) scatterChartInstance.destroy();
    scatterChartInstance = new Chart(document.getElementById('scatterChart'), {
        type:'scatter', data:{datasets:[{label:'Daily returns',data:rc.scatter.map(p=>({x:p.market,y:p.asset})),backgroundColor:'rgba(34,211,238,.35)',pointRadius:2},{type:'line',label:'Regression',data:[{x:rc.regression.x0,y:rc.regression.y0},{x:rc.regression.x1,y:rc.regression.y1}],borderColor:'#fbbf24',borderWidth:2,pointRadius:0}]},
        options:{responsive:true,maintainAspectRatio:false,scales:{x:{title:{display:true,text:'Market daily return (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{title:{display:true,text:'Asset daily return (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}
    });
    document.getElementById('scatterStats').innerHTML = `
        <div class="scenario"><div class="text-[10px] muted">Beta</div><strong class="mono">${Number(rc.beta).toFixed(3)}</strong></div>
        <div class="scenario"><div class="text-[10px] muted">Correlation</div><strong class="mono">${Number(rc.correlation).toFixed(3)}</strong></div>
        <div class="scenario"><div class="text-[10px] muted">Market annualized</div><strong class="mono">${(Number(rc.market_annual_return)*100).toFixed(2)}%</strong></div>
        <div class="scenario"><div class="text-[10px] muted">Asset annualized</div><strong class="mono">${(Number(rc.asset_annual_return)*100).toFixed(2)}%</strong></div>`;
}

function recalculateTrade() {
    if (!currentData) return;
    const equity = Number(document.getElementById('tradeEquity').value || 10000);
    const usedMargin = Math.max(Number(document.getElementById('tradeUsedMargin').value || 0), 0);
    const units = Math.max(Number(document.getElementById('tradeUnitsInput').value || 0), 0);
    const chosenEntry = Number(document.getElementById('tradeEntryPrice').value || 0);
    const direction = document.getElementById('tradeDirection').value;
    const t = currentData.technicals || {};
    const current = Number(currentData.current_price);
    const atr = Math.max(Number(t.atr14 || 0), current * .01);
    const longEntry = Number(currentData.trade_setup?.long_suitable_entry || current);
    const shortEntry = Number(currentData.trade_setup?.short_suitable_entry || current);
    const suitableEntry = direction === 'LONG' ? longEntry : shortEntry;
    const entry = chosenEntry > 0 ? chosenEntry : suitableEntry;
    const sl = direction === 'LONG' ? entry - atr : entry + atr;
    const tp = direction === 'LONG' ? entry + 2*atr : entry - 2*atr;
    const riskDollars = units*Math.abs(entry-sl);
    const notional = units*entry;
    const pnlTP = units*Math.abs(tp-entry);
    const pnlSL = -units*Math.abs(sl-entry);
    const freeMargin = equity - usedMargin;
    const impliedLeverage = usedMargin > 0 ? notional/usedMargin : null;

    document.getElementById('tradeEntry').innerText='$'+entry.toFixed(2);
    document.getElementById('tradeSL').innerText='$'+sl.toFixed(2);
    document.getElementById('tradeTP').innerText='$'+tp.toFixed(2);
    document.getElementById('tradeRR').innerText='1 : 2.0';
    document.getElementById('tradeSuitableEntry').innerText='$'+suitableEntry.toFixed(2);
    document.getElementById('tradeLongEntry').innerText='$'+longEntry.toFixed(2);
    document.getElementById('tradeShortEntry').innerText='$'+shortEntry.toFixed(2);
    document.getElementById('tradeRiskDollars').innerText='$'+riskDollars.toFixed(2);
    document.getElementById('tradeNotional').innerText='$'+notional.toFixed(2);
    document.getElementById('tradeFreeMargin').innerText='$'+freeMargin.toFixed(2)+(freeMargin<0 ? ' ⚠' : '');
    document.getElementById('tradeATR').innerText='$'+atr.toFixed(2);
    document.getElementById('tradeUsedMarginOut').innerText='$'+usedMargin.toFixed(2);
    document.getElementById('tradeImpliedLeverage').innerText = impliedLeverage!=null ? impliedLeverage.toFixed(2)+'x' : '--';
    document.getElementById('eqStart').innerText='$'+equity.toFixed(2);
    document.getElementById('eqTP').innerText='$'+(equity+pnlTP).toFixed(2);
    document.getElementById('eqSL').innerText='$'+(equity+pnlSL).toFixed(2);
    document.getElementById('tradeNote').innerText='Scenario: '+direction+' · '+units+' units · fitted '+direction.toLowerCase()+' entry $'+suitableEntry.toFixed(2)+'. This direction-specific price uses the same multi-factor analysis stack (technical structure, DCF/peer/ML/analyst valuation, RSI momentum and beta/ATR risk), but separates the long and short entry logic. Stop is 1 ATR from entry and target is 2 ATR. Educational scenario only.';
}
function useSuitableEntry() {
    const suitableText = document.getElementById('tradeSuitableEntry').innerText;
    const suitable = Number(String(suitableText).replace('$',''));
    if (!isNaN(suitable) && suitable > 0) {
        document.getElementById('tradeEntryPrice').value = suitable.toFixed(2);
        recalculateTrade();
    }
}


function applyFuturePreset() {
    const p = {
        CUSTOM:[75,1], CL:[75,1000], GC:[3400,100], SI:[39,5000], NG:[3.0,10000], ZC:[6500,50], ZW:[650,50], ES:[6500,50], NQ:[24000,20]
    };
    const v = p[document.getElementById('futureContract').value] || p.CUSTOM;
    document.getElementById('futureSpot').value = v[0];
    document.getElementById('futureMultiplier').value = v[1];
    calculateFuturesSimulation();
}

function calculateFuturesSimulation() {
    const spot = Math.max(Number(document.getElementById('futureSpot').value || 0), 0.000001);
    const multiplier = Math.max(Number(document.getElementById('futureMultiplier').value || 1), 1);
    const contracts = Math.max(Number(document.getElementById('futureContracts').value || 1), 1);
    const rate = Number(document.getElementById('futureRate').value || 0)/100;
    const carry = Number(document.getElementById('futureCarry').value || 0)/100;
    const months = Math.max(Number(document.getElementById('futureExpiry').value || 3), 0.1);
    const T = months/12;
    const direction = document.getElementById('futureDirection').value;
    const shock = Math.max(Number(document.getElementById('futureShock').value || 5), 0.1)/100;
    const theo = spot * Math.exp((rate + carry) * T);
    const basis = theo - spot;
    const notional = theo * multiplier * contracts;
    const breakeven = theo;
    const upper = theo*(1+shock), lower = theo*(1-shock);
    const scenarioPrice = direction === 'LONG' ? upper : lower;
    const pnl = (scenarioPrice-theo) * multiplier * contracts * (direction === 'LONG' ? 1 : -1);

    document.getElementById('futureTheo').innerText = '$'+theo.toFixed(2);
    document.getElementById('futureNotional').innerText = '$'+notional.toLocaleString(undefined,{maximumFractionDigits:2});
    document.getElementById('futureBasis').innerText = (basis>=0?'+':'')+'$'+basis.toFixed(2);
    document.getElementById('futureBreakeven').innerText = '$'+breakeven.toFixed(2);
    document.getElementById('futureScenarioPnl').innerText = (pnl>=0?'+':'')+'$'+pnl.toLocaleString(undefined,{maximumFractionDigits:2});

    const prices=[]; const pnlData=[];
    const step = theo * shock / 5;
    for(let i=-25;i<=25;i++){
        const x = theo + i*step; prices.push(x);
        pnlData.push((x-theo)*multiplier*contracts*(direction === 'LONG' ? 1 : -1));
    }
    if (window.futuresPayoffChartInstance) window.futuresPayoffChartInstance.destroy();
    window.futuresPayoffChartInstance = new Chart(document.getElementById('futuresPayoffChart'), {type:'line',data:{labels:prices.map(x=>x.toFixed(2)),datasets:[{label:(direction==='LONG'?'Long':'Short')+' futures P/L',data:pnlData,borderWidth:2,pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},scales:{x:{title:{display:true,text:'Expiry / Settlement Price'}},y:{title:{display:true,text:'P/L'}}}}});

    const labels=[], curve=[];
    for(let i=0;i<=24;i++){ const t=T*i/24; labels.push((t*12).toFixed(2)+'m'); curve.push(spot*Math.exp((rate+carry)*t)); }
    if (window.futuresCurveChartInstance) window.futuresCurveChartInstance.destroy();
    window.futuresCurveChartInstance = new Chart(document.getElementById('futuresCurveChart'), {type:'line',data:{labels,datasets:[{label:'Theoretical futures price',data:curve,borderWidth:2,pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,scales:{x:{title:{display:true,text:'Time to expiry'}},y:{title:{display:true,text:'Futures price'}}}}});

    const contract = document.getElementById('futureContract').selectedOptions[0].text;
    const marketView = direction === 'LONG' ? 'bullish settlement scenario' : 'bearish settlement scenario';
    document.getElementById('futureHypothesis').innerText = contract + ': the modelled theoretical futures price is $'+theo.toFixed(2)+' with a basis of $'+basis.toFixed(2)+' over spot. Under the selected '+(direction==='LONG'?'long':'short')+' position, the '+(shock*100).toFixed(1)+'% stress scenario assumes a '+marketView+' and produces an illustrative P/L of '+(pnl>=0?'+':'')+'$'+pnl.toFixed(2)+' for '+contracts+' contract(s). This is a hypothetical payoff simulation; actual futures prices depend on market expectations, convenience yield, storage, financing, liquidity, contract specifications and exchange settlement rules.';
}

function renderCfdScenario(data) {
    const c = data.cfd_scenario || {};
    const d = data.directional_decision || {};
    const decisionEl = document.getElementById('directionalDecision');
    if (decisionEl) {
        decisionEl.innerText = d.decision || 'NO CLEAR DIRECTION';
        decisionEl.className = 'text-2xl mono ' + (d.decision === 'LONG BIAS' ? 'green' : d.decision === 'SHORT BIAS' ? 'red' : 'cyan');
    }
    const confEl = document.getElementById('decisionConfidence');
    if (confEl) { confEl.innerText = d.confidence || '--'; confEl.className = 'badge ' + ((d.confidence||'').startsWith('HIGH') ? 'green' : (d.confidence||'').startsWith('LOW') ? 'amber' : ''); }
    const ds = document.getElementById('decisionScore'); if (ds) ds.innerText = (Number(d.score||0)>=0?'+':'') + Number(d.score||0).toFixed(1) + ' / 100';
    const da = document.getElementById('decisionAction'); if (da) da.innerText = d.action || '--';
    const dc = document.getElementById('decisionCatalyst'); if (dc) dc.innerText = Number(d.catalyst_pressure||0).toFixed(0) + ' / 100';
    const lc = document.getElementById('decisionLongCase'); if (lc) lc.innerHTML = (d.long_case||[]).length ? d.long_case.map(x=>'<div>• '+x+'</div>').join('') : '<span class="muted">No strong bullish factor.</span>';
    const scase = document.getElementById('decisionShortCase'); if (scase) scase.innerHTML = (d.short_case||[]).length ? d.short_case.map(x=>'<div>• '+x+'</div>').join('') : '<span class="muted">No strong bearish factor.</span>';
    const fmtMoney = v => '$' + Number(v || 0).toFixed(2);
    const pct = v => Number(v || 0).toFixed(1) + '%';
    document.getElementById('cfdSpot').innerText = fmtMoney(c.spot);
    document.getElementById('cfdAtr').innerText = fmtMoney(c.atr14);
    document.getElementById('cfdVol').innerText = pct(Number(c.volatility_5y||0)*100);
    document.getElementById('cfdBeta').innerText = Number(c.beta||0).toFixed(3);
    document.getElementById('cfdCatalyst').innerText = Number(c.catalyst_pressure_score||0).toFixed(0);
    document.getElementById('cfdEventRisk').innerText = Number(c.event_risk_score||0).toFixed(0);
    document.getElementById('cfdGeoRisk').innerText = Number(c.geopolitical_risk_score||0).toFixed(0);
    document.getElementById('cfdBias').innerText = (Number(c.headline_bias_score||0)>=0?'+':'') + Number(c.headline_bias_score||0).toFixed(0);
    document.getElementById('cfdEventCount').innerText = `${c.high_impact_event_count||0} high-impact / ${c.event_count||0} total events`;
    document.getElementById('cfdGeoCount').innerText = `${c.geopolitical_headline_count||0} related headlines`;
    const regime = document.getElementById('cfdRiskRegime');
    regime.innerText = c.risk_regime || 'WATCH';
    regime.className = 'badge ' + ((c.risk_regime==='ELEVATED')?'red':(c.risk_regime==='NORMAL'?'green':'amber'));

    document.getElementById('cfdLongEntry').innerText = fmtMoney(c.long_fitted_entry);
    document.getElementById('cfdShortEntry').innerText = fmtMoney(c.short_fitted_entry);

    const rows = (c.event_focus || []).map(e => `<div class="scenario flex flex-col md:flex-row md:items-center gap-3"><div class="w-28 mono text-xs">${e.date}</div><div class="flex-1"><strong class="text-sm">${e.event}</strong><div class="text-xs muted mt-1">${e.possible_result||''}</div></div><span class="badge">${e.asset_relevance||'MEDIUM'}</span></div>`).join('');
    document.getElementById('cfdEvents').innerHTML = rows || '<div class="text-sm muted">No upcoming macro events returned.</div>';

    const geo = data.geopolitical_news || [];
    document.getElementById('cfdGeoNews').innerHTML = geo.length ? geo.map(n=>`<div class="scenario news-row"><div class="flex flex-col md:flex-row md:justify-between gap-2"><strong class="text-sm">${n.title}</strong><span class="badge">${(n.tags||[]).join(' · ')}</span></div><div class="text-xs muted mt-2">${n.summary||'No summary available.'}</div><div class="text-[10px] muted mt-2">${n.source||'Source'} · ${n.published||''}</div>${n.url ? `<a class="text-[10px] cyan mt-2 inline-block" href="${n.url}" target="_blank" rel="noopener">Open source ↗</a>` : ''}</div>`).join('') : '<div class="text-sm muted">No geopolitically relevant headlines were returned for this asset.</div>';

    const interp = `Catalyst pressure is ${Number(c.catalyst_pressure_score||0).toFixed(0)}/100 (${c.risk_regime||'WATCH'}). The score combines the count of high-impact macro events, geopolitically tagged headlines, and absolute headline/trend bias. The fitted long/short prices come from the existing multi-factor Trade Lab model. Scenario graph values show hypothetical P/L for 1 unit under ±1 ATR and ±2 ATR shocks; spreads, financing, slippage, margin rules and liquidation are not modeled.`;
    document.getElementById('cfdInterpretation').innerText = interp;

    const sc = c.scenario_table || [];
    const long = sc.filter(x=>x.direction==='LONG');
    const shrt = sc.filter(x=>x.direction==='SHORT');
    const labels = long.map(x=> (x.shock_atr>0?'+':'') + x.shock_atr + ' ATR');
    const longData = long.map(x=>x.pnl_per_unit);
    const shortData = shrt.map(x=>x.pnl_per_unit);
    if (cfdShockChartInstance) cfdShockChartInstance.destroy();
    cfdShockChartInstance = new Chart(document.getElementById('cfdShockChart'), {type:'line', data:{labels,datasets:[{label:'Long P/L / unit',data:longData,borderColor:'#19d58b',borderWidth:2,pointRadius:3},{label:'Short P/L / unit',data:shortData,borderColor:'#ff667c',borderWidth:2,pointRadius:3}]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},scales:{x:{title:{display:true,text:'Price shock'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{title:{display:true,text:'P/L per unit'},ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}});

    if (cfdCatalystChartInstance) cfdCatalystChartInstance.destroy();
    cfdCatalystChartInstance = new Chart(document.getElementById('cfdCatalystChart'), {type:'bar',data:{labels:['Macro event','Geopolitical','Headline/trend'],datasets:[{label:'Pressure score',data:[Number(c.event_risk_score||0),Number(c.geopolitical_risk_score||0),Math.abs(Number(c.headline_bias_score||0))],borderWidth:1}]},options:{responsive:true,maintainAspectRatio:false,scales:{x:{ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{beginAtZero:true,max:100,title:{display:true,text:'0–100'},ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}});
}



function renderOpportunities(data) {
    const rows=data.rows||[], long=data.long_candidates||[], short=data.short_candidates||[], watch=data.watchlist||[];
    const byId=id=>document.getElementById(id);
    if(byId('oppUniverse')) byId('oppUniverse').innerText=rows.length;
    if(byId('oppLongCount')) byId('oppLongCount').innerText=long.length;
    if(byId('oppShortCount')) byId('oppShortCount').innerText=short.length;
    if(byId('oppWatchCount')) byId('oppWatchCount').innerText=watch.length;
    const cls=v=>Number(v)>=25?'green':Number(v)<=-25?'red':'cyan';
    const dcls=d=>d==='LONG CANDIDATE'?'green':d==='SHORT CANDIDATE'?'red':'cyan';
    const tbody=document.querySelector('#opportunityTable tbody');
    if(tbody) tbody.innerHTML=rows.map(r=>`<tr><td><strong>${r.ticker}</strong><div class="text-[10px] muted">${r.name}</div></td><td class="${cls(r.day_score)}">${Number(r.day_score).toFixed(1)}</td><td class="${cls(r.week_score)}">${Number(r.week_score).toFixed(1)}</td><td class="${cls(r.composite_score)}"><strong>${Number(r.composite_score).toFixed(1)}</strong></td><td>${Number(r.ret_1d_pct).toFixed(2)}%</td><td>${Number(r.ret_5d_pct).toFixed(2)}%</td><td>${Number(r.ret_20d_pct).toFixed(2)}%</td><td>${Number(r.rsi14).toFixed(1)}</td><td>${r.risk_flag}</td><td class="${dcls(r.direction)}"><strong>${r.direction}</strong></td></tr>`).join('')||'<tr><td colspan="10" class="muted">No data returned.</td></tr>';
    const top=rows.slice(0,12);
    if(opportunityBarChartInstance) opportunityBarChartInstance.destroy();
    opportunityBarChartInstance=new Chart(document.getElementById('opportunityBarChart'),{type:'bar',data:{labels:top.map(r=>r.ticker),datasets:[{label:'Composite score',data:top.map(r=>r.composite_score),borderWidth:1}]},options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,scales:{x:{min:-100,max:100,title:{display:true,text:'-100 bearish · +100 bullish'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{ticks:{color:'#c5cfdb'},grid:{color:'#17212d'}}}}});
    if(opportunityScatterChartInstance) opportunityScatterChartInstance.destroy();
    opportunityScatterChartInstance=new Chart(document.getElementById('opportunityScatterChart'),{type:'scatter',data:{datasets:[{label:'Assets',data:rows.map(r=>({x:r.day_score,y:r.week_score})),pointRadius:5}]},options:{responsive:true,maintainAspectRatio:false,plugins:{tooltip:{callbacks:{label:ctx=>{const r=rows[ctx.dataIndex];return `${r.ticker}: Day ${Number(r.day_score).toFixed(1)}, Week ${Number(r.week_score).toFixed(1)}`;}}}},scales:{x:{min:-100,max:100,title:{display:true,text:'Day trend score'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{min:-100,max:100,title:{display:true,text:'Week trend score'},ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}});
    if(byId('opportunityHypotheses')) byId('opportunityHypotheses').innerHTML=rows.slice(0,8).map(r=>`<div class="scenario"><div class="flex justify-between gap-3"><strong>${r.ticker} · ${r.direction}</strong><span class="mono ${dcls(r.direction)}">${Number(r.composite_score).toFixed(1)}</span></div><div class="text-xs muted mt-2">${r.hypothesis}</div><div class="text-[10px] muted mono mt-2">${r.scenario} · ${r.timeframe_alignment} · risk ${r.risk_flag}</div></div>`).join('')||'<div class="text-sm muted">No hypotheses available.</div>';
}

async function fetchOpportunities(){
    const raw=(document.getElementById('opportunityTickers')?.value||'').trim();
    const url=raw?API_BASE+'/api/market-opportunities?tickers='+encodeURIComponent(raw):API_BASE+'/api/market-opportunities';
    try{const res=await fetch(url);if(!res.ok)throw new Error('scanner request failed');renderOpportunities(await res.json());}catch(err){console.error(err);alert('Market scanner could not load data. Check the server/network and try again.');}
}

function useCurrentTickerForPortfolio(){
    const t=(document.getElementById('tickerInput')?.value||currentData?.ticker||'').trim().toUpperCase();
    if(t){document.getElementById('portfolioTickers').value=t; fetchPortfolioOptimizer();}
}

function renderPortfolioOptimizer(data){
    portfolioData=data;
    const s=data.selected||{};
    document.getElementById('portfolioStatus').innerText=(data.method||'OPTIMIZER')+' · '+(data.assets_used||0)+' ASSETS';
    document.getElementById('portReturn').innerText=(Number(s.expected_return||0)*100).toFixed(2)+'%';
    document.getElementById('portVol').innerText=(Number(s.volatility||0)*100).toFixed(2)+'%';
    document.getElementById('portVar').innerText=Number(s.variance||0).toFixed(6);
    document.getElementById('portSharpe').innerText=Number(s.sharpe||0).toFixed(3);
    document.getElementById('portBeta').innerText=Number(s.beta||0).toFixed(3);
    document.getElementById('portfolioMethodLabel').innerText=(data.method==='MIN_RISK'?'MINIMUM-VARIANCE WEIGHTS':data.method==='TANGENCY'?'TANGENCY / MAX-SHARPE WEIGHTS':'COMPARE MODE · SELECTED TABLE = MINIMUM-RISK WEIGHTS');
    document.getElementById('portfolioAssetCount').innerText=Number(data.assets_used||0);
    document.getElementById('portfolioWeightSum').innerText=(Number(s.weight_sum||0)*100).toFixed(2)+'%';

    const rows=data.rows||[];
    const tbody=document.querySelector('#portfolioTable tbody');
    tbody.innerHTML=rows.map(r=>`<tr>
        <td><strong>${r.ticker}</strong><div class="text-[10px] muted">${r.name||''}</div></td>
        <td class="cyan"><strong>${(Number(r.weight)*100).toFixed(2)}%</strong></td>
        <td>${(Number(r.min_risk_weight)*100).toFixed(2)}%</td>
        <td>${(Number(r.tangency_weight)*100).toFixed(2)}%</td>
        <td>${(Number(r.expected_return)*100).toFixed(2)}%</td>
        <td>${(Number(r.volatility)*100).toFixed(2)}%</td>
        <td>${Number(r.beta).toFixed(3)}</td>
        <td>${Number(r.asset_sharpe).toFixed(3)}</td>
        <td>${(Number(r.return_contribution)*100).toFixed(2)}%</td>
        <td>${(Number(r.risk_contribution)*100).toFixed(2)}%</td>
    </tr>`).join('') || '<tr><td colspan="10" class="muted">No valid assets.</td></tr>';

    const comp=document.getElementById('portfolioMethodComparison');
    const card=(title,m,active)=>`<div class="scenario ${active?'border border-emerald-800':''}"><div class="text-[10px] muted">${title}</div><div class="grid grid-cols-2 gap-2 mt-3 text-xs"><div><span class="muted">Return</span><div class="mono mt-1">${(Number(m.expected_return||0)*100).toFixed(2)}%</div></div><div><span class="muted">Volatility</span><div class="mono mt-1">${(Number(m.volatility||0)*100).toFixed(2)}%</div></div><div><span class="muted">Sharpe</span><div class="mono mt-1">${Number(m.sharpe||0).toFixed(3)}</div></div><div><span class="muted">Beta</span><div class="mono mt-1">${Number(m.beta||0).toFixed(3)}</div></div></div></div>`;
    comp.innerHTML=card('Minimum Risk',data.minimum_risk,data.method==='MIN_RISK')+card('Tangency / Max Sharpe',data.tangency,data.method==='TANGENCY');

    const failed=data.failed_assets||[];
    document.getElementById('portfolioNotes').innerHTML=`<div>Data window: <strong>${data.data_start||'N/A'}</strong> → <strong>${data.data_end||'N/A'}</strong> · Benchmark: <strong>${data.benchmark||'^GSPC'}</strong> · Risk-free: <strong>${(Number(data.risk_free_rate||0)*100).toFixed(2)}%</strong>.</div><div class="mt-2">${data.methodology||''}</div>${failed.length?`<div class="mt-2 red">Excluded assets: ${failed.map(x=>x.ticker+' — '+x.reason).join('; ')}</div>`:''}`;

    // Scatter
    if(portfolioScatterChartInstance) portfolioScatterChartInstance.destroy();
    portfolioScatterChartInstance=new Chart(document.getElementById('portfolioScatterChart'),{
        type:'scatter',
        data:{datasets:[{label:'Selected assets',data:(data.scatter||[]).map(r=>({x:Number(r.volatility)*100,y:Number(r.expected_return)*100,radius:Math.max(4,Math.min(14,4+Number(r.weight)*80)),ticker:r.ticker})),pointRadius:ctx=>ctx.raw?.radius||5}]},
        options:{responsive:true,maintainAspectRatio:false,plugins:{tooltip:{callbacks:{label:ctx=>{const r=ctx.raw;return `${r.ticker}: return ${Number(r.y).toFixed(2)}%, vol ${Number(r.x).toFixed(2)}%`;}}}},scales:{x:{title:{display:true,text:'Annualized Volatility (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{title:{display:true,text:'Expected Return (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}
    });

    // SML + points
    if(portfolioSmlChartInstance) portfolioSmlChartInstance.destroy();
    portfolioSmlChartInstance=new Chart(document.getElementById('portfolioSmlChart'),{
        type:'line',data:{datasets:[
            {label:'SML',data:(data.sml.beta||[]).map((b,i)=>({x:Number(b),y:Number(data.sml.return[i])*100})),borderWidth:2,pointRadius:0},
            {type:'scatter',label:'Assets',data:(data.scatter||[]).map(r=>({x:Number(r.beta),y:Number(r.expected_return)*100})),pointRadius:5}
        ]},options:{responsive:true,maintainAspectRatio:false,scales:{x:{type:'linear',title:{display:true,text:'Beta'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{title:{display:true,text:'Expected Return (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}
    });

    // CAL through selected portfolio
    if(portfolioCalChartInstance) portfolioCalChartInstance.destroy();
    portfolioCalChartInstance=new Chart(document.getElementById('portfolioCalChart'),{
        type:'line',data:{datasets:[
            {label:'CAL',data:(data.cal.volatility||[]).map((v,i)=>({x:Number(v)*100,y:Number(data.cal.return[i])*100})),borderWidth:2,pointRadius:0},
            {type:'scatter',label:'Selected portfolio',data:[{x:Number(s.volatility)*100,y:Number(s.expected_return)*100}],pointRadius:7},
            {type:'scatter',label:'Tangency portfolio',data:[{x:Number(data.tangency.volatility)*100,y:Number(data.tangency.expected_return)*100}],pointRadius:6}
        ]},options:{responsive:true,maintainAspectRatio:false,scales:{x:{type:'linear',title:{display:true,text:'Volatility (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{title:{display:true,text:'Expected Return (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}
    });

    // Efficient frontier, with selected point
    if(portfolioFrontierChartInstance) portfolioFrontierChartInstance.destroy();
    portfolioFrontierChartInstance=new Chart(document.getElementById('portfolioFrontierChart'),{
        type:'line',data:{datasets:[
            {label:'Efficient frontier',data:(data.efficient_frontier||[]).map(p=>({x:Number(p.volatility)*100,y:Number(p.return)*100})),borderWidth:2,pointRadius:0,tension:.15},
            {type:'scatter',label:'Selected portfolio',data:[{x:Number(s.volatility)*100,y:Number(s.expected_return)*100}],pointRadius:7}
        ]},options:{responsive:true,maintainAspectRatio:false,scales:{x:{type:'linear',title:{display:true,text:'Volatility (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{title:{display:true,text:'Expected Return (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}
    });
}

async function fetchPortfolioOptimizer(){
    const raw=(document.getElementById('portfolioTickers')?.value||'').trim();
    if(!raw){alert('Enter at least one asset ticker.');return;}
    const method=document.getElementById('portfolioMethod').value;
    const lookback=document.getElementById('portfolioLookback').value;
    const url=API_BASE+'/api/portfolio/optimize?tickers='+encodeURIComponent(raw)+'&method='+encodeURIComponent(method)+'&lookback='+encodeURIComponent(lookback);
    const status=document.getElementById('portfolioStatus');
    status.innerText='LOADING…';
    try{const res=await fetch(url);const data=await res.json();if(!res.ok)throw new Error(data.detail||'Portfolio optimization failed');renderPortfolioOptimizer(data);}catch(err){console.error(err);status.innerText='ERROR';alert(err.message||'Portfolio optimizer could not load data.');}
}

function renderNewsCalendar(data) {
    const cal = data.economic_calendar || [];
    document.getElementById('macroCalendar').innerHTML = cal.map(e=>`<div class="scenario flex flex-col md:flex-row md:items-center gap-3"><div class="w-28 mono text-xs">${e.date}</div><div class="flex-1"><strong class="text-sm">${e.event}</strong><div class="text-xs muted mt-1">${e.possible_result}</div></div><div class="text-[10px] mono">${e.importance}</div></div>`).join('');
    const news = data.news || [];
    document.getElementById('assetNews').innerHTML = news.length ? news.map(n=>`<div class="scenario news-row"><div class="flex flex-col md:flex-row md:justify-between gap-2"><strong class="text-sm">${n.title}</strong><span class="badge">${n.impact}</span></div><div class="text-xs muted mt-2">${n.summary || 'No summary available.'}</div><div class="text-xs text-slate-300 mt-2"><strong>Possible result:</strong> ${n.possible_result}</div>${n.url ? `<a class="text-[10px] cyan mt-2 inline-block" href="${n.url}" target="_blank" rel="noopener">Open source ↗</a>` : ''}</div>`).join('') : '<div class="text-sm muted">No recent headlines returned by the provider.</div>';
}

function updateSliders() {
    const wacc = parseFloat(document.getElementById('waccSlider').value);
    const growth = parseFloat(document.getElementById('growthSlider').value);
    document.getElementById('waccVal').innerText = (wacc * 100).toFixed(1) + '%';
    document.getElementById('growthVal').innerText = (growth * 100).toFixed(2) + '%';
    if (currentData) recalculateDCF(wacc, growth);
}

function recalculateDCF(wacc, growth) {
    if (growth >= wacc) return;
    const fcf = currentData.inputs.latest_free_cash_flow;
    const g = currentData.inputs.projected_growth_rate;
    const proj = Array.from({length: 5}, (_, i) => fcf * Math.pow(1 + g, i + 1));
    const pvFcf = proj.reduce((sum, cf, i) => sum + (cf / Math.pow(1 + wacc, i + 1)), 0);
    const tv = (proj[4] * (1 + growth)) / (wacc - growth);
    const pvTv = tv / Math.pow(1 + wacc, 5);
    const ev = pvFcf + pvTv;

    const estimatedFair = currentData.fair_value_per_share > 0 ? (ev / (ev / currentData.fair_value_per_share)) : currentData.fair_value_per_share;
    const upside = ((estimatedFair - currentData.current_price) / currentData.current_price) * 100;

    document.getElementById('fairValue').innerText = '$' + estimatedFair.toFixed(2);
    document.getElementById('dcfTargetCopy').innerText = '$' + estimatedFair.toFixed(2);
    const dcfElem = document.getElementById('dcfUpside');
    dcfElem.innerText = (upside >= 0 ? '+' : '') + upside.toFixed(2) + '% Implied';
    dcfElem.className = 'text-xs mt-2 ' + (upside >= 0 ? 'green' : 'red');

    document.getElementById('baseVal').innerText = '$' + estimatedFair.toFixed(2);
    document.getElementById('bearVal').innerText = '$' + (estimatedFair * 0.82).toFixed(2);
    document.getElementById('bullVal').innerText = '$' + (estimatedFair * 1.24).toFixed(2);
}


function currentTickerForExport(){
    return (document.getElementById('tickerInput')?.value || currentData?.ticker || '').trim().toUpperCase();
}
function updateExportBadge(){
    const badge=document.getElementById('exportTickerBadge');
    if(badge){ const t=currentTickerForExport(); badge.innerText=t ? ('TICKER · '+t) : 'SELECT A TICKER'; }
}
function downloadExport(kind){
    const t=currentTickerForExport();
    if(!t){ alert('Enter a ticker first.'); return; }
    const base = kind==='market' ? API_BASE+`/api/download/excel/${encodeURIComponent(t)}/market` : kind==='fundamentals' ? API_BASE+`/api/download/excel/${encodeURIComponent(t)}/fundamentals` : API_BASE+`/api/download/excel/${encodeURIComponent(t)}`;
    window.location.href = base;
}
function downloadPowerBI(){
    const t=currentTickerForExport();
    if(!t){ alert('Enter a ticker first.'); return; }
    window.location.href = API_BASE+`/api/download/powerbi/${encodeURIComponent(t)}`;
}

let opportunitiesLoaded = false;
function renderQuick(q) {
    document.getElementById('companyName').innerText = q.company_name;
    document.getElementById('tickerSymbol').innerText = q.ticker;
    setAssetAwareLabels(q);
    document.getElementById('currentPrice').innerText = analysisPriceText(q, q.current_price, q.current_price < 10 ? 4 : 2);
    document.getElementById('currencyBadge').innerText = (q.quote_unit || q.currency) + ' · ' +
        String(q.sector || q.asset_class || '').toUpperCase() + ' · FULL ANALYSIS LOADING…';
    if (q.technicals) {
        document.getElementById('rsiVal').innerText = q.technicals.rsi14;
        document.getElementById('macdVal').innerText = q.technicals.macd;
        document.getElementById('volumeRatio').innerText = q.technicals.volume_ratio + 'x';
        renderTechnicalChart(q);
    }
    renderMainLineChart(q);
}
async function fetchValuation(options = {}) {
    // Auto-refresh runs silently: it must not pop alerts, jump the user back to the
    // dashboard, or re-scan the whole market every minute.
    const silent = options.silent === true;
    const ticker = document.getElementById('tickerInput').value.trim().toUpperCase();
    if(!ticker){ if(!silent) alert('Enter an asset ticker (e.g. AAPL, EURUSD=X, GC=F, CL=F, BTC-USD, ^TNX).'); return; }
    const wacc = document.getElementById('waccSlider').value;
    const growth = document.getElementById('growthSlider').value;

    let fullLoaded = false;
    if (!silent) {
        // Fast first paint: price, chart and technicals arrive in about a second while
        // the full analysis is still computing. Both share the same server-side downloads.
        fetch(API_BASE+'/api/quick/' + encodeURIComponent(ticker))
            .then(r => r.ok ? r.json() : null)
            .then(q => {
                const stillSelected = document.getElementById('tickerInput').value.trim().toUpperCase() === ticker;
                if (q && !fullLoaded && stillSelected) renderQuick(q);
            })
            .catch(() => {});
    }
    try {
        const url = API_BASE+'/api/analysis/' + encodeURIComponent(ticker) + '?discount_rate=' + wacc + '&perpetual_growth=' + growth;
        const response = await fetch(url);
        if (!response.ok) {
            let detail = 'Failed to fetch valuation data';
            try { detail = (await response.json()).detail || detail; } catch (e) {}
            if (!silent) alert(detail);
            return;
        }
        const data = await response.json();
        fullLoaded = true;
        currentData = data;
        const ds=document.getElementById('derivSpot'); if(ds) ds.value=Number(data.current_price).toFixed(2);
        const dv=document.getElementById('derivVol'); if(dv) dv.value=(Number(data.risk_metrics?.std_dev_5y||0)*100).toFixed(2);
        const dk=document.getElementById('derivStrike'); if(dk) dk.value=Number(data.current_price).toFixed(2);
        renderDashboardData(data);
        if (!silent) switchView('dashboard', document.querySelector('.navlink[data-view="dashboard"]'));
        if (!opportunitiesLoaded) { opportunitiesLoaded = true; fetchOpportunities(); }
    } catch (err) {
        console.error('Fetch error:', err);
    }
}


let ratioProfitChartInstance = null;
let ratioMarginChartInstance = null;

function fmtRatio(v, suffix='') {
    return v === null || v === undefined || Number.isNaN(Number(v)) ? 'N/A' : Number(v).toFixed(2) + suffix;
}

let statementTab = 'income';
function fmtStmt(v) {
    if (v == null || !isFinite(Number(v))) return 'N/A';
    const n = Number(v);
    const a = Math.abs(n);
    if (a >= 1e12) return (n/1e12).toFixed(2)+'T';
    if (a >= 1e9) return (n/1e9).toFixed(2)+'B';
    if (a >= 1e6) return (n/1e6).toFixed(2)+'M';
    if (a >= 1e3) return (n/1e3).toFixed(2)+'K';
    return n.toFixed(2);
}
function showStatementTab(tab) {
    statementTab = tab;
    if (currentData) renderFinancialStatements(currentData);
}
function renderFinancialStatements(data) {
    const fs = data.financial_statements || {};
    const years = fs.years || [];
    const status = document.getElementById('statementDataStatus');
    if (status) { status.innerText = years.length ? `${years.length} FY LOADED` : 'NO DATA'; status.className='badge '+(years.length?'green':''); }
    const sm = fs.analysis || {};
    const summary = document.getElementById('statementAnalysisSummary');
    if (summary) summary.innerHTML = [
        ['Revenue YoY', sm.revenue_growth_yoy], ['Net Income YoY', sm.net_income_growth_yoy],
        ['Assets YoY', sm.assets_growth_yoy], ['Debt YoY', sm.debt_growth_yoy]
    ].map(([k,v])=>`<div class="scenario"><div class="text-[10px] muted">${k}</div><strong class="block mt-2 mono">${v==null?'N/A':(Number(v)*100).toFixed(2)+'%'}</strong></div>`).join('');
    const table=document.getElementById('financialStatementTable');
    if (!table) return;
    const key = statementTab==='balance'?'balance_sheet':statementTab==='cash'?'cash_flow':'income_statement';
    const rows=fs[key]||[];
    table.querySelector('thead').innerHTML='<tr><th>Metric</th>'+years.map(y=>`<th>${y}</th>`).join('')+'</tr>';
    table.querySelector('tbody').innerHTML=rows.map(r=>`<tr><td>${r.metric}</td>${(r.values||[]).map((v,i)=>`<td class="mono">${r.metric==='Diluted EPS' && v!=null?'$'+Number(v).toFixed(2):fmtStmt(v)}</td>`).join('')}</tr>`).join('');
    const interp=document.getElementById('statementInterpretation');
    if (interp) interp.innerHTML=(sm.interpretation||[]).length ? '<strong>Analysis:</strong> '+sm.interpretation.join(' · ') : 'Provider did not return enough annual data for interpretation.';
}

function renderRatioHistory(data) {
    const history = data.ratio_history || [];
    const tbody = document.querySelector('#ratioHistoryTable tbody');
    if (!tbody) return;
    const years = history.map(x => x.year);
    document.querySelector('#ratioHistoryTable thead tr').innerHTML =
        '<th>Ratio</th>' + years.map(y => '<th>' + y + '</th>').join('') +
        (years.length < 5 ? Array(5-years.length).fill('<th>N/A</th>').join('') : '');

    const rows = [
        ['ROIC', 'roic', '%'], ['ROA', 'roa', '%'], ['ROE', 'roe', '%'],
        ['Gross Margin', 'gross_margin', '%'], ['Operating Margin', 'operating_margin', '%'],
        ['Net Margin', 'net_margin', '%'], ['Current Ratio', 'current_ratio', 'x'],
        ['Quick Ratio', 'quick_ratio', 'x'], ['Cash Ratio', 'cash_ratio', 'x'],
        ['Debt / Equity', 'debt_to_equity', 'x'], ['Interest Coverage', 'interest_coverage', 'x'],
        ['Asset Turnover', 'asset_turnover', 'x'], ['Inventory Turnover', 'inventory_turnover', 'x'],
        ['Receivables Turnover', 'receivables_turnover', 'x'], ['FCF Yield*', 'fcf_yield', '%']
    ];
    tbody.innerHTML = rows.map(r => {
        return '<tr><td>' + r[0] + '</td>' + history.map(h => {
            const v = h[r[1]];
            return '<td>' + fmtRatio(v, r[2]) + '</td>';
        }).join('') + (history.length < 5 ? Array(5-history.length).fill('<td>N/A</td>').join('') : '') + '</tr>';
    }).join('');

    document.getElementById('ratioDataStatus').innerText = history.length ? (history.length + ' FY AVAILABLE') : 'NO ANNUAL DATA';

    const labels = history.map(h => String(h.year)).reverse();
    const pick = key => history.map(h => h[key] == null ? null : Number(h[key])*100).reverse();

    if (ratioProfitChartInstance) ratioProfitChartInstance.destroy();
    ratioProfitChartInstance = new Chart(document.getElementById('ratioProfitChart'), {
        type:'line', data:{labels, datasets:[
            {label:'ROIC',data:pick('roic'),borderColor:'#19d58b',pointRadius:2,tension:.2},
            {label:'ROA',data:pick('roa'),borderColor:'#22d3ee',pointRadius:2,tension:.2},
            {label:'ROE',data:pick('roe'),borderColor:'#fbbf24',pointRadius:2,tension:.2}
        ]}, options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8492a3'}}},scales:{x:{grid:{color:'#17212d'},ticks:{color:'#64748b'}},y:{grid:{color:'#17212d'},ticks:{color:'#64748b',callback:v=>v+'%'}}}}
    });

    if (ratioMarginChartInstance) ratioMarginChartInstance.destroy();
    ratioMarginChartInstance = new Chart(document.getElementById('ratioMarginChart'), {
        type:'line', data:{labels, datasets:[
            {label:'Gross Margin',data:pick('gross_margin'),borderColor:'#19d58b',pointRadius:2,tension:.2},
            {label:'Operating Margin',data:pick('operating_margin'),borderColor:'#22d3ee',pointRadius:2,tension:.2},
            {label:'Net Margin',data:pick('net_margin'),borderColor:'#fbbf24',pointRadius:2,tension:.2}
        ]}, options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8492a3'}}},scales:{x:{grid:{color:'#17212d'},ticks:{color:'#64748b'}},y:{grid:{color:'#17212d'},ticks:{color:'#64748b',callback:v=>v+'%'}}}}
    });
}

let fundamentalFactorChartInstance = null;
let fundamentalValuationChartInstance = null;
let fundamentalTrendChartInstance = null;

function renderFundamentalAnalysis(data, recreateCharts = false) {
    const f = data?.fundamental_analysis || {};
    const byId = id => document.getElementById(id);
    const band = byId('fundamentalBand');
    if (!band) return;

    // Render the non-chart content first so a chart sizing problem can never
    // prevent the table/KPIs/notes from appearing.
    const scoreRaw = Number(f.score);
    const score = Number.isFinite(scoreRaw) ? scoreRaw : 0;
    band.innerText = f.rating_band ? String(f.rating_band).toUpperCase() : 'NO DATA';
    band.className = 'badge ' + (score >= 75 ? 'green' : score >= 60 ? 'cyan' : score >= 45 ? 'amber' : 'red');
    if (byId('fundamentalCoverage')) {
        byId('fundamentalCoverage').innerText = `${Number(f.data_coverage || 0).toFixed(0)}% DATA COVERAGE`;
    }

    const p = f.profitability || {}, g = f.growth || {}, b = f.balance_sheet || {}, c = f.cash_flow || {}, v = f.valuation || {};
    const risk = f.risk_context || {};
    const kpis = [
        ['Score', score, '', score >= 60 ? 'green' : score >= 45 ? 'amber' : 'red'],
        ['Valuation gap', Number(v.gap_median || 0) * 100, '%', Number(v.gap_median || 0) >= 0 ? 'green' : 'red'],
        ['ROIC', Number(p.roic || 0) * 100, '%', Number(p.roic || 0) >= 0.10 ? 'green' : 'amber'],
        ['ROE', Number(p.roe || 0) * 100, '%', Number(p.roe || 0) >= 0.12 ? 'green' : 'amber'],
        ['Revenue CAGR', Number(g.revenue_cagr || 0) * 100, '%', Number(g.revenue_cagr || 0) >= 0 ? 'green' : 'red'],
        ['Debt / Equity', Number(b.debt_to_equity || 0), 'x', Number(b.debt_to_equity || 0) <= 1 ? 'green' : 'amber'],
        ['FCF positive', `${c.positive_fcf_years || 0}/${c.available_fcf_years || 0}`, '', 'cyan'],
        ['Beta', Number(risk.beta || 0), 'x', Number(risk.beta || 0) <= 1.4 ? 'cyan' : 'amber']
    ];
    const kpiBox = byId('fundamentalKpis');
    if (kpiBox) {
        kpiBox.innerHTML = kpis.map(k => `<div class="scenario"><div class="text-[10px] muted">${k[0]}</div><strong class="block mt-2 mono ${k[3]}">${typeof k[1] === 'number' ? k[1].toFixed(2) + k[2] : k[1]}</strong></div>`).join('');
    }

    const rows = [];
    const addRow = (cat, metric, value, note) => rows.push(`<tr><td>${cat}</td><td>${metric}</td><td class="mono">${value}</td><td class="muted">${note}</td></tr>`);
    addRow('Valuation','DCF',v.dcf==null?'N/A':'$'+Number(v.dcf).toFixed(2),'Intrinsic-value cross-check from DCF.');
    addRow('Valuation','Peer target',v.peer_target==null?'N/A':'$'+Number(v.peer_target).toFixed(2),'Sector multiple cross-check.');
    addRow('Valuation','ML forecast',v.ml_target==null?'N/A':'$'+Number(v.ml_target).toFixed(2),'Model forecast cross-check.');
    addRow('Valuation','Analyst target',v.analyst_target==null?'N/A':'$'+Number(v.analyst_target).toFixed(2),'Provider consensus target when available.');
    addRow('Valuation','Composite',v.composite_target==null?'N/A':'$'+Number(v.composite_target).toFixed(2),'Platform composite valuation.');
    addRow('Valuation','Forward P/E',v.forward_pe==null?'N/A':Number(v.forward_pe).toFixed(2)+'x',v.peer_forward_pe_median ? `Peer median ${Number(v.peer_forward_pe_median).toFixed(2)}x.` : 'Peer multiple unavailable.');
    addRow('Profitability','ROIC',p.roic==null?'N/A':(Number(p.roic)*100).toFixed(2)+'%','Capital efficiency.');
    addRow('Profitability','ROE',p.roe==null?'N/A':(Number(p.roe)*100).toFixed(2)+'%','Return on equity.');
    addRow('Profitability','Net margin',p.net_margin==null?'N/A':(Number(p.net_margin)*100).toFixed(2)+'%','Latest net profitability.');
    addRow('Growth','Revenue CAGR',g.revenue_cagr==null?'N/A':(Number(g.revenue_cagr)*100).toFixed(2)+'%','CAGR across available annual history.');
    addRow('Growth','Net income CAGR',g.net_income_cagr==null?'N/A':(Number(g.net_income_cagr)*100).toFixed(2)+'%','CAGR across available annual history.');
    addRow('Balance Sheet','Current ratio',b.current_ratio==null?'N/A':Number(b.current_ratio).toFixed(2)+'x','Short-term liquidity.');
    addRow('Balance Sheet','Debt / Equity',b.debt_to_equity==null?'N/A':Number(b.debt_to_equity).toFixed(2)+'x','Financial leverage.');
    addRow('Balance Sheet','Interest coverage',b.interest_coverage==null?'N/A':Number(b.interest_coverage).toFixed(2)+'x','Ability to cover interest expense.');
    addRow('Cash Flow','FCF positive years',`${c.positive_fcf_years||0}/${c.available_fcf_years||0}`,'Consistency of positive free cash flow.');
    addRow('Risk context','5Y Sharpe',data.risk_metrics?.sharpe_ratio==null?'N/A':Number(data.risk_metrics.sharpe_ratio).toFixed(2),'Risk-adjusted historical return context; not part of the fundamental score.');
    addRow('Risk context','Information Ratio',data.risk_metrics?.information_ratio==null?'N/A':Number(data.risk_metrics.information_ratio).toFixed(3),'Annualized active return divided by tracking error vs S&P 500; not part of the fundamental score.');
    addRow('Risk context','Jensen Alpha',data.risk_metrics?.jensen_alpha==null?'N/A':(Number(data.risk_metrics.jensen_alpha)*100).toFixed(2)+'%','Annualized return above/below CAPM required return; not part of the fundamental score.');
    addRow('Financial health','Altman Z-Score',data.altman?.score==null?'N/A':Number(data.altman.score).toFixed(2),data.altman?.zone || data.altman?.status || 'N/A');
    const evidenceBody = byId('fundamentalEvidenceTable')?.querySelector('tbody');
    if (evidenceBody) evidenceBody.innerHTML = rows.join('');

    const notes = byId('fundamentalNotes');
    if (notes) {
        const section = (title, items, cls) => `<div class="scenario"><div class="text-[10px] ${cls} font-semibold uppercase tracking-wider">${title}</div><div class="mt-2">${(items||[]).length ? items.map(x=>`<div>• ${x}</div>`).join('') : '<div class="muted">No specific item detected from available data.</div>'}</div></div>`;
        notes.innerHTML = section('Strengths', f.strengths, 'green') + section('Risks', f.risks, 'red') + section('Watch items', f.watch_items, 'amber');
    }

    // Render charts after the section is visible. requestAnimationFrame is important
    // when the function is called from the initial dashboard load.
    const chartWork = () => {
        const factorCanvas = byId('fundamentalFactorChart');
        const valuationCanvas = byId('fundamentalValuationChart');
        const trendCanvas = byId('fundamentalTrendChart');
        if (!factorCanvas || !valuationCanvas || !trendCanvas || typeof Chart === 'undefined') return;

        const factorLabels = Object.keys(f.factor_scores || {});
        const factorData = factorLabels.map(k => Number(f.factor_scores[k] || 0));
        if (fundamentalFactorChartInstance) fundamentalFactorChartInstance.destroy();
        fundamentalFactorChartInstance = new Chart(factorCanvas, {
            type:'radar',
            data:{labels:factorLabels,datasets:[{label:'Fundamental score',data:factorData,fill:true,borderColor:'#22d3ee',backgroundColor:'rgba(34,211,238,.12)',pointBackgroundColor:'#22d3ee'}]},
            options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8492a3'}}},scales:{r:{min:0,max:100,grid:{color:'#233241'},angleLines:{color:'#233241'},ticks:{color:'#64748b',backdropColor:'transparent'}}}}
        });

        const cross = (f.valuation_cross_check || []).filter(x => x && x.target != null && Number.isFinite(Number(x.target)));
        const valLabels = ['Current', ...cross.map(x=>x.method)];
        const valData = [Number(data.current_price || 0), ...cross.map(x=>Number(x.target))];
        if (fundamentalValuationChartInstance) fundamentalValuationChartInstance.destroy();
        fundamentalValuationChartInstance = new Chart(valuationCanvas, {
            type:'bar',
            data:{labels:valLabels,datasets:[{label:'Price / Target',data:valData,borderWidth:1}]},
            options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8492a3'}}},scales:{x:{grid:{color:'#17212d'},ticks:{color:'#64748b'}},y:{grid:{color:'#17212d'},ticks:{color:'#64748b'}}}}
        });

        const t = f.trends || {};
        const tl = (t.years || []).map(String);
        const align = (arr, n) => { const a = Array.isArray(arr) ? arr.slice() : []; while (a.length < n) a.unshift(null); return a.slice(-n).map(x => x == null ? null : Number(x)*100); };
        const n = tl.length;
        if (fundamentalTrendChartInstance) fundamentalTrendChartInstance.destroy();
        fundamentalTrendChartInstance = new Chart(trendCanvas, {
            type:'line',
            data:{labels:tl,datasets:[
                {label:'Revenue YoY',data:align(t.revenue_yoy,n),borderColor:'#19d58b',pointRadius:2,tension:.2,spanGaps:true},
                {label:'Net Income YoY',data:align(t.net_income_yoy,n),borderColor:'#22d3ee',pointRadius:2,tension:.2,spanGaps:true},
                {label:'FCF YoY',data:align(t.fcf_yoy,n),borderColor:'#fbbf24',pointRadius:2,tension:.2,spanGaps:true},
                {label:'Debt YoY',data:align(t.debt_yoy,n),borderColor:'#fb7185',pointRadius:2,tension:.2,spanGaps:true}
            ]},
            options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8492a3'}}},scales:{x:{grid:{color:'#17212d'},ticks:{color:'#64748b'}},y:{grid:{color:'#17212d'},ticks:{color:'#64748b',callback:v=>v+'%'}}}}
        });
    };

    if (recreateCharts || !fundamentalFactorChartInstance || !fundamentalValuationChartInstance || !fundamentalTrendChartInstance) {
        requestAnimationFrame(chartWork);
    } else {
        chartWork();
    }
}

function analysisPriceText(data, value, digits=4){
    if(value==null || !Number.isFinite(Number(value))) return 'N/A';
    const n=Number(value);
    const unit=String(data?.quote_unit||data?.currency||'USD');
    if(unit==='USD') return '$'+n.toLocaleString(undefined,{minimumFractionDigits:digits,maximumFractionDigits:digits});
    return n.toLocaleString(undefined,{minimumFractionDigits:digits,maximumFractionDigits:digits})+' '+unit;
}

function setAssetAwareLabels(data){
    const asset=String(data?.asset_class||'Equity');
    const marketMode=String(data?.analysis_mode||'ISSUER')==='MARKET_ASSET';
    const badge=document.getElementById('assetClassBadge');
    if(badge) badge.innerText=asset.toUpperCase()+' · '+(marketMode?'MARKET ASSET':'ISSUER');
    const fv=document.querySelector('#fairValue')?.previousElementSibling;
    if(fv) fv.innerText=marketMode?'MODEL FAIR LEVEL':'DCF FAIR VALUE';
    const up=document.querySelector('#dcfUpside');
    if(up) up.dataset.marketMode=marketMode?'1':'0';
    const mlLabel=document.querySelector('#mlPrice')?.previousElementSibling;
    if(mlLabel) mlLabel.innerText=marketMode?'ML 20D MODEL LEVEL':'XGBoost 20D Price';
    const priceTitle=document.getElementById('currentPrice')?.previousElementSibling;
    if(priceTitle) priceTitle.innerText=marketMode?'Current quote':'Current market price';
}

function renderDashboardData(data) {
    document.getElementById('companyName').innerText = data.company_name;
    document.getElementById('tickerSymbol').innerText = data.ticker;
    setAssetAwareLabels(data);
    document.getElementById('currencyBadge').innerText = (data.quote_unit || data.currency) + ' · ' + String(data.sector || data.asset_class || '').toUpperCase();
    document.getElementById('currentPrice').innerText = analysisPriceText(data, data.current_price, data.current_price < 10 ? 4 : 2);
    document.getElementById('peerSubtitle').innerText = data.analysis_mode === 'MARKET_ASSET' ? 'ASSET-CLASS MODE · EQUITY FUNDAMENTALS / PEER MULTIPLES NOT APPLIED' : 'SECTOR: ' + data.sector.toUpperCase() + ' · DIRECT INDUSTRY COMPS';
    updateExportBadge();
    const portInput = document.getElementById('portfolioTickers');
    if (portInput && !portInput.dataset.userEdited) { portInput.value = data.ticker; }

    document.getElementById('fairValue').innerText = analysisPriceText(data, data.fair_value_per_share, data.current_price < 10 ? 4 : 2);
    document.getElementById('dcfTargetCopy').innerText = analysisPriceText(data, data.fair_value_per_share, data.current_price < 10 ? 4 : 2);
    const dcfElem = document.getElementById('dcfUpside');
    dcfElem.innerText = (data.dcf_implied_upside_percent >= 0 ? '+' : '') + data.dcf_implied_upside_percent.toFixed(2) + '% Implied';
    dcfElem.className = 'text-xs mt-2 ' + (data.dcf_implied_upside_percent >= 0 ? 'green' : 'red');

    if (document.getElementById('mlPrice')) {
        document.getElementById('mlPrice').innerText = analysisPriceText(data, data.ml_forecast.predicted_target_price, data.current_price < 10 ? 4 : 2);
    }
    document.getElementById('mlForecastTarget').innerText = analysisPriceText(data, data.ml_forecast.predicted_target_price, data.current_price < 10 ? 4 : 2);

    const rm = data.risk_metrics;
    document.getElementById('capmReturn').innerText = (rm.capm_expected_return * 100).toFixed(2) + '%';
    document.getElementById('rfRate').innerText = (rm.risk_free_rate * 100).toFixed(2) + '%';
    document.getElementById('rmReturn').innerText = (rm.market_return_5y * 100).toFixed(2) + '%';
    document.getElementById('capmBeta').innerText = rm.beta.toFixed(2);
    document.getElementById('capmResult').innerText = (rm.capm_expected_return * 100).toFixed(2) + '%';
    document.getElementById('var5y').innerText = rm.variance_5y.toFixed(4);
    document.getElementById('sd5y').innerText = (rm.std_dev_5y * 100).toFixed(2) + '%';
    document.getElementById('sharpeRatio').innerText = rm.sharpe_ratio.toFixed(2);
    document.getElementById('sharpeVal').innerText = rm.sharpe_ratio.toFixed(2);
    document.getElementById('mSquaredVal').innerText = (rm.m_squared * 100).toFixed(2) + '%';
    document.getElementById('treynorVal').innerText = (rm.treynor_ratio * 100).toFixed(2) + '%';
    document.getElementById('dist52w').innerText = (data.technicals.distance_from_52w_high).toFixed(2) + '%';

    const fr = data.financial_ratios;
    document.getElementById('roicVal').innerText = (fr.roic * 100).toFixed(2) + '%';
    document.getElementById('currentRatioVal').innerText = fr.current_ratio.toFixed(2) + 'x';
    document.getElementById('quickRatioVal').innerText = fr.quick_ratio.toFixed(2) + 'x';
    document.getElementById('debtEquityVal').innerText = fr.debt_to_equity.toFixed(2) + 'x';
    document.getElementById('interestCoverageVal').innerText = fr.interest_coverage.toFixed(2) + 'x';
    document.getElementById('fcfYieldVal').innerText = (fr.fcf_yield * 100).toFixed(2) + '%';
    document.getElementById('roaVal').innerText = (fr.roa * 100).toFixed(2) + '%';
    document.getElementById('roeVal').innerText = (fr.roe * 100).toFixed(2) + '%';
    document.getElementById('grossMarginVal').innerText = (fr.gross_margin * 100).toFixed(2) + '%';
    document.getElementById('operatingMarginVal').innerText = (fr.operating_margin * 100).toFixed(2) + '%';
    document.getElementById('netMarginVal').innerText = (fr.net_margin * 100).toFixed(2) + '%';
    document.getElementById('cashRatioVal').innerText = fr.cash_ratio > 0 ? fr.cash_ratio.toFixed(2) + 'x' : 'N/A';
    document.getElementById('assetTurnoverVal').innerText = fr.asset_turnover > 0 ? fr.asset_turnover.toFixed(2) + 'x' : 'N/A';
    document.getElementById('inventoryTurnoverVal').innerText = fr.inventory_turnover > 0 ? fr.inventory_turnover.toFixed(2) + 'x' : 'N/A';
    document.getElementById('receivablesTurnoverVal').innerText = fr.receivables_turnover > 0 ? fr.receivables_turnover.toFixed(2) + 'x' : 'N/A';

    const az = data.altman || {};
    const azScore = az.score == null ? null : Number(az.score);
    const azText = azScore == null || !Number.isFinite(azScore) ? 'N/A' : azScore.toFixed(2);
    const azZone = az.zone && az.zone !== 'N/A' ? az.zone : (az.status || 'N/A');
    const azEl = document.getElementById('altmanZVal');
    const azZoneEl = document.getElementById('altmanZoneVal');
    const azKpi = document.getElementById('altmanKpi');
    const azKpiNote = document.getElementById('altmanKpiNote');
    if (azEl) azEl.innerText = azText;
    if (azZoneEl) azZoneEl.innerText = azZone;
    if (azKpi) azKpi.innerText = azText;
    if (azKpiNote) azKpiNote.innerText = azZone;

    const avgUpside = (data.dcf_implied_upside_percent + data.ml_forecast.ml_implied_upside) / 2;
    const signalElem = document.getElementById('signal');
    if (avgUpside > 15) { signalElem.innerText = 'STRONG BUY'; signalElem.className = 'value green'; }
    else if (avgUpside > 5) { signalElem.innerText = 'BUY'; signalElem.className = 'value green'; }
    else if (avgUpside > -5) { signalElem.innerText = 'HOLD'; signalElem.className = 'value amber'; }
    else { signalElem.innerText = 'SELL'; signalElem.className = 'value red'; }

    if (data.trade_setup) {
        const ts = data.trade_setup;
        const longEl=document.getElementById('tradeLongEntry');
        const shortEl=document.getElementById('tradeShortEntry');
        if(longEl) longEl.innerText='$'+Number(ts.long_suitable_entry ?? data.current_price).toFixed(2);
        if(shortEl) shortEl.innerText='$'+Number(ts.short_suitable_entry ?? data.current_price).toFixed(2);
        const dir=document.getElementById('tradeDirection')?.value || 'LONG';
        const sel=document.getElementById('tradeSuitableEntry');
        if(sel) sel.innerText='$'+Number(dir==='LONG' ? (ts.long_suitable_entry ?? data.current_price) : (ts.short_suitable_entry ?? data.current_price)).toFixed(2);
    }
    if (document.getElementById('derivTicker')) {
        document.getElementById('derivTicker').innerText=data.ticker || '--';
        document.getElementById('derivSpotOut').innerText='$'+Number(data.current_price).toFixed(2);
        document.getElementById('derivVolOut').innerText=(Number(data.risk_metrics?.std_dev_5y||0)*100).toFixed(2)+'%';
        document.getElementById('derivBetaOut').innerText=Number(data.risk_metrics?.beta||0).toFixed(3);
        document.getElementById('derivLongEntry').innerText='$'+Number(data.trade_setup?.long_suitable_entry ?? data.current_price).toFixed(2);
        document.getElementById('derivShortEntry').innerText='$'+Number(data.trade_setup?.short_suitable_entry ?? data.current_price).toFixed(2);
    }
    calculateFuturesSimulation();

    const fm = data.ml_forecast.feature_metrics;
    document.getElementById('peRatio').innerText = fm.pe_ratio ? fm.pe_ratio.toFixed(2) + 'x' : 'N/A';
    document.getElementById('margin').innerText = (fm.profit_margins * 100).toFixed(1) + '%';
    document.getElementById('roe').innerText = (fm.return_on_equity * 100).toFixed(1) + '%';
    document.getElementById('beta').innerText = fm.beta.toFixed(2);

    if (data.composite_target_price) {
        document.getElementById('compositeTarget').innerText = '$' + data.composite_target_price.toFixed(2);
        document.getElementById('compositeUpside').innerText = (data.composite_implied_upside_percent >= 0 ? '+' : '') + data.composite_implied_upside_percent.toFixed(2) + '% Implied Upside';
        document.getElementById('peerTarget').innerText = data.peer_target_price ? '$' + data.peer_target_price.toFixed(2) : 'N/A';
        document.getElementById('analystTarget').innerText = data.analyst_target_price ? '$' + data.analyst_target_price.toFixed(2) : 'N/A';
        document.getElementById('peerMethod').innerText = data.methodology || 'MULTIVARIATE';
    }

    if (data.technicals) {
        document.getElementById('rsiVal').innerText = data.technicals.rsi14;
        document.getElementById('rsiState').innerText = data.technicals.rsi14 > 70 ? 'Overbought' : (data.technicals.rsi14 < 30 ? 'Oversold' : 'Neutral');
        document.getElementById('macdVal').innerText = data.technicals.macd;
        document.getElementById('macdState').innerText = data.technicals.trend.toUpperCase();
        document.getElementById('volumeRatio').innerText = data.technicals.volume_ratio + 'x';
    }

    if (data.peers && data.peers.length) {
        const tbody = document.querySelector('#peerTable tbody');
        tbody.innerHTML = data.peers.map((p, idx) => `
            <tr ${idx === 0 ? 'class="bg-emerald-950/20 font-bold"' : ''}>
                <td>${p.name} <span class="muted font-mono">(${p.ticker})</span> ${idx === 0 ? '<span class="text-[9px] text-emerald-400 ml-1">(Target)</span>' : ''}</td>
                <td>$${p.price.toFixed(2)}</td>
                <td>${p.forward_pe > 0 ? p.forward_pe.toFixed(2) + 'x' : 'N/A'}</td>
                <td>${p.revenue_growth.toFixed(1)}%</td>
                <td>${p.profit_margin.toFixed(1)}%</td>
                <td>${p.beta.toFixed(2)}</td>
            </tr>
        `).join('');
    }

    document.getElementById('baseVal').innerText = '$' + data.fair_value_per_share.toFixed(2);
    document.getElementById('bearVal').innerText = '$' + (data.fair_value_per_share * 0.82).toFixed(2);
    document.getElementById('bullVal').innerText = '$' + (data.fair_value_per_share * 1.24).toFixed(2);

    renderMainLineChart(data);
    renderBarChart(data);
    renderTechnicalChart(data);
    renderPeerCharts(data);
    renderRiskAnalytics(data);
    renderRatioHistory(data);
        renderFinancialStatements(data);
    renderFundamentalAnalysis(data);
    renderNewsCalendar(data);
    renderCfdScenario(data);
    recalculateTrade();
}

function smaSeries(values, n) {
    const out = new Array(values.length).fill(null);
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
        sum += Number(values[i]) || 0;
        if (i >= n) sum -= Number(values[i - n]) || 0;
        if (i >= n - 1) out[i] = +(sum / n).toFixed(6);
    }
    return out;
}

function renderMainLineChart(data) {
    const cData = data.chart_data;
    const ctxLine = document.getElementById('lineChart').getContext('2d');
    if (lineChartInstance) lineChartInstance.destroy();

    let sliceCount = cData.dates.length;
    if (selectedTimeframe === '1M') sliceCount = 21;
    else if (selectedTimeframe === '3M') sliceCount = 63;
    else if (selectedTimeframe === '6M') sliceCount = 126;
    else if (selectedTimeframe === '1Y') sliceCount = 252;

    const slicedDates = cData.dates.slice(-sliceCount);
    const slicedHist = cData.historical_close.slice(-sliceCount);
    const sma20Full = (cData.sma_20 && cData.sma_20.length) ? cData.sma_20 : smaSeries(cData.historical_close, 20);
    const sma50Full = (cData.sma_50 && cData.sma_50.length) ? cData.sma_50 : smaSeries(cData.historical_close, 50);
    const slicedSma20 = sma20Full.slice(-sliceCount);
    const slicedSma50 = sma50Full.slice(-sliceCount);

    const labels = [...slicedDates, ...cData.ml_forecast_dates.slice(1)];
    const hist = [...slicedHist, ...Array(cData.ml_forecast_dates.length - 1).fill(null)];
    const sma20 = [...slicedSma20, ...Array(cData.ml_forecast_dates.length - 1).fill(null)];
    const sma50 = [...slicedSma50, ...Array(cData.ml_forecast_dates.length - 1).fill(null)];
    const forecast = [...Array(slicedDates.length - 1).fill(null), ...cData.ml_forecast_prices];

    lineChartInstance = new Chart(ctxLine, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                { label: 'Close Price', data: hist, borderColor: '#19d58b', borderWidth: 2, pointRadius: 0 },
                { label: 'SMA 20', data: sma20, borderColor: '#64748b', borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0 },
                { label: 'SMA 50', data: sma50, borderColor: '#fbbf24', borderWidth: 1.5, borderDash: [2, 2], pointRadius: 0 },
                { label: 'XGBoost 20D Projection', data: forecast, borderColor: '#22d3ee', borderWidth: 2.5, borderDash: [6, 6], pointRadius: 0 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { labels: { color: '#8492a3', font: { family: 'JetBrains Mono', size: 11 } } } },
            scales: {
                x: { grid: { color: '#17212d' }, ticks: { color: '#64748b', maxTicksLimit: 12 } },
                y: { grid: { color: '#17212d' }, ticks: { color: '#64748b' } }
            }
        }
    });
}

function renderBarChart(data) {
    const ctxBar = document.getElementById('barChart').getContext('2d');
    if (barChartInstance) barChartInstance.destroy();

    barChartInstance = new Chart(ctxBar, {
        type: 'bar',
        data: {
            labels: ['Current Market', 'DCF Fair Value', 'XGBoost 20D', 'Composite Target'],
            datasets: [{
                data: [
                    data.current_price,
                    data.fair_value_per_share,
                    data.ml_forecast.predicted_target_price,
                    data.composite_target_price || data.fair_value_per_share
                ],
                backgroundColor: ['#475569', '#19d58b', '#22d3ee', '#fbbf24'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#8492a3', font: { family: 'JetBrains Mono', size: 10 } } },
                y: { grid: { color: '#17212d' }, ticks: { color: '#64748b' } }
            }
        }
    });
}

function renderTechnicalChart(data) {
    if (!data.technicals || !data.technicals.rsi_series) return;
    const ctxTech = document.getElementById('technicalChart').getContext('2d');
    if (techChartInstance) techChartInstance.destroy();

    techChartInstance = new Chart(ctxTech, {
        type: 'line',
        data: {
            labels: Array.from({length: data.technicals.rsi_series.length}, (_, i) => i + 1),
            datasets: [{
                label: 'RSI (14)', data: data.technicals.rsi_series,
                borderColor: '#fbbf24', borderWidth: 1.5, pointRadius: 0
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { min: 0, max: 100, grid: { color: '#17212d' }, ticks: { color: '#64748b', stepSize: 25 } }
            }
        }
    });
}

function renderPeerCharts(data) {
    if (!data.peers || !data.peers.length) return;
    const labels = data.peers.map(p => p.ticker);
    const peValues = data.peers.map(p => p.forward_pe > 0 ? p.forward_pe : 0);
    const marginValues = data.peers.map(p => p.profit_margin);
    const backgroundColors = data.peers.map((p, i) => i === 0 ? '#19d58b' : '#334155');

    // Forward P/E Chart
    const ctxPe = document.getElementById('peerPeChart').getContext('2d');
    if (peerPeChartInstance) peerPeChartInstance.destroy();
    peerPeChartInstance = new Chart(ctxPe, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{ label: 'Forward P/E', data: peValues, backgroundColor: backgroundColors, borderRadius: 4 }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#8492a3', font: { family: 'JetBrains Mono', size: 10 } } },
                y: { grid: { color: '#17212d' }, ticks: { color: '#64748b' } }
            }
        }
    });

    // Profit Margin Chart
    const ctxMargin = document.getElementById('peerMarginChart').getContext('2d');
    if (peerMarginChartInstance) peerMarginChartInstance.destroy();
    peerMarginChartInstance = new Chart(ctxMargin, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{ label: 'Profit Margin (%)', data: marginValues, backgroundColor: backgroundColors, borderRadius: 4 }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#8492a3', font: { family: 'JetBrains Mono', size: 10 } } },
                y: { grid: { color: '#17212d' }, ticks: { color: '#64748b' } }
            }
        }
    });
}

const portInputInit = document.getElementById('portfolioTickers');
if (portInputInit) portInputInit.addEventListener('input', ()=>{portInputInit.dataset.userEdited='1';});
setupMovableSidebar();
loadInstrumentBoard('Equities');
// (setDerivativeMode was called here but never defined -- the ReferenceError stopped
// the rest of this script, so the dashboard stayed on "Loading..." forever.)
fetchValuation();
// Refresh every 5 minutes, only while the tab is visible; the server caches for 15.
setInterval(() => { if (!document.hidden) fetchValuation({silent: true}); }, 300000);
</script>
</body>
</html>
"""




@app.get("/portfolio-lab", response_class=HTMLResponse)
def portfolio_lab():
    """Local-first portfolio scenario dashboard; live ticker reads use existing optimizer API."""
    return HTMLResponse(PORTFOLIO_LAB_HTML)

PORTFOLIO_LAB_HTML = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Finmodel · Portfolio Lab</title><script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.8/dist/chart.umd.min.js"></script><style>\n:root{font-family:Inter,system-ui,Arial,sans-serif;color:#e8effb;background:#090f1e}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 65% -10%,#18365b 0%,#090f1e 53%)}.wrap{max-width:1450px;margin:auto;padding:28px}header{display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:28px}a{color:#9ccaff;text-decoration:none}.tag{font-size:12px;border:1px solid #314769;border-radius:20px;padding:8px 13px;color:#a8bfde}h1{font-size:clamp(24px,3vw,38px);letter-spacing:-1px;margin:8px 0}h2{font-size:16px;margin:0 0 15px;color:#dfecff}.muted,.small{color:#92a4be}.small{font-size:12px}.layout{display:grid;grid-template-columns:340px minmax(0,1fr);gap:18px}.panel,.metric{border:1px solid #263752;border-radius:18px;background:#121e32e8;padding:21px;min-width:0}.side{display:flex;flex-direction:column;gap:18px}.main{min-width:0;display:flex;flex-direction:column;gap:18px}.fields{display:grid;grid-template-columns:1fr 1fr;gap:11px}label{display:block;font-size:12px;color:#bacbe3;margin:0 0 6px}input,select,button{font:inherit;border-radius:10px}input,select{width:100%;border:1px solid #3a4e6a;background:#0c1627;color:#ecf4ff;padding:11px}input:focus,select:focus{outline:2px solid #5c9fe5}button{border:0;padding:12px 15px;background:#4ba5ff;color:#081323;font-weight:700;cursor:pointer}button.secondary{color:#b6d9ff;background:#223a5a}.btns{display:flex;gap:8px;flex-wrap:wrap;margin-top:13px}.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.metric{padding:18px}.metric .num{font-size:clamp(21px,2vw,30px);font-weight:730;letter-spacing:-.8px;margin:8px 0}.charts{display:grid;grid-template-columns:1fr 1fr;gap:18px}.chartbox{height:290px;position:relative}.weightrow{display:grid;grid-template-columns:62px minmax(0,1fr) 76px;align-items:center;gap:8px;margin:14px 0}.weightrow input[type=range]{accent-color:#60b0ff;padding:0}.weightrow input[type=number]{padding:8px}.weightrow strong{font-size:12px}#notice{min-height:42px;margin:10px 0 0;font-size:13px;line-height:1.5;color:#a8d4ff}#notice.error{color:#ffacac}table{width:100%;border-collapse:collapse;text-align:right;font-size:12px;white-space:nowrap}th,td{padding:12px 10px;border-bottom:1px solid #283750}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}.kicker{color:#74baff;font-size:12px;letter-spacing:2px}footer{font-size:12px;color:#9aacc4;margin:25px 0}.pill{color:#8ee6ba}.inline{display:flex;align-items:center;justify-content:space-between;gap:10px}.sum{font-weight:bold}body.loading button{opacity:.65;pointer-events:none}@media(max-width:980px){.layout{grid-template-columns:1fr}.side{display:grid;grid-template-columns:1fr 1fr}.metrics{grid-template-columns:repeat(2,1fr)}}@media(max-width:650px){.wrap{padding:15px}.side,.charts,.metrics{grid-template-columns:1fr}.panel{padding:16px}.chartbox{height:245px}}\n</style></head><body><div class="wrap"><header><div><div class="kicker">FINMODEL / PRIVATE WORKSPACE</div><h1>Portfolio Analytics Lab</h1><span class="muted">Explore your own asset mix · Risk / return / CAPM</span></div><div class="tag">LOCAL SESSION · NO ACCOUNT SYNC</div></header><div class="layout"><aside class="side"><section class="panel"><h2>Market universe</h2><label for="tickers">Asset tickers (comma-separated)</label><input id="tickers" value="AAPL,MSFT,NVDA,GOOGL,AMZN" maxlength="180" spellcheck="false"><div class="fields" style="margin-top:13px"><div><label for="lookback">Price history</label><select id="lookback"><option>1y</option><option>2y</option><option>3y</option><option selected>5y</option><option>10y</option></select></div><div><label for="benchmark">Benchmark</label><input id="benchmark" value="^GSPC" maxlength="15"></div></div><div class="btns"><button id="load">Analyze assets</button><button class="secondary" id="reset">Reset</button></div><div id="notice" role="status">Enter ticker symbols and select Analyze assets.</div></section><section class="panel"><div class="inline"><h2>Allocation lab</h2><span class="small" id="weightSum">—</span></div><div id="weights" class="small">Load market data to edit weights.</div><div class="btns"><button class="secondary" id="equal">Equal weights</button><button class="secondary" id="minimum">Min risk</button><button class="secondary" id="tangency">Max Sharpe</button></div><p class="small">Weights are long-only, must total 100%, and are normalized only when you choose a preset. Editing any weight updates the remaining allocation proportionally.</p></section></aside><main class="main"><section class="metrics"><div class="metric"><div class="small">Annualized return</div><div class="num" id="ret">—</div><div class="small">Historical mean estimate</div></div><div class="metric"><div class="small">Volatility</div><div class="num" id="vol">—</div><div class="small">Annualized standard deviation</div></div><div class="metric"><div class="small">Sharpe ratio</div><div class="num" id="sharpe">—</div><div class="small">Excess return per unit risk</div></div><div class="metric"><div class="small">Portfolio beta</div><div class="num" id="beta">—</div><div class="small">Relative to benchmark</div></div><div class="metric"><div class="small">CAPM expected return</div><div class="num" id="capm">—</div><div class="small">Rf + β × market premium</div></div><div class="metric"><div class="small">Portfolio variance</div><div class="num" id="variance">—</div><div class="small">Annualized return variance</div></div></section><section class="charts"><div class="panel"><h2>Allocation breakdown</h2><div class="chartbox"><canvas id="allocationChart"></canvas></div></div><div class="panel"><h2>Risk / return frontier</h2><div class="chartbox"><canvas id="frontierChart"></canvas></div></div><div class="panel"><h2>Security Market Line</h2><div class="chartbox"><canvas id="smlChart"></canvas></div></div><div class="panel"><h2>Asset comparison</h2><div class="chartbox"><canvas id="assetChart"></canvas></div></div></section><section class="panel"><div class="inline"><h2>Asset-level metrics</h2><button class="secondary" id="export">Export CSV</button></div><div class="scroll"><table><thead><tr><th>Ticker</th><th>Weight</th><th>Return</th><th>Volatility</th><th>Beta</th><th>Sharpe</th><th>Return contribution</th></tr></thead><tbody id="assetRows"><tr><td colspan="7">No assets loaded</td></tr></tbody></table></div></section><div class="small" id="period"></div></main></div><footer>Models use historical adjusted daily prices; expected returns are arithmetic daily means × 252, with annual covariance × 252. CAPM is a separate equilibrium estimate. Estimates are not predictions or investment advice. Your symbols are sent to the configured Finmodel market-data service when you analyze; this page does not upload portfolio holdings to a third-party account.</footer></div><script>\nlet data=null,weights=[],charts={};const $=id=>document.getElementById(id),pct=v=>Number.isFinite(v)?(v*100).toFixed(2)+\'%\':\'—\',dec=(v,n=2)=>Number.isFinite(v)?v.toFixed(n):\'—\';\nconst colors=[\'#5eabff\',\'#65e3bb\',\'#fbcb78\',\'#c0a1ff\',\'#fa8f96\',\'#85d7e8\',\'#d3dc8b\',\'#e5a9e2\',\'#a7b6ff\',\'#ffb874\'];\nfunction notice(msg,error=false){$(\'notice\').textContent=msg;$(\'notice\').className=error?\'error\':\'\'}\nfunction calc(){const m=data.model_inputs;const mu=m.annual_returns,C=m.annual_covariance,b=m.betas;const ret=weights.reduce((s,w,i)=>s+w*mu[i],0);const variance=weights.reduce((s,w,i)=>s+w*weights.reduce((z,v,j)=>z+v*C[i][j],0),0);const vol=Math.sqrt(Math.max(0,variance)),beta=weights.reduce((s,w,i)=>s+w*b[i],0);const rf=data.risk_free_rate;return {ret,variance,vol,beta,sharpe:vol>0?(ret-rf)/vol:NaN,capm:rf+beta*(data.market.annual_return-rf)}}\nfunction chart(id,type,datasets,labels=[],opts={}){if(typeof Chart===\'undefined\')return;if(charts[id])charts[id].destroy();charts[id]=new Chart($(id),{type,data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{color:\'#b9cce4\',boxWidth:11}}},scales:type===\'doughnut\'?{}:{x:{ticks:{color:\'#9db0ca\'},grid:{color:\'#263852\'}},y:{ticks:{color:\'#9db0ca\'},grid:{color:\'#263852\'}}},...opts}})}\nfunction redraw(){if(!data)return;const r=calc(),rows=data.model_inputs.tickers;for(const [id,val] of Object.entries({ret:pct(r.ret),vol:pct(r.vol),sharpe:dec(r.sharpe,3),beta:dec(r.beta,3),capm:pct(r.capm),variance:dec(r.variance,5)}))$(id).textContent=val; $(\'weightSum\').textContent=pct(weights.reduce((a,b)=>a+b,0));\nchart(\'allocationChart\',\'doughnut\',[{data:weights.map(w=>w*100),backgroundColor:rows.map((_,i)=>colors[i%colors.length]),borderWidth:0}],rows,{plugins:{legend:{position:\'bottom\',labels:{color:\'#c5d5ea\'}}}});\nchart(\'frontierChart\',\'scatter\',[{label:\'Simulated frontier\',data:(data.efficient_frontier||[]).map(x=>({x:x.volatility*100,y:x.return*100})),borderColor:\'#4a9ded\',backgroundColor:\'#4a9ded\',pointRadius:2,showLine:true},{label:\'Your portfolio\',data:[{x:r.vol*100,y:r.ret*100}],backgroundColor:\'#f6c26c\',pointRadius:8}],[],{scales:{x:{title:{display:true,text:\'Volatility %\',color:\'#c0d0e4\'},ticks:{color:\'#9db0ca\'}},y:{title:{display:true,text:\'Return %\',color:\'#c0d0e4\'},ticks:{color:\'#9db0ca\'}}}});\nchart(\'smlChart\',\'scatter\',[{label:\'CAPM line\',data:data.sml.beta.map((b,i)=>({x:b,y:data.sml.return[i]*100})),borderColor:\'#65e3bb\',showLine:true,pointRadius:0},{label:\'Your portfolio (historical mean)\',data:[{x:r.beta,y:r.ret*100}],backgroundColor:\'#ffc975\',pointRadius:7},{label:\'CAPM estimate\',data:[{x:r.beta,y:r.capm*100}],backgroundColor:\'#8eafff\',pointRadius:7}]);\nchart(\'assetChart\',\'bar\',[{label:\'Return %\',data:data.model_inputs.annual_returns.map(x=>x*100),backgroundColor:\'#5eabff\'},{label:\'Volatility %\',data:data.model_inputs.annual_covariance.map((a,i)=>Math.sqrt(a[i])*100),backgroundColor:\'#65e3bb\'}],rows);\n$(\'assetRows\').innerHTML=rows.map((sym,i)=>`<tr><td>${sym}</td><td>${pct(weights[i])}</td><td>${pct(data.model_inputs.annual_returns[i])}</td><td>${pct(Math.sqrt(data.model_inputs.annual_covariance[i][i]))}</td><td>${dec(data.model_inputs.betas[i],3)}</td><td>${dec((data.model_inputs.annual_returns[i]-data.risk_free_rate)/Math.sqrt(data.model_inputs.annual_covariance[i][i]),3)}</td><td>${pct(weights[i]*data.model_inputs.annual_returns[i])}</td></tr>`).join(\'\');}\nfunction edit(i,value){if(!Number.isFinite(value)||value<0||value>100)return;const target=value/100,others=weights.reduce((s,w,j)=>s+(i===j?0:w),0),remainder=1-target; if(weights.length===1){weights=[1]}else{weights=weights.map((w,j)=>j===i?target:others>1e-12?w/others*remainder:remainder/(weights.length-1))}renderWeights();redraw()}\nfunction renderWeights(){const node=$(\'weights\');node.replaceChildren();data.model_inputs.tickers.forEach((sym,i)=>{const el=document.createElement(\'div\');el.className=\'weightrow\';const name=document.createElement(\'strong\');name.textContent=sym;const range=document.createElement(\'input\');range.type=\'range\';range.min=\'0\';range.max=\'100\';range.step=\'.1\';range.value=weights[i]*100;range.setAttribute(\'aria-label\',sym+\' weight slider\');const num=document.createElement(\'input\');num.type=\'number\';num.min=\'0\';num.max=\'100\';num.step=\'.1\';num.value=(weights[i]*100).toFixed(1);num.setAttribute(\'aria-label\',sym+\' weight percent\');range.oninput=()=>edit(i,Number(range.value));num.onchange=()=>{const v=Number(num.value);if(num.value.trim()===\'\'||!Number.isFinite(v)||v<0||v>100){notice(\'Weights must be between 0% and 100%.\',true);renderWeights();return}edit(i,v)};el.append(name,range,num);node.append(el)})}\nfunction preset(mode){if(!data)return;const symbols=data.model_inputs.tickers;weights=mode===\'equal\'?symbols.map(()=>1/symbols.length):symbols.map(sym=>{const row=data.rows.find(r=>r.ticker===sym);return mode===\'minimum\'?row.min_risk_weight:row.tangency_weight});const total=weights.reduce((a,b)=>a+b,0);weights=weights.map(x=>x/total);renderWeights();redraw()}\nasync function load(){const syms=$(\'tickers\').value.toUpperCase().split(\',\').map(x=>x.trim()).filter(Boolean);if(!syms.length||syms.length>20||new Set(syms).size!==syms.length||syms.some(s=>!/^\\^?[A-Z0-9.\\-=]{1,15}$/.test(s))){notice(\'Enter 1–20 unique, valid ticker symbols.\',true);return}const bench=$(\'benchmark\').value.trim().toUpperCase();if(!/^\\^?[A-Z0-9.\\-=]{1,15}$/.test(bench)){notice(\'Enter a valid benchmark ticker.\',true);return}notice(\'Fetching market data…\');document.body.classList.add(\'loading\');try{const q=new URLSearchParams({tickers:syms.join(\',\'),method:\'COMPARE\',lookback:$(\'lookback\').value,benchmark:bench});const res=await fetch(\'/api/portfolio/optimize?\'+q);const out=await res.json();if(!res.ok)throw new Error(out.detail||\'Market data unavailable\');if(!out.model_inputs)throw new Error(\'Server does not support custom weights. Use the updated Finmodel server.\');data=out;weights=data.model_inputs.tickers.map(()=>1/data.model_inputs.tickers.length);renderWeights();redraw();$(\'period\').textContent=`Sample: ${out.data_start} – ${out.data_end} · Benchmark: ${out.benchmark} · Risk-free: ${pct(out.risk_free_rate)} · ${out.assets_used} valid assets`;notice(out.failed_assets?.length?\'Excluded: \'+out.failed_assets.map(x=>x.ticker+\' (\'+x.reason+\')\').join(\', \'):\'Analysis complete. Change weights to explore scenarios.\',false)}catch(err){notice(err.message||String(err),true)}finally{document.body.classList.remove(\'loading\')}}\n$(\'load\').onclick=load;$(\'reset\').onclick=()=>{data=null;weights=[];$(\'tickers\').value=\'AAPL,MSFT,NVDA,GOOGL,AMZN\';$(\'weights\').textContent=\'Load market data to edit weights.\';notice(\'Reset. Analyze assets to begin.\')};$(\'equal\').onclick=()=>preset(\'equal\');$(\'minimum\').onclick=()=>preset(\'minimum\');$(\'tangency\').onclick=()=>preset(\'tangency\');$(\'export\').onclick=()=>{if(!data){notice(\'Analyze assets before exporting.\',true);return}const rows=[[\'Ticker\',\'Weight\',\'AnnualReturn\',\'Volatility\',\'Beta\'],...data.model_inputs.tickers.map((x,i)=>[x,weights[i],data.model_inputs.annual_returns[i],Math.sqrt(data.model_inputs.annual_covariance[i][i]),data.model_inputs.betas[i]])];const blob=new Blob([rows.map(r=>r.join(\',\')).join(\'\\n\')],{type:\'text/csv\'});const a=document.createElement(\'a\');a.href=URL.createObjectURL(blob);a.download=\'finmodel-portfolio.csv\';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};\n</script></body></html>'

# ------------------------------------------------------------------------------
# 6. LOCAL SERVER EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # FIX: was "Platform:app" -- only correct if the file is literally named
    # Platform.py. Save this file as main.py (as instructed below) and this
    # matches; if you rename the file, change the string before the colon
    # to match your filename.
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=port, reload=False)
