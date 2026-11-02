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
    version="17.1.0"
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
        rows = calendar_rows_for_pipeline()
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


# ------------------------------------------------------------------------------
# PORTFOLIO CUSTOMIZER: user constraints + choice of objective
# ------------------------------------------------------------------------------
CONSTRAINED_OBJECTIVES = {
    'MIN_RISK': 'Minimum variance within your limits',
    'MAX_SHARPE': 'Maximum Sharpe ratio within your limits',
    'TARGET_RETURN': 'Lowest risk that reaches your target return',
    'TARGET_VOL': 'Highest return that stays within your volatility cap',
    'UTILITY': 'Mean-variance utility: return - (risk aversion / 2) x variance',
    'RISK_PARITY': 'Equal risk contribution from every asset',
    'MAX_DIVERSIFICATION': 'Maximum diversification ratio (weighted vol / portfolio vol)',
}


def _pct_or_none(value) -> float | None:
    """Accept 12, '12', '12%' (percent) and return a fraction; blank -> None."""
    if value is None:
        return None
    s = str(value).strip().replace('%', '')
    if s == '' or s.lower() in {'none', 'null', 'nan'}:
        return None
    v = float(s)
    if not np.isfinite(v):
        raise ValueError('Constraint values must be finite numbers.')
    return v / 100.0


def _parse_weight_bounds(spec: str, symbols: list[str], global_min: float | None,
                         global_max: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-asset weight limits in percent.

    spec: 'AAPL:5-30,MSFT:0-40,NVDA:-25' ('-25' = no minimum, max 25; '10-' = min 10).
    Global min/max apply to every asset not listed in spec.
    """
    n = len(symbols)
    lo = np.full(n, 0.0 if global_min is None else global_min, dtype=float)
    hi = np.full(n, 1.0 if global_max is None else global_max, dtype=float)
    idx = {s: i for i, s in enumerate(symbols)}
    for entry in (spec or '').replace(';', ',').split(','):
        entry = entry.strip()
        if not entry:
            continue
        if ':' not in entry:
            raise ValueError('Weight limits look like AAPL:5-30 (min-max in %).')
        sym, rng = entry.split(':', 1)
        sym = sym.strip().upper()
        if sym not in idx:
            continue                      # asset dropped for missing history: ignore its limit
        rng = rng.strip()
        if '-' in rng:
            a, b = rng.split('-', 1)
        else:
            a, b = rng, rng               # a single number pins the weight
        a_v = _pct_or_none(a)
        b_v = _pct_or_none(b)
        i = idx[sym]
        if a_v is not None:
            lo[i] = a_v
        if b_v is not None:
            hi[i] = b_v
    if np.any(lo < -1e-12) or np.any(hi > 1 + 1e-12):
        raise ValueError('Weight limits must be between 0% and 100% (long-only).')
    lo = np.clip(lo, 0.0, 1.0); hi = np.clip(hi, 0.0, 1.0)
    bad = [symbols[i] for i in range(n) if lo[i] > hi[i] + 1e-12]
    if bad:
        raise ValueError('Minimum weight is above maximum weight for: ' + ', '.join(bad))
    if lo.sum() > 1 + 1e-9:
        raise ValueError('Your minimum weights add up to %.2f%%, above 100%%. Lower some minimums.' % (lo.sum() * 100))
    if hi.sum() < 1 - 1e-9:
        raise ValueError('Your maximum weights add up to only %.2f%%, so the portfolio cannot reach 100%%. '
                         'Raise some maximums or add assets.' % (hi.sum() * 100))
    return lo, hi


def _project_box_simplex(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w : sum(w)=1, lo<=w<=hi} by bisection on the shift."""
    x = np.asarray(x, dtype=float)
    a, b = float(np.min(x - hi)) - 1.0, float(np.max(x - lo)) + 1.0
    for _ in range(100):
        t = 0.5 * (a + b)
        if np.clip(x - t, lo, hi).sum() > 1.0:
            a = t
        else:
            b = t
    return np.clip(x - 0.5 * (a + b), lo, hi)


def _return_range(mu: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    """Exact min/max portfolio return over the box-constrained simplex (greedy fill)."""
    def fill(order):
        w = lo.copy(); room = 1.0 - w.sum()
        for i in order:
            add = min(hi[i] - w[i], room)
            w[i] += add; room -= add
            if room <= 1e-15:
                break
        return float(w @ mu)
    return fill(np.argsort(mu)), fill(np.argsort(-mu))


def _solve_constrained_portfolio(mu: np.ndarray, cov: np.ndarray, rf: float, objective: str,
                                 lo: np.ndarray, hi: np.ndarray, target_return: float | None = None,
                                 target_vol: float | None = None, risk_aversion: float = 3.0) -> tuple[np.ndarray, dict]:
    """Long-only optimizer honouring per-asset limits and an optional return/volatility target."""
    n = len(mu)
    notes: list[str] = []
    x0 = _project_box_simplex(np.full(n, 1.0 / n), lo, hi)
    sig = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    var = lambda w: float(w @ cov @ w)
    min_ret, max_ret = _return_range(mu, lo, hi)

    extra: list[dict] = []
    if objective == 'TARGET_RETURN':
        if target_return is None:
            raise ValueError('Enter a target return for the TARGET_RETURN objective.')
        if target_return > max_ret + 1e-9:
            notes.append('Target return %.2f%% is above the %.2f%% your limits allow; used the maximum instead.'
                         % (target_return * 100, max_ret * 100))
            target_return = max_ret
        extra.append({'type': 'ineq', 'fun': lambda w, t=target_return: float(w @ mu - t)})
    if objective == 'TARGET_VOL':
        if target_vol is None or target_vol <= 0:
            raise ValueError('Enter a positive volatility cap for the TARGET_VOL objective.')

    def objective_fn(w):
        if objective in ('MIN_RISK', 'TARGET_RETURN'):
            return var(w)
        if objective == 'MAX_SHARPE':
            return -float((w @ mu - rf) / max(np.sqrt(var(w)), 1e-10))
        if objective == 'TARGET_VOL':
            return -float(w @ mu)
        if objective == 'UTILITY':
            return -float(w @ mu - 0.5 * risk_aversion * var(w))
        if objective == 'RISK_PARITY':
            v = max(var(w), 1e-16)
            rc = w * (cov @ w) / v
            return float(np.sum((rc - 1.0 / n) ** 2)) * 1e3
        if objective == 'MAX_DIVERSIFICATION':
            return -float((w @ sig) / max(np.sqrt(var(w)), 1e-10))
        raise ValueError('Unknown objective')

    def feasible(w, tol=1e-6):
        if abs(w.sum() - 1) > 1e-6 or np.any(w < lo - tol) or np.any(w > hi + tol):
            return False
        if objective == 'TARGET_RETURN' and w @ mu < target_return - 1e-6:
            return False
        if objective == 'TARGET_VOL' and np.sqrt(var(w)) > target_vol + 1e-6:
            return False
        return True

    # TARGET_VOL: if even the least-risky allowed portfolio exceeds the cap, say so and use it.
    if objective == 'TARGET_VOL':
        w_min, _ = _solve_constrained_portfolio(mu, cov, rf, 'MIN_RISK', lo, hi)
        floor_vol = float(np.sqrt(var(w_min)))
        if floor_vol > target_vol + 1e-9:
            notes.append('Volatility cap %.2f%% is below the %.2f%% minimum your limits allow; '
                         'returned the minimum-risk portfolio instead.' % (target_vol * 100, floor_vol * 100))
            return w_min, {'notes': notes, 'solver': 'SLSQP' if _scipy_minimize else 'sampling',
                           'return_range': [min_ret, max_ret], 'min_volatility': floor_vol}
        extra.append({'type': 'ineq', 'fun': lambda w: float(target_vol ** 2 - var(w))})
        x0 = w_min                         # feasible start

    best = None
    solver = 'sampling'
    if _scipy_minimize is not None and n > 1:
        cons = [{'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)}] + extra
        starts = [x0, _project_box_simplex(hi / max(hi.sum(), 1e-12), lo, hi),
                  _project_box_simplex(np.maximum(mu - mu.min(), 1e-6), lo, hi)]
        for start in starts:
            try:
                res = _scipy_minimize(objective_fn, start, method='SLSQP', bounds=list(zip(lo, hi)),
                                      constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            except Exception:
                continue
            if not np.all(np.isfinite(res.x)):
                continue
            w = _project_box_simplex(np.asarray(res.x, dtype=float), lo, hi)
            if feasible(w, 1e-5) and (best is None or objective_fn(w) < objective_fn(best)):
                best = w; solver = 'SLSQP'
    if best is None:
        # Sampling fallback (no scipy, or SLSQP failed): random portfolios projected into the limits.
        rng = np.random.default_rng(11)
        cands = [_project_box_simplex(c, lo, hi) for c in rng.dirichlet(np.ones(n), size=6000)] + [x0]
        ok = [c for c in cands if feasible(c, 1e-4)] or cands
        best = min(ok, key=objective_fn)
        if n > 1:
            notes.append('Numerical optimizer unavailable or did not converge; used the best of 6,000 sampled portfolios.')
    return best, {'notes': notes, 'solver': solver if n > 1 else 'single asset',
                  'return_range': [min_ret, max_ret]}


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
                       lookback: str = '5y', benchmark_symbol: str = '^GSPC', custom_weights: str = '',
                       objective: str = 'MAX_SHARPE', min_weight=None, max_weight=None, bounds: str = '',
                       target_return=None, target_vol=None, risk_aversion=None) -> dict:
    """Estimate long-only portfolio weights for an arbitrary user-selected basket.

    METHODS
    - MIN_RISK: minimum-variance portfolio, sum(weights)=1, weights>=0.
    - TANGENCY: max Sharpe portfolio using the same constraints.
    - COMPARE: calculates both and returns the selected table for comparison UI.
    - CUSTOM: evaluates user-entered weights without optimization.
    - CONSTRAINED: the customizer. Optimizes the chosen objective (see
      CONSTRAINED_OBJECTIVES) subject to the user's per-asset minimum/maximum
      weights and an optional target return or volatility cap. Limits are in
      percent: min_weight/max_weight apply to every asset, bounds overrides per
      asset ('AAPL:5-30,MSFT:0-40').

    Expected returns are annualized arithmetic daily returns (mean * 252),
    covariance is annualized daily covariance, and the risk-free rate matches
    the platform CAPM assumption. This is a model/simulation, not a trade order.
    """
    symbols = _normalize_ticker_list(tickers)
    if not symbols:
        raise ValueError('Enter at least one asset ticker.')
    method = str(method or 'MIN_RISK').upper().replace('-', '_').replace(' ', '_')
    if method not in {'MIN_RISK','TANGENCY','COMPARE','CUSTOM','CONSTRAINED'}:
        raise ValueError('method must be MIN_RISK, TANGENCY, COMPARE, CUSTOM, or CONSTRAINED')
    objective = str(objective or 'MAX_SHARPE').upper().replace('-', '_').replace(' ', '_')
    if objective == 'TANGENCY':
        objective = 'MAX_SHARPE'
    if method == 'CONSTRAINED' and objective not in CONSTRAINED_OBJECTIVES:
        raise ValueError('objective must be one of ' + ', '.join(CONSTRAINED_OBJECTIVES))
    # Validate number formats before any download, so typos fail fast.
    g_min = _pct_or_none(min_weight); g_max = _pct_or_none(max_weight)
    t_ret = _pct_or_none(target_return); t_vol = _pct_or_none(target_vol)
    try:
        lam = 3.0 if risk_aversion in (None, '') else float(risk_aversion)
    except (TypeError, ValueError):
        raise ValueError('Risk aversion must be a number (typical range 1-10).')
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError('Risk aversion must be a positive number (typical range 1-10).')

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
    selected_w = min_w if method in ('MIN_RISK', 'COMPARE') else tangency_w
    if method == 'CUSTOM':
        try:
            allocations = {}
            for entry in custom_weights.split(','):
                if not entry.strip():
                    continue
                symbol, percentage = entry.strip().split(':', 1)
                symbol = symbol.strip().upper()
                if symbol in allocations:
                    raise ValueError('Duplicate allocation for ' + symbol)
                allocations[symbol] = float(percentage)
            if set(allocations) != set(symbols):
                raise ValueError('Custom weights must include exactly the valid assets: ' + ', '.join(symbols))
            if any(not np.isfinite(v) or v < 0 or v > 100 for v in allocations.values()):
                raise ValueError('Each custom weight must be between 0 and 100%.')
            if abs(sum(allocations.values()) - 100) > 0.01:
                raise ValueError('Custom weights must sum to 100%. Current total: %.2f%%' % sum(allocations.values()))
            selected_w = np.array([allocations[s] / sum(allocations.values()) for s in symbols])
        except ValueError:
            raise
        except Exception:
            raise ValueError('Enter custom weights as AAPL:40,MSFT:60 (percentages).')

    constrained_w = None
    constraint_info = None
    if method == 'CONSTRAINED':
        lo, hi = _parse_weight_bounds(bounds, symbols, g_min, g_max)
        constrained_w, solve_info = _solve_constrained_portfolio(
            mu, cov, rf, objective, lo, hi, target_return=t_ret, target_vol=t_vol, risk_aversion=lam)
        selected_w = constrained_w
        binding = []
        for i, sym in enumerate(symbols):
            if hi[i] < 1 - 1e-9 and constrained_w[i] >= hi[i] - 1e-4:
                binding.append(f'{sym} at its {hi[i]*100:.1f}% maximum')
            elif lo[i] > 1e-9 and constrained_w[i] <= lo[i] + 1e-4:
                binding.append(f'{sym} at its {lo[i]*100:.1f}% minimum')
        constraint_info = {
            'objective': objective,
            'objective_label': CONSTRAINED_OBJECTIVES[objective],
            'min_weight': None if g_min is None else round(g_min, 6),
            'max_weight': None if g_max is None else round(g_max, 6),
            'bounds': {sym: [round(float(lo[i]), 6), round(float(hi[i]), 6)] for i, sym in enumerate(symbols)},
            'target_return': None if t_ret is None else round(t_ret, 6),
            'target_vol': None if t_vol is None else round(t_vol, 6),
            'risk_aversion': lam,
            'achievable_return_range': [round(v, 6) for v in solve_info['return_range']],
            'binding': binding,
            'notes': solve_info['notes'],
            'solver': solve_info['solver'],
        }

    def metrics(w):
        ret = float(w @ mu)
        var = float(w @ cov @ w)
        vol = float(np.sqrt(max(var, 0.0)))
        sharpe = float((ret-rf)/vol) if vol > 0 else 0.0
        beta = float(w @ betas)
        return {'expected_return':ret,'variance':var,'volatility':vol,'sharpe':sharpe,'beta':beta}

    min_m = metrics(min_w); tan_m = metrics(tangency_w); sel_m = metrics(selected_w)
    frontier = _efficient_frontier_samples(mu, cov, rf, np.vstack([selected_w, min_w, tangency_w]))
    equal_w = np.full(len(symbols), 1.0 / len(symbols))
    eq_m = metrics(equal_w)

    asset_vol = returns.std(ddof=1).to_numpy(dtype=float) * np.sqrt(252)
    rows=[]
    for i,sym in enumerate(symbols):
        rows.append({
            'ticker':sym,
            'name':names.get(sym,sym),
            'weight':round(float(selected_w[i]),6),
            'min_risk_weight':round(float(min_w[i]),6),
            'tangency_weight':round(float(tangency_w[i]),6),
            'constrained_weight':None if constrained_w is None else round(float(constrained_w[i]),6),
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
        'equal_weight':{**{k:round(v,6) for k,v in eq_m.items()},'weight_sum':1.0},
        'constrained':None if constrained_w is None else {
            **{k:round(v,6) for k,v in metrics(constrained_w).items()},
            'weight_sum':round(float(constrained_w.sum()),6),
            'weights':{sym: round(float(constrained_w[i]),6) for i, sym in enumerate(symbols)},
        },
        'constraints':constraint_info,
        # Inputs the browser customizer needs to re-price any weight mix instantly,
        # with no round trip: annualized expected returns, covariance and betas.
        'model':{
            'tickers':symbols,
            'mu':[round(float(v),8) for v in mu],
            'cov':[[round(float(v),10) for v in row] for row in cov],
            'beta':[round(float(v),6) for v in betas],
            'vol':[round(float(v),8) for v in asset_vol],
            'names':[names.get(sym, sym) for sym in symbols],
        },
        'objectives':CONSTRAINED_OBJECTIVES,
        'market':{'annual_return':round(float(market_ann),6)},
        'rows':rows,
        'scatter':scatter,
        'sml':{'beta':beta_axis.tolist(),'return':sml_return.tolist()},
        'cal':{'volatility':cal_vols.tolist(),'return':cal_returns.tolist()},
        'efficient_frontier':frontier,
        'methodology':(
            'Long-only mean-variance portfolio simulation. MIN_RISK minimizes annualized portfolio variance. '
            'TANGENCY maximizes estimated Sharpe ratio against the platform risk-free assumption. '
            'Weights sum to 100% and no short weights are allowed in this portfolio optimizer. CUSTOM evaluates user-entered weights without optimization. '
            'CONSTRAINED (the customizer) optimizes your chosen objective inside your per-asset minimum/maximum weights and optional return target or volatility cap. '
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
        "status": "healthy", "engine": "finmodel", "version": "17.1.0",
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
    benchmark: str = Query('^GSPC'),
    custom_weights: str = Query(''),
    objective: str = Query('MAX_SHARPE', description="CONSTRAINED objective: " + ", ".join(CONSTRAINED_OBJECTIVES)),
    min_weight: str = Query('', description="Minimum weight per asset, %"),
    max_weight: str = Query('', description="Maximum weight per asset, %"),
    bounds: str = Query('', description="Per-asset limits in %, e.g. AAPL:5-30,MSFT:0-40"),
    target_return: str = Query('', description="Target annual return, % (TARGET_RETURN)"),
    target_vol: str = Query('', description="Volatility cap, % (TARGET_VOL)"),
    risk_aversion: str = Query('', description="Risk aversion for UTILITY (default 3)"),
):
    if lookback not in {'1y', '2y', '3y', '5y', '10y'}:
        raise HTTPException(status_code=400, detail="lookback must be one of 1y, 2y, 3y, 5y, 10y")
    extra = ''
    if method.strip().upper() == 'CONSTRAINED':
        extra = '::' + '::'.join(str(x).strip().upper() for x in
                                 (objective, min_weight, max_weight, bounds, target_return, target_vol, risk_aversion))
    cache_key = f"portfolio::{tickers.strip().upper()}::{method.upper()}::{lookback}::{benchmark.upper()}::{custom_weights.strip().upper()}{extra}"
    try:
        return _cached_call(cache_key, lambda: optimize_portfolio(
            tickers, method=method, lookback=lookback, benchmark_symbol=benchmark, custom_weights=custom_weights,
            objective=objective, min_weight=min_weight, max_weight=max_weight, bounds=bounds,
            target_return=target_return, target_vol=target_vol, risk_aversion=risk_aversion))
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
            calendar_snapshot(block=True)
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

# ==============================================================================
# INTELLIGENCE LAYER (macro calendar, news intelligence, geopolitical simulator,
# indicator lab) -- formerly finmodel_intel.py
# ==============================================================================
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


# ==============================================================================
# INTELLIGENCE PAGE VIEWS -- formerly finmodel_intel_ui.py
# ==============================================================================
"""Front-end for finmodel_intel: three dashboard views injected into the main page.

inject(page) rewrites the main dashboard HTML:
  * the old "Macro Calendar & News" view becomes "Macro & News Intelligence"
  * new views: "Geopolitical Simulator" and "Indicator Lab"
  * extra CSS and one extra <script> that hooks into switchView / renderNewsCalendar
Every server-provided string is HTML-escaped before it is inserted.
"""
import re

_NAV_OLD = '<div class="navlink" data-view="news" onclick="switchView(\'news\', this)">◉ <span>Macro Calendar & News</span></div>'
_NAV_NEW = ('<div class="navlink" data-view="news" onclick="switchView(\'news\', this)">◉ <span>Macro & News Intelligence</span></div>\n'
            '        <div class="navlink" data-view="geo" onclick="switchView(\'geo\', this)">⊕ <span>Geopolitical Simulator</span></div>\n'
            '        <div class="navlink" data-view="indicators" onclick="switchView(\'indicators\', this)">≋ <span>Indicator Lab</span></div>')

CSS = r'''
.chip{display:inline-flex;align-items:center;gap:4px;border:1px solid var(--line);border-radius:999px;padding:2px 8px;font-size:10px;font-family:"JetBrains Mono",monospace;white-space:nowrap}
.chip.pos{border-color:#14532d;color:#34d399}.chip.neg{border-color:#7f1d1d;color:#fb7185}.chip.neu{color:#94a3b8}
.chip.vh{border-color:#7f1d1d;color:#fda4af}.chip.hi{border-color:#78350f;color:#fbbf24}.chip.md{color:#94a3b8}
.chip.btn{cursor:pointer}.chip.btn.on{background:#0a2d23;border-color:#155d49;color:var(--green)}
.intel-card{border:1px solid var(--line);border-radius:12px;padding:14px;background:rgba(7,12,18,.6)}
.intel-bar{height:5px;border-radius:999px;background:#17212d;overflow:hidden}.intel-bar>span{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}
.intel-ev summary{cursor:pointer;list-style:none}.intel-ev summary::-webkit-details-marker{display:none}
.intel-ev[open]{background:rgba(13,24,34,.75)}
.intel-day{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:14px 0 6px}
.intel-link{color:var(--cyan);font-size:11px}.intel-link:hover{text-decoration:underline}
.intel-p{font-size:12.5px;line-height:1.65;color:#cbd5e1}.intel-p b{color:#fff}
.intel-table td,.intel-table th{padding:7px 8px;font-size:11px}
.intel-input{background:#070c12;border:1px solid #263444;color:#fff;border-radius:8px;padding:6px 9px;font-size:12px;outline:none;width:100%}
.intel-check{display:flex;align-items:center;gap:6px;font-size:11px;color:#cbd5e1;cursor:pointer}
#geoNetwork svg text{font-family:Inter,sans-serif}
'''

HTML_NEWS = r'''
        <div id="view-news" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div><div class="font-semibold text-lg">Macro &amp; News Intelligence</div>
                    <div class="text-[10px] muted mono mt-1">THEMES · TONE · FACTOR LINKS · CALENDAR · SOURCES</div></div>
                    <div class="flex items-center gap-2"><span id="intelNewsStatus" class="badge">OPEN TO LOAD</span>
                    <button class="btn px-3 py-1.5 text-xs" onclick="loadIntelNews(true)">Refresh ↻</button></div>
                </div>
                <div id="intelBrief" class="mt-4 space-y-3"></div>
            </section>
            <section class="grid grid-cols-1 xl:grid-cols-3 gap-5">
                <div class="panel p-5 xl:col-span-2">
                    <div class="font-semibold">Themes driving the news</div>
                    <div class="text-[10px] muted mono mt-1 mb-4">RANKED BY RECENCY-WEIGHTED COVERAGE · EACH WITH ITS MECHANISM AND LINK TO THE ASSET</div>
                    <div id="intelThemes" class="space-y-3"></div>
                </div>
                <div class="space-y-5">
                    <div class="panel p-5"><div class="font-semibold">What moved the asset last week</div>
                        <div class="text-[10px] muted mono mt-1 mb-3">FACTOR ATTRIBUTION (WEEKLY REGRESSION)</div>
                        <div class="h-[230px]"><canvas id="intelAttribChart"></canvas></div>
                        <div id="intelAttribText" class="text-xs muted mt-3 leading-5"></div></div>
                    <div class="panel p-5"><div class="font-semibold">Macro exposures</div>
                        <div class="text-[10px] muted mono mt-1 mb-3">3Y WEEKLY BETAS · t-STATS</div>
                        <div class="table-wrap"><table class="intel-table" id="intelBetaTable"><tbody></tbody></table></div></div>
                </div>
            </section>
            <section class="panel p-5">
                <div class="flex flex-col xl:flex-row xl:items-end justify-between gap-3">
                    <div><div class="font-semibold text-lg">Economic Calendar</div>
                    <div class="text-[10px] muted mono mt-1">AUTOMATIC · OFFICIAL DATES (FRED) · CONSENSUS (WEEKLY FEED) · YOUR LOCAL TIME</div>
                    <div id="calStatus" class="text-[10px] muted mt-1"></div></div>
                    <div class="flex flex-wrap gap-2 items-center">
                        <select id="calDays" class="intel-input" style="width:auto" onchange="loadCalendar(true)"><option value="7">7 days</option><option value="21" selected>21 days</option><option value="45">45 days</option><option value="90">90 days</option></select>
                        <select id="calImportance" class="intel-input" style="width:auto" onchange="loadCalendar(true)"><option value="MEDIUM">Medium+</option><option value="HIGH" selected>High+</option><option value="VERY HIGH">Very high</option></select>
                        <select id="calRegion" class="intel-input" style="width:auto" onchange="loadCalendar(true)"><option value="all">All regions</option><option value="us">United States</option><option value="global">Outside US</option></select>
                        <label class="intel-check"><input type="checkbox" id="calSensitive" onchange="renderCalendar()"> Only events that move this asset</label>
                        <a id="calIcs" class="btn px-3 py-1.5 text-xs" href="#" title="Download or subscribe in Google/Apple/Outlook calendar">Add to my calendar (.ics)</a>
                    </div>
                </div>
                <div id="calFocus" class="mt-4"></div>
                <div id="calList" class="mt-2"></div>
            </section>
            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div><div class="font-semibold text-lg">Headline stream</div><div class="text-[10px] muted mono mt-1">DE-DUPLICATED ACROSS OUTLETS · TONE-SCORED · LINKED</div></div>
                    <div id="intelFilters" class="flex flex-wrap gap-2"></div>
                </div>
                <div id="intelHeadlines" class="mt-4 space-y-2"></div>
                <div id="intelMethod" class="text-[10px] muted mt-4 leading-5"></div>
            </section>
        </div>

        <div id="view-geo" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div><div class="font-semibold text-lg">Geopolitical Shock Simulator</div>
                    <div class="text-[10px] muted mono mt-1">SCENARIO → FACTOR SPILLOVERS → ASSET · MONTE CARLO ON MEASURED EXPOSURES</div></div>
                    <span id="geoStatus" class="badge">OPEN TO LOAD</span>
                </div>
                <div class="grid grid-cols-1 lg:grid-cols-4 gap-3 mt-5">
                    <label class="text-xs muted lg:col-span-2">Scenario<select id="geoScenario" class="intel-input mt-1" onchange="geoFillShocks(); runGeo()"></select></label>
                    <label class="text-xs muted">Horizon<select id="geoHorizon" class="intel-input mt-1" onchange="runGeo()"><option value="1">1 week</option><option value="2">2 weeks</option><option value="4" selected>1 month</option><option value="8">2 months</option><option value="13">3 months</option></select></label>
                    <label class="text-xs muted">Correlations<select id="geoRegime" class="intel-input mt-1" onchange="loadGeo(true)"><option value="stress" selected>Stress regime (crisis weeks)</option><option value="normal">Full history</option></select></label>
                </div>
                <div class="text-[10px] muted mt-4 mb-2">Primary shocks (edit to build your own; leave blank to let a factor follow its historical co-movement)</div>
                <div id="geoShockInputs" class="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3"></div>
                <div class="flex flex-wrap gap-2 mt-4 items-center">
                    <button class="btn px-5 py-2.5" onclick="runGeo()">Run simulation →</button>
                    <button class="btn px-4 py-2.5 text-xs" onclick="geoFillShocks(); runGeo()">Reset to scenario</button>
                    <label class="text-xs muted flex items-center gap-2 ml-2">Shock uncertainty <input id="geoUnc" type="range" min="0" max="0.8" step="0.05" value="0.35" onchange="runGeo()"> <span id="geoUncVal" class="mono">35%</span></label>
                </div>
            </section>
            <section class="grid grid-cols-2 md:grid-cols-6 gap-3" id="geoKpis"></section>
            <section class="panel p-5">
                <div class="font-semibold">Propagation network</div>
                <div class="text-[10px] muted mono mt-1 mb-3">TRIGGER → SHOCKED FACTORS → SPILLOVERS (CONDITIONAL ON HISTORY) → ASSET · EDGE LABELS = EXPECTED EFFECT</div>
                <div id="geoNetwork" class="w-full overflow-x-auto"></div>
            </section>
            <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold">Outcome distribution</div><div class="text-[10px] muted mono mt-1 mb-3">SIMULATED RETURN OVER THE HORIZON</div><div class="h-[300px]"><canvas id="geoHist"></canvas></div></div>
                <div class="panel p-5"><div class="font-semibold">Contribution by channel</div><div class="text-[10px] muted mono mt-1 mb-3">BETA × EXPECTED FACTOR MOVE</div><div class="h-[300px]"><canvas id="geoContrib"></canvas></div></div>
            </section>
            <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold">Analysis</div><div id="geoNarrative" class="space-y-3 mt-3"></div>
                    <div class="font-semibold mt-5">Causal chain</div><ol id="geoChain" class="list-decimal ml-5 mt-2 space-y-1 intel-p"></ol>
                    <div id="geoAnalogues" class="text-xs muted mt-3"></div></div>
                <div class="panel p-5"><div class="font-semibold">Factor detail</div><div class="table-wrap mt-3"><table class="intel-table" id="geoFactorTable"><thead><tr><th>Factor</th><th>Scenario move</th><th>Beta</th><th>t</th><th>Contribution</th></tr></thead><tbody></tbody></table></div></div>
            </section>
            <section class="panel p-5">
                <div class="font-semibold">Stress-test matrix</div>
                <div class="text-[10px] muted mono mt-1 mb-3">EVERY SCENARIO FOR THIS ASSET · HEADLINE RELEVANCE FROM LIVE NEWS · CLICK A ROW TO SIMULATE</div>
                <div class="table-wrap"><table class="intel-table" id="geoMatrix"><thead><tr><th>Scenario</th><th>Region</th><th>Headline relevance</th><th>Primary shocks</th><th>Expected move</th></tr></thead><tbody></tbody></table></div>
            </section>
        </div>

        <div id="view-indicators" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div><div class="font-semibold text-lg">Indicator Lab</div><div class="text-[10px] muted mono mt-1">20+ INDICATORS · SIGNAL SUMMARY · RELIABILITY ON THIS ASSET</div></div>
                    <span id="indStatus" class="badge">OPEN TO LOAD</span>
                </div>
                <div class="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3 mt-5">
                    <label class="text-xs muted">Interval<select id="indInterval" class="intel-input mt-1" onchange="loadIndicators(true)"><option value="1d" selected>Daily</option><option value="1wk">Weekly</option><option value="1h">Hourly</option></select></label>
                    <label class="text-xs muted">Window<select id="indPeriod" class="intel-input mt-1" onchange="loadIndicators(true)"><option value="3mo">3 months</option><option value="6mo">6 months</option><option value="1y" selected>1 year</option><option value="2y">2 years</option><option value="5y">5 years</option></select></label>
                    <label class="text-xs muted">Oscillator<select id="indOsc" class="intel-input mt-1" onchange="renderIndicators()"><option>RSI</option><option>MACD</option><option>Stochastic</option><option>ADX / DMI</option><option>CCI</option><option>Williams %R</option><option>MFI</option><option>OBV</option><option>ATR %</option></select></label>
                    <label class="text-xs muted">SMA periods<input id="indSma" class="intel-input mt-1" value="50,200"></label>
                    <label class="text-xs muted">EMA periods<input id="indEma" class="intel-input mt-1" value="20"></label>
                    <label class="text-xs muted">Forward test (bars)<input id="indFwd" class="intel-input mt-1" type="number" value="10" min="2" max="120"></label>
                </div>
                <details class="mt-3"><summary class="text-xs cyan cursor-pointer">More parameters</summary>
                <div class="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3 mt-3">
                    <label class="text-xs muted">BB length<input id="ind_bb_n" class="intel-input mt-1" value="20"></label>
                    <label class="text-xs muted">BB std dev<input id="ind_bb_k" class="intel-input mt-1" value="2"></label>
                    <label class="text-xs muted">Supertrend ATR<input id="ind_st_n" class="intel-input mt-1" value="10"></label>
                    <label class="text-xs muted">Supertrend mult<input id="ind_st_mult" class="intel-input mt-1" value="3"></label>
                    <label class="text-xs muted">RSI length<input id="ind_rsi_n" class="intel-input mt-1" value="14"></label>
                    <label class="text-xs muted">MACD fast<input id="ind_macd_fast" class="intel-input mt-1" value="12"></label>
                    <label class="text-xs muted">MACD slow<input id="ind_macd_slow" class="intel-input mt-1" value="26"></label>
                    <label class="text-xs muted">MACD signal<input id="ind_macd_signal" class="intel-input mt-1" value="9"></label>
                    <label class="text-xs muted">Stochastic %K<input id="ind_stoch_k" class="intel-input mt-1" value="14"></label>
                    <label class="text-xs muted">ADX length<input id="ind_adx_n" class="intel-input mt-1" value="14"></label>
                    <label class="text-xs muted">CCI length<input id="ind_cci_n" class="intel-input mt-1" value="20"></label>
                    <label class="text-xs muted">Donchian length<input id="ind_dc_n" class="intel-input mt-1" value="20"></label>
                    <label class="text-xs muted">Keltner length<input id="ind_kc_n" class="intel-input mt-1" value="20"></label>
                    <label class="text-xs muted">VWAP window<input id="ind_vwap_n" class="intel-input mt-1" value="20"></label>
                    <label class="text-xs muted">MFI length<input id="ind_mfi_n" class="intel-input mt-1" value="14"></label>
                    <label class="text-xs muted">Williams length<input id="ind_wr_n" class="intel-input mt-1" value="14"></label>
                </div></details>
                <div class="flex flex-wrap gap-3 mt-4" id="indOverlays"></div>
                <div class="mt-4"><button class="btn px-5 py-2.5" onclick="loadIndicators(true)">Apply parameters →</button></div>
            </section>
            <section class="panel p-4 md:p-5"><div class="h-[420px]"><canvas id="indPriceChart"></canvas></div><div class="h-[200px] mt-4"><canvas id="indOscChart"></canvas></div></section>
            <section class="grid grid-cols-1 xl:grid-cols-3 gap-5">
                <div class="panel p-5"><div class="font-semibold">Technical summary</div><div id="indSummary" class="mt-4"></div></div>
                <div class="panel p-5 xl:col-span-2"><div class="font-semibold">Signals &amp; reliability on this asset</div>
                    <div class="text-[10px] muted mono mt-1 mb-3">AVERAGE FORWARD RETURN AFTER EACH SIGNAL VS THE ASSET'S NORMAL DRIFT</div>
                    <div class="table-wrap"><table class="intel-table" id="indSignals"><thead><tr><th>Indicator</th><th>Now</th><th>Reading</th><th>After bullish signals</th><th>After bearish signals</th></tr></thead><tbody></tbody></table></div>
                    <div id="indNote" class="text-[10px] muted mt-3 leading-5"></div></div>
            </section>
        </div>
'''

JS = r'''
<script>
/* ===================== finmodel intelligence layer ===================== */
const INTEL = {ticker: null, news: null, cal: null, geo: null, sim: null, ind: null, headFilter: 'all', charts: {}};
function iesc(s){ return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function iurl(u){ return /^https?:\/\//i.test(String(u||'')) ? iesc(u) : '#'; }
function itk(){ return String((typeof currentData !== 'undefined' && currentData && currentData.ticker) || (document.getElementById('tickerInput')||{}).value || '').trim().toUpperCase(); }
async function ifetch(path){ const r = await fetch(API_BASE + path); let j = {}; try { j = await r.json(); } catch(e) {} if(!r.ok) throw new Error(j.detail || ('HTTP ' + r.status)); return j; }
function iLocal(iso){ try { return new Date(iso).toLocaleString([], {weekday:'short', month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'}); } catch(e){ return iso; } }
function iDay(iso){ try { return new Date(iso).toLocaleDateString([], {weekday:'long', month:'long', day:'numeric'}); } catch(e){ return iso.slice(0,10); } }
function iCountdown(iso){ const ms = new Date(iso) - new Date(); const a = Math.abs(ms)/36e5; const txt = a < 1 ? Math.round(a*60)+'m' : a < 48 ? Math.round(a)+'h' : Math.round(a/24)+'d'; return ms >= 0 ? 'in '+txt : txt+' ago'; }
function iToneChip(t){ const v = Number(t||0); const cls = v > 0.08 ? 'pos' : v < -0.08 ? 'neg' : 'neu'; return `<span class="chip ${cls}">tone ${v>=0?'+':''}${v.toFixed(2)}</span>`; }
function iImpChip(i){ const c = i==='VERY HIGH'?'vh':i==='HIGH'?'hi':'md'; return `<span class="chip ${c}">${iesc(i)}</span>`; }
function iChart(id, cfg){ if(INTEL.charts[id]) INTEL.charts[id].destroy(); const el = document.getElementById(id); if(!el) return null; INTEL.charts[id] = new Chart(el, cfg); return INTEL.charts[id]; }
const ICOL = {grid:'#17212d', tick:'#64748b', green:'#19d58b', red:'#fb7185', cyan:'#22d3ee', amber:'#fbbf24', violet:'#a78bfa', slate:'#94a3b8'};
function iAxes(extra){ return Object.assign({x:{grid:{color:ICOL.grid},ticks:{color:ICOL.tick,maxTicksLimit:10}}, y:{grid:{color:ICOL.grid},ticks:{color:ICOL.tick}}}, extra||{}); }

/* ---- hooks into the existing dashboard ---- */
const _intelOrigSwitch = switchView;
switchView = function(v, el){ _intelOrigSwitch(v, el); if(v==='news') loadIntelNews(); if(v==='geo') loadGeo(); if(v==='indicators') loadIndicators(); };
renderNewsCalendar = function(data){
    const t = (data && data.ticker) ? String(data.ticker).toUpperCase() : itk();
    if (t && t !== INTEL.ticker) { INTEL.ticker = t; INTEL.news = INTEL.cal = INTEL.geo = INTEL.sim = INTEL.ind = null; }
    const active = (document.querySelector('.view-section.active-view')||{}).id;
    if (active === 'view-news') loadIntelNews(); else if (active === 'view-geo') loadGeo(); else if (active === 'view-indicators') loadIndicators();
};

/* ===================== Macro & News ===================== */
async function loadIntelNews(force){
    const t = itk(); if(!t) return;
    INTEL.ticker = t;
    loadCalendar(force);
    if (INTEL.news && INTEL.news.ticker === t && !force) return renderIntelNews();
    const st = document.getElementById('intelNewsStatus'); st.innerText = 'READING THE NEWS…';
    try { INTEL.news = await ifetch('/api/intel/' + encodeURIComponent(t)); st.innerText = INTEL.news.story_count + ' STORIES'; renderIntelNews(); }
    catch(e){ st.innerText = 'UNAVAILABLE'; document.getElementById('intelBrief').innerHTML = `<div class="text-sm red">News intelligence unavailable: ${iesc(e.message)}</div>`; }
}
function renderIntelNews(){
    const d = INTEL.news; if(!d) return;
    document.getElementById('intelBrief').innerHTML = (d.brief||[]).map(p => {
        const m = String(p).match(/^([^:]{3,40}):\s(.*)$/);
        return `<p class="intel-p">${m ? `<b>${iesc(m[1])}:</b> ${iesc(m[2])}` : iesc(p)}</p>`;
    }).join('');
    document.getElementById('intelThemes').innerHTML = (d.themes||[]).map(th => `
        <div class="intel-card">
            <div class="flex flex-wrap items-center justify-between gap-2"><strong>${iesc(th.label)}</strong>
              <div class="flex gap-2">${iToneChip(th.tone)}<span class="chip">${th.count} stories · ${th.outlets} outlets</span></div></div>
            <div class="intel-bar mt-2"><span style="width:${Math.max(4, th.intensity_pct)}%"></span></div>
            <div class="mt-3 space-y-2">${(th.paragraphs||[]).map(p => `<p class="intel-p">${iesc(p)}</p>`).join('')}</div>
            <div class="mt-3 space-y-1">${(th.stories||[]).map(s => `<div class="text-xs"><a class="intel-link" href="${iurl(s.url)}" target="_blank" rel="noopener">${iesc(s.title)}</a>
                <span class="muted"> — ${iesc(s.source)}, ${Math.round(s.age_h)}h ago${s.outlets>1?`, ${s.outlets} outlets`:''}</span></div>`).join('')}</div>
        </div>`).join('') || '<div class="text-sm muted">No themed coverage returned by the feeds.</div>';

    // filters + headline stream
    const themes = [...new Set((d.headlines||[]).flatMap(h => h.themes||[]))].slice(0, 8);
    const filters = [['all','All'],['asset','About '+d.ticker],['macro','Macro'],['geo','Geopolitical'], ...themes.map(t => ['theme:'+t, t])];
    document.getElementById('intelFilters').innerHTML = filters.map(([k,l]) => `<span class="chip btn ${INTEL.headFilter===k?'on':''}" onclick="INTEL.headFilter='${iesc(k).replace(/'/g,'')}';renderIntelNews()">${iesc(l)}</span>`).join('');
    const f = INTEL.headFilter;
    const rows = (d.headlines||[]).filter(h => f==='all' || (f==='asset' && h.about_asset) || (f==='macro' && h.origin==='macro') || (f==='geo' && h.origin==='geo') || (f.startsWith('theme:') && (h.themes||[]).includes(f.slice(6))));
    document.getElementById('intelHeadlines').innerHTML = rows.map(h => `
        <div class="scenario news-row"><div class="flex flex-col md:flex-row md:justify-between gap-2">
            <a class="text-sm intel-link" style="font-size:13px" href="${iurl(h.url)}" target="_blank" rel="noopener">${iesc(h.title)}</a>
            <div class="flex gap-1 flex-wrap">${iToneChip(h.tone)}${(h.themes||[]).map(t => `<span class="chip">${iesc(t)}</span>`).join('')}</div></div>
            <div class="text-[10px] muted mt-1">${iesc(h.source)} · ${Math.round(h.age_h)}h ago${h.about_asset?' · mentions '+iesc(d.ticker):''}
            ${(h.also||[]).length ? ' · also: ' + h.also.map(a => `<a class="intel-link" href="${iurl(a.url)}" target="_blank" rel="noopener">${iesc(a.source)}</a>`).join(', ') : ''}</div></div>`).join('') || '<div class="text-sm muted">No headlines for this filter.</div>';
    document.getElementById('intelMethod').innerText = d.method || '';

    // factor attribution + betas
    const fm = d.factor_model;
    const bt = document.querySelector('#intelBetaTable tbody');
    if (fm && fm.ok) {
        const lw = fm.last_week || {};
        const keys = Object.keys(lw.contributions||{});
        const labels = keys.map(k => (fm.labels||{})[k] || k).concat(['Stock-specific', 'Actual']);
        const vals = keys.map(k => lw.contributions[k]).concat([lw.stock_specific, lw.actual]);
        iChart('intelAttribChart', {type:'bar', data:{labels, datasets:[{data:vals, backgroundColor: vals.map((v,i) => i===vals.length-1 ? ICOL.cyan : (v>=0?ICOL.green:ICOL.red))}]},
            options:{indexAxis:'y', responsive:true, maintainAspectRatio:false, animation:false, plugins:{legend:{display:false}}, scales:iAxes({x:{grid:{color:ICOL.grid},ticks:{color:ICOL.tick,callback:v=>v+'%'}}})}});
        document.getElementById('intelAttribText').innerText = `Week ${lw.start} → ${lw.end}. Factors explain ${Math.round((fm.r2||0)*100)}% of ${fm.ticker}'s weekly moves over ${fm.weeks} weeks (${fm.start} → ${fm.end}).`;
        bt.innerHTML = '<tr><th>Factor</th><th>Effect</th><th>t</th></tr>' + Object.entries(fm.betas||{}).map(([k,b]) => {
            if (b.self) return `<tr><td>${iesc(b.label)}</td><td colspan="2">this asset</td></tr>`;
            const per = b.unit==='bp' ? 10 : 1; const unit = b.unit==='bp' ? 'per +10bp' : b.unit==='pts' ? 'per +1 pt' : 'per +1%';
            const v = (b.beta||0)*per; const sig = Math.abs(b.t||0) >= 2;
            return `<tr><td>${iesc(b.label)}</td><td class="${v>=0?'green':'red'}">${v>=0?'+':''}${v.toFixed(2)}% <span class="muted">${unit}</span></td><td class="${sig?'':'muted'}">${b.t==null?'–':Number(b.t).toFixed(1)}</td></tr>`;
        }).join('');
    } else {
        bt.innerHTML = `<tr><td class="muted">${iesc((fm && fm.reason) || 'Factor model unavailable')}</td></tr>`;
    }
}

async function loadCalendar(force){
    const t = itk();
    const days = document.getElementById('calDays').value, imp = document.getElementById('calImportance').value, reg = document.getElementById('calRegion').value;
    const key = [t, days, imp, reg].join('|');
    document.getElementById('calIcs').href = API_BASE + '/api/calendar.ics?days=60&importance=HIGH';
    if (INTEL.cal && INTEL.cal._key === key && !force) return renderCalendar();
    document.getElementById('calList').innerHTML = '<div class="text-sm muted">Loading calendar…</div>';
    try { INTEL.cal = await ifetch(`/api/calendar?ticker=${encodeURIComponent(t)}&days=${days}&importance=${encodeURIComponent(imp)}&region=${reg}`); INTEL.cal._key = key; renderCalendar(); }
    catch(e){ document.getElementById('calList').innerHTML = `<div class="text-sm red">Calendar unavailable: ${iesc(e.message)}</div>`; }
}
function renderCalendar(){
    const d = INTEL.cal; if(!d) return;
    const s = d.status || {};
    document.getElementById('calStatus').innerHTML = `Official schedule: <b>${iesc(s.fred_schedule||'?')}</b> · Consensus feed: <b>${iesc(s.weekly_feed||'?')}</b>` +
        (d.fred_key_configured ? '' : ' · <span class="amber">Set FRED_API_KEY for official dates and event-day statistics on every release</span>');
    const f = d.focus;
    document.getElementById('calFocus').innerHTML = f ? `<div class="intel-card"><span class="chip hi">MOST RELEVANT FOR ${iesc(d.ticker||'')}</span>
        <div class="mt-2"><b>${iesc(f.title)}</b> <span class="muted text-xs">${iLocal(f.datetime_utc)} (${iCountdown(f.datetime_utc)})</span></div><div class="text-xs muted mt-1">${iesc(f.why)}</div></div>` : '';
    const only = document.getElementById('calSensitive').checked;
    let events = (d.events||[]);
    if (only) events = events.filter(e => e.sensitivity && e.sensitivity.ratio >= 1.15);
    let html = '', lastDay = '';
    for (const e of events) {
        const day = iDay(e.datetime_utc);
        if (day !== lastDay) { html += `<div class="intel-day">${iesc(day)}</div>`; lastDay = day; }
        const sens = e.sensitivity;
        const sensChip = sens && sens.ratio ? `<span class="chip ${sens.ratio>=1.3?'hi':'md'}" title="Average move on past release days vs other days">${iesc(d.ticker)} ${sens.ratio}× normal</span>` : '';
        const cons = (e.consensus||[]).slice(0,3).map(c => `<span class="chip" title="${iesc(c.label)}">${iesc(c.label.length>22?c.label.slice(0,22)+'…':c.label)}: f ${iesc(c.forecast||'–')} · p ${iesc(c.previous||'–')}${c.actual?' · a '+iesc(c.actual):''}</span>`).join('');
        const a = e.analysis || {};
        html += `<details class="intel-ev scenario mb-2"><summary><div class="flex flex-col md:flex-row md:items-center gap-3">
            <div class="w-40 mono text-xs">${iLocal(e.datetime_utc)}<div class="muted text-[10px]">${iesc(e.time)} ET · ${iCountdown(e.datetime_utc)}</div></div>
            <div class="flex-1"><strong class="text-sm">${iesc(e.title)}</strong>${e.estimated?' <span class="chip md">est.</span>':''}
                <div class="flex flex-wrap gap-1 mt-1">${iImpChip(e.importance)}<span class="chip">${iesc(e.region||'')}</span><span class="chip">${iesc(a.category||'')}</span>${sensChip}${cons}</div></div>
            <div class="text-[10px] cyan">Analysis ▾</div></div></summary>
            <div class="mt-3 space-y-2">${(a.paragraphs||[]).map(p => { const m = String(p).match(/^([^:]{3,32}):\s(.*)$/); return `<p class="intel-p">${m?`<b>${iesc(m[1])}:</b> ${iesc(m[2])}`:iesc(p)}</p>`; }).join('')}
            <div class="flex flex-wrap gap-3 mt-2">${(a.links||[]).map(l => `<a class="intel-link" href="${iurl(l.url)}" target="_blank" rel="noopener">${iesc(l.label)} ↗</a>`).join('')}
            ${(a.readings||[]).map(r => `<a class="intel-link" href="${iurl(r.url)}" target="_blank" rel="noopener">${iesc(r.label)} data (FRED) ↗</a>`).join('')}</div>
            <div class="text-[10px] muted">Sources: ${iesc((e.sources||[]).join(', '))}</div></div></details>`;
    }
    document.getElementById('calList').innerHTML = html || '<div class="text-sm muted">No events match these filters.</div>';
}

/* ===================== Geopolitical simulator ===================== */
async function loadGeo(force){
    const t = itk(); if(!t) return;
    const regime = document.getElementById('geoRegime').value;
    const st = document.getElementById('geoStatus');
    if (!INTEL.geo || INTEL.geo.ticker !== t || INTEL.geo.regime !== regime || force) {
        st.innerText = 'ESTIMATING EXPOSURES…';
        try { INTEL.geo = await ifetch(`/api/geo/scenarios?ticker=${encodeURIComponent(t)}&regime=${regime}`); }
        catch(e){ st.innerText = 'UNAVAILABLE'; document.getElementById('geoNarrative').innerHTML = `<div class="text-sm red">${iesc(e.message)}</div>`; return; }
        const sel = document.getElementById('geoScenario');
        const prev = sel.value;
        const sorted = [...INTEL.geo.scenarios].sort((a,b) => b.relevance_pct - a.relevance_pct);
        sel.innerHTML = sorted.map(s => `<option value="${iesc(s.key)}">${iesc(s.name)}${s.relevance_pct>=60?'  — in the news':''}</option>`).join('') + '<option value="custom">Custom scenario (set shocks below)</option>';
        sel.value = prev && [...sel.options].some(o => o.value===prev) ? prev : sorted[0].key;
        geoFillShocks();
        renderGeoMatrix();
    }
    if (!INTEL.sim || INTEL.sim.ticker !== t || force) runGeo();
}
function geoFillShocks(){
    const g = INTEL.geo; if(!g) return;
    const sc = g.scenarios.find(s => s.key === document.getElementById('geoScenario').value) || {shock_values:{}};
    document.getElementById('geoShockInputs').innerHTML = g.factors.map(f => {
        const v = (sc.shock_values||{})[f.symbol];
        const unit = f.unit==='bp' ? 'bp' : f.unit==='pts' ? 'pts' : '%';
        return `<label class="text-xs muted">${iesc(f.label)} (${unit})<input class="intel-input mt-1 mono" data-f="${iesc(f.symbol)}" value="${v==null?'':v}" placeholder="follows history"></label>`;
    }).join('');
}
function renderGeoMatrix(){
    const g = INTEL.geo; if(!g) return;
    document.querySelector('#geoMatrix tbody').innerHTML = [...g.scenarios].sort((a,b) => (a.expected_move??0) - (b.expected_move??0)).map(s => `
        <tr style="cursor:pointer" onclick="document.getElementById('geoScenario').value='${iesc(s.key)}';geoFillShocks();runGeo();window.scrollTo({top:0,behavior:'smooth'})">
        <td>${iesc(s.name)}</td><td>${iesc(s.region)}</td>
        <td><div class="intel-bar" style="width:90px;display:inline-block;vertical-align:middle"><span style="width:${s.relevance_pct}%"></span></div> <span class="muted">${s.headline_hits} stories</span></td>
        <td class="muted">${Object.entries(s.shocks).map(([k,v]) => iesc(((g.factors.find(f=>f.symbol===k)||{}).label||k) + ' ' + v)).join(', ')}</td>
        <td class="${(s.expected_move??0)>=0?'green':'red'}">${s.expected_move==null?'–':(s.expected_move>=0?'+':'')+s.expected_move.toFixed(2)+'%'}</td></tr>`).join('');
}
async function runGeo(){
    const t = itk(); if(!t || !INTEL.geo) return;
    const st = document.getElementById('geoStatus'); st.innerText = 'SIMULATING…';
    const unc = document.getElementById('geoUnc').value; document.getElementById('geoUncVal').innerText = Math.round(unc*100)+'%';
    const shocks = [...document.querySelectorAll('#geoShockInputs input')].filter(i => i.value.trim()!=='' && !isNaN(Number(i.value))).map(i => i.dataset.f + ':' + Number(i.value)).join(',');
    const q = new URLSearchParams({scenario: document.getElementById('geoScenario').value, horizon_weeks: document.getElementById('geoHorizon').value,
        regime: document.getElementById('geoRegime').value, uncertainty: unc});
    if (shocks) q.set('shocks', shocks);
    try { INTEL.sim = await ifetch(`/api/geo/simulate/${encodeURIComponent(t)}?` + q.toString()); st.innerText = 'MODEL R² ' + Math.round((INTEL.sim.model.r2||0)*100) + '%'; renderGeo(); }
    catch(e){ st.innerText = 'ERROR'; document.getElementById('geoNarrative').innerHTML = `<div class="text-sm red">${iesc(e.message)}</div>`; }
}
function renderGeo(){
    const d = INTEL.sim; if(!d) return;
    const s = d.stats;
    const k = [['Expected', s.mean], ['Median', s.median], ['5th pct', s.p5], ['95th pct', s.p95], ['Chance of loss', s.prob_loss, '%'], ['Avg worst 5%', s.es5]];
    document.getElementById('geoKpis').innerHTML = k.map(([l,v,u]) => `<div class="panel p-4"><div class="text-[10px] muted">${l}</div><div class="metric-big mt-2 ${u?'':(v>=0?'green':'red')}" style="font-size:22px">${u ? Number(v).toFixed(0)+'%' : (v>=0?'+':'')+Number(v).toFixed(2)+'%'}</div></div>`).join('');
    document.getElementById('geoNarrative').innerHTML = (d.narrative||[]).map(p => { const m = String(p).match(/^([^:]{3,32}):\s(.*)$/); return `<p class="intel-p">${m?`<b>${iesc(m[1])}:</b> ${iesc(m[2])}`:iesc(p)}</p>`; }).join('');
    document.getElementById('geoChain').innerHTML = (d.scenario.chain||[]).map(c => `<li>${iesc(c)}</li>`).join('') || '<li class="muted">Custom shocks: no predefined chain.</li>';
    document.getElementById('geoAnalogues').innerText = d.scenario.analogues ? 'Historical analogues: ' + d.scenario.analogues : '';
    document.querySelector('#geoFactorTable tbody').innerHTML = (d.factor_table||[]).map(r => `<tr><td>${iesc(r.label)}${r.fixed?' <span class="chip hi">shocked</span>':''}</td>
        <td class="${(r.expected_move||0)>=0?'green':'red'}">${iesc(r.move_text)}</td><td>${r.beta==null?'–':Number(r.beta).toFixed(4)}</td><td class="muted">${r.t==null?'–':Number(r.t).toFixed(1)}</td>
        <td class="${(r.contribution||0)>=0?'green':'red'}">${r.contribution==null?'–':(r.contribution>=0?'+':'')+Number(r.contribution).toFixed(2)+'%'}</td></tr>`).join('');
    const e = d.histogram.edges, c = d.histogram.counts;
    const mids = c.map((_,i) => ((e[i]+e[i+1])/2));
    iChart('geoHist', {type:'bar', data:{labels: mids.map(v => v.toFixed(1)+'%'), datasets:[{data:c, backgroundColor: mids.map(v => v<0?ICOL.red:ICOL.green), borderWidth:0, barPercentage:1, categoryPercentage:1}]},
        options:{responsive:true, maintainAspectRatio:false, animation:false, plugins:{legend:{display:false}}, scales:iAxes({y:{grid:{color:ICOL.grid},ticks:{color:ICOL.tick},title:{display:true,text:'paths',color:ICOL.tick}}})}});
    const cb = Object.entries(d.contributions||{}).filter(([,v]) => v!=null);
    iChart('geoContrib', {type:'bar', data:{labels: cb.map(([k]) => ((d.factor_table.find(r=>r.factor===k)||{}).label||k)), datasets:[{data: cb.map(([,v]) => v), backgroundColor: cb.map(([,v]) => v>=0?ICOL.green:ICOL.red)}]},
        options:{indexAxis:'y', responsive:true, maintainAspectRatio:false, animation:false, plugins:{legend:{display:false}}, scales:iAxes({x:{grid:{color:ICOL.grid},ticks:{color:ICOL.tick,callback:v=>v+'%'}}})}});
    drawGeoNetwork(d.network);
}
function drawGeoNetwork(net){
    const box = document.getElementById('geoNetwork'); if(!net) { box.innerHTML=''; return; }
    const cols = {trigger:0, primary:1, secondary:2, asset:3};
    const byCol = [[],[],[],[]]; net.nodes.forEach(n => byCol[cols[n.kind]].push(n));
    const W = 980, rowH = 74, H = Math.max(260, Math.max(...byCol.map(c => c.length)) * rowH + 40);
    const X = [20, 270, 540, 800], NW = [190, 170, 170, 160], NH = 48;
    const pos = {};
    byCol.forEach((col, ci) => { const total = col.length * rowH; col.forEach((n, i) => { pos[n.id] = {x: X[ci], y: (H - total)/2 + i*rowH + (rowH-NH)/2, w: NW[ci]}; }); });
    const maxAsset = Math.max(0.01, ...net.edges.filter(e => e.to==='asset').map(e => Math.abs(e.value||0)));
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="min-width:760px" xmlns="http://www.w3.org/2000/svg">`;
    svg += `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>`;
    for (const e of net.edges) {
        const a = pos[e.from], b = pos[e.to]; if(!a || !b) continue;
        const x1 = a.x + a.w, y1 = a.y + NH/2, x2 = b.x, y2 = b.y + NH/2, mx = (x1+x2)/2;
        const v = Number(e.value||0); const col = v>=0 ? ICOL.green : ICOL.red;
        const w = e.to==='asset' ? 1.5 + 6*Math.abs(v)/maxAsset : 2;
        svg += `<path d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2-4},${y2}" fill="none" stroke="${col}" stroke-opacity=".55" stroke-width="${w.toFixed(1)}" marker-end="url(#arr)"/>`;
        svg += `<text x="${mx}" y="${(y1+y2)/2 - 4}" fill="${col}" font-size="10" text-anchor="middle">${iesc(e.label)}</text>`;
    }
    for (const n of net.nodes) {
        const p = pos[n.id];
        const stroke = n.kind==='trigger' ? ICOL.amber : n.kind==='asset' ? ICOL.cyan : n.kind==='primary' ? '#f59e0b' : '#334155';
        const mv = n.move_text ? `<text x="${p.x+10}" y="${p.y+37}" fill="${(n.move||0)>=0?ICOL.green:ICOL.red}" font-size="12" font-family="JetBrains Mono,monospace">${iesc(n.move_text)}${n.z!=null && n.kind!=='asset' ? ' ('+Number(n.z).toFixed(1)+'σ)' : ''}</text>` : '';
        const label = String(n.label); const short = label.length > 28 ? label.slice(0,27)+'…' : label;
        svg += `<rect x="${p.x}" y="${p.y}" width="${p.w}" height="${NH}" rx="10" fill="#0b1119" stroke="${stroke}" stroke-width="1.4"/>`;
        svg += `<text x="${p.x+10}" y="${p.y+18}" fill="#e7edf5" font-size="11" font-weight="600">${iesc(short)}</text>${mv}`;
    }
    ['Trigger','Shocked factors','Spillovers','Asset'].forEach((t,i) => { svg += `<text x="${X[i]}" y="14" fill="#64748b" font-size="9" letter-spacing="1.5">${t.toUpperCase()}</text>`; });
    box.innerHTML = svg + '</svg>';
}

/* ===================== Indicator Lab ===================== */
const IND_OVERLAYS = [
    ['ma','Moving averages',true], ['bb','Bollinger Bands',true], ['kc','Keltner Channels',false], ['dc','Donchian Channel',false],
    ['st','Supertrend',false], ['psar','Parabolic SAR',false], ['vwap','VWAP (rolling)',false], ['ichi','Ichimoku',false]];
INTEL.indOn = Object.fromEntries(IND_OVERLAYS.map(([k,,on]) => [k, on]));
function indOverlayControls(){
    document.getElementById('indOverlays').innerHTML = IND_OVERLAYS.map(([k,l]) => `<label class="intel-check"><input type="checkbox" ${INTEL.indOn[k]?'checked':''} onchange="INTEL.indOn['${k}']=this.checked;renderIndicators()"> ${l}</label>`).join('');
}
async function loadIndicators(force){
    const t = itk(); if(!t) return;
    indOverlayControls();
    const q = new URLSearchParams({period: document.getElementById('indPeriod').value, interval: document.getElementById('indInterval').value,
        sma: document.getElementById('indSma').value, ema: document.getElementById('indEma').value, fwd: document.getElementById('indFwd').value});
    ['bb_n','bb_k','st_n','st_mult','rsi_n','macd_fast','macd_slow','macd_signal','stoch_k','adx_n','cci_n','dc_n','kc_n','vwap_n','mfi_n','wr_n'].forEach(k => { const v = (document.getElementById('ind_'+k)||{}).value; if (v) q.set(k, v); });
    const key = t + '?' + q.toString();
    if (INTEL.ind && INTEL.ind._key === key && !force) return renderIndicators();
    const st = document.getElementById('indStatus'); st.innerText = 'COMPUTING…';
    try { INTEL.ind = await ifetch(`/api/indicators/${encodeURIComponent(t)}?` + q.toString()); INTEL.ind._key = key; st.innerText = INTEL.ind.dates.length + ' BARS'; renderIndicators(); }
    catch(e){ st.innerText = 'ERROR'; document.getElementById('indSummary').innerHTML = `<div class="text-sm red">${iesc(e.message)}</div>`; }
}
function renderIndicators(){
    const d = INTEL.ind; if(!d) return;
    const L = d.dates, O = d.overlays, S = d.oscillators;
    const ds = [{label:'Close', data:d.ohlc.close, borderColor:'#e7edf5', borderWidth:1.6, pointRadius:0}];
    const line = (label, data, color, dash, fill) => ds.push({label, data, borderColor:color, borderWidth:1.2, pointRadius:0, borderDash:dash||[], fill:fill||false, backgroundColor: fill ? 'rgba(34,211,238,.06)' : undefined, spanGaps:true});
    const palette = [ICOL.amber, ICOL.violet, ICOL.green, '#f472b6'];
    if (INTEL.indOn.ma) Object.keys(O).filter(k => /^(SMA|EMA) /.test(k)).forEach((k,i) => line(k, O[k], palette[i%4], k.startsWith('EMA')?[4,3]:[]));
    if (INTEL.indOn.bb) { line('BB upper', O['BB upper'], ICOL.cyan, [2,2]); line('BB lower', O['BB lower'], ICOL.cyan, [2,2], '-1'); line('BB middle', O['BB middle'], 'rgba(34,211,238,.5)', [1,3]); }
    if (INTEL.indOn.kc) { line('Keltner upper', O['Keltner upper'], '#f472b6', [5,3]); line('Keltner lower', O['Keltner lower'], '#f472b6', [5,3]); }
    if (INTEL.indOn.dc) { line('Donchian upper', O['Donchian upper'], ICOL.slate, [1,2]); line('Donchian lower', O['Donchian lower'], ICOL.slate, [1,2]); }
    if (INTEL.indOn.st) line('Supertrend', O['Supertrend'], ICOL.green);
    if (INTEL.indOn.vwap) line('VWAP (rolling)', O['VWAP (rolling)'], '#fb923c');
    if (INTEL.indOn.ichi) { line('Tenkan', O['Ichimoku tenkan'], '#60a5fa'); line('Kijun', O['Ichimoku kijun'], '#f87171'); line('Span A', O['Ichimoku span A'], 'rgba(52,211,153,.6)'); line('Span B', O['Ichimoku span B'], 'rgba(251,113,133,.6)', [], '-1'); }
    if (INTEL.indOn.psar) ds.push({label:'Parabolic SAR', data:O['Parabolic SAR'], showLine:false, pointRadius:1.6, pointBackgroundColor:ICOL.amber, borderColor:ICOL.amber});
    iChart('indPriceChart', {type:'line', data:{labels:L, datasets:ds}, options:{responsive:true, maintainAspectRatio:false, animation:false, interaction:{mode:'index', intersect:false},
        plugins:{legend:{labels:{color:'#8492a3', boxWidth:12, font:{size:10}}}}, scales:iAxes()}});

    const osc = document.getElementById('indOsc').value;
    const od = [], flat = (v, c) => od.push({label:String(v), data:L.map(() => v), borderColor:c||'#334155', borderWidth:1, pointRadius:0, borderDash:[3,3]});
    const add = (label, data, color, type) => od.push({label, data, borderColor:color, backgroundColor: type==='bar' ? (data||[]).map(v => (v||0)>=0?'rgba(25,213,139,.5)':'rgba(251,113,133,.5)') : undefined, type: type||'line', borderWidth:1.3, pointRadius:0, spanGaps:true});
    if (osc==='RSI') { add('RSI', S['RSI'], ICOL.amber); flat(70, ICOL.red); flat(30, ICOL.green); }
    if (osc==='MACD') { add('MACD', S['MACD'], ICOL.cyan); add('Signal', S['MACD signal'], ICOL.amber); add('Histogram', S['MACD histogram'], ICOL.slate, 'bar'); }
    if (osc==='Stochastic') { add('%K', S['Stochastic %K'], ICOL.cyan); add('%D', S['Stochastic %D'], ICOL.amber); flat(80, ICOL.red); flat(20, ICOL.green); }
    if (osc==='ADX / DMI') { add('ADX', S['ADX'], '#e7edf5'); add('+DI', S['+DI'], ICOL.green); add('-DI', S['-DI'], ICOL.red); flat(25); }
    if (osc==='CCI') { add('CCI', S['CCI'], ICOL.violet); flat(100, ICOL.red); flat(-100, ICOL.green); }
    if (osc==='Williams %R') { add('Williams %R', S['Williams %R'], ICOL.cyan); flat(-20, ICOL.red); flat(-80, ICOL.green); }
    if (osc==='MFI') { add('MFI', S['MFI'], ICOL.green); flat(80, ICOL.red); flat(20, ICOL.green); }
    if (osc==='OBV') add('OBV', S['OBV'], ICOL.cyan);
    if (osc==='ATR %') add('ATR % of price', S['ATR %'], ICOL.amber);
    iChart('indOscChart', {type:'line', data:{labels:L, datasets:od}, options:{responsive:true, maintainAspectRatio:false, animation:false, interaction:{mode:'index', intersect:false},
        plugins:{legend:{labels:{color:'#8492a3', boxWidth:12, font:{size:10}, filter: it => isNaN(Number(it.text))}}}, scales:iAxes({x:{display:false}})}});

    const sm = d.summary, b = d.baseline;
    const col = /bullish/i.test(sm.label) ? 'green' : /bearish/i.test(sm.label) ? 'red' : 'cyan';
    document.getElementById('indSummary').innerHTML = `<div class="metric-big ${col}">${iesc(sm.label)}</div>
        <div class="intel-bar mt-3" style="height:8px"><span style="width:${Math.round((sm.score+1)*50)}%"></span></div>
        <div class="flex gap-2 mt-3"><span class="chip pos">${sm.buy} bullish</span><span class="chip neu">${sm.neutral} neutral</span><span class="chip neg">${sm.sell} bearish</span></div>
        <p class="intel-p mt-4">Baseline: over ${b.bars} bars this asset rose ${b.hit_up}% of the time across ${b.forward_bars}-bar windows, averaging ${b.avg_fwd>=0?'+':''}${b.avg_fwd}%. A signal is only useful if its "edge" beats that.</p>`;
    const rel = (r, side) => r ? `<span class="${(r.edge||0)>0?'green':'muted'}">${r.avg_fwd>=0?'+':''}${r.avg_fwd}% · hit ${r.hit}% · edge ${r.edge>=0?'+':''}${r.edge}%</span> <span class="muted">(n=${r.n})</span>` : '<span class="muted">too few signals</span>';
    document.querySelector('#indSignals tbody').innerHTML = d.signals.map(s => `<tr><td>${iesc(s.indicator)}</td>
        <td><span class="chip ${s.signal==='BUY'?'pos':s.signal==='SELL'?'neg':'neu'}">${s.signal==='BUY'?'BULLISH':s.signal==='SELL'?'BEARISH':'NEUTRAL'}</span></td>
        <td class="muted" style="font-family:Inter,sans-serif">${iesc(s.reason)}</td><td>${rel(s.reliability.bullish)}</td><td>${rel(s.reliability.bearish)}</td></tr>`).join('');
    document.getElementById('indNote').innerText = d.note;
}
</script>
'''


def inject(page: str) -> str:
    """Return the dashboard page with the intelligence views wired in."""
    out = page
    if _NAV_OLD in out:
        out = out.replace(_NAV_OLD, _NAV_NEW, 1)
    out = re.sub(r'\n\s*<div id="view-news" class="view-section space-y-6">.*?\n        </div>\n',
                 lambda m: "\n" + HTML_NEWS, out, count=1, flags=re.S)
    out = out.replace("</style>", CSS + "\n</style>", 1)
    idx = out.rfind("</body>")
    if idx != -1:
        out = out[:idx] + JS + "\n" + out[idx:]
    return out


install(app, _sys.modules[__name__])


@app.get("/", response_class=HTMLResponse)
def render_dashboard():
    return inject(_dashboard_base())


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
/* Portfolio customizer */
.cust-slider { accent-color:#f59e0b; width:100%; min-width:110px; cursor:pointer; }
.cust-num { padding:5px 7px !important; font-family:"JetBrains Mono",monospace; font-size:12px; text-align:right; }
.cust-bad { border-color:#fb7185 !important; color:#fb7185 !important; }
.cust-table td { padding:7px 8px; vertical-align:middle; }
.cust-table tr.cust-locked td { background:rgba(245,158,11,.05); }
.cust-chip { border:1px solid #263444; background:#0b141d; color:#c5cfdb; border-radius:999px; padding:5px 12px; font-size:12px; font-weight:600; cursor:pointer; }
.cust-chip:hover { border-color:#f59e0b; color:#fbbf24; }
.cust-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.cust-compare th, .cust-compare td { padding:7px 8px; }
.cust-input-off { opacity:.4; }
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
                            <option value="CUSTOM">Custom Allocation</option>
                            <option value="CONSTRAINED">Custom Optimizer (your limits)</option>
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
                <div id="customPortfolioControls" class="panel p-4 mt-4" style="display:none">
                    <div class="text-sm font-semibold">Your custom allocation</div>
                    <div class="text-xs muted mt-1">Enter weights in percentages, e.g. AAPL:40,MSFT:60. Include each selected asset. Total must be 100%.</div>
                    <input id="portfolioCustomWeights" class="trade-input mt-3" type="text" value="AAPL:20,MSFT:20,NVDA:20,GOOGL:20,AMZN:20" placeholder="AAPL:40,MSFT:60" oninput="scheduleCustomPortfolioUpdate()">
                    <div id="customPortfolioTotal" class="text-xs mono muted mt-2">Total: 100.00%</div>
                </div>
                <div class="flex flex-wrap gap-2 mt-4">
                    <button type="button" class="btn px-5 py-2.5" onclick="fetchPortfolioOptimizer()">Calculate Portfolio →</button>
                    <button type="button" class="btn px-4 py-2.5" onclick="useCurrentTickerForPortfolio()">Use current company</button>
                </div>
                <div class="text-[10px] muted mt-4 leading-5">
                    Minimum Risk minimizes estimated variance. Tangency maximizes estimated Sharpe ratio. The optimizer uses all valid assets you enter, with weights constrained to 0–100% and total weight = 100%. Use the Portfolio Customizer below to set your own weights or your own limits and objective. This is a quantitative scenario model, not an order or personalized investment instruction.
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

            <section id="portfolioCustomizer" class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-start justify-between gap-3">
                    <div>
                        <div class="font-semibold text-lg">Portfolio Customizer</div>
                        <div class="text-[10px] muted mono mt-1">YOUR WEIGHTS AND LIMITS BESIDE THE AUTOMATIC MIN-RISK / TANGENCY RESULTS · LIVE RE-PRICING</div>
                    </div>
                    <span id="custStatus" class="badge">WAITING FOR A RUN</span>
                </div>

                <div class="text-sm font-semibold mt-5">1 · Optimize with your own limits</div>
                <div class="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3 mt-3">
                    <label class="text-xs muted col-span-2 md:col-span-1 xl:col-span-2">Objective
                        <select id="custObjective" class="trade-input mt-1" onchange="custObjectiveChanged()">
                            <option value="MAX_SHARPE">Max Sharpe within limits</option>
                            <option value="MIN_RISK">Minimum risk within limits</option>
                            <option value="TARGET_RETURN">Reach a target return at lowest risk</option>
                            <option value="TARGET_VOL">Highest return under a volatility cap</option>
                            <option value="UTILITY">Risk-aversion utility (return − λ/2 · variance)</option>
                            <option value="RISK_PARITY">Risk parity (equal risk contributions)</option>
                            <option value="MAX_DIVERSIFICATION">Maximum diversification</option>
                        </select>
                    </label>
                    <label class="text-xs muted">Min weight / asset %
                        <input id="custMinW" class="trade-input mt-1" type="number" min="0" max="100" step="0.5" placeholder="0">
                    </label>
                    <label class="text-xs muted">Max weight / asset %
                        <input id="custMaxW" class="trade-input mt-1" type="number" min="0" max="100" step="0.5" placeholder="100">
                    </label>
                    <label id="custTargetRetWrap" class="text-xs muted">Target return %
                        <input id="custTargetRet" class="trade-input mt-1" type="number" step="0.5" placeholder="e.g. 15">
                    </label>
                    <label id="custTargetVolWrap" class="text-xs muted">Volatility cap %
                        <input id="custTargetVol" class="trade-input mt-1" type="number" min="0" step="0.5" placeholder="e.g. 20">
                    </label>
                    <label id="custLambdaWrap" class="text-xs muted">Risk aversion λ
                        <input id="custLambda" class="trade-input mt-1" type="number" min="0.1" step="0.5" value="3">
                    </label>
                </div>
                <div class="flex flex-wrap items-center gap-2 mt-3">
                    <button type="button" class="btn px-5 py-2.5" onclick="runConstrainedPortfolio()">Optimize with my limits →</button>
                    <button type="button" class="cust-chip" onclick="custClearLimits()">Clear limits</button>
                    <span class="text-[10px] muted">Per-asset Min/Max in the table override the global limits. Locked assets keep your weight while the rest is optimized.</span>
                </div>
                <div id="custConstraintNote" class="text-xs muted mt-3 leading-5"></div>

                <div class="text-sm font-semibold mt-6">2 · Set weights yourself</div>
                <div class="flex flex-wrap items-center gap-2 mt-3">
                    <span class="text-xs muted">Start from:</span>
                    <button type="button" class="cust-chip" onclick="custSeed('min')">Min Risk</button>
                    <button type="button" class="cust-chip" onclick="custSeed('tan')">Tangency</button>
                    <button type="button" class="cust-chip" id="custSeedCon" onclick="custSeed('con')">Constrained</button>
                    <button type="button" class="cust-chip" onclick="custSeed('eq')">Equal weight</button>
                    <button type="button" class="cust-chip" onclick="custSeed('blend')">50/50 Min Risk + Tangency</button>
                    <span class="mx-1 muted">|</span>
                    <button type="button" class="cust-chip" onclick="custNormalize()">Normalize to 100%</button>
                    <label class="text-xs muted flex items-center gap-2 ml-1"><input id="custAutoBalance" type="checkbox" checked> Auto-balance other weights</label>
                    <button type="button" class="btn px-4 py-2 ml-auto" onclick="custSendToServer()">Analyse my mix in the main view →</button>
                </div>

                <div class="mt-4">
                    <div class="table-wrap">
                        <table id="custTable" class="cust-table">
                            <thead><tr><th>Asset</th><th>Min Risk</th><th>Tangency</th><th>Constrained</th><th style="text-align:left">Your weight</th><th>Lock</th><th>Min %</th><th>Max %</th><th>Risk share</th></tr></thead>
                            <tbody><tr><td colspan="9" class="muted">Run the optimizer to load assets.</td></tr></tbody>
                            <tfoot><tr><td><strong>Total</strong></td><td id="custTotMin" class="mono">--</td><td id="custTotTan" class="mono">--</td><td id="custTotCon" class="mono">--</td><td id="custTotal" class="mono" style="text-align:left">--</td><td></td><td id="custTotLo" class="mono">--</td><td id="custTotHi" class="mono">--</td><td class="mono">100%</td></tr></tfoot>
                        </table>
                    </div>
                    <div class="grid grid-cols-1 xl:grid-cols-2 gap-5 mt-5">
                        <div>
                            <div class="text-sm font-semibold">Your mix vs the automatic portfolios</div>
                            <div class="table-wrap mt-2"><table id="custCompare" class="cust-compare"><tbody><tr><td class="muted">Live metrics appear after the first run.</td></tr></tbody></table></div>
                            <div id="custVerdict" class="text-xs muted leading-5 mt-3"></div>
                        </div>
                        <div>
                            <div class="text-sm font-semibold">Weights side by side</div>
                            <div id="custWeightsChartWrap" class="mt-2" style="height:280px"><canvas id="custWeightsChart"></canvas></div>
                        </div>
                    </div>
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
    document.getElementById('portfolioStatus').innerText=(data.method||'OPTIMIZER')+(data.constraints?' · '+data.constraints.objective:'')+' · '+(data.assets_used||0)+' ASSETS';
    document.getElementById('portReturn').innerText=(Number(s.expected_return||0)*100).toFixed(2)+'%';
    document.getElementById('portVol').innerText=(Number(s.volatility||0)*100).toFixed(2)+'%';
    document.getElementById('portVar').innerText=Number(s.variance||0).toFixed(6);
    document.getElementById('portSharpe').innerText=Number(s.sharpe||0).toFixed(3);
    document.getElementById('portBeta').innerText=Number(s.beta||0).toFixed(3);
    document.getElementById('portfolioMethodLabel').innerText=(data.method==='MIN_RISK'?'MINIMUM-VARIANCE WEIGHTS':data.method==='TANGENCY'?'TANGENCY / MAX-SHARPE WEIGHTS':data.method==='CUSTOM'?'YOUR CUSTOM WEIGHTS':data.method==='CONSTRAINED'?('CUSTOM OPTIMIZER · '+String((data.constraints||{}).objective_label||'').toUpperCase()):'COMPARE MODE · SELECTED TABLE = MINIMUM-RISK WEIGHTS');
    document.getElementById('portfolioAssetCount').innerText=Number(data.assets_used||0);
    document.getElementById('portfolioWeightSum').innerText=(Number(s.weight_sum||0)*100).toFixed(2)+'%';

    const rows=data.rows||[];
    const hasCon=!!data.constrained;
    document.querySelector('#portfolioTable thead').innerHTML='<tr><th>Asset</th><th>Weight</th><th>Min Risk</th><th>Tangency</th>'+(hasCon?'<th>Constrained</th>':'')+'<th>Exp Return</th><th>Volatility</th><th>Beta</th><th>Asset Sharpe</th><th>Return Contribution</th><th>Risk Contribution</th></tr>';
    const tbody=document.querySelector('#portfolioTable tbody');
    tbody.innerHTML=rows.map(r=>`<tr>
        <td><strong>${custEsc(r.ticker)}</strong><div class="text-[10px] muted">${custEsc(r.name||'')}</div></td>
        <td class="cyan"><strong>${(Number(r.weight)*100).toFixed(2)}%</strong></td>
        <td>${(Number(r.min_risk_weight)*100).toFixed(2)}%</td>
        <td>${(Number(r.tangency_weight)*100).toFixed(2)}%</td>${hasCon?`<td>${(Number(r.constrained_weight)*100).toFixed(2)}%</td>`:''}
        <td>${(Number(r.expected_return)*100).toFixed(2)}%</td>
        <td>${(Number(r.volatility)*100).toFixed(2)}%</td>
        <td>${Number(r.beta).toFixed(3)}</td>
        <td>${Number(r.asset_sharpe).toFixed(3)}</td>
        <td>${(Number(r.return_contribution)*100).toFixed(2)}%</td>
        <td>${(Number(r.risk_contribution)*100).toFixed(2)}%</td>
    </tr>`).join('') || '<tr><td colspan="11" class="muted">No valid assets.</td></tr>';

    const comp=document.getElementById('portfolioMethodComparison');
    const card=(title,m,active)=>`<div class="scenario ${active?'border border-emerald-800':''}"><div class="text-[10px] muted">${title}</div><div class="grid grid-cols-2 gap-2 mt-3 text-xs"><div><span class="muted">Return</span><div class="mono mt-1">${(Number(m.expected_return||0)*100).toFixed(2)}%</div></div><div><span class="muted">Volatility</span><div class="mono mt-1">${(Number(m.volatility||0)*100).toFixed(2)}%</div></div><div><span class="muted">Sharpe</span><div class="mono mt-1">${Number(m.sharpe||0).toFixed(3)}</div></div><div><span class="muted">Beta</span><div class="mono mt-1">${Number(m.beta||0).toFixed(3)}</div></div></div></div>`;
    comp.innerHTML=card('Minimum Risk',data.minimum_risk,data.method==='MIN_RISK')+card('Tangency / Max Sharpe',data.tangency,data.method==='TANGENCY')
        +(data.constrained?card('Custom Optimizer · '+custEsc((data.constraints||{}).objective||''),data.constrained,data.method==='CONSTRAINED'):'')
        +(data.equal_weight?card('Equal Weight (reference)',data.equal_weight,false):'');

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
            {label:'CAL',data:(data.cal.volatility||[]).map((v,i)=>({x:Number(v)*100,y:Number(data.cal.return[i])*100})),borderWidth:2,pointRadius:0,borderColor:'#22d3ee',backgroundColor:'#22d3ee'},
            {type:'scatter',label:'Selected portfolio',data:[{x:Number(s.volatility)*100,y:Number(s.expected_return)*100}],pointRadius:7,backgroundColor:'#f472b6',borderColor:'#f472b6'},
            {type:'scatter',label:'Tangency portfolio',data:[{x:Number(data.tangency.volatility)*100,y:Number(data.tangency.expected_return)*100}],pointRadius:6,backgroundColor:CUST_COLORS.tan,borderColor:CUST_COLORS.tan},
            {type:'scatter',label:'Your mix',data:[],pointRadius:7,pointStyle:'rectRot',backgroundColor:CUST_COLORS.mine,borderColor:CUST_COLORS.mine}
        ]},options:{responsive:true,maintainAspectRatio:false,scales:{x:{type:'linear',title:{display:true,text:'Volatility (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{title:{display:true,text:'Expected Return (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}
    });

    // Efficient frontier, with selected point
    if(portfolioFrontierChartInstance) portfolioFrontierChartInstance.destroy();
    portfolioFrontierChartInstance=new Chart(document.getElementById('portfolioFrontierChart'),{
        type:'line',data:{datasets:[
            {label:'Efficient frontier',data:(data.efficient_frontier||[]).map(p=>({x:Number(p.volatility)*100,y:Number(p.return)*100})),borderWidth:2,pointRadius:0,tension:.15,borderColor:'#22d3ee',backgroundColor:'#22d3ee'},
            {type:'scatter',label:'Selected portfolio',data:[{x:Number(s.volatility)*100,y:Number(s.expected_return)*100}],pointRadius:7,backgroundColor:'#f472b6',borderColor:'#f472b6'},
            {type:'scatter',label:'Min Risk',data:[{x:Number(data.minimum_risk.volatility)*100,y:Number(data.minimum_risk.expected_return)*100}],pointRadius:5,backgroundColor:CUST_COLORS.min,borderColor:CUST_COLORS.min},
            {type:'scatter',label:'Tangency',data:[{x:Number(data.tangency.volatility)*100,y:Number(data.tangency.expected_return)*100}],pointRadius:5,backgroundColor:CUST_COLORS.tan,borderColor:CUST_COLORS.tan},
            ...(data.constrained?[{type:'scatter',label:'Constrained',data:[{x:Number(data.constrained.volatility)*100,y:Number(data.constrained.expected_return)*100}],pointRadius:6,pointStyle:'triangle',backgroundColor:CUST_COLORS.con,borderColor:CUST_COLORS.con}]:[]),
            {type:'scatter',label:'Your mix',data:[],pointRadius:7,pointStyle:'rectRot',backgroundColor:CUST_COLORS.mine,borderColor:CUST_COLORS.mine}
        ]},options:{responsive:true,maintainAspectRatio:false,scales:{x:{type:'linear',title:{display:true,text:'Volatility (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{title:{display:true,text:'Expected Return (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}}}}
    });

    custBuild(data);
}

let customPortfolioTimer=null;
function scheduleCustomPortfolioUpdate(){
    clearTimeout(customPortfolioTimer);
    const input=document.getElementById('portfolioCustomWeights');
    const values=(input?.value||'').split(',').filter(x=>x.trim()).map(x=>Number(x.split(':')[1]));
    const total=values.reduce((a,b)=>a+b,0);
    const label=document.getElementById('customPortfolioTotal');
    if(label) label.innerText='Total: '+(Number.isFinite(total)?total.toFixed(2):'Invalid')+'%';
    if(values.length && values.every(Number.isFinite) && Math.abs(total-100)<0.01)
        customPortfolioTimer=setTimeout(fetchPortfolioOptimizer,650);
}
document.getElementById('portfolioMethod')?.addEventListener('change',()=>{
    document.getElementById('customPortfolioControls').style.display=document.getElementById('portfolioMethod').value==='CUSTOM'?'block':'none';
});
async function fetchPortfolioOptimizer(){
    const raw=(document.getElementById('portfolioTickers')?.value||'').trim();
    if(!raw){alert('Enter at least one asset ticker.');return;}
    const method=document.getElementById('portfolioMethod').value;
    const lookback=document.getElementById('portfolioLookback').value;
    const url=API_BASE+'/api/portfolio/optimize?tickers='+encodeURIComponent(raw)+'&method='+encodeURIComponent(method)+'&lookback='+encodeURIComponent(lookback)+(method==='CUSTOM'?'&custom_weights='+encodeURIComponent(document.getElementById('portfolioCustomWeights').value):'')+(method==='CONSTRAINED'?custConstraintParams():'');
    const status=document.getElementById('portfolioStatus');
    status.innerText='LOADING…';
    const custBadge=document.getElementById('custStatus');
    if(method==='CONSTRAINED'&&custBadge){custBadge.className='badge amber';custBadge.innerText='OPTIMIZING…';}
    try{const res=await fetch(url);const data=await res.json();if(!res.ok)throw new Error(data.detail||'Portfolio optimization failed');renderPortfolioOptimizer(data);}catch(err){console.error(err);status.innerText='ERROR';if(method==='CONSTRAINED'&&custBadge){custBadge.className='badge red';custBadge.innerText='CHECK YOUR LIMITS';const n=document.getElementById('custConstraintNote');if(n)n.innerHTML='<span class="red">'+custEsc(err.message||'Optimization failed')+'</span>';}alert(err.message||'Portfolio optimizer could not load data.');}
}

// ---------------------------------------------------------------------------
// PORTFOLIO CUSTOMIZER
// Works next to the automatic Min-Risk / Tangency results. The server sends the
// model inputs (expected returns, covariance, betas), so any weight mix is
// re-priced in the browser instantly while you drag a slider.
// ---------------------------------------------------------------------------
const CUST_COLORS={mine:'#f59e0b',min:'#19d58b',tan:'#38bdf8',con:'#a78bfa',eq:'#94a3b8'};
let custState={key:'',tickers:[],names:{},w:{},lock:{},lo:{},hi:{}};
let custWeightsChartInstance=null;

function custEsc(v){return String(v==null?'':v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function custPct(x,d=2){return Number.isFinite(x)?(x*100).toFixed(d)+'%':'--';}
function custModel(){return portfolioData&&portfolioData.model||null;}
function custNum(v){const s=String(v==null?'':v).trim();if(s==='')return null;const n=Number(s);return Number.isFinite(n)?n:null;}

function custAutoWeights(src){
    // Returns {ticker: percent} for an automatic portfolio.
    const T=custState.tickers, out={};
    const byT={}; (portfolioData?.rows||[]).forEach(r=>byT[r.ticker]=r);
    T.forEach(t=>{
        const r=byT[t]||{};
        let w=0;
        if(src==='min') w=Number(r.min_risk_weight||0);
        else if(src==='tan') w=Number(r.tangency_weight||0);
        else if(src==='con') w=Number(portfolioData?.constrained?.weights?.[t]||0);
        else if(src==='eq') w=1/T.length;
        else if(src==='blend') w=0.5*Number(r.min_risk_weight||0)+0.5*Number(r.tangency_weight||0);
        else w=Number(r.weight||0);
        out[t]=w*100;
    });
    return out;
}

function custSeed(src, silent){
    if(!custState.tickers.length){if(!silent)alert('Run the optimizer first to load assets.');return;}
    if(src==='con'&&!portfolioData?.constrained){if(!silent)alert('Run "Optimize with my limits" first to get a constrained portfolio.');return;}
    const w=custAutoWeights(src);
    custState.tickers.forEach(t=>{custState.w[t]=w[t];});
    custRefresh();
}

function custNormalize(){
    const T=custState.tickers; if(!T.length)return;
    const tot=T.reduce((a,t)=>a+Number(custState.w[t]||0),0);
    T.forEach(t=>{custState.w[t]=tot>0?Number(custState.w[t]||0)/tot*100:100/T.length;});
    custRefresh();
}

function custSetWeight(t, v){
    const T=custState.tickers, W=custState.w, L=custState.lock;
    v=Math.max(0,Math.min(100,Number(v)||0));
    if(!document.getElementById('custAutoBalance')?.checked){W[t]=v;custRefresh();return;}
    const others=T.filter(x=>x!==t&&!L[x]);
    const lockedSum=T.filter(x=>x!==t&&L[x]).reduce((a,x)=>a+Number(W[x]||0),0);
    const room=Math.max(0,100-lockedSum);
    if(!others.length){W[t]=room;custRefresh();return;}      // nothing free to rebalance against
    v=Math.min(v,room);
    W[t]=v;
    // Scale the free assets proportionally so the total stays exactly 100%.
    const target=room-v, cur=others.reduce((a,x)=>a+Number(W[x]||0),0);
    others.forEach(x=>{W[x]=cur>1e-12?Number(W[x]||0)*target/cur:target/others.length;});
    custRefresh();
}

function custBoundsFor(t){
    const gMin=custNum(document.getElementById('custMinW')?.value), gMax=custNum(document.getElementById('custMaxW')?.value);
    const lo=custNum(custState.lo[t]), hi=custNum(custState.hi[t]);
    return {lo:lo!=null?lo:(gMin!=null?gMin:0), hi:hi!=null?hi:(gMax!=null?gMax:100)};
}

function custMineFrac(){
    const T=custState.tickers, raw=T.map(t=>Math.max(0,Number(custState.w[t]||0)));
    const s=raw.reduce((a,b)=>a+b,0);
    return s>0?raw.map(x=>x/s):raw.map(()=>1/T.length);
}

function custFrontierGap(vol, ret){
    // Return you give up versus the efficient frontier at the same volatility.
    const pts=(portfolioData?.efficient_frontier||[]).slice().sort((a,b)=>a.volatility-b.volatility);
    let best=null;
    for(const p of pts){ if(p.volatility<=vol+1e-9) best=(best==null?p.return:Math.max(best,p.return)); }
    [portfolioData?.minimum_risk,portfolioData?.tangency,portfolioData?.constrained].forEach(p=>{
        if(p&&Number(p.volatility)<=vol+1e-9) best=(best==null?Number(p.expected_return):Math.max(best,Number(p.expected_return)));
    });
    return best==null?0:Math.max(0,best-ret);
}

function custStats(w){
    const m=custModel(), n=w.length, rf=Number(portfolioData.risk_free_rate||0);
    let ret=0,beta=0,wsig=0,hh=0; const cw=new Array(n).fill(0);
    for(let i=0;i<n;i++){
        ret+=w[i]*m.mu[i]; beta+=w[i]*m.beta[i]; wsig+=w[i]*Math.sqrt(Math.max(m.cov[i][i],0)); hh+=w[i]*w[i];
        let acc=0; for(let j=0;j<n;j++) acc+=m.cov[i][j]*w[j]; cw[i]=acc;
    }
    let v=0; for(let i=0;i<n;i++) v+=w[i]*cw[i];
    const vol=Math.sqrt(Math.max(v,0));
    return {ret,vol,variance:v,sharpe:vol>0?(ret-rf)/vol:0,beta,effN:hh>0?1/hh:0,maxW:Math.max(...w),
            divRatio:vol>0?wsig/vol:0,rc:cw.map((c,i)=>v>0?w[i]*c/v:0),gap:custFrontierGap(vol,ret)};
}

function custPortfolios(){
    const T=custState.tickers, frac=src=>{const w=custAutoWeights(src);return T.map(t=>w[t]/100);};
    const cols=[{key:'mine',label:'Your mix',w:custMineFrac()},{key:'min',label:'Min Risk',w:frac('min')},{key:'tan',label:'Tangency',w:frac('tan')}];
    if(portfolioData?.constrained) cols.push({key:'con',label:'Constrained',w:frac('con')});
    cols.push({key:'eq',label:'Equal',w:frac('eq')});
    cols.forEach(c=>c.s=custStats(c.w));
    return cols;
}

function custBuild(data){
    const m=data.model;
    const badge=document.getElementById('custStatus');
    if(!m||!m.tickers||!m.tickers.length){if(badge)badge.innerText='NO MODEL DATA';return;}
    const key=m.tickers.join(',');
    const fresh=custState.key!==key;
    const prev=custState;
    custState={key,tickers:m.tickers.slice(),names:{},w:{},lock:{},lo:{},hi:{}};
    m.tickers.forEach((t,i)=>{
        custState.names[t]=(m.names||[])[i]||t;
        custState.lock[t]=!!prev.lock[t];
        custState.lo[t]=prev.lo[t]!=null?prev.lo[t]:'';
        custState.hi[t]=prev.hi[t]!=null?prev.hi[t]:'';
        custState.w[t]=fresh?0:Number(prev.w[t]||0);
    });
    if(fresh) custSeed(data.method==='CONSTRAINED'?'con':'sel',true);

    const auto={min:custAutoWeights('min'),tan:custAutoWeights('tan'),con:data.constrained?custAutoWeights('con'):null};
    const tbody=document.querySelector('#custTable tbody');
    tbody.innerHTML=custState.tickers.map((t,i)=>`<tr data-i="${i}">
        <td><strong>${custEsc(t)}</strong><div class="text-[10px] muted">${custEsc(custState.names[t])}</div></td>
        <td>${auto.min[t].toFixed(2)}%</td>
        <td>${auto.tan[t].toFixed(2)}%</td>
        <td>${auto.con?auto.con[t].toFixed(2)+'%':'<span class="muted">—</span>'}</td>
        <td style="text-align:left;min-width:220px"><div class="flex items-center gap-2">
            <input type="range" min="0" max="100" step="0.25" data-role="slider" class="cust-slider" aria-label="${custEsc(t)} weight">
            <input type="number" min="0" max="100" step="0.1" data-role="num" class="trade-input cust-num" style="width:78px" aria-label="${custEsc(t)} weight %"></div></td>
        <td style="text-align:center"><input type="checkbox" data-role="lock" aria-label="Lock ${custEsc(t)}"></td>
        <td><input type="number" min="0" max="100" step="0.5" data-role="lo" class="trade-input cust-num" style="width:64px"></td>
        <td><input type="number" min="0" max="100" step="0.5" data-role="hi" class="trade-input cust-num" style="width:64px"></td>
        <td data-role="rc">--</td>
    </tr>`).join('');
    tbody.querySelectorAll('tr[data-i]').forEach(tr=>{
        const t=custState.tickers[Number(tr.dataset.i)];
        tr.querySelector('[data-role=lock]').checked=custState.lock[t];
        tr.querySelector('[data-role=lo]').value=custState.lo[t];
        tr.querySelector('[data-role=hi]').value=custState.hi[t];
    });
    document.getElementById('custTotMin').innerText='100.00%';
    document.getElementById('custTotTan').innerText='100.00%';
    document.getElementById('custTotCon').innerText=data.constrained?'100.00%':'—';
    document.getElementById('custSeedCon').style.opacity=data.constrained?'1':'.45';

    // Weights chart: automatic portfolios next to yours.
    const wrap=document.getElementById('custWeightsChartWrap');
    if(wrap) wrap.style.height=Math.max(260,Math.min(900,custState.tickers.length*(data.constrained?34:28)+70))+'px';
    if(custWeightsChartInstance) custWeightsChartInstance.destroy();
    const ds=[
        {label:'Min Risk',data:custState.tickers.map(t=>auto.min[t]),backgroundColor:CUST_COLORS.min},
        {label:'Tangency',data:custState.tickers.map(t=>auto.tan[t]),backgroundColor:CUST_COLORS.tan},
    ];
    if(auto.con) ds.push({label:'Constrained',data:custState.tickers.map(t=>auto.con[t]),backgroundColor:CUST_COLORS.con});
    ds.push({label:'Your mix',data:[],backgroundColor:CUST_COLORS.mine});
    custWeightsChartInstance=new Chart(document.getElementById('custWeightsChart'),{type:'bar',
        data:{labels:custState.tickers,datasets:ds},
        options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,animation:{duration:150},
            plugins:{legend:{labels:{color:'#c5cfdb',boxWidth:12}},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${Number(c.raw).toFixed(2)}%`}}},
            scales:{x:{min:0,suggestedMax:50,title:{display:true,text:'Weight (%)'},ticks:{color:'#64748b'},grid:{color:'#17212d'}},y:{ticks:{color:'#c5cfdb'},grid:{display:false}}}}});

    custRenderConstraintNote(data);
    if(badge){badge.className='badge green';badge.innerText=data.method==='CONSTRAINED'?'CONSTRAINED RESULT READY':'READY · DRAG TO CUSTOMIZE';}
    custRefresh();
}

function custRenderConstraintNote(data){
    const el=document.getElementById('custConstraintNote'); if(!el)return;
    const c=data.constraints;
    if(!c){el.innerHTML='Set limits and an objective, then press <strong>Optimize with my limits</strong>. The result appears as the <span style="color:'+CUST_COLORS.con+'">Constrained</span> column and as a triangle on the efficient frontier.';return;}
    const r=c.achievable_return_range||[];
    const lines=[`<strong>${custEsc(c.objective_label)}</strong> · solver ${custEsc(c.solver)}`
        +(c.target_return!=null?` · target return ${custPct(c.target_return)}`:'')
        +(c.target_vol!=null?` · volatility cap ${custPct(c.target_vol)}`:'')
        +(c.objective==='UTILITY'?` · λ = ${Number(c.risk_aversion).toFixed(2)}`:'')
        +(r.length===2?` · returns reachable within your limits: ${custPct(r[0])} to ${custPct(r[1])}`:'')];
    if((c.binding||[]).length) lines.push('Limits that bind: '+c.binding.map(custEsc).join(', ')+'.');
    (c.notes||[]).forEach(n=>lines.push('<span style="color:#fbbf24">'+custEsc(n)+'</span>'));
    el.innerHTML=lines.map(l=>'<div>'+l+'</div>').join('');
}

function custRefresh(){
    if(!custModel()||!custState.tickers.length)return;
    const T=custState.tickers, active=document.activeElement;
    const cols=custPortfolios(), mine=cols[0].s;
    const total=T.reduce((a,t)=>a+Number(custState.w[t]||0),0);
    let loSum=0,hiSum=0;
    document.querySelectorAll('#custTable tbody tr[data-i]').forEach(tr=>{
        const i=Number(tr.dataset.i), t=T[i], w=Number(custState.w[t]||0), b=custBoundsFor(t);
        loSum+=b.lo; hiSum+=Math.min(b.hi,100);
        const sl=tr.querySelector('[data-role=slider]'), nm=tr.querySelector('[data-role=num]');
        if(sl!==active) sl.value=w;
        if(nm!==active) nm.value=w.toFixed(2);
        const bad=w<b.lo-0.005||w>b.hi+0.005;
        nm.classList.toggle('cust-bad',bad);
        nm.title=bad?`Outside your ${b.lo}–${b.hi}% limit`:'';
        tr.classList.toggle('cust-locked',!!custState.lock[t]);
        const g=custNum(document.getElementById('custMinW')?.value), h=custNum(document.getElementById('custMaxW')?.value);
        tr.querySelector('[data-role=lo]').placeholder=g!=null?g:'0';
        tr.querySelector('[data-role=hi]').placeholder=h!=null?h:'100';
        const rc=mine.rc[i];
        tr.querySelector('[data-role=rc]').innerHTML=`<span class="${rc>0.5?'red':''}">${custPct(rc,1)}</span>`;
    });
    const totEl=document.getElementById('custTotal');
    const ok=Math.abs(total-100)<0.01;
    totEl.innerHTML=`<strong class="${ok?'green':'red'}">${total.toFixed(2)}%</strong>`+(ok?'':' <span class="text-[10px] muted">metrics use weights rescaled to 100%</span>');
    document.getElementById('custTotLo').innerHTML=`<span class="${loSum>100.001?'red':''}">${loSum.toFixed(1)}%</span>`;
    document.getElementById('custTotHi').innerHTML=`<span class="${hiSum<99.999?'red':''}">${hiSum.toFixed(1)}%</span>`;

    // Live comparison table: your mix vs the automatic portfolios.
    const best={ret:Math.max(...cols.map(c=>c.s.ret)),vol:Math.min(...cols.map(c=>c.s.vol)),sharpe:Math.max(...cols.map(c=>c.s.sharpe))};
    const rowsDef=[
        ['Expected return',c=>custPct(c.s.ret),c=>Math.abs(c.s.ret-best.ret)<1e-9],
        ['Volatility',c=>custPct(c.s.vol),c=>Math.abs(c.s.vol-best.vol)<1e-9],
        ['Sharpe',c=>c.s.sharpe.toFixed(3),c=>Math.abs(c.s.sharpe-best.sharpe)<1e-9],
        ['Beta',c=>c.s.beta.toFixed(3)],
        ['Effective # of assets',c=>c.s.effN.toFixed(1)],
        ['Largest weight',c=>custPct(c.s.maxW,1)],
        ['Diversification ratio',c=>c.s.divRatio.toFixed(2)],
        ['Return gap vs frontier',c=>c.s.gap<0.0005?'on frontier':'−'+custPct(c.s.gap)],
    ];
    document.getElementById('custCompare').innerHTML=
        '<thead><tr><th>Metric</th>'+cols.map(c=>`<th><span class="cust-dot" style="background:${CUST_COLORS[c.key]}"></span>${c.label}</th>`).join('')+'</tr></thead><tbody>'
        +rowsDef.map(([label,f,isBest])=>'<tr><td>'+label+'</td>'+cols.map(c=>`<td class="${c.key==='mine'?'':'muted'}" style="${c.key==='mine'?'color:'+CUST_COLORS.mine+';font-weight:700':''}">${f(c)}${isBest&&isBest(c)?' ★':''}</td>`).join('')+'</tr>').join('')+'</tbody>';

    const tan=cols.find(c=>c.key==='tan').s, mn=cols.find(c=>c.key==='min').s;
    const v=[];
    v.push(`Your mix: ${custPct(mine.ret)} expected return at ${custPct(mine.vol)} volatility (Sharpe ${mine.sharpe.toFixed(2)} vs ${tan.sharpe.toFixed(2)} for Tangency).`);
    if(mine.gap>=0.0005) v.push(`At the same risk the efficient frontier offers about ${custPct(mine.gap)} more return a year.`);
    else v.push('Your mix sits on (or very near) the efficient frontier.');
    if(mine.vol<mn.vol-1e-6) v.push('Lower volatility than Min Risk is not possible with these assets; check the weights.');
    const topI=mine.rc.indexOf(Math.max(...mine.rc));
    if(topI>=0&&mine.rc[topI]>0.4) v.push(`${custEsc(T[topI])} drives ${custPct(mine.rc[topI],0)} of your portfolio risk.`);
    document.getElementById('custVerdict').innerHTML=v.join(' ');

    // Charts: move the "Your mix" markers.
    if(custWeightsChartInstance){
        const d=custWeightsChartInstance.data.datasets.find(x=>x.label==='Your mix');
        if(d){d.data=T.map(t=>Number(custState.w[t]||0));custWeightsChartInstance.update('none');}
    }
    const pt=[{x:mine.vol*100,y:mine.ret*100}];
    [portfolioFrontierChartInstance,portfolioCalChartInstance].forEach(ch=>{
        if(!ch)return;
        const d=ch.data.datasets.find(x=>x.label==='Your mix');
        if(d){d.data=pt;ch.update('none');}
    });
}

function custConstraintParams(){
    const g=id=>(document.getElementById(id)?.value||'').trim();
    const parts=custState.tickers.map(t=>{
        if(custState.lock[t]){const w=Number(custState.w[t]||0).toFixed(2);return `${t}:${w}-${w}`;}   // locked = pinned
        const lo=String(custState.lo[t]??'').trim(), hi=String(custState.hi[t]??'').trim();
        return (lo!==''||hi!=='')?`${t}:${lo}-${hi}`:'';
    }).filter(Boolean);
    const p={objective:g('custObjective'),min_weight:g('custMinW'),max_weight:g('custMaxW'),bounds:parts.join(','),
             target_return:g('custTargetRet'),target_vol:g('custTargetVol'),risk_aversion:g('custLambda')};
    const obj=p.objective;
    if(obj!=='TARGET_RETURN') p.target_return='';
    if(obj!=='TARGET_VOL') p.target_vol='';
    if(obj!=='UTILITY') p.risk_aversion='';
    return Object.entries(p).map(([k,v])=>'&'+k+'='+encodeURIComponent(v)).join('');
}

function custObjectiveChanged(){
    const obj=document.getElementById('custObjective')?.value;
    const set=(id,on)=>{const el=document.getElementById(id);if(el)el.classList.toggle('cust-input-off',!on);};
    set('custTargetRetWrap',obj==='TARGET_RETURN');
    set('custTargetVolWrap',obj==='TARGET_VOL');
    set('custLambdaWrap',obj==='UTILITY');
}

function runConstrainedPortfolio(){
    const obj=document.getElementById('custObjective').value;
    if(obj==='TARGET_RETURN'&&custNum(document.getElementById('custTargetRet').value)==null){alert('Enter a target return (%) for this objective.');return;}
    if(obj==='TARGET_VOL'&&!(custNum(document.getElementById('custTargetVol').value)>0)){alert('Enter a volatility cap (%) for this objective.');return;}
    document.getElementById('portfolioMethod').value='CONSTRAINED';
    document.getElementById('customPortfolioControls').style.display='none';
    fetchPortfolioOptimizer();
}

function custClearLimits(){
    ['custMinW','custMaxW','custTargetRet','custTargetVol'].forEach(id=>{const el=document.getElementById(id);if(el)el.value='';});
    custState.tickers.forEach(t=>{custState.lo[t]='';custState.hi[t]='';custState.lock[t]=false;});
    document.querySelectorAll('#custTable tbody tr[data-i]').forEach(tr=>{
        tr.querySelector('[data-role=lo]').value='';tr.querySelector('[data-role=hi]').value='';tr.querySelector('[data-role=lock]').checked=false;});
    custRefresh();
}

function custSendToServer(){
    const T=custState.tickers; if(!T.length){alert('Run the optimizer first to load assets.');return;}
    // Round to 2 decimals and put any rounding remainder on the largest weight so the total is exactly 100.
    const f=custMineFrac().map(x=>Math.round(x*10000)/100);
    const diff=Math.round((100-f.reduce((a,b)=>a+b,0))*100)/100;
    const k=f.indexOf(Math.max(...f)); f[k]=Math.round((f[k]+diff)*100)/100;
    const txt=T.map((t,i)=>`${t}:${f[i].toFixed(2)}`).join(',');
    document.getElementById('portfolioCustomWeights').value=txt;
    document.getElementById('portfolioMethod').value='CUSTOM';
    document.getElementById('customPortfolioControls').style.display='block';
    const lbl=document.getElementById('customPortfolioTotal'); if(lbl) lbl.innerText='Total: 100.00%';
    fetchPortfolioOptimizer().then(()=>document.getElementById('view-portfolio')?.scrollIntoView({behavior:'smooth'}));
}

(function custWire(){
    const tb=document.querySelector('#custTable tbody'); if(!tb)return;
    const rowT=e=>{const tr=e.target.closest('tr[data-i]');return tr?custState.tickers[Number(tr.dataset.i)]:null;};
    tb.addEventListener('input',e=>{
        const t=rowT(e); if(!t)return;
        if(e.target.dataset.role==='slider') custSetWeight(t,e.target.value);
    });
    tb.addEventListener('change',e=>{
        const t=rowT(e); if(!t)return;
        const role=e.target.dataset.role;
        if(role==='num'){custSetWeight(t,e.target.value);e.target.value=Number(custState.w[t]||0).toFixed(2);}
        else if(role==='lock'){custState.lock[t]=e.target.checked;custRefresh();}
        else if(role==='lo'||role==='hi'){custState[role][t]=e.target.value.trim();custRefresh();}
    });
    tb.addEventListener('keydown',e=>{if(e.key==='Enter'&&e.target.dataset.role==='num'){e.target.blur();}});
    ['custMinW','custMaxW'].forEach(id=>document.getElementById(id)?.addEventListener('input',()=>custRefresh()));
    document.getElementById('portfolioMethod')?.addEventListener('change',()=>{
        if(document.getElementById('portfolioMethod').value==='CONSTRAINED') document.getElementById('portfolioCustomizer')?.scrollIntoView({behavior:'smooth',block:'start'});
    });
    custObjectiveChanged();
})();

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


# ------------------------------------------------------------------------------
# 6. LOCAL SERVER EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Running the file directly = local use. Listen on this computer only
    # (127.0.0.1) and open the browser at the address that actually works.
    # 0.0.0.0 is a "listen everywhere" setting, not an address a browser can
    # open. Cloud hosts and Docker start the app with their own --host 0.0.0.0
    # command, so they are unaffected. Set HOST=0.0.0.0 to share on your network.
    import webbrowser

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    local_url = f"http://localhost:{port}"
    print("\n" + "=" * 60)
    print(f"  Finmodel is starting. Open this address in your browser:")
    print(f"      {local_url}")
    print(f"  Keep this window open while you use it (Ctrl+C stops it).")
    print("=" * 60 + "\n", flush=True)
    if os.environ.get("NO_BROWSER", "").strip() == "":
        threading.Timer(2.5, lambda: webbrowser.open(local_url)).start()
    uvicorn.run(app, host=host, port=port, reload=False)