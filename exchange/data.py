"""
Binance Futures historical data fetcher.

Pulls OHLCV klines and funding-rate history from the public REST endpoints,
caches to parquet under data/historical/, and incrementally tops up the cache
on subsequent calls so we don't re-download 3 years of bars every run.

No API key required — these are public endpoints.

Usage:
    from exchange.data import load_klines, load_funding
    df15m = load_klines("BTC/USDT:USDT", "15m", days_back=1095)
    df1h  = load_klines("BTC/USDT:USDT", "1h",  days_back=1095)
    fund  = load_funding("BTC/USDT:USDT",       days_back=1095)

Or from the CLI:
    python -m exchange.data --symbol BTC/USDT:USDT --tf 15m --days 1095
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from .env import REPO_ROOT

log = logging.getLogger(__name__)

HISTORICAL_DIR = REPO_ROOT / "data" / "historical"
HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_FUTURES = "https://fapi.binance.com"
REQUEST_TIMEOUT_S = 30
POLITE_SLEEP_S = 0.1
KLINE_LIMIT = 1500
FUNDING_LIMIT = 1000

Timeframe = Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

TF_MS: dict[Timeframe, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _api_symbol(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTCUSDT' (Binance API format)."""
    return symbol.replace("/", "").split(":")[0]


def _cache_path(symbol: str, label: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return HISTORICAL_DIR / f"{safe}_{label}.parquet"


def fetch_klines(
    symbol: str, timeframe: Timeframe, start: datetime, end: datetime
) -> pd.DataFrame:
    """Fetch OHLCV klines from Binance Futures public REST, paginated."""
    sym = _api_symbol(symbol)
    cursor = _to_ms(start)
    end_ms = _to_ms(end)
    tf_ms = TF_MS[timeframe]
    rows: list[list] = []

    while cursor < end_ms:
        resp = requests.get(
            f"{BINANCE_FUTURES}/fapi/v1/klines",
            params={
                "symbol": sym,
                "interval": timeframe,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": KLINE_LIMIT,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + tf_ms
        time.sleep(POLITE_SLEEP_S)

    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        ).rename_axis("open_time")

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "n_trades",
            "taker_buy_base", "taker_buy_quote", "_ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col])
    return df.set_index("open_time")[["open", "high", "low", "close", "volume"]]


def fetch_funding(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch funding-rate history (typically every 8h on perps)."""
    sym = _api_symbol(symbol)
    cursor = _to_ms(start)
    end_ms = _to_ms(end)
    rows: list[dict] = []

    while cursor < end_ms:
        resp = requests.get(
            f"{BINANCE_FUTURES}/fapi/v1/fundingRate",
            params={
                "symbol": sym,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": FUNDING_LIMIT,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_ms = batch[-1]["fundingTime"]
        if last_ms <= cursor:  # pathological — avoid infinite loop
            break
        cursor = last_ms + 1
        time.sleep(POLITE_SLEEP_S)

    if not rows:
        return pd.DataFrame(columns=["funding_rate"]).rename_axis("funding_time")

    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"])
    return df.set_index("funding_time")[["funding_rate"]].sort_index()


def load_klines(
    symbol: str = "BTC/USDT:USDT",
    timeframe: Timeframe = "15m",
    days_back: int = 1095,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Load klines from cache, fetching forward AND backward gaps if absent.

    Two-sided cache: if the requested window extends past either end of the
    cache, fetch the missing piece and merge. Without the backward extension,
    a first call with small days_back would pin the cache start; later calls
    with larger days_back would silently return short slices.
    """
    end = end or datetime.now(UTC)
    start = end - timedelta(days=days_back)
    cache_path = _cache_path(symbol, timeframe)

    if not cache_path.exists():
        fresh = fetch_klines(symbol, timeframe, start, end)
        if not fresh.empty:
            fresh.to_parquet(cache_path)
        return fresh

    cached = pd.read_parquet(cache_path)
    if cached.empty:
        fresh = fetch_klines(symbol, timeframe, start, end)
        if not fresh.empty:
            fresh.to_parquet(cache_path)
        return fresh

    cached_start = cached.index.min().to_pydatetime()
    cached_end = cached.index.max().to_pydatetime()
    chunks = [cached]
    changed = False

    # Extend backwards if we asked for history older than the cache.
    if start < cached_start - timedelta(hours=1):
        back = fetch_klines(symbol, timeframe, start, cached_start - timedelta(milliseconds=1))
        if not back.empty:
            chunks.insert(0, back)
            changed = True

    # Extend forwards if the cache hasn't reached the requested end.
    if end > cached_end + timedelta(hours=1):
        fwd = fetch_klines(symbol, timeframe, cached_end + timedelta(milliseconds=1), end)
        if not fwd.empty:
            chunks.append(fwd)
            changed = True

    if changed:
        combined = pd.concat(chunks).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(cache_path)
    else:
        combined = cached

    return combined.loc[start:end]


def load_funding(
    symbol: str = "BTC/USDT:USDT",
    days_back: int = 1095,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Load funding history from cache, fetching forward AND backward gaps."""
    end = end or datetime.now(UTC)
    start = end - timedelta(days=days_back)
    cache_path = _cache_path(symbol, "funding")

    if not cache_path.exists():
        fresh = fetch_funding(symbol, start, end)
        if not fresh.empty:
            fresh.to_parquet(cache_path)
        return fresh

    cached = pd.read_parquet(cache_path)
    if cached.empty:
        fresh = fetch_funding(symbol, start, end)
        if not fresh.empty:
            fresh.to_parquet(cache_path)
        return fresh

    cached_start = cached.index.min().to_pydatetime()
    cached_end = cached.index.max().to_pydatetime()
    chunks = [cached]
    changed = False

    if start < cached_start - timedelta(hours=8):
        back = fetch_funding(symbol, start, cached_start - timedelta(milliseconds=1))
        if not back.empty:
            chunks.insert(0, back)
            changed = True
    if end > cached_end + timedelta(hours=8):
        fwd = fetch_funding(symbol, cached_end + timedelta(milliseconds=1), end)
        if not fwd.empty:
            chunks.append(fwd)
            changed = True

    if changed:
        combined = pd.concat(chunks).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(cache_path)
    else:
        combined = cached
    return combined.loc[start:end]


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Pull Binance Futures history into the parquet cache.")
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--tf", default="15m", choices=list(TF_MS.keys()))
    p.add_argument("--days", type=int, default=1095, help="lookback days (default 3y)")
    p.add_argument("--funding", action="store_true", help="also pull funding-rate history")
    args = p.parse_args()

    print(f"Fetching {args.symbol} {args.tf} for last {args.days} days...")
    df = load_klines(args.symbol, args.tf, days_back=args.days)
    print(f"  klines: {len(df)} rows, {df.index.min()} → {df.index.max()}")

    if args.funding:
        print("Fetching funding rate history...")
        fdf = load_funding(args.symbol, days_back=args.days)
        print(f"  funding: {len(fdf)} rows, avg rate {fdf['funding_rate'].mean():.6f} per 8h")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
