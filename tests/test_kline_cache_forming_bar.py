"""load_klines must never write Binance's still-forming bar to the cache.

Regression for 2026-11-04: the forming row was cached, the next forward fetch
started after it, so the partial bar was frozen in the parquet for good.
"""
from datetime import UTC, datetime, timedelta

import pandas as pd

from exchange import data


def _bars(start: pd.Timestamp, n: int, close: float) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="4h", tz="UTC", name="open_time")
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1.0}, index=idx)


def test_forming_bar_returned_but_not_cached(tmp_path, monkeypatch):
    now = pd.Timestamp(datetime.now(UTC)).floor("4h")
    path = tmp_path / "BTC_4h.parquet"
    monkeypatch.setattr(data, "_cache_path", lambda *_: path)

    # 5 closed bars + the forming one (open_time == current 4h boundary)
    frame = _bars(now - pd.Timedelta(hours=20), 6, 100.0)
    monkeypatch.setattr(data, "fetch_klines", lambda *a, **k: frame)
    out = data.load_klines("BTC/USDT:USDT", "4h", days_back=2)

    assert out.index.max() == now                      # caller still sees it
    assert pd.read_parquet(path).index.max() == now - pd.Timedelta(hours=4)


def test_partial_bar_is_refetched_on_next_call(tmp_path, monkeypatch):
    now = pd.Timestamp(datetime.now(UTC)).floor("4h")
    path = tmp_path / "BTC_4h.parquet"
    monkeypatch.setattr(data, "_cache_path", lambda *_: path)

    first = _bars(now - pd.Timedelta(hours=20), 6, 100.0)
    first.iloc[-1, first.columns.get_loc("close")] = 1.0   # partial snapshot
    monkeypatch.setattr(data, "fetch_klines", lambda *a, **k: first)
    data.load_klines("BTC/USDT:USDT", "4h", days_back=2)

    calls = []
    def fwd(symbol, timeframe, start, end):
        calls.append(start)
        return _bars(now, 1, 100.0)                          # the real values
    monkeypatch.setattr(data, "fetch_klines", fwd)
    out = data.load_klines("BTC/USDT:USDT", "4h", days_back=2)

    assert calls and calls[0] <= now.to_pydatetime()
    assert out.loc[now, "close"] == 100.0
