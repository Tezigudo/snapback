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


def test_bar_that_closes_during_fetch_is_not_cached(tmp_path, monkeypatch):
    """The cutoff is taken before the request, not after it returns."""
    now = pd.Timestamp(datetime.now(UTC)).floor("4h")
    path = tmp_path / "BTC_4h.parquet"
    monkeypatch.setattr(data, "_cache_path", lambda *_: path)
    frame = _bars(now - pd.Timedelta(hours=20), 6, 100.0)   # last row = forming

    real_datetime = data.datetime
    class SlowClock(real_datetime):                         # 2nd now() is 5h later
        _n = 0
        @classmethod
        def now(cls, tz=None):
            cls._n += 1
            t = real_datetime.now(tz)
            return t + timedelta(hours=5) if cls._n > 1 else t
    monkeypatch.setattr(data, "datetime", SlowClock)
    monkeypatch.setattr(data, "fetch_klines", lambda *a, **k: frame)
    data.load_klines("BTC/USDT:USDT", "4h", days_back=2)

    assert pd.read_parquet(path).index.max() == now - pd.Timedelta(hours=4)


def test_sub_hour_cache_fetches_the_next_closed_bar(tmp_path, monkeypatch):
    """15m: a flat 1h forward slack used to hide up to 3 new closed bars."""
    now = pd.Timestamp(datetime.now(UTC)).floor("15min")
    path = tmp_path / "BTC_15m.parquet"
    idx = pd.date_range(now - pd.Timedelta(minutes=60), periods=3, freq="15min",
                        tz="UTC", name="open_time")              # ends 30 min ago
    pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                  "volume": 1.0}, index=idx).to_parquet(path)
    monkeypatch.setattr(data, "_cache_path", lambda *_: path)
    calls = []
    monkeypatch.setattr(data, "fetch_klines",
                        lambda *a, **k: calls.append(a) or pd.DataFrame())
    data.load_klines("BTC/USDT:USDT", "15m", days_back=1)
    cached_end = idx[-1].to_pydatetime()
    forward = [a for a in calls if a[2] > cached_end]   # (symbol, tf, start, end)
    assert forward, "forward fetch skipped although a newer closed bar exists"


def test_cache_write_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "BTC_4h.parquet"
    _bars(pd.Timestamp("2026-01-01", tz="UTC"), 3, 100.0).to_parquet(path)
    def boom(self, p, *a, **k):
        open(p, "wb").write(b"half")
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    try:
        data._write_cache(_bars(pd.Timestamp("2026-01-01", tz="UTC"), 4, 1.0), path)
    except OSError:
        pass
    monkeypatch.undo()
    assert len(pd.read_parquet(path)) == 3      # old cache intact
