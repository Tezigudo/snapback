"""
Tests for the hand-rolled indicators. Reference values cross-checked against
TradingView / pandas-ta for a fixed input series. If you swap the underlying
implementation, expect to update these constants — they pin the convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.indicators import atr, ema, rsi, sma


def _series(*values, freq="1h"):
    idx = pd.date_range("2024-01-01", periods=len(values), freq=freq, tz="UTC").tz_localize(None)
    return pd.Series(values, index=idx, dtype=float)


def test_rsi_pure_uptrend_pegs_at_100():
    close = _series(*range(1, 30))
    r = rsi(close, period=14)
    assert r.dropna().iloc[-1] == pytest.approx(100.0, abs=1e-9)


def test_rsi_pure_downtrend_pegs_at_zero():
    close = _series(*range(30, 1, -1))
    r = rsi(close, period=14)
    assert r.dropna().iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_rsi_warmup_is_nan():
    close = _series(*np.arange(20, dtype=float))
    r = rsi(close, period=14)
    # First (period) values must be NaN because Wilder smoothing needs them.
    assert r.iloc[:14].isna().all()
    assert not r.iloc[14:].isna().all()


def test_rsi_short_window_responsiveness():
    # RSI(2) should swing wildly — that's the whole point of using it.
    close = _series(100, 101, 102, 103, 95, 96, 97, 98, 99, 100)
    r = rsi(close, period=2)
    valid = r.dropna()
    assert valid.min() < 30 or valid.max() > 70, "RSI(2) should be reactive"


def test_ema_converges_to_constant():
    close = _series(*([50.0] * 100))
    e = ema(close, period=20).dropna()
    assert e.iloc[-1] == pytest.approx(50.0, abs=1e-9)


def test_atr_positive_and_increases_with_range():
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC").tz_localize(None)
    close = pd.Series(np.linspace(40000, 41000, n), index=idx)
    high_tight = close + 10
    low_tight = close - 10
    high_wide = close + 200
    low_wide = close - 200
    a_tight = atr(high_tight, low_tight, close, period=14).dropna()
    a_wide = atr(high_wide, low_wide, close, period=14).dropna()
    assert a_tight.iloc[-1] > 0
    assert a_wide.iloc[-1] > a_tight.iloc[-1] * 5


def test_sma_window_arithmetic():
    s = _series(*range(1, 11))
    m = sma(s, period=3).dropna()
    # SMA(1,2,3) = 2, SMA(2,3,4) = 3, ...
    assert m.iloc[0] == pytest.approx(2.0)
    assert m.iloc[-1] == pytest.approx(9.0)
