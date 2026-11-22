"""Donchian backtest uses LIVE's ATR convention (2026-11 review, §A3 item 7).

live_donchian_v3 sizes the stop with the signal bar's own ATR
(`atr(...).iloc[-1]` over closed bars). The backtest used `.shift(1)` — the bar
before the breakout — which overstated the baseline. Same-timeframe frames now
get the unshifted ATR; a higher-timeframe frame keeps the shift (no lookahead).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot as bot_mod  # noqa: E402
from strategy.indicators import atr  # noqa: E402
from strategy.signals_donchian import attach_donchian  # noqa: E402


def _bars(n=200, freq="4h", seed=3):
    rng = np.random.default_rng(seed)
    close = 30_000 + np.cumsum(rng.normal(0, 150, n))
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    high = close + rng.uniform(10, 400, n)
    low = close - rng.uniform(10, 400, n)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "Volume": 1.0}, index=idx)


def test_backtest_atr_equals_live_signal_bar_atr_on_every_bar():
    df = _bars()
    out = attach_donchian(df, df, period_entry=20, period_exit=10, atr_period=14)
    for i in range(30, len(df)):
        live = atr(df["High"].iloc[:i + 1], df["Low"].iloc[:i + 1],
                   df["Close"].iloc[:i + 1], 14).iloc[-1]   # live: closed bars up to i
        assert out["ATR_1h"].iloc[i] == live


def test_higher_timeframe_frame_keeps_the_shift():
    one_h = _bars(n=100, freq="1h")
    m15 = one_h.resample("15min").ffill()
    out = attach_donchian(m15, one_h, period_entry=20, period_exit=10, atr_period=14)
    shifted = atr(one_h["High"], one_h["Low"], one_h["Close"], 14).shift(1)
    t = one_h.index[50]
    assert out.loc[t, "ATR_1h"] == shifted.loc[t]


def _b(strategy):
    b = object.__new__(bot_mod.Bot)
    b.params = {"strategy": strategy}
    b.log = logging.getLogger("t")
    return b


def test_time_stop_parity_check(caplog):
    assert _b({"time_stop_bars": 48, "max_hold_bars": 48})._check_time_stop_parity()
    assert _b({"max_hold_bars": 0})._check_time_stop_parity()       # one side unset
    with caplog.at_level(logging.ERROR, logger="t"):
        assert not _b({"time_stop_bars": 0, "max_hold_bars": 48})._check_time_stop_parity()
    assert "PARAMS PARITY" in caplog.text
