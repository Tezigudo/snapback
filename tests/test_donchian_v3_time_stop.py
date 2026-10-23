"""Donchian-v3 backtest must honour the live time stop (max_hold_bars).

DonchianBreakoutBTCv3.next() overrides the parent's next() and used to omit the
time stop (only the parent had it), so backtests let positions run until the
channel exit / SL while live force-closes at max_hold_bars (N bars on 4h).
The tests use a synthetic N (42), not any leg's configured value.

Live semantics (bot._maybe_time_stop + time_stop_due): close at market once
age since the entry FILL >= N bars; tick order is time_stop -> channel_exit ->
_maybe_enter, so the same tick may re-enter off the just-closed bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Backtest

from strategy.signals_donchian import DonchianBreakoutBTCv3


def _frame(n: int = 130, short: bool = False, exit_lower_from: int | None = None) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    px = np.full(n, 40_000.0)
    df = pd.DataFrame({"Open": px, "High": px + 10, "Low": px - 10, "Close": px,
                       "Volume": 1.0}, index=idx)
    df["DonchianUpper"] = 39_000.0    # close > upper on every bar -> always a long signal
    df["DonchianLower"] = 0.0
    df["ExitUpper"] = 0.0
    df["DonchianExitUpper"] = 1e9
    df["DonchianExitLower"] = 0.0     # close never < 0 -> channel exit never fires
    if exit_lower_from is not None:   # channel exit due (close < 1e9) from this bar on
        df.iloc[exit_lower_from:, df.columns.get_loc("DonchianExitLower")] = 1e9
    if short:                         # close < lower on every bar -> always a short signal
        df["DonchianUpper"] = 1e9
        df["DonchianLower"] = 41_000.0
        df["DonchianExitUpper"] = 1e9
        df["DonchianExitLower"] = 0.0
    df["ATR_1h"] = 400.0              # SL 600 away; bars span only +/-10 so SL never hits
    return df.drop(columns=["ExitUpper"])


def _run(time_stop_bars: int, frame: pd.DataFrame | None = None, **attrs):
    cls = type("V3T", (DonchianBreakoutBTCv3,), {})
    cls.time_stop_bars = time_stop_bars
    cls.atr_sl_multiple = 1.7
    cls.slope_trend_threshold_pct = 0.0
    cls.allow_shorts = False
    cls.donchian_period_entry = 5
    cls.donchian_period_exit = 5      # strategy values have no class default
    cls.atr_trail_multiple = 0.0
    cls.leverage = 5
    for k, v in attrs.items():
        setattr(cls, k, v)
    stats = Backtest(_frame() if frame is None else frame, cls, cash=1_000_000, commission=0.0, margin=0.05,
                     trade_on_close=False, exclusive_orders=True,
                     finalize_trades=True).run()
    return stats._trades


def test_time_stop_closes_48_bars_after_fill_and_reenters_same_bar():
    t = _run(42)
    first = t.iloc[0]
    # first signal is on the first bar the engine calls next() on; fill = next open
    # live: age >= 42 bars after the fill -> close at market at fill + 42 bars
    assert first.ExitBar - first.EntryBar == 42
    # live re-enters on the same tick off the bar that just closed
    assert t.iloc[1].EntryBar == first.ExitBar


def test_time_stop_disabled_holds_until_end():
    t = _run(0)
    assert len(t) == 1
    assert t.iloc[0].ExitBar == len(_frame()) - 1   # only the finalize close


def test_time_stop_short_side_closes_and_reenters_same_bar():
    t = _run(42, _frame(short=True), allow_shorts=True)
    first = t.iloc[0]
    assert first.Size < 0
    assert first.ExitBar - first.EntryBar == 42
    assert t.iloc[1].EntryBar == first.ExitBar and t.iloc[1].Size < 0


def test_time_stop_takes_precedence_over_channel_exit_on_same_bar():
    first = _run(42).iloc[0]
    # channel exit also due on the very bar the time stop fires (bar ExitBar-1)
    t = _run(42, _frame(exit_lower_from=int(first.ExitBar) - 1))
    assert t.iloc[0].ExitBar == first.ExitBar
    # time stop falls through to entry; a channel-exit return would skip re-entry
    assert t.iloc[1].EntryBar == first.ExitBar


def test_time_stop_with_slope_gate_blocking_reentry_does_not_reenter():
    class Gated(DonchianBreakoutBTCv3):
        def _slope_now(self):   # uptrend for the first signal, flat afterwards
            return 1.0 if len(self.data) <= 5 else 0.0

    t = Backtest(_frame(), Gated, cash=1_000_000, commission=0.0, margin=0.05,
                 trade_on_close=False, exclusive_orders=True, finalize_trades=True
                 ).run(time_stop_bars=42, atr_sl_multiple=1.7,
                       slope_trend_threshold_pct=0.5, allow_shorts=False,
                       donchian_period_entry=5, donchian_period_exit=5, atr_trail_multiple=0.0, leverage=5,
                       regime_ema_period=10, regime_slope_window=5)._trades
    assert len(t) == 1
    assert t.iloc[0].ExitBar - t.iloc[0].EntryBar == 42
