"""No entry on a bar that closed while a position was held (2026-10-27).

Live 2026-10-22, v1: the SL filled at 02:27:55 and the bot went long at
02:28:00 on the 02:00 bar, which had closed 13 min before, with the position
still open. The backtest never enters on such a bar. These tests replay that
tick sequence and the bar-boundary edges with a frozen clock.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import bot

from ._bot_seed import seed_no_stop_state

T = pd.Timestamp


def _epoch(ts: str) -> float:
    return T(ts, tz="UTC").timestamp()


def _bars(forming_open: str) -> pd.DataFrame:
    """300 15m bars ending with the still-FORMING bar opened at forming_open."""
    idx = pd.date_range(end=T(forming_open), periods=300, freq="15min")
    n = len(idx)
    return pd.DataFrame({"open": np.ones(n), "high": np.ones(n), "low": np.ones(n),
                         "close": np.ones(n), "volume": np.ones(n)}, index=idx)


class Harness:
    def __init__(self, last_signal: str | None) -> None:
        b = object.__new__(bot.Bot)
        seed_no_stop_state(b, bot.Bot)
        b.instance, b.symbol, b.entry_tf, b.bar_seconds = "v1", "BTC/USDT:USDT", "15m", 900
        b.strategy_name, b.params = "multifactor-v1", {}
        b.log = logging.getLogger("test_stale_bar_reentry")
        b.client = MagicMock()
        b._entry_blocked = lambda: False
        b._entries_paused = lambda: False
        b._daily_loss_blocks_entry = lambda equity: False
        b._persist_last_signal_ts = MagicMock()
        b._latest_gates, b._last_waiting_for = None, ""
        b._last_signal_ts = T(last_signal) if last_signal else None
        self.b = b
        self.evaluated: list[pd.Timestamp] = []

    def tick(self, at: str, side: str, forming_open: str) -> None:
        self.b.client.fetch_position.return_value = SimpleNamespace(side=side)
        self.b.client.fetch_ohlcv.return_value = _bars(forming_open)

        def _eval(name, df, funding, params):
            self.evaluated.append(df.index[-1])
            return SimpleNamespace(side=None, debug={})

        with patch("bot.time.time", return_value=_epoch(at)), \
                patch("bot.evaluate_for_strategy", side_effect=_eval), \
                patch("bot.gate_status", return_value={}):
            self.b._maybe_enter(150.0)


def test_live_2026_10_22_sl_mid_bar_does_not_reenter_on_stale_bar() -> None:
    h = Harness(last_signal="2026-10-21 23:45")
    h.tick("2026-10-22 02:27:51", "long", "2026-10-22 02:15")   # SL not yet hit
    h.tick("2026-10-22 02:28:00", "flat", "2026-10-22 02:15")   # SL filled 02:27:55
    assert h.evaluated == []                                    # the bug entered here
    h.tick("2026-10-22 02:30:05", "flat", "2026-10-22 02:30")
    assert h.evaluated == [T("2026-10-22 02:15")]               # backtest's first chance
    h.b._persist_last_signal_ts.assert_any_call(T("2026-10-22 02:00"))


def test_fill_just_before_bar_close_still_evaluates_that_bar() -> None:
    # SL fills at 02:29:59, inside the 02:15 bar; detected at 02:30:01. The
    # backtest evaluates the 02:15 bar (the position closed inside it).
    h = Harness(last_signal="2026-10-22 01:00")
    h.tick("2026-10-22 02:29:56", "long", "2026-10-22 02:15")
    h.tick("2026-10-22 02:30:01", "flat", "2026-10-22 02:30")
    assert h.evaluated == [T("2026-10-22 02:15")]


def test_held_across_a_bar_close_skips_that_bar() -> None:
    # Still open just after 02:30 → the 02:15 bar closed with the position on.
    h = Harness(last_signal="2026-10-22 01:00")
    h.tick("2026-10-22 02:30:02", "long", "2026-10-22 02:30")
    h.tick("2026-10-22 02:31:00", "flat", "2026-10-22 02:30")
    assert h.evaluated == []
    h.tick("2026-10-22 02:45:03", "flat", "2026-10-22 02:45")
    assert h.evaluated == [T("2026-10-22 02:30")]


def test_bot_close_on_this_tick_consumes_its_bar() -> None:
    # Trend exit at the 02:30 close: _maybe_channel_exit stamps the close,
    # then _maybe_enter runs flat on the same tick.
    h = Harness(last_signal="2026-10-22 01:00")
    h.tick("2026-10-22 02:29:57", "long", "2026-10-22 02:15")
    h.b._position_held_at = _epoch("2026-10-22 02:30:01")      # the close itself
    h.tick("2026-10-22 02:30:01", "flat", "2026-10-22 02:30")
    assert h.evaluated == []
    h.tick("2026-10-22 02:45:02", "flat", "2026-10-22 02:45")
    assert h.evaluated == [T("2026-10-22 02:30")]


def test_flat_all_along_is_unchanged() -> None:
    h = Harness(last_signal="2026-10-22 02:00")
    h.tick("2026-10-22 02:30:04", "flat", "2026-10-22 02:30")
    h.tick("2026-10-22 02:30:09", "flat", "2026-10-22 02:30")
    assert h.evaluated == [T("2026-10-22 02:15")]
    assert h.b._position_held_at is None


def test_never_moves_the_guard_backwards() -> None:
    h = Harness(last_signal="2026-10-22 03:00")
    h.b._position_held_at = _epoch("2026-10-22 02:28:00")
    h.tick("2026-10-22 03:20:00", "flat", "2026-10-22 03:15")
    assert h.b._last_signal_ts == T("2026-10-22 03:00")
    assert h.evaluated == []


@pytest.mark.parametrize("site", ["time_stop", "trend_exit"])
def test_bot_close_sites_stamp_before_closing(site) -> None:
    src = open(bot.__file__).read()
    leg = {"time_stop": 'close_leg="x")', "trend_exit": 'close_leg="ce")'}[site]
    head = src[:src.index(leg)]
    call = head.rindex("self.client.close_position(")
    stamp = head.rindex("self._position_held_at = time.time()")
    assert 0 < call - stamp < 400
