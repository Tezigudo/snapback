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


def test_stamp_uses_the_time_before_the_position_read() -> None:
    # The read is sent at 02:29:58 (held), the SL fills 02:29:59.5 inside the
    # 02:15 bar, and a slow reply lands at 02:30:01. Stamping the reply time
    # would consume the 02:15 bar, which the backtest can trade.
    h = Harness(last_signal="2026-10-22 01:00")
    clock = [_epoch("2026-10-22 02:29:58")]

    def _slow_read(_symbol):
        clock[0] = _epoch("2026-10-22 02:30:01")
        return SimpleNamespace(side="long")

    h.b.client.fetch_position.side_effect = _slow_read
    with patch("bot.time.time", side_effect=lambda: clock[0]):
        h.b._maybe_enter(150.0)
    assert h.b._position_held_at == _epoch("2026-10-22 02:29:58")
    h.b.client.fetch_position.side_effect = None
    h.tick("2026-10-22 02:30:06", "flat", "2026-10-22 02:30")
    assert h.evaluated == [T("2026-10-22 02:15")]


# --------------------------------------------------------------------------
# The real close paths, on a real Bot (tests/test_orphan_bracket_cleanup.py's
# factory; conftest gives every test an isolated state DB).
# --------------------------------------------------------------------------

from exchange import state  # noqa: E402

from .test_orphan_bracket_cleanup import _flat, _make_bot, _open  # noqa: E402


def _real_bot(strategy: str = "multifactor-v1"):
    b, mc = _make_bot()
    b.strategy_name = strategy
    b.bar_seconds = 900
    return b, mc


def _stamp_at_close(b, mc) -> list:
    seen: list = []

    def _close(*a, **k):
        seen.append(b._position_held_at)
        raise RuntimeError("stop after the close call")

    mc.close_position.side_effect = _close
    return seen


def test_detector_stamps_the_exit_fill_time_through_failing_ticks() -> None:
    # Ticks fail 02:14:50-02:15:40 (last open tick 02:14:50); the SL fills at
    # 02:15:10; the first good tick is 02:15:45. Without the fill time the
    # guard would consume only the 01:45 bar and the 02:00 bar (closed 02:15
    # with the position open) would be entered: the 2026-10-22 bug again.
    b, mc = _real_bot()
    b._last_position_side, b._last_position_entry = "long", 65000.0
    b._last_position_qty, b._last_entry_root = 0.01, "1234567890123"
    b._position_held_at = _epoch("2026-10-22 02:14:50")
    state.record_fill(side="long", qty=0.01, price=65000.0, reason="entry",
                      client_order_id_root="1234567890123")
    mc.fetch_position.return_value = _flat()
    mc.ex.fetch_my_trades.return_value = [
        {"side": "sell", "price": 64000.0,
         "timestamp": int(_epoch("2026-10-22 02:15:10") * 1000)}]
    with patch("bot.send_alert"), \
            patch("bot.time.time", return_value=_epoch("2026-10-22 02:15:45")):
        b._detect_bracket_exit(equity=140.0)
    assert b._position_held_at == _epoch("2026-10-22 02:15:10")
    b._consume_held_bars()
    assert b._last_signal_ts == T("2026-10-22 02:00")


def test_detector_stamps_every_open_tick_with_the_pre_read_time() -> None:
    b, mc = _real_bot()
    b._last_position_side = "long"
    clock = [_epoch("2026-10-22 02:29:58")]

    def _slow_read(_symbol):
        clock[0] = _epoch("2026-10-22 02:30:01")
        return _open("long")

    mc.fetch_position.side_effect = _slow_read
    with patch("bot.time.time", side_effect=lambda: clock[0]):
        b._detect_bracket_exit(equity=140.0)
    assert b._position_held_at == _epoch("2026-10-22 02:29:58")


@pytest.mark.parametrize(("strategy", "consumed"), [
    ("multifactor-v1", "2026-10-22 02:15"),   # backtest returns after the close
    ("donchian-v3", "2026-10-22 02:00"),      # backtest may enter on the same bar
])
def test_time_stop_follows_each_legs_backtest(strategy, consumed) -> None:
    b, mc = _real_bot(strategy)
    b.params = {**b.params, "strategy": {**(b.params.get("strategy") or {}),
                                         "max_hold_bars": 4}}
    b._position_held_at = _epoch("2026-10-22 02:30:01")    # this tick's detector stamp
    state.record_fill(side="long", qty=0.01, price=65000.0, reason="entry")
    mc.fetch_position.return_value = _open("long")
    seen = _stamp_at_close(b, mc)
    with patch("bot.time_stop_due", return_value=True), \
            patch("bot.time.time", return_value=_epoch("2026-10-22 02:30:02")), \
            patch("bot.send_alert"):
        try:
            b._maybe_time_stop(equity=140.0)
        except RuntimeError:
            pass
    assert len(seen) == 1                                    # stamped BEFORE closing
    b._consume_held_bars()
    assert b._last_signal_ts == T(consumed)


def test_trend_exit_consumes_its_own_bar_before_closing() -> None:
    b, mc = _real_bot("multifactor-v1")
    mc.fetch_position.return_value = _open("long")
    mc.fetch_ohlcv.return_value = _bars("2026-10-22 02:30")
    seen = _stamp_at_close(b, mc)
    with patch("bot.trend_exit_signal", return_value=(True, {"reason": "test"})), \
            patch("bot.time.time", return_value=_epoch("2026-10-22 02:30:01")), \
            patch("bot.send_alert"):
        try:
            b._maybe_channel_exit(equity=140.0)
        except RuntimeError:
            pass
    assert seen == [_epoch("2026-10-22 02:30:01")]
    b._consume_held_bars()
    assert b._last_signal_ts == T("2026-10-22 02:15")
