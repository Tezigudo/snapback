"""Operator entry pause (data/PAUSE_<instance>, 2026-10-27).

The flag blocks NEW entries only: no ohlcv read, no signal evaluation, no
order. It alerts once per pause/resume edge, and removing the file resumes
entries on the next tick without a restart.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import bot
from exchange.env import leg_pause_path

from ._bot_seed import seed_no_stop_state


def _bot(tmp_path, side: str = "flat"):
    b = object.__new__(bot.Bot)
    seed_no_stop_state(b, bot.Bot)
    b.instance = "v1"
    b.symbol = "BTC/USDT:USDT"
    b.entry_tf = "15m"
    b.bar_seconds = 900
    b.strategy_name, b.params = "multifactor-v1", {}
    b._latest_gates, b._last_waiting_for = None, ""
    b._last_signal_ts = None
    b._persist_last_signal_ts = MagicMock()
    b.log = logging.getLogger("test_entry_pause")
    b.client = MagicMock()
    b.client.fetch_position.return_value = SimpleNamespace(side=side)
    b._entry_blocked = lambda: False
    b._daily_loss_blocks_entry = lambda equity: False
    return b


@pytest.fixture
def db(tmp_path):
    from exchange import state
    with patch("exchange.state.DB_PATH", tmp_path / "state.db"):
        state.init_db()
        yield


@pytest.fixture
def flag(tmp_path):
    p = tmp_path / "PAUSE_v1"
    with patch("bot.leg_pause_path", lambda inst: tmp_path / f"PAUSE_{inst}"):
        yield p


def test_pause_path_is_per_leg() -> None:
    assert leg_pause_path("v1").name == "PAUSE_v1"
    assert leg_pause_path("v1").parent.name == "data"
    assert leg_pause_path("v1") != leg_pause_path("donchian")


def test_paused_skips_evaluation_and_alerts_once(tmp_path, flag) -> None:
    flag.touch()
    b = _bot(tmp_path)
    with patch("bot.send_alert") as alert:
        for _ in range(3):
            b._maybe_enter(150.0)
    b.client.fetch_ohlcv.assert_not_called()
    assert alert.call_count == 1
    assert "PAUSED" in alert.call_args[0][0]


def test_removing_flag_resumes_without_restart(tmp_path, flag) -> None:
    flag.touch()
    b = _bot(tmp_path)
    b.client.fetch_ohlcv.side_effect = RuntimeError("evaluated")
    with patch("bot.send_alert") as alert:
        b._maybe_enter(150.0)
        flag.unlink()
        with pytest.raises(RuntimeError, match="evaluated"):
            b._maybe_enter(150.0)
    assert [c[0][0] for c in alert.call_args_list] == [
        "v1: entries PAUSED", "v1: entries RESUMED"]


def test_resume_never_enters_on_a_bar_closed_during_the_pause(tmp_path, flag) -> None:
    """Paused across the 02:15 close, resumed at 02:28: the 02:00 bar (closed
    02:15, during the pause) must not be evaluated; the 02:15 bar is, at 02:30."""
    import pandas as pd

    from .test_stale_bar_reentry import _bars, _epoch

    flag.touch()
    b = _bot(tmp_path)
    b._last_signal_ts = pd.Timestamp("2026-10-22 01:00")
    seen: list = []

    def _eval(name, df, funding, params):
        seen.append(df.index[-1])
        return SimpleNamespace(side=None, debug={})

    def tick(at, forming):
        b.client.fetch_ohlcv.return_value = _bars(forming)
        with patch("bot.time.time", return_value=_epoch(at)), \
                patch("bot.evaluate_for_strategy", side_effect=_eval), \
                patch("bot.gate_status", return_value={}), patch("bot.send_alert"):
            b._maybe_enter(150.0)

    tick("2026-10-22 02:16:00", "2026-10-22 02:15")
    flag.unlink()
    tick("2026-10-22 02:28:00", "2026-10-22 02:15")
    assert seen == []
    tick("2026-10-22 02:30:04", "2026-10-22 02:30")
    assert seen == [pd.Timestamp("2026-10-22 02:15")]


def test_loop_checks_the_pause_edge_after_the_exits_while_held(db) -> None:
    """One real loop() tick holding a position: the edge check runs (so a
    PAUSED alert isn't held back until flat), and only AFTER every
    risk-reducing exit, so a slow alert send can't delay a close."""
    from .test_daily_anchor_midnight import D1, _make_bot, _ready_principal, _tick

    b, bot_mod = _make_bot()
    _ready_principal()
    order: list[str] = []
    for name in ("_maybe_time_stop", "_maybe_channel_exit", "_maybe_breakeven",
                 "_maybe_reprotect"):
        setattr(b, name, lambda *a, _n=name, **k: order.append(_n))
    b._entries_paused = lambda: order.append("_entries_paused") or False
    _tick(b, bot_mod, D1, 1000.0, "long")
    assert order[-1] == "_entries_paused"
    assert order.count("_entries_paused") == 1


def test_unreadable_flag_counts_as_paused_and_never_raises(tmp_path) -> None:
    class _Unreadable:
        def exists(self):
            raise PermissionError(13, "Permission denied")

    b = _bot(tmp_path)
    with patch("bot.leg_pause_path", lambda inst: _Unreadable()), \
            patch("bot.send_alert", return_value=True):
        assert b._pause_flag_present() is True
        b._maybe_enter(150.0)
    b.client.fetch_ohlcv.assert_not_called()


def test_alert_retries_are_capped(tmp_path, flag) -> None:
    flag.touch()
    b = _bot(tmp_path)
    tries = bot.Bot.PAUSE_ALERT_MAX_TRIES
    with patch("bot.send_alert", return_value=False) as alert:
        for i in range(tries + 3):
            with patch("bot.time.time",
                       return_value=1000.0 + i * bot.Bot.PAUSE_ALERT_RETRY_S):
                assert b._entries_paused() is True
    assert alert.call_count == tries
    assert b._entries_paused_seen is True


def test_failed_alert_is_retried_not_dropped(tmp_path, flag) -> None:
    flag.touch()
    b = _bot(tmp_path)
    with patch("bot.send_alert", return_value=False) as alert, \
            patch("bot.time.time", return_value=1000.0):
        assert b._entries_paused() is True          # gate holds even if unsent
        assert b._entries_paused() is True          # throttled: no resend yet
    assert alert.call_count == 1
    with patch("bot.send_alert", return_value=True) as alert, \
            patch("bot.time.time", return_value=1000.0 + bot.Bot.PAUSE_ALERT_RETRY_S):
        b._entries_paused()
        b._entries_paused()
    assert alert.call_count == 1                    # retried once, then reported


def test_flag_created_during_evaluation_blocks_the_order(tmp_path, flag) -> None:
    b = _bot(tmp_path)
    b.dry_run = False
    b.risk_pct, b.leverage, b.order_type = 1.0, 5, "market"
    b.constraints = bot.DEFAULT_CONSTRAINTS
    b._place_live_entry = MagicMock()
    b.client.fetch_ohlcv.return_value = __import__(
        "tests.test_stale_bar_reentry", fromlist=["_bars"])._bars("2026-10-22 02:30")
    decision = SimpleNamespace(side="long", price=65000.0, sl_distance=650.0,
                               sl_price=64350.0, tp_price=66950.0, debug={})

    def _eval(*a, **k):
        flag.touch()                                # operator pauses mid-evaluation
        return decision

    with patch("bot.evaluate_for_strategy", side_effect=_eval), \
            patch("bot.gate_status", return_value={}), patch("bot.send_alert"), \
            patch("bot.check_notional"):
        b._maybe_enter(150.0)
    b._place_live_entry.assert_not_called()


def test_no_flag_no_alert(tmp_path, flag) -> None:
    b = _bot(tmp_path)
    b.client.fetch_ohlcv.side_effect = RuntimeError("evaluated")
    with patch("bot.send_alert") as alert, pytest.raises(RuntimeError):
        b._maybe_enter(150.0)
    alert.assert_not_called()


def test_pause_does_not_touch_an_open_position(tmp_path, flag) -> None:
    flag.touch()
    b = _bot(tmp_path, side="long")
    with patch("bot.send_alert"):
        b._maybe_enter(150.0)
    b.client.close_position.assert_not_called()
    b.client.fetch_ohlcv.assert_not_called()
