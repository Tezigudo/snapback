"""Bot-initiated closes: send the close FIRST, sweep the stops AFTER, alert on
failure (2026-11 review, FINAL §A3 item 3 / N4).

Before: close_position(sweep_first=True) cancelled the leg's SL/TP and THEN
sent the reduce-only close. A rejected close (margin, -2022, -1021, timeout)
left a leveraged position with no stop, no TP and no alert — including on the
HALT and kill-switch paths.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from exchange import state
from exchange.binance_client import PositionUnknown
from tests.test_orphan_bracket_cleanup import _make_bot, _open


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch("bot.time.sleep"):
        yield


def _bot():
    b, mc = _make_bot()
    mc.fetch_position.return_value = _open("long", 65_000.0, 0.01)
    return b, mc


# --- the helper ------------------------------------------------------------

def test_close_is_sent_before_the_sweep_and_with_sweep_first_false():
    b, mc = _bot()
    parent = MagicMock()
    parent.attach_mock(mc.close_position, "close_position")
    parent.attach_mock(mc.cancel_open_orders, "cancel_open_orders")
    with patch("bot.send_alert") as alert:
        order, flat = b._close_then_sweep("r1", "x", "time-stop")
    assert flat is True and order is not None
    names = [c[0] for c in parent.mock_calls]
    assert names == ["close_position", "cancel_open_orders"], names
    assert parent.mock_calls[0] == call.close_position(
        "BTC/USDT:USDT", client_order_id_root="r1", close_leg="x", sweep_first=False)
    alert.assert_not_called()


def test_close_raises_keeps_stops_alerts_and_propagates():
    b, mc = _bot()
    mc.close_position.side_effect = RuntimeError("-2022 ReduceOnly rejected")
    with patch("bot.send_alert") as alert, pytest.raises(RuntimeError):
        b._close_then_sweep("r1", "k", "kill-switch")
    mc.cancel_open_orders.assert_not_called()
    assert "FAILED" in alert.call_args.args[0]
    assert "NOT cancelled" in alert.call_args.args[1]


def test_close_sent_but_still_open_does_not_sweep():
    b, mc = _bot()
    mc.close_position.side_effect = None
    mc.close_position.return_value = {"id": "1"}
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _open("long", 65_000.0, 0.01)
    with patch("bot.send_alert") as alert:
        _order, flat = b._close_then_sweep("r1", "ce", "trend-exit")
    assert flat is False
    assert mc.fetch_position_checked.call_count == b.CLOSE_CONFIRM_READS
    mc.cancel_open_orders.assert_not_called()
    assert "NOT confirmed flat" in alert.call_args.args[1]


def test_unreadable_confirm_is_not_flat():
    b, mc = _bot()
    mc.fetch_position_checked.side_effect = PositionUnknown("no contracts")
    with patch("bot.send_alert"):
        _order, flat = b._close_then_sweep("r1", "x", "time-stop")
    assert flat is False
    mc.cancel_open_orders.assert_not_called()


def test_none_return_on_a_false_flat_sweeps_nothing():
    """close_position's own lenient read said flat and sent nothing; a strict
    re-read says open. Nothing may be cancelled."""
    b, mc = _bot()
    mc.close_position.side_effect = None
    mc.close_position.return_value = None
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _open("long", 65_000.0, 0.01)
    with patch("bot.send_alert") as alert:
        order, flat = b._close_then_sweep("r1", "h", "HALT")
    assert (order, flat) == (None, False)
    mc.cancel_open_orders.assert_not_called()
    alert.assert_called_once()


def test_failure_alerts_are_rate_limited_per_leg():
    b, mc = _bot()
    mc.close_position.side_effect = RuntimeError("boom")
    with patch("bot.send_alert") as alert:
        for _ in range(3):
            with pytest.raises(RuntimeError):
                b._close_then_sweep("r1", "h", "HALT")
        assert alert.call_count == 1
        with pytest.raises(RuntimeError):
            b._close_then_sweep("r1", "k", "kill-switch")   # other leg: mails
        assert alert.call_count == 2


# --- callers ---------------------------------------------------------------

def test_time_stop_close_failure_keeps_stops_and_alerts():
    b, mc = _bot()
    b.params = {**b.params, "strategy": {**(b.params.get("strategy") or {}),
                                          "max_hold_bars": 1}}
    state.record_fill(side="long", qty=0.01, price=65_000.0, reason="entry",
                      client_order_id_root="r1")
    mc.close_position.side_effect = RuntimeError("timeout")
    with patch("bot.send_alert") as alert, \
         patch("bot.time_stop_due", return_value=True):
        b._maybe_time_stop(equity=1000.0)
    mc.cancel_open_orders.assert_not_called()
    assert any("close FAILED" in c.args[0] for c in alert.call_args_list)


def _loop_once(b):
    """One loop() iteration; returns loop()'s value or None if it kept going."""
    calls = {"n": 0}

    def _sleep(_s):
        calls["n"] += 1
        b._stopped = True

    with patch("bot.time.sleep", _sleep):
        return b.loop()


def test_halt_close_failure_does_not_claim_flattened_and_keeps_looping():
    b, mc = _bot()
    mc.close_position.side_effect = RuntimeError("-1021")
    with patch("bot.halt_source", return_value="self"), \
         patch("bot.send_alert") as alert:
        rv = _loop_once(b)
    subjects = [c.args[0] for c in alert.call_args_list]
    assert "Bot HALTED" not in subjects, subjects
    assert any("HALT close FAILED" in s for s in subjects), subjects
    mc.cancel_open_orders.assert_not_called()
    assert rv == 0 and b._stopped  # loop exited only via the patched sleep (retry path)


def test_halt_close_unconfirmed_does_not_exit_until_confirmed():
    """F5: an unconfirmed HALT close keeps the process alive and retries; it
    exits 0 (and mails "Bot HALTED") only once flat is confirmed."""
    b, mc = _bot()
    mc.close_position.side_effect = None
    mc.close_position.return_value = {"id": "1"}
    mc.fetch_position_checked.side_effect = None
    reads = {"n": 0}

    def _checked(*a, **k):
        reads["n"] += 1
        # tick 1 (2 confirm reads): still open; tick 2: flat
        return _open("long", 65_000.0, 0.01) if reads["n"] <= 2 else \
            __import__("tests.test_orphan_bracket_cleanup", fromlist=["_flat"])._flat()
    mc.fetch_position_checked.side_effect = _checked
    sleeps = []
    with patch("bot.halt_source", return_value="self"), \
         patch("bot.send_alert") as alert, patch("bot.consolidate_push"), \
         patch("bot.time.sleep", side_effect=sleeps.append):
        assert b.loop() == 0
    assert mc.close_position.call_count == 2
    subjects = [c.args[0] for c in alert.call_args_list]
    assert subjects.count("Bot HALTED") == 1 and subjects[-1] == "Bot HALTED"
    assert mc.cancel_open_orders.call_count == 1          # swept only once confirmed


def test_kill_close_unconfirmed_does_not_exit():
    b, mc = _bot()
    mc.close_position.side_effect = None
    mc.close_position.return_value = {"id": "1"}
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _open("long", 65_000.0, 0.01)
    b._kill_switch_after_reconcile = lambda equity: True
    mc.fetch_equity_usdt.return_value = 10.0
    def _sleep(_s):
        if mc.close_position.call_count >= 2:
            b._stopped = True
    with patch("bot.halt_source", return_value=None), patch("bot.send_alert"), \
         patch("bot.time.sleep", side_effect=_sleep):
        rv = b.loop()
    assert mc.close_position.call_count >= 2               # retried, not exited
    mc.cancel_open_orders.assert_not_called()
    assert rv == 0 and b._stopped                          # left only via the test stop


def test_kill_switch_close_failure_alerts_and_keeps_stops():
    b, mc = _bot()
    mc.close_position.side_effect = RuntimeError("margin")
    b._check_kill_switch = lambda equity: True
    mc.fetch_equity_usdt.return_value = 10.0
    with patch("bot.halt_source", return_value=None), \
         patch("bot.send_alert") as alert:
        _loop_once(b)
    assert any("kill-switch close FAILED" in c.args[0] for c in alert.call_args_list)
    mc.cancel_open_orders.assert_not_called()


def _boot_ctx():
    from contextlib import ExitStack

    from tests.test_orphan_bracket_cleanup import _boot_patches
    st = ExitStack()
    st.enter_context(_boot_patches())
    st.enter_context(patch.multiple(
        "bot.state", init_db=MagicMock(), get_float=MagicMock(return_value=0.0),
        set_float=MagicMock(), set_meta=MagicMock(), enqueue_bot_event=MagicMock(),
        record_event=MagicMock(), latest_entry_coid_root=MagicMock(return_value=None)))
    st.enter_context(patch("bot.principal.initialize", MagicMock(return_value=None)))
    st.enter_context(patch("bot.principal.get_principal", MagicMock(return_value=None)))
    st.enter_context(patch("bot.send_alert"))
    return st


def test_boot_flatten_none_on_false_flat_sweeps_nothing():
    b, mc = _bot()
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.close_position.side_effect = None
    mc.close_position.return_value = None
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _open("long", 65_000.0, 0.01)
    with _boot_ctx():
        b.boot()
    mc.cancel_open_orders.assert_not_called()


def test_boot_flatten_happy_path_closes_then_sweeps():
    b, mc = _bot()
    mc.fetch_equity_usdt.return_value = 1000.0
    parent = MagicMock()
    parent.attach_mock(mc.close_position, "close_position")
    parent.attach_mock(mc.cancel_open_orders, "cancel_open_orders")
    with _boot_ctx():
        b.boot()
    names = [c[0] for c in parent.mock_calls]
    assert names[:2] == ["close_position", "cancel_open_orders"], names
    assert mc.fetch_position.return_value.side == "flat"


# --- F2: time-stop / trend-exit book nothing unless flat is confirmed ---------

def _time_stop_bot():
    b, mc = _bot()
    b.params = {**b.params, "strategy": {**(b.params.get("strategy") or {}),
                                          "max_hold_bars": 1}}
    state.record_fill(side="long", qty=0.01, price=65_000.0, reason="entry",
                      client_order_id_root="r1")
    return b, mc


def _reasons():
    import sqlite3
    with sqlite3.connect(state.DB_PATH) as c:
        return [r[0] for r in c.execute("SELECT reason FROM fills ORDER BY id")]


@pytest.mark.parametrize("case", ["unconfirmed", "already_flat"])
def test_time_stop_books_nothing_unless_flat_confirmed(case):
    b, mc = _time_stop_bot()
    mc.close_position.side_effect = None
    mc.close_position.return_value = {"id": "1"} if case == "unconfirmed" else None
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = (
        _open("long", 65_000.0, 0.01) if case == "unconfirmed" else
        __import__("tests.test_orphan_bracket_cleanup", fromlist=["_flat"])._flat())
    with patch("bot.send_alert") as alert, patch("bot.time_stop_due", return_value=True):
        b._maybe_time_stop(equity=1000.0)
    mc.close_position.assert_called_once()
    assert _reasons() == ["entry"]                       # no close row, no price-0 row
    assert not any("time-stop close" in c.args[0] and "FAILED" not in c.args[0]
                   for c in alert.call_args_list)


def test_time_stop_confirmed_close_still_books():
    b, _mc = _time_stop_bot()
    with patch("bot.send_alert"), patch("bot.time_stop_due", return_value=True):
        b._maybe_time_stop(equity=1000.0)
    assert _reasons() == ["entry", "time_stop"]


def test_unconfirmed_time_stop_exit_is_recorded_later_by_the_flat_edge():
    """The SL/close fill's PnL is not dropped: the fills row is still `entry`,
    so the next flat edge prices the exit from the exchange's own trade."""
    b, mc = _time_stop_bot()
    mc.close_position.side_effect = None
    mc.close_position.return_value = {"id": "1"}
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _open("long", 65_000.0, 0.01)
    with patch("bot.send_alert"), patch("bot.time_stop_due", return_value=True):
        b._maybe_time_stop(equity=1000.0)
    # next tick: the position IS flat now, bracket legs swept/filled
    from tests.test_orphan_bracket_cleanup import _flat
    b._last_position_side, b._last_position_entry = "long", 65_000.0
    b._last_position_qty, b._last_entry_root = 0.01, "r1"
    mc.fetch_position.return_value = _flat()
    mc.fetch_position_checked.return_value = _flat()
    mc.fetch_algo_orders.side_effect = None
    mc.fetch_algo_orders.return_value = ([], True)
    mc.ex.fetch_my_trades.return_value = [{"side": "sell", "price": 64_000.0}]
    with patch("bot.send_alert"):
        b._detect_bracket_exit(equity=990.0)
    assert _reasons() == ["entry", "bracket_exit"]


def test_trend_exit_books_nothing_when_unconfirmed():
    b, mc = _bot()
    b.strategy_name = "donchian-v3"
    state.record_fill(side="long", qty=0.01, price=65_000.0, reason="entry",
                      client_order_id_root="r1")
    mc.close_position.side_effect = None
    mc.close_position.return_value = {"id": "1"}
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _open("long", 65_000.0, 0.01)
    import pandas as pd
    mc.fetch_ohlcv.return_value = pd.DataFrame({"Close": [1.0, 1.0]})
    with patch("bot.trend_exit_signal", return_value=(True, {"reason": "x"})), \
         patch("bot.send_alert"):
        b._maybe_channel_exit(equity=1000.0)
    mc.close_position.assert_called_once()
    assert _reasons() == ["entry"]


# --- G2 (round 3): breakeven close books nothing unless flat is confirmed -----

def test_breakeven_close_unconfirmed_books_nothing_and_keeps_be_moved_false():
    import sqlite3

    from exchange.binance_client import Position
    from tests.test_sol_breakeven import ENTRY, QTY, _ab, _ohlcv, _seed
    from tests.test_sol_breakeven import _bot as _sol_bot
    bot, mc = _sol_bot()
    _seed()
    _ohlcv(mc, [104, 120, 106])
    mc.fetch_mark_price.return_value = 100.1            # already through breakeven
    mc.close_position.side_effect = None
    mc.close_position.return_value = {"average": 0.0}   # price unknown too
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = Position("SOL/USDT:USDT", "long", QTY,
                                                      ENTRY, 0.0, 0.0)
    with patch("bot.send_alert") as alert:
        bot._maybe_breakeven(100.0)
    with sqlite3.connect(state.DB_PATH) as c:
        reasons = [r[0] for r in c.execute("SELECT reason FROM fills ORDER BY id")]
    assert "breakeven_exit" not in reasons
    assert not _ab().get("be_moved")
    subjects = [c.args[0] for c in alert.call_args_list]
    assert "Bot breakeven close" not in subjects
    assert any("breakeven close FAILED" in s for s in subjects)
    mc.cancel_open_orders.assert_not_called()


# --- G3 (round 3): HALT/kill that find the position already flat ------------

def test_halt_found_flat_sweeps_books_and_alerts_before_exiting():
    import sqlite3

    from tests.test_orphan_bracket_cleanup import _flat
    b, mc = _bot()
    state.record_fill(side="long", qty=0.01, price=65_000.0, reason="entry",
                      client_order_id_root="r1")
    b._last_position_side, b._last_position_entry = "long", 65_000.0
    b._last_position_qty, b._last_entry_root = 0.01, "r1"
    mc.close_position.side_effect = None
    mc.close_position.return_value = None            # already flat: nothing sent
    mc.fetch_position.return_value = _flat()
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _flat()
    mc.fetch_algo_orders.side_effect = None
    mc.fetch_algo_orders.return_value = ([], True)   # the SL filled
    mc.ex.fetch_my_trades.return_value = [{"side": "sell", "price": 64_000.0}]
    with patch("bot.halt_source", return_value="self"), \
         patch("bot.send_alert") as alert, patch("bot.consolidate_push"):
        assert b.loop() == 0
    with sqlite3.connect(state.DB_PATH) as c:
        reasons = [r[0] for r in c.execute("SELECT reason FROM fills ORDER BY id")]
    assert reasons == ["entry", "bracket_exit"]                 # real exit booked
    assert mc.cancel_open_orders.call_count >= 1                # brackets swept
    subjects = [c.args[0] for c in alert.call_args_list]
    assert any("HALT found the position already flat" in s for s in subjects)


def test_kill_found_flat_with_unbookable_exit_alerts_reconcile():
    from tests.test_orphan_bracket_cleanup import _flat
    b, mc = _bot()
    state.record_fill(side="long", qty=0.01, price=65_000.0, reason="entry",
                      client_order_id_root="r1")
    b._last_position_side = "unknown"                # nothing to detect from
    mc.close_position.side_effect = None
    mc.close_position.return_value = None
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _flat()
    b._kill_switch_after_reconcile = lambda equity: True
    mc.fetch_equity_usdt.return_value = 10.0
    with patch("bot.halt_source", return_value=None), patch("bot.send_alert") as alert:
        assert b.loop() == 0
    bodies = [c.args[1] for c in alert.call_args_list
              if "kill-switch found the position already flat" in c.args[0]]
    assert bodies and "NOT in the ledger" in bodies[0]
    mc.cancel_open_orders.assert_called_once()


# --- G5 (round 3): HALT retry loop keeps managing risk, escalates ------------

def test_halt_retry_runs_risk_reducing_hooks_only_and_escalates():
    b, mc = _bot()
    mc.close_position.side_effect = None
    mc.close_position.return_value = {"id": "1"}
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _open("long", 65_000.0, 0.01)
    calls = []
    b._maybe_finish_no_stop_close = lambda eq: calls.append("no_stop")
    b._maybe_reprotect = lambda eq: calls.append("reprotect")
    b._maybe_breakeven = lambda eq: calls.append("breakeven")
    b._maybe_enter = lambda eq: calls.append("enter")
    b._maybe_time_stop = lambda eq: calls.append("time_stop")
    t = [1_000_000.0]
    ticks = {"n": 0}

    def _sleep(_s):
        ticks["n"] += 1
        t[0] += 300.0                                  # 5 min per retry
        if ticks["n"] >= 6:
            b._stopped = True
    with patch("bot.halt_source", return_value="self"), \
         patch("bot.send_alert") as alert, patch("bot.time.sleep", side_effect=_sleep), \
         patch("bot.time.time", side_effect=lambda: t[0]):
        b.loop()
    assert "no_stop" in calls and "reprotect" in calls
    assert not {"breakeven", "enter", "time_stop"} & set(calls)
    urgent = [c for c in alert.call_args_list if c.args[0].startswith("URGENT")]
    assert len(urgent) == 1                            # escalated once, not every tick


# --- G11 (round 3): same-bar held stamp rolled back on an unconfirmed time stop

def _same_bar_time_stop_bot(strategy):
    b, mc = _time_stop_bot()
    b.strategy_name = strategy
    b.bar_seconds = 3600
    return b, mc


def test_unconfirmed_time_stop_rolls_back_the_same_bar_stamp():
    from bot_internals import _TIME_STOP_REENTERS_SAME_BAR
    strat = sorted(_TIME_STOP_REENTERS_SAME_BAR)[0]
    b, mc = _same_bar_time_stop_bot(strat)
    b._position_held_at = None
    mc.close_position.side_effect = None
    mc.close_position.return_value = {"id": "1"}
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _open("long", 65_000.0, 0.01)
    now = 1_800_000_000.0 + 1800                       # mid-bar
    with patch("bot.send_alert"), patch("bot.time_stop_due", return_value=True), \
         patch("bot.time.time", return_value=now):
        b._maybe_time_stop(equity=1000.0)
    assert b._position_held_at == now                  # held NOW, not bar_open - 1


def test_confirmed_time_stop_keeps_the_same_bar_stamp():
    from bot_internals import _TIME_STOP_REENTERS_SAME_BAR
    strat = sorted(_TIME_STOP_REENTERS_SAME_BAR)[0]
    b, _mc = _same_bar_time_stop_bot(strat)
    b._position_held_at = None
    now = 1_800_000_000.0 + 1800
    with patch("bot.send_alert"), patch("bot.time_stop_due", return_value=True), \
         patch("bot.time.time", return_value=now):
        b._maybe_time_stop(equity=1000.0)
    assert b._position_held_at == (int(now) // 3600) * 3600 - 1


# --- H2 (round 4): a RAISING HALT close still runs the hooks + escalation ----

def test_halt_close_raising_runs_risk_hooks_and_escalates():
    b, mc = _bot()
    mc.close_position.side_effect = RuntimeError("-2022 ReduceOnly rejected")
    calls = []
    b._maybe_finish_no_stop_close = lambda eq: calls.append("no_stop")
    b._maybe_reprotect = lambda eq: calls.append("reprotect")
    b._maybe_enter = lambda eq: calls.append("enter")
    t = [1_000_000.0]
    ticks = {"n": 0}

    def _sleep(_s):
        ticks["n"] += 1
        t[0] += 300.0
        if ticks["n"] >= 6:
            b._stopped = True
    with patch("bot.halt_source", return_value="self"), \
         patch("bot.send_alert") as alert, patch("bot.time.sleep", side_effect=_sleep), \
         patch("bot.time.time", side_effect=lambda: t[0]):
        b.loop()
    assert "no_stop" in calls and "reprotect" in calls and "enter" not in calls
    assert sum(1 for c in alert.call_args_list if c.args[0].startswith("URGENT")) == 1
    mc.cancel_open_orders.assert_not_called()


# --- H3 (round 4): the no-stop close finish is None-equity safe --------------

def test_finish_no_stop_with_unknown_equity_still_mails_records_then_clears():
    import json as _json
    import sqlite3
    b, _mc = _bot()
    rec = {"kind": "no_stop_close", "root": "r1", "side": "long", "qty": 0.01,
           "entry_price": 65_000.0, "stop_price": None, "why": "t"}
    b._no_stop_pending = rec
    state.set_meta(b.NO_STOP_PENDING_KEY, _json.dumps(rec))
    state.record_fill(side="long", qty=0.01, price=65_000.0, reason="entry",
                      client_order_id_root="r1")
    with patch("bot.send_alert") as alert:
        b._finish_no_stop("long", 0.01, 65_000.0, "r1", None, closed=True,
                          order={"id": "1", "average": 64_500.0}, close_error=None,
                          why="HALT retry reprotect", record_if_flat=True)
    (mail,) = [c for c in alert.call_args_list if c.args[0] == "NO STOP — position closed"]
    assert "Equity: unknown" in mail.args[1]
    assert b._pending_record() is None
    with sqlite3.connect(state.DB_PATH) as c:
        row = c.execute("SELECT reason, equity_after FROM fills ORDER BY id DESC LIMIT 1").fetchone()
    assert row == ("no_stop_close", None)          # never a 0 / stale equity


def test_halt_retry_marks_equity_stale_so_nothing_persists_it():
    b, mc = _bot()
    b._last_known_equity = 123.0
    b._equity_stale = False
    seen = []
    b._maybe_finish_no_stop_close = lambda eq: seen.append(b._ledger_equity(eq))
    b._maybe_reprotect = lambda eq: None
    with patch("bot.send_alert"):
        b._halt_retry_tick("self data/HALT_v1")
    assert seen == [None]
