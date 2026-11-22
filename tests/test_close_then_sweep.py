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


def test_halt_close_unconfirmed_says_so():
    b, mc = _bot()
    mc.close_position.side_effect = None
    mc.close_position.return_value = {"id": "1"}
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _open("long", 65_000.0, 0.01)
    with patch("bot.halt_source", return_value="self"), \
         patch("bot.send_alert") as alert, patch("bot.consolidate_push"):
        assert _loop_once(b) == 0
    halted = [c.args[1] for c in alert.call_args_list if c.args[0] == "Bot HALTED"]
    assert halted and "NOT confirmed" in halted[0]
    mc.cancel_open_orders.assert_not_called()


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
