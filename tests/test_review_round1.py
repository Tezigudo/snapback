"""Review round 1 fixes (sourcery PRs #6-#11): entry-order watch on every
emergency-episode exit path, retry-close pricing, close-failed mail retry,
HALT/kill booking, boot-flatten price, flat-confirm unknown ticks."""

from __future__ import annotations

import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest

from exchange import state
from exchange.binance_client import BinanceClient
from tests.test_orphan_bracket_cleanup import _flat, _make_bot, _open
from tests.test_review_fixes_guard import REC, _pending_bot


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch("bot.time.sleep"):
        yield


# --- 356: every path that ends an emergency episode watches `-e` --------------

@pytest.mark.parametrize("gone,watched", [(None, True), (False, True), (True, False)])
def test_pending_retry_flat_branch_watches_an_unproven_entry_order(gone, watched):
    b, mc = _pending_bot(0.010)
    mc.fetch_position.return_value = _flat()
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _flat()
    mc.cancel_entry_order.return_value = gone
    b._flat_seen = {"no_stop_retry": (1, None)}
    with patch("bot.send_alert"):
        b._maybe_finish_no_stop_close(1000.0)
        b._maybe_finish_no_stop_close(1000.0)
    assert b._pending_record() is None
    u = b._unverified_record()
    assert (u is not None) is watched
    if watched:
        assert u["root"] == "r1" and u["after_emergency_close"] is True


@pytest.mark.parametrize("gone,watched", [(None, True), (True, False)])
def test_abandon_branch_watches_an_unproven_entry_order(gone, watched):
    b, mc = _pending_bot(0.010)
    mc.fetch_position.return_value = _open("short", 65_000.0, 0.01)
    mc.cancel_entry_order.return_value = gone
    with patch("bot.send_alert"):
        b._maybe_finish_no_stop_close(1000.0)
    mc.emergency_close.assert_not_called()
    assert b._pending_record() is None
    assert (b._unverified_record() is not None) is watched


@pytest.mark.parametrize("extra", [{}, {"retry_protect": True}], ids=["late_fill", "retry_protect"])
@pytest.mark.parametrize("gone,kept", [(None, True), (True, False)])
def test_guard_keeps_the_record_until_the_entry_order_is_proven_gone(extra, gone, kept):
    b, mc = _make_bot()
    b._set_unverified({**REC, **extra})
    b._unverified_next = 0.0
    b.constraints = MagicMock(qty_step=0.001)
    mc.fetch_position.return_value = _open("long", 65_000.0, 0.01)
    mc.entry_order_filled.return_value = 0.01
    mc.cancel_entry_order.return_value = gone
    with patch("bot.send_alert"), \
         patch.object(b, "_protect_or_close_open", return_value=True):
        b._maybe_guard_unverified_entry(1000.0)
    u = b._unverified_record()
    assert (u is not None) is kept
    if kept:
        assert u["after_emergency_close"] is True    # no false "outside the ledger" mail


def test_unresolved_record_escalates_once_after_30_minutes():
    b, mc = _make_bot()
    b._set_unverified({**REC, "after_emergency_close": True, "since": 0.0})
    b._unverified_next = 0.0
    b.constraints = MagicMock(qty_step=0.001)
    mc.fetch_position.return_value = _flat()
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _flat()
    mc.cancel_entry_order.return_value = True
    mc.entry_order_filled.return_value = None          # unreadable
    t = [1_000_000.0]
    with patch("bot.send_alert") as alert, patch("bot.time.time", lambda: t[0]):
        b._maybe_guard_unverified_entry(1000.0)        # starts the clock
        t[0] += 1500
        b._unverified_next = 0.0
        b._maybe_guard_unverified_entry(1000.0)
        alert.assert_not_called()                      # < 30 min: WARNING only
        t[0] += 400
        b._unverified_next = 0.0
        b._maybe_guard_unverified_entry(1000.0)
        assert alert.call_count == 1
        t[0] += 3600
        b._unverified_next = 0.0
        b._maybe_guard_unverified_entry(1000.0)
        assert alert.call_count == 1                   # one mail only
    assert b._unverified_record() is not None          # still blocks entries


def test_watch_runs_after_the_close_so_it_never_delays_it():
    b, mc = _make_bot()
    order = []
    mc.emergency_close.side_effect = lambda *a, **k: (order.append("close"),
                                                       (True, {"id": "1"}, None))[1]
    mc.cancel_entry_order.side_effect = lambda *a, **k: (order.append("cancel"), None)[1]
    with patch("bot.send_alert"):
        b._emergency_close(_open("long", 65_000.0, 0.01), "r1", 1000.0, "t")
    assert order == ["close", "cancel"]


# --- 381: the retry's own close order is the one priced -------------------------

def test_retry_persists_the_order_of_the_close_that_was_sent():
    b, mc = _pending_bot(0.010, close_coid="old-ns", close_order_id="1")
    mc.fetch_position.return_value = _open("long", 65_000.0, 0.010)
    mc.emergency_close.return_value = (False, {"clientOrderId": "new-ns", "id": "2"},
                                       RuntimeError("flat not confirmed"))
    with patch("bot.send_alert"), patch.object(b, "_alert_no_stop_failed"):
        b._maybe_finish_no_stop_close(1000.0)
    p = b._pending_record()
    assert p["close_coid"] == "new-ns" and p["close_order_id"] == "2"


def test_retry_failure_without_an_order_keeps_the_earlier_close_order():
    b, mc = _pending_bot(0.010, close_coid="old-ns", close_order_id="1")
    mc.fetch_position.return_value = _open("long", 65_000.0, 0.010)
    mc.emergency_close.return_value = (False, None, RuntimeError("rejected"))
    with patch("bot.send_alert"), patch.object(b, "_alert_no_stop_failed"):
        b._maybe_finish_no_stop_close(1000.0)
    assert b._pending_record()["close_coid"] == "old-ns"


# --- close-failed mail: short retry gap after a FAILED send --------------------

def test_failed_close_mail_is_not_retried_on_every_pass():
    b, mc = _make_bot()
    with patch("bot.send_alert", return_value=False) as alert:
        for _ in range(5):
            b._alert_close_failed("h", "HALT", "x")
    assert alert.call_count == 1


# --- HALT / kill booking uses the close order, not stale position state --------

def test_halt_close_booking_uses_the_orders_own_side_and_qty():
    b, mc = _make_bot()
    state.record_fill(side="long", qty=0.01, price=65_000.0, reason="entry",
                      client_order_id_root="r1")
    b._last_position_side, b._last_position_qty = "unknown", None   # stale / boot HALT
    order = {"id": "1", "average": 64_500.0, "side": "sell", "amount": 0.01}
    with patch("bot.send_alert"):
        b._book_halt_kill_close(order, "halt_close", "r1", None)
    with sqlite3.connect(state.DB_PATH) as c:
        row = c.execute("SELECT qty, price, pnl_usd FROM fills "
                        "WHERE reason='halt_close'").fetchone()
    assert row[0] == 0.01 and row[1] == 64_500.0 and row[2] == pytest.approx(-5.0)


# --- close_from_account_amt: hedge mode refuses -------------------------------------

def test_account_sized_close_refuses_in_hedge_mode():
    ex = MagicMock()
    c = BinanceClient(ex=ex, env="testnet", coid_prefix="snap-v1-", hedge_mode=True)
    with patch.object(c, "_account_position_amt", return_value=0.01), \
         pytest.raises(RuntimeError, match="hedge"):
        c.close_from_account_amt("BTC/USDT:USDT", "r1", "h")
    ex.create_order.assert_not_called()


# --- 1036: boot-flatten never books the entry price as the close price ---------------

def test_boot_flatten_with_an_unresolved_close_price_stays_unknown():
    from tests.test_close_then_sweep import _boot_ctx, _bot
    b, mc = _bot()
    mc.fetch_equity_usdt.return_value = 1000.0
    with _boot_ctx(), patch.object(b, "_resolve_fill_price", return_value=None), \
         patch.object(b, "_alert_boot_flatten") as alert_fn:
        b.boot()
        import bot
        ev = bot.state.enqueue_bot_event.call_args_list
    flat_ev = [c for c in ev if c.args and c.args[0] == "boot_flatten"]
    assert flat_ev and flat_ev[0].kwargs["price_usd"] is None
    assert flat_ev[0].kwargs["payload"]["pnl_usd"] is None
    assert alert_fn.call_args.args[2] is None


# --- 277: an unreadable strict re-read does not count toward acceptance ---------------

def test_flat_unknown_flat_is_not_accepted():
    b, mc = _make_bot()
    mc.fetch_position_checked.side_effect = [_flat(), RuntimeError("garbled"), _flat()]
    results = []
    for seq in (1, 2, 3):
        b._tick_seq = seq
        results.append(b._confirm_flat("site", stop_evidence=False))
    assert results == [False, False, False]
    assert b._flat_pending("site")                     # still pending, not forgotten
    mc.fetch_position_checked.side_effect = [_flat()]
    b._tick_seq = 4
    assert b._confirm_flat("site", stop_evidence=False) is True   # two clean ticks


# --- 4657588390: the pending-retry flat branch writes NULL equity when stale ----------

def test_pending_retry_flat_close_row_has_null_equity_when_equity_is_stale():
    b, mc = _pending_bot(0.010, close_coid="snap-x-1-ns", close_order_id="42")
    state.record_fill(side="long", qty=0.01, price=65_000.0, reason="entry",
                      client_order_id_root="r1")
    mc.fetch_position.return_value = _flat()
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _flat()
    mc.cancel_entry_order.return_value = True
    mc.ex.fetch_order.return_value = {"average": 64_000.0, "status": "closed"}
    b._flat_seen = {"no_stop_retry": (1, None)}
    b._equity_stale = True                      # a HALT-tick stand-in, not a real read
    with patch("bot.send_alert"):
        b._maybe_finish_no_stop_close(1234.5)
    with sqlite3.connect(state.DB_PATH) as c:
        eq = c.execute("SELECT equity_after FROM fills "
                       "WHERE reason='no_stop_close'").fetchone()
    assert eq == (None,)
