"""Sourcery review fixes (PRs #9-#11): unverified-entry guard cluster."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from exchange.binance_client import UnprotectedFillError
from tests.test_orphan_bracket_cleanup import _flat, _make_bot, _open

REC = {"kind": "entry_unverified", "root": "r1", "side": "long", "since": 0.0,
       "sl_distance": 100.0, "tp_distance": 200.0, "place_tp": True}


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch("bot.time.sleep"):
        yield


def _guard_bot(*, filled, recheck, rec=None):
    b, mc = _make_bot()
    b._set_unverified(dict(rec or REC))
    b._unverified_next = 0.0
    mc.fetch_position.return_value = _flat()
    # 1st strict read (flat, before the cancel), 2nd = the re-read after it
    mc.fetch_position_checked.side_effect = [_flat(), recheck]
    mc.cancel_entry_order.return_value = True
    mc.entry_order_filled.return_value = filled
    b.constraints = MagicMock(qty_step=0.001)
    return b, mc


def test_fill_missed_by_a_lagging_order_status_is_still_protected():
    # 4657588376: filled reads 0 but the position re-read shows the fill
    b, mc = _guard_bot(filled=0.0, recheck=_open("long", 65_000.0, 0.01))
    with patch("bot.send_alert"), \
         patch.object(b, "_protect_or_close_open", return_value=False) as prot:
        b._maybe_guard_unverified_entry(1000.0)
    prot.assert_called_once()
    assert b._unverified_record() is not None        # kept: stop not yet placed


def test_opposite_side_at_the_cancel_is_left_untouched():
    # 4657588360: no entry row, no bracket on a foreign position
    b, mc = _guard_bot(filled=0.01, recheck=_open("short", 65_000.0, 0.01))
    with patch("bot.send_alert") as alert, \
         patch.object(b, "_protect_or_close_open") as prot:
        b._maybe_guard_unverified_entry(1000.0)
    prot.assert_not_called()
    mc.protect_open_position.assert_not_called()
    assert b._unverified_record() is None
    assert "not ours" in alert.call_args.args[0]


def test_after_emergency_close_marker_suppresses_the_false_ledger_mail():
    rec = {**REC, "after_emergency_close": True}
    b, mc = _guard_bot(filled=0.01, recheck=_flat(), rec=rec)
    with patch("bot.send_alert") as alert:
        b._maybe_guard_unverified_entry(1000.0)
    assert b._unverified_record() is None
    alert.assert_not_called()


def test_partly_attributable_position_keeps_the_record_and_alerts_once():
    # 4657588366: stop cannot be placed, position not wholly ours
    b, mc = _make_bot()
    mc.ex.fetch_open_orders.return_value = []
    mc.fetch_algo_orders.return_value = ([], True)
    mc.protect_open_position.side_effect = UnprotectedFillError(
        side="long", filled_qty=0.01, fill_price=65_000.0, filled_as="market",
        entry=None, sl_error=RuntimeError("rejected"), closed=False,
        close_order=None, close_error=None)
    pos = _open("long", 65_000.0, 0.02)
    with patch("bot.send_alert") as alert:
        r1 = b._protect_or_close_open(pos, "r1", 100.0, 200.0, True, 1000.0,
                                      why="t", record_if_flat=False,
                                      may_close=False, entry_qty=0.01)
        r2 = b._protect_or_close_open(pos, "r1", 100.0, 200.0, True, 1000.0,
                                      why="t", record_if_flat=False,
                                      may_close=False, entry_qty=0.01)
    assert (r1, r2) == (False, False)
    assert mc.protect_open_position.call_count == 2    # retried
    assert alert.call_count == 1                       # alerted once


# --- entry-cancel-race / fill-price (no-stop emergency close) -----------------

def _pending_bot(qty=0.010, **extra):
    b, mc = _make_bot()
    rec = {"kind": "no_stop_close", "root": "r1", "side": "long", "qty": qty,
           "entry_price": 65_000.0, "stop_price": 64_000.0, "why": "t", **extra}
    b._no_stop_pending = rec
    b._pending_meta_stale = False
    b._no_stop_next_try = 0.0
    b.constraints = MagicMock(qty_step=0.001)
    return b, mc


def test_residual_attributable_to_the_entry_order_is_closed_not_abandoned():
    # 4657588370: pending 0.010, position 0.004 same side, entry filled 0.014
    b, mc = _pending_bot(0.010)
    mc.fetch_position.return_value = _open("long", 65_000.0, 0.004)
    mc.entry_order_filled.return_value = 0.014
    mc.emergency_close.return_value = (True, {"id": "1", "average": 64_900.0}, None)
    with patch("bot.send_alert") as alert:
        b._maybe_finish_no_stop_close(1000.0)
    assert mc.emergency_close.call_args.kwargs["qty"] == 0.004
    assert not any("ABANDONED" in c.args[0] for c in alert.call_args_list)


def test_unattributable_size_mismatch_is_still_abandoned():
    b, mc = _pending_bot(0.010)
    mc.fetch_position.return_value = _open("long", 65_000.0, 0.004)
    mc.entry_order_filled.return_value = 0.0
    with patch("bot.send_alert") as alert:
        b._maybe_finish_no_stop_close(1000.0)
    mc.emergency_close.assert_not_called()
    assert any("ABANDONED" in c.args[0] for c in alert.call_args_list)


def test_flat_confirmed_retry_prices_the_close_from_the_persisted_order():
    # 4657588381
    import sqlite3

    from exchange import state
    b, mc = _pending_bot(0.010, close_coid="snap-x-1-ns", close_order_id="42")
    state.record_fill(side="long", qty=0.01, price=65_000.0, reason="entry",
                      client_order_id_root="r1")
    mc.fetch_position.return_value = _flat()
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _flat()
    b._flat_seen = {"no_stop_retry": (1, None)}
    mc.ex.fetch_order.return_value = {"average": 64_000.0, "status": "closed"}
    with patch("bot.send_alert"):
        b._maybe_finish_no_stop_close(1000.0)
        b._maybe_finish_no_stop_close(1000.0)
    with sqlite3.connect(state.DB_PATH) as c:
        px, pnl = c.execute("SELECT price, pnl_usd FROM fills "
                            "WHERE reason='no_stop_close'").fetchone()
    assert px == 64_000.0 and pnl == pytest.approx(-10.0)


def test_unproven_entry_cancel_after_emergency_close_rearms_the_guard():
    # 4657581356 (NEEDS-GOD decision): close first, then guard `-e`
    b, mc = _make_bot()
    mc.cancel_entry_order.return_value = None
    with patch("bot.send_alert"):
        b._finish_no_stop("long", 0.01, 65_000.0, "r1", 1000.0, closed=True,
                          order={"id": "1", "average": 64_900.0}, close_error=None,
                          why="t", record_if_flat=False, stop_price=64_000.0)
    u = b._unverified_record()
    assert u and u["root"] == "r1" and u["after_emergency_close"] is True
    assert b._pending_record() is None


# --- stale equity (4657588390) -------------------------------------------------

@pytest.mark.parametrize("equity,stale", [(None, False), (1000.0, True)])
def test_entry_row_is_written_with_null_equity_when_equity_is_unknown_or_stale(equity, stale):
    import sqlite3

    from exchange import state
    b, mc = _make_bot()
    b._equity_stale = stale
    b._record_entry_fill("long", 0.01, 65_000.0, "r1", equity, "market")
    with sqlite3.connect(state.DB_PATH) as c:
        rows = c.execute("SELECT reason, equity_after FROM fills").fetchall()
    assert rows == [("entry", None)]



# --- small alert / audit / book-read fixes -------------------------------------

def test_resize_mail_still_sent_when_the_event_write_fails_and_event_links_the_signal():
    b, mc = _make_bot()
    with patch("bot.state.record_event", side_effect=RuntimeError("db")), \
         patch("bot.send_alert") as alert:
        b._note_notional_resize("long", 0.02, 1300.0, 0.01, 650.0, 1000.0,
                                signal_id="s1")
    alert.assert_called_once()
    with patch("bot.state.record_event") as rec, patch("bot.send_alert"):
        b._note_notional_resize("long", 0.02, 1300.0, 0.01, 650.0, 1000.0,
                                signal_id="s1")
    assert rec.call_args.kwargs["signal_id"] == "s1"


def test_close_failed_throttle_is_set_only_after_a_successful_send():
    b, mc = _make_bot()
    with patch("bot.send_alert", return_value=False) as alert:
        b._alert_close_failed("h", "HALT", "x")
        b._alert_close_failed("h", "HALT", "x")
    assert alert.call_count == 2                       # failed send: not throttled
    with patch("bot.send_alert", return_value=True) as alert:
        b._alert_close_failed("h", "HALT", "x")
        b._alert_close_failed("h", "HALT", "x")
    assert alert.call_count == 1


def test_close_failed_throttle_survives_a_restart():
    b1, _ = _make_bot()
    b2, _ = _make_bot()                                # a restarted process
    with patch("bot.send_alert", return_value=True) as alert:
        b1._alert_close_failed("h", "boot-flatten", "x")
        b2._alert_close_failed("h", "boot-flatten", "x")
    assert alert.call_count == 1


def test_malformed_algo_row_makes_the_book_unreadable():
    from exchange.binance_client import BinanceClient
    ex = MagicMock()
    ex.fapiPrivateGetOpenAlgoOrders.return_value = [{"clientAlgoId": "a"}, "junk"]
    c = BinanceClient(ex=ex, env="testnet", coid_prefix="snap-v1-")
    ex.market.return_value = {"id": "BTCUSDT"}
    assert c.fetch_algo_orders("BTC/USDT:USDT") == ([], False)
