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
