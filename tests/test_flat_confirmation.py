"""A single "flat" position read must not strip a live position's stop.

Regression for the 2026-11 review finding N1 (A-blind Pass 2 item A,
B-informed N1): `_detect_bracket_exit` acted on ONE flat read — it swept the
leg's SL and TP, advanced the snapshot, and (elsewhere) `_maybe_reprotect`
erased `active_bracket` and `_maybe_finish_no_stop_close` cleared the pending
close with a false "now flat" mail. fetch_position folds empty / malformed
payloads into flat, and donchian / SOL have no reprotect, so nothing put the
stop back.

These tests script every read explicitly (they do NOT use agree_on_flat).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from exchange import state
from exchange.binance_client import BinanceClient, PositionUnknown
from tests.test_orphan_bracket_cleanup import _flat, _make_bot, _open

ENTRY, QTY, EXIT_PX, ROOT = 65_000.0, 0.01, 64_000.0, "1787375710921"
SL = {"algoId": "1", "clientAlgoId": f"snap-v1-{ROOT}-s", "side": "SELL",
      "reduceOnly": True, "algoStatus": "NEW"}
TP = {"algoId": "2", "clientAlgoId": f"snap-v1-{ROOT}-t", "side": "SELL",
      "reduceOnly": True, "algoStatus": "NEW"}


def _bot():
    b, mc = _make_bot()
    state.record_fill(side="long", qty=QTY, price=ENTRY, reason="entry",
                      client_order_id_root=ROOT)
    b._last_position_side = "long"
    b._last_position_entry = ENTRY
    b._last_position_qty = QTY
    b._last_entry_root = ROOT
    mc.fetch_position.return_value = _flat()
    mc.fetch_position_checked.side_effect = None   # scripted per test below
    mc.fetch_algo_orders.side_effect = None
    mc.fetch_algo_orders.return_value = ([], True)
    mc.ex.fetch_my_trades.return_value = [{"side": "sell", "price": EXIT_PX}]
    return b, mc


def _tick(b):
    with patch("bot.send_alert") as alert:
        b._detect_bracket_exit(equity=1000.0)
    return alert


def _fills():
    import sqlite3
    with sqlite3.connect(state.DB_PATH) as c:
        return [r[0] for r in c.execute("SELECT reason FROM fills ORDER BY id")]


def _assert_untouched(b, mc):
    mc.cancel_open_orders.assert_not_called()
    assert _fills() == ["entry"]
    assert b._last_position_side == "long"
    assert b._last_position_entry == ENTRY
    assert b._last_position_qty == QTY
    assert b._last_entry_root == ROOT
    assert b._exit_retry_n == 0


# --- bracket-exit site --------------------------------------------------------

def test_flat_read_contradicted_by_reread_cancels_nothing():
    b, mc = _bot()
    mc.fetch_position_checked.return_value = _open("long", ENTRY, QTY)
    b._ext_pending = 0.005  # a size-change debounce in flight must survive
    _tick(b)
    _assert_untouched(b, mc)
    assert b._ext_pending == 0.005
    # and the next (healthy) tick is an ordinary open tick, not a "first look"
    mc.fetch_position.return_value = _open("long", ENTRY, QTY)
    _tick(b)
    mc.cancel_open_orders.assert_not_called()
    assert not b._flat_pending(b.BRACKET_SITE)


def test_reread_error_is_unknown_not_flat():
    b, mc = _bot()
    mc.fetch_position_checked.side_effect = PositionUnknown("no contracts field")
    _tick(b)
    _assert_untouched(b, mc)
    assert b._entry_blocked()  # no entry while the flat is unconfirmed


def test_confirmed_flat_with_a_bracket_leg_gone_sweeps_same_tick():
    """TP filled: the TP is gone from the algo book, the SL sibling rests.
    That IS the fill evidence — sweep and record on this tick (no latency)."""
    b, mc = _bot()
    state.set_meta(b.TP_PLACED_KEY, ROOT)        # this position's TP was placed
    mc.fetch_position_checked.return_value = _flat()
    mc.fetch_algo_orders.return_value = ([SL], True)
    _tick(b)
    mc.cancel_open_orders.assert_called_once_with("BTC/USDT:USDT", coid_prefix="snap-v1-")
    assert _fills() == ["entry", "bracket_exit"]
    assert b._last_position_side == "flat"
    assert not b._flat_pending(b.BRACKET_SITE)


@pytest.mark.parametrize("book", [([SL, TP], True), ([], False), ([SL], True)],
                         ids=["bracket-intact", "algo-book-unreadable",
                              "tp-missing-but-never-recorded-placed"])
def test_no_fill_evidence_defers_then_accepts_with_an_alert(book):
    b, mc = _bot()
    mc.fetch_position_checked.return_value = _flat()
    mc.fetch_algo_orders.return_value = book
    alert = _tick(b)
    _assert_untouched(b, mc)
    alert.assert_not_called()
    assert b._entry_blocked()
    # flat again next tick → accepted (manual close / liquidation), mailed once
    alert = _tick(b)
    mc.cancel_open_orders.assert_called_once()
    assert _fills() == ["entry", "bracket_exit"]
    subjects = [c.args[0] for c in alert.call_args_list]
    assert any("position flat" in s for s in subjects), subjects
    assert not b._flat_pending(b.BRACKET_SITE)


@pytest.mark.parametrize("recorded", ["", "1700000000000"], ids=["tp-failed", "other-root"])
def test_missing_tp_is_not_fill_evidence_unless_this_positions_tp_was_placed(recorded):
    """F6: TP placement failed ("stop is resting, position kept") or a stale
    record for another root — a missing TP must not bypass the debounce."""
    b, mc = _bot()
    state.set_meta(b.TP_PLACED_KEY, recorded)
    mc.fetch_position_checked.return_value = _flat()
    mc.fetch_algo_orders.return_value = ([SL], True)
    _tick(b)
    _assert_untouched(b, mc)


def test_record_tp_placed_at_entry():
    b, _mc = _bot()
    b._record_tp_placed({"sl": {}, "tp": {}}, "r9", place_tp=True)
    assert state.get_meta(b.TP_PLACED_KEY) == "r9"
    b._record_tp_placed({"sl": {}, "tp_error": "-2021"}, "r10", place_tp=True)
    assert state.get_meta(b.TP_PLACED_KEY) == ""
    b._record_tp_placed({"sl": {}}, "r11", place_tp=False)
    assert state.get_meta(b.TP_PLACED_KEY) == ""


def test_flat_then_open_resets_the_debounce():
    b, mc = _bot()
    mc.fetch_position_checked.return_value = _flat()
    mc.fetch_algo_orders.return_value = ([SL, TP], True)
    _tick(b)                                         # 1/2
    mc.fetch_position.return_value = _open("long", ENTRY, QTY)
    _tick(b)                                         # open again → reset
    mc.fetch_position.return_value = _flat()
    _tick(b)                                         # 1/2 again, not 2/2
    _assert_untouched(b, mc)


def test_retry_of_an_already_swept_exit_still_confirms_on_the_fast_path():
    """A flat-edge retry (first pass swept, lookup failed) re-confirms: the
    algo book is empty, so it records on the same tick."""
    b, mc = _bot()
    b._exit_retry_n = 1
    mc.fetch_position_checked.return_value = _flat()
    mc.fetch_algo_orders.return_value = ([], True)
    _tick(b)
    assert _fills() == ["entry", "bracket_exit"]
    mc.fetch_position_checked.assert_called_once()


def test_pending_replacement_retry_is_not_swept_on_a_false_flat():
    """Replacement retry pending (_exit_retry_n=1, snapshot = OLD position) and
    the REPLACEMENT position carries a fresh bracket. One false flat read must
    not strip it, nor book a bracket_exit off the replacement's entry row."""
    b, mc = _bot()
    b._exit_retry_n = 1
    mc.fetch_position_checked.return_value = _open("long", 66_000.0, QTY)
    _tick(b)
    mc.cancel_open_orders.assert_not_called()
    assert _fills() == ["entry"]


def test_persistently_unreadable_reread_escalates_to_error(caplog):
    import logging
    b, mc = _bot()
    mc.fetch_position_checked.side_effect = PositionUnknown("no contracts")
    with caplog.at_level(logging.WARNING):
        for _ in range(b.FLAT_UNKNOWN_ERROR_TICKS):
            _tick(b)
    assert any(r.levelno == logging.ERROR and "NOT confirmed" in r.getMessage()
               for r in caplog.records)
    _assert_untouched(b, mc)


# --- no-stop retry site -------------------------------------------------------

def test_no_stop_retry_flat_contradicted_keeps_pending():
    from tests.test_no_stop_fixpass import _pending_after_failed_close
    b, _ex = _pending_after_failed_close()
    pending = b._pending_record()
    b.client.fetch_position = MagicMock(return_value=_flat())
    b.client.fetch_position_checked = MagicMock(
        return_value=_open(pending["side"], 65_000.0, float(pending["qty"])))
    with patch("bot.send_alert") as alert:
        b._maybe_finish_no_stop_close(150.0)
    assert b._pending_record() is not None
    assert not any("flat" in c.args[0].lower() for c in alert.call_args_list)


# --- reprotect site -----------------------------------------------------------

def test_reprotect_single_flat_keeps_active_bracket():
    b, mc = _bot()
    b.params = {**b.params, "reprotect": {"enabled": True}}
    state.set_meta("active_bracket", '{"side": "long"}')
    mc.fetch_position_checked.return_value = _open("long", ENTRY, QTY)
    b._maybe_reprotect(equity=1000.0)
    assert state.get_meta("active_bracket") == '{"side": "long"}'


# --- the strict read itself -------------------------------------------------

def _client(payload):
    ex = MagicMock()
    ex.fetch_positions.return_value = payload
    return BinanceClient(ex=ex, env="testnet", coid_prefix="snap-v1-")


def _acct(ex, rows):
    ex.fapiPrivateV2GetAccount.return_value = {"positions": rows}


def test_empty_list_is_flat_only_if_the_account_endpoint_confirms():
    # V3 positionRisk omits flat symbols; the INDEPENDENT account endpoint
    # (fapi v2 account, lists every symbol) must say positionAmt == 0 (F8)
    c = _client([])
    c.ex.market.return_value = {"id": "BTCUSDT"}
    _acct(c.ex, [{"symbol": "BTCUSDT", "positionAmt": "0.000"},
                 {"symbol": "ETHUSDT", "positionAmt": "1.0"}])
    assert c.fetch_position_checked("BTC/USDT:USDT").side == "flat"


@pytest.mark.parametrize("acct", [
    "raises", "no-method", "no-row", "nonzero", "garbage",
], ids=lambda a: a)
def test_empty_list_without_independent_confirmation_is_unknown(acct):
    c = _client([])
    c.ex.market.return_value = {"id": "BTCUSDT"}
    if acct == "raises":
        c.ex.fapiPrivateV2GetAccount.side_effect = RuntimeError("timeout")
    elif acct == "no-method":
        del c.ex.fapiPrivateV2GetAccount
    elif acct == "no-row":
        _acct(c.ex, [{"symbol": "ETHUSDT", "positionAmt": "0"}])
    elif acct == "nonzero":   # positionRisk stale-empty, account shows the position
        _acct(c.ex, [{"symbol": "BTCUSDT", "positionAmt": "0.010"}])
    else:
        _acct(c.ex, [{"symbol": "BTCUSDT", "positionAmt": "abc"}])
    with pytest.raises(PositionUnknown):
        c.fetch_position_checked("BTC/USDT:USDT")


def test_explicit_zero_row_is_flat_without_the_account_read():
    c = _client([{"symbol": "BTC/USDT:USDT", "contracts": 0.0, "side": "long"}])
    assert c.fetch_position_checked("BTC/USDT:USDT").side == "flat"
    c.ex.fapiPrivateV2GetAccount.assert_not_called()


@pytest.mark.parametrize("payload", [
    None,
    [{"symbol": "BTC/USDT:USDT", "contracts": None, "side": "long"}],
    [{"symbol": "BTC/USDT:USDT", "side": "long"}],
    [{"symbol": "BTC/USDT:USDT", "contracts": "abc", "side": "long"}],
], ids=["not-a-list", "contracts-None", "contracts-missing", "contracts-garbage"])
def test_checked_read_unknown_on_malformed(payload):
    with pytest.raises(PositionUnknown):
        _client(payload).fetch_position_checked("BTC/USDT:USDT")


def test_checked_read_open():
    p = _client([{"symbol": "BTC/USDT:USDT", "contracts": 0.01, "side": "long",
                  "entryPrice": 65000.0}]).fetch_position_checked("BTC/USDT:USDT")
    assert (p.side, p.qty) == ("long", 0.01)


# --- F9: the debounce counts STRICTLY consecutive loop ticks ------------------

def test_debounce_resets_across_a_skipped_tick():
    b, mc = _bot()
    mc.fetch_position_checked.return_value = _flat()
    mc.fetch_algo_orders.return_value = ([SL, TP], True)
    b._tick_seq = 10
    _tick(b)                                  # 1/2 at tick 10
    b._tick_seq = 12                          # tick 11: detector skipped (equity None)
    _tick(b)                                  # streak broken: 1/2 again, NOT accepted
    _assert_untouched(b, mc)
    b._tick_seq = 13
    _tick(b)                                  # 2/2 on consecutive ticks: accepted
    mc.cancel_open_orders.assert_called_once()


def test_two_calls_in_one_tick_do_not_confirm():
    b, mc = _bot()
    mc.fetch_position_checked.return_value = _flat()
    mc.fetch_algo_orders.return_value = ([SL, TP], True)
    b._tick_seq = 5
    _tick(b)
    _tick(b)                                  # same tick: still 1/2
    _assert_untouched(b, mc)


# --- F10: a bot close confirmed late is not called "manual close?" -----------

def test_late_confirmation_of_a_bot_close_says_bot_close():
    b, mc = _bot()
    b._bot_close_unconfirmed = {"why": "time-stop", "leg": "x", "ts": 0.0}
    mc.fetch_position_checked.return_value = _flat()
    mc.fetch_algo_orders.return_value = ([SL, TP], True)   # sweep never ran
    _tick(b)
    alert = _tick(b)
    subjects = [c.args[0] for c in alert.call_args_list]
    assert any("bot time-stop close now confirmed flat" in s for s in subjects), subjects
    assert not any("position flat, bracket" in s for s in subjects)
    assert b._bot_close_unconfirmed is None


def test_without_a_bot_close_it_still_asks_manual_close():
    b, mc = _bot()
    b._bot_close_unconfirmed = None
    mc.fetch_position_checked.return_value = _flat()
    mc.fetch_algo_orders.return_value = ([SL, TP], True)
    _tick(b)
    alert = _tick(b)
    body = " ".join(c.args[1] for c in alert.call_args_list)
    assert "manual close or liquidation?" in body


# --- G10 (round 3): account-endpoint fallback V2 -> V3 -----------------------

def test_v2_failing_falls_back_to_v3_present_zero_row():
    c = _client([])
    c.ex.market.return_value = {"id": "BTCUSDT"}
    c.ex.fapiPrivateV2GetAccount.side_effect = RuntimeError("410 gone")
    c.ex.fapiPrivateV3GetAccount.return_value = {
        "positions": [{"symbol": "BTCUSDT", "positionAmt": "0"}]}
    assert c.fetch_position_checked("BTC/USDT:USDT").side == "flat"


def test_v3_absent_row_is_not_flat():
    # V3 omits flat symbols with no orders: absence there cannot confirm flat
    c = _client([])
    c.ex.market.return_value = {"id": "BTCUSDT"}
    del c.ex.fapiPrivateV2GetAccount
    c.ex.fapiPrivateV3GetAccount.return_value = {"positions": []}
    with pytest.raises(PositionUnknown):
        c.fetch_position_checked("BTC/USDT:USDT")


def test_v2_used_first_when_it_answers():
    c = _client([])
    c.ex.market.return_value = {"id": "BTCUSDT"}
    _acct(c.ex, [{"symbol": "BTCUSDT", "positionAmt": "0"}])
    c.fetch_position_checked("BTC/USDT:USDT")
    c.ex.fapiPrivateV3GetAccount.assert_not_called()
