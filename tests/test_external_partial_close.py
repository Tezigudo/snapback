"""A position's size changed OUTSIDE the bot (manual / other-script reduce).

Regression: sol_supertrend 2026-09-24. Long 0.66 @ 104.74, God sold 0.33 @
114.94 by hand, then the TP filled the other 0.33 @ 124.10. The bot priced the
exit as the FULL entry qty x the TP fill (bot.py _detect_bracket_exit used the
entry fill row's qty). Record-keeping only: nothing is ever placed/cancelled.
"""
from __future__ import annotations

import sqlite3
import time
from unittest.mock import patch

from exchange import state
from tests.test_boot_resume import _flat, _make_bot, _open

ENTRY, QTY = 104.74, 0.66


def _rows():
    with sqlite3.connect(state.DB_PATH) as c:
        return c.execute(
            "SELECT side, reason, qty, price, pnl_usd FROM fills ORDER BY id"
        ).fetchall()


def _armed():
    state.record_fill(side="long", qty=QTY, price=ENTRY, reason="entry",
                      client_order_id_root="sol-st-1")
    bot, mc = _make_bot()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=QTY)
    with patch("bot.send_alert"):
        bot._detect_bracket_exit(1000.0)  # first observation: seeds tracking
    mc.ex.fetch_my_trades.reset_mock()
    return bot, mc


def _trade(side, amount, price, coid="x-cvBPrNm9abc", order="o1", ts=None):
    return {"side": side, "amount": amount, "price": price, "order": order,
            "timestamp": ts if ts is not None else time.time() * 1000 + 5000,
            "info": {"clientOrderId": coid}}


def _tick(bot, n=2):
    with patch("bot.send_alert") as alert:
        for _ in range(n):
            bot._detect_bracket_exit(1000.0)
    return alert


def test_incident_2026_09_24_final_pnl_prices_only_remaining_qty():
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.33, 114.94)]
    alert = _tick(bot, 4)  # debounce, record, then unchanged: ONE alert
    assert alert.call_count == 1
    assert _rows()[1][:4] == ("partial_close", "external_partial", 0.33, 114.94)
    assert abs(_rows()[1][4] - 3.366) < 1e-6
    assert bot._last_position_qty == 0.33

    mc.fetch_position.return_value = _flat()
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.33, 114.94),
                                          _trade("sell", 0.33, 124.10)]
    _tick(bot, 1)
    close = [r for r in _rows() if r[1] == "bracket_exit"][0]
    assert close[2] == 0.33 and abs(close[4] - 6.3888) < 1e-6
    total = sum(r[4] for r in _rows() if r[4] is not None)
    assert abs(total - 9.7548) < 1e-6
    assert abs(total - 0.66 * (124.10 - ENTRY)) > 2  # the old, wrong number
    mc.close_position.assert_not_called()


def test_no_extra_exchange_call_while_size_is_unchanged():
    bot, mc = _armed()
    with patch("bot.send_alert") as alert:
        bot._detect_bracket_exit(1000.0)
    mc.ex.fetch_my_trades.assert_not_called()
    mc.ex.fetch_ticker.assert_not_called()
    alert.assert_not_called()
    assert len(_rows()) == 1


def test_unmatched_trades_fall_back_to_ticker_and_say_so():
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.10, 99.0)]
    mc.ex.fetch_ticker.return_value = {"last": 115.0}
    alert = _tick(bot, 4)  # 2 attempts raise "not visible yet", 3rd falls back
    assert _rows()[1][3] == 115.0
    assert "ticker" in alert.call_args[0][1]


def test_one_tick_stale_read_writes_nothing():
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    _tick(bot, 1)
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=QTY)
    _tick(bot, 3)
    assert len(_rows()) == 1
    mc.ex.fetch_my_trades.assert_not_called()


def test_pricing_retries_are_bounded_then_row_is_written_unpriced():
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    mc.ex.fetch_my_trades.side_effect = RuntimeError("down")
    mc.ex.fetch_ticker.side_effect = RuntimeError("down")
    alert = _tick(bot, 8)
    assert mc.ex.fetch_my_trades.call_count == 3  # 3 attempts, never again
    assert alert.call_count == 1
    assert _rows()[1] == ("partial_close", "external_partial", 0.33, 0.0, None)
    assert bot._last_position_qty == 0.33


def test_our_own_bracket_fill_is_not_external():
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    mc.ex.fetch_my_trades.return_value = [
        _trade("sell", 0.33, 124.10, coid=bot.coid_prefix + "abc-t")]
    alert = _tick(bot, 3)
    alert.assert_not_called()
    assert [r[1] for r in _rows()] == ["entry", "bot_partial"]
    assert bot._last_position_qty == 0.33


def test_order_id_lookup_is_used_when_trade_has_no_coid():
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    t = _trade("sell", 0.33, 124.10)
    t["info"] = {}
    mc.ex.fetch_my_trades.return_value = [t]
    mc.ex.fetch_order.return_value = {"clientOrderId": bot.coid_prefix + "x-t"}
    alert = _tick(bot)
    alert.assert_not_called()
    assert [r[1] for r in _rows()] == ["entry", "bot_partial"]


def test_external_increase_is_alerted_recorded_and_not_an_exit():
    bot, mc = _armed()
    # an add moves the average entry price, which looks like a replacement
    mc.fetch_position.return_value = _open(entry_price=108.0, qty=0.9)
    alert = _tick(bot)
    assert alert.call_count == 1
    assert [r[1] for r in _rows()] == ["entry", "external_add"]  # no fake close
    assert bot._last_position_qty == 0.9
    mc.close_position.assert_not_called()


def test_reduce_then_add_then_tp_prices_the_net_qty():
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.65)
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.01, 114.0)]
    _tick(bot)
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.66)
    _tick(bot)
    assert [r[1] for r in _rows()] == ["entry", "external_partial", "external_add"]
    mc.fetch_position.return_value = _flat()
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.66, 124.10)]
    _tick(bot, 1)
    close = [r for r in _rows() if r[1] == "bracket_exit"][0]
    assert close[2] == 0.66


def test_open_entry_fill_looks_past_the_partial_rows():
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.33, 114.94)]
    _tick(bot)
    assert bot._open_entry_fill() == ("long", QTY, ENTRY)


def test_older_own_trade_of_equal_qty_is_not_adopted_when_sell_not_visible():
    """09-24 re-created: a previous position's `-e` sell of the same qty sits in
    the window and the manual sell has not reached userTrades yet."""
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    old = _trade("sell", 0.33, 99.0, coid=bot.coid_prefix + "old-e",
                 ts=time.time() * 1000 - 3600_000)
    mc.ex.fetch_my_trades.return_value = [old]
    mc.ex.fetch_ticker.return_value = {"last": 115.0}
    alert = _tick(bot, 2)  # confirmed, attempt 1: retry, nothing written
    assert len(_rows()) == 1 and alert.call_count == 0
    # the manual sell becomes visible
    mc.ex.fetch_my_trades.return_value = [old, _trade("sell", 0.33, 114.94)]
    alert = _tick(bot, 1)
    assert _rows()[1][:4] == ("partial_close", "external_partial", 0.33, 114.94)
    assert alert.call_count == 1


def test_flat_edge_uses_running_cost_basis_after_an_add():
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.33, 114.94)]
    _tick(bot)
    # add 0.33 @ 107.26 -> exchange average (0.33*104.74 + 0.33*107.26)/0.66 = 106.0
    mc.fetch_position.return_value = _open(entry_price=106.0, qty=0.66)
    _tick(bot)
    assert abs(_rows()[2][3] - 107.26) < 1e-6
    mc.fetch_position.return_value = _flat()
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.66, 124.10)]
    _tick(bot, 1)
    close = [r for r in _rows() if r[1] == "bracket_exit"][0]
    assert close[2] == 0.66 and abs(close[4] - 0.66 * (124.10 - 106.0)) < 1e-6


def test_add_without_prior_partial_is_priced_at_the_weighted_basis_on_tp():
    """entry 0.33 @ 104.74, manual add +0.33 (avg 106.0), TP fills 0.66."""
    state.record_fill(side="long", qty=0.33, price=ENTRY, reason="entry",
                      client_order_id_root="sol-st-1")
    bot, mc = _make_bot()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    with patch("bot.send_alert"):
        bot._detect_bracket_exit(1000.0)
    mc.fetch_position.return_value = _open(entry_price=106.0, qty=0.66)
    _tick(bot)
    assert abs(_rows()[1][3] - 107.26) < 1e-6
    mc.fetch_position.return_value = _flat()
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.66, 124.10)]
    _tick(bot, 1)
    close = [r for r in _rows() if r[1] == "bracket_exit"][0]
    assert close[2] == 0.66 and abs(close[4] - 0.66 * (124.10 - 106.0)) < 1e-6


def test_add_then_manual_reduce_then_tp_does_not_fall_back_to_prev_qty():
    state.record_fill(side="long", qty=0.33, price=ENTRY, reason="entry",
                      client_order_id_root="sol-st-1")
    bot, mc = _make_bot()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    with patch("bot.send_alert"):
        bot._detect_bracket_exit(1000.0)
    mc.fetch_position.return_value = _open(entry_price=106.0, qty=0.66)
    _tick(bot)
    mc.fetch_position.return_value = _open(entry_price=106.0, qty=0.33)
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.33, 114.94)]
    _tick(bot)
    mc.fetch_position.return_value = _flat()
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.33, 124.10)]
    _tick(bot, 1)
    close = [r for r in _rows() if r[1] == "bracket_exit"][0]
    assert close[2] == 0.33 and abs(close[4] - 0.33 * (124.10 - 106.0)) < 1e-6
    total = sum(r[4] for r in _rows() if r[4] is not None)
    assert abs(total - 0.33 * (114.94 - 106.0 + 124.10 - 106.0)) < 1e-6


def test_open_entry_fill_returns_running_basis_after_reduce_then_add():
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.33, 114.94)]
    _tick(bot)
    mc.fetch_position.return_value = _open(entry_price=106.0, qty=0.66)
    _tick(bot)
    side, _, basis = bot._open_entry_fill()
    assert side == "long" and abs(basis - 106.0) < 1e-6


def test_own_partial_reduction_is_recorded_so_flat_edge_prices_the_rest():
    """Our TP child fills 0.33 @ 120, then the remaining 0.33 @ 124.10."""
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    mc.ex.fetch_my_trades.return_value = [
        _trade("sell", 0.33, 120.0, coid=bot.coid_prefix + "abc-t")]
    alert = _tick(bot, 3)
    alert.assert_not_called()
    assert _rows()[1][:3] == ("partial_close", "bot_partial", 0.33)
    assert abs(_rows()[1][4] - 0.33 * (120.0 - ENTRY)) < 1e-6
    mc.fetch_position.return_value = _flat()
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.33, 124.10)]
    _tick(bot, 1)
    close = [r for r in _rows() if r[1] == "bracket_exit"][0]
    assert close[2] == 0.33
    total = sum(r[4] for r in _rows() if r[4] is not None)
    assert abs(total - (0.33 * (120.0 - ENTRY) + 0.33 * (124.10 - ENTRY))) < 1e-6


def test_pending_size_change_does_not_leak_into_the_next_position():
    bot, mc = _armed()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    _tick(bot, 1)  # pending = 0.33, nothing written
    mc.fetch_position.return_value = _flat()
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.66, 124.10)]
    _tick(bot, 1)
    assert bot._ext_pending is None and bot._ext_attempts == 0
    state.record_fill(side="long", qty=QTY, price=ENTRY, reason="entry",
                      client_order_id_root="sol-st-2")
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=QTY)
    _tick(bot, 1)
    mc.ex.fetch_my_trades.reset_mock()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    _tick(bot, 1)  # first tick on the NEW position: debounce, no write
    mc.ex.fetch_my_trades.assert_not_called()
    assert [r[1] for r in _rows()].count("external_partial") == 0


def test_tp_during_debounce_prices_the_pending_reduction_from_trades():
    """Entry 0.33; manual -0.11 @ 110 seen once (pending); TP 0.22 @ 124 fills."""
    state.record_fill(side="long", qty=0.33, price=ENTRY, reason="entry",
                      client_order_id_root="sol-st-9")
    bot, mc = _make_bot()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    with patch("bot.send_alert"):
        bot._detect_bracket_exit(1000.0)
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.22)
    _tick(bot, 1)  # debounce tick 1: nothing written
    assert len(_rows()) == 1
    mc.fetch_position.return_value = _flat()
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.11, 110.0, order="o1"),
                                          _trade("sell", 0.22, 124.0, order="o2")]
    _tick(bot, 1)
    rows = _rows()
    assert rows[1][:4] == ("partial_close", "external_partial", 0.11, 110.0)
    close = [r for r in rows if r[1] == "bracket_exit"][0]
    assert close[2] == 0.22 and close[3] == 124.0
    total = sum(r[4] for r in rows if r[4] is not None)
    assert abs(total - (0.11 * (110.0 - ENTRY) + 0.22 * (124.0 - ENTRY))) < 1e-6


def test_tp_during_debounce_with_unreadable_trades_keeps_old_pricing_and_alerts():
    state.record_fill(side="long", qty=0.33, price=ENTRY, reason="entry",
                      client_order_id_root="sol-st-9")
    bot, mc = _make_bot()
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.33)
    with patch("bot.send_alert"):
        bot._detect_bracket_exit(1000.0)
    mc.fetch_position.return_value = _open(entry_price=ENTRY, qty=0.22)
    _tick(bot, 1)
    mc.fetch_position.return_value = _flat()
    mc.ex.fetch_my_trades.return_value = [_trade("sell", 0.33, 124.0)]
    alert = _tick(bot, 1)
    assert [r[1] for r in _rows()] == ["entry", "bracket_exit"]
    assert _rows()[1][2] == 0.33
    assert any("Ledger may be off" in c[0][0] for c in alert.call_args_list)
