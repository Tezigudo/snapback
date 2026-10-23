"""Zero-price bot close rows are repaired from exchange trades.

Live 2026-10-21 (SOL): channel_exit coid sol-st-1791403209120-ce, fetch_order
raised once -> close row price=0.0 pnl=None. Exchange truth: buy 0.99 @ 108.50,
realizedPnl +7.2369.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime, timedelta

import pytest

from exchange import state
from tests.test_channel_exit import _make_bot

PREFIX = "snap-v1-"
ROOT = "1791403209120"


class FakeEx:
    def __init__(self, orders=None, trades=None, fail=False):
        self.orders = orders or []
        self.trades = trades or []
        self.fail = fail
        self.calls = 0

    def fetch_orders(self, symbol, since=None, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("exchange down")
        return self.orders

    def fetch_my_trades(self, symbol, since=None, **kw):
        if self.fail:
            raise RuntimeError("exchange down")
        return self.trades


def _order(oid, leg, root=ROOT):
    return {"id": oid, "clientOrderId": f"{PREFIX}{root}-{leg}"}


def _trade(oid, qty, px, pnl, ts_ms, **info):
    return {"order": oid, "amount": qty, "price": px, "timestamp": ts_ms,
            "info": {"realizedPnl": str(pnl), "orderId": oid, **info}}


def _row(price=0.0, root=ROOT, reason="channel_exit", pnl=None, ts=None, qty=0.99):
    ts = ts or datetime.now(UTC).isoformat()
    with sqlite3.connect(state.DB_PATH) as c:
        cur = c.execute(
            "INSERT INTO fills(ts, side, qty, price, pnl_usd, reason, "
            "client_order_id_root) VALUES (?,?,?,?,?,?,?)",
            (ts, "close", qty, price, pnl, reason, root))
        return cur.lastrowid, ts


def _get(fid):
    with sqlite3.connect(state.DB_PATH) as c:
        return c.execute("SELECT price, pnl_usd FROM fills WHERE id=?", (fid,)).fetchone()


def _now_ms(ts):
    return int(datetime.fromisoformat(ts).timestamp() * 1000) - 2000


def _bot(ex):
    bot, mc = _make_bot(strategy_name="donchian-v3")
    mc.ex = ex
    return bot


def test_zero_price_channel_exit_is_repaired():
    fid, ts = _row()
    ex = FakeEx([_order("9", "ce")], [_trade("9", 0.99, 108.50, 7.2369, _now_ms(ts))])
    bot = _bot(ex)
    bot._maybe_backfill_close_prices()
    price, pnl = _get(fid)
    assert price == pytest.approx(108.50)
    assert pnl == pytest.approx(7.2369)


def test_multi_trade_fill_is_qty_weighted_and_pnl_summed():
    fid, ts = _row(qty=1.0)
    t = _now_ms(ts)
    ex = FakeEx([_order("9", "ce")], [
        _trade("9", 0.30, 100.0, 1.0, t), _trade("9", 0.70, 110.0, 2.5, t)])
    _bot(ex)._maybe_backfill_close_prices()
    price, pnl = _get(fid)
    assert price == pytest.approx(0.3 * 100 + 0.7 * 110)
    assert pnl == pytest.approx(3.5)


def test_unrelated_trades_ignored():
    fid, ts = _row(qty=1.0)
    t = _now_ms(ts)
    ex = FakeEx(
        [_order("9", "ce"), _order("8", "e"), _order("7", "ce", root="999"),
         _order("5", "ce")],
        [_trade("9", 1.0, 108.5, 7.0, t),
         _trade("8", 1.0, 200.0, 0.0, t),                    # entry, same root
         _trade("7", 1.0, 300.0, 9.0, t),                    # other root
         _trade("6", 1.0, 400.0, 9.0, t),                    # unknown order
         _trade("5", 1.0, 500.0, 9.0, t - 3 * 3600 * 1000)])  # outside window
    _bot(ex)._maybe_backfill_close_prices()
    assert _get(fid) == (pytest.approx(108.5), pytest.approx(7.0))


def test_direct_client_order_id_on_trade_matches():
    fid, ts = _row(qty=1.0)
    ex = FakeEx([], [_trade("1", 1.0, 50.0, 2.0, _now_ms(ts),
                            clientOrderId=f"{PREFIX}{ROOT}-x")])
    _bot(ex)._maybe_backfill_close_prices()
    assert _get(fid) == (pytest.approx(50.0), pytest.approx(2.0))


def test_exchange_failure_leaves_row_untouched_and_does_not_raise():
    fid, _ = _row(pnl=None)
    bot = _bot(FakeEx(fail=True))
    bot._maybe_backfill_close_prices()
    assert _get(fid) == (0.0, None)


def test_no_matching_trades_leaves_row_untouched():
    fid, _ = _row()
    _bot(FakeEx([], []))._maybe_backfill_close_prices()
    assert _get(fid) == (0.0, None)


def test_rate_limited():
    _row()
    ex = FakeEx([], [])
    bot = _bot(ex)
    bot._maybe_backfill_close_prices()
    bot._maybe_backfill_close_prices()
    assert ex.calls == 1
    bot._backfill_next = 0.0
    bot._maybe_backfill_close_prices()
    assert ex.calls == 2


def test_no_exchange_call_without_work():
    _row(price=108.5, pnl=7.0)
    ex = FakeEx([], [])
    _bot(ex)._maybe_backfill_close_prices()
    assert ex.calls == 0


def test_priced_rows_never_touched():
    fid, ts = _row(price=111.0, pnl=1.5)
    ex = FakeEx([_order("9", "ce")], [_trade("9", 0.99, 108.5, 7.2, _now_ms(ts))])
    _bot(ex)._maybe_backfill_close_prices()
    assert _get(fid) == (111.0, 1.5)
    assert state.set_close_price(fid, 1.0, 1.0) is False
    assert _get(fid) == (111.0, 1.5)


def test_rows_without_root_skipped():
    fid, _ = _row(root=None)
    ex = FakeEx([], [])
    _bot(ex)._maybe_backfill_close_prices()
    assert ex.calls == 0 and _get(fid) == (0.0, None)


def test_expired_row_alerts_once_after_repeated_tries(monkeypatch):
    old = (datetime.now(UTC) - timedelta(days=9)).isoformat()
    _row(ts=old)
    _row(ts=old, root="222")
    alerts = []
    monkeypatch.setattr("bot.send_alert", lambda s, b: alerts.append(s) or True)
    bot = _bot(FakeEx([], []))
    bot._maybe_backfill_close_prices()
    assert alerts == []                      # never attempted by this process
    bot._backfill_tries = {1: 3, 2: 3}
    bot._backfill_next = 0.0
    bot._maybe_backfill_close_prices()
    bot._backfill_next = 0.0
    bot._maybe_backfill_close_prices()
    assert len(alerts) == 1                  # one summary for both rows, once


def test_alert_send_returning_false_is_retried(monkeypatch):
    _row(ts=(datetime.now(UTC) - timedelta(days=9)).isoformat())
    calls = []

    def send(s, b):
        calls.append(s)
        return len(calls) > 1          # real send_alert returns False, never raises
    monkeypatch.setattr("bot.send_alert", send)
    bot = _bot(FakeEx([], []))
    bot._backfill_tries = {1: 3}
    for _ in range(3):
        bot._backfill_next = 0.0
        bot._maybe_backfill_close_prices()
    assert len(calls) == 2


def test_very_old_history_does_not_alert(monkeypatch):
    _row(ts=(datetime.now(UTC) - timedelta(days=60)).isoformat())
    alerts = []
    monkeypatch.setattr("bot.send_alert", lambda s, b: alerts.append(s) or True)
    ex = FakeEx([], [])
    _bot(ex)._maybe_backfill_close_prices()
    assert alerts == [] and ex.calls == 0


def test_one_row_per_pass_and_abort_on_error_with_cooldown():
    for r in ("1", "2", "3"):
        _row(root=r)
    ex = FakeEx(fail=True)
    bot = _bot(ex)
    bot._maybe_backfill_close_prices()
    assert ex.calls == 1                     # one row, aborted at first error
    assert bot._backfill_tries == {}         # an errored lookup is not a "try"
    assert bot._backfill_cooldown_s == 300.0
    bot._backfill_next = time.monotonic() + 1
    bot._maybe_backfill_close_prices()
    assert ex.calls == 1                     # cooling down
    bot._backfill_next = 0.0
    bot._maybe_backfill_close_prices()
    assert bot._backfill_cooldown_s == 600.0  # doubles


def test_rate_limit_errors_do_not_propagate():
    import ccxt
    _row()
    ex = FakeEx()
    ex.fetch_orders = lambda *a, **k: (_ for _ in ()).throw(ccxt.RateLimitExceeded("418"))
    bot = _bot(ex)
    bot._maybe_backfill_close_prices()
    assert bot._backfill_cooldown_s == 300.0


def test_rotates_so_unmatched_row_does_not_starve_others():
    _row(root="111")                          # never matches
    fid2, ts = _row(root="222", qty=1.0)
    ex = FakeEx([{"id": "9", "clientOrderId": f"{PREFIX}222-ce"}],
                [_trade("9", 1.0, 42.0, 1.0, _now_ms(ts))])
    bot = _bot(ex)
    bot._maybe_backfill_close_prices()
    bot._backfill_next = 0.0
    bot._maybe_backfill_close_prices()
    assert _get(fid2) == (pytest.approx(42.0), pytest.approx(1.0))


def test_qty_mismatch_is_not_matched():
    fid, ts = _row(qty=0.99)
    ex = FakeEx([_order("9", "ce")], [_trade("9", 0.40, 108.5, 3.0, _now_ms(ts))])
    _bot(ex)._maybe_backfill_close_prices()
    assert _get(fid) == (0.0, None)


def test_two_rows_same_root_each_get_their_own_order():
    t0 = datetime.now(UTC) - timedelta(minutes=20)
    ts1, ts2 = t0.isoformat(), (t0 + timedelta(minutes=10)).isoformat()
    f1, _ = _row(ts=ts1, qty=1.0)
    f2, _ = _row(ts=ts2, qty=1.0)
    ex = FakeEx([_order("A", "x"), _order("B", "ce")],
                [_trade("A", 1.0, 10.0, 1.0, _now_ms(ts1)),
                 _trade("B", 1.0, 20.0, 2.0, _now_ms(ts2))])
    bot = _bot(ex)
    for _ in range(2):
        bot._backfill_next = 0.0
        bot._maybe_backfill_close_prices()
    assert _get(f1)[0] == pytest.approx(10.0)
    assert _get(f2)[0] == pytest.approx(20.0)


def test_no_stop_close_with_fresh_root_matches():
    fid, ts = _row(reason="no_stop_close", qty=1.0)
    ex = FakeEx([{"id": "7", "clientOrderId": f"{PREFIX}1791999999999-ns"}],
                [_trade("7", 1.0, 77.0, -2.0, _now_ms(ts))])
    _bot(ex)._maybe_backfill_close_prices()
    assert _get(fid) == (pytest.approx(77.0), pytest.approx(-2.0))


def test_fresh_root_ns_not_matched_for_other_reasons():
    fid, ts = _row(reason="channel_exit", qty=1.0)
    ex = FakeEx([{"id": "7", "clientOrderId": f"{PREFIX}1791999999999-ns"}],
                [_trade("7", 1.0, 77.0, -2.0, _now_ms(ts))])
    _bot(ex)._maybe_backfill_close_prices()
    assert _get(fid) == (0.0, None)


def test_default_c_leg_matches():
    fid, ts = _row(qty=1.0)
    ex = FakeEx([_order("9", "c")], [_trade("9", 1.0, 5.0, 0.5, _now_ms(ts))])
    _bot(ex)._maybe_backfill_close_prices()
    assert _get(fid)[0] == pytest.approx(5.0)


def test_unmatched_warning_rate_limited(caplog):
    _row()
    bot = _bot(FakeEx([], []))
    with caplog.at_level("WARNING"):
        for _ in range(3):
            bot._backfill_next = 0.0
            bot._maybe_backfill_close_prices()
    assert sum("no matching close trade" in r.message for r in caplog.records) == 1


def test_split_close_across_orders_is_priced():
    fid, ts = _row(qty=1.0, reason="no_stop_close")
    t = _now_ms(ts)
    ex = FakeEx([_order("A", "x"), {"id": "B", "clientOrderId": f"{PREFIX}1791999999999-ns"}],
                [_trade("A", 0.4, 100.0, 1.0, t), _trade("B", 0.6, 110.0, 2.0, t + 500)])
    _bot(ex)._maybe_backfill_close_prices()
    price, pnl = _get(fid)
    assert price == pytest.approx(0.4 * 100 + 0.6 * 110)
    assert pnl == pytest.approx(3.0)


def test_two_same_qty_no_stop_rows_do_not_share_an_order():
    f1, ts1 = _row(qty=1.0, reason="no_stop_close")
    f2, ts2 = _row(qty=1.0, reason="no_stop_close")
    ex = FakeEx([{"id": "7", "clientOrderId": f"{PREFIX}1791999999999-ns"}],
                [_trade("7", 1.0, 77.0, -2.0, _now_ms(ts1))])
    bot = _bot(ex)
    for _ in range(3):
        bot._backfill_next = 0.0
        bot._maybe_backfill_close_prices()
    assert _get(f1) == (pytest.approx(77.0), pytest.approx(-2.0))
    assert _get(f2) == (0.0, None)


def test_claims_survive_restart_via_meta():
    f1, ts1 = _row(qty=1.0, reason="no_stop_close")
    ex = FakeEx([{"id": "7", "clientOrderId": f"{PREFIX}1791999999999-ns"}],
                [_trade("7", 1.0, 77.0, -2.0, _now_ms(ts1))])
    _bot(ex)._maybe_backfill_close_prices()
    f2, _ = _row(qty=1.0, reason="no_stop_close")
    _bot(ex)._maybe_backfill_close_prices()      # fresh Bot, same DB
    assert _get(f2) == (0.0, None)


def test_more_than_20_unpriced_rows_all_reachable():
    for i in range(25):
        _row(root=str(1000 + i))
    ex = FakeEx([], [])
    bot = _bot(ex)
    for _ in range(25):
        bot._backfill_next = 0.0
        bot._maybe_backfill_close_prices()
    assert sorted(bot._backfill_tries) == list(range(1, 26))


def test_giveup_sees_new_row_behind_20_untried_old_rows(monkeypatch):
    for i in range(22):
        _row(root=str(500 + i), ts=(datetime.now(UTC) - timedelta(days=10)).isoformat())
    alerts = []
    monkeypatch.setattr("bot.send_alert", lambda s, b: alerts.append(b) or True)
    bot = _bot(FakeEx([], []))
    bot._backfill_tries = {22: 3}
    bot._maybe_backfill_close_prices()
    assert len(alerts) == 1 and "fill #22" in alerts[0] and "fill #1 " not in alerts[0]


def test_bad_ts_row_skipped_without_cooldown(caplog):
    _row(ts="not-a-timestamp")
    ex = FakeEx([], [])
    bot = _bot(ex)
    bot._maybe_backfill_close_prices()
    assert ex.calls == 0 and getattr(bot, "_backfill_cooldown_s", 0.0) == 0.0
    assert bot._backfill_tries == {}
