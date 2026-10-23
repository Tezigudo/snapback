"""F3: limit-entry path must only ever REDUCE exposure under races.

limit_order_with_bracket polls a limit order, cancels at timeout, then either
brackets the fill or falls back to a market order. The poll value is up to
poll_s stale, and cancel/re-fetch can fail. Old behaviour read that as "no fill"
and market-bought the full qty on top of a late limit fill (double size, brackets
for one leg only), and could place a 0-qty or step-floored (under-sized) bracket.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from exchange.binance_client import BinanceClient

SYM = "BTC/USDT:USDT"
QTY = 0.05


class FakeEx:
    def __init__(self, *, polls, final_order=None, cancel_raises=False,
                 refetch_raises=False, position=0.0, position_raises=False,
                 position_after_market=None):
        self.polls = list(polls)            # fetch_order answers during polling
        self.final_order = final_order      # answer to post-cancel fetch_order
        self.cancel_raises = cancel_raises
        self.refetch_raises = refetch_raises
        self.position = position
        self.position_raises = position_raises
        self.position_after_market = position_after_market
        self.created: list[tuple] = []
        self._polled = 0

    def market(self, symbol):
        return {"precision": {"amount": 0.001}}

    def create_order(self, symbol, typ, side, amount, price, params=None):
        self.created.append((typ, side, amount))
        if typ == "market" and self.position_after_market is not None:
            self.position = self.position_after_market
        return {"id": f"{typ}-{len(self.created)}", "average": 65000.0}

    def fetch_order(self, oid, symbol):
        if oid.startswith("market"):
            return {"id": oid, "average": 65000.0, "status": "closed", "filled": QTY}
        if self._polled < len(self.polls):
            self._polled += 1
            return self.polls[self._polled - 1]
        if self.refetch_raises:
            raise RuntimeError("fetch_order down")
        return self.final_order

    def cancel_order(self, oid, symbol):
        if self.cancel_raises:
            raise RuntimeError("order already filled")
        return {}

    def fetch_positions(self, symbols):
        if self.position_raises:
            raise RuntimeError("positions down")
        if not self.position:
            return []
        return [{"symbol": SYM, "contracts": self.position, "side": "long",
                 "entryPrice": 65000.0, "unrealizedPnl": 0.0, "initialMargin": 1.0}]

    # helpers
    def types(self):
        return [c[0] for c in self.created]

    def stop_qtys(self):
        return [c[2] for c in self.created if c[0] == "STOP_MARKET"]


OPEN0 = {"status": "open", "filled": 0.0}


def _run(ex, qty=QTY):
    c = BinanceClient(ex, "testnet")
    with patch("exchange.binance_client.time.sleep"):
        return c.limit_order_with_bracket(
            SYM, "long", qty, 65000.0, sl_distance=500.0, tp_distance=1000.0,
            timeout_s=2.0, poll_s=2.0, client_order_id_root="1716120000000")


def test_cancel_ok_but_late_partial_fill_is_not_market_doubled():
    # Poll saw 0 fill; cancel SUCCEEDED but 0.03 had filled in between.
    ex = FakeEx(polls=[OPEN0], position=0.03,
                final_order={"status": "canceled", "filled": 0.03, "average": 65000.0})
    out = _run(ex)
    assert "market" not in ex.types()
    assert ex.stop_qtys() == [pytest.approx(0.03)]
    assert out["filled_as"] == "limit_partial"


def test_cancel_raises_and_refetch_fails_but_position_shows_the_fill():
    ex = FakeEx(polls=[OPEN0], cancel_raises=True, refetch_raises=True, position=QTY)
    out = _run(ex)
    assert "market" not in ex.types()
    assert ex.stop_qtys() == [pytest.approx(QTY)]
    assert out["filled_as"] == "limit"


def test_cancel_raises_refetch_fails_position_unknown_places_nothing_more():
    ex = FakeEx(polls=[OPEN0], cancel_raises=True, refetch_raises=True,
                position_raises=True)
    with pytest.raises(RuntimeError):
        _run(ex)
    assert ex.types() == ["limit"]          # only the original limit order


def test_cancel_raises_and_order_still_open_places_no_market():
    ex = FakeEx(polls=[OPEN0], cancel_raises=True,
                final_order={"status": "open", "filled": 0.0}, position=0.0)
    with pytest.raises(RuntimeError):
        _run(ex)
    assert "market" not in ex.types()


def test_partial_fill_brackets_the_partial_and_skips_market():
    ex = FakeEx(polls=[{"status": "open", "filled": 0.02}], position=0.02,
                final_order={"status": "canceled", "filled": 0.02, "average": 65000.0})
    out = _run(ex)
    assert "market" not in ex.types()
    assert ex.stop_qtys() == [pytest.approx(0.02)]
    assert out["filled_qty"] == pytest.approx(0.02)


def test_exact_step_partial_is_not_floored_one_step_short():
    # floor(0.043/0.001) == 42 in float -> old code left 0.001 BTC with no stop.
    ex = FakeEx(polls=[{"status": "open", "filled": 0.043}], position=0.043,
                final_order={"status": "canceled", "filled": 0.043, "average": 65000.0})
    _run(ex)
    assert ex.stop_qtys() == [pytest.approx(0.043)]


def test_partial_that_rounds_to_zero_places_no_zero_qty_bracket_or_market():
    ex = FakeEx(polls=[{"status": "open", "filled": 0.0004}], position=0.0004,
                final_order={"status": "canceled", "filled": 0.0004, "average": 65000.0})
    out = _run(ex)
    assert ex.types() == ["limit"]
    assert out["sl"] is None and out["tp"] is None


def test_genuinely_unfilled_falls_back_to_market_and_brackets_position_qty():
    ex = FakeEx(polls=[OPEN0], position=0.0,
                final_order={"status": "canceled", "filled": 0.0},
                position_after_market=QTY)
    out = _run(ex)
    assert ex.types()[:2] == ["limit", "market"]
    assert ex.created[1][2] == pytest.approx(QTY)
    assert ex.stop_qtys() == [pytest.approx(QTY)]
    assert out["filled_as"] == "market_fallback"


def _stuck(sweep_ok):
    ex = FakeEx(polls=[OPEN0], cancel_raises=True, refetch_raises=True,
                position_raises=True)
    c = BinanceClient(ex, "testnet")
    swept = []

    def _sweep(sym, coid_prefix=None):
        swept.append(coid_prefix)
        if not sweep_ok:
            raise RuntimeError("down")
        return 1
    c.cancel_open_orders = _sweep
    def _read(symbol=None):
        if not sweep_ok:
            raise RuntimeError("down")
        return []                                      # verification read: gone
    ex.fetch_open_orders = _read
    with patch("exchange.binance_client.time.sleep"), \
         patch("alerts.send_alert") as alert:
        with pytest.raises(RuntimeError) as ei:
            c.limit_order_with_bracket(
                SYM, "long", QTY, 65000.0, sl_distance=500.0, tp_distance=1000.0,
                timeout_s=2.0, poll_s=2.0, client_order_id_root="1716120000000")
    assert swept == [c.coid_prefix]          # own prefix only
    assert alert.call_count == 1
    assert "market" not in ex.types()
    return str(ei.value)


def test_stuck_limit_is_swept_by_own_prefix_before_raising():
    assert "was cancelled" in _stuck(sweep_ok=True)


def test_stuck_limit_error_says_order_may_still_rest():
    assert "MAY STILL REST" in _stuck(sweep_ok=False)


def test_stuck_limit_with_real_sweep_over_unreachable_exchange_says_may_rest():
    # REAL cancel_open_orders swallows read failures and returns 0: it must not
    # be taken as "cancelled".
    ex = FakeEx(polls=[OPEN0], cancel_raises=True, refetch_raises=True,
                position_raises=True)

    def _down(symbol=None):
        raise RuntimeError("exchange unreachable")
    ex.fetch_open_orders = _down
    c = BinanceClient(ex, "testnet")
    c.fetch_algo_orders = lambda sym: ([], False)
    with patch("exchange.binance_client.time.sleep"), patch("alerts.send_alert"):
        with pytest.raises(RuntimeError) as ei:
            c.limit_order_with_bracket(
                SYM, "long", QTY, 65000.0, sl_distance=500.0, tp_distance=1000.0,
                timeout_s=2.0, poll_s=2.0, client_order_id_root="1716120000000")
    assert "MAY STILL REST" in str(ei.value)


def test_partial_with_failed_cancel_brackets_alerts_and_returns_for_recording():
    # Poll saw a partial; every cancel fails and the book can't be read: the
    # remainder may fill later. Bracket the actual qty, alert, and RETURN (not
    # raise) so the bot still records the entry row for the live position.
    ex = FakeEx(polls=[{"status": "open", "filled": 0.02}], cancel_raises=True,
                refetch_raises=True, position=0.02)

    def _down(symbol=None):
        raise RuntimeError("down")
    ex.fetch_open_orders = _down
    c = BinanceClient(ex, "testnet")
    with patch("exchange.binance_client.time.sleep"), \
         patch("alerts.send_alert") as alert:
        out = c.limit_order_with_bracket(
            SYM, "long", QTY, 65000.0, sl_distance=500.0, tp_distance=1000.0,
            timeout_s=2.0, poll_s=2.0, client_order_id_root="1716120000000")
    assert ex.stop_qtys() == [pytest.approx(0.02)]
    assert "market" not in ex.types()
    assert alert.call_count == 1
    assert "MAY STILL REST" in alert.call_args[0][1]
    assert out["filled_as"] == "limit_partial"
    assert out["filled_qty"] == pytest.approx(0.02)
    assert out["remainder_may_rest"] is True


def test_partial_fill_growing_during_kill_is_bracketed_in_full():
    # 0.02 held at the first read; the remainder fills to 0.03 before the kill
    # lands. The bracket must cover 0.03, not the stale 0.02.
    ex = FakeEx(polls=[{"status": "open", "filled": 0.02}], cancel_raises=True,
                refetch_raises=True, position=0.02)

    def _gone(symbol=None):
        ex.position = 0.03
        return []
    ex.fetch_open_orders = _gone
    c = BinanceClient(ex, "testnet")
    with patch("exchange.binance_client.time.sleep"), patch("alerts.send_alert"):
        out = c.limit_order_with_bracket(
            SYM, "long", QTY, 65000.0, sl_distance=500.0, tp_distance=1000.0,
            timeout_s=2.0, poll_s=2.0, client_order_id_root="1716120000000")
    assert ex.stop_qtys() == [pytest.approx(0.03)]
    assert out["filled_qty"] == pytest.approx(0.03)
    assert out["remainder_may_rest"] is False


def test_partial_with_failed_cancel_but_verified_gone_returns_normally():
    ex = FakeEx(polls=[{"status": "open", "filled": 0.02}], cancel_raises=True,
                refetch_raises=True, position=0.02)
    ex.fetch_open_orders = lambda symbol=None: []
    c = BinanceClient(ex, "testnet")
    with patch("exchange.binance_client.time.sleep"):
        out = c.limit_order_with_bracket(
            SYM, "long", QTY, 65000.0, sl_distance=500.0, tp_distance=1000.0,
            timeout_s=2.0, poll_s=2.0, client_order_id_root="1716120000000")
    assert out["filled_as"] == "limit_partial"
