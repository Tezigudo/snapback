"""Exit price must be the REAL average fill, not 0.0.

THE BUG (measured live 2026-08-23, all three legs, every close since launch):

    v1        2026-08-23  close  price=0.0  pnl=None  trend_exit
    v1        2026-08-19  close  price=0.0  pnl=None  trend_exit
    donchian  2026-08-23  close  price=0.0  pnl=None  channel_exit
    sol       2026-08-08  close  price=0.0  pnl=None  time_stop

`close_position()` returns Binance's CREATE response for a reduce-only MARKET
order, and that response carries `avgPrice: "0.00"` — the fill is not attributed
to the order yet. So `order_avg_price()` returns None at every bot-initiated
close, and:

  * the time-stop and trend/channel-exit paths recorded `price=float(None or 0.0)`
    -> 0.0, and never computed pnl_usd at all;
  * the boot-flatten path's `order_avg_price(o) or pos.entry_price` fallback
    silently collapsed to the ENTRY price, so its PnL was always exactly 0 —
    the very bug its comment claims to have fixed.

The bracket-exit path was unaffected because it runs on a LATER tick and reads
the trade tape, by which time the fill exists.

THE FIX: re-fetch the same order by clientOrderId. Verified against the two real
closes from 2026-08-23 — v1 `snap-v1-1787462070917-ce` returns avgPrice
76003.10, donchian `snap-d3-1787256026107-ce` returns 76010.00 — whereas both
create-responses had carried "0.00".

These tests reproduce that exact two-stage shape: create says 0.00, refetch says
the truth.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.test_channel_exit import _make_bot


def _create_response(coid: str = "snap-v1-1787462070917-ce") -> dict:
    """Exactly what Binance returns when the close order is CREATED."""
    return {
        "id": "1112343799276",
        "clientOrderId": coid,
        "average": None,
        "price": None,
        "filled": 0.0,
        "info": {"avgPrice": "0.00", "executedQty": "0", "status": "NEW"},
    }


def _fetched_response(avg: float, qty: float) -> dict:
    """What fetch_order returns once the fill is attributed."""
    return {
        "id": "1112343799276",
        "average": avg,
        "price": avg,
        "filled": qty,
        "status": "closed",
        "info": {"avgPrice": f"{avg:.5f}", "executedQty": f"{qty}", "status": "FILLED"},
    }


class TestResolveFillPrice:
    def test_create_response_alone_yields_nothing(self):
        """Guard the premise: this is why every exit recorded 0.0."""
        from bot_internals import order_avg_price
        assert order_avg_price(_create_response()) is None

    def test_refetch_recovers_the_real_average(self):
        bot, mc = _make_bot(strategy_name="multifactor-v1")
        mc.ex.fetch_order.return_value = _fetched_response(76003.1, 0.004)
        assert bot._resolve_fill_price(_create_response()) == pytest.approx(76003.1)

    def test_uses_create_price_when_already_populated(self):
        """No refetch when the create response already carries a real average."""
        bot, mc = _make_bot(strategy_name="multifactor-v1")
        order = {**_create_response(), "average": 71234.5}
        assert bot._resolve_fill_price(order) == pytest.approx(71234.5)
        mc.ex.fetch_order.assert_not_called()

    def test_refetches_by_client_order_id(self):
        """Must key on THIS order, not scan the tape and side-match."""
        bot, mc = _make_bot(strategy_name="multifactor-v1")
        mc.ex.fetch_order.return_value = _fetched_response(76010.0, 0.003)
        bot._resolve_fill_price(_create_response(coid="snap-d3-1787256026107-ce"))
        _, kwargs = mc.ex.fetch_order.call_args[0], mc.ex.fetch_order.call_args[1]
        assert kwargs["params"]["origClientOrderId"] == "snap-d3-1787256026107-ce"

    def test_retries_while_fill_is_not_yet_attributed(self):
        """The tape can lag the create by a beat; one bounded retry loop."""
        bot, mc = _make_bot(strategy_name="multifactor-v1")
        mc.ex.fetch_order.side_effect = [
            _create_response(),                  # still 0.00
            _fetched_response(76003.1, 0.004),   # now populated
        ]
        with patch("bot.time.sleep"):
            assert bot._resolve_fill_price(_create_response()) == pytest.approx(76003.1)
        assert mc.ex.fetch_order.call_count == 2

    def test_returns_none_rather_than_raising_when_fetch_fails(self):
        """Bookkeeping must never break a close the exchange already executed."""
        bot, mc = _make_bot(strategy_name="multifactor-v1")
        mc.ex.fetch_order.side_effect = Exception("timeout")
        with patch("bot.time.sleep"):
            assert bot._resolve_fill_price(_create_response()) is None

    def test_does_not_retry_a_failed_call(self):
        """ccxt's timeout is 15s; retrying a hung call would stall the tick
        loop ~46s for a bookkeeping value. Retries are for the
        'fetched but not filled yet' case only."""
        bot, mc = _make_bot(strategy_name="multifactor-v1")
        mc.ex.fetch_order.side_effect = Exception("timeout")
        with patch("bot.time.sleep"):
            bot._resolve_fill_price(_create_response())
        assert mc.ex.fetch_order.call_count == 1

    def test_returns_none_when_order_is_none(self):
        """close_position returns None when already flat."""
        bot, _ = _make_bot(strategy_name="multifactor-v1")
        assert bot._resolve_fill_price(None) is None


class TestExitPnl:
    """PnL is GROSS, matching the bracket-exit path's existing convention."""

    def test_long_profit(self):
        from bot import _exit_pnl_usd
        # donchian's real trade: 0.003 BTC, 72645.50 -> 76010.00
        assert _exit_pnl_usd("long", 72645.50, 76010.0, 0.003) == pytest.approx(10.0935)

    def test_short_profit(self):
        from bot import _exit_pnl_usd
        assert _exit_pnl_usd("short", 100.0, 90.0, 2.0) == pytest.approx(20.0)

    def test_long_loss_is_negative(self):
        from bot import _exit_pnl_usd
        assert _exit_pnl_usd("long", 100.0, 90.0, 2.0) == pytest.approx(-20.0)

    def test_unknown_price_yields_none(self):
        from bot import _exit_pnl_usd
        assert _exit_pnl_usd("long", 100.0, None, 2.0) is None
        assert _exit_pnl_usd("long", 0.0, 90.0, 2.0) is None


class TestChannelExitRecordsRealPrice:
    """End-to-end wiring: the helpers are useless if plumbed in wrong.

    Drives the real _maybe_channel_exit path with the exact live shape --
    close_position returns avgPrice "0.00", the refetch returns the truth --
    and asserts what actually lands in the fills row.
    """

    def _drive(self, exit_avg: float):
        from exchange import state
        from tests.test_channel_exit import _big_df, _open

        bot, mc = _make_bot(strategy_name="donchian-v3", entry_tf="4h")
        # Seed the entry the exit will be measured against, in the isolated DB
        # that tests/conftest.py hands us.
        state.record_fill(side="long", qty=0.02, price=65000.0, reason="entry",
                          equity_after=10000.0, client_order_id_root="sig-1")
        mc.fetch_position.return_value = _open("long")
        mc.fetch_ohlcv.return_value = _big_df(260)
        mc.close_position.return_value = _create_response("snap-d3-sig-1-ce")
        mc.ex.fetch_order.return_value = _fetched_response(exit_avg, 0.02)

        captured = {}

        def _spy(**kw):
            if kw.get("side") == "close":
                captured.update(kw)

        with patch("bot.trend_exit_signal", return_value=(True, {"reason": "channel"})), \
             patch("bot.state.record_fill", side_effect=_spy), \
             patch("bot.state.enqueue_bot_event"), \
             patch("bot.time.sleep"):
            bot._maybe_channel_exit(equity=10250.0)
        return captured

    def test_records_the_refetched_price_not_zero(self):
        got = self._drive(76010.0)
        assert got, "no close fill was recorded"
        assert got["price"] == pytest.approx(76010.0), (
            f"expected the refetched average, got {got.get('price')!r} "
            "-- this is the 0.0 bug"
        )

    def test_records_gross_pnl_against_the_entry_fill(self):
        got = self._drive(76010.0)
        # (76010 - 65000) * 0.02 long
        assert got["pnl_usd"] == pytest.approx(220.20)

    def test_losing_exit_records_negative_pnl(self):
        got = self._drive(60000.0)
        assert got["price"] == pytest.approx(60000.0)
        assert got["pnl_usd"] == pytest.approx(-100.0)


class TestStaleEntryIsNotAttributed:
    """A closed trade's entry must never be used to price a later exit."""

    def test_returns_none_when_newest_fill_is_a_close(self):
        """Mirrors _detect_bracket_exit's guard. Filtering on reason='entry'
        would skip backwards over this close and return the stale entry."""
        from exchange import state
        bot, _ = _make_bot(strategy_name="donchian-v3", entry_tf="4h")
        state.record_fill(side="long", qty=0.02, price=65000.0, reason="entry",
                          equity_after=10000.0, client_order_id_root="old")
        state.record_fill(side="close", qty=0.02, price=70000.0, reason="channel_exit",
                          equity_after=10100.0, client_order_id_root="old")
        assert bot._open_entry_fill() is None

    def test_returns_the_entry_when_it_is_newest(self):
        from exchange import state
        bot, _ = _make_bot(strategy_name="donchian-v3", entry_tf="4h")
        state.record_fill(side="long", qty=0.02, price=65000.0, reason="entry",
                          equity_after=10000.0, client_order_id_root="cur")
        assert bot._open_entry_fill() == ("long", 0.02, 65000.0)

    def test_returns_none_on_empty_db(self):
        bot, _ = _make_bot(strategy_name="donchian-v3", entry_tf="4h")
        assert bot._open_entry_fill() is None
