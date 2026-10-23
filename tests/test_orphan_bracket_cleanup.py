"""Tests for orphan bracket cleanup (fix/orphan-bracket-cleanup).

Covers:
1. cancel_open_orders COID-prefix filtering — only matching orders cancelled
2. _detect_bracket_exit sweeps surviving bracket on open->flat transition
3. _detect_bracket_exit: cancel is called BEFORE PnL recording
4. _place_live_entry pre-clears stale orders (market + limit paths)
5. boot() flat+live -> orphan sweep; flat+dry_run -> no sweep
6. Cancel failure is non-fatal in both _detect_bracket_exit and _place_live_entry
7. _detect_bracket_exit spots a SAME-POLL replacement (open->open, different
   position) and, crucially, does NOT sweep on that path
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bot_internals import SignalDecision
from exchange.binance_client import BinanceClient, Position


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MINIMAL_PARAMS: dict = {
    "symbol": "BTC/USDT:USDT",
    "hedge": {"enabled": False, "client_order_id_prefix": "snap-v1-"},
    "execution": {
        "poll_interval_s": "5",
        "order_type": "market",
        "limit_offset_bps": "0",
        "limit_timeout_s": "25",
    },
    "timeframes": {"entry": "15m"},
    "sizing": {"leverage": 5, "risk_per_trade_pct": 1.0},
    "deploy": {"kill_switch_equity_fraction": 0.5, "min_capital_warn_usdt": 1.0},
    "strategy": {},
}


def _mock_client(coid_prefix: str = "snap-v1-") -> MagicMock:
    # Don't use spec=BinanceClient — `ex` is set in __init__, not the class body,
    # so spec would hide it.  Plain MagicMock gives us free attribute creation.
    mc = MagicMock()
    mc.ex.parse_timeframe.return_value = 900  # 15m = 900 s
    mc.coid_prefix = coid_prefix
    mc.env = "testnet"
    mc.cancel_open_orders.return_value = 0
    return mc


def _make_bot(dry_run: bool = False, order_type: str = "market"):
    from bot import Bot
    params = {
        **_MINIMAL_PARAMS,
        "execution": {**_MINIMAL_PARAMS["execution"], "order_type": order_type},
    }
    mc = _mock_client()
    with patch("bot.BinanceClient.from_env", return_value=mc):
        bot = Bot(params=params, dry_run=dry_run)
    return bot, mc


def _flat() -> Position:
    return Position(symbol="BTC/USDT:USDT", side="flat", qty=0.0,
                    entry_price=0.0, unrealized_pnl=0.0, margin_used=0.0)


def _open(side: str = "long", entry_price: float = 65000.0,
          qty: float = 0.01) -> Position:
    return Position(symbol="BTC/USDT:USDT", side=side, qty=qty,
                    entry_price=entry_price, unrealized_pnl=10.0,
                    margin_used=100.0)


def _boot_patches():
    """Context manager that patches all side-effects in boot()."""
    return patch.multiple(
        "bot",
        check_symbol=MagicMock(),
        check_leverage=MagicMock(),
        send_alert=MagicMock(),
    )


# ---------------------------------------------------------------------------
# 1. cancel_open_orders COID-prefix filtering (BinanceClient unit tests)
# ---------------------------------------------------------------------------

class TestCancelOpenOrdersCoidPrefix:
    """BinanceClient.cancel_open_orders should filter by prefix when given one."""

    def _client_with_orders(self, orders: list) -> BinanceClient:
        mc_ex = MagicMock()
        mc_ex.fetch_open_orders.return_value = orders
        return BinanceClient(ex=mc_ex, env="testnet", coid_prefix="snap-v1-")

    def _order(self, order_id: str, coid: str | None = None,
                via_info: bool = False) -> dict:
        o: dict = {"id": order_id}
        if coid is not None:
            if via_info:
                o["info"] = {"clientOrderId": coid}
            else:
                o["clientOrderId"] = coid
        return o

    def test_prefix_cancels_only_matching_top_level_coid(self):
        orders = [
            self._order("1", "snap-v1-1234-s"),   # should cancel
            self._order("2", "snap-d3-1234-s"),   # different prefix — skip
            self._order("3", "web_manual"),        # manual — skip
            self._order("4"),                      # no coid — skip
        ]
        client = self._client_with_orders(orders)
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 1
        client.ex.cancel_order.assert_called_once_with("1", "BTC/USDT:USDT")

    def test_prefix_cancels_matching_coid_from_info_dict(self):
        """ccxt sometimes nests clientOrderId inside the `info` dict."""
        orders = [
            self._order("1", "snap-v1-1234-s", via_info=True),  # matches
            self._order("2", "snap-d3-9999-t", via_info=True),  # different prefix
        ]
        client = self._client_with_orders(orders)
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 1
        client.ex.cancel_order.assert_called_once_with("1", "BTC/USDT:USDT")

    def test_no_prefix_cancels_all_orders(self):
        orders = [
            self._order("1", "snap-v1-1234-s"),
            self._order("2", "snap-d3-9999-t"),
            self._order("3"),
        ]
        client = self._client_with_orders(orders)
        n = client.cancel_open_orders("BTC/USDT:USDT")  # coid_prefix=None
        assert n == 3

    def test_fetch_failure_returns_zero_without_cancelling(self):
        mc_ex = MagicMock()
        mc_ex.fetch_open_orders.side_effect = Exception("network error")
        client = BinanceClient(ex=mc_ex, env="testnet", coid_prefix="snap-v1-")
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 0
        mc_ex.cancel_order.assert_not_called()

    def test_returns_count_of_cancelled_orders(self):
        orders = [
            self._order("1", "snap-v1-aaa-s"),
            self._order("2", "snap-v1-bbb-t"),
        ]
        client = self._client_with_orders(orders)
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 2

    def test_info_present_but_none_does_not_crash(self):
        """Regression: an order with info=None (key present, value None) must
        not raise AttributeError under prefix filtering. `o.get("info", {})`
        does NOT fall back to {} when the value is None, so the code guards with
        `(o.get("info") or {})`. The loop must skip the no-coid order and keep
        going."""
        orders = [
            {"id": "1", "info": None},                      # no coid, info None → skip
            {"id": "2", "clientOrderId": "snap-v1-aaa-s"},  # matches → cancel
        ]
        client = self._client_with_orders(orders)
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 1
        client.ex.cancel_order.assert_called_once_with("2", "BTC/USDT:USDT")

    def test_non_string_coid_is_skipped_not_crashed(self):
        """A non-string clientOrderId (e.g. an adapter returning an int) must be
        treated as non-matching, not raise out of the sweep on `.startswith`."""
        orders = [
            {"id": "1", "clientOrderId": 12345},            # non-string → skip
            {"id": "2", "clientOrderId": "snap-v1-aaa-s"},  # matches → cancel
        ]
        client = self._client_with_orders(orders)
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 1
        client.ex.cancel_order.assert_called_once_with("2", "BTC/USDT:USDT")

    def test_partial_cancel_failure_is_non_fatal(self):
        """One cancel_order raising must not abort the sweep: the loop continues
        to the remaining orders, and the returned count reflects only the
        successful cancels."""
        orders = [
            self._order("1", "snap-v1-aaa-s"),
            self._order("2", "snap-v1-bbb-s"),
            self._order("3", "snap-v1-ccc-t"),
        ]
        client = self._client_with_orders(orders)
        client.ex.cancel_order.side_effect = [None, Exception("rejected"), None]
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 2
        assert client.ex.cancel_order.call_count == 3
        assert [c.args[0]
                for c in client.ex.cancel_order.call_args_list] == ["1", "2", "3"]


class TestCancelAlgoOrders:
    """The bracket sweep must ALSO clear conditional orders: Binance parks
    STOP_MARKET/TAKE_PROFIT_MARKET triggers under /fapi/v1/openAlgoOrders,
    invisible to fetch_open_orders — a plain-only sweep orphans the surviving
    bracket sibling (found live on v1's account 2026-07-26)."""

    def _client(self, algo_rows: list, plain_orders: list | None = None) -> BinanceClient:
        mc_ex = MagicMock()
        mc_ex.fetch_open_orders.return_value = plain_orders or []
        mc_ex.market.return_value = {"id": "BTCUSDT"}
        mc_ex.fapiPrivateGetOpenAlgoOrders.return_value = algo_rows
        return BinanceClient(ex=mc_ex, env="testnet", coid_prefix="snap-v1-")

    def test_algo_orders_swept_with_coid_scoping(self):
        rows = [
            {"algoId": "111", "clientAlgoId": "snap-v1-1234-s"},   # ours → cancel
            {"algoId": "222", "clientAlgoId": "snap-d3-1234-s"},   # other leg → skip
            {"algoId": "333", "clientAlgoId": "x-cvBPrNm9abc"},    # ccxt default → skip
            {"algoId": "444"},                                     # no coid → skip
        ]
        client = self._client(rows)
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 1
        client.ex.fapiPrivateDeleteAlgoOrder.assert_called_once_with(
            {"symbol": "BTCUSDT", "algoId": "111"})

    def test_counts_combine_plain_and_algo(self):
        plain = [{"id": "9", "clientOrderId": "snap-v1-aaa-s"}]
        algo = [{"algoId": "111", "clientAlgoId": "snap-v1-aaa-t"}]
        client = self._client(algo, plain_orders=plain)
        assert client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-") == 2

    def test_no_prefix_cancels_all_algo_orders(self):
        rows = [
            {"algoId": "111", "clientAlgoId": "snap-v1-1234-s"},
            {"algoId": "222", "clientAlgoId": "x-cvBPrNm9abc"},
        ]
        client = self._client(rows)
        assert client.cancel_open_orders("BTC/USDT:USDT") == 2

    def test_algo_fetch_failure_does_not_block_plain_sweep(self):
        plain = [{"id": "9", "clientOrderId": "snap-v1-aaa-s"}]
        client = self._client([], plain_orders=plain)
        client.ex.fapiPrivateGetOpenAlgoOrders.side_effect = Exception("boom")
        assert client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-") == 1

    def test_partial_algo_cancel_failure_is_non_fatal(self):
        rows = [
            {"algoId": "111", "clientAlgoId": "snap-v1-a-s"},
            {"algoId": "222", "clientAlgoId": "snap-v1-b-s"},
            {"algoId": "333", "clientAlgoId": "snap-v1-c-t"},
        ]
        client = self._client(rows)
        client.ex.fapiPrivateDeleteAlgoOrder.side_effect = [
            None, Exception("rejected"), None]
        assert client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-") == 2

    def test_report_unknown_returns_none_on_unreadable_book(self):
        client = self._client([])
        client.ex.fapiPrivateGetOpenAlgoOrders.side_effect = Exception("blip")
        assert client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-") == 0
        assert client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-",
                                         report_unknown=True) is None
        client.ex.fapiPrivateGetOpenAlgoOrders.side_effect = None
        client.ex.fapiPrivateGetOpenAlgoOrders.return_value = []
        client.ex.fetch_open_orders.side_effect = Exception("blip")
        assert client.cancel_open_orders("BTC/USDT:USDT", report_unknown=True) is None
        client.ex.fetch_open_orders.side_effect = None
        client.ex.fetch_open_orders.return_value = []
        assert client.cancel_open_orders("BTC/USDT:USDT", report_unknown=True) == 0

    def test_market_id_fallback_when_markets_not_loaded(self):
        rows = [{"algoId": "111", "clientAlgoId": "snap-v1-a-s"}]
        client = self._client(rows)
        client.ex.market.side_effect = Exception("markets not loaded")
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 1
        client.ex.fapiPrivateGetOpenAlgoOrders.assert_called_once_with(
            {"symbol": "BTCUSDT"})

    def test_missing_algo_endpoints_degrade_to_plain_only(self):
        # Old ccxt without the algo endpoints: spec-limited mock has no such attrs.
        mc_ex = MagicMock(spec=["fetch_open_orders", "cancel_order", "market"])
        mc_ex.fetch_open_orders.return_value = [
            {"id": "9", "clientOrderId": "snap-v1-aaa-s"}]
        client = BinanceClient(ex=mc_ex, env="testnet", coid_prefix="snap-v1-")
        assert client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-") == 1


# ---------------------------------------------------------------------------
# 2. _detect_bracket_exit: sweep on open->flat transition
# ---------------------------------------------------------------------------

class TestDetectBracketExit:

    def _setup_transition(self, bot, mc) -> None:
        """Configure mocks for a long->flat bracket-exit scenario."""
        bot._last_position_side = "long"
        mc.fetch_position.return_value = _flat()
        mc.ex.fetch_my_trades.return_value = [{"side": "sell", "price": 66000.0}]

    def _run(self, bot, mc, record_fill_spy=None) -> None:
        fill_patch = patch("bot.state.record_fill",
                           side_effect=record_fill_spy)
        with patch("bot.sqlite3.connect") as mock_db, \
             fill_patch, \
             patch("bot.state.enqueue_bot_event"), \
             patch("bot.send_alert"):
            (mock_db.return_value
             .__enter__.return_value
             .execute.return_value
             .fetchone.side_effect) = [
                ("entry", "long", 0.01, 65000.0, "1234567890123", 1)]
            bot._detect_bracket_exit(equity=10000.0)

    def test_cancel_called_once_with_prefix(self):
        bot, mc = _make_bot()
        self._setup_transition(bot, mc)
        self._run(bot, mc)
        mc.cancel_open_orders.assert_called_once_with(
            "BTC/USDT:USDT", coid_prefix="snap-v1-"
        )

    def test_cancel_called_before_pnl_recording(self):
        """Cancel must precede state.record_fill so the orphan is swept even
        if the PnL lookup later errors out."""
        bot, mc = _make_bot()
        self._setup_transition(bot, mc)
        call_order: list[str] = []
        mc.cancel_open_orders.side_effect = (
            lambda *a, **kw: call_order.append("cancel") or 0
        )
        self._run(bot, mc,
                  record_fill_spy=lambda *a, **kw: call_order.append("record_fill"))
        assert "cancel" in call_order, "cancel_open_orders was not called"
        assert "record_fill" in call_order, "state.record_fill was not called"
        assert (call_order.index("cancel") < call_order.index("record_fill")), (
            f"cancel must precede record_fill; got order: {call_order}"
        )

    def test_no_cancel_when_position_was_already_flat(self):
        """No open->flat transition → no cancel."""
        bot, mc = _make_bot()
        bot._last_position_side = "flat"
        mc.fetch_position.return_value = _flat()
        bot._detect_bracket_exit(equity=10000.0)
        mc.cancel_open_orders.assert_not_called()

    def test_no_cancel_when_position_still_open(self):
        """Transition open->open → no cancel."""
        bot, mc = _make_bot()
        bot._last_position_side = "long"
        mc.fetch_position.return_value = _open()
        bot._detect_bracket_exit(equity=10000.0)
        mc.cancel_open_orders.assert_not_called()

    def test_cancel_failure_does_not_block_pnl_recording(self):
        """A cancel exception must not swallow the PnL recording that follows."""
        bot, mc = _make_bot()
        self._setup_transition(bot, mc)
        mc.cancel_open_orders.side_effect = Exception("exchange timeout")
        with patch("bot.sqlite3.connect") as mock_db, \
             patch("bot.state.record_fill") as mock_fill, \
             patch("bot.state.enqueue_bot_event"), \
             patch("bot.send_alert"):
            (mock_db.return_value
             .__enter__.return_value
             .execute.return_value
             .fetchone.side_effect) = [
                ("entry", "long", 0.01, 65000.0, "1234567890123", 1)]
            bot._detect_bracket_exit(equity=10000.0)
        mock_fill.assert_called_once()


# ---------------------------------------------------------------------------
# 3. _place_live_entry: pre-clear before entry placement
# ---------------------------------------------------------------------------

class TestPlaceLiveEntry:

    def _decision(self) -> SignalDecision:
        return SignalDecision(side="long", price=65000.0,
                              sl_distance=975.0, tp_distance=1950.0, debug={})

    def test_cancel_called_before_market_order(self):
        bot, mc = _make_bot(order_type="market")
        mc.reset_mock()  # clear __init__ call history
        mc.cancel_open_orders.return_value = 0
        mc.market_order_with_bracket.return_value = {"entry": {}, "sl": {}, "tp": {}}

        with patch("bot.trade_events.record_market_entry"), \
             patch("bot.state.set_meta"):
            bot._place_live_entry(self._decision(), qty=0.01,
                                  signal_id="1234567890123", equity=10000.0)

        call_names = [c[0] for c in mc.method_calls]
        assert "cancel_open_orders" in call_names
        assert "market_order_with_bracket" in call_names
        assert (call_names.index("cancel_open_orders")
                < call_names.index("market_order_with_bracket")), (
            f"cancel must precede market_order_with_bracket; got: {call_names}"
        )
        mc.cancel_open_orders.assert_called_once_with(
            "BTC/USDT:USDT", coid_prefix="snap-v1-"
        )

    def test_cancel_called_before_limit_order(self):
        bot, mc = _make_bot(order_type="limit")
        mc.reset_mock()
        mc.cancel_open_orders.return_value = 0
        mc.limit_order_with_bracket.return_value = {
            "entry": {}, "sl": {}, "tp": {},
            "filled_as": "limit", "fill_price": 64990.0, "filled_qty": 0.01,
        }

        with patch("bot.trade_events.record_limit_entry"), \
             patch("bot.state.set_meta"):
            bot._place_live_entry(self._decision(), qty=0.01,
                                  signal_id="1234567890123", equity=10000.0)

        call_names = [c[0] for c in mc.method_calls]
        assert "cancel_open_orders" in call_names
        assert "limit_order_with_bracket" in call_names
        assert (call_names.index("cancel_open_orders")
                < call_names.index("limit_order_with_bracket")), (
            f"cancel must precede limit_order_with_bracket; got: {call_names}"
        )
        mc.cancel_open_orders.assert_called_once_with(
            "BTC/USDT:USDT", coid_prefix="snap-v1-"
        )

    def test_cancel_failure_does_not_block_entry_placement(self):
        """A cancel exception before the entry must not prevent the order."""
        bot, mc = _make_bot(order_type="market")
        mc.cancel_open_orders.side_effect = Exception("timeout")
        mc.market_order_with_bracket.return_value = {"entry": {}, "sl": {}, "tp": {}}

        with patch("bot.trade_events.record_market_entry"), \
             patch("bot.state.set_meta"):
            bot._place_live_entry(self._decision(), qty=0.01,
                                  signal_id="1234567890123", equity=10000.0)

        mc.market_order_with_bracket.assert_called_once()


# ---------------------------------------------------------------------------
# 4. boot(): orphan sweep when flat at startup
# ---------------------------------------------------------------------------

class TestBootOrphanSweep:

    def _state_patches(self):
        return patch.multiple(
            "bot.state",
            init_db=MagicMock(),
            get_float=MagicMock(return_value=0.0),
            set_float=MagicMock(),
            set_meta=MagicMock(),
            enqueue_bot_event=MagicMock(),
            record_event=MagicMock(),
            latest_entry_coid_root=MagicMock(return_value=None),
        )

    def test_flat_live_sweeps_orphaned_orders(self):
        bot, mc = _make_bot(dry_run=False)
        mc.fetch_equity_usdt.return_value = 1000.0
        mc.fetch_position.return_value = _flat()
        mc.cancel_open_orders.return_value = 1  # one orphan found

        with _boot_patches(), self._state_patches(), \
             patch("bot.principal.initialize", MagicMock(return_value=None)), \
             patch("bot.principal.get_principal", MagicMock(return_value=None)):
            bot.boot()

        mc.cancel_open_orders.assert_called_once_with(
            "BTC/USDT:USDT", coid_prefix="snap-v1-"
        )

    def test_flat_dry_run_does_not_sweep(self):
        bot, mc = _make_bot(dry_run=True)
        mc.fetch_equity_usdt.return_value = 1000.0
        mc.fetch_position.return_value = _flat()

        with _boot_patches(), self._state_patches(), \
             patch("bot.principal.initialize", MagicMock(return_value=None)), \
             patch("bot.principal.get_principal", MagicMock(return_value=None)):
            bot.boot()

        mc.cancel_open_orders.assert_not_called()

    def test_open_position_does_not_trigger_flat_sweep(self):
        """With an open position at boot, the boot-flatten branch runs (calling
        close_position), NOT the new flat-path else-branch sweep.

        NOTE: in PRODUCTION close_position itself calls cancel_open_orders
        (binance_client.py) — but here the client is a MagicMock whose
        close_position is a no-op stub that does not call through. So this test
        only asserts the NEW else-branch sweep does not fire on the open-position
        path; it is NOT a claim that production cancels nothing here."""
        bot, mc = _make_bot(dry_run=False)
        mc.fetch_equity_usdt.return_value = 1000.0
        mc.fetch_position.return_value = _open()

        with _boot_patches(), self._state_patches(), \
             patch("bot.principal.initialize", MagicMock(return_value=None)), \
             patch("bot.principal.get_principal", MagicMock(return_value=None)):
            bot.boot()

        # The new flat-path else-branch must not run when a position is open.
        # (mc.close_position is a mock stub, so it doesn't cancel through here.)
        mc.cancel_open_orders.assert_not_called()


# ---------------------------------------------------------------------------
# 7. _detect_bracket_exit: same-poll replacement (open -> open, not the same
#    position).  Live 2026-08-23: the bracket SL filled at 05:14:30.039 and
#    _maybe_enter re-entered at 05:14:31.290 — inside one 5 s poll, so the loop
#    never observed `flat` and the exit was dropped entirely.
# ---------------------------------------------------------------------------

class TestSamePollReplacement:

    OLD_ENTRY = 77_467.90
    NEW_ENTRY = 76_122.80
    EXIT_PX = 76_126.30
    QTY = 0.004
    OLD_ROOT = "1787375710921"
    NEW_ROOT = "1787462070039"

    # The conftest `isolated_state_db` fixture gives every test a real, empty
    # sqlite DB, so these seed genuine `fills` rows rather than mocking the
    # cursor. That matters: the double-write guard runs its own queries, and a
    # blanket sqlite mock would answer all of them with the same row.
    def _seed_entry(self, root, side="long", qty=None, price=None):
        from exchange import state as st
        st.record_fill(side=side, qty=self.QTY if qty is None else qty,
                       price=self.OLD_ENTRY if price is None else price,
                       reason="entry", client_order_id_root=root)

    def _seed_close(self, root, reason="trend_exit"):
        from exchange import state as st
        st.record_fill(side="close", qty=self.QTY, price=self.EXIT_PX,
                       reason=reason, pnl_usd=-5.37, client_order_id_root=root)

    def _tracking(self, bot, side="long", entry=None, qty=None, root=None):
        """Put the bot in the state of already tracking an open position."""
        bot._last_position_side = side
        bot._last_position_entry = self.OLD_ENTRY if entry is None else entry
        bot._last_position_qty = self.QTY if qty is None else qty
        bot._last_entry_root = self.OLD_ROOT if root is None else root

    def _run(self, bot, record_fill_spy=None, event_spy=None):
        with patch("bot.state.record_fill", side_effect=record_fill_spy), \
             patch("bot.state.enqueue_bot_event", side_effect=event_spy), \
             patch("bot.state.latest_entry_coid_root", return_value="9999"), \
             patch("bot.send_alert"):
            bot._detect_bracket_exit(equity=141.87)

    def _replacement(self, mc, side="long"):
        mc.fetch_position.return_value = _open(side, self.NEW_ENTRY, self.QTY)
        mc.ex.fetch_my_trades.return_value = [
            {"side": "sell", "price": self.EXIT_PX},
            {"side": "buy", "price": self.NEW_ENTRY},
        ]

    # --- the safety property -----------------------------------------------

    def test_replacement_does_NOT_sweep_the_new_bracket(self):
        """_place_live_entry has already bracketed the position that is open
        right now; a prefix-scoped sweep here would strip its SL and TP."""
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        self._replacement(mc)
        self._run(bot)
        mc.cancel_open_orders.assert_not_called()

    # --- it records the exit at all ----------------------------------------

    def test_replacement_records_the_dropped_exit(self):
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        self._replacement(mc)
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert len(calls) == 1, "the replacement exit was dropped again"
        assert calls[0]["reason"] == "bracket_exit"
        assert calls[0]["price"] == pytest.approx(self.EXIT_PX)

    def test_replacement_prices_from_the_snapshot_not_the_fills_row(self):
        """The fills table already names the REPLACEMENT entry by the time this
        runs (_maybe_enter is last in the tick), so it cannot price this exit."""
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        self._replacement(mc)
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        expected = (self.EXIT_PX - self.OLD_ENTRY) * self.QTY
        assert calls[0]["pnl_usd"] == pytest.approx(expected), (
            "PnL was computed off the new entry price, not the closed position's"
        )
        assert calls[0]["pnl_usd"] < 0, "the 08-23 trade was a loss"
        assert calls[0]["client_order_id_root"] == self.OLD_ROOT, (
            "the exit was attributed to the replacement entry's root"
        )

    def test_replacement_event_is_tagged_as_such(self):
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        self._replacement(mc)
        events = []
        self._run(bot, event_spy=lambda *a, **kw: events.append(kw))
        assert events[0]["payload"]["detected_by"] == "replacement"

    # --- and it must not DOUBLE-write one the bot already closed ------------

    def test_bot_initiated_close_plus_reentry_is_not_written_twice(self):
        """_maybe_channel_exit / _maybe_time_stop can close AND _maybe_enter can
        re-open inside one tick (both run after _detect_bracket_exit). The next
        tick sees a changed entry_price -- and must not emit a second exit for a
        position whose close is already in the ledger."""
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._seed_close(self.OLD_ROOT)              # the bot's own exit
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        self._replacement(mc)
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert calls == [], "wrote a duplicate exit over the bot's own close"

    def test_untagged_entry_still_blocks_a_double_write(self):
        """No coid root to anchor on -- fall back to the row before the newest."""
        bot, mc = _make_bot()
        self._seed_entry(None)
        self._seed_close(None)
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot, root=None)
        self._replacement(mc)
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert calls == []

    # --- and it stays quiet when nothing changed ---------------------------

    def test_same_position_still_open_does_not_trigger(self):
        """The identical position, tick after tick, must stay silent."""
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._tracking(bot)
        mc.fetch_position.return_value = _open("long", self.OLD_ENTRY, self.QTY)
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert calls == []
        mc.cancel_open_orders.assert_not_called()

    def test_zero_entry_price_does_not_fabricate_an_exit(self):
        """fetch_position defaults a missing entryPrice to 0.0 -- that must not
        read as a replacement."""
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._tracking(bot)
        mc.fetch_position.return_value = _open("long", 0.0, self.QTY)
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert calls == []

    def test_no_snapshot_yet_does_not_trigger(self):
        """Pre-existing state (side only, no snapshot) must not fire."""
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        bot._last_position_side = "long"
        self._replacement(mc)
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert calls == []

    def test_first_observation_populates_the_full_snapshot(self):
        """Boot must leave side AND identity consistent, not just side."""
        bot, mc = _make_bot()
        mc.fetch_position.return_value = _open("long", self.OLD_ENTRY, self.QTY)
        self._run(bot)
        assert bot._last_position_side == "long"
        assert bot._last_position_entry == pytest.approx(self.OLD_ENTRY)
        assert bot._last_position_qty == pytest.approx(self.QTY)
        assert bot._last_entry_root == "9999"

    def test_going_flat_clears_the_snapshot(self):
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._tracking(bot)
        mc.fetch_position.return_value = _flat()
        mc.ex.fetch_my_trades.return_value = [{"side": "sell", "price": self.EXIT_PX}]
        self._run(bot)
        assert bot._last_position_entry is None
        assert bot._last_position_qty is None
        assert bot._last_entry_root is None

    def test_side_flip_replacement_is_still_recorded(self):
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._seed_entry(self.NEW_ROOT, side="short", price=self.NEW_ENTRY)
        self._tracking(bot, side="long")
        self._replacement(mc, side="short")
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert len(calls) == 1
        mc.cancel_open_orders.assert_not_called()

    # --- a detected exit must survive a failed lookup ----------------------

    def test_failed_trade_lookup_keeps_the_old_snapshot_for_a_retry(self):
        """fills already names the REPLACEMENT entry, so the snapshot is the only
        record the closed position existed. Advancing past a failed lookup would
        lose the exit permanently -- the very bug this detector exists to end."""
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        mc.fetch_position.return_value = _open("long", self.NEW_ENTRY, self.QTY)
        mc.ex.fetch_my_trades.side_effect = RuntimeError("ccxt: request timed out")
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert calls == []
        assert bot._last_position_entry == pytest.approx(self.OLD_ENTRY), (
            "snapshot advanced past a failure; the exit is now unrecoverable"
        )
        assert bot._last_entry_root == self.OLD_ROOT
        assert bot._exit_retry_n == 1

    def test_the_retry_then_succeeds_on_the_next_tick(self):
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        mc.fetch_position.return_value = _open("long", self.NEW_ENTRY, self.QTY)
        mc.ex.fetch_my_trades.side_effect = RuntimeError("ccxt: request timed out")
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        # ...the venue comes back
        mc.ex.fetch_my_trades.side_effect = None
        mc.ex.fetch_my_trades.return_value = [{"side": "sell", "price": self.EXIT_PX}]
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert len(calls) == 1, "the retry did not recover the exit"
        assert calls[0]["pnl_usd"] == pytest.approx(
            (self.EXIT_PX - self.OLD_ENTRY) * self.QTY)
        assert bot._exit_retry_n == 0, "retry counter not reset after success"

    def test_no_matching_trade_also_retries(self):
        """An empty/unmatched trade list is the same failure as an exception."""
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        mc.fetch_position.return_value = _open("long", self.NEW_ENTRY, self.QTY)
        mc.ex.fetch_my_trades.return_value = [{"side": "buy", "price": self.NEW_ENTRY}]
        self._run(bot)
        assert bot._last_position_entry == pytest.approx(self.OLD_ENTRY)
        assert bot._exit_retry_n == 1

    def test_retry_is_bounded_and_alerts_instead_of_looping_forever(self):
        """fetch_my_trades only returns 10 trades, so an unbounded retry would
        end up matching a window that no longer holds the close."""
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        mc.fetch_position.return_value = _open("long", self.NEW_ENTRY, self.QTY)
        mc.ex.fetch_my_trades.side_effect = RuntimeError("still down")
        bot._exit_retry_n = bot.EXIT_RETRY_LIMIT      # one short of giving up
        alerts = []
        with patch("bot.state.record_fill"), \
             patch("bot.state.enqueue_bot_event"), \
             patch("bot.state.latest_entry_coid_root", return_value="9999"), \
             patch("bot.send_alert", side_effect=lambda *a, **kw: alerts.append(a)):
            bot._detect_bracket_exit(equity=141.87)
        assert len(alerts) == 1, "gave up silently"
        assert "NOT recorded" in alerts[0][0]
        assert bot._exit_retry_n == 0
        # snapshot released, so it stops re-detecting the same replacement
        assert bot._last_position_entry == pytest.approx(self.NEW_ENTRY)

    def test_a_deliberate_skip_is_not_retried(self):
        """An already-recorded close is a correct skip, not a failure -- it must
        not hold the snapshot open."""
        bot, mc = _make_bot()
        self._seed_entry(self.OLD_ROOT)
        self._seed_close(self.OLD_ROOT)
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        self._replacement(mc)
        self._run(bot)
        assert bot._exit_retry_n == 0
        assert bot._last_position_entry == pytest.approx(self.NEW_ENTRY)

    # --- the classic path must be untouched --------------------------------

    def test_flat_edge_still_sweeps_and_still_reads_fills(self):
        bot, mc = _make_bot()
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        mc.fetch_position.return_value = _flat()
        mc.ex.fetch_my_trades.return_value = [{"side": "sell", "price": self.EXIT_PX}]
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        mc.cancel_open_orders.assert_called_once_with(
            "BTC/USDT:USDT", coid_prefix="snap-v1-")
        # priced from the fills row (NEW_ENTRY here), not the snapshot
        assert calls[0]["pnl_usd"] == pytest.approx(
            (self.EXIT_PX - self.NEW_ENTRY) * self.QTY)

    # --- ...and it must retry too (live 2026-09-04, donchian) --------------

    def test_flat_edge_failed_lookup_keeps_the_snapshot_for_a_retry(self):
        """The replacement path has retried since #26; the flat edge never did.

        donchian's algo SL filled at 12:31:49 and fetch_my_trades still did not
        list the closing trade at 12:32:04. One lookup, one warning, a bare
        return -- and because the snapshot had already advanced to flat there
        was no edge left for the next tick to re-detect, so the -$4.78 close was
        gone from the ledger for good.
        """
        bot, mc = _make_bot()
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        mc.fetch_position.return_value = _flat()
        # The close is real on the venue but not yet visible on the account.
        mc.ex.fetch_my_trades.return_value = [{"side": "buy", "price": self.NEW_ENTRY}]
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert calls == []
        assert bot._exit_retry_n == 1
        assert bot._last_position_side == "long", (
            "snapshot left flat after a failed lookup — the next tick sees no "
            "edge to re-detect and the exit is unrecoverable"
        )

    def test_flat_edge_retry_records_the_exit_on_the_next_tick(self):
        bot, mc = _make_bot()
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        mc.fetch_position.return_value = _flat()
        mc.ex.fetch_my_trades.return_value = [{"side": "buy", "price": self.NEW_ENTRY}]
        calls = []
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert calls == []
        # ...the algo fill surfaces a tick later, exactly as it did on 09-04.
        mc.ex.fetch_my_trades.return_value = [
            {"side": "buy", "price": self.NEW_ENTRY},
            {"side": "sell", "price": self.EXIT_PX},
        ]
        self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert len(calls) == 1, "the retry did not recover the exit"
        assert calls[0]["pnl_usd"] == pytest.approx(
            (self.EXIT_PX - self.NEW_ENTRY) * self.QTY)
        assert bot._exit_retry_n == 0, "retry counter not reset after success"

    def test_flat_edge_sqlite_failure_still_arms_a_retry(self):
        """Sourcery caught this on #28. The `fills` read sits between "the
        position ended" and "the exit is written", so a locked or unavailable DB
        raised straight past the arming point into the outer handler with
        `retain` unset -- the same permanent loss, a different trigger. Arming
        now happens off the exchange snapshot, before anything that can raise.
        """
        import sqlite3

        bot, mc = _make_bot()
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        mc.fetch_position.return_value = _flat()
        # The trade IS visible -- the DB is what fails, so nothing else excuses
        # the loss.
        mc.ex.fetch_my_trades.return_value = [{"side": "sell", "price": self.EXIT_PX}]
        calls = []
        with patch("bot.sqlite3.connect",
                   side_effect=sqlite3.OperationalError("database is locked")):
            self._run(bot, record_fill_spy=lambda *a, **kw: calls.append(kw))
        assert calls == []
        assert bot._exit_retry_n == 1
        assert bot._last_position_side == "long", (
            "a raising fills read left the snapshot flat — the exit is gone"
        )

    def test_flat_edge_retry_gives_up_and_leaves_the_snapshot_flat(self):
        """Arming a retry also means it has to let go. On this path the release
        is implicit -- _hold_unrecorded_exit returns WITHOUT restoring, leaving
        the flat snapshot the top of the method already wrote -- so pin it: if
        it ever re-armed instead, the detector would re-fire every tick forever.
        """
        bot, mc = _make_bot()
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._tracking(bot)
        mc.fetch_position.return_value = _flat()
        mc.ex.fetch_my_trades.return_value = [{"side": "buy", "price": self.NEW_ENTRY}]
        bot._exit_retry_n = bot.EXIT_RETRY_LIMIT      # one short of giving up
        alerts = []
        with patch("bot.state.record_fill"), \
             patch("bot.state.enqueue_bot_event"), \
             patch("bot.state.latest_entry_coid_root", return_value="9999"), \
             patch("bot.send_alert", side_effect=lambda *a, **kw: alerts.append(a)):
            bot._detect_bracket_exit(equity=141.87)
        assert len(alerts) == 1, "gave up silently"
        assert "NOT recorded" in alerts[0][0]
        assert bot._exit_retry_n == 0
        assert bot._last_position_side == "flat", (
            "snapshot re-armed after giving up — the detector will re-fire "
            "on every tick from here"
        )

    def test_flat_edge_deliberate_skip_is_still_not_retried(self):
        """Pins the arming point BELOW the `reason='entry'` guard: a position
        whose close is already in the ledger is a correct skip, not a failed
        lookup, and must not hold the snapshot open for 60 ticks."""
        bot, mc = _make_bot()
        self._seed_entry(self.NEW_ROOT, price=self.NEW_ENTRY)
        self._seed_close(self.NEW_ROOT)
        self._tracking(bot)
        mc.fetch_position.return_value = _flat()
        self._run(bot)
        assert bot._exit_retry_n == 0
        assert bot._last_position_side == "flat"
