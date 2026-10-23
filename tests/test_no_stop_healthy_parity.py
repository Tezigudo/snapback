"""Healthy-path parity for the no-stop change (2026-10-13).

The no-stop fix only acts when a stop CANNOT be placed. While everything is
healthy it must make exactly the exchange calls it made before. Each test pins
the literal, ordered call list against the scripted FakeEx. The file runs
unchanged on the base commit (e6244e6) and on the fix, and passes on both.
That is the proof.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from exchange import state
from exchange.binance_client import BinanceClient

from ._fake_exchange import SYMBOL, FakeEx

PREFIX = "snap-v1-"
ROOT = "1791000000000"
E, S, T = (f"{PREFIX}{ROOT}-{x}" for x in "est")
ALGO = ("fapiPrivateGetOpenAlgoOrders", {"symbol": "BTCUSDT"})


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch("exchange.binance_client.time.sleep"), patch("bot.time.sleep"):
        yield


def _client(ex: FakeEx) -> BinanceClient:
    return BinanceClient(ex=ex, env="testnet", coid_prefix=PREFIX)


def _stop(stop: float, qty: float, coid: str) -> tuple:
    return ("create_order", SYMBOL, "STOP_MARKET", "sell", qty, None,
            {"stopPrice": stop, "reduceOnly": True, "workingType": "MARK_PRICE",
             "newClientOrderId": coid})


def _tp(stop: float, qty: float, coid: str) -> tuple:
    return ("create_order", SYMBOL, "TAKE_PROFIT_MARKET", "sell", qty, None,
            {"stopPrice": stop, "reduceOnly": True, "workingType": "MARK_PRICE",
             "newClientOrderId": coid})


class TestEntryCalls:

    @pytest.mark.parametrize("place_tp", [True, False], ids=["v1/sol", "donchian"])
    def test_market_entry(self, place_tp) -> None:
        ex = FakeEx()
        out = _client(ex).market_order_with_bracket(
            SYMBOL, "long", 0.005, 64000.0, 67000.0,
            client_order_id_root=ROOT, place_tp=place_tp)
        expected = [
            ("create_order", SYMBOL, "market", "buy", 0.005, None,
             {"reduceOnly": False, "newClientOrderId": E}),
            _stop(64000.0, 0.005, S),
        ] + ([_tp(67000.0, 0.005, T)] if place_tp else [])
        assert ex.calls == expected
        assert set(out) == {"entry", "sl", "tp"}

    def _limit(self, ex: FakeEx) -> dict:
        return _client(ex).limit_order_with_bracket(
            SYMBOL, "long", 0.005, 65000.0, 1000.0, 2000.0,
            timeout_s=4.0, poll_s=2.0, client_order_id_root=ROOT)

    LIMIT = ("create_order", SYMBOL, "limit", "buy", 0.005, 65000.0,
             {"reduceOnly": False, "timeInForce": "GTC", "newClientOrderId": E})
    POLL = ("fetch_order", "lim-1", SYMBOL, {})
    POS = ("fetch_positions", (SYMBOL,))   # reads added by #45 (post-cancel/post-fill/post-kill)

    def test_limit_full_fill(self) -> None:
        ex = FakeEx()
        ex.limit_polls = [{"status": "closed", "filled": 0.005, "average": 64990.0,
                           "_side": "long"}]
        out = self._limit(ex)
        assert ex.calls == [self.LIMIT, self.POLL,
                            _stop(63990.0, 0.005, S), _tp(66990.0, 0.005, T)]
        assert out["filled_as"] == "limit" and "tp_error" not in out

    def test_limit_partial_fill(self) -> None:
        ex = FakeEx()
        ex.limit_polls = [{"status": "open", "filled": 0.002, "average": 64990.0,
                           "_side": "long"}]
        out = self._limit(ex)
        assert ex.calls == [self.LIMIT, self.POLL, self.POLL,
                            ("cancel_order", "lim-1", SYMBOL),
                            self.POLL, self.POS, self.POS,
                            _stop(63990.0, 0.002, S), _tp(66990.0, 0.002, T)]
        assert out["filled_as"] == "limit_partial"

    def test_limit_market_fallback(self) -> None:
        ex = FakeEx()
        ex.limit_polls = [{"status": "open", "filled": 0.0, "average": None,
                           "_side": "long"}]
        out = self._limit(ex)
        assert ex.calls == [self.LIMIT, self.POLL, self.POLL,
                            ("cancel_order", "lim-1", SYMBOL),
                            self.POLL, self.POS,
                            ("create_order", SYMBOL, "market", "buy", 0.005, None,
                             {"reduceOnly": False, "newClientOrderId": E}),
                            ("fetch_order", "1001", SYMBOL, {}),
                            self.POS,
                            _stop(64000.0, 0.005, S), _tp(67000.0, 0.005, T)]
        assert out["filled_as"] == "market_fallback"


V1_PARAMS: dict = {
    "symbol": SYMBOL,
    "strategy_name": "multifactor-v1",
    "hedge": {"enabled": False, "client_order_id_prefix": PREFIX},
    "execution": {"poll_interval_s": "5", "order_type": "market",
                  "limit_offset_bps": "0", "limit_timeout_s": "4"},
    "timeframes": {"entry": "15m"},
    "sizing": {"leverage": 5, "risk_per_trade_pct": 1.0},
    "deploy": {"kill_switch_equity_fraction": 0.5, "min_capital_warn_usdt": 1.0},
    "strategy": {},
    "reprotect": {"enabled": True, "observe_only": False,
                  "max_replaces_per_position": 3},
}


class TestHealthyTickCalls:

    def _healthy(self) -> tuple:
        from bot import Bot
        ex = FakeEx()
        with patch("bot.BinanceClient.from_env", return_value=_client(ex)):
            bot = Bot(params=V1_PARAMS, dry_run=False)
        ex.pos = ("long", 0.005, 65000.0)
        state.record_fill(side="long", qty=0.005, price=65000.0, reason="entry",
                          client_order_id_root=ROOT)
        state.set_meta("active_bracket", json.dumps({
            "signal_id": ROOT, "side": "long", "entry_price": 65000.0, "qty": 0.005,
            "sl_distance": 1000.0, "tp_distance": 2000.0, "place_tp": True}))
        ex._rest_algo(S, "SELL", 0.005, 64000.0)
        ex._rest_algo(T, "SELL", 0.005, 67000.0)
        return bot, ex

    def test_reprotect_on_an_intact_bracket(self) -> None:
        bot, ex = self._healthy()
        for _ in range(3):
            bot._last_reprotect_ts = 0.0
            bot._maybe_reprotect(150.0)
        one = [("fetch_positions", (SYMBOL,)), ("fetch_open_orders", SYMBOL), ALGO]
        assert ex.calls == one * 3

    def test_tick_replay_on_a_healthy_position(self) -> None:
        """The loop's per-tick hooks that touch the exchange on a v1 position
        (the trend exit needs OHLCV and is unchanged by this fix), replayed
        three times with nothing wrong."""
        bot, ex = self._healthy()
        with patch("bot.send_alert") as alert:
            for _ in range(3):
                bot._detect_bracket_exit(150.0)
                if hasattr(bot, "_maybe_finish_no_stop_close"):
                    bot._maybe_finish_no_stop_close(150.0)
                bot._maybe_time_stop(150.0)
                bot._maybe_breakeven(150.0)
                bot._last_reprotect_ts = 0.0
                bot._maybe_reprotect(150.0)
                bot._maybe_enter(150.0)
        tick = [("fetch_positions", (SYMBOL,)),
                ("fetch_positions", (SYMBOL,)), ("fetch_open_orders", SYMBOL), ALGO,
                ("fetch_positions", (SYMBOL,))]
        assert ex.calls == tick * 3
        alert.assert_not_called()
        assert len(ex.algo) == 2
