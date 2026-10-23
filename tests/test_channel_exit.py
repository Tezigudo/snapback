"""Tests for the donchian-v3 live Donchian channel-exit (feat/donchian-channel-exit).

Covers:
1. PARITY — channel_exit_signal reproduces DonchianBreakoutBTCv3.next()'s exit
   branch bar-for-bar on the same synthetic series, using signals_donchian's own
   attach_donchian column builder as the reference (long AND short).
2. Edge cases — exactly-at-channel (strict inequality → no exit), NaN indicator,
   insufficient bars (warmup), flat/unknown side.
3. binance_client place_tp=False omits the TP leg (entry + SL only) while the
   default place_tp=True still places TP — so v1/multifactor is untouched.
4. strategy_uses_channel_exit predicate + bot._place_live_entry passes the right
   place_tp per strategy (v1 keeps TP; donchian omits it).
5. bot._maybe_channel_exit wiring: strategy-gated, drops the forming bar, and on
   a trigger closes reduce-only (close_leg='ce'), records reason='channel_exit',
   enqueues the exit event, and alerts. Dry-run only logs.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot_internals import SignalDecision, strategy_uses_channel_exit
from exchange.binance_client import BinanceClient, Position
from strategy.live_donchian_v3 import channel_exit_signal
from strategy.signals_donchian import attach_donchian

# A realistic long-entry / short-exit geometry (synthetic, not a leg's config).
PARAMS = {"strategy": {"donchian_period_entry": 70, "donchian_period_exit": 12,
                       "atr_period": 18}}
# Small periods for isolated edge-case math (period-independent behaviour).
SMALL = {"strategy": {"donchian_period_entry": 3, "donchian_period_exit": 3,
                      "atr_period": 3}}


def _ohlc(close: np.ndarray) -> pd.DataFrame:
    """Capitalised, tz-naive 4h OHLCV frame from a close path."""
    idx = pd.date_range("2024-01-01", periods=len(close), freq="4h",
                        tz="UTC").tz_localize(None)
    return pd.DataFrame(
        {"Open": close, "High": close + 30.0, "Low": close - 30.0,
         "Close": close, "Volume": 100.0},
        index=idx,
    )


def _df_from_closes(closes: list[float]) -> pd.DataFrame:
    return _ohlc(np.asarray(closes, dtype=float))


# ---------------------------------------------------------------------------
# 1. PARITY vs strategy/signals_donchian.py
# ---------------------------------------------------------------------------

class TestChannelExitParity:
    """channel_exit_signal must match next()'s exit branch on every bar.

    Reference = attach_donchian (signals_donchian's own column builder, called
    single-TF by passing the 4h frame as both args) + the class's exit rule:
        long  exits when close < DonchianExitLower  (guarded on all 5 finite)
        short exits when close > DonchianExitUpper
    channel_exit_signal computes the same channels on each prefix df.iloc[:i+1];
    rolling/shift are causal so the prefix value at bar i equals the full-series
    column at bar i — that equivalence is the whole point of the live port.
    """

    def _series(self) -> pd.DataFrame:
        i = np.arange(220)
        close = (30000.0 + 15.0 * i
                 + 800.0 * np.sin(i / 4.0)
                 + 400.0 * np.sin(i / 11.0))
        return _ohlc(close)

    def _reference(self, df: pd.DataFrame):
        ref = attach_donchian(df, df, period_entry=70, period_exit=12,
                              atr_period=18)
        longs, shorts = [], []
        for i in range(len(df)):
            u = ref["DonchianUpper"].iloc[i]
            lo = ref["DonchianLower"].iloc[i]
            eu = ref["DonchianExitUpper"].iloc[i]
            el = ref["DonchianExitLower"].iloc[i]
            a = ref["ATR_1h"].iloc[i]
            c = df["Close"].iloc[i]
            guard = bool(np.all(np.isfinite([u, lo, eu, el, a])))
            longs.append(guard and c < el)
            shorts.append(guard and c > eu)
        return longs, shorts

    def test_long_and_short_parity_bar_for_bar(self):
        df = self._series()
        ref_long, ref_short = self._reference(df)
        for i in range(len(df)):
            got_long, _ = channel_exit_signal(df.iloc[:i + 1], "long", PARAMS)
            got_short, _ = channel_exit_signal(df.iloc[:i + 1], "short", PARAMS)
            assert got_long == ref_long[i], f"long mismatch at bar {i}"
            assert got_short == ref_short[i], f"short mismatch at bar {i}"

    def test_series_actually_exercises_both_exits(self):
        """Guard against a vacuous all-False parity pass."""
        df = self._series()
        ref_long, ref_short = self._reference(df)
        assert sum(ref_long) >= 3, "synthetic series triggers too few long exits"
        assert sum(ref_short) >= 3, "synthetic series triggers too few short exits"


# ---------------------------------------------------------------------------
# 2. Edge cases (strict inequality, NaN, warmup, flat side)
# ---------------------------------------------------------------------------

class TestChannelExitEdges:

    def test_long_exactly_at_channel_does_not_exit(self):
        # prev-3 min = 100; close == 100 → strict '<' means HOLD.
        df = _df_from_closes([110, 105, 100, 100])
        should, dbg = channel_exit_signal(df, "long", SMALL)
        assert should is False
        assert dbg["reason"] == "hold"
        assert dbg["exit_lower"] == pytest.approx(100.0)

    def test_long_just_below_channel_exits(self):
        df = _df_from_closes([110, 105, 100, 99])
        should, dbg = channel_exit_signal(df, "long", SMALL)
        assert should is True
        assert dbg["reason"] == "channel_exit_long"

    def test_short_exactly_at_channel_does_not_exit(self):
        # prev-3 max = 110; close == 110 → HOLD.
        df = _df_from_closes([100, 105, 110, 110])
        should, dbg = channel_exit_signal(df, "short", SMALL)
        assert should is False
        assert dbg["reason"] == "hold"
        assert dbg["exit_upper"] == pytest.approx(110.0)

    def test_short_just_above_channel_exits(self):
        df = _df_from_closes([100, 105, 110, 111])
        should, dbg = channel_exit_signal(df, "short", SMALL)
        assert should is True
        assert dbg["reason"] == "channel_exit_short"

    def test_nan_indicator_holds(self):
        # A NaN inside the rolling window (min_periods) → NaN channel → guard trips.
        df = _df_from_closes([100, 101, float("nan"), 102, 103])
        should, dbg = channel_exit_signal(df, "long", SMALL)
        assert should is False
        assert dbg["reason"] == "nan_indicators"

    def test_insufficient_bars_is_warmup(self):
        df = _df_from_closes([100, 101, 102])  # len 3 < need 4
        should, dbg = channel_exit_signal(df, "long", SMALL)
        assert should is False
        assert dbg["reason"] == "warmup"
        assert dbg["need"] == 4 and dbg["have"] == 3

    def test_flat_or_unknown_side_never_exits(self):
        df = _df_from_closes([110, 105, 100, 99])  # would trigger a long exit
        for side in ("flat", "", None, "unknown"):
            should, dbg = channel_exit_signal(df, side, SMALL)
            assert should is False
            assert dbg["reason"] == "not_in_position"

    def test_warmup_uses_the_configured_periods(self):
        # With the 70/12 PARAMS geometry, need = 71; a 40-bar frame is warmup.
        df = _ohlc(30000.0 + np.arange(40, dtype=float))
        should, dbg = channel_exit_signal(df, "long", PARAMS)
        assert should is False
        assert dbg["reason"] == "warmup"
        assert dbg["need"] == 71


# ---------------------------------------------------------------------------
# 3. binance_client: place_tp omits/keeps the TP leg
# ---------------------------------------------------------------------------

class TestBracketPlaceTp:

    def _client(self) -> BinanceClient:
        c = BinanceClient(ex=MagicMock(), env="testnet", coid_prefix="snap-d3-")
        c._round_qty = lambda symbol, qty: qty  # type: ignore[assignment]
        c._create_order_with_coid_retry = MagicMock(  # type: ignore[assignment]
            side_effect=lambda *a, **kw: {"id": f"{a[1]}"})
        return c

    def _order_types(self, client) -> list[str]:
        # positional signature: (symbol, order_type, side, qty, price, params)
        return [call.args[1]
                for call in client._create_order_with_coid_retry.call_args_list]

    def test_market_default_places_tp(self):
        c = self._client()
        out = c.market_order_with_bracket(
            "BTC/USDT:USDT", "long", 0.01, 64000.0, 66000.0,
            client_order_id_root="1")
        types = self._order_types(c)
        assert "TAKE_PROFIT_MARKET" in types
        assert out["tp"] is not None

    def test_market_place_tp_false_omits_tp(self):
        c = self._client()
        out = c.market_order_with_bracket(
            "BTC/USDT:USDT", "long", 0.01, 64000.0, 66000.0,
            client_order_id_root="1", place_tp=False)
        types = self._order_types(c)
        assert "market" in types and "STOP_MARKET" in types
        assert "TAKE_PROFIT_MARKET" not in types
        assert out["tp"] is None

    def test_place_brackets_false_returns_none_tp(self):
        c = self._client()
        sl, tp = c._place_brackets(
            "BTC/USDT:USDT", "long", 0.01, 65000.0, 1000.0, 5000.0,
            client_order_id_root="1", place_tp=False)
        assert sl is not None and tp is None
        assert "TAKE_PROFIT_MARKET" not in self._order_types(c)

    def test_place_brackets_default_places_tp(self):
        c = self._client()
        sl, tp = c._place_brackets(
            "BTC/USDT:USDT", "short", 0.01, 65000.0, 1000.0, 5000.0,
            client_order_id_root="1")
        assert sl is not None and tp is not None
        assert "TAKE_PROFIT_MARKET" in self._order_types(c)


# ---------------------------------------------------------------------------
# 4. strategy gate + _place_live_entry threads place_tp per strategy
# ---------------------------------------------------------------------------

_BASE_PARAMS: dict = {
    "symbol": "BTC/USDT:USDT",
    "hedge": {"enabled": False, "client_order_id_prefix": "snap-v1-"},
    "execution": {"poll_interval_s": "5", "order_type": "market",
                  "limit_offset_bps": "0", "limit_timeout_s": "25"},
    "timeframes": {"entry": "15m"},
    "sizing": {"leverage": 5, "risk_per_trade_pct": 1.0},
    "deploy": {"kill_switch_equity_fraction": 0.5, "min_capital_warn_usdt": 1.0},
    "strategy": {},
    # observe_only False so the TestMaybeReprotect cases below still exercise
    # the acting path they were written for; production ships observe_only True
    # for its rollout phase (see config/params.yaml and
    # tests/test_reprotect_algo_aware.py).
    "reprotect": {"enabled": True, "observe_only": False,
                  "max_replaces_per_position": 3},
}


def _mock_client() -> MagicMock:
    mc = MagicMock()
    mc.ex.parse_timeframe.return_value = 900
    mc.coid_prefix = "snap-v1-"
    mc.env = "testnet"
    mc.cancel_open_orders.return_value = 0
    # Algo order book: reprotect reads BOTH books since 2026-08-11. Default to
    # "readable, and empty" so these tests keep describing the PLAIN book only —
    # a bare MagicMock here would be iterated as if it were rows.
    mc.fetch_algo_orders.return_value = ([], True)
    return mc


def _make_bot(strategy_name: str | None, order_type: str = "market",
              entry_tf: str = "15m", dry_run: bool = False):
    from bot import Bot
    params = {**_BASE_PARAMS,
              "execution": {**_BASE_PARAMS["execution"], "order_type": order_type},
              "timeframes": {"entry": entry_tf}}
    if strategy_name is not None:
        params["strategy_name"] = strategy_name
    mc = _mock_client()
    with patch("bot.BinanceClient.from_env", return_value=mc):
        bot = Bot(params=params, dry_run=dry_run)
    return bot, mc


def test_strategy_uses_channel_exit_predicate():
    assert strategy_uses_channel_exit("donchian-v3") is True
    for other in ("multifactor-v1", "cnh-hybrid-short-v1", "v3-all-wider-4", ""):
        assert strategy_uses_channel_exit(other) is False


class TestPlaceLiveEntryPlaceTp:

    def _decision(self) -> SignalDecision:
        return SignalDecision(side="long", price=65000.0,
                              sl_distance=975.0, tp_distance=4875.0, debug={})

    def test_v1_market_entry_places_tp(self):
        bot, mc = _make_bot(strategy_name="multifactor-v1", order_type="market")
        mc.market_order_with_bracket.return_value = {"entry": {}, "sl": {}, "tp": {}}
        with patch("bot.trade_events.record_market_entry"), \
             patch("bot.state.set_meta"):
            bot._place_live_entry(self._decision(), qty=0.01,
                                  signal_id="1", equity=10000.0)
        assert mc.market_order_with_bracket.call_args.kwargs["place_tp"] is True

    def test_supertrend_entry_KEEPS_tp(self):
        """supertrend's TP is a real bracket leg — only donchian omits it."""
        bot, mc = _make_bot(strategy_name="supertrend", order_type="market",
                            entry_tf="4h")
        mc.market_order_with_bracket.return_value = {"entry": {}, "sl": {}, "tp": {}}
        with patch("bot.trade_events.record_market_entry"), \
             patch("bot.state.set_meta"):
            bot._place_live_entry(self._decision(), qty=0.01,
                                  signal_id="1", equity=10000.0)
        assert mc.market_order_with_bracket.call_args.kwargs["place_tp"] is True

    def test_donchian_market_entry_omits_tp(self):
        bot, mc = _make_bot(strategy_name="donchian-v3", order_type="market",
                            entry_tf="4h")
        mc.market_order_with_bracket.return_value = {"entry": {}, "sl": {}, "tp": None}
        with patch("bot.trade_events.record_market_entry"), \
             patch("bot.state.set_meta"):
            bot._place_live_entry(self._decision(), qty=0.01,
                                  signal_id="1", equity=10000.0)
        assert mc.market_order_with_bracket.call_args.kwargs["place_tp"] is False

    def test_donchian_limit_entry_omits_tp(self):
        bot, mc = _make_bot(strategy_name="donchian-v3", order_type="limit",
                            entry_tf="4h")
        mc.limit_order_with_bracket.return_value = {
            "entry": {}, "sl": {}, "tp": None, "filled_as": "limit",
            "fill_price": 64990.0, "filled_qty": 0.01}
        with patch("bot.trade_events.record_limit_entry"), \
             patch("bot.state.set_meta"):
            bot._place_live_entry(self._decision(), qty=0.01,
                                  signal_id="1", equity=10000.0)
        assert mc.limit_order_with_bracket.call_args.kwargs["place_tp"] is False


# ---------------------------------------------------------------------------
# 5. bot._maybe_channel_exit wiring
# ---------------------------------------------------------------------------

def _open(side: str = "long") -> Position:
    return Position(symbol="BTC/USDT:USDT", side=side, qty=0.02,
                    entry_price=65000.0, unrealized_pnl=5.0, margin_used=100.0)


def _flat() -> Position:
    return Position(symbol="BTC/USDT:USDT", side="flat", qty=0.0,
                    entry_price=0.0, unrealized_pnl=0.0, margin_used=0.0)


def _big_df(rows: int = 260) -> pd.DataFrame:
    return _ohlc(30000.0 + np.arange(rows, dtype=float))


class TestMaybeChannelExit:

    def test_noop_for_a_leg_without_a_trend_exit(self):
        """Was parameterised on multifactor-v1 until 2026-08-10, when v1 gained
        its adverse-EMA exit (see tests/test_v1_trend_exit.py). Re-pointed at
        legs that genuinely have no trend exit, so the no-op path stays covered
        instead of the assertion just being deleted."""
        for name in ("v3-all-wider-4", "cnh-hybrid-short"):
            bot, mc = _make_bot(strategy_name=name)
            mc.reset_mock()
            bot._maybe_channel_exit(equity=10000.0)
            mc.fetch_position.assert_not_called()
            mc.fetch_ohlcv.assert_not_called()
            mc.close_position.assert_not_called()

    def test_v1_leg_also_reaches_the_exit_hook(self):
        """v1's adverse-EMA(200) exit is the rule every v1 sign-off measured but
        the live bot never ran. Like supertrend, v1 keeps its TP bracket AND
        needs the hook — and it must be evaluated on v1's 15m entry timeframe,
        not the 4h the other two trend-exit legs use."""
        bot, mc = _make_bot(strategy_name="multifactor-v1", entry_tf="15m")
        mc.fetch_position.return_value = _open("long")
        mc.fetch_ohlcv.return_value = _big_df(260)
        with patch("bot.trend_exit_signal", return_value=(False, {})) as ce:
            bot._maybe_channel_exit(equity=10000.0)
        ce.assert_called_once()
        assert ce.call_args.args[0] == "multifactor-v1"
        assert mc.fetch_ohlcv.call_args.args[1] == "15m"

    def test_noop_when_flat(self):
        bot, mc = _make_bot(strategy_name="donchian-v3", entry_tf="4h")
        mc.fetch_position.return_value = _flat()
        bot._maybe_channel_exit(equity=10000.0)
        mc.fetch_ohlcv.assert_not_called()
        mc.close_position.assert_not_called()

    def test_drops_forming_bar_before_evaluating(self):
        bot, mc = _make_bot(strategy_name="donchian-v3", entry_tf="4h")
        mc.fetch_position.return_value = _open("long")
        raw = _big_df(260)
        mc.fetch_ohlcv.return_value = raw
        with patch("bot.trend_exit_signal", return_value=(False, {})) as ce:
            bot._maybe_channel_exit(equity=10000.0)
        # dispatcher signature is (strategy_name, bars, side, params)
        assert ce.call_args.args[0] == "donchian-v3"
        passed_df = ce.call_args.args[1]
        assert len(passed_df) == len(raw) - 1  # forming last row dropped
        mc.close_position.assert_not_called()

    def test_supertrend_leg_also_reaches_the_exit_hook(self):
        """The hook is gated on strategy_uses_trend_exit, NOT on TP omission.

        supertrend keeps its TP bracket and ALSO needs the flip exit, so it must
        reach the hook. The regression this guards: re-gating the hook on
        strategy_uses_channel_exit would silently disable the flip exit and
        leave supertrend positions running to SL/TP only.
        """
        bot, mc = _make_bot(strategy_name="supertrend", entry_tf="4h")
        mc.fetch_position.return_value = _open("long")
        mc.fetch_ohlcv.return_value = _big_df(260)
        with patch("bot.trend_exit_signal", return_value=(False, {})) as ce:
            bot._maybe_channel_exit(equity=10000.0)
        ce.assert_called_once()
        assert ce.call_args.args[0] == "supertrend"

    def test_trigger_closes_reduce_only_and_records(self):
        bot, mc = _make_bot(strategy_name="donchian-v3", entry_tf="4h")
        mc.fetch_position.return_value = _open("long")
        mc.fetch_ohlcv.return_value = _big_df(260)
        dbg = {"reason": "channel_exit_long", "cur_close": 29999.0,
               "exit_lower": 30000.0, "exit_upper": 30010.0}
        with patch("bot.trend_exit_signal", return_value=(True, dbg)), \
             patch("bot.state.latest_entry_coid_root", return_value="1700000000000"), \
             patch("bot.state.record_fill") as rec, \
             patch("bot.state.enqueue_bot_event") as enq, \
             patch("bot.send_alert") as alert:
            bot._maybe_channel_exit(equity=12345.0)
        mc.close_position.assert_called_once_with(
            "BTC/USDT:USDT", client_order_id_root="1700000000000", close_leg="ce")
        assert rec.call_args.kwargs["reason"] == "channel_exit"
        assert rec.call_args.kwargs["side"] == "close"
        assert enq.call_args.kwargs["payload"]["reason"] == "channel_exit"
        alert.assert_called_once()

    def test_dry_run_logs_and_does_not_close(self):
        bot, mc = _make_bot(strategy_name="donchian-v3", entry_tf="4h",
                            dry_run=True)
        mc.fetch_position.return_value = _open("short")
        mc.fetch_ohlcv.return_value = _big_df(260)
        dbg = {"reason": "channel_exit_short", "cur_close": 30011.0,
               "exit_lower": 30000.0, "exit_upper": 30010.0}
        with patch("bot.trend_exit_signal", return_value=(True, dbg)), \
             patch("bot.state.record_fill") as rec, \
             patch("bot.send_alert") as alert:
            bot._maybe_channel_exit(equity=10000.0)
        mc.close_position.assert_not_called()
        rec.assert_not_called()
        alert.assert_not_called()


# ---------------------------------------------------------------------------
# 6. bot._maybe_reprotect (bracket-health-check) wiring
# ---------------------------------------------------------------------------

_ACTIVE_BRACKET = json.dumps({
    "signal_id": "1700000000000", "side": "long", "entry_price": 65000.0,
    "sl_distance": 975.0, "tp_distance": 1950.0, "place_tp": True,
})


# Plain-book rows as ccxt returns them for OUR bracket: tagged with this
# position's exact root and on the closing side. Since 2026-10-13 reprotect
# counts a leg only with the exact root + closing side (adopt_bracket_state) —
# an untagged row is someone else's order and does not protect "our" position.
def _ro_sl() -> dict:
    return {"clientOrderId": "snap-v1-1700000000000-s", "side": "sell",
            "info": {"type": "STOP_MARKET", "reduceOnly": "true", "stopPrice": "64000"}}


def _ro_tp() -> dict:
    return {"clientOrderId": "snap-v1-1700000000000-t", "side": "sell",
            "info": {"type": "TAKE_PROFIT_MARKET", "reduceOnly": "true", "stopPrice": "67000"}}


class TestMaybeReprotect:

    def _bot(self, dry_run: bool = False):
        bot, mc = _make_bot(strategy_name="multifactor-v1", dry_run=dry_run)
        mc.reset_mock()
        mc.fetch_position.return_value = _open("long")  # qty 0.02 @ 65000
        mc.cancel_open_orders.return_value = 0
        return bot, mc

    def test_replaces_when_bracket_missing(self):
        bot, mc = self._bot()
        mc.ex.fetch_open_orders.return_value = []  # both legs gone
        with patch("bot.state.get_meta", return_value=_ACTIVE_BRACKET), \
             patch("bot.state.set_meta") as set_meta, \
             patch("bot.state.record_event"), patch("bot.send_alert") as alert:
            bot._maybe_reprotect(equity=10000.0)
        # The per-position re-place counter is persisted back into
        # active_bracket so the cap survives a restart (2026-08-11).
        persisted = json.loads(set_meta.call_args.args[1])
        assert persisted["reprotect_count"] == 1
        # Must go through the raw ccxt client (self.client.ex), NOT self.client —
        # guards the AttributeError-on-every-poll regression the reviewer caught.
        assert mc.ex.fetch_open_orders.called
        # 2026-10-13: only the MISSING legs are placed, tagged-only, with this
        # root's exact ids — no cancel-everything-first (that cancelled a
        # healthy stop to fix a TP).
        mc.cancel_open_orders.assert_not_called()
        mc._place_brackets.assert_not_called()
        mc.place_tagged_stop.assert_called_once_with(
            "BTC/USDT:USDT", "long", 0.02, 65000.0 - 975.0, "1700000000000", "s")
        mc.place_tagged_leg.assert_called_once_with(
            "BTC/USDT:USDT", "long", 0.02, 65000.0 + 1950.0, "1700000000000", "t")
        alert.assert_called_once()

    def test_noop_when_bracket_intact(self):
        bot, mc = self._bot()
        mc.ex.fetch_open_orders.return_value = [_ro_sl(), _ro_tp()]
        with patch("bot.state.get_meta", return_value=_ACTIVE_BRACKET), \
             patch("bot.send_alert"):
            bot._maybe_reprotect(equity=10000.0)
        mc.cancel_open_orders.assert_not_called()
        mc._place_brackets.assert_not_called()
        mc.place_tagged_stop.assert_not_called()
        mc.place_tagged_leg.assert_not_called()

    def test_noop_on_side_mismatch(self):
        bot, mc = self._bot()
        short_ab = json.dumps({**json.loads(_ACTIVE_BRACKET), "side": "short"})
        with patch("bot.state.get_meta", return_value=short_ab):
            bot._maybe_reprotect(equity=10000.0)
        mc.ex.fetch_open_orders.assert_not_called()
        mc._place_brackets.assert_not_called()

    def test_noop_on_entry_price_drift(self):
        bot, mc = self._bot()
        drifted = json.dumps({**json.loads(_ACTIVE_BRACKET), "entry_price": 50000.0})
        with patch("bot.state.get_meta", return_value=drifted):
            bot._maybe_reprotect(equity=10000.0)
        mc._place_brackets.assert_not_called()

    def test_a_resting_stop_is_never_cancelled_to_fix_the_tp(self):
        """Replaces test_skips_replace_if_cancel_leaves_a_leg (2026-10-13). The
        old flow cancelled every leg, re-read, and refused to place if one
        survived — which also meant it CANCELLED A HEALTHY STOP to fix a missing
        TP. Now only the missing leg is placed, under its exact id, so there is
        nothing to cancel and no duplicate to guard against."""
        bot, mc = self._bot()
        mc.ex.fetch_open_orders.return_value = [_ro_sl()]
        with patch("bot.state.get_meta", return_value=_ACTIVE_BRACKET), \
             patch("bot.state.set_meta"), \
             patch("bot.state.record_event"), patch("bot.send_alert"):
            bot._maybe_reprotect(equity=10000.0)
        mc.cancel_open_orders.assert_not_called()
        mc.ex.cancel_order.assert_not_called()
        mc.place_tagged_stop.assert_not_called()
        mc.place_tagged_leg.assert_called_once()

    def test_clears_stale_meta_when_flat(self):
        bot, mc = self._bot()
        mc.fetch_position.return_value = _flat()
        with patch("bot.state.set_meta") as set_meta:
            bot._maybe_reprotect(equity=10000.0)
        set_meta.assert_called_once_with("active_bracket", "")
        mc._place_brackets.assert_not_called()

    def test_dry_run_noop(self):
        bot, mc = self._bot(dry_run=True)
        with patch("bot.state.get_meta", return_value=_ACTIVE_BRACKET):
            bot._maybe_reprotect(equity=10000.0)
        mc.fetch_position.assert_not_called()
        mc._place_brackets.assert_not_called()
