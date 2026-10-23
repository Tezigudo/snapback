"""Algo-aware bracket detection for `_maybe_reprotect`.

WHY THIS FILE EXISTS
`_maybe_reprotect` was disabled 2026-07-22 after re-placing a HEALTHY bracket
until Binance returned `-4045 Reach max stop order limit`. The detector read
`fetch_open_orders()` (/fapi/v1/openOrders), which does not return the
conditional/algo orders `_place_brackets` actually creates — so a fully
protected position looked unprotected on every poll.

Re-enabling it is only safe if the detector is proven against the REAL algo
payload, so `ALGO_SL` / `ALGO_TP` below are copied verbatim from v1's own
resting bracket on 2026-08-10 rather than invented. The single most important
test here is `test_the_july_false_negative_is_gone`.
"""

from __future__ import annotations

import json

import pytest

from bot_internals import (
    algo_bracket_leg,
    bracket_is_intact,
    bracket_state,
    reduce_only_bracket_leg,
)

PREFIX = "snap-v1-"
ROOT = "1700000000000"

# Shape of a live v1 row while a healthy bracket rests (ids and prices are
# synthetic). Note what is NOT here: any type field. `strategyType` was null on BOTH legs.
ALGO_SL = {"algoId": "1000000000000001", "clientAlgoId": f"{PREFIX}{ROOT}-s",
           "side": "SELL", "positionSide": "BOTH", "triggerPrice": "48000.0",
           "strategyType": None, "origQty": None, "reduceOnly": True,
           "algoStatus": "NEW", "bookTime": None}
ALGO_TP = {"algoId": "1000000000000003", "clientAlgoId": f"{PREFIX}{ROOT}-t",
           "side": "SELL", "positionSide": "BOTH", "triggerPrice": "52000.0",
           "strategyType": None, "origQty": None, "reduceOnly": True,
           "algoStatus": "NEW", "bookTime": None}


def _plain(otype: str, reduce_only: bool = True) -> dict:
    """A ccxt /fapi/v1/openOrders row (the legacy shape)."""
    return {"info": {"type": otype, "reduceOnly": str(reduce_only).lower()},
            "type": otype.lower()}


class TestTheRegression:
    """The exact false negative that caused the -4045 spam."""

    def test_the_july_false_negative_is_gone(self) -> None:
        """A healthy bracket that rests ONLY on the algo endpoint — which is
        where Binance actually parks it — must read INTACT.

        Under the old plain-only detector this returned False on every 5s poll,
        which is what re-placed until the max-stop-order limit.
        """
        st = bracket_state([], [ALGO_SL, ALGO_TP], PREFIX, place_tp=True)
        assert st.sl and st.tp
        assert st.intact is True

    def test_the_old_detector_still_demonstrates_the_bug(self) -> None:
        """Pin WHY a new helper was needed rather than reusing the old one.

        If this ever starts passing, the plain classifier learned the algo shape
        and `algo_bracket_leg` may be redundant — but silently keeping both
        would be worse than knowing.
        """
        assert bracket_is_intact([ALGO_SL, ALGO_TP], place_tp=True) is False
        assert reduce_only_bracket_leg(ALGO_SL) is None
        assert reduce_only_bracket_leg(ALGO_TP) is None


class TestAlgoLegClassification:

    def test_classifies_by_coid_suffix(self) -> None:
        assert algo_bracket_leg(ALGO_SL, PREFIX) == "sl"
        assert algo_bracket_leg(ALGO_TP, PREFIX) == "tp"

    def test_does_not_depend_on_a_type_field(self) -> None:
        """The live rows carry `strategyType: None` and no `type` at all —
        classification must not regress to reading one."""
        for row in (ALGO_SL, ALGO_TP):
            assert "type" not in row
            assert row["strategyType"] is None
        assert algo_bracket_leg(ALGO_SL, PREFIX) == "sl"

    def test_foreign_prefix_is_not_our_bracket(self) -> None:
        """An order placed by hand in the Binance app must NOT count as
        protection — reprotect could not re-place it faithfully anyway."""
        manual = {**ALGO_SL, "clientAlgoId": "web_abc123-s"}
        assert algo_bracket_leg(manual, PREFIX) is None
        st = bracket_state([], [manual], PREFIX, place_tp=False)
        assert st.intact is False

    def test_another_legs_prefix_is_not_ours(self) -> None:
        """Sub-accounts are isolated, but prefix-scoping is the guard that makes
        that structural rather than incidental."""
        other = {**ALGO_SL, "clientAlgoId": f"snap-don-{ROOT}-s"}
        assert algo_bracket_leg(other, PREFIX) is None

    def test_non_reduce_only_is_not_a_bracket_leg(self) -> None:
        entry = {**ALGO_SL, "clientAlgoId": f"{PREFIX}{ROOT}-e", "reduceOnly": False}
        assert algo_bracket_leg(entry, PREFIX) is None

    def test_unknown_leg_suffix_is_ignored(self) -> None:
        assert algo_bracket_leg({**ALGO_SL, "clientAlgoId": f"{PREFIX}{ROOT}-x"},
                                PREFIX) is None

    @pytest.mark.parametrize("bad", [
        {}, {"clientAlgoId": None}, {"clientAlgoId": 123},
        {"clientAlgoId": f"{PREFIX}{ROOT}-s"},          # reduceOnly absent
    ])
    def test_malformed_rows_never_raise(self, bad: dict) -> None:
        assert algo_bracket_leg(bad, PREFIX) is None

    def test_string_reduce_only_is_tolerated(self) -> None:
        """Live rows use a real bool; tolerate the plain endpoint's string form
        in case Binance ever aligns the two shapes."""
        assert algo_bracket_leg({**ALGO_SL, "reduceOnly": "true"}, PREFIX) == "sl"


class TestBracketStateMergesBothBooks:

    def test_plain_only_bracket_still_detected(self) -> None:
        """The legacy path must keep working — donchian/supertrend rely on it."""
        st = bracket_state([_plain("STOP_MARKET"), _plain("TAKE_PROFIT_MARKET")],
                           [], PREFIX, place_tp=True)
        assert st.intact is True

    def test_split_across_both_books(self) -> None:
        """SL plain + TP algo. Reading either book alone under-reports; this is
        why the merge exists rather than a straight endpoint swap."""
        st = bracket_state([_plain("STOP_MARKET")], [ALGO_TP], PREFIX, place_tp=True)
        assert st.sl and st.tp and st.intact

    def test_missing_sl_is_not_intact(self) -> None:
        st = bracket_state([], [ALGO_TP], PREFIX, place_tp=True)
        assert st.intact is False
        assert "SL=MISSING" in st.describe()

    def test_missing_tp_is_not_intact_when_tp_expected(self) -> None:
        st = bracket_state([], [ALGO_SL], PREFIX, place_tp=True)
        assert st.intact is False

    def test_sl_alone_is_intact_when_no_tp_is_placed(self) -> None:
        """donchian-v3 omits the TP leg by design — its channel exit IS the
        profit-taking mechanism. Demanding a TP would re-place forever."""
        st = bracket_state([], [ALGO_SL], PREFIX, place_tp=False)
        assert st.intact is True
        assert "TP=n/a" in st.describe()

    def test_empty_books_are_not_intact(self) -> None:
        assert bracket_state([], [], PREFIX, place_tp=True).intact is False

    def test_any_leg_guards_the_post_cancel_recheck(self) -> None:
        """After cancelling, a SURVIVING leg must block the re-place or we place
        a duplicate pair beside it. A plain-only re-check would miss an algo
        survivor — the same blind spot on the more dangerous side."""
        assert bracket_state([], [ALGO_SL], PREFIX, place_tp=True).any_leg is True
        assert bracket_state([], [], PREFIX, place_tp=True).any_leg is False

    def test_none_inputs_are_tolerated(self) -> None:
        st = bracket_state(None, None, PREFIX, place_tp=True)
        assert st.intact is False and st.any_leg is False


# ---------------------------------------------------------------------------
# bot._maybe_reprotect — the brakes
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch  # noqa: E402

from exchange.binance_client import Position  # noqa: E402

_PARAMS: dict = {
    "symbol": "BTC/USDT:USDT",
    "strategy_name": "multifactor-v1",
    "hedge": {"enabled": False, "client_order_id_prefix": PREFIX},
    "execution": {"poll_interval_s": "5", "order_type": "market",
                  "limit_offset_bps": "0", "limit_timeout_s": "25"},
    "timeframes": {"entry": "15m"},
    "sizing": {"leverage": 5, "risk_per_trade_pct": 1.0},
    "deploy": {"kill_switch_equity_fraction": 0.5, "min_capital_warn_usdt": 1.0},
    "strategy": {},
    "reprotect": {"enabled": True, "observe_only": False,
                  "max_replaces_per_position": 3},
}

_OPEN = Position(symbol="BTC/USDT:USDT", side="long", qty=0.005,
                 entry_price=64911.2, unrealized_pnl=0.0, margin_used=50.0)
_FLAT = Position(symbol="BTC/USDT:USDT", side="flat", qty=0.0,
                 entry_price=0.0, unrealized_pnl=0.0, margin_used=0.0)

_BRACKET = {"signal_id": ROOT, "side": "long", "entry_price": 64911.2,
            "sl_distance": 973.67, "tp_distance": 1947.33, "place_tp": True}


def _bot(reprotect: dict | None = None, algo_rows=(), plain_orders=(),
         readable: bool = True):
    from bot import Bot
    params = {**_PARAMS, "reprotect": {**_PARAMS["reprotect"], **(reprotect or {})}}
    mc = MagicMock()
    mc.ex.parse_timeframe.return_value = 900
    mc.coid_prefix = PREFIX
    mc.env = "testnet"
    mc.cancel_open_orders.return_value = 0
    mc.fetch_position.return_value = _OPEN
    mc.ex.fetch_open_orders.return_value = list(plain_orders)
    mc.fetch_algo_orders.return_value = (list(algo_rows), readable)
    with patch("bot.BinanceClient.from_env", return_value=mc):
        b = Bot(params=params, dry_run=False)
    return b, mc


class TestMaybeReprotectBrakes:

    def test_healthy_algo_bracket_places_nothing(self) -> None:
        """The July scenario end to end: position open, bracket resting on the
        algo endpoint only. Must be a complete no-op."""
        bot, mc = _bot(algo_rows=[ALGO_SL, ALGO_TP])
        with patch("bot.state.get_meta", return_value=json.dumps(_BRACKET)):
            bot._maybe_reprotect(equity=147.0)
        mc._place_brackets.assert_not_called()
        mc.cancel_open_orders.assert_not_called()

    def test_observe_only_logs_and_places_nothing(self) -> None:
        bot, mc = _bot({"observe_only": True}, algo_rows=[])
        with patch("bot.state.get_meta", return_value=json.dumps(_BRACKET)), \
             patch.object(bot.log, "warning") as warn:
            bot._maybe_reprotect(equity=147.0)
        mc._place_brackets.assert_not_called()
        assert any("WOULD re-place" in str(c) for c in warn.call_args_list)

    def test_disabled_is_a_total_noop(self) -> None:
        bot, mc = _bot({"enabled": False}, algo_rows=[])
        bot._maybe_reprotect(equity=147.0)
        mc.fetch_position.assert_not_called()

    def test_failed_algo_read_never_acts(self) -> None:
        """THE bug Sourcery caught on PR #20, and it is the July bug again.

        fetch_algo_orders never raises, so "no algo orders" and "the call
        failed" are the same empty list. An earlier version only abstained when
        the ccxt METHOD was missing — but the endpoint can exist and the CALL
        still fail (network blip, 5xx, rate limit), and then a healthy bracket
        reads as missing and gets re-placed. ok=False must stop it dead.
        """
        bot, mc = _bot({"observe_only": False}, algo_rows=[], readable=False)
        with patch("bot.state.get_meta", return_value=json.dumps(_BRACKET)), \
             patch("bot.state.set_meta"), patch("bot.state.record_event"), \
             patch("bot.send_alert"):
            bot._maybe_reprotect(equity=147.0)
        mc._place_brackets.assert_not_called()
        mc.cancel_open_orders.assert_not_called()

    def test_missing_legs_are_placed_tagged_only_without_a_sweep(self) -> None:
        """Replaces test_failed_algo_read_after_cancel_blocks_the_replace
        (2026-10-13). There is no cancel-then-re-read any more: only the legs
        whose EXACT id is absent are placed, tagged-only (no untagged
        fallback), so a duplicate of a resting leg of ours cannot be created and
        no post-cancel read is needed."""
        bot, mc = _bot({"observe_only": False}, algo_rows=[])
        with patch("bot.state.get_meta", return_value=json.dumps(_BRACKET)), \
             patch("bot.state.set_meta"), patch("bot.state.record_event"), \
             patch("bot.send_alert"):
            bot._maybe_reprotect(equity=147.0)
        mc.cancel_open_orders.assert_not_called()
        mc._place_brackets.assert_not_called()
        assert mc.fetch_algo_orders.call_count == 1
        mc.place_tagged_stop.assert_called_once()
        mc.place_tagged_leg.assert_called_once()

    def test_cap_with_the_stop_missing_closes_the_position(self) -> None:
        """God's rule (2026-10-13): a stop that cannot be kept ⇒ close. At the
        cap the stop has vanished cap+1 times; the old code alerted once and
        left the position naked. It must not place more orders either."""
        capped = {**_BRACKET, "reprotect_count": 3}
        bot, mc = _bot({"observe_only": False}, algo_rows=[ALGO_TP])
        mc.emergency_close.return_value = (True, None, None)
        with patch("bot.state.get_meta", return_value=json.dumps(capped)), \
             patch("bot.state.set_meta"), patch("bot.state.record_fill"), \
             patch("bot.send_alert") as alert:
            bot._maybe_reprotect(equity=147.0)
        mc.emergency_close.assert_called_once_with("BTC/USDT:USDT", ROOT,
                                                   side="long", qty=0.005)
        mc.place_tagged_stop.assert_not_called()
        assert "NO STOP" in alert.call_args.args[0]

    def test_cap_stops_replacing_and_alerts_once(self) -> None:
        """The brake that makes a detector bug survivable: -4045 needs many
        placements, and the cap makes many impossible. (Stop resting, TP
        missing — the case where the cap still only alerts.)"""
        capped = {**_BRACKET, "tp_reprotect_count": 3}
        bot, mc = _bot({"observe_only": False}, algo_rows=[ALGO_SL])
        with patch("bot.state.get_meta", return_value=json.dumps(capped)), \
             patch("bot.send_alert") as alert:
            for _ in range(5):
                bot._last_reprotect_ts = 0.0   # defeat the 60s throttle
                bot._maybe_reprotect(equity=147.0)
        mc._place_brackets.assert_not_called()
        mc.place_tagged_leg.assert_not_called()
        mc.emergency_close.assert_not_called()
        assert alert.call_count == 1, "cap alert must not spam"

    def test_flat_position_clears_the_stale_bracket_record(self) -> None:
        """The second reason to re-enable this: nothing else clears
        active_bracket, so it went stale after every trade while disabled."""
        bot, mc = _bot()
        mc.fetch_position.return_value = _FLAT
        with patch("bot.state.set_meta") as set_meta:
            bot._maybe_reprotect(equity=147.0)
        set_meta.assert_called_once_with("active_bracket", "")
