"""Red-team fix-pass for the no-stop rule (2026-10-13 panel, items 1–8).

Each class names the panel item. Scenarios C / D / F / G mirror the red team's
reproductions (/private/tmp/nostop_rt/test_rt.py) and assert the fixed
behaviour.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import ccxt
import pytest

from exchange import state

from ._fake_exchange import WOULD_TRIGGER, FakeEx, rejected, timeout
from .test_no_stop_close import (
    DECISION,
    DONCH_PARAMS,
    OLD,
    PREFIX,
    ROOT,
    V1_PARAMS,
    _bot,
    _enter,
    _fills,
    _open,
    _reprotect,
    _rest,
    _subjects,
)

NEW = "1792000000000"


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch("exchange.binance_client.time.sleep"), patch("bot.time.sleep"):
        yield


def _pending_after_failed_close(params=DONCH_PARAMS):
    """Entry fills, no stop possible, every close attempt fails."""
    ex = FakeEx()
    ex.fail["STOP_MARKET"] = [WOULD_TRIGGER]
    ex.fail["close"] = [rejected(-1001)] * 3
    bot = _bot(ex, params)
    with patch("bot.send_alert"):
        bot._place_live_entry(DECISION, 0.005, ROOT, 150.0)
    assert ex.pos is not None and bot._pending_record() is not None
    bot._no_stop_next_try = 0.0
    return bot, ex


class TestItem1StalePendingNeverTouchesAnotherPosition:

    def test_entries_are_blocked_while_a_close_is_pending(self) -> None:
        bot, ex = _pending_after_failed_close()
        ex.pos = None                       # e.g. God closed it by hand
        n = len(ex.calls)
        with patch("bot.send_alert"):
            _enter(bot)
        assert len(ex.calls) == n, "no exchange call, no entry while pending"

    def test_manual_close_clears_the_record_without_touching_anything(self) -> None:
        bot, ex = _pending_after_failed_close()
        ex.pos = None
        with patch("bot.send_alert") as alert:
            bot._maybe_finish_no_stop_close(150.0)
        assert bot._pending_record() is None
        assert state.get_meta("no_stop_close_pending") == ""
        assert ex.closes() == [] or all(c for c in ex.closes())
        assert [f[0] for f in _fills()] == ["entry", "no_stop_close"]
        assert any("flat" in s.lower() for s in _subjects(alert))

    @pytest.mark.parametrize("pos", [("long", 0.008, 65100.0), ("short", 0.005, 65100.0)],
                             ids=["bigger", "other-side"])
    def test_a_different_position_is_left_alone(self, pos) -> None:
        """Red team C/C2: the old retry swept the NEW position's SL/TP and
        closed part of it under the old root."""
        bot, ex = _pending_after_failed_close()
        ex.pos = pos
        _rest(ex, NEW, "s", prefix="snap-d3-")
        before_algo = list(ex.algo)
        n = len(ex.calls)
        with patch("bot.send_alert") as alert:
            bot._maybe_finish_no_stop_close(150.0)
        new_calls = ex.calls[n:]
        assert all(c[0] == "fetch_positions" for c in new_calls), new_calls
        assert ex.algo == before_algo and ex.pos == pos
        assert bot._pending_record() is None
        assert any("ABANDONED" in s for s in _subjects(alert))

    def test_a_new_root_is_left_alone_even_at_the_same_size(self) -> None:
        bot, ex = _pending_after_failed_close()
        state.record_fill(side="long", qty=0.005, price=65000.0, reason="entry",
                          client_order_id_root=NEW)
        n = len(ex.calls)
        with patch("bot.send_alert"):
            bot._maybe_finish_no_stop_close(150.0)
        assert all(c[0] == "fetch_positions" for c in ex.calls[n:])
        assert ex.pos is not None

    def test_the_emergency_sweep_is_scoped_to_its_own_root(self) -> None:
        from .test_no_stop_close import _client
        ex = FakeEx()
        ex.pos = ("long", 0.005, 65000.0)
        other = ex._rest_algo(f"{PREFIX}{OLD}-s", "SELL", 0.005, 1.0)
        mine = ex._rest_algo(f"{PREFIX}{ROOT}-t", "SELL", 0.005, 1.0)
        closed, _order, _err = _client(ex).emergency_close(
            "BTC/USDT:USDT", ROOT, side="long", qty=0.005)
        assert closed and ex.pos is None
        assert mine not in ex.algo
        assert other in ex.algo, "another root's order was swept by the emergency close"

    def test_boot_clears_a_pending_record_it_made_moot(self) -> None:
        bot, _ex = _pending_after_failed_close()
        bot._last_position_side = "unknown"
        bot._clear_no_stop_records_at_boot()
        assert bot._pending_record() is None


class TestItem2RestingEntryOrder:

    def _limit_bot(self, params):
        p = {**params, "execution": {**params["execution"], "order_type": "limit"}}
        ex = FakeEx()
        ex.fail["entry"] = [timeout()]
        ex.entry_timeout_lands = True
        return _bot(ex, p), ex

    @pytest.mark.parametrize("params", [V1_PARAMS, DONCH_PARAMS], ids=["v1", "donchian"])
    def test_a_timed_out_limit_that_rests_is_cancelled(self, params) -> None:
        bot, ex = self._limit_bot(params)
        with patch("bot.send_alert"):
            _enter(bot)
        assert ex.plain == [], "the resting GTC entry must be cancelled"
        assert any(c[0] == "cancel_order" and len(c) == 4 and
                   c[3]["origClientOrderId"].endswith("-e") for c in ex.calls)
        # watched for the settle window even though the read said "gone"
        assert json.loads(state.get_meta("entry_unverified"))["kind"] == "entry_unverified"

    def test_unconfirmed_cancel_is_watched_and_a_late_fill_gets_a_stop(self) -> None:
        bot, ex = self._limit_bot(DONCH_PARAMS)
        real = ex.fetch_open_orders
        reads = {"n": 0}

        def flaky(symbol=None):
            reads["n"] += 1
            if reads["n"] == 1:
                raise ccxt.NetworkError("open-orders read failed")
            return real(symbol)
        ex.fail["entry"] = [timeout()]
        with patch.object(ex, "cancel_order", side_effect=ccxt.NetworkError("x")), \
             patch.object(ex, "fetch_open_orders", side_effect=flaky), \
             patch("bot.send_alert") as alert:
            _enter(bot)
        assert json.loads(state.get_meta("entry_unverified"))["kind"] == "entry_unverified"
        assert any("may still be resting" in s for s in _subjects(alert))
        # blocked while unverified
        n = len(ex.calls)
        _enter(bot)
        assert len(ex.calls) == n
        # the order fills a minute later
        ex.plain = []
        ex.pos = ("long", 0.005, 65000.0)
        ex.entry_filled = 0.005
        bot._unverified_next = 0.0
        with patch("bot.send_alert") as alert2:
            bot._maybe_guard_unverified_entry(150.0)
        stops = [r["clientAlgoId"] for r in ex.algo]
        assert any(c.startswith("snap-d3-") and c.endswith("-s") for c in stops)
        assert ex.pos is not None and ex.closes() == []
        assert state.get_meta("entry_unverified") == ""
        assert [f[0] for f in _fills()] == ["entry"]
        assert any("stop placed" in s.lower() for s in _subjects(alert2))

    def test_guard_makes_no_call_when_nothing_is_unverified(self) -> None:
        ex = FakeEx()
        bot = _bot(ex, DONCH_PARAMS)
        bot._maybe_guard_unverified_entry(150.0)
        bot._maybe_finish_no_stop_close(150.0)
        assert ex.calls == []


class TestItem3SeparateCaps:

    def test_tp_failures_do_not_spend_the_stop_budget(self) -> None:
        """Red team D: 3 TP failures, then the stop is cancelled externally.
        The stop must still get its re-place attempts, not an instant close."""
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex)
        _rest(ex, ROOT, "s")
        ex.fail["TAKE_PROFIT_MARKET"] = [rejected(-1001)] * 3
        with patch("bot.send_alert"):
            _reprotect(bot, 3)
        ex.algo = [r for r in ex.algo if not r["clientAlgoId"].endswith("-s")]
        with patch("bot.send_alert"):
            _reprotect(bot, 1)
        assert ex.pos is not None and ex.closes() == []
        assert [c[6]["newClientOrderId"] for c in ex.creates("STOP_MARKET")] == \
            [f"{PREFIX}{ROOT}-s"]


class TestItem4ClockDrift:

    def test_nonce_errors_resync_before_retrying(self) -> None:
        ex = FakeEx()
        ex.fail["STOP_MARKET"] = [ccxt.InvalidNonce("-1021")] * 3
        ex.fail["close"] = [ccxt.InvalidNonce("-1021")]
        bot = _bot(ex, DONCH_PARAMS)
        with patch("bot.send_alert"):
            bot._place_live_entry(DECISION, 0.005, ROOT, 150.0)
        assert sum(1 for c in ex.calls if c[0] == "load_time_difference") >= 4
        assert ex.pos is None

    def test_transient_entry_closes_do_not_latch(self) -> None:
        ex = FakeEx()
        bot = _bot(ex, DONCH_PARAMS)
        for i in range(3):
            ex.fail["STOP_MARKET"] = [timeout()] * 3
            with patch("bot.send_alert"):
                bot._place_live_entry(DECISION, 0.005, str(int(ROOT) + i), 150.0)
        assert bot._latch_active() is False


class TestItem5NoAlertOrRetryStorm:

    def test_repeated_cap_close_failure(self) -> None:
        """Cap reached with the stop missing, and the close keeps failing."""
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex, reprotect_count=3)
        ex.fail["close"] = [rejected(-1001)] * 100
        t = [1_000_000.0]
        with patch("bot.time.time", side_effect=lambda: t[0]), \
             patch("bot.send_alert") as alert:
            for _ in range(40):                 # 40 ticks, 5 s apart = 200 s
                bot._maybe_finish_no_stop_close(150.0)
                bot._maybe_reprotect(150.0)
                t[0] += 5.0
        closes = len([c for c in ex.creates("market") if c[6].get("reduceOnly")])
        failed = [s for s in _subjects(alert) if "FAILED" in s]
        assert ex.pos is not None
        assert len(failed) == 1, failed
        # 1 cap-close + retries at 15, 30, 60, 120 s backoff — not 2/min × 3
        assert closes <= 3 * 5, closes

    def test_failed_alert_repeats_after_15_minutes(self) -> None:
        bot, ex = _pending_after_failed_close()
        ex.fail["close"] = [rejected(-1001)] * 100
        import time as _time
        t = [_time.time() + 1.0]
        with patch("bot.time.time", side_effect=lambda: t[0]), \
             patch("bot.send_alert") as alert:
            for _ in range(20):                  # a retry every 100 s, 2000 s
                bot._no_stop_next_try = 0.0
                bot._maybe_finish_no_stop_close(150.0)
                t[0] += 100.0
        # the first alert went out at entry; repeats at ~900 s and ~1800 s only
        assert len([s for s in _subjects(alert) if "FAILED" in s]) == 2


class TestItem6Latch:

    def _no_stop_entry(self, bot, ex, i: int, err=None) -> None:
        ex.fail["STOP_MARKET"] = [err or WOULD_TRIGGER] * 3
        with patch("bot.send_alert") as alert:
            bot._place_live_entry(DECISION, 0.005, str(int(ROOT) + i), 150.0)
        return _subjects(alert)

    def test_two_non_transient_no_stop_entries_latch_for_four_hours(self) -> None:
        ex = FakeEx()
        bot = _bot(ex, DONCH_PARAMS)
        self._no_stop_entry(bot, ex, 0)
        assert bot._latch_active() is False
        subjects = self._no_stop_entry(bot, ex, 1, rejected())
        assert sum("LATCHED" in s for s in subjects) == 1
        n = len(ex.calls)
        with patch("bot.send_alert"):
            _enter(bot)
        assert len(ex.calls) == n, "latched: no entry, no exchange call"
        # 4 h later it releases itself
        later = json.loads(state.get_meta("no_stop_latch"))["until"] + 1
        with patch("bot.time.time", return_value=later), \
             patch("bot.send_alert") as alert:
            assert bot._latch_active() is False
        assert any("released" in s.lower() for s in _subjects(alert))
        # ...and two more re-latch it
        self._no_stop_entry(bot, ex, 2)
        self._no_stop_entry(bot, ex, 3)
        assert bot._latch_active() is True

    def test_a_protected_entry_resets_the_streak(self) -> None:
        ex = FakeEx()
        bot = _bot(ex, DONCH_PARAMS)
        self._no_stop_entry(bot, ex, 0)
        ex.pos = None
        ex.fail["STOP_MARKET"] = []
        with patch("bot.send_alert"):
            bot._place_live_entry(DECISION, 0.005, str(int(ROOT) + 1), 150.0)
        ex.pos = None
        ex.algo = []
        self._no_stop_entry(bot, ex, 2)
        assert bot._latch_active() is False


class TestItem7UntaggedStopIsNotOurs:

    def test_an_untagged_reduce_only_stop_now_reads_as_missing(self) -> None:
        """BEHAVIOUR CHANGE (2026-10-13), pinned on purpose. Before, any
        reduce-only STOP in the plain book counted as "our SL present". Now a
        leg counts only with this position's EXACT coid root on the closing
        side (adopt_bracket_state), so an untagged / manual stop reads as
        MISSING and reprotect ADDS ours beside it. Over-protection, not under:
        the old rule let a stale or foreign stop hide a missing one."""
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex)
        ex.plain.append({"id": "manual-1", "side": "sell", "type": "stop_market",
                         "info": {"type": "STOP_MARKET", "reduceOnly": "true",
                                  "stopPrice": "64000"}})
        _rest(ex, ROOT, "t")
        with patch("bot.send_alert"):
            _reprotect(bot, 1)
        assert [c[6]["newClientOrderId"] for c in ex.creates("STOP_MARKET")] == \
            [f"{PREFIX}{ROOT}-s"]
        assert ex.plain and ex.plain[0]["id"] == "manual-1", "manual order untouched"


class TestItem8AlertContent:

    def test_failed_alert_has_stop_mark_and_unrealized(self) -> None:
        with patch("bot.send_alert") as alert:
            ex = FakeEx()
            ex.fail["STOP_MARKET"] = [WOULD_TRIGGER]
            ex.fail["close"] = [rejected(-1001)] * 3
            bot = _bot(ex, DONCH_PARAMS)
            bot._place_live_entry(DECISION, 0.005, ROOT, 150.0)
        body = next(c.args[1] for c in alert.call_args_list if "FAILED" in c.args[0])
        assert "Intended stop: 64,000" in body
        assert "Mark: 64,500" in body and "Unrealized: $-2.50" in body

    def test_closed_alert_has_pnl_and_intended_stop(self) -> None:
        ex = FakeEx()
        ex.fail["STOP_MARKET"] = [WOULD_TRIGGER]
        bot = _bot(ex, DONCH_PARAMS)
        with patch("bot.send_alert") as alert:
            bot._place_live_entry(DECISION, 0.005, ROOT, 150.0)
        body = next(c.args[1] for c in alert.call_args_list
                    if c.args[0] == "NO STOP — position closed")
        assert "PnL: $-0.50" in body and "Intended stop: 64,000" in body


class TestRedTeam2:
    """Re-red-team of f15a1b3 (/private/tmp/fc2_rt/test_rt2.py)."""

    def _donch_limit(self):
        p = {**DONCH_PARAMS, "execution": {**DONCH_PARAMS["execution"],
                                           "order_type": "limit"}}
        ex = FakeEx()
        ex.fail["entry"] = [timeout()]
        return _bot(ex, p), ex

    def _tick(self, bot, now):
        bot._unverified_next = 0.0
        with patch("bot.time.time", return_value=now):
            bot._maybe_guard_unverified_entry(150.0)

    def test_Fa_limit_that_lands_after_the_first_read_still_gets_a_stop(self) -> None:
        """The create timed out and the order was NOT on the book at the first
        read (so 'gone' == True). It lands later and fills. Before: no record,
        no stop, no alert, forever on donchian/SOL."""
        import time as _t
        bot, ex = self._donch_limit()
        with patch("bot.send_alert"):
            _enter(bot)
        assert state.get_meta("entry_unverified")
        # lands + fills after the check
        ex.pos = ("long", 0.005, 65000.0)
        ex.entry_filled = 0.005
        with patch("bot.send_alert") as alert:
            self._tick(bot, _t.time() + 30)
        assert any(r["clientAlgoId"].endswith("-s") for r in ex.algo)
        assert any("stop placed" in s.lower() for s in _subjects(alert))
        assert state.get_meta("entry_unverified") == ""

    def test_Fa_record_clears_only_after_the_settle_window(self) -> None:
        import time as _t
        bot, _ex = self._donch_limit()
        with patch("bot.send_alert"):
            _enter(bot)
        self._tick(bot, _t.time() + 10)
        assert state.get_meta("entry_unverified"), "cleared before settling"
        with patch("bot.send_alert"):
            self._tick(bot, _t.time() + 120)
        assert state.get_meta("entry_unverified") == ""

    def _watching_with(self, pos, filled):
        import time as _t
        bot, ex = self._donch_limit()
        with patch("bot.send_alert"):
            _enter(bot)
        ex.pos = pos
        ex.entry_filled = filled
        n = len(ex.calls)
        with patch("bot.send_alert") as alert:
            self._tick(bot, _t.time() + 30)
        return bot, ex, ex.calls[n:], _subjects(alert)

    def test_2_a_manual_same_side_position_is_untouched(self) -> None:
        _b, ex, calls, subjects = self._watching_with(("long", 0.005, 65000.0), 0.0)
        assert not [c for c in calls if c[0] == "create_order"]
        assert ex.pos is not None and ex.algo == []
        assert any("not ours" in s for s in subjects)

    def test_2_unattributable_position_is_untouched(self) -> None:
        _b, _ex, calls, subjects = self._watching_with(("long", 0.005, 65000.0), None)
        assert not [c for c in calls if c[0] == "create_order"]
        assert state.get_meta("entry_unverified"), "keeps watching"
        assert any("cannot be attributed" in s for s in subjects)

    def test_2_larger_position_gets_a_stop_for_the_actual_qty(self) -> None:
        _b, ex, calls, _ = self._watching_with(("long", 0.010, 65000.0), 0.005)
        stops = [c for c in calls if c[0] == "create_order" and c[2] == "STOP_MARKET"]
        assert len(stops) == 1 and stops[0][4] == 0.010
        assert ex.closes() == []

    def test_2_larger_position_is_never_closed_when_no_stop_is_possible(self) -> None:
        import time as _t
        bot, ex = self._donch_limit()
        with patch("bot.send_alert"):
            _enter(bot)
        ex.pos = ("long", 0.010, 65000.0)
        ex.entry_filled = 0.005
        ex.fail["STOP_MARKET"] = [WOULD_TRIGGER]
        with patch("bot.send_alert") as alert:
            self._tick(bot, _t.time() + 30)
        assert ex.closes() == [] and ex.pos is not None
        assert any("NOT closed" in s for s in _subjects(alert))

    def test_3_boot_keeps_the_record_when_the_read_fails(self) -> None:
        bot, ex = self._donch_limit()
        with patch("bot.send_alert"):
            _enter(bot)
        bot._last_position_side = "unknown"
        with patch.object(ex, "fetch_open_orders", side_effect=ccxt.NetworkError("x")):
            bot._clear_no_stop_records_at_boot()
        assert state.get_meta("entry_unverified")
        bot._clear_no_stop_records_at_boot()
        assert state.get_meta("entry_unverified") == ""
