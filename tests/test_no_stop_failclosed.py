"""PR #49 review: the no-stop policy (close at market + alert; 4 h latch) must
hold when the bookkeeping itself fails."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

import pytest

from exchange import state

from ._fake_exchange import WOULD_TRIGGER, FakeEx, rejected
from .test_no_stop_close import (
    DECISION,
    DONCH_PARAMS,
    ROOT,
    _bot,
    _fills,
    _open,
    _reprotect,
    _subjects,
)
from .test_no_stop_fixpass import _pending_after_failed_close  # noqa: F401

BOOM = sqlite3.OperationalError("database is locked")


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch("exchange.binance_client.time.sleep"), patch("bot.time.sleep"):
        yield


def _latch_write_fails():
    """set_meta raising for the latch row only (other writes behave)."""
    real = state.set_meta

    def flaky(k, v):
        if k == "no_stop_latch":
            raise BOOM
        return real(k, v)
    return patch("bot.state.set_meta", side_effect=flaky)


def _no_stop_entry(bot, ex, i: int):
    ex.fail["STOP_MARKET"] = [WOULD_TRIGGER] * 3
    with patch("bot.send_alert") as alert:
        bot._place_live_entry(DECISION, 0.005, str(int(ROOT) + i), 150.0)
    return alert


class TestLatchFailsClosed:

    def test_latch_holds_in_memory_when_the_write_fails_and_alert_says_so(self) -> None:
        ex = FakeEx()
        bot = _bot(ex, DONCH_PARAMS)
        _no_stop_entry(bot, ex, 0)
        with _latch_write_fails():
            alert = _no_stop_entry(bot, ex, 1)
            assert bot._latch_active() is True
        body = next(c.args[1] for c in alert.call_args_list if "LATCHED" in c.args[0])
        assert "FAILED" in body and "memory only" in body

    def test_streak_survives_a_failing_write(self) -> None:
        ex = FakeEx()
        bot = _bot(ex, DONCH_PARAMS)
        with _latch_write_fails():
            _no_stop_entry(bot, ex, 0)
            _no_stop_entry(bot, ex, 1)
            assert bot._latch_active() is True

    def test_unreadable_latch_blocks_entries(self) -> None:
        bot = _bot(FakeEx(), DONCH_PARAMS)
        with patch("bot.state.get_meta", side_effect=BOOM):
            assert bot._latch_active() is True
            assert bot._entry_blocked() is True

    def test_released_alert_is_sent_once_even_if_the_reset_write_fails(self) -> None:
        bot = _bot(FakeEx(), DONCH_PARAMS)
        state.set_meta("no_stop_latch", json.dumps(
            {"kind": "no_stop_latch", "streak": 0, "until": 1000.0}))
        with patch("bot.state.set_meta", side_effect=BOOM), \
             patch("bot.time.time", return_value=2000.0), \
             patch("bot.send_alert") as alert:
            for _ in range(5):
                assert bot._latch_active() is False
        assert len(_subjects(alert)) == 1

    def test_a_cleared_pending_close_is_not_resurrected_by_a_failed_clear(self) -> None:
        bot, _ex = _pending_after_failed_close()
        with patch("bot.state.set_meta", side_effect=BOOM):
            bot._clear_pending()
        assert state.get_meta("no_stop_close_pending")      # the write really failed
        assert bot._pending_record() is None


class TestUnknownBookIsNotNoStop:

    def _open_bot(self):
        ex = FakeEx()
        bot = _bot(ex, DONCH_PARAMS)
        ex.pos = ("long", 0.005, 65000.0)
        return bot, ex

    @pytest.mark.parametrize("which", ["algo", "plain"])
    def test_unreadable_book_places_and_closes_nothing(self, which) -> None:
        bot, ex = self._open_bot()
        pos = bot.client.fetch_position(bot.symbol)
        n = len(ex.calls)
        patches = ([patch.object(bot.client, "fetch_algo_orders", return_value=([], False))]
                   if which == "algo" else
                   [patch.object(ex, "fetch_open_orders", side_effect=BOOM)])
        with patches[0], patch("bot.send_alert"):
            handled = bot._protect_or_close_open(pos, ROOT, 1000.0, 2000.0, True, 150.0,
                                                 why="t", record_if_flat=True)
        assert handled is False
        assert not [c for c in ex.calls[n:] if c[0] == "create_order"]
        assert ex.pos is not None

    def test_alert_when_it_persists_but_not_every_tick(self) -> None:
        bot, _ex = self._open_bot()
        pos = bot.client.fetch_position(bot.symbol)
        with patch.object(bot.client, "fetch_algo_orders", return_value=([], False)), \
             patch("bot.send_alert") as alert:
            for _ in range(8):
                bot._protect_or_close_open(pos, ROOT, 1000.0, 2000.0, True, 150.0,
                                           why="t", record_if_flat=True)
        assert len(_subjects(alert)) == 1

    def test_entry_error_with_unreadable_books_is_retried_by_the_guard(self) -> None:
        bot, ex = self._open_bot()
        bad = patch.object(bot.client, "fetch_algo_orders", return_value=([], False))
        with bad, patch("bot.send_alert"):
            assert bot._entry_error_left_position(DECISION, ROOT, 150.0, RuntimeError("x"))
        u = json.loads(state.get_meta("entry_unverified"))
        assert u["retry_protect"] is True and ex.algo == []
        bot._unverified_next = 0.0
        with patch("bot.send_alert"):
            bot._maybe_guard_unverified_entry(150.0)       # books readable now
        assert any(r["clientAlgoId"].endswith("-s") for r in ex.algo)
        assert state.get_meta("entry_unverified") == ""
        assert ex.closes() == []


class TestBookkeepingNeverBlocksTheClose:

    def test_reprotect_closes_at_the_cap_even_if_the_counter_cannot_persist(self) -> None:
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex, sl_place_failures=2)
        ex.fail["STOP_MARKET"] = [rejected(-1001)] * 5
        real = state.set_meta

        def flaky(k, v):
            if k == "active_bracket":
                raise BOOM
            return real(k, v)
        with patch("bot.state.set_meta", side_effect=flaky), patch("bot.send_alert"):
            _reprotect(bot, 1)
        assert ex.pos is None and len(ex.closes()) == 1

    def test_in_memory_counter_holds_the_cap_when_nothing_persists(self) -> None:
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex)
        ex.fail["STOP_MARKET"] = [rejected(-1001)] * 10
        with patch("bot.state.set_meta", side_effect=BOOM), patch("bot.send_alert"):
            for _ in range(3):
                _reprotect(bot, 1)
        assert ex.pos is None and len(ex.closes()) == 1


class TestAlertsAreTruthful:

    @pytest.mark.parametrize("observe,expect", [(True, "No automatic TP retry"),
                                                (False, "Reprotect will retry")])
    def test_tp_alert_respects_observe_only(self, observe, expect) -> None:
        ex = FakeEx()
        params = {**DONCH_PARAMS, "reprotect": {"enabled": True, "observe_only": observe}}
        bot = _bot(ex, params)
        with patch("bot.send_alert") as alert:
            bot._alert_tp_error({"tp_error": "boom"}, "long", ROOT)
        assert expect in alert.call_args.args[1]


class TestPendingRetryDoesNotAbandonANakedPosition:

    def _lose_entry_row(self):
        bot, ex = _pending_after_failed_close()
        with sqlite3.connect(state.DB_PATH) as c:
            c.execute("DELETE FROM fills")
        state.record_fill(side="long", qty=0.005, price=60000.0, reason="entry",
                          client_order_id_root="1700000000000")   # the PREVIOUS trade
        ex.fail["close"] = []
        return bot, ex

    def test_missing_entry_row_closes_with_a_fresh_id_and_alerts(self) -> None:
        bot, ex = self._lose_entry_row()
        with patch("bot.send_alert") as alert:
            bot._maybe_finish_no_stop_close(150.0)
        assert ex.pos is None
        coid = ex.closes()[-1][6]["newClientOrderId"]
        assert coid.endswith("-ns") and ROOT not in coid
        assert any("entry record is missing" in s for s in _subjects(alert))
        assert not any("ABANDONED" in s for s in _subjects(alert))

    def test_unreadable_ledger_closes_too(self) -> None:
        bot, ex = _pending_after_failed_close()
        ex.fail["close"] = []
        with patch("bot.state.latest_entry_coid_root", side_effect=BOOM), \
             patch.object(bot, "_entry_row_exists", return_value=None), \
             patch("bot.send_alert"):
            bot._maybe_finish_no_stop_close(150.0)
        assert ex.pos is None

    def test_a_provably_newer_entry_is_still_left_alone(self) -> None:
        bot, ex = _pending_after_failed_close()
        state.record_fill(side="long", qty=0.005, price=65000.0, reason="entry",
                          client_order_id_root="1792000000000")
        n = len(ex.calls)
        with patch("bot.send_alert"):
            bot._maybe_finish_no_stop_close(150.0)
        assert all(c[0] == "fetch_positions" for c in ex.calls[n:])
        assert ex.pos is not None


class TestUnverifiedRecordSurvivesAFailedWrite:

    def test_guard_still_protects_on_the_next_tick_when_the_write_failed(self) -> None:
        ex = FakeEx()
        bot = _bot(ex, DONCH_PARAMS)
        ex.pos = ("long", 0.005, 65000.0)
        real = state.set_meta

        def flaky(k, v):
            if k == "entry_unverified":
                raise BOOM
            return real(k, v)
        with patch.object(bot.client, "fetch_algo_orders", return_value=([], False)), \
             patch("bot.state.set_meta", side_effect=flaky), patch("bot.send_alert"):
            assert bot._entry_error_left_position(DECISION, ROOT, 150.0, RuntimeError("x"))
            assert not state.get_meta("entry_unverified")      # nothing persisted
            assert bot._entry_blocked() is True                # memory holds it
        bot._unverified_next = 0.0
        with patch("bot.state.set_meta", side_effect=flaky), patch("bot.send_alert"):
            bot._maybe_guard_unverified_entry(150.0)           # books readable now
        assert any(r["clientAlgoId"].endswith("-s") for r in ex.algo)
        assert bot._unverified_record() is None

    def test_a_failed_clear_does_not_resurrect_the_record(self) -> None:
        bot = _bot(FakeEx(), DONCH_PARAMS)
        bot._set_unverified({"kind": "entry_unverified", "root": ROOT, "side": "long"})
        with patch("bot.state.set_meta", side_effect=BOOM):
            bot._clear_unverified()
        assert state.get_meta("entry_unverified")
        assert bot._unverified_record() is None


class TestUnknownBookEscalates:

    def _setup(self):
        ex = FakeEx()
        bot = _bot(ex, DONCH_PARAMS)
        ex.pos = ("long", 0.005, 65000.0)
        pos = bot.client.fetch_position(bot.symbol)
        return bot, ex, pos

    def _tick(self, bot, pos, t, now, readable=False):
        t[0] = now
        books = ([], True) if readable else ([], False)
        with patch.object(bot.client, "fetch_algo_orders", return_value=books), \
             patch("bot.send_alert") as alert:
            handled = bot._protect_or_close_open(pos, ROOT, 1000.0, 2000.0, True, 150.0,
                                                 why="t", record_if_flat=True)
        return handled, alert

    def test_no_action_before_300s_then_exactly_one_close(self) -> None:
        bot, ex, pos = self._setup()
        t = [1000.0]
        with patch("bot.time.monotonic", side_effect=lambda: t[0]):
            for now in (1000.0, 1100.0, 1299.0):
                handled, _a = self._tick(bot, pos, t, now)
                assert handled is False and ex.closes() == []
            handled, alert = self._tick(bot, pos, t, 1300.0)
            assert handled is True and len(ex.closes()) == 1 and ex.pos is None
            assert ex.closes()[0][6]["newClientOrderId"].endswith("-ns")
            assert any("could not be verified for 300s" in c.args[1]
                       for c in alert.call_args_list)
        assert not [c for c in ex.creates("STOP_MARKET")]

    def test_a_successful_read_resets_the_timer(self) -> None:
        bot, ex, pos = self._setup()
        t = [1000.0]
        with patch("bot.time.monotonic", side_effect=lambda: t[0]):
            self._tick(bot, pos, t, 1000.0)
            self._tick(bot, pos, t, 1200.0)
            # readable (stop absent => a stop is placed, position stays open)
            self._tick(bot, pos, t, 1250.0, readable=True)
            assert ex.pos is not None
            handled, _a = self._tick(bot, pos, t, 1400.0)       # 150 s into a new streak
        assert handled is False and ex.closes() == []


class TestLatchOperatorControls:

    def test_deleting_the_persisted_latch_row_releases_it(self) -> None:
        ex = FakeEx()
        bot = _bot(ex, DONCH_PARAMS)
        _no_stop_entry(bot, ex, 0)
        _no_stop_entry(bot, ex, 1)
        assert bot._latch_active() is True
        state.set_meta("no_stop_latch", "")           # "delete meta to release early"
        assert bot._latch_active() is False

    def test_memory_latch_wins_when_the_persist_failed(self) -> None:
        ex = FakeEx()
        bot = _bot(ex, DONCH_PARAMS)
        with _latch_write_fails():
            _no_stop_entry(bot, ex, 0)
            _no_stop_entry(bot, ex, 1)
        assert bot._latch_active() is True            # row absent, but never persisted

    def test_unreadable_latch_alerts_once_per_hour(self) -> None:
        bot = _bot(FakeEx(), DONCH_PARAMS)
        t = [10_000.0]
        with patch("bot.state.get_meta", side_effect=BOOM), \
             patch("bot.time.time", side_effect=lambda: t[0]), \
             patch("bot.send_alert") as alert:
            for _ in range(5):
                assert bot._latch_active() is True
                t[0] += 60
            t[0] += 3600
            assert bot._latch_active() is True
        assert len(_subjects(alert)) == 2
