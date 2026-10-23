"""Boot-resume Phase B — the fixes that only matter once adoption is ARMED.

Plan: docs/BOOT_RESUME_PHASE_B_PLAN.md. These run against the REAL per-test
sqlite DB that tests/conftest.py provides, not patched `state.*` calls: B1 and
B2 are about what gets PERSISTED (a close row, a bar timestamp, a guard
counter), so patching the persistence away would test nothing.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from exchange import state
from tests.test_boot_resume import (
    _ARMED_BRACKET,
    _MINIMAL_PARAMS,
    _boot_patches,
    _flat,
    _make_bot,
    _open,
    _principal_patches,
)

@pytest.fixture(autouse=True)
def _lift_arming_interlock():
    """This module exercises the ARMED code path, which production cannot
    reach while bot.ARMING_PREREQS_BUILT is False (see
    tests/test_boot_resume_flag_off.py for the interlock itself)."""
    with patch("bot.ARMING_PREREQS_BUILT", True):
        yield


_ARMED_PARAMS = {**_MINIMAL_PARAMS,
                 "boot_resume": {"enabled": True, "observe_only": False}}


def _seed_db(bracket: dict | None = _ARMED_BRACKET, root: str = "sig-1",
             entry_ts: str | None = None) -> None:
    """An entry fill + its stashed bracket, as _place_live_entry leaves them."""
    state.record_fill(side="long", qty=0.01, price=65000.0, reason="entry",
                      client_order_id_root=root)
    if entry_ts is not None:
        with sqlite3.connect(state.DB_PATH) as c:
            c.execute("UPDATE fills SET ts=? WHERE reason='entry'", (entry_ts,))
    if bracket is not None:
        state.set_meta("active_bracket", json.dumps(bracket))


def _boot(bot, mc, position):
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.fetch_position.return_value = position
    with _boot_patches(), _principal_patches():
        bot.boot()


def _fills() -> list[tuple]:
    with sqlite3.connect(state.DB_PATH) as c:
        return c.execute(
            "SELECT side, reason, price, client_order_id_root FROM fills ORDER BY id"
        ).fetchall()


def _outbox_kinds() -> list[str]:
    with sqlite3.connect(state.DB_PATH) as c:
        return [r[0] for r in c.execute("SELECT kind FROM outbox ORDER BY id")]


# ---------------------------------------------------------------------------
# B1 — an exit between the boot read and the first tick must be recorded
# ---------------------------------------------------------------------------

class TestB1TrackingSeed:

    def test_bracket_fill_before_first_tick_writes_the_close(self):
        """T1. Adopt, then the SL fills before the loop's first tick.

        Teeth: with `_seed_adopted_tracking`'s assignments removed, the first
        tick sees unknown→flat, returns without writing, and this fails (checked
        by hand while building Phase B)."""
        _seed_db()
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        _boot(bot, mc, _open())
        mc.close_position.assert_not_called()
        assert "boot_adopt" in _outbox_kinds()

        # The bracket SL fills while boot() is still finishing.
        mc.fetch_position.return_value = _flat()
        mc.ex.fetch_my_trades.return_value = [
            {"side": "sell", "price": 64500.0, "amount": 0.01}]
        with patch("bot.send_alert"):
            bot._detect_bracket_exit(995.0)

        closes = [f for f in _fills() if f[0] == "close"]
        assert closes == [("close", "bracket_exit", 64500.0, "sig-1")]
        assert "exit" in _outbox_kinds()

    def test_seed_names_the_adopted_position(self):
        _seed_db()
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        _boot(bot, mc, _open(entry_price=65010.0, qty=0.01))
        assert bot._last_position_side == "long"
        assert bot._last_position_entry == 65010.0
        assert bot._last_position_qty == 0.01
        assert bot._last_entry_root == "sig-1"

    def test_an_unchanged_position_on_the_first_tick_is_not_an_exit(self):
        """The seed must not fabricate an exit when nothing happened."""
        _seed_db()
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        _boot(bot, mc, _open())
        bot._detect_bracket_exit(1000.0)
        assert [f for f in _fills() if f[0] == "close"] == []
        assert "exit" not in _outbox_kinds()

    def test_bookkeeping_failure_after_adopt_does_not_abort_boot(self):
        """A sqlite error in record_event must not flatten or abort an adopt."""
        _seed_db()
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        bot.log = MagicMock()
        with patch.object(state, "record_event",
                          side_effect=sqlite3.OperationalError("database is locked")):
            _boot(bot, mc, _open())
        mc.close_position.assert_not_called()
        assert bot._last_position_side == "long"
        assert bot._last_entry_root == "sig-1"
        bot.log.exception.assert_called()

    def test_adopt_branch_never_re_reads_the_root(self):
        """C5 + root=None nit: the root is read ONCE, inside the gate, and the
        adopt branch uses the proven value. A later read failure therefore
        cannot produce an untagged adopted position (the old fallback did)."""
        _seed_db()
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        bot.log = MagicMock()
        real = state.latest_entry_coid_root
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] > 1:      # first call is the gate's own read
                raise sqlite3.OperationalError("disk I/O error")
            return real()

        with patch.object(state, "latest_entry_coid_root", side_effect=flaky):
            _boot(bot, mc, _open())
        mc.close_position.assert_not_called()
        assert calls["n"] == 1
        assert bot._last_position_side == "long"
        assert bot._last_entry_root == "sig-1"

    def test_flatten_path_does_not_seed(self):
        """Refused → flatten must leave the tracking exactly as before."""
        _seed_db(bracket=None)          # no active_bracket → refuse
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        mc.close_position.return_value = {}
        _boot(bot, mc, _open())
        mc.close_position.assert_called_once()
        assert bot._last_position_side == "unknown"
        assert bot._last_entry_root is None


# ---------------------------------------------------------------------------
# B2 — no same-bar re-entry after an adopted position closes
# ---------------------------------------------------------------------------

def _bars(last_closed: str, n: int = 300) -> pd.DataFrame:
    """15m OHLCV, naive-UTC bar-OPEN index (fetch_ohlcv's convention), with
    one extra FORMING bar after `last_closed` that _maybe_enter drops."""
    end = pd.Timestamp(last_closed) + pd.Timedelta(minutes=15)
    idx = pd.date_range(end=end, periods=n, freq="15min")
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                         "Volume": 1.0}, index=idx)


def _run_maybe_enter(bot, mc, last_closed: str):
    """One _maybe_enter pass while flat. Returns the evaluate mock."""
    mc.fetch_position.return_value = _flat()
    mc.fetch_ohlcv.return_value = _bars(last_closed)
    mc.fetch_funding_rate.return_value = 0.0001
    ev = MagicMock(side_effect=RuntimeError("stop after evaluate"))
    with patch("bot.evaluate_for_strategy", ev), \
         patch.object(bot, "_daily_loss_blocks_entry", return_value=False), \
         contextlib.suppress(RuntimeError):
        bot._maybe_enter(1000.0)
    return ev


class TestB2SignalBarSeed:

    def test_adopted_then_flat_in_the_same_bar_does_not_re_evaluate(self):
        """T2. The bar evaluated before the restart must not be re-taken."""
        _seed_db(entry_ts="2026-09-29T09:45:00+00:00")  # pin the fill: a wall-clock ts beat the meta seed after ~10:00 UTC
        state.set_meta("last_entry_bar_ts", "2026-09-29T09:45:00")
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        _boot(bot, mc, _open())
        assert bot._last_signal_ts == pd.Timestamp("2026-09-29 09:45:00")

        ev = _run_maybe_enter(bot, mc, last_closed="2026-09-29 09:45:00")
        ev.assert_not_called()
        mc.fetch_funding_rate.assert_not_called()

    def test_the_next_bar_is_still_evaluated(self):
        """The seed must suppress ONLY bars already seen, not the next one."""
        _seed_db(entry_ts="2026-09-29T09:45:00+00:00")  # pin the fill: a wall-clock ts beat the meta seed after ~10:00 UTC
        state.set_meta("last_entry_bar_ts", "2026-09-29T09:45:00")
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        _boot(bot, mc, _open())
        ev = _run_maybe_enter(bot, mc, last_closed="2026-09-29 10:00:00")
        ev.assert_called_once()

    def test_fallback_from_an_aware_entry_fill_is_naive_and_comparable(self):
        """No meta key (entry predates B2): seed = fill bar − 1 bar, naive UTC.

        Aware-vs-naive would raise TypeError inside _maybe_enter on every tick
        — a crash loop — so this runs a real comparison, not just the seed."""
        _seed_db(entry_ts="2026-09-29T10:05:03.120000+00:00")
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        _boot(bot, mc, _open())
        assert bot._last_signal_ts == pd.Timestamp("2026-09-29 09:45:00")
        assert bot._last_signal_ts.tzinfo is None

        _run_maybe_enter(bot, mc, "2026-09-29 09:45:00").assert_not_called()
        # The fill's own bar (10:00) is NOT suppressed: a plain floor would.
        _run_maybe_enter(bot, mc, "2026-09-29 10:00:00").assert_called_once()

    def test_later_of_meta_and_fallback_wins(self):
        _seed_db(entry_ts="2026-09-29T10:05:03+00:00")      # fallback 09:45
        state.set_meta("last_entry_bar_ts", "2026-09-29T11:30:00")
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        _boot(bot, mc, _open())
        assert bot._last_signal_ts == pd.Timestamp("2026-09-29 11:30:00")

    def test_armed_with_no_seed_available_refuses_and_flattens(self):
        """No meta key and no entry fill: cannot prove the bar → flatten."""
        state.set_meta("active_bracket", json.dumps(_ARMED_BRACKET))
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        mc.close_position.return_value = {}
        _boot(bot, mc, _open())
        mc.close_position.assert_called_once()
        assert "boot_adopt" not in _outbox_kinds()
        assert bot._last_signal_ts is None

    def test_maybe_enter_persists_the_bar_whenever_it_advances(self):
        bot, mc = _make_bot()
        _run_maybe_enter(bot, mc, "2026-09-29 09:45:00")
        # Evaluate raised (our stop), so _last_signal_ts did NOT advance and
        # nothing may be persisted — the bar is re-tried next tick, as today.
        assert state.get_meta("last_entry_bar_ts") is None

        ok = MagicMock(return_value=MagicMock(side=None, reason="none"))
        mc.fetch_ohlcv.return_value = _bars("2026-09-29 10:00:00")
        with patch("bot.evaluate_for_strategy", ok), \
             patch("bot.gate_status", return_value={}), \
             patch.object(bot, "_daily_loss_blocks_entry", return_value=False), \
             contextlib.suppress(Exception):
            bot._maybe_enter(1000.0)
        assert bot._last_signal_ts == pd.Timestamp("2026-09-29 10:00:00")
        assert state.get_meta("last_entry_bar_ts") == "2026-09-29T10:00:00"

    def test_a_failing_persist_never_blocks_evaluation(self):
        bot, mc = _make_bot()
        ok = MagicMock(return_value=MagicMock(side=None, reason="none"))
        mc.fetch_position.return_value = _flat()
        mc.fetch_ohlcv.return_value = _bars("2026-09-29 10:00:00")
        mc.fetch_funding_rate.return_value = 0.0
        with patch("bot.evaluate_for_strategy", ok), \
             patch("bot.gate_status", return_value={}), \
             patch("bot.state.set_meta", side_effect=sqlite3.OperationalError("locked")), \
             patch.object(bot, "_daily_loss_blocks_entry", return_value=False):
            try:
                bot._maybe_enter(1000.0)
            except sqlite3.OperationalError as e:
                raise AssertionError("persist failure escaped _maybe_enter") from e
            except Exception:
                pass
        ok.assert_called_once()
        assert bot._last_signal_ts == pd.Timestamp("2026-09-29 10:00:00")


class TestSignalBarSeedPure:

    def test_none_when_nothing_usable(self):
        from bot_internals import signal_bar_seed
        assert signal_bar_seed(None, None, 900) is None
        assert signal_bar_seed("garbage", "also garbage", 900) is None

    def test_nat_persisted_falls_back_to_the_fill(self):
        from bot_internals import signal_bar_seed
        got = signal_bar_seed("NaT", "2026-09-29T10:07:00+00:00", 900)
        assert got == pd.Timestamp("2026-09-29 09:45:00")

    def test_nat_on_both_is_none(self):
        from bot_internals import signal_bar_seed
        assert signal_bar_seed("NaT", "NaT", 900) is None

    def test_valid_persisted_beats_nat_fill(self):
        from bot_internals import signal_bar_seed
        got = signal_bar_seed("2026-09-29T10:00:00+00:00", "NaT", 900)
        assert got == pd.Timestamp("2026-09-29 10:00:00")

    def test_fill_exactly_on_a_bar_boundary(self):
        from bot_internals import signal_bar_seed
        got = signal_bar_seed(None, "2026-09-29T10:00:00+00:00", 900)
        assert got == pd.Timestamp("2026-09-29 09:45:00")

    def test_4h_bars(self):
        from bot_internals import signal_bar_seed
        got = signal_bar_seed(None, "2026-09-29T08:00:07+00:00", 4 * 3600)
        assert got == pd.Timestamp("2026-09-29 04:00:00")

    def test_non_utc_offset_is_converted(self):
        from bot_internals import signal_bar_seed
        got = signal_bar_seed("2026-09-29T16:45:00+07:00", None, 900)
        assert got == pd.Timestamp("2026-09-29 09:45:00")
        assert got.tzinfo is None


# ---------------------------------------------------------------------------
# D5 — adopt-loop guard: max 3 adopts per position (no time reset; blocker 5)
# ---------------------------------------------------------------------------

def _adopt_log() -> dict | None:
    raw = state.get_meta("boot_adopt_log")
    return json.loads(raw) if raw else None


def _boot_once(params=None, position=None):
    bot, mc = _make_bot(params=dict(params or _ARMED_PARAMS))
    mc.close_position.return_value = {}
    alert = MagicMock()
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.fetch_position.return_value = position or _open()
    with _boot_patches(), _principal_patches(), patch("bot.send_alert", alert):
        bot.boot()
    return bot, mc, alert


def _guard_alerts(alert: MagicMock) -> list:
    return [c for c in alert.call_args_list
            if c.args and "adopt-loop guard" in c.args[0]]


class TestD5AdoptLoopGuard:

    def test_three_adopts_then_the_fourth_boot_flattens_and_alerts(self):
        """T6 per God's D5: MAX 3 adopts (per position since blocker 5), the 4th flattens.
        (The plan's T6 text says 'the 3rd flattens'; God's decision wins.)"""
        _seed_db()
        state.set_meta("last_entry_bar_ts", "2026-09-29T09:45:00")
        for n in (1, 2, 3):
            _bot, mc, alert = _boot_once()
            mc.close_position.assert_not_called()
            assert _adopt_log()["count"] == n
            assert _guard_alerts(alert) == []

        _bot, mc, alert = _boot_once()
        mc.close_position.assert_called_once_with(
            "BTC/USDT:USDT", client_order_id_root="sig-1", close_leg="bf")
        assert len(_guard_alerts(alert)) == 1
        assert _adopt_log()["count"] == 3        # refusal does not bump it
        assert _outbox_kinds().count("boot_adopt") == 3

    def test_time_never_resets_the_count(self):
        """Arming blocker 5: a SLOW crash loop must not re-arm the guard. Under
        the old first-adopt-anchored window this adopted again at 31 min."""
        import time as _t
        _seed_db()
        state.set_meta("last_entry_bar_ts", "2026-09-29T09:45:00")
        state.set_meta("boot_adopt_log", json.dumps(
            {"signal_id": "sig-1", "count": 3, "first_ts": _t.time() - 31 * 60}))
        _bot, mc, alert = _boot_once()
        mc.close_position.assert_called_once()
        assert len(_guard_alerts(alert)) == 1
        assert _adopt_log()["count"] == 3

    def test_inside_the_window_a_full_count_flattens(self):
        import time as _t
        _seed_db()
        state.set_meta("last_entry_bar_ts", "2026-09-29T09:45:00")
        state.set_meta("boot_adopt_log", json.dumps(
            {"signal_id": "sig-1", "count": 3, "first_ts": _t.time() - 29 * 60}))
        _bot, mc, alert = _boot_once()
        mc.close_position.assert_called_once()
        assert len(_guard_alerts(alert)) == 1

    def test_a_different_position_starts_a_fresh_count(self):
        import time as _t
        _seed_db()
        state.set_meta("last_entry_bar_ts", "2026-09-29T09:45:00")
        state.set_meta("boot_adopt_log", json.dumps(
            {"signal_id": "sig-OLD", "count": 3, "first_ts": _t.time() - 60}))
        _bot, mc, _ = _boot_once()
        mc.close_position.assert_not_called()
        assert _adopt_log() == {"signal_id": "sig-1", "count": 1,
                                "first_ts": _adopt_log()["first_ts"]}

    def test_observe_only_never_touches_the_counter(self):
        _seed_db()
        state.set_meta("last_entry_bar_ts", "2026-09-29T09:45:00")
        _bot, mc, _ = _boot_once(params=_MINIMAL_PARAMS)     # observe_only
        mc.close_position.assert_called_once()
        assert _adopt_log() is None

    def test_a_seed_refusal_does_not_spend_budget(self):
        state.set_meta("active_bracket", json.dumps(_ARMED_BRACKET))  # no fill
        _bot, mc, _ = _boot_once()
        mc.close_position.assert_called_once()
        assert _adopt_log() is None

    def test_an_unwritable_counter_flattens(self):
        _seed_db()
        state.set_meta("last_entry_bar_ts", "2026-09-29T09:45:00")
        real = state.set_meta

        def boom(key, value, **kw):
            if key == "boot_adopt_log":
                raise sqlite3.OperationalError("locked")
            return real(key, value, **kw)
        with patch("bot.state.set_meta", side_effect=boom):
            _bot, mc, _ = _boot_once()
        mc.close_position.assert_called_once()
        assert "boot_adopt" not in _outbox_kinds()


class TestAdoptLoopGuardPure:

    def test_fresh(self):
        from bot_internals import adopt_loop_guard
        ok, _why, log = adopt_loop_guard(None, "s", 1000.0)
        assert ok and log == {"signal_id": "s", "count": 1, "first_ts": 1000.0}

    def test_counts_up_to_three_then_refuses(self):
        from bot_internals import adopt_loop_guard
        raw = None
        for n in (1, 2, 3):
            ok, _why, log = adopt_loop_guard(raw, "s", 1000.0 + n)
            assert ok and log["count"] == n
            raw = json.dumps(log)
        ok, why, _log = adopt_loop_guard(raw, "s", 1000.0 + 60)
        assert not ok and "adopt-loop guard" in why

    def test_slow_crash_loop_every_11_minutes_is_still_capped(self):
        """Blocker 5: one crash every 11 min — the window-anchored guard adopted
        forever. Per-position: the 4th boot refuses whatever the spacing."""
        from bot_internals import adopt_loop_guard
        raw, t = None, 0.0
        for n in (1, 2, 3):
            ok, _why, log = adopt_loop_guard(raw, "s", t)
            assert ok and log["count"] == n
            raw, t = json.dumps(log), t + 11 * 60
        for _ in range(10):                      # and stays refused, for hours
            assert adopt_loop_guard(raw, "s", t)[0] is False
            t += 11 * 60

    def test_a_week_later_the_same_position_is_still_refused(self):
        from bot_internals import adopt_loop_guard
        raw = json.dumps({"signal_id": "s", "count": 3, "first_ts": 0.0})
        assert adopt_loop_guard(raw, "s", 7 * 86400.0)[0] is False
        assert adopt_loop_guard(raw, "other", 7 * 86400.0)[0] is True

    def test_garbage_log_is_treated_as_empty(self):
        from bot_internals import adopt_loop_guard
        for raw in ("{", "[]", json.dumps({"count": "x"})):
            ok, _why, log = adopt_loop_guard(raw, "s", 5.0)
            assert ok and log["count"] == 1

    def test_clock_going_backwards_stays_inside_the_window(self):
        """Fail-closed: a backwards clock must not reset a tripped guard."""
        from bot_internals import adopt_loop_guard
        raw = json.dumps({"signal_id": "s", "count": 3, "first_ts": 5000.0})
        ok, _why, _log = adopt_loop_guard(raw, "s", 1000.0)
        assert ok is False


def test_a_locked_db_cannot_delay_entry_more_than_the_cap():
    """A2: another writer holds the DB; the per-bar persist gives up within
    ~0.5 s instead of sqlite's default 10 s, and never raises."""
    import time as _t
    bot, _mc = _make_bot()
    blocker = sqlite3.connect(state.DB_PATH, timeout=0.1, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        t0 = _t.monotonic()
        bot._persist_last_signal_ts(pd.Timestamp("2026-09-29 10:00:00"))
        elapsed = _t.monotonic() - t0
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert elapsed < 2.0, elapsed
    assert state.get_meta("last_entry_bar_ts") is None     # write was abandoned
