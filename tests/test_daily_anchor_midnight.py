"""Daily anchor = live equity snapshotted at EVERY UTC midnight, flat or not.

God's decision (2026-10-13): "snapshot the day's start equity at UTC midnight
every day, flat or not, using live equity incl. open P&L; the daily-loss breaker
then counts overnight-held losses; the monitor sees open-trade drawdown."

Before: the anchor was set lazily at the first FLAT `_maybe_enter` after
midnight. A position carried over midnight that stopped out at 03:00 anchored
AFTER the loss, so the breaker never counted it (donchian holds for days).
The monitor, during a hold, read the entry fill's equity_after (no unrealized
PnL) — sol sat frozen at 97.39 from 10-07.

These tests drive the REAL `Bot.loop()` one tick at a time (exits/reprotect/
consolidate stubbed out; `_maybe_enter` and the breaker real), so they pin
where the snapshot happens, not just that a helper exists.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import monitor
from risk import ceilings

CFG = {**monitor.DEFAULTS}
LOSS = ceilings().MAX_DAILY_LOSS_PCT  # synthetic fixture ceiling (conftest); tests derive from it


def _fixed_datetime(fixed_now):
    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now
    return _FakeDatetime


class _Log:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def _rec(self, lvl, msg, *a, **k):
        self.lines.append((lvl, msg % a if a else msg))

    def info(self, m, *a, **k): self._rec("info", m, *a)
    def warning(self, m, *a, **k): self._rec("warning", m, *a)
    def error(self, m, *a, **k): self._rec("error", m, *a)
    def exception(self, m, *a, **k): self._rec("error", m, *a)
    def debug(self, m, *a, **k): pass


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "state.db"
    with patch("exchange.state.DB_PATH", p):
        from exchange import state
        state.init_db()
        yield p


def _ready_principal(seed: float = 1000.0) -> None:
    """Initialise Part C so the breaker's fail-safe gate is open."""
    from exchange import principal, state
    state.set_meta(principal.META_SOURCE, "income_backfill")
    state.set_meta("principal_base", "0.0")
    state.principal_ledger_upsert([(1, "TRANSFER", seed, "USDT",
                                    int(datetime(2026, 6, 1, tzinfo=UTC).timestamp() * 1000))])


def _transfer(tran_id: int, amount: float, when: datetime) -> None:
    from exchange import state
    state.principal_ledger_upsert([(tran_id, "TRANSFER", amount, "USDT",
                                    int(when.timestamp() * 1000))])


def _make_bot():
    import bot as bot_mod
    b = object.__new__(bot_mod.Bot)
    b.log = _Log()
    b._daily_loss_blocked = False
    b._stopped = False
    b._last_signal_ts = None
    b._last_live_equity_write = None
    b.poll_s = 60.0
    b.instance = "test"
    b.symbol = "BTC/USDT:USDT"
    b.entry_tf = "4h"
    b.bar_seconds = 4 * 3600
    b.dry_run = False
    b.client = MagicMock()
    b.client.fetch_ohlcv.return_value = pd.DataFrame()   # len<250 → returns
    for name in ("_heartbeat", "_maybe_reconcile_principal", "_detect_bracket_exit",
                 "_maybe_time_stop", "_maybe_channel_exit", "_maybe_breakeven",
                 "_maybe_reprotect", "_maybe_push_consolidate",
                 "_maybe_finish_no_stop_close", "_maybe_guard_unverified_entry"):
        setattr(b, name, lambda *a, **k: None)
    b._check_kill_switch = lambda equity: False
    from tests._bot_seed import seed_no_stop_state
    seed_no_stop_state(b, bot_mod.Bot)
    return b, bot_mod


def _tick(b, bot_mod, now: datetime, equity: float, side: str,
          wallet: float | None = None, pos: SimpleNamespace | None = None) -> bool:
    """Run exactly one loop() iteration at `now`. Returns True if _maybe_enter
    reached signal evaluation (i.e. the breaker did NOT block)."""
    b._stopped = False
    b.client.fetch_equity_usdt.return_value = equity
    b.client.last_wallet_usdt = wallet      # None = component unavailable
    b.client.fetch_position.return_value = pos or SimpleNamespace(side=side)
    before = b.client.fetch_ohlcv.call_count

    def _sleep(_s):
        b._stopped = True

    with patch.object(bot_mod, "datetime", _fixed_datetime(now)), \
         patch.object(bot_mod, "halt_source", lambda inst: None), \
         patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod.time, "sleep", _sleep):
        assert b.loop() == 0
    return b.client.fetch_ohlcv.call_count > before


D1 = datetime(2026, 10, 7, 23, 0, tzinfo=UTC)
MID = datetime(2026, 10, 8, 0, 0, 30, tzinfo=UTC)
STOP = datetime(2026, 10, 8, 3, 0, tzinfo=UTC)


# --------------------------------------------------------------------------
# (a) FAILING ON BASE: overnight hold, stop-out at 03:00 → breaker must count it
# --------------------------------------------------------------------------

def test_overnight_hold_stopout_trips_breaker(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    assert _tick(b, bot_mod, D1, 1000.0, "flat")          # day-1 anchor, enters
    assert _tick(b, bot_mod, MID, 1000.0, "long") is False  # held over midnight
    # Midnight tick must snapshot the new day even though a position is open.
    assert state.get_meta("daily_anchor_date") == "2026-10-08"
    assert state.get_float("daily_anchor_equity") == pytest.approx(1000.0)
    # 03:00: stopped out at -(LOSS+0.5)% → flat; the breaker must block.
    eq = 1000.0 * (1 - (LOSS + 0.5) / 100)
    assert _tick(b, bot_mod, STOP, eq, "flat") is False
    assert state.get_meta("daily_loss_breaker_date") == "2026-10-08"


def test_midnight_snapshot_includes_unrealized(db):
    """Loss accrued BEFORE midnight is yesterday's; only post-midnight counts."""
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    _tick(b, bot_mod, D1, 1000.0, "flat")
    _tick(b, bot_mod, MID, 960.0, "long")                    # -4% unrealized at 00:00
    assert state.get_float("daily_anchor_equity") == pytest.approx(960.0)
    # Stops out at 940: -2.08% vs 960 → under the ceiling → entries allowed.
    assert _tick(b, bot_mod, STOP, 940.0, "flat") is True


def test_flat_day_does_not_trip(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    for now, eq in ((D1, 1000.0), (MID, 1000.0), (STOP, 995.0),
                    (datetime(2026, 10, 8, 20, 0, tzinfo=UTC), 990.0)):
        assert _tick(b, bot_mod, now, eq, "flat") is True


def test_breaker_resets_next_midnight_even_while_holding(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    _tick(b, bot_mod, D1, 1000.0, "flat")
    _tick(b, bot_mod, MID, 1000.0, "flat")
    assert _tick(b, bot_mod, STOP, 900.0, "flat") is False   # tripped day 2
    _tick(b, bot_mod, datetime(2026, 10, 9, 0, 0, 30, tzinfo=UTC), 900.0, "long")
    assert state.get_meta("daily_anchor_date") == "2026-10-09"
    assert _tick(b, bot_mod, datetime(2026, 10, 9, 5, 0, tzinfo=UTC), 890.0, "flat") is True


# --------------------------------------------------------------------------
# Transfers during a hold: no false drawdown (breaker) — semantics unchanged
# --------------------------------------------------------------------------

def test_deposit_during_hold_does_not_mask_loss(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    _tick(b, bot_mod, MID, 1000.0, "long")
    _transfer(2, 100.0, datetime(2026, 10, 8, 1, 0, tzinfo=UTC))
    # Real trading loss of (LOSS+0.5)% hidden by a +100 deposit → still trips.
    eq = 1000.0 * (1 - (LOSS + 0.5) / 100) + 100.0
    assert _tick(b, bot_mod, STOP, eq, "flat") is False


def test_deposit_during_hold_no_false_trip_in_monitor_or_breaker(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    _tick(b, bot_mod, MID, 1000.0, "long")
    _transfer(2, 100.0, datetime(2026, 10, 8, 1, 0, tzinfo=UTC))
    assert _tick(b, bot_mod, STOP, 1100.0, "flat") is True   # flat P&L day
    cur, anchor = monitor._equity_from_db(db, now=STOP)
    assert cur == pytest.approx(1100.0) and anchor == pytest.approx(1100.0)


def test_withdrawal_during_hold_does_not_false_trip(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    _tick(b, bot_mod, MID, 1000.0, "long")
    _transfer(2, -200.0, datetime(2026, 10, 8, 1, 0, tzinfo=UTC))
    assert _tick(b, bot_mod, STOP, 800.0, "flat") is True    # -200 is a withdrawal
    cur, anchor = monitor._equity_from_db(db, now=STOP)
    assert cur == pytest.approx(800.0) and anchor == pytest.approx(800.0)


def test_withdrawal_does_not_mask_real_loss(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    _tick(b, bot_mod, MID, 1000.0, "long")
    _transfer(2, -200.0, datetime(2026, 10, 8, 1, 0, tzinfo=UTC))
    eq = 800.0 - 1000.0 * (LOSS + 0.5) / 100
    assert _tick(b, bot_mod, STOP, eq, "flat") is False


# --------------------------------------------------------------------------
# Bot down over midnight / first-ever run / uninitialised principal
# --------------------------------------------------------------------------

def test_bot_down_over_midnight_snapshots_late_and_logs(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    _tick(b, bot_mod, D1, 1000.0, "long")
    late = datetime(2026, 10, 8, 2, 0, tzinfo=UTC)
    _tick(b, bot_mod, late, 970.0, "long")
    assert state.get_meta("daily_anchor_date") == "2026-10-08"
    assert state.get_float("daily_anchor_equity") == pytest.approx(970.0)
    assert any(lvl == "warning" and "LATE" in msg for lvl, msg in b.log.lines)
    with sqlite3.connect(db) as c:
        kinds = [r[0] for r in c.execute("SELECT kind FROM events")]
    assert "daily_anchor_late" in kinds


def test_on_time_snapshot_is_not_flagged_late(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    _tick(b, bot_mod, D1, 1000.0, "flat")
    _tick(b, bot_mod, MID, 1000.0, "long")
    assert not any("LATE" in msg for _, msg in b.log.lines)


def test_first_ever_run_anchors_on_first_tick(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    assert state.get_meta("daily_anchor_date") is None
    _tick(b, bot_mod, STOP, 1000.0, "long")
    assert state.get_meta("daily_anchor_date") == "2026-10-08"
    assert state.get_float("daily_anchor_equity") == pytest.approx(1000.0)
    assert float(state.get_meta("daily_anchor_principal_sum")) == pytest.approx(1000.0)


def test_uninitialised_principal_does_not_snapshot(db):
    b, bot_mod = _make_bot()
    from exchange import state
    _tick(b, bot_mod, MID, 1000.0, "long")
    assert state.get_meta("daily_anchor_date") is None
    assert state.get_meta("daily_anchor_principal_sum") is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -5.0])
def test_non_finite_or_nonpositive_equity_is_never_anchored(db, bad):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    _tick(b, bot_mod, MID, bad, "long")
    assert state.get_meta("daily_anchor_date") is None


def test_snapshot_failure_does_not_break_the_tick(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    calls = []
    b._maybe_reprotect = lambda eq: calls.append(eq)
    with patch.object(bot_mod.state, "set_meta", side_effect=sqlite3.OperationalError("locked")):
        _tick(b, bot_mod, MID, 1000.0, "long")
    assert calls == [1000.0]   # reprotect still ran on this tick


# --------------------------------------------------------------------------
# (b) FAILING ON BASE: monitor during a multi-day hold reflects unrealized PnL
# --------------------------------------------------------------------------

def test_monitor_sees_open_position_drawdown_during_multiday_hold(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    _tick(b, bot_mod, D1, 1000.0, "flat")
    state.record_fill(side="buy", qty=1.0, price=100.0, reason="entry",
                      equity_after=1000.0)
    # Held for days; unrealized -12% at 10-10 06:00.
    now = datetime(2026, 10, 10, 6, 0, tzinfo=UTC)
    _tick(b, bot_mod, now, 880.0, "long")
    cur, anchor = monitor._equity_from_db(db, now=now + dt.timedelta(minutes=3))
    assert anchor == pytest.approx(1000.0)
    assert cur == pytest.approx(880.0)
    assert monitor._equity_band((1 - cur / anchor) * 100, CFG) == "alert"


def test_monitor_live_equity_written_inside_rollover_grace(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    _tick(b, bot_mod, MID, 930.0, "long")
    cur, _ = monitor._equity_from_db(db, now=MID + dt.timedelta(minutes=1))
    assert cur == pytest.approx(930.0)


def test_monitor_ignores_stale_live_equity_and_falls_back(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    _tick(b, bot_mod, MID, 930.0, "long")       # live + anchor dated 10-08
    later = datetime(2026, 10, 9, 6, 0, tzinfo=UTC)   # bot dead since
    state.record_fill(side="close", qty=1.0, price=1.0, reason="x", equity_after=950.0)
    cur, _ = monitor._equity_from_db(db, now=later)
    assert cur == pytest.approx(950.0)          # fill fallback, not the stale 930


@pytest.mark.parametrize("raw", ["garbage", json.dumps({"ts": "x", "equity": 1}),
                                 json.dumps({"ts": 1e12, "equity": float("nan"), "ledger_sum": 1000.0, "transfer_pending": False}),
                                 json.dumps([1, 2])])
def test_monitor_malformed_live_equity_falls_back(db, raw):
    _ready_principal()
    from exchange import state
    state.set_meta("live_equity", raw)
    state.record_fill(side="close", qty=1.0, price=1.0, reason="x", equity_after=950.0)
    cur, _ = monitor._equity_from_db(db, now=STOP)
    assert cur == pytest.approx(950.0)


def test_live_equity_write_is_rate_limited(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    _tick(b, bot_mod, MID, 1000.0, "long")
    _tick(b, bot_mod, MID + dt.timedelta(seconds=5), 990.0, "long")
    assert json.loads(state.get_meta("live_equity"))["equity"] == pytest.approx(1000.0)
    _tick(b, bot_mod, MID + dt.timedelta(seconds=65), 980.0, "long")
    assert json.loads(state.get_meta("live_equity"))["equity"] == pytest.approx(980.0)


# --------------------------------------------------------------------------
# Panel fix-pass (2026-10-13): unreconciled transfers, hysteresis, equity > 0
# --------------------------------------------------------------------------

def _epoch(d: datetime) -> float:
    return d.timestamp()


def _replay_0828(db, *, detector: bool, side: str = "long", wallet: bool = True,
                 lag_min: float = 0.0, fill_at_deposit: bool = False,
                 end_hm: tuple[int, int] = (3, 15)) -> list[str]:
    """v1, 2026-08-28: +$21 deposit at 02:28:45, hourly reconcile lands it at
    02:59:51. Equity 137.70 vs P 162.80 (-15.4%, alert) the whole time in
    trading terms. Bot ticks every 60s; monitor cron every 5 min. Returns the
    subjects of every mail the monitor sent.

    lag_min: the income API only shows the deposit this long after it lands.
    fill_at_deposit: a bot fill is recorded on the same tick the deposit shows.
    """
    from exchange import principal, state
    b, bot_mod = _make_bot()
    _ready_principal(seed=162.80)
    b._last_position_side = side
    dep = datetime(2026, 8, 28, 2, 28, 45, tzinfo=UTC)
    visible = dep + dt.timedelta(minutes=lag_min)
    b._principal_reconcile_interval_s = 3600.0
    b._principal_retry_interval_s = 60.0
    b._last_principal_reconcile_ts = _epoch(datetime(2026, 8, 28, 1, 59, 51, tzinfo=UTC))
    del b._maybe_reconcile_principal            # use the REAL reconcile cadence
    if not detector:
        b.TRANSFER_JUMP_FRAC = float("inf")
    clock = {"t": 0.0}

    def _reconcile(client, log=None):
        if clock["t"] >= _epoch(visible):
            _transfer(2, 21.0, dep)
        return {}

    sent: list[str] = []
    mon_state: dict = {"alerts": {}, "equity_bands": {"v1": "alert"}}
    monitor._stamp_alert(mon_state, "equity_alert:v1")   # reminder not due
    t = datetime(2026, 8, 28, 2, 20, tzinfo=UTC)
    end = datetime(2026, 8, 28, *end_hm, tzinfo=UTC)
    filled = False
    with patch.object(principal, "reconcile_recent", _reconcile):
        while t <= end:
            clock["t"] = _epoch(t)
            eq = 137.70 + (21.0 if t >= dep else 0.0)
            if fill_at_deposit and t >= dep and not filled:
                state.record_fill(side="close", qty=0.0, price=1.0, reason="x",
                                  equity_after=eq)
                filled = True
            w = (eq + 12.3 if side != "flat" else eq) if wallet else None
            with patch.object(bot_mod.time, "time", lambda: clock["t"]):
                _tick(b, bot_mod, t, eq, side, wallet=w)
            if t.minute % 5 == 0:
                real = monitor._equity_from_db
                with patch.object(monitor, "_equity_from_db",
                                  lambda p, g=10, now=None, _t=t, _r=real: _r(p, g, _t)), \
                     patch.object(monitor, "send_alert",
                                  lambda subj, body, **k: sent.append(subj) or True):
                    monitor._check_equity("v1", db, CFG, mon_state)
            t += dt.timedelta(minutes=1)
    assert state.principal_ledger_sum("USDT") == pytest.approx(183.80)
    return sent


def test_0828_replay_without_detector_sends_two_spurious_mails(db):
    sent = _replay_0828(db, detector=False)
    assert len(sent) == 2, sent          # false RECOVERED, then fresh DROP
    assert sent[0].startswith("EQUITY RECOVERED")


def test_0828_replay_with_detector_sends_nothing(db):
    assert _replay_0828(db, detector=True) == []


def test_jump_with_position_change_is_not_a_transfer(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    b._last_position_side, b._last_position_qty = "long", 1.0
    _tick(b, bot_mod, MID, 1000.0, "long")
    b._last_position_side, b._last_position_qty = "flat", 0.0   # exit last tick
    _tick(b, bot_mod, MID + dt.timedelta(seconds=65), 940.0, "flat")
    assert json.loads(state.get_meta("live_equity"))["transfer_pending"] is False


def test_jump_with_new_fill_is_not_a_transfer(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    _tick(b, bot_mod, MID, 1000.0, "long")
    state.record_fill(side="close", qty=1.0, price=1.0, reason="x", equity_after=940.0)
    _tick(b, bot_mod, MID + dt.timedelta(seconds=65), 940.0, "long")
    assert json.loads(state.get_meta("live_equity"))["transfer_pending"] is False


def test_suspect_jump_forces_reconcile_then_clears_on_ledger_move(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    b._last_principal_reconcile_ts = 1e18     # "just reconciled" far in the future
    _tick(b, bot_mod, MID, 1000.0, "long", wallet=1000.0)
    b._last_principal_reconcile_ts = 0.0 + 1.0
    _tick(b, bot_mod, MID + dt.timedelta(seconds=5), 1100.0, "long", wallet=1100.0)
    assert json.loads(state.get_meta("live_equity"))["transfer_pending"] is True
    assert b._last_principal_reconcile_ts == 0.0          # next tick reconciles
    cur, _ = monitor._equity_from_db(db, now=MID + dt.timedelta(minutes=1))
    assert cur == pytest.approx(1000.0)                   # anchor, not raw 1100
    _transfer(2, 100.0, MID)
    _tick(b, bot_mod, MID + dt.timedelta(seconds=10), 1100.0, "long", wallet=1100.0)
    assert json.loads(state.get_meta("live_equity"))["transfer_pending"] is False
    cur, anchor = monitor._equity_from_db(db, now=MID + dt.timedelta(minutes=1))
    assert cur == pytest.approx(1100.0) and anchor == pytest.approx(1100.0)


def test_suspect_expires_when_ledger_never_moves(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    _tick(b, bot_mod, MID, 1000.0, "long", wallet=1000.0)
    # wallet move never followed by a ledger row (e.g. income API never shows it)
    _tick(b, bot_mod, MID + dt.timedelta(seconds=60), 1030.0, "long", wallet=1030.0)
    assert json.loads(state.get_meta("live_equity"))["transfer_pending"] is True
    _tick(b, bot_mod, MID + dt.timedelta(seconds=420), 1030.0, "long", wallet=1030.0)
    assert json.loads(state.get_meta("live_equity"))["transfer_pending"] is False


def test_monitor_ignores_sample_taken_against_older_ledger(db):
    _ready_principal()
    from exchange import state
    state.set_meta("live_equity", json.dumps(
        {"ts": _epoch(STOP), "equity": 900.0, "ledger_sum": 1000.0,
         "transfer_pending": False}))
    state.set_meta("daily_anchor_date", "2026-10-08")
    state.set_float("daily_anchor_equity", 1000.0)
    state.set_float("daily_anchor_principal_sum", 1000.0)
    assert monitor._equity_from_db(db, now=STOP)[0] == pytest.approx(900.0)
    _transfer(2, 50.0, STOP)                    # reconcile landed after the sample
    # Doubtful sample → min(adjusted anchor 1050, live 900): never hide a drop.
    # RESIDUAL: if the transfer truly post-dates the sample this reads 50 low
    # for at most one bot write interval (60 s).
    assert monitor._equity_from_db(db, now=STOP)[0] == pytest.approx(900.0)


@pytest.mark.parametrize("eq", [0.0, -1.0])
def test_zero_equity_never_written_or_read(db, eq):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    _tick(b, bot_mod, MID, eq, "long")
    assert state.get_meta("live_equity") is None
    state.set_meta("live_equity", json.dumps(
        {"ts": _epoch(STOP), "equity": eq, "ledger_sum": 1000.0,
         "transfer_pending": False}))
    state.record_fill(side="close", qty=1.0, price=1.0, reason="x", equity_after=950.0)
    assert monitor._equity_from_db(db, now=STOP)[0] == pytest.approx(950.0)


@pytest.mark.parametrize("prev,drop,expected", [
    (None, 5.0, "warn"), ("ok", 4.9, "ok"),
    ("warn", 4.5, "warn"), ("warn", 4.01, "warn"), ("warn", 3.99, "ok"),
    ("alert", 9.5, "alert"), ("alert", 8.99, "warn"), ("alert", 3.5, "ok"),
    ("warn", 10.0, "alert"), ("ok", 10.0, "alert"),
])
def test_band_hysteresis(prev, drop, expected):
    assert monitor._equity_band(drop, CFG, prev) == expected


def test_hold_hovering_at_warn_line_mails_once(tmp_path):
    sent: list[str] = []
    mon_state: dict = {"alerts": {}}
    for drop in (5.2, 4.8, 5.1, 4.6, 5.3, 4.9):
        with patch.object(monitor, "_equity_from_db",
                          lambda p, g=10, _d=drop: (100.0 - _d, 100.0)), \
             patch.object(monitor, "send_alert",
                          lambda subj, body, **k: sent.append(subj) or True):
            monitor._check_equity("donchian", tmp_path / "x.db", CFG, mon_state)
    assert len(sent) == 1 and sent[0].startswith("EQUITY WARN")


def test_0828_deposit_while_flat_no_wallet_read_sends_nothing(db):
    assert _replay_0828(db, detector=True, side="flat", wallet=False) == []


def test_0828_deposit_while_flat_with_wallet_sends_nothing(db):
    assert _replay_0828(db, detector=True, side="flat") == []


def test_residual_income_api_lag_over_5_min(db):
    """RESIDUAL (documented): if the income API shows the deposit >5 min late,
    the suspect window expires first and the monitor reverts to the 7b4d6d5
    behaviour for this transfer — a false RECOVERED, then a fresh DROP once
    the next hourly reconcile lands it."""
    sent = _replay_0828(db, detector=True, lag_min=6.0, end_hm=(3, 45))
    assert [x.split(" ")[1] for x in sent] == ["RECOVERED:", "DROP"]


def test_residual_deposit_on_same_tick_as_fill(db):
    """RESIDUAL (documented): a fill on the deposit's tick explains the jump
    away, so it is not flagged; same two mails as without the detector."""
    sent = _replay_0828(db, detector=True, fill_at_deposit=True)
    assert [x.split(" ")[1] for x in sent] == ["RECOVERED:", "DROP"]


def _crash_run(db, equities: dict[int, float], *, wallet_mode: str):
    """Ticks every 60s from 01:00 for 15 min, monitor every 5 min. equities:
    minute -> live equity (carried forward). wallet_mode: 'const' (pure price
    move), 'none' (no wallet read), 'tracks' (wallet moves with equity, e.g. a
    stop-out realized on the exchange before the bot records the fill — the
    detector flags it). Returns [(minute, subject)]."""
    b, bot_mod = _make_bot()
    _ready_principal(seed=1000.0)
    b._last_position_side, b._last_position_qty = "long", 1.0
    sent: list[tuple[int, str]] = []
    mon_state: dict = {"alerts": {}}
    eq = 1000.0
    t0 = datetime(2026, 10, 8, 1, 0, tzinfo=UTC)
    for m in range(16):
        eq = equities.get(m, eq)
        w = {"const": 1000.0, "none": None, "tracks": eq}[wallet_mode]
        t = t0 + dt.timedelta(minutes=m)
        _tick(b, bot_mod, t, eq, "long", wallet=w)
        if m % 5 == 0:
            real = monitor._equity_from_db
            with patch.object(monitor, "_equity_from_db",
                              lambda p, g=10, now=None, _t=t, _r=real: _r(p, g, _t)), \
                 patch.object(monitor, "send_alert",
                              lambda subj, body, _m=m, **k: sent.append((_m, subj)) or True):
                monitor._check_equity("v1", db, CFG, mon_state)
    return sent


@pytest.mark.parametrize("mode", ["const", "none", "tracks"])
def test_one_tick_minus_8pct_warns_on_next_monitor_run(db, mode):
    sent = _crash_run(db, {1: 920.0}, wallet_mode=mode)
    assert sent and sent[0][0] == 5 and sent[0][1].startswith("EQUITY WARN")


@pytest.mark.parametrize("mode", ["const", "none", "tracks"])
def test_minus_2pct_per_min_cascade_warns_and_alerts_on_time(db, mode):
    # 1000·0.98^k: 5.9% at m3, 11.4% at m6 → first crons past the lines: 5, 10.
    sent = _crash_run(db, {k: 1000.0 * 0.98 ** k for k in range(1, 16)},
                      wallet_mode=mode)
    assert [(m, x.split(" ")[1]) for m, x in sent] == [(5, "WARN"), (10, "DROP")]


def test_price_move_in_position_is_not_flagged(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    b._last_position_side, b._last_position_qty = "long", 1.0
    _tick(b, bot_mod, MID, 1000.0, "long", wallet=1000.0)
    _tick(b, bot_mod, MID + dt.timedelta(seconds=65), 920.0, "long", wallet=1000.0)
    assert json.loads(state.get_meta("live_equity"))["transfer_pending"] is False


# --------------------------------------------------------------------------
# Sourcery-style review: tick ordering, lock exposure, atomic anchor write
# --------------------------------------------------------------------------

def test_real_bracket_fill_is_not_flagged_as_a_transfer(db):
    """Real _detect_bracket_exit (NOT stubbed): a bracket SL fills on Binance
    between ticks, so the wallet drops ~3.5% while the bot's position snapshot
    and fills table are still pre-exit when equity is fetched. The live-equity
    writer must run after the detector so the exit explains the move."""
    import bot as bot_mod
    from exchange import state
    b, _ = _make_bot()
    _ready_principal()
    del b._detect_bracket_exit                      # use the real one
    b.coid_prefix, b.base_asset = "snap-v1-", "BTC"
    b._last_position_side, b._last_position_entry = "unknown", None
    b._last_position_qty, b._last_entry_root = None, None
    b.client.ex.fetch_my_trades.return_value = [
        {"side": "sell", "price": 96.5, "amount": 1.0}]
    state.record_fill(side="long", qty=1.0, price=100.0, reason="entry",
                      equity_after=1000.0, client_order_id_root="r1")
    held = SimpleNamespace(side="long", entry_price=100.0, qty=1.0)
    _tick(b, bot_mod, MID, 1000.0, "long", wallet=1000.0, pos=held)   # adopts
    _tick(b, bot_mod, MID + dt.timedelta(seconds=30), 1000.0, "long", wallet=1000.0,
          pos=held)                                                   # steady state
    _tick(b, bot_mod, MID + dt.timedelta(seconds=60), 965.0, "flat", wallet=965.0,
          pos=SimpleNamespace(side="flat", entry_price=0.0, qty=0.0))
    assert not any("unexplained equity move" in m for _, m in b.log.lines), b.log.lines
    assert b._transfer_suspect is None
    assert json.loads(state.get_meta("live_equity"))["transfer_pending"] is False


def test_unrelated_fill_does_not_clear_a_pending_suspect(db):
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    _tick(b, bot_mod, MID, 1000.0, "long", wallet=1000.0)
    _tick(b, bot_mod, MID + dt.timedelta(seconds=60), 1100.0, "long", wallet=1100.0)
    assert b._transfer_suspect is not None
    state.record_fill(side="close", qty=1.0, price=1.0, reason="x", equity_after=1100.0)
    _tick(b, bot_mod, MID + dt.timedelta(seconds=120), 1100.0, "long", wallet=1100.0)
    assert json.loads(state.get_meta("live_equity"))["transfer_pending"] is True


def _hold_write_lock(path):
    c = sqlite3.connect(path, timeout=0.1, isolation_level=None)
    c.execute("BEGIN IMMEDIATE")
    return c


def test_locked_db_bookkeeping_skips_fast_and_never_blocks_the_tick(db):
    import time as _time
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    locker = _hold_write_lock(db)
    try:
        t0 = _time.monotonic()
        _tick(b, bot_mod, MID, 1000.0, "long")
        elapsed = _time.monotonic() - t0
    finally:
        locker.rollback()
        locker.close()
    # Two capped writes (anchor + live_equity) at 0.5 s each; the default busy
    # wait would be 10 s apiece.
    assert elapsed < 4.0, elapsed
    assert state.get_meta("daily_anchor_date") is None      # skipped, retried later
    assert state.get_meta("live_equity") is None
    _tick(b, bot_mod, MID + dt.timedelta(seconds=60), 1000.0, "long")
    assert state.get_meta("daily_anchor_date") == "2026-10-08"


def test_anchor_date_is_written_last(db):
    """A failure between the anchor writes must not leave today's date paired
    with yesterday's equity/baseline (the guard would trust it all day)."""
    b, bot_mod = _make_bot()
    _ready_principal()
    from exchange import state
    state.set_meta("daily_anchor_date", "2026-10-07")
    state.set_float("daily_anchor_equity", 900.0)
    real = state.set_float

    def flaky(key, value, **kw):
        if key == "daily_anchor_equity":
            raise sqlite3.OperationalError("locked")
        return real(key, value, **kw)
    with patch.object(bot_mod.state, "set_float", side_effect=flaky):
        _tick(b, bot_mod, MID, 1000.0, "long")
    assert state.get_meta("daily_anchor_date") == "2026-10-07"   # still stale
    _tick(b, bot_mod, MID + dt.timedelta(seconds=60), 1000.0, "long")
    assert state.get_meta("daily_anchor_date") == "2026-10-08"
    assert state.get_float("daily_anchor_equity") == pytest.approx(1000.0)


# --------------------------------------------------------------------------
# Transfer detector seeded across a restart
# --------------------------------------------------------------------------

def _restart_after_sample(wallet_after: float, persisted: bool = True):
    from exchange import state
    b1, bot_mod = _make_bot()
    _ready_principal()
    b1._last_position_side, b1._last_position_qty = "flat", 0.0
    flat = SimpleNamespace(side="flat", entry_price=0.0, qty=0.0)
    _tick(b1, bot_mod, MID, 1000.0, "flat", wallet=1000.0, pos=flat)
    if not persisted:
        state.set_meta("live_equity", "")
    b2, _ = _make_bot()                       # fresh process: no in-memory sample
    b2._last_position_side, b2._last_position_qty = "flat", 0.0
    _tick(b2, bot_mod, MID + dt.timedelta(seconds=90), wallet_after, "flat",
          wallet=wallet_after, pos=flat)
    return b2


def test_restart_with_persisted_sample_flags_transfer_made_while_down(db):
    b2 = _restart_after_sample(1100.0)
    assert b2._transfer_suspect is not None
    assert any("unexplained equity move" in m for _, m in b2.log.lines)


def test_restart_without_persisted_sample_is_blind_as_before(db):
    b2 = _restart_after_sample(1100.0, persisted=False)
    assert b2._transfer_suspect is None


def test_restart_with_stale_persisted_sample_is_not_used(db):
    from exchange import state
    b1, bot_mod = _make_bot()
    _ready_principal()
    flat = SimpleNamespace(side="flat", entry_price=0.0, qty=0.0)
    _tick(b1, bot_mod, MID, 1000.0, "flat", wallet=1000.0, pos=flat)
    b2, _ = _make_bot()
    _tick(b2, bot_mod, MID + dt.timedelta(seconds=1200), 1100.0, "flat",
          wallet=1100.0, pos=flat)
    assert b2._transfer_suspect is None
