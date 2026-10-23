"""Tests for the daily-loss circuit breaker.

The breaker (risk.check_daily_loss + bot._daily_loss_blocks_entry) blocks NEW
entries for the rest of the UTC day once intraday drawdown hits
MAX_DAILY_LOSS_PCT (params `risk:` block; tests run on the synthetic fixture
ceiling seeded by conftest). It is a tighter, daily-resetting sibling of the cumulative
kill-switch (_check_kill_switch) and does NOT flatten or HALT.

Covers:
  1. risk.check_daily_loss: just-under threshold passes, at/over raises.
  2. The UTC-day anchor (state.meta) re-anchors on date rollover.
  3. The bot-level gate: under → entries allowed, at/over → blocked + event
     logged once, then a new UTC day re-allows.
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest


def _fixed_datetime(fixed_now):
    """Return a mock datetime class where now(UTC) returns `fixed_now`
    but other datetime functionality (strftime, fromisoformat, etc.)
    still works via the real datetime class."""
    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now
    return _FakeDatetime


# --------------------------------------------------------------------------
# 1. Pure ceiling function
# --------------------------------------------------------------------------

def test_check_daily_loss_under_threshold_passes() -> None:
    from risk import ceilings, check_daily_loss
    # Loss 0.01pp under the ceiling → allowed (no raise).
    day_start = 1000.0
    equity = day_start * (1 - (ceilings().MAX_DAILY_LOSS_PCT - 0.01) / 100.0)
    check_daily_loss(equity, day_start)  # must not raise


def test_check_daily_loss_at_threshold_raises() -> None:
    from risk import RiskBreach, ceilings, check_daily_loss
    day_start = 1000.0
    equity = day_start * (1 - ceilings().MAX_DAILY_LOSS_PCT / 100.0)  # exactly at ceiling
    with pytest.raises(RiskBreach):
        check_daily_loss(equity, day_start)


def test_check_daily_loss_over_threshold_raises() -> None:
    from risk import RiskBreach, check_daily_loss
    with pytest.raises(RiskBreach):
        check_daily_loss(950.0, 1000.0)  # -5%


def test_check_daily_loss_noop_on_bad_anchor() -> None:
    from risk import check_daily_loss
    # Zero/negative day-start equity is treated as "no anchor yet" → never trips.
    check_daily_loss(500.0, 0.0)
    check_daily_loss(500.0, -1.0)


# --------------------------------------------------------------------------
# Helpers: a minimal Bot stub exercising only the breaker plumbing, plus a
# temp state.db so we never touch the real one.
# --------------------------------------------------------------------------

def _with_temp_db(test_fn):
    def wrapped():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "state.db"
            with patch("exchange.state.DB_PATH", tmp):
                from exchange import state
                state.init_db()
                test_fn(tmp)
    return wrapped


class _StubLog:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


def _make_breaker_bot(principal_ready: bool = True):
    """Build a bare object exposing bot.Bot's breaker methods bound to it,
    without running Bot.__init__ (which needs an exchange client + config).

    principal_ready: mark Part C's principal ledger initialised (the default) so
    the breaker's fail-safe gate is open and the behaviour tests exercise the
    active path. Pass False to exercise the not-yet-initialised (boot) path.
    """
    import bot as bot_mod
    from exchange import principal, state
    b = object.__new__(bot_mod.Bot)
    b.log = _StubLog()
    b._daily_loss_blocked = False
    if principal_ready:
        state.set_meta(principal.META_SOURCE, "manual")
    return b, bot_mod


# --------------------------------------------------------------------------
# 2. UTC-day anchor + rollover
# --------------------------------------------------------------------------

@_with_temp_db
def test_anchor_set_then_stable_within_day(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    from exchange import state
    fixed_dt = _fixed_datetime(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", fixed_dt):
        # First call anchors to current equity.
        assert b._daily_anchor_equity(1000.0) == 1000.0
        assert state.get_meta("daily_anchor_date") == "2026-06-15"
        # Same day, different equity → anchor must NOT move.
        assert b._daily_anchor_equity(1234.0) == 1000.0


@_with_temp_db
def test_anchor_resets_on_utc_date_rollover(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    from exchange import state
    day1_dt = _fixed_datetime(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    day2_dt = _fixed_datetime(datetime(2026, 6, 16, 12, 0, 0, tzinfo=UTC))
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", day1_dt):
        b._daily_anchor_equity(1000.0)
        assert state.get_meta("daily_anchor_date") == "2026-06-15"
    # New UTC day → re-anchor to today's equity.
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", day2_dt):
        assert b._daily_anchor_equity(880.0) == 880.0
        assert state.get_meta("daily_anchor_date") == "2026-06-16"


# --------------------------------------------------------------------------
# 3. Bot-level gate behaviour
# --------------------------------------------------------------------------

@_with_temp_db
def test_under_threshold_allows_entry(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    fixed_dt = _fixed_datetime(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", fixed_dt):
        b._daily_anchor_equity(1000.0)
        # -1.5% loss, under the fixture ceiling → not blocked.
        assert b._daily_loss_blocks_entry(985.0) is False
        assert b._daily_loss_blocked is False


@_with_temp_db
def test_over_threshold_blocks_and_logs_event_once(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    from exchange import state
    alerts: list = []
    fixed_dt = _fixed_datetime(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    with patch.object(bot_mod, "send_alert", lambda *a, **k: alerts.append(a)), \
         patch.object(bot_mod, "datetime", fixed_dt):
        b._daily_anchor_equity(1000.0)
        # -5% loss → blocked (over the fixture ceiling).
        assert b._daily_loss_blocks_entry(950.0) is True
        assert b._daily_loss_blocked is True
        # Still blocked on subsequent polls the same day...
        assert b._daily_loss_blocks_entry(945.0) is True
        # ...but the event/alert fires only ONCE (latched).
        with sqlite3.connect(tmp) as c:
            n = c.execute(
                "SELECT COUNT(*) FROM events WHERE kind='daily_loss_breaker'"
            ).fetchone()[0]
        assert n == 1
        assert len(alerts) == 1


@_with_temp_db
def test_new_utc_day_reallows_after_block(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    from exchange import state
    day1_dt = _fixed_datetime(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    day2_dt = _fixed_datetime(datetime(2026, 6, 16, 12, 0, 0, tzinfo=UTC))
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", day1_dt):
        b._daily_anchor_equity(1000.0)
        assert b._daily_loss_blocks_entry(950.0) is True  # -5% → blocked today
    # New UTC day → re-anchor to today's equity → fresh 0% loss → allowed.
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", day2_dt):
        assert b._daily_loss_blocks_entry(950.0) is False
        assert b._daily_loss_blocked is False


# --------------------------------------------------------------------------
# 4. Restart-dedup: persisted latch survives a simulated bot restart
# --------------------------------------------------------------------------

@_with_temp_db
def test_breaker_does_not_re_emit_after_restart_same_day(tmp: Path) -> None:
    """After the breaker fires, a simulated restart within the same UTC day
    must NOT enqueue a second daily_loss_breaker outbox event.

    Before the fix, self._daily_loss_blocked started as False on each restart,
    causing a duplicate event every restart while equity stayed below threshold.
    The fix persists 'daily_loss_breaker_date' to state.meta.
    """
    from exchange import state
    alerts: list = []
    fixed_dt = _fixed_datetime(datetime(2026, 6, 30, 14, 0, 0, tzinfo=UTC))

    def run_bot():
        b, bot_mod = _make_breaker_bot()
        with patch.object(bot_mod, "send_alert", lambda *a, **k: alerts.append(a)), \
             patch.object(bot_mod, "datetime", fixed_dt):
            b._daily_anchor_equity(1000.0)
            return b._daily_loss_blocks_entry(950.0)  # -5% → should block

    # First run: breaker trips, event + alert emitted.
    blocked = run_bot()
    assert blocked is True
    with sqlite3.connect(tmp) as c:
        n_events = c.execute(
            "SELECT COUNT(*) FROM outbox WHERE kind='daily_loss_breaker'"
        ).fetchone()[0]
    assert n_events == 1, "breaker should emit exactly one outbox event on first trip"
    assert len(alerts) == 1

    # Simulated restart: create a fresh bot object (self._daily_loss_blocked=False).
    # The persisted latch in state.meta should prevent re-emission.
    blocked_again = run_bot()
    assert blocked_again is True
    with sqlite3.connect(tmp) as c:
        n_events_after = c.execute(
            "SELECT COUNT(*) FROM outbox WHERE kind='daily_loss_breaker'"
        ).fetchone()[0]
    assert n_events_after == 1, (
        "restart within same UTC day must NOT enqueue a second event; "
        f"got {n_events_after}"
    )
    assert len(alerts) == 1, "alert must not fire again on restart"


@_with_temp_db
def test_breaker_emits_again_on_next_utc_day_after_restart(tmp: Path) -> None:
    """The persistent latch is date-scoped: a breach on day 2 (after a restart)
    should emit a fresh event, even though a latch from day 1 is still in meta.
    """
    from exchange import state
    day1_dt = _fixed_datetime(datetime(2026, 6, 30, 14, 0, 0, tzinfo=UTC))
    day2_dt = _fixed_datetime(datetime(2026, 7, 1, 8, 0, 0, tzinfo=UTC))
    alerts: list = []

    def run_bot(fixed_dt):
        b, bot_mod = _make_breaker_bot()
        with patch.object(bot_mod, "send_alert", lambda *a, **k: alerts.append(a)), \
             patch.object(bot_mod, "datetime", fixed_dt):
            b._daily_anchor_equity(1000.0)
            return b._daily_loss_blocks_entry(950.0)

    # Day 1: breaker trips.
    run_bot(day1_dt)
    with sqlite3.connect(tmp) as c:
        n_day1 = c.execute(
            "SELECT COUNT(*) FROM outbox WHERE kind='daily_loss_breaker'"
        ).fetchone()[0]
    assert n_day1 == 1

    # Day 2 (new UTC day, anchor resets): fresh breach → should emit again.
    run_bot(day2_dt)
    with sqlite3.connect(tmp) as c:
        n_total = c.execute(
            "SELECT COUNT(*) FROM outbox WHERE kind='daily_loss_breaker'"
        ).fetchone()[0]
    assert n_total == 2, (
        "a new UTC day must allow the breaker to emit again; "
        f"got {n_total} total outbox events"
    )


# --------------------------------------------------------------------------
# 5. Transfer-immune BOOK-equity baseline (Part C / C1 daily-loss migration)
#
# The daily baseline is BOOK equity: the raw UTC-midnight anchor shifted by the
# NET principal (TRANSFER/DEPOSIT/WITHDRAW) moved intraday, read from the Part C
# principal ledger (bot._daily_book_anchor). So an intraday deposit/withdrawal
# moves equity AND the baseline by the same amount -> only true trading P&L
# counts toward the daily-loss ceiling. Consistent with the principal-anchored kill
# switch; still block-only (no flatten, no HALT); still resets at UTC midnight.
# --------------------------------------------------------------------------

_INCOME_TS = 1_782_600_000_000  # arbitrary intraday income timestamp (ms)


def _reconcile_transfer(income_usd: float, tran_id: int, asset: str = "USDT") -> None:
    """Simulate a principal-moving income row landing in the ledger exactly as
    principal.reconcile_recent would: tranId-keyed, Binance-signed income
    (deposit +, withdrawal -). Bumps principal_ledger_sum by income_usd."""
    from exchange import state
    state.principal_ledger_upsert(
        [(tran_id, "TRANSFER", income_usd, asset, _INCOME_TS)])


# (b) A real at-ceiling trading loss (no transfers) still blocks entries.
#     Confirms the book-anchor path does not weaken genuine-loss detection.
@_with_temp_db
def test_real_at_ceiling_trading_loss_still_blocks(tmp: Path) -> None:
    from risk import ceilings
    b, bot_mod = _make_breaker_bot()
    fixed_dt = _fixed_datetime(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    at_ceiling = 1000.0 * (1 - ceilings().MAX_DAILY_LOSS_PCT / 100.0)
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", fixed_dt):
        b._daily_anchor_equity(1000.0)          # empty ledger -> baseline sum 0
        # No transfers -> book anchor == raw anchor.
        assert b._daily_book_anchor(at_ceiling) == 1000.0
        # Exactly at the ceiling from pure trading -> blocked.
        assert b._daily_loss_blocks_entry(at_ceiling) is True
        assert b._daily_loss_blocked is True


# (a-i) An intraday WITHDRAWAL must NOT falsely TRIP the breaker. A raw anchor
#       reads the withdrawal as a loss; the book anchor tracks it out.
@_with_temp_db
def test_intraday_withdrawal_does_not_falsely_trip(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    fixed_dt = _fixed_datetime(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", fixed_dt):
        b._daily_anchor_equity(1000.0)          # baseline ledger sum = 0
        # A $100 withdrawal is reconciled into the ledger (signed negative)...
        _reconcile_transfer(-100.0, tran_id=1)
        # ...equity drops to 900 SOLELY from the withdrawal, no trading loss.
        # Raw anchor would read -10% and trip; book anchor drops to 900 too.
        assert b._daily_book_anchor(900.0) == 900.0
        assert b._daily_loss_blocks_entry(900.0) is False
        assert b._daily_loss_blocked is False


# (a-ii) An intraday DEPOSIT must NOT falsely CLEAR a real trading loss. A raw
#        anchor would see the deposit push equity back above the anchor and stop
#        blocking; the book anchor rises with the deposit so the loss still shows.
@_with_temp_db
def test_intraday_deposit_does_not_mask_real_loss(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    fixed_dt = _fixed_datetime(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", fixed_dt):
        b._daily_anchor_equity(1000.0)          # baseline ledger sum = 0
        # Trading lost 5.5% (1000 -> 945); THEN a $100 deposit is reconciled,
        # lifting raw equity to 1045 (ABOVE the raw 1000 anchor).
        _reconcile_transfer(100.0, tran_id=2)
        # Raw anchor would report +4.5% and ALLOW entries (masking the loss).
        # Book anchor = 1000 + 100 = 1100; (1100-1045)/1100 = 5.0% -> blocked.
        assert b._daily_book_anchor(1045.0) == 1100.0
        assert b._daily_loss_blocks_entry(1045.0) is True   # latch starts False
        assert b._daily_loss_blocked is True


# (c) Midnight reset: both the raw anchor AND the principal baseline re-snapshot
#     at the UTC rollover, so a PRIOR-day transfer sitting in the ledger is not
#     mistaken for an intraday move (which would inflate the baseline and
#     suppress the breaker forever).
@_with_temp_db
def test_book_anchor_resets_at_midnight_ignoring_prior_day_transfer(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    day1 = _fixed_datetime(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    day2 = _fixed_datetime(datetime(2026, 6, 16, 12, 0, 0, tzinfo=UTC))
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", day1):
        b._daily_anchor_equity(1000.0)
        _reconcile_transfer(100.0, tran_id=3)   # a deposit happened on day 1
        assert b._daily_loss_blocks_entry(1045.0) is True   # 5.0% book loss -> blocked
    # New UTC day: anchor + principal baseline BOTH re-snapshot to today.
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", day2):
        # Baseline re-snaps to the CURRENT ledger sum (100); the day-1 deposit
        # is NOT re-counted as an intraday move -> book anchor == today's equity.
        assert b._daily_book_anchor(1100.0) == 1100.0
        assert b._daily_loss_blocks_entry(1100.0) is False  # fresh 0% day -> allowed
        assert b._daily_loss_blocked is False
        # A fresh 5% trading loss on day 2 still blocks (baseline uninflated).
        assert b._daily_loss_blocks_entry(1045.0) is True


# (d) The breaker is block-ONLY: it must NEVER write a per-leg or global HALT,
#     and must NEVER enqueue a kill_switch event — even on a catastrophic loss.
@_with_temp_db
def test_breaker_never_writes_halt(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    fixed_dt = _fixed_datetime(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    halt_dir = tmp.parent            # tmp == <td>/state.db, so <td> is clean
    b.instance = "v1"
    b.halt_path = halt_dir / "HALT_v1"
    global_halt = halt_dir / "HALT"
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", fixed_dt):
        b._daily_anchor_equity(1000.0)
        # A brutal 50% loss -> the breaker blocks new entries but must NOT
        # flatten or HALT (that is the kill switch's job, not this guard).
        assert b._daily_loss_blocks_entry(500.0) is True
    assert not b.halt_path.exists(), \
        "daily-loss breaker must never write a per-leg data/HALT_<instance>"
    assert not global_halt.exists(), \
        "daily-loss breaker must never write the global data/HALT"
    with sqlite3.connect(tmp) as c:
        n_kill = c.execute(
            "SELECT COUNT(*) FROM outbox WHERE kind='kill_switch'").fetchone()[0]
    assert n_kill == 0, "daily-loss breaker must never emit a kill_switch event"


# (compat) A legacy daily anchor written by PRE-migration code has no
#          daily_anchor_principal_sum key. The book anchor must seed the baseline
#          to the CURRENT ledger sum (delta 0) so pre-existing principal is never
#          mistaken for an intraday transfer.
@_with_temp_db
def test_legacy_anchor_without_principal_baseline_seeds_safely(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    from exchange import state
    fixed_dt = _fixed_datetime(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    # Ledger already holds historical principal (as the Part C machinery would),
    # but the daily anchor was written by old code with NO principal-sum key.
    _reconcile_transfer(800.0, tran_id=9)
    state.set_meta("daily_anchor_date", "2026-06-15")
    state.set_float("daily_anchor_equity", 1000.0)
    assert state.get_meta("daily_anchor_principal_sum") is None
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", fixed_dt):
        # First book-anchor call seeds the baseline to the current sum (800),
        # yielding a zero delta -> book anchor == raw anchor.
        assert b._daily_book_anchor(1000.0) == 1000.0
        assert float(state.get_meta("daily_anchor_principal_sum")) == 800.0
        # A real 5% loss then still blocks: the 800 pre-existing principal did
        # NOT inflate the baseline.
        assert b._daily_loss_blocks_entry(950.0) is True


# (cold-start) The breaker is INACTIVE until Part C's principal ledger is
# initialised — the same fail-safe window the kill switch is disabled in. This
# closes the boot race: if the daily anchor were snapshotted with an empty
# ledger and the full-history backfill landed afterwards, that backfill would
# look like a giant intraday deposit and inflate the book anchor (false
# entry-block all day). Instead no anchor is taken until P is ready, so the
# baseline captures the post-backfill principal (delta 0).
@_with_temp_db
def test_breaker_inactive_until_principal_initialised(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot(principal_ready=False)
    from exchange import principal, state
    fixed_dt = _fixed_datetime(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod, "datetime", fixed_dt):
        # Backfill still pending: even a catastrophic loss must NOT block, and no
        # daily anchor may be snapshotted yet (a stale sum=0 baseline is the bug).
        assert b._daily_loss_blocks_entry(500.0) is False
        assert state.get_meta("daily_anchor_date") is None
        assert state.get_meta("daily_anchor_principal_sum") is None
        # Backfill completes: the full history lands as one principal sum, and P
        # is marked initialised.
        _reconcile_transfer(130.0, tran_id=101)
        state.set_meta(principal.META_SOURCE, "income_backfill")
        # First active call anchors NOW, so the 130 backfill is captured in the
        # baseline (not read as an intraday deposit) → a flat day is allowed...
        assert b._daily_loss_blocks_entry(1000.0) is False
        assert float(state.get_meta("daily_anchor_principal_sum")) == 130.0
        # ...and a genuine 5% trading loss still blocks.
        assert b._daily_loss_blocks_entry(950.0) is True
