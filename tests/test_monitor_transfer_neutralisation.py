"""The monitor must not read a deposit as a drawdown.

Regression cover for 2026-08-28. `daily_anchor_equity` is a RAW snapshot of
equity taken at UTC midnight and frozen for the rest of the day, while
`principal_anchor` is refreshed hourly from the income ledger. Funding a leg
mid-day therefore raised the DENOMINATOR and left the numerator behind, and the
5-minute monitor reported a drawdown that never happened:

    leg              raw       P      reported   reality
    donchian      161.56  172.53      -6.36%     +6.4%   -> crossed into warn
    sol_supertrend 59.58   80.00     -25.52%     -0.5%   -> crossed into alert
    v1            137.68  183.93     -25.15%    -13.7%   -> alert, inflated

The fix mirrors `bot._daily_book_anchor` at read time: shift the raw anchor by
the net principal moved since it was set. `daily_digest.py` imports the same
reader, so the daily mail carried the same phantom numbers.
"""

from __future__ import annotations

import datetime as dt
import math
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import monitor  # noqa: E402

CFG = {
    "equity_drop_warn_pct": 5.0,
    "equity_drop_alert_pct": 10.0,
    "equity_alert_reminder_hours": 24,
}

# Pinned clocks. `_equity_from_db` now branches on how long the UTC day has been
# running, so any test using a deliberately stale anchor must say WHEN it reads
# or it fails only when the suite happens to run in the first ten minutes of a
# UTC day — the worst kind of flake to chase.
MIDDAY = dt.datetime(2026, 9, 3, 12, 0, 0, tzinfo=dt.timezone.utc)
ROLLOVER = dt.datetime(2026, 9, 3, 0, 0, 4, tzinfo=dt.timezone.utc)


@pytest.fixture
def make_db(tmp_path: Path):
    """Build a leg DB with the Part C principal ledger the live legs carry."""
    counter = iter(range(1000))

    def _make(
        *,
        raw_equity: float,
        principal: float,
        baseline: float | None,
        ledger: list[tuple[float, str]] | None,
        anchor_date: str | None = None,
        with_ledger_table: bool = True,
        fill_equity: float | None = None,
        principal_source: str | None = "income_backfill",
        principal_base: float | None = 0.0,
    ) -> Path:
        path = tmp_path / f"state_{next(counter)}.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE fills (id INTEGER PRIMARY KEY, equity_after REAL)")
        if anchor_date is None:
            anchor_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        meta = [
            ("principal_anchor", str(principal)),
            ("deploy_start_equity", str(principal)),
            ("daily_anchor_equity", str(raw_equity)),
            ("daily_anchor_date", anchor_date),
        ]
        if baseline is not None:
            meta.append(("daily_anchor_principal_sum", str(baseline)))
        if principal_source is not None:
            meta.append(("principal_source", principal_source))
        if principal_base is not None:
            meta.append(("principal_base", str(principal_base)))
        conn.executemany("INSERT INTO meta VALUES (?, ?)", meta)
        if with_ledger_table:
            conn.execute(
                "CREATE TABLE principal_ledger (tran_id INTEGER PRIMARY KEY, "
                "income_type TEXT, income_usd REAL, asset TEXT, ts_ms INTEGER)")
            for i, (amount, asset) in enumerate(ledger or []):
                conn.execute(
                    "INSERT INTO principal_ledger VALUES (?, 'TRANSFER', ?, ?, ?)",
                    (i + 1, amount, asset, 1_700_000_000_000 + i))
        if fill_equity is not None:
            conn.execute("INSERT INTO fills (equity_after) VALUES (?)", (fill_equity,))
        conn.commit()
        conn.close()
        return path
    return _make


def _drop_pct(db_path: Path) -> float:
    cur, anchor = monitor._equity_from_db(db_path)
    assert cur is not None and anchor is not None
    return (1 - cur / anchor) * 100


# --- the bug itself ----------------------------------------------------------

def test_intraday_deposit_is_not_a_drawdown(make_db):
    """Deposit +20 after the anchor was set: numerator moves with denominator."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=100.0,
                 ledger=[(100.0, "USDT"), (20.0, "USDT")])
    cur, anchor = monitor._equity_from_db(db)
    assert cur == pytest.approx(120.0)
    assert anchor == pytest.approx(120.0)
    assert monitor._equity_band(_drop_pct(db), CFG) == "ok"


def test_intraday_withdrawal_is_not_a_gain(make_db):
    """The mirror image — a withdrawal must not manufacture headroom."""
    db = make_db(raw_equity=100.0, principal=80.0, baseline=100.0,
                 ledger=[(100.0, "USDT"), (-20.0, "USDT")])
    cur, _ = monitor._equity_from_db(db)
    assert cur == pytest.approx(80.0)
    assert _drop_pct(db) == pytest.approx(0.0)


def test_real_trading_loss_still_reported_through_a_deposit(make_db):
    """Funding a losing leg must not paper over the loss it already carries.

    v1's real 2026-08-28 numbers: down 13.7% against deposited principal both
    before and after the +21 deposit. The deposit changes neither the loss nor
    the band — it only stopped the monitor from doubling it.
    """
    db = make_db(raw_equity=137.67903059, principal=183.93261979,
                 baseline=162.93261979,
                 ledger=[(162.93261979, "USDT"), (21.0, "USDT")])
    assert _drop_pct(db) == pytest.approx(13.73, abs=0.01)
    assert monitor._equity_band(_drop_pct(db), CFG) == "alert"


@pytest.mark.parametrize("leg, raw, principal, baseline, deposit, before, after", [
    ("donchian",       161.55832398, 172.53125, 150.53125, 22.0, "warn",  "ok"),
    ("sol_supertrend",  59.58291866,  80.0,      60.0,     20.0, "alert", "ok"),
])
def test_2026_08_28_false_alarms_are_gone(make_db, leg, raw, principal,
                                          baseline, deposit, before, after):
    """The two legs that actually crossed a band on the funding, by their real
    numbers. `before` is what the pre-fix reader produced, `after` the truth."""
    db = make_db(raw_equity=raw, principal=principal, baseline=baseline,
                 ledger=[(baseline, "USDT"), (deposit, "USDT")])
    assert monitor._equity_band((1 - raw / principal) * 100, CFG) == before
    assert monitor._equity_band(_drop_pct(db), CFG) == after


def test_only_usdt_rows_count_toward_the_delta(make_db):
    """P is USDT-denominated, so the shift must be too — every live leg carries
    a small BNB fee-transfer row that principal.py deliberately excludes."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=100.0,
                 ledger=[(100.0, "USDT"), (20.0, "USDT"), (0.016, "BNB")])
    cur, _ = monitor._equity_from_db(db)
    assert cur == pytest.approx(120.0)


# --- fail-safes: an unknown delta must restore the old reading, never guess ---

def test_missing_baseline_key_falls_back_to_raw(make_db):
    """A daily anchor set before the baseline key existed. The bot seeds it on
    its next tick; until then we must not invent a correction."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=None,
                 ledger=[(120.0, "USDT")])
    cur, _ = monitor._equity_from_db(db)
    assert cur == pytest.approx(100.0)


def test_unparseable_baseline_falls_back_to_raw(make_db):
    db = make_db(raw_equity=100.0, principal=120.0, baseline=None,
                 ledger=[(120.0, "USDT")])
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO meta VALUES ('daily_anchor_principal_sum', 'oops')")
    conn.commit()
    conn.close()
    cur, _ = monitor._equity_from_db(db)
    assert cur == pytest.approx(100.0)


def test_pre_part_c_db_without_ledger_table_still_reads(make_db):
    """Older DBs have no principal_ledger. The read must not raise — that would
    return (None, None) and silence the equity check entirely."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=None,
                 ledger=None, with_ledger_table=False)
    cur, anchor = monitor._equity_from_db(db)
    assert cur == pytest.approx(100.0)
    assert anchor == pytest.approx(120.0)


def test_fill_branch_is_not_shifted(make_db):
    """`fills.equity_after` is already post-transfer when the fill is newer than
    the deposit; shifting it would double-count. Reached when today's daily
    anchor is absent (here: a stale anchor date) and the rollover grace has
    passed — `now` is pinned to midday so this cannot depend on the wall clock
    the suite happens to run at."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=100.0,
                 ledger=[(100.0, "USDT"), (20.0, "USDT")],
                 anchor_date="2020-01-01", fill_equity=118.0)
    cur, _ = monitor._equity_from_db(db, now=MIDDAY)
    assert cur == pytest.approx(118.0)


# --- the rollover false alarm (2026-09-03) -----------------------------------
#
# The 2026-08-28 fix above left a door open: it corrected the daily-anchor
# branch, and the fill branch it deliberately left raw turned out to be reached
# for a few seconds at every UTC midnight. donchian then mailed a warn at 00:00
# and a recovery at 00:05 for six consecutive days, with identical numbers.

def test_stale_anchor_at_the_rollover_reports_nothing(make_db):
    """donchian's real 00:00:04 UTC read. The date has turned over, the bot's
    60s poll has not re-anchored yet, and the newest fill predates the
    2026-08-28 deposit — so the only available reading is $160.46 against a
    $172.53 anchor: -7.00% mailed while the leg was actually UP 6.39%.
    """
    db = make_db(raw_equity=160.46, principal=172.53, baseline=150.53,
                 ledger=[(150.53, "USDT"), (22.0, "USDT")],
                 anchor_date="2026-09-02", fill_equity=160.46)
    cur, anchor = monitor._equity_from_db(db, now=ROLLOVER)
    assert cur is None, "a rollover-stale anchor must not fall through to a fill"
    assert anchor == pytest.approx(172.53), "the anchor is still knowable"


def test_grace_zero_restores_the_old_stale_fill_reading(make_db):
    """The gate is the ONLY thing silencing this. With the grace disabled the
    pre-fix reading comes back verbatim, which pins the blame on the window
    rather than on some other change to the branch.
    """
    db = make_db(raw_equity=160.46, principal=172.53, baseline=150.53,
                 ledger=[(150.53, "USDT"), (22.0, "USDT")],
                 anchor_date="2026-09-02", fill_equity=160.46)
    cur, anchor = monitor._equity_from_db(db, grace_min=0.0, now=ROLLOVER)
    assert cur == pytest.approx(160.46)
    assert monitor._equity_band((1 - cur / anchor) * 100, CFG) == "warn"


def test_a_leg_stuck_in_a_position_is_still_read_after_the_grace(make_db):
    """The fill fallback is load-bearing, so the fix must not delete it.
    `_maybe_enter` short-circuits on a non-flat position, so a leg in a
    multi-day trade never re-anchors at all — the stale fill is the only reading
    there is, and that is exactly when a real bleed matters most.
    """
    db = make_db(raw_equity=200.0, principal=200.0, baseline=200.0,
                 ledger=[(200.0, "USDT")],
                 anchor_date="2026-08-30", fill_equity=120.0)
    cur, anchor = monitor._equity_from_db(db, now=MIDDAY)
    assert cur == pytest.approx(120.0)
    assert monitor._equity_band((1 - cur / anchor) * 100, CFG) == "alert"


def test_a_fresh_anchor_is_never_gated(make_db):
    """The grace keys off the anchor being stale, not off the hour. A leg that
    re-anchored on time reads normally at 00:00:04 like any other minute.
    """
    db = make_db(raw_equity=100.0, principal=120.0, baseline=100.0,
                 ledger=[(100.0, "USDT"), (20.0, "USDT")],
                 anchor_date="2026-09-03", fill_equity=1.0)
    cur, _ = monitor._equity_from_db(db, now=ROLLOVER)
    assert cur == pytest.approx(120.0)


def test_check_equity_sends_nothing_and_keeps_its_band_through_the_rollover(
        make_db):
    """End to end: no mail, and the band state machine is left untouched so the
    next genuine reading is still judged against the band it actually left.
    A grace of a full day makes this independent of when the suite runs.
    """
    db = make_db(raw_equity=59.87, principal=80.0, baseline=60.0,
                 ledger=[(60.0, "USDT"), (20.0, "USDT")],
                 anchor_date="2026-09-02", fill_equity=59.87)
    sent: list[tuple[str, str]] = []
    state = {"alerts": {}, "equity_bands": {"sol_supertrend": "ok"}}
    cfg = dict(CFG, equity_anchor_grace_min=24 * 60)

    def _fake_send(subject, body, tag=None):
        sent.append((subject, body))
        return True

    with patch.object(monitor, "send_alert", _fake_send):
        monitor._check_equity("sol_supertrend", db, cfg, state)
    assert sent == [], f"expected silence through the rollover, got {sent}"
    assert state["equity_bands"] == {"sol_supertrend": "ok"}


# --- denominator provenance (Sourcery PR #25, comment 1) ---------------------

def test_anchor_is_derived_from_the_same_ledger_read(make_db):
    """P must be `principal_base + Sum(ledger)` — what the kill switch actually
    computes — not the cached `principal_anchor` copy. Here the cache is stale
    on purpose: numerator and denominator must still agree."""
    db = make_db(raw_equity=100.0, principal=100.0, baseline=100.0,
                 ledger=[(100.0, "USDT"), (20.0, "USDT")])
    cur, anchor = monitor._equity_from_db(db)
    assert anchor == pytest.approx(120.0), "stale principal_anchor was used"
    assert cur == pytest.approx(120.0)
    assert monitor._equity_band(_drop_pct(db), CFG) == "ok"


def test_manual_principal_base_is_included(make_db):
    """`manual_principal_usdt` mode seeds a non-zero base; P is base + ledger."""
    db = make_db(raw_equity=150.0, principal=1.0, baseline=50.0,
                 ledger=[(50.0, "USDT")], principal_base=100.0)
    _, anchor = monitor._equity_from_db(db)
    assert anchor == pytest.approx(150.0)


def test_zero_principal_base_is_not_treated_as_missing(make_db):
    """income_backfill mode stores base 0.0. A `> 0` filter would discard it and
    silently fall back to the cached anchor."""
    db = make_db(raw_equity=100.0, principal=999.0, baseline=100.0,
                 ledger=[(120.0, "USDT")], principal_base=0.0)
    _, anchor = monitor._equity_from_db(db)
    assert anchor == pytest.approx(120.0)


def test_uninitialised_ledger_disables_both_shift_and_derived_p(make_db):
    """Mirrors `principal.is_initialized()`. Mid-backfill the ledger holds the
    whole deposit history while the seed is unrecorded — shifting by that would
    move the numerator by the ENTIRE principal."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=0.0,
                 ledger=[(120.0, "USDT")], principal_source=None)
    cur, anchor = monitor._equity_from_db(db)
    assert cur == pytest.approx(100.0), "shifted while backfill was in flight"
    assert anchor == pytest.approx(120.0), "derived P used before initialisation"


# --- non-finite values must never go quiet (Sourcery PR #25, comment 2) ------

@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_non_finite_baseline_falls_back_to_raw(make_db, bad):
    """A NaN/inf delta yields NaN equity, and `_equity_band` scores NaN as `ok`
    — a real drawdown would be silently suppressed."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=None,
                 ledger=[(120.0, "USDT")])
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO meta VALUES ('daily_anchor_principal_sum', ?)", (bad,))
    conn.commit()
    conn.close()
    cur, _ = monitor._equity_from_db(db)
    assert cur == pytest.approx(100.0)
    assert math.isfinite(cur)


def test_infinite_meta_value_is_rejected_by_pos_float(make_db):
    """`inf` survives a bare `> 0` test. Left in, it propagates to a NaN band."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=100.0,
                 ledger=None, with_ledger_table=False, principal_source=None)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE meta SET value = 'inf' WHERE key = 'principal_anchor'")
    conn.commit()
    conn.close()
    _, anchor = monitor._equity_from_db(db)
    assert anchor == pytest.approx(120.0), "inf anchor leaked past _pos_float"


def test_equity_band_scores_nan_as_ok_which_is_why_we_guard():
    """Pins the reason the guards above exist."""
    assert monitor._equity_band(float("nan"), CFG) == "ok"
    assert monitor._equity_band(float("-inf"), CFG) == "ok"
