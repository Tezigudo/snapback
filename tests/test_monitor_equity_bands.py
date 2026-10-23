"""Equity alerting fires on band transitions, not on every tick.

Regression cover for the 2026-07-28 alert-fatigue fix: v1 sat 6.3% below its
principal anchor for 30 hours and the monitor sent 52 identical EQUITY WARN
emails (one per 30-min cooldown window). Equity-vs-anchor is a state, not an
event — it must alert on change, stay quiet while unchanged, and never go
silent about a severe drawdown.
"""

from __future__ import annotations

import datetime as dt
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


@pytest.fixture
def db(tmp_path: Path):
    """Factory for a state.db shaped like a live leg's: principal anchor,
    today's daily-anchor equity, and the fills table _equity_from_db reads."""
    counter = iter(range(1000))

    def _make(equity: float, anchor: float = 100.0) -> Path:
        path = tmp_path / f"state_{next(counter)}.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE fills (id INTEGER PRIMARY KEY, equity_after REAL)")
        today_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        conn.executemany(
            "INSERT INTO meta VALUES (?, ?)",
            [("principal_anchor", str(anchor)),
             ("deploy_start_equity", str(anchor)),
             ("daily_anchor_equity", str(equity)),
             ("daily_anchor_date", today_utc)],
        )
        conn.commit()
        conn.close()
        return path
    return _make


def _run(db_path: Path, state: dict, sent: list, ok: bool = True) -> None:
    """Invoke the equity check, recording (subject, body) of anything sent."""
    def _fake_send(subject, body, tag=None):
        sent.append((subject, body))
        return ok
    with patch.object(monitor, "send_alert", _fake_send):
        monitor._check_equity("v1", db_path, CFG, state)


# --- band classification -----------------------------------------------------

@pytest.mark.parametrize("drop, expected", [
    (0.0, "ok"), (4.99, "ok"),
    (5.0, "warn"), (9.99, "warn"),
    (10.0, "alert"), (35.0, "alert"),
    (-2.0, "ok"),  # above anchor (profit)
])
def test_band_boundaries(drop, expected):
    assert monitor._equity_band(drop, CFG) == expected


# --- the actual bug: a persistent band must not re-alert ---------------------

def test_warn_alerts_once_then_stays_silent(db):
    """THE regression. 6.3% below anchor for many ticks = exactly one email."""
    path = db(equity=93.7)          # -6.3%
    state, sent = {"alerts": {}}, []
    for _ in range(20):
        _run(path, state, sent)
    assert len(sent) == 1
    assert sent[0][0] == "EQUITY WARN 6.3%: v1"
    assert state["equity_bands"]["v1"] == "warn"


def test_ok_never_alerts(db):
    path = db(equity=99.0)          # -1%
    state, sent = {"alerts": {}}, []
    for _ in range(5):
        _run(path, state, sent)
    assert sent == []
    assert state["equity_bands"]["v1"] == "ok"


# --- transitions -------------------------------------------------------------

def test_warn_to_alert_emits_on_deepening(db):
    state, sent = {"alerts": {}}, []
    _run(db(equity=93.0), state, sent)       # -7% → warn
    _run(db(equity=85.0), state, sent)       # -15% → alert
    assert [s[0] for s in sent] == ["EQUITY WARN 7.0%: v1", "EQUITY DROP 15.0%: v1"]
    assert state["equity_bands"]["v1"] == "alert"


def test_recovery_emits_once(db):
    state, sent = {"alerts": {}}, []
    _run(db(equity=93.0), state, sent)       # warn
    _run(db(equity=99.0), state, sent)       # recovered
    _run(db(equity=99.0), state, sent)       # still fine — silent
    assert [s[0] for s in sent] == ["EQUITY WARN 7.0%: v1", "EQUITY RECOVERED: v1"]
    assert state["equity_bands"]["v1"] == "ok"


def test_recovery_body_reports_a_gain_as_a_gain(db):
    """The body hardcoded a minus in front of drop_pct, so a leg ABOVE its
    anchor recovered with "= --6.39%" — a double minus announcing a 6.4% gain as
    a loss, on the one mail whose job is to say the problem is over. donchian
    sent exactly that every morning for six days.
    """
    state, sent = {"alerts": {}}, []
    _run(db(equity=93.0), state, sent)        # warn, to have something to leave
    _run(db(equity=110.0), state, sent)       # +10% — above anchor
    subject, body = sent[-1]
    assert subject == "EQUITY RECOVERED: v1"
    # "--" alone would match the "\n--\n" signature rule _emit appends.
    assert "= --" not in body, f"double minus is back: {body}"
    assert "= +10.00%." in body


def test_loss_body_still_carries_one_minus(db):
    """Guard the other direction: the sign fix must not drop the minus from a
    genuine drawdown, which is v1's real -13.73% case.
    """
    state, sent = {"alerts": {}}, []
    _run(db(equity=85.0), state, sent)        # -15% → alert
    assert "= -15.00%." in sent[-1][1]


def test_alert_to_warn_is_reported_as_improving(db):
    state, sent = {"alerts": {}}, []
    _run(db(equity=85.0), state, sent)       # alert
    _run(db(equity=93.0), state, sent)       # partial recovery → warn
    assert sent[1][0] == "EQUITY WARN 7.0%: v1"
    assert "improving" in sent[1][1]


# --- never silently swallow a pre-existing problem ---------------------------

def test_first_observation_in_warn_still_alerts(db):
    """No prior band recorded (fresh deploy) + already unhealthy → must alert."""
    state, sent = {"alerts": {}}, []
    _run(db(equity=93.7), state, sent)
    assert len(sent) == 1
    assert sent[0][0] == "EQUITY WARN 6.3%: v1"


def test_first_observation_ok_records_band_without_alerting(db):
    state, sent = {"alerts": {}}, []
    _run(db(equity=100.0), state, sent)
    assert sent == []
    assert state["equity_bands"]["v1"] == "ok"


# --- severe band keeps a slow heartbeat --------------------------------------

def test_alert_band_repeats_after_reminder_window(db):
    path = db(equity=85.0)                   # -15% → alert
    state, sent = {"alerts": {}}, []
    _run(path, state, sent)                  # initial alert
    _run(path, state, sent)                  # within reminder window → silent
    assert len(sent) == 1

    # Backdate the stamp past the 24h reminder window.
    old = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(hours=25)
    state["alerts"]["equity_alert:v1"] = old.isoformat()
    _run(path, state, sent)
    assert len(sent) == 2
    assert sent[1][0] == "EQUITY DROP 15.0%: v1 (ongoing)"


def test_warn_band_never_repeats_even_after_days(db):
    """Contrast with the alert band: warn has no reminder heartbeat."""
    path = db(equity=93.0)
    state, sent = {"alerts": {}}, []
    _run(path, state, sent)
    old = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=7)
    state["alerts"]["equity_warn:v1"] = old.isoformat()
    _run(path, state, sent)
    assert len(sent) == 1


# --- delivery failures must not lose the alert -------------------------------

def test_failed_send_does_not_commit_band(db):
    """SMTP down → band uncommitted → retried next tick rather than lost."""
    path = db(equity=93.0)
    state, sent = {"alerts": {}}, []
    _run(path, state, sent, ok=False)
    assert "v1" not in state.get("equity_bands", {})
    _run(path, state, sent, ok=True)         # SMTP back
    assert state["equity_bands"]["v1"] == "warn"
    assert len(sent) == 2                    # attempted twice, delivered once


# --- degradation -------------------------------------------------------------

def test_unreadable_db_is_silent(tmp_path):
    state, sent = {"alerts": {}}, []
    _run(tmp_path / "nonexistent.db", state, sent)
    assert sent == []


def test_zero_anchor_is_silent(db):
    state, sent = {"alerts": {}}, []
    _run(db(equity=50.0, anchor=0.0), state, sent)
    assert sent == []


# --- silent DB-read failure ---------------------------------------------------

def test_persistent_read_failure_alerts_once_then_recovers(tmp_path, db):
    """(None, None) every run used to be fully silent. N consecutive failures
    send ONE 'blind' alert; success resets the counter and sends a recovery."""
    missing = tmp_path / "nope.db"            # unreadable (mode=ro, no file)
    state, sent = {"alerts": {}}, []
    n = monitor.DEFAULTS["equity_read_fail_alert_runs"]
    cfg = {**CFG, "equity_read_fail_alert_runs": n}

    def run(path):
        with patch.object(monitor, "send_alert",
                          lambda s, b, tag=None: sent.append((s, b)) or True):
            monitor._check_equity("v1", path, cfg, state)

    for _ in range(n - 1):
        run(missing)
    assert sent == []
    for _ in range(5):
        run(missing)
    assert [s for s, _ in sent] == ["EQUITY CHECK BLIND: v1"]
    run(db(equity=100.0))
    assert state["equity_read_fails"]["v1"] == 0
    assert [s for s, _ in sent][-1] == "EQUITY CHECK RECOVERED: v1"


def test_short_read_failure_is_silent(tmp_path, db):
    state, sent = {"alerts": {}}, []
    with patch.object(monitor, "send_alert",
                      lambda s, b, tag=None: sent.append(s) or True):
        monitor._check_equity("v1", tmp_path / "nope.db", CFG, state)
        monitor._check_equity("v1", db(equity=100.0), CFG, state)
    assert sent == []
    assert state["equity_read_fails"]["v1"] == 0


# --- kill threshold quoted in the alert mail ----------------------------------

def test_alert_mail_quotes_kill_threshold_from_leg_yaml(db, monkeypatch):
    """The threshold is computed at send time from the leg's params YAML
    (fixture: kill_switch_equity_fraction 0.55 → -45.0%), never hardcoded."""
    monkeypatch.setattr(monitor, "PARAMS_DIR", ROOT / "tests" / "fixtures")
    state, sent = {"alerts": {}}, []
    _run(db(equity=85.0), state, sent)                       # -15% → alert
    assert sent and "Bot kill switch flattens at -45.0% from principal." in sent[-1][1]
    # Ongoing-alert reminder carries the same computed sentence.
    state["alerts"].clear()
    state["alerts"]["equity_alert:v1"] = "2000-01-01T00:00:00+00:00"
    _run(db(equity=85.0), state, sent)
    assert len(sent) == 2 and sent[-1][0].endswith("(ongoing)")
    assert "flattens at -45.0% from principal." in sent[-1][1]


@pytest.mark.parametrize("raw", ["deploy: {}\n", "deploy: {kill_switch_equity_fraction: 42}\n",
                                 ": not yaml ["])
def test_alert_mail_never_invents_a_threshold(db, monkeypatch, tmp_path, raw):
    (tmp_path / "params.yaml").write_text(raw)
    monkeypatch.setattr(monitor, "PARAMS_DIR", tmp_path)
    state, sent = {"alerts": {}}, []
    _run(db(equity=85.0), state, sent)
    body = sent[-1][1]
    assert "configured kill_switch_equity_fraction of principal" in body
    assert "flattens at -" not in body


def test_leg_params_files_match_bot_instance_profiles():
    import bot
    for leg in monitor.LEGS:
        name = leg["name"]
        assert monitor.LEG_PARAMS_FILES[name] == bot.INSTANCE_PROFILES[name]["config"].name
