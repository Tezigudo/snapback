"""Alert-only losing-streak monitor (CONSECUTIVE_LOSS_BREAKER_PLAN.md, E3)."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import monitor  # noqa: E402

CFG = {"streak_alert_thresholds": {"v1": 13, "donchian": 9, "sol_supertrend": 7}}


def _db(tmp_path: Path, pnls, name="s.db") -> Path:
    p = tmp_path / name
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE fills (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, "
              "side TEXT, qty REAL, price REAL, pnl_usd REAL, reason TEXT, "
              "equity_after REAL, client_order_id_root TEXT)")
    for v in pnls:
        c.execute("INSERT INTO fills(ts, side, qty, price, pnl_usd, reason) "
                  "VALUES ('t', 'close', 1, 1, ?, 'bracket_exit')", (v,))
        c.execute("INSERT INTO fills(ts, side, qty, price, reason) "
                  "VALUES ('t', 'long', 1, 1, 'entry')")  # entries never count
    c.commit()
    c.close()
    return p


def test_counts_trailing_losses_and_win_resets(tmp_path):
    assert monitor._losing_streak(_db(tmp_path, [-1, -1, -1])) == 3
    assert monitor._losing_streak(_db(tmp_path, [-1, -1, 2, -1], "b.db")) == 1
    assert monitor._losing_streak(_db(tmp_path, [-1, -1, -1, 2], "c.db")) == 0
    assert monitor._losing_streak(_db(tmp_path, [], "d.db")) == 0


def test_breakeven_and_unknown_are_neutral(tmp_path):
    # 0.0 and NULL neither extend nor reset.
    assert monitor._losing_streak(_db(tmp_path, [-1, 0.0, -1, None, -1])) == 3


def test_unreadable_db_returns_none(tmp_path):
    assert monitor._losing_streak(tmp_path / "missing.db") is None


def test_alert_fires_once_then_rearms_after_win(tmp_path):
    state = {"alerts": {}}
    with patch.object(monitor, "send_alert", return_value=True) as send:
        db = _db(tmp_path, [-1] * 7)
        monitor._check_streak("sol_supertrend", db, CFG, state)
        monitor._check_streak("sol_supertrend", db, CFG, state)   # re-run: silent
        assert send.call_count == 1
        db2 = _db(tmp_path, [-1] * 9, "more.db")                   # grows: still silent
        monitor._check_streak("sol_supertrend", db2, CFG, state)
        assert send.call_count == 1
        db3 = _db(tmp_path, [-1] * 9 + [5], "win.db")              # win re-arms
        monitor._check_streak("sol_supertrend", db3, CFG, state)
        db4 = _db(tmp_path, [-1] * 9 + [5] + [-1] * 7, "again.db")
        monitor._check_streak("sol_supertrend", db4, CFG, state)
        assert send.call_count == 2


def test_below_threshold_is_silent(tmp_path):
    with patch.object(monitor, "send_alert", return_value=True) as send:
        monitor._check_streak("sol_supertrend", _db(tmp_path, [-1] * 6), CFG,
                              {"alerts": {}})
    send.assert_not_called()


def test_per_leg_thresholds(tmp_path):
    db = _db(tmp_path, [-1] * 9)
    with patch.object(monitor, "send_alert", return_value=True) as send:
        monitor._check_streak("v1", db, CFG, {"alerts": {}})             # 9 < 13
        assert send.call_count == 0
        monitor._check_streak("donchian", db, CFG, {"alerts": {}})       # 9 >= 9
        assert send.call_count == 1
        monitor._check_streak("unlisted", db, CFG, {"alerts": {}})       # unmonitored
        assert send.call_count == 1


def test_failed_send_retries_next_tick(tmp_path):
    db = _db(tmp_path, [-1] * 7)
    state = {"alerts": {}}
    with patch.object(monitor, "send_alert", return_value=False) as send:
        monitor._check_streak("sol_supertrend", db, CFG, state)
        monitor._check_streak("sol_supertrend", db, CFG, state)
    assert send.call_count == 2


def test_default_thresholds_and_no_side_effects(tmp_path):
    assert monitor.DEFAULTS["streak_alert_thresholds"] == {
        "v1": 13, "donchian": 9, "sol_supertrend": 7}
    db = _db(tmp_path, [-1] * 20)
    before = db.read_bytes()
    with patch.object(monitor, "send_alert", return_value=True):
        monitor._check_streak("v1", db, CFG, {"alerts": {}})
    assert db.read_bytes() == before                       # DB untouched
    assert not list(tmp_path.glob("HALT*"))                 # no HALT files


def test_realert_at_plus_3_levels(tmp_path):
    state = {"alerts": {}}
    with patch.object(monitor, "send_alert", return_value=True) as send:
        for n, expected in [(7, 1), (9, 1), (10, 2), (12, 2), (13, 3), (20, 4), (21, 4)]:
            monitor._check_streak("sol_supertrend",
                                  _db(tmp_path, [-1] * n, f"l{n}.db"), CFG, state)
            assert send.call_count == expected, n
    assert state["streak_alerted"]["sol_supertrend"] == 19   # 7+3*4


def test_bool_latch_migrates(tmp_path):
    state = {"alerts": {}, "streak_alerted": {"sol_supertrend": True}}
    with patch.object(monitor, "send_alert", return_value=True) as send:
        monitor._check_streak("sol_supertrend", _db(tmp_path, [-1] * 9), CFG, state)
        assert send.call_count == 0
        assert state["streak_alerted"]["sol_supertrend"] == 7
        monitor._check_streak("sol_supertrend", _db(tmp_path, [-1] * 10, "b.db"), CFG, state)
        assert send.call_count == 1


def test_yaml_thresholds_merge_per_leg(tmp_path):
    cfgfile = tmp_path / "monitor.yaml"
    cfgfile.write_text("streak_alert_thresholds:\n  v1: 20\n")
    with patch.object(monitor, "CONFIG_PATH", cfgfile):
        cfg = monitor._load_config()
    assert cfg["streak_alert_thresholds"] == {"v1": 20, "donchian": 9, "sol_supertrend": 7}


def test_check_leg_end_to_end_and_crash_isolated(tmp_path):
    (tmp_path / "logs").mkdir()
    _db(tmp_path, [-1] * 7, "state_x.db")
    (tmp_path / "hb").write_text("x")
    leg = {"name": "sol_supertrend", "heartbeat": "hb", "log": "x.jsonl",
           "state": "state_x.db", "systemd": "u", "live": True}
    cfg = dict(monitor.DEFAULTS)
    state = {"alerts": {}, "log_offsets": {}}
    with patch.object(monitor, "DATA", tmp_path), \
         patch.object(monitor, "LOGS", tmp_path / "logs"), \
         patch.object(monitor, "_systemd_active", return_value=True), \
         patch.object(monitor, "send_alert", return_value=True) as send:
        monitor._check_leg(leg, cfg, state)
        subjects = [c.args[0] for c in send.call_args_list]
        assert any(s.startswith("LOSING STREAK 7") for s in subjects)
        with patch.object(monitor, "_check_streak", side_effect=RuntimeError("boom")):
            monitor._check_leg(leg, cfg, state)   # must not raise
