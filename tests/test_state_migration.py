"""Tests for state.db schema migration (adds client_order_id_root + signal_id).

Verifies:
  1. Fresh init_db() creates the new columns
  2. Pre-existing db (older schema) gets the columns added without data loss
  3. record_fill/record_event with new kwargs persist correctly
  4. latest_entry_coid_root() returns the right value
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch


def _with_temp_db(test_fn):
    """Patch DB_PATH to a temp file so tests don't touch real state.db."""
    def wrapped():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "state.db"
            with patch("exchange.state.DB_PATH", tmp):
                test_fn(tmp)
    return wrapped


@_with_temp_db
def test_fresh_init_has_new_columns(tmp: Path) -> None:
    from exchange import state
    state.init_db()
    with sqlite3.connect(tmp) as c:
        fills_cols = {r[1] for r in c.execute("PRAGMA table_info(fills)")}
        events_cols = {r[1] for r in c.execute("PRAGMA table_info(events)")}
    assert "client_order_id_root" in fills_cols
    assert "signal_id" in events_cols


@_with_temp_db
def test_migration_from_old_schema(tmp: Path) -> None:
    """Simulate a pre-existing db without the new columns; init_db should
    add them without losing data."""
    with sqlite3.connect(tmp) as c:
        c.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, side TEXT NOT NULL,
            qty REAL NOT NULL, price REAL NOT NULL,
            pnl_usd REAL, reason TEXT, equity_after REAL
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, level TEXT NOT NULL,
            kind TEXT NOT NULL, msg TEXT NOT NULL
        );
        INSERT INTO fills (ts, side, qty, price, reason)
            VALUES ('2026-05-19T08:00:00', 'long', 0.001, 65000, 'entry');
        INSERT INTO events (ts, level, kind, msg)
            VALUES ('2026-05-19T08:00:01', 'INFO', 'boot', 'started');
        """)

    from exchange import state
    state.init_db()

    with sqlite3.connect(tmp) as c:
        fills_cols = {r[1] for r in c.execute("PRAGMA table_info(fills)")}
        events_cols = {r[1] for r in c.execute("PRAGMA table_info(events)")}
        # New columns present
        assert "client_order_id_root" in fills_cols
        assert "signal_id" in events_cols
        # Old data still there
        fill_count = c.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
        event_count = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert fill_count == 1
        assert event_count == 1
        # Old rows have NULL in new columns
        coid_val = c.execute(
            "SELECT client_order_id_root FROM fills WHERE id=1").fetchone()[0]
        assert coid_val is None


@_with_temp_db
def test_record_fill_with_coid_root(tmp: Path) -> None:
    from exchange import state
    state.init_db()
    state.record_fill(side="long", qty=0.001, price=65000.0,
                      reason="entry", equity_after=101.0,
                      client_order_id_root="1716120000000")
    with sqlite3.connect(tmp) as c:
        row = c.execute("SELECT client_order_id_root FROM fills "
                        "ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] == "1716120000000"


@_with_temp_db
def test_record_fill_without_coid_defaults_null(tmp: Path) -> None:
    from exchange import state
    state.init_db()
    state.record_fill(side="long", qty=0.001, price=65000.0,
                      reason="entry", equity_after=101.0)
    with sqlite3.connect(tmp) as c:
        row = c.execute("SELECT client_order_id_root FROM fills "
                        "ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] is None


@_with_temp_db
def test_record_event_with_signal_id(tmp: Path) -> None:
    from exchange import state
    state.init_db()
    state.record_event("INFO", "entry", {"side": "long"},
                       signal_id="1716120000000")
    with sqlite3.connect(tmp) as c:
        row = c.execute("SELECT signal_id FROM events "
                        "ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] == "1716120000000"


@_with_temp_db
def test_latest_entry_coid_root(tmp: Path) -> None:
    from exchange import state
    state.init_db()
    # Several fills, only the most recent entry with a non-null root should win
    state.record_fill(side="long", qty=0.001, price=65000.0,
                      reason="entry", equity_after=100.0,
                      client_order_id_root="111")
    state.record_fill(side="close", qty=0.001, price=66000.0,
                      reason="time_stop", equity_after=101.0,
                      client_order_id_root="111")
    state.record_fill(side="long", qty=0.001, price=65500.0,
                      reason="entry", equity_after=101.0,
                      client_order_id_root="222")
    assert state.latest_entry_coid_root() == "222"


@_with_temp_db
def test_latest_entry_coid_root_returns_none_when_latest_untagged(tmp: Path) -> None:
    """If the most recent entry has no root (recorded by a pre-COID bot
    version), return None — better to close untagged than mis-attribute
    to an older, already-closed position's root."""
    from exchange import state
    state.init_db()
    state.record_fill(side="long", qty=0.001, price=65000.0,
                      reason="entry", equity_after=100.0,
                      client_order_id_root="111")
    state.record_fill(side="long", qty=0.001, price=66000.0,
                      reason="entry", equity_after=101.0,
                      client_order_id_root=None)
    assert state.latest_entry_coid_root() is None


@_with_temp_db
def test_latest_entry_coid_root_empty(tmp: Path) -> None:
    from exchange import state
    state.init_db()
    assert state.latest_entry_coid_root() is None
