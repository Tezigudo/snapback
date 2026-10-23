"""A leg holding a position across >=2 UTC midnights must not false-warn at 00:15.

A held position short-circuits `_maybe_enter`, so the bot never re-anchors
`daily_anchor_date`. Once the 10-min rollover grace ends, the monitor falls back
to the last fill's `equity_after`, which was NOT transfer-adjusted: any deposit
made after that fill read as a phantom drawdown (a false warn at 00:15).
This is a PREDICTED residual of the rollover-grace fix (not seen live;
the 09-01..03 warns were the 00:00 race). The stale anchor persists all day
during a hold, not just at 00:15. Live-DB replay at 10-14 00:15: donchian
-3.78% old vs +6.61% new.

NOT covered (separate gap): during a hold the fill ignores unrealized PnL.

The fix shifts the fill by USDT principal moved AFTER the fill, at read time.
The stored anchor is never touched and the fill fallback is kept.
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

UTC = dt.timezone.utc
CFG = {**monitor.DEFAULTS}
# Two full UTC midnights after the anchor (08-26) and the deposit (08-28 02:28).
NOW = dt.datetime(2026, 8, 30, 0, 15, 0, tzinfo=UTC)
FILL_TS = dt.datetime(2026, 8, 27, 9, 0, 0, tzinfo=UTC)
DEPOSIT_MS = int(dt.datetime(2026, 8, 28, 2, 28, tzinfo=UTC).timestamp() * 1000)
SEED_MS = int(dt.datetime(2026, 8, 1, tzinfo=UTC).timestamp() * 1000)


def _db(tmp_path: Path, *, fill_equity: float, fill_ts: str | None = FILL_TS.isoformat(),
        deposit_ms: int = DEPOSIT_MS, with_anchor: bool = True,
        with_ts_column: bool = True) -> Path:
    path = tmp_path / "leg_under_test.db"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    if with_ts_column:
        c.execute("CREATE TABLE fills (id INTEGER PRIMARY KEY, ts TEXT, equity_after REAL)")
        c.execute("INSERT INTO fills (ts, equity_after) VALUES (?, ?)", (fill_ts, fill_equity))
    else:
        c.execute("CREATE TABLE fills (id INTEGER PRIMARY KEY, equity_after REAL)")
        c.execute("INSERT INTO fills (equity_after) VALUES (?)", (fill_equity,))
    c.execute("CREATE TABLE principal_ledger (tran_id INTEGER PRIMARY KEY, "
              "income_type TEXT, income_usd REAL, asset TEXT, ts_ms INTEGER)")
    c.execute("INSERT INTO principal_ledger VALUES (1, 'TRANSFER', 150.53, 'USDT', ?)", (SEED_MS,))
    c.execute("INSERT INTO principal_ledger VALUES (2, 'TRANSFER', 22.0, 'USDT', ?)", (deposit_ms,))
    meta = [("principal_source", "income_backfill"), ("principal_base", "0.0"),
            ("principal_anchor", "172.53"), ("deploy_start_equity", "150.53")]
    if with_anchor:   # stale: set on 08-26, never refreshed while the position is open
        meta += [("daily_anchor_equity", "158.0"), ("daily_anchor_date", "2026-08-26"),
                 ("daily_anchor_principal_sum", "150.53")]
    c.executemany("INSERT INTO meta VALUES (?, ?)", meta)
    c.commit()
    c.close()
    return path


def _drop(db: Path) -> float:
    cur, anchor = monitor._equity_from_db(db, now=NOW)
    assert cur is not None and anchor is not None
    return (1 - cur / anchor) * 100


def test_deposit_after_stale_fill_is_not_a_drawdown(tmp_path):
    # Fixture numbers: raw 160.46 / 172.53 = -7.0% (phantom); adjusted 182.46 -> ok.
    db = _db(tmp_path, fill_equity=160.46)
    cur, anchor = monitor._equity_from_db(db, now=NOW)
    assert cur == pytest.approx(182.46)
    assert anchor == pytest.approx(172.53)
    assert monitor._equity_band(_drop(db), CFG) == "ok"


def test_deposit_before_the_fill_is_not_double_counted(tmp_path):
    early = int(dt.datetime(2026, 8, 26, tzinfo=UTC).timestamp() * 1000)
    db = _db(tmp_path, fill_equity=182.46, deposit_ms=early)
    cur, _ = monitor._equity_from_db(db, now=NOW)
    assert cur == pytest.approx(182.46)


def test_real_drop_during_hold_still_alerts(tmp_path):
    db = _db(tmp_path, fill_equity=130.0)   # 152.0 vs 172.53 = -11.9%
    assert monitor._equity_band(_drop(db), CFG) == "alert"


def test_missing_anchor_while_flat_still_reads_the_fill(tmp_path):
    db = _db(tmp_path, fill_equity=130.0, with_anchor=False)
    assert monitor._equity_band(_drop(db), CFG) == "alert"


def test_unparseable_fill_ts_falls_back_to_raw_fill(tmp_path):
    db = _db(tmp_path, fill_equity=160.46, fill_ts="not-a-date")
    cur, _ = monitor._equity_from_db(db, now=NOW)
    assert cur == pytest.approx(160.46)


def test_fills_without_ts_column_falls_back_to_raw_fill(tmp_path):
    db = _db(tmp_path, fill_equity=160.46, with_ts_column=False)
    cur, _ = monitor._equity_from_db(db, now=NOW)
    assert cur == pytest.approx(160.46)


def test_unreadable_ledger_falls_back_to_raw_fill_and_logs(tmp_path, caplog):
    db = _db(tmp_path, fill_equity=160.46)
    c = sqlite3.connect(db)
    c.execute("ALTER TABLE principal_ledger RENAME COLUMN asset TO asset_x")
    c.commit()
    c.close()
    with caplog.at_level("WARNING", logger="snapback.monitor"):
        cur, _ = monitor._equity_from_db(db, now=NOW)
    assert cur == pytest.approx(160.46)
    assert "principal_ledger sum skipped" in caplog.text


def test_check_equity_sends_nothing_at_0015_on_a_multiday_hold(tmp_path):
    db = _db(tmp_path, fill_equity=160.46)
    state: dict = {"alerts": {}}
    real = monitor._equity_from_db
    with patch.object(monitor, "_equity_from_db",
                      lambda p, g=10, now=None: real(p, g, NOW)), \
         patch.object(monitor, "send_alert", return_value=True) as send:
        monitor._check_equity("donchian", db, CFG, state)
    send.assert_not_called()
    assert state.get("equity_bands", {}).get("donchian") in (None, "ok")


def test_check_equity_still_warns_on_a_real_drop_at_0015(tmp_path):
    db = _db(tmp_path, fill_equity=130.0)
    state: dict = {"alerts": {}}
    real = monitor._equity_from_db
    with patch.object(monitor, "_equity_from_db",
                      lambda p, g=10, now=None: real(p, g, NOW)), \
         patch.object(monitor, "send_alert", return_value=True) as send:
        monitor._check_equity("donchian", db, CFG, state)
    send.assert_called_once()
