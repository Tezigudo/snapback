"""SQLite state store for the bot. WAL mode. Survives restarts.

Schema:
  meta(key TEXT PRIMARY KEY, value TEXT)
    'deploy_start_equity'       : float
    'deploy_start_ts'           : ISO ts
    'last_entry_bar_ts'         : ISO ts (naive UTC, bar OPEN) of the last CLOSED bar
                                  _maybe_enter evaluated, or consumed because it
                                  closed while a position was held / entries
                                  were paused. Written every time
                                  _last_signal_ts advances; read by boot-resume
                                  (adopt path ONLY) to stop a same-bar re-entry.
    'boot_adopt_log'            : JSON {signal_id, count, first_ts} — boot-resume
                                  adopt-loop guard (D5). Written on adopt only.
    'consecutive_losses'        : int  UNENFORCED/DEAD - nothing writes or reads it
                                  (docs/CONSECUTIVE_LOSS_BREAKER_PLAN.md)
    'daily_anchor_date'         : YYYY-MM-DD UTC date of today's equity anchor
    'daily_anchor_equity'       : float LIVE equity (wallet + unrealized) at the first
                                  tick after UTC midnight, flat or holding (2026-10-13)
    'live_equity'               : JSON {"ts": epoch_s, "equity": float}, bot's last
                                  tick equity (≤1 write/60s) — monitor reads it
    'daily_loss_breaker_date'   : YYYY-MM-DD UTC date when breaker was last emitted
                                  (persisted latch; prevents re-emit on restart)
    'principal_source'          : 'manual' | 'income_backfill' (kill-switch anchor)
    'principal_base'            : float seed for P (manual base, else 0.0)
    'principal_anchor'          : cached float P = base + Σ principal_ledger
    'principal_income_watermark_ms' : int, newest income ts folded into P

  principal_ledger(tran_id INTEGER PRIMARY KEY, income_type TEXT, income_usd REAL,
        asset TEXT, ts_ms INTEGER, applied_at TEXT)
    Net-deposited-principal ledger for the kill switch. Holds only
    principal-moving income (TRANSFER/DEPOSIT/WITHDRAW) keyed by Binance's
    stable tranId so re-fetching an overlapping income window on restart cannot
    double-count. P = principal_base + SUM(income_usd) over USDT rows. See
    exchange/principal.py.

  fills(id INTEGER PRIMARY KEY, ts TEXT, side TEXT, qty REAL, price REAL,
        pnl_usd REAL, reason TEXT, equity_after REAL,
        client_order_id_root TEXT NULL)

  events(id INTEGER PRIMARY KEY, ts TEXT, level TEXT, kind TEXT, msg TEXT,
         signal_id TEXT NULL)

  client_order_id_root: the per-signal millisecond id that anchors all 3 legs
    of a bot trade (entry + SL + TP) on Binance. See exchange/binance_client.py
    `_coid()`. Lets us join state.db fills to Binance-reported trades by
    clientOrderId prefix `snap-v1-<root>-{e|s|t|x|bf|h|k}`.

  outbox(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, payload TEXT,
         created_at TEXT, attempts INTEGER, last_error TEXT NULL)
    Pending events queued for push to investing-consolidate's /bot-event API.
    Rows are inserted by enqueue_bot_event() and removed by
    tools.consolidate_push.drain() once the API acknowledges. The bot's local
    state.db is the source of truth; consolidate is a downstream read-only
    view. If consolidate is unreachable, events queue here and replay
    automatically when the API comes back.

  dead_letter(id INTEGER PRIMARY KEY, outbox_id INTEGER, kind TEXT,
              payload TEXT, reason TEXT, dead_at TEXT)
    Outbox rows that were permanently rejected by the backend (persistent 4xx
    after DEAD_LETTER_AFTER_ATTEMPTS retries). The payload is preserved here
    so no data is lost silently — operators can inspect and replay manually
    if needed. drain() moves rows here rather than discarding them.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .env import REPO_ROOT

# Env-var override so a second bot instance (Donchian leg) can point at
# data/state_donchian.db without code changes. Default matches the single-bot
# v1 deploy.
DB_PATH = Path(os.environ.get("SNAPBACK_STATE_DB", REPO_ROOT / "data" / "state.db"))


def set_db_path(path: str | Path) -> None:
    """Override DB_PATH from bot.main() after CLI args are parsed.

    Used by the second bot instance (Donchian leg) to point at
    data/state_donchian.db. Call BEFORE any other state.* function.
    """
    global DB_PATH
    DB_PATH = Path(path)


def _conn(timeout: float = 10.0) -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=timeout)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


def _columns(c: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            pnl_usd REAL,
            reason TEXT,
            equity_after REAL,
            client_order_id_root TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            level TEXT NOT NULL,
            kind TEXT NOT NULL,
            msg TEXT NOT NULL,
            signal_id TEXT
        );
        CREATE TABLE IF NOT EXISTS outbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_outbox_id ON outbox(id);
        CREATE TABLE IF NOT EXISTS dead_letter (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            outbox_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            reason TEXT NOT NULL,
            dead_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_dead_letter_dead_at ON dead_letter(dead_at);
        CREATE TABLE IF NOT EXISTS principal_ledger (
            tran_id INTEGER PRIMARY KEY,
            income_type TEXT NOT NULL,
            income_usd REAL NOT NULL,
            asset TEXT NOT NULL,
            ts_ms INTEGER NOT NULL,
            applied_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_principal_ledger_ts ON principal_ledger(ts_ms);
        """)
        # Additive migrations for pre-existing databases.
        if "client_order_id_root" not in _columns(c, "fills"):
            c.execute("ALTER TABLE fills ADD COLUMN client_order_id_root TEXT")
        if "signal_id" not in _columns(c, "events"):
            c.execute("ALTER TABLE events ADD COLUMN signal_id TEXT")
        # dead_letter table is created by the IF NOT EXISTS above; no ALTER needed.


def get_meta(key: str, default: str | None = None, timeout: float = 10.0) -> str | None:
    with _conn(timeout) as c:
        row = c.execute(META_GET_SQL, (key,)).fetchone()
    return row[0] if row else default


def set_meta(key: str, value: str, timeout: float = 10.0) -> None:
    """Upsert a meta key. `timeout` is sqlite's busy wait; pass a short one
    from any path that must never stall trading (see bot._persist_last_signal_ts)."""
    with _conn(timeout) as c:
        c.execute("INSERT INTO meta(key, value) VALUES (?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, value))


def get_float(key: str, default: float = 0.0, timeout: float = 10.0) -> float:
    v = get_meta(key, timeout=timeout)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def set_float(key: str, value: float, timeout: float = 10.0) -> None:
    set_meta(key, repr(float(value)), timeout=timeout)


def record_fill(side: str, qty: float, price: float, reason: str,
                pnl_usd: float | None = None, equity_after: float | None = None,
                client_order_id_root: str | None = None) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO fills(ts, side, qty, price, pnl_usd, reason, "
            "equity_after, client_order_id_root) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(UTC).isoformat(), side, qty, price,
             pnl_usd, reason, equity_after, client_order_id_root),
        )


def record_event(level: str, kind: str, msg: str | dict,
                 signal_id: str | None = None) -> None:
    if isinstance(msg, dict):
        msg = json.dumps(msg, default=str)
    with _conn() as c:
        c.execute(
            "INSERT INTO events(ts, level, kind, msg, signal_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now(UTC).isoformat(), level, kind, msg, signal_id),
        )


def enqueue_bot_event(
    kind: str,
    *,
    signal_id: str | None = None,
    strategy: str | None = None,
    side: str | None = None,
    qty: float | None = None,
    price_usd: float | None = None,
    notional_usd: float | None = None,
    equity_usd: float | None = None,
    payload: dict | None = None,
) -> int:
    """Queue a bot event for push to consolidate's /bot-event API.

    The push happens out-of-band via tools.consolidate_push.drain(). This
    function just inserts to the local outbox table and returns the row id.

    bot_ts_ms is captured here (at enqueue time) and stored in the payload
    so a queued event keeps its original timestamp even if the push is
    delayed by an API outage.
    """
    import time as _time
    body = json.dumps({
        "bot_ts_ms": int(_time.time() * 1000),
        "kind": kind,
        "signal_id": signal_id,
        "strategy": strategy,
        "side": side,
        "qty": qty,
        "price_usd": price_usd,
        "notional_usd": notional_usd,
        "equity_usd": equity_usd,
        "payload": payload or {},
    }, default=str)
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO outbox(kind, payload, created_at) VALUES (?, ?, ?)",
            (kind, body, datetime.now(UTC).isoformat()),
        )
        return int(cur.lastrowid or 0)


def outbox_pending(limit: int = 50) -> list[tuple[int, str, str, int]]:
    """Return up to `limit` pending outbox rows oldest-first.

    Each row: (id, kind, payload_json, attempts).
    """
    with _conn() as c:
        return [
            (int(r[0]), str(r[1]), str(r[2]), int(r[3]))
            for r in c.execute(
                "SELECT id, kind, payload, attempts FROM outbox "
                "ORDER BY id ASC LIMIT ?", (limit,)
            ).fetchall()
        ]


def outbox_delete(ids: list[int]) -> int:
    """Hard-delete outbox rows by id (after successful push)."""
    if not ids:
        return 0
    with _conn() as c:
        placeholders = ",".join("?" * len(ids))
        cur = c.execute(f"DELETE FROM outbox WHERE id IN ({placeholders})", ids)
        return cur.rowcount or 0


def outbox_mark_failed(ids: list[int], error: str) -> None:
    """Increment attempts + record last_error for the given outbox rows."""
    if not ids:
        return
    with _conn() as c:
        placeholders = ",".join("?" * len(ids))
        c.execute(
            f"UPDATE outbox SET attempts = attempts + 1, last_error = ? "
            f"WHERE id IN ({placeholders})",
            [error, *ids],
        )


def outbox_dead_letter_over_limit(
    ids: list[int], reason: str, max_attempts: int,
) -> list[tuple[int, str, str]]:
    """Move rows with attempts >= max_attempts from outbox to dead_letter.

    Called by drain() after a persistent 4xx so poison rows cannot block
    the entire queue forever. The full payload is preserved in dead_letter
    so no data is lost silently — operators can inspect / replay manually.

    Returns list of (outbox_id, kind, payload) for each moved row so the
    caller can log them at ERROR level.
    """
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    dead_at = datetime.now(UTC).isoformat()
    with _conn() as c:
        rows = c.execute(
            f"SELECT id, kind, payload FROM outbox "
            f"WHERE id IN ({placeholders}) AND attempts >= ?",
            [*ids, max_attempts],
        ).fetchall()
        if not rows:
            return []
        dead_ids = [int(r[0]) for r in rows]
        c.executemany(
            "INSERT INTO dead_letter(outbox_id, kind, payload, reason, dead_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(r[0], r[1], r[2], reason, dead_at) for r in rows],
        )
        dead_placeholders = ",".join("?" * len(dead_ids))
        c.execute(
            f"DELETE FROM outbox WHERE id IN ({dead_placeholders})", dead_ids,
        )
        return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]


def dead_letter_size() -> int:
    """Count rows currently in the dead_letter table (for monitoring)."""
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) FROM dead_letter").fetchone()
    return int(row[0]) if row else 0


def outbox_size() -> int:
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) FROM outbox").fetchone()
    return int(row[0]) if row else 0


def principal_ledger_upsert(rows: list[tuple[int, str, float, str, int]]) -> int:
    """Idempotently insert principal-moving income rows keyed by tran_id.

    rows: (tran_id, income_type, income_usd, asset, ts_ms).
    INSERT OR IGNORE on the tran_id PRIMARY KEY means a re-fetched (overlapping)
    income window can never double-count. Returns the count of NEWLY inserted
    rows (0 if every row was already present)."""
    if not rows:
        return 0
    applied_at = datetime.now(UTC).isoformat()
    with _conn() as c:
        before = c.execute("SELECT COUNT(*) FROM principal_ledger").fetchone()[0]
        c.executemany(
            "INSERT OR IGNORE INTO principal_ledger"
            "(tran_id, income_type, income_usd, asset, ts_ms, applied_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(int(t), str(k), float(v), str(a), int(ts), applied_at)
             for (t, k, v, a, ts) in rows],
        )
        after = c.execute("SELECT COUNT(*) FROM principal_ledger").fetchone()[0]
    return int(after - before)


def max_fill_id(timeout: float = 10.0) -> int:
    """Highest fills.id (0 if none). Cheap change-detector for 'did the bot
    record a fill since I last looked' (bot._persist_live_equity)."""
    with _conn(timeout) as c:
        row = c.execute("SELECT COALESCE(MAX(id), 0) FROM fills").fetchone()
    return int(row[0]) if row else 0


def principal_ledger_sum(asset: str = "USDT", timeout: float = 10.0) -> float:
    """Signed sum of principal-moving income for one asset (the P contribution)."""
    with _conn(timeout) as c:
        row = c.execute(
            "SELECT COALESCE(SUM(income_usd), 0) FROM principal_ledger WHERE asset=?",
            (asset,),
        ).fetchone()
    return float(row[0]) if row else 0.0


def principal_ledger_max_ts() -> int:
    """Newest income ts_ms in the ledger (0 if empty) — the reconcile watermark."""
    with _conn() as c:
        row = c.execute("SELECT COALESCE(MAX(ts_ms), 0) FROM principal_ledger").fetchone()
    return int(row[0]) if row else 0


def principal_ledger_count() -> int:
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) FROM principal_ledger").fetchone()
    return int(row[0]) if row else 0


def principal_ledger_non_usdt_count() -> int:
    """Count of principal-moving rows in a non-USDT asset (excluded from P)."""
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) FROM principal_ledger WHERE asset<>?", ("USDT",)
        ).fetchone()
    return int(row[0]) if row else 0


# Shared with tools/boot_resume_probe.py, which runs these same reads against a
# `mode=ro` connection. One string per question, so the probe cannot drift from
# what boot() actually asks.
META_GET_SQL = "SELECT value FROM meta WHERE key=?"
LATEST_ENTRY_ROOT_SQL = (
    "SELECT client_order_id_root FROM fills "
    "WHERE reason='entry' "
    "ORDER BY id DESC LIMIT 1"
)
LATEST_ENTRY_TS_SQL = (
    "SELECT ts FROM fills WHERE reason='entry' ORDER BY id DESC LIMIT 1"
)


def latest_entry_coid_root() -> str | None:
    """The most recent entry fill's client_order_id_root, or None.

    Used by close paths (time-stop, boot-flatten, HALT, kill-switch) to tag
    closing orders with the same root as the entry — so consolidate's
    importer can join entry+exit fills via the shared root.

    Returns None if no entry has been recorded OR if the latest entry fill
    has no root (e.g. recorded by a pre-COID bot version). In that case the
    close is placed untagged, which is honest: better than mis-attributing
    to an older, already-closed position's root.
    """
    with _conn() as c:
        row = c.execute(LATEST_ENTRY_ROOT_SQL).fetchone()
    return row[0] if row else None


def latest_entry_fill_ts() -> str | None:
    """ISO ts of the most recent `reason='entry'` fill, or None.

    Boot-resume (Phase B, B2) fallback for seeding the entry-dedup bar when the
    `last_entry_bar_ts` meta key is absent. Read-only.
    """
    with _conn() as c:
        row = c.execute(LATEST_ENTRY_TS_SQL).fetchone()
    return row[0] if row else None


def unpriced_close_rows(since_iso: str, until_iso: str | None = None,
                        limit: int | None = None) -> list[tuple[int, str, float, str, str | None]]:
    """Bot `close` rows recorded without a price (0.0 / NULL) that still carry a
    client_order_id_root: (id, ts, qty, root, reason), oldest first.

    `ts` is written as UTC isoformat, so string comparison orders correctly.
    Used by bot._maybe_backfill_close_prices to repair rows whose exit price
    could not be resolved at close time."""
    sql = ("SELECT id, ts, qty, client_order_id_root, reason FROM fills "
           "WHERE side='close' AND (price IS NULL OR price <= 0) "
           "AND client_order_id_root IS NOT NULL AND client_order_id_root <> '' "
           "AND ts >= ?")
    args: list = [since_iso]
    if until_iso is not None:
        sql += " AND ts < ?"
        args.append(until_iso)
    sql += " ORDER BY id ASC"
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
    with _conn() as c:
        return [(int(r[0]), str(r[1]), float(r[2] or 0.0), str(r[3]), r[4])
                for r in c.execute(sql, args).fetchall()]


def set_close_price(fill_id: int, price: float, pnl_usd: float | None) -> bool:
    """Fill in a close row's price (and pnl) -- ONLY if it is still unpriced, so
    a row that already has a real price can never be overwritten. Returns True
    if a row was updated."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE fills SET price=?, pnl_usd=COALESCE(?, pnl_usd) "
            "WHERE id=? AND side='close' AND (price IS NULL OR price <= 0)",
            (float(price), pnl_usd, int(fill_id)))
        return (cur.rowcount or 0) > 0
