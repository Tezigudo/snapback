"""
Cron-invoked health checker. Designed for cron */5 * * * *.

Checks (per leg):
  1. Heartbeat freshness (alert if > HEARTBEAT_STALE_S)
  2. ERROR / WARNING / mtf_4h_gate_nan events in last poll window
  3. Equity drop from deploy_start_equity (LIVE legs only)
  4. systemd unit state (active vs failed)
  5. Losing streak (ALERT-ONLY, never blocks trading): consecutive losing
     closes read read-only from the leg's fills DB; alerts at the leg's
     threshold (v1 13 / donchian 9 / sol 7) and again every +3. An estimate
     from local fills, not exchange truth (see _losing_streak).

State persistence:
  data/monitor_state.json — last seen log offsets per leg, last alert ts per
  kind, the last equity band (ok/warn/alert) per leg, and the last alerted
  losing-streak level per leg (streak_alerted; a win re-arms it).
  Prevents duplicate alerts every 5 min for the same condition. Event-style
  checks are rate-limited by cooldown; the equity check fires only when its
  band changes (see _check_equity).

Email alerts via alerts.send_alert(). NO LLM calls.

Per-leg config: pulls thresholds from config/monitor.yaml if present, else uses
DEFAULTS below. Times reported as GMT+7 (Asia/Bangkok) in alert bodies.

CRITICAL: never raises. Cron-invoked; a crash is silent failure on the box.
Every exception is caught + logged + suppressed.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # config/monitor.yaml is optional

from alerts import send_alert, is_configured

REPO_ROOT = Path(__file__).resolve().parent
DATA = REPO_ROOT / "data"
LOGS = REPO_ROOT / "logs"
STATE_PATH = DATA / "monitor_state.json"
CONFIG_PATH = REPO_ROOT / "config" / "monitor.yaml"
# Each leg's params YAML (bot.INSTANCE_PROFILES names the same files). Read only
# to quote the leg's configured kill threshold in the equity-alert mail.
PARAMS_DIR = REPO_ROOT / "config"
LEG_PARAMS_FILES: dict[str, str] = {
    "v1": "params.yaml",
    "donchian": "params_donchian.yaml",
    "sol_supertrend": "params_sol_supertrend.yaml",
}

ICT = dt.timezone(dt.timedelta(hours=7), name="ICT")

log = logging.getLogger("snapback.monitor")

DEFAULTS: dict[str, Any] = {
    "heartbeat_stale_s":          120,   # v1 polls 5s, donchian/cnh_short 60s. 120s = comfortable margin.
    "equity_drop_warn_pct":       5.0,   # 5% from anchor → warn
    "equity_drop_alert_pct":      10.0,  # 10% → alert (kill switch fires at 15%)
    "error_alert":                True,  # any ERROR-level log line → alert
    "nan_4h_gate_alert":          True,  # any mtf_4h_gate_nan event → alert
    "alert_cooldown_min":         30,    # don't re-alert same kind within 30 min
    # Equity alerts fire on BAND TRANSITIONS, not on a cooldown (see
    # _check_equity). The severe band still repeats this often so a sustained
    # drawdown can't go silent.
    "equity_alert_reminder_hours": 24,
    # Minutes after UTC midnight during which a stale daily anchor means "the
    # bot has not re-anchored yet", not "here is a reading worth alerting on".
    # See the ROLLOVER GRACE section of _equity_from_db. Must comfortably clear
    # the slowest leg's poll interval (60s for donchian/sol_supertrend).
    "equity_anchor_grace_min":    10,
    # Max age of the bot's per-tick `live_equity` meta value before the reader
    # falls back to the daily anchor / fill. Covers the 60s polls + a 300s
    # catch-all backoff with margin; older means the bot is down.
    "equity_live_max_age_min":    15,
    # Leave a band only once the drop is this many points back past its line.
    "equity_band_hysteresis_pct": 1.0,
    # Consecutive monitor runs whose equity DB read fails ((None, None)) before
    # one "equity check blind" alert. Cron is */5, so 6 runs = ~30 min.
    "equity_read_fail_alert_runs": 6,
    # ALERT-ONLY losing-streak monitor (docs/CONSECUTIVE_LOSS_BREAKER_PLAN.md,
    # decision E3). Per-leg count of consecutive losing closed trades at which
    # to alert once. Each is the leg's longest backtest streak + 1, i.e. "worse
    # than anything in 6 years of history". Never blocks entries or changes
    # risk; a leg missing from this map is not monitored.
    "streak_alert_thresholds":    {"v1": 13, "donchian": 9, "sol_supertrend": 7},
}

LEGS: list[dict[str, str]] = [
    {"name": "v1",         "heartbeat": "heartbeat",            "log": "bot.jsonl",        "state": "state.db",            "systemd": "snapback-btc",                   "live": True},
    {"name": "donchian",   "heartbeat": "heartbeat_donchian",   "log": "donchian.jsonl",   "state": "state_donchian.db",   "systemd": "snapback-btc-donchian",          "live": True},   # real-money since 2026-07-02
    {"name": "sol_supertrend", "heartbeat": "heartbeat_sol_supertrend", "log": "sol_supertrend.jsonl", "state": "state_sol_supertrend.db", "systemd": "snapback-sol-supertrend", "live": True},  # real-money since 2026-07-27 (God's go)
    # cnh_short retired 2026-07-12 (archive/cnh_short_retired_20260712). Its
    # permanent DOWN/STALE alerts every 5 min were failing sends that kept the
    # MailerSend account paused for "high API error rate" — do not re-add a
    # leg here unless its systemd unit actually runs.
    #
    # sol_supertrend flipped to real money 2026-07-27 (DRY_RUN=0 in
    # .env.sol_supertrend on the droplet); `live: True` keeps the equity-drop
    # and error checks armed. Keep both flags in sync if it ever goes back to
    # dry-run.
]


def _now_ict() -> str:
    return dt.datetime.now(tz=ICT).strftime("%Y-%m-%d %H:%M:%S ICT")


def _mins_since_utc_midnight(now: dt.datetime) -> float:
    """Minutes elapsed since the most recent UTC midnight.

    How long the bot has HAD to re-anchor `daily_anchor_date` for the new UTC
    day. Deliberately measured from the rollover rather than from the anchor's
    own date: a leg that has held a position for three days has an anchor three
    days old, but right after midnight it is no more overdue than any other leg.
    """
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return (now - midnight).total_seconds() / 60.0


def _load_state() -> dict[str, Any]:
    """Load monitor state. On first run, initialize log offsets to current
    EOF so we don't re-fire alerts on historical events (e.g. pre-deploy
    KILL_SWITCH or Tracebacks from days/weeks ago). The first cron tick
    establishes the baseline; alerts start firing on the SECOND tick.
    """
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("monitor: state file unreadable, starting fresh: %s", e)
    # First run — pin offsets to current EOF per leg so historical events
    # don't trigger false-positive alerts.
    initial_offsets: dict[str, int] = {}
    for leg in LEGS:
        log_path = LOGS / leg["log"]
        if log_path.exists():
            initial_offsets[f"offset:{leg['name']}"] = log_path.stat().st_size
        else:
            initial_offsets[f"offset:{leg['name']}"] = 0
    return {"alerts": {}, "log_offsets": initial_offsets}


def _save_state(state: dict[str, Any]) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
    except OSError as e:
        log.warning("monitor: failed to save state: %s", e)


def _load_config() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if yaml is not None and CONFIG_PATH.exists():
        try:
            user_cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            # streak_alert_thresholds merges PER LEG so overriding one leg
            # does not silently un-monitor the others.
            thr = dict(cfg["streak_alert_thresholds"])
            thr.update(user_cfg.get("streak_alert_thresholds") or {})
            cfg.update(user_cfg)
            cfg["streak_alert_thresholds"] = thr
        except Exception as e:
            log.warning("monitor: monitor.yaml unreadable, using defaults: %s", e)
    return cfg


def _can_alert(state: dict[str, Any], kind: str, cooldown_min: int) -> bool:
    """Return True if we have not alerted on this kind within the cooldown window."""
    last_ts_str = state["alerts"].get(kind)
    if not last_ts_str:
        return True
    try:
        last_ts = dt.datetime.fromisoformat(last_ts_str)
    except ValueError:
        return True
    return (dt.datetime.now(tz=dt.timezone.utc) - last_ts) > dt.timedelta(minutes=cooldown_min)


def _stamp_alert(state: dict[str, Any], kind: str) -> None:
    state["alerts"][kind] = dt.datetime.now(tz=dt.timezone.utc).isoformat()


def _emit(subject: str, body: str, state: dict[str, Any], kind: str,
          cooldown_min: int, force: bool = False) -> bool:
    """Send an alert unless it is within its cooldown. Returns True if sent.

    `force` bypasses the cooldown — used by the equity band checks, which are
    already rate-limited by only firing on state transitions.
    """
    if not force and not _can_alert(state, kind, cooldown_min):
        log.info("monitor: %s within cooldown, suppressing", kind)
        return False
    full = f"{body}\n\n--\nGenerated {_now_ict()} by monitor.py"
    sent = send_alert(subject, full, tag="snapback-monitor")
    if sent:
        _stamp_alert(state, kind)
    else:
        log.warning("monitor: alert send failed for %s", kind)
    return bool(sent)


def _heartbeat_age_s(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(dt.datetime.now().timestamp() - path.stat().st_mtime)


def _count_log_events(log_path: Path, since_offset: int, patterns: dict[str, str]) -> tuple[dict[str, int], int]:
    """Scan log file from `since_offset` to EOF. Returns (per-pattern counts, new offset).

    Patterns is {kind: substring}. A line counts if substring is present.
    """
    counts = {k: 0 for k in patterns}
    if not log_path.exists():
        return counts, since_offset
    try:
        size = log_path.stat().st_size
        if since_offset > size:
            since_offset = 0  # log rotated
        with log_path.open("rb") as f:
            f.seek(since_offset)
            data = f.read()
            new_offset = f.tell()
        text = data.decode("utf-8", errors="replace")
        for line in text.splitlines():
            for kind, needle in patterns.items():
                if needle in line:
                    counts[kind] += 1
        return counts, new_offset
    except OSError as e:
        log.warning("monitor: log scan failed for %s: %s", log_path, e)
        return counts, since_offset


def _systemd_active(unit: str) -> bool:
    if shutil.which("systemctl") is None:
        return True  # no systemd (dev box) — don't alert
    try:
        r = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "active"
    except (OSError, subprocess.SubprocessError):
        return True  # transient failure — don't alert


def _usdt_ledger_sum(conn: Any, after_ms: float | None = None) -> float | None:
    """Σ USDT principal_ledger income, optionally only rows with ts_ms > after_ms.

    None on a pre-Part-C DB (no table). One query for both callers so the
    asset filter can never drift between the anchor shift and the fill shift.
    """
    import sqlite3
    sql = ("SELECT COALESCE(SUM(income_usd), 0.0) FROM principal_ledger "
           "WHERE asset = 'USDT'")
    params: tuple = ()
    if after_ms is not None:
        sql += " AND ts_ms > ?"
        params = (after_ms,)
    try:
        return conn.execute(sql, params).fetchone()[0]
    except sqlite3.Error as e:
        if "no such table" not in str(e):
            log.warning("monitor: principal_ledger sum skipped: %s", e)
        return None


def _fresh_live_equity(raw: str | None, now: dt.datetime, max_age_min: float,
                       ledger_now: float | None = None) -> tuple[float, bool] | None:
    """Parse the bot's `live_equity` meta value (bot._persist_live_equity):
    {"ts": epoch_s, "equity": x, "ledger_sum": Σledger at write,
     "transfer_pending": bool}.

    Returns None (unusable) unless the equity is finite and > 0 (a transient 0
    would read as -100%) and no older than `max_age_min` (nor >1 min in the
    future). Otherwise returns (equity, trusted). `trusted` is False when the
    sample is flagged transfer_pending, lacks a ledger_sum, or was measured
    against a different principal-ledger state than the caller's denominator
    (`ledger_now`): live equity moves the instant a deposit/withdrawal lands,
    P only on the bot's reconcile (2026-08-28 replay: the drop swung down, up
    and down again — two false mails). The caller then takes min(fallback, live) — see
    _equity_from_db. Absent key ⇒ None ⇒ the pre-2026-10-13 behaviour.
    """
    if not raw:
        return None
    try:
        d = json.loads(raw)
        ts = float(d["ts"])
        eq = float(d["equity"])
    except (TypeError, ValueError, KeyError, IndexError, json.JSONDecodeError):
        return None
    if not (math.isfinite(ts) and math.isfinite(eq)) or eq <= 0:
        return None
    age_s = now.timestamp() - ts
    if age_s > max_age_min * 60.0 or age_s < -60.0:
        return None
    try:
        sample_ledger = float(d.get("ledger_sum"))
    except (TypeError, ValueError):
        sample_ledger = None
    trusted = (d.get("transfer_pending") is False
               and sample_ledger is not None and math.isfinite(sample_ledger)
               and (ledger_now is None
                    or (math.isfinite(ledger_now)
                        and abs(ledger_now - sample_ledger) <= 1e-6)))
    return eq, trusted


def _equity_from_db(
    db_path: Path,
    grace_min: float = DEFAULTS["equity_anchor_grace_min"],
    now: dt.datetime | None = None,
    live_max_age_min: float = DEFAULTS["equity_live_max_age_min"],
) -> tuple[float | None, float | None]:
    """Return (current_equity, anchor_equity) or (None, None) on any failure.

    Anchor: `principal_anchor` (net deposited principal, what the bot's kill
    switch actually uses) when the leg has one, else `deploy_start_equity`.
    The old code returned deploy_start_equity for BOTH values, so drop_pct
    was always 0 and the equity check could never fire.

    Current, in order (2026-10-13):
      1. `live_equity` — the bot's wallet + unrealized from its last tick, if
         <= `live_max_age_min` old. Sees an OPEN position's drawdown.
      2. today's per-UTC-day breaker anchor (`daily_anchor_equity`, snapshotted
         by the bot on the first tick after every UTC midnight, flat or not),
         transfer-adjusted.
      3. rollover grace → None.
      4. the last fill's `equity_after`, transfer-adjusted (bot down).
    Intraday protection still lives in-process (daily-loss breaker + kill
    switch); this is the out-of-process view.

    TRANSFER NEUTRALISATION (2026-08-28). `daily_anchor_equity` is a RAW
    snapshot of equity at UTC midnight and never moves again that day, but
    `principal_anchor` is refreshed hourly from the income ledger. An intraday
    deposit therefore raises the DENOMINATOR while the numerator stays at its
    pre-deposit value, and the leg reads as a drawdown it never took. This
    fired for real: all three legs were topped up at 02:28 UTC on 2026-08-28
    and within 32 minutes donchian crossed into `warn` (it was actually UP)
    and sol_supertrend into `alert` (it was actually about flat). A withdrawal produces the mirror-image failure —
    a false all-clear — which is the more dangerous direction.

    The bot already solves this for its own daily-loss breaker in
    `bot._daily_book_anchor`: shift the raw anchor by the net principal moved
    since that anchor was set. We mirror the same computation at READ time —

        current = raw_anchor + (ledger_sum_now - daily_anchor_principal_sum)

    — and deliberately write nothing back. The stored anchor must stay RAW,
    because `_daily_book_anchor` applies this delta itself on every tick;
    pre-shifting the stored value would apply it twice and corrupt the live
    daily-loss breaker.

    Both sides come from ONE read of ONE table on ONE connection. The
    denominator is `principal_base + Σledger` — P exactly as
    `principal.get_principal()` derives it for the kill switch — rather than
    the cached `principal_anchor` copy, so the numerator and denominator can
    never be observed a reconcile apart from each other.

    Neither the shift nor the derived P is applied until `principal_source` is
    set, mirroring `principal.is_initialized()` and the bot's matching guard in
    `_daily_loss_blocks_entry`. During a first-run income backfill the ledger
    sum climbs from 0 to the entire deposit history while the seed is still
    unrecorded; treating that as an intraday transfer would shift the numerator
    by the whole principal.

    Fail-safe, matching the bot: a missing, unparseable or non-finite baseline,
    or a pre-Part-C DB with no `principal_ledger` table, yields delta 0 —
    exactly the previous behaviour. The shift applies only to the daily-anchor
    readings. `fills.equity_after` is already post-transfer whenever the fill
    is newer than the deposit; a fill OLDER than the deposit stays uncorrected.

    ROLLOVER GRACE (2026-09-03). The paragraph above used to end "...which is
    acceptable because that branch is only reached when today's daily anchor is
    missing, and it self-heals at the next UTC rollover." Both halves were
    wrong, and they cost a daily false alarm for six days.

    The fill branch is reached AT every rollover, not only when the anchor is
    absent: at 00:00 UTC `today` becomes the new date while `daily_anchor_date`
    still holds the old one, because only the bot writes that key
    (`bot._daily_anchor_equity`, then reached via `_maybe_enter` on a 60s poll
    for donchian/sol_supertrend; since 2026-10-13 via the per-tick
    `_roll_daily_anchor`, still on that poll). The monitor's `*/5` cron fires inside that gap,
    reads a stale pre-deposit fill, and alerts. The bot re-anchors seconds later
    and the next tick "recovers" — so the rollover is the CAUSE, not the cure.

    Observed 2026-08-29..09-03: donchian warned at 00:00 and recovered at
    00:05, every day, with byte-identical numbers; sol_supertrend did the same on
    the days its own poll happened to lose the race. Each leg's error equalled
    ITS OWN 2026-08-28 deposit — donchian 23.10 (+22), sol 19.71 (+20), v1
    21.00 (+21) — which is the signature of this branch specifically, the only
    one that skips the transfer shift. `daily_digest.py` imports this reader and
    ran an hour later on the same data: it saw the correct figures throughout.

    So for `grace_min` after midnight, a stale anchor yields (None, anchor) and
    `_check_equity` returns without alerting. Past that window the fill is used
    exactly as before. (Until 2026-10-13 a leg holding for days never
    re-anchored; now the bot anchors every midnight and writes `live_equity`,
    so the fill is reached only when the bot is down — and a stale fill is
    then the only reading there is, which is precisely when it matters most.)
    That fill is shifted by USDT principal moved AFTER its `ts` (ledger rows
    with `ts_ms > fill`), so a deposit made while the bot was down is not read
    as a drawdown; an unparseable `ts` or unreadable ledger falls back to the
    raw fill (logged), and a locked DB yields (None, None). The gate is on how
    long the anchor has been overdue, not on the
    branch: deleting the fallback would blind the check to the case it exists
    for. One tick of silence costs nothing — this is a day/trade-granularity
    check, and intraday protection lives in-process.
    """
    try:
        import sqlite3
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0) as conn:
            cur = conn.execute(
                "SELECT key, value FROM meta WHERE key IN "
                "('principal_anchor', 'deploy_start_equity', "
                "'daily_anchor_equity', 'daily_anchor_date', "
                "'daily_anchor_principal_sum', 'principal_base', "
                "'principal_source', 'live_equity')")
            rows = dict(cur.fetchall())
            fill_ts: str | None = None
            try:
                fill = conn.execute(
                    "SELECT equity_after, ts FROM fills WHERE equity_after IS NOT NULL "
                    "ORDER BY id DESC LIMIT 1").fetchone()
                if fill is not None:
                    fill_ts = fill[1]
            except sqlite3.OperationalError as e:
                if "no such column" not in str(e):
                    raise   # lock/corruption: outer handler → (None, None), not a stale fill
                # fills table without `ts` (old schema/tests)
                fill = conn.execute(
                    "SELECT equity_after FROM fills WHERE equity_after IS NOT NULL "
                    "ORDER BY id DESC LIMIT 1").fetchone()
            # Must match bot's state.principal_ledger_sum(PRINCIPAL_ASSET): USDT
            # rows ONLY. Non-USDT transfers are excluded from P as well, so
            # excluding them here keeps numerator and denominator in the same
            # units. Pre-Part-C DB (no table) → None → no neutralisation.
            ledger_sum: float | None = _usdt_ledger_sum(conn)
            # Mirror `principal.is_initialized()` / the bot's own guard in
            # `_daily_loss_blocks_entry`: until the ledger has been marked
            # initialised, a full-history backfill is still in flight and the sum
            # climbs from 0 to the whole deposit history. Treating that as an
            # intraday transfer would shift the numerator by the ENTIRE principal.
            # No neutralisation and no derived P until the seed is recorded.
            ledger_ready = rows.get("principal_source") is not None

            def _pos_float(key: str) -> float | None:
                """Parse one meta key defensively — a malformed value in one key
                must not invalidate the whole equity read (Sourcery, PR #6).

                `inf` must be rejected as well as unparseable text: it survives the
                `> 0` test, and an infinite reading propagates into `drop_pct` as
                NaN, which `_equity_band` classifies as `ok` — silently suppressing
                an alert rather than failing loudly (Sourcery, PR #25).
                """
                try:
                    v = float(rows.get(key))
                except (TypeError, ValueError):
                    return None
                return v if math.isfinite(v) and v > 0 else None

            def _adjusted(raw: float | None) -> float | None:
                """Raw daily anchor shifted by principal moved since it was set.

                Returns `raw` unchanged whenever the delta cannot be established,
                so this can only ever restore the pre-fix reading, never invent a
                correction from an unknown baseline.
                """
                if raw is None or ledger_sum is None or not ledger_ready:
                    return raw
                try:
                    baseline = float(rows.get("daily_anchor_principal_sum"))
                except (TypeError, ValueError):
                    # Anchor predates the baseline key; the bot seeds it lazily on
                    # its next tick. No baseline → no defensible delta → don't guess.
                    return raw
                if not math.isfinite(baseline) or not math.isfinite(ledger_sum):
                    # A non-finite delta yields NaN equity, and _equity_band scores
                    # NaN as `ok`. Refuse to compute rather than go quiet.
                    return raw
                return raw + (ledger_sum - baseline)

            # P as the KILL SWITCH computes it. `bot._check_kill_switch` calls
            # `principal.get_principal()`, which derives `principal_base + Σledger`
            # LIVE; the cached `principal_anchor` meta key is a convenience copy for
            # dashboards. Deriving it here from the SAME connection and the SAME
            # ledger read as the numerator means the two can never be measured a
            # reconcile apart from each other (Sourcery, PR #25), and makes the
            # monitor agree with the ceiling it is reporting against. `principal_base`
            # is legitimately 0.0 in income_backfill mode, so it cannot go through
            # `_pos_float`, which rejects non-positive values.
            derived_principal: float | None = None
            if ledger_ready and ledger_sum is not None and math.isfinite(ledger_sum):
                try:
                    base = float(rows.get("principal_base", 0.0) or 0.0)
                except (TypeError, ValueError):
                    base = None
                if base is not None and math.isfinite(base):
                    candidate = base + ledger_sum
                    if candidate > 0:
                        derived_principal = candidate

            anchor = (derived_principal or _pos_float("principal_anchor")
                      or _pos_float("deploy_start_equity"))
            if now is None:
                now = dt.datetime.now(dt.timezone.utc)
            today = now.strftime("%Y-%m-%d")
            daily = _pos_float("daily_anchor_equity")
            live_read = _fresh_live_equity(rows.get("live_equity"), now,
                                           live_max_age_min, ledger_sum)
            live, live_trusted = live_read if live_read else (None, False)
            if live is not None and live_trusted:
                # LIVE EQUITY (2026-10-13). The bot's own wallet + unrealized from
                # its last tick, so an OPEN position's drawdown is visible (before
                # this, a hold read the entry fill: sol sat frozen at 97.39 from
                # 10-07). Already post-transfer — NOT passed through _adjusted or
                # shifted by fill transfers, which would double-count. Valid inside
                # the rollover grace too: it is not a stale snapshot.
                current: float | None = live
            elif rows.get("daily_anchor_date") == today and daily is not None:
                current = _adjusted(daily)
            elif _mins_since_utc_midnight(now) < grace_min:
                # ROLLOVER GRACE — see docstring. The UTC day just turned over and
                # the bot has not re-anchored yet. Every fallback below is stale by
                # at least a day here, and the fill may predate a transfer, so
                # report nothing rather than something wrong.
                current = None
            elif fill is not None and fill[0] is not None:
                # equity_after of 0.0 is a VALID (and alarming) reading — do not
                # truthiness-filter it away. Transfer-adjusted below: see docstring.
                try:
                    current = float(fill[0])
                    # MULTI-DAY HOLD (2026-10-13), a PREDICTED residual: not seen
                    # live (no 00:10-00:20 warns in Sep/Oct; the 09-01..03 warns
                    # were the 00:00 rollover race, fixed by the grace). Since the
                    # bot now snapshots the anchor at every midnight and writes
                    # live equity each tick, this branch is reached only when the
                    # bot is DOWN (live_equity stale AND today's anchor missing).
                    # A fill older than a deposit/withdrawal then reads as a phantom
                    # move. Replay on live-DB copies at a simulated 10-14 00:15:
                    # donchian flipped sign old vs new (a 10-01 deposit after
                    # the 09-23 fill). Shift by USDT principal moved AFTER the fill
                    # — the same read-time neutralisation as `_adjusted`; nothing
                    # is written back. Unknown ts / no ledger / not ready → raw fill.
                    # Computed lazily: only this branch needs it.
                    fill_transfers: float | None = None
                    if ledger_ready and fill_ts is not None:
                        try:
                            fill_ms = (dt.datetime.fromisoformat(str(fill_ts)).timestamp()
                                       * 1000.0)
                            fill_transfers = _usdt_ledger_sum(conn, after_ms=fill_ms)
                        except (ValueError, TypeError):
                            fill_transfers = None   # unparseable ts → raw fill
                    if (fill_transfers is not None and math.isfinite(fill_transfers)
                            and math.isfinite(current)):
                        current += fill_transfers
                except (TypeError, ValueError):
                    current = _adjusted(daily)
            else:
                current = _adjusted(daily)
            if live is not None and not live_trusted:
                # DOUBTFUL LIVE SAMPLE (possible unreconciled transfer). Never
                # let the doubt HIDE a drawdown: take the LOWER of the fallback
                # and live. A deposit (live too high) → fallback, no false
                # recovery; a real crash (live lower) → live, alerted on time;
                # a withdrawal can at worst cause one early warn.
                current = live if current is None else min(current, live)
            return current, anchor
    except Exception as e:
        log.warning("monitor: db read failed for %s: %s", db_path, e)
        return None, None


def _losing_streak(db_path: Path) -> int | None:
    """Current run of consecutive losing closed trades, or None on read failure.

    Recomputed from scratch on every call (no incremental counter), so it is
    idempotent across re-runs and restarts.

    ESTIMATE, not exchange truth: local fills, gross PnL, strategy exits only.
    Restart/HALT/kill closes and pre-23-Aug NULL-pnl closes are invisible, so
    the count can run late or early. See docs/CONSECUTIVE_LOSS_BREAKER_PLAN.md.

    Source: `fills` rows with side='close', in id order. Those are written only
    for strategy exits (bracket_exit, time_stop, breakeven_exit, channel/flip
    exits). Infrastructure closes (boot_flatten, HALT, kill) write NO fills row
    — they are events only — so they are excluded by construction.

    Classification by `pnl_usd` (gross; see bot._exit_pnl_usd):
      < 0   loss      -> extends the streak
      > 0   win       -> resets it to 0
      == 0 or NULL    -> breakeven / unknown: NEUTRAL, neither extends nor
                         resets. A trade whose outcome we cannot establish
                         must not manufacture or erase a streak.
    """
    try:
        import sqlite3
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0) as conn:
            rows = conn.execute(
                "SELECT pnl_usd FROM fills WHERE side='close' ORDER BY id").fetchall()
    except Exception as e:
        log.warning("monitor: streak read failed for %s: %s", db_path, e)
        return None
    streak = 0
    for (pnl,) in rows:
        try:
            v = float(pnl)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or v == 0:
            continue
        streak = streak + 1 if v < 0 else 0
    return streak


def _check_streak(name: str, db_path: Path, cfg: dict[str, Any],
                  state: dict[str, Any]) -> None:
    """ALERT-ONLY. Alert when a leg's losing streak reaches its threshold, then
    once more at each +3 beyond it (threshold+3, +6, ...). A win re-arms fully.

    Touches nothing but monitor state and the alert path: no HALT file, no
    bot state write, no risk change.

    The latch `state["streak_alerted"][leg]` is the LAST ALERTED LEVEL (int),
    committed only after a successful send so an SMTP failure retries next
    tick. If the streak jumps past several levels between ticks, one alert is
    sent and the latch takes the highest level reached. A legacy bool latch
    (True) is migrated to the threshold level.

    The count is an ESTIMATE from local fills (see _losing_streak), so it can
    run late or early versus the exchange.
    """
    threshold = (cfg.get("streak_alert_thresholds") or {}).get(name)
    if not threshold:
        return
    threshold = int(threshold)
    streak = _losing_streak(db_path)
    if streak is None:
        return
    latched = state.setdefault("streak_alerted", {})
    if streak < threshold:
        latched.pop(name, None)   # streak broke (or never reached it): re-arm
        return
    last = latched.get(name)
    if isinstance(last, bool):    # legacy bool latch
        last = threshold if last else None
    if last is not None and not isinstance(last, int):
        last = None
    if last is not None and last > streak:
        last = None               # stale latch from a streak that has since reset
    level = threshold + 3 * ((streak - threshold) // 3)
    if last is not None:
        latched[name] = last      # persist migration
        if level <= last:
            return
    sent = _emit(
        f"LOSING STREAK {streak}: {name}",
        f"{name} has an estimated {streak} consecutive losing closed trades "
        f"(alert level {level}; threshold {threshold} = longest backtest streak + 1).\n"
        f"Estimated from local fills (gross PnL, strategy exits only; "
        f"restart/HALT/kill closes and pre-23-Aug NULL-pnl closes are invisible), "
        f"so it can run late or early.\n"
        f"This is worse than anything in the backtest history. It does NOT "
        f"block trading; re-run the live-vs-backtest parity review.\n"
        f"See docs/CONSECUTIVE_LOSS_BREAKER_PLAN.md. Next alert at {level + 3}, "
        f"or after a win resets the streak.",
        state, kind=f"streak:{name}", cooldown_min=0, force=True,
    )
    if sent:
        latched[name] = level


def _check_leg(leg: dict[str, str], cfg: dict[str, Any], state: dict[str, Any]) -> None:
    name = leg["name"]
    hb_path = DATA / leg["heartbeat"]
    log_path = LOGS / leg["log"]
    db_path = DATA / leg["state"]
    unit = leg["systemd"]
    cooldown = int(cfg["alert_cooldown_min"])

    # 1. systemd unit state
    if not _systemd_active(unit):
        _emit(
            f"DOWN: {unit}",
            f"systemd reports {unit} is not active. Check `systemctl status {unit}` on the droplet.",
            state, kind=f"systemd:{name}", cooldown_min=cooldown,
        )

    # 2. heartbeat age
    age = _heartbeat_age_s(hb_path)
    if age is None:
        _emit(
            f"NO HEARTBEAT: {name}",
            f"Heartbeat file {hb_path.name} missing. Bot may not have started.",
            state, kind=f"hb_missing:{name}", cooldown_min=cooldown,
        )
    elif age > int(cfg["heartbeat_stale_s"]):
        _emit(
            f"STALE HEARTBEAT: {name} ({age}s)",
            f"Heartbeat for {name} is {age}s old (threshold {cfg['heartbeat_stale_s']}s).\n"
            f"Bot may be hung or crashed.",
            state, kind=f"hb_stale:{name}", cooldown_min=cooldown,
        )

    # 3. log event scan
    offset_key = f"offset:{name}"
    since = int(state["log_offsets"].get(offset_key, 0))
    counts, new_offset = _count_log_events(
        log_path, since,
        {
            "ERROR":      '"level":"ERROR"',
            "TRACEBACK":  "Traceback (most recent call last)",
            "NAN_4H":     "mtf_4h_gate_nan",
            "KILL_SWITCH":"KILL SWITCH",
        },
    )
    state["log_offsets"][offset_key] = new_offset

    if counts["ERROR"] > 0 and cfg.get("error_alert"):
        _emit(
            f"ERROR x{counts['ERROR']}: {name}",
            f"{counts['ERROR']} ERROR-level events in {log_path.name} since last poll. "
            f"Investigate `/diagnose` or tail the log.",
            state, kind=f"errors:{name}", cooldown_min=cooldown,
        )
    if counts["TRACEBACK"] > 0:
        _emit(
            f"TRACEBACK: {name}",
            f"{counts['TRACEBACK']} Python tracebacks in {log_path.name}. Unhandled exception.",
            state, kind=f"trace:{name}", cooldown_min=cooldown,
        )
    if counts["NAN_4H"] > 0 and cfg.get("nan_4h_gate_alert"):
        _emit(
            f"4H GATE NAN: {name}",
            f"{counts['NAN_4H']} mtf_4h_gate_nan events. The 4H EMA200 isn't being computed; "
            f"signals are being silently skipped. Check 4H parquet feed.",
            state, kind=f"nan_4h:{name}", cooldown_min=cooldown,
        )
    if counts["KILL_SWITCH"] > 0:
        _emit(
            f"KILL SWITCH FIRED: {name}",
            f"{counts['KILL_SWITCH']} kill-switch events in {log_path.name}. "
            f"Bot has flattened positions and exited. data/HALT may now exist.",
            state, kind=f"kill:{name}", cooldown_min=cooldown,
        )

    # 4. equity check (LIVE only — for DRY legs, balance changes are paper)
    if leg.get("live") and db_path.exists():
        _check_equity(name, db_path, cfg, state)
        try:
            _check_streak(name, db_path, cfg, state)
        except Exception:
            log.error("monitor: streak check crashed for %s:\n%s", name, traceback.format_exc())


def _equity_band(drop_pct: float, cfg: dict[str, Any], prev: str | None = None) -> str:
    """Classify a drawdown-from-anchor into ok / warn / alert.

    HYSTERESIS (2026-10-13). With live equity a hold hovering at the line
    would flip bands every */5 cron, and each flip mails (force=True). Given
    the previous band, stay in it until the drop retreats
    `equity_band_hysteresis_pct` (default 1 pt) PAST its threshold: e.g. leave
    warn only once drop < warn − 1. Entering a worse band is unchanged.
    """
    h = float(cfg.get("equity_band_hysteresis_pct",
                      DEFAULTS.get("equity_band_hysteresis_pct", 0.0)))
    alert_thr = float(cfg["equity_drop_alert_pct"]) - (h if prev == "alert" else 0.0)
    warn_thr = float(cfg["equity_drop_warn_pct"]) - (h if prev in ("warn", "alert") else 0.0)
    if drop_pct >= alert_thr:
        return "alert"
    if drop_pct >= warn_thr:
        return "warn"
    return "ok"


def _kill_switch_text(name: str) -> str:
    """'Bot kill switch flattens at -X% from principal.' with X computed from
    the leg's params YAML (deploy.kill_switch_equity_fraction) — the threshold
    the bot's _check_kill_switch enforces. Read at send time, never hardcoded.
    If the YAML or key can't be read, or the fraction is not in (0, 1), the
    sentence names the key instead of inventing a number. It quotes the YAML
    ON DISK, which matches what the bot enforces once the leg has been
    restarted onto it (tools/leg_safe_restart.py); between an edit and that
    restart the mail can name the new value while the bot holds the old one."""
    frac = None
    fname = LEG_PARAMS_FILES.get(name)
    if fname and yaml is not None:
        try:
            raw = yaml.safe_load((PARAMS_DIR / fname).read_text()) or {}
            v = float((raw.get("deploy") or {})["kill_switch_equity_fraction"])
            frac = v if 0.0 < v < 1.0 else None
        except Exception:  # noqa: BLE001 — a mail must still go out
            frac = None
    if frac is None:
        return ("Bot kill switch flattens at the leg's configured "
                "kill_switch_equity_fraction of principal.")
    return f"Bot kill switch flattens at -{(1.0 - frac) * 100:.1f}% from principal."


def _check_equity(name: str, db_path: Path, cfg: dict[str, Any],
                  state: dict[str, Any]) -> None:
    """Alert on equity-band TRANSITIONS rather than every tick the band holds.

    Equity-vs-anchor is a state, not an event. The previous version re-emitted
    on the 30-min cooldown for as long as the condition persisted — v1 sat
    below its anchor for 30 hours and sent 52 identical warns, which is how a
    real alert gets buried. Behaviour now:

      - crossing into warn/alert emits once,
      - holding that band is silent, except `alert` repeats every
        `equity_alert_reminder_hours` so a sustained drawdown can't go quiet,
      - returning to ok emits a recovery notice,
      - a non-ok band seen with no prior history still emits, so a problem that
        already exists at deploy time is never silently swallowed.

    The band is committed only after a successful send, so an SMTP failure is
    retried on the next tick instead of being lost.
    """
    cur, anchor = _equity_from_db(
        db_path,
        float(cfg.get("equity_anchor_grace_min",
                      DEFAULTS["equity_anchor_grace_min"])),
    )
    # (None, None) = the DB read itself failed (locked/corrupt/missing). Rollover
    # grace returns (None, anchor) and is a legitimate silence, not counted.
    fails = state.setdefault("equity_read_fails", {})
    if cur is None and anchor is None:
        n = int(fails.get(name, 0)) + 1
        fails[name] = n
        limit = int(cfg.get("equity_read_fail_alert_runs",
                            DEFAULTS["equity_read_fail_alert_runs"]))
        sent_flags = state.setdefault("equity_blind_sent", {})
        if n >= limit and not sent_flags.get(name):   # once per streak; retried if send fails
            sent_flags[name] = _emit(
                f"EQUITY CHECK BLIND: {name}",
                f"The equity DB read has failed on {n} consecutive monitor runs "
                f"(~{n * 5} min). Drawdown alerts for {name} are NOT being "
                f"evaluated. Check the state DB (lock/corruption) and "
                f"monitor logs.",
                state, kind=f"equity_blind:{name}", cooldown_min=0, force=True,
            )
        return
    if fails.get(name):
        if state.setdefault("equity_blind_sent", {}).pop(name, False):
            _emit(
                f"EQUITY CHECK RECOVERED: {name}",
                f"The equity DB read works again after {fails[name]} failed runs.",
                state, kind=f"equity_blind_ok:{name}", cooldown_min=0, force=True,
            )
        fails[name] = 0
    if cur is None or anchor is None or anchor <= 0:
        return
    drop_pct = (1 - cur / anchor) * 100
    bands = state.setdefault("equity_bands", {})
    prev = bands.get(name)
    band = _equity_band(drop_pct, cfg, prev)
    # Signed DELTA, not the drop. A hardcoded "-" printed the recovery notice as
    # "= --6.39%" whenever equity was above anchor — a double minus reporting a
    # 6.4% GAIN, on the very mail meant to reassure. Matches daily_digest's
    # "{delta_pct:+.2f}%".
    position = f"Equity ${cur:.2f} vs anchor ${anchor:.2f} = {-drop_pct:+.2f}%."

    if band == prev:
        if band == "alert":
            _emit(
                f"EQUITY DROP {drop_pct:.1f}%: {name} (ongoing)",
                f"{position} Still past the {cfg['equity_drop_alert_pct']}% alert line. "
                f"{_kill_switch_text(name)}",
                state, kind=f"equity_alert:{name}",
                cooldown_min=int(float(cfg["equity_alert_reminder_hours"]) * 60),
            )
        return

    if band == "ok":
        if prev is None:
            bands[name] = band  # healthy and nothing to compare against
            return
        sent = _emit(
            f"EQUITY RECOVERED: {name}",
            f"{position} Back inside the {cfg['equity_drop_warn_pct']}% warn threshold.",
            state, kind=f"equity_ok:{name}", cooldown_min=0, force=True,
        )
    elif band == "alert":
        sent = _emit(
            f"EQUITY DROP {drop_pct:.1f}%: {name}",
            f"{position} {_kill_switch_text(name)} "
            f"Investigate immediately.",
            state, kind=f"equity_alert:{name}", cooldown_min=0, force=True,
        )
    else:
        origin = ("improving — was past the alert line" if prev == "alert"
                  else "newly below the warn line")
        sent = _emit(
            f"EQUITY WARN {drop_pct:.1f}%: {name}",
            f"{position} ({origin}) No further warns unless this changes band.",
            state, kind=f"equity_warn:{name}", cooldown_min=0, force=True,
        )
    if sent:
        bands[name] = band


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not is_configured():
        log.warning("monitor: SMTP not configured; running checks but cannot alert")

    cfg = _load_config()
    state = _load_state()

    for leg in LEGS:
        try:
            _check_leg(leg, cfg, state)
        except Exception:
            log.error("monitor: leg %s check crashed:\n%s", leg["name"], traceback.format_exc())

    _save_state(state)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Cron output goes to mail; print so the operator sees it via mail/cron
        sys.stderr.write(traceback.format_exc())
        sys.exit(0)  # never let cron retry; we don't want a flapping monitor
