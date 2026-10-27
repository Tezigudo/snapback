"""Snapback-BTC live trading daemon.

Strategy: multifactor-v1 (see config/params.yaml).
Modes:
  - LIVE   (default): real orders against the configured exchange env.
  - DRY-RUN (--dry-run or DRY_RUN=1): fetches real market+balance data,
    evaluates signals, logs everything it WOULD do, but never calls create_order.

Binance Futures testnet/sandbox is no longer supported by ccxt for private
endpoints (deprecated late 2025). Run dry-run first against real account to
verify wiring before flipping to live orders.

Loop (every poll_interval_s):
  1. heartbeat: touch data/heartbeat
  2. check data/HALT → if present, flatten + exit
  3. fetch equity → if equity / deploy_start < kill_switch_fraction → HALT
  4. fetch latest 15m bars + funding
  5. if no open position AND signal fires on last closed bar → place bracket
  6. time-stop: close positions older than max_hold_bars
  7. log to logs/bot.jsonl + data/state.db

DOES NOT:
  - run any LLM call
  - place orders unless all gates pass
  - touch mainnet without confirm_mainnet.lock present
  - place orders that fail exchange minimums (min qty / min notional)

USAGE:
  uv run python -m bot --dry-run     # SAFE: no orders placed
  uv run python -m bot                # live (real orders)
  # Ctrl+C to stop cleanly. Or: touch data/HALT in another terminal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import signal
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import ccxt
import pandas as pd
import yaml

import risk
from alerts import send_alert
from bot_internals import (
    BracketState,
    SignalDecision,
    adopt_bracket_state,
    adopt_loop_guard,
    adopt_precheck,
    breakeven_due,
    breakeven_stop_price,
    can_adopt,
    channel_exit_signal,  # noqa: F401  (re-exported; tools import it from here)
    evaluate_for_strategy,
    gate_status,
    limit_entry_price,
    order_avg_price,
    require_strategy_available,
    resolve_strategy_name,
    signal_bar_seed,
    stale_root_legs,
    strategy_uses_channel_exit,
    strategy_uses_trend_exit,
    time_stop_due,
    trend_exit_fill_reason,
    trend_exit_signal,
)
from config_check import EXIT_CONFIG, validate_params
from exchange import principal, state, trade_events
from exchange.binance_client import (
    BinanceClient,
    EquityUnavailable,
    UnprotectedFillError,
    is_nonce_error,
    is_transient_error,
    is_would_trigger,
)
from exchange.constraints import (
    DEFAULT_CONSTRAINTS,
    ExchangeConstraints,
    fallbacks_for_symbol,
    merge_with_live,
    passes_minimums,
    round_qty_down,
)
from exchange.env import (
    REPO_ROOT,
    get_env,
    halt_source,
    is_halted,
    leg_halt_path,
    leg_pause_path,
    load_env_for_instance,
)
from risk import (
    RiskBreach,
    check_daily_loss,
    check_leverage,
    check_notional,
    check_symbol,
)

# Re-export for tools/preflight_live.py and any downstream importers.
from strategy.live_multifactor_v1 import evaluate_signal  # noqa: F401
from tools import consolidate_push

LOG_DIR = REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "bot.jsonl"
HEARTBEAT = REPO_ROOT / "data" / "heartbeat"
# CONFIG_PATH default — overridable via --config CLI flag (or --instance).
CONFIG_PATH = REPO_ROOT / "config" / "params.yaml"

# Named instance profiles for the multi-leg deploy. Picking `--instance donchian`
# derives all four paths (config / state.db / log / heartbeat) at once so the
# systemd unit and tmux commands don't have to spell out four flags each.
# Individual --config / --state-db / --log-file / --heartbeat overrides still
# work and take precedence — useful for one-off experiments.
INSTANCE_PROFILES: dict[str, dict[str, Path]] = {
    "v1": {
        "config":    REPO_ROOT / "config" / "params.yaml",
        "state_db":  REPO_ROOT / "data" / "state.db",
        "log_file":  REPO_ROOT / "logs" / "bot.jsonl",
        "heartbeat": REPO_ROOT / "data" / "heartbeat",
    },
    "donchian": {
        "config":    REPO_ROOT / "config" / "params_donchian.yaml",
        "state_db":  REPO_ROOT / "data" / "state_donchian.db",
        "log_file":  REPO_ROOT / "logs" / "donchian.jsonl",
        "heartbeat": REPO_ROOT / "data" / "heartbeat_donchian",
    },
    "cnh_short": {
        "config":    REPO_ROOT / "config" / "params_cnh_hybrid_short.yaml",
        "state_db":  REPO_ROOT / "data" / "state_cnh_short.db",
        "log_file":  REPO_ROOT / "logs" / "cnh_short.jsonl",
        "heartbeat": REPO_ROOT / "data" / "heartbeat_cnh_short",
    },
    # NEW LEG (2026-05-30): SOL cnh-hybrid-short. Validated Phase 1-5 on SOL
    # (memory: snapback-sol-hybrid-short-phase2to5). BLOCKED until risk.py
    # ALLOWED_SYMBOLS includes "SOL/USDT:USDT" — check_symbol() runs at startup
    # so this gate blocks even --dry-run. Tier-3 edit (RISK_REVIEW=1 + user OK).
    "cnh_short_sol": {
        "config":    REPO_ROOT / "config" / "params_cnh_hybrid_short_sol.yaml",
        "state_db":  REPO_ROOT / "data" / "state_cnh_short_sol.db",
        "log_file":  REPO_ROOT / "logs" / "cnh_short_sol.jsonl",
        "heartbeat": REPO_ROOT / "data" / "heartbeat_cnh_short_sol",
    },
    # NEW LEG (2026-07-25): SOL supertrend — replaces the never-funded
    # cnh_short slot per God's decision. Winner of the round-3 win-rate-blended
    # walk-forward (9/9 OOS folds positive); see SOL_LEG_VERDICT.md.
    # risk.py ALLOWED_SYMBOLS now includes SOL/USDT:USDT (RISK_REVIEW 2026-07-25).
    # Needs .env.sol_supertrend with its OWN sub-account key — there is no
    # cnh_short key to inherit (that leg was never keyed), and booting without
    # the file is refused outright, see _main()'s per-instance env guard.
    "sol_supertrend": {
        "config":    REPO_ROOT / "config" / "params_sol_supertrend.yaml",
        "state_db":  REPO_ROOT / "data" / "state_sol_supertrend.db",
        "log_file":  REPO_ROOT / "logs" / "sol_supertrend.jsonl",
        "heartbeat": REPO_ROOT / "data" / "heartbeat_sol_supertrend",
    },
}

# Derive each leg's SELF-halt flag (data/HALT_<instance>) the same way as the
# other per-leg paths, from a single source of truth (env.leg_halt_path). The
# shared data/HALT stays a GLOBAL manual stop-all and is NOT in the profile —
# no leg's kill switch may write it (that shared write is what caused the 07-01
# cascade). See exchange/env.py :: is_halted / leg_halt_path.
for _name, _profile in INSTANCE_PROFILES.items():
    _profile["halt"] = leg_halt_path(_name)

# Boot-resume HARD ARMING INTERLOCK. Adoption can only be armed when BOTH this
# constant is True AND a leg's config sets `boot_resume.observe_only: false`,
# so arming can never be a one-line config edit. The prerequisites from
# docs/BOOT_RESUME_PHASE_B_PLAN.md ("Blockers before arming") are now BUILT and
# tested (tests/test_boot_resume_armed_path.py):
#   C5  require active_bracket.signal_id == latest entry coid root (root=None refuses)
#   C6  re-read the position immediately before committing to adopt
#   C9  bounded retry of an unreadable order book at boot (3 x, 5 s apart)
#   C10 seed the reprotect cap-alert latch on adopt
#   C11 send_alert on every adopt ("Position RESUMED after restart")
#   B3  a missing STOP is never adoptable; B4 plain legs scoped to our prefix
#   B5  adopt-loop guard is per position, never reset on time; B8 T9/T10 tests
# BUILT IS NOT APPROVED. Flipping this is God's decision, after the probe
# evidence gate (plan §2). While False, `observe_only: false` is logged at
# ERROR, alerted, and treated as observe_only (the position flattens).
ARMING_PREREQS_BUILT = False

# C9 (boot-resume, ARMED path only): an unreadable order book at boot is
# retried this many times in total, this many seconds apart, before it counts
# as a refusal (→ flatten). Sized for the post-reboot network flap after
# network-online.target. Observe mode never reads the books, so it never waits.
BOOT_BOOK_READ_ATTEMPTS = 3
BOOT_BOOK_RETRY_S = 5.0


def _boot_book_sleep(seconds: float) -> None:
    """Indirection so tests can patch the C9 back-off without patching time."""
    time.sleep(seconds)


# Console logs display in Bangkok time (GMT+7) for human readability.
# JSONL `ts` field and state.db remain UTC for alignment with Binance candles.
LOCAL_TZ = ZoneInfo("Asia/Bangkok")


class _LocalTimeFormatter(logging.Formatter):
    """Formats `%(asctime)s` in Asia/Bangkok local time."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=LOCAL_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S %z")


# Ledger rows that change a position's size without being an entry or an exit:
# manual reduce / add, and a partial fill of one of OUR orders seen mid-way.
_SIZE_ROWS_SQL = "('external_partial', 'external_add', 'bot_partial')"


def _replay_size_rows(qty: float, basis: float, rows) -> tuple[float, float]:
    """(open qty, cost basis) after replaying size-change rows in id order,
    like the exchange's running average: a reduction keeps the basis, an add
    re-weights it. `rows` are (reason, qty, price)."""
    for r, q, p in rows:
        q, p = float(q), float(p)
        if r == "external_add":
            basis = (qty * basis + q * p) / (qty + q)
            qty += q
        else:
            qty -= q
    return qty, basis


def _exit_pnl_usd(side: str, entry_price: float | None,
                  exit_price: float | None, qty: float) -> float | None:
    """GROSS realised PnL for a closed position, or None if it cannot be known.

    Deliberately gross (no fees/funding), because that is the convention the
    bracket-exit path at _detect_bracket_exit already writes into the same
    `fills.pnl_usd` column, and one column must mean one thing. Expect it to
    read a few tenths of a percent above the equity ledger (the difference
    being round-trip fees). For true net PnL, difference consecutive
    `equity_after` values instead.
    """
    if not entry_price or not exit_price or not qty:
        return None
    direction = 1.0 if side == "long" else -1.0
    return round((float(exit_price) - float(entry_price)) * float(qty) * direction, 4)


def _rel_to_repo(p: Path) -> str:
    """Repo-relative form of `p` for the startup banner, absolute if outside.

    The state DB, log file and heartbeat are all env-overridable
    (SNAPBACK_STATE_DB / LOG_DIR / …) and those overrides accept ABSOLUTE
    paths pointing anywhere on the box. `Path.relative_to` raises ValueError in
    that case, which would take down boot() over a cosmetic banner line.

    Found 2026-08-23 by the new tests/conftest.py isolation fixture, which
    points the state DB at a pytest tmp dir — the first time anything had ever
    run this code with a DB outside the repo. Latent in production only because
    both live legs happen to use data/*.db.
    """
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _setup_logging(level: str = "INFO") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("snapback")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_LocalTimeFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(sh)

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "ts": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload, default=str)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)
    return logger


def load_params(path: str | None = None) -> dict:
    """Load YAML params. Default path = config/params.yaml. Override via --config."""
    p = path or CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def compute_qty(equity: float, price: float, sl_pct: float,
                risk_pct: float, leverage: int) -> float:
    """Risk-based sizing matching backtest: target_btc = risk / sl_distance.

    Kept for backward-compat with multifactor-v1 (fixed-pct SL).
    """
    if price <= 0 or sl_pct <= 0:
        return 0.0
    return compute_qty_from_distance(equity, price, sl_pct * price, risk_pct, leverage)


def compute_qty_from_distance(equity: float, price: float, sl_distance: float,
                              risk_pct: float, leverage: int) -> float:
    """Sizing with sl_distance in PRICE units (works for ATR-based stops)."""
    if price <= 0 or sl_distance <= 0:
        return 0.0
    risk_amount = equity * (risk_pct / 100.0)
    target = risk_amount / sl_distance
    cap = (equity * leverage * 0.95) / price
    return max(min(target, cap), 0.0)


# --- Main loop ---------------------------------------------------------------
class Bot:
    _principal_unset_alerted: bool = False   # class default: tests build Bot via __new__
    def __init__(self, params: dict, dry_run: bool = False,
                 instance: str = "v1") -> None:
        self.params = params
        self.symbol = params["symbol"]
        # "SOL/USDT:USDT" -> "SOL". Alerts used to hardcode "BTC", so the
        # sol_supertrend leg emailed "Closed long 1.2000 BTC" for a SOL trade.
        self.base_asset = str(self.symbol).split("/")[0]
        self.strategy_name = resolve_strategy_name(params)
        self.dry_run = dry_run
        self.instance = instance
        # This leg's SELF-halt flag. The kill switch touches ONLY this file, so
        # a leg self-halting can never take a sibling down.
        self.halt_path = leg_halt_path(instance)
        self.log = logging.getLogger("snapback.bot")
        # Hedge mode + clientOrderId prefix come from `hedge` block in params YAML.
        # When unset, behaves exactly like the legacy single-bot deploy.
        hedge_cfg = params.get("hedge", {}) or {}
        self.hedge_enabled = bool(hedge_cfg.get("enabled", False))
        self.coid_prefix = str(hedge_cfg.get("client_order_id_prefix", "snap-v1-"))
        self.client = BinanceClient.from_env(
            hedge_mode=self.hedge_enabled,
            coid_prefix=self.coid_prefix,
        )
        exec_cfg = params.get("execution", {})
        self.poll_s = float(exec_cfg["poll_interval_s"])
        self.order_type = str(exec_cfg.get("order_type", "market")).lower()
        self.limit_offset_bps = float(exec_cfg.get("limit_offset_bps", 0.0))
        self.limit_timeout_s = float(exec_cfg.get("limit_timeout_s", 20.0))
        # Entry timeframe from the params YAML — drives both fetch_ohlcv and
        # the time-stop window. v1=15m, donchian=4h, cnh_short=4h. Defaulting
        # to 15m preserves legacy behavior for any config that omits the block.
        self.entry_tf = str((params.get("timeframes") or {}).get("entry", "15m"))
        # parse_timeframe returns seconds — ccxt's standard converter.
        self.bar_seconds = int(self.client.ex.parse_timeframe(self.entry_tf))
        self.leverage = int(params["sizing"]["leverage"])
        self.risk_pct = float(params["sizing"]["risk_per_trade_pct"])
        self.kill_fraction = float(params["deploy"]["kill_switch_equity_fraction"])
        # Min-capital warning threshold (required; config_check.REQUIRED_KEYS).
        self.min_capital_warn = float(params["deploy"]["min_capital_warn_usdt"])
        self.constraints: ExchangeConstraints = DEFAULT_CONSTRAINTS
        self._stopped = False
        self._last_signal_ts: pd.Timestamp | None = None
        self._entries_paused_seen = False
        # B2 (boot-resume): prepared by _boot_resume_verdict on the ARMED adopt
        # path only, consumed by _seed_adopted_tracking. None otherwise.
        self._adopt_signal_ts_seed: pd.Timestamp | None = None
        # C5: the entry coid root the gate PROVED this position belongs to.
        # Read ONCE by _can_adopt (armed path), carried through _prepare_adopt
        # into boot()'s adopt branch, so the branch never re-reads it — a
        # re-read that failed there could only fall back to an untagged root.
        self._adopt_root: str | None = None
        # The ARMED verdict's reason, for the C11 resume alert; and whether the
        # adopted position's reprotect budget is already spent (C10).
        self._adopt_reason: str | None = None
        self._adopt_reprotect_capped: bool = False
        # Push to consolidate every PUSH_INTERVAL_S; also send a heartbeat
        # event at the same cadence so the dashboard's "alive" check works.
        # 30s is well under consolidate's 60s healthy-threshold.
        self._push_interval_s = 30.0
        self._last_push_ts: float = 0.0
        # Most recent gate-status snapshot from the live evaluator. Updated on
        # every bar-close evaluation. Pushed to consolidate as part of the
        # heartbeat payload so the dashboard can show "what's true now / waiting
        # on what" without you SSHing in.
        self._latest_gates: dict | None = None
        self._last_waiting_for: str | None = None
        # Last observed position side, used by _detect_bracket_exit to spot
        # open→flat transitions. "unknown" until the first loop call so we
        # don't emit a spurious exit alert for a historical entry in state.db.
        self._last_position_side: str = "unknown"
        # ...and the rest of that position's IDENTITY, because the side alone
        # cannot see a same-poll replacement. Live on 2026-08-23: v1's bracket
        # SL filled at 05:14:30.039 and _maybe_enter re-entered at 05:14:31.290
        # — 1.25 s later, inside one 5 s poll. The loop therefore never observed
        # `flat`, the open→flat edge never fired, and the exit was dropped on
        # the floor: no `close` row in fills, no `exit` bot_event, no alert, and
        # a loss missing from the ledger until it was backfilled by hand weeks
        # later. `entry_price` is the discriminator — side and qty are usually
        # identical across a re-entry, and it can ONLY move when the position is
        # replaced, because _maybe_enter returns early unless flat (there is no
        # scale-in or averaging anywhere in this bot).
        self._last_position_entry: float | None = None
        self._last_position_qty: float | None = None
        # The coid root of the entry that opened the position we are tracking.
        # Captured when we first observe it open, because by the next tick
        # state.latest_entry_coid_root() already names the REPLACEMENT entry.
        self._last_entry_root: str | None = None
        # external size-change debounce / retry state (_note_external_size_change)
        self._ext_pending: float | None = None
        self._ext_attempts: int = 0
        # post-reduction qty that was still pending when the flat edge fired
        self._flat_pending_q: float | None = None
        # Consecutive ticks a detected exit has failed to be RECORDED (see
        # _hold_unrecorded_exit). Reset on success and on giving up.
        self._exit_retry_n: int = 0
        # Throttle bracket re-placement (see _maybe_reprotect) so a persistent
        # placement failure can't spam orders/alerts every poll.
        self._last_reprotect_ts: float = 0.0
        # Latch so the daily-loss breaker logs/alerts once per trip, not every
        # poll. Cleared on drawdown recovery or UTC-day rollover.
        self._daily_loss_blocked: bool = False
        # Principal-anchor reconcile cadence. The kill switch compares equity to
        # NET DEPOSITED PRINCIPAL (exchange/principal.py), refreshed from the
        # Binance income ledger. DCA is monthly so hourly is ample once the
        # anchor exists; retry faster while it is still uninitialised.
        self._principal_reconcile_interval_s: float = 3600.0
        self._principal_retry_interval_s: float = 60.0
        self._last_principal_reconcile_ts: float = 0.0
        # Epoch s of the last `live_equity` meta write (see _persist_live_equity).
        self._last_live_equity_write: float | None = None
        # Unreconciled-transfer detector state (see _persist_live_equity).
        self._prev_tick_equity: float | None = None
        self._prev_tick_fingerprint: tuple | None = None
        self._prev_tick_wallet: float | None = None
        self._transfer_suspect: dict | None = None
        # One alert per process when the per-position re-place cap is hit — the
        # cap itself is persisted in active_bracket; this only de-dupes the mail.
        self._reprotect_capped_alerted: bool = False
        # Alert de-dupe: unknown-equity mail is rate-limited (epoch s of last
        # send); the "kill switch inert, principal unset" mail is once per unset spell.
        self._equity_unknown_alert_ts: float = 0.0
        self._equity_unknown_streak: int = 0
        self._last_known_equity: float | None = None
        # In-memory copy of meta no_stop_close_pending (see _finish_no_stop),
        # plus its retry backoff and alert rate-limit.
        self._no_stop_pending: dict | None = None
        self._no_stop_backoff_s: float = self.NO_STOP_RETRY_MIN_S
        self._no_stop_next_try: float = 0.0
        # In-memory twins of the no-stop meta rows: a failing sqlite write must
        # neither un-latch entries nor resurrect a cleared pending close.
        self._latch_until: float = 0.0
        self._latch_streak: int = 0
        self._latch_persisted: bool = False
        self._latch_unreadable_alert_ts: float = 0.0
        self._latch_released_for: float = 0.0
        self._pending_meta_stale: bool = False
        self._book_unknown_n: int = 0
        self._book_unknown_since: float | None = None
        self._unverified: dict | None = None
        self._unverified_meta_stale: bool = False
        self._unproven_alerted_root: str | None = None
        self._sl_fail_mem: dict[str, int] = {}
        self._no_stop_alert_ts: float = 0.0
        self._unverified_next: float = 0.0
        # True while the loop's `equity` is the last-known stand-in, not a fresh read.
        self._equity_stale: bool = False
        # Breakeven stop (see _maybe_breakeven). Open-time (epoch s) of the last
        # CLOSED bar fully evaluated — the hook costs zero API calls until the
        # next bar closes. Advanced only on success or a definite no-op, so a
        # failed placement is retried (throttled by _be_retry_after).
        self._be_evaluated_bar: int | None = None
        self._be_retry_after: float = 0.0
        self._be_fail_n: int = 0
        if self.dry_run:
            self.log.warning("DRY-RUN MODE: no real orders will be placed")

    def boot(self) -> None:
        check_symbol(self.symbol)
        check_leverage(self.leverage)

        # Load live exchange constraints (min qty, min notional). Tighter
        # of hard-coded fallbacks vs live values wins.
        try:
            live_market = self.client.ex.market(self.symbol)
            # Per-symbol fallbacks: BTC's $50 min-notional / $0.10 tick are not
            # a sane default for SOL. merge_with_live still takes the TIGHTER of
            # fallback vs live, so this cannot loosen anything.
            self.constraints = merge_with_live(
                fallbacks_for_symbol(self.symbol), live_market)
            self.log.info("Exchange constraints for %s: min_qty=%s, min_notional=$%s, "
                          "price_step=%s, qty_step=%s",
                          self.symbol, self.constraints.min_qty_base,
                          self.constraints.min_notional_usdt,
                          self.constraints.price_step, self.constraints.qty_step)
        except Exception as e:
            self.log.warning("Could not fetch live market constraints, using defaults: %s", e)

        if not self.dry_run:
            self.client.set_leverage(self.symbol, self.leverage)

        state.init_db()

        # Establish the principal anchor P (kill-switch denominator = NET
        # DEPOSITED PRINCIPAL, not a balance snapshot). Best-effort: if the
        # income backfill fails here, the loop retries and the kill switch
        # stays fail-safe (disabled) until P is known. NEVER snapshots balance.
        try:
            P = principal.initialize(self.client, self.params, log=self.log)
            self._last_principal_reconcile_ts = time.time()
            if P is not None:
                self.log.info(
                    "Principal anchor P=%.2f USDT (kill floor=%.2f at fraction %.3f)",
                    P, P * self.kill_fraction, self.kill_fraction)
            else:
                self.log.warning(
                    "Principal anchor NOT established at boot — kill switch "
                    "DISABLED until income reconcile succeeds.")
        except Exception as e:
            self.log.warning("Principal init failed at boot (will retry in loop): %s", e)

        for _attempt in range(4):
            try:
                equity = self.client.fetch_equity_usdt()
                break
            except EquityUnavailable as e:
                if _attempt == 3:
                    # Still before any flatten/adopt: the position (if any) keeps
                    # its exchange-side brackets while systemd retries.
                    self.log.error("boot: equity unreadable after retries: %s", e)
                    send_alert("Bot boot aborted: equity unreadable", str(e))
                    raise
                self.log.warning("boot: equity unreadable (try %d/4): %s",
                                 _attempt + 1, e)
                time.sleep(2.0 * (_attempt + 1))
        if equity < self.min_capital_warn:
            self.log.warning("LOW CAPITAL: equity $%.2f < $%.2f recommended. "
                             "Some signals may be skipped due to exchange minimums.",
                             equity, self.min_capital_warn)

        start_eq = state.get_float("deploy_start_equity", 0.0)
        if start_eq <= 0:
            state.set_float("deploy_start_equity", equity)
            state.set_meta("deploy_start_ts", datetime.now(UTC).isoformat())
            self.log.info("Recorded deploy_start_equity=%.2f USDT", equity)
            mode = "DRY-RUN" if self.dry_run else "LIVE"
            send_alert(
                f"Bot deploy start [{mode}]",
                self._format_deploy_summary(equity, mode),
            )
            start_eq = equity
        else:
            # Report against PRINCIPAL, not against deploy_start_equity.
            # start_eq is a raw equity snapshot frozen at deploy time, so every
            # later deposit or withdrawal moves the numerator and leaves the
            # denominator behind -- the same two-clocks defect PR #25 fixed in
            # the monitor, and it flatters in the dangerous direction.
            # Live 2026-09-09: sol_supertrend booted announcing
            # a large gain vs its deploy-start equity for a leg that was DOWN
            # vs principal -- a later top-up was counted as profit.
            # P is the denominator the kill switch already names (see
            # _check_kill_switch), so reporting against it also stops the boot
            # line and the kill-switch line describing the same leg
            # differently. start_eq stays RAW and is still printed: do NOT
            # re-baseline it, _daily_book_anchor applies its own delta and
            # pre-shifting the stored value would double-count.
            P = principal.get_principal()
            if P is not None and P > 0:
                self.log.info(
                    "Resuming deploy. start=%.2f current=%.2f principal=%.2f "
                    "(%+.2f%% vs principal)",
                    start_eq, equity, P, (equity / P - 1) * 100)
            else:
                # Two distinct reasons land here and an operator has to tell
                # them apart: P is None until the income backfill lands,
                # whereas a P that is present but <= 0 means the ledger HAS
                # been read and nets to nothing -- a fully withdrawn leg, which
                # is not "pending" and will never resolve on its own.
                # breached() treats both as unknown; only the first is waiting
                # on anything. Either way, say the number is unadjusted rather
                # than print a transfer-contaminated percentage bare.
                why = ("principal pending" if P is None
                       else f"principal non-positive ({P:.2f})")
                self.log.info(
                    "Resuming deploy. start=%.2f current=%.2f (%+.2f%% vs "
                    "start; %s, NOT transfer-adjusted)",
                    start_eq, equity, (equity / start_eq - 1) * 100, why)

        # Push a boot event to consolidate so the dashboard knows the bot
        # is alive and which strategy/env it's running. The deploy-start
        # equity and kill-switch fraction let consolidate compute the
        # kill-switch level without needing a separate config endpoint.
        state.enqueue_bot_event(
            "boot",
            strategy=self.strategy_name,
            equity_usd=float(equity),
            payload={
                "env": self.client.env,
                "dry_run": bool(self.dry_run),
                "strategy_name": self.strategy_name,
                "deploy_start_equity": float(start_eq),
                "principal_anchor": principal.get_principal(),
                "kill_switch_fraction": float(self.kill_fraction),
                "leverage": int(self.leverage),
                "order_type": self.order_type,
            },
        )

        pos = self.client.fetch_position(self.symbol)
        if pos.side != "flat":
            if self.dry_run:
                self.log.warning("Boot found open position %s qty=%.4f @ %.2f. "
                                 "DRY-RUN: leaving it alone.",
                                 pos.side, pos.qty, pos.entry_price)
            elif self._boot_resume_verdict(pos)[0]:
                # RESUME. The tick loop owns this position from here exactly as
                # if no restart had happened: its age still reads from the fills
                # table, the daily anchor still reads from state.meta, and the
                # first _maybe_reprotect will restore the bracket if it is gone.
                # The root is the one the gate PROVED (C5) and stashed; it is
                # never re-read here, so it can be neither None nor different.
                root = self._adopt_root
                # B1: seed the exit-tracking snapshot BEFORE anything else can
                # run (or raise). Without it _last_position_side stays
                # "unknown", and a bracket that fills between this read and the
                # first tick reads as unknown→flat — NOT an exit — so no close
                # row, no exit event, no alert: the donchian 09-04 dropped-exit
                # shape through a new door. Plain assignments; cannot raise.
                self._seed_adopted_tracking(pos, root)
                # C11: an unchosen restart while in a position must reach God's
                # inbox, not only the log. Its own try, BEFORE the bookkeeping,
                # so a sqlite failure below cannot swallow it; non-fatal.
                self._alert_resumed(pos, root)
                # Bookkeeping only: the position is already adopted, so a failure
                # here is logged and the tick loop carries on owning it.
                try:
                    self.log.warning("Boot found open position %s qty=%.4f @ %.2f. "
                                     "RESUMING it (root=%s).",
                                     pos.side, pos.qty, pos.entry_price, root or "—")
                    state.record_event("WARN", "boot_adopt",
                                       {"side": pos.side, "qty": pos.qty,
                                        "entry": pos.entry_price, "signal_id": root},
                                       signal_id=root)
                    state.enqueue_bot_event(
                        "boot_adopt",
                        signal_id=root,
                        side=pos.side,
                        qty=float(pos.qty),
                        price_usd=float(pos.entry_price),
                        payload={"reason": "resumed_at_boot",
                                 "entry_price": float(pos.entry_price)},
                    )
                except Exception:
                    self.log.exception("boot_adopt bookkeeping failed; position "
                                       "stays adopted, tick loop owns it")
            else:
                root = state.latest_entry_coid_root()
                self.log.warning("Boot found open position %s qty=%.4f @ %.2f. "
                                 "Flattening (root=%s).",
                                 pos.side, pos.qty, pos.entry_price, root or "—")
                close_order = self.client.close_position(
                    self.symbol, client_order_id_root=root, close_leg="bf")
                if close_order is None:
                    self._boot_close_found_flat(pos, root)
                    return
                # Emit the REAL flatten fill + post-close equity so the dashboard
                # attributes the actual PnL. Previously this sent the ENTRY price
                # and no equity, so a boot-flattened WIN showed as $0/loss on the
                # dashboard (makeTrade fell back to (entry-entry)*qty = 0, which
                # deriveLegStats then counts as a loss).
                # Refetch: the create response carries avgPrice "0.00", so the
                # old `or pos.entry_price` fallback always fired and made every
                # boot-flatten report PnL of exactly 0.
                close_price = (self._resolve_fill_price(close_order)
                               or pos.entry_price)
                try:
                    equity_after = float(self.client.fetch_equity_usdt())
                except Exception:
                    equity_after = None
                pnl_usd = _exit_pnl_usd(pos.side, pos.entry_price,
                                        close_price, pos.qty)
                state.record_event("WARN", "boot_flatten",
                                   {"side": pos.side, "qty": pos.qty,
                                    "entry": pos.entry_price, "exit": close_price,
                                    "pnl_usd": pnl_usd, "signal_id": root},
                                   signal_id=root)
                state.enqueue_bot_event(
                    "boot_flatten",
                    signal_id=root,
                    side=pos.side,
                    qty=float(pos.qty),
                    price_usd=float(close_price),
                    equity_usd=equity_after,
                    payload={"reason": "stale_position_at_boot",
                             "exit_price": float(close_price),
                             "entry_price": float(pos.entry_price),
                             "pnl_usd": pnl_usd},
                )
        else:
            # Position is already flat at boot.  Sweep any orphaned reduce-only
            # orders left over from a prior run (e.g. a surviving bracket sibling
            # after an exchange-driven TP/SL fill).  Only our own COID prefix.
            if not self.dry_run:
                try:
                    n = self.client.cancel_open_orders(
                        self.symbol, coid_prefix=self.coid_prefix)
                    if n:
                        self.log.warning(
                            "boot: swept %d orphaned order(s) while flat", n)
                except Exception:
                    self.log.exception("boot: cancel_open_orders failed")
        self._clear_no_stop_records_at_boot()

    def _boot_close_found_flat(self, pos, root: str | None) -> None:
        """The boot flatten found the exchange ALREADY flat. NEVER raises.

        close_position re-reads the position and returns None when it is flat.
        Something closed it after boot's own read, most likely a bracket leg
        filling (droplet has this window today; the C6 re-read widens it on the
        armed path). The old code then recorded a `boot_flatten` with exit =
        entry and PnL $0 for a close that never happened, and swept nothing,
        because close_position returns before its own sweep. The surviving
        bracket sibling stayed resting.

        Here: sweep our own resting orders (same call as the flat-at-boot
        branch), record a local WARN event, and alert. Do NOT emit a fabricated
        exit. The real exit price/PnL is NOT recorded; income reconciliation is
        the backstop, as for "bracket fills while down".
        """
        try:
            self._clear_no_stop_records_at_boot()
        except Exception:
            self.log.exception("boot: clearing no-stop records failed")
        try:
            n = self.client.cancel_open_orders(self.symbol,
                                               coid_prefix=self.coid_prefix,
                                               report_unknown=True)
        except Exception:
            self.log.exception("boot: already-flat sweep failed")
            n = None
        swept = (f"sweep cancelled {n} resting order(s)" if n is not None else
                 "order book was UNREADABLE so the sweep is unverified: a "
                 "bracket may still be resting, check the exchange")
        msg = (f"Boot found an open {pos.side} {float(pos.qty):.4f} @ "
               f"{float(pos.entry_price):.2f} but the position was ALREADY "
               f"CLOSED when the flatten ran (root={root or '—'}); {swept}. "
               f"The exit was NOT recorded: reconcile it via the income "
               f"ledger.")
        try:
            self.log.warning("%s", msg)
            state.record_event("WARN", "boot_flatten_already_flat",
                               {"side": pos.side, "qty": pos.qty,
                                "entry": pos.entry_price, "signal_id": root,
                                "swept": n},
                               signal_id=root)
        except Exception:
            self.log.exception("boot: already-flat bookkeeping failed")
        try:
            send_alert("Position already closed during boot — exit not recorded",
                       f"{self.instance} leg: {msg}")
        except Exception:
            self.log.exception("boot: already-flat alert failed")

    def stop(self, *_args) -> None:
        self._stopped = True
        self.log.info("Stop requested.")

    def _heartbeat(self) -> None:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.touch()

    def _format_deploy_summary(self, equity: float, mode: str) -> str:
        kill_pct = self.kill_fraction * 100.0
        # Kill floor is anchored on DEPOSITED PRINCIPAL, not this boot's equity.
        # Fall back to the equity figure only if the anchor isn't established yet
        # (income backfill pending) — marked provisional so it isn't mistaken
        # for the real floor.
        P = principal.get_principal()
        kill_anchor = P if (P is not None and P > 0) else equity
        kill_usd = kill_anchor * self.kill_fraction
        kill_anchor_label = ("principal" if (P is not None and P > 0)
                             else "equity (PROVISIONAL — principal pending)")
        max_hold = int(
            (self.params.get("strategy") or {}).get("max_hold_bars", 0)
        )
        max_hold_days = (max_hold * self.bar_seconds) / 86400.0 if max_hold else 0.0
        dedup_bars = int(
            (self.params.get("strategy") or {}).get("dedup_bars", 0)
        )
        consolidate_source = (os.environ.get("CONSOLIDATE_SOURCE") or "").strip() or "(default snapback-btc)"
        be_cfg = self.params.get("breakeven") or {}
        return (
            f"snapback-btc deploy start\n"
            f"=========================\n"
            f"Instance      : {self.instance}\n"
            f"Strategy      : {self.strategy_name}\n"
            f"Symbol        : {self.symbol}\n"
            f"Config        : {CONFIG_PATH.relative_to(REPO_ROOT)}\n"
            f"Mode          : {mode}\n"
            f"Environment   : {self.client.env}\n"
            f"PID           : {os.getpid()}\n"
            f"Boot ts (UTC) : {datetime.now(UTC).isoformat(timespec='seconds')}\n"
            f"\n"
            f"Capital\n"
            f"  Start equity   : ${equity:,.2f} USDT\n"
            f"  Min recommended: ${self.min_capital_warn:,.2f} USDT"
            + ("  ⚠ BELOW FLOOR" if equity < self.min_capital_warn else "")
            + f"\n"
            f"  Kill switch    : {kill_pct:.1f}% of {kill_anchor_label}\n"
            f"                   → exit at ${kill_usd:,.2f} USDT "
            f"(anchor ${kill_anchor:,.2f})\n"
            f"\n"
            f"Risk\n"
            f"  Per trade      : {self.risk_pct:.2f}% equity\n"
            f"  Leverage       : {self.leverage}x\n"
            f"  Hedge mode     : {'on' if self.hedge_enabled else 'off'}\n"
            + (f"  Dedup          : {dedup_bars} bars\n" if dedup_bars else "")
            + (f"  Breakeven stop : at +{float(be_cfg['at_r']):g}R "
               f"(buffer {float(be_cfg['buffer_pct']):g}%)\n"
               if be_cfg.get("enabled") else "  Breakeven stop : off\n")
            + f"\n"
            f"Timing\n"
            f"  Entry timeframe: {self.entry_tf} ({self.bar_seconds}s/bar)\n"
            f"  Poll interval  : {self.poll_s:.0f}s\n"
            + (f"  Max hold       : {max_hold} bars ({max_hold_days:.1f} days)\n"
               if max_hold else "")
            + f"\n"
            f"Routing\n"
            f"  Consolidate src: {consolidate_source}\n"
            f"  Alert tag      : {os.environ.get('ALERT_TAG', '(default snapback-btc)')}\n"
            f"\n"
            f"Paths\n"
            f"  State DB       : {_rel_to_repo(state.DB_PATH)}\n"
            f"  Log file       : {_rel_to_repo(LOG_FILE)}\n"
            f"  Heartbeat      : {_rel_to_repo(HEARTBEAT)}\n"
        )

    def _maybe_push_consolidate(self, equity: float | None) -> None:
        """If enough time elapsed since last push: enqueue a heartbeat event
        and drain the outbox. Never blocks for more than the HTTP timeout
        (3s by default). Failures are logged and retried on the next tick.
        """
        now = time.time()
        if now - self._last_push_ts < self._push_interval_s:
            return
        self._last_push_ts = now
        if not consolidate_push.is_configured():
            return  # don't even enqueue heartbeats if no consumer
        try:
            payload: dict = {"halt_present": is_halted(self.instance),
                             "entries_paused": leg_pause_path(self.instance).exists(),
                             "outbox_size_before_drain": state.outbox_size()}
            # Include the most recent gate-status snapshot so the dashboard
            # can show "what's true now / waiting on what" without operator
            # intervention. Falls back to absent key when no bar has been
            # evaluated yet (first 5-10s of boot).
            if self._latest_gates is not None:
                payload["gates"] = self._latest_gates
            state.enqueue_bot_event(
                "heartbeat",
                equity_usd=float(equity) if equity is not None else None,
                payload=payload,
            )
            result = consolidate_push.drain()
            if result.get("error"):
                self.log.debug("consolidate push deferred: %s", result.get("error"))
        except Exception as e:
            # Push must NEVER affect the trading loop. Log + move on.
            self.log.warning("consolidate push raised (continuing): %s", e)

    def _check_kill_switch(self, equity: float) -> bool:
        # Anchor = NET DEPOSITED PRINCIPAL (God's rule), NOT a balance snapshot
        # and NOT a high-water mark. principal.breached() is fail-safe: it never
        # trips on an unknown/degenerate anchor, so a stale/wrong equity read or
        # a not-yet-initialised P cannot cause a false-positive kill (the
        # donchian wrong-account class of incident).
        P = principal.get_principal()
        if P is not None and P > 0:
            self._principal_unset_alerted = False   # re-arm: alert again if it is lost later
        elif not self._principal_unset_alerted:
            self._principal_unset_alerted = True
            self.log.warning("principal anchor unset (P=%s): kill switch is INERT "
                             "(daily-loss breaker too while the ledger is uninitialised).", P)
            send_alert(
                "Kill switch INERT: principal anchor unset",
                f"principal={P}. The principal kill switch is disabled until the "
                f"income reconcile establishes P (the daily-loss breaker too "
                f"while the ledger is uninitialised). Alert only; no action taken.")
        if not principal.breached(equity, P, self.kill_fraction):
            return False
        # breached() only returns True for a positive float anchor. Make that
        # invariant explicit with a real guard — an `assert` here is stripped
        # under `python -O`, which would let `P * fraction` raise TypeError (or
        # touch the HALT on a None anchor) in the hot trading path.
        if P is None:
            raise RuntimeError(
                "kill-switch invariant violated: breached() true but principal "
                "is None — refusing to act on an unknown anchor.")
        floor = P * self.kill_fraction
        self.log.error("KILL SWITCH: equity %.2f < %.2f (principal %.2f * %.2f)",
                       equity, floor, P, self.kill_fraction)
        # Touch ONLY this leg's self-halt flag — never the shared data/HALT.
        self.halt_path.touch()
        state.enqueue_bot_event(
            "kill_switch", equity_usd=float(equity),
            payload={"principal_anchor": float(P),
                     "kill_switch_fraction": float(self.kill_fraction),
                     "kill_floor": float(floor),
                     "drawdown_vs_principal_pct": (equity / P - 1) * 100},
        )
        # Force a push so the dashboard sees this before the bot exits.
        try:
            consolidate_push.drain()
        except Exception as e:
            self.log.warning("kill-switch push failed: %s", e)
        send_alert(
            "BOT KILL SWITCH FIRED",
            f"Equity fell below -{(1-self.kill_fraction)*100:.0f}% of DEPOSITED "
            f"PRINCIPAL.\n"
            f"principal={P:.2f}  floor={floor:.2f}  current={equity:.2f}  "
            f"({(equity / P - 1)*100:+.2f}% vs principal)\n"
            f"Position will be flattened, data/HALT_{self.instance} created, "
            f"bot will exit.",
        )
        return True

    def _maybe_reconcile_principal(self) -> None:
        """Periodically refresh the principal anchor from the Binance income
        ledger. Idempotent (tranId-keyed) so overlapping windows never
        double-count. Never blocks or crashes the loop — failures are logged."""
        now = time.time()
        initialized = principal.is_initialized()
        interval = (self._principal_reconcile_interval_s if initialized
                    else self._principal_retry_interval_s)
        if now - self._last_principal_reconcile_ts < interval:
            return
        self._last_principal_reconcile_ts = now
        try:
            if not initialized:
                principal.initialize(self.client, self.params, log=self.log)
            else:
                principal.reconcile_recent(self.client, log=self.log)
        except Exception as e:
            self.log.warning("principal reconcile failed (continuing): %s", e)

    # A snapshot taken later than this after UTC midnight is logged as LATE:
    # the bot was down (or principal not yet ready) at the rollover, so any
    # P&L between 00:00 and now is NOT in today's breaker window. Two polls
    # (donchian/sol poll every 60s) plus a margin for one transient backoff.
    DAILY_ANCHOR_LATE_MARGIN_S = 120.0

    def _daily_anchor_equity(self, equity: float, timeout: float = 10.0) -> float:
        """Return the UTC-day's RAW starting equity, re-anchoring on date rollover.

        SNAPSHOT AT MIDNIGHT, FLAT OR NOT (2026-10-13, God's decision). The tick
        loop calls this via _roll_daily_anchor right after fetch_equity_usdt(),
        so the first tick after 00:00 UTC anchors to LIVE equity (wallet +
        unrealized) whether or not a position is open. Before, it was reached
        only from _maybe_enter's flat path, so a position carried over midnight
        and stopped out at 03:00 anchored AFTER the loss and the breaker never
        counted it. Consequence, by design: P&L accrued before midnight is
        yesterday's; an open gain marked at 00:00 and later given back IS
        today's loss.

        Bot down at midnight: the first tick after restart anchors (late, WARN +
        `daily_anchor_late` event); P&L between 00:00 and that tick is not
        reconstructed. A non-finite or non-positive equity is never anchored.

        Persisted in state.meta so it survives restarts within a day. On each
        new UTC day we ALSO snapshot the cumulative principal-ledger sum
        (daily_anchor_principal_sum) so the breaker can neutralise intraday
        transfers — see _daily_book_anchor. The two ceilings stay separate.

        `timeout` is sqlite's busy wait; the per-tick caller caps it so a
        locked DB skips the snapshot (retried next tick) instead of stalling exits.
        """
        now = datetime.now(UTC)
        today = now.strftime("%Y-%m-%d")
        anchor_date = state.get_meta("daily_anchor_date", timeout=timeout)
        anchor_eq = state.get_float("daily_anchor_equity", 0.0, timeout=timeout)
        if anchor_date != today or anchor_eq <= 0:
            if not (math.isfinite(equity) and equity > 0):
                # A NaN passes `<= 0` and would sit as the anchor all day with
                # the breaker silently unable to trip. Keep today un-anchored
                # (0.0 → check_daily_loss no-op) and retry on the next tick.
                self.log.warning("Daily anchor NOT set: equity=%r is not a usable "
                                 "reading; retrying next tick", equity)
                return anchor_eq if anchor_date == today else 0.0
            # Baseline for intraday-transfer neutralisation: net deposited
            # principal (Part C ledger) as of the anchor. Intraday transfers move
            # the ledger sum; _daily_book_anchor adds the delta back so only
            # trading P&L counts toward the threshold.
            # DATE LAST: it is the "anchored today" marker the guard above trusts,
            # so a crash or lock between these writes must leave it stale (→ the
            # whole snapshot is retried) rather than pair today's date with
            # yesterday's equity and principal baseline.
            state.set_float("daily_anchor_principal_sum",
                            state.principal_ledger_sum(principal.PRINCIPAL_ASSET,
                                                       timeout=timeout),
                            timeout=timeout)
            state.set_float("daily_anchor_equity", equity, timeout=timeout)
            state.set_meta("daily_anchor_date", today, timeout=timeout)
            secs = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)
                    ).total_seconds()
            late_after = 2.0 * float(getattr(self, "poll_s", 60.0)) + self.DAILY_ANCHOR_LATE_MARGIN_S
            if anchor_date is not None and secs > late_after:
                self.log.warning(
                    "Daily anchor set LATE: date=%s equity=%.2f USDT, %.0f min after "
                    "UTC midnight (prev anchor %s) — P&L since 00:00 not counted today",
                    today, equity, secs / 60.0, anchor_date)
                state.record_event("WARN", "daily_anchor_late",
                                   {"date": today, "equity": equity,
                                    "minutes_late": round(secs / 60.0, 1),
                                    "prev_anchor_date": anchor_date})
            else:
                self.log.info("Daily anchor set: date=%s equity=%.2f USDT", today, equity)
            return equity
        return anchor_eq

    def _roll_daily_anchor(self, equity: float) -> None:
        """Per-tick hook: snapshot today's anchor on UTC rollover, flat or not.

        Gated on principal.is_initialized() for the same reason as the breaker
        (a partial income backfill would poison daily_anchor_principal_sum).
        Never raises: a sqlite hiccup here must not push the tick into the
        catch-all backoff and delay exits on a real-money position; the lazy
        call in _daily_loss_blocks_entry retries the snapshot anyway.
        """
        try:
            timeout = self.LAST_BAR_PERSIST_TIMEOUT_S
            if principal.is_initialized(timeout):
                self._daily_anchor_equity(equity, timeout)
        except Exception as e:
            self.log.warning("daily anchor rollover failed (continuing): %s", e)

    LIVE_EQUITY_WRITE_INTERVAL_S = 60.0
    # Tick-to-tick equity jump (fraction) with NO position change and NO new
    # fill that is treated as a possible deposit/withdrawal not yet in the
    # principal ledger (reconciled hourly). 1.5%: well above a minute of
    # unrealized drift on these legs, well below a monthly DCA (~13%).
    TRANSFER_JUMP_FRAC = 0.015
    # How long a suspected transfer suppresses the monitor's live reading if
    # the ledger never moves (then it was price, not a transfer).
    TRANSFER_SUSPECT_MAX_S = 300.0
    # Oldest persisted `live_equity` sample usable to seed the transfer detector
    # after a restart (older = a gap too long to call a jump "unexplained").
    TRANSFER_SEED_MAX_AGE_S = 600.0

    def _persist_live_equity(self, equity: float) -> None:
        """Mirror the tick's live equity (wallet + unrealized) into meta key
        `live_equity` as ONE JSON value so the monitor never reads a torn set:

            {"ts": epoch_s, "equity": x, "ledger_sum": Σ USDT principal_ledger
             at write time, "transfer_pending": bool}

        Lets monitor._equity_from_db see an open position's drawdown without
        any exchange call. Rate-limited to one write per
        LIVE_EQUITY_WRITE_INTERVAL_S (immediate when transfer_pending flips).

        UNRECONCILED TRANSFERS (2026-10-13 panel; refined after re-verify: a
        raw >1.5% equity jump also fires on a real fast drawdown, so only the
        UNEXPLAINED part — Δwallet, or Δequity when flat — counts, and a
        fingerprint change clears the flag). Live equity moves the
        instant God deposits/withdraws; the principal ledger only on the hourly
        reconcile. Replaying a real v1 deposit (ledger applied ~30 min later):
        live equity vs ledger-derived P swung down, back up, then down again,
        i.e. a false recovery mail then a fresh alert. So: a jump >
        TRANSFER_JUMP_FRAC since the previous tick with the position and the
        fills table unchanged forces the existing reconcile on the next tick
        (`_last_principal_reconcile_ts = 0.0`; retried each minute) and marks
        the sample transfer_pending, which the monitor ignores. Cleared when
        the ledger sum moves (transfer landed) or after
        TRANSFER_SUSPECT_MAX_S (it was price). Bonus: the kill switch and the
        daily breaker see the reconciled P an hour sooner.

        Known blind spots (the flag is a heuristic, not a ledger): a transfer
        in the same tick as a fill/position change, a transfer during bot
        downtime unless a fresh persisted sample seeds the detector at startup
        (_seed_transfer_detector), and a transfer under TRANSFER_JUMP_FRAC
        (a small top-up on a small account) are not flagged and read as trading P&L until the hourly
        reconcile. Must run AFTER _detect_bracket_exit in the tick (see loop).

        Only finite equity > 0 is written: a transient 0 would read as -100%.
        Never raises (best-effort like _persist_last_signal_ts).
        """
        try:
            if not (math.isfinite(equity) and equity > 0):
                return
            timeout = self.LAST_BAR_PERSIST_TIMEOUT_S
            now_s = datetime.now(UTC).timestamp()
            ledger = state.principal_ledger_sum(principal.PRINCIPAL_ASSET, timeout)
            fp = (getattr(self, "_last_position_side", None),
                  getattr(self, "_last_position_qty", None),
                  state.max_fill_id(timeout))
            if not getattr(self, "_live_equity_seeded", False):
                self._live_equity_seeded = True
                self._seed_transfer_detector(ledger, now_s, timeout)
            prev_eq = getattr(self, "_prev_tick_equity", None)
            prev_fp = getattr(self, "_prev_tick_fingerprint", None)
            prev_wallet = getattr(self, "_prev_tick_wallet", None)
            suspect = getattr(self, "_transfer_suspect", None)
            was_pending = suspect is not None
            wallet = getattr(self.client, "last_wallet_usdt", None)
            if not (isinstance(wallet, (int, float)) and math.isfinite(wallet)):
                wallet = None
            self._prev_tick_equity, self._prev_tick_fingerprint = equity, fp
            self._prev_tick_wallet = wallet

            if fp != prev_fp:
                # A fill or position change since last tick explains the move, so
                # no NEW suspect is raised. An already-raised one is NOT cleared
                # here: an unrelated fill must not re-trust a sample that may
                # still include an unreconciled transfer; it expires below on
                # the ledger move or TRANSFER_SUSPECT_MAX_S.
                pass
            elif prev_eq is not None and prev_eq > 0:
                # UNEXPLAINED move. Price moves live in unrealized PnL, so with
                # both wallet reads available the residual is Δwallet (realized
                # PnL, fees, funding, transfers). Flat: unrealized is 0, Δequity
                # is the residual. Holding without a wallet read: cannot tell
                # price from transfer → never flag (a flag must not hide a crash).
                if wallet is not None and prev_wallet is not None:
                    residual: float | None = wallet - prev_wallet
                elif fp[0] == "flat":
                    residual = equity - prev_eq
                else:
                    residual = None
                if residual is not None and abs(residual) / prev_eq > self.TRANSFER_JUMP_FRAC:
                    suspect = {"since": now_s, "ledger_sum": ledger}
                    self.log.warning(
                        "unexplained equity move %+.2f USDT (%+.2f%%; equity %.2f -> "
                        "%.2f) with no position change or fill — possible "
                        "unreconciled transfer; forcing principal reconcile",
                        residual, residual / prev_eq * 100, prev_eq, equity)
            if suspect is not None:
                if ledger != suspect["ledger_sum"] \
                        or now_s - suspect["since"] > self.TRANSFER_SUSPECT_MAX_S:
                    suspect = None
                elif time.time() - getattr(self, "_last_principal_reconcile_ts", 0.0) >= 60.0:
                    self._last_principal_reconcile_ts = 0.0   # next tick reconciles
            self._transfer_suspect = suspect
            pending = suspect is not None

            last = getattr(self, "_last_live_equity_write", None)
            if (pending == was_pending and last is not None
                    and now_s - last < self.LIVE_EQUITY_WRITE_INTERVAL_S):
                return
            self._last_live_equity_write = now_s
            state.set_meta("live_equity",
                           json.dumps({"ts": now_s, "equity": float(equity),
                                       "ledger_sum": float(ledger),
                                       "transfer_pending": pending,
                                       "wallet": wallet, "fp": list(fp)}),
                           timeout=timeout)
        except Exception as e:
            self.log.warning("could not persist live_equity: %s", e)

    def _seed_transfer_detector(self, ledger: float, now_s: float, timeout: float) -> None:
        """Seed the previous-tick sample from the last persisted `live_equity`
        so a transfer made while the bot was down/restarting is still compared
        on the first tick. Used only if the sample is fresh, not already
        transfer_pending, and taken at the same principal-ledger sum; anything
        else leaves the detector unseeded (first tick blind, as before)."""
        raw = state.get_meta("live_equity", timeout=timeout)
        if not raw:
            return
        try:
            d = json.loads(raw)
            ts, eq, led = float(d["ts"]), float(d["equity"]), float(d["ledger_sum"])
            fp = tuple(d["fp"])
            wallet = d.get("wallet")
            wallet = float(wallet) if wallet is not None else None
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            return
        if not (math.isfinite(eq) and eq > 0 and math.isfinite(ts) and math.isfinite(led)):
            return
        if d.get("transfer_pending") is not False or abs(led - ledger) > 1e-6:
            return
        if not (-60.0 <= now_s - ts <= self.TRANSFER_SEED_MAX_AGE_S):
            return
        if wallet is not None and not math.isfinite(wallet):
            wallet = None
        self._prev_tick_equity, self._prev_tick_fingerprint = eq, fp
        self._prev_tick_wallet = wallet

    def _daily_book_anchor(self, equity: float) -> float:
        """Transfer-immune daily baseline (BOOK equity at day start).

        A raw equity anchor misreads an intraday deposit/withdrawal as P&L: a
        deposit inflates equity (can MASK a real trading loss), a withdrawal
        deflates it (can FALSELY trip the breaker). Neutralise both by shifting
        the baseline by the net principal moved since the anchor was set:

            book_anchor = raw_anchor + (ledger_sum_now − ledger_sum_at_anchor)

        A transfer then moves equity and the baseline by the same amount, so only
        true trading P&L counts toward the 2% threshold — consistent with the
        principal-derived kill switch. Reuses the Part C principal ledger.

        Fail-safe to legacy behaviour: an empty/uninitialised ledger gives a 0
        delta (== raw anchor). A legacy anchor set before this migration (no
        baseline key) is seeded to the current sum, so PRE-existing principal is
        never mistaken for an intraday transfer (which would inflate the baseline
        and stop the breaker from ever tripping).
        """
        raw_anchor = self._daily_anchor_equity(equity)
        now_sum = state.principal_ledger_sum(principal.PRINCIPAL_ASSET)
        baseline = state.get_meta("daily_anchor_principal_sum")
        if baseline is None:
            state.set_float("daily_anchor_principal_sum", now_sum)
            return raw_anchor
        try:
            anchor_sum = float(baseline)
        except ValueError:
            anchor_sum = now_sum
        return raw_anchor + (now_sum - anchor_sum)

    def _daily_loss_blocks_entry(self, equity: float) -> bool:
        """True if today's TRADING drawdown has hit MAX_DAILY_LOSS_PCT — block new
        entries for the rest of the UTC day. Does NOT flatten or HALT: existing
        brackets keep managing any open position, and the anchor resets at the
        next UTC midnight even mid-hold (see _daily_anchor_equity /
        _roll_daily_anchor), so an overnight-held loss counts. The baseline is transfer-
        immune book equity (_daily_book_anchor), so an intraday deposit/withdrawal
        neither trips nor clears the breaker. Tighter, daily-resetting sibling of
        the principal-anchored kill switch.
        """
        # Fail-safe (mirrors the kill switch): the transfer-immune baseline is
        # derived from Part C's principal ledger. Until that ledger is
        # initialised, principal_ledger_sum() is 0 and a later full-history
        # backfill would look like one huge intraday deposit — inflating the book
        # anchor and falsely blocking entries for the rest of the UTC day. So the
        # breaker stays INACTIVE until P is ready (the same window the kill switch
        # is fail-safe/disabled), and the daily anchor is not snapshotted yet.
        if not principal.is_initialized():
            self._daily_loss_blocked = False
            return False
        day_start = self._daily_book_anchor(equity)
        try:
            check_daily_loss(equity, day_start)
        except RiskBreach as e:
            loss_pct = (day_start - equity) / day_start * 100.0 if day_start > 0 else 0.0
            # Dedupe: emit the outbox event + alert at most once per UTC day.
            # self._daily_loss_blocked is an in-memory fast path; the persisted
            # meta key 'daily_loss_breaker_date' ensures the latch survives a
            # bot restart within the same UTC day (prevents the dashboard from
            # receiving a duplicate event on every restart after a breach).
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            if not self._daily_loss_blocked:
                if state.get_meta("daily_loss_breaker_date") == today:
                    self._daily_loss_blocked = True  # sync in-memory with persisted latch
                else:
                    # Log + event once per day, on the bar the breaker trips.
                    self.log.warning("DAILY LOSS BREAKER: %s — blocking new entries "
                                     "until next UTC day.", e)
                    state.record_event("WARN", "daily_loss_breaker",
                                       {"equity": equity, "day_start": day_start,
                                        "loss_pct": loss_pct,
                                        "threshold_pct": risk.ceilings().MAX_DAILY_LOSS_PCT})
                    state.enqueue_bot_event(
                        "daily_loss_breaker", equity_usd=float(equity),
                        payload={"day_start_equity": float(day_start),
                                 "loss_pct": loss_pct,
                                 "threshold_pct": risk.ceilings().MAX_DAILY_LOSS_PCT},
                    )
                    send_alert(
                        "Bot daily-loss breaker tripped",
                        f"Daily drawdown breached -{risk.ceilings().MAX_DAILY_LOSS_PCT:.0f}%.\n"
                        f"day_start={day_start:.2f}  current={equity:.2f}  "
                        f"loss={loss_pct:.2f}%\n"
                        f"New entries blocked until next UTC day. "
                        f"Open position (if any) left to its brackets; no flatten.",
                    )
                    state.set_meta("daily_loss_breaker_date", today)
                    self._daily_loss_blocked = True
            return True
        # Drawdown recovered or new day anchored — clear the in-memory latch.
        # (The persisted key expires naturally: next breach day uses a new date.)
        self._daily_loss_blocked = False
        return False

    LAST_BAR_PERSIST_TIMEOUT_S = 0.5

    def _persist_last_signal_ts(self, last_ts) -> None:
        """B2: mirror `_last_signal_ts` into the `last_entry_bar_ts` meta key.

        The in-memory value dies with the process, so after a restart that
        ADOPTS a position, _maybe_enter would re-evaluate the same closed bar
        once the position goes flat and re-enter on the signal it already
        took. boot() reads this back on the adopt path only.

        Best-effort by construction: a locked or failing DB must never skip
        OR DELAY this bar's evaluation or entry, so the sqlite busy wait is
        capped at LAST_BAR_PERSIST_TIMEOUT_S (the default is 10 s, which a
        lock would add straight onto entry latency), and any failure logs and
        carries on. The cost of a missed write is only the entry-fill fallback
        in signal_bar_seed.
        """
        try:
            state.set_meta("last_entry_bar_ts", pd.Timestamp(last_ts).isoformat(),
                           timeout=self.LAST_BAR_PERSIST_TIMEOUT_S)
        except Exception as e:
            self.log.warning("could not persist last_entry_bar_ts=%s: %s", last_ts, e)

    def _entries_paused(self) -> bool:
        """Operator entry pause: data/PAUSE_<instance> exists. Blocks NEW
        entries only; exits, stops and monitoring are untouched. Alerts once
        on each pause/resume edge so a forgotten flag is visible."""
        paused = leg_pause_path(self.instance).exists()
        if paused != self._entries_paused_seen:
            self._entries_paused_seen = paused
            flag = f"data/PAUSE_{self.instance}"
            if paused:
                self.log.warning("entries PAUSED (%s present)", flag)
                send_alert(f"{self.instance}: entries PAUSED",
                           f"{flag} is present, so no new position will be opened. "
                           "Exits and stops still run. Remove the file to resume.")
            else:
                self.log.warning("entries RESUMED (%s removed)", flag)
                send_alert(f"{self.instance}: entries RESUMED",
                           f"{flag} was removed; new entries are allowed again.")
        return paused

    def _maybe_enter(self, equity: float) -> None:
        # No new entry while a no-stop emergency is unresolved or the no-stop
        # latch is on. Local-DB reads only — no exchange call on a healthy tick.
        if self._entry_blocked():
            return
        # Skip entry evaluation if already in a position — bracket SL/TP
        # manage the existing trade. Matches backtest's exclusive_orders=True.
        if self.client.fetch_position(self.symbol).side != "flat":
            return
        if self._entries_paused():
            return

        # Daily-loss breaker: tighter (params risk.max_daily_loss_pct) daily-resetting
        # guard, separate from the cumulative kill-switch
        # (deploy.kill_switch_equity_fraction). Blocks NEW entries only.
        if self._daily_loss_blocks_entry(equity):
            return

        # 1500 bars covers warmup on every supported timeframe:
        #   15m → ~15 days (v1: 200-EMA + 50 warmup, vol-regime needs ~30d
        #        of daily ATR — NaN until accumulated)
        #   4h  → ~250 days (donchian-v3: 80-bar channel, cnh-hybrid: 200-bar
        #        admission walk)
        # Binance Futures klines cap at 1500/call so 1500 is also the ceiling.
        df = self.client.fetch_ohlcv(self.symbol, self.entry_tf, limit=1500)
        # Binance returns the still-FORMING current bar as the last row (~5s
        # after it opens it holds ~0.5% of its eventual volume). The backtest
        # only ever sees CLOSED bars (backtesting.py never shows a forming bar;
        # signals_multifactor.next() reads Close[-1] of a closed bar). Drop the
        # forming bar HERE, before anything downstream touches the frame: every
        # evaluator, the entry price, SL/TP, the 4H-gate alignment timestamp,
        # the warmup count, and the dedup ts all read iloc[-1]/index[-1], so a
        # single slice at the source keeps live↔backtest parity intact. Without
        # this, the volume gate (cur_vol > 2×SMA20) can never be satisfied →
        # 0 trades, and entry/SL/TP would anchor to an incomplete bar.
        df = df.iloc[:-1]
        if len(df) < 250:
            return
        last_ts = df.index[-1]
        if self._last_signal_ts is not None and last_ts <= self._last_signal_ts:
            return

        funding = self.client.fetch_funding_rate(self.symbol)
        decision = evaluate_for_strategy(self.strategy_name, df, funding, self.params)
        self._last_signal_ts = last_ts
        self._persist_last_signal_ts(last_ts)

        # Snapshot the gate state for heartbeat-payload + log. Computed on every
        # bar evaluation so the dashboard's "current state" panel can always
        # answer "why isn't this firing?". Log only when the waiting-for
        # summary *changes*, to avoid spamming the JSONL with identical lines
        # bar after bar in long quiet stretches.
        self._latest_gates = gate_status(self.strategy_name, decision, self.params)
        cur_waiting = self._latest_gates.get("waiting_for", "")
        if cur_waiting and cur_waiting != self._last_waiting_for:
            self.log.info("gates: %s", cur_waiting)
            self._last_waiting_for = cur_waiting

        if decision.side is None:
            return

        qty = round_qty_down(
            compute_qty_from_distance(
                equity, decision.price, decision.sl_distance,
                self.risk_pct, self.leverage),
            self.constraints.qty_step,
        )
        notional = qty * decision.price

        # Exchange minimums (min qty + min notional) come from the live market
        # spec — tighter than our own caps. If a signal can't be filled, SKIP
        # it. Do NOT scale up to meet the minimum; that would violate the
        # risk budget.
        ok, reason = passes_minimums(qty, decision.price, self.constraints)
        if not ok:
            self.log.warning("Skipping %s signal: %s (equity=$%.2f). "
                             "Need more capital or accept smaller positions.",
                             decision.side, reason, equity)
            state.record_event("WARN", "signal_skipped_minimum",
                               {**decision.debug, "side": decision.side,
                                "qty": qty, "reason": reason})
            return

        try:
            check_notional(notional)
        except RiskBreach as e:
            self.log.warning("Skipping signal: %s", e)
            return

        # signal_id anchors all 3 legs of this trade on Binance via clientOrderId.
        # Format: snap-v1-<signal_id>-{e|s|t|x|bf|h|k}. Investing-consolidate's
        # importer joins entry+exit fills via this root. ms precision avoids
        # collisions across bot restarts.
        signal_id = str(int(time.time() * 1000))

        self.log.info(
            "%s %s [%s] sid=%s qty=%.4f price=%.2f sl=%.2f tp=%.2f notional=$%.2f",
            "DRY-RUN would" if self.dry_run else "Signal",
            decision.side, self.order_type, signal_id, qty,
            decision.price, decision.sl_price, decision.tp_price, notional,
        )

        if self.dry_run:
            trade_events.record_dry_run_entry(
                side=decision.side, qty=qty, price=decision.price,
                sl_price=decision.sl_price, tp_price=decision.tp_price,
                notional=notional, equity=equity,
                signal_id=signal_id, strategy_name=self.strategy_name,
                order_type=self.order_type, dbg=decision.debug,
            )
            return

        try:
            self._place_live_entry(decision, qty, signal_id, equity)
        except Exception as e:
            self.log.exception("order placement failed: %s", e)
            state.record_event("ERROR", "order_failed", str(e), signal_id=signal_id)
            # "Entry failed" is only true if nothing filled. A market entry can
            # time out on our side and still fill, leaving a position with no
            # stop — check before saying so.
            if not self._entry_error_left_position(decision, signal_id, equity, e):
                send_alert("Bot order failed", f"{decision.side} entry failed: {e}")

    def _place_live_entry(
        self, decision: SignalDecision, qty: float, signal_id: str, equity: float,
    ) -> None:
        """Place a live entry (market or limit per config) + brackets, then
        record the fill via trade_events. Raises on order-placement failure
        so the caller can log + alert."""
        # Pre-clear any stale bot orders before placing a new entry so we
        # don't leave a stale bracket from a prior trade orphaned alongside
        # the new one.  Only cancels orders with our own COID prefix.
        try:
            self.client.cancel_open_orders(self.symbol, coid_prefix=self.coid_prefix)
        except Exception:
            self.log.exception("pre-entry cancel_open_orders failed (continuing)")

        # Channel-exit strategies (donchian-v3) place entry + SL ONLY — the
        # live Donchian channel cross (see _maybe_channel_exit) closes the
        # trade, so there is no TP leg. Every other strategy keeps its TP.
        place_tp = not strategy_uses_channel_exit(self.strategy_name)

        # Remember this trade's bracket params so _maybe_reprotect can restore
        # the SL/TP if they later go missing while the position is still open
        # (external cancel, or a leverage change → Binance auto-cancels orders).
        # `qty` is recorded for _can_adopt, NOT for reprotect — reprotect always
        # re-places against the live pos.qty, deliberately. Adoption is the only
        # caller that needs to notice a partially-closed position, and without
        # this field its size check has nothing to compare against. A record
        # WITHOUT it (written before Phase A) is refused and flattened (D2).
        state.set_meta("active_bracket", json.dumps({
            "signal_id": signal_id, "side": decision.side,
            "entry_price": float(decision.price),
            "qty": float(qty),
            "sl_distance": float(decision.sl_distance),
            "tp_distance": float(decision.tp_distance),
            "place_tp": bool(place_tp),
        }))

        if self.order_type == "limit":
            limit_price = limit_entry_price(
                decision.side, decision.price, self.limit_offset_bps)
            try:
                orders = self.client.limit_order_with_bracket(
                    self.symbol, decision.side, qty, limit_price,
                    sl_distance=decision.sl_distance, tp_distance=decision.tp_distance,
                    timeout_s=self.limit_timeout_s,
                    client_order_id_root=signal_id,
                    place_tp=place_tp,
                )
            except UnprotectedFillError as e:
                self._handle_unprotected_fill(e, decision, signal_id, equity)
                return
            trade_events.record_limit_entry(
                side=decision.side,
                filled_qty=float(orders.get("filled_qty", qty)),
                fill_price=float(orders.get("fill_price", decision.price)),
                sl_distance=decision.sl_distance,
                tp_distance=decision.tp_distance,
                limit_price=limit_price,
                signal_price=decision.price,
                limit_offset_bps=self.limit_offset_bps,
                equity=equity, signal_id=signal_id,
                strategy_name=self.strategy_name,
                filled_as=str(orders.get("filled_as", "limit")),
                orders=orders, dbg=decision.debug,
            )
            self._alert_tp_error(orders, decision.side, signal_id)
            self._note_protected_entry()
            return

        try:
            orders = self.client.market_order_with_bracket(
                self.symbol, decision.side, qty,
                decision.sl_price, decision.tp_price,
                client_order_id_root=signal_id,
                place_tp=place_tp,
            )
        except UnprotectedFillError as e:
            self._handle_unprotected_fill(e, decision, signal_id, equity)
            return
        trade_events.record_market_entry(
            side=decision.side, qty=qty, price=decision.price,
            sl_price=decision.sl_price, tp_price=decision.tp_price,
            equity=equity, signal_id=signal_id,
            strategy_name=self.strategy_name,
            orders=orders, dbg=decision.debug,
        )
        self._alert_tp_error(orders, decision.side, signal_id)
        self._note_protected_entry()

    # --- no-stop handling (God's rule, 2026-10-13) ---------------------------
    # "If a stop-loss can't be placed (or re-placed), retry 2–3×; if it still
    # can't, close the position at market and alert — all 3 legs." The client
    # does the retries and the close; everything below RECORDS what happened and
    # keeps trying to close if the close itself failed.
    NO_STOP_PENDING_KEY = "no_stop_close_pending"
    UNVERIFIED_KEY = "entry_unverified"
    LATCH_KEY = "no_stop_latch"
    NO_STOP_ALERT_EVERY_S = 900.0      # repeat a FAILED alert at most every 15 min
    NO_STOP_RETRY_MIN_S = 15.0
    NO_STOP_RETRY_MAX_S = 300.0
    LATCH_AFTER = 2                    # consecutive non-transient no-stop entries
    LATCH_HOLD_S = 4 * 3600.0          # God, 2026-10-13: auto-release after 4 h
    UNVERIFIED_POLL_S = 15.0
    UNVERIFIED_SETTLE_S = 60.0

    # --- small meta helpers ------------------------------------------------
    def _meta_record_ex(self, key: str, kind: str) -> tuple[dict | None, bool]:
        """(record, read_ok). read_ok False = the read itself failed: unknown,
        which is NOT the same as "no record"."""
        try:
            raw = state.get_meta(key)
        except Exception:
            self.log.exception("no-stop: meta read %s failed", key)
            return None, False
        if not raw:
            return None, True
        try:
            d = json.loads(raw)
        except (ValueError, TypeError):
            return None, True
        return (d if isinstance(d, dict) and d.get("kind") == kind else None), True

    def _meta_record(self, key: str, kind: str) -> dict | None:
        return self._meta_record_ex(key, kind)[0]

    def _set_meta_quietly(self, key: str, value: str) -> bool:
        try:
            state.set_meta(key, value)
            return True
        except Exception:
            self.log.exception("no-stop: meta write %s failed", key)
            return False

    def _pending_record(self) -> dict | None:
        if self._no_stop_pending is not None:
            return self._no_stop_pending
        if self._pending_meta_stale:
            return None     # cleared in memory; the meta row is a failed-to-clear leftover
        return self._meta_record(self.NO_STOP_PENDING_KEY, "no_stop_close")

    def _clear_pending(self) -> None:
        self._no_stop_pending = None
        self._no_stop_backoff_s = self.NO_STOP_RETRY_MIN_S
        self._no_stop_next_try = 0.0
        self._no_stop_alert_ts = 0.0
        self._pending_meta_stale = not self._set_meta_quietly(self.NO_STOP_PENDING_KEY, "")

    # entry_unverified: memory first (set before the persist attempt), meta as
    # the restart-surviving fallback; cleared in both.
    def _unverified_record_ex(self) -> tuple[dict | None, bool]:
        if self._unverified is not None:
            return self._unverified, True
        if self._unverified_meta_stale:
            return None, True
        return self._meta_record_ex(self.UNVERIFIED_KEY, "entry_unverified")

    def _unverified_record(self) -> dict | None:
        return self._unverified_record_ex()[0]

    def _set_unverified(self, rec: dict) -> None:
        self._unverified = rec
        self._unverified_meta_stale = False
        self._set_meta_quietly(self.UNVERIFIED_KEY, json.dumps(rec))

    def _clear_unverified(self) -> None:
        self._unverified = None
        self._unverified_meta_stale = not self._set_meta_quietly(self.UNVERIFIED_KEY, "")

    def _entry_blocked(self) -> bool:
        """Fails CLOSED: an unreadable record blocks entries (the safe side)."""
        if self._pending_record() is not None:
            return True
        if not self._pending_meta_stale and \
                not self._meta_record_ex(self.NO_STOP_PENDING_KEY, "no_stop_close")[1]:
            return True
        u, ok = self._unverified_record_ex()
        if u is not None or not ok:
            return True
        return self._latch_active()

    # --- latch (God, 2026-10-13) -----------------------------------------
    # Two consecutive entries closed by the no-stop rule for a NON-transient
    # reason (not timeout / network / -1021) ⇒ refuse entries for 4 h, alert
    # once. Releases itself; two more re-latch it. A protected entry resets
    # the streak. Without this, a persistent rejection (e.g. -4045) would enter
    # and immediately close on every signal.
    def _latch_active(self) -> bool:
        lt, ok = self._meta_record_ex(self.LATCH_KEY, "no_stop_latch")
        if not ok:
            # unknown != unlatched: block entries (the safe side), and say so
            now = time.time()
            if now - self._latch_unreadable_alert_ts >= 3600.0:
                self._latch_unreadable_alert_ts = now
                send_alert("Entries BLOCKED — no-stop latch state unreadable",
                           "The DB read of the no-stop latch failed, so entries are "
                           "refused until it can be read (fails closed). Check the "
                           "bot's database.")
            return True
        if (lt is None or not lt.get("until")) and self._latch_persisted:
            # the persisted row was deleted by hand (the documented early
            # release): honour it. Memory alone wins only if the persist failed.
            self._latch_until = 0.0
            self._latch_persisted = False
        until = max(self._latch_until,
                    float(lt["until"]) if lt and lt.get("until") else 0.0)
        if not until:
            return False
        if time.time() < until:
            return True
        self._latch_until = 0.0
        self._latch_streak = 0
        self._set_meta_quietly(self.LATCH_KEY, json.dumps(
            {"kind": "no_stop_latch", "streak": 0, "until": None}))
        if self._latch_released_for == until:
            return False    # already announced; a failed reset write must not repeat it
        self._latch_released_for = until
        self.log.warning("no-stop latch RELEASED after %.0f h — entries resume",
                         self.LATCH_HOLD_S / 3600)
        send_alert("No-stop latch released — entries resume",
                   f"The {self.LATCH_HOLD_S / 3600:.0f} h no-stop entry latch expired. "
                   f"It re-latches after {self.LATCH_AFTER} more no-stop entries.")
        return False

    def _note_no_stop_entry(self, err: BaseException | None) -> None:
        if is_transient_error(err):
            return
        try:
            lt, ok = self._meta_record_ex(self.LATCH_KEY, "no_stop_latch")
            lt = lt or {"kind": "no_stop_latch", "streak": 0, "until": None}
            lt["streak"] = max(int(lt.get("streak") or 0) if ok else 0,
                               self._latch_streak) + 1
            self._latch_streak = lt["streak"]
            latched = lt["streak"] >= self.LATCH_AFTER
            if latched:
                lt["until"] = self._latch_until = time.time() + self.LATCH_HOLD_S
                lt["streak"] = self._latch_streak = 0
            # Persist BEFORE alerting "latched"; memory already holds it, so a
            # failed write still blocks entries in this process.
            persisted = self._set_meta_quietly(self.LATCH_KEY, json.dumps(lt))
            if latched:
                self._latch_persisted = persisted
            if not latched:
                return
            self.log.critical("no-stop latch SET: %d consecutive entries could not "
                              "get a stop (%s) — no entries for %.0f h",
                              self.LATCH_AFTER, err, self.LATCH_HOLD_S / 3600)
            send_alert("ENTRIES LATCHED — stops keep failing",
                       f"{self.LATCH_AFTER} consecutive entries were closed because no "
                       f"stop-loss could be placed (last error: {err}).\n"
                       f"New entries are refused for {self.LATCH_HOLD_S / 3600:.0f} h, "
                       f"then resume automatically. "
                       + (f"Delete meta '{self.LATCH_KEY}' to release early."
                          if persisted else
                          "The DB write FAILED: the latch is held in memory only and "
                          "is lost if the bot restarts."))
        except Exception:
            self.log.exception("no-stop: latch bookkeeping failed")

    def _note_protected_entry(self) -> None:
        self._latch_streak = 0
        lt = self._meta_record(self.LATCH_KEY, "no_stop_latch")
        if lt and int(lt.get("streak") or 0):
            lt["streak"] = 0
            self._set_meta_quietly(self.LATCH_KEY, json.dumps(lt))

    # --- alerts / recording ------------------------------------------------
    def _alert_tp_error(self, orders: dict, side: str, signal_id: str) -> None:
        err = (orders or {}).get("tp_error")
        if not err:
            return
        self.log.error("TP not placed for %s (signal_id=%s): %s — stop IS resting",
                       side, signal_id, err)
        state.record_event("WARN", "tp_place_failed", {"side": side, "error": err},
                           signal_id=signal_id)
        send_alert("Bot TP NOT placed — stop is in place",
                   f"{side.upper()} entry filled and its stop-loss is resting, but the "
                   f"take-profit could not be placed: {err}\n"
                   f"The position is protected by the stop. "
                   + ("Reprotect will retry the TP."
                      if ((rp := self.params.get("reprotect") or {}).get("enabled")
                          and not rp.get("observe_only", True))
                      else "No automatic TP retry on this leg — the trend exit "
                           "still applies.")
                   + f"\nsignal_id: {signal_id}")

    def _record_entry_fill(self, side: str, qty: float, price: float,
                           signal_id: str, equity: float, filled_as: str) -> None:
        """Entry row for a fill that never got its stop at entry time. Recorded
        so the ledger pairs it with its close — the old path wrote nothing."""
        state.record_fill(side=side, qty=qty, price=price, reason="entry",
                          equity_after=equity, client_order_id_root=signal_id)
        state.enqueue_bot_event(
            "entry", signal_id=signal_id, strategy=self.strategy_name,
            side=side, qty=float(qty), price_usd=float(price),
            notional_usd=float(qty * price), equity_usd=float(equity),
            payload={"filled_as": filled_as, "no_stop": True})

    def _latest_entry_root_quietly(self) -> str | None:
        try:
            return state.latest_entry_coid_root()
        except Exception:
            self.log.exception("no-stop: latest entry root read failed")
            return None

    def _entry_row_exists(self, root: str | None) -> bool | None:
        """True/False = an entry row for `root` is/isn't in the ledger; None = unreadable."""
        try:
            with sqlite3.connect(state.DB_PATH) as c:
                return c.execute("SELECT 1 FROM fills WHERE reason = 'entry' AND "
                                 "client_order_id_root = ? LIMIT 1",
                                 (str(root),)).fetchone() is not None
        except Exception:
            self.log.exception("no-stop: entry row lookup failed")
            return None

    def _newest_fill_is_entry_of(self, root: str | None) -> bool:
        try:
            with sqlite3.connect(state.DB_PATH) as c:
                row = c.execute("SELECT reason, client_order_id_root FROM fills "
                                "ORDER BY id DESC LIMIT 1").fetchone()
        except Exception:
            self.log.exception("no-stop: fills read failed")
            return False
        return bool(row) and row[0] == "entry" and str(row[1]) == str(root)

    def _alert_no_stop_failed(self, side: str, qty: float, entry_price: float | None,
                              stop_price: float | None, root: str | None,
                              err: BaseException | None, why: str, force: bool) -> None:
        now = time.time()
        if not force and now - self._no_stop_alert_ts < self.NO_STOP_ALERT_EVERY_S:
            return
        self._no_stop_alert_ts = now
        mark = self.client.fetch_mark_price(self.symbol)   # never raises
        unreal = None
        if mark and entry_price:
            unreal = (mark - float(entry_price)) * qty * (1.0 if side == "long" else -1.0)
        send_alert("NO STOP — emergency close FAILED, position OPEN",
                   f"{side.upper()} {qty:.4f} {self.base_asset} has NO stop-loss and "
                   f"the market close failed: {err}\n"
                   f"Why: {why}\n"
                   + (f"Entry: {float(entry_price):,.4f}  " if entry_price else "")
                   + (f"Intended stop: {float(stop_price):,.4f}\n" if stop_price else "\n")
                   + (f"Mark: {mark:,.4f}  Unrealized: ${unreal:+,.2f}\n"
                      if mark and unreal is not None else "Mark: unavailable\n")
                   + f"The bot keeps retrying (backoff up to "
                   f"{self.NO_STOP_RETRY_MAX_S:.0f}s; this alert repeats at most every "
                   f"{self.NO_STOP_ALERT_EVERY_S / 60:.0f} min). CHECK BINANCE NOW.\n"
                   f"signal_id: {root or '(untagged)'}")

    def _handle_unprotected_fill(self, e: UnprotectedFillError,
                                 decision: SignalDecision, signal_id: str,
                                 equity: float) -> None:
        price = float(e.fill_price or decision.price)
        try:
            self._record_entry_fill(e.side, e.filled_qty, price, signal_id,
                                    equity, e.filled_as)
            state.record_event("ERROR", "no_stop",
                               {"side": e.side, "qty": e.filled_qty, "price": price,
                                "sl_price": e.sl_price,
                                "sl_error": str(e.sl_error), "closed": e.closed,
                                "close_error": str(e.close_error or "")},
                               signal_id=signal_id)
        except Exception:
            self.log.exception("no-stop: recording the entry fill failed")
        self._note_no_stop_entry(e.sl_error)
        self._finish_no_stop(e.side, e.filled_qty, price, signal_id, equity,
                             closed=e.closed, order=e.close_order,
                             close_error=e.close_error,
                             why=f"stop could not be placed after the entry filled: "
                                 f"{e.sl_error}",
                             record_if_flat=True, stop_price=e.sl_price)

    def _finish_no_stop(self, side: str, qty: float, entry_price: float | None,
                        root: str | None, equity: float, *, closed: bool,
                        order: dict | None, close_error: BaseException | None,
                        why: str, record_if_flat: bool,
                        stop_price: float | None = None) -> None:
        """Record + alert the outcome of an emergency (no-stop) close.

        `order` None with `closed` True means the exchange was already flat.
        Whether that still needs a close row depends on the caller: if the
        bracket-exit detector never saw the position open, nobody else will
        write it (record_if_flat=True); if it did, its flat edge records it.
        """
        if not closed:
            pending = {"kind": "no_stop_close", "root": root, "side": side,
                       "qty": float(qty), "entry_price": entry_price,
                       "stop_price": stop_price, "why": why}
            first = self._pending_record() is None
            # In memory FIRST: if the sqlite write fails this process still
            # retries (a restart then goes through boot(), which flattens).
            self._no_stop_pending = pending
            self._pending_meta_stale = False
            self._set_meta_quietly(self.NO_STOP_PENDING_KEY, json.dumps(pending))
            now = time.time()
            if first:
                self._no_stop_backoff_s = self.NO_STOP_RETRY_MIN_S
            else:
                self._no_stop_backoff_s = min(self._no_stop_backoff_s * 2,
                                              self.NO_STOP_RETRY_MAX_S)
            self._no_stop_next_try = now + self._no_stop_backoff_s
            self.log.critical("NO STOP and the emergency close FAILED: %s %.4f "
                              "(root=%s): %s — retry in %.0fs",
                              side, qty, root, close_error, self._no_stop_backoff_s)
            self._alert_no_stop_failed(side, qty, entry_price, stop_price, root,
                                       close_error, why, force=first)
            return
        self._clear_pending()
        self._set_meta_quietly("active_bracket", "")
        exit_price = self._resolve_fill_price(order) if order is not None else None
        pnl = _exit_pnl_usd(side, entry_price, exit_price, qty)
        if order is not None or record_if_flat:
            try:
                state.record_fill(side="close", qty=qty, price=float(exit_price or 0.0),
                                  pnl_usd=pnl, reason="no_stop_close",
                                  equity_after=equity, client_order_id_root=root)
                state.enqueue_bot_event(
                    "exit", signal_id=root, side=side, qty=float(qty),
                    price_usd=exit_price, equity_usd=float(equity),
                    payload={"reason": "no_stop_close", "why": why,
                             "entry_price": entry_price, "exit_price": exit_price,
                             "intended_stop": stop_price,
                             "pnl_usd": pnl, "already_flat": order is None})
            except Exception:
                self.log.exception("no-stop: recording the close failed")
        self.log.critical("NO STOP — position closed: %s %.4f (root=%s) @ %s pnl=%s — %s",
                          side, qty, root, exit_price, pnl, why)
        send_alert("NO STOP — position closed",
                   f"{side.upper()} {qty:.4f} {self.base_asset} could not be given a "
                   f"stop-loss, so it was closed at market"
                   + (f" @ {exit_price:,.4f}" if exit_price else
                      (" (exchange was already flat)" if order is None else ""))
                   + f".\nWhy: {why}\n"
                   + (f"Entry: {float(entry_price):,.4f}  " if entry_price else "")
                   + (f"Intended stop: {float(stop_price):,.4f}\n" if stop_price else "\n")
                   + (f"PnL: ${pnl:+,.2f}\n" if pnl is not None else "PnL: unknown\n")
                   + f"Equity: {equity:.2f}\nsignal_id: {root or '(untagged)'}")

    def _emergency_close(self, pos, root: str | None, equity: float, why: str,
                         stop_price: float | None = None,
                         record_if_flat: bool = False,
                         coid_root: str | None = None) -> None:
        if self.dry_run:
            return
        extra = {"coid_root": coid_root} if coid_root else {}
        closed, order, err = self.client.emergency_close(
            self.symbol, root, side=pos.side, qty=pos.qty, **extra)
        entry = self._open_entry_fill()
        self._finish_no_stop(pos.side, pos.qty,
                             entry[2] if entry else pos.entry_price, root, equity,
                             closed=closed, order=order, close_error=err, why=why,
                             record_if_flat=record_if_flat, stop_price=stop_price)

    def _maybe_finish_no_stop_close(self, equity: float) -> None:
        """Retry an emergency close that failed — ONLY for the position it was
        for. Local reads only while nothing is pending.

        Never touches a position it cannot prove is the pending one (side, qty,
        and the newest entry root): a stale pending record used to sweep and
        close the NEXT trade. Entries are blocked while a record is pending, so
        a mismatch means a manual position — left alone, with an alert.
        """
        if self.dry_run:
            return
        p = self._pending_record()
        if p is None:
            return
        now = time.time()
        if now < self._no_stop_next_try:
            return
        root = p.get("root")
        side = str(p.get("side"))
        qty = float(p.get("qty") or 0.0)
        try:
            pos = self.client.fetch_position(self.symbol)
        except Exception as e:
            self.log.warning("no-stop retry: position read failed: %s", e)
            self._no_stop_next_try = now + self._no_stop_backoff_s
            return
        if pos.side == "flat" or not pos.qty:
            self._clear_pending()
            if self._newest_fill_is_entry_of(root):
                try:
                    state.record_fill(side="close", qty=qty, price=0.0, pnl_usd=None,
                                      reason="no_stop_close", equity_after=equity,
                                      client_order_id_root=root)
                except Exception:
                    self.log.exception("no-stop: recording the outside close failed")
            self.log.warning("no-stop retry: position is FLAT — closed outside the "
                             "bot; pending close cleared (root=%s)", root)
            send_alert("NO STOP — position now flat (closed outside the bot)",
                       f"The {side.upper()} {qty:.4f} {self.base_asset} position the bot "
                       f"was trying to close is flat. Pending close cleared; entries "
                       f"resume.\nsignal_id: {root or '(untagged)'}")
            return
        tol = max(float(self.constraints.qty_step or 0.0) / 2, 1e-12)
        # Abandon only on PROOF it is not ours: other side/size, or a NEWER
        # entry row exists. A missing/unreadable entry row for `root` proves
        # nothing (its record write can fail) — entries are blocked while this
        # is pending, so a position of the same side and size is the naked one:
        # close it (and say so) rather than leave it open.
        latest = self._latest_entry_root_quietly()
        has_row = self._entry_row_exists(root)
        newer_entry = bool(has_row) and latest is not None and str(latest) != str(root)
        if pos.side != side or abs(pos.qty - qty) > tol or newer_entry:
            self._clear_pending()
            self.log.error("no-stop retry ABANDONED: open %s %.4f (latest root %s) is "
                           "not the pending %s %.4f (root %s) — NOT touched",
                           pos.side, pos.qty, latest, side, qty, root)
            send_alert("NO STOP — pending close ABANDONED (position differs)",
                       f"The bot was retrying a close of {side.upper()} {qty:.4f} "
                       f"(signal_id {root}), but the open position is {pos.side.upper()} "
                       f"{pos.qty:.4f}. It was NOT touched. Check Binance.")
            return
        if not has_row and self._unproven_alerted_root != root:
            self._unproven_alerted_root = root
            send_alert("NO STOP — closing a position whose entry record is missing",
                       f"Retrying the close of {side.upper()} {qty:.4f} "
                       f"(signal_id {root}): its entry row could not be confirmed in the "
                       f"ledger, but the open position matches side and size and entries "
                       f"are blocked, so it is treated as the naked one.")
        # A fresh -ns id every retry: a reused id that already landed (position
        # read lagging) would hit the duplicate path and report a failed close.
        closed, order, err = self.client.emergency_close(
            self.symbol, root, side=pos.side, qty=pos.qty,
            coid_root=str(int(time.time() * 1000)))
        if not closed:
            self._finish_no_stop(side, qty, p.get("entry_price"), root, equity,
                                 closed=False, order=None, close_error=err,
                                 why=str(p.get("why") or "pending no-stop close"),
                                 record_if_flat=False, stop_price=p.get("stop_price"))
            return
        self._finish_no_stop(side, qty, p.get("entry_price"), root, equity,
                             closed=True, order=order, close_error=None,
                             why=str(p.get("why") or "pending no-stop close"),
                             record_if_flat=False, stop_price=p.get("stop_price"))

    def _protect_or_close_open(self, pos, root: str, sl_distance: float,
                               tp_distance: float, place_tp: bool, equity: float,
                               why: str, record_if_flat: bool,
                               may_close: bool = True,
                               entry_qty: float | None = None) -> bool:
        """The no-stop rule for a position the normal entry path did not
        bracket (an entry that raised, or a limit that filled late): keep it if
        its exact-root stop rests, else place one (with retry), else close.

        Returns False when the order books could not be read: unknown is not
        "no stop" (that would place a duplicate or close a protected position),
        so nothing is placed or closed and the caller must call again. If the
        books stay unreadable for `no_stop.unknown_book_close_after_s` (300 s)
        the position is closed at market (reduce-only needs no book read):
        God's rule is "no verifiable stop => close"."""
        try:
            if not self._newest_fill_is_entry_of(root):
                self._record_entry_fill(pos.side, float(entry_qty or pos.qty),
                                        pos.entry_price, root, equity,
                                        "unknown_after_error")
            if self._last_position_side == pos.side:
                # the detector already snapshotted this position with the
                # PREVIOUS root (the entry row did not exist yet)
                self._last_entry_root = root
        except Exception:
            self.log.exception("no-stop: recording the entry fill failed")
        try:
            open_orders = self.client.ex.fetch_open_orders(self.symbol)
            algo_rows, algo_ok = self.client.fetch_algo_orders(self.symbol)
            if not algo_ok:
                raise RuntimeError("algo order book unreadable")
            has_stop = adopt_bracket_state(open_orders, algo_rows, self.coid_prefix,
                                           root, pos.side).sl
        except Exception:
            self._book_unknown_n += 1
            now_m = time.monotonic()
            if self._book_unknown_since is None:
                self._book_unknown_since = now_m
            unknown_s = now_m - self._book_unknown_since
            limit_s = float((self.params.get("no_stop") or {})
                            .get("unknown_book_close_after_s", 300.0))
            if unknown_s >= limit_s and may_close:
                self.log.critical("no-stop: books unreadable for %.0fs — closing %s "
                                  "%.4f (root=%s)", unknown_s, pos.side, pos.qty, root)
                self._book_unknown_since = None
                self._book_unknown_n = 0
                self._emergency_close(
                    pos, root, equity, record_if_flat=record_if_flat,
                    coid_root=str(int(time.time() * 1000)),
                    why=f"{why}; protection could not be verified for "
                        f"{unknown_s:.0f}s (order books unreadable)")
                return True
            self.log.exception("no-stop: book read failed — cannot tell if a stop "
                               "rests; nothing placed or closed (%d)", self._book_unknown_n)
            if self._book_unknown_n == 3:
                send_alert("NO STOP check blocked — order book unreadable",
                           f"{pos.side.upper()} {pos.qty:.4f} {self.base_asset} is open "
                           f"({why}) and the bot cannot read the order books to see "
                           f"whether its stop rests. It keeps retrying and places / "
                           f"closes nothing until it can read them. CHECK BINANCE.\n"
                           f"signal_id: {root}")
            return False
        self._book_unknown_n = 0
        self._book_unknown_since = None
        if has_stop:
            self._note_protected_entry()
            send_alert("Entry error — position open WITH its stop",
                       f"{pos.side.upper()} {pos.qty:.4f} {self.base_asset} is open "
                       f"({why}) and its stop is resting. Left open.\n"
                       f"signal_id: {root}")
            return True
        sign = 1.0 if pos.side == "long" else -1.0
        sl_price = pos.entry_price - sign * sl_distance
        tp_price = pos.entry_price + sign * tp_distance
        try:
            _sl, _tp, tp_err = self.client.protect_open_position(
                self.symbol, pos.side, pos.qty, sl_price, tp_price, root, place_tp,
                pos.entry_price, close_on_failure=may_close)
        except UnprotectedFillError as e:
            if not may_close:
                self.log.critical("NO STOP on a partly-attributable %s %.4f (root=%s): "
                                  "%s — NOT closed (not wholly ours)",
                                  pos.side, pos.qty, root, e.sl_error)
                send_alert("NO STOP — position NOT closed (not wholly the bot's)",
                           f"{pos.side.upper()} {pos.qty:.4f} {self.base_asset} is open, "
                           f"only {float(entry_qty or 0):.4f} of it is attributable to "
                           f"entry {root}, and no stop could be placed: {e.sl_error}.\n"
                           f"Intended stop: {sl_price:,.4f}. The bot does NOT close a "
                           f"position that is not wholly its own. CHECK BINANCE NOW.")
                return True
            self._note_no_stop_entry(e.sl_error)
            self._finish_no_stop(pos.side, pos.qty, pos.entry_price, root, equity,
                                 closed=e.closed, order=e.close_order,
                                 close_error=e.close_error,
                                 why=f"{why}; stop could not be placed: {e.sl_error}",
                                 record_if_flat=record_if_flat, stop_price=sl_price)
            return True
        self._note_protected_entry()
        send_alert("Entry error — stop placed on the open position",
                   f"{pos.side.upper()} {pos.qty:.4f} {self.base_asset} @ "
                   f"{pos.entry_price:,.4f} opened without its bracket ({why}). "
                   f"Stop placed @ {sl_price:,.4f}"
                   + (f"; TP NOT placed: {tp_err}" if tp_err else "")
                   + f".\nsignal_id: {root}")
        return True

    def _entry_error_left_position(self, decision: SignalDecision, signal_id: str,
                                   equity: float, err: BaseException) -> bool:
        """After an entry ERROR: did anything fill, or can it still? True =
        handled (caller must not also say "entry failed").

        1. Cancel `{root}-e` and confirm it is gone — a limit whose create
           timed out can still be RESTING and fill minutes later.
        2. Re-read the position (twice: positionRisk can lag a fill).
        3. Open ⇒ the no-stop rule (stop with retry, else close).
           Flat but the entry order could not be confirmed gone ⇒ record it as
           UNVERIFIED: entries are blocked and _maybe_guard_unverified_entry
           watches for a late fill.
        """
        if self.dry_run:
            return False
        gone = self.client.cancel_entry_order(self.symbol, signal_id)
        pos = None
        for attempt in range(2):
            if attempt:
                time.sleep(2.0)
            try:
                pos = self.client.fetch_position(self.symbol)
            except Exception:
                self.log.exception("post-entry-error position check failed")
                continue
            if pos.side == decision.side and pos.qty:
                break
        place_tp = not strategy_uses_channel_exit(self.strategy_name)
        if pos is not None and pos.side == decision.side and pos.qty:
            if not self._protect_or_close_open(
                    pos, signal_id, decision.sl_distance, decision.tp_distance,
                    place_tp, equity, why=f"entry raised ({err})",
                    record_if_flat=True):
                # Books unreadable: hand to the 15 s guard, which retries the
                # same call (flag = the side alone was trusted here, as above).
                self._set_unverified({
                    "kind": "entry_unverified", "root": signal_id,
                    "side": decision.side, "retry_protect": True,
                    "sl_distance": float(decision.sl_distance),
                    "tp_distance": float(decision.tp_distance),
                    "place_tp": bool(place_tp), "since": time.time(),
                    "error": str(err)})
            return True
        # ALWAYS watch for a settle window, even when the first read says the
        # entry order is gone: a timed-out create can still be in flight and
        # land on the book AFTER that read, then fill with no stop. The guard
        # re-checks every 15 s and clears only after UNVERIFIED_SETTLE_S with
        # the order confirmed gone and the position flat.
        self._set_unverified({
            "kind": "entry_unverified", "root": signal_id, "side": decision.side,
            "sl_distance": float(decision.sl_distance),
            "tp_distance": float(decision.tp_distance),
            "place_tp": bool(place_tp), "since": time.time(), "error": str(err)})
        if gone is True:
            return False     # the caller's "entry failed" alert is accurate
        send_alert("Entry error — entry order may still be resting",
                   f"{decision.side} entry raised ({err}) and the bot could not "
                   f"confirm the entry order is gone. Entries are blocked; it is "
                   f"watched every {self.UNVERIFIED_POLL_S:.0f}s and gets a stop "
                   f"(or is closed) if it fills.\nsignal_id: {signal_id}")
        return True

    def _maybe_guard_unverified_entry(self, equity: float) -> None:
        """Watch an entry whose order state is unknown after an error (see
        _entry_error_left_position). Local read only while nothing is
        unverified — this is the ONLY per-tick no-stop guard on donchian/SOL,
        deliberately scoped to the case the bot itself created, so a healthy
        tick on any leg makes no extra exchange call. A general "is the stop
        still there" poll on those legs is reprotect, which needs its own
        observe-only pass first (params.yaml)."""
        if self.dry_run:
            return
        u = self._unverified_record()
        if u is None:
            return
        now = time.time()
        if now < self._unverified_next:
            return
        self._unverified_next = now + self.UNVERIFIED_POLL_S
        root = str(u.get("root"))
        try:
            pos = self.client.fetch_position(self.symbol)
        except Exception as e:
            self.log.warning("unverified entry: position read failed: %s", e)
            return
        if pos.side == "flat" or not pos.qty:
            if now - float(u.get("since") or 0) >= self.UNVERIFIED_SETTLE_S \
                    and self.client.cancel_entry_order(self.symbol, root) is True:
                self._clear_unverified()
                self.log.warning("unverified entry %s: confirmed gone, nothing filled", root)
                send_alert("Entry error resolved — nothing filled",
                           f"The entry order {root} is confirmed gone and no position "
                           f"opened. Entries resume.")
            return
        if u.get("retry_protect") and pos.side == u.get("side"):
            if self._protect_or_close_open(
                    pos, root, float(u["sl_distance"]), float(u["tp_distance"]),
                    bool(u.get("place_tp", True)), equity,
                    why=f"entry raised ({u.get('error')})", record_if_flat=True):
                self._clear_unverified()
            return
        if pos.side != u.get("side"):
            self._clear_unverified()
            send_alert("Entry error — open position is not ours, untouched",
                       f"Watching entry {root} ({u.get('side')}) but a {pos.side} "
                       f"{pos.qty:.4f} position is open. NOT touched. Check Binance.")
            return
        # Possibly a late fill. Stop any remainder from adding to it, then
        # ATTRIBUTE: the side alone could be a manual position. The entry
        # order's own filled qty (read by its client id) is the root proof.
        self.client.cancel_entry_order(self.symbol, root)
        filled = self.client.entry_order_filled(self.symbol, root)
        try:
            pos = self.client.fetch_position(self.symbol)
        except Exception as e:
            self.log.warning("unverified entry: re-read failed: %s", e)
            return
        if pos.side == "flat" or not pos.qty:
            return
        tol = max(float(self.constraints.qty_step or 0.0) / 2, 1e-12)
        if filled is None:
            if not u.get("unattributed_alerted"):
                u["unattributed_alerted"] = True
                self._set_unverified(u)
                send_alert("Entry error — open position cannot be attributed, untouched",
                           f"A {pos.side} {pos.qty:.4f} position is open while entry "
                           f"{root} is unverified, and that order cannot be read. NOT "
                           f"touched; the bot keeps checking. Check Binance.")
            return
        if filled <= tol:
            self._clear_unverified()
            send_alert("Entry error — open position is not ours, untouched",
                       f"Entry {root} filled nothing, but a {pos.side} {pos.qty:.4f} "
                       f"position is open. NOT touched. Check Binance.")
            return
        wholly_ours = pos.qty <= filled + tol
        # The record stays until the books were readable, so an unreadable tick
        # is retried 15 s later instead of being dropped with the position naked.
        if self._protect_or_close_open(pos, root, float(u["sl_distance"]),
                                       float(u["tp_distance"]),
                                       bool(u.get("place_tp", True)), equity,
                                       why=("entry order filled late after an error"
                                            if wholly_ours else
                                            f"entry order filled late ({filled:.4f}) "
                                            f"inside a larger {pos.qty:.4f} position"),
                                       record_if_flat=False, may_close=wholly_ours,
                                       entry_qty=filled):
            self._clear_unverified()

    def _clear_no_stop_records_at_boot(self) -> None:
        """boot() either flattened (sweeping our orders, incl. a resting `-e`)
        or found flat: an old pending close / unverified entry is moot, and
        retrying it would only raise a redundant alert. Kept when boot ADOPTED
        the position — the pending hook then re-verifies and closes it."""
        if self.dry_run or self._last_position_side not in ("unknown", "flat"):
            return
        if self._pending_record() is not None:
            self.log.warning("boot: clearing a pending no-stop close (boot flattened)")
            self._clear_pending()
        u = self._unverified_record()
        if u is not None:
            # Only on PROOF the entry order is gone: the boot sweep swallows a
            # failed open-orders read and reports 0, which is not proof.
            if self.client.cancel_entry_order(self.symbol, str(u.get("root"))) is True:
                self._clear_unverified()
            else:
                self.log.warning("boot: entry %s still unverified — kept; the guard "
                                 "keeps watching", u.get("root"))

    def _no_stop_record_blocks_adopt(self) -> str | None:
        """Reason boot must NOT adopt, or None. A pending no-stop close / unverified
        entry means the position may be naked or not wholly ours: flatten it (the
        records are then cleared by _clear_no_stop_records_at_boot). Fails CLOSED:
        an unreadable meta row blocks."""
        if self._no_stop_pending is not None:
            return "no_stop_close_pending record exists"
        if not self._pending_meta_stale:
            rec, ok = self._meta_record_ex(self.NO_STOP_PENDING_KEY, "no_stop_close")
            if rec is not None:
                return "no_stop_close_pending record exists"
            if not ok:
                return "no_stop_close_pending meta unreadable"
        u, ok = self._unverified_record_ex()
        if u is not None:
            return "entry_unverified record exists"
        if not ok:
            return "entry_unverified meta unreadable"
        return None

    def _can_adopt(self, pos) -> tuple[bool, str]:
        """Decide whether boot() may RESUME an open position instead of closing it.

        boot() flattens whatever it finds, because a fresh process cannot assume
        a position it did not open still has a live bracket behind it. That is a
        real cost, not a theoretical one: it has killed four live positions
        (sol 08-10 and 09-11, donchian 08-26, v1 07-21), and for sol it accounts
        for two of the three infrastructure deaths in a five-entry record.

        Arming reprotect is what makes resuming defensible — the bot can now see
        a missing bracket across BOTH order books and restore it. So this gate is
        deliberately not new machinery: it is the SAME identity + readability
        checks `_maybe_reprotect` already applies, asked once at boot.

        Returns (verdict, reason). The reason is the whole point during the
        observe-only phase: this event is far too rare to prove anything by
        silence, so the log line IS the evidence.

        FAIL-CLOSED. Every unknown, every exception, every missing record returns
        False, and the caller flattens exactly as it does today. The bar to
        resume is affirmative proof; the bar to flatten is anything less.
        """
        # The decision itself lives in bot_internals.can_adopt (C1), so the
        # read-only probe (tools/boot_resume_probe.py) runs the SAME code. This
        # wrapper only does the I/O, and only reads the books once the cheap
        # checks have passed — the same order Phase A used inline.
        ab_raw = state.get_meta("active_bracket")
        qty_step = float(getattr(self.constraints, "qty_step", 0.0) or 0.0)
        self._adopt_root = None
        self._adopt_reason = None
        self._adopt_reprotect_capped = False
        try:
            latest_root = state.latest_entry_coid_root()
        except Exception:
            self.log.exception("boot-resume: latest_entry_coid_root failed")
            return False, "latest entry root unreadable"
        why, _ab = adopt_precheck(self.params, pos, ab_raw, qty_step, latest_root)
        if why is not None:
            return False, why
        open_orders, algo_rows, algo_ok = self._read_books_with_retry()
        verdict = can_adopt(self.params, pos, ab_raw, open_orders, algo_rows,
                            algo_ok, qty_step, self.coid_prefix,
                            latest_root=latest_root)
        if verdict[0]:
            self._adopt_root = latest_root
            # C10: the cap is persisted per position; if it is already spent the
            # pre-restart process has sent the cap alert, so do not send it again.
            try:
                cap = int((self.params.get("reprotect") or {})
                          .get("max_replaces_per_position", 3))
                done = int((_ab or {}).get("reprotect_count", 0))
                self._adopt_reprotect_capped = done >= cap
            except (TypeError, ValueError):
                return False, "reprotect_count unparseable"
        return verdict

    def _read_books_with_retry(self) -> tuple[list | None, list | None, bool]:
        """C9: read BOTH order books, retrying an unreadable read at boot.

        Returns (open_orders, algo_rows, algo_ok) in the shape `can_adopt`
        expects: `open_orders=None` / `algo_ok=False` still mean "could not
        read", and after BOOT_BOOK_READ_ATTEMPTS failures that is exactly what
        is returned, so the gate refuses and boot flattens as before. Each
        attempt re-reads BOTH books, so the verdict is always taken on a pair
        read in the same attempt, never a stale plain book glued to a fresh
        algo book. No sleep after the last attempt.
        """
        open_orders: list | None = None
        algo_rows: list | None = None
        algo_ok = False
        for attempt in range(1, BOOT_BOOK_READ_ATTEMPTS + 1):
            try:
                open_orders = self.client.ex.fetch_open_orders(self.symbol)
            except Exception:
                self.log.exception("boot-resume: fetch_open_orders failed "
                                   "(attempt %d/%d)", attempt,
                                   BOOT_BOOK_READ_ATTEMPTS)
                open_orders = None
            if open_orders is None:
                algo_rows, algo_ok = None, False
            else:
                algo_rows, algo_ok = self.client.fetch_algo_orders(self.symbol)
                if algo_ok:
                    return open_orders, algo_rows, True
                self.log.warning("boot-resume: algo book unreadable "
                                 "(attempt %d/%d)", attempt,
                                 BOOT_BOOK_READ_ATTEMPTS)
            if attempt < BOOT_BOOK_READ_ATTEMPTS:
                _boot_book_sleep(BOOT_BOOK_RETRY_S)
        return open_orders, algo_rows, algo_ok

    def _seed_adopted_tracking(self, pos, root: str | None) -> None:
        """Make the tick loop treat an adopted position as one it already knew.

        B1: _detect_bracket_exit only emits an exit on a tracked open→flat (or
        replaced) edge; seeding here makes a bracket fill before the first tick
        a normal flat-edge exit. B2: `_last_signal_ts` is seeded from the value
        `_boot_resume_verdict` prepared, so the bar that opened (or was last
        evaluated for) this position is not re-evaluated once it closes.

        Called on the ADOPT path only. The flatten path is deliberately left
        as it was (plan §9: its same-bar exposure is a separate decision).
        """
        self._last_position_side = pos.side
        self._last_position_entry = float(pos.entry_price)
        self._last_position_qty = float(pos.qty)
        self._last_entry_root = root
        self._reprotect_capped_alerted = bool(self._adopt_reprotect_capped)
        seed = self._adopt_signal_ts_seed
        if seed is not None:
            self._last_signal_ts = seed
        # C12 — the Phase B evidence line.
        self.log.warning(
            "adopt: tracking seeded side=%s entry=%.2f qty=%.4f root=%s "
            "last_signal_ts=%s",
            pos.side, float(pos.entry_price), float(pos.qty), root or "—",
            seed.isoformat() if seed is not None else "—")

    def _boot_resume_verdict(self, pos) -> tuple[bool, str]:
        """Config + rollout wrapper around `_can_adopt`. NEVER raises.

        Returns (adopt, reason). Only (True, ...) makes boot() resume; anything
        else — disabled, malformed config, observe mode, the arming interlock,
        a refusing or raising gate — falls through to today's flatten.

        OBSERVE mode (`observe_only: true`, what ships) must be behaviour-
        neutral, so it makes NO exchange call at all: it runs only the offline
        precheck (config + the stashed active_bracket in the local DB), logs
        that result, says the order-book check was SKIPPED, and returns False.
        The full verdict against the live books is what
        `tools/boot_resume_probe.py` produces, read-only, without a restart —
        that probe is the counted arming evidence (Phase B plan §2), not
        restarts.

        ARMED requires BOTH `observe_only: false` in config AND the code
        constant ARMING_PREREQS_BUILT. Until the latter flips, a config that
        asks for armed mode is logged at ERROR, alerted, and treated as observe.
        """
        try:
            br = self.params.get("boot_resume")
            if br is None:
                return False, "boot_resume disabled"
            if not isinstance(br, dict):
                # Malformed config must never abort boot() before the flatten.
                self.log.warning("boot-resume: config is %s, not a mapping — "
                                 "treating as disabled", type(br).__name__)
                return False, "boot_resume config malformed — treated as disabled"
            # Strict booleans, fail-closed: `enabled: "yes"` does not enable and
            # `observe_only: "false"` does not arm.
            if br.get("enabled", False) is not True:
                return False, "boot_resume disabled"
            wants_armed = br.get("observe_only", True) is False
            if wants_armed and not ARMING_PREREQS_BUILT:
                self._alert_arming_interlock(pos)
            if not (wants_armed and ARMING_PREREQS_BUILT):
                return self._observe_verdict(pos)
        except Exception:
            self.log.exception("boot-resume: rollout wrapper raised — flattening")
            return False, "boot_resume evaluation raised"

        # ---- ARMED from here: config asks for it AND the prereqs are built. ----
        # A HALT flag present at boot means the first tick will close the
        # position anyway. Adopting first would spend an adopt-loop slot and
        # send a misleading RESUMED email, so refuse up front. Local file check
        # only; any doubt refuses.
        try:
            halt_by = halt_source(self.instance)
        except Exception:
            self.log.exception("boot-resume: HALT check raised — flattening")
            return False, "HALT check raised"
        if halt_by is not None:
            self.log.warning("boot-resume ARMED: FLATTEN %s %.4f @ %.2f — HALT "
                             "flag present (%s)", pos.side, pos.qty,
                             pos.entry_price, halt_by)
            return False, f"HALT flag present ({halt_by})"
        try:
            blocked = self._no_stop_record_blocks_adopt()
        except Exception:
            self.log.exception("boot-resume: no-stop record check raised — flattening")
            return False, "no-stop record check raised"
        if blocked is not None:
            self.log.warning("boot-resume ARMED: FLATTEN %s %.4f @ %.2f — %s",
                             pos.side, pos.qty, pos.entry_price, blocked)
            try:
                send_alert("boot-resume: adopt refused, flattening",
                           f"{self.instance} leg booted holding a {pos.side} "
                           f"{float(pos.qty):.4f} position with {blocked}; "
                           f"flattening instead of resuming.")
            except Exception:
                self.log.exception("boot-resume: refusal alert failed")
            return False, blocked
        try:
            adopt, why = self._can_adopt(pos)
        except Exception:
            # A gate that raises must not take the position with it.
            self.log.exception("boot-resume: gate raised — flattening")
            return False, "gate raised"
        if adopt:
            adopt, why = self._prepare_adopt(pos, why)
        self.log.warning(
            "boot-resume ARMED: %s %s %.4f @ %.2f — %s",
            "ADOPT" if adopt else "FLATTEN",
            pos.side, pos.qty, pos.entry_price, why)
        return adopt, why

    def _observe_verdict(self, pos) -> tuple[bool, str]:
        """Observe mode: offline precheck only, ZERO exchange calls, always False."""
        try:
            qty_step = float(getattr(self.constraints, "qty_step", 0.0) or 0.0)
            why, _ab = adopt_precheck(self.params, pos,
                                      state.get_meta("active_bracket"), qty_step,
                                      state.latest_entry_coid_root())
        except Exception:
            self.log.exception("boot-resume: gate raised — flattening")
            return False, "gate raised"
        if why is not None:
            self.log.warning("boot-resume OBSERVE: WOULD FLATTEN %s %.4f @ %.2f — %s",
                             pos.side, pos.qty, pos.entry_price, why)
            return False, why
        why = ("precheck passed; order-book check SKIPPED in observe mode "
               "(no exchange calls) — run tools/boot_resume_probe.py for the "
               "full verdict")
        self.log.warning("boot-resume OBSERVE: PRECHECK OK %s %.4f @ %.2f — %s",
                         pos.side, pos.qty, pos.entry_price, why)
        return False, why

    def _alert_resumed(self, pos, root: str | None) -> None:
        """C11: one email per adopt. NEVER raises — the position is adopted."""
        try:
            send_alert(
                "Position RESUMED after restart",
                f"The {self.instance} leg restarted holding a "
                f"{pos.side.upper()} {float(pos.qty):.4f} {self.base_asset} "
                f"position @ {float(pos.entry_price):,.2f} and RESUMED it "
                f"instead of closing it.\n"
                f"signal_id: {root or '(untagged)'}\n"
                f"gate: {self._adopt_reason or '?'}\n"
                + ("reprotect cap is ALREADY SPENT for this position — if the "
                   "bracket goes missing it will NOT be re-placed.\n"
                   if self._adopt_reprotect_capped else "")
                + "Restart while holding a position (cause unknown to the bot, "
                "may be a crash or a manual restart). To close instead: touch data/HALT_"
                f"{self.instance} (the next tick flattens and exits).",
            )
        except Exception:
            self.log.exception("boot-resume: resume alert failed")

    def _alert_arming_interlock(self, pos) -> None:
        """Config says `observe_only: false` but the code is not ready to arm."""
        msg = ("boot_resume.observe_only is false, but ARMING_PREREQS_BUILT is "
               "False in bot.py (the arming prerequisites are built but arming "
               "has not been approved: it needs the probe evidence gate in "
               "docs/BOOT_RESUME_PHASE_B_PLAN.md §2). Arming REFUSED — treating as "
               "observe_only, so this position is FLATTENED as before.")
        self.log.error("boot-resume: ARMING INTERLOCK — %s", msg)
        try:
            send_alert(
                "boot-resume: arming interlock tripped",
                f"The {self.instance} leg booted holding a {pos.side.upper()} "
                f"{float(pos.qty):.4f} {self.base_asset} position @ "
                f"{float(pos.entry_price):,.2f}.\n{msg}\n"
                f"Set boot_resume.observe_only back to true in the config.",
            )
        except Exception:
            self.log.exception("boot-resume: interlock alert failed")

    def _prepare_adopt(self, pos, why: str) -> tuple[bool, str]:
        """Everything the adopt branch needs, computed BEFORE committing.

        boot()'s adopt branch runs after the verdict; if it raised there, the
        process would die and systemd would adopt again. So anything that can
        fail runs here, and any failure becomes a refusal → today's flatten.
        """
        try:
            seed = signal_bar_seed(state.get_meta("last_entry_bar_ts"),
                                   state.latest_entry_fill_ts(),
                                   int(self.bar_seconds))
        except Exception:
            self.log.exception("boot-resume: entry-dedup seed unreadable — flattening")
            return False, "entry-dedup bar unreadable"
        if seed is None:
            return False, ("no entry-dedup bar to seed (no last_entry_bar_ts "
                           "and no entry fill)")

        # C6 (B4): RE-READ the position as the last exchange call before
        # committing. boot() read `pos` once, then the gate spent up to three
        # book round-trips (plus C9 back-off) on it; a bracket that filled or a
        # manual close in that window must not be adopted from the stale read.
        # Exact identity on the same exchange fields: this is the SAME position
        # or it is not — the 2% band is for stash-vs-fill, not read-vs-read.
        try:
            now_pos = self.client.fetch_position(self.symbol)
        except Exception:
            self.log.exception("boot-resume: position re-read failed — flattening")
            return False, "position re-read failed"
        if (now_pos.side != pos.side
                or abs(float(now_pos.qty) - float(pos.qty)) > 1e-12
                or abs(float(now_pos.entry_price) - float(pos.entry_price)) > 1e-9):
            return False, (f"position changed between read and adopt: "
                           f"{pos.side} {float(pos.qty):.4f} @ {float(pos.entry_price):.2f}"
                           f" -> {now_pos.side} {float(now_pos.qty):.4f} @ "
                           f"{float(now_pos.entry_price):.2f}")

        # D5 adopt-loop guard — LAST, so a refusal above never spends budget.
        # The counter is persisted BEFORE returning True: a crash anywhere
        # after this point must still count toward the limit.
        root = self._adopt_root
        if not root:
            # _can_adopt only stashes a root it matched (C5). Reaching here
            # without one means the caller skipped the gate: refuse.
            return False, "no proven entry root for this position"
        try:
            allowed, gwhy, new_log = adopt_loop_guard(
                state.get_meta("boot_adopt_log"), root, time.time())
            if allowed:
                state.set_meta("boot_adopt_log", json.dumps(new_log))
        except Exception:
            self.log.exception("boot-resume: adopt-loop guard unreadable — flattening")
            return False, "adopt-loop guard unreadable"
        if not allowed:
            self.log.error("boot-resume: %s — flattening instead of adopting", gwhy)
            try:
                send_alert(
                    "boot-resume: adopt-loop guard tripped",
                    f"The {self.instance} leg restarted holding a "
                    f"{pos.side.upper()} {float(pos.qty):.4f} {self.base_asset} "
                    f"position @ {float(pos.entry_price):,.2f} and would have "
                    f"adopted it again.\n{gwhy}.\n"
                    f"It is being FLATTENED at market instead (today's "
                    f"pre-resume behaviour). Something is restarting this leg "
                    f"repeatedly while it holds this position — read the "
                    f"journal before the next entry.\n"
                    f"signal_id: {root or '(untagged)'}",
                )
            except Exception:
                self.log.exception("boot-resume: guard alert failed")
            return False, gwhy
        self._adopt_signal_ts_seed = seed
        self._adopt_reason = f"{why}; {gwhy}"
        return True, self._adopt_reason

    def _ledger_equity(self, equity: float | None) -> float | None:
        """Equity to PERSIST on a fill/event: None while it is a stale stand-in.

        A stale value written as equity_after would corrupt net-PnL differencing
        of consecutive equity_after (and show stale equity as current); NULL is
        skipped by every reader (monitor filters IS NOT NULL, dashboard shows a dash).
        """
        if self._equity_stale or equity is None:
            return None
        return float(equity)

    def _maybe_reprotect(self, equity: float) -> None:
        """Restore a missing SL/TP bracket while a position is still open.

        The bot places a reduce-only bracket at entry, but an EXTERNAL event can
        remove it without the bot knowing — a manual cancel on Binance, or a
        leverage change (Binance auto-cancels ALL open orders on a leverage
        change). _detect_bracket_exit only reacts to a bracket FILL (whether it
        leaves the book flat or is replaced inside one poll by a re-entry),
        never a cancel, so a cancelled bracket would leave the position silently
        unprotected until the time stop — which, measured 2026-08-11, fires 0
        times in 4.6 years. This guard notices the gap and, from the params
        stashed at entry, places ONLY the missing leg(s) under this position's exact
        coid (so it never duplicates and never cancels a healthy stop), then
        alerts. A stop it cannot place — -2021, or max_sl_place_failures (3)
        failures, or a stop still missing at the cap — closes the position at
        market (God's rule, 2026-10-13). Legs are matched by EXACT root +
        closing side (adopt_bracket_state); our reduce-only legs from an older
        root are cancelled by id.

        RE-ENABLED 2026-08-11 with algo-aware detection. It was disabled
        2026-07-22 because `bracket_is_intact(fetch_open_orders(...))` cannot see
        the conditional/algo orders `_place_brackets` actually creates, so it
        read a healthy bracket as missing and re-placed every cooldown until
        `-4045 Reach max stop order limit`. Detection now merges BOTH order books
        via `bracket_state`, and `algo_bracket_leg` classifies the algo shape by
        client-order-id suffix because those rows carry no type field at all
        (see its docstring for the live payload).

        Three independent brakes, because the failure mode is a detector that is
        wrong in the unsafe direction:
          - `reprotect.observe_only` — log the decision, place nothing. The
            rollout runs here until a full live position passes with zero
            "would re-place" lines.
          - `reprotect.max_replaces_per_position` — hard cap per signal_id,
            persisted in active_bracket so a restart cannot reset it. The 60s
            throttle did NOT stop July; it only paced the spam.
          - an empty algo read is treated as UNKNOWN, not as "no bracket" — a
            transient endpoint failure looks identical to a cancelled bracket,
            and acting on it is precisely the July bug.
        """
        if self.dry_run:
            return
        rp = (self.params.get("reprotect") or {})
        if not rp.get("enabled", False):
            return
        # A failed no-stop close owns the position: _maybe_finish_no_stop_close
        # retries it with backoff. Reprotect running too would double the
        # close attempts and the alerts. Local read only.
        if self._pending_record() is not None:
            return
        pos = self.client.fetch_position(self.symbol)
        if pos.side == "flat" or pos.qty == 0:
            # No position → drop any stale bracket record so it can never be
            # applied to a later, unrelated position (the side + entry-price
            # guards below are the second line of defence).
            state.set_meta("active_bracket", "")
            return
        raw = state.get_meta("active_bracket")
        if not raw:
            return
        try:
            ab = json.loads(raw)
        except (ValueError, TypeError):
            return
        # Only act on stashed params that clearly belong to THIS position —
        # matching side + entry price (2%) — so a stale record never fabricates
        # a bracket at the wrong levels for a different/manual position.
        if ab.get("side") != pos.side:
            return
        ep = float(ab.get("entry_price") or 0.0)
        if ep <= 0 or pos.entry_price <= 0 or abs(ep - pos.entry_price) / pos.entry_price > 0.02:
            return
        place_tp = bool(ab.get("place_tp", True))
        root = ab.get("signal_id")
        if not root:
            self.log.warning("reprotect: active_bracket has no signal_id — cannot "
                             "identify this position's own legs; skipping")
            return
        root = str(root)
        try:
            open_orders = self.client.ex.fetch_open_orders(self.symbol)
        except Exception:
            self.log.exception("reprotect: fetch_open_orders failed")
            return
        # Read BOTH books. fetch_algo_orders never raises, so it reports success
        # separately: `ok=False` means missing endpoint, failed call, or a bad
        # payload. Acting on a read we could not make is the July bug wearing a
        # different hat — a 5-second network blip would re-place a live bracket.
        algo_rows, algo_ok = self.client.fetch_algo_orders(self.symbol)
        if not algo_ok:
            self.log.warning("reprotect: algo book unreadable — skipping "
                             "(cannot distinguish 'no bracket' from 'no answer')")
            return
        # EXACT root + closing side (adopt_bracket_state), not prefix-only: an
        # OLDER trade's leftover `-s` used to read as "our SL present", so the
        # new position got no stop. Untagged / foreign / other-root legs now
        # read as missing ⇒ our own stop is ADDED (more protection, still
        # capped). Its place_tp is hard-coded True, so rebuild with ours.
        own = adopt_bracket_state(open_orders, algo_rows, self.coid_prefix,
                                  root, pos.side)
        state_ = BracketState(sl=own.sl, tp=own.tp, place_tp=place_tp)
        # Reduce-only legs of OUR prefix from a DIFFERENT root can close this
        # position at a stale price. Computed from the rows already read, so a
        # healthy tick makes no extra call.
        stale = stale_root_legs(open_orders, algo_rows, self.coid_prefix, root)
        if state_.intact and not stale:
            return
        now = time.time()
        if now - self._last_reprotect_ts < 60:
            return
        observe = bool(rp.get("observe_only", True))

        # Hard caps per position, persisted so a restart cannot reset them. A
        # detector bug must not be able to walk to -4045 again. SEPARATE for
        # the two legs: with one shared counter, three TP fixes used up the
        # budget and a later missing STOP was closed with zero re-place
        # attempts — not the "retry 2–3×" God asked for.
        cap = int(rp.get("max_replaces_per_position", 3))
        done = int(ab.get("reprotect_count", 0))          # stop re-places
        tp_done = int(ab.get("tp_reprotect_count", 0))    # TP re-place attempts
        need_sl = not state_.sl
        need_tp = place_tp and not state_.tp
        sl_d = float(ab["sl_distance"])
        tp_d = float(ab["tp_distance"])
        sign = 1.0 if pos.side == "long" else -1.0
        sl_price = pos.entry_price - sign * sl_d
        if done >= cap and need_sl and not observe:
            # The stop keeps disappearing and we may not re-place it again.
            # God's rule (2026-10-13): a position that cannot keep a stop is
            # closed, not left naked behind a once-only alert.
            self._last_reprotect_ts = now
            self._emergency_close(
                pos, root, equity, stop_price=sl_price,
                why=f"stop missing again after {done} re-place(s) (cap {cap})")
            return
        tp_capped = need_tp and tp_done >= cap
        if ((need_sl and done >= cap) or (tp_capped and not need_sl)) and not stale:
            if not self._reprotect_capped_alerted:
                self._reprotect_capped_alerted = True
                self.log.error("reprotect: hit cap %d for signal_id=%s — STOPPING. %s",
                               cap, root, state_.describe())
                send_alert(
                    "Bracket re-place CAP reached — position may be unprotected",
                    f"{pos.side} {pos.qty:.4f} @ {pos.entry_price:.2f}\n"
                    f"{state_.describe()}\n"
                    f"Stop re-placed {done}×, TP {tp_done}×; cap is {cap} each. No "
                    f"further attempts will be made for this leg — check the "
                    f"bracket manually.")
            return

        self._last_reprotect_ts = now  # throttle (relaxed to a 15s retry on failure below)

        if observe:
            # Rollout phase 1: prove the detector against a real resting bracket
            # without placing anything. Any line here during a healthy position
            # is the July false-negative reappearing.
            self.log.warning(
                "reprotect OBSERVE: WOULD re-place for %s %.4f @ %.2f — %s "
                "stale=%s (plain=%d algo=%d)",
                pos.side, pos.qty, pos.entry_price, state_.describe(),
                [c for _, _, c in stale],
                len(open_orders or []), len(algo_rows or []))
            return

        # Stale legs are cancelled only while THIS root's stop rests (now, or
        # once it has been placed below). Cancelling an old-root stop before
        # the new one is resting would remove the only stop the position has —
        # the same no-stop window as cancel-then-replace.
        if stale and state_.sl:
            self._sweep_stale_legs(stale, root)
        if state_.intact:
            return
        if tp_capped and not need_sl:
            return    # only the TP is missing and its cap is spent (alerted above)

        # Place ONLY the missing leg(s). The old cancel-everything-then-replace
        # cancelled a HEALTHY stop to fix a missing TP, leaving a window with
        # no stop at all. Legs go out tagged-only with this root's exact id,
        # so nothing placed here can duplicate a resting leg of ours.
        self.log.warning(
            "Bracket MISSING while holding %s %.4f @ %.2f (equity=$%.2f) — %s; "
            "placing the missing leg(s).",
            pos.side, pos.qty, pos.entry_price, equity, state_.describe())
        if need_sl:
            try:
                self.client.place_tagged_stop(
                    self.symbol, pos.side, pos.qty, sl_price, root, "s")
            except Exception as e:
                self.log.exception("reprotect: stop re-place failed: %s", e)
                if is_nonce_error(e):
                    self.client.resync_time()
                # Count in memory too: if sqlite is unwritable the cap must
                # still hold, and bookkeeping must never skip the close below.
                fails = max(int(ab.get("sl_place_failures", 0)),
                            self._sl_fail_mem.get(root, 0)) + 1
                self._sl_fail_mem[root] = fails
                max_fails = int(rp.get("max_sl_place_failures", 3))
                ab["sl_place_failures"] = fails
                self._set_meta_quietly("active_bracket", json.dumps(ab))
                if is_would_trigger(e) or fails >= max_fails:
                    # -2021: price is already through the stop — retrying cannot
                    # help. Otherwise: retried enough. Close (God's rule).
                    self._emergency_close(
                        pos, root, equity, stop_price=sl_price,
                        why=(f"stop re-place rejected: price already through the "
                             f"stop ({e})" if is_would_trigger(e) else
                             f"stop re-place failed {fails}× ({e})"))
                    return
                # An unprotected live position is urgent — retry ~15s (not the
                # 60s success cooldown, and not every 5s poll).
                self._last_reprotect_ts = now - 45
                send_alert(
                    "Bracket re-place FAILED",
                    f"{pos.side} {pos.qty:.4f} @ {pos.entry_price:.2f} — the stop-loss "
                    f"@ {sl_price:,.4f} could not be re-placed (attempt {fails}/{max_fails}; the "
                    f"position is closed at market after {max_fails}): {e}")
                return
            ab["sl_place_failures"] = 0
            self._sl_fail_mem.pop(root, None)
            ab["reprotect_count"] = done + 1
            if stale:
                self._sweep_stale_legs(stale, root)
        tp_err: Exception | None = None
        if need_tp and not tp_capped:
            ab["tp_reprotect_count"] = tp_done + 1
            try:
                self.client.place_tagged_leg(
                    self.symbol, pos.side, pos.qty, pos.entry_price + sign * tp_d,
                    root, "t")
            except Exception as e:
                self.log.exception("reprotect: TP re-place failed: %s", e)
                tp_err = e
                self._last_reprotect_ts = now - 45
        # Persist the counts INSIDE active_bracket so the caps survive a
        # restart — an in-memory counter would reset on every boot and let a
        # detector bug resume walking toward -4045.
        self._set_meta_quietly("active_bracket", json.dumps(ab))
        state.record_event("WARN", "bracket_reprotect",
                           {"side": pos.side, "qty": pos.qty,
                            "entry": pos.entry_price, "equity": equity,
                            "attempt": int(ab.get("reprotect_count", 0)),
                            "tp_attempt": int(ab.get("tp_reprotect_count", 0)),
                            "cap": cap,
                            "was": state_.describe(),
                            "tp_error": str(tp_err) if tp_err else None,
                            "signal_id": root},
                           signal_id=root)
        if tp_err is not None:
            send_alert(
                "Bracket re-place FAILED (TP only — stop is in place)",
                f"{pos.side} {pos.qty:.4f} @ {pos.entry_price:.2f}: the take-profit "
                f"could not be re-placed ({tp_err}). The stop-loss is resting.")
            return
        send_alert(
            "Bracket re-placed",
            f"SL/TP for {pos.side} {pos.qty:.4f} @ {pos.entry_price:.2f} went missing "
            f"(external cancel?) and was automatically restored ({state_.describe()} "
            f"before).")

    def _sweep_stale_legs(self, stale: list[tuple[str, str, str]], root: str) -> None:
        """Cancel OUR reduce-only legs left by an earlier root — by id, one at
        a time. Never cancel_open_orders(prefix): that would take the CURRENT
        root's legs with them."""
        done: list[str] = []
        for book, oid, coid in stale:
            try:
                if book == "algo":
                    ok = self.client.cancel_algo_by_coid(self.symbol, coid)
                else:
                    self.client.ex.cancel_order(oid, self.symbol)
                    ok = True
            except Exception as e:
                self.log.warning("reprotect: cancel stale leg %s failed: %s", coid, e)
                ok = False
            if ok:
                done.append(coid)
        self.log.warning("reprotect: cancelled %d/%d stale leg(s) from an earlier "
                         "trade (current root %s): %s", len(done), len(stale), root, done)
        state.record_event("WARN", "stale_legs_swept",
                           {"root": root, "cancelled": done,
                            "found": [c for _, _, c in stale]}, signal_id=root)
        send_alert("Stale bracket legs cancelled",
                   f"{len(done)}/{len(stale)} reduce-only order(s) from an earlier "
                   f"trade were still resting and could have closed the current "
                   f"position at a stale price: {done}")

    # ~5 min at a 5 s poll. Bounded on purpose: fetch_my_trades only returns the
    # last 10 trades, so a retry that ran for hours would eventually be matching
    # against a window that no longer contains the close at all.
    EXIT_RETRY_LIMIT = 60

    def _hold_unrecorded_exit(self, side: str, entry: float | None,
                              qty: float | None, root: str | None) -> None:
        """Keep a DETECTED but UNRECORDED exit alive for a retry — either path.

        If the exit is detected but cannot be written (fetch_my_trades down, the
        closing trade not yet visible on the account, the `fills` read raising),
        advancing the snapshot past that failure loses the exit *permanently*,
        which is precisely the bug this detector was written to end. Putting the
        OLD identity back makes the next tick re-detect the same edge and try
        again.

        Why the snapshot and not the ledger, on each path:
          - REPLACED: the snapshot is the ONLY record the closed position ever
            existed, because `fills` already names the replacement entry.
          - FLAT EDGE: `fills` still names the closed position, so the restored
            snapshot only has to be non-flat — it re-creates the open→flat edge,
            and the retry re-reads `fills` for pricing.

        Known limitation, left deliberately: if the REPLACEMENT position also
        closes while a retry is outstanding, the next tick takes the flat path
        and prices the exit off the replacement's `fills` row. That mis-prices a
        double failure rather than losing it, which is the better of the two.
        """
        self._exit_retry_n += 1
        if self._exit_retry_n > self.EXIT_RETRY_LIMIT:
            attempts = self._exit_retry_n - 1
            self._exit_retry_n = 0
            self.log.error(
                "bracket-exit: GAVE UP recording the exit of %s %.4f @ %.2f "
                "(root=%s) after %d attempts — the ledger is missing this close",
                side, qty or 0.0, entry or 0.0, root, attempts)
            send_alert(
                "Bot exit NOT recorded",
                f"A {side.upper()} {qty or 0.0:.4f} {self.base_asset} position "
                f"closed and its exit could not be recorded after {attempts} "
                f"attempts.\n"
                f"Entry: {entry or 0.0:,.2f}  signal_id: {root or '(untagged)'}\n"
                f"The ledger is missing this close — it needs a manual backfill.",
            )
            return
        self._last_position_side = side
        self._last_position_entry = entry
        self._last_position_qty = qty
        self._last_entry_root = root

    def _detect_bracket_exit(self, equity: float) -> None:
        """Bracket SL/TP fills close the position on Binance's side without
        a bot-initiated close. Detect that the tracked position has ENDED and
        emit an exit alert with PnL.

        Two ways it can end, and they are priced from different sources:

          - FLAT EDGE (open→flat). Sweeps the surviving bracket sibling, then
            prices the exit from the latest `fills` row, which must be
            `reason='entry'` — i.e. no bot-initiated close (time_stop, kill,
            halt) has been recorded since the entry.

          - REPLACED (open→open, different position). The bracket filled and
            _maybe_enter re-entered inside one poll, so the loop never sees
            `flat`. Detected by a change in `entry_price`, priced from the
            in-memory snapshot, and it must NOT sweep — see the branch comment.
            Before this existed the exit was dropped entirely: no `close` row,
            no `exit` event, no alert (live 2026-08-23).

        Also required either way: not dry-run (no real brackets in dry-run
        mode), and a matching opposite-side trade on Binance.

        BOTH paths are retried. Once either has decided the position ended, the
        exit is owed, and every failure before it is written — a raising sqlite
        read, an unmatched or empty trade list, a zero fill price, any ccxt
        error — re-arms `retain` so the next tick re-detects instead of losing
        the close. A branch that decides no exit is owed (a bot-initiated close
        already in the ledger) disarms explicitly. Bounded by EXIT_RETRY_LIMIT
        ticks, then it alerts rather than giving up silently.
        """
        if self.dry_run:
            return
        # Set once an exit is DETECTED on EITHER path — replacement since #26,
        # flat edge since the 2026-09-04 donchian loss — and cleared the moment
        # it is written, or when a branch decides no exit is owed at all.
        # Anything still holding it at `finally` is an exit we found and failed
        # to record — see _hold_unrecorded_exit.
        retain: tuple[str, float | None, float | None, str | None] | None = None
        try:
            pos = self.client.fetch_position(self.symbol)
            current = pos.side

            prev_side = self._last_position_side
            prev_entry = self._last_position_entry
            prev_qty = self._last_position_qty
            prev_root = self._last_entry_root
            had_open = prev_side not in ("flat", "unknown")

            # A position we were tracking can disappear two ways:
            #   flat      — the classic open→flat edge.
            #   REPLACED  — still open, but it is not the same position: the
            #               bracket filled and _maybe_enter re-entered inside a
            #               single poll (live 2026-08-23, 1.25 s apart).
            # entry_price is the discriminator, compared at float-noise
            # tolerance: Binance echoes the identical number for an untouched
            # position, while a re-entry seconds later can be only a few dollars
            # away — so a percentage threshold would miss exactly the case this
            # branch exists to catch. Both prices must be non-zero, or a missing
            # `entryPrice` field would fabricate an exit out of nothing.
            replaced = (
                had_open
                and current != "flat"
                and prev_entry is not None
                and prev_entry > 0.0
                and pos.entry_price > 0.0
                and (current != prev_side
                     or abs(pos.entry_price - prev_entry)
                     > 1e-9 * max(prev_entry, pos.entry_price))
            )

            # A same-side SIZE INCREASE made outside the bot (manual add) moves
            # the average entry_price and so looks exactly like a replacement.
            # A real bot re-entry always writes a NEW entry row, so an
            # unchanged latest entry root says nobody re-entered: not an exit.
            if (replaced and prev_root is not None and current == prev_side
                    and pos.qty > float(prev_qty or 0.0)
                    and state.latest_entry_coid_root() == prev_root):
                replaced = False

            # Refresh the snapshot BEFORE any early return, so every path leaves
            # it agreeing with what the exchange just said.
            self._last_position_side = current
            if current == "flat" or replaced or not had_open:
                # A reduction still in the debounce / pricing window when the
                # position goes flat is unrecorded: keep its size for the flat
                # edge (which may need retries) before dropping the state.
                if current == "flat" and had_open and self._ext_pending is not None:
                    self._flat_pending_q = self._ext_pending
                # a pending size change belongs to the position that was open
                self._ext_pending, self._ext_attempts = None, 0
            if current == "flat":
                self._last_position_entry = None
                self._last_position_qty = None
                self._last_entry_root = None
            else:
                self._last_position_entry = pos.entry_price
                self._last_position_qty = pos.qty
                if replaced or not had_open:
                    # Only correct at the moment the position opens; one tick
                    # later this already names the replacement entry.
                    self._last_entry_root = state.latest_entry_coid_root()

            # Same position, different SIZE: somebody other than the bot changed
            # it. Record-keeping only — nothing is placed, cancelled or resized.
            if (had_open and current == prev_side and not replaced
                    and prev_qty and pos.qty > 0.0):
                self._note_external_size_change(
                    pos, float(prev_qty), prev_entry, prev_root, equity)

            # First observation since boot — initialise state without emitting.
            # A historical "entry" row in state.db whose position already closed
            # on the exchange must not trigger an alert.
            if prev_side == "unknown" or not had_open:
                return
            if current != "flat" and not replaced:
                return

            if replaced:
                # ⚠️ NO SWEEP ON THIS PATH, AND DO NOT "UNIFY" IT WITH THE ONE
                # BELOW. _maybe_enter → _place_live_entry has already placed a
                # fresh bracket for the position that is open RIGHT NOW, and
                # cancel_open_orders is scoped to the whole leg by coid prefix —
                # calling it here would strip the SL and TP off a live
                # real-money position, which is strictly worse than the dropped
                # exit this branch fixes.
                #
                # The fills table cannot price this exit either: _maybe_enter
                # runs last in the tick, so its `entry` row is already the most
                # recent one by the time we get here. The in-memory snapshot is
                # the only record of the position that just closed.
                entry_side = prev_side
                entry_qty = float(prev_qty or 0.0)
                entry_price = float(prev_entry)
                signal_id = prev_root
                if entry_qty <= 0.0:
                    self.log.warning(
                        "bracket-exit: replacement %s→%s but no tracked qty; "
                        "cannot price the exit", prev_side, current)
                    return
                # ⚠️ A BOT-INITIATED close can share a tick with a re-entry too,
                # and without this guard that would be written TWICE. The tick
                # order is _detect_bracket_exit → _maybe_time_stop →
                # _maybe_channel_exit → _maybe_enter, so on one tick the channel
                # or trend exit can record its own close and _maybe_enter can
                # then open at a new price (it never advanced _last_signal_ts
                # while the position was open, so the bar guard lets it through).
                # The next tick sees a changed entry_price and would emit a
                # SECOND exit for a position already closed in the ledger —
                # corrupting it in the opposite direction from the dropped exit
                # this branch fixes. The classic path gets this for free from its
                # `reason='entry'` check on the newest row; the replacement path
                # has to ask explicitly, because the newest row is the re-entry.
                with sqlite3.connect(state.DB_PATH) as c:
                    anchor = c.execute(
                        "SELECT id FROM fills WHERE reason = 'entry' "
                        "AND client_order_id_root IS ? ORDER BY id DESC LIMIT 1",
                        (signal_id,),
                    ).fetchone()
                    if anchor:
                        already_closed = c.execute(
                            "SELECT 1 FROM fills WHERE id > ? AND reason NOT IN ('entry', "
                            "'external_partial', 'external_add', 'bot_partial') "
                            "LIMIT 1", (anchor[0],),
                        ).fetchone() is not None
                    else:
                        # Untagged entry, so there is nothing to anchor on. Fall
                        # back to the row just BEFORE the newest one (the newest
                        # being the re-entry): if that is a close, it is ours.
                        prior = c.execute(
                            "SELECT reason FROM fills WHERE id < "
                            "(SELECT MAX(id) FROM fills) AND reason NOT IN "
                            + _SIZE_ROWS_SQL + " ORDER BY id DESC LIMIT 1"
                        ).fetchone()
                        already_closed = bool(prior) and prior[0] != "entry"
                if already_closed:
                    self.log.info(
                        "bracket-exit: replacement %s→%s, but this position "
                        "already has a recorded close — not double-writing",
                        prev_side, current)
                    return
                retain = (prev_side, prev_entry, prev_qty, prev_root)
            else:
                # Arm the retry BEFORE anything that can fail. The flat edge is
                # already confirmed, so the position HAS ended and an exit is
                # owed; from here every early exit — a raising sqlite read, the
                # trade lookup below, a zero fill price — is a failure to record
                # something real. The snapshot advanced to flat at the top of
                # this method, so returning without retaining loses it for good.
                # The replacement path has been retried since #26; the flat edge
                # — the older and far more common one — never was.
                # Live 2026-09-04: donchian's algo SL filled at 12:31:49 and
                # fetch_my_trades still did not list the closing trade at
                # 12:32:04, so the single lookup missed, this branch returned,
                # and the ledger lost a losing close (no fills row, no exit
                # event, no alert) until it was backfilled by hand.
                #
                # prev_* (the exchange snapshot of the position that just
                # closed) rather than the `fills` row, precisely so the arming
                # happens before the read that can raise. It is also all the
                # next tick needs: restoring a non-flat side re-creates this
                # same edge, and the retry re-reads `fills` for pricing.
                retain = (prev_side, prev_entry, prev_qty, prev_root)

                # Sweep the surviving bracket leg (the sibling that Binance did NOT
                # fill).  Do this before the PnL lookup so even if the fill-lookup
                # errors out, the orphan is still cancelled.
                try:
                    self.client.cancel_open_orders(self.symbol, coid_prefix=self.coid_prefix)
                except Exception:
                    self.log.exception("bracket-exit: cancel_open_orders failed")

                with sqlite3.connect(state.DB_PATH) as c:
                    # external_partial / external_add / bot_partial rows are
                    # bookkeeping for a size change that is not an exit: look past them to the entry.
                    row = c.execute(
                        "SELECT reason, side, qty, price, client_order_id_root, id "
                        "FROM fills WHERE reason NOT IN "
                        + _SIZE_ROWS_SQL + " ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                    ext_rows = []
                    if row and row[0] == "entry":
                        ext_rows = c.execute(
                            "SELECT reason, qty, price FROM fills WHERE id > ? "
                            "AND reason IN " + _SIZE_ROWS_SQL + " "
                            "ORDER BY id", (row[5],)).fetchall()
                if not row or row[0] != "entry":
                    # DELIBERATE skip, not a failure: a bot-initiated close
                    # (time_stop, kill, halt) is already in the ledger, so no
                    # exit is owed. Disarm, or this holds the snapshot open for
                    # EXIT_RETRY_LIMIT ticks over nothing and then alerts.
                    retain = None
                    return
                _, entry_side, entry_qty, entry_price, signal_id, _id = row
                # Price only what was still open, at the running cost basis (see
                # the replay below). 09-24: 0.66 x TP gave
                # 12.78; the honest figure prices 0.33 + 0.33 = 9.75.
                entry_qty = float(entry_qty)
                entry_price = float(entry_price)
                if ext_rows:
                    qty_run, entry_price = _replay_size_rows(
                        entry_qty, entry_price, ext_rows)
                    entry_qty = qty_run if qty_run > 0 else float(prev_qty or 0.0)
                entry_qty = self._record_pending_reduction(
                    entry_side, entry_qty, entry_price, signal_id, equity)

            opposite = "sell" if entry_side == "long" else "buy"
            trades = self.client.ex.fetch_my_trades(self.symbol, limit=10)
            if replaced and current != prev_side:
                # A flip closes a long with a SELL and opens the short with a
                # SELL too, so the newest opposite-side trade may be the new
                # ENTRY rather than our close. Both land inside one poll, so any
                # mispricing is bounded by a second or two of movement — worth
                # logging loudly and recording, not worth guessing at.
                self.log.warning(
                    "bracket-exit: side-flip replacement %s→%s; exit price may "
                    "name the new entry (prev %.4f @ %.2f, now %.4f @ %.2f)",
                    prev_side, current, entry_qty, entry_price,
                    pos.qty, pos.entry_price)
            exit_trade = next(
                (t for t in reversed(trades) if t.get("side") == opposite),
                None,
            )
            if exit_trade is None:
                self.log.warning(
                    "bracket-exit: no opposite-side trade found for signal_id=%s",
                    signal_id,
                )
                return
            exit_price = float(exit_trade.get("price") or 0.0)
            if exit_price <= 0:
                return

            if entry_side == "long":
                pnl = (exit_price - entry_price) * entry_qty
            else:
                pnl = (entry_price - exit_price) * entry_qty
            notional = entry_price * entry_qty
            pnl_pct = (pnl / notional) * 100.0 if notional > 0 else 0.0

            state.record_fill(
                side="close", qty=entry_qty, price=exit_price,
                pnl_usd=pnl, reason="bracket_exit",
                equity_after=self._ledger_equity(equity), client_order_id_root=signal_id,
            )
            # Durable now — stop holding the old snapshot open for a retry.
            retain = None
            self._exit_retry_n = 0
            state.enqueue_bot_event(
                "exit", signal_id=signal_id, side=entry_side,
                qty=float(entry_qty), price_usd=float(exit_price),
                equity_usd=self._ledger_equity(equity),
                payload={"reason": "bracket_exit",
                         "detected_by": "replacement" if replaced else "flat_edge",
                         "entry_price": float(entry_price),
                         "exit_price": float(exit_price),
                         "pnl_usd": float(pnl),
                         "pnl_pct": float(pnl_pct)},
            )
            send_alert(
                f"Bot {entry_side.upper()} exit",
                f"{entry_side.upper()} {entry_qty:.4f} {self.base_asset} closed "
                f"by bracket (SL or TP)\n"
                f"Entry: {entry_price:,.2f}  Exit: {exit_price:,.2f}\n"
                f"PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
                f"Equity now: ${equity:,.2f}\n"
                f"signal_id: {signal_id or '(untagged)'}",
            )
        except Exception as e:
            self.log.warning("bracket-exit detection failed: %s", e)
        finally:
            if retain is not None:
                self._hold_unrecorded_exit(*retain)
            else:
                self._flat_pending_q = None

    def _record_pending_reduction(self, side: str, open_qty: float,
                                  basis: float, signal_id: str | None,
                                  equity: float) -> float:
        """Flat edge only. If a size change was still pending (debounce /
        pricing retries) when the exit arrived, `open_qty` still includes the
        unrecorded reduction and would be priced at the exit fill. Price it from
        the exchange's own trades (oldest first, after the last ledger row),
        write it as external_partial and return the qty the exit really closed.
        If the trades cannot be read or do not add up, keep `open_qty` (never
        block the exit) and alert that the ledger may be off.
        """
        pend, self._flat_pending_q = self._flat_pending_q, None
        step = float(getattr(self.constraints, "qty_step", 0.0) or 0.0)
        tol = max(step * 0.5, 1e-9)
        delta = round(open_qty - pend, 8) if pend is not None else 0.0
        if delta <= tol:
            return open_qty
        price = 0.0
        try:
            with sqlite3.connect(state.DB_PATH) as c:
                ts_row = c.execute(
                    "SELECT MAX(ts) FROM fills WHERE id >= (SELECT COALESCE("
                    "MAX(id), 0) FROM fills WHERE reason = 'entry')"
                ).fetchone()
            since = datetime.fromisoformat(ts_row[0]) if ts_row and ts_row[0] else None
            if since is not None and since.tzinfo is None:
                since = since.replace(tzinfo=UTC)
            since_ms = since.timestamp() * 1000.0 if since else 0.0
            opposite = "sell" if side == "long" else "buy"
            got, picked = 0.0, []
            for t in self.client.ex.fetch_my_trades(self.symbol, limit=10):
                if t.get("side") != opposite or float(t.get("timestamp") or 0.0) <= since_ms:
                    continue
                picked.append(t)
                got += float(t.get("amount") or 0.0)
                if got >= delta - tol:
                    break
            if picked and abs(got - delta) <= tol:
                price = sum(float(t["amount"]) * float(t.get("price") or 0.0)
                            for t in picked) / got
        except Exception as e:
            self.log.warning("flat edge: pending reduction not priced (%s)", e)
        if price <= 0:
            self.log.warning("flat edge: reduction %.4f was pending and could "
                             "not be matched to trades; pricing the full qty",
                             delta)
            try:
                send_alert("Ledger may be off: unrecorded partial close",
                           f"{self.base_asset}: {delta:.4f} was reduced before "
                           f"the exit but its trade could not be read; the exit "
                           f"is priced on the full {open_qty:.4f}.\n"
                           f"signal_id: {signal_id or '(untagged)'}")
            except Exception:
                self.log.exception("flat edge: alert failed")
            return open_qty
        state.record_fill(
            side="partial_close", qty=delta, price=price,
            pnl_usd=_exit_pnl_usd(side, basis, price, delta),
            reason="external_partial", equity_after=equity,
            client_order_id_root=signal_id)
        self.log.warning("flat edge: recorded pending reduction %.4f @ %.4f",
                         delta, price)
        return round(open_qty - delta, 8)

    EXT_PRICE_ATTEMPTS = 3

    def _price_external_reduction(self, pos, delta: float, tol: float,
                                  since_ms: float, final: bool):
        """(price, source, ours) for a confirmed reduction of `delta`.

        `ours` is True when every matched trade belongs to an order whose
        clientOrderId starts with this leg's coid prefix (our TP/SL algo fills,
        our market closes, partial fills of our own orders). userTrades rows
        carry no clientOrderId, so it is read from the row if present, else one
        fetch_order per distinct order id. RAISES if that lookup fails, so the
        caller retries (bounded) rather than guessing.

        Only trades newer than `since_ms` (the current entry / last recorded
        external row) are considered, so an older equal-qty trade of ours
        (a previous position's entry) can never be mistaken for this one.
        If none sums to `delta` the fill is "not visible yet" (userTrades lag):
        RAISES to retry, and only on the `final` attempt falls back to the
        ticker, flagged, as external. Never adopts silently on a guess.

        KNOWN RESIDUAL: an order placed via the untagged fallback in
        binance_client._create_order_with_coid_retry carries no coid and would
        be classed external. Triggered algo children DO carry our coid (live
        sol-st-...-s SL fills, 2026-09-03 / 09-08).
        """
        opposite = "sell" if pos.side == "long" else "buy"
        got, picked = 0.0, []
        for t in reversed(self.client.ex.fetch_my_trades(self.symbol, limit=10)):
            if t.get("side") != opposite:
                continue
            if float(t.get("timestamp") or 0.0) <= since_ms:
                continue
            picked.append(t)
            got += float(t.get("amount") or 0.0)
            if got >= delta - tol:
                break
        prices = [float(t.get("price") or 0.0) for t in picked]
        if picked and abs(got - delta) <= tol and all(p > 0 for p in prices):
            price = sum(float(t["amount"]) * p for t, p in zip(picked, prices)) / got
            ours = True
            for oid in {t.get("order") for t in picked}:
                t0 = next(t for t in picked if t.get("order") == oid)
                coid = (t0.get("clientOrderId")
                        or (t0.get("info") or {}).get("clientOrderId"))
                if not coid and oid:
                    coid = (self.client.ex.fetch_order(oid, self.symbol)
                            or {}).get("clientOrderId")
                if not str(coid or "").startswith(self.coid_prefix):
                    ours = False
            return price, "userTrades", ours
        if not final:
            raise RuntimeError("reduction not visible in userTrades yet")
        last = float(self.client.ex.fetch_ticker(self.symbol).get("last") or 0.0)
        if last <= 0:
            raise RuntimeError("no price for the external reduction")
        return last, "ticker (flagged: userTrades did not match)", False

    def _note_external_size_change(self, pos, prev_qty: float,
                                   prev_entry: float | None,
                                   prev_root: str | None, equity: float) -> None:
        """RECORD-KEEPING ONLY: the position's size moved and the bot did not
        do it (manual / other-script reduce-only close, or an add). Nothing is
        placed, cancelled or resized here — brackets stay as they are
        (reduceOnly caps them).

        Detected from the position `_detect_bracket_exit` already reads every
        tick (zero extra calls in steady state), DEBOUNCED: the same new qty
        must be seen on 2 consecutive ticks before anything is written, so a
        stale one-tick read cannot write a row. Until then (and while a write
        is being retried) the tracked qty is rewound to the old baseline.

        DECREASE: priced by _price_external_reduction (one fetch_my_trades, plus
        fetch_order per order id; ticker fallback). Trades from our own orders
        are not external: the qty is adopted silently. Pricing is attempted at
        most EXT_PRICE_ATTEMPTS times per change; then the row is written
        unpriced (price 0, pnl NULL) with a single alert. Written as
        side='partial_close' / reason='external_partial' (not 'close': monitor's
        loss streak counts those).
        INCREASE: alert, and a side='partial_add' / reason='external_add' row
        is recorded so the exit paths' qty and cost basis (replayed by
        _replay_size_rows) stay right.
        A partial fill of one of OUR orders is recorded the same way as
        reason='bot_partial' (no alert) so a later flat edge prices only what
        was still open.
        Readers that look for "newest fill is the entry" skip all three reasons.

        KNOWN RESIDUAL: a manual reduce made while the bot is DOWN is not
        captured; tracking is seeded from the exchange at boot.
        """
        try:
            step = float(getattr(self.constraints, "qty_step", 0.0) or 0.0)
            tol = max(step * 0.5, 1e-9)
            q = float(pos.qty)
            diff = q - prev_qty
            if abs(diff) <= tol:
                self._ext_pending, self._ext_attempts = None, 0
                return
            self._last_position_qty = prev_qty  # hold the baseline until settled
            self._last_position_entry = prev_entry
            pend = self._ext_pending
            if pend is None or abs(pend - q) > tol:
                self._ext_pending, self._ext_attempts = q, 0
                return
            price, src, pnl, kind = 0.0, "", None, "decrease"
            if diff > 0:
                kind = "increase"
                # Execution price of the add, backed out of the exchange's
                # new average: the exit paths need the weighted cost basis.
                add_px = float(pos.entry_price)
                if prev_entry and prev_entry > 0:
                    px = (float(pos.entry_price) * q - prev_entry * prev_qty) / diff
                    add_px = px if px > 0 else add_px
                state.record_fill(
                    side="partial_add", qty=diff,
                    price=add_px, pnl_usd=None,
                    reason="external_add", equity_after=equity,
                    client_order_id_root=prev_root)
            else:
                delta = -diff
                self._ext_attempts += 1
                with sqlite3.connect(state.DB_PATH) as c:
                    ts_row = c.execute(
                        "SELECT MAX(ts) FROM fills WHERE id >= (SELECT COALESCE("
                        "MAX(id), 0) FROM fills WHERE reason = 'entry')"
                    ).fetchone()
                since = datetime.fromisoformat(ts_row[0]) if ts_row and ts_row[0] else None
                if since is not None and since.tzinfo is None:
                    since = since.replace(tzinfo=UTC)
                since_ms = since.timestamp() * 1000.0 if since else 0.0
                try:
                    price, src, ours = self._price_external_reduction(
                        pos, delta, tol, since_ms,
                        self._ext_attempts >= self.EXT_PRICE_ATTEMPTS)
                except Exception as e:
                    if self._ext_attempts < self.EXT_PRICE_ATTEMPTS:
                        self.log.warning("external partial close: pricing failed "
                                         "(%d/%d): %s", self._ext_attempts,
                                         self.EXT_PRICE_ATTEMPTS, e)
                        return
                    price, src, ours = 0.0, "UNPRICED after retries", False
                if ours:
                    self.log.info("size change %.4f -> %.4f is our own order's "
                                  "fill — tracking updated, recorded as "
                                  "bot_partial, no alert", prev_qty, q)
                    state.record_fill(
                        side="partial_close", qty=delta, price=price,
                        pnl_usd=_exit_pnl_usd(pos.side, prev_entry, price, delta),
                        reason="bot_partial", equity_after=equity,
                        client_order_id_root=prev_root)
                    self._last_position_qty = q
                    self._last_position_entry = float(pos.entry_price)
                    self._ext_pending, self._ext_attempts = None, 0
                    return
                pnl = _exit_pnl_usd(pos.side, prev_entry, price, delta)
                state.record_fill(
                    side="partial_close", qty=delta, price=price, pnl_usd=pnl,
                    reason="external_partial", equity_after=equity,
                    client_order_id_root=prev_root)
        except Exception as e:
            self._last_position_qty = prev_qty
            self._last_position_entry = prev_entry
            self.log.warning("external size change: not recorded yet (%s)", e)
            return
        self._last_position_qty = q
        self._last_position_entry = float(pos.entry_price)
        self._ext_pending, self._ext_attempts = None, 0
        if kind == "increase":
            self.log.warning("external size increase detected: %s %.4f -> %.4f %s",
                             pos.side, prev_qty, q, self.base_asset)
            subject = "Bot position grew outside the bot"
            body = (f"{pos.side.upper()} {self.base_asset} size {prev_qty:.4f} -> "
                    f"{q:.4f} without a bot order. Brackets untouched.")
        else:
            self.log.warning(
                "external partial close detected: %s %.4f -> %.4f %s, sold %.4f @ "
                "%.4f (%s) pnl=%s", pos.side, prev_qty, q, self.base_asset,
                -diff, price, src, pnl)
            subject = f"Bot {pos.side.upper()} partially closed outside the bot"
            body = (f"{self.base_asset} {prev_qty:.4f} -> {q:.4f}; {-diff:.4f} @ "
                    f"{price:,.4f} ({src}).\nGross PnL on that part: "
                    f"{'n/a' if pnl is None else f'${pnl:+,.2f}'}. Recorded as "
                    f"external_partial; brackets untouched.")
        try:
            send_alert(subject, body + f"\nsignal_id: {prev_root or '(untagged)'}")
        except Exception:
            self.log.exception("external size change: alert failed")

    def _resolve_fill_price(self, order: dict | None,
                            attempts: int = 3,
                            delay_s: float = 0.4) -> float | None:
        """True average fill price of an order the bot just placed.

        Binance's CREATE response for a reduce-only MARKET order carries
        avgPrice "0.00" -- the fill is not attributed to the order yet -- so
        `order_avg_price()` returns None at EVERY bot-initiated close. Measured
        2026-08-23: every `close` row in all three legs' fills tables since
        launch has price=0.0, and the boot-flatten path's
        `order_avg_price(o) or pos.entry_price` fallback therefore always
        reported PnL of exactly 0.

        Re-fetching the SAME order by clientOrderId returns the real average --
        verified against both real closes that day (v1
        snap-v1-1787462070917-ce -> 76003.10, donchian
        snap-d3-1787256026107-ce -> 76010.00). Keying on the order is
        authoritative; scanning fetch_my_trades and side-matching (what
        _detect_bracket_exit does, correctly, a tick later) can pick up an
        unrelated fill when run immediately after placing.

        Returns None if the price cannot be established, leaving the decision
        to the caller. NEVER raises: the exchange has already executed the
        close, and bookkeeping must not turn that into an exception.
        """
        px = order_avg_price(order)
        if px:
            return px
        o = order or {}
        coid = str(o.get("clientOrderId") or "")
        oid = str(o.get("id") or "")
        if not coid and not oid:
            return None
        params = {"origClientOrderId": coid} if coid else {}
        for i in range(max(1, attempts)):
            if i:
                time.sleep(delay_s)
            try:
                fetched = self.client.ex.fetch_order(oid or None, self.symbol,
                                                     params=params)
            except Exception:
                # Do NOT retry a failed CALL. ccxt's timeout is 15s, so three
                # attempts would block the tick loop ~46s -- and this path is
                # pure bookkeeping running AFTER the exchange has already closed
                # the position. A stalled loop is worse than a missing price,
                # which _maybe_backfill_close_prices repairs later. Retries exist
                # only for the "fetched, but the fill is not attributed yet"
                # case handled below.
                self.log.warning("exit-price: fetch_order failed for %s -- "
                                 "not retrying (bookkeeping must not stall the "
                                 "loop); the close row will be written with "
                                 "price 0.0 and repaired by the close-price "
                                 "backfill from exchange trades", coid or oid)
                return None
            px = order_avg_price(fetched)
            if px:
                return px
        self.log.warning("exit-price: could not resolve fill price for %s -- "
                         "recording without it; the close-price backfill will "
                         "repair the row from exchange trades", coid or oid)
        return None

    BACKFILL_INTERVAL_S = 60.0
    BACKFILL_LOOKBACK_DAYS = 7
    BACKFILL_COOLDOWN_MIN_S = 300.0
    BACKFILL_COOLDOWN_MAX_S = 3600.0
    BACKFILL_UNMATCHED_LOG_S = 6 * 3600.0
    BACKFILL_GIVEUP_TRIES = 3

    BACKFILL_CLAIMS_KEY = "close_backfill_claimed"

    def _backfill_claims(self, utc_now) -> dict[str, float]:
        """Exchange orders already used to price a close row ({order key:
        epoch s}), pruned to the lookback window -- stops two rows claiming the
        same order."""
        try:
            raw = json.loads(state.get_meta(self.BACKFILL_CLAIMS_KEY) or "{}")
        except ValueError:
            raw = {}
        horizon = utc_now.timestamp() - 2 * self.BACKFILL_LOOKBACK_DAYS * 86400
        return {str(k): float(v) for k, v in raw.items() if float(v) >= horizon}

    def _backfill_claim(self, claims: dict[str, float], keys, utc_now) -> None:
        for k in keys:
            claims[str(k)] = utc_now.timestamp()
        state.set_meta(self.BACKFILL_CLAIMS_KEY, json.dumps(claims))

    def _maybe_backfill_close_prices(self) -> None:
        """Repair `close` fills recorded with price 0.0 (see close_backfill.py).

        Bookkeeping only and NEVER raises. Bounded per tick: at most one pass
        per BACKFILL_INTERVAL_S, one row per pass (2 REST calls, fewest-tried
        row first so an unmatchable row cannot starve the others). ANY exchange
        error ends the pass and starts a cooldown (5 min, doubling to 1 h) so a
        nonce/rate-limit/ban condition is left to the trading calls; it is
        logged once per cooldown, not swallowed."""
        now = time.monotonic()
        if now < getattr(self, "_backfill_next", 0.0):
            return
        self._backfill_next = now + self.BACKFILL_INTERVAL_S
        try:
            from datetime import timedelta

            import close_backfill
            utc_now = datetime.now(UTC)
            cutoff = utc_now - timedelta(days=self.BACKFILL_LOOKBACK_DAYS)
            tries = self.__dict__.setdefault("_backfill_tries", {})
            unmatched_logged = self.__dict__.setdefault("_backfill_unmatched_logged", {})
            bad = self.__dict__.setdefault("_backfill_bad_rows", set())
            rows = [r for r in state.unpriced_close_rows(cutoff.isoformat())
                    if r[0] not in bad]
            if rows:
                fid, ts, qty, root, reason = min(rows, key=lambda r: (tries.get(r[0], 0), r[0]))
                claims = self._backfill_claims(utc_now)
                try:
                    datetime.fromisoformat(ts)
                except ValueError:
                    bad.add(fid)
                    self.log.warning("close-backfill: fill #%s has an unreadable ts "
                                     "%r; skipping it", fid, ts)
                    return
                try:
                    res = close_backfill.price_close_row(
                        self.client.ex, self.symbol, self.client.coid_prefix,
                        root, ts, qty, reason, claimed=set(claims))
                except Exception as e:
                    cd = min(max(getattr(self, "_backfill_cooldown_s", 0.0) * 2,
                                 self.BACKFILL_COOLDOWN_MIN_S),
                             self.BACKFILL_COOLDOWN_MAX_S)
                    self._backfill_cooldown_s = cd
                    self._backfill_next = now + cd
                    self.log.warning("close-backfill: exchange error (%s: %s); "
                                     "pausing backfill %.0fs", type(e).__name__, e, cd)
                    return
                self._backfill_cooldown_s = 0.0
                if res is None:
                    tries[fid] = tries.get(fid, 0) + 1
                    if now - unmatched_logged.get(fid, -1e18) >= self.BACKFILL_UNMATCHED_LOG_S:
                        unmatched_logged[fid] = now
                        self.log.warning("close-backfill: fill #%s (root=%s, qty %.4f) "
                                         "has no matching close trade on the exchange "
                                         "yet; will retry", fid, root, qty)
                else:
                    price, pnl, keys = res
                    if state.set_close_price(fid, price, pnl):
                        self._backfill_claim(claims, keys, utc_now)
                        self.log.info("close-backfill: fill #%s (%s, root=%s) -> "
                                      "price %.4f pnl %s", fid, reason, root, price,
                                      "n/a" if pnl is None else f"{pnl:+.4f}")
            # Give up only on rows THIS process looked up repeatedly (and found
            # no match for) that have aged out of the lookback; one summary alert
            # per pass. The meta key is written only if the alert was sent.
            old = [r for r in state.unpriced_close_rows(
                       (utc_now - timedelta(days=2 * self.BACKFILL_LOOKBACK_DAYS)).isoformat(),
                       until_iso=cutoff.isoformat())
                   if tries.get(r[0], 0) >= self.BACKFILL_GIVEUP_TRIES
                   and not state.get_meta(f"close_backfill_gaveup:{r[0]}")]
            if old:
                lines = "\n".join(f"  fill #{r[0]} ({r[4]}, {r[1]}, signal_id {r[3]})"
                                  for r in old)
                if send_alert(
                        "Close rows still have no price",
                        f"{self.base_asset}: {len(old)} close row(s) were never repaired "
                        f"by the backfill and are older than {self.BACKFILL_LOOKBACK_DAYS} "
                        f"days; giving up:\n{lines}"):
                    for r in old:
                        state.set_meta(f"close_backfill_gaveup:{r[0]}", utc_now.isoformat())
        except Exception:
            self.log.exception("close-backfill: pass failed (ignored)")

    def _open_entry_fill(self) -> tuple[str, float, float] | None:
        """(side, qty, price) of the entry for the position that is OPEN NOW;
        `price` is the running cost basis (manual adds re-weight it), `qty` the
        entry row's.

        Reads the most recent fill of ANY kind and requires it to be an entry,
        exactly as _detect_bracket_exit does. Filtering on `reason='entry'`
        instead would skip BACKWARDS over intervening closes and hand back an
        entry that is already closed -- attributing a stale entry price to the
        current position and writing a wrong number into the trade record.

        That is reachable: a position opened manually, or one whose own entry
        was never recorded, has no entry row of its own, so the newest fill is
        the PREVIOUS trade's close. Returning None there is correct -- the
        caller records the exit without a PnL rather than with a false one.
        """
        try:
            with sqlite3.connect(state.DB_PATH) as c:
                row = c.execute(
                    "SELECT reason, side, qty, price, id FROM fills "
                    "WHERE reason NOT IN " + _SIZE_ROWS_SQL + " "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
                ext_rows = []
                if row and row[0] == "entry":
                    ext_rows = c.execute(
                        "SELECT reason, qty, price FROM fills WHERE id > ? "
                        "AND reason IN " + _SIZE_ROWS_SQL + " ORDER BY id",
                        (row[4],)).fetchall()
        except Exception:
            self.log.exception("exit-pnl: entry-fill lookup failed")
            return None
        if not row or row[0] != "entry":
            return None
        _, basis = _replay_size_rows(float(row[2]), float(row[3]), ext_rows)
        return str(row[1]), float(row[2]), basis

    def _maybe_time_stop(self, equity: float) -> None:
        # Read the config BEFORE fetch_position so a leg with no time stop
        # costs zero API calls per tick instead of one. That is supertrend
        # (max_hold_bars: 0); donchian-v3 sets a positive value and DOES pay the fetch.
        max_hold = int((self.params.get("strategy") or {}).get("max_hold_bars", 0))
        if max_hold <= 0:
            return
        pos = self.client.fetch_position(self.symbol)
        if pos.side == "flat":
            return
        try:
            with sqlite3.connect(state.DB_PATH) as c:
                row = c.execute(
                    "SELECT ts FROM fills WHERE reason='entry' "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if not row:
                return
            entry_ts = datetime.fromisoformat(row[0])
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=UTC)
            age_s = (datetime.now(UTC) - entry_ts).total_seconds()
            if time_stop_due(max_hold, self.bar_seconds, age_s):
                if self.dry_run:
                    self.log.info("DRY-RUN: would time-stop close after %.1f hours", age_s / 3600)
                    return
                root = state.latest_entry_coid_root()
                self.log.info("Time-stop firing after %.1f hours (root=%s)",
                              age_s / 3600, root or "—")
                order = self.client.close_position(
                    self.symbol, client_order_id_root=root, close_leg="x")
                exit_price = self._resolve_fill_price(order)
                _entry = self._open_entry_fill()
                state.record_fill(side="close", qty=pos.qty,
                                  price=float(exit_price or 0.0),
                                  pnl_usd=_exit_pnl_usd(
                                      pos.side, _entry[2] if _entry else None,
                                      exit_price, pos.qty),
                                  reason="time_stop", equity_after=self._ledger_equity(equity),
                                  client_order_id_root=root)
                state.enqueue_bot_event(
                    "exit", signal_id=root, side=pos.side, qty=float(pos.qty),
                    price_usd=exit_price,
                    equity_usd=self._ledger_equity(equity),
                    payload={"reason": "time_stop", "age_h": age_s / 3600},
                )
                send_alert("Bot time-stop close",
                           f"Closed {pos.side} {pos.qty:.4f} {self.base_asset} after "
                           f"{age_s/3600:.1f}h hold"
                           + (f" @ {exit_price:,.4f}" if exit_price else "")
                           + f". Equity: {equity:.2f}\n"
                           f"signal_id: {root or '(untagged)'}")
        except Exception as e:
            self.log.warning("time-stop check failed: %s", e)

    def _maybe_channel_exit(self, equity: float) -> None:
        """The generic live trend-exit hook. Name is historical — donchian-v3
        was the first leg to need it; three legs use it now, each with its own
        rule (see bot_internals.trend_exit_signal):

          - donchian-v3:   opposite Donchian exit-channel cross on 4h. Its ONLY
                           profit-taking mechanism — the entry places no TP leg.
          - supertrend:    opposite STDir flip on 4h, alongside its TP bracket.
          - multifactor-v1: adverse 15m EMA(200) cross, alongside its TP bracket
                           (added 2026-08-10 — see MULTIFACTOR_V1_LIVE_EXIT_VERDICT.md).

        Strategy-gated: a no-op for every leg not in strategy_uses_trend_exit
        (cnh, v3all), which keep their brackets and never reach this code.

        Structurally mirrors _maybe_time_stop: fetch position, evaluate, and on
        a trigger close reduce-only (COID-tagged, close_leg='ce'), record the
        fill, enqueue the exit event, alert. The surviving bracket sibling is
        swept by close_position's own COID-scoped cancel_open_orders before the
        reduce-only close is placed.

        Bars are fetched on self.entry_tf, so each leg is evaluated on the
        timeframe its own rule is defined on (4h for donchian/supertrend, 15m
        for v1) without any per-strategy branching here.
        """
        if not strategy_uses_trend_exit(self.strategy_name):
            return
        try:
            pos = self.client.fetch_position(self.symbol)
            if pos.side == "flat":
                return
            df = self.client.fetch_ohlcv(self.symbol, self.entry_tf, limit=1500)
            # Evaluate CLOSED bars only. Binance returns the still-forming
            # current bar as the last row; the backtest only ever sees closed
            # bars, so drop it here to keep the channel-exit decision in parity.
            # No extra length precheck: channel_exit_signal carries the
            # backtest's own warmup/NaN guard (max(entry, exit, atr)+1 bars),
            # and suppressing an EXIT on a short fetch is worse than checking
            # (Sourcery, PR #7).
            df = df.iloc[:-1]
            should_exit, dbg = trend_exit_signal(
                self.strategy_name, df, pos.side, self.params)
            if not should_exit:
                return
            if self.dry_run:
                self.log.info("DRY-RUN: would trend-exit close %s %.4f (%s)",
                              pos.side, pos.qty, dbg.get("reason"))
                return
            root = state.latest_entry_coid_root()
            fill_reason = trend_exit_fill_reason(self.strategy_name)
            self.log.info("Trend-exit firing (%s): closing %s (root=%s) rule=%s "
                          "close=%.2f dbg=%s",
                          fill_reason, pos.side, root or "—", dbg.get("reason"),
                          dbg.get("cur_close", float("nan")), dbg)
            order = self.client.close_position(
                self.symbol, client_order_id_root=root, close_leg="ce")
            exit_price = self._resolve_fill_price(order)
            _entry = self._open_entry_fill()
            state.record_fill(side="close", qty=pos.qty,
                              price=float(exit_price or 0.0),
                              pnl_usd=_exit_pnl_usd(
                                  pos.side, _entry[2] if _entry else None,
                                  exit_price, pos.qty),
                              reason=fill_reason, equity_after=self._ledger_equity(equity),
                              client_order_id_root=root)
            state.enqueue_bot_event(
                "exit", signal_id=root, side=pos.side, qty=float(pos.qty),
                price_usd=exit_price,
                equity_usd=self._ledger_equity(equity),
                # Rule-specific keys are absent per strategy (donchian has no
                # trend_ema, v1 has no channel) — .get() yields None, which the
                # dashboard renders as blank rather than a wrong number.
                payload={"reason": fill_reason,
                         "rule": dbg.get("reason"),
                         "cur_close": dbg.get("cur_close"),
                         "exit_lower": dbg.get("exit_lower"),
                         "exit_upper": dbg.get("exit_upper"),
                         "trend_ema": dbg.get("trend_ema")},
            )
            send_alert("Bot trend-exit close",
                       f"Closed {pos.side} {pos.qty:.4f} {self.base_asset} on "
                       f"{dbg.get('reason', 'trend exit')}"
                       + (f" @ {exit_price:,.4f}" if exit_price else "")
                       + f". Equity: {equity:.2f}\n"
                       f"signal_id: {root or '(untagged)'}")
        except Exception as e:
            self.log.warning("channel-exit check failed: %s", e)

    BE_RETRY_S = 30.0
    BE_FAIL_LIMIT = 10
    BE_FILL_GRACE_S = 300   # a fill this soon after the bar open counts that bar (parity)

    def _maybe_breakeven(self, equity: float) -> None:
        """Move the stop to breakeven ONCE, when the open trade's max favourable
        excursion over CLOSED bars reaches `breakeven.at_r` R.

        The live half of TRAILING_STOP_VERDICT.md's one surviving candidate
        (SOL, at_r 2.0). Parity target: tools/trailing_stop_study.TrailMixin
        with be_at_r — evaluated at each bar close, stop = entry ± buffer,
        and if price is already back through that level the backtest fills at
        the next open, so live closes at market.

        Config-gated: legs without `breakeven.enabled` return before any API
        call. Evaluates once per newly closed bar.

        Order of operations is chosen so the position is never unprotected:
          1. read the algo book (unreadable → retry later; acting blind is the
             July reprotect bug);
          2. place the new `-sb` stop unless one already rests (a retry after
             a placement that timed out but was accepted must not double it —
             and place_tagged_stop never falls back to an untagged order);
          3. cancel ONLY the original `-s` stop, by exact id — the `-t` TP stays;
          4. persist `be_moved` in active_bracket once the new stop rests,
             even if step 3 failed (alerted), so it never fires twice.
        """
        be = self.params.get("breakeven") or {}
        if not be.get("enabled", False):
            return
        # Both required when enabled (config_check.CONDITIONAL_KEYS) — no
        # literal fallbacks.
        at_r = float(be["at_r"])
        buf = float(be["buffer_pct"]) / 100.0
        if at_r <= 0:
            return
        now = time.time()
        last_closed = (int(now // self.bar_seconds) - 1) * self.bar_seconds
        if self._be_evaluated_bar == last_closed or now < self._be_retry_after:
            return
        try:
            self._breakeven_step(equity, at_r, buf, last_closed, now)
        except Exception as e:
            self._breakeven_failed(f"unexpected error: {e}", now)
            self.log.exception("breakeven check failed")

    def _breakeven_failed(self, why: str, now: float) -> None:
        """Retry in BE_RETRY_S; after BE_FAIL_LIMIT in a row, alert once and
        wait for the next bar (which retries from scratch)."""
        self._be_fail_n += 1
        self._be_retry_after = now + self.BE_RETRY_S
        self.log.warning("breakeven: %s (attempt %d)", why, self._be_fail_n)
        if self._be_fail_n == self.BE_FAIL_LIMIT:
            send_alert("Breakeven stop move FAILING",
                       f"{self.base_asset}: {why}\n{self._be_fail_n} attempts in a "
                       f"row. The original stop is still in place; will retry "
                       f"next bar.")
            self._be_fail_n = 0
            self._be_evaluated_bar = (int(now // self.bar_seconds) - 1) * self.bar_seconds

    def _breakeven_step(self, equity: float, at_r: float, buf: float,
                        last_closed: int, now: float) -> None:
        pos = self.client.fetch_position(self.symbol)
        if pos.side == "flat" or pos.qty == 0:
            self._be_evaluated_bar = last_closed
            self._be_fail_n = 0
            return
        raw = state.get_meta("active_bracket")
        try:
            ab = json.loads(raw) if raw else None
        except (ValueError, TypeError):
            ab = None
        root = state.latest_entry_coid_root()
        # Only act on a bracket record that provably belongs to THIS position
        # (same guards as reprotect, plus the entry coid) — active_bracket is
        # never cleared on legs that run without reprotect.
        ep = float((ab or {}).get("entry_price") or 0.0)
        if (not ab or ab.get("side") != pos.side or ep <= 0 or pos.entry_price <= 0
                or abs(ep - pos.entry_price) / pos.entry_price > 0.02
                or not root or str(ab.get("signal_id")) != str(root)):
            self.log.info("breakeven: no bracket record matching the open %s "
                          "(root=%s) — not acting", pos.side, root)
            self._be_evaluated_bar = last_closed
            return
        if ab.get("be_moved"):
            self._be_evaluated_bar = last_closed
            return
        sl_distance = float(ab.get("sl_distance") or 0.0)

        with sqlite3.connect(state.DB_PATH) as c:
            row = c.execute("SELECT ts FROM fills WHERE reason='entry' "
                            "ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            self._be_evaluated_bar = last_closed
            return
        entry_ts = datetime.fromisoformat(row[0])
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=UTC)
        fill_bar = int(entry_ts.timestamp() // self.bar_seconds) * self.bar_seconds
        # The backtest fills at the bar's OPEN and counts that bar's High/Low.
        # Live entries land ~10-30 s after the open, so counting the fill bar
        # keeps parity. A LATE fill (e.g. an entry evaluated right after a
        # mid-bar restart) would credit extremes printed before the position
        # existed, so the fill bar only counts inside a short grace window
        # (Sourcery, PR #32 — its "always exclude" fix would break parity).
        first_bar = (fill_bar if entry_ts.timestamp() - fill_bar <= self.BE_FILL_GRACE_S
                     else fill_bar + self.bar_seconds)
        n_bars = (last_closed - first_bar) // self.bar_seconds + 1
        if n_bars <= 0:
            self._be_evaluated_bar = last_closed
            return
        df = self.client.fetch_ohlcv(self.symbol, self.entry_tf,
                                     limit=int(min(1500, n_bars + 3)))
        df = df.iloc[:-1]   # drop the still-forming bar — closed bars only
        if df.empty or int(df.index[-1].timestamp()) < last_closed:
            # The exchange has not published the bar that just closed yet.
            self._be_retry_after = now + 10.0
            return
        win = df[(df.index >= pd.Timestamp(first_bar, unit="s"))
                 & (df.index <= pd.Timestamp(last_closed, unit="s"))]
        highs, lows = win["High"].tolist(), win["Low"].tolist()
        # Running extreme persisted per position, so a window truncated by the
        # 1500-bar fetch cap (a trade older than ~250 days on 4h) can never
        # forget an earlier excursion (Sourcery, PR #32).
        stored_ext = ab.get("be_ext")
        if isinstance(stored_ext, (int, float)):
            (highs if pos.side == "long" else lows).append(float(stored_ext))
        due, mfe = breakeven_due(pos.side, pos.entry_price, sl_distance,
                                 highs, lows, at_r)
        if not due:
            ext = pos.entry_price + (mfe if pos.side == "long" else -mfe) * sl_distance
            if ext != stored_ext:
                ab["be_ext"] = ext
                state.set_meta("active_bracket", json.dumps(ab))
            self._be_evaluated_bar = last_closed
            return
        be_price = breakeven_stop_price(pos.side, pos.entry_price, buf)
        if self.dry_run:
            self.log.info("DRY-RUN: would move %s stop to breakeven %.4f (MFE %.2fR)",
                          pos.side, be_price, mfe)
            self._be_evaluated_bar = last_closed
            return

        sb_coid = f"{self.coid_prefix}{root}-sb"
        s_coid = f"{self.coid_prefix}{root}-s"
        rows, ok = self.client.fetch_algo_orders(self.symbol)
        if not ok:
            self._breakeven_failed("algo book unreadable", now)
            return
        if not any(r.get("clientAlgoId") == sb_coid for r in rows):
            if self._breakeven_through(pos.side, be_price):
                self._breakeven_close(pos, root, be_price, mfe, equity)
                ab["be_moved"] = True
                state.set_meta("active_bracket", json.dumps(ab))
                self._be_evaluated_bar = last_closed
                self._be_fail_n = 0
                return
            try:
                self.client.place_tagged_stop(self.symbol, pos.side, pos.qty,
                                              be_price, str(root), "sb")
            except Exception as e:
                # Placement can be refused because price crossed the level
                # between the mark read and the order — decide by MARK price,
                # not by matching an error string the algo endpoint may word
                # differently.
                if self._breakeven_through(pos.side, be_price):
                    self._breakeven_close(pos, root, be_price, mfe, equity)
                    ab["be_moved"] = True
                    state.set_meta("active_bracket", json.dumps(ab))
                    self._be_evaluated_bar = last_closed
                    self._be_fail_n = 0
                    return
                self._breakeven_failed(f"placing the breakeven stop failed: {e}", now)
                return

        old_cancelled = self.client.cancel_algo_by_coid(self.symbol, s_coid)
        if not old_cancelled:
            # False means "not found" OR "cancel failed". Re-read: if the book
            # is readable and -s is not there, it is already gone (an earlier
            # attempt cancelled it before crashing) — protected by -sb, no alarm.
            rows2, ok2 = self.client.fetch_algo_orders(self.symbol)
            if ok2 and not any(r.get("clientAlgoId") == s_coid for r in rows2):
                old_cancelled = True
        ab["be_moved"] = True
        ab["be_price"] = be_price
        state.set_meta("active_bracket", json.dumps(ab))
        self._be_evaluated_bar = last_closed
        self._be_fail_n = 0
        state.record_event("INFO", "breakeven_stop",
                           {"side": pos.side, "qty": pos.qty, "entry": pos.entry_price,
                            "be_price": be_price, "mfe_r": round(mfe, 3),
                            "old_stop_cancelled": old_cancelled},
                           signal_id=root)
        self.log.warning("Breakeven: %s %.4f stop moved to %.4f (MFE %.2fR, old -s "
                         "cancelled=%s)", pos.side, pos.qty, be_price, mfe, old_cancelled)
        send_alert("Stop moved to breakeven",
                   f"{pos.side} {pos.qty:.4f} {self.base_asset} @ {pos.entry_price:,.4f} "
                   f"reached +{mfe:.2f}R — stop now {be_price:,.4f}. TP unchanged."
                   + ("" if old_cancelled else
                      "\n⚠ The ORIGINAL stop could not be cancelled and may still "
                      "rest (reduce-only, further away — the breakeven stop fires "
                      "first). Check the Binance app."))

    def _breakeven_through(self, side: str, be_price: float) -> bool:
        """Is MARK already at/through the breakeven level? Unknown → False
        (then placement is attempted, and a refusal re-asks this)."""
        mark = self.client.fetch_mark_price(self.symbol)
        if mark is None:
            return False
        return mark <= be_price if side == "long" else mark >= be_price

    def _breakeven_close(self, pos, root: str, be_price: float, mfe: float,
                         equity: float) -> None:
        """Price is already back through breakeven: close at market — the
        backtest fills this case at the next bar's open. Mirrors
        _maybe_channel_exit's close + ledger path.

        If anything below close_position() raises, the exit is NOT lost:
        the next tick's _detect_bracket_exit sees open→flat with the newest
        fills row still `entry`, and records the close from the exchange's own
        trades (labelled bracket_exit). Pinned by
        test_ledger_failure_after_close_is_recovered (Sourcery, PR #32, asked
        for a pending-close record; the existing flat-edge path already is one)."""
        self.log.warning("Breakeven: mark already through %.4f — closing %s at market",
                         be_price, pos.side)
        # Close FIRST, sweep AFTER. If the close raises, the original -s/-t
        # are still resting and the retry starts from a protected position.
        # (close_position's default sweep-then-close would leave it with no
        # stop and no TP, and a retry that finds price back above breakeven
        # would restore only the -sb stop — the TP lost for good.)
        order = self.client.close_position(self.symbol, client_order_id_root=root,
                                           close_leg="be", sweep_first=False)
        try:
            self.client.cancel_open_orders(self.symbol, coid_prefix=self.coid_prefix)
        except Exception:
            # Leftover reduce-only brackets are harmless on a flat position,
            # and _detect_bracket_exit / the pre-entry sweep remove them.
            self.log.exception("breakeven close: post-close sweep failed")
        if order is None:
            # Already flat by the time close_position looked (e.g. a manual
            # close in between). Write NOTHING: a bogus `close` row here would
            # make _detect_bracket_exit treat the real exit as already recorded
            # and suppress its true price/PnL. It records the exit next tick.
            self.log.warning("Breakeven: position already flat at close time — "
                             "leaving the exit to _detect_bracket_exit")
            return
        exit_price = self._resolve_fill_price(order)
        _entry = self._open_entry_fill()
        state.record_fill(side="close", qty=pos.qty, price=float(exit_price or 0.0),
                          pnl_usd=_exit_pnl_usd(pos.side, _entry[2] if _entry else None,
                                                exit_price, pos.qty),
                          reason="breakeven_exit", equity_after=self._ledger_equity(equity),
                          client_order_id_root=root)
        state.enqueue_bot_event("exit", signal_id=root, side=pos.side, qty=float(pos.qty),
                                price_usd=exit_price, equity_usd=self._ledger_equity(equity),
                                payload={"reason": "breakeven_exit", "be_price": be_price,
                                         "mfe_r": round(mfe, 3)})
        send_alert("Bot breakeven close",
                   f"Closed {pos.side} {pos.qty:.4f} {self.base_asset} — it reached "
                   f"+{mfe:.2f}R but price was already back through breakeven "
                   f"{be_price:,.4f}"
                   + (f" @ {exit_price:,.4f}" if exit_price else "")
                   + f". Equity: {equity:.2f}\nsignal_id: {root}")

    def loop(self) -> int:
        self.log.info("Bot loop started. poll=%.1fs symbol=%s", self.poll_s, self.symbol)
        backoff_s = self.poll_s
        transient_backoff_s = self.poll_s
        while not self._stopped:
            try:
                self._heartbeat()
                halt_by = halt_source(self.instance)
                if halt_by is not None:
                    which = ("GLOBAL data/HALT (manual stop-all)"
                             if halt_by == "global"
                             else f"self data/HALT_{self.instance}")
                    self.log.warning("HALT flag present [%s] — %s.", which,
                                     "exiting (DRY-RUN, no flatten)" if self.dry_run
                                     else "flattening and exiting")
                    if not self.dry_run:
                        root = state.latest_entry_coid_root()
                        self.client.close_position(
                            self.symbol,
                            client_order_id_root=root, close_leg="h")
                    state.enqueue_bot_event(
                        "halt",
                        payload={"dry_run": bool(self.dry_run),
                                 "halt_source": halt_by},
                    )
                    try:
                        consolidate_push.drain()
                    except Exception as e:
                        self.log.warning("halt push failed: %s", e)
                    send_alert("Bot HALTED",
                               f"{which} detected. "
                               f"{'DRY-RUN exit, no flatten.' if self.dry_run else 'Position flattened.'} "
                               f"Bot exiting.")
                    return 0

                self._maybe_reconcile_principal()
                equity_known = True
                self._equity_stale = False
                try:
                    equity = self.client.fetch_equity_usdt()
                    self._equity_unknown_streak = 0
                    self._last_known_equity = equity
                except EquityUnavailable as e:
                    # Equity UNKNOWN this tick: no kill switch / breaker / entry on
                    # a guessed number. Risk-REDUCING exits still run (they only
                    # use equity for bookkeeping, so the last known value stands in).
                    equity_known = False
                    equity = self._last_known_equity
                    self._equity_stale = True
                    self._equity_unknown_streak += 1
                    self.log.warning("equity unavailable (streak=%d) — %s: %s",
                                     self._equity_unknown_streak,
                                     "exits only" if equity is not None else "skipping tick", e)
                    now = time.time()
                    if (now - self._equity_unknown_alert_ts >= 900.0
                            or self._equity_unknown_streak == 10):
                        self._equity_unknown_alert_ts = now
                        send_alert("Equity unreadable" + (
                            f" ({self._equity_unknown_streak} ticks in a row)"
                            if self._equity_unknown_streak >= 10 else " (entries paused)"),
                            str(e))
                if equity_known:
                    # Best-effort, no exchange calls: UTC-midnight anchor snapshot
                    # (flat OR holding); the monitor's live-equity feed follows
                    # _detect_bracket_exit below. Never on a guessed equity.
                    self._roll_daily_anchor(equity)
                if equity_known and self._check_kill_switch(equity):
                    if not self.dry_run:
                        root = state.latest_entry_coid_root()
                        self.client.close_position(
                            self.symbol,
                            client_order_id_root=root, close_leg="k")
                    return 0

                # #49 no-stop hooks are risk-REDUCING (close / protect a naked
                # position) and use equity for bookkeeping only (None-safe), so
                # they run even when equity has never been read (#45).
                if equity is None:
                    self._maybe_finish_no_stop_close(equity)
                    self._maybe_guard_unverified_entry(equity)
                else:
                    self._detect_bracket_exit(equity)
                    # AFTER the detector, never before: a bracket fill moves the
                    # wallet on the exchange before the bot's position snapshot and
                    # fills table know, so persisting first reads every SL/TP exit as
                    # an unexplained move (a false "possible transfer" flag).
                    if equity_known:
                        self._persist_live_equity(equity)
                    self._maybe_finish_no_stop_close(equity)
                    self._maybe_guard_unverified_entry(equity)
                    self._maybe_time_stop(equity)
                    self._maybe_channel_exit(equity)
                    self._maybe_breakeven(equity)
                    # RE-ENABLED 2026-08-11 with algo-aware detection (bracket_state
                    # merges /fapi/v1/openOrders AND /fapi/v1/openAlgoOrders). It was
                    # disabled 2026-07-22 because the plain-only check read a healthy
                    # conditional bracket as missing and re-placed to -4045. Gated on
                    # `reprotect.enabled`, and ships `observe_only: true` — it logs
                    # the decision and places nothing until a full live position
                    # passes with zero "WOULD re-place" lines.
                    self._maybe_reprotect(equity)
                    self._maybe_backfill_close_prices()
                if not equity_known:
                    transient_backoff_s = min(transient_backoff_s * 2, 30)
                    time.sleep(transient_backoff_s)
                    continue
                self._maybe_enter(equity)
                self._maybe_push_consolidate(equity)
                backoff_s = self.poll_s
                transient_backoff_s = self.poll_s
                time.sleep(self.poll_s)
            except KeyboardInterrupt:
                self.log.info("KeyboardInterrupt — clean stop.")
                return 0
            except ccxt.InvalidNonce as e:
                # Binance error -1021: request timestamp outside recvWindow.
                # Caused by droplet clock drift. ccxt's adjustForTimeDifference
                # only re-syncs on the next fetch_time call, so force one now
                # before retrying. Short capped backoff — drift normally clears
                # within a few seconds after the resync.
                self.log.warning("InvalidNonce (clock drift) — forcing time re-sync: %s", e)
                try:
                    self.client.ex.load_time_difference()
                except Exception as sync_err:
                    self.log.warning("load_time_difference() failed: %s", sync_err)
                transient_backoff_s = min(transient_backoff_s * 2, 30)
                time.sleep(transient_backoff_s)
            except ccxt.NetworkError as e:
                # Connection drop, timeout, DDoSProtection, ExchangeNotAvailable.
                # All transient — retry on a short capped backoff so a 10-second
                # Binance hiccup doesn't escalate to the 300s catch-all backoff.
                self.log.warning("NetworkError (transient) — retrying: %s", e)
                transient_backoff_s = min(transient_backoff_s * 2, 30)
                time.sleep(transient_backoff_s)
            except Exception as e:
                self.log.exception("loop error: %s", e)
                try:
                    state.record_event("ERROR", "loop_error", str(e))
                except Exception:
                    # Must never raise out of loop(): exit -> systemd restart ->
                    # boot() flattens a live position. Bookkeeping only.
                    self.log.exception("loop error: record_event failed")
                backoff_s = min(backoff_s * 2, 300)
                time.sleep(backoff_s)
        return 0


RISK_FP_META_KEY = "risk_ceilings_fingerprint"
RISK_VALUES_META_KEY = "risk_ceilings_values"


def _note_risk_ceilings_change(log: logging.Logger) -> None:
    """Compensating control for the ceilings leaving the RISK_REVIEW-hooked
    risk.py (2026-10-22): fingerprint this leg's installed `risk:` ceilings and
    compare with the last ACKNOWLEDGED one in the leg's state DB.

    Changed -> WARNING with old->new per key + an alert. The new fingerprint is
    stored in the same boot ONLY if send_alert returned True; otherwise the old
    one is kept (ERROR logged) so the next boot re-alerts until delivered. If
    the alert went out but storing fails, ERROR is logged and the next boot
    repeats the alert (a repeat beats silence). No stored fingerprint (first
    boot after migration) -> store silently. Observability only: never blocks
    boot (the gate is config_check); any failure is logged, not raised."""
    try:
        c = risk.ceilings()
        values = {key: getattr(c, field) for field, (key, _t) in risk.RISK_FIELDS.items()}
        blob = json.dumps(values, sort_keys=True)
        fp = hashlib.sha256(blob.encode()).hexdigest()
        state.init_db()
        old_fp = state.get_meta(RISK_FP_META_KEY)
        if old_fp == fp:
            return
        if old_fp is not None:
            try:
                old_vals = json.loads(state.get_meta(RISK_VALUES_META_KEY) or "{}")
            except ValueError:
                old_vals = {}
            changed = [f"{k}: {old_vals.get(k)!r} -> {values[k]!r}"
                       for k in sorted(values) if old_vals.get(k) != values[k]]
            log.warning("RISK CEILINGS CHANGED since last boot: %s",
                        "; ".join(changed) or "(fingerprint only)")
            delivered = send_alert(
                "risk ceilings changed since last boot",
                "The params `risk:` ceilings differ from the last boot:\n  "
                + "\n  ".join(changed or ["(fingerprint only)"])
                + "\nIf this was not an approved risk review, HALT the leg.")
            if delivered is not True:
                log.error("risk ceilings change alert NOT delivered — keeping the "
                          "previous fingerprint so the next boot re-alerts")
                return
        try:
            state.set_meta(RISK_VALUES_META_KEY, blob)
            state.set_meta(RISK_FP_META_KEY, fp)
        except Exception as e:
            log.error("risk ceilings fingerprint NOT stored (%s: %s) — the next boot "
                      "will repeat the change alert", type(e).__name__, e)
    except Exception as e:  # observability must not block boot
        log.error("risk ceilings fingerprint check failed: %s: %s", type(e).__name__, e)


def _refuse_boot(instance: str, reason: str, log: logging.Logger | None = None) -> int:
    """Refuse to boot: say why (stderr, log if set up), attempt ONE alert, and
    return EXIT_CONFIG (78). The units list 78 in RestartPreventExitStatus, so
    the leg stays down instead of crash-looping and re-alerting."""
    msg = f"instance={instance}: {reason}"
    print(f"FATAL: BOOT REFUSED {msg}", file=sys.stderr)
    if log is not None:
        log.critical("BOOT REFUSED %s", msg)
    try:  # send_alert is a logged no-op when SMTP is not configured
        send_alert(f"BOOT REFUSED — {instance} did not start", msg)
    except Exception:  # alerting must never mask the refusal itself
        print("FATAL: boot-refused alert failed", file=sys.stderr)
    return EXIT_CONFIG


def main() -> int:
    global LOG_FILE, HEARTBEAT, CONFIG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Observe-only: fetch real data, evaluate signals, log what WOULD happen, "
                         "but never place real orders. Also honored via DRY_RUN=1 env var.")
    ap.add_argument("--log-level", default="INFO")
    # Canonical multi-instance selector. `--instance donchian` derives config,
    # state.db, log, heartbeat paths in one shot. Individual --config etc.
    # below override single paths if you need to mix and match.
    ap.add_argument("--instance", default="v1", choices=list(INSTANCE_PROFILES.keys()),
                    help="Named instance profile (v1 | donchian). Derives all four paths "
                         "from INSTANCE_PROFILES. Individual --config / --state-db / "
                         "--log-file / --heartbeat take precedence if also set.")
    ap.add_argument("--config", default=None,
                    help="Path to params YAML (overrides --instance default).")
    ap.add_argument("--state-db", default=None,
                    help="Path to SQLite state.db (overrides --instance default).")
    ap.add_argument("--log-file", default=None,
                    help="Path to JSONL log file (overrides --instance default).")
    ap.add_argument("--heartbeat", default=None,
                    help="Path to heartbeat file (overrides --instance default).")
    ap.add_argument("--check-config", action="store_true",
                    help="Validate the instance's params YAML against REQUIRED_KEYS "
                         "and exit (0 = OK, 78 = refused). Touches no exchange, "
                         "env, state or logs.")
    args = ap.parse_args()

    # Resolve effective paths: profile defaults, then per-flag overrides.
    profile = INSTANCE_PROFILES[args.instance]
    config_path = Path(args.config) if args.config else profile["config"]

    if args.check_config:
        try:
            _params = load_params(config_path)
            validate_params(_params, source=str(config_path))
            risk.ceilings_from_params(_params)  # CONFIG OK => the boot loader succeeds too
            require_strategy_available(resolve_strategy_name(_params))
        except Exception as e:  # any load/parse/validate failure is a refusal
            print(f"CONFIG REFUSED [{args.instance}] {type(e).__name__}: {e}",
                  file=sys.stderr)
            return EXIT_CONFIG
        print(f"CONFIG OK [{args.instance}] {config_path}")
        return 0
    state_db_path = Path(args.state_db) if args.state_db else profile["state_db"]
    log_file_path = Path(args.log_file) if args.log_file else profile["log_file"]
    heartbeat_path = Path(args.heartbeat) if args.heartbeat else profile["heartbeat"]

    # Apply path overrides BEFORE the rest of main runs — state, LOG_FILE,
    # and HEARTBEAT are module-level constants that the rest of the bot
    # reads, so write them before _setup_logging() / bot.boot() / bot.loop().
    #
    # Every pre-trading boot failure from here to the params check exits 78
    # (EX_CONFIG) via _refuse_boot — the only status in the units'
    # RestartPreventExitStatus — so a leg that cannot boot alerts once and stays
    # down instead of crash-looping (and re-alerting) every RestartSec.
    try:
        CONFIG_PATH = config_path
        state.set_db_path(state_db_path)
        LOG_FILE = log_file_path
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT = heartbeat_path
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)

        # Overlay .env.{instance} (donchian's sub-account API key, cnh_short's
        # ALERT_TAG, etc.) on top of the base .env loaded at exchange.env import.
        instance_env_path = load_env_for_instance(args.instance)
    except Exception as e:
        return _refuse_boot(args.instance, f"boot setup failed: {type(e).__name__}: {e}")
    # Fail loudly if a NON-v1 leg has no per-instance env file.
    #
    # load_env_for_instance returns None when `.env.<instance>` is absent, which
    # is correct for v1 (the base .env IS its env) but a live-money hazard for
    # every other leg: without the overlay the leg silently inherits v1's
    # BINANCE_API_KEY and trades a different symbol inside v1's account, pushes
    # telemetry under v1's CONSOLIDATE_SOURCE, and emails under v1's ALERT_TAG.
    # env.py's own docstring names this as the thing the overlay exists to
    # prevent, but nothing enforced it. Enforced here, before any client is
    # constructed.
    #
    # Not exempted for --dry-run on purpose: a dry run against the wrong account
    # reads the wrong equity (so sizing and the min-notional check are both
    # meaningless) and still emits events under the wrong source.
    if args.instance != "v1" and instance_env_path is None:
        print(
            f"FATAL: instance {args.instance!r} has no .env.{args.instance}.\n"
            f"  Without it this leg would use the BASE .env — i.e. v1's Binance\n"
            f"  API key and account — and place {args.instance} orders there.\n"
            f"  Create {REPO_ROOT / f'.env.{args.instance}'} holding that leg's own\n"
            f"  sub-account BINANCE_API_KEY / BINANCE_API_SECRET, plus\n"
            f"  CONSOLIDATE_SOURCE=... and ALERT_TAG=... so its dashboard events\n"
            f"  and alert emails do not collide with v1's.",
            file=sys.stderr,
        )
        return _refuse_boot(args.instance, f"no .env.{args.instance} (would trade "
                                           "on v1's account) — create it, then start")
    # Make the instance name discoverable by anything that emits alerts
    # later (alerts.send_alert prepends it to the subject so every leg's
    # mail threads cleanly in Gmail).
    os.environ["SNAPBACK_INSTANCE"] = args.instance

    dry_run = args.dry_run or os.environ.get("DRY_RUN") in ("1", "true", "yes")

    try:
        log = _setup_logging(args.log_level)
    except Exception as e:
        return _refuse_boot(args.instance, f"logging setup failed: {type(e).__name__}: {e}")
    # Refuse to boot on a missing/incomplete params file — never trade on code
    # defaults. Runs before any exchange client exists. Exit 78 (EX_CONFIG) is
    # listed in the units' RestartPreventExitStatus, so this alerts once and
    # stays down instead of crash-looping.
    try:
        params = load_params(config_path)
        validate_params(params, source=str(config_path))
        # Hard ceilings come from the params `risk:` block (no code defaults);
        # installed before any Bot exists, so boot()'s checks read this leg's.
        risk.configure(params)
        # A lazily-imported strategy evaluator must import NOW, not on the
        # first bar inside the loop's error backoff (no-op for the live legs).
        require_strategy_available(resolve_strategy_name(params))
    except Exception as e:  # ANY failure here (bad UTF-8, ...) refuses with 78,
        # never a traceback exit 1 that Restart=on-failure would crash-loop on.
        return _refuse_boot(args.instance, f"CONFIG REFUSED {type(e).__name__}: {e}", log)
    log.info("snapback-btc booting instance=%s strategy=%s config=%s state=%s log=%s",
             args.instance, resolve_strategy_name(params),
             config_path, state.DB_PATH, LOG_FILE)
    if instance_env_path is not None:
        log.info("snapback-btc loaded per-instance env: %s",
                 instance_env_path.relative_to(REPO_ROOT))
    env = get_env()
    log.info("snapback-btc booting env=%s dry_run=%s", env, dry_run)
    if env == "mainnet" and not dry_run:
        log.warning("=" * 60)
        log.warning("MAINNET LIVE MODE. Real money at risk.")
        log.warning("=" * 60)

    _note_risk_ceilings_change(log)
    bot = Bot(params, dry_run=dry_run, instance=args.instance)
    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    bot.boot()
    return bot.loop()


if __name__ == "__main__":
    sys.exit(main())
