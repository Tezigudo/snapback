"""
Capitulation-bounce alert watcher — ALERT-ONLY, no order placement.

Watches a multi-coin USDT-perp universe on the 1h timeframe for the
validated LONG-capitulation signal and emails the operator when one fires.
This is a discretionary-trade alarm, NOT a bot leg: it never touches
`risk.py`, never places an order, and reads only public kline data.

Signal (validated by walk-forward, see reports/CAPITULATION_ALERT.html):
    gain over 48h  <  -15%          (deep drop = capitulation)
    AND Parabolic-SAR flipped to UP within the last 3 bars
    AND MACD histogram crossed >0   within the last 3 bars
Trade plan attached to each alert (the operator executes by hand):
    SL = entry - 2.0 * ATR(14)
    TP = entry + 3.0 * ATR(14)
    time-stop = 24 bars (~24h)

Walk-forward out-of-sample edge: +2.77% EV/trade, ~61% WR, z ~= 7 vs random.

State / anti-spam:
    A small JSON state file (data/capitulation_alerts.json) records the
    timestamp of the last alert per coin. A new alert for the same coin is
    suppressed for COOLDOWN_BARS hours. NOTE: this is a per-coin alert
    debounce, NOT the backtest's exit-based cooldown — there is no position
    to track because we place no orders.

Designed to run hourly via cron ON THE DROPLET (SMTP is blocked locally).
    # crontab -e  (on the droplet, in the repo, with the venv python)
    7 * * * * cd /root/snapback-btc && .venv/bin/python -m tools.capitulation_watch >> logs/capitulation_watch.log 2>&1

Usage:
    python -m tools.capitulation_watch              # live: scan, alert, persist state
    python -m tools.capitulation_watch --dry-run    # scan + print, no email, no state write
    python -m tools.capitulation_watch --backfill    # full-history scan (reproduction / report data)
    python -m tools.capitulation_watch --seed-state  # mark current bar as seen for every coin, no alerts
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from alerts import send_alert
from exchange.data import load_klines
from strategy.indicators import atr, macd, parabolic_sar

log = logging.getLogger("capitulation_watch")

# --- validated parameters (do not tune without re-running walk-forward) ------
GAIN_WINDOW = 48          # bars (~48h) over which the drop is measured
DROP_THRESHOLD = 0.15     # require <= -15% over the window
CONFIRM_WINDOW = 3        # SAR flip and MACD cross must be within last N bars
SCAN_LOOKBACK = 12        # live: catch signals up to N closed bars back, so a
                          # cron outage of up to ~N hours doesn't lose a signal
                          # (decoupled from CONFIRM_WINDOW, which gates the signal itself)
SL_ATR_MULT = 2.0
TP_ATR_MULT = 3.0
TIME_STOP_BARS = 24
ATR_PERIOD = 14
COOLDOWN_BARS = 48        # per-coin alert debounce (hours)
TIMEFRAME = "1h"

# Live watchlist. MATIC is intentionally excluded — delisted from Binance
# Futures 2024-09-11, so a live fetch returns stale/empty data. It remains in
# the backtest universe as a survivor-bias counter-test only.
WATCHLIST = [
    "BTC", "ETH", "SOL", "ADA", "WLD",
    "XRP", "DOGE", "BNB", "AVAX", "LINK", "DOT", "LTC", "BCH",
    "NEAR", "APT", "SUI", "ATOM", "ARB", "INJ",
]

# Enough history that path-dependent SAR matches the backtest, and indicators
# are fully warm. Cheap because load_klines is an incremental cached fetcher.
HISTORY_DAYS = 540

_REPO = Path(__file__).resolve().parent.parent
STATE_PATH = _REPO / "data" / "capitulation_alerts.json"


# --- indicator wiring (reuses strategy/indicators.py — single source of truth)
def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["sar"] = parabolic_sar(df["high"], df["low"])
    _, _, df["mhist"] = macd(df["close"])
    return df


def _signal_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series, True on each bar where the capitulation signal completes.

    Mirrors the validated walk-forward signal; see reports/CAPITULATION_ALERT.html.
    """
    close = df["close"]
    gain = close / close.shift(GAIN_WINDOW) - 1.0

    sar_up = df["sar"] < close                       # SAR below price = uptrend
    flip = sar_up & ~sar_up.shift(1).fillna(False)   # flip bar
    flip_recent = flip.rolling(CONFIRM_WINDOW).max().astype(bool)

    cross = (df["mhist"] > 0) & (df["mhist"].shift(1) <= 0)
    cross_recent = cross.rolling(CONFIRM_WINDOW).max().astype(bool)

    return ((gain < -DROP_THRESHOLD) & flip_recent & cross_recent).fillna(False)


def _trade_plan(row: pd.Series) -> dict:
    entry = float(row["close"])
    a = float(row["atr"])
    return {
        "entry": entry,
        "atr": a,
        "stop_loss": entry - SL_ATR_MULT * a,
        "take_profit": entry + TP_ATR_MULT * a,
        "risk_pct": SL_ATR_MULT * a / entry * 100,
        "reward_pct": TP_ATR_MULT * a / entry * 100,
    }


# --- state -------------------------------------------------------------------
def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("state file unreadable (%s); starting fresh", e)
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def _in_cooldown(last_iso: str | None, bar_ts: pd.Timestamp) -> bool:
    if not last_iso:
        return False
    try:
        last = pd.Timestamp(last_iso)
    except ValueError:
        return False
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    return (bar_ts - last) < pd.Timedelta(hours=COOLDOWN_BARS)


# --- core --------------------------------------------------------------------
def _load_coin(coin: str, *, backfill: bool) -> pd.DataFrame | None:
    symbol = f"{coin}/USDT:USDT"
    days = 365 * 6 if backfill else HISTORY_DAYS
    try:
        df = load_klines(symbol, TIMEFRAME, days_back=days)
    except Exception as e:  # network / exchange errors must never crash the run
        log.warning("%s: load_klines failed: %s", coin, e)
        return None
    if df is None or df.empty or len(df) < GAIN_WINDOW + ATR_PERIOD + 5:
        log.warning("%s: insufficient data (%s rows)", coin, 0 if df is None else len(df))
        return None
    # Drop the last bar: it may be the still-forming current hour.
    return df.iloc[:-1]


def scan(*, backfill: bool = False) -> list[dict]:
    """Return every signal across the watchlist.

    Live mode reports only signals on the most recent SCAN_LOOKBACK closed
    bars (so a cron outage of up to ~SCAN_LOOKBACK hours doesn't lose a fresh
    signal). Backfill mode reports every historical signal — used for
    reproduction and the report.
    """
    hits: list[dict] = []
    for coin in WATCHLIST:
        df = _load_coin(coin, backfill=backfill)
        if df is None:
            continue
        df = _enrich(df)
        mask = _signal_mask(df)

        if backfill:
            idxs = list(df.index[mask])
        else:
            recent = df.index[-SCAN_LOOKBACK:]
            idxs = [ts for ts in df.index[mask] if ts in recent]

        for ts in idxs:
            row = df.loc[ts]
            hits.append({"coin": coin, "bar_ts": ts, **_trade_plan(row),
                         "drop_48h_pct": float((row["close"] / df["close"].shift(GAIN_WINDOW).loc[ts] - 1) * 100)})
    return hits


def _format_alert(hit: dict) -> tuple[str, str]:
    subject = f"capitulation LONG — {hit['coin']} dropped {hit['drop_48h_pct']:.1f}% / 48h"
    body = (
        f"Capitulation-bounce signal fired on {hit['coin']}/USDT (1h).\n"
        f"Bar (UTC): {hit['bar_ts']}\n\n"
        f"  48h move:   {hit['drop_48h_pct']:+.1f}%\n"
        f"  entry:      {hit['entry']:.6g}\n"
        f"  ATR(14):    {hit['atr']:.6g}\n\n"
        f"Suggested discretionary plan (you place the order, the bot does not):\n"
        f"  stop loss:  {hit['stop_loss']:.6g}   (-{hit['risk_pct']:.2f}%, 2.0x ATR)\n"
        f"  take profit:{hit['take_profit']:.6g}   (+{hit['reward_pct']:.2f}%, 3.0x ATR)\n"
        f"  time stop:  {TIME_STOP_BARS} bars (~{TIME_STOP_BARS}h)\n\n"
        f"Walk-forward edge: +2.77% EV/trade, ~61% WR (out-of-sample).\n"
        f"This is an ALERT only. No order has been or will be placed automatically.\n"
    )
    return subject, body


def run(*, dry_run: bool, backfill: bool) -> int:
    hits = scan(backfill=backfill)
    if backfill:
        print(f"[backfill] {len(hits)} signals across {len(WATCHLIST)} coins (full history)")
        by_coin: dict[str, int] = {}
        for h in hits:
            by_coin[h["coin"]] = by_coin.get(h["coin"], 0) + 1
        for c in sorted(by_coin, key=lambda k: -by_coin[k]):
            print(f"  {c:6s} {by_coin[c]}")
        return 0

    state = _load_state()
    fired = 0
    for hit in sorted(hits, key=lambda h: h["bar_ts"]):
        coin = hit["coin"]
        bar_ts = hit["bar_ts"]
        last = state.get(coin, {}).get("last_alert_ts")
        if _in_cooldown(last, bar_ts):
            log.info("%s: signal at %s suppressed (cooldown)", coin, bar_ts)
            continue
        # NOTE: the cooldown check above already covers "already alerted this
        # bar or newer" (a last >= bar_ts implies a <48h gap), so no separate
        # — and crash-prone — re-parse of `last` is needed here.

        subject, body = _format_alert(hit)
        if dry_run:
            print(f"\n--- WOULD ALERT ---\n{subject}\n{body}")
        else:
            ok = send_alert(subject, body, tag="snapback-capitulation")
            log.info("%s: alert %s (bar %s)", coin, "sent" if ok else "FAILED", bar_ts)
            if not ok:
                # Do NOT advance state on a failed send, or the signal is lost
                # forever; leave it un-recorded so the next run retries.
                continue
            state.setdefault(coin, {})["last_alert_ts"] = bar_ts.isoformat()
        fired += 1

    if not dry_run:
        _save_state(state)
    print(f"{'[dry-run] ' if dry_run else ''}scanned {len(WATCHLIST)} coins, {fired} new alert(s)")
    return 0


def _seed_state() -> int:
    """Record the latest closed bar per coin without alerting — use once before
    going live so the first cron run doesn't email a backlog of stale signals."""
    state = _load_state()
    now = datetime.now(UTC).isoformat()
    for coin in WATCHLIST:
        state.setdefault(coin, {})["last_alert_ts"] = now
    _save_state(state)
    print(f"seeded state for {len(WATCHLIST)} coins at {now}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="scan and print; no email, no state write")
    p.add_argument("--backfill", action="store_true", help="full-history scan for reproduction/report")
    p.add_argument("--seed-state", action="store_true", help="mark current bar seen for all coins, no alerts")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.seed_state:
        return _seed_state()
    return run(dry_run=args.dry_run, backfill=args.backfill)


if __name__ == "__main__":
    raise SystemExit(main())
