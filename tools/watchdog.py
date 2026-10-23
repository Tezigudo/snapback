"""Per-leg heartbeat watchdog.

Each bot leg touches its `data/heartbeat_{instance}` file every loop tick.
This watchdog (run via cron) checks each file's mtime against a staleness
threshold and emails the operator on down/up transitions.

Design choices:

- **Per-leg ALERT_TAG**: each leg has its own subject thread in Gmail, so the
  down alert lands next to that leg's other emails. We set SNAPBACK_INSTANCE
  before each send so alerts.send_alert prepends the right instance bracket.

- **Re-alert cadence**: while a leg stays down, we re-alert at most once per
  RENOTIFY_INTERVAL_S (default 30 min). Otherwise a single 8h outage spams
  the inbox via cron.

- **Recovery alerts**: when a leg goes from down → up, we send one "recovered"
  email so the operator knows manual intervention isn't needed.

- **State file**: `data/watchdog_state.json` tracks each leg's current state
  and last_alert_ts. Updated atomically (write-tmp + rename) so a crash mid
  write can't corrupt it.

- **Failure isolation**: a problem with one leg's check (e.g., missing
  heartbeat file) never blocks the others.

Cron usage:
    */2 * * * * /root/snapback-btc/.venv/bin/python -m tools.watchdog \
        >> /tmp/watchdog-cron.log 2>&1
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# We're invoked as a script via `python -m tools.watchdog`; import siblings
# the same way bot.py does.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from alerts import send_alert  # noqa: E402

log = logging.getLogger("snapback.watchdog")

# Stale threshold: bots touch heartbeat every poll (60s default). 300s = 5
# polls missed, generous enough that a restart loop doesn't trigger noise.
STALE_THRESHOLD_S = int(os.environ.get("WATCHDOG_STALE_S", "300"))
# Re-alert cadence while DOWN: don't spam more than once per 30 min.
RENOTIFY_INTERVAL_S = int(os.environ.get("WATCHDOG_RENOTIFY_S", "1800"))

STATE_FILE = REPO_ROOT / "data" / "watchdog_state.json"


@dataclass(frozen=True)
class LegSpec:
    instance: str          # matches bot.py's --instance value
    heartbeat: Path
    alert_tag: str         # subject prefix the leg uses for its own alerts


LEGS: tuple[LegSpec, ...] = (
    LegSpec(
        instance="v1",
        heartbeat=REPO_ROOT / "data" / "heartbeat",
        alert_tag="snapback-btc",
    ),
    LegSpec(
        instance="donchian",
        heartbeat=REPO_ROOT / "data" / "heartbeat_donchian",
        alert_tag="snapback-btc-donchian",
    ),
    # SOL supertrend leg — deployed 2026-07-25. Its systemd unit
    # (snapback-sol-supertrend) is enabled and running, so unlike cnh_short below
    # it will NOT generate permanent-DOWN spam. Added the same day it started so
    # the leg is not silently unmonitored.
    LegSpec(
        instance="sol_supertrend",
        heartbeat=REPO_ROOT / "data" / "heartbeat_sol_supertrend",
        alert_tag="snapback-sol-st",
    ),
    # cnh_short is intentionally SHELVED/offline (paper, down since the 2026-07-01
    # halt). Monitoring it made the watchdog email "cnh_short DOWN" every ~30 min
    # forever (harmless when MailerSend was dead; became inbox spam once Brevo
    # went live 2026-07-22). Re-add this LegSpec only if cnh_short is brought back
    # to real live trading.
)


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: tmp file + rename, so a crash mid-write can't leave a
    # half-written JSON that we'd fail to parse next tick.
    with tempfile.NamedTemporaryFile(
        "w", dir=STATE_FILE.parent, delete=False, encoding="utf-8",
    ) as f:
        json.dump(state, f, indent=2, sort_keys=True)
        tmp_path = Path(f.name)
    tmp_path.replace(STATE_FILE)


def _heartbeat_age_s(path: Path) -> float | None:
    """Age in seconds since the file was last touched. None if missing."""
    if not path.exists():
        return None
    return (datetime.now().timestamp() - path.stat().st_mtime)


def _emit(leg: LegSpec, subject: str, body: str) -> None:
    """Send an alert with the leg's subject thread."""
    # Set SNAPBACK_INSTANCE so alerts.send_alert prepends [{instance}] —
    # threading matches each leg's other emails in Gmail.
    os.environ["SNAPBACK_INSTANCE"] = leg.instance
    ok = send_alert(subject, body, tag=leg.alert_tag)
    if not ok:
        log.warning("watchdog: send_alert returned False for %s", leg.instance)


def _check_leg(leg: LegSpec, state: dict, now_ts: float) -> None:
    age = _heartbeat_age_s(leg.heartbeat)
    is_down = age is None or age > STALE_THRESHOLD_S

    leg_state = state.get(leg.instance, {"state": "unknown", "last_alert_ts": 0.0})
    prev_state = leg_state.get("state", "unknown")
    last_alert_ts = float(leg_state.get("last_alert_ts") or 0.0)

    new_state = "down" if is_down else "up"
    transition = (prev_state != new_state)
    renotify_due = (
        new_state == "down"
        and (now_ts - last_alert_ts) >= RENOTIFY_INTERVAL_S
    )

    if is_down and (transition or renotify_due):
        age_str = "MISSING" if age is None else f"{age:.0f}s ({age/60:.1f}min)"
        subj = f"watchdog: {leg.instance} DOWN — heartbeat stale"
        body = (
            f"snapback-btc/{leg.instance} appears DOWN.\n"
            f"\n"
            f"Heartbeat path : {leg.heartbeat}\n"
            f"Age            : {age_str}\n"
            f"Threshold      : {STALE_THRESHOLD_S}s\n"
            f"Time (UTC)     : {datetime.now(UTC).isoformat(timespec='seconds')}\n"
            f"\n"
            f"Restart command (run on droplet):\n"
            f"  tmux kill-session -t bot_{leg.instance} 2>/dev/null; \\\n"
            f"  tmux new-session -ds bot_{leg.instance} "
            f"-c /root/snapback-btc \\\n"
            f"    '/root/snapback-btc/.venv/bin/python -m bot --dry-run "
            f"--instance {leg.instance}'\n"
            f"\n"
            f"(Drop --dry-run for live mode.)\n"
        )
        _emit(leg, subj, body)
        leg_state["last_alert_ts"] = now_ts

    elif transition and new_state == "up" and prev_state == "down":
        subj = f"watchdog: {leg.instance} RECOVERED"
        body = (
            f"snapback-btc/{leg.instance} is alive again.\n"
            f"Heartbeat age: {age:.0f}s (< threshold {STALE_THRESHOLD_S}s).\n"
            f"Time (UTC): {datetime.now(UTC).isoformat(timespec='seconds')}.\n"
        )
        _emit(leg, subj, body)

    leg_state["state"] = new_state
    state[leg.instance] = leg_state


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    state = _load_state()
    now_ts = datetime.now().timestamp()
    for leg in LEGS:
        try:
            _check_leg(leg, state, now_ts)
        except Exception:
            log.exception("watchdog: check failed for %s", leg.instance)
    _save_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
