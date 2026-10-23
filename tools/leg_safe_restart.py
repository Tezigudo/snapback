#!/usr/bin/env python3
"""Restart ONE snapback leg, but only while it is provably flat and clean.

    ./.venv/bin/python /root/snapback-btc/tools/leg_safe_restart.py <instance>

WHY THIS EXISTS. boot() FLATTENS any open position in live mode, so restarting
a leg mid-trade silently closes it at market. /root/v1_deferred_restart.py
enforced that check for v1 only -- donchian and sol_supertrend had no sanctioned
path at all, so they got raw `systemctl restart`, which performs no check
whatsoever. This generalises the v1 guard to every leg.

FLAT means all three, checked against the EXCHANGE, not the local DB:
  - position side == flat
  - zero plain open orders
  - zero resting ALGO orders  (SL/TP live on /fapi/v1/openAlgoOrders, which
    fetch_open_orders does NOT return -- the blind spot behind the July -4045
    spam. Checking only plain orders would call "flat" with a bracket resting.)
An UNREADABLE algo book is treated as "unknown", never as "clean".

Also fixes a real footgun in the v1 script: it logged a HARDCODED target commit
("restarting onto 8bbe584") that had gone stale, so its output claimed a deploy
that was not what was on disk. Here the target is read from git at run time.

On anything unexpected: do nothing and leave the bot alone.

EXIT CODES are deliberately distinct, because "I safely did nothing" and "I may
have just broken a real-money leg" must never share a code:
    0  restarted and healthy   (or, with --check, the leg IS restartable)
    1  RESTARTED BUT NOT ACTIVE -- needs a human NOW
    2  usage error
    3  refused; the bot was NOT touched (in position, orders resting, or the
       algo book could not be read).  This is the SAFE outcome.

RACE: a leg could open a position between the flat check and the restart, and
boot() would then flatten it. Mitigated by RE-VERIFYING position and both order
books immediately before `systemctl restart`, which cuts the window from the
several seconds the exchange round-trips take down to one syscall. It is not
eliminated -- doing that would need the bot to stop producing orders first, and
`systemctl stop` then finding the leg NOT flat would strand an open position
with no supervising process, which is strictly worse than refusing while it
runs. Narrowed and disclosed, deliberately not closed.

Usage:
    tools/leg_safe_restart.py <instance>          restart if flat
    tools/leg_safe_restart.py <instance> --check report only, never restart
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/root/snapback-btc")
sys.path.insert(0, str(REPO))

# instance -> (systemd unit, symbol)
LEGS = {
    "v1":             ("snapback-btc",            "BTC/USDT:USDT"),
    "donchian":       ("snapback-btc-donchian",   "BTC/USDT:USDT"),
    "sol_supertrend": ("snapback-sol-supertrend", "SOL/USDT:USDT"),
}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}", flush=True)


RESTART_ISSUED = False   # so the crash handler cannot claim "untouched" wrongly


def _git_rev(repo: Path) -> str | None:
    """Short HEAD, or None if git could not answer."""
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    rev = r.stdout.strip()
    return rev if r.returncode == 0 and rev else None


def _tracked_dirty(repo: Path) -> bool:
    """True if TRACKED files are modified.

    --untracked-files=no is essential, not cosmetic: the droplet legitimately
    carries 24 untracked runtime files (data/*.db, logs/, *.bak), so a plain
    `git status --porcelain` is never empty there and a naive dirty-check would
    refuse every restart forever.
    """
    r = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                        "--untracked-files=no"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def _unit_prop(unit: str, prop: str) -> str:
    return subprocess.run(["systemctl", "show", unit, "-p", prop, "--value"],
                          capture_output=True, text=True).stdout.strip()


def main() -> int:
    args = sys.argv[1:]
    check_only = "--check" in args
    args = [a for a in args if a != "--check"]
    if len(args) != 1 or args[0] not in LEGS:
        log(f"usage: leg_safe_restart.py <{'|'.join(LEGS)}> [--check]")
        return 2
    inst = args[0]
    unit, symbol = LEGS[inst]

    # Real deployed commit, not a hardcoded string that can rot. If we cannot
    # establish it we must NOT restart: the whole point of this tool is an
    # audit record naming what shipped, and "unknown" is not that.
    target = _git_rev(REPO)
    if target is None:
        log(f"[{inst}] cannot determine target revision -- refusing "
            f"(bot NOT touched)")
        return 3

    from exchange.env import load_env_for_instance
    load_env_for_instance(inst)
    from exchange.binance_client import BinanceClient

    c = BinanceClient.from_env()
    pos = c.fetch_position(symbol)
    if pos.side != "flat" or pos.qty:
        log(f"[{inst}] IN POSITION: {pos.side} qty={pos.qty} @ {pos.entry_price} "
            f"-- refusing (boot() would flatten it)")
        return 3

    plain = c.ex.fetch_open_orders(symbol)
    algo, algo_ok = c.fetch_algo_orders(symbol)
    if not algo_ok:
        log(f"[{inst}] algo book UNREADABLE -- refusing "
            f"(cannot prove the book is clean)")
        return 3
    if plain or algo:
        log(f"[{inst}] flat but {len(plain)} plain + {len(algo)} algo orders "
            f"resting -- refusing")
        return 3

    try:
        equity = c.fetch_equity_usdt()
    except Exception as e:
        log(f"[{inst}] equity UNREADABLE ({e}) -- refusing (bot NOT touched)")
        return 3
    if check_only:
        log(f"[{inst}] FLAT and clean (equity=${equity:.2f}) -- WOULD restart "
            f"{unit} onto {target}. --check given, doing nothing.")
        return 0
    log(f"[{inst}] FLAT and clean (equity=${equity:.2f}) -- restarting {unit} "
        f"onto {target}")

    # Re-verify EVERYTHING immediately before pulling the trigger.
    # (a) the repo: a concurrent deploy between rev-parse and now would restart
    #     the leg onto code the audit stamp does not name.
    if _git_rev(REPO) != target or _tracked_dirty(REPO):
        log(f"[{inst}] repo changed or tracked files dirty since the check "
            f"-- refusing (bot NOT touched)")
        return 3
    # (b) the exchange: the bot has been free to trade this whole time.
    pos2 = c.fetch_position(symbol)
    if pos2.side != "flat" or pos2.qty:
        log(f"[{inst}] opened {pos2.side} qty={pos2.qty} during the check window "
            f"-- refusing (bot NOT touched)")
        return 3
    algo2, algo2_ok = c.fetch_algo_orders(symbol)
    if not algo2_ok or algo2 or c.ex.fetch_open_orders(symbol):
        log(f"[{inst}] orders appeared during the check window (or the algo book "
            f"went unreadable) -- refusing (bot NOT touched)")
        return 3

    old_pid = _unit_prop(unit, "MainPID")
    global RESTART_ISSUED
    RESTART_ISSUED = True
    subprocess.run(["systemctl", "restart", unit], check=True)
    time.sleep(15)
    new_pid = _unit_prop(unit, "MainPID")
    active = subprocess.run(["systemctl", "is-active", unit],
                            capture_output=True, text=True).stdout.strip()
    log(f"[{inst}] restarted: pid {old_pid} -> {new_pid}, is-active={active}")

    # Bookkeeping ONLY, and it runs after the restart has already happened, so a
    # failure here must never be reported as "leaving bot untouched".
    stamp = REPO / "data" / f"{inst}_safe_restart.done"
    try:
        stamp.write_text(json.dumps({
            "instance": inst, "unit": unit,
            "fired_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "target": target, "old_pid": old_pid, "new_pid": new_pid,
            "is_active": active, "equity_at_restart": equity,
        }, indent=2))
        log(f"[{inst}] wrote {stamp}")
    except Exception as e:
        log(f"[{inst}] RESTART SUCCEEDED (pid {old_pid} -> {new_pid}, "
            f"is-active={active}) but the stamp could not be written: {e} "
            f"-- do NOT re-run on the strength of this message")
    if active != "active":
        log(f"[{inst}] *** RESTARTED BUT is-active={active} -- NEEDS ATTENTION ***")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        if RESTART_ISSUED:
            log(f"ERROR AFTER THE RESTART WAS ISSUED -- the service was already "
                f"restarted, do NOT assume it was untouched: {e}")
            raise SystemExit(1) from e
        log(f"ERROR (leaving bot untouched): {e}")
        raise SystemExit(3) from e
