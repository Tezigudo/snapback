"""
Environment + mainnet lockfile gate.

Reads BINANCE_ENV from .env (default: testnet). If BINANCE_ENV=mainnet, also
requires a `confirm_mainnet.lock` file at the repo root — created manually by
the user, never committed.

This is the first defence against accidentally running real-money trades.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
except ImportError:  # P0: deps may not be installed yet
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False

Env = Literal["testnet", "mainnet"]

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKFILE = REPO_ROOT / "confirm_mainnet.lock"

# Load .env once at import time. Missing file is fine — env vars may be set elsewhere.
load_dotenv(REPO_ROOT / ".env", override=False)


# Instances that legitimately run on the BASE .env (the main Binance account)
# and therefore do NOT require a `.env.{instance}` overlay. Only v1 is the
# base-account leg (confirmed on the droplet: `.env.v1` does not exist, while
# `.env.donchian` / `.env.cnh_short` do, each holding that sub-account's own
# keys). EVERY other instance is a Binance sub-account leg and MUST supply its
# own `.env.{instance}`; booting one without it is a hard error (see below).
BASE_ACCOUNT_INSTANCES: frozenset[str] = frozenset({"v1"})


class EnvError(RuntimeError):
    pass


def load_env_for_instance(instance: str) -> Path | None:
    """Overlay per-instance secrets on top of the base .env.

    Each sub-account leg of the multi-leg deploy (donchian, cnh_short, …) has
    its own Binance sub-account API key, ALERT_TAG, and CONSOLIDATE_SOURCE in
    `.env.{instance}`. Without this overlay a leg launched via
    `python -m bot --instance X` would silently inherit v1's base `.env` =
    the MAIN-account keys, read the whole main-account balance, and anchor /
    kill-switch on it. That silent fallback is the root cause of donchian's
    wrong-account read, so it is now a FAIL-LOUD condition.

    Behaviour:
      - `.env.{instance}` exists → load it (override=True) and return its path.
      - file absent AND instance is a base-account leg (v1) → return None; the
        base `.env` is legitimately its environment.
      - file absent AND instance requires its own sub-account env → raise
        EnvError. NEVER silently inherit the main-account keys.
    """
    candidate = REPO_ROOT / f".env.{instance}"
    if candidate.exists():
        # override=True so instance values WIN over the base .env load above.
        load_dotenv(candidate, override=True)
        return candidate
    if instance in BASE_ACCOUNT_INSTANCES:
        # v1 (base account): the base .env IS its env. Not an error.
        return None
    raise EnvError(
        f"Instance {instance!r} is a sub-account leg but its env file is "
        f"missing: {candidate}\n"
        "Refusing to boot: without it the leg would silently inherit the base "
        ".env (v1's MAIN-account keys), read the wrong account's balance, and "
        "could trip its kill switch on a foreign equity (the donchian "
        "wrong-account failure mode).\n"
        f"Create .env.{instance} holding THIS sub-account's own "
        "BINANCE_API_KEY / BINANCE_API_SECRET (see DEPLOY_COMBINED.md), or add "
        f"{instance!r} to BASE_ACCOUNT_INSTANCES if it is meant to run on the "
        "main account."
    )


def get_env() -> Env:
    """Return the active environment, enforcing the mainnet lockfile gate."""
    raw = os.environ.get("BINANCE_ENV", "testnet").strip().lower()
    if raw not in ("testnet", "mainnet"):
        raise EnvError(
            f"BINANCE_ENV must be 'testnet' or 'mainnet', got {raw!r}"
        )

    if raw == "mainnet" and not LOCKFILE.exists():
        raise EnvError(
            f"BINANCE_ENV=mainnet but lockfile missing: {LOCKFILE}\n"
            "Refusing to run on mainnet without explicit confirmation.\n"
            "If you really mean it, run `/promote-mainnet` and follow the checklist."
        )

    return raw  # type: ignore[return-value]


def get_api_credentials() -> tuple[str, str]:
    """Return (api_key, api_secret), erroring if missing."""
    key = os.environ.get("BINANCE_API_KEY", "").strip()
    secret = os.environ.get("BINANCE_API_SECRET", "").strip()
    if not key or not secret:
        raise EnvError(
            "BINANCE_API_KEY / BINANCE_API_SECRET not set. "
            f"Edit {REPO_ROOT / '.env'} (copy from .env.example)."
        )
    return key, secret


def global_halt_path() -> Path:
    """The GLOBAL manual stop-all flag. Operator-only: nothing automated writes
    it. Its presence halts EVERY leg (used for a deliberate stop-everything)."""
    return REPO_ROOT / "data" / "HALT"


def leg_halt_path(instance: str) -> Path:
    """This leg's SELF-halt flag (`data/HALT_<instance>`). Written by the leg's
    own kill switch. It halts ONLY this leg — never a sibling."""
    return REPO_ROOT / "data" / f"HALT_{instance}"


def halt_source(instance: str | None = None) -> str | None:
    """Which HALT flag (if any) is active for this leg.

    Returns:
      - "global"  → the shared data/HALT exists (manual stop-all).
      - instance  → this leg's data/HALT_<instance> exists (self-halt).
      - None      → no halt applies.

    Global is checked first so a stop-all is reported as global even if the
    leg also self-halted.
    """
    if global_halt_path().exists():
        return "global"
    if instance and leg_halt_path(instance).exists():
        return instance
    return None


def is_halted(instance: str | None = None) -> bool:
    """True if the bot must close all + exit clean.

    A leg is halted when the GLOBAL data/HALT exists OR its own
    data/HALT_<instance> exists. Passing no instance checks only the global
    flag (used by generic tools that have no leg context). A leg's self-halt
    can never stop another leg — that isolation is the whole point of the
    per-leg scheme (prevents the 07-01 cascade where one leg's kill switch
    touched the shared HALT and took every leg down)."""
    return halt_source(instance) is not None


if __name__ == "__main__":
    print(get_env())
