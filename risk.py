"""
Hard risk ceilings. DO NOT EDIT without git env RISK_REVIEW=1.

These limits are enforced at boot (symbol, leverage), before every entry
(notional) and on every tick (daily loss). The bot crashes loudly if a boot
check fails.

WHERE THE NUMBERS LIVE (RISK_REVIEW 2026-10-22): the numeric ceilings are NOT in
this file. They are read from the `risk:` block of the leg's params YAML
(config/params*.yaml, untracked — see config/README.md), so the public repo does
not publish them. There are NO code defaults: config_check.REQUIRED_KEYS lists
every `risk.*` key and RANGE_CHECKS bounds them, so a missing / null /
out-of-range ceiling refuses boot (exit 78) — never a fallback. bot.main()
installs them with `configure(params)` after validation; reading the ceilings
before that raises RiskNotConfigured (a crash, never a silent pass).

Consequence of the move: the ceilings are now edited in the params YAML, which
the RISK_REVIEW pre-commit hook does not guard. Mechanical backstops instead:
config_check's generic range bounds and cross-checks against the leg's own
config (sizing.leverage <= max_leverage; max_notional_usd >= the static
fallback minimum order), plus a boot-time fingerprint that alerts when the
ceilings changed since the last boot (bot._note_risk_ceilings_change). Treat
any `risk:` edit as a risk review.

The symbol allowlist stays in code: it is not a sizing number, and keeping it
code-reviewed is what stops a params edit from enabling a new market.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# dataclass field -> (params key under `risk:`, python type the code used).
# The types matter: the RiskBreach messages format these values, so an int
# ceiling read back as a float would change log/alert text ("Nx" -> "N.0x").
RISK_FIELDS: dict[str, tuple[str, type]] = {
    "MAX_NOTIONAL_USD": ("max_notional_usd", float),
    "MAX_LEVERAGE": ("max_leverage", int),
    "MAX_DAILY_LOSS_PCT": ("max_daily_loss_pct", float),
    "MAX_OPEN_POSITIONS": ("max_open_positions", int),
    "MAX_ORDERS_PER_MINUTE": ("max_orders_per_minute", int),
    "MAX_CONSECUTIVE_LOSSES": ("max_consecutive_losses", int),
    "MIN_TIME_BETWEEN_TRADES_S": ("min_time_between_trades_s", int),
}

# Dotted params paths, for config_check.REQUIRED_KEYS (every leg).
RISK_KEYS: tuple[str, ...] = tuple(f"risk.{k}" for k, _ in RISK_FIELDS.values())


@dataclass(frozen=True)
class RiskCeilings:
    # Absolute notional cap per single order, in USDT. ENFORCED: check_notional
    # before every entry; an oversize signal is skipped, not resized.
    MAX_NOTIONAL_USD: float | None = None

    # Hard leverage cap. Bot may request less, never more. ENFORCED: check_leverage
    # in Bot.boot() against sizing.leverage. At high leverage the liquidation
    # distance is a few percent and flash crashes can gap past SLs, so this
    # ceiling is load-bearing for tail-risk control — DO NOT raise it without a
    # separate RISK_REVIEW.
    MAX_LEVERAGE: int | None = None

    # If equity drawdown since this UTC day's start reaches this %, block NEW
    # entries for the rest of the day (bot._daily_loss_blocks_entry). ENFORCED
    # via check_daily_loss. Does NOT flatten or HALT — open positions keep their
    # exchange-native brackets, and the daily anchor resets at the next UTC
    # midnight. The tighter, daily-resetting sibling of the cumulative
    # kill-switch. Sizing rule (RISK_REVIEW 2026-07-12): it must exceed one full
    # stop-loss (sizing.risk_per_trade_pct + slippage) or a single normal SL
    # freezes the leg for the day, and should still halt before a second
    # consecutive one — so it moves in lockstep with risk_per_trade_pct.
    MAX_DAILY_LOSS_PCT: float | None = None

    # UNENFORCED - informational only (printed by tools/preflight_live.py).
    # Nothing caps open positions; each leg trades one symbol, one-way.
    MAX_OPEN_POSITIONS: int | None = None

    # UNENFORCED - informational only. Nothing in the bot reads this value;
    # there is no order-rate limiter.
    MAX_ORDERS_PER_MINUTE: int | None = None

    # UNENFORCED - informational only. Nothing in the bot reads this value; it
    # does NOT trip any kill-switch. Decided 2026-09-29 (E1/E2) not to wire it
    # as a trading breaker: see docs/CONSECUTIVE_LOSS_BREAKER_PLAN.md. Streaks
    # are only alerted on, in monitor.py (streak_alert_thresholds).
    MAX_CONSECUTIVE_LOSSES: int | None = None

    # UNENFORCED - informational only. Nothing in the bot reads this value;
    # there is no minimum spacing between trades beyond the strategy's bar logic.
    MIN_TIME_BETWEEN_TRADES_S: int | None = None

    # Hard symbol allowlist. ENFORCED: check_symbol in Bot.boot().
    # BTC: v1 (multifactor) + donchian-v3 legs.
    # SOL: added 2026-07-25 (RISK_REVIEW, user-approved) for the sol_supertrend
    #      leg — round-3 win-rate-blended walk-forward winner, 9/9 OOS folds
    #      positive, ~uncorrelated with both BTC legs (-0.02 vs multifactor-v1,
    #      +0.06 vs donchian-v3). Evidence in SOL_LEG_VERDICT.md; live↔backtest
    #      parity in tools/supertrend_parity.py (119/119 entries, 0 spurious).
    #      NOTE: this also un-blocks config/params_cnh_hybrid_short_sol.yaml,
    #      which measured CAGR 1.3% with 1,360 days underwater — that unit must
    #      NOT be enabled. The allowlist is no longer what prevents it.
    #      MAX_NOTIONAL_USD stays the per-order backstop for both symbols.
    ALLOWED_SYMBOLS: tuple[str, ...] = ("BTC/USDT:USDT", "SOL/USDT:USDT")


class RiskBreach(Exception):
    """Raised when an action would violate a hard ceiling."""


class RiskNotConfigured(RuntimeError):
    """The ceilings were read before configure(). Deliberately NOT a RiskBreach:
    callers that catch RiskBreach to skip a signal must not swallow this."""


def ceilings_from_params(params: Any) -> RiskCeilings:
    """Build the ceilings from a params dict's `risk:` block. No defaults: a
    missing / null / non-numeric value raises ValueError naming every bad key.
    Range checks live in config_check.RANGE_CHECKS (run first, at boot)."""
    block = params.get("risk") if isinstance(params, dict) else None
    if not isinstance(block, dict):
        raise ValueError("params has no `risk:` mapping "
                         "(run tools/migrate_risk_params.py)")
    values: dict[str, Any] = {}
    bad: list[str] = []
    for field, (key, typ) in RISK_FIELDS.items():
        v = block.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            bad.append(f"risk.{key}={v!r}")
        elif typ is int and not float(v).is_integer():
            bad.append(f"risk.{key}={v!r} (must be an integer)")
        else:
            values[field] = typ(v)
    if bad:
        raise ValueError("risk ceilings missing/invalid: " + ", ".join(bad))
    return RiskCeilings(**values)


_ACTIVE: RiskCeilings | None = None


def configure(params: Any) -> RiskCeilings:
    """Install this process's ceilings from its (validated) params. One leg per
    process, so the module-level value is that leg's value."""
    global _ACTIVE
    _ACTIVE = ceilings_from_params(params)
    return _ACTIVE


def ceilings() -> RiskCeilings:
    if _ACTIVE is None:
        raise RiskNotConfigured("risk ceilings not configured — risk.configure(params) "
                                "must run before any risk check")
    return _ACTIVE


def check_notional(notional_usd: float) -> None:
    c = ceilings()
    if notional_usd > c.MAX_NOTIONAL_USD:
        raise RiskBreach(
            f"notional {notional_usd:.2f} USDT exceeds MAX_NOTIONAL_USD={c.MAX_NOTIONAL_USD}"
        )


def check_leverage(requested: int) -> None:
    c = ceilings()
    if requested > c.MAX_LEVERAGE:
        raise RiskBreach(
            f"leverage {requested}x exceeds MAX_LEVERAGE={c.MAX_LEVERAGE}x"
        )


def check_symbol(symbol: str) -> None:
    c = ceilings()
    if symbol not in c.ALLOWED_SYMBOLS:
        raise RiskBreach(
            f"symbol {symbol!r} not in ALLOWED_SYMBOLS={c.ALLOWED_SYMBOLS}"
        )


def check_daily_loss(equity_now: float, equity_day_start: float) -> None:
    c = ceilings()
    if equity_day_start <= 0:
        return
    loss_pct = (equity_day_start - equity_now) / equity_day_start * 100.0
    if loss_pct >= c.MAX_DAILY_LOSS_PCT:
        raise RiskBreach(
            f"daily loss {loss_pct:.2f}% reached MAX_DAILY_LOSS_PCT={c.MAX_DAILY_LOSS_PCT}%"
        )
