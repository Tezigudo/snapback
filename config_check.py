"""Boot-time params validation: refuse to trade on an incomplete config.

The per-leg params YAMLs (config/params*.yaml) are NOT tracked in git — only
their `*.example.yaml` skeletons are, with every strategy value null. A leg
booting on a missing / skeleton / truncated file must refuse loudly rather than
silently fall back to some code default, because a "default" for a live
trading parameter is just a different strategy nobody signed off.

REQUIRED_KEYS lists, per strategy_name, every dotted params path the live code
for that strategy reads (bot.py, bot_internals.py, strategy/live_*.py). A key
counts as missing when it is absent OR null. Falsy values (0, 0.0, false, "")
are valid: sol's `max_hold_bars: 0` and `hedge.enabled: false` are live config.

Per-strategy SAFETY blocks are required where the live leg runs them, so a
missing or typo'd block can never silently switch the feature off:
  multifactor-v1: reprotect.{enabled,observe_only,max_replaces_per_position},
                  boot_resume.{enabled,observe_only}
  supertrend:     breakeven.{enabled,at_r,buffer_pct}
Elsewhere these blocks stay optional (absent == off, by design): breakeven on
v1/donchian, reprotect/boot_resume on donchian/sol, no_stop everywhere,
strategy.dedup_bars, reprotect.max_sl_place_failures. A *present and enabled*
breakeven block always requires its tuning keys (CONDITIONAL_KEYS).

Beyond presence: BOOL_KEYS must be a real YAML bool (not "true", 1 or null) and
RANGE_CHECKS bound the few safety-critical numerics (finite; kill fraction in
(0,1), leverage 1..125, risk > 0, poll in (0, 3600], breakeven at_r > 0, and
every risk.* hard ceiling) — a placeholder 0 must not read as "CONFIG OK".

The `risk:` block (risk.RISK_KEYS) holds the hard ceilings that used to be
constants in risk.py; it is required on every leg. Cross-checks against the
leg's own config: sizing.leverage <= risk.max_leverage, and
risk.max_notional_usd >= the symbol's STATIC fallback minimum order
(exchange.constraints; the live exchange minimum is only known at boot).

Pure module: no exchange client, env or network access (exchange.constraints
is static data only), so `python -m bot
--check-config` can run it anywhere.
"""

from __future__ import annotations

import math
from typing import Any

from exchange.constraints import fallbacks_for_symbol
from risk import RISK_KEYS

# Exit status for "config refused". 78 = EX_CONFIG (sysexits.h). The systemd
# units list it in RestartPreventExitStatus so a refused boot stops instead of
# crash-looping (and re-alerting) every RestartSec.
EXIT_CONFIG = 78

_MISSING = object()

# Read by bot.Bot.__init__ / main loop for every leg.
COMMON_KEYS: tuple[str, ...] = (
    *RISK_KEYS,   # risk.py hard ceilings: every leg, no code defaults (RISK_REVIEW 2026-10-22)
    "symbol",
    "strategy_name",
    "timeframes.entry",
    "sizing.leverage",
    "sizing.risk_per_trade_pct",
    "deploy.kill_switch_equity_fraction",
    "deploy.min_capital_warn_usdt",
    "execution.order_type",
    "execution.limit_offset_bps",
    "execution.limit_timeout_s",
    "execution.poll_interval_s",
    "hedge.enabled",
    "hedge.client_order_id_prefix",
    "strategy.max_hold_bars",
)

# Strategy-specific keys: what that strategy's evaluator / exit / gate_status
# read from params["strategy"].
_STRATEGY_KEYS: dict[str, tuple[str, ...]] = {
    "multifactor-v1": (
        "rsi_period",
        "rsi_long_threshold",
        "rsi_short_threshold",
        "mf_trend_ema_period",
        "volume_ma_period",
        "volume_multiple",
        "require_funding_not_extreme",
        "funding_extreme_threshold",
        "use_mtf_4h_gate",
        "mtf_4h_ema_period",
        "require_trend",
        "sl_pct",
        "tp_pct",
    ),
    "donchian-v3": (
        "donchian_period_entry",
        "donchian_period_exit",
        "atr_period",
        "atr_sl_multiple",
        "regime_ema_period",
        "regime_slope_window",
        "slope_trend_threshold_pct",
    ),
    "supertrend": (
        "st_period",
        "st_multiplier",
        "st_atr_period",
        "st_sl_atr",
        "st_tp_atr",
        "allow_shorts",
    ),
    "cnh-hybrid-short-v1": (
        "uptrend_bars",
        "chop_bars",
        "min_rise_pct",
        "max_chop_ratio",
        "require_chop_at_top",
        "breakdown_mode",
        "cup_len",
        "handle_len",
        "min_r2",
        "min_cup_depth_atr",
        "handle_max_depth_frac",
        "peak_tolerance",
        "entry_max_bars_after_handle",
        "entry_emas",
        "sl_atr_mult",
        "tp_emas",
        "atr_period",
    ),
    "v3-all-wider-4": (
        "rsi_period",
        "rsi_long_threshold",
        "rsi_short_threshold",
        "mf_trend_ema_period",
        "volume_ma_period",
        "volume_multiple",
        "require_funding_not_extreme",
        "funding_extreme_threshold",
    ),
}

# Safety blocks the live leg of that strategy runs (see module docstring).
_SAFETY_KEYS: dict[str, tuple[str, ...]] = {
    "multifactor-v1": (
        "reprotect.enabled",
        "reprotect.observe_only",
        "reprotect.max_replaces_per_position",
        "boot_resume.enabled",
        "boot_resume.observe_only",
    ),
    "supertrend": (
        "breakeven.enabled",
        "breakeven.at_r",
        "breakeven.buffer_pct",
    ),
}

REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    name: (COMMON_KEYS + tuple(f"strategy.{k}" for k in keys)
           + _SAFETY_KEYS.get(name, ()))
    for name, keys in _STRATEGY_KEYS.items()
}

# (gate path, gate value) -> keys required when the gate holds. The gate is a
# BOOL_KEY, so `is True` here and the bot's truthiness agree.
CONDITIONAL_KEYS: tuple[tuple[str, Any, tuple[str, ...]], ...] = (
    ("breakeven.enabled", True, ("breakeven.at_r", "breakeven.buffer_pct")),
)

# Must be a real bool whenever present and non-null (required or optional).
BOOL_KEYS: frozenset[str] = frozenset({
    "hedge.enabled",
    "reprotect.enabled",
    "reprotect.observe_only",
    "boot_resume.enabled",
    "boot_resume.observe_only",
    "breakeven.enabled",
    "strategy.require_funding_not_extreme",
    "strategy.use_mtf_4h_gate",
    "strategy.require_trend",
    "strategy.allow_shorts",
    "strategy.require_chop_at_top",
})

# Safety-critical numerics: (predicate, description). Checked whenever present
# and non-null; must be a FINITE int/float (not bool, not a quoted string, not
# .inf/.nan). Upper bounds catch fat-finger units, not strategy choices.
RANGE_CHECKS: dict[str, tuple[Any, str]] = {
    "deploy.kill_switch_equity_fraction": (lambda v: 0 < v < 1, "in (0, 1)"),
    "sizing.leverage": (lambda v: 1 <= v <= 125 and float(v).is_integer(),
                        "an integer in [1, 125]"),
    "sizing.risk_per_trade_pct": (lambda v: v > 0, "> 0"),
    "execution.poll_interval_s": (lambda v: 0 < v <= 3600, "in (0, 3600]"),
    "breakeven.at_r": (lambda v: v > 0, "> 0"),
    "breakeven.buffer_pct": (lambda v: v >= 0, ">= 0"),
    # risk.py hard ceilings. Bounds are fat-finger guards, not the live values;
    # alongside _cross_check_problems and bot's boot-time fingerprint alert they
    # replace the RISK_REVIEW hook that guarded the old risk.py constants.
    "risk.max_notional_usd": (lambda v: 0 < v <= 1_000_000, "in (0, 1000000]"),
    "risk.max_leverage": (lambda v: 1 <= v <= 125 and float(v).is_integer(),
                          "an integer in [1, 125]"),
    "risk.max_daily_loss_pct": (lambda v: 0 < v < 100, "in (0, 100)"),
    "risk.max_open_positions": (lambda v: 1 <= v <= 100 and float(v).is_integer(),
                                "an integer in [1, 100]"),
    "risk.max_orders_per_minute": (lambda v: 1 <= v <= 1200 and float(v).is_integer(),
                                   "an integer in [1, 1200]"),
    "risk.max_consecutive_losses": (lambda v: 1 <= v <= 1000 and float(v).is_integer(),
                                    "an integer in [1, 1000]"),
    "risk.min_time_between_trades_s": (lambda v: 0 < v <= 86400 and float(v).is_integer(),
                                       "an integer in (0, 86400]"),
}


class ParamsError(Exception):
    """The params dict cannot be traded on. Message lists every problem."""


def _lookup(params: Any, dotted: str) -> Any:
    cur = params
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def config_problems(params: Any) -> list[str]:
    """Every reason `params` is not bootable; [] means OK. Never raises."""
    if not isinstance(params, dict):
        return [f"params is {type(params).__name__}, expected a mapping "
                "(empty or unparsable YAML?)"]
    name = params.get("strategy_name")
    if name is None or name == "":
        return ["strategy_name: missing or null (refusing to guess a strategy)"]
    if name not in REQUIRED_KEYS:
        return [f"strategy_name: unknown {name!r} "
                f"(known: {', '.join(sorted(REQUIRED_KEYS))})"]
    problems: list[str] = []
    required = list(REQUIRED_KEYS[name])
    for gate, want, keys in CONDITIONAL_KEYS:
        if _lookup(params, gate) is want:
            required.extend(keys)
    for key in dict.fromkeys(required):          # de-dup, keep order
        v = _lookup(params, key)
        if v is _MISSING:
            problems.append(f"{key}: missing")
        elif v is None:
            problems.append(f"{key}: null")
    for key in sorted(BOOL_KEYS):
        v = _lookup(params, key)
        if v is not _MISSING and v is not None and not isinstance(v, bool):
            problems.append(f"{key}: must be true/false, got {v!r}")
    for key, (ok, desc) in RANGE_CHECKS.items():
        v = _lookup(params, key)
        if v is _MISSING or v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            problems.append(f"{key}: must be a finite number {desc}, got {v!r}")
        elif not ok(v):
            problems.append(f"{key}: must be {desc}, got {v!r}")
    problems.extend(_cross_check_problems(params, problems))
    return problems


def _cross_check_problems(params: dict, problems: list[str]) -> list[str]:
    """risk.* ceilings vs the leg's OWN config. The public RANGE_CHECKS bounds
    stay generic on purpose (tightening them toward the live values would
    publish them); these checks catch a ceiling that contradicts the leg it
    guards. Skipped for a key that already has a presence/range problem."""
    flagged = {p.split(":", 1)[0] for p in problems}
    out: list[str] = []
    lev, max_lev = _lookup(params, "sizing.leverage"), _lookup(params, "risk.max_leverage")
    if not flagged & {"sizing.leverage", "risk.max_leverage"} \
            and isinstance(lev, (int, float)) and isinstance(max_lev, (int, float)) \
            and lev > max_lev:
        # Bot.boot() check_leverage would raise anyway; refuse at config time
        # so --check-config sees it and boot exits 78 instead of crashing.
        out.append(f"risk.max_leverage: below this leg's sizing.leverage "
                   f"({max_lev!r} < {lev!r})")
    cap, symbol = _lookup(params, "risk.max_notional_usd"), _lookup(params, "symbol")
    if "risk.max_notional_usd" not in flagged and isinstance(cap, (int, float)) \
            and isinstance(symbol, str) and symbol:
        # STATIC fallback minimum only (no network here). Boot merges in the
        # live exchange limits (merge_with_live, tighter wins), so a cap above
        # this floor but below the live minimum still passes CONFIG OK — and
        # then check_notional skips every signal. A sanity floor, not a proof.
        floor = fallbacks_for_symbol(symbol).min_notional_usdt
        if cap < floor:
            out.append(f"risk.max_notional_usd: below the static fallback minimum "
                       f"order for {symbol} ({cap!r} < {floor!r}); boot also "
                       f"enforces the live exchange minimum")
    return out


def validate_params(params: Any, source: str = "params") -> None:
    """Raise ParamsError naming every missing/null required key."""
    problems = config_problems(params)
    if problems:
        raise ParamsError(
            f"{source}: refusing to boot — {len(problems)} required param(s) "
            "missing or null (no code defaults are used for these):\n  "
            + "\n  ".join(problems)
            + "\nCreate/fix the file from its config/*.example.yaml "
              "(see config/README.md).")
