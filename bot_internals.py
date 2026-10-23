"""Pure-function helpers used by bot.py — extracted to keep the trading
loop focused on orchestration rather than dispatch and arithmetic.

No side effects: nothing here touches the exchange, state.db, alerts, or
consolidate. The Bot class calls these to make decisions, then handles
the I/O itself.
"""

from __future__ import annotations

import contextlib
import importlib
import json
from dataclasses import dataclass

import pandas as pd

# channel_exit_signal is re-exported here on purpose: bot.py imports it from
# bot_internals (byte-identical between main and droplet) rather than adding a
# new import to the heavily-diverging exchange/env/risk import block — keeps the
# eventual droplet cherry-pick's bot.py diff inside non-diverging regions.
from strategy.live_donchian_v3 import (  # noqa: F401  (channel_exit_signal re-exported for bot.py)
    channel_exit_signal,
    evaluate_signal_donchian_v3,
)
from strategy.live_multifactor_v1 import evaluate_signal, trend_exit_signal_multifactor_v1
from strategy.live_supertrend import evaluate_signal_supertrend, flip_exit_signal


def _optional_evaluator(module: str, attr: str):
    """Import a strategy evaluator that not every checkout ships.

    `v3-all-wider-4` and `cnh-hybrid-short-v1` are imported here, on first use,
    instead of at module top, so this module (and bot.py) import cleanly in a
    checkout that omits them (the public snapshot). Where the module exists the
    first call imports it and later calls are a sys.modules lookup, so the
    evaluator and its output are unchanged. Only the module's OWN absence is
    translated; an ImportError raised from inside it propagates as-is.
    """
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name != module:
            raise
        raise ModuleNotFoundError(
            f"strategy evaluator {module}.{attr} is not in this checkout "
            f"(module {module!r} missing); this strategy cannot run here",
            name=module) from e
    return getattr(mod, attr)


# strategy_name -> (module, evaluator) for the evaluators imported on first use.
_OPTIONAL_EVALUATORS: dict[str, tuple[str, str]] = {
    "v3-all-wider-4": ("strategy.live_v3all_wider4", "evaluate_signal_v3all_wider4"),
    "cnh-hybrid-short-v1": ("strategy.live_cnh_hybrid_short",
                            "evaluate_signal_cnh_hybrid_short"),
}


def require_strategy_available(strategy_name: str) -> None:
    """Boot-time probe: import the configured strategy's evaluator NOW if it is
    one of the lazily-imported ones, so a checkout without it refuses to boot
    (bot._main -> _refuse_boot, exit 78) instead of failing on the first bar
    inside the loop's generic error backoff. A no-op for every other strategy
    (their evaluators are imported at module top)."""
    if strategy_name in _OPTIONAL_EVALUATORS:
        _optional_evaluator(*_OPTIONAL_EVALUATORS[strategy_name])


def strategy_uses_channel_exit(strategy_name: str) -> bool:
    """True for strategies that omit the TP bracket leg because a live trend
    exit IS their profit-taking mechanism. Currently only donchian-v3.

    NOT the same question as "does this leg run a trend-exit check each tick" —
    that is `strategy_uses_trend_exit`. supertrend keeps its TP bracket AND runs
    a trend exit, so it is deliberately absent here.

    Every other strategy (v1/multifactor, cnh, v3all) is untouched: it keeps its
    TP bracket.
    """
    return strategy_name == "donchian-v3"


def strategy_uses_trend_exit(strategy_name: str) -> bool:
    """True for strategies whose loop must run a trend-exit check each tick.

    - donchian-v3: Donchian channel cross — its ONLY profit-taking mechanism.
    - supertrend: opposite STDir flip, which closes the position even though a
      TP bracket also exists. Both exits are live at once.
    - multifactor-v1: adverse 15m EMA(200) cross, gated on `require_trend`.
      Added 2026-08-10 — every v1 sign-off measured the model WITH this exit,
      but it had never run live. See MULTIFACTOR_V1_LIVE_EXIT_VERDICT.md and
      strategy.live_multifactor_v1.trend_exit_signal_multifactor_v1.

    Kept separate from `strategy_uses_channel_exit` so donchian's TP-omission
    behaviour is unchanged by adding a leg that needs the hook but keeps its TP.
    v1, like supertrend, KEEPS its TP bracket — it is deliberately absent there.
    """
    return strategy_name in ("donchian-v3", "supertrend", "multifactor-v1")


def trend_exit_signal(
    strategy_name: str,
    bars: pd.DataFrame,
    position_side: str,
    params: dict,
) -> tuple[bool, dict]:
    """Dispatch the per-strategy trend exit. Returns (should_exit, debug).

    Thin dispatcher so bot._maybe_trend_exit has one callsite, and the donchian
    path is reached by exactly the same call as before this leg was added.
    """
    if strategy_name == "supertrend":
        return flip_exit_signal(bars, position_side, params)
    if strategy_name == "multifactor-v1":
        return trend_exit_signal_multifactor_v1(bars, position_side, params)
    return channel_exit_signal(bars, position_side, params)


def trend_exit_fill_reason(strategy_name: str) -> str:
    """The `reason` string recorded on the fill/telemetry when the trend-exit
    hook fires.

    donchian-v3 and supertrend keep "channel_exit" — that value is already live
    in the fills table, the consolidate dashboard and its fixtures, and renaming
    it would orphan the history. v1's exit is a different mechanism (EMA cross,
    not a channel), so it gets its own value rather than lying about which rule
    closed the trade. Consumer-side `exitReason` is a pass-through string with
    no enum validation, so a new value renders as-is.
    """
    return "trend_exit" if strategy_name == "multifactor-v1" else "channel_exit"


def resolve_strategy_name(params: dict) -> str:
    """Default to multifactor-v1 if `strategy_name` is missing/empty/None.

    Five callsites in bot.py used to spell this `self.params.get(...) or "..."` —
    one place to look now.
    """
    name = params.get("strategy_name")
    return str(name) if name else "multifactor-v1"


def limit_entry_price(side: str, close_price: float, offset_bps: float) -> float:
    """Maker-style limit price: place BELOW close for long buys, ABOVE for short
    sells, so the fill (if it happens) earns the maker rebate.

    offset_bps=0 means "at close". Positive offsets push further from close
    (lower fill probability, better price if filled).
    """
    offset = offset_bps / 10000.0
    return close_price * (1.0 - offset) if side == "long" else close_price * (1.0 + offset)


@dataclass(frozen=True)
class SignalDecision:
    """What a strategy evaluator returned: the side to trade (or None), the
    price it observed, and the SL/TP distances in absolute price units.

    sl_price / tp_price are computed on demand — only meaningful when
    `side` is non-None.
    """

    side: str | None
    price: float
    sl_distance: float
    tp_distance: float
    debug: dict

    @property
    def sl_price(self) -> float:
        if self.side == "long":
            return self.price - self.sl_distance
        return self.price + self.sl_distance

    @property
    def tp_price(self) -> float:
        if self.side == "long":
            return self.price + self.tp_distance
        return self.price - self.tp_distance


def evaluate_for_strategy(
    strategy_name: str,
    bars_15m: pd.DataFrame,
    funding_rate: float,
    params: dict,
) -> SignalDecision:
    """Dispatch to the live signal evaluator for the configured strategy.

    Three paths today:
      - "v3-all-wider-4": evaluator returns (side, sl_dist, tp_dist, dbg) — SL/TP
        are already in price units from ATR×k math.
      - "donchian-v3": Donchian-cons breakout on 4h bars. Returns (side, sl_dist,
        tp_dist, dbg). SL = atr_sl_multiple×ATR. The entry places NO TP leg — the live
        Donchian channel cross (bot._maybe_channel_exit / channel_exit_signal)
        closes the trade; tp_dist here is advisory telemetry only.
      - default ("multifactor-v1"): evaluator returns (side, dbg); SL/TP come
        from fixed-pct multipliers in params (`sl_pct`, `tp_pct`) applied to
        the close price.

    The `bars_15m` argument name is historical — for donchian-v3 the bot passes
    4h bars in this slot (entry timeframe from config). Each strategy reads
    whatever its config says to.
    """
    fallback_price = float(bars_15m["Close"].iloc[-1])

    if strategy_name == "v3-all-wider-4":
        evaluate_signal_v3all_wider4 = _optional_evaluator(
            *_OPTIONAL_EVALUATORS["v3-all-wider-4"])
        side, sl_dist, tp_dist, dbg = evaluate_signal_v3all_wider4(
            bars_15m, funding_rate, params)
        price = (dbg.get("cur_close", fallback_price)
                 if isinstance(dbg, dict) else fallback_price)
        return SignalDecision(
            side=side, price=price,
            sl_distance=float(sl_dist), tp_distance=float(tp_dist),
            debug=dbg if isinstance(dbg, dict) else {},
        )

    if strategy_name == "donchian-v3":
        side, sl_dist, tp_dist, dbg = evaluate_signal_donchian_v3(
            bars_15m, funding_rate, params)
        price = (dbg.get("cur_close", fallback_price)
                 if isinstance(dbg, dict) else fallback_price)
        return SignalDecision(
            side=side, price=price,
            sl_distance=float(sl_dist), tp_distance=float(tp_dist),
            debug=dbg if isinstance(dbg, dict) else {},
        )

    if strategy_name == "supertrend":
        # Supertrend flip on native 4h. SL and TP are both real bracket legs
        # (unlike donchian, whose tp_dist is advisory only), and the opposite
        # flip closes the position on top of them via trend_exit_signal.
        side, sl_dist, tp_dist, dbg = evaluate_signal_supertrend(
            bars_15m, funding_rate, params)
        price = (dbg.get("cur_close", fallback_price)
                 if isinstance(dbg, dict) else fallback_price)
        return SignalDecision(
            side=side, price=price,
            sl_distance=float(sl_dist), tp_distance=float(tp_dist),
            debug=dbg if isinstance(dbg, dict) else {},
        )

    if strategy_name == "cnh-hybrid-short-v1":
        # HYBRID short pattern detector on 4h. The bot passes 4h bars in the
        # bars_15m slot (entry timeframe = 4h per params YAML). Returns
        # (side, sl_dist, tp_dist, dbg) — SL = sl_atr_mult × ATR(14, 4h),
        # TP = distance from entry to the configured EMA (default EMA100).
        evaluate_signal_cnh_hybrid_short = _optional_evaluator(
            *_OPTIONAL_EVALUATORS["cnh-hybrid-short-v1"])
        side, sl_dist, tp_dist, dbg = evaluate_signal_cnh_hybrid_short(
            bars_15m, funding_rate, params)
        price = (dbg.get("close", fallback_price)
                 if isinstance(dbg, dict) else fallback_price)
        return SignalDecision(
            side=side, price=price,
            sl_distance=float(sl_dist), tp_distance=float(tp_dist),
            debug=dbg if isinstance(dbg, dict) else {},
        )

    # multifactor-v1 (and any future fixed-pct variant)
    side, dbg = evaluate_signal(bars_15m, funding_rate, params)
    price = (dbg.get("cur_close", fallback_price)
             if isinstance(dbg, dict) else fallback_price)
    sl_pct = float(params["strategy"]["sl_pct"])
    tp_pct = float(params["strategy"]["tp_pct"])
    return SignalDecision(
        side=side, price=price,
        sl_distance=sl_pct * price, tp_distance=tp_pct * price,
        debug=dbg if isinstance(dbg, dict) else {},
    )


def gate_status(strategy_name: str, decision: SignalDecision, params: dict) -> dict:
    """Build a structured 'what's true now, what are we waiting for' snapshot
    from a strategy evaluator's debug output. The bot logs this on every bar
    evaluation and includes it in heartbeat-event payloads pushed to consolidate,
    so the dashboard can answer 'why isn't this firing?' without you SSHing
    into the droplet.

    Returns a dict with stable JSON-serializable shape:
      {
        "strategy": "<name>",
        "would_fire": "long" | "short" | None,
        "values":  {<indicator name>: <numeric>},
        "thresholds": {<threshold name>: <numeric>},
        "gates_long":  {<gate name>: <bool>},
        "gates_short": {<gate name>: <bool>},
        "missing_long":  [<gate name>, ...],
        "missing_short": [<gate name>, ...],
        "waiting_for": "<human-readable summary>",
      }
    """
    dbg = decision.debug or {}
    s = params.get("strategy", {}) if isinstance(params, dict) else {}

    if strategy_name == "multifactor-v1":
        rsi = dbg.get("rsi")
        close = dbg.get("cur_close")
        ema = dbg.get("trend_ema")
        vol_sma = dbg.get("vol_sma")
        cur_vol = dbg.get("cur_vol")
        funding = dbg.get("funding_rate")

        # Same keys the evaluator already read strictly (no fallbacks; enforced
        # at boot by config_check.REQUIRED_KEYS).
        rsi_lt_long = float(s["rsi_long_threshold"])
        rsi_gt_short = float(s["rsi_short_threshold"])
        vol_mult = float(s["volume_multiple"])
        funding_extreme = float(s["funding_extreme_threshold"])
        require_funding = bool(s["require_funding_not_extreme"])

        def _safe_lt(a, b):
            return a is not None and b is not None and a < b
        def _safe_gt(a, b):
            return a is not None and b is not None and a > b

        gates_long = {
            "rsi_oversold":  _safe_lt(rsi, rsi_lt_long),
            "trend_up":      _safe_gt(close, ema),
            "volume_spike":  _safe_gt(cur_vol, vol_mult * vol_sma) if vol_sma else False,
            "funding_ok":    (not require_funding) or (funding is not None and funding <= funding_extreme),
        }
        gates_short = {
            "rsi_overbought": _safe_gt(rsi, rsi_gt_short),
            "trend_down":     _safe_lt(close, ema),
            "volume_spike":   gates_long["volume_spike"],  # same volume rule
            "funding_ok":     (not require_funding) or (funding is not None and funding >= -funding_extreme),
        }
        missing_long  = [k for k, v in gates_long.items()  if not v]
        missing_short = [k for k, v in gates_short.items() if not v]
        vol_ratio = (cur_vol / vol_sma) if (cur_vol is not None and vol_sma) else None
        return {
            "strategy": strategy_name,
            "would_fire": decision.side,
            "values": {
                "rsi":          float(rsi) if rsi is not None else None,
                "close":        float(close) if close is not None else None,
                "ema200":       float(ema) if ema is not None else None,
                "vol_ratio":    float(vol_ratio) if vol_ratio is not None else None,
                "funding_rate": float(funding) if funding is not None else None,
            },
            "thresholds": {
                "rsi_long_lt":   rsi_lt_long,
                "rsi_short_gt":  rsi_gt_short,
                "vol_multiple":  vol_mult,
                "funding_extreme": funding_extreme,
            },
            "gates_long":  gates_long,
            "gates_short": gates_short,
            "missing_long":  missing_long,
            "missing_short": missing_short,
            "waiting_for": _format_waiting(missing_long, missing_short, decision.side),
        }

    if strategy_name == "donchian-v3":
        close = dbg.get("cur_close")
        upper = dbg.get("upper")
        lower = dbg.get("lower")
        slope = dbg.get("slope")
        slope_thr = (dbg.get("slope_threshold")
                     if dbg.get("slope_threshold") is not None
                     else float(s["slope_trend_threshold_pct"]))
        gate_on = bool(dbg.get("gate_on", True))

        breakout_ok = close is not None and upper is not None and close > upper
        breakdown_ok = close is not None and lower is not None and close < lower
        slope_long_ok  = (not gate_on) or (slope is not None and slope >=  slope_thr)
        slope_short_ok = (not gate_on) or (slope is not None and slope <= -slope_thr)

        gates_long  = {"breakout_above_80bar": breakout_ok,  "slope_up":   slope_long_ok}
        gates_short = {"breakdown_below_80bar": breakdown_ok, "slope_down": slope_short_ok}
        missing_long  = [k for k, v in gates_long.items()  if not v]
        missing_short = [k for k, v in gates_short.items() if not v]
        return {
            "strategy": strategy_name,
            "would_fire": decision.side,
            "values": {
                "close":      float(close) if close is not None else None,
                "upper_80bar": float(upper) if upper is not None else None,
                "lower_80bar": float(lower) if lower is not None else None,
                "slope":       float(slope) if slope is not None else None,
            },
            "thresholds": {
                "slope_threshold": float(slope_thr),
                "gate_on":         gate_on,
            },
            "gates_long":  gates_long,
            "gates_short": gates_short,
            "missing_long":  missing_long,
            "missing_short": missing_short,
            "waiting_for": _format_waiting(missing_long, missing_short, decision.side),
        }

    if strategy_name == "supertrend":
        close = dbg.get("cur_close")
        dir_now = dbg.get("st_dir")
        dir_prev = dbg.get("st_dir_prev")
        allow_shorts = bool(dbg.get("allow_shorts", True))

        flip_long = dir_prev == -1.0 and dir_now == 1.0
        flip_short = dir_prev == 1.0 and dir_now == -1.0
        # `shorts_enabled` deliberately does NOT live in gates_short. It is
        # static config, so as a gate it would read ✓ forever and dilute a list
        # whose whole job is "what is changing / what am I waiting on". It goes
        # in thresholds alongside the other config, matching how donchian-v3
        # surfaces its own on/off switch (`gate_on`) above. gates_short now
        # contains only the one thing that actually moves.
        gates_long = {"st_flip_up": bool(flip_long)}
        gates_short = {"st_flip_down": bool(flip_short and allow_shorts)}
        missing_long = [k for k, v in gates_long.items() if not v]
        missing_short = [k for k, v in gates_short.items() if not v]

        def _num(key):
            v = dbg.get(key)
            return float(v) if isinstance(v, (int, float)) else None

        def _dbg_or_cfg(dbg_key, cfg_key):
            # Prefer what the evaluator actually used; else the config value.
            # No literal fallback — the key is boot-validated.
            return dbg[dbg_key] if dbg_key in dbg else s[cfg_key]

        return {
            "strategy": strategy_name,
            "would_fire": decision.side,
            "values": {
                "close": float(close) if close is not None else None,
                "st_dir": float(dir_now) if dir_now is not None else None,
                "st_dir_prev": float(dir_prev) if dir_prev is not None else None,
                # The band + distance to it: the level price must cross for the
                # next flip. Without this the card can only say "direction is
                # -1"; with it, "needs +4.1%".
                "st_line": _num("st_line"),
                "dist_to_flip_pct": _num("dist_to_flip_pct"),
                "atr": _num("atr"),
                "atr_pct": _num("atr_pct"),
                "bars_since_flip": _num("bars_since_flip"),
                # What a LONG firing on this bar would be bracketed at, so the
                # reader does not have to do ATR arithmetic.
                "would_sl_price": _num("would_sl_price"),
                "would_tp_price": _num("would_tp_price"),
            },
            "thresholds": {
                "st_period": float(_dbg_or_cfg("st_period", "st_period")),
                "st_multiplier": float(_dbg_or_cfg("st_multiplier", "st_multiplier")),
                "sl_atr": float(_dbg_or_cfg("sl_atr", "st_sl_atr")),
                "tp_atr": float(_dbg_or_cfg("tp_atr", "st_tp_atr")),
                "shorts_enabled": allow_shorts,
            },
            "gates_long": gates_long,
            "gates_short": gates_short,
            "missing_long": missing_long,
            "missing_short": missing_short,
            "waiting_for": _format_waiting(missing_long, missing_short, decision.side),
        }

    if strategy_name == "cnh-hybrid-short-v1":
        close = dbg.get("close")
        ema24 = dbg.get("ema24")
        ema100 = dbg.get("ema100")
        atr_v = dbg.get("atr")
        last_admitted = dbg.get("last_admitted_pattern")
        pattern_fired = dbg.get("pattern")   # "DT" | "ICNH" | None
        cross_down = dbg.get("entry_ema_cross_down")

        # HYBRID is SHORT-only; "would_fire" is short or None.
        tp_slot_ok = (close is not None and ema100 is not None
                      and ema100 < close)
        # "Admission actionable this bar" — true for DT when admission happened
        # at the current bar (the only window DT can fire), and true for ICnH
        # whenever the evaluator chose to fire (admission is in the lookback +
        # cross-down just happened). The earlier `last_admitted.ts == dbg.ts`
        # check was always False for ICnH entries because ICnH admits at the
        # handle-end bar, not the cross-down bar — so the dashboard reported
        # "waiting on pattern_admitted_this_bar" while simultaneously firing.
        gates_short = {
            "pattern_admitted_this_bar": bool(
                last_admitted is not None
                and (
                    last_admitted.get("ts") == dbg.get("ts")
                    or pattern_fired == "ICNH"
                )
            ),
            "tp_slot_below_entry":  tp_slot_ok,
            "icnh_lookback_ema_xd": bool(cross_down) if cross_down is not None else False,
        }
        # The 3rd gate is only meaningful when ICnH is the pending pattern.
        missing_short = [k for k, v in gates_short.items() if not v]
        return {
            "strategy": strategy_name,
            "would_fire": decision.side,
            "values": {
                "close":  float(close) if close is not None else None,
                "ema24":  float(ema24) if ema24 is not None else None,
                "ema100": float(ema100) if ema100 is not None else None,
                "atr":    float(atr_v) if atr_v is not None else None,
                "last_admitted_pattern": last_admitted,
                "pattern_fired": pattern_fired,
            },
            "thresholds": {
                # Telemetry only, read from the params YAML; None when absent
                # (no code-side default for a tunable).
                "sl_atr_mult": (float(s["sl_atr_mult"])
                                if s.get("sl_atr_mult") is not None else None),
                "tp_ema":      (s.get("tp_emas", ["ema100"]) or ["ema100"])[0],
                "dedup_bars":  (int(s["dedup_bars"])
                                if s.get("dedup_bars") is not None else None),
            },
            # HYBRID has no long side — keep gates_long empty for shape compat.
            "gates_long":  {},
            "gates_short": gates_short,
            "missing_long":  [],
            "missing_short": missing_short,
            "waiting_for": _format_waiting([], missing_short, decision.side),
        }

    # Unknown strategy — return minimal envelope so the dashboard doesn't blow up.
    return {
        "strategy": strategy_name,
        "would_fire": decision.side,
        "values": {},
        "thresholds": {},
        "gates_long": {},
        "gates_short": {},
        "missing_long": [],
        "missing_short": [],
        "waiting_for": "no gate introspection available for this strategy",
    }


def _format_waiting(missing_long: list, missing_short: list, fired: str | None) -> str:
    if fired:
        return f"signal fired: {fired}"
    long_part = "long ready" if not missing_long else "long waiting on " + ", ".join(missing_long)
    short_part = "short ready" if not missing_short else "short waiting on " + ", ".join(missing_short)
    return f"{long_part} | {short_part}"


def time_stop_due(max_hold_bars: int, bar_seconds: int, age_s: float) -> bool:
    """True when an open position has outlived its time stop.

    max_hold_bars <= 0 means THERE IS NO TIME STOP — not "close after zero
    bars". supertrend sets it to 0 on purpose (its exit is the STDir flip) and
    config/params_sol_supertrend.yaml labels the field "Unused by supertrend".
    donchian-v3 does NOT: it sets a positive max_hold_bars (N 4h bars) as a
    max-hold safety on top of its channel exit — do not "align" it to 0 to
    match supertrend, that is precisely the edit this docstring exists to
    prevent. Reading 0 as a due time stop made
    `age_s >= 0` true for every position the leg could ever open: on
    2026-08-08 the sol_supertrend leg entered a LONG and was
    flattened 7 seconds later, and would have been on every subsequent entry.

    max_hold_bars counts ENTRY-TF bars, not 15m bars, so the caller passes
    bar_seconds — otherwise a 4h leg time-stops 16x too early.
    """
    if max_hold_bars <= 0 or bar_seconds <= 0:
        return False
    return age_s >= max_hold_bars * bar_seconds


def order_avg_price(order: dict | None) -> float | None:
    """Best-effort average fill price from a ccxt order dict (market fills).
    Returns None if no positive price is present."""
    if not order:
        return None
    info = order.get("info") or {}
    for v in (order.get("average"), info.get("avgPrice"), order.get("price")):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            return f
    return None


def reduce_only_bracket_leg(order: dict) -> str | None:
    """Classify a ccxt order as a reduce-only bracket leg: 'sl' (STOP*), 'tp'
    (TAKE_PROFIT*), or None (not a reduce-only stop/take-profit — e.g. an
    unfilled limit entry). Handles reduceOnly/closePosition as bool or string."""
    info = order.get("info") or {}
    reduce_only = (
        str(info.get("reduceOnly")).lower() == "true"
        or str(info.get("closePosition")).lower() == "true"
    )
    if not reduce_only:
        return None
    otype = str(info.get("type") or info.get("origType") or order.get("type") or "").upper()
    if "TAKE_PROFIT" in otype:
        return "tp"
    if "STOP" in otype:
        return "sl"
    return None


def bracket_is_intact(open_orders: list[dict], place_tp: bool) -> bool:
    """True when the resting reduce-only bracket is complete for an open
    position: a stop (SL) leg is present, plus a take-profit (TP) leg when the
    strategy places one (place_tp). Pure — unit-tested."""
    legs = {reduce_only_bracket_leg(o) for o in (open_orders or [])}
    return "sl" in legs and ("tp" in legs or not place_tp)


def has_bracket_leg(open_orders: list[dict]) -> bool:
    """True if ANY reduce-only bracket leg (SL or TP) is present. Used to verify
    a cancel actually cleared the bracket before re-placing (cancel_open_orders
    swallows per-order failures, so a partial cancel could otherwise leave a
    surviving leg alongside a freshly-placed pair)."""
    return any(reduce_only_bracket_leg(o) is not None for o in (open_orders or []))


def algo_bracket_leg(row: dict, coid_prefix: str) -> str | None:
    """Classify a RAW /fapi/v1/openAlgoOrders row as 'sl' | 'tp' | None.

    Separate from `reduce_only_bracket_leg` because the algo endpoint returns a
    different shape, and the difference is not cosmetic. The shape of a live row
    while a healthy bracket rests (ids, timestamp and price illustrative):

        {"algoId": "1000000000000001",
         "clientAlgoId": "snap-v1-1700000000000-t",
         "side": "SELL", "triggerPrice": "50000.0",
         "strategyType": null, "reduceOnly": true, "algoStatus": "NEW"}

    Versus what `reduce_only_bracket_leg` needs: `info.type`/`origType`
    containing STOP/TAKE_PROFIT, and `reduceOnly` nested under `info`. Here
    there is **no type field at all** (`strategyType` was null on BOTH resting
    legs), `reduceOnly` is a top-level bool, and the id is `clientAlgoId`.
    Passing algo rows to the plain classifier returns None for every leg — i.e.
    "bracket missing" — which is the same false negative that produced the
    2026-07-22 `-4045` re-place spam, reached by a different route. That is why
    this function exists and why it does not key on type.

    Discriminator is the bot's OWN client-order-id suffix: `_place_brackets`
    tags the stop `-s` and the take-profit `-t` (binance_client `_coid`). This
    also scopes detection to our own orders — a bracket placed by hand in the
    Binance app carries a different prefix and must NOT count as "intact",
    because reprotect could not re-place it faithfully anyway.
    """
    coid = row.get("clientAlgoId")
    if not isinstance(coid, str) or not coid.startswith(coid_prefix):
        return None
    # reduceOnly / closePosition arrive as real bools here, but tolerate the
    # string form the plain endpoint uses in case Binance ever aligns them.
    if not (str(row.get("reduceOnly")).lower() == "true"
            or str(row.get("closePosition")).lower() == "true"):
        return None
    leg = coid.rsplit("-", 1)[-1]
    if leg == "t":
        return "tp"
    # `sb` = the breakeven stop that _maybe_breakeven places in place of `s`
    # (a distinct id, never an untagged order — see
    # BinanceClient.place_tagged_stop). It IS the position's stop, so it counts
    # as SL here. ⚠️ Reprotect would re-place a missing stop at the ORIGINAL
    # distance from `active_bracket` and undo the breakeven move — before
    # enabling reprotect on a breakeven leg, teach it to read `be_moved`.
    if leg in ("s", "sb"):
        return "sl"
    return None


def breakeven_due(
    side: str,
    entry_price: float,
    sl_distance: float,
    closed_highs: list[float],
    closed_lows: list[float],
    at_r: float,
) -> tuple[bool, float]:
    """Has the open trade's max favourable excursion reached `at_r` R?

    Live port of `tools/trailing_stop_study.TrailMixin` with `be_at_r` set (the
    arm validated in TRAILING_STOP_VERDICT.md): the extreme starts at the
    entry price and extends with the High (long) / Low (short) of every CLOSED
    bar from the fill bar onward — the fill bar included, because the backtest
    fills at that bar's open and sees its High at its close. 1R is the
    initial stop distance.

    Returns (due, mfe_in_R). Never raises; bad inputs are "not due".
    """
    if at_r <= 0 or sl_distance <= 0 or entry_price <= 0:
        return False, 0.0
    if side == "long":
        vals = [float(h) for h in closed_highs if h == h]   # drop NaN
        ext = max([entry_price, *vals])
        mfe = (ext - entry_price) / sl_distance
    elif side == "short":
        vals = [float(v) for v in closed_lows if v == v]
        ext = min([entry_price, *vals])
        mfe = (entry_price - ext) / sl_distance
    else:
        return False, 0.0
    return mfe >= at_r, mfe


def breakeven_stop_price(side: str, entry_price: float, buffer_frac: float) -> float:
    """Entry ± a small buffer so a scratch covers fees (buffer_pct from the YAML)."""
    return entry_price * (1 + buffer_frac) if side == "long" else entry_price * (1 - buffer_frac)


@dataclass(frozen=True)
class BracketState:
    """What is actually resting for an open position, across BOTH order books.

    `intact` answers "is this position protected?"; `any_leg` answers "did a
    cancel really clear everything?" — the pre-re-place guard that stops a
    partial cancel leaving a stale leg beside a fresh pair.
    """

    sl: bool
    tp: bool
    place_tp: bool

    @property
    def intact(self) -> bool:
        return self.sl and (self.tp or not self.place_tp)

    @property
    def any_leg(self) -> bool:
        return self.sl or self.tp

    def describe(self) -> str:
        return (f"SL={'present' if self.sl else 'MISSING'} "
                f"TP={'present' if self.tp else ('MISSING' if self.place_tp else 'n/a')}")


def bracket_state(
    plain_orders: list[dict] | None,
    algo_rows: list[dict] | None,
    coid_prefix: str,
    place_tp: bool,
) -> BracketState:
    """Merge both order books into one answer.

    BOTH sources are required. Plain-only is what broke in July (algo brackets
    invisible → "missing" → re-place loop). Algo-only would be the mirror bug if
    Binance ever moves brackets back, or for a leg whose SL rests as a plain
    order. Reading both is the only shape that cannot silently under-report.
    """
    legs: set[str | None] = {reduce_only_bracket_leg(o) for o in (plain_orders or [])}
    legs |= {algo_bracket_leg(r, coid_prefix) for r in (algo_rows or [])}
    return BracketState(sl="sl" in legs, tp="tp" in legs, place_tp=place_tp)


# ---------------------------------------------------------------------------
# Boot-resume gate (Phase B, C1). PURE: the caller does every fetch.
#
# Split out of Bot._can_adopt so that tools/boot_resume_probe.py and boot() run
# the SAME code. A probe that re-implemented the gate would prove nothing about
# the gate. The reason strings are the evidence (the boot's "boot-resume
# OBSERVE" line and the probe's output must be byte-comparable), so do not
# reword them casually.
#
# Split in two because the order-book reads are network round-trips. The Bot
# wrapper runs `adopt_precheck` first and only fetches the books if it passes,
# exactly as Phase A did inline. `can_adopt` re-runs the precheck itself, so a
# caller that fetches up front (the probe) gets the identical verdict.
# ---------------------------------------------------------------------------

def adopt_precheck(params: dict, pos, ab_raw: str | None,
                   qty_step: float,
                   latest_root: str | None) -> tuple[str | None, dict | None]:
    """Everything the gate can decide without an order book.

    `latest_root` is `state.latest_entry_coid_root()` — the coid root of the
    newest entry fill, read by the CALLER (local DB only, no exchange call).

    Returns (refusal_reason, parsed_active_bracket). A None reason means "no
    refusal yet; go and read the books".
    """
    rp = (params.get("reprotect") or {})
    # Without an ARMED re-placer, adopting a possibly-unprotected position is
    # strictly worse than closing it — nothing would ever restore the bracket.
    # This is also what keeps donchian and sol fail-closed by construction:
    # neither config has a `reprotect:` key at all.
    if not rp.get("enabled", False):
        return "reprotect not enabled for this leg", None
    if rp.get("observe_only", True):
        return "reprotect still observe-only — no armed re-placer", None

    if not ab_raw:
        # REACHABLE, and it must flatten. _maybe_reprotect clears this to ''
        # on the first flat tick, and a dropped exit leaves the same shape
        # (donchian 2026-09-04 wrote no fill and no event). An open position
        # with no stashed bracket is one we cannot identify as ours.
        return "no active_bracket record (cleared, or never ours)", None
    try:
        ab = json.loads(ab_raw)
    except (ValueError, TypeError):
        return "active_bracket unparseable", None
    if not isinstance(ab, dict):
        return "active_bracket unparseable", None

    # C5 (B3): IDENTITY. The stashed bracket must belong to the newest entry
    # this leg recorded — the same test _breakeven_step applies before it
    # touches a stop. Side / ±2% entry / qty cannot tell a stale record from a
    # previous position, or a manual position that lands inside the band, apart
    # from ours. BOTH sides must be non-empty: `None == None` is not proof, and
    # an untagged root also means the adopted position's exit/close orders
    # would carry no signal_id (consolidate could not join them).
    ab_sid = ab.get("signal_id")
    if not latest_root:
        return "no latest entry coid root — cannot prove the position is ours", ab
    if not ab_sid:
        return "active_bracket has no signal_id — cannot prove the position is ours", ab
    if str(ab_sid) != str(latest_root):
        return (f"identity mismatch: active_bracket.signal_id {ab_sid} vs "
                f"latest entry root {latest_root}"), ab

    if ab.get("side") != pos.side:
        return f"side mismatch: stashed {ab.get('side')} vs live {pos.side}", ab
    ep = float(ab.get("entry_price") or 0.0)
    if ep <= 0 or pos.entry_price <= 0:
        return "entry price missing on one side", ab
    if abs(ep - pos.entry_price) / pos.entry_price > 0.02:
        return (f"entry price drift {ep:.2f} vs {pos.entry_price:.2f} "
                f"exceeds 2%"), ab

    # Quantity is NOT checked by _maybe_reprotect, and that is safe there
    # because it re-places against the live pos.qty whatever the stash says.
    # Adoption inherits more than a bracket, so a partially-closed position
    # must not be resumed against a stale size. One qty_step of tolerance,
    # because the stash records the requested qty and the fill is rounded.
    #
    # D2 (God, 2026-09-29): a record with NO stashed qty is REFUSED, not
    # skipped. Every position opened before Phase A deployed lacks the field,
    # and "cannot check the size" is not affirmative proof. Cost: at most the
    # first position after deploy is flattened, exactly as today.
    try:
        stashed_qty = float(ab.get("qty") or 0.0)
    except (TypeError, ValueError):
        stashed_qty = 0.0
    if stashed_qty <= 0:
        return "no stashed qty (position opened before Phase A) — cannot verify size", ab
    step = float(qty_step or 0.0)
    if abs(stashed_qty - float(pos.qty)) > max(step, 1e-12):
        return (f"qty mismatch: stashed {stashed_qty} vs live "
                f"{pos.qty} (step {step})"), ab

    # A channel-exit strategy (donchian-v3) places an SL and no TP, so
    # `bracket_state` would be asked whether a HALF bracket is intact —
    # a shape nothing has been exercised against. Refuse it in code, not
    # in a comment.
    #
    # Note this is the OPPOSITE safety direction from reprotect's decision
    # not to gate on strategy. There, a hard gate would silently DISABLE
    # protection on a legitimate switch. Here, refusing simply falls through
    # to flatten — today's behaviour — so the gate costs a resumed trade,
    # never an unprotected one. Lift it once the half-bracket path is tested.
    if not bool(ab.get("place_tp", True)):
        return "SL-only bracket (channel-exit strategy) — half-bracket path untested", ab
    return None, ab


# The leg suffixes a position's own bracket carries: `-s` stop, `-sb` the
# breakeven replacement stop (place_tagged_stop), `-t` take-profit.
_ADOPT_LEG_SUFFIXES = ("s", "sb", "t")


def adopt_bracket_state(open_orders: list[dict] | None, algo_rows: list[dict] | None,
                        coid_prefix: str, latest_root: str | None,
                        pos_side: str) -> BracketState:
    """The adopt gate's view of the bracket: ONLY this position's own legs. PURE.

    A row counts only if (a) its client id is EXACTLY
    `{prefix}{latest_root}-{s|sb|t}` and (b) its side CLOSES the position
    (SELL for a long, BUY for a short). Prefix-only matching (what
    `algo_bracket_leg` does for the live re-placer) accepts a stale stop from a
    PREVIOUS position — e.g. `snap-v1-OLD-s` left by a failed pre-entry cancel —
    as "our SL present", and the gate would adopt a position with no real stop.
    Untagged, foreign, other-leg and other-root rows all read as missing:
    the fail-closed direction.

    Scoped here, NOT in the shared `bracket_state` / `algo_bracket_leg`: the
    live, armed `_maybe_reprotect` also calls those. Shared with the probe so its
    printed bracket line is the same view the gate decided on.
    """
    if not latest_root or pos_side not in ("long", "short"):
        return BracketState(sl=False, tp=False, place_tp=True)
    wanted = {f"{coid_prefix}{latest_root}-{leg}" for leg in _ADOPT_LEG_SUFFIXES}
    close_side = "SELL" if pos_side == "long" else "BUY"

    def plain_ok(o: dict) -> bool:
        info = o.get("info") or {}
        coid = o.get("clientOrderId") or info.get("clientOrderId")
        side = str(o.get("side") or info.get("side") or "").upper()
        return coid in wanted and side == close_side

    def algo_ok(r: dict) -> bool:
        return (r.get("clientAlgoId") in wanted
                and str(r.get("side") or "").upper() == close_side)

    return bracket_state([o for o in (open_orders or []) if plain_ok(o)],
                         [r for r in (algo_rows or []) if algo_ok(r)],
                         coid_prefix, True)


def stale_root_legs(open_orders: list[dict] | None, algo_rows: list[dict] | None,
                    coid_prefix: str, root: str | None) -> list[tuple[str, str, str]]:
    """Reduce-only legs of OUR prefix that belong to a DIFFERENT entry root. PURE.

    Returns [(book, order_id, coid)] with book "plain" | "algo". These are
    leftovers of an earlier trade (e.g. a pre-entry sweep that failed), and a
    reduce-only stop/TP from that trade can CLOSE the current position at a
    stale price. Only our own prefix is ever returned: manual orders, other
    legs' prefixes and untagged orders are never touched. With no current
    root nothing is returned — without it "different" cannot be decided.
    """
    if not root:
        return []

    def root_of(coid: str) -> str:
        return coid[len(coid_prefix):].rsplit("-", 1)[0]

    def is_true(v) -> bool:
        return str(v).lower() == "true"

    out: list[tuple[str, str, str]] = []
    for o in open_orders or []:
        info = o.get("info") or {}
        coid = o.get("clientOrderId") or info.get("clientOrderId")
        if not isinstance(coid, str) or not coid.startswith(coid_prefix):
            continue
        if not (is_true(o.get("reduceOnly")) or is_true(info.get("reduceOnly"))
                or is_true(info.get("closePosition"))):
            continue
        if root_of(coid) != str(root):
            out.append(("plain", str(o.get("id")), coid))
    for r in algo_rows or []:
        coid = r.get("clientAlgoId")
        if not isinstance(coid, str) or not coid.startswith(coid_prefix):
            continue
        if not (is_true(r.get("reduceOnly")) or is_true(r.get("closePosition"))):
            continue
        if root_of(coid) != str(root):
            out.append(("algo", str(r.get("algoId")), coid))
    return out


def can_adopt(params: dict, pos, ab_raw: str | None,
              open_orders: list[dict] | None, algo_rows: list[dict] | None,
              algo_ok: bool, qty_step: float,
              coid_prefix: str, *, latest_root: str | None) -> tuple[bool, str]:
    """The whole boot-resume gate. Returns (verdict, reason). FAIL-CLOSED.

    `open_orders=None` means the plain book could not be read.
    `algo_ok=False` means the algo book could not be read — never round that
    down to "no bracket" (the July -4045 bug; at boot it would be worse,
    because we would resume rather than merely re-place).
    """
    why, ab = adopt_precheck(params, pos, ab_raw, qty_step, latest_root)
    if why is not None:
        return False, why
    assert ab is not None
    if open_orders is None:
        return False, "plain order book unreadable"
    if not algo_ok:
        return False, "algo book unreadable — cannot distinguish 'no bracket' from 'no answer'"

    # Blocker 4 + root/side identity: only THIS position's own legs count
    # (exact coid root + closing side) — see adopt_bracket_state.
    st = adopt_bracket_state(open_orders, algo_rows, coid_prefix, latest_root,
                             pos.side)
    if st.intact:
        return True, f"bracket intact ({st.describe()})"

    # Blocker 3: an absent STOP is never adoptable. Reprotect would re-place
    # it on the next tick, but if the mark is already through the stop price
    # Binance rejects the placement (it would trigger immediately) and the
    # position sits UNPROTECTED while reprotect retries. Deciding that needs a
    # mark-price read this gate does not have; refusing needs nothing.
    if not st.sl:
        return False, (f"stop leg missing ({st.describe()}) — refusing: "
                       f"cannot prove the mark has not passed the stop")

    # Only the TP is missing. Still adoptable IF the armed re-placer is allowed
    # to act on the very next tick. If it has already spent its cap, nothing
    # will restore the bracket, so refuse rather than resume half a bracket.
    rp = (params.get("reprotect") or {})
    cap = int(rp.get("max_replaces_per_position", 3))
    done = int(ab.get("reprotect_count", 0))
    if done >= cap:
        return False, (f"bracket missing ({st.describe()}) and reprotect "
                       f"cap spent ({done}/{cap})")
    return True, (f"bracket missing ({st.describe()}) but reprotect can "
                  f"restore it ({done}/{cap} used)")


# D5 (God, 2026-09-29): "max 3 adopts / 30 min, then stop adopting".
# Implemented STRICTER than the literal wording (arming blocker 5): at most 3
# adopts per position (signal_id), with NO time reset. A window anchored on the
# first adopt re-arms every 30 min, so a SLOW crash loop (one crash every 11+
# min) would adopt forever; a sliding window has the same hole at 11 min. A
# per-position cap closes it. Cost: a 4th organic restart inside one position
# flattens it — today's behaviour. Legitimate in-position restarts are rare,
# because deploys go through leg_safe_restart.py, which requires the leg flat.
ADOPT_LOOP_MAX = 3


def adopt_loop_guard(log_raw: str | None, signal_id: str | None, now_s: float,
                     max_adopts: int = ADOPT_LOOP_MAX) -> tuple[bool, str, dict]:
    """Crash-loop breaker for boot-resume. PURE.

    A bug that crashes the tick loop BECAUSE of the adopted position would,
    under systemd `Restart=on-failure` / `RestartSec=10`, become
    crash → adopt → crash forever, and the time stop / trend exit / reprotect
    would never run. Today's flatten breaks that loop; this keeps a breaker.

    `log_raw` is the persisted `boot_adopt_log` meta value. Returns
    (allowed, reason, new_log). `new_log` is what the caller must persist
    BEFORE adopting when allowed is True (so a crash after it still counts).

    The count is per signal_id and resets ONLY when the signal_id changes (a
    different position) — never on time. `first_ts` is kept for the message.
    An unreadable log is treated as empty: it is our own counter, not proof of
    anything, and failing closed on it would refuse every future adopt.
    """
    key = signal_id or ""
    prev: dict = {}
    if log_raw:
        try:
            parsed = json.loads(log_raw)
            if isinstance(parsed, dict):
                prev = parsed
        except (ValueError, TypeError):
            prev = {}
    try:
        prev_count = int(prev.get("count", 0))
        prev_first = float(prev.get("first_ts", 0.0))
    except (TypeError, ValueError):
        prev_count, prev_first = 0, 0.0
    same = prev.get("signal_id") == key and prev_count > 0
    if not same:
        return True, f"adopt 1/{max_adopts} for this position", {
            "signal_id": key, "count": 1, "first_ts": float(now_s)}
    if prev_count >= max_adopts:
        mins = max(0, int((now_s - prev_first) // 60))
        return False, (f"adopt-loop guard: {prev_count} adopts of "
                       f"{key or '(untagged)'} already (first {mins} min ago; "
                       f"max {max_adopts} per position)"), prev
    n = prev_count + 1
    return True, f"adopt {n}/{max_adopts} for this position", {
        "signal_id": key, "count": n, "first_ts": prev_first}


def _naive_utc(ts) -> pd.Timestamp:
    """pd.Timestamp in NAIVE UTC — the convention of fetch_ohlcv's index.

    `fills.ts` is tz-AWARE (datetime.now(UTC).isoformat()) while df.index is
    naive, and comparing the two raises TypeError. In _maybe_enter that would
    crash the tick loop on every bar — i.e. exactly the crash→adopt loop D5
    exists to stop — so every seed is normalised here.
    """
    t = pd.Timestamp(ts)
    if pd.isna(t):
        # pd.Timestamp('NaT') does not raise, and NaT would win max() in
        # signal_bar_seed — reject it like any other unparseable value.
        raise ValueError(f"non-finite timestamp: {ts!r}")
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def signal_bar_seed(persisted_raw: str | None, entry_fill_ts_raw: str | None,
                    bar_seconds: int) -> pd.Timestamp | None:
    """B2: the `_last_signal_ts` an ADOPTED position must resume with. PURE.

    Candidates, latest wins:
      - `last_entry_bar_ts` meta — the last closed bar _maybe_enter evaluated
        before the restart (exactly what the in-memory value held).
      - fallback from the newest entry fill: floor(fill_ts, bar) − 1 bar. The
        fill lands inside the bar AFTER the signal bar (entry is evaluated on
        the last CLOSED bar, index = bar OPEN time), so the signal bar is one
        bar before the fill's bar. A plain floor (the plan's wording) would name
        the fill's own bar and suppress that bar's legitimate evaluation once it
        closes.
    Returns None when neither is usable; the caller then refuses to adopt.
    """
    out: list[pd.Timestamp] = []
    if persisted_raw:
        with contextlib.suppress(ValueError, TypeError):
            out.append(_naive_utc(persisted_raw))
    if entry_fill_ts_raw and bar_seconds > 0:
        try:
            t = _naive_utc(entry_fill_ts_raw)
            out.append(t.floor(f"{int(bar_seconds)}s")
                       - pd.Timedelta(seconds=int(bar_seconds)))
        except (ValueError, TypeError):
            pass  # unusable fallback; the meta key alone may still seed
    return max(out) if out else None
