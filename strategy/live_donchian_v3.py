"""Live signal evaluator + channel-exit check for the Donchian-v3 leg.

Pure-function ports of `DonchianBreakoutBTCv3` (strategy/signals_donchian.py)
for use in `bot.py`:
  - evaluate_signal_donchian_v3() : entry evaluation on the last CLOSED bar.
  - channel_exit_signal()         : in-position Donchian-channel EXIT check.

EXIT SEMANTICS (channel_exit_signal) — byte-for-byte with the backtest.
The backtest closes an open position on the Donchian EXIT-channel cross,
computed exactly as attach_donchian / DonchianBreakoutBTCv3.next():

    exit_upper = Close.rolling(N_exit, min_periods=N_exit).max().shift(1)
    exit_lower = Close.rolling(N_exit, min_periods=N_exit).min().shift(1)
    long  exits when close <  exit_lower      (STRICT '<')
    short exits when close >  exit_upper      (STRICT '>')

The `.shift(1)` makes the channel the max/min of the N closes STRICTLY BEFORE
the current bar; `close` is the current (last CLOSED) bar's close, so the
channel can never peek at the bar being tested. channel_exit_signal mirrors
this AND next()'s NaN guard over (upper, lower, exit_upper, exit_lower, atr).
Live geometry (entry / exit channel lengths) comes from the params YAML
(donchian_period_entry / donchian_period_exit; the exit length was a sweep winner).

CALLER CONTRACT: pass a frame whose last row is a CLOSED bar — drop Binance's
still-forming last row first. bot._maybe_channel_exit does exactly that, so the
live exit sees the same closed-bar series the backtest steps over.

KNOWN LIVE↔BACKTEST DIVERGENCE (same-bar flip, accepted): the bot's loop runs
the channel-exit check and then entry evaluation in the SAME iteration, so a
close that pierces both the exit channel AND the opposite entry
channel (with the slope gate agreeing) can exit and enter the opposite side on
one bar. The backtest returns after its exit branch and can only re-enter at
the NEXT bar. Verification panel 2026-07-18: narrow, directionally sensible,
mirrors the accepted time-stop→re-enter behavior — documented, not gated.

NO TP: the Donchian entry places entry + SL ONLY (no TP leg — "let the channel
exit close the trade"). ATR_TP_K below is retained purely as an advisory
reference level in logs/telemetry; it is NEVER placed as an order.

TIME-STOP: both sides honour it, at the configured bar count. The live bot reads
`max_hold_bars` (bot._maybe_time_stop); the backtest reads `time_stop_bars` and
closes on it in DonchianBreakoutBTCv3.next() (strategy/signals_donchian.py
:139-143), and every donchian backtest entry point passes it explicitly
(tools/full_history_backtest.py, tools/v1_plus_donchian_backtest.py,
tools/_postfrac_donchian_variants_sweep.py). Earlier revisions of this docstring
claimed the backtest ignored it — it does not; there is no divergence here.
CAUTION: the backtest CLASS has no time_stop_bars default (it must come from the
params YAML and raises if missing), so a new backtest path cannot silently run a
different hold than live.
Either way the time stop can only close EARLIER than the channel, never override
a channel-exit decision.

Returns (evaluate_signal_donchian_v3):
  side ∈ {'long', 'short', None}
  sl_distance, tp_distance — price units (NaN if side is None)
  debug — dict for logging
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.indicators import atr, ema

# Every tunable comes from params["strategy"] (config/params_donchian.yaml,
# untracked; keys enforced at boot by config_check.REQUIRED_KEYS). There are
# deliberately no fallback values here.
ATR_TP_K = 5.0               # advisory telemetry only — entry places NO TP order


def _ema_slope_signed(close: pd.Series, ema_period: int, slope_window: int) -> float:
    """Mirror of strategy.regime_classifier.ema_slope_signed at the last bar.

    Byte-for-byte port of the backtest formula evaluated on the last bar:

        ema = close.ewm(span=ema_period, adjust=False).mean()
        slope_per_bar = (ema[-1] - ema[-1 - slope_window]) / slope_window
        signed_pct    = (slope_per_bar / close[-1]) * 100.0

    i.e. the signed EMA endpoint-difference per bar, divided by the CURRENT
    close (not the EMA window mean), expressed in percent-per-bar. Positive =
    uptrend, negative = downtrend. The gate compares this against
    `slope_trend_threshold_pct` (from config), so units must be
    percent-per-bar to match the backtest's directional gate exactly.

    NOTE on EWM seeding: the live bot feeds ~1500 4h bars (bot.py fetch_ohlcv
    limit=1500), well past the ~400-bar (1-alpha)^n seed-decay horizon for
    the configured regime_ema_period, so the last-bar EMA matches the backtest's full-history
    EMA at the same timestamp. The `ema()` helper uses the identical
    ewm(span=..., adjust=False) recursion; min_periods only affects head NaNs.
    """
    e = ema(close, ema_period)
    if len(e) < slope_window + 1:
        return float("nan")
    ema_last = float(e.iloc[-1])
    ema_prev = float(e.iloc[-1 - slope_window])
    if not (np.isfinite(ema_last) and np.isfinite(ema_prev)):
        return float("nan")
    cur_close = float(close.iloc[-1])
    if cur_close <= 0:
        return float("nan")
    slope_per_bar = (ema_last - ema_prev) / slope_window
    return (slope_per_bar / cur_close) * 100.0


def evaluate_signal_donchian_v3(
    bars_4h: pd.DataFrame,
    funding_rate: float,             # unused; Donchian doesn't gate on funding
    params: dict,
) -> tuple[str | None, float, float, dict]:
    """Evaluate Donchian-v3 cons entry on the last closed 4h bar.

    bars_4h must have columns Open, High, Low, Close, Volume and at least
    max(donchian_period_entry, regime_ema_period + regime_slope_window) + 10 rows.

    Returns (side, sl_distance, tp_distance, debug).
    """
    _ = funding_rate   # signature compat with other evaluators
    s = params["strategy"]
    period_entry = int(s["donchian_period_entry"])
    period_exit = int(s["donchian_period_exit"])
    atr_period = int(s["atr_period"])
    atr_sl_k = float(s["atr_sl_multiple"])
    regime_ema_p = int(s["regime_ema_period"])
    regime_slope_w = int(s["regime_slope_window"])
    slope_thr = float(s["slope_trend_threshold_pct"])

    warmup = max(period_entry, regime_ema_p + regime_slope_w, atr_period) + 5
    if len(bars_4h) < warmup:
        return None, float("nan"), float("nan"), {"reason": "warmup", "have": len(bars_4h), "need": warmup}

    close, high, low = bars_4h["Close"], bars_4h["High"], bars_4h["Low"]

    # Donchian channel: rolling max/min of CLOSE over last N bars, shifted 1
    # to use bars strictly BEFORE the current bar (no lookahead).
    upper = close.rolling(period_entry, min_periods=period_entry).max().shift(1).iloc[-1]
    lower = close.rolling(period_entry, min_periods=period_entry).min().shift(1).iloc[-1]
    exit_upper = close.rolling(period_exit, min_periods=period_exit).max().shift(1).iloc[-1]
    exit_lower = close.rolling(period_exit, min_periods=period_exit).min().shift(1).iloc[-1]
    atr_v = atr(high, low, close, atr_period).iloc[-1]
    cur_close = float(close.iloc[-1])

    if not all(np.isfinite([upper, lower, atr_v, cur_close])):
        return None, float("nan"), float("nan"), {
            "reason": "nan_indicators",
            "upper": float(upper) if np.isfinite(upper) else None,
            "lower": float(lower) if np.isfinite(lower) else None,
            "atr": float(atr_v) if np.isfinite(atr_v) else None,
        }

    sl_dist = atr_sl_k * float(atr_v)
    tp_dist = ATR_TP_K * float(atr_v)

    # Regime gate (signed EMA-slope). Only active when slope_thr > 0.
    gate_on = slope_thr > 0
    slope = _ema_slope_signed(close, regime_ema_p, regime_slope_w) if gate_on else 0.0

    debug = {
        "cur_close": cur_close, "upper": float(upper), "lower": float(lower),
        "exit_upper": float(exit_upper), "exit_lower": float(exit_lower),
        "atr": float(atr_v), "sl_dist": float(sl_dist), "tp_dist": float(tp_dist),
        "slope": float(slope) if np.isfinite(slope) else None,
        "gate_on": gate_on, "slope_threshold": slope_thr,
    }

    # LONG: close > upper-channel AND (gate off OR slope >= +threshold)
    if cur_close > upper:
        if gate_on and (not np.isfinite(slope) or slope < slope_thr):
            return None, float("nan"), float("nan"), {**debug, "reason": "long_blocked_by_slope"}
        return "long", sl_dist, tp_dist, debug

    # SHORT: close < lower-channel AND (gate off OR slope <= -threshold)
    if cur_close < lower:
        if gate_on and (not np.isfinite(slope) or slope > -slope_thr):
            return None, float("nan"), float("nan"), {**debug, "reason": "short_blocked_by_slope"}
        return "short", sl_dist, tp_dist, debug

    return None, float("nan"), float("nan"), {**debug, "reason": "no_breakout"}


def channel_exit_signal(
    bars_4h: pd.DataFrame,
    position_side: str,
    params: dict,
) -> tuple[bool, dict]:
    """Decide whether an OPEN donchian-v3 position closes on the Donchian
    exit-channel cross, evaluated on the LAST CLOSED 4h bar.

    The channel-exit test is identical to DonchianBreakoutBTCv3.next()'s
    in-position channel-exit branch (strategy/signals_donchian.py); the
    time stop is handled separately by bot._maybe_time_stop (next() evaluates
    it first and only runs this test when the time stop is not due). next() guards at the top on all of
    (upper, lower, exit_upper, exit_lower, atr) being finite, then closes:

        long  when close_v <  exit_lower
        short when close_v >  exit_upper

    STRICT comparisons — a close exactly AT the channel does NOT exit. The
    exit-channel columns use the same `.rolling(N).max()/min().shift(1)`
    convention as attach_donchian, so the channel is the extreme of the N
    closes strictly BEFORE the current bar and never peeks at it.

    The full NaN guard (incl. the entry channel and ATR) is replicated
    even though only exit_upper/exit_lower drive the decision: it makes the
    warmup window match the backtest bar-for-bar. In practice all indicators
    are finite by the time a position is open (entry needs the entry channel),
    so the entry-channel/ATR terms never suppress a real exit. ATR is a
    guard-only input here and is computed with the live (unshifted) convention,
    consistent with evaluate_signal_donchian_v3.

    CALLER CONTRACT: bars_4h.iloc[-1] MUST be a CLOSED bar (drop Binance's
    forming last row first). Returns (should_exit, debug). should_exit is
    False for a flat/unknown side, during warmup, or on any NaN indicator. It
    does not raise on market data; a params dict missing a required key raises
    KeyError (boot validation guarantees the live dict is complete).
    """
    if position_side not in ("long", "short"):
        return False, {"reason": "not_in_position", "side": position_side}

    s = params["strategy"]
    period_entry = int(s["donchian_period_entry"])
    period_exit = int(s["donchian_period_exit"])
    atr_period = int(s["atr_period"])

    # +1 for the shift(1): the newest channel value needs N closes STRICTLY
    # before the current bar, i.e. N+1 rows total.
    need = max(period_entry, period_exit, atr_period) + 1
    if len(bars_4h) < need:
        return False, {"reason": "warmup", "have": len(bars_4h), "need": need}

    close, high, low = bars_4h["Close"], bars_4h["High"], bars_4h["Low"]

    # Same shift convention as attach_donchian: rolling extreme of the N closes
    # ending at the PREVIOUS bar (shift(1)), read at the last (current) bar.
    upper = close.rolling(period_entry, min_periods=period_entry).max().shift(1).iloc[-1]
    lower = close.rolling(period_entry, min_periods=period_entry).min().shift(1).iloc[-1]
    exit_upper = close.rolling(period_exit, min_periods=period_exit).max().shift(1).iloc[-1]
    exit_lower = close.rolling(period_exit, min_periods=period_exit).min().shift(1).iloc[-1]
    atr_v = atr(high, low, close, atr_period).iloc[-1]
    cur_close = float(close.iloc[-1])

    # next()'s NaN guard — same five indicators (+ the current close).
    if not all(np.isfinite([upper, lower, exit_upper, exit_lower, atr_v, cur_close])):
        return False, {"reason": "nan_indicators", "side": position_side}

    debug = {
        "side": position_side,
        "cur_close": cur_close,
        "exit_upper": float(exit_upper),
        "exit_lower": float(exit_lower),
        "period_exit": period_exit,
    }

    # Exit branch — mirrors next() exactly (strict inequalities).
    if position_side == "long" and cur_close < exit_lower:
        return True, {**debug, "reason": "channel_exit_long"}
    if position_side == "short" and cur_close > exit_upper:
        return True, {**debug, "reason": "channel_exit_short"}
    return False, {**debug, "reason": "hold"}
