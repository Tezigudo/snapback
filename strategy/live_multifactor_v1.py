"""multifactor-v1 — live signal evaluator (pure function).

Mirror of `live_v3all_wider4.py` for the v1 strategy. Both modules export
`evaluate_signal*(bars_15m, funding_rate, params)` callable from the bot
loop and from validation harnesses.

v1 is the deployable default. Backtest evidence (2026-05-17 lock):
  5 OOS windows 2022-2025 H1 → +50.48% compounded, 5 of 5 windows positive.
  (CLAUDE.md historically cites +55.73% from an earlier $100k footing; the
   $1M footing in reports/multifactor_v1_deepening.json is +50.48%.)

LIVE↔BACKTEST PARITY (after 4H gate add, 2026-06-02):
  The 4H EMA200 gate is causally evaluated identically to the backtest:
    - backtest: full-history 4H EMA(200), aligned to 15m via merge_asof
      on close-time timestamps (lookahead-safe, see signals_multifactor.py
      :_build_4h_ema_aligned)
    - live: rolling fetch of ~180d of 4H bars from the SAME parquet/REST
      source the backtest uses; same EMA function (strategy.indicators.ema);
      same "use most-recently CLOSED 4H bar" rule applied at the current
      15m timestamp.
  Validated by tools/multifactor_validate.py to ≥99.5% per-bar signal parity.

LONG entry, ALL of:
  RSI(rsi_period, 15m) < rsi_long_threshold
  close > EMA(ema_period, 15m)                                 (trend filter)
  volume(15m) > volume_multiple × SMA(volume_ma_period, vol)
  funding_rate <= funding_extreme_threshold                    (per 8h)
  close > EMA(mtf_4h_ema_period, 4H) AT MOST-RECENTLY-CLOSED 4H BAR (if gated)  ← NEW 2026-06-02
(thresholds and periods are read from the params YAML)

SHORT entry is the mirror.

Exit (sized by bot.py, not here):
  TP at +/- tp_pct × close
  SL at -/+ sl_pct × close
  Time-stop at time_stop_bars
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from strategy.indicators import ema, rsi, sma

log = logging.getLogger(__name__)

# Live evaluator loads this many days of 4H bars per tick. EMA(200) over 4h
# bars needs ~33 days just to leave the warm-up window; we load ~180 days so
# the EMA has fully converged to its full-history value (verified to ~1e-6 by
# the parity harness in tools/multifactor_validate.py). The cache is two-sided
# in exchange/data.py — repeated calls only fetch the tail.
_LIVE_4H_DAYS_BACK = 180


def _fetch_4h_bars(symbol: str = "BTC/USDT:USDT") -> pd.DataFrame | None:
    """Live-path 4H bar fetcher. Uses exchange.data.load_klines (cached + REST).

    Returns None if the fetch fails for any reason — caller treats this as
    "gate cannot be evaluated, block the entry" (NOT "fall back to no-gate").
    Local import keeps the strategy module importable in offline test contexts
    that don't have `requests` available.
    """
    try:
        from exchange.data import load_klines
    except ImportError as e:
        log.warning("live_multifactor_v1: exchange.data unavailable (%s)", e)
        return None
    try:
        df = load_klines(symbol, "4h", days_back=_LIVE_4H_DAYS_BACK)
        if df is None or df.empty:
            log.warning("live_multifactor_v1: 4H fetch returned empty")
            return None
        return df
    except Exception as e:
        log.warning("live_multifactor_v1: 4H fetch failed: %s", e)
        return None


def _compute_4h_ema_at_15m_close(
    bars_4h: pd.DataFrame,
    close_ts_15m: pd.Timestamp,
    ema_period: int,
) -> float:
    """Return the 4H EMA(ema_period) of the most-recently CLOSED 4H bar at
    `close_ts_15m`. NaN if no closed 4H bar precedes the 15m timestamp or if
    insufficient warm-up.

    Causally identical to the backtest's merge_asof(direction="backward") on
    bar-close timestamps. We compute the EMA on the full 4H history we
    received, then pick the value whose close-time is the largest <= 15m ts.
    """
    if bars_4h is None or bars_4h.empty:
        return float("nan")
    df = bars_4h.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # parquet uses lowercase, ccxt fetches may capitalise — handle both.
    close_col = "close" if "close" in df.columns else "Close"
    ema_full = ema(df[close_col], ema_period)
    # Re-index by CLOSE timestamp (open + 4h). The most recent value whose
    # close-time <= 15m timestamp is the one we use.
    close_times = df.index + pd.Timedelta(hours=4)
    ema_at_close = pd.Series(ema_full.values, index=close_times)
    ts = close_ts_15m
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    # Most-recent close-time <= ts. Equivalent to merge_asof backward.
    mask = ema_at_close.index <= ts
    if not mask.any():
        return float("nan")
    return float(ema_at_close[mask].iloc[-1])


def evaluate_signal(
    bars_15m: pd.DataFrame,
    funding_rate: float,
    params: dict,
    bars_4h: pd.DataFrame | None = None,
) -> tuple[str | None, dict]:
    """Return ('long'/'short'/None, debug_dict) for the last CLOSED 15m bar.

    Pure-function port of `DayTradeMultiFactorBTC._long_signal/_short_signal`
    (see strategy/signals_multifactor.py for the backtest version).

    Parameters
    ----------
    bars_15m:
        15m OHLCV frame, tz-naive index. Last row is the bar to evaluate.
    funding_rate:
        Most-recent funding rate (decimal, e.g. 0.0001 = 0.01%).
    params:
        Full params dict from config/params.yaml.
    bars_4h:
        OPTIONAL injected 4H frame. If None and the 4H gate is enabled,
        evaluator calls exchange.data.load_klines() to self-fetch. Harnesses
        inject cached slices to avoid network and to back-test deterministically.
    """
    s = params["strategy"]
    warm = max(s["mf_trend_ema_period"], s["volume_ma_period"], s["rsi_period"]) + 5
    if len(bars_15m) < warm:
        return None, {"reason": "warmup"}

    close = bars_15m["Close"]
    vol = bars_15m["Volume"]

    rsi_v = rsi(close, s["rsi_period"]).iloc[-1]
    vol_sma_v = sma(vol, s["volume_ma_period"]).iloc[-1]
    trend_ema_v = ema(close, s["mf_trend_ema_period"]).iloc[-1]
    cur_vol = vol.iloc[-1]
    cur_close = close.iloc[-1]

    if not all(np.isfinite([rsi_v, vol_sma_v, trend_ema_v, cur_vol, cur_close])):
        return None, {"reason": "nan_indicators"}

    vol_ok = cur_vol > s["volume_multiple"] * vol_sma_v
    trend_up = cur_close > trend_ema_v
    funding_long_blocked = (s["require_funding_not_extreme"]
                            and funding_rate > s["funding_extreme_threshold"])
    funding_short_blocked = (s["require_funding_not_extreme"]
                             and funding_rate < -s["funding_extreme_threshold"])

    # --- 4H EMA200 regime gate (additive, 2026-06-02). Default OFF for back-compat;
    # operator enables via params.yaml `strategy.use_mtf_4h_gate: true`.
    use_mtf_4h_gate = bool(s.get("use_mtf_4h_gate", False))
    ema_4h_v = float("nan")
    if use_mtf_4h_gate:
        mtf_4h_ema_period = int(s["mtf_4h_ema_period"])   # required when gated
        if bars_4h is None:
            symbol = params.get("symbol", "BTC/USDT:USDT")
            bars_4h = _fetch_4h_bars(symbol)
        ema_4h_v = _compute_4h_ema_at_15m_close(
            bars_4h, bars_15m.index[-1], mtf_4h_ema_period,
        )

    debug = {
        "ts": bars_15m.index[-1].isoformat(),
        "rsi": float(rsi_v), "vol_sma": float(vol_sma_v), "trend_ema": float(trend_ema_v),
        "cur_vol": float(cur_vol), "cur_close": float(cur_close),
        "vol_ok": bool(vol_ok), "trend_up": bool(trend_up),
        "funding_rate": funding_rate,
        "use_mtf_4h_gate": use_mtf_4h_gate,
        "ema_4h_200": ema_4h_v if np.isfinite(ema_4h_v) else None,
    }

    # 4H regime gate: long requires close > 4H EMA, short requires close <.
    # If gate is enabled and the 4H value is NaN (fetch failed or warmup),
    # BLOCK both sides. We do NOT silently fall back to ungated — that would
    # cause a re-locked "4H-gated" strategy to run ungated whenever the feed
    # hiccups, a silent live↔backtest divergence.
    if use_mtf_4h_gate:
        if not np.isfinite(ema_4h_v):
            return None, {**debug, "reason": "mtf_4h_gate_nan"}
        regime_long_ok = cur_close > ema_4h_v
        regime_short_ok = cur_close < ema_4h_v
        debug["regime_long_ok"] = bool(regime_long_ok)
        debug["regime_short_ok"] = bool(regime_short_ok)
    else:
        regime_long_ok = True
        regime_short_ok = True

    if (rsi_v < s["rsi_long_threshold"] and vol_ok and trend_up
            and not funding_long_blocked and regime_long_ok):
        return "long", debug
    if (rsi_v > s["rsi_short_threshold"] and vol_ok and not trend_up
            and not funding_short_blocked and regime_short_ok):
        return "short", debug
    return None, debug


def trend_exit_signal_multifactor_v1(
    bars_15m: pd.DataFrame,
    position_side: str,
    params: dict,
) -> tuple[bool, dict]:
    """Should an OPEN multifactor-v1 position close on an adverse EMA(200) cross?

    Byte-for-byte with `DayTradeMultiFactorBTC.next()`'s in-position branch
    (strategy/signals_multifactor.py:317-326), which runs whenever
    `require_trend` is true:

        t = self._trend_ema[i]
        if np.isfinite(t):
            long  closes when close_v < t
            short closes when close_v > t

    STRICT comparisons — a close exactly AT the EMA does NOT exit. `t` is the
    EMA on the ENTRY timeframe (15m, `mf_trend_ema_period`), NOT the 4H EMA200
    that gates entries (`mtf_4h_ema_period`); those are different lines and
    confusing them would exit on the wrong one.

    Why this exists: every v1 sign-off measured the model WITH this exit, but
    the live bot never ran it — `strategy_uses_trend_exit()` returned True only
    for donchian-v3 and supertrend, so live v1 exited on SL/TP/time-stop alone.
    The 2026-08-01 re-validation (MULTIFACTOR_V1_LIVE_EXIT_VERDICT.md) measured
    what that gap costs: walk-forward 64% vs the 70% gate, OOS 3/5, and a
    start-anchored drawdown that breaches the kill floor on 0.41% of deploy
    dates where the as-validated model breaches on 0.00%. This function closes
    the gap in the direction of the model that was actually validated.

    `require_trend: false` disables the exit, exactly as it disables the branch
    in the backtest — the two stay in lockstep off one config key. The DEFAULT
    when the key is absent must also match, and it is `True`
    (`DayTradeMultiFactorBTC.require_trend`). Defaulting to False here would
    mean a config that lost the key runs the exit in the backtest and NOT live —
    which is this module's own bug re-created silently, on the same leg.

    KNOWN DIVERGENCES (both shared with the donchian channel exit, both
    accepted):

    1. backtesting.py executes `position.close()` at the NEXT bar's open, while
       the bot closes at market as soon as it sees the triggering CLOSED bar.
       Sub-bar timing only; the decision itself is identical.
    2. Same-bar exit -> OPPOSITE entry. `bot.loop` calls `_maybe_channel_exit`
       and then `_maybe_enter` in the same tick, and `_maybe_enter`'s
       one-eval-per-bar guard (`_last_signal_ts`) is not yet set for this bar
       because it early-returns while a position is open. So live CAN close on
       an adverse cross and open the opposite side on the same bar; `next()`
       returns straight after `position.close()` and cannot re-enter until T+1.
       MEASURED 2026-08-11 over 2022-01-01..2026-08-11: 236 trades, 146 of them
       closed by this exit, and the opposite entry signal was true on the exit
       bar **0 times**. The entry gates (RSI extremes plus a volume spike and
       the 4H regime) simply never coincide with an adverse cross on a held
       position. Real in principle, never observed in 4.6 years.

    CALLER CONTRACT: bars_15m.iloc[-1] MUST be a CLOSED bar (drop Binance's
    forming last row first). Returns (should_exit, debug). Does not raise on
    market data; returns False for a flat/unknown side, during warmup, or on a
    NaN EMA. With require_trend on, a params dict lacking mf_trend_ema_period
    raises KeyError (boot validation guarantees the live dict has it).
    """
    if position_side not in ("long", "short"):
        return False, {"reason": "not_in_position", "side": position_side}

    s = params.get("strategy", {})
    # Default True to match DayTradeMultiFactorBTC.require_trend — see docstring.
    # Deliberately a literal, NOT `DayTradeMultiFactorBTC.require_trend`: that
    # class subclasses backtesting.Strategy, and this module is on the LIVE
    # import chain (bot.py -> bot_internals -> here), which today does not pull
    # `backtesting` in at all. Importing it for one bool would put a backtest-only
    # dependency inside the trading process. Drift is instead made loud by
    # tests/test_v1_trend_exit.py, which asserts this default equals the class
    # attribute — detection without the coupling.
    if not s.get("require_trend", True):
        return False, {"reason": "require_trend_off", "side": position_side}

    period = int(s["mf_trend_ema_period"])
    if len(bars_15m) < period:
        return False, {"reason": "warmup", "have": len(bars_15m), "need": period}

    close = bars_15m["Close"]
    trend_ema_v = ema(close, period).iloc[-1]
    cur_close = float(close.iloc[-1])

    # next() guards on isfinite(t) only — min_periods=period already makes the
    # warmup rows NaN, so this catches a short/ragged frame that slipped the
    # length check above.
    if not np.isfinite(trend_ema_v):
        return False, {"reason": "nan_indicators", "side": position_side}

    debug = {
        "side": position_side,
        "cur_close": cur_close,
        "trend_ema": float(trend_ema_v),
        "trend_ema_period": period,
    }

    # A NaN close makes BOTH strict comparisons below False, so next() holds
    # through it as well — it never guards close_v. Labelling it is therefore a
    # zero-behaviour-change addition that makes "held on bad data"
    # distinguishable from "held because price is on the right side" in logs.
    if not np.isfinite(cur_close):
        return False, {**debug, "reason": "nan_close"}

    if position_side == "long" and cur_close < trend_ema_v:
        return True, {**debug, "reason": "trend_exit_long"}
    if position_side == "short" and cur_close > trend_ema_v:
        return True, {**debug, "reason": "trend_exit_short"}
    return False, {**debug, "reason": "hold"}
