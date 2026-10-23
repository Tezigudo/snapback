"""
Strategy v1: "RSI Extreme + EMA Confluence + Volume + Funding"

Multi-timeframe deterministic logic. Entry timeframe = 15m, trend filter +
ATR-based exits come from 1h. Funding rate (8h cadence on Binance perps) is
forward-filled to the entry timeframe and used as a filter / cost.

Two pieces:

  1. `prepare_strategy_data()` — pure-pandas, no backtesting.py.
     Takes raw OHLCV at 15m + 1h plus funding history, returns ONE DataFrame
     indexed at 15m with all indicators attached. Crucially, ALL 1h indicators
     are computed at the close of each 1h bar and then shifted by one bar
     before reindexing so that a 15m bar T sees only the last FULLY CLOSED
     1h bar's value — no lookahead.

  2. `SnapbackBTC` — backtesting.py Strategy that reads the prepared columns.
     Order placed in next() at bar T fills at open of T+1 (trade_on_close=False
     in backtest.py), so the strategy never trades on information it shouldn't
     have at decision time.

LONG entry, ALL of:
    RSI(rsi_period, 15m) < rsi_long_threshold
    close > EMA(ema_period, 1h)
    volume(15m) > volume_multiple * SMA(volume_ma_period, vol)
    funding_rate <= funding_long_max

SHORT is the mirror.

Exit:
    TP at +/- atr_tp_multiple * ATR(atr_period, 1h)
    SL at -/+ atr_sl_multiple * ATR(atr_period, 1h)
    Time-stop at time_stop_bars entry-timeframe bars

For the snapback-v1 / multifactor-v1 / donchian-v3 families those values come
from the per-leg params YAML (config/README.md) and have no code default.
Generic indicator constants (e.g. MACD periods) and the multifactor-v2/v3
research variants keep their own class-level constants.

Sizing:
    risk_per_trade_pct * equity / sl_distance = BTC quantity
    Expressed as integer "fractional units" (1 unit = FRACTIONAL_UNIT BTC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from backtesting import Strategy

from exchange.env import REPO_ROOT

from .indicators import atr, ema, rsi, sma

FRACTIONAL_UNIT = 1e-6  # matches FractionalBacktest config in backtest.py


# Fields that every per-leg params YAML is expected to supply. They default to
# None here on purpose: a research run on a YAML that omits one must fail loudly
# (require_params) instead of silently using a number baked into the code.
_REQUIRED_NOTE = "see config/README.md"


@dataclass(frozen=True)
class StrategyParams:
    rsi_period: int | None = None
    rsi_long_threshold: float | None = None
    rsi_short_threshold: float | None = None
    ema_period: int | None = None
    volume_ma_period: int | None = None
    volume_multiple: float | None = None
    funding_long_max: float | None = None
    funding_short_min: float | None = None
    atr_period: int | None = None
    atr_tp_multiple: float | None = None
    atr_sl_multiple: float | None = None
    time_stop_bars: int | None = None
    risk_per_trade_pct: float | None = None
    leverage: int | None = None
    # Donchian-specific (ignored by snapback strategies)
    donchian_period_entry: int | None = None
    donchian_period_exit: int | None = None
    atr_trail_multiple: float | None = None
    # Carry-specific (ignored by other strategies)
    funding_threshold: float = 0.0002
    funding_exit_threshold: float = 0.00005
    sl_pct: float | None = None
    max_24h_change_pct: float = 100.0     # 100 = disabled (no filter)
    # Carry-v3 NEW tail-risk gates (100 / 100 = both disabled = v2 behaviour)
    atr_percentile_threshold: float = 100.0
    dd_halt_pct: float = 100.0
    # Carry-v4 NEW trend gate (0 = disabled = v3 behaviour)
    trend_ema_period: int = 0
    # FundingMomentum (fmom-v1) params
    momentum_lookback_bars: int = 96
    momentum_threshold: float = 0.0002
    tp_pct: float | None = None
    require_trend_align: bool = False
    # Donchian-v3 regime-gate params
    regime_ema_period: int | None = None
    regime_slope_window: int | None = None
    slope_trend_threshold_pct: float | None = None   # 0 = gate OFF
    # DayTradeMultiFactorBTC (multifactor-v1) params
    # NOTE: trend_ema_period above defaults to 0 for carry-v4 compat.
    # multifactor uses mf_trend_ema_period instead.
    mf_trend_ema_period: int | None = None
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    require_candlestick: bool | None = None
    require_macd: bool | None = None
    require_trend: bool | None = None
    require_funding_not_extreme: bool | None = None
    funding_extreme_threshold: float | None = None
    max_hold_bars: int | None = None
    use_mtf_4h_gate: bool | None = None
    # rider-v1 indicator periods (only read by DonchianRiderV1)
    rider_atr_period: int | None = None
    rider_ema_period: int | None = None
    mtf_4h_ema_period: int | None = None
    # multifactor-v2 (trailing stop) — legacy
    trail_activate_atr: float = 1.0
    trail_atr_multiple: float = 2.0
    # multifactor-v2 (TA confirmation) — current
    confirmations_required: int = 2
    swing_k: int = 3
    swing_lookback_bars: int | None = None   # v2/v3 keep their own class constant unless the YAML sets it
    trendline_max_distance_pct: float | None = None
    sr_max_distance_pct: float = 0.010
    sr_cluster_tolerance_pct: float = 0.005
    fib_max_distance_pct: float = 0.010
    # multifactor-v3 — adaptive risk gates (THRESHOLDS only; enable_* are
    # class-level switches on the V3* variant subclasses).
    max_distance_above_ema_pct: float = 0.20
    max_distance_below_ema_pct: float = 0.20
    atr_sl_k: float | None = None     # class-level per variant; never applied
    atr_tp_k: float | None = None     # class-level per variant; never applied
    vol_regime_lookback_days: int = 30
    vol_regime_max_pctile: float = 0.85

    @classmethod
    def from_yaml(cls, path: str | None = None) -> StrategyParams:
        """Research/backtest loader for a per-leg params YAML.

        - Missing file -> FileNotFoundError pointing at config/README.md (the
          real params files are untracked; copy a *.example.yaml and fill it).
        - A key that is PRESENT but null -> ValueError. An unfilled template must
          never silently backtest on code defaults that look like live params.
        - A key ABSENT from the file -> the dataclass default, which is None for
          every strategy-specific field. Whichever strategy/helper then needs it
          raises via require_params(), naming the key (e.g. donchian keys in a
          v1 file are None and only matter if you run a donchian strategy).
        """
        path = path or str(REPO_ROOT / "config" / "params.yaml")
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{path} not found. The per-leg params YAMLs are untracked: copy "
                "the matching config/*.example.yaml and fill in every null "
                "(see config/README.md).") from None
        if not isinstance(cfg, dict):
            raise ValueError(f"{path}: expected a mapping, got {type(cfg).__name__}")
        s = cfg.get("strategy")
        z = cfg.get("sizing")
        if not isinstance(s, dict) or not isinstance(z, dict):
            raise ValueError(f"{path}: 'strategy' and 'sizing' blocks are required "
                             "(see config/README.md)")
        nulls = sorted([f"strategy.{k}" for k, v in s.items() if v is None]
                       + [f"sizing.{k}" for k, v in z.items() if v is None])
        if nulls:
            raise ValueError(f"{path}: null value(s) {', '.join(nulls)} — fill them "
                             "in (no code default is substituted; see config/README.md)")

        def g(key, cast):
            # Absent key -> the dataclass default (None for strategy-specific
            # fields). Null values were refused above.
            v = s.get(key, getattr(cls, key))
            return v if v is None else cast(v)

        def gz(key, cast):
            return cast(z[key]) if key in z else getattr(cls, key)

        return cls(
            rsi_period=g("rsi_period", int),
            rsi_long_threshold=g("rsi_long_threshold", float),
            rsi_short_threshold=g("rsi_short_threshold", float),
            ema_period=g("ema_period", int),
            volume_ma_period=g("volume_ma_period", int),
            volume_multiple=g("volume_multiple", float),
            funding_long_max=g("funding_long_max", float),
            funding_short_min=g("funding_short_min", float),
            atr_period=g("atr_period", int),
            atr_tp_multiple=g("atr_tp_multiple", float),
            atr_sl_multiple=g("atr_sl_multiple", float),
            time_stop_bars=g("time_stop_bars", int),
            risk_per_trade_pct=gz("risk_per_trade_pct", float),
            leverage=gz("leverage", int),
            # multifactor-v1 fields
            sl_pct=g("sl_pct", float),
            tp_pct=g("tp_pct", float),
            mf_trend_ema_period=g("mf_trend_ema_period", int),
            require_trend=g("require_trend", bool),
            require_candlestick=g("require_candlestick", bool),
            require_macd=g("require_macd", bool),
            require_funding_not_extreme=g("require_funding_not_extreme", bool),
            funding_extreme_threshold=g("funding_extreme_threshold", float),
            max_hold_bars=g("max_hold_bars", int),
            use_mtf_4h_gate=g("use_mtf_4h_gate", bool),
            rider_atr_period=g("rider_atr_period", int),
            rider_ema_period=g("rider_ema_period", int),
            mtf_4h_ema_period=g("mtf_4h_ema_period", int),
            # donchian-v3 fields
            donchian_period_entry=g("donchian_period_entry", int),
            donchian_period_exit=g("donchian_period_exit", int),
            atr_trail_multiple=g("atr_trail_multiple", float),
            regime_ema_period=g("regime_ema_period", int),
            regime_slope_window=g("regime_slope_window", int),
            slope_trend_threshold_pct=g("slope_trend_threshold_pct", float),
        )


def require_params(obj, names, where: str) -> None:
    """Raise a clear ValueError if any of `names` is None on `obj`.

    `obj` is a StrategyParams or a Strategy class/instance whose attributes are
    filled from the params YAML. The message names the YAML key(s) so a research
    run on an incomplete YAML says exactly what to add.
    """
    missing = [n for n in names if getattr(obj, n, None) is None]
    if missing:
        keys = ", ".join(f"strategy.{n}" for n in missing)
        raise ValueError(
            f"{where}: {keys} not set. Strategy-specific values have no code "
            f"default: put them in the params YAML you are running ({_REQUIRED_NOTE}).")


def prepare_strategy_data(
    klines_15m: pd.DataFrame,
    klines_1h: pd.DataFrame,
    funding: pd.DataFrame,
    params: StrategyParams,
) -> pd.DataFrame:
    """Attach precomputed indicators to the 15m bars. No lookahead.

    Expects 15m and 1h DataFrames with columns open/high/low/close/volume
    (lowercase) indexed by tz-naive UTC time. Funding DataFrame has a
    `funding_rate` column indexed by funding_time.

    Returns a 15m DataFrame with capitalised OHLCV columns plus:
      RSI         : RSI(period) on 15m close
      VolSMA      : SMA(volume_ma_period) on 15m volume
      EMA_1h      : EMA(ema_period) on 1h close, shifted one 1h bar then
                    reindexed to 15m via ffill (so 15m at T uses last
                    fully-closed 1h bar)
      ATR_1h      : ATR(atr_period) on 1h H/L/C, same shift-then-ffill
      Funding     : funding_rate forward-filled from last known event
    """
    require_params(params, ("rsi_period", "volume_ma_period", "ema_period",
                            "atr_period"), "prepare_strategy_data")
    if klines_15m.empty:
        raise ValueError("klines_15m is empty")
    if klines_1h.empty:
        raise ValueError("klines_1h is empty")

    df = klines_15m.copy()
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    one_h = klines_1h.copy()
    one_h.columns = [c.capitalize() for c in one_h.columns]
    if one_h.index.tz is not None:
        one_h.index = one_h.index.tz_convert("UTC").tz_localize(None)

    # 15m indicators — directly usable since RSI[T] reflects close at T+15m,
    # and backtesting.py with trade_on_close=False fills at open of T+1.
    df["RSI"] = rsi(df["Close"], params.rsi_period)
    df["VolSMA"] = sma(df["Volume"], params.volume_ma_period)

    # 1h indicators — shift by 1 so the value at bar T represents the
    # information that became available at T-1's close (= T's open).
    # Then reindex to 15m with ffill.
    ema_1h = ema(one_h["Close"], params.ema_period).shift(1)
    atr_1h = atr(one_h["High"], one_h["Low"], one_h["Close"], params.atr_period).shift(1)
    df["EMA_1h"] = ema_1h.reindex(df.index, method="ffill")
    df["ATR_1h"] = atr_1h.reindex(df.index, method="ffill")

    # Funding rate — most recent known event, ffilled. No shift: the rate is
    # published at fundingTime; a 15m bar at or after fundingTime can see it.
    if funding is None or funding.empty:
        df["Funding"] = np.nan
    else:
        f = funding.copy()
        if f.index.tz is not None:
            f.index = f.index.tz_convert("UTC").tz_localize(None)
        df["Funding"] = f["funding_rate"].reindex(df.index, method="ffill")

    return df


class SnapbackBTC(Strategy):
    """v1 strategy. Params come from class attributes — backtest.py overrides
    them from `config/params.yaml` before instantiating."""

    # Filled by backtest.py before Backtest() is constructed.
    rsi_long_threshold = None
    rsi_short_threshold = None
    volume_multiple = None
    funding_long_max = None
    funding_short_min = None
    atr_tp_multiple = None
    atr_sl_multiple = None
    time_stop_bars = None
    risk_per_trade_pct = None
    leverage = None
    allow_shorts = True

    def init(self) -> None:
        require_params(self, ("rsi_long_threshold", "rsi_short_threshold",
                              "volume_multiple", "funding_long_max",
                              "funding_short_min", "atr_tp_multiple",
                              "atr_sl_multiple", "time_stop_bars",
                              "risk_per_trade_pct", "leverage"), "SnapbackBTC")
        # Indicators are read directly from self.data in next(). We deliberately
        # do NOT register them via self.I() — FractionalBacktest would scale
        # every registered indicator by 1/fractional_unit. We use plain Backtest
        # with a large starting cash instead so 1+ BTC fits naturally.
        self._entry_bar: int | None = None

    def _position_units(self, sl_distance: float, price: float) -> int:
        """Risk-based sizing in integer BTC units, clamped to broker margin.

        Backtest size = int N means N units of the asset. We compute N from the
        risk-per-trade target, then clamp to 95% of max-margin notional so the
        broker won't reject for insufficient margin.
        """
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        units = int(min(target_btc, max_btc))
        return max(units, 0)

    def next(self) -> None:
        # Skip until all warm-up indicators are ready.
        rsi_v = self.data.RSI[-1]
        ema_v = self.data.EMA_1h[-1]
        atr_v = self.data.ATR_1h[-1]
        vol_sma_v = self.data.VolSMA[-1]
        funding_v = self.data.Funding[-1]
        close_v = self.data.Close[-1]
        vol_v = self.data.Volume[-1]

        if any(
            v is None or not np.isfinite(v)
            for v in (rsi_v, ema_v, atr_v, vol_sma_v, funding_v)
        ):
            return

        # Position management — handle time-stop while in a trade.
        if self.position:
            if self._entry_bar is not None:
                age = len(self.data) - self._entry_bar
                if age >= self.time_stop_bars:
                    self.position.close()
                    self._entry_bar = None
            return

        # Entry filters
        volume_ok = vol_v > self.volume_multiple * vol_sma_v
        long_filters = (
            rsi_v < self.rsi_long_threshold
            and close_v > ema_v
            and volume_ok
            and funding_v <= self.funding_long_max
        )
        short_filters = (
            self.allow_shorts
            and rsi_v > self.rsi_short_threshold
            and close_v < ema_v
            and volume_ok
            and funding_v >= self.funding_short_min
        )

        if long_filters:
            sl_dist = self.atr_sl_multiple * atr_v
            sl = close_v - sl_dist
            tp = close_v + self.atr_tp_multiple * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v < tp:
                self.buy(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)
        elif short_filters:
            sl_dist = self.atr_sl_multiple * atr_v
            sl = close_v + sl_dist
            tp = close_v - self.atr_tp_multiple * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and tp < close_v < sl:
                self.sell(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)
