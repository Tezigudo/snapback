"""
DayTradeMultiFactorBTC — multi-factor day-trading-style strategy on 15m.

Every tunable (indicator periods, thresholds, stop/target geometry, hold
limit, which filters are on) comes from the per-leg params YAML; this class
carries no numeric defaults for them (config/README.md). A run on a YAML that
omits one raises a ValueError naming the key.

Filters, each switchable from the YAML:
  - RSI(rsi_period) asymmetric thresholds
  - Volume confirmation (volume > volume_multiple x SMA(volume_ma_period))
  - Candlestick pattern (bullish/bearish engulfing or hammer)
  - Trend filter (close vs EMA(mf_trend_ema_period) on the entry timeframe)
  - Funding-rate "sentiment" gate (refuse crowded-trade direction)
  - MACD histogram confirmation (momentum agreement)
  - Higher-timeframe EMA regime gate (use_mtf_4h_gate / mtf_4h_ema_period)

Entry: every enabled filter must agree for a LONG; SHORT is the mirror
(RSI above its short threshold, bearish engulfing, close below the EMAs,
funding not extremely negative, MACD histogram < 0).

Exit rules:
  - fixed-percentage stop loss (sl_pct) and take profit (tp_pct)
  - time stop (max_hold_bars)
  - adverse trend cross (close moves to the wrong side of the trend EMA)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import (
    bearish_engulfing,
    bullish_engulfing,
    ema,
    hammer,
    macd,
    rsi,
    sma,
)
from strategy.signals import require_params

_REPO_ROOT = Path(__file__).resolve().parent.parent
_4H_PARQUET_DEFAULT = _REPO_ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"


def _build_4h_ema_aligned(
    dates_15m: pd.DatetimeIndex,
    parquet_path: Path,
    ema_period: int,
) -> np.ndarray:
    """Load 4H bars, compute EMA(ema_period), align to 15m timestamps with lookahead safety.

    TIMING CONVENTION (critical):
        The parquet index is bar-OPEN time. A 4H bar opening at T closes at T+4h.
        We may only use the EMA of that bar at 15m timestamps >= T+4h.

    Implementation mirrors the divergence-v2 strategy's 4H EMA alignment:
        1. Load FULL 4H parquet (all history → ensures the EMA is warmed up
           before any OOS window begins).
        2. Compute EMA(N) on 4H closes.
        3. Re-index the EMA series by CLOSE timestamps (open_time + 4h).
        4. pd.merge_asof(direction="backward") against the 15m index — each 15m
           bar receives the EMA of the most-recently CLOSED 4H bar. The EMA of
           the still-open 4H bar is unreachable, hence no lookahead.
        5. Both indices stripped to tz-naive + datetime64[us] precision (pandas
           refuses merges across mixed precision / tz).

    Returns an ndarray aligned 1:1 to dates_15m (NaN during warm-up).
    """
    df4h = pd.read_parquet(parquet_path)
    if df4h.index.tz is not None:
        df4h.index = df4h.index.tz_localize(None)
    # parquet columns are lowercased on disk; backtesting.py capitalises for
    # the 15m frame but we read 4H directly here so use the on-disk name.
    close_col = "close" if "close" in df4h.columns else "Close"
    ema_4h = ema(df4h[close_col], ema_period).values

    close_times = df4h.index + pd.Timedelta(hours=4)
    close_times_us = pd.DatetimeIndex(close_times.astype("datetime64[us]"))

    left_idx_raw = dates_15m
    if left_idx_raw.tz is not None:
        left_idx_raw = left_idx_raw.tz_localize(None)
    left_idx = pd.DatetimeIndex(left_idx_raw.astype("datetime64[us]"))

    right = pd.DataFrame({"ema4h": ema_4h}, index=close_times_us).sort_index()
    left = pd.DataFrame(index=left_idx)
    merged = pd.merge_asof(
        left, right,
        left_index=True, right_index=True,
        direction="backward",
    )
    return merged["ema4h"].values


class DayTradeMultiFactorBTC(Strategy):
    # Everything strategy-specific is filled from the params YAML by
    # backtest._apply_params_to_class (None = must be supplied).
    # --- entry thresholds ---
    rsi_period = None
    rsi_long_threshold = None
    rsi_short_threshold = None

    # --- volume confirmation ---
    volume_ma_period = None
    volume_multiple = None

    # --- candlestick / trend ---
    require_candlestick = None
    mf_trend_ema_period = None      # (name avoids collision with carry-v4 trend_ema_period
                                    #  which can be 0 in shared StrategyParams)
    require_trend = None

    # --- MACD momentum filter ---
    require_macd = None
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # --- funding sentiment proxy (refuse crowded trade) ---
    funding_extreme_threshold = None
    require_funding_not_extreme = None

    # --- exits ---
    sl_pct = None
    tp_pct = None
    max_hold_bars = None

    # --- sizing ---
    risk_per_trade_pct = 2.0
    leverage = None                 # set by run_backtest
    allow_shorts = True

    # --- higher-timeframe EMA regime gate (additive) ---
    # Lookahead-safe alignment: see _build_4h_ema_aligned() module docstring.
    # Long entries require the entry-TF close above that EMA; shorts below.
    use_mtf_4h_gate = None
    mtf_4h_ema_period = None
    mtf_4h_parquet_path = str(_4H_PARQUET_DEFAULT)

    _REQUIRED = ("rsi_period", "rsi_long_threshold", "rsi_short_threshold",
                 "volume_ma_period", "volume_multiple", "require_candlestick",
                 "mf_trend_ema_period", "require_trend", "require_macd",
                 "funding_extreme_threshold", "require_funding_not_extreme",
                 "sl_pct", "tp_pct", "max_hold_bars", "leverage",
                 "use_mtf_4h_gate")

    def init(self) -> None:
        require_params(self, self._REQUIRED, type(self).__name__)
        if self.use_mtf_4h_gate:
            require_params(self, ("mtf_4h_ema_period",), type(self).__name__)
        close = pd.Series(self.data.Close)
        open_ = pd.Series(self.data.Open)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        self._rsi = rsi(close, self.rsi_period).values
        self._vol_sma = sma(volume, self.volume_ma_period).values
        self._trend_ema = ema(close, self.mf_trend_ema_period).values
        _, _, self._macd_hist = (s.values for s in
                                  macd(close, self.macd_fast, self.macd_slow, self.macd_signal))
        self._bull_engulf = bullish_engulfing(open_, high, low, close).values
        self._bear_engulf = bearish_engulfing(open_, high, low, close).values
        self._hammer = hammer(open_, high, low, close).values
        self._entry_bar: int | None = None

        # Higher-TF EMA regime filter (additive). Loaded from the FULL 4H parquet so
        # the EMA is warmed-up before any backtest slice begins. Aligned to the
        # 15m index via backward merge_asof on bar-CLOSE timestamps — same
        # convention as strategy/signals_divergence_v2.py.
        if self.use_mtf_4h_gate:
            dates = pd.DatetimeIndex(self.data.index)
            self._ema_4h = _build_4h_ema_aligned(
                dates,
                Path(self.mtf_4h_parquet_path),
                self.mtf_4h_ema_period,
            )
        else:
            self._ema_4h = None

    def _position_units(self, price: float, sl_distance: float) -> int:
        # NOTE: backtesting.py 0.6.5 only accepts integer units.
        # Fractional 0.001-BTC sizing is implemented via HARNESS-level price
        # scaling (see tools/_fractional_run.py). Under scaling: 1 returned
        # "unit" == 0.001 BTC, matching Binance USDT-M perp qty_step.
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    def _long_signal(self, i: int) -> bool:
        # 1. RSI oversold
        if not (np.isfinite(self._rsi[i]) and self._rsi[i] < self.rsi_long_threshold):
            return False
        # 2. Volume confirmation
        vol_sma_v = self._vol_sma[i]
        if not (np.isfinite(vol_sma_v) and self.data.Volume[-1] > self.volume_multiple * vol_sma_v):
            return False
        # 3. Trend filter
        if self.require_trend:
            t = self._trend_ema[i]
            if not (np.isfinite(t) and self.data.Close[-1] > t):
                return False
        # 4. Candlestick pattern
        if self.require_candlestick:
            if not (bool(self._bull_engulf[i]) or bool(self._hammer[i])):
                return False
        # 5. MACD agreement
        if self.require_macd:
            h = self._macd_hist[i]
            if not (np.isfinite(h) and h > 0):
                return False
        # 6. Funding sentiment (refuse crowded long)
        if self.require_funding_not_extreme:
            try:
                f = self.data.Funding[-1]
                if np.isfinite(f) and f > self.funding_extreme_threshold:
                    return False
            except (AttributeError, IndexError):
                pass
        # 7. Higher-TF EMA regime gate. Closes the loop on the trend gate
        # already present by also requiring the higher timeframe regime to
        # agree. Long requires entry-TF close > the higher-TF EMA.
        if self.use_mtf_4h_gate and self._ema_4h is not None:
            ema_4h_v = self._ema_4h[i]
            if not (np.isfinite(ema_4h_v) and self.data.Close[-1] > ema_4h_v):
                return False
        return True

    def _short_signal(self, i: int) -> bool:
        if not self.allow_shorts:
            return False
        if not (np.isfinite(self._rsi[i]) and self._rsi[i] > self.rsi_short_threshold):
            return False
        vol_sma_v = self._vol_sma[i]
        if not (np.isfinite(vol_sma_v) and self.data.Volume[-1] > self.volume_multiple * vol_sma_v):
            return False
        if self.require_trend:
            t = self._trend_ema[i]
            if not (np.isfinite(t) and self.data.Close[-1] < t):
                return False
        if self.require_candlestick:
            if not bool(self._bear_engulf[i]):
                return False
        if self.require_macd:
            h = self._macd_hist[i]
            if not (np.isfinite(h) and h < 0):
                return False
        if self.require_funding_not_extreme:
            try:
                f = self.data.Funding[-1]
                if np.isfinite(f) and f < -self.funding_extreme_threshold:
                    return False
            except (AttributeError, IndexError):
                pass
        # 7. Higher-TF EMA regime gate (short mirror).
        if self.use_mtf_4h_gate and self._ema_4h is not None:
            ema_4h_v = self._ema_4h[i]
            if not (np.isfinite(ema_4h_v) and self.data.Close[-1] < ema_4h_v):
                return False
        return True

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = self.data.Close[-1]

        # Position management: time stop + adverse-trend exit
        if self.position:
            if self._entry_bar is not None:
                if (i - self._entry_bar) >= self.max_hold_bars:
                    self.position.close()
                    self._entry_bar = None
                    return
            # Adverse-trend exit
            if self.require_trend:
                t = self._trend_ema[i]
                if np.isfinite(t):
                    if self.position.is_long and close_v < t:
                        self.position.close()
                        self._entry_bar = None
                        return
                    if self.position.is_short and close_v > t:
                        self.position.close()
                        self._entry_bar = None
                        return
            return

        # Entry
        sl_dist = self.sl_pct * close_v
        tp_dist = self.tp_pct * close_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if self._long_signal(i):
            self.buy(size=units, sl=close_v - sl_dist, tp=close_v + tp_dist)
            self._entry_bar = i
        elif self._short_signal(i):
            self.sell(size=units, sl=close_v + sl_dist, tp=close_v - tp_dist)
            self._entry_bar = i
