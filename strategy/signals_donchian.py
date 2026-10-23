"""
DonchianBreakoutBTC — classic turtle-style trend-following on 1h Donchian
channels, evaluated on 15m bars.

OPPOSITE hypothesis vs snapback. snapback assumes BTC mean-reverts on 15m
inside an EMA-defined trend; Donchian assumes that when price punches
through a multi-day high/low it keeps going. If both fail OOS, the
strategy family is unrelated to whether you mean-revert or trend-follow —
it's the timeframe / asset combination that has no easy edge.

Entry (long):  15m close > rolling Donchian-upper of last N 1h closes
Entry (short): 15m close < rolling Donchian-lower of last N 1h closes
Exit:          opposite direction's M-bar Donchian channel (M < N)
SL:            entry_price ± atr_sl_multiple × ATR(atr_period, 1h)
No TP — let winners run; this is trend-following's whole pitch.

All 1h-derived columns are computed at 1h close and SHIFTED BY ONE 1h bar
before reindexing onto 15m, so a 15m bar at time T only sees a Donchian
channel computed from 1h bars strictly before T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import atr, ema
from strategy.signals import require_params


def attach_donchian(
    df_15m: pd.DataFrame,
    klines_1h: pd.DataFrame,
    period_entry: int | None = None,
    period_exit: int | None = None,
    atr_period: int | None = None,
) -> pd.DataFrame:
    """Attach DonchianUpper/Lower/ExitUpper/ExitLower/ATR_1h columns to df_15m.

    df_15m should already be capitalised + tz-naive (from prepare_strategy_data).
    `period_entry`, `period_exit` and `atr_period` have no default: pass them from the params
    YAML (donchian_period_exit, atr_period).
    """
    if period_entry is None or period_exit is None or atr_period is None:
        raise ValueError(
            "attach_donchian: donchian_period_entry, donchian_period_exit and atr_period have no code "
            "default — pass them from the params YAML (see config/README.md).")
    one_h = klines_1h.copy()
    one_h.columns = [c.capitalize() for c in one_h.columns]
    if one_h.index.tz is not None:
        one_h.index = one_h.index.tz_convert("UTC").tz_localize(None)

    upper = one_h["Close"].rolling(period_entry, min_periods=period_entry).max().shift(1)
    lower = one_h["Close"].rolling(period_entry, min_periods=period_entry).min().shift(1)
    exit_u = one_h["Close"].rolling(period_exit, min_periods=period_exit).max().shift(1)
    exit_l = one_h["Close"].rolling(period_exit, min_periods=period_exit).min().shift(1)
    atr_v = atr(one_h["High"], one_h["Low"], one_h["Close"], atr_period).shift(1)

    out = df_15m.copy()
    out["DonchianUpper"] = upper.reindex(out.index, method="ffill")
    out["DonchianLower"] = lower.reindex(out.index, method="ffill")
    out["DonchianExitUpper"] = exit_u.reindex(out.index, method="ffill")
    out["DonchianExitLower"] = exit_l.reindex(out.index, method="ffill")
    if "ATR_1h" not in out.columns:
        out["ATR_1h"] = atr_v.reindex(out.index, method="ffill")
    return out


class DonchianBreakoutBTC(Strategy):
    """v1 + ATR trailing stop and high-water-mark exit.

    The Donchian exit channel is *itself* a trailing mechanism, but it only
    updates once per 1h bar and only reflects the lowest of the last M
    closes. A separate ATR-distance trailing stop catches faster reversals
    that happen within a 1h bar (or that don't quite breach the channel
    but blow through the entry's risk envelope). Both exits coexist —
    whichever fires first wins.
    """

    # Strategy-specific values come from the params YAML (None = must be set).
    donchian_period_entry = None
    donchian_period_exit = None
    atr_sl_multiple = None     # initial stop distance
    atr_trail_multiple = None  # 0 = no trailing; >0 = trail SL at high - K*ATR
    risk_per_trade_pct = 2.0
    leverage = None            # set by run_backtest
    allow_shorts = True
    # Max hold time in entry-TF bars. 0 disables.
    time_stop_bars = None

    def init(self) -> None:
        require_params(self, ("donchian_period_entry", "donchian_period_exit",
                              "time_stop_bars", "atr_sl_multiple",
                              "atr_trail_multiple", "leverage"),
                       type(self).__name__)
        self._entry_bar: int | None = None
        self._high_water: float = 0.0   # for long trailing
        self._low_water: float = 0.0    # for short trailing

    def _position_units(self, sl_distance: float, price: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    def _time_stop_due(self) -> bool:
        return (self.time_stop_bars > 0 and self._entry_bar is not None
                and (len(self.data) - self._entry_bar) >= self.time_stop_bars)

    def _maybe_trail(self, atr_v: float) -> None:
        """Ratchet the trade SL toward high_water - K*ATR (long) or low_water + K*ATR (short)."""
        if self.atr_trail_multiple <= 0 or not self.trades:
            return
        trade = self.trades[-1]
        if trade.is_long:
            new_sl = self._high_water - self.atr_trail_multiple * atr_v
            if trade.sl is None or new_sl > trade.sl:
                trade.sl = new_sl
        else:
            new_sl = self._low_water + self.atr_trail_multiple * atr_v
            if trade.sl is None or new_sl < trade.sl:
                trade.sl = new_sl

    def next(self) -> None:
        upper = self.data.DonchianUpper[-1]
        lower = self.data.DonchianLower[-1]
        exit_upper = self.data.DonchianExitUpper[-1]
        exit_lower = self.data.DonchianExitLower[-1]
        atr_v = self.data.ATR_1h[-1]
        close_v = self.data.Close[-1]
        high_v = self.data.High[-1]
        low_v = self.data.Low[-1]

        if any(
            v is None or not np.isfinite(v)
            for v in (upper, lower, exit_upper, exit_lower, atr_v)
        ):
            return

        if self.position:
            if self.position.is_long:
                self._high_water = max(self._high_water, high_v)
            else:
                self._low_water = min(self._low_water if self._low_water > 0 else low_v, low_v)
            self._maybe_trail(atr_v)

            # Time stop — max-hold safety. Prevents indefinite positions when
            # the Donchian exit channel never fires (e.g. a sustained micro-trend
            # that doesn't pierce the opposite N-bar channel for weeks).
            if self._time_stop_due():
                self.position.close()
                self._entry_bar = None
                return

            # Donchian channel exit
            if self.position.is_long and close_v < exit_lower:
                self.position.close()
                self._entry_bar = None
            elif self.position.is_short and close_v > exit_upper:
                self.position.close()
                self._entry_bar = None
            return

        sl_dist = self.atr_sl_multiple * atr_v

        if close_v > upper:
            sl = close_v - sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl)
                self._entry_bar = len(self.data)
                self._high_water = high_v
                self._low_water = 0.0
        elif self.allow_shorts and close_v < lower:
            sl = close_v + sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl > close_v:
                self.sell(size=units, sl=sl)
                self._entry_bar = len(self.data)
                self._high_water = 0.0
                self._low_water = low_v


class DonchianBreakoutBTCv2(DonchianBreakoutBTC):
    """Same behaviour as v1 — separate class so the sweep machinery can mutate
    class attributes without touching the v1 baseline. v2 enables ATR trailing
    via the wider sweep grid in `config/sweep_donchian_v2.yaml`."""


class DonchianBreakoutBTCv3(DonchianBreakoutBTC):
    """v2 + DIRECTIONAL regime gate.

    Uses signed EMA-slope: positive = uptrend, negative = downtrend.
    Entry rules:
      - LONG breakout requires slope >= +slope_trend_threshold_pct
      - SHORT breakout requires slope <= -slope_trend_threshold_pct
      - Chop (|slope| < threshold) = no entry in either direction

    Two reasons this is better than the |slope| gate:
      1. It refuses to short during an uptrend (and vice versa), which is
         the most common Donchian failure in chop — a small downtick that
         technically pierces the lower channel but is just a pullback in
         a larger uptrend.
      2. The |slope| gate's "high threshold blocks early-trend entries"
         pathology is avoided — we don't gate on magnitude alone.

    slope_trend_threshold_pct=0 disables the gate (v2 behaviour)."""

    regime_ema_period: int | None = None
    regime_slope_window: int | None = None
    slope_trend_threshold_pct: float | None = None   # 0 = gate OFF

    def init(self) -> None:
        super().init()
        require_params(self, ("slope_trend_threshold_pct",), type(self).__name__)
        if self.slope_trend_threshold_pct > 0:
            import pandas as pd_
            from strategy.regime_classifier import ema_slope_signed
            require_params(self, ("regime_ema_period", "regime_slope_window"),
                           type(self).__name__)
            close = pd_.Series(self.data.Close)
            self._regime_slope = ema_slope_signed(
                close,
                ema_period=self.regime_ema_period,
                slope_window=self.regime_slope_window,
            ).values
        else:
            self._regime_slope = None

    def _slope_now(self) -> float | None:
        if self._regime_slope is None:
            return None
        s = self._regime_slope[len(self.data) - 1]
        import numpy as _np
        return float(s) if _np.isfinite(s) else None

    def next(self) -> None:
        upper = self.data.DonchianUpper[-1]
        lower = self.data.DonchianLower[-1]
        exit_upper = self.data.DonchianExitUpper[-1]
        exit_lower = self.data.DonchianExitLower[-1]
        atr_v = self.data.ATR_1h[-1]
        close_v = self.data.Close[-1]
        high_v = self.data.High[-1]
        low_v = self.data.Low[-1]

        import numpy as _np
        if any(
            v is None or not _np.isfinite(v)
            for v in (upper, lower, exit_upper, exit_lower, atr_v)
        ):
            return

        if self.position:
            if self.position.is_long:
                self._high_water = max(self._high_water, high_v)
            else:
                self._low_water = min(self._low_water if self._low_water > 0 else low_v, low_v)
            self._maybe_trail(atr_v)
            # Time stop — mirrors live (bot._maybe_time_stop).
            # Live ticks time_stop -> channel_exit -> _maybe_enter, so after a
            # time-stop close the SAME tick evaluates entry on the just-closed
            # bar. Hence no `return` after the time stop: fall through to the
            # entry logic. The channel-exit path still returns (known gap: live
            # can re-enter on the same tick after a channel exit, this cannot).
            # Bar math: signal bar E, fills at open E+1; this fires on the close
            # of bar E+N (N = time_stop_bars) and fills at open E+N+1, i.e. N bars
            # after the entry fill = live age >= N entry bars.
            if self._time_stop_due():
                self.position.close()
                self._entry_bar = None
            else:
                if self.position.is_long and close_v < exit_lower:
                    self.position.close()
                    self._entry_bar = None
                elif self.position.is_short and close_v > exit_upper:
                    self.position.close()
                    self._entry_bar = None
                return

        sl_dist = self.atr_sl_multiple * atr_v
        slope = self._slope_now()
        gate_on = self.slope_trend_threshold_pct > 0

        if close_v > upper:
            if gate_on and (slope is None or slope < self.slope_trend_threshold_pct):
                return  # don't long unless in confirmed uptrend
            sl = close_v - sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl)
                self._entry_bar = len(self.data)
                self._high_water = high_v
                self._low_water = 0.0
        elif self.allow_shorts and close_v < lower:
            if gate_on and (slope is None or slope > -self.slope_trend_threshold_pct):
                return  # don't short unless in confirmed downtrend
            sl = close_v + sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl > close_v:
                self.sell(size=units, sl=sl)
                self._entry_bar = len(self.data)
                self._high_water = 0.0
                self._low_water = low_v


# ---------------------------------------------------------------------------
# Donchian Trend-Rider v1  (native 4h, LONG-only, fixed TP bracket)
# ---------------------------------------------------------------------------

def attach_rider(
    df: pd.DataFrame,
    donchian_n: int = 55,
    atr_period: int | None = None,
    ema_period: int | None = None,
) -> pd.DataFrame:
    """Attach RiderDonHi / RiderEma / RiderAtr columns to a capitalised, tz-naive
    DataFrame (native 4h bars).

    RiderDonHi = rolling max of HIGH over `donchian_n` bars, shifted 1 (channel
    excludes the current bar — mirrors hiwr build_breakout and rider_port_validate).
    RiderEma   = EMA(close, ema_period).
    RiderAtr   = Wilder ATR(atr_period) on High/Low/Close.

    Prefixed names avoid any collision with the existing DonchianUpper/ATR_1h
    columns used by donchian-v3.
    """
    if atr_period is None or ema_period is None:
        raise ValueError("attach_rider: strategy.rider_atr_period and strategy.rider_ema_period "
                         "have no code default — set them in the params YAML "
                         "(config/README.md) or pass them.")
    out = df.copy()
    out["RiderDonHi"] = out["High"].rolling(donchian_n, min_periods=donchian_n).max().shift(1)
    out["RiderEma"] = ema(out["Close"], ema_period)
    out["RiderAtr"] = atr(out["High"], out["Low"], out["Close"], atr_period)
    return out


class DonchianRiderV1(Strategy):
    """4h Donchian trend-rider: long-only, small ATR stop, large fixed ATR TP.

    Entry:    close > RiderDonHi  AND  close > RiderEma
    Stop:     sl = close_v - rider_sl_atr * ATR(rider_atr_period)   (anchored to signal-bar close)
    Target:   tp = close_v + rider_tp_atr * ATR(rider_atr_period)   (fixed bracket — no trailing exit)
    Trail:    optional chandelier (rider_trail_atr >= 5); default off (0.0)
    Time-stop: rider_time_stop_bars bars

    Sizing mirrors donchian-v3: risk_per_trade_pct of equity / sl_distance,
    capped at leverage * equity * 0.95 / price.

    Leverage is set by run_backtest from the params YAML (sizing.leverage).
    """

    # All params are rider-prefixed so _apply_params_to_class cannot
    # accidentally clobber them with values from params.yaml.
    rider_sl_atr: float = 1.0
    rider_tp_atr: float = 8.0
    rider_trail_atr: float = 0.0       # 0 = no trail; >=5 = chandelier
    rider_time_stop_bars: int = 200
    rider_donchian_n: int = 55
    rider_atr_period: int | None = None
    rider_ema_period: int | None = None
    rider_risk_per_trade_pct: float = 2.0  # prefixed to avoid apply-list clobber
    # leverage IS set by run_backtest via STRATEGIES[name].leverage = eff_leverage
    leverage: int | None = None

    def init(self) -> None:
        require_params(self, ("rider_atr_period", "rider_ema_period", "leverage"),
                       type(self).__name__)
        self._entry_bar: int | None = None
        self._high_water: float = 0.0

    def _position_units(self, sl_distance: float, price: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.rider_risk_per_trade_pct / 100.0)
        target = risk_amount / sl_distance
        cap = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target, cap)), 0)

    def next(self) -> None:
        close_v = self.data.Close[-1]
        high_v = self.data.High[-1]
        don_hi = self.data.RiderDonHi[-1]
        ema_v = self.data.RiderEma[-1]
        atr_v = self.data.RiderAtr[-1]

        if any(v is None or not np.isfinite(v) for v in (don_hi, ema_v, atr_v)):
            return

        if self.position:
            if self.rider_time_stop_bars > 0 and self._entry_bar is not None:
                if (len(self.data) - self._entry_bar) >= self.rider_time_stop_bars:
                    self.position.close()
                    self._entry_bar = None
                    return
            if self.rider_trail_atr > 0 and self.trades:
                self._high_water = max(self._high_water, high_v)
                new_sl = self._high_water - self.rider_trail_atr * atr_v
                tr = self.trades[-1]
                if tr.sl is None or new_sl > tr.sl:
                    tr.sl = new_sl
            return

        if close_v > don_hi and close_v > ema_v:
            sl_dist = self.rider_sl_atr * atr_v
            sl = close_v - sl_dist
            tp = close_v + self.rider_tp_atr * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)
                self._high_water = high_v
