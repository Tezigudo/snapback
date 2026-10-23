"""
Indicators with no external dependency beyond pandas/numpy.

We intentionally avoid pandas-ta here because (a) it ships unstable across
numpy 1/2, (b) it's easy to misuse without realising it inserts lookahead via
fillna, and (c) writing these by hand documents the exact convention used by
the strategy. If you swap implementations later, update the tests in
tests/test_indicators.py to match.

All functions take a pandas Series indexed by time and return a Series of the
same shape. NaN-fill behaviour: leave NaNs at the head (warm-up). Never fill
NaNs in the middle of a series — that hides data gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RSI. Standard formula:
        RS = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)
    Uses Wilder smoothing (EWM with alpha=1/period, adjust=False) — matches
    most charting platforms.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0 (only gains), RSI = 100.
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return out


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average via pandas .ewm(span=...)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Wilder's ATR: smoothed True Range.
        TR = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = Wilder EMA(TR, period)
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("period must be > 0")
    return series.rolling(window=period, min_periods=period).mean()


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    n_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands. Returns (upper, mid, lower).

    mid   = SMA(close, period)
    upper = mid + n_std * rolling_std(close, period, ddof=0)
    lower = mid - n_std * rolling_std(close, period, ddof=0)

    Uses population std (ddof=0) to match TradingView / most charting platforms.
    NaN values populate the head for the first `period - 1` bars (warm-up).
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    mid = close.rolling(window=period, min_periods=period).mean()
    sd = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return upper, mid, lower


def keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 20,
    mult: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels. Returns (upper, mid, lower).

    mid   = EMA(close, ema_period)
    upper = mid + mult * ATR(high, low, close, atr_period)
    lower = mid - mult * ATR(high, low, close, atr_period)

    Convention matches Chester Keltner's original (EMA mid) with Linda Raschke's
    ATR-based bands. Used jointly with Bollinger Bands to detect "squeezes":
    when BB upper < KC upper AND BB lower > KC lower, realized volatility is
    suppressed and breakouts that follow tend to be outsized.

    NaN values populate the head until both EMA and ATR are warmed up.
    """
    if ema_period <= 0 or atr_period <= 0:
        raise ValueError("ema_period and atr_period must be > 0")
    mid = ema(close, ema_period)
    atr_v = atr(high, low, close, atr_period)
    upper = mid + mult * atr_v
    lower = mid - mult * atr_v
    return upper, mid, lower


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Classic MACD (Moving Average Convergence Divergence).

    Returns (macd_line, signal_line, histogram).
      macd_line = EMA(close, fast) - EMA(close, slow)
      signal_line = EMA(macd_line, signal)
      histogram = macd_line - signal_line

    Histogram > 0 = bullish momentum, < 0 = bearish.
    Histogram CROSSING zero is a tradeable signal (momentum direction change).
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bullish_engulfing(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Bullish engulfing pattern: today's GREEN body fully engulfs yesterday's RED body.

    Web research (multiple backtests on crypto): one of the highest win-rate
    reversal patterns when paired with volume confirmation (~65-75% in
    BTC/major pairs). Used here as a CONFIRMATION filter, not a sole entry
    signal.

    Returns a boolean Series — True on the bar where the engulfing completes.
    """
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    # Previous bar must be red (close < open).
    prev_red = prev_close < prev_open
    # Current bar must be green (close > open).
    cur_green = close > open_
    # Current body engulfs previous body (open below prev close, close above prev open).
    engulf = (open_ <= prev_close) & (close >= prev_open)
    return prev_red & cur_green & engulf


def bearish_engulfing(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Bearish engulfing: today's RED body fully engulfs yesterday's GREEN body."""
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    prev_green = prev_close > prev_open
    cur_red = close < open_
    engulf = (open_ >= prev_close) & (close <= prev_open)
    return prev_green & cur_red & engulf


def hammer(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    body_max_pct: float = 0.3,
    lower_shadow_min_x: float = 2.0,
) -> pd.Series:
    """Hammer pattern (bullish reversal): small body at top, long lower shadow.

    body_max_pct: body size as fraction of total range (0.3 = small body)
    lower_shadow_min_x: lower shadow must be >= N × body size
    """
    body = (close - open_).abs()
    rng = high - low
    upper_shadow = high - close.where(close > open_, open_)
    lower_shadow = open_.where(close > open_, close) - low
    small_body = body <= body_max_pct * rng.replace(0, np.nan)
    long_lower = lower_shadow >= lower_shadow_min_x * body.replace(0, np.nan)
    short_upper = upper_shadow <= body  # rejection from below
    return small_body & long_lower & short_upper


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    step: float = 0.02,
    max_af: float = 0.2,
) -> pd.Series:
    """Wilder's Parabolic SAR. Returns the SAR value series.

    Trend flips when price crosses the SAR. SAR sits below price in uptrend,
    above in downtrend. The acceleration factor (AF) grows by `step` each
    time a new extreme is made, capped at `max_af`.

    Used by traders as both a trailing stop and a trend-following signal.
    Sign convention: returned values are positive prices; the trend direction
    can be derived by comparing SAR to close (close > SAR = uptrend).
    """
    n = len(high)
    sar = np.full(n, np.nan, dtype=float)
    if n < 2:
        return pd.Series(sar, index=high.index)

    # Initial state: pick direction by first two bars
    is_long = high.iloc[1] >= high.iloc[0]
    af = step
    ep = high.iloc[0] if is_long else low.iloc[0]  # extreme point
    sar[1] = low.iloc[0] if is_long else high.iloc[0]

    for i in range(2, n):
        prev_sar = sar[i - 1]
        if is_long:
            new_sar = prev_sar + af * (ep - prev_sar)
            # SAR must not exceed prior two lows
            new_sar = min(new_sar, low.iloc[i - 1], low.iloc[i - 2])
            if low.iloc[i] < new_sar:
                # Flip to short
                is_long = False
                new_sar = ep
                ep = low.iloc[i]
                af = step
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_af)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = max(new_sar, high.iloc[i - 1], high.iloc[i - 2])
            if high.iloc[i] > new_sar:
                is_long = True
                new_sar = ep
                ep = high.iloc[i]
                af = step
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_af)
        sar[i] = new_sar
    return pd.Series(sar, index=high.index)


def swing_high_low(
    high: pd.Series, low: pd.Series, k: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """N-bar fractal swing high / low detector.

    A bar is a swing HIGH if its high is strictly greater than the high of
    the `k` bars on each side. Returns (swing_highs, swing_lows) as boolean
    Series aligned with the input index.

    k=3 is the classic Bill Williams fractal. Use larger k for "major"
    swings on bigger timeframes; smaller k for more sensitive intraday.
    """
    n = len(high)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    h = high.values
    lo = low.values
    for i in range(k, n - k):
        if h[i] == max(h[i - k:i + k + 1]) and (h[i] > h[i - 1] or h[i] > h[i + 1]):
            sh[i] = True
        if lo[i] == min(lo[i - k:i + k + 1]) and (lo[i] < lo[i - 1] or lo[i] < lo[i + 1]):
            sl[i] = True
    return pd.Series(sh, index=high.index), pd.Series(sl, index=low.index)


def trendline_from_swings(
    swing_mask: pd.Series, price: pd.Series, n_recent: int = 3,
) -> tuple[float, float] | None:
    """Fit a line through the most recent N swing points.

    Returns (slope_per_bar, intercept) for the line, suitable for
    `slope * bar_idx + intercept` to project the line.
    slope_per_bar is in price units per bar.

    Use with swing_highs for resistance lines, swing_lows for support.
    Returns None if fewer than n_recent swing points found.
    """
    idx = np.where(swing_mask.values)[0]
    if len(idx) < n_recent:
        return None
    idx = idx[-n_recent:]
    prices = price.values[idx]
    # Linear regression: slope, intercept = polyfit(x, y, 1)
    slope, intercept = np.polyfit(idx.astype(float), prices, 1)
    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# Multifactor-v2 helpers: trendline proximity, S/R zones, Fibonacci
# ---------------------------------------------------------------------------
def trendline_proximity_pct(
    price_now: float,
    slope: float,
    intercept: float,
    bar_now: int,
) -> float | None:
    """% distance from price_now to the projected trendline value at bar_now.

    Returns signed distance: price > line → positive, price < line → negative.
    Used by v2: "long signal valid only if price is within X% above a support
    trendline" (small positive distance), and the mirror for short.
    """
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    line_value = slope * bar_now + intercept
    if line_value <= 0:
        return None
    return (price_now - line_value) / line_value


def sr_zones(
    swing_prices: np.ndarray, cluster_tolerance_pct: float = 0.005,
) -> list[float]:
    """Cluster swing prices into S/R zones.

    Two swings within `cluster_tolerance_pct` of each other (default 0.5%) are
    merged into one zone, represented by their mean. Returns sorted list of
    zone centers. The more swings in a zone, the more significant the level —
    but this function returns one entry per zone regardless of weight.
    """
    if len(swing_prices) == 0:
        return []
    sorted_prices = np.sort(swing_prices)
    zones: list[list[float]] = [[float(sorted_prices[0])]]
    for p in sorted_prices[1:]:
        p = float(p)
        ref = zones[-1][0]
        if abs(p - ref) / ref <= cluster_tolerance_pct:
            zones[-1].append(p)
        else:
            zones.append([p])
    return [float(np.mean(z)) for z in zones]


def nearest_sr_zone_distance_pct(
    price: float, zones: list[float], direction: str = "below",
) -> float | None:
    """Signed distance from `price` to the NEAREST zone in the given direction.

    direction='below': only consider zones BELOW price (potential support).
    direction='above': only consider zones ABOVE price (potential resistance).
    Returns abs distance / price (positive). None if no zone in that direction.
    """
    if not zones or price <= 0:
        return None
    if direction == "below":
        candidates = [z for z in zones if z <= price]
        if not candidates:
            return None
        nearest = max(candidates)  # closest one below
    else:
        candidates = [z for z in zones if z >= price]
        if not candidates:
            return None
        nearest = min(candidates)  # closest one above
    return abs(price - nearest) / price


# Standard Fibonacci retracement levels (proper subset; 0% and 100% omitted).
FIB_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.786)


def fib_retracement_distance_pct(
    price: float,
    swing_high: float,
    swing_low: float,
    levels: tuple[float, ...] = FIB_LEVELS,
) -> tuple[float, float] | None:
    """Distance from `price` to nearest Fibonacci retracement level.

    Returns (nearest_level, distance_pct) where:
      - nearest_level ∈ levels (the Fib ratio that price is closest to)
      - distance_pct = |price - level_price| / price (always positive)

    For an uptrend retracement: levels measured DOWN from swing_high by
    `level * (swing_high - swing_low)`. For downtrend retracement: levels
    measured UP from swing_low. Direction is auto-inferred:
      - if swing_high > swing_low (price recently made a high then pulled back): UP retrace
      - if swing_low > swing_high (price recently made a low then bounced up): DOWN retrace
    """
    if swing_high == swing_low or price <= 0:
        return None
    rng = abs(swing_high - swing_low)
    if swing_high > swing_low:
        # Uptrend retracement: support levels below the high.
        level_prices = [swing_high - lvl * rng for lvl in levels]
    else:
        # Downtrend retracement: resistance levels above the low.
        # We swap names but the math is symmetric.
        level_prices = [swing_low - lvl * rng for lvl in levels]
        # Note: swing_low here is actually the lower of the two passed.
    best_level = None
    best_dist = float("inf")
    for lvl, lp in zip(levels, level_prices):
        d = abs(price - lp) / price
        if d < best_dist:
            best_dist = d
            best_level = lvl
    return (float(best_level), float(best_dist)) if best_level is not None else None


def recent_swing_pair(
    swing_high_mask: pd.Series,
    swing_low_mask: pd.Series,
    high: pd.Series,
    low: pd.Series,
    lookback_bars: int = 200,
) -> tuple[float, float] | None:
    """Return (most_recent_swing_high_price, most_recent_swing_low_price) within
    the last `lookback_bars` bars. None if either is missing.

    Used to compute the Fibonacci range. The CALLER decides which one is
    "first" (which side price retraced from).
    """
    n = len(high)
    start = max(0, n - lookback_bars)
    high_idx = np.where(swing_high_mask.values[start:])[0]
    low_idx = np.where(swing_low_mask.values[start:])[0]
    if len(high_idx) == 0 or len(low_idx) == 0:
        return None
    last_high = float(high.values[start + high_idx[-1]])
    last_low = float(low.values[start + low_idx[-1]])
    return last_high, last_low


# ---------------------------------------------------------------------------
# ADX dual-regime helpers (adx-dual-regime-v1)
# ---------------------------------------------------------------------------

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index. Returns the ADX line only (not +DI / -DI).

    Computation is two Wilder-smoothing passes so the warm-up period is roughly
    2 × `period` bars — NaN values populate the head until both passes converge.
    Traders use ADX above 25 as confirmation that a trend is strong enough to
    follow; below 25 indicates range / chop where mean-reversion strategies
    outperform breakout ones.

    Convention identical to ``atr()``: Wilder EWM with alpha = 1/period and
    adjust=False. No NaN fill mid-series — gaps propagate.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    # True Range (same as atr's TR)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional movement
    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder smooth: pass 1
    alpha = 1.0 / period
    sm_tr       = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    sm_plus_dm  = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    sm_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # Directional Indicators
    plus_di  = 100.0 * sm_plus_dm / sm_tr.replace(0.0, np.nan)
    minus_di = 100.0 * sm_minus_dm / sm_tr.replace(0.0, np.nan)

    # DX (directional index) — undefined when both DIs are zero
    di_sum  = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)

    # Wilder smooth: pass 2 → ADX
    adx_line = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return adx_line


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Supertrend indicator. Returns a DataFrame with columns `supertrend`
    (the trailing line) and `direction` (+1 = uptrend/long, -1 = downtrend/short).

    Standard construction:
        mid = (high + low) / 2
        basic_upper = mid + multiplier * ATR(period)
        basic_lower = mid - multiplier * ATR(period)
        final_upper[i] = basic_upper[i] if (basic_upper[i] < final_upper[i-1] or
                                             close[i-1] > final_upper[i-1])
                         else final_upper[i-1]
        final_lower[i] = basic_lower[i] if (basic_lower[i] > final_lower[i-1] or
                                             close[i-1] < final_lower[i-1])
                         else final_lower[i-1]
        direction flips to +1 when close crosses above final_upper, flips to -1
        when close crosses below final_lower, else carries forward.
        supertrend = final_lower when direction == +1, else final_upper.

    LOOKAHEAD: every value at bar i is computed from data through bar i only
    (close[i], high[i], low[i], ATR[i] which itself only uses bars <= i). The
    caller is responsible for reading `[-1]` (the last CLOSED bar) in
    `next()` — same convention as the rest of this file's indicators.

    NaN values populate the head until ATR(period) warms up (first `period`
    bars). Never filled mid-series.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    mid = (high + low) / 2.0
    atr_v = atr(high, low, close, period)
    basic_upper = mid + multiplier * atr_v
    basic_lower = mid - multiplier * atr_v

    n = len(close)
    final_upper = np.full(n, np.nan, dtype=float)
    final_lower = np.full(n, np.nan, dtype=float)
    direction = np.full(n, np.nan, dtype=float)
    st = np.full(n, np.nan, dtype=float)

    bu = basic_upper.values
    bl = basic_lower.values
    c = close.values

    first_valid = atr_v.first_valid_index()
    if first_valid is None:
        return pd.DataFrame(
            {"supertrend": st, "direction": direction}, index=close.index
        )
    start = close.index.get_loc(first_valid)

    final_upper[start] = bu[start]
    final_lower[start] = bl[start]
    # Initial direction: uptrend if close is above the lower band, else downtrend.
    direction[start] = 1.0 if c[start] > final_lower[start] else -1.0
    st[start] = final_lower[start] if direction[start] == 1.0 else final_upper[start]

    for i in range(start + 1, n):
        final_upper[i] = (
            bu[i]
            if (bu[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            bl[i]
            if (bl[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )

        if direction[i - 1] == 1.0:
            direction[i] = -1.0 if c[i] < final_lower[i] else 1.0
        else:
            direction[i] = 1.0 if c[i] > final_upper[i] else -1.0

        st[i] = final_lower[i] if direction[i] == 1.0 else final_upper[i]

    return pd.DataFrame({"supertrend": st, "direction": direction}, index=close.index)


def donchian_channel(
    high: pd.Series, low: pd.Series, period: int = 20
) -> tuple[pd.Series, pd.Series]:
    """Donchian channel — rolling high and rolling low over `period` bars.

    Returns ``(upper, lower)`` where:
      - ``upper = rolling_max(high, period)``
      - ``lower = rolling_min(low,  period)``

    The channel is NOT shifted internally; the strategy is responsible for
    shifting by 1 when judging breakouts so that bar ``i`` is compared to the
    channel formed by bars ``[i-period, i-1]`` rather than the current bar's
    own high/low. Traders use Donchian channels in trend-following systems
    (Richard Dennis / Turtle Trading) — a close above the upper band signals
    a 20-bar breakout and triggers a long entry; below lower band triggers short.

    NaN values populate the head for the first ``period - 1`` bars (warm-up).
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    return upper, lower


# ---------------------------------------------------------------------------
# Divergence helpers (divergence-v1 / divergence-v2)
# ---------------------------------------------------------------------------

def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index — RSI-style volume oscillator (0-100).

    Bounded indicator suitable for divergence pivot detection across regimes
    (unlike cumulative OBV which drifts unboundedly). Computation:

        typical_price = (high + low + close) / 3
        money_flow    = typical_price * volume
        positive_mf   = money_flow where typical_price > prev typical_price else 0
        negative_mf   = money_flow where typical_price < prev typical_price else 0
        mf_ratio      = rolling_sum(positive_mf, period) / rolling_sum(negative_mf, period)
        mfi           = 100 - (100 / (1 + mf_ratio))

    Edge cases (matching RSI convention):
        - When negative_mf_sum == 0 (all periods positive): MFI = 100.
        - Head NaNs: populated for first `period` bars (warmup). Never mid-filled.

    Pure pandas/numpy — no pandas-ta dependency.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    tp = (high + low + close) / 3.0
    mf = tp * volume
    tp_diff = tp.diff()
    pos_mf = mf.where(tp_diff > 0, 0.0)
    neg_mf = mf.where(tp_diff < 0, 0.0)
    # Rolling sums — min_periods=period so head is NaN until warm-up completes.
    pos_sum = pos_mf.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(window=period, min_periods=period).sum()
    mf_ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + mf_ratio))
    # All-positive window (neg_sum == 0 and pos_sum > 0): MFI = 100.
    result = result.where(~((neg_sum == 0) & (pos_sum > 0)), 100.0)
    return result


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume — cumulative signed volume.

    OBV[0] = 0 by convention. Each subsequent bar adds volume when price
    closed higher, subtracts it when lower, and leaves OBV unchanged on a
    flat close. Traders use OBV divergence (price makes a new extreme while
    OBV fails to confirm) as a leading signal of trend exhaustion. When OBV
    diverges from price *and* RSI diverges the same way, the combined signal
    is much stronger than either indicator alone.

    NaN-fill rule (per file convention): only bar 0 is pinned to 0 to
    establish the initial accumulator value. Mid-series NaNs (data gaps) are
    NOT filled — they propagate through the cumsum so the caller can see
    the gap rather than silently hiding it.
    """
    signed_vol = np.sign(close.diff()) * volume
    # Pin bar 0 to 0 (convention: OBV starts at 0, no prior close to compare).
    # Only this one head element is set — mid-series NaNs are NOT filled so
    # data gaps propagate forward (per the file's NaN discipline).
    signed_vol.iloc[0] = 0.0
    return signed_vol.cumsum()


def session_poc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    session: str = "UTC_day",
    n_bins: int = 50,
) -> pd.Series:
    """Previous-session Point of Control (highest-volume price bin from the
    most-recent CLOSED session).

    For each bar i, return the POC of the session that ended STRICTLY BEFORE
    bar i. Lookahead-safe: bar i never sees its own session's volume.

    session = "UTC_day" → 00:00 UTC daily session boundary.

    Implementation:
      1. Group bars by UTC day (floor to midnight).
      2. For each session, compute a volume-by-price histogram with n_bins bins
         spanning [session_low, session_high]. Each bar contributes its volume to
         the bin that contains its typical price (high+low+close)/3. When
         session_high == session_low (single price, degenerate session), POC is
         set to session_low directly.
      3. POC = midpoint of the highest-volume bin.
      4. Forward-fill the POC across the FOLLOWING session's bars. Bars in the
         very first session see NaN (no prior session exists).

    The lookahead guard is enforced by computing each session's POC from the
    bars WITHIN that session, then mapping each bar to the POC of the session
    IMMEDIATELY PRIOR (via shift(1) on the per-session POC series before
    reindex). Bars in the current session can never see the current session's
    still-accumulating volume.

    Parameters
    ----------
    high, low, close, volume : pd.Series
        OHLCV, must share a DatetimeIndex. Index must be timezone-naive (UTC
        convention in this repo — tz is stripped by the data loader).
    session : str
        Session granularity. Only "UTC_day" is implemented; other values raise
        NotImplementedError.
    n_bins : int
        Number of price bins for the volume histogram. Default 50. Sensitivity
        note: bin_width = (session_high - session_low) / n_bins, so larger
        n_bins gives finer resolution at the cost of more noise per bin.

    Returns
    -------
    pd.Series indexed same as inputs.
        NaN for bars in the first session (no prior session POC available).
    """
    if session != "UTC_day":
        raise NotImplementedError(f"session={session!r} not implemented; only 'UTC_day' is supported")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")

    # Group bars by UTC calendar day. Index is timezone-naive in this repo.
    day_key = high.index.floor("D")

    # Build per-session POC: one value per unique day label.
    unique_days = np.sort(np.unique(day_key))

    poc_by_day: dict[pd.Timestamp, float] = {}

    for day in unique_days:
        mask = day_key == day
        h = high.values[mask]
        lo = low.values[mask]
        c = close.values[mask]
        v = volume.values[mask]

        if len(h) == 0:
            continue

        s_high = float(h.max())
        s_low = float(lo.min())

        if s_high <= s_low or not np.isfinite(s_high) or not np.isfinite(s_low):
            # Degenerate session (single bar or flat prices) — POC is the midpoint.
            poc_by_day[day] = (s_high + s_low) / 2.0
            continue

        # Typical price for each bar in the session.
        tp = (h + lo + c) / 3.0

        # Histogram: n_bins equally-spaced bins over [s_low, s_high].
        # bin_edges has n_bins+1 elements; bin i covers [edges[i], edges[i+1]).
        bin_edges = np.linspace(s_low, s_high, n_bins + 1)
        bin_volume = np.zeros(n_bins, dtype=float)

        for k in range(len(tp)):
            if not np.isfinite(tp[k]) or not np.isfinite(v[k]):
                continue
            # Find bin index — clip to [0, n_bins-1] to handle boundary exactly.
            idx = int(np.searchsorted(bin_edges[1:], tp[k], side="right"))
            idx = min(idx, n_bins - 1)
            bin_volume[idx] += v[k]

        best_bin = int(np.argmax(bin_volume))
        poc_price = (bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2.0
        poc_by_day[day] = poc_price

    # Convert to a Series indexed by day timestamp.
    if not poc_by_day:
        return pd.Series(np.nan, index=high.index)

    poc_series = pd.Series(poc_by_day)
    poc_series.index = pd.DatetimeIndex(poc_series.index)
    poc_series = poc_series.sort_index()

    # Shift by 1 so each day maps to the PRIOR day's POC (lookahead guard).
    poc_prior = poc_series.shift(1)

    # Reindex to bar-level: each bar gets the POC of its session's prior session.
    # Use forward-fill: the POC is constant within a session (it never changes
    # mid-session since it's from the prior session).
    bar_days = pd.DatetimeIndex(day_key)
    poc_bar = poc_prior.reindex(bar_days, method="ffill")
    poc_bar.index = high.index

    return poc_bar


def find_divergence(
    price: pd.Series,
    indicator: pd.Series,
    swing_mask: pd.Series,
    kind: str,
    k: int,
    min_separation: int,
    max_separation: int,
) -> pd.Series:
    """Lookahead-safe divergence detector based on causally-registered swing fractals.

    A swing detected at bar ``i`` by ``swing_high_low(k)`` is not knowable
    until bar ``i + k`` (its *confirmation bar*). This function shifts the
    swing mask forward by ``k`` so that ``registered_mask[j]`` is True only
    if ``j - k`` was a swing, and bar ``j`` is the earliest we could know
    that.  The divergence fires at bar ``j == b2 + k`` — exactly once — so
    downstream callers get a single True on the confirmation bar of the
    more-recent swing, never a sticky True.

    Parameters
    ----------
    price:
        ``low`` for bullish divergence; ``high`` for bearish divergence.
    indicator:
        RSI, OBV, or any other Series aligned to the same index.
    swing_mask:
        UNSHIFTED boolean mask from ``swing_high_low()`` — True on the bar
        that IS the swing. The shift-by-``k`` is applied internally.
    kind:
        ``"regular_bullish"``  — price LL + indicator HL → long signal
        ``"regular_bearish"``  — price HH + indicator LH → short signal
    k:
        Same ``k`` passed to ``swing_high_low()``. Must be ≥ 1.
    min_separation:
        Minimum bar distance between the two swing bars (b2 - b1 ≥ this).
    max_separation:
        Maximum bar distance between the two swing bars (b2 - b1 ≤ this).

    Returns
    -------
    pd.Series of bool, same index as ``price``.
        True on bar ``b2 + k`` when a qualifying divergence is confirmed;
        False everywhere else.
    """
    if kind not in ("regular_bullish", "regular_bearish"):
        raise ValueError(f"kind must be 'regular_bullish' or 'regular_bearish', got {kind!r}")
    if k < 1:
        raise ValueError("k must be >= 1")

    n = len(price)
    result = np.zeros(n, dtype=bool)

    price_vals = price.values
    ind_vals = indicator.values
    mask_vals = swing_mask.values  # unshifted — swing at position i

    # Build a list of swing bar positions (integer positions, not index labels).
    swing_positions = np.where(mask_vals)[0]

    # For each swing bar i, it registers at i+k. Walk all potential firing bars.
    # A firing bar is j = b2 + k, where b2 is a swing bar and b1 < b2 is an
    # earlier swing bar satisfying separation constraints + divergence geometry.
    # We need only the two most-recent swing bars up to the firing bar.

    for pos, b2 in enumerate(swing_positions):
        j = b2 + k  # the firing/confirmation bar for this swing
        if j >= n:
            break  # b2+k is out of range

        # Find b1: the most-recent swing bar strictly before b2 whose
        # registration bar (b1+k) is also ≤ j (i.e. b1 <= b2 means b1+k ≤ j).
        # Of the swings preceding b2, find the one immediately before it.
        if pos == 0:
            continue  # no prior swing to pair with

        b1 = swing_positions[pos - 1]
        sep = b2 - b1

        if sep < min_separation or sep > max_separation:
            continue

        # Both b1 and b2 must have valid (non-NaN) indicator values.
        ind_b1 = ind_vals[b1]
        ind_b2 = ind_vals[b2]
        if not (np.isfinite(ind_b1) and np.isfinite(ind_b2)):
            continue

        price_b1 = price_vals[b1]
        price_b2 = price_vals[b2]

        if kind == "regular_bullish":
            # Price: lower low; Indicator: higher low
            if price_b2 < price_b1 and ind_b2 > ind_b1:
                result[j] = True
        else:  # regular_bearish
            # Price: higher high; Indicator: lower high
            if price_b2 > price_b1 and ind_b2 < ind_b1:
                result[j] = True

    return pd.Series(result, index=price.index)
