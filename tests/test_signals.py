"""
Tests for prepare_strategy_data and SnapbackBTC. The big invariant: a 15m
bar at time T must NEVER see a 1h indicator computed from data after T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.signals import StrategyParams, prepare_strategy_data


def _klines(periods: int, freq: str, start_price: float = 40_000.0, drift: float = 0.5):
    idx = pd.date_range("2024-01-01", periods=periods, freq=freq, tz="UTC").tz_localize(None)
    rng = np.random.default_rng(seed=7)
    noise = np.cumsum(rng.standard_normal(periods) * 20.0)
    close = start_price + np.arange(periods) * drift + noise
    return pd.DataFrame(
        {
            "open": close - 2,
            "high": close + 12,
            "low": close - 12,
            "close": close,
            "volume": 100 + rng.random(periods) * 20,
        },
        index=idx,
    )


def _funding(periods: int):
    # 8h cadence
    idx = pd.date_range("2024-01-01", periods=periods, freq="8h", tz="UTC").tz_localize(None)
    rng = np.random.default_rng(seed=11)
    rates = rng.uniform(-0.0005, 0.0005, periods)
    return pd.DataFrame({"funding_rate": rates}, index=idx).rename_axis("funding_time")


def _params(**over) -> StrategyParams:
    """Synthetic indicator periods — StrategyParams has no strategy defaults."""
    base = dict(rsi_period=9, volume_ma_period=13, ema_period=30, atr_period=11)
    return StrategyParams(**{**base, **over})


def test_prepare_data_attaches_expected_columns():
    p = _params()
    k15 = _klines(periods=4 * 24 * 30, freq="15min")  # 30 days
    k1h = _klines(periods=24 * 30, freq="1h")
    fund = _funding(periods=3 * 30)
    df = prepare_strategy_data(k15, k1h, fund, p)
    for col in ("Open", "High", "Low", "Close", "Volume", "RSI", "VolSMA",
                "EMA_1h", "ATR_1h", "Funding"):
        assert col in df.columns, f"missing column {col}"
    # After warm-up there must be SOME non-NaN indicator values.
    assert df["EMA_1h"].notna().sum() > 0
    assert df["ATR_1h"].notna().sum() > 0
    assert df["RSI"].notna().sum() > 0
    assert df["Funding"].notna().sum() > 0


def test_1h_indicators_have_no_lookahead():
    """
    The 1h EMA value reindexed onto a 15m bar at time T must NOT use any
    1h bar whose close timestamp is > T. We verify by asserting that the
    EMA_1h value at a 15m bar at time T equals the EMA computed from 1h
    bars indexed strictly BEFORE T.
    """
    p = _params(ema_period=20)
    k1h = _klines(periods=24 * 7, freq="1h")
    k15 = _klines(periods=4 * 24 * 7, freq="15min")
    fund = _funding(periods=3 * 7)
    df = prepare_strategy_data(k15, k1h, fund, p)

    # Sample many 15m timestamps; for each, verify the visible EMA_1h
    # corresponds to a 1h bar at a STRICTLY earlier index.
    from strategy.indicators import ema
    expected_1h = ema(k1h["close"], period=p.ema_period).shift(1)

    sample_times = df.index[100:130]  # well past warm-up
    for t in sample_times:
        visible = df.at[t, "EMA_1h"]
        if pd.isna(visible):
            continue
        # The 1h bar whose label is the largest <= t
        prior_idx = expected_1h.index[expected_1h.index <= t]
        assert len(prior_idx) > 0
        expected = expected_1h.loc[prior_idx[-1]]
        assert visible == pytest.approx(expected, rel=1e-9), (
            f"EMA_1h at 15m bar {t} diverges from shifted 1h value"
        )


def test_funding_is_ffilled_not_interpolated():
    """All Funding values between two events must equal the EARLIER event's rate."""
    p = _params()
    k1h = _klines(periods=24 * 5, freq="1h")
    k15 = _klines(periods=4 * 24 * 5, freq="15min")
    # Three sparse funding events with very different rates so ffill is obvious
    idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00", "2024-01-04 00:00"])
    fund = pd.DataFrame({"funding_rate": [0.0001, -0.0003, 0.0005]}, index=idx).rename_axis(
        "funding_time"
    )
    df = prepare_strategy_data(k15, k1h, fund, p)
    # All 15m bars in [2024-01-02 00:00, 2024-01-04 00:00) must read -0.0003
    span = df.loc["2024-01-02 00:00":"2024-01-03 23:45", "Funding"]
    assert (span == -0.0003).all(), "ffill broken — bars between events drifted"


def test_short_window_throws_on_empty_inputs():
    with pytest.raises(ValueError):
        prepare_strategy_data(pd.DataFrame(), _klines(10, "1h"), _funding(3), _params())
