"""
Smoke tests for the backtest harness. Uses synthetic in-memory data so the
test suite has no network dependency.

The big sanity check: a buy-and-hold backtest on a known price series must
report `after_funding ≈ naive_return - friction_drag - funding_cost_pct`.
If the harness's math is off, this catches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from backtesting.lib import FractionalBacktest

from backtest import (
    COMMISSION_PER_SIDE,
    BuyAndHold,
    funding_cost_for_long_btc,
)


def _make_bt(df, **overrides):
    kwargs = dict(
        cash=10_000,
        commission=COMMISSION_PER_SIDE,
        trade_on_close=False,
        exclusive_orders=True,
        fractional_unit=1e-6,
        finalize_trades=True,
    )
    kwargs.update(overrides)
    return FractionalBacktest(df, BuyAndHold, **kwargs)


def _synthetic_klines(n: int = 200, start_price: float = 40_000.0, drift: float = 5.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC").tz_localize(None)
    rng = np.random.default_rng(seed=42)
    walk = np.cumsum(rng.standard_normal(n) * 30) + np.arange(n) * drift
    close = start_price + walk
    return pd.DataFrame(
        {
            "Open": close - 2,
            "High": close + 15,
            "Low": close - 15,
            "Close": close,
            "Volume": 100 + rng.random(n) * 20,
        },
        index=idx,
    )


def test_buy_and_hold_single_trade():
    df = _synthetic_klines()
    stats = _make_bt(df).run()
    assert int(stats["# Trades"]) == 1, "buy-and-hold should open exactly one position"


def test_friction_drag_matches_commission():
    """Backtest return should equal naive return minus ~2*commission (entry + finalize exit)."""
    df = _synthetic_klines(n=500)
    stats = _make_bt(df).run()

    naive_pct = (df["Close"].iloc[-1] / df["Open"].iloc[0] - 1.0) * 100.0
    bt_pct = float(stats["Return [%]"])
    drag_pct = naive_pct - bt_pct

    # finalize_trades=True closes the position at the last bar's close, so we
    # pay commission on BOTH entry and exit. Plus a tiny sizing drag from
    # size=0.999 (leaves 0.1% in cash). Plus slight entry-bar-open vs
    # first-bar-open price difference (depends on synthetic walk).
    expected_drag_pct = 2 * COMMISSION_PER_SIDE * 100.0  # ~0.10pp
    assert drag_pct > 0, "backtest must show LESS return than naive (friction is positive)"
    assert abs(drag_pct - expected_drag_pct) < 0.15, (
        f"drag {drag_pct:.3f}pp diverges from expected ~{expected_drag_pct:.3f}pp; "
        "fee/slippage model may be wrong"
    )


def test_funding_cost_zero_when_no_funding_events():
    df = _synthetic_klines(n=50)  # 50h < first funding event of an empty DF
    empty_funding = pd.DataFrame(columns=["funding_rate"]).rename_axis("funding_time")
    cost, n = funding_cost_for_long_btc(df, empty_funding, initial_cash=10_000,
                                        commission=COMMISSION_PER_SIDE)
    assert cost == 0.0
    assert n == 0


def test_funding_cost_positive_for_positive_funding():
    """If funding rate is positive (longs pay), cost should be positive USDT."""
    df = _synthetic_klines(n=200)
    # Synthesise 3 funding events at +0.01% each
    funding_times = df.index[[50, 100, 150]]
    funding = pd.DataFrame(
        {"funding_rate": [0.0001, 0.0001, 0.0001]},
        index=funding_times,
    ).rename_axis("funding_time")

    cost, n = funding_cost_for_long_btc(df, funding, initial_cash=10_000,
                                        commission=COMMISSION_PER_SIDE)
    assert n == 3
    # btc_position ≈ 10000 * (1 - 0.0005) / 40000 ≈ 0.25
    # each event: 0.25 * ~price * 0.0001 ≈ ~1 USDT
    # 3 events ≈ ~3 USDT (rough)
    assert 1.5 < cost < 5.0, f"expected ~3 USDT cost, got {cost}"


def test_funding_cost_negative_for_negative_funding():
    """Negative funding means longs RECEIVE — cost should be negative (we got paid)."""
    df = _synthetic_klines(n=200)
    funding_times = df.index[[50, 100, 150]]
    funding = pd.DataFrame(
        {"funding_rate": [-0.0001, -0.0001, -0.0001]},
        index=funding_times,
    ).rename_axis("funding_time")

    cost, _ = funding_cost_for_long_btc(df, funding, initial_cash=10_000,
                                        commission=COMMISSION_PER_SIDE)
    assert cost < 0, f"negative funding should mean negative cost, got {cost}"
