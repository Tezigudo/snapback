"""Tests for the supertrend leg's flip exit.

WHY THIS FILE EXISTS
`flip_exit_signal` is the supertrend leg's third exit — "an opposite STDir flip
closes the position even if SL/TP haven't hit". It has no exchange-native
equivalent, so unlike the SL/TP brackets it only works if this code runs.

It didn't. `_st_frame` was widened to return (STDir, ATR, STLine) when the
st_line telemetry was added; the entry caller was updated, this one was not, so
`direction, atr_s = _st_frame(...)` raised ValueError on EVERY call.
`bot._maybe_channel_exit` catches broadly and logs a WARNING, so the leg ran in
production with its flip exit silently dead, protected only by SL/TP.
`_st_frame` now returns a NamedTuple read by field, so widening it again cannot
break a caller the same way.

`tools/supertrend_parity.py` does exercise this function, but it's a tool — the
suite never runs it, so nothing failed. Hence these tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.live_supertrend import (
    StFrame,
    _cfg,
    _st_frame,
    _warmup_bars,
    flip_exit_signal,
)

# Synthetic values (not the live params); the tests use clean monotone trends.
PARAMS = {"strategy": {"st_period": 9, "st_multiplier": 2.25,
                       "st_atr_period": 9, "st_sl_atr": 1.75, "st_tp_atr": 6.5,
                       "allow_shorts": True}}


def _trend_bars(n: int, start: float, step: float) -> pd.DataFrame:
    """Monotone trend — a clean, unambiguous Supertrend direction.

    step > 0 drives STDir to +1, step < 0 to -1. Range is a fixed fraction of
    price so ATR stays well-defined without dominating the band.
    """
    close = start + step * np.arange(n, dtype=float)
    span = abs(step) * 0.5 + start * 0.001
    return pd.DataFrame({
        "Open": close - step * 0.5,
        "High": close + span,
        "Low": close - span,
        "Close": close,
        "Volume": np.full(n, 1_000.0),
    }, index=pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC"))


def _direction(bars: pd.DataFrame) -> float:
    return float(_st_frame(bars, _cfg(PARAMS)).direction.iloc[-1])


class TestStFrameContract:
    """The shape that broke the exit. _st_frame is now a NamedTuple so callers
    read by field and a future widening can't break them — but the fields
    themselves are load-bearing, so pin those."""

    def test_is_a_named_frame_with_all_three_series(self) -> None:
        out = _st_frame(_trend_bars(80, 100.0, 1.0), _cfg(PARAMS))
        assert isinstance(out, StFrame)
        for name in ("direction", "atr", "line"):
            assert isinstance(getattr(out, name), pd.Series), name

    def test_fields_are_not_interchangeable(self) -> None:
        # Guards against a swapped construction: direction is ±1, the line is a
        # price, ATR is a small positive spread.
        out = _st_frame(_trend_bars(80, 100.0, 1.0), _cfg(PARAMS))
        assert set(np.unique(out.direction.dropna())) <= {-1.0, 1.0}
        assert out.atr.dropna().iloc[-1] > 0
        assert out.line.dropna().iloc[-1] > out.atr.dropna().iloc[-1]


class TestFlipExitSignal:
    def test_does_not_raise_and_returns_pair(self) -> None:
        # THE regression: this raised ValueError("too many values to unpack").
        out = flip_exit_signal(_trend_bars(80, 100.0, 1.0), "long", PARAMS)
        assert isinstance(out, tuple) and len(out) == 2
        assert isinstance(out[0], bool)
        assert isinstance(out[1], dict)

    def test_long_holds_while_trend_is_up(self) -> None:
        bars = _trend_bars(80, 100.0, 1.0)
        assert _direction(bars) == 1.0
        should_exit, dbg = flip_exit_signal(bars, "long", PARAMS)
        assert should_exit is False
        assert dbg["reason"] == "no_flip_exit"

    def test_long_exits_when_direction_is_down(self) -> None:
        bars = _trend_bars(80, 200.0, -1.0)
        assert _direction(bars) == -1.0
        should_exit, dbg = flip_exit_signal(bars, "long", PARAMS)
        assert should_exit is True
        assert dbg["reason"] == "flip_exit_long"

    def test_short_exits_when_direction_is_up(self) -> None:
        bars = _trend_bars(80, 100.0, 1.0)
        should_exit, dbg = flip_exit_signal(bars, "short", PARAMS)
        assert should_exit is True
        assert dbg["reason"] == "flip_exit_short"

    def test_short_holds_while_trend_is_down(self) -> None:
        bars = _trend_bars(80, 200.0, -1.0)
        should_exit, dbg = flip_exit_signal(bars, "short", PARAMS)
        assert should_exit is False
        assert dbg["reason"] == "no_flip_exit"

    def test_warmup_short_history_returns_false(self) -> None:
        need = _warmup_bars(_cfg(PARAMS))
        should_exit, dbg = flip_exit_signal(_trend_bars(need - 1, 100.0, 1.0),
                                            "long", PARAMS)
        assert should_exit is False
        assert dbg["reason"] == "warmup"

    @pytest.mark.parametrize("side", ["long", "short"])
    def test_debug_carries_the_decision_inputs(self, side: str) -> None:
        _, dbg = flip_exit_signal(_trend_bars(80, 100.0, 1.0), side, PARAMS)
        # st_line is what a human reads the decision against — the price level
        # that must be crossed to flip. It was unreachable while this raised.
        for k in ("cur_close", "st_dir", "atr", "st_line", "position_side"):
            assert k in dbg, k
        assert dbg["position_side"] == side
        assert dbg["st_line"] is not None
        assert np.isfinite(dbg["st_line"])

    def test_st_line_agrees_with_st_frame(self) -> None:
        bars = _trend_bars(80, 100.0, 1.0)
        _, dbg = flip_exit_signal(bars, "long", PARAMS)
        expected = float(_st_frame(bars, _cfg(PARAMS)).line.iloc[-1])
        assert dbg["st_line"] == pytest.approx(expected)

    def test_st_line_sits_below_price_in_an_uptrend(self) -> None:
        # Sanity on which band is reported: in an uptrend the trailing stop is
        # under price, so a swapped return order would show up here.
        _, dbg = flip_exit_signal(_trend_bars(80, 100.0, 1.0), "long", PARAMS)
        assert dbg["st_line"] < dbg["cur_close"]
