"""Tests for multifactor-v1's adverse-trend exit.

WHY THIS FILE EXISTS
Every v1 sign-off — the 5 locked OOS windows, the walk-forward, the PSR — was
measured on `DayTradeMultiFactorBTC`, which closes an open position on an
adverse EMA(200) cross whenever `require_trend` is true. The live bot never ran
that rule: `strategy_uses_trend_exit()` returned True only for donchian-v3 and
supertrend, so live v1 exited on SL/TP/time-stop alone.

Nothing caught it for months because `tools/multifactor_validate.py` stage 2
compares ENTRY SIGNAL BARS ONLY — "100% parity across 25,702 bars" was true and
simply did not cover exits. The 2026-08-01 re-validation
(MULTIFACTOR_V1_LIVE_EXIT_VERDICT.md) priced the gap: walk-forward 64% vs the
70% gate, OOS 3/5, PSR `insufficient_evidence`, and a start-anchored drawdown
breaching the kill floor on 0.41% of deploy dates vs 0.00% as-validated.

So the load-bearing tests here are the two the old harness lacked:
  1. bar-for-bar EXIT parity against the backtest's own branch, and
  2. the predicate assertion that v1 is actually wired into the hook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot_internals import (
    strategy_uses_channel_exit,
    strategy_uses_trend_exit,
    trend_exit_fill_reason,
    trend_exit_signal,
)
from strategy.indicators import ema
from strategy.live_multifactor_v1 import trend_exit_signal_multifactor_v1
from tests._live_config import live_config

PERIOD = 170   # synthetic trend-EMA period
PARAMS = {"strategy": {"mf_trend_ema_period": PERIOD, "require_trend": True}}


def _ohlc(close: np.ndarray) -> pd.DataFrame:
    """Minimal OHLCV frame — only Close drives this exit."""
    return pd.DataFrame({
        "Open": close,
        "High": close * 1.001,
        "Low": close * 0.999,
        "Close": close,
        "Volume": np.full(len(close), 1_000.0),
    }, index=pd.date_range("2026-01-01", periods=len(close), freq="15min", tz="UTC"))


def _wavy(n: int = 400) -> pd.DataFrame:
    """Series that crosses its own EMA(200) repeatedly in both directions."""
    i = np.arange(n)
    close = (64_000.0
             + 900.0 * np.sin(i / 17.0)
             + 350.0 * np.sin(i / 5.0)
             + 4.0 * i)
    return _ohlc(close)


# ---------------------------------------------------------------------------
# 1. Parity with the backtest's in-position branch
# ---------------------------------------------------------------------------

class TestTrendExitParity:
    """`trend_exit_signal_multifactor_v1` must match `DayTradeMultiFactorBTC
    .next()`'s adverse-trend branch (signals_multifactor.py:317-326) bar for bar:

        t = self._trend_ema[i]          # init(): ema(close, mf_trend_ema_period)
        if np.isfinite(t):
            long  closes when close_v < t
            short closes when close_v > t

    The reference below rebuilds `_trend_ema` the way `init()` does rather than
    hardcoding numbers, so a change to the backtest's indicator breaks this test
    instead of silently re-opening the gap.
    """

    def _reference(self, df: pd.DataFrame) -> tuple[list[bool], list[bool]]:
        trend_ema = ema(pd.Series(df["Close"].values), PERIOD).values
        longs, shorts = [], []
        for i in range(len(df)):
            t = trend_ema[i]
            close_v = df["Close"].iloc[i]
            finite = bool(np.isfinite(t))
            longs.append(finite and close_v < t)
            shorts.append(finite and close_v > t)
        return longs, shorts

    def test_long_and_short_parity_bar_for_bar(self) -> None:
        df = self._wavy = _wavy()
        ref_long, ref_short = self._reference(df)
        for i in range(len(df)):
            prefix = df.iloc[:i + 1]
            got_long, _ = trend_exit_signal_multifactor_v1(prefix, "long", PARAMS)
            got_short, _ = trend_exit_signal_multifactor_v1(prefix, "short", PARAMS)
            assert got_long == ref_long[i], f"long mismatch at bar {i}"
            assert got_short == ref_short[i], f"short mismatch at bar {i}"

    def test_series_actually_exercises_both_exits(self) -> None:
        """Guard against a vacuous all-False parity pass."""
        ref_long, ref_short = self._reference(_wavy())
        assert sum(ref_long) >= 10, "synthetic series triggers too few long exits"
        assert sum(ref_short) >= 10, "synthetic series triggers too few short exits"

    def test_exit_reads_the_15m_ema_not_the_4h_gate_ema(self) -> None:
        """`mf_trend_ema_period` (entry TF) and `mtf_4h_ema_period` (regime gate)
        are DIFFERENT lines. Reading the 4H one here would exit on the wrong
        rule while still looking plausible."""
        df = _wavy()
        params = {"strategy": {"mf_trend_ema_period": PERIOD,
                               "mtf_4h_ema_period": 50,  # decoy, must be ignored
                               "require_trend": True}}
        _, dbg = trend_exit_signal_multifactor_v1(df, "long", params)
        expected = float(ema(pd.Series(df["Close"].values), PERIOD).iloc[-1])
        assert dbg["trend_ema_period"] == PERIOD
        assert dbg["trend_ema"] == expected


# ---------------------------------------------------------------------------
# 2. Live fetch window — the bot does NOT see genesis
# ---------------------------------------------------------------------------

class TestRollingWindowSeeding:
    """The backtest's EMA is seeded at bar 0 of the whole history; the bot
    fetches a rolling ~1500-bar window, so its EMA is seeded at the window
    start. `ewm(adjust=False)` decays the initial condition geometrically, so
    the two converge — but this exit is a THRESHOLD CROSSING, where a small
    numeric drift can flip a decision. Pin how small it actually is.
    """

    def test_window_seeded_ema_matches_genesis_seeded_ema(self) -> None:
        full = _ohlc(64_000.0 + 900.0 * np.sin(np.arange(3000) / 23.0)
                     + 3.0 * np.arange(3000))
        window = full.iloc[-1500:]

        ema_genesis = float(ema(pd.Series(full["Close"].values), PERIOD).iloc[-1])
        ema_window = float(ema(pd.Series(window["Close"].values), PERIOD).iloc[-1])

        # (1 - 2/201)^1299 ~ 3e-6 of the seed error survives to the last bar.
        assert abs(ema_genesis - ema_window) < 0.01, (
            f"window seeding drifts {abs(ema_genesis - ema_window):.6f} — "
            "large enough to flip a threshold crossing")

    def test_decision_agrees_between_window_and_genesis(self) -> None:
        full = _wavy(2200)
        for side in ("long", "short"):
            got_full, _ = trend_exit_signal_multifactor_v1(full, side, PARAMS)
            got_win, _ = trend_exit_signal_multifactor_v1(
                full.iloc[-1500:], side, PARAMS)
            assert got_full == got_win, f"{side}: window and genesis disagree"


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------

class TestTrendExitEdges:

    def _at_ema(self) -> tuple[pd.DataFrame, float]:
        """A frame whose last close sits EXACTLY on its own EMA(200)."""
        df = _wavy()
        close = df["Close"].to_numpy(copy=True)
        # Solve for the last close c s.t. c == alpha*c + (1-alpha)*prev_ema,
        # which holds iff c == prev_ema.
        prev_ema = float(ema(pd.Series(close[:-1]), PERIOD).iloc[-1])
        close[-1] = prev_ema
        out = _ohlc(close)
        return out, prev_ema

    def test_long_exactly_at_ema_does_not_exit(self) -> None:
        df, level = self._at_ema()
        should, dbg = trend_exit_signal_multifactor_v1(df, "long", PARAMS)
        assert dbg["cur_close"] == dbg["trend_ema"] == level
        assert should is False
        assert dbg["reason"] == "hold"

    def test_short_exactly_at_ema_does_not_exit(self) -> None:
        df, _ = self._at_ema()
        should, dbg = trend_exit_signal_multifactor_v1(df, "short", PARAMS)
        assert should is False
        assert dbg["reason"] == "hold"

    def test_long_just_below_ema_exits(self) -> None:
        df, level = self._at_ema()
        close = df["Close"].to_numpy(copy=True)
        close[-1] = level - 1.0
        should, dbg = trend_exit_signal_multifactor_v1(_ohlc(close), "long", PARAMS)
        assert should is True
        assert dbg["reason"] == "trend_exit_long"

    def test_short_just_above_ema_exits(self) -> None:
        df, level = self._at_ema()
        close = df["Close"].to_numpy(copy=True)
        close[-1] = level + 1.0
        should, dbg = trend_exit_signal_multifactor_v1(_ohlc(close), "short", PARAMS)
        assert should is True
        assert dbg["reason"] == "trend_exit_short"

    def test_long_below_ema_does_not_exit_a_short(self) -> None:
        """Sides are not symmetric — a long-adverse bar must leave a short open."""
        df, level = self._at_ema()
        close = df["Close"].to_numpy(copy=True)
        close[-1] = level - 1.0
        should, _ = trend_exit_signal_multifactor_v1(_ohlc(close), "short", PARAMS)
        assert should is False

    def test_require_trend_off_disables_the_exit(self) -> None:
        """Same config key that disables the branch in the backtest — the two
        must stay in lockstep or live/backtest diverge again."""
        df, level = self._at_ema()
        close = df["Close"].to_numpy(copy=True)
        close[-1] = level - 500.0  # deeply adverse
        params = {"strategy": {"mf_trend_ema_period": PERIOD, "require_trend": False}}
        should, dbg = trend_exit_signal_multifactor_v1(_ohlc(close), "long", params)
        assert should is False
        assert dbg["reason"] == "require_trend_off"

    def test_require_trend_missing_defaults_to_the_backtest_default(self) -> None:
        """An absent key must fall back to "on" in the live evaluator, not to "off".

        This is the bug this whole module exists to fix, in miniature: a live
        default of False would mean a config that lost the key runs the adverse
        exit in the backtest and NOT live — silently, on the same leg, with no
        test failing.
        """
        df, level = self._at_ema()
        close = df["Close"].to_numpy(copy=True)
        close[-1] = level - 500.0  # deeply adverse — must exit
        should, dbg = trend_exit_signal_multifactor_v1(
            _ohlc(close), "long", {"strategy": {"mf_trend_ema_period": PERIOD}})
        assert should is True, "absent require_trend silently disabled the exit"
        assert dbg["reason"] == "trend_exit_long"

    def test_deployed_config_actually_sets_require_trend(self) -> None:
        """The default above is a safety net, not the plan — production must set
        the key explicitly.

        Deliberately reads the REAL config/params.yaml rather than a fixture:
        asserting a fixture sets the key would prove nothing about production,
        and "production quietly stopped setting it" is the exact scenario the
        safety net exists for. The file is untracked, so this skips where it is
        absent; config_check.REQUIRED_KEYS additionally refuses boot when the
        key is missing or null.
        """
        cfg = live_config("params.yaml")
        assert cfg["strategy"]["require_trend"] is True

    def test_require_trend_is_a_required_key(self) -> None:
        from config_check import REQUIRED_KEYS
        assert "strategy.require_trend" in REQUIRED_KEYS["multifactor-v1"]

    def test_time_stop_keys_stay_equal(self) -> None:
        """`max_hold_bars` and `time_stop_bars` are two names for one concept
        with different readers — the bot and backtest read the former, while
        tools/data_parity_check.py and tools/build_deployed_strategies_viz.py
        read the latter. Retuning one alone desynchronises the tools from the
        bot silently, and data_parity_check's fallback is a far shorter window
        than the deployed one, so a deletion is worse than a mismatch.
        """
        s = live_config("params.yaml")["strategy"]
        assert "max_hold_bars" in s, "the live bot's time-stop key is missing"
        assert "time_stop_bars" in s, (
            "deleting time_stop_bars silently drops data_parity_check to its "
            "short fallback time stop")
        assert s["time_stop_bars"] == s["max_hold_bars"]

    def test_insufficient_bars_is_warmup(self) -> None:
        should, dbg = trend_exit_signal_multifactor_v1(_wavy(50), "long", PARAMS)
        assert should is False
        assert dbg["reason"] == "warmup"
        assert dbg["need"] == PERIOD

    def test_nan_close_holds_rather_than_raising(self) -> None:
        """Parity detail: next() guards only `t`, so a NaN close falls through
        both strict comparisons and the backtest holds too. Holding is the
        parity-correct answer AND the safe one — never close on garbage data."""
        close = _wavy()["Close"].to_numpy(copy=True)
        close[-1] = np.nan
        for side in ("long", "short"):
            should, dbg = trend_exit_signal_multifactor_v1(_ohlc(close), side, PARAMS)
            assert should is False, side
            assert dbg["reason"] == "nan_close", side

    def test_nan_ema_holds(self) -> None:
        """Ragged frame that clears the length check but not the EMA warmup."""
        close = _wavy(PERIOD + 5)["Close"].to_numpy(copy=True)
        close[:10] = np.nan
        should, dbg = trend_exit_signal_multifactor_v1(_ohlc(close), "long", PARAMS)
        assert should is False
        assert dbg["reason"] == "nan_indicators"

    def test_flat_or_unknown_side_never_exits(self) -> None:
        for side in ("flat", "", "LONG", None):
            should, dbg = trend_exit_signal_multifactor_v1(_wavy(), side, PARAMS)  # type: ignore[arg-type]
            assert should is False, side
            assert dbg["reason"] == "not_in_position"

    def test_warmup_uses_the_configured_period(self) -> None:
        params = {"strategy": {"mf_trend_ema_period": 30, "require_trend": True}}
        should, dbg = trend_exit_signal_multifactor_v1(_wavy(40), "long", params)
        assert dbg["reason"] != "warmup"
        assert isinstance(should, bool)


# ---------------------------------------------------------------------------
# 4. Wiring — the actual regression
# ---------------------------------------------------------------------------

class TestWiring:
    """The exit function existing is worth nothing if the loop never calls it.
    This is the assertion whose absence let live v1 run unvalidated for months.
    """

    def test_v1_is_wired_into_the_trend_exit_hook(self) -> None:
        assert strategy_uses_trend_exit("multifactor-v1") is True

    def test_existing_trend_exit_legs_are_unchanged(self) -> None:
        assert strategy_uses_trend_exit("donchian-v3") is True
        assert strategy_uses_trend_exit("supertrend") is True

    def test_legs_without_a_trend_exit_stay_out(self) -> None:
        for name in ("v3-all-wider-4", "cnh-hybrid-short", "unknown"):
            assert strategy_uses_trend_exit(name) is False, name

    def test_v1_still_places_its_tp_bracket(self) -> None:
        """v1 gains a trend exit but KEEPS its TP leg — only donchian omits TP.
        Wiring v1 into channel-exit instead would silently drop its TP."""
        assert strategy_uses_channel_exit("multifactor-v1") is False
        assert strategy_uses_channel_exit("donchian-v3") is True
        assert strategy_uses_channel_exit("supertrend") is False

    def test_dispatcher_routes_v1_to_the_ema_rule(self) -> None:
        """Not the donchian fallback — that would need channel columns v1's
        frame doesn't have, and would return a wrong answer, not an error."""
        df, level = TestTrendExitEdges()._at_ema()
        close = df["Close"].to_numpy(copy=True)
        close[-1] = level - 1.0
        should, dbg = trend_exit_signal("multifactor-v1", _ohlc(close), "long", PARAMS)
        assert should is True
        assert dbg["reason"] == "trend_exit_long"
        assert "trend_ema" in dbg and "exit_lower" not in dbg

    def test_fill_reason_is_distinct_from_channel_exit(self) -> None:
        """v1's exit is an EMA cross, not a channel cross. donchian/supertrend
        keep "channel_exit" — that value is already in the fills table and the
        consolidate fixtures, and renaming it would orphan the history."""
        assert trend_exit_fill_reason("multifactor-v1") == "trend_exit"
        assert trend_exit_fill_reason("donchian-v3") == "channel_exit"
        assert trend_exit_fill_reason("supertrend") == "channel_exit"
