"""Tests for the pure helpers in bot_internals.py.

These functions are extracted from bot.py and have no side effects, so
they're easy to unit-test without mocking the exchange or state.db.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bot_internals import (
    SignalDecision,
    evaluate_for_strategy,
    limit_entry_price,
    resolve_strategy_name,
    time_stop_due,
)

H4_S = 4 * 3600
M15_S = 15 * 60


class TestResolveStrategyName:
    def test_default_when_missing(self) -> None:
        assert resolve_strategy_name({}) == "multifactor-v1"

    def test_default_when_none(self) -> None:
        assert resolve_strategy_name({"strategy_name": None}) == "multifactor-v1"

    def test_default_when_empty_string(self) -> None:
        # Falsy values fall through to the default — same behavior as the
        # pre-extraction inline expression `... or "multifactor-v1"`.
        assert resolve_strategy_name({"strategy_name": ""}) == "multifactor-v1"

    def test_passthrough_v1(self) -> None:
        assert resolve_strategy_name({"strategy_name": "multifactor-v1"}) == "multifactor-v1"

    def test_passthrough_v3(self) -> None:
        assert resolve_strategy_name({"strategy_name": "v3-all-wider-4"}) == "v3-all-wider-4"


class TestTimeStopDue:
    """Regression cover for the 2026-08-08 sol_supertrend incident: the leg
    entered LONG 1.2 SOL @ 75.63 and was time-stopped 7 seconds later because
    max_hold_bars=0 was read as "close after zero bars" rather than "no time
    stop". This path had no test at all, which is how it reached real money."""

    def test_zero_means_disabled_not_immediate(self) -> None:
        # THE incident: a position 7s old, config says 0.
        assert time_stop_due(0, H4_S, age_s=7.0) is False

    def test_zero_stays_disabled_however_old(self) -> None:
        # Not merely "not yet due" — never due, at any age.
        assert time_stop_due(0, H4_S, age_s=365 * 86400.0) is False

    def test_negative_also_disabled(self) -> None:
        assert time_stop_due(-1, H4_S, age_s=1e9) is False

    def test_not_due_before_window(self) -> None:
        # a synthetic 54-bar stop on 4h = 9 days.
        assert time_stop_due(54, H4_S, age_s=7 * 86400.0) is False

    def test_due_at_exact_boundary(self) -> None:
        assert time_stop_due(54, H4_S, age_s=54 * H4_S) is True

    def test_due_past_window(self) -> None:
        assert time_stop_due(54, H4_S, age_s=54 * H4_S + 1) is True

    def test_bar_seconds_scales_the_window(self) -> None:
        """max_hold_bars counts ENTRY-TF bars. The same 54 bars is 9 days on a
        4h leg but only 13.5 hours on 15m — a 4h leg reading 15m seconds would
        time-stop 16x too early."""
        age = 24 * 3600.0  # one day
        assert time_stop_due(54, M15_S, age_s=age) is True    # 13.5h window
        assert time_stop_due(54, H4_S, age_s=age) is False    # 9d window

    def test_long_15m_window_in_days(self) -> None:
        # a synthetic 960-bar stop × 15m = 10 days.
        assert time_stop_due(960, M15_S, age_s=9.9 * 86400.0) is False
        assert time_stop_due(960, M15_S, age_s=10.1 * 86400.0) is True

    def test_zero_bar_seconds_is_disabled_not_crash(self) -> None:
        # Defensive: a bad/unparsed timeframe must not make every tick "due".
        assert time_stop_due(54, 0, age_s=1e9) is False


class TestLimitEntryPrice:
    """LONG entries place limit BELOW close (try to fill at a better price);
    SHORT entries place limit ABOVE close. This is the maker-rebate
    direction — verify the sign doesn't flip."""

    def test_long_at_close(self) -> None:
        assert limit_entry_price("long", 100.0, 0.0) == pytest.approx(100.0)

    def test_short_at_close(self) -> None:
        assert limit_entry_price("short", 100.0, 0.0) == pytest.approx(100.0)

    def test_long_below_close_with_offset(self) -> None:
        # 1 bp = 0.01% → 100 × (1 - 0.0001) = 99.99
        assert limit_entry_price("long", 100.0, 1.0) == pytest.approx(99.99)

    def test_short_above_close_with_offset(self) -> None:
        assert limit_entry_price("short", 100.0, 1.0) == pytest.approx(100.01)

    def test_realistic_btc_price(self) -> None:
        # 1 bp offset on $65,000 BTC = $6.50
        long_p = limit_entry_price("long", 65000.0, 1.0)
        short_p = limit_entry_price("short", 65000.0, 1.0)
        assert long_p == pytest.approx(65000.0 - 6.5)
        assert short_p == pytest.approx(65000.0 + 6.5)


class TestSignalDecision:
    def test_sl_tp_long(self) -> None:
        d = SignalDecision(side="long", price=100.0, sl_distance=1.5,
                           tp_distance=3.0, debug={})
        assert d.sl_price == 98.5
        assert d.tp_price == 103.0

    def test_sl_tp_short(self) -> None:
        d = SignalDecision(side="short", price=100.0, sl_distance=1.5,
                           tp_distance=3.0, debug={})
        assert d.sl_price == 101.5
        assert d.tp_price == 97.0

    def test_frozen(self) -> None:
        # Decisions are immutable — accidental mutation should fail loudly.
        d = SignalDecision(side="long", price=100.0, sl_distance=1.0,
                           tp_distance=2.0, debug={})
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            d.price = 200.0  # type: ignore[misc]


def _synthetic_bars(n: int = 300, base: float = 65000.0) -> pd.DataFrame:
    """Build a deterministic OHLCV frame long enough to clear v1 warmup
    (EMA warm-up + buffer). No signal will actually fire — we only care about
    the dispatch shape, not the entry verdict."""
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC").tz_localize(None)
    rng = np.random.default_rng(seed=42)
    walk = base + np.cumsum(rng.standard_normal(n) * 5)
    return pd.DataFrame({
        "Open": walk - 1, "High": walk + 5, "Low": walk - 5,
        "Close": walk, "Volume": 100 + rng.random(n) * 10,
    }, index=idx)


class TestEvaluateForStrategy:
    """Verify the dispatch shape — both strategy paths must return a
    SignalDecision with the right distances. Whether side is non-None
    depends on the bars; we don't pin that here (varies with random walk)."""

    def _v1_params(self) -> dict:
        return {
            "strategy": {
                # Synthetic values (not any leg's config); the tests check
                # dispatch mechanics, not the numbers.
                "rsi_period": 12, "rsi_long_threshold": 38, "rsi_short_threshold": 67,
                "ema_period": 150, "mf_trend_ema_period": 150,
                "volume_ma_period": 18, "volume_multiple": 2.3,
                "funding_extreme_threshold": 0.0007,
                "require_funding_not_extreme": True,
                "sl_pct": 0.012, "tp_pct": 0.027,
            },
        }

    def test_v1_returns_signal_decision(self) -> None:
        bars = _synthetic_bars()
        d = evaluate_for_strategy("multifactor-v1", bars, 0.0, self._v1_params())
        assert isinstance(d, SignalDecision)
        # v1 derives distances from sl_pct/tp_pct × price.
        assert d.sl_distance == pytest.approx(0.012 * d.price)
        assert d.tp_distance == pytest.approx(0.027 * d.price)

    def test_default_strategy_routes_to_v1(self) -> None:
        """An unknown strategy name should fall through to v1 (the default),
        matching pre-refactor behavior of the `else` branch."""
        bars = _synthetic_bars()
        d = evaluate_for_strategy("some-unknown-strategy", bars, 0.0, self._v1_params())
        # v1 path → distances derived from sl_pct/tp_pct
        assert d.sl_distance == pytest.approx(0.012 * d.price)

    def test_v1_returns_warmup_with_short_history(self) -> None:
        # Fewer than warm-up bars → side=None, debug has 'reason': 'warmup'.
        bars = _synthetic_bars(n=10)
        d = evaluate_for_strategy("multifactor-v1", bars, 0.0, self._v1_params())
        assert d.side is None
        assert d.debug.get("reason") == "warmup"

    def test_decision_price_falls_back_when_dbg_lacks_cur_close(self) -> None:
        # When debug doesn't carry cur_close (warmup), price falls back to the
        # last close. Verify that fallback is what the bar actually shows.
        bars = _synthetic_bars(n=10)
        d = evaluate_for_strategy("multifactor-v1", bars, 0.0, self._v1_params())
        assert d.price == pytest.approx(float(bars["Close"].iloc[-1]))


def _firing_long_bars(n: int = 300, base: float = 65000.0) -> pd.DataFrame:
    """Build a frame whose LAST bar satisfies every v1 long-entry gate:
      RSI < rsi_long_threshold, close > EMA, volume spike, funding not extreme.

    Construction: a long slow uptrend (so close > the EMA) that dips on the
    final few bars (so RSI is oversold), with a volume spike on the last bar (so the
    volume gate passes). This is the closed-bar state the backtest validated on.
    """
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC").tz_localize(None)
    # Slow uptrend keeps close above the EMA.
    close = base + np.arange(n, dtype=float) * 8.0
    # Final 6-bar dip drives RSI down without crossing back under the EMA.
    close[-6:] = close[-7] - np.arange(1, 7) * 25.0
    vol = np.full(n, 100.0)
    vol[-1] = 100.0 * 5.0  # last bar: ~5× the volume SMA baseline → vol gate passes
    return pd.DataFrame({
        "Open": close - 1, "High": close + 5, "Low": close - 30,
        "Close": close, "Volume": vol,
    }, index=idx)


def _firing_short_bars(n: int = 300, base: float = 65000.0) -> pd.DataFrame:
    """Build a frame whose LAST bar satisfies every v1 short-entry gate:
      RSI > rsi_short_threshold, close < EMA, volume spike, funding not extreme.

    Construction: a long slow downtrend (so close < the EMA) that rallies sharply
    on the final few bars (so RSI is overbought), with a volume spike on the last bar.
    Mirror of _firing_long_bars for the short side.
    """
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC").tz_localize(None)
    # Slow downtrend keeps close below the EMA.
    close = base - np.arange(n, dtype=float) * 8.0
    # Final 10-bar rally drives RSI overbought. The rally is steep enough to push
    # RSI over the threshold but the EMA still sits well above (lagging the
    # long downtrend), so close < the EMA holds for the short trend gate.
    close[-10:] = close[-11] + np.arange(1, 11) * 50.0
    vol = np.full(n, 100.0)
    vol[-1] = 100.0 * 5.0  # last bar: ~5× the volume SMA baseline → vol gate passes
    return pd.DataFrame({
        "Open": close + 1, "High": close + 30, "Low": close - 5,
        "Close": close, "Volume": vol,
    }, index=idx)


class TestFormingBarParity:
    """Regression for the forming-bar bug (fix/forming-bar-closed-bar-eval).

    The live bot fetched a frame whose last row is the still-FORMING bar
    (~0.5% of eventual volume), so the volume gate could never fire. The fix
    drops that row (`df = df.iloc[:-1]`) in bot._maybe_enter before any
    evaluator/price/SL-TP logic reads iloc[-1]. These tests pin both halves:
    the closed bar DOES fire, and a forming bar appended on top does NOT —
    but only because the slice is applied. The existing validators only ever
    fed closed bars, so this gap was untested."""

    def _params(self) -> dict:
        return {
            "strategy": {
                # Synthetic values (not any leg's config); the tests check
                # dispatch mechanics, not the numbers.
                "rsi_period": 12, "rsi_long_threshold": 38, "rsi_short_threshold": 67,
                "ema_period": 150, "mf_trend_ema_period": 150,
                "volume_ma_period": 18, "volume_multiple": 2.3,
                "funding_extreme_threshold": 0.0007,
                "require_funding_not_extreme": True,
                "sl_pct": 0.012, "tp_pct": 0.027,
            },
        }

    def test_closed_bar_fires_long(self) -> None:
        # Sanity: the engineered closed-bar frame produces a long signal.
        bars = _firing_long_bars()
        d = evaluate_for_strategy("multifactor-v1", bars, 0.0, self._params())
        assert d.side == "long", f"expected long on closed bar, got {d.debug}"
        assert d.debug["vol_ok"] is True
        assert d.debug["trend_up"] is True

    def test_forming_bar_alone_does_not_fire(self) -> None:
        # If the bot evaluated the RAW frame (forming bar last, near-zero
        # volume), the volume gate fails → no signal. This is the BUG state,
        # asserted here so the fix's necessity is documented.
        bars = _firing_long_bars()
        forming = bars.iloc[[-1]].copy()
        forming.index = forming.index + pd.Timedelta(minutes=15)
        forming["Close"] = float(bars["Close"].iloc[-1]) - 40.0  # partial-bar price
        forming["Volume"] = 0.5  # ~0.5% of a full bar, as Binance returns at open
        raw = pd.concat([bars, forming])
        d = evaluate_for_strategy("multifactor-v1", raw, 0.0, self._params())
        assert d.side is None, "forming bar must NOT satisfy the volume gate"
        assert d.debug["vol_ok"] is False

    def test_slice_restores_closed_bar_signal(self) -> None:
        # The fix: drop the forming bar (mirrors bot._maybe_enter `df.iloc[:-1]`)
        # and the underlying closed bar fires again — restoring live↔backtest
        # parity. Entry price must anchor to the CLOSED bar, not the partial.
        bars = _firing_long_bars()
        forming = bars.iloc[[-1]].copy()
        forming.index = forming.index + pd.Timedelta(minutes=15)
        forming["Close"] = float(bars["Close"].iloc[-1]) - 40.0
        forming["Volume"] = 0.5
        raw = pd.concat([bars, forming])

        deformed = raw.iloc[:-1]  # the surgical fix, applied at the test level
        d = evaluate_for_strategy("multifactor-v1", deformed, 0.0, self._params())
        assert d.side == "long"
        # Price/SL/TP must come from the CLOSED bar, never the forming one.
        assert d.price == pytest.approx(float(bars["Close"].iloc[-1]))
        assert d.price != pytest.approx(float(forming["Close"].iloc[-1]))

    def test_closed_bar_fires_short(self) -> None:
        # Sanity: the engineered closed-bar frame produces a short signal.
        bars = _firing_short_bars()
        d = evaluate_for_strategy("multifactor-v1", bars, 0.0, self._params())
        assert d.side == "short", f"expected short on closed bar, got {d.debug}"
        assert d.debug["vol_ok"] is True
        assert d.debug["trend_up"] is False

    def test_forming_bar_suppresses_short(self) -> None:
        # Forming bar with near-zero volume appended → volume gate fails → no
        # signal. Same mechanism as the long-side bug.
        bars = _firing_short_bars()
        forming = bars.iloc[[-1]].copy()
        forming.index = forming.index + pd.Timedelta(minutes=15)
        forming["Close"] = float(bars["Close"].iloc[-1]) + 40.0
        forming["Volume"] = 0.5
        raw = pd.concat([bars, forming])
        d = evaluate_for_strategy("multifactor-v1", raw, 0.0, self._params())
        assert d.side is None, "forming bar must NOT satisfy the volume gate"
        assert d.debug["vol_ok"] is False

    def test_slice_restores_closed_bar_short_signal(self) -> None:
        # The fix applied: drop the forming bar → closed bar fires short again.
        bars = _firing_short_bars()
        forming = bars.iloc[[-1]].copy()
        forming.index = forming.index + pd.Timedelta(minutes=15)
        forming["Close"] = float(bars["Close"].iloc[-1]) + 40.0
        forming["Volume"] = 0.5
        raw = pd.concat([bars, forming])

        deformed = raw.iloc[:-1]
        d = evaluate_for_strategy("multifactor-v1", deformed, 0.0, self._params())
        assert d.side == "short"
        # Price/SL/TP must come from the CLOSED bar, never the forming one.
        assert d.price == pytest.approx(float(bars["Close"].iloc[-1]))
        assert d.price != pytest.approx(float(forming["Close"].iloc[-1]))
class TestRelToRepo:
    """bot._rel_to_repo: the startup banner must never crash on path layout.

    Regression for a latent boot() crash found 2026-08-23: the banner rendered
    `state.DB_PATH.relative_to(REPO_ROOT)`, which raises ValueError whenever the
    DB lives outside the repo. SNAPBACK_STATE_DB accepts an absolute path
    pointing anywhere, so this was reachable by configuration alone — it had
    simply never been exercised, because both live legs use data/*.db.
    """

    def test_path_inside_repo_renders_relative(self):
        import bot
        p = bot.REPO_ROOT / "data" / "state.db"
        assert bot._rel_to_repo(p) == "data/state.db"

    def test_path_outside_repo_falls_back_to_absolute(self):
        import bot
        p = Path("/tmp/pytest-somewhere-else/state.db")
        assert bot._rel_to_repo(p) == "/tmp/pytest-somewhere-else/state.db"

    def test_banner_does_not_raise_for_out_of_repo_db(self, tmp_path):
        """The whole point: an out-of-repo DB must not take down boot()."""
        import bot
        outside = tmp_path / "state.db"
        assert bot._rel_to_repo(outside) == str(outside)
        assert not str(outside).startswith(str(bot.REPO_ROOT))
