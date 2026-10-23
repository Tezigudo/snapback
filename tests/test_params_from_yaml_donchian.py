"""StrategyParams.from_yaml must carry the donchian-v3 fields.

Before the fix it silently dropped donchian_period_entry (to the class default)
and slope_trend_threshold_pct (to 0.0, regime gate OFF), so any backtest that
loaded config/params_donchian.yaml measured a 20-bar ungated churner instead of
the deployed gated breakout. Live reads the yaml dict directly and was never
affected.

The real params YAMLs are untracked (config/README.md); these tests load the
synthetic fixtures in tests/fixtures/ and assert the loader carries THEIR
values through, which is the property under test.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from strategy.signals import StrategyParams
from tests._live_config import live_config

CFG = Path(__file__).resolve().parent / "fixtures"
_DONCHIAN_FIELDS = (
    "donchian_period_entry", "donchian_period_exit", "atr_trail_multiple",
    "regime_ema_period", "regime_slope_window", "slope_trend_threshold_pct",
)


def test_donchian_yaml_loads_configured_values():
    path = CFG / "params_donchian.yaml"
    raw = yaml.safe_load(path.read_text())
    s, z = raw["strategy"], raw["sizing"]
    p = StrategyParams.from_yaml(str(path))
    default = StrategyParams()
    for f in _DONCHIAN_FIELDS + ("time_stop_bars",):
        assert getattr(p, f) == s[f], f
    # Guard against a fixture that coincides with the dataclass defaults —
    # then "carried through" and "silently dropped" would look identical.
    for f in ("donchian_period_entry", "slope_trend_threshold_pct"):
        assert s[f] != getattr(default, f), f
    assert p.risk_per_trade_pct == z["risk_per_trade_pct"]
    assert p.leverage == z["leverage"]


def test_yamls_without_donchian_keys_leave_them_unset():
    # v1 / multifactor carries no donchian keys: they load as the dataclass
    # default (None = "must be supplied"), never as a code-side number.
    default = StrategyParams()
    p = StrategyParams.from_yaml(str(CFG / "params.yaml"))
    for f in _DONCHIAN_FIELDS:
        assert getattr(default, f) is None, f
        assert getattr(p, f) is None, f


def test_backtest_time_stop_matches_live_max_hold():
    # Live reads strategy.max_hold_bars, the backtest reads time_stop_bars;
    # nothing else keeps the two in step. Deployed-config guard: reads the real
    # (untracked) file, skips where it is absent.
    s = live_config("params_donchian.yaml")["strategy"]
    assert s["time_stop_bars"] == s["max_hold_bars"]
