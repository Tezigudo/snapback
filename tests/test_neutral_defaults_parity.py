"""Neutral research defaults: no behaviour change for the deployed legs.

The research modules (backtest.py, strategy/signals*.py, regime_classifier)
used to carry class/dataclass defaults equal to the deployed parameters. Those
defaults are now None ("must come from the params YAML"). This file proves:

  1. the live bot never imports those modules (so the change cannot reach it);
  2. with each live YAML, the research path resolves the same params and
     produces byte-identical backtest trades/equity as the pinned old tree
     (OLD_REF) on a fixed synthetic frame (v1 and donchian research classes;
     the SOL leg has no research class in these files, so only its resolved
     params are compared);
  3. an incomplete YAML fails with an error naming the missing key.

The old tree is extracted with `git archive`; missing history FAILS unless
SNAPBACK_PUBLIC_SNAPSHOT=1 (same gate as test_no_live_fallbacks_parity).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tests.test_no_live_fallbacks_parity import _live_params, _missing

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "tests" / "_neutral_parity_worker.py"
OLD_REF = os.environ.get("SNAPBACK_NEUTRAL_OLD_REF",
                         "80a1f6fc16aaf02c2dc4be72b9611180ae6d7303")  # droplet at live 80a1f6f
LEGS = ("params.yaml", "params_donchian.yaml", "params_sol_supertrend.yaml")
RESEARCH_MODULES = ("backtest", "strategy.signals", "strategy.signals_multifactor",
                    "strategy.signals_donchian", "strategy.regime_classifier")


def test_live_path_does_not_import_research_modules():
    code = ("import sys, bot, bot_internals, config_check, monitor\n"
            f"print([m for m in {RESEARCH_MODULES!r} if m in sys.modules])")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "[]"


def _worker(tree: Path, yaml_dir: Path, data_dir: Path) -> dict:
    data_dir.mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, str(WORKER), str(tree), str(yaml_dir), str(data_dir)],
                       cwd=tree, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("neutral_parity")
    yaml_dir = base / "yaml"
    yaml_dir.mkdir()
    for name in LEGS:
        (yaml_dir / name).write_text(yaml.safe_dump(_live_params(name)))
    arc = subprocess.run(["git", "archive", OLD_REF, "strategy", "backtest.py", "exchange"],
                         cwd=ROOT, capture_output=True)
    if arc.returncode != 0:
        _missing(f"old research tree {OLD_REF} not in this repo's history")
    old_tree = base / "old"
    old_tree.mkdir()
    with tarfile.open(fileobj=io.BytesIO(arc.stdout)) as tf:
        tf.extractall(old_tree, filter="data")
    return (_worker(old_tree, yaml_dir, base / "d_old"),
            _worker(ROOT, yaml_dir, base / "d_new"))


@pytest.mark.parametrize("leg", ["v1", "v1_gate_on", "donchian"])
def test_backtest_trades_identical_with_live_yaml(runs, leg):
    old, new = runs
    assert len(old[leg]["trades"]) > 0, "vacuous parity: synthetic frame produced no trades"
    assert new[leg] == old[leg]


@pytest.mark.parametrize("strategy", ["snapback-v1", "multifactor-v1", "multifactor-v2-loose",
                                      "multifactor-v3", "v3-all-wider-4", "donchian-v3"])
def test_run_backtest_path_identical_with_live_yaml(runs, strategy):
    """run_backtest -> prepare_strategy_data / _prepare_*_data -> _apply_params_to_class,
    including the v2/v3 variants, whose class constants must not be overwritten."""
    old, new = runs
    assert old["run_backtest"][strategy]["n"] > 0, "vacuous parity"
    assert new["run_backtest"][strategy] == old["run_backtest"][strategy]


@pytest.mark.parametrize("name", LEGS)
def test_resolved_params_identical_with_live_yaml(runs, name):
    old, new = runs
    o, n = old[f"resolved:{name}"], new[f"resolved:{name}"]
    assert o, "nothing compared"
    assert {k: n[k] for k in o} == o


def _fixture(name: str) -> dict:
    return yaml.safe_load((ROOT / "tests" / "fixtures" / name).read_text())


def test_injection_skips_none_and_keeps_variant_constants():
    from backtest import UNAVAILABLE_STRATEGIES, _apply_params_to_class
    from strategy.signals import StrategyParams
    if "multifactor-v2-loose" in UNAVAILABLE_STRATEGIES:
        _missing("research module strategy.signals_multifactor_v2 not in this checkout")
    from strategy.signals_multifactor_v2 import DayTradeMultiFactorBTCv2Loose as V2

    before = {k: getattr(V2, k) for k in ("rsi_period", "volume_ma_period",
                                          "swing_lookback_bars", "trendline_max_distance_pct")}
    _apply_params_to_class(V2, StrategyParams(rsi_period=3, volume_ma_period=4))
    assert {k: getattr(V2, k) for k in before} == before


def test_incomplete_yaml_names_the_missing_key(tmp_path):
    from strategy.signals import StrategyParams, require_params
    from strategy.signals_donchian import attach_donchian
    from strategy.signals_multifactor import DayTradeMultiFactorBTC

    raw = _fixture("params.yaml")
    raw["strategy"].pop("sl_pct")
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(raw))
    loaded = StrategyParams.from_yaml(str(p))
    assert loaded.sl_pct is None
    with pytest.raises(ValueError, match=r"strategy\.sl_pct.*config/README\.md"):
        require_params(loaded, ("sl_pct",), "test")
    # The strategy class has no default either: init() refuses before any trade.
    cls = type("Bare", (DayTradeMultiFactorBTC,), {})
    with pytest.raises(ValueError, match=r"strategy\.rsi_period"):
        cls.init(cls.__new__(cls))
    with pytest.raises(ValueError, match="donchian_period_exit"):
        attach_donchian(pd.DataFrame(), pd.DataFrame(), period_entry=5)


def _stub_loaders(monkeypatch):
    import numpy as np
    import backtest as bt

    n = 1500
    rng = np.random.default_rng(5)
    close = 30000.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, n)))
    idx = pd.date_range("2023-01-01", periods=n, freq="4h")
    frame = pd.DataFrame({"open": close, "high": close * 1.004, "low": close * 0.996,
                          "close": close, "volume": 10.0}, index=idx)
    monkeypatch.setattr(bt, "load_klines", lambda *a, **k: frame.copy())
    monkeypatch.setattr(bt, "load_funding", lambda *a, **k: pd.DataFrame(
        {"funding_rate": [0.0001] * 3}, index=idx[:3]))
    return bt


def _rider_params(tmp_path, with_rider_keys: bool):
    from strategy.signals import StrategyParams
    raw = _fixture("params_donchian.yaml")
    if with_rider_keys:
        raw["strategy"]["rider_atr_period"] = 9
        raw["strategy"]["rider_ema_period"] = 31
    p = tmp_path / "rider.yaml"
    p.write_text(yaml.safe_dump(raw))
    return StrategyParams.from_yaml(str(p))


def test_rider_runs_with_periods_from_yaml(monkeypatch, tmp_path):
    from datetime import datetime
    bt = _stub_loaders(monkeypatch)
    params = _rider_params(tmp_path, True)
    r = bt.run_backtest("rider-v1", "BTC/USDT:USDT", "4h", datetime(2023, 1, 5),
                        datetime(2023, 9, 1), params_override=params, quiet=True)
    assert r["bars"] > 0
    from strategy.signals_donchian import DonchianRiderV1
    assert (DonchianRiderV1.rider_atr_period, DonchianRiderV1.rider_ema_period) == (9, 31)


def test_rider_without_periods_raises_naming_the_yaml_key(monkeypatch, tmp_path):
    from datetime import datetime
    bt = _stub_loaders(monkeypatch)
    params = _rider_params(tmp_path, False)
    with pytest.raises(ValueError, match=r"strategy\.rider_atr_period"):
        bt.run_backtest("rider-v1", "BTC/USDT:USDT", "4h", datetime(2023, 1, 5),
                        datetime(2023, 9, 1), params_override=params, quiet=True)


def test_missing_key_does_not_inherit_previous_run_values(tmp_path):
    from backtest import _apply_params_to_class
    from strategy.signals import StrategyParams, require_params
    from strategy.signals_donchian import DonchianBreakoutBTCv3 as D

    run_a = StrategyParams(time_stop_bars=17, donchian_period_entry=23)
    _apply_params_to_class(D, run_a)
    assert D.time_stop_bars == 17
    run_b = StrategyParams()          # YAML omitting those keys
    _apply_params_to_class(D, run_b)
    assert D.time_stop_bars is None and D.donchian_period_entry is None
    with pytest.raises(ValueError, match=r"strategy\.time_stop_bars"):
        require_params(D, ("time_stop_bars",), "D")
