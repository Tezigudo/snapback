"""
Tests for the walk-forward engine — splitter shape, scoring invariants,
researcher non-emptiness, and an end-to-end smoke run on synthetic data
(no network).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

research = pytest.importorskip(
    "research",
    reason="research package not present on droplet branch",
)

from research.agents.base import FoldResult  # noqa: E402
from research.agents.deterministic import DeterministicResearcher  # noqa: E402
from research.scoring import (  # noqa: E402
    bootstrap_sharpe_p5,
    deflated_sharpe,
    fold_stability_score,
)
from research.walk_forward import (  # noqa: E402
    make_params,
    split_windows,
    sweep_grid,
)
from strategy.signals import StrategyParams  # noqa: E402


# --- splitter ---------------------------------------------------------------
def test_splitter_yields_non_overlapping_train_test():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 4, 1, tzinfo=timezone.utc)
    folds = list(split_windows(start, end, train_days=30, test_days=15, step_days=15))
    assert len(folds) > 0
    for ts, te, vs, ve in folds:
        assert te == vs, "test must start exactly where train ends"
        assert te - ts == timedelta(days=30)
        assert ve - vs == timedelta(days=15)


def test_splitter_stops_before_overrunning_end():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 2, 1, tzinfo=timezone.utc)
    folds = list(split_windows(start, end, train_days=20, test_days=10, step_days=10))
    for _, _, _, ve in folds:
        assert ve <= end


def test_splitter_rejects_nonpositive_args():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 4, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        list(split_windows(start, end, 0, 30, 30))


def test_splitter_sliding_step_advances():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 5, 1, tzinfo=timezone.utc)
    folds = list(split_windows(start, end, train_days=30, test_days=10, step_days=15))
    starts = [ts for ts, _, _, _ in folds]
    diffs = {(b - a).days for a, b in zip(starts, starts[1:])}
    assert diffs == {15}, f"step should be 15 days everywhere, got {diffs}"


# --- sweep -------------------------------------------------------------------
def test_sweep_grid_cartesian_product():
    grid = {"a": [1, 2], "b": [10, 20, 30]}
    combos = list(sweep_grid(grid))
    assert len(combos) == 6
    pairs = {(c["a"], c["b"]) for c in combos}
    assert pairs == {(1, 10), (1, 20), (1, 30), (2, 10), (2, 20), (2, 30)}


def test_sweep_grid_empty():
    assert list(sweep_grid({})) == []


def test_sweep_grid_deterministic_order():
    grid = {"b": [1], "a": [10, 20]}
    combos1 = list(sweep_grid(grid))
    combos2 = list(sweep_grid(grid))
    assert combos1 == combos2


# --- make_params ------------------------------------------------------------
def test_make_params_mirrors_rsi_short_threshold_when_absent():
    base = StrategyParams()
    out = make_params({"rsi_long_threshold": 15.0}, base)
    assert out.rsi_long_threshold == 15.0
    assert out.rsi_short_threshold == 85.0


def test_make_params_respects_explicit_short_threshold():
    base = StrategyParams()
    out = make_params(
        {"rsi_long_threshold": 15.0, "rsi_short_threshold": 75.0}, base
    )
    assert out.rsi_short_threshold == 75.0


def test_make_params_drops_unknown_keys():
    base = StrategyParams()
    # No TypeError — bogus key is silently dropped.
    out = make_params({"nonsense_field": 42, "rsi_period": 7}, base)
    assert out.rsi_period == 7


# --- scoring -----------------------------------------------------------------
def test_deflated_sharpe_le_raw_for_multitest():
    raw = 1.5
    for n in (2, 10, 100, 1000):
        assert deflated_sharpe(raw, n) <= raw


def test_deflated_sharpe_equal_when_single_trial():
    assert deflated_sharpe(1.2, 1) == pytest.approx(1.2)
    assert deflated_sharpe(1.2, 0) == pytest.approx(1.2)


def test_bootstrap_sharpe_p5_deterministic_with_seed():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.01, 0.02, size=50)
    a = bootstrap_sharpe_p5(returns, n_iter=500, seed=42)
    b = bootstrap_sharpe_p5(returns, n_iter=500, seed=42)
    assert a == b


def test_bootstrap_sharpe_p5_zero_for_empty():
    assert bootstrap_sharpe_p5([], n_iter=100) == 0.0
    assert bootstrap_sharpe_p5([0.5], n_iter=100) == 0.0


def test_fold_stability_score_range():
    assert fold_stability_score([]) == 0.0
    assert fold_stability_score([1.0, 1.0, 1.0]) == 1.0
    assert fold_stability_score([-1.0, -1.0]) == 0.0
    assert fold_stability_score([1.0, -1.0, 1.0]) == pytest.approx(2 / 3)


# --- deterministic researcher -----------------------------------------------
def _fake_fold(i: int, train_sh: float, test_sh: float, params=None):
    return FoldResult(
        fold_index=i,
        train_start=f"2024-{i+1:02d}-01T00:00:00",
        train_end=f"2024-{i+1:02d}-15T00:00:00",
        test_start=f"2024-{i+1:02d}-15T00:00:00",
        test_end=f"2024-{i+2:02d}-01T00:00:00",
        chosen_params=params or {"rsi_period": 2, "rsi_long_threshold": 10},
        train_sharpe=train_sh,
        test_sharpe=test_sh,
        test_return_pct=test_sh * 5.0,
        test_after_funding_pct=test_sh * 5.0,
        trades=15,
        max_drawdown_pct=8.0,
    )


def test_deterministic_researcher_handles_empty_folds():
    r = DeterministicResearcher()
    assert "No folds" in r.commentary([])
    assert r.next_sweep_ranges([]) == {}


def test_deterministic_researcher_returns_non_empty_commentary():
    r = DeterministicResearcher()
    folds = [
        _fake_fold(0, 1.5, 0.8),
        _fake_fold(1, 1.2, -0.3, {"rsi_period": 14, "rsi_long_threshold": 30}),
        _fake_fold(2, 1.0, 1.1),
    ]
    out = r.commentary(folds)
    assert "Folds evaluated: 3" in out
    assert "Stability" in out
    assert "rsi_period" in out
    next_ranges = r.next_sweep_ranges(folds)
    assert "rsi_period" in next_ranges
    assert 2 in next_ranges["rsi_period"]  # 2 won twice, 14 won once


# --- end-to-end smoke (mocked backtest, no network) -------------------------
def test_run_walk_forward_smoke_with_mocked_backtest(monkeypatch):
    """A 2-combo sweep across 2 folds, with run_backtest mocked.

    Verifies the engine wires train→winner→test, builds FoldResults of the
    right shape, and survives an OOS leg even when sharpe drops in test.
    """
    from research import walk_forward

    call_log = []

    def fake_run_backtest(strategy_name, symbol, timeframe, start, end,
                          quiet=False, params_override=None, **kw):
        call_log.append((str(start.date()), str(end.date()), params_override.rsi_period))
        # Discriminator: train windows are 30d, test windows are 15d.
        is_train = (end - start).days >= 25
        if params_override.rsi_period == 2:
            sharpe = 1.5 if is_train else 0.3
            ret = 5.0 if is_train else 1.0
        else:
            sharpe = 0.5 if is_train else 0.4
            ret = 2.0 if is_train else 1.5
        return {
            "strategy": strategy_name,
            "symbol": symbol,
            "timeframe": "15m+1h",
            "start": start, "end": end,
            "bars": 100, "trades": 15,
            "naive_return_pct": ret, "backtest_return_pct": ret,
            "after_funding_pct": ret, "funding_cost_usdt": 0.0, "funding_events": 0,
            "sharpe": sharpe, "max_drawdown_pct": -5.0,
            "profit_factor": 1.2, "win_rate_pct": 50.0, "avg_trade_pct": 0.1,
            "commission_per_side": 0.0005, "leverage": 3,
        }

    monkeypatch.setattr(walk_forward, "run_backtest", fake_run_backtest)

    grid = {"rsi_period": [2, 14], "rsi_long_threshold": [10]}
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 4, 1, tzinfo=timezone.utc)
    folds = walk_forward.run_walk_forward(
        start, end, train_days=30, test_days=15, step_days=15,
        grid=grid, min_trades_train=1,
    )

    assert len(folds) >= 2, "should produce at least 2 folds"
    # Winner on every train should be rsi_period=2 (higher train sharpe)
    for f in folds:
        assert f.chosen_params["rsi_period"] == 2
        assert f.test_sharpe == pytest.approx(0.3)
        assert f.train_sharpe == pytest.approx(1.5)
    # 2 combos × N train evals + N OOS evals worth of backtest calls
    assert len(call_log) >= 4


def test_write_reports_creates_json_md(tmp_path):
    from research.walk_forward import write_reports

    folds = [_fake_fold(0, 1.0, 0.5), _fake_fold(1, 0.8, 0.4)]
    sweep_cfg = {
        "promotion": {
            "min_median_test_sharpe": 0.3,
            "min_fold_stability": 0.5,
            "max_train_test_drift_pct": 100.0,
        }
    }
    paths = write_reports(folds, {"rsi_period": [2]}, sweep_cfg, out_dir=tmp_path)
    assert paths["json"].exists()
    assert paths["md"].exists()
    md = paths["md"].read_text()
    assert "Walk-forward report" in md
    assert "Per-fold table" in md
    assert "PASS" in md  # this synthetic should pass
