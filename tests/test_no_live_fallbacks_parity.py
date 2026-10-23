"""S3 parity: removing the code-side numeric fallbacks must not change a single
decision when the params YAML is present and complete.

Method: load the PRE-S3 source of each touched live module straight out of git
(`git show <BASE>:<path>`) as a throwaway module, then run old and new side by
side on the same deterministic synthetic market data with the same params and
assert identical outputs (side, SL/TP distances, full debug dict) on every
window, for entries, trend exits and gate_status.

Params under test:
  * the LIVE per-leg YAMLs — the untracked working copy config/<name> when it
    exists, else `git show <LIVE_REF>:config/<name>` (the last droplet commit
    that still tracked them);
  * the synthetic tests/fixtures/ configs, which always run.

BASE defaults to the S2 tip (the parent of the S3 change); override with
SNAPBACK_PARITY_BASE / SNAPBACK_PARITY_LIVE_REF. Missing history FAILS unless
SNAPBACK_PUBLIC_SNAPSHOT=1 (the public snapshot sets it, or drops this file).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import bot_internals as new_bi
from bot_internals import SignalDecision
from config_check import config_problems
from strategy import live_donchian_v3 as new_don
from strategy import live_multifactor_v1 as new_v1
from strategy import live_supertrend as new_st

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
BASE = os.environ.get("SNAPBACK_PARITY_BASE",
                      "2a45e9ae6509b1421a8774b619b2ff75da1d0b4b")       # S2 tip, pre-S3
LIVE_REF = os.environ.get("SNAPBACK_PARITY_LIVE_REF",
                          "b9cb2d072633cc855fec06d3ebc0e74326d4befd")  # last tracked live YAMLs

LEGS = {  # instance file -> strategy_name
    "params.yaml": "multifactor-v1",
    "params_donchian.yaml": "donchian-v3",
    "params_sol_supertrend.yaml": "supertrend",
}


def _git_show(ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _missing(what: str):
    """Missing pinned history is a FAILURE (a squash/rewrite/shallow clone would
    otherwise turn every parity test into a silent green SKIP). The public
    snapshot, which has no private history, sets SNAPBACK_PUBLIC_SNAPSHOT=1 —
    or drops this file; see PUBLIC_REPO_PLAN."""
    if os.environ.get("SNAPBACK_PUBLIC_SNAPSHOT") == "1":
        pytest.skip(f"public snapshot: {what}")
    pytest.fail(f"{what} — parity proof needs the pinned history "
                "(set SNAPBACK_PUBLIC_SNAPSHOT=1 only in the public snapshot)")


def _old_module(path: str) -> types.ModuleType:
    src = _git_show(BASE, path)
    if src is None:
        _missing(f"pre-S3 source {BASE}:{path} not in this repo's history")
    mod = types.ModuleType("_pre_s3_" + path.replace("/", "_").removesuffix(".py"))
    mod.__file__ = f"<{BASE}:{path}>"
    sys.modules[mod.__name__] = mod   # dataclasses resolve their module here
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def _live_params(name: str) -> dict:
    p = ROOT / "config" / name
    if p.is_file():
        return yaml.safe_load(p.read_text())
    src = _git_show(LIVE_REF, f"config/{name}")
    if src is None:
        _missing(f"live config/{name} unavailable (untracked, not in history)")
    # The pre-untrack copies predate the `risk:` block (RISK_REVIEW 2026-10-22);
    # add it exactly as the droplet migration does.
    from tools.migrate_risk_params import PRE_CHANGE_SHA, migrate_text, values_from_git
    try:
        values = values_from_git(PRE_CHANGE_SHA)
    except Exception as e:  # noqa: BLE001 — missing history is _missing's call
        _missing(f"pre-change risk.py unavailable: {e}")
    return yaml.safe_load(migrate_text(src, values, "test") or src)


def _params(source: str, name: str) -> dict:
    if source == "fixture":
        return yaml.safe_load((FIXTURES / name).read_text())
    return _live_params(name)


# ---------------------------------------------------------------------------
# Deterministic synthetic market data with real trends, chop and volume spikes
# ---------------------------------------------------------------------------

def _ohlcv(n: int, freq: str, seed: int, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=n // 60 + 1), 60)[:n]
    rets = rng.normal(0.0, 0.006, n) + regime * 0.0012
    close = 30000.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.004, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.lognormal(3.0, 0.5, n) * np.where(rng.random(n) < 0.08, 4.0, 1.0)
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _windows(df: pd.DataFrame, length: int, step: int):
    for end in range(length, len(df) + 1, step):
        yield df.iloc[end - length:end]


SOURCES = ["fixture", "live"]


# ---------------------------------------------------------------------------
# Parity per strategy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", SOURCES)
def test_donchian_parity(source):
    old = _old_module("strategy/live_donchian_v3.py")
    params = _params(source, "params_donchian.yaml")
    assert config_problems(params) == []
    df = _ohlcv(1400, "4h", seed=11)
    fired = 0
    for w in _windows(df, 400, 3):
        o, n = old.evaluate_signal_donchian_v3(w, 0.0, params), \
            new_don.evaluate_signal_donchian_v3(w, 0.0, params)
        assert _same(o, n), (w.index[-1], o, n)
        fired += n[0] is not None
        for side in ("long", "short"):
            assert _same(old.channel_exit_signal(w, side, params),
                         new_don.channel_exit_signal(w, side, params))
    assert fired > 0, "no window fired — parity never exercised the entry branch"


@pytest.mark.parametrize("source", SOURCES)
def test_supertrend_parity(source):
    old = _old_module("strategy/live_supertrend.py")
    params = _params(source, "params_sol_supertrend.yaml")
    assert config_problems(params) == []
    assert old._cfg(params) == new_st._cfg(params)
    df = _ohlcv(1200, "4h", seed=23)
    fired = 0
    for w in _windows(df, 300, 2):
        o, n = old.evaluate_signal_supertrend(w, 0.0, params), \
            new_st.evaluate_signal_supertrend(w, 0.0, params)
        assert _same(o, n), (w.index[-1], o, n)
        fired += n[0] is not None
        for side in ("long", "short"):
            assert _same(old.flip_exit_signal(w, side, params),
                         new_st.flip_exit_signal(w, side, params))
    assert fired > 0, "no window fired — parity never exercised the entry branch"


@pytest.mark.parametrize("gate", ["as_configured", "forced_on"])
@pytest.mark.parametrize("source", SOURCES)
def test_multifactor_v1_parity(source, gate):
    old = _old_module("strategy/live_multifactor_v1.py")
    params = _params(source, "params.yaml")
    assert config_problems(params) == []
    if gate == "forced_on":
        params = {**params, "strategy": {**params["strategy"], "use_mtf_4h_gate": True}}
    m15 = _ohlcv(5200, "15min", seed=37)
    h4 = m15.resample("4h").agg({"Open": "first", "High": "max", "Low": "min",
                                 "Close": "last", "Volume": "sum"}).dropna()
    for w in _windows(m15.iloc[3600:], 600, 4):
        o = old.evaluate_signal(w, 0.0002, params, bars_4h=h4)
        n = new_v1.evaluate_signal(w, 0.0002, params, bars_4h=h4)
        assert _same(o, n), (w.index[-1], o, n)
        for side in ("long", "short"):
            assert _same(old.trend_exit_signal_multifactor_v1(w, side, params),
                         new_v1.trend_exit_signal_multifactor_v1(w, side, params))


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("name", sorted(LEGS))
def test_gate_status_parity(source, name):
    """gate_status (telemetry pushed every bar) — old vs new on real decisions
    and on an empty-debug decision (exercises the config-side reads)."""
    old = _old_module("bot_internals.py")
    params = _params(source, name)
    strat = LEGS[name]
    tf = "15min" if strat == "multifactor-v1" else "4h"
    df = _ohlcv(900, tf, seed=5)
    decisions = [SignalDecision(side=None, price=1.0, sl_distance=0.0,
                                tp_distance=0.0, debug={})]
    for w in _windows(df, 450, 15):
        d = new_bi.evaluate_for_strategy(strat, w, 0.0001, {
            **params, "strategy": {**params["strategy"], "use_mtf_4h_gate": False}})
        decisions.append(d)
    for d in decisions:
        assert _same(old.gate_status(strat, d, params), new_bi.gate_status(strat, d, params))


@pytest.mark.parametrize(("name", "fn", "key"), [
    ("params_donchian.yaml", lambda w, p: new_don.evaluate_signal_donchian_v3(w, 0.0, p),
     "donchian_period_entry"),
    ("params_donchian.yaml", lambda w, p: new_don.channel_exit_signal(w, "long", p),
     "donchian_period_exit"),
    ("params_sol_supertrend.yaml", lambda w, p: new_st.evaluate_signal_supertrend(w, 0.0, p),
     "st_multiplier"),
    ("params.yaml", lambda w, p: new_v1.trend_exit_signal_multifactor_v1(w, "long", p),
     "mf_trend_ema_period"),
])
def test_no_silent_default_when_key_missing(name, fn, key):
    """The point of S3: a missing tunable is an error, never a baked-in value."""
    params = yaml.safe_load((FIXTURES / name).read_text())
    del params["strategy"][key]
    with pytest.raises(KeyError, match=key):
        fn(_ohlcv(500, "4h", seed=3), params)


def test_missing_pinned_history_fails_not_skips(monkeypatch):
    mod = sys.modules[__name__]
    monkeypatch.setattr(mod, "BASE", "0" * 40)
    monkeypatch.delenv("SNAPBACK_PUBLIC_SNAPSHOT", raising=False)
    with pytest.raises(pytest.fail.Exception, match="pinned history"):
        _old_module("strategy/live_supertrend.py")
    monkeypatch.setenv("SNAPBACK_PUBLIC_SNAPSHOT", "1")
    with pytest.raises(pytest.skip.Exception):
        _old_module("strategy/live_supertrend.py")
