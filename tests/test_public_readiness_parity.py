"""Public-readiness change: no behaviour change where every module is present.

The change (pinned against PUB_BASE, the droplet tip it was cut from):
  * bot_internals imports the v3-all-wider-4 and cnh-hybrid-short evaluators on
    first use instead of at module top;
  * backtest registers the v2 / v3 / v1-tuned research classes only when their
    modules exist;
  * docstrings / comments lose live numbers; one telemetry fallback (cnh
    gate_status thresholds) no longer invents a value for a key the YAML omits.

This file proves:
  1. with those modules ABSENT, the live import chain still imports and selecting
     a missing strategy raises an error naming the module (runs everywhere, on
     the working tree — a standing regression test);
  2. ONE-TIME PROOF of this change, comparing two pinned commits out of history
     (PUB_BASE = before, PUB_CHANGE / MONITOR_CHANGE = the commits that made
     it), never the working tree: with the optional modules PRESENT the
     backtest registry and the injection snapshot are identical, bot_internals
     decides identically on every live leg (live YAMLs + fixtures) and on the
     optional strategies, the comment/docstring-only files are AST-identical
     once docstrings are blanked, and the monitor's computed kill sentence
     equals the old hardcoded one for the live YAMLs. Later edits to these
     files do not touch these tests.

Missing pinned history or a missing optional module FAILS unless
SNAPBACK_PUBLIC_SNAPSHOT=1 (same gate as test_no_live_fallbacks_parity).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.test_no_live_fallbacks_parity import (
    _git_show,
    _live_params,
    _missing,
    _ohlcv,
    _same,
    _windows,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
PUB_BASE = os.environ.get("SNAPBACK_PUBREADY_BASE",
                          "a57fc7a4888113956a78181925156051489c03ed")  # droplet tip, pre-change
# The commits that MADE the change. Every base-vs-change proof below compares
# two pinned commits out of git history, never the working tree, so later
# legitimate edits to these files are not frozen by this one-time proof.
PUB_CHANGE = os.environ.get("SNAPBACK_PUBREADY_CHANGE",
                            "fbee040b8e23b9fa61f35e3e8f7cf7c698848db6")  # PR #56 main commit
MONITOR_CHANGE = os.environ.get("SNAPBACK_PUBREADY_MONITOR_CHANGE",
                                "3d19069f7328bb5fb45779cd96fe73391ea8ea5a")  # computed kill text

# Modules the public snapshot omits; the live chain must not need them.
OPTIONAL_MODULES = (
    "strategy.live_v3all_wider4",
    "strategy.live_cnh_hybrid_short",
    "strategy.cnh_detectors",
    "strategy.signals_multifactor_v2",
    "strategy.signals_multifactor_v3",
    "strategy.signals_multifactor_tuned",
)
LIVE_LEGS = {
    "params.yaml": "multifactor-v1",
    "params_donchian.yaml": "donchian-v3",
    "params_sol_supertrend.yaml": "supertrend",
}
# Files whose only change is in comments / docstrings.
DOCSTRING_ONLY = (
    "bot.py",
    "strategy/live_donchian_v3.py",
    "strategy/live_multifactor_v1.py",
    "tools/consolidate_push.py",
    "tools/consolidate_futures_push.py",
    "tools/build_dashboard.py",
)
# Files whose only change is in string literals (message wording).
STRINGS_ONLY = ("exchange/env.py",)


def _run_py(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                          text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})


def _optional_present() -> bool:
    import importlib.util
    present = all(importlib.util.find_spec(m) is not None for m in OPTIONAL_MODULES)
    if not present and os.environ.get("SNAPBACK_PUBLIC_SNAPSHOT") != "1":
        pytest.fail("optional strategy modules missing outside the public snapshot")
    return present


def _ref_module(ref: str, path: str, name: str) -> types.ModuleType:
    src = _git_show(ref, path)
    if src is None:
        _missing(f"pinned source {ref}:{path} not in this repo's history")
    mod = types.ModuleType(name)
    mod.__file__ = f"<{ref}:{path}>"
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------------------
# 1. Absent optional modules: clean import, clear error on selection
# ---------------------------------------------------------------------------

def test_live_chain_imports_without_optional_modules():
    code = f"""
import sys
for m in {OPTIONAL_MODULES!r}:
    sys.modules[m] = None          # import of m now raises ModuleNotFoundError
import bot, bot_internals, backtest, config_check, monitor, tools.preflight_live
import pandas as pd
from datetime import datetime
out = {{"unavailable": sorted(set(backtest.UNAVAILABLE_STRATEGIES.values()))}}
bars = pd.DataFrame({{"Close": [1.0]}})
for name in ("v3-all-wider-4", "cnh-hybrid-short-v1"):
    try:
        bot_internals.evaluate_for_strategy(name, bars, 0.0, {{}})
        out[name] = "no error"
    except ModuleNotFoundError as e:
        out[name] = [e.name, str(e)]
try:
    backtest.run_backtest("v3-all", "BTC/USDT:USDT", "15m",
                          datetime(2024, 1, 1), datetime(2024, 2, 1), quiet=True)
    out["bt"] = "no error"
except ModuleNotFoundError as e:
    out["bt"] = [e.name, str(e)]
print(__import__("json").dumps(out))
"""
    r = _run_py(code)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["unavailable"] == ["strategy.signals_multifactor_tuned",
                                  "strategy.signals_multifactor_v2",
                                  "strategy.signals_multifactor_v3"]
    assert out["v3-all-wider-4"][0] == "strategy.live_v3all_wider4"
    assert "strategy.live_v3all_wider4" in out["v3-all-wider-4"][1]
    assert out["cnh-hybrid-short-v1"][0] == "strategy.live_cnh_hybrid_short"
    assert "strategy.live_cnh_hybrid_short" in out["cnh-hybrid-short-v1"][1]
    assert out["bt"][0] == "strategy.signals_multifactor_v3"


def test_live_chain_does_not_import_optional_modules_at_import_time():
    """Where they exist, importing the live chain still does not pull the
    lazily-imported evaluators in (they load on first use only)."""
    code = ("import sys, bot, bot_internals, config_check, monitor\n"
            "print([m for m in ('strategy.live_v3all_wider4', "
            "'strategy.live_cnh_hybrid_short') if m in sys.modules])")
    r = _run_py(code)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "[]"


# ---------------------------------------------------------------------------
# 2. Present: identical registry and identical decisions
# ---------------------------------------------------------------------------

def test_backtest_registry_and_pristine_identical_to_base():
    if not _optional_present():
        pytest.skip("public snapshot: optional research modules absent")
    for ref in (PUB_BASE, PUB_CHANGE):
        if _git_show(ref, "backtest.py") is None:
            _missing(f"pinned source {ref}:backtest.py not in this repo's history")
    code = f"""
import json, re, subprocess, sys, types
def load(ref, name):
    src = subprocess.run(["git", "show", ref + ":backtest.py"], capture_output=True,
                         text=True, check=True).stdout
    m = types.ModuleType(name)
    m.__file__ = "<" + ref + ":backtest.py>"
    sys.modules[name] = m
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m
old = load("{PUB_BASE}", "_pub_base_backtest")
new = load("{PUB_CHANGE}", "_pub_change_backtest")
def _q(c):  # classes defined IN backtest.py carry the exec'd module's name
    return re.sub(r"^_pub_(base|change)_backtest\\.", "backtest.",
                  f"{{c.__module__}}.{{c.__qualname__}}")
def snap(m):
    return {{"keys": list(m.STRATEGIES),
             "classes": [_q(c) for c in m.STRATEGIES.values()],
             "pristine": {{_q(c): {{k: repr(v) for k, v in d.items()}}
                          for c, d in m._PRISTINE.items()}}}}
print(json.dumps({{"old": snap(old), "new": snap(new),
                   "same_objects": all(new.STRATEGIES[k] is old.STRATEGIES[k] for k in old.STRATEGIES
                                       if k != "buy-and-hold"),
                   "unavailable": new.UNAVAILABLE_STRATEGIES}}))
"""
    r = _run_py(code)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["unavailable"] == {}
    assert out["new"] == out["old"]
    assert out["same_objects"] is True
    assert len(out["new"]["keys"]) > 20


def _params(source: str, name: str) -> dict:
    if source == "fixture":
        return yaml.safe_load((FIXTURES / name).read_text())
    return _live_params(name)


@pytest.mark.parametrize("source", ["fixture", "live"])
@pytest.mark.parametrize("name", sorted(LIVE_LEGS))
def test_live_leg_decisions_identical_to_base(source, name):
    old_bi = _ref_module(PUB_BASE, "bot_internals.py", "_pub_base_bot_internals")
    new_bi = _ref_module(PUB_CHANGE, "bot_internals.py", "_pub_change_bot_internals")
    params = _params(source, name)
    strat = LIVE_LEGS[name]
    assert new_bi.resolve_strategy_name(params) == old_bi.resolve_strategy_name(params) == strat
    tf = "15min" if strat == "multifactor-v1" else "4h"
    df = _ohlcv(1100, tf, seed=29)
    runs = [{**params, "strategy": {**params["strategy"], "use_mtf_4h_gate": False}}]
    if strat == "multifactor-v1":
        # v1's gates are rare on random-walk data; a loosened copy makes the
        # entry branch (sl_pct/tp_pct distances) fire as well. Same code path.
        runs.append({**params, "strategy": {
            **params["strategy"], "use_mtf_4h_gate": False, "rsi_long_threshold": 55,
            "rsi_short_threshold": 45, "volume_multiple": 0.0,
            "require_funding_not_extreme": False}})
    fired = 0
    for run in runs:
        for w in _windows(df, 450, 7):
            o = old_bi.evaluate_for_strategy(strat, w, 0.0001, run)
            n = new_bi.evaluate_for_strategy(strat, w, 0.0001, run)
            assert _same(o.__dict__, n.__dict__), (w.index[-1], o, n)
            assert _same(old_bi.gate_status(strat, o, run), new_bi.gate_status(strat, n, run))
            fired += n.side is not None
    for flag in ("strategy_uses_channel_exit", "strategy_uses_trend_exit"):
        assert getattr(old_bi, flag)(strat) == getattr(new_bi, flag)(strat)
    assert fired > 0, "no window fired — parity never exercised the entry branch"


@pytest.mark.parametrize(("strat", "fixture", "tf", "seed"), [
    ("v3-all-wider-4", "params.yaml", "15min", 41),
    ("cnh-hybrid-short-v1", "params_cnh_hybrid_short.yaml", "4h", 43),
])
def test_optional_strategy_decisions_identical_to_base(strat, fixture, tf, seed):
    if not _optional_present():
        pytest.skip("public snapshot: optional strategy modules absent")
    old_bi = _ref_module(PUB_BASE, "bot_internals.py", "_pub_base_bot_internals")
    new_bi = _ref_module(PUB_CHANGE, "bot_internals.py", "_pub_change_bot_internals")
    params = yaml.safe_load((FIXTURES / fixture).read_text())
    df = _ohlcv(1300, tf, seed=seed)
    for w in _windows(df, 600, 25):
        o = old_bi.evaluate_for_strategy(strat, w, 0.0001, params)
        n = new_bi.evaluate_for_strategy(strat, w, 0.0001, params)
        assert _same(o.__dict__, n.__dict__), (w.index[-1], o, n)
        assert _same(old_bi.gate_status(strat, o, params), new_bi.gate_status(strat, n, params))


def test_time_stop_due_identical_to_base():
    old_bi = _ref_module(PUB_BASE, "bot_internals.py", "_pub_base_bot_internals")
    new_bi = _ref_module(PUB_CHANGE, "bot_internals.py", "_pub_change_bot_internals")
    for bars in (-1, 0, 1, 7, 42, 54, 960):
        for bar_s in (0, 900, 14400):
            edge = float(bars * bar_s)
            for age in [*np.linspace(0, 2e6, 41), edge - 1, edge, edge + 1]:
                assert (old_bi.time_stop_due(bars, bar_s, float(age))
                        == new_bi.time_stop_due(bars, bar_s, float(age)))


# ---------------------------------------------------------------------------
# 3. Comment/docstring-only files are AST-identical
# ---------------------------------------------------------------------------

class _Blank(ast.NodeTransformer):
    def __init__(self, all_strings: bool):
        self.all_strings = all_strings

    def _doc(self, node):
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body[0].value.value = ""
        return node

    visit_Module = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _doc

    def visit_Constant(self, node):
        if self.all_strings and isinstance(node.value, str):
            node.value = ""
        return node

    def visit_JoinedStr(self, node):
        self.generic_visit(node)
        return node


def _norm(src: str, all_strings: bool) -> str:
    return ast.dump(_Blank(all_strings).visit(ast.parse(src)), include_attributes=False)


@pytest.mark.parametrize(("path", "all_strings"),
                         [(p, False) for p in DOCSTRING_ONLY] + [(p, True) for p in STRINGS_ONLY])
def test_file_ast_identical_modulo_docs(path, all_strings):
    old, new = _git_show(PUB_BASE, path), _git_show(PUB_CHANGE, path)
    for ref, src in ((PUB_BASE, old), (PUB_CHANGE, new)):
        if src is None:
            _missing(f"pinned source {ref}:{path} not in this repo's history")
    assert _norm(old, all_strings) == _norm(new, all_strings), (
        f"{path}: a non-comment change crept in — this file is meant to be "
        "docstring/comment-only for the public-readiness change")


# ---------------------------------------------------------------------------
# 4. monitor's equity-alert mail: threshold now computed, text unchanged live
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(LIVE_LEGS))
def test_monitor_kill_sentence_identical_to_base_with_live_yaml(name, tmp_path, monkeypatch):
    """With each leg's live YAML, the computed sentence is the exact one the
    PUB_BASE monitor.py hardcoded — the droplet's mail reads as before."""
    old = _git_show(PUB_BASE, "monitor.py")
    if old is None:
        _missing(f"pinned source {PUB_BASE}:monitor.py not in this repo's history")
    monitor = _ref_module(MONITOR_CHANGE, "monitor.py", "_pub_change_monitor")
    leg = {v: k for k, v in monitor.LEG_PARAMS_FILES.items()}[name]
    (tmp_path / name).write_text(yaml.safe_dump(_live_params(name)))
    monkeypatch.setattr(monitor, "PARAMS_DIR", tmp_path)
    sentence = monitor._kill_switch_text(leg)
    assert "flattens at -" in sentence, sentence
    assert sentence in old


def test_backtest_imports_with_v3_present_but_v2_absent():
    """v3 imports v2 at module top: a checkout trimmed to v3-without-v2 must
    still `import backtest`, marking the v3 names unavailable (naming v2)."""
    code = """
import sys, json
sys.modules["strategy.signals_multifactor_v2"] = None
import backtest
u = backtest.UNAVAILABLE_STRATEGIES
print(json.dumps({"v3": u.get("multifactor-v3"), "v2": u.get("multifactor-v2-loose"),
                  "tuned_loaded": "v1-deluxe" in backtest.STRATEGIES,
                  "v3_loaded": "multifactor-v3" in backtest.STRATEGIES}))
"""
    r = _run_py(code)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["v2"] == "strategy.signals_multifactor_v2"
    assert out["v3_loaded"] is False
    if _optional_present():   # v3 shipped: it is unavailable BECAUSE v2 is missing
        assert out["v3"] == "strategy.signals_multifactor_v2"
        assert out["tuned_loaded"] is True
    else:                     # public snapshot: v3 itself is absent
        assert out["v3"] == "strategy.signals_multifactor_v3"


def test_missing_pinned_ref_fails_not_skips(monkeypatch):
    monkeypatch.delenv("SNAPBACK_PUBLIC_SNAPSHOT", raising=False)
    with pytest.raises(pytest.fail.Exception, match="pinned history"):
        _ref_module("0" * 40, "bot_internals.py", "_pub_missing")
    monkeypatch.setenv("SNAPBACK_PUBLIC_SNAPSHOT", "1")
    with pytest.raises(pytest.skip.Exception):
        _ref_module("0" * 40, "bot_internals.py", "_pub_missing")


# ---------------------------------------------------------------------------
# Standing (working tree): the optional branches dispatch to the right evaluator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("strat", "module", "attr", "fixture", "tf"), [
    ("v3-all-wider-4", "strategy.live_v3all_wider4", "evaluate_signal_v3all_wider4",
     "params.yaml", "15min"),
    ("cnh-hybrid-short-v1", "strategy.live_cnh_hybrid_short",
     "evaluate_signal_cnh_hybrid_short", "params_cnh_hybrid_short.yaml", "4h"),
])
def test_optional_branch_dispatches_to_its_evaluator(strat, module, attr, fixture, tf,
                                                     monkeypatch):
    if not _optional_present():
        pytest.skip("public snapshot: optional strategy modules absent")
    import importlib

    import bot_internals
    mod = importlib.import_module(module)
    real = getattr(mod, attr)
    calls = []

    def spy(*a, **k):
        out = real(*a, **k)
        calls.append(out)
        return out

    monkeypatch.setattr(mod, attr, spy)
    params = yaml.safe_load((FIXTURES / fixture).read_text())
    w = _ohlcv(700, tf, seed=47)
    d = bot_internals.evaluate_for_strategy(strat, w, 0.0001, params)
    assert len(calls) == 1, f"{strat} did not dispatch to {module}.{attr}"
    side, sl, tp, dbg = calls[0]
    assert isinstance(d, bot_internals.SignalDecision)
    assert d.side == side
    assert _same(d.sl_distance, float(sl)) and _same(d.tp_distance, float(tp))
    assert d.debug is dbg and isinstance(d.price, float)
