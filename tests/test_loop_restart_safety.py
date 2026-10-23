"""Restart-safety and main-loop guard tests.

Covers:
  1. tools/leg_safe_restart.py refusal logic (position / resting plain orders /
     resting algo orders / unreadable algo book / dirty repo -> exit 3, never
     touches systemctl; --check -> 0 clean / 3 refused; unknown leg -> 2),
     plus the restart happy path and the re-verify / stamp-failure branches.
  2. Bot.loop(): HALT flatten (leg 'h'), kill-switch flatten (leg 'k'), HALT
     close retry, error backoff doubling (cap 300 s) and network cap (30 s).
  3. CHARACTERIZATION of the naked-position window after SL rejection, driven
     through Bot._place_live_entry (market + limit paths). Callers of
     _place_live_entry (_maybe_enter's handler) are out of scope here.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ccxt
import pytest

import bot as bot_mod
from exchange.binance_client import BinanceClient, Position

ROOT = Path(__file__).resolve().parent.parent


def P(side="flat", qty=0.0):
    return Position("BTC/USDT:USDT", side, qty, 60000.0, 0.0, 0.0)


def load_lsr():
    spec = importlib.util.spec_from_file_location("lsr", ROOT / "tools" / "leg_safe_restart.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_lsr(monkeypatch, pos, plain, algo, algo_ok, argv=("v1",), dirty=False,
            configure=None, revs=("abc1234",), repo=None, from_env=True):
    """Drive leg_safe_restart.main() with a mocked client + subprocess.

    `revs`: successive _git_rev results (last repeats). `configure(c)` may set
    side_effect sequences on the client. Returns (rc, sp, client, module); `sp`
    is the subprocess.run mock — only "systemctl restart" calls count as a restart.
    """
    m = load_lsr()
    c = MagicMock()
    c.fetch_position.return_value = pos
    c.ex.fetch_open_orders.return_value = plain
    c.fetch_algo_orders.return_value = (algo, algo_ok)
    c.fetch_equity_usdt.return_value = 100.0
    if configure:
        configure(c)
    it = iter(revs)
    last = {"v": revs[-1]}
    monkeypatch.setattr(m, "_git_rev", lambda r: next(it, last["v"]))
    monkeypatch.setattr(m, "_tracked_dirty", lambda r: dirty)
    if repo is not None:
        monkeypatch.setattr(m, "REPO", repo)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)

    def fake_run(cmd, **kw):
        out = {"show": "4242", "is-active": "active"}.get(cmd[1], "")
        return SimpleNamespace(stdout=out, returncode=0)
    sp = MagicMock(side_effect=fake_run)
    monkeypatch.setattr(m.subprocess, "run", sp)
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    with patch("exchange.env.load_env_for_instance"), \
         patch.object(BinanceClient, "from_env", return_value=c) as fe:
        rc = m.main()
    if not from_env:
        fe.assert_not_called()
    return rc, sp, c, m


def restarts(sp):
    return [x for x in sp.call_args_list if x.args[0][:2] == ["systemctl", "restart"]]


# ---- 1. leg_safe_restart refusal logic ----
@pytest.mark.parametrize("pos,plain,algo,ok", [
    (P("long", 0.01), [], [], True),          # in position
    (P("flat", 0.0), [{"id": 1}], [], True),  # plain orders resting
    (P("flat", 0.0), [], [{"id": 1}], True),  # algo bracket resting
    (P("flat", 0.0), [], [], False),          # algo book unreadable
    (P("flat", 0.003), [], [], True),         # side flat but qty nonzero
])
def test_lsr_refuses_and_never_touches_systemctl(monkeypatch, pos, plain, algo, ok):
    rc, sp, _, _ = run_lsr(monkeypatch, pos, plain, algo, ok)
    assert rc == 3
    sp.assert_not_called()


@pytest.mark.parametrize("pos,plain,algo,ok", [
    (P("long", 0.01), [], [], True),
    (P("flat", 0.0), [{"id": 1}], [], True),
    (P("flat", 0.0), [], [{"id": 1}], True),
    (P("flat", 0.0), [], [], False),
])
def test_lsr_check_on_refusal_rows_still_returns_3(monkeypatch, pos, plain, algo, ok):
    rc, sp, _, _ = run_lsr(monkeypatch, pos, plain, algo, ok, ("v1", "--check"))
    assert rc == 3
    sp.assert_not_called()


def test_lsr_dirty_repo_refuses_and_never_touches_systemctl(monkeypatch):
    rc, sp, _, _ = run_lsr(monkeypatch, P(), [], [], True, dirty=True)
    assert rc == 3
    assert restarts(sp) == []


def test_lsr_check_only_never_restarts(monkeypatch):
    rc, sp, _, _ = run_lsr(monkeypatch, P(), [], [], True, ("v1", "--check"))
    assert rc == 0
    sp.assert_not_called()


def test_lsr_unknown_leg_usage(monkeypatch):
    rc, sp, _, _ = run_lsr(monkeypatch, P(), [], [], True, ("nope",), from_env=False)
    assert rc == 2
    sp.assert_not_called()


def test_lsr_happy_path_restarts_once_and_writes_stamp(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    rc, sp, _, _ = run_lsr(monkeypatch, P(), [], [], True, repo=tmp_path)
    assert rc == 0
    assert [x.args[0] for x in restarts(sp)] == [["systemctl", "restart", "snapback-btc"]]
    assert (tmp_path / "data" / "v1_safe_restart.done").exists()


def test_lsr_git_rev_none_refuses_before_touching_exchange(monkeypatch):
    rc, sp, c, _ = run_lsr(monkeypatch, P(), [], [], True, revs=(None,), from_env=False)
    assert rc == 3
    sp.assert_not_called()


def test_lsr_rev_changed_between_check_and_restart_refuses(monkeypatch):
    rc, sp, _, _ = run_lsr(monkeypatch, P(), [], [], True, revs=("abc1234", "def5678"))
    assert rc == 3
    assert restarts(sp) == []


def test_lsr_position_opened_in_window_refuses(monkeypatch):
    def cfg(c):
        c.fetch_position.side_effect = [P(), P("long", 0.01)]
    rc, sp, _, _ = run_lsr(monkeypatch, P(), [], [], True, configure=cfg)
    assert rc == 3
    assert restarts(sp) == []


def test_lsr_algo_order_appeared_in_window_refuses(monkeypatch):
    def cfg(c):
        c.fetch_algo_orders.side_effect = [([], True), ([{"id": 1}], True)]
    rc, sp, _, _ = run_lsr(monkeypatch, P(), [], [], True, configure=cfg)
    assert rc == 3
    assert restarts(sp) == []


def test_lsr_algo_book_unreadable_in_window_refuses(monkeypatch):
    def cfg(c):
        c.fetch_algo_orders.side_effect = [([], True), ([], False)]
    rc, sp, _, _ = run_lsr(monkeypatch, P(), [], [], True, configure=cfg)
    assert rc == 3
    assert restarts(sp) == []


def test_lsr_stamp_write_failure_is_reported_truthfully(monkeypatch, tmp_path, capsys):
    # no data/ dir under the fake REPO -> write_text raises after the restart
    rc, sp, _, _ = run_lsr(monkeypatch, P(), [], [], True, repo=tmp_path)
    out = capsys.readouterr().out
    assert len(restarts(sp)) == 1
    assert rc == 0                      # restart succeeded and is active
    assert "RESTART SUCCEEDED" in out and "stamp could not be written" in out
    assert "untouched" not in out and "NOT touched" not in out


# ---- 2. loop(): HALT flatten / kill switch / error backoff ----
def mkbot(inst="v1"):
    b = object.__new__(bot_mod.Bot)
    b.log = logging.getLogger("t"); b._stopped = False; b.poll_s = 0.0
    b.symbol = "BTC/USDT:USDT"; b.instance = inst; b.dry_run = False
    b.client = MagicMock(); b.client.fetch_equity_usdt.return_value = 100.0
    b._heartbeat = lambda: None
    for n in ("_maybe_reconcile_principal","_detect_bracket_exit","_maybe_time_stop",
              "_maybe_channel_exit","_maybe_breakeven","_maybe_reprotect",
              "_maybe_enter","_maybe_push_consolidate",
              "_maybe_finish_no_stop_close","_maybe_guard_unverified_entry"):
        setattr(b, n, MagicMock())
    from tests._bot_seed import seed_no_stop_state
    seed_no_stop_state(b, bot_mod.Bot)
    b._check_kill_switch = MagicMock(return_value=False)
    return b

def test_loop_halt_flattens_and_exits_0():
    b = mkbot()
    with patch.object(bot_mod, "halt_source", return_value="self"), \
         patch.object(bot_mod, "send_alert") as al, patch.object(bot_mod.consolidate_push, "drain"):
        assert b.loop() == 0
    b.client.close_position.assert_called_once()
    assert b.client.close_position.call_args.kwargs["close_leg"] == "h"
    b._maybe_enter.assert_not_called()
    al.assert_called_once()

def test_loop_halt_dry_run_does_not_flatten():
    b = mkbot(); b.dry_run = True
    with patch.object(bot_mod, "halt_source", return_value="global"), \
         patch.object(bot_mod, "send_alert") as al, patch.object(bot_mod.consolidate_push, "drain"):
        assert b.loop() == 0
    b.client.close_position.assert_not_called()
    assert "DRY" in str(al.call_args).upper()

def test_loop_kill_switch_flattens_with_k_and_never_enters():
    b = mkbot(); b._check_kill_switch.return_value = True
    with patch.object(bot_mod, "halt_source", return_value=None):
        assert b.loop() == 0
    assert b.client.close_position.call_args.kwargs["close_leg"] == "k"
    b._maybe_enter.assert_not_called(); b._detect_bracket_exit.assert_not_called()

def test_loop_halt_close_failure_is_retried_not_swallowed():
    """close_position raising in HALT path must NOT exit 0 silently; loop retries."""
    b = mkbot(); calls = {"n": 0}
    def cp(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1: raise ccxt.ExchangeError("boom")
    b.client.close_position.side_effect = cp
    order = []
    with patch.object(bot_mod, "halt_source", return_value="self"), \
         patch.object(bot_mod, "send_alert",
                      side_effect=lambda *a, **k: order.append(("alert", calls["n"]))) as al, \
         patch.object(bot_mod.consolidate_push, "drain"), \
         patch.object(bot_mod.time, "sleep", side_effect=lambda s: order.append(("sleep", calls["n"]))):
        assert b.loop() == 0
    assert calls["n"] == 2
    # the failed attempt backs off once and sends NO alert; the single alert
    # only comes after the second (successful) close
    assert order == [("sleep", 1), ("alert", 2)]
    al.assert_called_once()

def test_loop_generic_error_backs_off_doubling_cap_300_and_continues():
    b = mkbot(); b.poll_s = 100.0
    n = {"i": 0}
    def boom(eq):
        n["i"] += 1
        if n["i"] >= 4: b._stopped = True
        raise RuntimeError("x")
    b._maybe_enter.side_effect = boom
    sleeps = []
    with patch.object(bot_mod, "halt_source", return_value=None), \
         patch.object(bot_mod.time, "sleep", side_effect=sleeps.append):
        b.loop()
    assert sleeps == [200.0, 300, 300, 300]

def test_loop_network_error_capped_30():
    b = mkbot(); b.poll_s = 10.0; n = {"i": 0}
    def boom(eq):
        n["i"] += 1
        if n["i"] >= 3: b._stopped = True
        raise ccxt.NetworkError("x")
    b._maybe_enter.side_effect = boom
    s = []
    with patch.object(bot_mod, "halt_source", return_value=None), patch.object(bot_mod.time, "sleep", side_effect=s.append):
        b.loop()
    assert s == [20.0, 30, 30]

# ---- 3. naked position after SL placement failure ----
def mk_entry_bot(order_type):
    b = object.__new__(bot_mod.Bot)
    b.log = logging.getLogger("t"); b.symbol = "BTC/USDT:USDT"
    b.coid_prefix = "snap-v1-"; b.strategy_name = "multifactor-v1"
    b.order_type = order_type; b.limit_offset_bps = 5.0; b.limit_timeout_s = 1
    b.client = MagicMock()
    return b


@pytest.mark.parametrize("order_type,method", [
    ("market", "market_order_with_bracket"),
    ("limit", "limit_order_with_bracket"),
])
def test_characterize_sl_rejection_leaves_position_open_CURRENT_BEHAVIOUR(order_type, method):
    """CHARACTERIZATION of CURRENT behaviour, not a statement of desired behaviour.

    Bot._place_live_entry: the client's *_order_with_bracket raises after the
    entry filled (SL rejected). The exception propagates and the bot issues NO
    close_position; safety then depends entirely on reprotect. A pending PR
    intends to change this (flatten on SL rejection); that PR is EXPECTED to
    update/replace this test. Scope: _place_live_entry only, not its caller.
    """
    from bot_internals import SignalDecision
    b = mk_entry_bot(order_type)
    getattr(b.client, method).side_effect = ccxt.InvalidOrder(
        "-2021 Order would immediately trigger")
    d = SignalDecision(side="long", price=65000.0, sl_distance=975.0,
                       tp_distance=1950.0, debug={})
    with patch("bot.state.set_meta"), pytest.raises(ccxt.InvalidOrder):
        b._place_live_entry(d, qty=0.01, signal_id="1234567890123", equity=10000.0)
    b.client.close_position.assert_not_called()


def test_sl_rejection_client_now_closes_at_market():
    """Was a CHARACTERIZATION of the naked-position window (entry fills, SL
    -2021 -> no close). #49 replaced it: -2021 is not retried, the client
    issues a reduce-only market close and raises UnprotectedFillError."""
    from exchange.binance_client import UnprotectedFillError
    ex = MagicMock()
    ex.market.return_value = {"id": "BTCUSDT"}
    calls = []
    def co(sym, typ, side, qty, price, params=None):
        calls.append((typ, (params or {}).get("reduceOnly"),
                      (params or {}).get("newClientOrderId")))
        if typ == "STOP_MARKET": raise ccxt.InvalidOrder("-2021 Order would immediately trigger")
        return {"id": "1"}
    ex.create_order.side_effect = co
    ex.fetch_open_orders.return_value = []
    ex.fapiPrivateGetOpenAlgoOrders.return_value = []
    c = BinanceClient.__new__(BinanceClient)
    c.ex = ex; c.hedge_mode = False; c.coid_prefix = "snap-v1-"
    c._round_qty = lambda s, q: q
    with patch("exchange.binance_client.time.sleep"), \
         pytest.raises(UnprotectedFillError) as ei:
        c.market_order_with_bracket("BTC/USDT:USDT", "long", 0.01, 59000, 62000, "123")
    assert ei.value.closed is True
    assert [t for t, _, _ in calls] == ["market", "STOP_MARKET", "market"]
    assert calls[-1] == ("market", True, "snap-v1-123-ns")
