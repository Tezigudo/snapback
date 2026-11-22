"""Kill switch vs. a withdrawal the principal ledger has not booked yet.

Regression for the 2026-11 review (FINAL §A3 item 6, v1 2026-11-04): the kill
check ran on the hourly-reconciled principal before the transfer detector, so
a withdrawal larger than the cushion (1 - kill_fraction) flattened and HALTed
the leg. Now a breach first reconciles the principal in the same tick, and a
bounded exemption covers only a wallet drop that nothing else explains.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import bot as bot_mod
from exchange import principal as principal_mod

KF = 0.645
P0 = 300.0                      # deposited principal → floor 193.5


class _Principal:
    def __init__(self, p):
        self.p = p
        self.reconciles = 0
        self.on_reconcile = None

    def get(self):
        return self.p

    def reconcile(self, client, log=None):
        self.reconciles += 1
        if self.on_reconcile:
            self.on_reconcile(self)
        return {}


def _bot(*, wallet_now: float, pos_side="long", pos_qty=0.01):
    b = object.__new__(bot_mod.Bot)
    b.log = logging.getLogger("test_kill_withdrawal")
    b.symbol, b.instance = "BTC/USDT:USDT", "donchian"
    b.kill_fraction = KF
    b._kill_grace = None
    b._last_principal_reconcile_ts = 0.0
    b._last_position_side, b._last_position_qty = "long", 0.01
    b._prev_tick_fingerprint = ("long", 0.01, 7)
    b._prev_tick_wallet, b._prev_tick_equity = 300.0, 300.0
    b.client = MagicMock()
    b.client.last_wallet_usdt = wallet_now
    b.client.fetch_position_checked.return_value = SimpleNamespace(side=pos_side, qty=pos_qty)
    # the real kill action is replaced by "would it fire on the CURRENT anchor"
    b._check_kill_switch = MagicMock(
        side_effect=lambda eq: principal_mod.breached(eq, principal_mod.get_principal(), KF))
    return b


@pytest.fixture
def P():
    pr = _Principal(P0)
    ledger = {"sum": 300.0}
    with patch.object(bot_mod.principal, "get_principal", pr.get), \
         patch.object(bot_mod.principal, "reconcile_recent", pr.reconcile), \
         patch.object(bot_mod.principal, "is_initialized", return_value=True), \
         patch.object(bot_mod.state, "principal_ledger_sum", lambda *a, **k: ledger["sum"]), \
         patch.object(bot_mod.state, "max_fill_id", lambda *a, **k: 7), \
         patch.object(bot_mod, "send_alert") as alert:
        pr.ledger, pr.alert = ledger, alert
        yield pr


def test_withdrawal_already_in_the_ledger_does_not_kill(P):
    """Withdraw 150 of 300: equity 150 < floor 193.5 on the stale anchor. The
    same-tick reconcile books it (P → 150, floor 96.75): no kill."""
    def _book(pr):
        pr.p = 150.0
        pr.ledger["sum"] = 150.0
    P.on_reconcile = _book
    b = _bot(wallet_now=150.0)
    assert b._kill_switch_after_reconcile(150.0) is False
    assert P.reconciles == 1
    assert b._kill_grace is None


def test_ledger_lagging_defers_then_kills_when_the_grace_expires(P):
    b = _bot(wallet_now=150.0)
    t = [1_000_000.0]
    with patch.object(bot_mod.time, "time", lambda: t[0]):
        assert b._kill_switch_after_reconcile(150.0) is False     # deferred
        assert b._kill_grace is not None
        assert any("DEFERRED" in c.args[0] for c in P.alert.call_args_list)
        # next tick: _prev_tick_wallet is now post-withdrawal; grace still holds
        b._prev_tick_wallet = 150.0
        t[0] += 60
        assert b._kill_switch_after_reconcile(150.0) is False
        assert P.alert.call_count == 1                            # mailed once
        t[0] += b.TRANSFER_SUSPECT_MAX_S
        assert b._kill_switch_after_reconcile(150.0) is True      # bounded
    assert P.reconciles == 3                                      # every breach tick


def test_withdrawal_booked_this_tick_is_not_counted_again_as_unbooked(P):
    """sourcery 4657581919: 150 withdrawn AND a price loss; the same-tick
    reconcile books the 150 (P2=150, floor 96.75). Equity 90 still breaches
    P2, but old code added the raw wallet drop back (90+150 > 96.75) and
    deferred. The booked part is already in P2: the kill fires, no DEFERRED."""
    def _book(pr):
        pr.p = 150.0
        pr.ledger["sum"] = 150.0
    P.on_reconcile = _book
    b = _bot(wallet_now=150.0)
    assert b._kill_switch_after_reconcile(90.0) is True
    assert b._kill_grace is None
    assert not any("DEFERRED" in c.args[0] for c in P.alert.call_args_list)


def test_unverified_position_read_during_grace_check_kills(P):
    """sourcery 4657581932: the grace uses the strict read; PositionUnknown
    means no deferral."""
    from exchange.binance_client import PositionUnknown
    b = _bot(wallet_now=150.0)
    b.client.fetch_position_checked.side_effect = PositionUnknown("garbled")
    assert b._kill_switch_after_reconcile(150.0) is True
    assert b._kill_grace is None


def test_same_wallet_drop_with_the_position_now_flat_kills_at_once(P):
    """A stop-out between ticks also drops the wallet with no fill row yet —
    exactly when the kill must fire. Position changed → no exemption."""
    b = _bot(wallet_now=150.0, pos_side="flat", pos_qty=0.0)
    assert b._kill_switch_after_reconcile(150.0) is True
    assert b._kill_grace is None


def test_drawdown_that_breaches_on_its_own_kills_even_with_a_withdrawal(P):
    # withdrew 150, AND price took equity to 40: 40 + 150 = 190 < 193.5
    b = _bot(wallet_now=150.0)
    assert b._kill_switch_after_reconcile(40.0) is True


def test_price_drawdown_during_an_open_grace_kills(P):
    b = _bot(wallet_now=150.0)
    assert b._kill_switch_after_reconcile(150.0) is False
    b._prev_tick_wallet = 150.0
    assert b._kill_switch_after_reconcile(40.0) is True
    assert b._kill_grace is None


@pytest.mark.parametrize("change", ["new_fill", "no_wallet", "small_drop", "unknown_snapshot"])
def test_any_missing_or_changed_read_fails_toward_the_kill(P, change):
    b = _bot(wallet_now=150.0)
    if change == "new_fill":
        b._prev_tick_fingerprint = ("long", 0.01, 6)
    elif change == "no_wallet":
        b.client.last_wallet_usdt = None
    elif change == "small_drop":
        b._prev_tick_wallet = 151.0   # wallet barely moved: not a withdrawal
    else:
        b._last_position_side = "unknown"
    assert b._kill_switch_after_reconcile(150.0) is True


def test_no_breach_is_untouched(P):
    b = _bot(wallet_now=300.0)
    assert b._kill_switch_after_reconcile(290.0) is False
    assert P.reconciles == 0
    b._check_kill_switch.assert_called_once_with(290.0)


# --- F7: entries blocked during the grace; loop()-level grace -> kill ---------

def test_entries_blocked_while_the_kill_grace_is_active(P):
    from tests._bot_seed import seed_no_stop_state
    b = _bot(wallet_now=150.0)
    seed_no_stop_state(b, bot_mod.Bot)
    b._pending_record = lambda: None
    b._meta_record_ex = lambda key, kind: (None, True)
    b._unverified_record_ex = lambda: (None, True)
    b._latch_active = lambda: False
    assert b._entry_blocked() is False
    assert b._kill_switch_after_reconcile(150.0) is False      # grace opens
    assert b._entry_blocked() is True
    b._kill_grace = None
    assert b._entry_blocked() is False


def test_loop_withdrawal_grace_defers_then_kills_on_a_real_breach(P):
    """Drive loop(): tick 1 = 150 USDT withdrawn (unbooked) -> deferred, the
    leg keeps running with entries blocked; tick 2 = price drawdown that
    breaches on its own -> kill switch fires, 'k' close, loop returns 0."""
    from tests.test_loop_restart_safety import mkbot
    b = mkbot("donchian")
    b.kill_fraction = KF
    b._kill_grace = None
    b._last_principal_reconcile_ts = 0.0
    b._last_position_side, b._last_position_qty = "long", 0.01
    b._prev_tick_fingerprint = ("long", 0.01, 7)
    b._prev_tick_wallet, b._prev_tick_equity = 300.0, 300.0
    b._persist_live_equity = MagicMock()          # keep _prev_tick_* as seeded
    b.client.fetch_position_checked.return_value = SimpleNamespace(side="long", qty=0.01)
    seq = iter([(150.0, 150.0), (40.0, 150.0)])    # (equity, wallet) per tick

    def _equity():
        eq, w = next(seq)
        b.client.last_wallet_usdt = w
        return eq
    b.client.fetch_equity_usdt.side_effect = _equity
    entered = []
    b._maybe_enter = MagicMock(side_effect=lambda eq: entered.append(b._kill_grace is not None))
    b._check_kill_switch = MagicMock(
        side_effect=lambda eq: principal_mod.breached(eq, principal_mod.get_principal(), KF))
    with patch.object(bot_mod, "halt_source", return_value=None), \
         patch.object(bot_mod.time, "sleep"):
        assert b.loop() == 0
    assert entered == [True]                       # tick 1 ran on, grace active
    assert b.client.close_position.call_args.kwargs["close_leg"] == "k"
    assert any("DEFERRED" in c.args[0] for c in P.alert.call_args_list)
