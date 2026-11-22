"""Notional cap: RESIZE to the cap instead of skipping (2026-11, God approved).

Shared sizing path Bot._maybe_enter -> every leg (v1, donchian, SOL). Stop and
TP prices are unchanged; only qty shrinks, so effective risk < target, never
above. The cap value and every other ceiling are untouched.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import bot
import risk
from exchange.constraints import DEFAULT_CONSTRAINTS
from tests._bot_seed import seed_no_stop_state

CAP = 500.0
PRICE = 65_000.0


@pytest.fixture(autouse=True)
def _cap(fixture_risk_ceilings):
    risk._ACTIVE = replace(fixture_risk_ceilings, MAX_NOTIONAL_USD=CAP)
    yield


# --- pure clamp ---------------------------------------------------------------

def test_over_cap_is_clamped_to_step_and_never_above_cap():
    q, clamped = risk.clamp_qty_to_notional_cap(0.020, PRICE, 0.001)   # $1300
    assert clamped and q == pytest.approx(0.007)                     # $455 <= $500
    assert q * PRICE <= CAP


def test_under_cap_is_unchanged():
    assert risk.clamp_qty_to_notional_cap(0.007, PRICE, 0.001) == (0.007, False)


def test_clamp_never_increases_size():
    for qty in (0.0071, 0.0079, 0.5, 3.0):
        q, _ = risk.clamp_qty_to_notional_cap(qty, PRICE, 0.001)
        assert q <= qty and q * PRICE <= CAP + 1e-9


def test_exact_step_boundary_float_guard():
    # cap/price exactly on a step: 500/50000 = 0.01 -> stays 0.01 (= cap, allowed)
    q, clamped = risk.clamp_qty_to_notional_cap(0.05, 50_000.0, 0.001)
    assert clamped and q * 50_000.0 <= CAP and q == pytest.approx(0.010)


# --- the bot's sizing path ----------------------------------------------------

def _bot(equity_risk_qty_price=PRICE, constraints=DEFAULT_CONSTRAINTS):
    b = object.__new__(bot.Bot)
    seed_no_stop_state(b, bot.Bot)
    b.instance, b.symbol, b.base_asset = "donchian", "BTC/USDT:USDT", "BTC"
    b.entry_tf, b.bar_seconds = "15m", 900
    b.strategy_name, b.params = "multifactor-v1", {}
    b._latest_gates, b._last_waiting_for = None, ""
    b._last_signal_ts = None
    b._persist_last_signal_ts = MagicMock()
    b.log = logging.getLogger("test_notional_cap_resize")
    b.client = MagicMock()
    b.client.fetch_position.return_value = SimpleNamespace(side="flat")
    b._entry_blocked = lambda: False
    b._entries_paused = lambda: False
    b._daily_loss_blocks_entry = lambda equity: False
    b.dry_run = False
    b.risk_pct, b.leverage, b.order_type = 2.75, 20, "market"
    b.constraints = constraints
    b._place_live_entry = MagicMock()
    b.client.fetch_ohlcv.return_value = __import__(
        "tests.test_stale_bar_reentry", fromlist=["_bars"])._bars("2026-10-22 02:30")
    return b


def _decision(sl_distance):
    return SimpleNamespace(side="long", price=PRICE, sl_distance=sl_distance,
                           sl_price=PRICE - sl_distance, tp_price=PRICE + 2 * sl_distance,
                           debug={})


def _enter(b, equity, decision):
    with patch("bot.evaluate_for_strategy", return_value=decision), \
         patch("bot.gate_status", return_value={}), \
         patch("bot.send_alert") as alert:
        b._maybe_enter(equity)
    return alert


def test_bot_over_cap_places_the_clamped_size_and_alerts():
    # $271 * 2.75% = $7.45 risk / $300 stop = 0.0248 BTC = $1615 notional > $500
    b = _bot()
    d = _decision(300.0)
    alert = _enter(b, 271.0, d)
    b._place_live_entry.assert_called_once()
    _dec, qty, _sid, _eq = b._place_live_entry.call_args.args
    assert qty == pytest.approx(0.007) and qty * PRICE <= CAP
    assert _dec.sl_price == PRICE - 300.0 and _dec.tp_price == PRICE + 600.0   # prices unchanged
    (mail,) = [c for c in alert.call_args_list if "resized" in c.args[0]]
    body = mail.args[1]
    assert "donchian" in body and "intended" in body and "effective risk" in body
    # effective risk = 2.75% * 0.007/0.024 (floored intended qty) < target
    assert "0.80%" in body


def test_bot_clamped_below_exchange_minimum_is_skipped():
    tight = replace(DEFAULT_CONSTRAINTS, min_notional_usdt=600.0)   # min > cap
    b = _bot(constraints=tight)
    alert = _enter(b, 271.0, _decision(300.0))
    b._place_live_entry.assert_not_called()
    assert not [c for c in alert.call_args_list if "resized" in c.args[0]]


def test_bot_under_cap_is_unchanged_and_silent():
    # $271 * 2.75% / $3000 stop = 0.00248 -> 0.002 BTC = $130 < $500
    b = _bot(constraints=replace(DEFAULT_CONSTRAINTS, min_notional_usdt=50.0))
    alert = _enter(b, 271.0, _decision(3000.0))
    b._place_live_entry.assert_called_once()
    assert b._place_live_entry.call_args.args[1] == pytest.approx(0.002)
    assert not [c for c in alert.call_args_list if "resized" in c.args[0]]
