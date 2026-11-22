"""F1 (2026-11 fix-pass): an entry must not act on ONE lenient flat read.

A false flat used to reach _place_live_entry, whose pre-entry sweep cancels
the leg's live SL/TP, and a second non-reduce-only entry then stacked or
flipped the open position.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from exchange.binance_client import PositionUnknown
from tests.test_notional_cap_resize import _bot, _decision, _enter
from tests.test_orphan_bracket_cleanup import _open


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch("bot.time.sleep"):
        yield


@pytest.mark.parametrize("strict", ["open", "unknown"])
def test_lenient_flat_contradicted_sends_and_cancels_nothing(strict):
    b = _bot()
    b._place_live_entry.side_effect = None
    real_place = type(b)._place_live_entry
    b._place_live_entry = lambda *a, **k: real_place(b, *a, **k)
    b.client.fetch_position_checked.side_effect = (
        PositionUnknown("stale") if strict == "unknown" else None)
    b.client.fetch_position_checked.return_value = _open("long", 65_000.0, 0.01)
    _enter(b, 271.0, _decision(3000.0))
    b.client.cancel_open_orders.assert_not_called()
    b.client.market_order_with_bracket.assert_not_called()
    b.client.limit_order_with_bracket.assert_not_called()
    b.client.ex.create_order.assert_not_called()


def test_confirmed_flat_still_enters():
    b = _bot()
    _enter(b, 271.0, _decision(3000.0))
    b._place_live_entry.assert_called_once()


# --- G4 (round 3): escalate repeated unconfirmed-flat entry skips -------------

def test_three_unconfirmed_skips_in_a_row_log_error_and_alert_once():
    b = _bot()
    b.client.fetch_position_checked.side_effect = PositionUnknown("stale")
    alerts = []
    for _ in range(5):
        b._last_signal_ts = None                     # a new bar each time
        alerts += [c.args[0] for c in _enter(b, 271.0, _decision(3000.0)).call_args_list]
    blocked = [a for a in alerts if "entries blocked" in a]
    assert len(blocked) == 1                         # rate-limited after the 3rd skip
    assert b._entry_skip_unconfirmed_n == 5
    b._place_live_entry.assert_not_called()


def test_streak_resets_on_a_confirmed_flat():
    b = _bot()
    b.client.fetch_position_checked.side_effect = PositionUnknown("stale")
    for _ in range(2):
        b._last_signal_ts = None
        _enter(b, 271.0, _decision(3000.0))
    assert b._entry_skip_unconfirmed_n == 2
    b.client.fetch_position_checked.side_effect = None
    b.client.fetch_position_checked.return_value = __import__(
        "tests.test_orphan_bracket_cleanup", fromlist=["_flat"])._flat()
    b._last_signal_ts = None
    _enter(b, 271.0, _decision(3000.0))
    assert b._entry_skip_unconfirmed_n == 0


def test_an_unconfirmed_skip_consumes_the_bar():
    """G4: documented behaviour — the skipped signal is not retried on a later
    tick of the same bar (the bar was consumed before evaluation)."""
    b = _bot()
    b.client.fetch_position_checked.side_effect = PositionUnknown("stale")
    _enter(b, 271.0, _decision(3000.0))
    b.client.fetch_position_checked.side_effect = None
    b.client.fetch_position_checked.return_value = __import__(
        "tests.test_orphan_bracket_cleanup", fromlist=["_flat"])._flat()
    _enter(b, 271.0, _decision(3000.0))              # same bar: not re-evaluated
    b._place_live_entry.assert_not_called()
