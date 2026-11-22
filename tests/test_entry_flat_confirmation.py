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
