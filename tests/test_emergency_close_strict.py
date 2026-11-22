"""F4 (2026-11 fix-pass): emergency_close must not accept a lenient flat.

A position read that lags (positionRisk reads flat, position is open) used to
make attempts >= 1 sweep the root and report closed=True; _finish_no_stop then
cleared the record and mailed "closed" with the position still open.
"""

from __future__ import annotations

from unittest.mock import patch

from exchange.binance_client import BinanceClient

from ._fake_exchange import SYMBOL, FakeEx


def _client(ex):
    return BinanceClient(ex=ex, env="testnet", coid_prefix="snap-d3-")


def test_lagging_flat_read_is_retried_not_reported_closed():
    ex = FakeEx()
    ex.pos = ("long", 0.005, 65000.0)
    ex.position_lag_reads = 1                 # first positionRisk read: flat (stale)
    with patch("exchange.binance_client.time.sleep"):
        closed, order, err = _client(ex).emergency_close(SYMBOL, "1791000000000")
    assert closed and order is not None       # the close was actually SENT
    assert ex.pos is None
    assert ex.closes()                        # a reduce-only close hit the exchange


def test_truly_flat_is_still_closed_without_an_order():
    ex = FakeEx()
    ex.pos = None
    with patch("exchange.binance_client.time.sleep"):
        closed, order, _err = _client(ex).emergency_close(SYMBOL, "1791000000000")
    assert closed and order is None
