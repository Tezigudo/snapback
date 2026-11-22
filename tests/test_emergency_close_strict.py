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
        closed, order, _err = _client(ex).emergency_close(SYMBOL, "1791000000000")
    assert closed and order is not None       # the close was actually SENT
    assert ex.pos is None
    assert ex.closes()                        # a reduce-only close hit the exchange


def test_truly_flat_is_still_closed_without_an_order():
    ex = FakeEx()
    ex.pos = None
    with patch("exchange.binance_client.time.sleep"):
        closed, order, _err = _client(ex).emergency_close(SYMBOL, "1791000000000")
    assert closed and order is None


# --- G9 (round 3): stops are never swept before the close, nor after an
#     unconfirmed one -----------------------------------------------------------

ROOT = "1791000000000"


def _with_resting_stop():
    ex = FakeEx()
    ex.pos = ("long", 0.005, 65000.0)
    ex._rest_algo(f"snap-d3-{ROOT}-s", "sell", 0.005, 64000.0)
    return ex


def test_resting_stop_is_cancelled_only_after_the_close():
    ex = _with_resting_stop()
    with patch("exchange.binance_client.time.sleep"):
        closed, order, _e = _client(ex).emergency_close(SYMBOL, ROOT, "long", 0.005)
    assert closed and order is not None
    names = [c[0] for c in ex.calls]
    i_close = next(i for i, c in enumerate(ex.calls)
                   if c[0] == "create_order" and c[2] == "market")
    deletes = [i for i, n in enumerate(names) if "Delete" in n or n == "cancel_algo"]
    assert all(i > i_close for i in deletes), ex.calls
    assert ex.algo == []                       # swept once flat was confirmed


def test_unconfirmed_close_keeps_the_stop():
    ex = _with_resting_stop()
    c = _client(ex)
    from exchange.binance_client import PositionUnknown
    c.fetch_position_checked = lambda symbol: (_ for _ in ()).throw(PositionUnknown("down"))
    with patch("exchange.binance_client.time.sleep"):
        closed, order, err = c.emergency_close(SYMBOL, ROOT, "long", 0.005)
    # H5 (round 4): the close WAS sent, but flat is unconfirmed -> NOT closed,
    # so _finish_no_stop keeps the pending record and active_bracket
    assert not closed and order is not None and err is not None
    assert [r["clientAlgoId"] for r in ex.algo] == [f"snap-d3-{ROOT}-s"]
    assert len(ex.closes()) == 1               # no second close from here
