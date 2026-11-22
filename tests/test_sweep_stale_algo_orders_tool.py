"""G6 (round 3): tools/sweep_stale_algo_orders.py never cancels a live SL,
never touches another prefix, and aborts on an UNKNOWN position."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import sweep_stale_algo_orders as tool  # noqa: E402


def _row(coid, side, aid):
    return {"algoId": aid, "clientAlgoId": coid, "side": side, "reduceOnly": True,
            "orderType": "STOP_MARKET", "triggerPrice": "1"}


ROWS = [
    _row("snap-d3-111-s", "SELL", "1"),     # live SL of the open long
    _row("snap-d3-111-t", "SELL", "2"),     # live TP
    _row("snap-d3-099-s", "BUY", "3"),      # an old SHORT's SL: wrong side, but an SL
    _row("snap-d3-099-t", "BUY", "4"),      # orphan TP of a closed short
    _row("snap-v1-555-s", "SELL", "5"),     # another leg
    _row("x-cvBPrNm9abc", "SELL", "6"),     # manual / spam
]


def _ex(positions, account_amt=None):
    ex = MagicMock()
    ex.fetch_positions.return_value = positions
    ex.market.return_value = {"id": "BTCUSDT"}
    if account_amt is None:
        del ex.fapiPrivateV2GetAccount
    else:
        ex.fapiPrivateV2GetAccount.return_value = {
            "positions": [{"symbol": "BTCUSDT", "positionAmt": str(account_amt)}]}
    ex.fapiPrivateGetOpenAlgoOrders.return_value = [dict(r) for r in ROWS]
    return ex


def _cancelled(ex):
    return sorted(c.args[0]["algoId"] for c in ex.fapiPrivateDeleteAlgoOrder.call_args_list)


def test_open_position_never_cancels_an_sl_or_another_prefix():
    ex = _ex([{"symbol": "BTC/USDT:USDT", "contracts": 0.01, "side": "long"}])
    assert tool.main(["--prefix", "snap-d3-", "--execute"], ex=ex) == 0
    assert _cancelled(ex) == ["4"]          # only our wrong-side non-SL orphan


def test_unknown_position_aborts_and_cancels_nothing():
    ex = _ex([], account_amt=None)          # empty positionRisk, no confirmation
    assert tool.main(["--prefix", "snap-d3-", "--execute"], ex=ex) == 3
    ex.fapiPrivateDeleteAlgoOrder.assert_not_called()


def test_stale_empty_positionrisk_with_open_account_position_aborts():
    ex = _ex([], account_amt=0.01)
    assert tool.main(["--prefix", "snap-d3-", "--execute"], ex=ex) == 3
    ex.fapiPrivateDeleteAlgoOrder.assert_not_called()


def test_confirmed_flat_cancels_only_our_prefix():
    ex = _ex([], account_amt=0.0)
    assert tool.main(["--prefix", "snap-d3-", "--execute"], ex=ex) == 0
    assert _cancelled(ex) == ["1", "2", "3", "4"]


def test_dry_run_cancels_nothing():
    ex = _ex([], account_amt=0.0)
    assert tool.main(["--prefix", "snap-d3-"], ex=ex) == 0
    ex.fapiPrivateDeleteAlgoOrder.assert_not_called()


def test_position_flipping_between_reads_cancels_nothing():
    # sourcery 4657588350: flat at the first read, a live long (with its
    # bracket) by the pre-execute re-read -> abort, cancel nothing
    zero = {"symbol": "BTC/USDT:USDT", "contracts": 0.0, "side": "long"}
    long_ = {"symbol": "BTC/USDT:USDT", "contracts": 0.01, "side": "long"}
    ex = _ex([zero], account_amt=0.0)
    ex.fetch_positions.side_effect = [[zero], [long_]]
    assert tool.main(["--prefix", "snap-d3-", "--execute"], ex=ex) == 3
    ex.fapiPrivateDeleteAlgoOrder.assert_not_called()


def test_algo_book_is_read_before_the_position():
    ex = _ex([], account_amt=0.0)
    order = []
    ex.fapiPrivateGetOpenAlgoOrders.side_effect = lambda *a, **k: (
        order.append("book"), [dict(r) for r in ROWS])[1]
    ex.fetch_positions.side_effect = lambda *a, **k: (order.append("pos"), [])[1]
    tool.main(["--prefix", "snap-d3-"], ex=ex)
    assert order[0] == "book" and "pos" in order
