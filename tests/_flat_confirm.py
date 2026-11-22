"""Fake-client helper for the flat-confirmation reads (bot._confirm_flat)."""

from __future__ import annotations

from unittest.mock import DEFAULT


def _flat_like(pos):
    from exchange.binance_client import Position
    return Position(symbol=getattr(pos, "symbol", "BTC/USDT:USDT"), side="flat",
                    qty=0.0, entry_price=0.0, unrealized_pnl=0.0, margin_used=0.0)


def agree_on_flat(mc, *, set_algo: bool = True):
    """Make a MagicMock exchange client answer the flat-confirmation reads the
    way a real exchange would after a genuine bracket fill (2026-11, N1):

      - `fetch_position_checked` (the confirming re-read) agrees with whatever
        `fetch_position` currently answers — it reads the same mock state, so a
        test that flips `fetch_position.return_value` flips both;
      - set_algo=True: `fetch_algo_orders` reads ([], True) — no bracket leg
        resting, so a flat edge confirms on the same tick, as before the fix;
      - `close_position` (unless the test scripted it) leaves the position
        flat, as a filled reduce-only market close does;
      - set_algo=False: the test's own algo book is kept WHILE the position is
        open, and reads empty once `fetch_position` says flat — a filled leg
        leaves the book (the sibling's sweep is the bot's job). A test that
        reassigns `fetch_algo_orders.return_value` later is still honoured.

    Tests about the confirmation itself (tests/test_flat_confirmation.py) do
    NOT use this; they script each read explicitly.
    """
    def _current():
        if mc.fetch_position.side_effect is None:
            return mc.fetch_position.return_value
        return None  # scripted sequence: unknowable without consuming it

    def _checked(*a, **k):
        if mc.fetch_position.side_effect is None:
            return mc.fetch_position.return_value
        return mc.fetch_position(*a, **k)

    def _algo(*a, **k):
        pos = _current()
        if pos is not None and getattr(pos, "side", None) == "flat":
            return [], True
        return DEFAULT  # → fetch_algo_orders.return_value, read at call time

    def _close(*a, **k):
        # A reduce-only market close fills: the position reads flat afterwards
        # (bot._close_then_sweep confirms that before it sweeps). Only when the
        # test has not scripted fetch_position itself.
        pos = _current()
        if pos is not None and getattr(pos, "side", "flat") != "flat":
            mc.fetch_position.return_value = _flat_like(pos)
        return DEFAULT

    mc.fetch_position_checked.side_effect = _checked
    if mc.close_position.side_effect is None:
        mc.close_position.side_effect = _close
    if set_algo:
        mc.fetch_algo_orders.return_value = ([], True)
    else:
        mc.fetch_algo_orders.side_effect = _algo
    return mc
