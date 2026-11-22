"""Fake-client helper for the flat-confirmation reads (bot._confirm_flat)."""

from __future__ import annotations

from unittest.mock import DEFAULT


def agree_on_flat(mc, *, set_algo: bool = True):
    """Make a MagicMock exchange client answer the flat-confirmation reads the
    way a real exchange would after a genuine bracket fill (2026-11, N1):

      - `fetch_position_checked` (the confirming re-read) agrees with whatever
        `fetch_position` currently answers — it reads the same mock state, so a
        test that flips `fetch_position.return_value` flips both;
      - set_algo=True: `fetch_algo_orders` reads ([], True) — no bracket leg
        resting, so a flat edge confirms on the same tick, as before the fix;
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

    mc.fetch_position_checked.side_effect = _checked
    if set_algo:
        mc.fetch_algo_orders.return_value = ([], True)
    else:
        mc.fetch_algo_orders.side_effect = _algo
    return mc
