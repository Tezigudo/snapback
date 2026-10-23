"""F8: the catch-all in loop() must never raise out of the loop.

If state.record_event raised inside `except Exception` (e.g. sqlite "database is
locked"), loop() died -> process exit -> systemd restart -> boot() flattens a
live position. The handler body must be unable to raise; backoff is unchanged.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

from tests.test_boot_resume import _make_bot


def test_record_event_failure_in_handler_does_not_exit_loop(tmp_path):
    bot, mc = _make_bot()
    mc.fetch_equity_usdt.side_effect = RuntimeError("boom")
    sleeps: list[float] = []

    def _sleep(s):
        sleeps.append(s)
        bot._stopped = True

    locked = MagicMock(side_effect=sqlite3.OperationalError("database is locked"))
    with patch("bot.halt_source", return_value=None), \
         patch("bot.HEARTBEAT", tmp_path / "hb"), \
         patch("bot.time.sleep", side_effect=_sleep), \
         patch("bot.state.record_event", locked), \
         patch.object(bot, "_maybe_reconcile_principal"):
        assert bot.loop() == 0
    locked.assert_called_once()
    assert sleeps == [min(bot.poll_s * 2, 300)]
