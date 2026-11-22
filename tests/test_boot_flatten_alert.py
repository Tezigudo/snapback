"""Every boot flatten mails (2026-11 review, N7). Boot-resume is NOT armed by
this change; the alert carries its refusal reason."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.test_close_then_sweep import _bot, _boot_ctx
from tests.test_orphan_bracket_cleanup import _open


def _boot(b):
    with _boot_ctx(), patch("bot.send_alert") as alert, patch("bot.time.sleep"):
        b.boot()
    return alert


def _flatten_mails(alert):
    return [c for c in alert.call_args_list if "BOOT FLATTENED" in c.args[0]]


def test_boot_flatten_alert_has_leg_side_size_reason():
    b, mc = _bot()
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.close_position.return_value = {"id": "1", "average": 64_900.0}
    alert = _boot(b)
    mails = _flatten_mails(alert)
    assert len(mails) == 1
    subj, body = mails[0].args
    assert "LONG" in subj and "0.0100" in subj
    assert b.instance in body and "Side: long" in body and "Size: 0.010000" in body
    assert "Reason:" in body and "not resumed" in body and "boot_resume" in body
    assert "Flat confirmed" in body


def test_unconfirmed_boot_flatten_books_nothing_and_alerts_close_failed():
    """F3: flat NOT confirmed -> no BOOT FLATTENED mail, no boot_flatten event /
    PnL, nothing swept, no-stop records kept; the close-failed alert instead."""
    import bot as bot_mod
    b, mc = _bot()
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.close_position.side_effect = None
    mc.close_position.return_value = {"id": "1", "average": 64_900.0}
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _open("long", 65_000.0, 0.01)
    b._clear_no_stop_records_at_boot = MagicMock()
    with _boot_ctx(), patch("bot.send_alert") as alert, patch("bot.time.sleep"):
        b.boot()
        kinds = [c.args[0] for c in bot_mod.state.enqueue_bot_event.call_args_list]
        events = [c.args[1] for c in bot_mod.state.record_event.call_args_list]
    assert "boot_flatten" not in kinds and "boot_flatten" not in events
    assert not _flatten_mails(alert)
    assert any("close FAILED" in c.args[0] for c in alert.call_args_list)
    mc.cancel_open_orders.assert_not_called()
    b._clear_no_stop_records_at_boot.assert_not_called()


def test_flat_at_boot_unconfirmed_leaves_orders_and_records():
    from exchange.binance_client import PositionUnknown
    from tests.test_orphan_bracket_cleanup import _flat
    b, mc = _bot()
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.fetch_position.return_value = _flat()
    mc.fetch_position_checked.side_effect = PositionUnknown("stale empty")
    b._clear_no_stop_records_at_boot = MagicMock()
    alert = _boot(b)
    mc.cancel_open_orders.assert_not_called()
    b._clear_no_stop_records_at_boot.assert_not_called()
    assert any("could not confirm flat" in c.args[0] for c in alert.call_args_list)


def test_flat_at_boot_confirmed_still_sweeps_and_clears():
    from tests.test_orphan_bracket_cleanup import _flat
    b, mc = _bot()
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.fetch_position.return_value = _flat()
    b._clear_no_stop_records_at_boot = MagicMock()
    _boot(b)
    mc.cancel_open_orders.assert_called_once()
    b._clear_no_stop_records_at_boot.assert_called_once()


def test_already_flat_at_flatten_clears_no_stop_records_once():
    b, mc = _bot()
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.close_position.side_effect = None
    mc.close_position.return_value = None          # flat by the time it looked
    from tests.test_orphan_bracket_cleanup import _flat
    mc.fetch_position_checked.side_effect = None
    mc.fetch_position_checked.return_value = _flat()
    b._clear_no_stop_records_at_boot = MagicMock()
    _boot(b)
    b._clear_no_stop_records_at_boot.assert_called_once()


def test_boot_flatten_alert_failure_never_blocks_boot():
    b, mc = _bot()
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.close_position.return_value = {"id": "1", "average": 64_900.0}
    def _raise_on_flatten_mail(subject, *a, **k):
        if "BOOT FLATTENED" in subject:
            raise RuntimeError("smtp down")
    boom = MagicMock(side_effect=_raise_on_flatten_mail)
    with _boot_ctx(), patch("bot.send_alert", boom), patch("bot.time.sleep"):
        b.boot()   # must not raise
    mc.close_position.assert_called_once()
    assert any("BOOT FLATTENED" in c.args[0] for c in boom.call_args_list)
