"""Operator entry pause (data/PAUSE_<instance>, 2026-10-27).

The flag blocks NEW entries only: no ohlcv read, no signal evaluation, no
order. It alerts once per pause/resume edge, and removing the file resumes
entries on the next tick without a restart.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import bot
from exchange.env import leg_pause_path

from ._bot_seed import seed_no_stop_state


def _bot(tmp_path, side: str = "flat"):
    b = object.__new__(bot.Bot)
    seed_no_stop_state(b, bot.Bot)
    b.instance = "v1"
    b.symbol = "BTC/USDT:USDT"
    b.entry_tf = "15m"
    b.log = logging.getLogger("test_entry_pause")
    b.client = MagicMock()
    b.client.fetch_position.return_value = SimpleNamespace(side=side)
    b._entry_blocked = lambda: False
    b._daily_loss_blocks_entry = lambda equity: False
    return b


@pytest.fixture
def flag(tmp_path):
    p = tmp_path / "PAUSE_v1"
    with patch("bot.leg_pause_path", lambda inst: tmp_path / f"PAUSE_{inst}"):
        yield p


def test_pause_path_is_per_leg() -> None:
    assert leg_pause_path("v1").name == "PAUSE_v1"
    assert leg_pause_path("v1").parent.name == "data"
    assert leg_pause_path("v1") != leg_pause_path("donchian")


def test_paused_skips_evaluation_and_alerts_once(tmp_path, flag) -> None:
    flag.touch()
    b = _bot(tmp_path)
    with patch("bot.send_alert") as alert:
        for _ in range(3):
            b._maybe_enter(150.0)
    b.client.fetch_ohlcv.assert_not_called()
    assert alert.call_count == 1
    assert "PAUSED" in alert.call_args[0][0]


def test_removing_flag_resumes_without_restart(tmp_path, flag) -> None:
    flag.touch()
    b = _bot(tmp_path)
    b.client.fetch_ohlcv.side_effect = RuntimeError("evaluated")
    with patch("bot.send_alert") as alert:
        b._maybe_enter(150.0)
        flag.unlink()
        with pytest.raises(RuntimeError, match="evaluated"):
            b._maybe_enter(150.0)
    assert [c[0][0] for c in alert.call_args_list] == [
        "v1: entries PAUSED", "v1: entries RESUMED"]


def test_no_flag_no_alert(tmp_path, flag) -> None:
    b = _bot(tmp_path)
    b.client.fetch_ohlcv.side_effect = RuntimeError("evaluated")
    with patch("bot.send_alert") as alert, pytest.raises(RuntimeError):
        b._maybe_enter(150.0)
    alert.assert_not_called()


def test_pause_does_not_touch_an_open_position(tmp_path, flag) -> None:
    flag.touch()
    b = _bot(tmp_path, side="long")
    with patch("bot.send_alert"):
        b._maybe_enter(150.0)
    b.client.close_position.assert_not_called()
    b.client.fetch_ohlcv.assert_not_called()
