"""Exit visibility (24 Nov 2026, SOL stop-out).

1. Every booked fill writes one log line (the bracket stop-out left none).
2. A FIRED gate snapshot stops claiming FIRED in heartbeats once a fill has
   been booked after that evaluation: "in position" after the entry,
   "trade closed" after the exit. No fill yet keeps FIRED.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

import bot
from exchange import state


@pytest.fixture
def db(tmp_path):
    with patch("exchange.state.DB_PATH", tmp_path / "state.db"):
        state.init_db()
        yield


def _bot(gates, eval_ts):
    b = object.__new__(bot.Bot)
    b._latest_gates = gates
    b._gates_eval_ts = eval_ts
    return b


FIRED = {"strategy": "supertrend", "would_fire": "long",
         "waiting_for": "signal fired: long", "gates_long": {"st_flip_up": True}}


def _past(seconds=1):
    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()


def test_record_fill_logs_a_line(db, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="snapback.ledger"):
        state.record_fill(side="close", qty=1.4, price=71.71, pnl_usd=-5.614,
                          reason="bracket_exit", equity_after=155.41821523,
                          client_order_id_root="1795449605503")
    msgs = [r.getMessage() for r in caplog.records if r.name == "snapback.ledger"]
    assert msgs == ["fill booked: close 1.4 @ 71.71 reason=bracket_exit "
                    "pnl=-5.614 equity_after=155.4182 signal_id=1795449605503"]


def test_record_fill_log_handles_missing_values(db, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="snapback.ledger"):
        state.record_fill(side="long", qty=1.4, price=75.72, reason="entry")
    assert "pnl=- equity_after=- signal_id=-" in caplog.records[-1].getMessage()


def test_fired_kept_while_no_fill_since_evaluation(db) -> None:
    state.record_fill(side="close", qty=1.0, price=85.0, reason="bracket_exit")
    b = _bot(dict(FIRED), datetime.now(UTC).isoformat())   # old fill predates eval
    assert b._gates_for_heartbeat()["would_fire"] == "long"


def test_fired_becomes_in_position_after_entry(db) -> None:
    b = _bot(dict(FIRED), _past())
    state.record_fill(side="long", qty=1.4, price=75.72, reason="entry")
    g = b._gates_for_heartbeat()
    assert g["would_fire"] is None and g["fired"] == "long"
    assert g["waiting_for"].startswith("in position: long (entered @ 75.72)")
    assert b._latest_gates["would_fire"] == "long"    # snapshot not mutated


def test_fired_becomes_closed_after_exit(db) -> None:
    b = _bot(dict(FIRED), _past())
    state.record_fill(side="long", qty=1.4, price=75.72, reason="entry")
    state.record_fill(side="close", qty=1.4, price=71.71, reason="bracket_exit")
    g = b._gates_for_heartbeat()
    assert g["would_fire"] is None
    assert g["waiting_for"].startswith("long trade closed (bracket_exit @ 71.71)")


def test_size_rows_do_not_count_as_the_exit(db) -> None:
    b = _bot(dict(FIRED), _past())
    state.record_fill(side="long", qty=1.4, price=75.72, reason="entry")
    state.record_fill(side="close", qty=0.4, price=80.0, reason="external_partial")
    assert b._gates_for_heartbeat()["waiting_for"].startswith("in position")


def test_not_fired_and_missing_eval_ts_pass_through(db) -> None:
    quiet = {"would_fire": None, "waiting_for": "long waiting on st_flip_up"}
    assert _bot(quiet, _past())._gates_for_heartbeat() is quiet
    b = object.__new__(bot.Bot)
    b._latest_gates = dict(FIRED)                         # attr never set
    assert b._gates_for_heartbeat()["would_fire"] == "long"
    assert _bot(None, None)._gates_for_heartbeat() is None


def test_db_error_keeps_snapshot(tmp_path) -> None:
    with patch("exchange.state.DB_PATH", tmp_path / "missing" / "x.db"):
        g = _bot(dict(FIRED), _past())._gates_for_heartbeat()
    assert g["would_fire"] == "long"
