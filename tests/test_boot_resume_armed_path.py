"""Boot-resume ARMING PREREQUISITES — the items that had to exist before
`bot.ARMING_PREREQS_BUILT` may ever flip (docs/BOOT_RESUME_PHASE_B_PLAN.md,
"Blockers before arming"). Each block pins the refusal (negative) case first:
a bug here that flattens costs one trade, a bug that adopts leaves a live
position the bot cannot prove is its own.

Runs against the per-test sqlite DB from tests/conftest.py, with the arming
interlock lifted, exactly like tests/test_boot_resume_phase_b.py.
"""
from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from bot_internals import adopt_precheck, can_adopt
from exchange import state
from tests.test_boot_resume import _ARMED_BRACKET, _MINIMAL_PARAMS, _make_bot, _open
from tests.test_boot_resume_phase_b import _ARMED_PARAMS, _boot, _outbox_kinds, _seed_db


@pytest.fixture(autouse=True)
def _lift_arming_interlock():
    with patch("bot.ARMING_PREREQS_BUILT", True):
        yield


def _armed_boot(position=None, **seed):
    _seed_db(**seed)
    bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
    mc.close_position.return_value = {}
    _boot(bot, mc, position if position is not None else _open())
    return bot, mc


def _adopted(mc) -> bool:
    return (not mc.close_position.called) and "boot_adopt" in _outbox_kinds()


def _flattened(mc) -> bool:
    return mc.close_position.called and "boot_flatten" in _outbox_kinds() \
        and "boot_adopt" not in _outbox_kinds()


# ---------------------------------------------------------------------------
# C5 — identity: active_bracket.signal_id == latest entry coid root
# ---------------------------------------------------------------------------

class TestC5Identity:

    def test_matching_root_adopts(self):
        bot, mc = _armed_boot()
        assert _adopted(mc)
        assert bot._last_entry_root == "sig-1"

    def test_mismatched_root_flattens(self):
        """T3: a stale active_bracket from the PREVIOUS position, same side and
        inside the ±2% / qty tolerance, must not be adopted."""
        bot, mc = _armed_boot(bracket={**_ARMED_BRACKET, "signal_id": "sig-OLD"})
        assert _flattened(mc)
        # The flatten still tags the close with the newest entry root.
        mc.close_position.assert_called_once_with(
            "BTC/USDT:USDT", client_order_id_root="sig-1", close_leg="bf")
        assert bot._last_position_side == "unknown"

    def test_bracket_without_signal_id_flattens(self):
        ab = {k: v for k, v in _ARMED_BRACKET.items() if k != "signal_id"}
        _bot, mc = _armed_boot(bracket=ab)
        assert _flattened(mc)

    def test_untagged_latest_entry_flattens(self):
        """root=None nit: an entry fill with no coid root (pre-COID writer) —
        None must never be treated as a match, even against a None sid."""
        ab = {**_ARMED_BRACKET, "signal_id": None}
        _seed_db(bracket=ab, root=None)
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        mc.close_position.return_value = {}
        _boot(bot, mc, _open())
        assert _flattened(mc)

    def test_root_lookup_raising_in_the_gate_flattens(self):
        """Fail closed: an unreadable root refuses; it never adopts untagged."""
        _seed_db()
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        mc.close_position.return_value = {}
        real = state.latest_entry_coid_root
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:          # the gate's read
                raise sqlite3.OperationalError("disk I/O error")
            return real()                # the (unchanged) flatten block's read

        with patch.object(state, "latest_entry_coid_root", side_effect=flaky):
            _boot(bot, mc, _open())
        assert _flattened(mc)
        assert bot._adopt_root is None

    def test_pure_gate_reasons(self):
        p, pos = dict(_MINIMAL_PARAMS), _open()
        raw = json.dumps(_ARMED_BRACKET)
        assert adopt_precheck(p, pos, raw, 0.001, "sig-1")[0] is None
        assert adopt_precheck(p, pos, raw, 0.001, None)[0].startswith(
            "no latest entry coid root")
        assert adopt_precheck(p, pos, raw, 0.001, "")[0].startswith(
            "no latest entry coid root")
        assert adopt_precheck(p, pos, raw, 0.001, "sig-2")[0].startswith(
            "identity mismatch")

    def test_can_adopt_requires_latest_root_keyword(self):
        """No caller can silently skip C5 by omitting the argument."""
        with pytest.raises(TypeError):
            can_adopt(dict(_MINIMAL_PARAMS), _open(), json.dumps(_ARMED_BRACKET),
                      [], [], True, 0.001, "snap-v1-")  # type: ignore[call-arg]

    def test_observe_mode_reports_the_identity_refusal_without_exchange_calls(self):
        """Observe mode now runs C5 too (a LOCAL DB read), still no book read."""
        _seed_db(bracket={**_ARMED_BRACKET, "signal_id": "sig-OLD"})
        bot, mc = _make_bot(params=dict(_MINIMAL_PARAMS))     # observe_only: true
        bot.log = MagicMock()
        ok, why = bot._boot_resume_verdict(_open())
        assert ok is False and why.startswith("identity mismatch")
        names = [c[0] for c in mc.mock_calls]
        assert "ex.fetch_open_orders" not in names
        assert "fetch_algo_orders" not in names
        assert "fetch_position" not in names


# ---------------------------------------------------------------------------
# C9 — bounded retry of an unreadable order book at boot (T7)
# ---------------------------------------------------------------------------

@pytest.fixture
def no_sleep():
    with patch("bot._boot_book_sleep") as sl:
        yield sl


_ALGO_INTACT = [
    {"algoId": "1", "clientAlgoId": "snap-v1-sig-1-s", "side": "SELL",
     "triggerPrice": "64500.0", "strategyType": None, "reduceOnly": True,
     "algoStatus": "NEW"},
    {"algoId": "2", "clientAlgoId": "snap-v1-sig-1-t", "side": "SELL",
     "triggerPrice": "67000.0", "strategyType": None, "reduceOnly": True,
     "algoStatus": "NEW"},
]


def _boot_with(mc_setup, no_sleep):
    _seed_db()
    bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
    mc.close_position.return_value = {}
    mc_setup(mc)
    _boot(bot, mc, _open())
    return bot, mc


class TestC9BookRetry:

    def test_algo_unreadable_twice_then_readable_adopts(self, no_sleep):
        def setup(mc):
            mc.fetch_algo_orders.side_effect = [([], False), ([], False),
                                                (_ALGO_INTACT, True)]
        _bot, mc = _boot_with(setup, no_sleep)
        assert _adopted(mc)
        assert mc.fetch_algo_orders.call_count == 3
        assert mc.ex.fetch_open_orders.call_count == 3   # both books re-read per attempt
        assert [c.args for c in no_sleep.call_args_list] == [(5.0,), (5.0,)]

    def test_plain_book_raising_twice_then_readable_adopts(self, no_sleep):
        def setup(mc):
            mc.ex.fetch_open_orders.side_effect = [RuntimeError("ECONNRESET"),
                                                   RuntimeError("ECONNRESET"), []]
        _bot, mc = _boot_with(setup, no_sleep)
        assert _adopted(mc)
        assert mc.ex.fetch_open_orders.call_count == 3
        assert mc.fetch_algo_orders.call_count == 1      # only read once plain was OK
        assert no_sleep.call_count == 2

    def test_algo_unreadable_three_times_flattens(self, no_sleep):
        def setup(mc):
            mc.fetch_algo_orders.return_value = ([], False)
        bot, mc = _boot_with(setup, no_sleep)
        assert _flattened(mc)
        assert mc.fetch_algo_orders.call_count == 3      # bounded: never a 4th
        assert no_sleep.call_count == 2                  # no sleep after the last
        assert bot._last_position_side == "unknown"

    def test_plain_book_raising_three_times_flattens(self, no_sleep):
        def setup(mc):
            mc.ex.fetch_open_orders.side_effect = RuntimeError("ECONNRESET")
        _bot, mc = _boot_with(setup, no_sleep)
        assert _flattened(mc)
        assert mc.ex.fetch_open_orders.call_count == 3
        mc.fetch_algo_orders.assert_not_called()

    def test_readable_first_time_never_sleeps(self, no_sleep):
        _bot, mc = _boot_with(lambda mc: None, no_sleep)
        assert _adopted(mc)
        no_sleep.assert_not_called()
        assert mc.ex.fetch_open_orders.call_count == 1

    def test_a_precheck_refusal_never_reads_or_waits(self, no_sleep):
        """Retry is spent only on a position the offline checks already accept."""
        _seed_db(bracket={**_ARMED_BRACKET, "signal_id": "sig-OLD"})
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        mc.close_position.return_value = {}
        mc.fetch_algo_orders.return_value = ([], False)
        _boot(bot, mc, _open())
        assert _flattened(mc)
        mc.ex.fetch_open_orders.assert_not_called()
        no_sleep.assert_not_called()

    def test_observe_mode_never_retries(self, no_sleep):
        _seed_db()
        bot, mc = _make_bot(params=dict(_MINIMAL_PARAMS))     # observe_only: true
        mc.close_position.return_value = {}
        mc.fetch_algo_orders.return_value = ([], False)
        _boot(bot, mc, _open())
        assert _flattened(mc)
        mc.ex.fetch_open_orders.assert_not_called()
        no_sleep.assert_not_called()

    def test_the_real_backoff_is_five_seconds_three_attempts(self):
        import bot
        assert bot.BOOT_BOOK_READ_ATTEMPTS == 3
        assert bot.BOOT_BOOK_RETRY_S == 5.0


# ---------------------------------------------------------------------------
# C6 — re-read the position immediately before committing to adopt (T4)
# ---------------------------------------------------------------------------

def _flat_pos():
    from exchange.binance_client import Position
    return Position(symbol="BTC/USDT:USDT", side="flat", qty=0.0, entry_price=0.0,
                    unrealized_pnl=0.0, margin_used=0.0)


def _boot_reads(*positions):
    """boot()'s own read, then the C6 re-read, then any later reads."""
    _seed_db()
    bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.close_position.return_value = None      # close_position no-ops when flat
    mc.fetch_position.side_effect = list(positions)
    from tests.test_boot_resume import _boot_patches, _principal_patches
    with _boot_patches(), _principal_patches():
        bot.boot()
    return bot, mc


class TestC6ReRead:

    def test_unchanged_position_adopts_and_the_re_read_is_the_last_exchange_call(self):
        bot, mc = _boot_reads(_open(), _open())
        assert _adopted(mc)
        names = [c[0] for c in mc.mock_calls]
        i_reread = max(i for i, n in enumerate(names) if n == "fetch_position")
        i_books = max(i for i, n in enumerate(names)
                      if n in ("ex.fetch_open_orders", "fetch_algo_orders"))
        assert i_books < i_reread
        assert mc.fetch_position.call_count == 2

    def test_position_closed_between_read_and_adopt_flattens(self):
        """The bracket filled while the gate read the books. Refuse → the
        flatten block runs on the stale `pos`; close_position re-reads and
        no-ops when flat."""
        bot, mc = _boot_reads(_open(), _flat_pos())
        assert "boot_adopt" not in _outbox_kinds()
        mc.close_position.assert_called_once_with(
            "BTC/USDT:USDT", client_order_id_root="sig-1", close_leg="bf")
        # close_position returned None (already flat): no fabricated $0 exit,
        # the surviving sibling is swept instead (see TestAlreadyFlatAtFlatten).
        assert "boot_flatten" not in _outbox_kinds()
        mc.cancel_open_orders.assert_called_once_with("BTC/USDT:USDT",
                                                      coid_prefix="snap-v1-", report_unknown=True)
        assert bot._last_position_side == "unknown"
        # A refusal here spent no adopt-loop budget.
        assert state.get_meta("boot_adopt_log") is None

    @pytest.mark.parametrize("changed", [
        dict(qty=0.005),                 # partial manual close
        dict(side="short"),              # replaced
        dict(entry_price=65010.0),       # re-entered inside the 2% band
    ])
    def test_any_identity_change_flattens(self, changed):
        _bot, mc = _boot_reads(_open(), _open(**changed))
        assert "boot_adopt" not in _outbox_kinds()
        mc.close_position.assert_called_once()

    def test_re_read_raising_flattens(self):
        _bot, mc = _boot_reads(_open(), RuntimeError("timeout"))
        assert "boot_adopt" not in _outbox_kinds()
        mc.close_position.assert_called_once()

    def test_pure_gate_parity_is_unaffected(self):
        """C6 lives in the armed wrapper (_prepare_adopt), not in can_adopt, so the
        probe's single-snapshot verdict stays byte-comparable with the gate."""
        import inspect
        import bot_internals
        assert "fetch_position" not in inspect.getsource(bot_internals.can_adopt)


# ---------------------------------------------------------------------------
# C11 — "Position RESUMED after restart" alert; C10 — cap-alert latch seed
# ---------------------------------------------------------------------------

def _boot_capture_alerts(position=None, bracket=_ARMED_BRACKET, alert_effect=None,
                         params=None):
    from tests.test_boot_resume import _principal_patches
    _seed_db(bracket=bracket)
    bot, mc = _make_bot(params=dict(params or _ARMED_PARAMS))
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.close_position.return_value = {}
    pos = position if position is not None else _open()
    mc.fetch_position.return_value = pos
    alert = MagicMock(side_effect=alert_effect)
    with patch.multiple("bot", check_symbol=MagicMock(), check_leverage=MagicMock(),
                        send_alert=alert), _principal_patches():
        bot.boot()
    return bot, mc, alert


_RESUMED = "Position RESUMED after restart"


def _subjects(alert) -> list[str]:
    return [c.args[0] for c in alert.call_args_list]


def _resumed(alert) -> list:
    """Only the C11 calls (boot also sends its usual 'Bot deploy start')."""
    return [c for c in alert.call_args_list if c.args[0] == _RESUMED]


def _smtp_down_for_resume(subject, *_a, **_k):
    if subject == _RESUMED:
        raise RuntimeError("SMTP down")


class TestC11ResumeAlert:

    def test_adopt_sends_exactly_one_resumed_alert(self):
        bot, mc, alert = _boot_capture_alerts()
        assert _adopted(mc)
        assert len(_resumed(alert)) == 1
        body = _resumed(alert)[0].args[1]
        assert "LONG 0.0100 BTC" in body and "65,000.00" in body
        assert "signal_id: sig-1" in body
        assert "bracket intact" in body               # the gate's reason
        assert "data/HALT_" in body                   # how to close instead
        assert "cap is ALREADY SPENT" not in body

    def test_refusal_sends_no_resumed_alert(self):
        _bot, mc, alert = _boot_capture_alerts(
            bracket={**_ARMED_BRACKET, "signal_id": "sig-OLD"})
        assert _flattened(mc)
        assert "Position RESUMED after restart" not in _subjects(alert)

    def test_observe_mode_sends_no_resumed_alert(self):
        _bot, mc, alert = _boot_capture_alerts(params=_MINIMAL_PARAMS)
        assert _flattened(mc)
        assert _resumed(alert) == []

    def test_a_failing_alert_never_aborts_the_adopt(self):
        bot, mc, alert = _boot_capture_alerts(alert_effect=_smtp_down_for_resume)
        assert _adopted(mc)
        assert len(_resumed(alert)) == 1
        assert bot._last_position_side == "long"

    def test_alert_survives_a_bookkeeping_failure(self):
        """The alert is sent BEFORE the sqlite bookkeeping, in its own try."""
        with patch.object(state, "record_event",
                          side_effect=sqlite3.OperationalError("locked")):
            _bot, mc, alert = _boot_capture_alerts()
        mc.close_position.assert_not_called()
        assert len(_resumed(alert)) == 1


class TestC10CapLatchSeed:

    def test_spent_cap_seeds_the_latch_and_says_so_in_the_alert(self):
        bot, mc, alert = _boot_capture_alerts(
            bracket={**_ARMED_BRACKET, "reprotect_count": 3})   # cap 3, intact
        assert _adopted(mc)
        assert bot._reprotect_capped_alerted is True
        assert "cap is ALREADY SPENT" in _resumed(alert)[0].args[1]

    def test_unspent_cap_leaves_the_latch_clear(self):
        bot, _mc, _alert = _boot_capture_alerts(
            bracket={**_ARMED_BRACKET, "reprotect_count": 2})
        assert bot._reprotect_capped_alerted is False

    def test_flatten_path_does_not_touch_the_latch(self):
        bot, mc, _alert = _boot_capture_alerts(
            bracket={**_ARMED_BRACKET, "reprotect_count": 3, "signal_id": "sig-OLD"})
        assert _flattened(mc)
        assert bot._reprotect_capped_alerted is False


# ---------------------------------------------------------------------------
# Blocker 3 — an absent STOP is never adoptable (mark may be through it)
# Blocker 4 — only OUR plain-book legs count as our bracket
# ---------------------------------------------------------------------------

_SL, _TP = _ALGO_INTACT


def _gate(open_orders=(), algo=(), bracket=_ARMED_BRACKET):
    return can_adopt(dict(_MINIMAL_PARAMS), _open(), json.dumps(bracket),
                     list(open_orders), list(algo), True, 0.001, "snap-v1-",
                     latest_root="sig-1")


def _plain(kind: str, coid: str | None, side: str = "sell") -> dict:
    info = {"reduceOnly": "true", "type": kind}
    o = {"info": info, "type": kind.lower(), "side": side}
    if coid is not None:
        o["clientOrderId"] = coid
    return o


class TestBlocker3StopMissing:

    def test_no_stop_at_all_refuses_even_with_reprotect_budget(self):
        ok, why = _gate(algo=[], bracket={**_ARMED_BRACKET, "reprotect_count": 0})
        assert ok is False and why.startswith("stop leg missing")

    def test_tp_present_stop_missing_refuses(self):
        ok, why = _gate(algo=[_TP])
        assert ok is False and "SL=MISSING" in why

    def test_stop_present_tp_missing_with_budget_adopts(self):
        ok, why = _gate(algo=[_SL])
        assert ok is True and "TP=MISSING" in why

    def test_armed_boot_with_no_stop_flattens(self):
        _seed_db()
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        mc.close_position.return_value = {}
        mc.fetch_algo_orders.return_value = ([_TP], True)
        _boot(bot, mc, _open())
        assert _flattened(mc)


class TestBlocker4PlainPrefixScope:

    def test_foreign_plain_stop_does_not_count_as_ours(self):
        """A manual reduce-only STOP_MARKET in the plain book + our TP only:
        before scoping this read as intact and adopted."""
        ok, why = _gate(open_orders=[_plain("STOP_MARKET", "manual-123")], algo=[_TP])
        assert ok is False and why.startswith("stop leg missing")

    def test_untagged_plain_stop_does_not_count_as_ours(self):
        ok, _why = _gate(open_orders=[_plain("STOP_MARKET", None)], algo=[_TP])
        assert ok is False

    def test_other_leg_prefix_does_not_count(self):
        ok, _why = _gate(open_orders=[_plain("STOP_MARKET", "snap-dc-sig-9-s")],
                         algo=[_TP])
        assert ok is False

    def test_our_plain_stop_counts(self):
        ok, why = _gate(open_orders=[_plain("STOP_MARKET", "snap-v1-sig-1-s")],
                        algo=[_TP])
        assert ok is True and "intact" in why

    def test_coid_in_info_only_counts(self):
        o = _plain("STOP_MARKET", None)
        o["info"]["clientOrderId"] = "snap-v1-sig-1-s"
        ok, _why = _gate(open_orders=[o], algo=[_TP])
        assert ok is True

    def test_shared_bracket_state_is_unchanged_for_reprotect(self):
        """Scoping lives in the adopt gate ONLY. The live re-placer's view of a
        foreign plain stop must not change (it still reads as SL present)."""
        from bot_internals import bracket_state
        st = bracket_state([_plain("STOP_MARKET", "manual-123")], [_TP],
                           "snap-v1-", True)
        assert st.sl is True and st.intact is True


# ---------------------------------------------------------------------------
# Blocker 8 — T9 / T10: HALT and the kill switch still win after an adopt
# ---------------------------------------------------------------------------

def _adopt_then_loop(tmp_path, *, halt_by=None, equity_in_loop=1000.0,
                     principal_p=None):
    """Adopt for real, then run loop() until it returns (it must, on tick 1)."""
    bot, mc, _alert = _boot_capture_alerts()
    assert _adopted(mc)
    mc.close_position.reset_mock()
    mc.fetch_equity_usdt.return_value = equity_in_loop
    bot.halt_path = tmp_path / "HALT_v1"
    with patch.multiple("bot", send_alert=MagicMock(),
                        halt_source=MagicMock(return_value=halt_by)), \
         patch("bot.consolidate_push.drain"), \
         patch.object(bot, "_heartbeat"), \
         patch.object(bot, "_maybe_reconcile_principal"), \
         patch("bot.principal.get_principal", return_value=principal_p), \
         patch.object(bot, "_detect_bracket_exit",
                      side_effect=AssertionError("tick ran past HALT/kill")):
        rc = bot.loop()
    return bot, mc, rc


class TestHaltAndKillAfterAdopt:

    def test_t9_halt_after_adopt_closes_on_the_first_tick(self, tmp_path):
        _bot, mc, rc = _adopt_then_loop(tmp_path, halt_by="self")
        assert rc == 0
        mc.close_position.assert_called_once_with(
            "BTC/USDT:USDT", client_order_id_root="sig-1", close_leg="h")
        assert "halt" in _outbox_kinds()

    def test_t10_kill_switch_after_adopt_closes_on_the_first_tick(self, tmp_path):
        # principal 1000, kill fraction 0.5 (test params) → floor 500.
        bot, mc, rc = _adopt_then_loop(tmp_path, equity_in_loop=100.0,
                                       principal_p=1000.0)
        assert rc == 0
        mc.close_position.assert_called_once_with(
            "BTC/USDT:USDT", client_order_id_root="sig-1", close_leg="k")
        assert bot.halt_path.exists()
        assert "kill_switch" in _outbox_kinds()


# ---------------------------------------------------------------------------
# Panel MUST-fix — a leg counts only with the EXACT root and the CLOSING side
# ---------------------------------------------------------------------------

def _algo(coid: str, side: str = "SELL") -> dict:
    return {"algoId": "9", "clientAlgoId": coid, "side": side,
            "triggerPrice": "64500.0", "strategyType": None, "reduceOnly": True,
            "algoStatus": "NEW"}


class TestExactRootAndSide:

    def test_red_team_old_root_stop_plus_new_tp_refuses(self):
        """Red-team repro: long, signal_id NEW, books = a stale BUY stop
        `snap-v1-OLD-s` + `NEW-t`. Prefix-only matching said 'bracket intact'."""
        ab = {**_ARMED_BRACKET, "signal_id": "NEW"}
        ok, why = can_adopt(dict(_MINIMAL_PARAMS), _open(), json.dumps(ab), [],
                            [_algo("snap-v1-OLD-s", "BUY"), _algo("snap-v1-NEW-t")],
                            True, 0.001, "snap-v1-", latest_root="NEW")
        assert ok is False and why.startswith("stop leg missing")

    def test_old_root_stop_on_the_right_side_refuses(self):
        ok, _why = _gate(algo=[_algo("snap-v1-OLD-s"), _TP])
        assert ok is False

    def test_wrong_side_stop_with_the_right_root_refuses(self):
        """A BUY stop cannot close a long."""
        ok, why = _gate(algo=[_algo("snap-v1-sig-1-s", "BUY"), _TP])
        assert ok is False and "SL=MISSING" in why

    def test_wrong_side_plain_stop_refuses(self):
        ok, _why = _gate(open_orders=[_plain("STOP_MARKET", "snap-v1-sig-1-s", "buy")],
                         algo=[_TP])
        assert ok is False

    def test_right_root_right_side_adopts(self):
        ok, why = _gate(algo=[_algo("snap-v1-sig-1-s"), _algo("snap-v1-sig-1-t")])
        assert ok is True and "intact" in why

    def test_breakeven_sb_stop_counts(self):
        ok, why = _gate(algo=[_algo("snap-v1-sig-1-sb"), _TP])
        assert ok is True and "intact" in why

    def test_short_position_needs_buy_legs(self):
        ab = {**_ARMED_BRACKET, "side": "short"}
        args = (dict(_MINIMAL_PARAMS), _open(side="short"), json.dumps(ab), [])
        ok, _ = can_adopt(*args, [_algo("snap-v1-sig-1-s", "BUY"),
                                  _algo("snap-v1-sig-1-t", "BUY")],
                          True, 0.001, "snap-v1-", latest_root="sig-1")
        assert ok is True
        ok, _ = can_adopt(*args, [_algo("snap-v1-sig-1-s", "SELL"),
                                  _algo("snap-v1-sig-1-t", "SELL")],
                          True, 0.001, "snap-v1-", latest_root="sig-1")
        assert ok is False

    def test_root_that_is_a_prefix_of_another_does_not_match(self):
        """Exact match, not startswith: root `sig-1` must not accept `sig-10-s`."""
        ok, _why = _gate(algo=[_algo("snap-v1-sig-10-s"), _TP])
        assert ok is False

    def test_live_reprotect_view_is_unchanged(self):
        """bracket_state (live reprotect) still counts the stale-root stop —
        the blind spot is reported to God, not changed here."""
        from bot_internals import bracket_state
        st = bracket_state([], [_algo("snap-v1-OLD-s", "BUY"), _TP], "snap-v1-", True)
        assert st.intact is True


# ---------------------------------------------------------------------------
# Panel SHOULD-fix — flatten finds the exchange already flat
# ---------------------------------------------------------------------------

def _observe_boot(close_ret, *, alert=None):
    """Observe mode (what ships): every open position goes to the flatten."""
    from tests.test_boot_resume import _principal_patches
    _seed_db()
    bot, mc = _make_bot(params=dict(_MINIMAL_PARAMS))
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.fetch_position.return_value = _open()
    mc.close_position.return_value = close_ret
    mc.cancel_open_orders.return_value = 1
    with patch.multiple("bot", check_symbol=MagicMock(), check_leverage=MagicMock(),
                        send_alert=alert or MagicMock()), _principal_patches():
        bot.boot()
    return bot, mc


def _smtp_down_for_flat(subject, *_a, **_k):
    if subject.startswith("Position already closed during boot"):
        raise RuntimeError("smtp")


def _events(kind: str) -> list:
    with sqlite3.connect(state.DB_PATH) as c:
        return c.execute("SELECT kind FROM events WHERE kind=?", (kind,)).fetchall()


class TestAlreadyFlatAtFlatten:

    def test_old_code_would_have_recorded_a_zero_dollar_exit(self):
        """Re-derive the pre-fix behaviour from the block itself: without the
        early return, close_order=None falls through to `or pos.entry_price`."""
        import bot as botmod
        assert botmod._exit_pnl_usd("long", 65000.0, None or 65000.0, 0.01) == 0.0

    def test_already_flat_emits_no_boot_flatten_and_sweeps(self):
        alert = MagicMock()
        _bot, mc = _observe_boot(None, alert=alert)
        assert "boot_flatten" not in _outbox_kinds()
        assert _events("boot_flatten") == []
        assert len(_events("boot_flatten_already_flat")) == 1
        mc.cancel_open_orders.assert_called_once_with("BTC/USDT:USDT",
                                                      coid_prefix="snap-v1-",
                                                      report_unknown=True)
        mc.fetch_equity_usdt.assert_called_once()        # boot's own read only
        subj = [c.args[0] for c in alert.call_args_list]
        assert "Position already closed during boot — exit not recorded" in subj

    def test_unreadable_book_is_reported_not_called_swept_zero(self):
        def _body(n):
            alert = MagicMock()
            _seed_db()
            bot, mc = _make_bot(params=dict(_MINIMAL_PARAMS))
            mc.fetch_equity_usdt.return_value = 1000.0
            mc.fetch_position.return_value = _open()
            mc.close_position.return_value = None
            mc.cancel_open_orders.return_value = n
            from tests.test_boot_resume import _principal_patches
            with patch.multiple("bot", check_symbol=MagicMock(),
                                check_leverage=MagicMock(), send_alert=alert), \
                 _principal_patches():
                bot.boot()
            return alert.call_args_list[-1].args[1]
        assert "UNREADABLE" in _body(None)
        clean = _body(0)
        assert "cancelled 0 resting" in clean and "UNREADABLE" not in clean

    def test_a_failing_sweep_or_alert_never_aborts_boot(self):
        _seed_db()
        bot, mc = _make_bot(params=dict(_MINIMAL_PARAMS))
        mc.fetch_equity_usdt.return_value = 1000.0
        mc.fetch_position.return_value = _open()
        mc.close_position.return_value = None
        mc.cancel_open_orders.side_effect = RuntimeError("503")
        from tests.test_boot_resume import _principal_patches
        with patch.multiple("bot", check_symbol=MagicMock(), check_leverage=MagicMock(),
                            send_alert=MagicMock(side_effect=_smtp_down_for_flat)), \
             _principal_patches():
            bot.boot()                                    # must not raise
        assert "boot_flatten" not in _outbox_kinds()

    def test_normal_flatten_is_unchanged(self):
        """A real close order: same exchange calls as droplet (close, then the
        post-close equity read), boot_flatten emitted, no extra sweep."""
        _bot, mc = _observe_boot({"id": "1", "average": 65100.0})
        names = [c[0] for c in mc.mock_calls if not c[0].startswith("ex.market")]
        assert names[-3:] == ["fetch_position", "close_position", "fetch_equity_usdt"]
        mc.cancel_open_orders.assert_not_called()
        assert "boot_flatten" in _outbox_kinds()
        assert _events("boot_flatten_already_flat") == []


class TestHaltAtBootRefusesBeforeAdopting:

    @pytest.mark.parametrize("halt_by", ["self", "global"])
    def test_halt_present_flattens_without_spending_a_slot_or_emailing(self, halt_by):
        _seed_db()
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        bot.log = MagicMock()
        with patch("bot.halt_source", return_value=halt_by), \
             patch.object(bot, "_can_adopt") as gate:
            ok, why = bot._boot_resume_verdict(_open())
        assert ok is False and why == f"HALT flag present ({halt_by})"
        gate.assert_not_called()                          # no book reads
        assert state.get_meta("boot_adopt_log") is None   # no slot spent

    def test_halt_check_raising_refuses(self):
        bot, _mc = _make_bot(params=dict(_ARMED_PARAMS))
        bot.log = MagicMock()
        with patch("bot.halt_source", side_effect=OSError("EIO")):
            assert bot._boot_resume_verdict(_open()) == (False, "HALT check raised")

    def test_observe_mode_never_checks_halt(self):
        bot, _mc = _make_bot(params=dict(_MINIMAL_PARAMS))
        bot.log = MagicMock()
        with patch("bot.halt_source") as hs:
            bot._boot_resume_verdict(_open())
        hs.assert_not_called()


class TestPendingNoStopRecordRefusesAdopt:
    """An armed boot must flatten, not adopt, while a no-stop close / unverified
    entry record exists (or its meta is unreadable); records clear after."""

    @pytest.mark.parametrize("key,rec", [
        ("no_stop_close_pending", {"kind": "no_stop_close", "root": "sig-1"}),
        ("entry_unverified", {"kind": "entry_unverified", "root": "sig-1"}),
    ])
    def test_record_flattens_and_is_cleared_after(self, key, rec):
        _seed_db()
        state.set_meta(key, json.dumps(rec))
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        mc.close_position.return_value = {}
        mc.cancel_entry_order.return_value = True
        _boot(bot, mc, _open())
        assert _flattened(mc)
        assert not state.get_meta(key)

    def test_unreadable_meta_flattens(self):
        _seed_db()
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        with patch.object(bot, "_meta_record_ex", return_value=(None, False)):
            ok, why = bot._boot_resume_verdict(_open())
        assert ok is False and "unreadable" in why

    def test_no_record_still_adopts(self):
        bot, mc = _armed_boot()
        assert _adopted(mc)
