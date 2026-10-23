"""Tests for boot-resume — adopting an open position instead of flattening it.

PHASE A ships with `observe_only: true`, so the behaviour under test is
deliberately "decide, log, then flatten anyway". The tests are split so the
gate's verdict and the rollout's effect are checked independently:

1. `_can_adopt` — the pure gate. Every refusal reason, and the two accept paths.
2. `_boot_resume_verdict` — the rollout wrapper. Observe-only must never return
   True, a disabled block must never call the gate, and a raising gate must not
   propagate.
3. `boot()` — Phase A must still flatten, and the armed path must not.

The refusal cases matter more than the accept cases: this gate is fail-closed by
design, and a bug that flattens is a lost trade, while a bug that adopts is an
unprotected live position.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from exchange.binance_client import Position


@pytest.fixture(autouse=True)
def _lift_arming_interlock():
    """This module exercises the ARMED code path, which production cannot
    reach while bot.ARMING_PREREQS_BUILT is False (see
    tests/test_boot_resume_flag_off.py for the interlock itself)."""
    with patch("bot.ARMING_PREREQS_BUILT", True):
        yield


@pytest.fixture(autouse=True)
def _no_c9_backoff():
    """C9 retries an unreadable book with a real 5 s back-off; never wait in tests."""
    with patch("bot._boot_book_sleep") as sl:
        yield sl


_MINIMAL_PARAMS: dict = {
    "symbol": "BTC/USDT:USDT",
    "hedge": {"enabled": False, "client_order_id_prefix": "snap-v1-"},
    "execution": {
        "poll_interval_s": "5",
        "order_type": "market",
        "limit_offset_bps": "0",
        "limit_timeout_s": "25",
    },
    "timeframes": {"entry": "15m"},
    "sizing": {"leverage": 5, "risk_per_trade_pct": 1.0},
    "deploy": {"kill_switch_equity_fraction": 0.5, "min_capital_warn_usdt": 1.0},
    "strategy": {},
    # Armed reprotect is the precondition for adoption.
    "reprotect": {"enabled": True, "observe_only": False,
                  "max_replaces_per_position": 3},
    "boot_resume": {"enabled": True, "observe_only": True},
}

_ARMED_BRACKET = {
    "signal_id": "sig-1", "side": "long", "entry_price": 65000.0,
    "qty": 0.01, "sl_distance": 500.0, "tp_distance": 2000.0, "place_tp": True,
}


def _mock_client(coid_prefix: str = "snap-v1-") -> MagicMock:
    mc = MagicMock()
    mc.ex.parse_timeframe.return_value = 900
    mc.coid_prefix = coid_prefix
    mc.env = "testnet"
    mc.cancel_open_orders.return_value = 0
    # Healthy bracket in the ALGO book by default: both legs present.
    # Shape copied from the live row in algo_bracket_leg's docstring —
    # reduceOnly is a top-level BOOL and there is no type field at all. An
    # under-specified fake here silently exercises the "bracket missing" path
    # instead of the intact one, which is exactly what it did on first run.
    mc.ex.fetch_open_orders.return_value = []
    mc.fetch_algo_orders.return_value = ([
        {"algoId": "1000000000000002", "clientAlgoId": "snap-v1-sig-1-s",
         "side": "SELL", "triggerPrice": "64500.0", "strategyType": None,
         "reduceOnly": True, "algoStatus": "NEW"},
        {"algoId": "1000000000000003", "clientAlgoId": "snap-v1-sig-1-t",
         "side": "SELL", "triggerPrice": "67000.0", "strategyType": None,
         "reduceOnly": True, "algoStatus": "NEW"},
    ], True)
    return mc


def _make_bot(dry_run: bool = False, params: dict | None = None):
    from bot import Bot
    mc = _mock_client()
    with patch("bot.BinanceClient.from_env", return_value=mc):
        bot = Bot(params=params or dict(_MINIMAL_PARAMS), dry_run=dry_run)
    return bot, mc


def _open(side: str = "long", entry_price: float = 65000.0, qty: float = 0.01):
    return Position(symbol="BTC/USDT:USDT", side=side, qty=qty,
                    entry_price=entry_price, unrealized_pnl=10.0,
                    margin_used=100.0)


def _flat():
    return Position(symbol="BTC/USDT:USDT", side="flat", qty=0.0,
                    entry_price=0.0, unrealized_pnl=0.0, margin_used=0.0)


def _keyed_meta(active_bracket: str | None, **extra):
    """get_meta stand-in that answers PER KEY.

    Phase A returned the bracket JSON for every key. Phase B also reads
    `last_entry_bar_ts` on the armed path, so a key-blind fake would hand it a
    bracket JSON (unparseable as a timestamp) and silently exercise the
    fallback instead of the thing under test.
    """
    values = {"active_bracket": active_bracket, **extra}
    return MagicMock(side_effect=lambda key, default=None: values.get(key, default))


# A plausible entry fill: 10:05:03Z lands in the 10:00 bar, so the signal bar
# (the last CLOSED bar at entry time) is 09:45.
_ENTRY_FILL_TS = "2026-09-29T10:05:03.120000+00:00"


def _with_bracket(raw, root: str | None = "sig-1"):
    """Patch meta so active_bracket reads `raw` (a dict, a str, or None).

    Also pins the entry-fill fallback so the ARMED path (Phase B) has a
    dedup bar to seed from; the gate itself never reads it.
    """
    value = raw if isinstance(raw, str) or raw is None else json.dumps(raw)
    return patch.multiple(
        "bot.state",
        get_meta=_keyed_meta(value),
        latest_entry_fill_ts=MagicMock(return_value=_ENTRY_FILL_TS),
        # C5: the newest entry fill's root matches the stashed bracket's
        # signal_id ("sig-1"), so only the thing under test can refuse.
        latest_entry_coid_root=MagicMock(return_value=root),
    )


# ---------------------------------------------------------------------------
# 1. _can_adopt — the pure gate
# ---------------------------------------------------------------------------

class TestCanAdopt:

    def test_accepts_when_bracket_intact(self):
        bot, _ = _make_bot()
        with _with_bracket(_ARMED_BRACKET):
            ok, why = bot._can_adopt(_open())
        assert ok is True
        assert "intact" in why

    def test_accepts_when_tp_missing_but_reprotect_has_budget(self):
        """A missing TP is adoptable ONLY because the armed re-placer runs next
        tick. (A missing STOP is never adoptable — tests/test_boot_resume_armed_path.py.)"""
        bot, mc = _make_bot()
        sl_only = [r for r in mc.fetch_algo_orders.return_value[0]
                   if r["clientAlgoId"].endswith("-s")]
        mc.fetch_algo_orders.return_value = (sl_only, True)
        with _with_bracket({**_ARMED_BRACKET, "reprotect_count": 1}):
            ok, why = bot._can_adopt(_open())
        assert ok is True
        assert "1/3" in why

    def test_refuses_when_reprotect_cap_spent(self):
        """Nothing would restore the TP, so resuming leaves half a bracket."""
        bot, mc = _make_bot()
        sl_only = [r for r in mc.fetch_algo_orders.return_value[0]
                   if r["clientAlgoId"].endswith("-s")]
        mc.fetch_algo_orders.return_value = (sl_only, True)
        with _with_bracket({**_ARMED_BRACKET, "reprotect_count": 3}):
            ok, why = bot._can_adopt(_open())
        assert ok is False
        assert "cap spent" in why

    def test_refuses_when_reprotect_disabled(self):
        params = {**_MINIMAL_PARAMS, "reprotect": {"enabled": False}}
        bot, _ = _make_bot(params=params)
        with _with_bracket(_ARMED_BRACKET):
            ok, why = bot._can_adopt(_open())
        assert ok is False
        assert "not enabled" in why

    def test_refuses_when_reprotect_still_observe_only(self):
        params = {**_MINIMAL_PARAMS,
                  "reprotect": {"enabled": True, "observe_only": True}}
        bot, _ = _make_bot(params=params)
        with _with_bracket(_ARMED_BRACKET):
            ok, why = bot._can_adopt(_open())
        assert ok is False
        assert "observe-only" in why

    def test_refuses_when_active_bracket_empty(self):
        """REACHABLE: reprotect clears it to '' on the first flat tick, and a
        dropped exit leaves the same shape (donchian 2026-09-04)."""
        bot, _ = _make_bot()
        with _with_bracket(""):
            ok, why = bot._can_adopt(_open())
        assert ok is False
        assert "no active_bracket" in why

    def test_refuses_when_active_bracket_unparseable(self):
        bot, _ = _make_bot()
        with _with_bracket("{not json"):
            ok, why = bot._can_adopt(_open())
        assert ok is False
        assert "unparseable" in why

    def test_refuses_on_side_mismatch(self):
        bot, _ = _make_bot()
        with _with_bracket(_ARMED_BRACKET):
            ok, why = bot._can_adopt(_open(side="short"))
        assert ok is False
        assert "side mismatch" in why

    def test_refuses_on_entry_price_drift_beyond_2pct(self):
        bot, _ = _make_bot()
        with _with_bracket(_ARMED_BRACKET):
            ok, why = bot._can_adopt(_open(entry_price=70000.0))
        assert ok is False
        assert "drift" in why

    def test_accepts_entry_price_drift_inside_2pct(self):
        bot, _ = _make_bot()
        with _with_bracket(_ARMED_BRACKET):
            ok, _why = bot._can_adopt(_open(entry_price=65650.0))  # +1.0%
        assert ok is True

    def test_refuses_on_qty_mismatch(self):
        """A partially-closed position must not be resumed against a stale size.
        reprotect does not check this; adoption must."""
        bot, _ = _make_bot()
        with _with_bracket(_ARMED_BRACKET):
            ok, why = bot._can_adopt(_open(qty=0.005))
        assert ok is False
        assert "qty mismatch" in why

    def test_refuses_a_record_written_before_the_qty_field_existed(self):
        """D2 (Phase B): a pre-Phase-A stash has no qty, so size cannot be
        verified. Phase A SKIPPED the check here; God chose refuse → flatten.
        Asserted with a MATCHING live qty and an intact bracket, so the only
        thing that can refuse is the missing field."""
        bot, mc = _make_bot()
        legacy = {k: v for k, v in _ARMED_BRACKET.items() if k != "qty"}
        with _with_bracket(legacy):
            ok, why = bot._can_adopt(_open(qty=0.01))
        assert ok is False
        assert "no stashed qty" in why
        # Refused before any network round-trip.
        mc.ex.fetch_open_orders.assert_not_called()
        mc.fetch_algo_orders.assert_not_called()

    def test_refuses_a_zero_or_garbage_stashed_qty(self):
        bot, _ = _make_bot()
        for bad in (0, 0.0, None, "abc"):
            with _with_bracket({**_ARMED_BRACKET, "qty": bad}):
                ok, why = bot._can_adopt(_open())
            assert ok is False, bad
            assert "no stashed qty" in why

    def test_armed_boot_flattens_a_pre_phase_a_position(self):
        """D2 end to end: armed, intact bracket, but no qty → today's flatten."""
        params = {**_MINIMAL_PARAMS,
                  "boot_resume": {"enabled": True, "observe_only": False}}
        bot, mc = _make_bot(params=params)
        mc.fetch_equity_usdt.return_value = 1000.0
        mc.fetch_position.return_value = _open()
        legacy = {k: v for k, v in _ARMED_BRACKET.items() if k != "qty"}
        with _boot_patches(), _state_patches(bracket=legacy), _principal_patches():
            bot.boot()
        mc.close_position.assert_called_once_with(
            "BTC/USDT:USDT", client_order_id_root="sig-1", close_leg="bf")

    def test_refuses_an_sl_only_bracket(self):
        """donchian-v3 places no TP leg. bracket_state would be asked about a
        HALF bracket, which nothing has been exercised against. Refusing costs a
        resumed trade; accepting could cost an unprotected position."""
        bot, _ = _make_bot()
        with _with_bracket({**_ARMED_BRACKET, "place_tp": False}):
            ok, why = bot._can_adopt(_open())
        assert ok is False
        assert "half-bracket" in why

    def test_adopting_does_not_reset_the_reprotect_budget(self):
        """max_replaces_per_position is persisted in active_bracket precisely so
        a restart cannot hand the re-placer a fresh budget. The gate must READ
        the stash and never rewrite it."""
        bot, _ = _make_bot()
        with _with_bracket({**_ARMED_BRACKET, "reprotect_count": 2}), \
             patch("bot.state.set_meta") as set_meta:
            ok, _why = bot._can_adopt(_open())
        assert ok is True
        set_meta.assert_not_called()

    def test_refuses_when_algo_book_unreadable(self):
        """The July -4045 bug by another route. Unknown must never mean 'gone'."""
        bot, mc = _make_bot()
        mc.fetch_algo_orders.return_value = ([], False)
        with _with_bracket(_ARMED_BRACKET):
            ok, why = bot._can_adopt(_open())
        assert ok is False
        assert "unreadable" in why

    def test_refuses_when_plain_book_raises(self):
        bot, mc = _make_bot()
        mc.ex.fetch_open_orders.side_effect = RuntimeError("boom")
        with _with_bracket(_ARMED_BRACKET):
            ok, why = bot._can_adopt(_open())
        assert ok is False
        assert "plain order book unreadable" in why


# ---------------------------------------------------------------------------
# 2. _boot_resume_verdict — the rollout wrapper
# ---------------------------------------------------------------------------

class TestBootResumeVerdict:

    def test_observe_only_never_adopts_even_when_gate_says_yes(self):
        bot, _ = _make_bot()
        with _with_bracket(_ARMED_BRACKET):
            adopt, _why = bot._boot_resume_verdict(_open())
        assert adopt is False

    def test_armed_adopts_when_gate_says_yes(self):
        params = {**_MINIMAL_PARAMS,
                  "boot_resume": {"enabled": True, "observe_only": False}}
        bot, mc = _make_bot(params=params)
        mc.fetch_position.return_value = _open()     # C6 re-read: unchanged
        with _with_bracket(_ARMED_BRACKET):
            adopt, _why = bot._boot_resume_verdict(_open())
        assert adopt is True

    def test_disabled_block_does_not_consult_the_gate(self):
        params = {**_MINIMAL_PARAMS, "boot_resume": {"enabled": False}}
        bot, _ = _make_bot(params=params)
        with patch.object(bot, "_can_adopt") as gate:
            adopt, why = bot._boot_resume_verdict(_open())
        assert adopt is False
        assert "disabled" in why
        gate.assert_not_called()

    def test_absent_block_is_fail_closed(self):
        params = {k: v for k, v in _MINIMAL_PARAMS.items() if k != "boot_resume"}
        bot, _ = _make_bot(params=params)
        adopt, _why = bot._boot_resume_verdict(_open())
        assert adopt is False

    def test_a_raising_gate_does_not_propagate(self):
        """A gate that throws must not take the position with it."""
        params = {**_MINIMAL_PARAMS,
                  "boot_resume": {"enabled": True, "observe_only": False}}
        bot, _ = _make_bot(params=params)
        with patch.object(bot, "_can_adopt", side_effect=RuntimeError("boom")):
            adopt, why = bot._boot_resume_verdict(_open())
        assert adopt is False
        assert "raised" in why


# ---------------------------------------------------------------------------
# 3. boot() integration — Phase A must still flatten
# ---------------------------------------------------------------------------

def _state_patches(bracket=_ARMED_BRACKET, enqueue=None):
    """Patch every state side-effect boot() touches.

    `enqueue` is taken as a parameter rather than read back off the context
    manager: patch.multiple only returns mocks for kwargs set to DEFAULT, so
    `with _state_patches() as st` yields an EMPTY dict here.
    """
    value = bracket if isinstance(bracket, str) else json.dumps(bracket)
    return patch.multiple(
        "bot.state",
        init_db=MagicMock(),
        get_float=MagicMock(return_value=0.0),
        set_float=MagicMock(),
        set_meta=MagicMock(),
        get_meta=_keyed_meta(value),
        enqueue_bot_event=enqueue or MagicMock(),
        record_event=MagicMock(),
        latest_entry_coid_root=MagicMock(return_value="sig-1"),
        latest_entry_fill_ts=MagicMock(return_value=_ENTRY_FILL_TS),
    )


def _boot_patches():
    return patch.multiple(
        "bot",
        check_symbol=MagicMock(),
        check_leverage=MagicMock(),
        send_alert=MagicMock(),
    )


def _principal_patches():
    return patch.multiple(
        "bot.principal",
        initialize=MagicMock(return_value=None),
        get_principal=MagicMock(return_value=None),
    )


class TestBootIntegration:

    def test_phase_a_still_flattens_an_adoptable_position(self):
        """The whole point of observe-only: zero behaviour change."""
        bot, mc = _make_bot()
        mc.fetch_equity_usdt.return_value = 1000.0
        mc.fetch_position.return_value = _open()

        with _boot_patches(), _state_patches(), _principal_patches():
            bot.boot()

        mc.close_position.assert_called_once()

    def test_armed_adopts_and_does_not_close(self):
        params = {**_MINIMAL_PARAMS,
                  "boot_resume": {"enabled": True, "observe_only": False}}
        bot, mc = _make_bot(params=params)
        mc.fetch_equity_usdt.return_value = 1000.0
        mc.fetch_position.return_value = _open()

        ev = MagicMock()
        with _boot_patches(), _state_patches(enqueue=ev), _principal_patches():
            bot.boot()

        mc.close_position.assert_not_called()
        # Not enough that it skipped the close — the adopt block must actually
        # have run. Without this, a gate that returns True while the
        # event-emitting block throws would still pass.
        kinds = [c.args[0] for c in ev.call_args_list if c.args]
        assert "boot_adopt" in kinds

    def test_armed_still_flattens_when_the_gate_refuses(self):
        params = {**_MINIMAL_PARAMS,
                  "boot_resume": {"enabled": True, "observe_only": False}}
        bot, mc = _make_bot(params=params)
        mc.fetch_equity_usdt.return_value = 1000.0
        mc.fetch_position.return_value = _open()

        with _boot_patches(), _state_patches(bracket=""), _principal_patches():
            bot.boot()

        mc.close_position.assert_called_once()

    def test_flat_boot_is_untouched_by_this_feature(self):
        bot, mc = _make_bot()
        mc.fetch_equity_usdt.return_value = 1000.0
        mc.fetch_position.return_value = _flat()

        with _boot_patches(), _state_patches(), _principal_patches():
            bot.boot()

        mc.close_position.assert_not_called()
        mc.cancel_open_orders.assert_called_once_with(
            "BTC/USDT:USDT", coid_prefix="snap-v1-")
