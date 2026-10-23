"""tools/boot_resume_probe.py — read-only guard (T13) and probe/boot parity (T8).

The probe is run on the droplet against the LIVE exchange by hand, so its
read-only property has to be structural and tested, not a promise.
"""
from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from exchange import state
from exchange.binance_client import BinanceClient
from exchange.constraints import fallbacks_for_symbol, merge_with_live
from tests.test_boot_resume import _ARMED_BRACKET, _MINIMAL_PARAMS, _make_bot, _open
from tools import boot_resume_probe as probe
from tools.boot_resume_probe import (
    ReadOnlyClient,
    ReadOnlyExchange,
    ReadOnlyViolation,
    evaluate,
    open_state_ro,
)

SYM = "BTC/USDT:USDT"
_MARKET = {"id": "BTCUSDT", "limits": {"amount": {"min": 0.001}, "cost": {"min": 50}},
           "precision": {"amount": 0.001, "price": 0.1}}
_ALGO_INTACT = [
    {"algoId": "1", "clientAlgoId": "snap-v1-sig-1-s", "side": "SELL",
     "triggerPrice": "64500.0", "strategyType": None, "reduceOnly": True,
     "algoStatus": "NEW"},
    {"algoId": "2", "clientAlgoId": "snap-v1-sig-1-t", "side": "SELL",
     "triggerPrice": "67000.0", "strategyType": None, "reduceOnly": True,
     "algoStatus": "NEW"},
]


@pytest.fixture(autouse=True)
def _no_c9_backoff():
    """C9 retries an unreadable book with a real 5 s back-off; never wait in tests."""
    with patch("bot._boot_book_sleep") as sl:
        yield sl



def _inner_exchange(side="long", qty=0.01, entry=65000.0, algo=None,
                    algo_raises=False, plain=None) -> MagicMock:
    ex = MagicMock()
    ex.fetch_positions.return_value = (
        [] if side == "flat" else
        [{"symbol": SYM, "contracts": qty, "side": side, "entryPrice": entry,
          "unrealizedPnl": 0.0, "initialMargin": 100.0}])
    ex.fetch_open_orders.return_value = list(plain or [])
    if algo_raises:
        ex.fapiPrivateGetOpenAlgoOrders.side_effect = RuntimeError("503")
    else:
        ex.fapiPrivateGetOpenAlgoOrders.return_value = (
            list(_ALGO_INTACT) if algo is None else algo)
    ex.market.return_value = _MARKET
    ex.parse_timeframe.return_value = 900
    return ex


def _ro_client(inner: MagicMock) -> ReadOnlyClient:
    return ReadOnlyClient(BinanceClient(ex=ReadOnlyExchange(inner), env="mainnet",
                                        coid_prefix="snap-v1-"))


def _called_names(inner: MagicMock) -> set[str]:
    return {c[0].split(".")[0] for c in inner.mock_calls if c[0]}


# ---------------------------------------------------------------------------
# T13 — the read-only guard
# ---------------------------------------------------------------------------

_MUTATORS = [
    ("close_position", (SYM,), {"client_order_id_root": "r", "close_leg": "c"}),
    ("cancel_open_orders", (SYM,), {"coid_prefix": "snap-v1-"}),
    ("set_leverage", (SYM, 5), {}),
    ("place_tagged_stop", (SYM, "long", 0.01, 64000.0, "r", "s"), {}),
    ("cancel_algo_by_coid", (SYM, "snap-v1-r-s"), {}),
    ("market_order_with_bracket", (SYM, "long", 0.01, 64000.0, 67000.0), {}),
    ("limit_order_with_bracket", (SYM, "long", 0.01, 65000.0, 500.0, 2000.0), {}),
]


class TestReadOnlyGuard:

    @pytest.mark.parametrize("name,args,kwargs", _MUTATORS, ids=[m[0] for m in _MUTATORS])
    def test_every_mutator_raises_and_reaches_no_write(self, name, args, kwargs):
        """Called on the INNER BinanceClient (bypassing ReadOnlyClient), so this
        proves the exchange-level whitelist alone stops the write."""
        inner = _inner_exchange()
        client = BinanceClient(ex=ReadOnlyExchange(inner), env="mainnet",
                               coid_prefix="snap-v1-")
        with pytest.raises(ReadOnlyViolation):
            getattr(client, name)(*args, **kwargs)
        assert _called_names(inner) <= ReadOnlyExchange.ALLOWED, _called_names(inner)
        assert client.ex.violations, "the guard never fired"

    def test_violation_escapes_a_broad_except_exception(self):
        """set_leverage wraps its call in `except Exception`; the violation
        must still surface rather than become a silent no-op."""
        assert not issubclass(ReadOnlyViolation, Exception)
        assert not issubclass(ReadOnlyViolation, AttributeError)

    def test_getattr_default_does_not_swallow_it(self):
        ro = ReadOnlyExchange(MagicMock())
        with pytest.raises(ReadOnlyViolation):
            getattr(ro, "create_order", None)

    @pytest.mark.parametrize("name", ["close_position", "cancel_open_orders",
                                      "set_leverage", "market_order_with_bracket",
                                      "limit_order_with_bracket", "place_tagged_stop",
                                      "cancel_algo_by_coid", "fetch_equity_usdt"])
    def test_client_wrapper_only_exposes_reads(self, name):
        ro = _ro_client(_inner_exchange())
        with pytest.raises(ReadOnlyViolation):
            getattr(ro, name)

    def test_cannot_rebind_the_exchange(self):
        ro = _ro_client(_inner_exchange())
        with pytest.raises(ReadOnlyViolation):
            ro.ex = MagicMock()
        with pytest.raises(ReadOnlyViolation):
            ro.ex.create_order = MagicMock()

    def test_client_wrapper_refuses_an_unguarded_exchange(self):
        with pytest.raises(ReadOnlyViolation):
            ReadOnlyClient(BinanceClient(ex=MagicMock(), env="mainnet"))

    def test_evaluate_refuses_a_plain_client(self):
        with pytest.raises(ReadOnlyViolation):
            evaluate(dict(_MINIMAL_PARAMS), MagicMock(), MagicMock())

    def test_state_db_is_opened_read_only(self, isolated_state_db):
        conn = open_state_ro(isolated_state_db)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO meta(key, value) VALUES ('x', 'y')")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM meta")
        conn.close()

    def test_a_missing_state_db_is_an_error_not_created(self, tmp_path):
        missing = tmp_path / "nope.db"
        with pytest.raises(sqlite3.OperationalError):
            open_state_ro(missing)
        assert not missing.exists()


# ---------------------------------------------------------------------------
# End to end against a fixture DB; never touches Bot, main() or logging
# ---------------------------------------------------------------------------

def _seed(bracket=_ARMED_BRACKET):
    state.record_fill(side="long", qty=0.01, price=65000.0, reason="entry",
                      client_order_id_root="sig-1")
    if bracket is not None:
        state.set_meta("active_bracket",
                       bracket if isinstance(bracket, str) else json.dumps(bracket))
    state.set_meta("last_entry_bar_ts", "2026-09-29T09:45:00")


def _db_snapshot(conn: sqlite3.Connection) -> tuple:
    return (conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall(),
            conn.execute("SELECT * FROM fills ORDER BY id").fetchall(),
            conn.execute("SELECT COUNT(*) FROM events").fetchone(),
            conn.execute("SELECT COUNT(*) FROM outbox").fetchone())


def _forbid_bot():
    return patch.multiple("bot",
                          main=MagicMock(side_effect=AssertionError("main() called")),
                          _setup_logging=MagicMock(
                              side_effect=AssertionError("_setup_logging called")))


class TestProbeEndToEnd:

    def test_adopt_verdict_with_no_bot_no_main_no_logging_no_writes(self, isolated_state_db):
        _seed()
        params = {**_MINIMAL_PARAMS,
                  "boot_resume": {"enabled": True, "observe_only": True}}
        conn = open_state_ro(isolated_state_db)
        before = _db_snapshot(conn)
        inner = _inner_exchange()
        with _forbid_bot(), patch("bot.Bot.__init__",
                                  side_effect=AssertionError("Bot constructed")):
            res = evaluate(params, _ro_client(inner), conn)
            out = probe.render("v1", res)
        assert res.verdict == "ADOPT"
        assert res.gate == (True, "bracket intact (SL=present TP=present)")
        assert "gate: ADOPT — bracket intact" in out
        assert "observe_only" in out            # a restart NOW would still flatten
        assert _called_names(inner) <= ReadOnlyExchange.ALLOWED
        assert _db_snapshot(conn) == before
        conn.close()

    def test_flat_position(self, isolated_state_db):
        conn = open_state_ro(isolated_state_db)
        res = evaluate(dict(_MINIMAL_PARAMS), _ro_client(_inner_exchange(side="flat")), conn)
        assert res.verdict == "FLAT"

    def test_disabled_leg_reports_flatten(self, isolated_state_db):
        _seed()
        params = {k: v for k, v in _MINIMAL_PARAMS.items() if k != "boot_resume"}
        conn = open_state_ro(isolated_state_db)
        res = evaluate(params, _ro_client(_inner_exchange()), conn)
        assert res.verdict == "FLATTEN"

    def test_pre_phase_a_stash_is_refused(self, isolated_state_db):
        _seed(bracket={k: v for k, v in _ARMED_BRACKET.items() if k != "qty"})
        conn = open_state_ro(isolated_state_db)
        res = evaluate(dict(_MINIMAL_PARAMS), _ro_client(_inner_exchange()), conn)
        assert res.verdict == "REFUSE"
        assert "no stashed qty" in res.reason

    def test_tripped_loop_guard_is_refused(self, isolated_state_db):
        _seed()
        state.set_meta("boot_adopt_log", json.dumps(
            {"signal_id": "sig-1", "count": 3, "first_ts": 1000.0}))
        conn = open_state_ro(isolated_state_db)
        res = evaluate(dict(_MINIMAL_PARAMS), _ro_client(_inner_exchange()), conn,
                       now_s=1060.0)
        assert res.verdict == "REFUSE"
        assert "adopt-loop guard" in res.reason


# ---------------------------------------------------------------------------
# T8 — probe and boot give byte-identical (verdict, reason)
# ---------------------------------------------------------------------------

_FOREIGN_PLAIN_STOP = {"clientOrderId": "manual-123", "type": "stop_market",
                       "side": "sell",
                       "info": {"reduceOnly": "true", "type": "STOP_MARKET"}}

_PARITY_CASES = {
    "intact": dict(bracket=_ARMED_BRACKET),
    "missing_with_budget": dict(bracket={**_ARMED_BRACKET, "reprotect_count": 1}, algo=[]),
    "missing_cap_spent": dict(bracket={**_ARMED_BRACKET, "reprotect_count": 3}, algo=[]),
    "tp_missing_with_budget": dict(bracket={**_ARMED_BRACKET, "reprotect_count": 1},
                                   algo=[_ALGO_INTACT[0]]),
    "tp_missing_cap_spent": dict(bracket={**_ARMED_BRACKET, "reprotect_count": 3},
                                 algo=[_ALGO_INTACT[0]]),
    "sl_missing_tp_present": dict(bracket=_ARMED_BRACKET, algo=[_ALGO_INTACT[1]]),
    "no_record": dict(bracket=None),
    "unparseable": dict(bracket="{not json"),
    "side_mismatch": dict(bracket={**_ARMED_BRACKET, "side": "short"}),
    "entry_drift": dict(bracket={**_ARMED_BRACKET, "entry_price": 70000.0}),
    "qty_mismatch": dict(bracket=_ARMED_BRACKET, qty=0.005),
    "no_qty": dict(bracket={k: v for k, v in _ARMED_BRACKET.items() if k != "qty"}),
    "sl_only": dict(bracket={**_ARMED_BRACKET, "place_tp": False}),
    "algo_unreadable": dict(bracket=_ARMED_BRACKET, algo_raises=True),
    "foreign_plain_stop": dict(bracket=_ARMED_BRACKET, algo=[_ALGO_INTACT[1]],
                               plain=[_FOREIGN_PLAIN_STOP]),
    "our_plain_stop": dict(bracket=_ARMED_BRACKET, algo=[_ALGO_INTACT[1]],
                           plain=[{**_FOREIGN_PLAIN_STOP,
                                   "clientOrderId": "snap-v1-sig-1-s"}]),
    "breakeven_mutated": dict(bracket={**_ARMED_BRACKET, "be_moved": True,
                                       "be_price": 65065.0, "be_ext": "sb"}),
}


@pytest.mark.parametrize("case", list(_PARITY_CASES), ids=list(_PARITY_CASES))
def test_probe_gate_matches_bot_can_adopt(case, isolated_state_db):
    c = _PARITY_CASES[case]
    _seed(bracket=c["bracket"])
    qty = c.get("qty", 0.01)
    inner = _inner_exchange(qty=qty, algo=c.get("algo"),
                            algo_raises=c.get("algo_raises", False),
                            plain=c.get("plain"))

    conn = open_state_ro(isolated_state_db)
    res = evaluate(dict(_MINIMAL_PARAMS), _ro_client(inner), conn)
    conn.close()

    bot, mc = _make_bot()
    bot.constraints = merge_with_live(fallbacks_for_symbol(SYM), _MARKET)
    mc.ex.fetch_open_orders.return_value = list(c.get("plain") or [])
    if c.get("algo_raises"):
        mc.fetch_algo_orders.return_value = ([], False)
    else:
        mc.fetch_algo_orders.return_value = (
            list(_ALGO_INTACT) if c.get("algo") is None else c["algo"], True)
    assert res.gate == bot._can_adopt(_open(qty=qty))


# ---------------------------------------------------------------------------
# CLI: usage and the per-instance env rule
# ---------------------------------------------------------------------------

class TestCli:

    @pytest.mark.parametrize("argv", [[], ["snapback-btc"], ["cnh_short"], ["v1", "x"]])
    def test_usage_errors_exit_2(self, argv, capsys):
        assert probe.main(argv) == 2

    def test_sub_account_leg_without_env_exits_2_before_any_client(self):
        with patch("exchange.env.load_env_for_instance", return_value=None), \
             patch.object(probe, "build_readonly_client",
                          side_effect=AssertionError("client built")):
            assert probe.main(["donchian"]) == 2

    @pytest.mark.parametrize("inst", ["donchian", "sol_supertrend"])
    def test_real_env_error_path_exits_2_before_any_client(self, inst, tmp_path, capsys):
        """No mock of the loader: point REPO_ROOT at an empty dir so the REAL
        load_env_for_instance raises EnvError for a sub-account leg."""
        from exchange.env import EnvError, load_env_for_instance
        with patch("exchange.env.REPO_ROOT", tmp_path):
            with pytest.raises(EnvError):
                load_env_for_instance(inst)
            with patch.object(probe, "build_readonly_client",
                              side_effect=AssertionError("client built")):
                assert probe.main([inst]) == 2
        assert "FATAL" in capsys.readouterr().err

    def test_help_says_run_as_root(self, capsys):
        assert probe.main(["--help"]) == 0
        assert "AS ROOT" in capsys.readouterr().out


class TestProbeGuardsAndOutput:

    def test_state_module_cannot_open_the_db_during_evaluate(self, isolated_state_db):
        """Anything reaching exchange.state while evaluate runs is refused —
        so no path inside the probe can open the DB read-write."""
        _seed()
        conn = open_state_ro(isolated_state_db)
        inner = _inner_exchange()
        inner.fetch_positions.side_effect = lambda *_a, **_k: state.set_meta("x", "y")
        with pytest.raises(ReadOnlyViolation):
            evaluate(dict(_MINIMAL_PARAMS), _ro_client(inner), conn)
        conn.close()
        # Restored afterwards, and the write never happened.
        assert state.get_meta("x") is None

    def test_client_log_lines_carry_exception_type_only(self, isolated_state_db, caplog):
        _seed()
        conn = open_state_ro(isolated_state_db)
        inner = _inner_exchange()
        inner.fapiPrivateGetOpenAlgoOrders.side_effect = RuntimeError(
            "GET /fapi/v1/openAlgoOrders?signature=LEAKME")
        with caplog.at_level("WARNING"):
            res = evaluate(dict(_MINIMAL_PARAMS), _ro_client(inner), conn)
        conn.close()
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "LEAKME" not in text
        assert "RuntimeError" in text
        assert res.verdict == "REFUSE"

    def test_output_has_probe_time_and_position_age(self, isolated_state_db):
        _seed()
        with open_state_ro(isolated_state_db) as conn:
            res = evaluate(dict(_MINIMAL_PARAMS), _ro_client(_inner_exchange()), conn)
        out = probe.render("v1", res)
        assert "probed at 20" in out and " UTC" in out
        assert "position_age" in out and "(from latest entry fill)" in out

    def test_headline_is_labelled_hypothetical_and_restart_line_follows(self, isolated_state_db):
        _seed()
        with open_state_ro(isolated_state_db) as conn:
            res = evaluate(dict(_MINIMAL_PARAMS), _ro_client(_inner_exchange()), conn)
        lines = probe.render("v1", res).splitlines()
        assert lines[1].startswith("GATE VERDICT (if armed): ADOPT")
        assert lines[2].startswith("a restart right now would:")

    def test_c5_mismatch_refuses(self, isolated_state_db):
        """C5 is enforced by the gate itself, so the probe and the boot refuse
        a stale active_bracket with the SAME reason string."""
        _seed(bracket={**_ARMED_BRACKET, "signal_id": "sig-STALE"})
        with open_state_ro(isolated_state_db) as conn:
            res = evaluate(dict(_MINIMAL_PARAMS), _ro_client(_inner_exchange()), conn)
        assert res.verdict == "REFUSE"
        assert res.reason.startswith("identity mismatch: active_bracket.signal_id sig-STALE")
        assert res.inputs["active_bracket_sid_matches_latest_root"] is False

    def test_roots_match_on_adopt(self, isolated_state_db):
        _seed()
        with open_state_ro(isolated_state_db) as conn:
            res = evaluate(dict(_MINIMAL_PARAMS), _ro_client(_inner_exchange()), conn)
        assert res.verdict == "ADOPT"
        assert res.inputs["active_bracket_sid_matches_latest_root"] is True

    def test_restart_line_reports_the_arming_interlock(self, isolated_state_db):
        _seed()
        params = {**_MINIMAL_PARAMS,
                  "boot_resume": {"enabled": True, "observe_only": False}}
        with open_state_ro(isolated_state_db) as conn:
            res = evaluate(params, _ro_client(_inner_exchange()), conn)
        assert "ARMING INTERLOCK" in probe.render("v1", res)

    def test_malformed_boot_resume_reads_as_disabled(self, isolated_state_db):
        _seed()
        with open_state_ro(isolated_state_db) as conn:
            res = evaluate({**_MINIMAL_PARAMS, "boot_resume": ["x"]},
                           _ro_client(_inner_exchange()), conn)
        assert res.verdict == "FLATTEN"


def test_probe_bracket_line_is_the_gates_view_not_the_raw_book(isolated_state_db):
    """Blocker 4 in the counted evidence: a manual plain stop must not make the
    probe print SL=present next to a gate REFUSE for a missing stop."""
    _seed()
    inner = _inner_exchange(algo=[_ALGO_INTACT[1]], plain=[_FOREIGN_PLAIN_STOP])
    with open_state_ro(isolated_state_db) as conn:
        res = evaluate(dict(_MINIMAL_PARAMS), _ro_client(inner), conn)
    assert res.verdict == "REFUSE" and res.reason.startswith("stop leg missing")
    assert res.inputs["bracket"].startswith("SL=MISSING")
    assert res.inputs["bracket_incl_foreign"].startswith("SL=present")
