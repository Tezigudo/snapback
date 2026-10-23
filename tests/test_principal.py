"""Tests for the principal-anchored kill switch (exchange/principal.py + bot).

Kill-switch anchor = NET DEPOSITED PRINCIPAL (God's rule), built from Binance
income TRANSFER/DEPOSIT/WITHDRAW events keyed idempotently by tranId — NOT a
balance snapshot and NOT a high-water mark.

Required coverage:
  - a deposit RAISES P; a withdrawal LOWERS P;
  - a large simulated bot profit does NOT raise P;
  - restart reconciliation is idempotent (no double-count);
  - a donchian-style stale/wrong-account read no longer trips the kill switch;
  - the kill switch, when it does fire, touches ONLY this leg's per-leg HALT.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

# Two fixed millisecond timestamps (ascending) for deterministic income rows.
T0 = 1_779_512_003_000
T1 = 1_782_559_672_000


def _income(tran_id, income_type, income, asset="USDT", time_ms=T0):
    # Binance returns `income` as a string, e.g. "50.50000000".
    return {"tranId": tran_id, "incomeType": income_type,
            "income": f"{income:.8f}", "asset": asset,
            "time": time_ms, "symbol": ""}


class _FakeClient:
    """Stands in for BinanceClient.fetch_income — returns rows at/after start."""
    def __init__(self, rows):
        self._rows = list(rows)

    def fetch_income(self, start_ms, **_kw):
        return [r for r in self._rows if int(r["time"]) >= int(start_ms)]


class _RecordingClient:
    """Like _FakeClient but records the incomeType filter of each fetch_income
    call AND honours it server-side — so a test can prove the backfill queries
    each principal type specifically (must-fix 3: no TRANSFER lost to the page
    cap behind a flood of REALIZED_PNL rows)."""
    def __init__(self, rows):
        self._rows = list(rows)
        self.income_types_requested: list[str | None] = []

    def fetch_income(self, start_ms, income_type=None, **_kw):
        self.income_types_requested.append(income_type)
        return [r for r in self._rows
                if int(r["time"]) >= int(start_ms)
                and (income_type is None or r["incomeType"] == income_type)]


def _with_temp_db(test_fn):
    def wrapped():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "state.db"
            with patch("exchange.state.DB_PATH", tmp):
                from exchange import state
                state.init_db()
                test_fn(Path(td))
    return wrapped


# --------------------------------------------------------------------------
# P construction from income
# --------------------------------------------------------------------------

@_with_temp_db
def test_deposit_raises_principal(_td: Path) -> None:
    from exchange import principal
    fc = _FakeClient([_income(1, "TRANSFER", 50.50, time_ms=T0)])
    principal.initialize(fc, {"deploy": {}})
    assert principal.get_principal() == 50.50
    # A monthly DCA deposit arrives → P rises.
    fc._rows.append(_income(2, "TRANSFER", 60.03, time_ms=T1))
    principal.reconcile_recent(fc)
    assert round(principal.get_principal(), 2) == 110.53


@_with_temp_db
def test_withdrawal_lowers_principal(_td: Path) -> None:
    from exchange import principal
    fc = _FakeClient([_income(1, "TRANSFER", 100.0, time_ms=T0)])
    principal.initialize(fc, {"deploy": {}})
    assert principal.get_principal() == 100.0
    # Withdrawal out of futures — Binance signs the income negative.
    fc._rows.append(_income(2, "TRANSFER", -30.0, time_ms=T1))
    principal.reconcile_recent(fc)
    assert principal.get_principal() == 70.0


@_with_temp_db
def test_bot_profit_does_not_raise_principal(_td: Path) -> None:
    from exchange import principal, state
    fc = _FakeClient([
        _income(1, "TRANSFER", 100.0, time_ms=T0),
        _income(2, "REALIZED_PNL", 500.0, time_ms=T1),   # huge win
        _income(3, "FUNDING_FEE", -2.0, time_ms=T1),
        _income(4, "COMMISSION", -1.0, time_ms=T1),
    ])
    principal.initialize(fc, {"deploy": {}})
    # Only the TRANSFER moves principal; trading income never does.
    assert principal.get_principal() == 100.0
    assert state.principal_ledger_count() == 1


@_with_temp_db
def test_restart_reconciliation_is_idempotent(_td: Path) -> None:
    from exchange import principal
    rows = [_income(1, "TRANSFER", 50.50, time_ms=T0),
            _income(2, "TRANSFER", 60.03, time_ms=T1)]
    fc = _FakeClient(rows)
    principal.initialize(fc, {"deploy": {}})
    p1 = principal.get_principal()
    # Simulate a restart: reconcile the SAME (overlapping) window again.
    res = principal.reconcile_recent(fc)
    assert res["applied"] == 0, "no new rows should be applied on a restart"
    assert principal.get_principal() == p1
    # A full re-backfill of the identical rows must also no-op (tranId PK).
    res2 = principal.reconcile_from_income(rows)
    assert res2["applied"] == 0
    assert principal.get_principal() == p1


@_with_temp_db
def test_manual_seed_then_accumulates(_td: Path) -> None:
    from exchange import principal, state
    principal.initialize(_FakeClient([]), {"deploy": {"manual_principal_usdt": 80.50}})
    assert principal.get_principal() == 80.50
    assert state.get_meta("principal_source") == "manual"
    # A later transfer accumulates on top of the manual base.
    principal.reconcile_from_income([_income(9, "TRANSFER", 20.0)])
    assert principal.get_principal() == 100.50


@_with_temp_db
def test_non_usdt_transfer_excluded_from_principal(_td: Path) -> None:
    from exchange import principal, state
    fc = _FakeClient([
        _income(1, "TRANSFER", 100.0, asset="USDT", time_ms=T0),
        _income(2, "TRANSFER", 5.0, asset="BNB", time_ms=T1),  # BNB moved in
    ])
    res = principal.initialize(fc, {"deploy": {}})
    # BNB principal transfer is stored but NOT summed into the USDT P.
    assert res == 100.0
    assert state.principal_ledger_count() == 2
    assert state.principal_ledger_non_usdt_count() == 1


@_with_temp_db
def test_initialize_filters_income_by_type(_td: Path) -> None:
    from exchange import principal, state
    # A busy account: one real TRANSFER buried under a flood of trading income.
    # An UNFILTERED whole-ledger fetch is what risks truncating (and silently
    # dropping) the TRANSFER; the per-type server-side filter cannot lose it.
    rows = [_income(1, "TRANSFER", 50.50, time_ms=T0),
            _income(2, "REALIZED_PNL", 9.0, time_ms=T1),
            _income(3, "COMMISSION", -0.5, time_ms=T1),
            _income(4, "FUNDING_FEE", -0.2, time_ms=T1)]
    rc = _RecordingClient(rows)
    P = principal.initialize(rc, {"deploy": {}})
    assert P == 50.50
    assert state.principal_ledger_count() == 1  # only the TRANSFER lands in P
    # The backfill issued a SERVER-SIDE incomeType filter (TRANSFER — the only
    # valid futures principal-income type), never an unfiltered whole-ledger
    # fetch (must-fix 3) and never the invalid DEPOSIT/WITHDRAW (-1130).
    assert set(rc.income_types_requested) == {"TRANSFER"}
    assert None not in rc.income_types_requested


# --------------------------------------------------------------------------
# Kill-switch predicate (fail-safe) + the donchian wrong-account regression
# --------------------------------------------------------------------------

def test_breached_fail_safe_on_unknown_anchor() -> None:
    from exchange import principal
    # None (uninitialised) or non-positive P must NEVER trip — this is the
    # fail-safe that stops a bad/absent anchor from causing a false kill.
    assert principal.breached(1.0, None, 0.6) is False
    assert principal.breached(1.0, 0.0, 0.6) is False
    assert principal.breached(1.0, -5.0, 0.6) is False


@_with_temp_db
def test_donchian_wrong_account_read_no_longer_trips(_td: Path) -> None:
    from exchange import principal
    frac = 0.6   # synthetic kill fraction
    # Synthetic amounts. The leg's TRUE deposited principal (its sub-account transfer).
    fc = _FakeClient([_income(1, "TRANSFER", 50.0, time_ms=T0)])
    principal.initialize(fc, {"deploy": {}})
    P = principal.get_principal()
    assert P == 50.0

    # OLD bug: the anchor was a wrong-account (larger) snapshot, so at the
    # real equity the OLD kill switch WOULD have fired.
    old_wrong_anchor = 120.0
    assert old_wrong_anchor * frac > 50.0  # old code would trip (false positive)

    # NEW: anchor is deposited principal. Equity == principal → no trip.
    assert principal.breached(50.0, P, frac) is False
    assert P * frac <= 50.0  # real floor is 30.00; equity sits above it
    # A transient WRONG-HIGH equity read can neither inflate P nor trip.
    assert principal.breached(120.0, P, frac) is False
    # Only a genuine loss of real principal trips: equity below 30.00.
    assert principal.breached(28.0, P, frac) is True


# --------------------------------------------------------------------------
# Bot-level kill switch: touches ONLY this leg's HALT (Part A + Part C)
# --------------------------------------------------------------------------

def _make_kill_bot(instance, halt_path, kill_fraction=0.6):
    import bot as bot_mod

    class _StubLog:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass

    b = object.__new__(bot_mod.Bot)
    b.instance = instance
    b.halt_path = halt_path
    b.kill_fraction = kill_fraction
    b.log = _StubLog()
    return b, bot_mod


@_with_temp_db
def test_kill_switch_touches_only_this_leg_halt(td: Path) -> None:
    from exchange import env as env_mod
    from exchange import principal, state
    principal.initialize(_FakeClient([_income(1, "TRANSFER", 50.50, time_ms=T0)]),
                         {"deploy": {}})
    P = principal.get_principal()

    # Point env.REPO_ROOT at the temp dir so global_halt_path()/leg_halt_path()
    # resolve UNDER td/data. The REAL global HALT is REPO_ROOT/data/HALT (NOT
    # td/HALT), so asserting against the env-derived paths is what makes the
    # 07-01-cascade guard actually bite: a regression touching the global HALT
    # would write td/data/HALT and fail `not halt_global.exists()`.
    with patch.object(env_mod, "REPO_ROOT", td):
        (td / "data").mkdir(parents=True, exist_ok=True)
        halt_self = env_mod.leg_halt_path("cnh_short")   # td/data/HALT_cnh_short
        halt_global = env_mod.global_halt_path()          # td/data/HALT (real global)
        halt_sibling = env_mod.leg_halt_path("v1")        # td/data/HALT_v1

        b, bot_mod = _make_kill_bot("cnh_short", halt_self)
        with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
             patch.object(bot_mod.consolidate_push, "drain", lambda *a, **k: {}):
            # Equity below the principal floor (32.57) → must trip.
            tripped = b._check_kill_switch(P * 0.5)

    assert tripped is True
    assert halt_self.exists(), "kill switch must write this leg's per-leg HALT"
    assert not halt_global.exists(), \
        "kill switch must NOT write the GLOBAL data/HALT (07-01 cascade guard)"
    assert not halt_sibling.exists(), "kill switch must NOT touch a sibling's HALT"
    # The emitted event carries the principal anchor, not a balance snapshot.
    import json
    row = state.outbox_pending(10)
    kinds = [r[1] for r in row]
    assert "kill_switch" in kinds
    payload = json.loads(next(r[2] for r in row if r[1] == "kill_switch"))
    assert payload["payload"]["principal_anchor"] == P


@_with_temp_db
def test_kill_switch_does_not_trip_above_floor(td: Path) -> None:
    from exchange import principal
    principal.initialize(_FakeClient([_income(1, "TRANSFER", 50.50, time_ms=T0)]),
                         {"deploy": {}})
    P = principal.get_principal()
    halt_self = td / "HALT_cnh_short"
    b, bot_mod = _make_kill_bot("cnh_short", halt_self)
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod.consolidate_push, "drain", lambda *a, **k: {}):
        # Equity at principal → above the 0.6 floor → no trip.
        assert b._check_kill_switch(P) is False
    assert not halt_self.exists()


@_with_temp_db
def test_kill_switch_disabled_when_principal_uninitialised(td: Path) -> None:
    from exchange import principal
    # No initialize() call → P is None → kill switch must be fail-safe (off),
    # even at an absurdly low equity read.
    assert principal.get_principal() is None
    halt_self = td / "HALT_cnh_short"
    b, bot_mod = _make_kill_bot("cnh_short", halt_self)
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None), \
         patch.object(bot_mod.consolidate_push, "drain", lambda *a, **k: {}):
        assert b._check_kill_switch(0.01) is False
    assert not halt_self.exists()
