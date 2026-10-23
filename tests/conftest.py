"""Suite-wide isolation for the sqlite state layer.

WHY THIS EXISTS
---------------
`exchange/state.py` resolves `DB_PATH` at import time to `data/state.db` (or
`$SNAPBACK_STATE_DB`). Nothing in the suite ever set that, so any test that
reached a `state.*` call — or `bot.py`'s direct `sqlite3.connect(state.DB_PATH)`
— operated on a REAL database file inside the repo, and created `data/state.db`
as a side effect of merely running the tests.

That is the hazard behind the droplet branch's red suite (fixed in `3c30737`):
six tests drove `bot._place_live_entry`, which persists `active_bracket` via
`state.set_meta`, against whatever `data/state.db` happened to exist on that
machine — and failed with

    sqlite3.OperationalError: no such table: meta

The tests that passed did so by patching `state.set_meta` one call site at a
time. That is a defence every future test author has to remember, and forgetting
it produces a failure whose message points at sqlite rather than at the missing
patch. Worse, on a developer machine where a populated `data/state.db` already
exists, the same test would silently PASS while writing to a real database.

This fixture removes the hazard rather than defending against it: every test
gets its own freshly-initialised database in its own tmp dir.

`set_db_path()` rebinds the module-level `DB_PATH` global (state.py:48-56), so
both `state._conn()` and `bot.py`'s direct read of `state.DB_PATH` follow it —
the direct read is why an accessor-only fix would not have been enough.

The same file also stops the suite from talking to the real SMTP relay — see
`no_outbound_alerts` below.

NOTE: this is isolation, not a behaviour change. Tests that patch `state.*` keep
working untouched; they simply no longer NEED to.
"""

from __future__ import annotations

import pytest

from exchange import state


@pytest.fixture(autouse=True)
def isolated_state_db(tmp_path, monkeypatch):
    """Point the state layer at a per-test sqlite file, then restore it.

    The env var is set too so that any code re-reading `SNAPBACK_STATE_DB`
    (or a subprocess) agrees with the rebound global.
    """
    original = state.DB_PATH
    db = tmp_path / "state.db"
    monkeypatch.setenv("SNAPBACK_STATE_DB", str(db))
    state.set_db_path(db)
    state.init_db()
    try:
        yield db
    finally:
        state.set_db_path(original)


# The SMTP vars alerts.is_configured() requires. Clearing any one of them is
# enough to make send_alert() short-circuit, but we clear all six so the intent
# survives a future change to that list.
_SMTP_ENV = (
    "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
    "ALERT_EMAIL_FROM", "ALERT_EMAIL_TO",
)


@pytest.fixture(autouse=True)
def no_outbound_alerts(monkeypatch):
    """Never let the test suite send a real alert email.

    `alerts.send_alert()` reads SMTP credentials straight from the process
    environment, and `uv run pytest` loads the repo's real `.env` — so any test
    that reached an alerting call was opening a live connection to the
    production relay. Observed 2026-08-23 while exercising the trend-exit path:

        alerts: failed to send 'Bot trend-exit close':
        (535, b'Too many failed login requests from <ip>. Try again later.')

    That is worse than noise. Repeated failed logins from a developer IP are
    exactly what gets a sender throttled or blocked, and this relay is the
    channel the live legs use to report a kill switch. A test run must not be
    able to degrade production alerting.

    `send_alert()` returns False early when `is_configured()` is False, so
    clearing the vars disables sending without patching every call site — tests
    that assert on alerting still patch `send_alert` themselves and are
    unaffected.
    """
    for var in _SMTP_ENV:
        monkeypatch.delenv(var, raising=False)


# --- risk ceilings ------------------------------------------------------------
# risk.py has no numeric defaults (RISK_REVIEW 2026-10-22): bot.main() installs
# the leg's ceilings from its params `risk:` block. Tests that drive Bot methods
# directly never run main(), so seed the SYNTHETIC fixture ceilings here (at
# import, so module-level reads in test files see them too) and restore them
# after every test in case one reconfigures. Tests assert relative to these.
import pathlib as _pathlib  # noqa: E402

import yaml as _yaml  # noqa: E402

import risk as _risk  # noqa: E402

_FIXTURE_PARAMS = _yaml.safe_load(
    (_pathlib.Path(__file__).resolve().parent / "fixtures" / "params.yaml").read_text())
FIXTURE_CEILINGS = _risk.configure(_FIXTURE_PARAMS)


@pytest.fixture(autouse=True)
def fixture_risk_ceilings():
    _risk._ACTIVE = FIXTURE_CEILINGS
    yield FIXTURE_CEILINGS
    _risk._ACTIVE = FIXTURE_CEILINGS
