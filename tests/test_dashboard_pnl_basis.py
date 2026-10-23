"""The leg dashboard must not read a deposit as profit, or invent a kill line.

Regression cover for 2026-09-09. `tools/build_dashboard.py` runs from cron every
60 s and measured P&L as `cur_equity / deploy_start_equity`, a RAW snapshot
frozen at first boot. Every later deposit moved the numerator and left the basis
behind, so the page contradicted itself:

    v1 dashboard card:   +10.46%  /  +$14.96      <- deploy_start_equity 142.93
    v1 bot log, SAME page: -14.25%                <- principal 183.93

donchian was worse -- start 50.50 against principal 172.53 renders +253.96% for
a leg that is up 3.60%.

Separately the kill card computed `deploy_start_equity * 0.82`, which was the
deploy-era rule. The live switch is `principal * kill_switch_equity_fraction`
(one configured fraction per leg), so the page named a floor the bot does not enforce. That
constant is now read from the leg's config, never hardcoded.

Fourth reader of the same pair, after PR #25 (monitor.py) and PR #30 (bot.py).
"""

from __future__ import annotations

import os

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from tools import build_dashboard as bd  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def _fixture_config_dir(monkeypatch):
    """The real params YAMLs are untracked; read the synthetic fixtures
    (kill_switch_equity_fraction 0.55 on every leg)."""
    monkeypatch.setattr(bd, "CONFIG_DIR", FIXTURES)


def _db(*, deploy_start: str, principal: str | None, equity_after: float,
        stale_cached_anchor: str | None = None) -> dict:
    """A minimal read_db() payload. meta values are TEXT, as sqlite returns.

    `principal` is the LIVE ledger derivation (read_db's "principal" key).
    `stale_cached_anchor` seeds meta['principal_anchor'] with a different value
    so a test can prove the renderer ignores the cache.
    """
    meta = {"deploy_start_equity": deploy_start,
            "deploy_start_ts": "2026-07-25T05:45:27+00:00"}
    if stale_cached_anchor is not None:
        meta["principal_anchor"] = stale_cached_anchor
    return {
        "meta": meta,
        # (ts, side, qty, price, pnl_usd, reason, equity_after)
        "fills": [("2026-09-09T03:00:00+00:00", "long", 0.01, 100.0,
                   None, "entry", equity_after)],
        "events": [],
        "principal": None if principal is None else float(principal),
    }


def _render(db: dict, instance: str = "v1") -> str:
    return bd.render_html(db, [], instance)


class TestPnlBasis:

    def test_v1_deposit_is_not_reported_as_profit(self, caplog):
        """The live numbers. $142.93 at deploy, $183.93 actually funded."""
        html = _render(_db(deploy_start="142.93", principal="183.93",
                           equity_after=157.89))
        assert "-14.1" in html          # truth
        assert "10.4" not in html       # the figure the bug printed
        assert "vs principal" in html

    def test_donchian_stale_start_does_not_render_a_fake_250_percent(self):
        """start 50.50 vs principal 172.53 -- the widest gap in the book."""
        html = _render(_db(deploy_start="50.50", principal="172.53",
                           equity_after=178.75), instance="donchian")
        assert "+3.6" in html
        assert "253" not in html

    def test_basis_is_principal_when_present(self):
        """Clean round numbers so the arithmetic is unambiguous:
        110/100 = +10.00%, while the old basis of 25 would give +340.00%."""
        html = _render(_db(deploy_start="25.00", principal="100.00",
                           equity_after=110.00))
        assert "+10.00%" in html
        assert "340" not in html

    def test_missing_principal_falls_back_but_says_so(self):
        """A pre-Part-C leg has no ledger. Falling back is correct; printing the
        unadjusted number WITHOUT the caveat is what kept this invisible."""
        html = _render(_db(deploy_start="100.00", principal=None,
                           equity_after=110.00))
        assert "+10.00%" in html
        assert "NOT transfer-adjusted" in html


class TestKillLine:

    def test_kill_line_uses_the_configured_fraction_not_0_82(self):
        """Fixture fraction 0.55 of principal 100.00 = 55.00. The old rule would
        have printed 0.82 of deploy-start 25.00 = 20.50."""
        html = _render(_db(deploy_start="25.00", principal="100.00",
                           equity_after=110.00))
        assert "$55.00" in html
        assert "$20.50" not in html
        assert "-45.0% from principal" in html

    def test_stale_18_percent_label_is_gone(self):
        html = _render(_db(deploy_start="100.00", principal="100.00",
                           equity_after=100.00))
        assert "-18% from start" not in html

    def test_unreadable_config_renders_a_dash_not_a_number(self):
        """An invented kill level is worse than none -- you would plan around
        it. read_kill_fraction returns None on any failure."""
        with patch.object(bd, "read_kill_fraction", return_value=None):
            html = _render(_db(deploy_start="100.00", principal="100.00",
                               equity_after=110.00))
        assert "config unreadable" in html

    def test_no_principal_says_the_kill_switch_is_disabled(self):
        """principal.breached() returns False when principal is None or <= 0,
        so the bot enforces NO floor. Printing one off deploy-start equity
        would repeat this bug's own mistake: naming a level nothing enforces."""
        html = _render(_db(deploy_start="100.00", principal=None,
                           equity_after=110.00))
        assert "kill switch DISABLED: no principal anchor" in html
        assert "$82.00" not in html   # 100.00 * 0.82, the old rule
        assert "$55.00" not in html   # 100.00 * 0.55, the plausible-but-wrong one


class TestPrincipalIsDerivedNotCached:
    """meta.principal_anchor is documented (principal.py:52) as a CACHE of
    base + sum(ledger), refreshed hourly. A transfer hits the ledger first, so
    rendering from the cache reports against a different quantity than the kill
    switch enforces -- the defect PR #25's review round found in monitor.py."""

    def test_renderer_ignores_a_stale_cached_anchor(self):
        """Live ledger says 100.00; the cache still says 80.00. 110/100 =
        +10.00%, whereas the stale cache would give +37.50%."""
        html = _render(_db(deploy_start="60.00", principal="100.00",
                           equity_after=110.00, stale_cached_anchor="80.00"))
        assert "+10.00%" in html
        assert "37.5" not in html
        assert "$55.00" in html        # 100.00 * 0.55, off the LIVE value
        assert "$44.00" not in html    # 80.00 * 0.55, off the stale cache

    def _leg_db(self, tmp_path, *, source: str | None,
                base: str, ledger: list[tuple[float, str]]):
        import sqlite3 as sq
        p = tmp_path / "state_x.db"
        c = sq.connect(p)
        c.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        c.execute("CREATE TABLE principal_ledger (tran_id INTEGER PRIMARY KEY, "
                  "income_type TEXT, income_usd REAL, asset TEXT, ts_ms INTEGER, "
                  "applied_at TEXT)")
        if source is not None:
            c.execute("INSERT INTO meta VALUES ('principal_source', ?)", (source,))
        c.execute("INSERT INTO meta VALUES ('principal_base', ?)", (base,))
        for i, (amt, asset) in enumerate(ledger):
            c.execute("INSERT INTO principal_ledger VALUES (?,?,?,?,?,?)",
                      (i, "TRANSFER", amt, asset, 0, ""))
        c.commit()
        meta = dict(c.execute("SELECT key, value FROM meta").fetchall())
        return c, meta

    def test_sums_usdt_ledger_onto_base(self, tmp_path):
        """sol's real shape: base 0.0, deposits of 60 then 20 -> P = 80."""
        c, meta = self._leg_db(tmp_path, source="income_backfill", base="0.0",
                               ledger=[(60.0, "USDT"), (20.0, "USDT")])
        assert bd._live_principal(c, meta) == 80.0

    def test_base_of_zero_is_not_treated_as_missing(self, tmp_path):
        """principal_base is legitimately 0.0 on every live leg."""
        c, meta = self._leg_db(tmp_path, source="income_backfill", base="0.0",
                               ledger=[(42.0, "USDT")])
        assert bd._live_principal(c, meta) == 42.0

    def test_non_usdt_rows_are_excluded(self, tmp_path):
        """Each leg carries a small BNB fee transfer that is not principal."""
        c, meta = self._leg_db(tmp_path, source="income_backfill", base="0.0",
                               ledger=[(60.0, "USDT"), (0.0035, "BNB")])
        assert bd._live_principal(c, meta) == 60.0

    def test_uninitialised_leg_returns_none(self, tmp_path):
        """Mirrors principal.is_initialized(): META_SOURCE unset -> None, which
        keeps the kill card fail-safe rather than inventing a floor."""
        c, meta = self._leg_db(tmp_path, source=None, base="0.0",
                               ledger=[(60.0, "USDT")])
        assert bd._live_principal(c, meta) is None

    def test_missing_ledger_table_returns_none(self, tmp_path):
        """A pre-Part-C leg DB has no principal_ledger at all."""
        import sqlite3 as sq
        p = tmp_path / "old.db"
        c = sq.connect(p)
        c.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        c.execute("INSERT INTO meta VALUES ('principal_source', 'manual')")
        c.commit()
        assert bd._live_principal(c, {"principal_source": "manual"}) is None

    def test_read_kill_fraction_reads_every_configured_leg(self):
        """Reads a YAML per leg from CONFIG_DIR (the fixtures here), so a config
        filename rename breaks the test rather than silently returning None and
        dashing out the kill card."""
        for inst, files in bd.INSTANCE_FILES.items():
            # The public snapshot omits the retired cnh legs' fixtures; only
            # there may a leg without one be skipped.
            if (os.environ.get("SNAPBACK_PUBLIC_SNAPSHOT") == "1"
                    and not (bd.CONFIG_DIR / files["cfg"]).is_file()):
                continue
            assert bd.read_kill_fraction(inst) == 0.55, inst

    def test_read_kill_fraction_rejects_a_nonsense_value(self):
        """Out-of-range means the config is wrong, not that we should multiply
        equity by it."""
        with patch.object(bd.yaml, "safe_load",
                          return_value={"deploy": {"kill_switch_equity_fraction": 42}}):
            assert bd.read_kill_fraction("v1") is None
