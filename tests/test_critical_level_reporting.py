"""F11 (2026-11 fix-pass): the daily digest and the dashboard count CRITICAL
log lines, not only ERROR (the monitor was fixed for this in fc3c62b)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import daily_digest  # noqa: E402
from tests.test_dashboard_pnl_basis import _db, _fixture_config_dir  # noqa: E402
from tools import build_dashboard as bd  # noqa: E402


def _summary(by_level):
    return {"name": "donchian", "live": True, "event_count": 1, "by_level": by_level,
            "heartbeat_mtime_age_s": None, "anchor_equity": None, "current_equity": None,
            "top_block_reasons": [], "nan_4h_count": 0}


def test_digest_counts_critical():
    assert daily_digest._alert_level_count({"CRITICAL": 1}) == 1
    assert daily_digest._alert_level_count({"ERROR": 2, "CRITICAL": 1, "INFO": 9}) == 3
    assert "ERROR/CRITICAL events: 1" in daily_digest._format_leg(_summary({"CRITICAL": 1}))


def test_dashboard_lists_critical_lines():
    logs = [{"ts": "2026-11-22T00:00:00+00:00", "level": "CRITICAL", "logger": "snapback",
             "msg": "position open with no stop UNIQUE_CRIT_MARKER"}]
    html = bd.render_html(_db(deploy_start="100", principal="100", equity_after=100.0),
                          logs, "v1")
    assert "No errors or warnings" not in html        # it is in the errors table
    assert "<tr class='row-err'>" in html             # and styled as an error row
