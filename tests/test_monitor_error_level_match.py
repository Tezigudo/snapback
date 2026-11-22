"""The monitor's ERROR alert must match what the bot actually writes.

Regression for 2026-11: monitor searched for the substring '"level":"ERROR"',
but bot.JsonFormatter writes json.dumps(...) output, '"level": "ERROR"' (space
after the colon), so the alert could never fire. CRITICAL matched nothing.

These tests write lines through the bot's REAL logging setup (not a
hand-written fixture string) so a future formatter change cannot silently
re-break the match.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402
import monitor  # noqa: E402


def _write_via_bot_logger(tmp_path: Path) -> Path:
    log_file = tmp_path / "donchian.jsonl"
    with patch.object(bot, "LOG_DIR", tmp_path), patch.object(bot, "LOG_FILE", log_file):
        logger = bot._setup_logging("DEBUG")
    try:
        logger.info("routine tick")
        logger.warning("a warning")
        logger.error("place_bracket failed: %s", "boom")
        logger.critical("position open with no stop")
        try:
            raise RuntimeError("x")
        except RuntimeError:
            logger.exception("crashed")  # level ERROR + exc field
        logger.info('msg that mentions "level":"ERROR" inside text is NOT an error')
    finally:
        for h in list(logger.handlers):
            h.flush()
            h.close()
            logger.removeHandler(h)
    return log_file


def test_bot_json_lines_really_have_a_space_after_colon(tmp_path):
    text = _write_via_bot_logger(tmp_path).read_text()
    assert '"level": "ERROR"' in text
    assert '"level":"ERROR"' not in text  # the old needle matches nothing


def test_error_and_critical_lines_counted(tmp_path):
    log_file = _write_via_bot_logger(tmp_path)
    counts, off = monitor._count_log_events(
        log_file, 0, {"ERROR": monitor._is_alert_level, "TB": "Traceback (most recent call last)"}
    )
    # error + critical + exception = 3; the INFO line quoting the old needle is not counted
    assert counts["ERROR"] == 3
    assert counts["TB"] == 1
    assert off == log_file.stat().st_size


def test_line_level_tolerates_whitespace_and_partial_lines():
    assert monitor._line_level('{"level":"ERROR","msg":"x"}') == "ERROR"
    assert monitor._line_level('{"level" :  "CRITICAL", "msg": "x"}') == "CRITICAL"
    # truncated JSON at EOF falls back to the regex
    assert monitor._line_level('{"ts": "t", "level": "ERROR", "msg": "trunc') == "ERROR"
    assert monitor._line_level("2026-11-22 ERROR snapback plain stdout line") is None
    assert not monitor._is_alert_level('{"level": "WARNING"}')


def test_check_leg_emits_error_alert(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    src = _write_via_bot_logger(tmp_path)
    (logs / "donchian.jsonl").write_bytes(src.read_bytes())
    leg = {"name": "donchian", "heartbeat": "hb", "log": "donchian.jsonl",
           "state": "s.db", "systemd": "u", "live": False}
    cfg = {"alert_cooldown_min": 30, "heartbeat_stale_s": 10**9, "error_alert": True}
    state = {"log_offsets": {}}
    emitted: list[str] = []
    with patch.object(monitor, "LOGS", logs), patch.object(monitor, "DATA", tmp_path), \
         patch.object(monitor, "_systemd_active", return_value=True), \
         patch.object(monitor, "_heartbeat_age_s", return_value=1), \
         patch.object(monitor, "_emit", side_effect=lambda subj, *a, **k: emitted.append(subj)):
        monitor._check_leg(leg, cfg, state)
    assert any(s.startswith("ERROR x3: donchian") for s in emitted), emitted
    logging.getLogger("snapback").handlers.clear()
