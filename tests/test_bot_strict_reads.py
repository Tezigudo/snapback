"""Direct cover for the three bot.py reads S3 made strict (no literal fallback):

  * Bot.__init__        deploy.min_capital_warn_usdt
  * _format_deploy_summary  breakeven.at_r / breakeven.buffer_pct (banner)
  * _maybe_breakeven    breakeven.at_r / breakeven.buffer_pct

Mocked exchange only — no SOL 4h cache, no network. Values come from the
synthetic sol fixture, so a typo'd key name or a reintroduced default shows up.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tests.test_sol_breakeven import ENTRY, PARAMS, QTY, R, ROOT, _bot, _ohlcv, _seed

FIXTURE = yaml.safe_load(
    (Path(__file__).resolve().parent / "fixtures" / "params_sol_supertrend.yaml").read_text())
BE = FIXTURE["breakeven"]                    # synthetic: at_r 1.5, buffer_pct 0.05
FIXTURE_PARAMS = {**PARAMS, "breakeven": dict(BE),
                  "deploy": {**PARAMS["deploy"],
                             "min_capital_warn_usdt": FIXTURE["deploy"]["min_capital_warn_usdt"]}}


def test_min_capital_warn_is_read_from_params():
    bot, _ = _bot(FIXTURE_PARAMS)
    assert bot.min_capital_warn == float(FIXTURE["deploy"]["min_capital_warn_usdt"])


def test_min_capital_warn_missing_is_an_error_not_a_default():
    p = {**FIXTURE_PARAMS, "deploy": {"kill_switch_equity_fraction": 0.5}}
    with pytest.raises(KeyError, match="min_capital_warn_usdt"):
        _bot(p)


def test_banner_shows_configured_breakeven():
    bot, _ = _bot(FIXTURE_PARAMS)
    text = bot._format_deploy_summary(100.0, "test")
    assert f"at +{float(BE['at_r']):g}R" in text
    assert f"(buffer {float(BE['buffer_pct']):g}%)" in text


def test_banner_breakeven_off_does_not_read_tuning_keys():
    bot, _ = _bot({**FIXTURE_PARAMS, "breakeven": {"enabled": False}})
    assert "Breakeven stop : off" in bot._format_deploy_summary(100.0, "test")


def test_maybe_breakeven_uses_configured_at_r_and_buffer():
    assert BE["enabled"] is True and float(BE["at_r"]) != 2.0      # not the old SOL value
    bot, mc = _bot(FIXTURE_PARAMS)
    _seed()
    reach = ENTRY + float(BE["at_r"]) * R + 0.1           # just past at_r R
    _ohlcv(mc, [ENTRY + 1, reach, ENTRY + 1])
    with patch("bot.send_alert"):
        bot._maybe_breakeven(100.0)
    mc.place_tagged_stop.assert_called_once()
    a = mc.place_tagged_stop.call_args.args
    assert a[2] == QTY and a[4] == ROOT
    assert a[3] == pytest.approx(ENTRY * (1 + float(BE["buffer_pct"]) / 100.0))


def test_maybe_breakeven_not_due_below_configured_at_r():
    bot, mc = _bot(FIXTURE_PARAMS)
    _seed()
    _ohlcv(mc, [ENTRY + 1, ENTRY + float(BE["at_r"]) * R - 0.5, ENTRY + 1])
    bot._maybe_breakeven(100.0)
    mc.place_tagged_stop.assert_not_called()


@pytest.mark.parametrize("drop", ["at_r", "buffer_pct"])
def test_maybe_breakeven_enabled_without_tuning_key_raises(drop):
    """No silent 0.0 / 0.1 fallback (boot validation refuses this config)."""
    be = {k: v for k, v in BE.items() if k != drop}
    bot, _ = _bot({**FIXTURE_PARAMS, "breakeven": be})
    with pytest.raises(KeyError, match=drop):
        bot._maybe_breakeven(100.0)
