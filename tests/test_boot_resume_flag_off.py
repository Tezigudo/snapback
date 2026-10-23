"""Boot-resume ships NOT ARMED: with the flag off, boot() is today's boot.

"Armed" means `boot_resume: {enabled: true, observe_only: false}`. Both keys
default to the safe side in code (`enabled` → False, `observe_only` → True),
v1's params.yaml ships `observe_only: true`, and donchian / sol carry no block
at all. These tests pin all three facts, then check the behaviour with a
position that the gate WOULD adopt: it must still be flattened, with the same
close call, and none of the Phase B state (seed, loop counter, tracking) may be
touched.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from exchange import state
from tests._live_config import live_config
from tests.test_boot_resume import (
    _ARMED_BRACKET,
    _MINIMAL_PARAMS,
    _boot_patches,
    _make_bot,
    _open,
    _principal_patches,
)

REPO = Path(__file__).resolve().parents[1]

_OBSERVE = dict(_MINIMAL_PARAMS)                                   # observe_only: true
_ABSENT = {k: v for k, v in _MINIMAL_PARAMS.items() if k != "boot_resume"}
_DISABLED = {**_MINIMAL_PARAMS, "boot_resume": {"enabled": False, "observe_only": False}}
_NO_OBSERVE_KEY = {**_MINIMAL_PARAMS, "boot_resume": {"enabled": True}}


def _seed_adoptable():
    """Everything an ARMED boot would need to adopt, so only the flag differs."""
    state.record_fill(side="long", qty=0.01, price=65000.0, reason="entry",
                      client_order_id_root="sig-1")
    state.set_meta("active_bracket", json.dumps(_ARMED_BRACKET))
    state.set_meta("last_entry_bar_ts", "2026-09-29T09:45:00")


def _boot(params):
    bot, mc = _make_bot(params=dict(params))
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.fetch_position.return_value = _open()
    mc.close_position.return_value = {}
    set_meta = MagicMock(side_effect=state.set_meta)
    with _boot_patches(), _principal_patches(), patch("bot.state.set_meta", set_meta):
        bot.boot()
    return bot, mc, [c.args[0] for c in set_meta.call_args_list]


def _outbox_kinds() -> list[str]:
    import sqlite3
    with sqlite3.connect(state.DB_PATH) as c:
        return [r[0] for r in c.execute("SELECT kind FROM outbox ORDER BY id")]


# ---------------------------------------------------------------------------
# 1. The flag's defaults
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_code_defaults_are_off(self):
        """Absent block, and a block with no observe_only key, never adopt."""
        for params in (_ABSENT, _NO_OBSERVE_KEY, _DISABLED,
                   {**_MINIMAL_PARAMS,
                    "boot_resume": {"enabled": True, "observe_only": False}}):
            bot, _ = _make_bot(params=dict(params))
            with patch.object(bot, "_can_adopt", return_value=(True, "forced yes")):
                assert bot._boot_resume_verdict(_open())[0] is False

    # The real params YAMLs are untracked; the shipped defaults now live in the
    # tracked templates (config/*.example.yaml), which is what a new operator
    # copies. The deployed files are checked too where present.
    @pytest.mark.parametrize("src", ["example", "live"])
    def test_v1_config_ships_observe_only(self, src):
        if src == "example":
            # Templates are all-null (fail-closed); the v1 template must carry the
            # block (it is REQUIRED on v1) so an operator has to decide it.
            p = yaml.safe_load((REPO / "config" / "params.example.yaml").read_text())
            assert set(p["boot_resume"]) == {"enabled", "observe_only"}
            return
        p = live_config("params.yaml")
        assert p["boot_resume"]["observe_only"] is True

    @pytest.mark.parametrize("src", ["example", "live"])
    @pytest.mark.parametrize("cfg", ["params_donchian.yaml", "params_sol_supertrend.yaml"])
    def test_other_legs_carry_no_block(self, cfg, src):
        if src == "example":
            p = yaml.safe_load((REPO / "config" / cfg.replace(".yaml", ".example.yaml")).read_text())
        else:
            p = live_config(cfg)
        assert "boot_resume" not in p
        assert "reprotect" not in p            # and so could never pass the gate


# ---------------------------------------------------------------------------
# 2. Flag off ⇒ boot flattens exactly as before
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("params", [_OBSERVE, _ABSENT, _DISABLED, _NO_OBSERVE_KEY],
                         ids=["observe_only", "block_absent", "disabled", "no_observe_key"])
def test_flag_off_flattens_an_adoptable_position_exactly_as_before(params):
    _seed_adoptable()
    bot, mc, meta_keys = _boot(params)

    mc.close_position.assert_called_once_with(
        "BTC/USDT:USDT", client_order_id_root="sig-1", close_leg="bf")
    kinds = _outbox_kinds()
    assert "boot_flatten" in kinds and "boot_adopt" not in kinds
    # None of Phase B's state is touched on the boot path.
    assert "boot_adopt_log" not in meta_keys
    assert "last_entry_bar_ts" not in meta_keys
    assert state.get_meta("boot_adopt_log") is None
    assert state.get_meta("last_entry_bar_ts") == "2026-09-29T09:45:00"
    assert bot._last_position_side == "unknown"
    assert bot._last_entry_root is None
    assert bot._last_signal_ts is None
    assert bot._adopt_signal_ts_seed is None


def _names(mc) -> list[str]:
    return [c[0] for c in mc.mock_calls]


def test_observe_only_makes_zero_extra_exchange_calls():
    """Same boot with the block absent vs observe_only: the client call
    sequence is IDENTICAL — observe mode reads no order book at all."""
    _seed_adoptable()
    _b, mc_absent, _ = _boot(_ABSENT)
    _b, mc_observe, _ = _boot(_OBSERVE)
    assert _names(mc_observe) == _names(mc_absent)
    for n in ("ex.fetch_open_orders", "fetch_algo_orders"):
        assert n not in _names(mc_observe)


def test_observe_verdict_says_the_book_check_was_skipped(caplog):
    _seed_adoptable()
    with caplog.at_level("WARNING"):
        _boot(_OBSERVE)
    lines = [r.getMessage() for r in caplog.records if "boot-resume OBSERVE" in r.getMessage()]
    assert len(lines) == 1
    assert "order-book check SKIPPED in observe mode" in lines[0]


@pytest.mark.parametrize("bad", [["enabled"], "true", 1, True])
def test_malformed_boot_resume_config_still_flattens(bad):
    """A non-mapping boot_resume must not abort boot() before the flatten."""
    _seed_adoptable()
    _bot, mc, meta_keys = _boot({**_MINIMAL_PARAMS, "boot_resume": bad})
    mc.close_position.assert_called_once_with(
        "BTC/USDT:USDT", client_order_id_root="sig-1", close_leg="bf")
    assert "boot_adopt_log" not in meta_keys


@pytest.mark.parametrize("br", [{"enabled": "yes", "observe_only": False},
                                {"enabled": True, "observe_only": "false"},
                                {"enabled": True, "observe_only": 0}])
def test_non_boolean_flags_never_arm(br):
    _seed_adoptable()
    with patch("bot.ARMING_PREREQS_BUILT", True):
        _bot, mc, _ = _boot({**_MINIMAL_PARAMS, "boot_resume": br})
    mc.close_position.assert_called_once()


# ---------------------------------------------------------------------------
# 2b. HARD ARMING INTERLOCK — one config line cannot arm adoption
# ---------------------------------------------------------------------------

_ARMED_CONFIG = {**_MINIMAL_PARAMS, "boot_resume": {"enabled": True, "observe_only": False}}


def test_interlock_constant_ships_false():
    import bot
    assert bot.ARMING_PREREQS_BUILT is False


def test_interlock_forces_flatten_when_config_says_armed(caplog):
    """observe_only: false + an ADOPTABLE position + prereqs NOT built ⇒
    ERROR log, alert, and today's flatten — with zero extra exchange calls."""
    _seed_adoptable()
    _b, mc_absent, _ = _boot(_ABSENT)

    bot, mc = _make_bot(params=dict(_ARMED_CONFIG))
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.fetch_position.return_value = _open()
    mc.close_position.return_value = {}
    alert = MagicMock()
    with caplog.at_level("WARNING"), _boot_patches(), _principal_patches(), \
         patch("bot.send_alert", alert):
        bot.boot()

    mc.close_position.assert_called_once_with(
        "BTC/USDT:USDT", client_order_id_root="sig-1", close_leg="bf")
    assert "boot_adopt" not in _outbox_kinds()
    assert [c for c in alert.call_args_list
            if c.args and "arming interlock" in c.args[0]]
    assert any(r.levelname == "ERROR" and "ARMING INTERLOCK" in r.getMessage()
               for r in caplog.records)
    assert _names(mc) == _names(mc_absent)
    assert state.get_meta("boot_adopt_log") is None
    assert bot._last_position_side == "unknown"


def test_the_same_config_adopts_only_once_the_constant_flips():
    """Proves the interlock is the ONLY thing stopping the adopt above."""
    _seed_adoptable()
    with patch("bot.ARMING_PREREQS_BUILT", True):
        _bot, mc, _ = _boot(_ARMED_CONFIG)
    mc.close_position.assert_not_called()
    assert "boot_adopt" in _outbox_kinds()


def test_dry_run_never_consults_the_gate():
    """T11: dry-run leaves a position alone and never asks the gate."""
    _seed_adoptable()
    bot, mc = _make_bot(dry_run=True,
                        params={**_MINIMAL_PARAMS,
                                "boot_resume": {"enabled": True, "observe_only": False}})
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.fetch_position.return_value = _open()
    with _boot_patches(), _principal_patches(), \
         patch.object(bot, "_boot_resume_verdict",
                      side_effect=AssertionError("gate consulted in dry-run")):
        bot.boot()
    mc.close_position.assert_not_called()


# ---------------------------------------------------------------------------
# 3. T15 — the flatten block is byte-identical to droplet 704df89
# ---------------------------------------------------------------------------

# sha256 of the flatten block as it stands on `droplet` @ 704df89 (pre-Phase-A),
# computed from `git show 704df89:bot.py` with _flatten_block below.
_DROPLET_704DF89_FLATTEN_SHA256 = (
    "c95930f5981a59581d5eb2f18ee180346a411b4b9b611b09e1d8924cdc4583c5")

_START = ('                root = state.latest_entry_coid_root()\n'
          '                self.log.warning("Boot found open position %s qty=%.4f @ %.2f. "\n'
          '                                 "Flattening (root=%s).",\n')
_END = '        else:\n            # Position is already flat at boot.'


def _flatten_block(src: str) -> str:
    assert src.count(_START) == 1, "flatten block start marker not unique"
    i = src.index(_START)
    return src[i:src.index(_END, i)]


# The ONE deliberate change to the flatten block since 704df89/d380405: an
# early return when close_position reports the exchange already flat (panel
# fix-pass, see Bot._boot_close_found_flat). Everything else is byte-identical:
# removing exactly this insertion must give back the droplet hash.
_ALREADY_FLAT_INSERT = ('                if close_order is None:\n'
                        '                    self._boot_close_found_flat(pos, root)\n'
                        '                    return\n')
_FLATTEN_SHA256_WITH_ALREADY_FLAT = (
    "a6c5f67681f2e699c31fe884657b371b3b5674bef0359f6650385c05abe6ce2f")


def test_flatten_block_is_byte_identical_to_droplet_baseline():
    block = _flatten_block((REPO / "bot.py").read_text())
    assert hashlib.sha256(block.encode()).hexdigest() == _FLATTEN_SHA256_WITH_ALREADY_FLAT
    assert block.count(_ALREADY_FLAT_INSERT) == 1
    stripped = block.replace(_ALREADY_FLAT_INSERT, "")
    assert hashlib.sha256(stripped.encode()).hexdigest() == _DROPLET_704DF89_FLATTEN_SHA256
    assert "close_leg=\"bf\"" in block


def test_baseline_hash_is_really_704df89():
    """Re-derive the pinned hash from git when the commit is reachable, so the
    constant cannot silently be re-pinned to the feature branch's own text."""
    try:
        src = subprocess.run(["git", "show", "704df89:bot.py"], cwd=REPO,
                             capture_output=True, text=True, check=True,
                             timeout=10).stdout
    except Exception:
        pytest.skip("704df89 not reachable from this checkout")
    got = hashlib.sha256(_flatten_block(src).encode()).hexdigest()
    assert got == _DROPLET_704DF89_FLATTEN_SHA256
