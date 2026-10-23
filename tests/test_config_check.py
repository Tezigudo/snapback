"""config_check.REQUIRED_KEYS: refuse to boot on a missing/null param, never
fall back to a code default.

Fixtures in tests/fixtures/ are complete, synthetic-valued configs with the same
structure as the tracked templates (config/*.example.yaml).
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import bot
from config_check import (
    EXIT_CONFIG,
    REQUIRED_KEYS,
    ParamsError,
    config_problems,
    validate_params,
)
from tests._live_config import live_config

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
CONFIG = ROOT / "config"

# instance -> params file name (bot.INSTANCE_PROFILES is the source of truth)
LEG_FILES = {inst: prof["config"].name for inst, prof in bot.INSTANCE_PROFILES.items()}
# The public snapshot omits the retired cnh legs' fixtures and templates; only
# there may a leg without them be dropped (elsewhere a missing fixture fails).
if os.environ.get("SNAPBACK_PUBLIC_SNAPSHOT") == "1":
    LEG_FILES = {inst: name for inst, name in LEG_FILES.items()
                 if (FIXTURES / name).is_file()}


def _fixture(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text())


def _drop(d: dict, dotted: str) -> None:
    *head, last = dotted.split(".")
    for part in head:
        d = d[part]
    del d[last]


def _set(d: dict, dotted: str, value) -> None:
    *head, last = dotted.split(".")
    for part in head:
        d = d[part]
    d[last] = value


# ---------------------------------------------------------------------------
# Shape: every leg has a template and a fixture; every strategy is covered
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inst", sorted(LEG_FILES))
def test_every_leg_has_template_and_fixture(inst):
    name = LEG_FILES[inst]
    assert (CONFIG / name.replace(".yaml", ".example.yaml")).is_file()
    assert (FIXTURES / name).is_file()


@pytest.mark.parametrize("inst", sorted(LEG_FILES))
def test_template_has_same_keys_as_fixture(inst):
    """Fixture = template with the nulls filled in; the key tree must match."""
    def keys(d, prefix=""):
        out = set()
        for k, v in d.items():
            out.add(prefix + k)
            if isinstance(v, dict):
                out |= keys(v, prefix + k + ".")
        return out
    name = LEG_FILES[inst]
    tmpl = yaml.safe_load((CONFIG / name.replace(".yaml", ".example.yaml")).read_text())
    assert keys(tmpl) == keys(_fixture(name))


def test_templates_are_gitignore_exempt_and_live_files_are_not():
    import subprocess
    def ignored(rel):
        return subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT).returncode == 0
    assert ignored("config/params.yaml")
    assert ignored("config/params_donchian.yaml")
    assert ignored("config/params_sol_supertrend.yaml")
    assert not ignored("config/params.example.yaml")
    assert not ignored("config/params_donchian.example.yaml")


# ---------------------------------------------------------------------------
# Complete config boots; templates (all-null) refuse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inst", sorted(LEG_FILES))
def test_complete_fixture_is_bootable(inst):
    params = _fixture(LEG_FILES[inst])
    assert config_problems(params) == []
    validate_params(params)  # does not raise


@pytest.mark.parametrize("inst", sorted(LEG_FILES))
def test_unfilled_template_refuses(inst):
    name = LEG_FILES[inst].replace(".yaml", ".example.yaml")
    params = yaml.safe_load((CONFIG / name).read_text())
    problems = config_problems(params)
    assert problems and all(p.endswith(": null") for p in problems), problems
    with pytest.raises(ParamsError):
        validate_params(params)


@pytest.mark.parametrize("inst", sorted(LEG_FILES))
def test_live_config_is_bootable(inst):
    """Where the real untracked file exists (operator machine / droplet), it
    must satisfy REQUIRED_KEYS — otherwise the next restart refuses to boot."""
    assert config_problems(live_config(LEG_FILES[inst])) == []


# ---------------------------------------------------------------------------
# Every single required key: missing -> refuse, null -> refuse
# ---------------------------------------------------------------------------

def _cases():
    for inst, name in sorted(LEG_FILES.items()):
        strat = _fixture(name)["strategy_name"]
        for key in REQUIRED_KEYS[strat]:
            if key == "strategy_name":
                continue  # covered by its own tests below
            yield pytest.param(name, key, id=f"{inst}:{key}")


@pytest.mark.parametrize(("name", "key"), list(_cases()))
def test_missing_required_key_refuses(name, key):
    params = _fixture(name)
    _drop(params, key)
    assert config_problems(params) == [f"{key}: missing"]
    with pytest.raises(ParamsError, match=key.replace(".", r"\.")):
        validate_params(params)


@pytest.mark.parametrize(("name", "key"), list(_cases()))
def test_null_required_key_refuses(name, key):
    params = _fixture(name)
    _set(params, key, None)
    assert config_problems(params) == [f"{key}: null"]


def test_all_problems_reported_at_once():
    params = _fixture("params.yaml")
    _drop(params, "sizing.risk_per_trade_pct")
    _set(params, "strategy.rsi_period", None)
    _drop(params, "deploy")
    problems = config_problems(params)
    assert "sizing.risk_per_trade_pct: missing" in problems
    assert "strategy.rsi_period: null" in problems
    assert "deploy.kill_switch_equity_fraction: missing" in problems


@pytest.mark.parametrize("value", [None, "", "no-such-strategy"])
def test_missing_or_unknown_strategy_name_refuses(value):
    """resolve_strategy_name() silently defaults to multifactor-v1; boot
    validation must not — a donchian file without its name would otherwise be
    traded as v1."""
    params = _fixture("params_donchian.yaml")
    params["strategy_name"] = value
    assert len(config_problems(params)) == 1
    params.pop("strategy_name")
    assert len(config_problems(params)) == 1


@pytest.mark.parametrize("bad", [None, [], "x", 3])
def test_non_mapping_refuses(bad):
    assert config_problems(bad)


def test_falsy_values_are_valid():
    """0 / 0.0 / false are live config (sol's max_hold_bars: 0, hedge off,
    limit_offset_bps 0.0) — only null/absent refuses."""
    params = _fixture("params_sol_supertrend.yaml")
    _set(params, "strategy.max_hold_bars", 0)
    _set(params, "strategy.allow_shorts", False)
    _set(params, "hedge.enabled", False)
    _set(params, "execution.limit_offset_bps", 0.0)
    assert config_problems(params) == []


def test_breakeven_keys_required_only_when_enabled():
    """On a leg where breakeven is optional (donchian), a present+enabled
    block still requires its tuning keys; disabled/absent is fine."""
    params = _fixture("params_donchian.yaml")
    params["breakeven"] = {"enabled": True, "at_r": None, "buffer_pct": 0.05}
    assert config_problems(params) == ["breakeven.at_r: null"]
    _set(params, "breakeven.enabled", False)
    assert config_problems(params) == []
    params.pop("breakeven")
    assert config_problems(params) == []          # absent block = off


@pytest.mark.parametrize(("name", "block"), [
    ("params_sol_supertrend.yaml", "breakeven"),
    ("params.yaml", "reprotect"),
    ("params.yaml", "boot_resume"),
])
def test_live_safety_block_is_required_on_its_leg(name, block):
    """sol's breakeven and v1's reprotect/boot_resume are LIVE: a missing or
    typo'd block must refuse, not silently switch the feature off."""
    params = _fixture(name)
    params.pop(block)
    problems = config_problems(params)
    assert problems and all(p.startswith(block + ".") for p in problems), problems
    params = _fixture(name)
    params[block] = {("enable" if k == "enabled" else k): v      # typo'd key
                     for k, v in params[block].items()}
    assert f"{block}.enabled: missing" in config_problems(params)


@pytest.mark.parametrize("name", ["params_donchian.yaml"])
def test_safety_blocks_stay_optional_where_not_live(name):
    params = _fixture(name)
    for block in ("breakeven", "reprotect", "boot_resume", "no_stop"):
        params.pop(block, None)
    assert config_problems(params) == []
    flat = {k for keys in REQUIRED_KEYS.values() for k in keys}
    assert not any(k.startswith("no_stop") for k in flat)
    assert "strategy.dedup_bars" not in flat


@pytest.mark.parametrize("bad", ["true", "false", 1, 0, "yes"])
@pytest.mark.parametrize(("name", "key"), [
    ("params_sol_supertrend.yaml", "breakeven.enabled"),
    ("params.yaml", "reprotect.observe_only"),
    ("params.yaml", "boot_resume.enabled"),
    ("params.yaml", "strategy.require_trend"),
    ("params_donchian.yaml", "hedge.enabled"),
])
def test_bool_keys_reject_non_bool(name, key, bad):
    params = _fixture(name)
    _set(params, key, bad)
    assert config_problems(params) == [f"{key}: must be true/false, got {bad!r}"]


@pytest.mark.parametrize(("key", "bad"), [
    ("deploy.kill_switch_equity_fraction", 0),
    ("deploy.kill_switch_equity_fraction", 1),
    ("deploy.kill_switch_equity_fraction", 1.5),
    ("deploy.kill_switch_equity_fraction", -0.2),
    ("deploy.kill_switch_equity_fraction", "0.5"),     # quoted string
    ("sizing.leverage", 0),
    ("sizing.leverage", 2.5),
    ("sizing.leverage", True),
    ("sizing.risk_per_trade_pct", 0),
    ("sizing.risk_per_trade_pct", -1.0),
    ("execution.poll_interval_s", 0),
    ("breakeven.at_r", 0),
    ("breakeven.buffer_pct", -0.01),
    ("execution.poll_interval_s", 3601),
    ("execution.poll_interval_s", float("inf")),
    ("sizing.leverage", 126),
    ("sizing.risk_per_trade_pct", float("inf")),
    ("deploy.kill_switch_equity_fraction", float("nan")),
    ("breakeven.at_r", float("inf")),
])
def test_safety_ranges_refuse_placeholders(key, bad):
    params = _fixture("params_sol_supertrend.yaml")
    _set(params, key, bad)
    problems = config_problems(params)
    assert len(problems) == 1 and problems[0].startswith(f"{key}: must be"), problems


# The three deployed legs' last tracked configs (b9cb2d0, before the untrack)
# must pass every check — no false refusal from the safety/bool/range rules.
_LIVE_REF = "b9cb2d072633cc855fec06d3ebc0e74326d4befd"


@pytest.mark.parametrize("name", sorted(set(LEG_FILES.values())))
def test_last_tracked_live_yamls_pass_all_checks(name):
    import os
    import subprocess
    r = subprocess.run(["git", "show", f"{_LIVE_REF}:config/{name}"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        if os.environ.get("SNAPBACK_PUBLIC_SNAPSHOT") == "1":
            pytest.skip("public snapshot: private history not available")
        pytest.fail(f"{_LIVE_REF}:config/{name} not in history — set "
                    "SNAPBACK_PUBLIC_SNAPSHOT=1 only in the public snapshot")
    # Pre-untrack copies predate the `risk:` block: un-migrated they refuse on
    # exactly the risk.* keys, and after the droplet migration they pass.
    from risk import RISK_KEYS
    from tools.migrate_risk_params import PRE_CHANGE_SHA, migrate_text, values_from_git
    assert config_problems(yaml.safe_load(r.stdout)) == [f"{k}: missing" for k in RISK_KEYS]
    migrated = migrate_text(r.stdout, values_from_git(PRE_CHANGE_SHA), "test")
    assert config_problems(yaml.safe_load(migrated)) == []


# Keys the live trading code may read without them being in REQUIRED_KEYS:
# optional-by-design blocks (absent = feature off) and their keys.
DOCUMENTED_OPTIONAL = frozenset({
    "breakeven", "reprotect", "boot_resume", "no_stop",
    "enabled", "observe_only", "at_r", "buffer_pct",
    "max_replaces_per_position", "max_sl_place_failures",
    "unknown_book_close_after_s", "dedup_bars", "use_mtf_4h_gate",
    "require_trend", "manual_principal_usdt",
})

_ALL_STRATS = tuple(sorted(REQUIRED_KEYS))

# module -> strategies whose live path runs the module's UNBRANCHED code.
# Strategy evaluators serve one strategy; bot.py / bot_internals.py serve every
# leg, so an unbranched read there must be allowed for ALL strategies.
_LIVE_MODULES = {
    "strategy/live_multifactor_v1.py": ("multifactor-v1",),
    "strategy/live_donchian_v3.py": ("donchian-v3",),
    "strategy/live_supertrend.py": ("supertrend",),
    "bot_internals.py": _ALL_STRATS,
    "bot.py": _ALL_STRATS,
}

# Reads whose strategy scope the branch tracker cannot infer, pinned by hand:
# evaluate_for_strategy's fall-through `return` is the multifactor-v1 path (no
# `if strategy_name == "multifactor-v1"` guard).
_SCOPE_OVERRIDES = {
    ("bot_internals.py", "sl_pct"): ("multifactor-v1",),
    ("bot_internals.py", "tp_pct"): ("multifactor-v1",),
}

# Name collisions with module-internal dicts (not params reads): supertrend's
# resolved `_cfg()` dict uses "atr_period" for the st_atr_period value.
_INTERNAL_DICT_KEYS = {"strategy/live_supertrend.py": {"atr_period"}}


def _param_universe() -> set[str]:
    """Every key name (any depth) that appears in a params template."""
    names: set[str] = set()

    def walk(d):
        for k, v in d.items():
            names.add(k)
            if isinstance(v, dict):
                walk(v)
    for f in CONFIG.glob("params*.example.yaml"):
        walk(yaml.safe_load(f.read_text()))
    return names


# Variables that hold a params (sub)dict by convention in the live modules.
# Keys read off these count even if no template has them (a brand-new key).
_PARAMS_VARS = frozenset({"params", "s", "be", "be_cfg", "rp", "br",
                          "exec_cfg", "hedge_cfg"})


def _receiver_is_params(node) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _PARAMS_VARS
    if isinstance(node, ast.Attribute):              # self.params
        return node.attr == "params"
    if isinstance(node, ast.BoolOp):                 # (x.get("blk") or {})
        return any(_receiver_is_params(v) for v in node.values)
    if isinstance(node, ast.Subscript):              # params["deploy"][...]
        return _receiver_is_params(node.value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr == "get" and _receiver_is_params(node.func.value)
    return False


def _branch_strategy(test) -> str | None:
    """'X' for an `if <...>strategy_name == "X":` test, else None."""
    if (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant)
            and isinstance(test.comparators[0].value, str)):
        left = test.left
        name = left.id if isinstance(left, ast.Name) else getattr(left, "attr", None)
        if name == "strategy_name":
            return test.comparators[0].value
    return None


def _keys_read(path: Path) -> list[tuple[str, bool, str | None]]:
    """[(key, read-off-a-params-variable?, enclosing strategy branch or None)]
    for every x["k"] / x.get("k", ...) in the module."""
    out: list[tuple[str, bool, str | None]] = []

    def visit(node, branch):
        if isinstance(node, ast.If) and (b := _branch_strategy(node.test)):
            visit(node.test, branch)
            for n in node.body:
                visit(n, b)
            for n in node.orelse:
                visit(n, branch)
            return
        key = recv = None
        if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            key, recv = node.slice.value, node.value
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "get" and node.args
              and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)):
            key, recv = node.args[0].value, node.func.value
        if key is not None:
            out.append((key, _receiver_is_params(recv), branch))
        for child in ast.iter_child_nodes(node):
            visit(child, branch)

    visit(ast.parse(path.read_text()), None)
    return out


def _allowed_for(strategies) -> set[str]:
    """Key names required by EVERY strategy in scope, plus documented optionals."""
    sets = [{part for dotted in REQUIRED_KEYS[st] for part in dotted.split(".")}
            for st in strategies]
    return set.intersection(*sets) | set(DOCUMENTED_OPTIONAL)


@pytest.mark.parametrize("module", sorted(_LIVE_MODULES))
def test_every_param_key_live_code_reads_is_required_or_documented_optional(module):
    """Drift guard: a future `s.get("new_key", 0.5)` in a live module fails here
    until the key is added to REQUIRED_KEYS (or deliberately to the optional
    allowlist). Static (AST) so it covers branches a runtime test never hits.

    Per-strategy: a read inside `if strategy_name == "X":` must be allowed for X;
    an unbranched read in bot.py / bot_internals.py must be allowed for EVERY
    strategy (that code runs on every leg); a strategy module's reads must be
    allowed for its own strategy.

    Best-effort — known blind spots:
      * keys built dynamically (variables, f-strings, loops over key lists);
      * a NEW key read through a receiver name not in _PARAMS_VARS (e.g. `cfg`)
        is only seen if that name already appears in some template;
      * matching is by key NAME, not full dotted path, so e.g. `enabled` read
        from the wrong block of an in-scope strategy passes;
      * branch inference only understands `strategy_name == "X"` tests (see
        _SCOPE_OVERRIDES for the one fall-through path pinned by hand).
    """
    universe = _param_universe()
    internal = _INTERNAL_DICT_KEYS.get(module, set())
    bad = []
    for key, strict, branch in _keys_read(ROOT / module):
        if not strict and key not in universe:
            continue
        scope = (_SCOPE_OVERRIDES.get((module, key))
                 or ((branch,) if branch in REQUIRED_KEYS else _LIVE_MODULES[module]))
        if key not in _allowed_for(scope) | internal:
            bad.append((key, scope))
    assert not bad, f"{module} reads params keys not required for their strategy scope: {sorted(set(bad))}"


def test_drift_guard_is_per_strategy():
    """A donchian-only key read in an unbranched bot.py path (runs on v1/sol
    too) is NOT allowed; inside a donchian branch it is."""
    assert "donchian_period_entry" not in _allowed_for(_ALL_STRATS)
    assert "donchian_period_entry" in _allowed_for(("donchian-v3",))
    assert "st_period" not in _allowed_for(("multifactor-v1",))


# ---------------------------------------------------------------------------
# bot.py wiring: --check-config and the boot refusal
# ---------------------------------------------------------------------------

def _run_main(argv, monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["bot", *argv,
                                     "--state-db", str(tmp_path / "s.db"),
                                     "--log-file", str(tmp_path / "bot.jsonl"),
                                     "--heartbeat", str(tmp_path / "hb")])
    for g in ("LOG_FILE", "HEARTBEAT", "CONFIG_PATH"):
        monkeypatch.setattr(bot, g, getattr(bot, g))
    monkeypatch.setattr(bot, "load_env_for_instance", lambda inst: None)
    monkeypatch.setattr(bot.signal, "signal", lambda *a: None)
    with patch.object(bot, "Bot") as bot_cls, \
         patch.object(bot, "send_alert", return_value=True) as alert:
        rc = bot.main()
    return rc, bot_cls, alert


def test_check_config_ok(monkeypatch, tmp_path, capsys):
    rc, bot_cls, alert = _run_main(
        ["--check-config", "--config", str(FIXTURES / "params.yaml")], monkeypatch, tmp_path)
    assert rc == 0
    assert "CONFIG OK" in capsys.readouterr().out
    bot_cls.assert_not_called()
    alert.assert_not_called()


@pytest.mark.parametrize("name", ["params.example.yaml", "does-not-exist.yaml"])
def test_check_config_refuses(monkeypatch, tmp_path, capsys, name):
    rc, bot_cls, _ = _run_main(
        ["--check-config", "--config", str(CONFIG / name)], monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    assert "CONFIG REFUSED" in capsys.readouterr().err
    bot_cls.assert_not_called()


def test_boot_refuses_incomplete_config_and_alerts(monkeypatch, tmp_path):
    params = _fixture("params.yaml")
    _set(params, "sizing.risk_per_trade_pct", None)
    cfg = tmp_path / "params.yaml"
    cfg.write_text(yaml.safe_dump(params))
    rc, bot_cls, alert = _run_main(["--config", str(cfg)], monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    bot_cls.assert_not_called()                     # no exchange client ever built
    alert.assert_called_once()
    assert "sizing.risk_per_trade_pct" in alert.call_args.args[1]


def test_boot_refuses_missing_file(monkeypatch, tmp_path):
    rc, bot_cls, alert = _run_main(["--config", str(tmp_path / "nope.yaml")],
                                   monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    bot_cls.assert_not_called()
    alert.assert_called_once()


def test_boot_with_complete_config_constructs_bot(monkeypatch, tmp_path):
    rc, bot_cls, alert = _run_main(["--config", str(FIXTURES / "params.yaml")],
                                   monkeypatch, tmp_path)
    bot_cls.assert_called_once()
    assert bot_cls.call_args.args[0] == _fixture("params.yaml")
    alert.assert_not_called()


def test_units_do_not_restart_on_config_refusal():
    for unit in (ROOT / "deploy").glob("*.service"):
        text = unit.read_text()
        if "Restart=" in text:
            assert f"RestartPreventExitStatus={EXIT_CONFIG}" in text, unit.name



@pytest.mark.parametrize("payload", [b"\xff\xfe\x00bad utf8", b"a: [unclosed\n", b"- just\n- a list\n"])
def test_any_load_failure_refuses_with_78_never_a_traceback(monkeypatch, tmp_path, payload):
    """Bad UTF-8 is a ValueError, not OSError/YAMLError: it must still exit 78
    (RestartPreventExitStatus) with one alert, on both paths."""
    cfg = tmp_path / "params.yaml"
    cfg.write_bytes(payload)
    rc, bot_cls, alert = _run_main(["--config", str(cfg)], monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    bot_cls.assert_not_called()
    alert.assert_called_once()
    rc, bot_cls, alert = _run_main(["--check-config", "--config", str(cfg)],
                                   monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    bot_cls.assert_not_called()


def test_unexpected_validator_exception_still_refuses_with_78(monkeypatch, tmp_path):
    with patch.object(bot, "validate_params", side_effect=RuntimeError("boom")):
        rc, bot_cls, alert = _run_main(["--config", str(FIXTURES / "params.yaml")],
                                       monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    bot_cls.assert_not_called()
    assert "boom" in alert.call_args.args[1]


# ---------------------------------------------------------------------------
# StrategyParams.from_yaml (research loader)
# ---------------------------------------------------------------------------

def test_from_yaml_missing_file_points_at_readme(tmp_path):
    from strategy.signals import StrategyParams
    with pytest.raises(FileNotFoundError, match="config/README.md"):
        StrategyParams.from_yaml(str(tmp_path / "params.yaml"))


@pytest.mark.parametrize("name", ["params.example.yaml", "params_donchian.example.yaml"])
def test_from_yaml_refuses_unfilled_template(name):
    """A null must never be silently replaced by a research default."""
    from strategy.signals import StrategyParams
    with pytest.raises(ValueError, match="null"):
        StrategyParams.from_yaml(str(CONFIG / name))


def test_from_yaml_null_key_refuses_absent_key_is_unset_and_required(tmp_path):
    from strategy.signals import StrategyParams, require_params
    params = _fixture("params_donchian.yaml")
    params["strategy"]["atr_period"] = None
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(params))
    with pytest.raises(ValueError, match=r"strategy\.atr_period"):
        StrategyParams.from_yaml(str(p))
    params["strategy"].pop("atr_period")
    p.write_text(yaml.safe_dump(params))
    loaded = StrategyParams.from_yaml(str(p))
    assert loaded.atr_period is None          # no code default is substituted
    with pytest.raises(ValueError, match=r"strategy\.atr_period.*config/README.md"):
        require_params(loaded, ("atr_period",), "test")


# ---------------------------------------------------------------------------
# Every pre-trading boot failure exits 78 (RestartPreventExitStatus) + alerts
# ---------------------------------------------------------------------------

def test_missing_instance_env_refuses_with_78(monkeypatch, tmp_path):
    rc, bot_cls, alert = _run_main(
        ["--instance", "donchian", "--config", str(FIXTURES / "params_donchian.yaml")],
        monkeypatch, tmp_path)                       # _run_main: no .env.<instance>
    assert rc == EXIT_CONFIG
    bot_cls.assert_not_called()
    alert.assert_called_once()
    assert ".env.donchian" in alert.call_args.args[1]


def test_boot_setup_failure_refuses_with_78(monkeypatch, tmp_path):
    with patch.object(bot.state, "set_db_path", side_effect=PermissionError("ro fs")):
        rc, bot_cls, alert = _run_main(["--config", str(FIXTURES / "params.yaml")],
                                       monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    bot_cls.assert_not_called()
    assert "ro fs" in alert.call_args.args[1]


def test_logging_setup_failure_refuses_with_78(monkeypatch, tmp_path):
    with patch.object(bot, "_setup_logging", side_effect=OSError("disk full")):
        rc, bot_cls, alert = _run_main(["--config", str(FIXTURES / "params.yaml")],
                                       monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    bot_cls.assert_not_called()
    assert "disk full" in alert.call_args.args[1]


# ---------------------------------------------------------------------------
# A lazily-imported strategy evaluator that is missing refuses at BOOT (78),
# not on the first bar inside the loop's generic error backoff
# ---------------------------------------------------------------------------

def _v3all_cfg(tmp_path) -> Path:
    params = _fixture("params.yaml")
    params["strategy_name"] = "v3-all-wider-4"
    cfg = tmp_path / "params_v3all.yaml"
    cfg.write_text(yaml.safe_dump(params))
    return cfg


def test_boot_refuses_missing_optional_evaluator_with_78(monkeypatch, tmp_path):
    import sys
    cfg = _v3all_cfg(tmp_path)
    monkeypatch.setitem(sys.modules, "strategy.live_v3all_wider4", None)  # module absent
    rc, bot_cls, alert = _run_main(["--config", str(cfg)], monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    bot_cls.assert_not_called()
    alert.assert_called_once()
    assert "strategy.live_v3all_wider4" in alert.call_args.args[1]


def test_check_config_refuses_missing_optional_evaluator(monkeypatch, tmp_path, capsys):
    import sys
    cfg = _v3all_cfg(tmp_path)
    monkeypatch.setitem(sys.modules, "strategy.live_v3all_wider4", None)
    rc, bot_cls, _ = _run_main(["--check-config", "--config", str(cfg)], monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    assert "strategy.live_v3all_wider4" in capsys.readouterr().err
    bot_cls.assert_not_called()


def test_live_strategies_boot_probe_is_a_noop(monkeypatch):
    """The live legs' evaluators are top-level imports: the probe imports nothing."""
    import sys

    from bot_internals import require_strategy_available
    for m in ("strategy.live_v3all_wider4", "strategy.live_cnh_hybrid_short"):
        monkeypatch.setitem(sys.modules, m, None)
    for name in ("multifactor-v1", "donchian-v3", "supertrend"):
        require_strategy_available(name)


def test_boot_with_optional_evaluator_present_constructs_bot(monkeypatch, tmp_path):
    import importlib.util
    if importlib.util.find_spec("strategy.live_v3all_wider4") is None:
        if os.environ.get("SNAPBACK_PUBLIC_SNAPSHOT") == "1":
            pytest.skip("public snapshot: strategy.live_v3all_wider4 is not shipped")
        pytest.fail("strategy.live_v3all_wider4 missing outside the public snapshot")
    rc, bot_cls, alert = _run_main(["--config", str(_v3all_cfg(tmp_path))], monkeypatch, tmp_path)
    bot_cls.assert_called_once()
    alert.assert_not_called()
