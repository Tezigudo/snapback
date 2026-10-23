"""RISK_REVIEW 2026-10-22 parity: moving the hard ceilings out of risk.py into the
params `risk:` block must not change a single value, type or decision.

Old side: the pre-change risk.py straight out of git (PRE_CHANGE_SHA), executed
as a throwaway module. New side: today's risk.py configured from a params dict
carrying the same values — a dict built from the old constants, the last
TRACKED per-leg YAMLs (pinned history) after tools/migrate_risk_params.py, and a
synthetic YAML migrated by the tool's CLI. The DEPLOYED files are deliberately
not pinned to these values: a later reviewed change there is legitimate (and is
surfaced by the boot-time fingerprint alert, bot._note_risk_ceilings_change).
Every field must match in value AND type, and every enforcement call site
(check_notional / check_leverage / check_symbol / check_daily_loss) must give
the identical outcome — pass, or RiskBreach with the identical message — on a
grid around each ceiling.

Missing pinned history FAILS unless SNAPBACK_PUBLIC_SNAPSHOT=1 (the public
snapshot sets it, or drops this file).

No live number appears in this file: every expected value is read from git.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

import bot
import risk
from config_check import EXIT_CONFIG, REQUIRED_KEYS, config_problems
from tools import migrate_risk_params as mig

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
BASE = os.environ.get("SNAPBACK_RISK_PARITY_BASE", mig.PRE_CHANGE_SHA)
LIVE_REF = "b9cb2d072633cc855fec06d3ebc0e74326d4befd"  # last tracked live YAMLs


def _git_show(ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _missing(what: str):
    if os.environ.get("SNAPBACK_PUBLIC_SNAPSHOT") == "1":
        pytest.skip(f"public snapshot: {what}")
    pytest.fail(f"{what} — parity proof needs the pinned history "
                "(set SNAPBACK_PUBLIC_SNAPSHOT=1 only in the public snapshot)")


@pytest.fixture(scope="module")
def old_risk() -> types.ModuleType:
    src = _git_show(BASE, "risk.py")
    if src is None:
        _missing(f"pre-change {BASE}:risk.py not in this repo's history")
    mod = types.ModuleType("_pre_risk_ext_risk")
    mod.__file__ = f"<{BASE}:risk.py>"
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def _old_values(old_risk) -> dict:
    return {key: getattr(old_risk.CEILINGS, field)
            for field, (key, _t) in risk.RISK_FIELDS.items()}


def _outcome(fn, *args):
    try:
        fn(*args)
    except Exception as e:
        return (type(e).__name__, str(e))
    return None


def _assert_same_ceilings(old, new) -> None:
    old_fields = {f.name for f in dataclasses.fields(old)}
    new_fields = {f.name for f in dataclasses.fields(new)}
    assert old_fields == new_fields
    for name in sorted(old_fields):
        a, b = getattr(old, name), getattr(new, name)
        assert a == b and type(a) is type(b), (name, a, b)


def _grid_parity(old_risk, new_ceilings, monkeypatch) -> None:
    monkeypatch.setattr(risk, "_ACTIVE", new_ceilings)
    c = old_risk.CEILINGS
    n, lev, dl = c.MAX_NOTIONAL_USD, c.MAX_LEVERAGE, c.MAX_DAILY_LOSS_PCT
    notionals = [0.0, 1.0, n / 2, n - 1e-6, n, n + 1e-6, n + 0.01, n * 2, 1e9]
    for x in notionals:
        assert _outcome(old_risk.check_notional, x) == _outcome(risk.check_notional, x), x
    for x in range(0, lev + 6):
        assert _outcome(old_risk.check_leverage, x) == _outcome(risk.check_leverage, x), x
    for s in ("BTC/USDT:USDT", "SOL/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT", ""):
        assert _outcome(old_risk.check_symbol, s) == _outcome(risk.check_symbol, s), s
    for start in (-1.0, 0.0, 1000.0, 123.45):
        for loss in (-5.0, 0.0, dl / 2, dl - 1e-9, dl - 0.01, dl, dl + 1e-9, dl + 0.01,
                     dl * 2, 100.0):
            now = start * (1 - loss / 100.0)
            assert (_outcome(old_risk.check_daily_loss, now, start)
                    == _outcome(risk.check_daily_loss, now, start)), (start, loss)
    # The one non-check reader: bot's breaker alert/event threshold.
    assert dl == risk.ceilings().MAX_DAILY_LOSS_PCT
    assert f"{risk.ceilings().MAX_DAILY_LOSS_PCT:.0f}" == f"{dl:.0f}"


def test_values_dict_builds_identical_ceilings(old_risk, monkeypatch):
    new = risk.ceilings_from_params({"risk": _old_values(old_risk)})
    _assert_same_ceilings(old_risk.CEILINGS, new)
    _grid_parity(old_risk, new, monkeypatch)


def test_migrate_tool_reads_the_same_values_from_git(old_risk):
    assert mig.values_from_git(BASE) == _old_values(old_risk)
    for k, v in mig.values_from_git(BASE).items():
        assert type(v) is type(_old_values(old_risk)[k]), k


_LEG_FILES = sorted({p["config"].name for p in bot.INSTANCE_PROFILES.values()})


@pytest.mark.parametrize("name", _LEG_FILES)
def test_migrated_live_yaml_gives_identical_ceilings(name, old_risk, monkeypatch):
    """The droplet procedure end to end, on the last tracked copy of each leg."""
    src = _git_show(LIVE_REF, f"config/{name}")
    if src is None:
        _missing(f"{LIVE_REF}:config/{name} not in history")
    migrated = mig.migrate_text(src, mig.values_from_git(BASE), "test")
    params = yaml.safe_load(migrated)
    assert config_problems(params) == []
    new = risk.ceilings_from_params(params)
    _assert_same_ceilings(old_risk.CEILINGS, new)
    _grid_parity(old_risk, new, monkeypatch)
    # Nothing but the risk block changed.
    params.pop("risk")
    assert params == yaml.safe_load(src)


def test_migrate_tool_from_pinned_git_on_synthetic_yaml(old_risk, tmp_path):
    """The migration proof, independent of any deployed file (which may later
    carry a legitimately reviewed change): the tool, run with --from-git <full
    pinned sha> on a synthetic un-migrated YAML, writes a `risk:` block equal in
    value and type to the pinned risk.py constants."""
    p = _unmigrated(tmp_path)
    assert mig.main([str(p), "--from-git", BASE, "--apply"]) == 0
    params = yaml.safe_load(p.read_text())
    assert config_problems(params) == []
    _assert_same_ceilings(old_risk.CEILINGS, risk.ceilings_from_params(params))


# ---------------------------------------------------------------------------
# No defaults: unconfigured / missing / bad values refuse
# ---------------------------------------------------------------------------

def test_dataclass_has_no_numeric_defaults():
    c = risk.RiskCeilings()
    for field in risk.RISK_FIELDS:
        assert getattr(c, field) is None, field


@pytest.mark.parametrize("call", [
    lambda: risk.check_notional(1.0), lambda: risk.check_leverage(1),
    lambda: risk.check_symbol("BTC/USDT:USDT"), lambda: risk.check_daily_loss(1.0, 1.0),
])
def test_unconfigured_is_a_crash_not_a_breach(call, monkeypatch):
    monkeypatch.setattr(risk, "_ACTIVE", None)
    with pytest.raises(risk.RiskNotConfigured):
        call()
    assert not issubclass(risk.RiskNotConfigured, risk.RiskBreach)


def test_risk_keys_required_for_every_strategy():
    for name, keys in REQUIRED_KEYS.items():
        assert set(risk.RISK_KEYS) <= set(keys), name


def _fixture(name="params.yaml"):
    return yaml.safe_load((FIXTURES / name).read_text())


@pytest.mark.parametrize("key", risk.RISK_KEYS)
@pytest.mark.parametrize("bad", ["missing", None, 0, -1, float("nan"), float("inf"),
                                 True, "5", 10**9])
def test_bad_risk_value_refuses(key, bad):
    params = _fixture()
    leaf = key.split(".")[1]
    if bad == "missing":
        del params["risk"][leaf]
    else:
        params["risk"][leaf] = bad
    problems = config_problems(params)
    assert len(problems) == 1 and problems[0].startswith(f"{key}:"), problems
    if bad == "missing" or bad is None or isinstance(bad, (bool, str)):
        with pytest.raises(ValueError):   # the loader itself never defaults either
            risk.ceilings_from_params(params)


@pytest.mark.parametrize("field", [f for f, (_k, t) in risk.RISK_FIELDS.items() if t is int])
def test_int_ceilings_refuse_fractions(field):
    params = _fixture()
    key = risk.RISK_FIELDS[field][0]
    params["risk"][key] = params["risk"][key] + 0.5
    assert any(p.startswith(f"risk.{key}:") for p in config_problems(params))
    with pytest.raises(ValueError):
        risk.ceilings_from_params(params)


def test_missing_risk_block_refuses():
    params = _fixture()
    del params["risk"]
    assert config_problems(params) == [f"{k}: missing" for k in risk.RISK_KEYS]


def test_types_are_coerced_like_the_old_constants():
    params = _fixture()
    params["risk"]["max_notional_usd"] = 750      # YAML int -> float field
    params["risk"]["max_leverage"] = 10.0         # YAML float -> int field
    c = risk.ceilings_from_params(params)
    assert type(c.MAX_NOTIONAL_USD) is float and type(c.MAX_LEVERAGE) is int


# ---------------------------------------------------------------------------
# main() installs the leg's ceilings; --check-config covers the loader
# ---------------------------------------------------------------------------

def _run_main(argv, monkeypatch, tmp_path, alert_ok=True):
    from unittest.mock import patch
    monkeypatch.setattr("sys.argv", ["bot", *argv,
                                     "--state-db", str(tmp_path / "s.db"),
                                     "--log-file", str(tmp_path / "bot.jsonl"),
                                     "--heartbeat", str(tmp_path / "hb")])
    for g in ("LOG_FILE", "HEARTBEAT", "CONFIG_PATH"):
        monkeypatch.setattr(bot, g, getattr(bot, g))
    monkeypatch.setattr(bot, "load_env_for_instance", lambda inst: None)
    monkeypatch.setattr(bot.signal, "signal", lambda *a: None)
    with patch.object(bot, "Bot") as bot_cls, \
         patch.object(bot, "send_alert", return_value=alert_ok) as alert:
        rc = bot.main()
    return rc, bot_cls, alert


def test_main_configures_ceilings_from_the_leg_params(monkeypatch, tmp_path):
    params = _fixture()
    params["risk"]["max_notional_usd"] = 321.0     # distinct from the conftest seed
    cfg = tmp_path / "params.yaml"
    cfg.write_text(yaml.safe_dump(params))
    monkeypatch.setattr(risk, "_ACTIVE", None)
    _rc, bot_cls, _alert = _run_main(["--config", str(cfg)], monkeypatch, tmp_path)
    bot_cls.assert_called_once()
    assert risk.ceilings() == risk.ceilings_from_params(params)
    assert risk.ceilings().MAX_NOTIONAL_USD == 321.0


def test_main_refuses_without_risk_block(monkeypatch, tmp_path):
    params = _fixture()
    del params["risk"]
    cfg = tmp_path / "params.yaml"
    cfg.write_text(yaml.safe_dump(params))
    rc, bot_cls, alert = _run_main(["--config", str(cfg)], monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    bot_cls.assert_not_called()
    alert.assert_called_once()


def test_check_config_refuses_out_of_range_ceiling(monkeypatch, tmp_path, capsys):
    params = _fixture()
    params["risk"]["max_daily_loss_pct"] = 0
    cfg = tmp_path / "params.yaml"
    cfg.write_text(yaml.safe_dump(params))
    rc, _b, _a = _run_main(["--check-config", "--config", str(cfg)], monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    assert "risk.max_daily_loss_pct" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# tools/migrate_risk_params.py
# ---------------------------------------------------------------------------

_SYNTH = {"max_notional_usd": 750.0, "max_leverage": 10, "max_daily_loss_pct": 2.75,
          "max_open_positions": 1, "max_orders_per_minute": 9,
          "max_consecutive_losses": 5, "min_time_between_trades_s": 45}


def _unmigrated(tmp_path) -> Path:
    text = (FIXTURES / "params.yaml").read_text()
    text = text[: text.index("\nrisk:")] + "\n"
    p = tmp_path / "params.yaml"
    p.write_text("# operator comment survives\n" + text)
    return p


def _values_file(tmp_path, values=_SYNTH) -> Path:
    v = tmp_path / "vals.yaml"
    v.write_text(yaml.safe_dump({"risk": values}))
    return v


def test_migrate_dry_run_writes_nothing(tmp_path, capsys):
    p = _unmigrated(tmp_path)
    before = p.read_text()
    assert mig.main([str(p), "--values-file", str(_values_file(tmp_path))]) == 0
    assert p.read_text() == before
    out = capsys.readouterr().out
    assert "+risk:" in out and "DRY RUN" in out
    assert not list(tmp_path.glob("*.bak"))


def test_migrate_apply_backup_idempotent(tmp_path, capsys):
    p = _unmigrated(tmp_path)
    before = p.read_text()
    vals = _values_file(tmp_path)
    assert mig.main([str(p), "--values-file", str(vals), "--apply"]) == 0
    baks = list(tmp_path.glob("params.yaml.*.bak"))
    assert len(baks) == 1 and baks[0].read_text() == before
    after = p.read_text()
    assert after.startswith(before) and "# operator comment survives" in after
    params = yaml.safe_load(after)
    assert config_problems(params) == []
    assert risk.ceilings_from_params(params) == risk.ceilings_from_params({"risk": _SYNTH})
    capsys.readouterr()
    assert mig.main([str(p), "--values-file", str(vals), "--apply"]) == 0
    assert "already migrated" in capsys.readouterr().out
    assert p.read_text() == after


def test_migrate_refuses_to_overwrite_a_different_ceiling(tmp_path):
    p = _unmigrated(tmp_path)
    mig.main([str(p), "--values-file", str(_values_file(tmp_path)), "--apply"])
    after = p.read_text()
    other = _values_file(tmp_path, {**_SYNTH, "max_leverage": 11})
    assert mig.main([str(p), "--values-file", str(other), "--apply"]) == 1
    assert p.read_text() == after


def test_migrate_refuses_post_change_source():
    """Pointing --from-git at a commit whose risk.py has no numbers must refuse,
    never write nulls."""
    with pytest.raises(mig.MigrateError):
        mig.values_from_risk_source((ROOT / "risk.py").read_text())


def test_migrate_backups_are_gitignored():
    for name in ("params.yaml", "params_donchian.yaml"):
        r = subprocess.run(["git", "check-ignore", "-q",
                            f"config/{name}.20261022T000000Z.bak"], cwd=ROOT)
        assert r.returncode == 0, name


def test_no_live_number_literals_in_risk_py():
    """risk.py carries structure only: no numeric literal in RiskCeilings."""
    import ast
    tree = ast.parse((ROOT / "risk.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RiskCeilings")
    for node in ast.walk(cls):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            pytest.fail(f"numeric literal {node.value!r} in RiskCeilings")


def test_migrate_refuses_non_full_sha(tmp_path):
    p = _unmigrated(tmp_path)
    before = p.read_text()
    for ref in ("droplet", "HEAD", BASE[:7], BASE.upper(), BASE + "0"):
        with pytest.raises(mig.MigrateError, match="full 40-hex"):
            mig.values_from_git(ref)
        assert mig.main([str(p), "--from-git", ref, "--apply"]) == 1
    assert p.read_text() == before


def test_migrate_temp_file_is_gitignored_and_never_left(tmp_path, monkeypatch):
    r = subprocess.run(["git", "check-ignore", "-q", "config/.params.yaml.migrate.tmp"],
                       cwd=ROOT)
    assert r.returncode == 0
    p = _unmigrated(tmp_path)
    assert mig.main([str(p), "--values-file", str(_values_file(tmp_path)), "--apply"]) == 0
    assert not list(tmp_path.glob(".*.migrate.tmp"))
    # A failure at the rename leaves the original untouched and no temp file.
    q = tmp_path / "q"
    q.mkdir()
    p2 = _unmigrated(q)
    before = p2.read_text()

    def boom(*a, **k):
        raise OSError("simulated crash at rename")
    monkeypatch.setattr(mig.os, "replace", boom)
    with pytest.raises(OSError):
        mig._apply(p2, before + "\nrisk: {}\n")
    assert p2.read_text() == before
    assert not list(q.glob(".*.migrate.tmp"))


# ---------------------------------------------------------------------------
# Cross-checks against the leg's own config
# ---------------------------------------------------------------------------

def test_max_leverage_below_leg_leverage_refuses():
    params = _fixture()
    params["risk"]["max_leverage"] = params["sizing"]["leverage"] - 1
    assert config_problems(params) == [
        f"risk.max_leverage: below this leg's sizing.leverage "
        f"({params['risk']['max_leverage']!r} < {params['sizing']['leverage']!r})"]
    params["risk"]["max_leverage"] = params["sizing"]["leverage"]   # equal is fine
    assert config_problems(params) == []


@pytest.mark.parametrize("name", ["params.yaml", "params_sol_supertrend.yaml"])
def test_max_notional_below_one_minimum_order_refuses(name):
    from exchange.constraints import fallbacks_for_symbol
    params = _fixture(name)
    floor = fallbacks_for_symbol(params["symbol"]).min_notional_usdt
    params["risk"]["max_notional_usd"] = floor - 0.01
    problems = config_problems(params)
    assert len(problems) == 1 and problems[0].startswith(
        "risk.max_notional_usd: below the static fallback minimum order"), problems
    params["risk"]["max_notional_usd"] = floor
    assert config_problems(params) == []


def test_cross_checks_skip_keys_already_refused():
    params = _fixture()
    params["risk"]["max_leverage"] = None
    assert config_problems(params) == ["risk.max_leverage: null"]


def test_check_config_refuses_leverage_above_ceiling(monkeypatch, tmp_path, capsys):
    params = _fixture()
    params["sizing"]["leverage"] = params["risk"]["max_leverage"] + 1
    cfg = tmp_path / "params.yaml"
    cfg.write_text(yaml.safe_dump(params))
    rc, _b, _a = _run_main(["--check-config", "--config", str(cfg)], monkeypatch, tmp_path)
    assert rc == EXIT_CONFIG
    assert "risk.max_leverage: below this leg's sizing.leverage" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Boot-time fingerprint: compensating control for the lost RISK_REVIEW hook
# ---------------------------------------------------------------------------

def _boot_with(params, monkeypatch, tmp_path, alert_ok=True):
    cfg = tmp_path / "params.yaml"
    cfg.write_text(yaml.safe_dump(params))
    return _run_main(["--config", str(cfg)], monkeypatch, tmp_path, alert_ok=alert_ok)


def _bumped(params):
    changed = yaml.safe_load(yaml.safe_dump(params))
    changed["risk"]["max_daily_loss_pct"] = params["risk"]["max_daily_loss_pct"] + 1
    return changed


def test_fingerprint_undelivered_alert_keeps_old_and_realerts(monkeypatch, tmp_path, caplog):
    from exchange import state
    params = _fixture()
    _boot_with(params, monkeypatch, tmp_path)
    fp0 = state.get_meta(bot.RISK_FP_META_KEY)
    with caplog.at_level("ERROR"):
        _rc, bot_cls, alert = _boot_with(_bumped(params), monkeypatch, tmp_path,
                                         alert_ok=False)
    bot_cls.assert_called_once()                       # still boots
    alert.assert_called_once()
    assert state.get_meta(bot.RISK_FP_META_KEY) == fp0  # not acknowledged
    assert any("NOT delivered" in r.getMessage() for r in caplog.records)
    _rc, _b, alert = _boot_with(_bumped(params), monkeypatch, tmp_path)  # delivered now
    alert.assert_called_once()
    assert "max_daily_loss_pct" in alert.call_args.args[1]
    assert state.get_meta(bot.RISK_FP_META_KEY) != fp0
    _rc, _b, alert = _boot_with(_bumped(params), monkeypatch, tmp_path)
    alert.assert_not_called()


def test_fingerprint_store_failure_after_alert_logs_error_and_repeats(
        monkeypatch, tmp_path, caplog):
    import sqlite3

    from exchange import state
    params = _fixture()
    _boot_with(params, monkeypatch, tmp_path)
    real_set = state.set_meta

    def failing_set(key, value, *a, **k):
        if key in (bot.RISK_FP_META_KEY, bot.RISK_VALUES_META_KEY):
            raise sqlite3.OperationalError("simulated")
        return real_set(key, value, *a, **k)
    monkeypatch.setattr(state, "set_meta", failing_set)
    with caplog.at_level("ERROR"):
        _rc, bot_cls, alert = _boot_with(_bumped(params), monkeypatch, tmp_path)
    bot_cls.assert_called_once()
    alert.assert_called_once()
    errs = [r for r in caplog.records if "fingerprint NOT stored" in r.getMessage()]
    assert len(errs) == 1 and errs[0].levelname == "ERROR"
    monkeypatch.setattr(state, "set_meta", real_set)
    _rc, _b, alert = _boot_with(_bumped(params), monkeypatch, tmp_path)
    alert.assert_called_once()                         # repeat beats silence


def test_fingerprint_first_boot_stores_silently_then_quiet(monkeypatch, tmp_path):
    from exchange import state
    params = _fixture()
    _rc, bot_cls, alert = _boot_with(params, monkeypatch, tmp_path)
    bot_cls.assert_called_once()
    alert.assert_not_called()
    assert state.get_meta(bot.RISK_FP_META_KEY)
    _rc, _b, alert = _boot_with(params, monkeypatch, tmp_path)   # unchanged
    alert.assert_not_called()


def test_fingerprint_change_alerts_once_with_changed_keys(monkeypatch, tmp_path, caplog):
    params = _fixture()
    _boot_with(params, monkeypatch, tmp_path)
    changed = yaml.safe_load(yaml.safe_dump(params))
    changed["risk"]["max_daily_loss_pct"] = params["risk"]["max_daily_loss_pct"] + 1
    with caplog.at_level("WARNING"):
        _rc, bot_cls, alert = _boot_with(changed, monkeypatch, tmp_path)
    bot_cls.assert_called_once()                       # never blocks boot
    alert.assert_called_once()
    subject, body = alert.call_args.args[:2]
    assert subject == "risk ceilings changed since last boot"
    assert "max_daily_loss_pct" in body and "max_leverage" not in body
    assert any("RISK CEILINGS CHANGED" in r.getMessage() for r in caplog.records)
    _rc, _b, alert = _boot_with(changed, monkeypatch, tmp_path)  # stored: no re-alert
    alert.assert_not_called()


def test_fingerprint_failure_never_blocks_boot(monkeypatch, tmp_path):
    from exchange import state

    import sqlite3

    def boom(*a, **k):
        raise sqlite3.OperationalError("simulated")
    monkeypatch.setattr(state, "get_meta", boom)
    _rc, bot_cls, alert = _boot_with(_fixture(), monkeypatch, tmp_path)
    bot_cls.assert_called_once()
    alert.assert_not_called()
