"""
Tests for the env + lockfile gate. These must pass before any other phase.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import exchange.env as env_mod
from exchange.env import (
    EnvError,
    LOCKFILE,
    get_env,
    halt_source,
    is_halted,
    leg_halt_path,
    load_env_for_instance,
)


def test_default_is_testnet(monkeypatch):
    monkeypatch.delenv("BINANCE_ENV", raising=False)
    assert get_env() == "testnet"


def test_explicit_testnet(monkeypatch):
    monkeypatch.setenv("BINANCE_ENV", "testnet")
    assert get_env() == "testnet"


def test_invalid_env_rejected(monkeypatch):
    monkeypatch.setenv("BINANCE_ENV", "production")
    with pytest.raises(EnvError):
        get_env()


def test_mainnet_without_lockfile_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("BINANCE_ENV", "mainnet")
    # Ensure no lockfile.
    if LOCKFILE.exists():
        pytest.skip("lockfile already exists in checkout; manual cleanup needed")
    with pytest.raises(EnvError, match="lockfile missing"):
        get_env()


def test_mainnet_with_lockfile_allowed(monkeypatch):
    monkeypatch.setenv("BINANCE_ENV", "mainnet")
    created = False
    if not LOCKFILE.exists():
        LOCKFILE.touch()
        created = True
    try:
        assert get_env() == "mainnet"
    finally:
        if created:
            LOCKFILE.unlink()


# --------------------------------------------------------------------------
# Part A: per-leg HALT isolation. A leg's self-halt (data/HALT_<instance>)
# must NEVER be able to halt a sibling leg; the shared data/HALT is a global
# stop-all that halts every leg.
# --------------------------------------------------------------------------

def _isolate_repo_root(monkeypatch, tmp_path):
    """Point env's REPO_ROOT at a tmp dir so HALT-file tests never touch the
    real repo's data/ directory. The HALT helpers read REPO_ROOT live."""
    monkeypatch.setattr(env_mod, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_no_halt_files_not_halted(monkeypatch, tmp_path):
    _isolate_repo_root(monkeypatch, tmp_path)
    assert is_halted("donchian") is False
    assert halt_source("donchian") is None


def test_leg_self_halt_only_halts_that_leg(monkeypatch, tmp_path):
    root = _isolate_repo_root(monkeypatch, tmp_path)
    # cnh_short self-halts (its own kill switch).
    leg_halt_path("cnh_short").touch()
    assert is_halted("cnh_short") is True
    assert halt_source("cnh_short") == "cnh_short"
    # Siblings are NOT affected — this is the 07-01 cascade fix.
    assert is_halted("v1") is False
    assert is_halted("donchian") is False
    # And the shared global HALT was NOT written.
    assert not (root / "data" / "HALT").exists()


def test_global_halt_halts_every_leg(monkeypatch, tmp_path):
    _isolate_repo_root(monkeypatch, tmp_path)
    env_mod.global_halt_path().touch()
    for leg in ("v1", "donchian", "cnh_short"):
        assert is_halted(leg) is True
        assert halt_source(leg) == "global"
    # Global takes precedence in the reason even without an instance arg.
    assert is_halted() is True
    assert halt_source() == "global"


def test_global_reported_over_self(monkeypatch, tmp_path):
    _isolate_repo_root(monkeypatch, tmp_path)
    env_mod.global_halt_path().touch()
    leg_halt_path("donchian").touch()
    # Both present → global wins the label.
    assert halt_source("donchian") == "global"


# --------------------------------------------------------------------------
# Part A: fail-loud per-instance env. A sub-account leg missing its
# .env.{instance} must raise (never silently inherit the base main-account
# keys — the donchian wrong-account root cause). v1 (base account,
# no .env.v1) must still boot.
# --------------------------------------------------------------------------

def test_load_env_missing_required_subaccount_raises(monkeypatch, tmp_path):
    # tmp repo root has NO .env.donchian.
    monkeypatch.setattr(env_mod, "REPO_ROOT", tmp_path)
    with pytest.raises(EnvError, match="sub-account leg"):
        load_env_for_instance("donchian")


def test_load_env_v1_base_account_no_file_ok(monkeypatch, tmp_path):
    # v1 has no .env.v1 on disk (confirmed on the droplet). It must NOT crash —
    # the base .env is legitimately its environment.
    monkeypatch.setattr(env_mod, "REPO_ROOT", tmp_path)
    assert load_env_for_instance("v1") is None


def test_load_env_subaccount_present_loads(monkeypatch, tmp_path):
    monkeypatch.setattr(env_mod, "REPO_ROOT", tmp_path)
    envfile = tmp_path / ".env.donchian"
    envfile.write_text("SNAPBACK_TEST_MARKER=donchian_loaded\n")
    try:
        loaded = load_env_for_instance("donchian")
        assert loaded == envfile
        assert os.environ.get("SNAPBACK_TEST_MARKER") == "donchian_loaded"
    finally:
        os.environ.pop("SNAPBACK_TEST_MARKER", None)
