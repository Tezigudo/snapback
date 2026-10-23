"""Access to the REAL (untracked) per-leg params YAMLs, for deployed-config guards.

config/params*.yaml are gitignored (config/README.md), so a fresh clone has
none. A few tests deliberately assert a property of the DEPLOYED config —
asserting it of a fixture would prove nothing about production. Those tests read
the real file when it is present (operator machines, the droplet) and SKIP when
it is not. The boot-time guard that covers the absent case is config_check's
REQUIRED_KEYS (bot refuses to boot on a missing/null key).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def live_config(name: str) -> dict:
    """yaml of config/<name>, or pytest.skip if the untracked file is absent."""
    path = CONFIG_DIR / name
    if not path.is_file():
        pytest.skip(f"config/{name} not present (untracked live params)")
    return yaml.safe_load(path.read_text())
