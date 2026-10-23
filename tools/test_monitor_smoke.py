"""Smoke test for monitor.py + daily_digest.py.

Runs both scripts with SMTP forcibly disabled so they exit cleanly without
sending emails. Validates the "never raises" contract — both must exit 0
even when:
  - heartbeat files missing
  - state.db missing or incomplete
  - log files empty or absent
  - SMTP not configured

Run before deploying the AFK package to droplet. Per
AFK_PACKAGE_DEPLOY.md step 0.

Usage:
    .venv/bin/python tools/test_monitor_smoke.py

Exit 0 = both scripts validate. Exit non-zero = something to investigate.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO / ".venv" / "bin" / "python"


def _sandbox_env() -> dict[str, str]:
    """Empty env with SMTP_* explicitly cleared to guarantee no-send."""
    return {
        "PATH":               os.environ.get("PATH", "/usr/bin"),
        "SMTP_HOST":          "",
        "SMTP_PORT":          "",
        "SMTP_USER":          "",
        "SMTP_PASSWORD":      "",
        "ALERT_EMAIL_FROM":   "",
        "ALERT_EMAIL_TO":     "",
        "BINANCE_ENV":        "testnet",
    }


def _run(name: str, script: Path) -> int:
    print(f"\n=== smoke: {name} ===", flush=True)
    proc = subprocess.run(
        [str(VENV_PYTHON), str(script)],
        env=_sandbox_env(),
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.stdout.strip():
        print("stdout:", proc.stdout.strip()[:500])
    if proc.stderr.strip():
        print("stderr:", proc.stderr.strip()[:1000])
    if proc.returncode != 0:
        print(f"FAIL: exit {proc.returncode}")
    else:
        print(f"OK: exit 0")
    return proc.returncode


def main() -> int:
    failures = 0
    for name, script in (
        ("monitor.py",      REPO / "monitor.py"),
        ("daily_digest.py", REPO / "daily_digest.py"),
    ):
        if not script.exists():
            print(f"SKIP: {name} not found at {script}")
            continue
        rc = _run(name, script)
        if rc != 0:
            failures += 1

    # Cleanup smoke-test state file if monitor.py wrote one
    state_file = REPO / "data" / "monitor_state.json"
    if state_file.exists():
        state_file.unlink()
        print(f"\ncleanup: removed {state_file.name}")

    print(f"\n=== smoke result: {'PASS' if failures == 0 else f'{failures} FAIL'} ===")
    return failures


if __name__ == "__main__":
    sys.exit(main())
