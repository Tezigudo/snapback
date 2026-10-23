"""tools/deploy_untrack_params.sh end-to-end in a throwaway clone.

systemctl, crontab and tar are replaced by fakes (PATH shims / env overrides),
so nothing on this machine's cron or systemd is touched. The clone starts at the
last commit that tracks the params (b9cb2d0) and fast-forwards to the untrack
commit.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "deploy_untrack_params.sh"
START = "b9cb2d072633cc855fec06d3ebc0e74326d4befd"      # last commit tracking params
UNTRACK = "2a45e9ae6509b1421a8774b619b2ff75da1d0b4b"    # git rm --cached config/params*.yaml
PYTHON = Path(sys.executable)

FAKE_SYSTEMCTL = r"""#!/usr/bin/env bash
D="$FAKE_SD"; echo "$*" >> "$D/log"
case "$1" in
  list-units) [[ -f "$D/list_fail" ]] && exit 1
              [[ -f "$D/timers" ]] && sed 's/$/ loaded active waiting x/' "$D/timers"; exit 0 ;;
  is-active) s="$(cat "$D/active_$2" 2>/dev/null || echo inactive)"; echo "$s"; [[ $s == active ]] ;;
  stop) echo inactive > "$D/active_$2" ;;
  start) [[ -f "$D/start_fail" ]] && exit 1; echo active > "$D/active_$2" ;;
  daemon-reload) touch "$D/reloaded" ;;
  show)
    prop="$3"; u="${@: -1}"
    case "$prop" in
      Transient) cat "$D/transient_$u" 2>/dev/null || echo no ;;
      Unit) cat "$D/unit_$u" 2>/dev/null ;;
      ExecStart) cat "$D/exec_$u" 2>/dev/null ;;
      LoadState) [[ -f "$D/frag_$u" ]] && echo loaded || echo not-found ;;
      FragmentPath) cat "$D/frag_$u" ;;
      RestartPreventExitStatus)
        [[ -f "$D/reloaded" ]] || { echo; exit 0; }
        f="$(cat "$D/frag_$u")"
        cat "$f" "$FAKE_UNIT_DIR/$u.d/"*.conf 2>/dev/null \
          | sed -n 's/^RestartPreventExitStatus=//p' | tr '\n' ' ' ;;
    esac ;;
esac
"""

FAKE_CRONTAB = r"""#!/usr/bin/env bash
D="$FAKE_SD"
case "$1" in
  -l) [[ -f "$D/crontab" ]] && cat "$D/crontab" || { echo "no crontab for user" >&2; exit 1; } ;;
  -r) rm -f "$D/crontab" ;;
  *) [[ -f "$D/crontab_fail" ]] && exit 1; cp "$1" "$D/crontab" ;;
esac
"""

FAKE_TAR = r"""#!/usr/bin/env bash
tar "$@" || exit $?
case "$(cat "$FAKE_SD/tar_mode" 2>/dev/null)" in
  changed) echo "tar: ./data/state.db: file changed as we read it" >&2; exit 1 ;;
  broken)  echo "tar: ./config: Cannot open: Permission denied" >&2; exit 1 ;;
  empty)   exit 1 ;;
esac
"""


def _have_history() -> bool:
    return all(subprocess.run(["git", "cat-file", "-e", f"{c}^{{commit}}"], cwd=ROOT).returncode == 0
               for c in (START, UNTRACK))


pytestmark = pytest.mark.skipif(
    not shutil.which("bash") or (os.environ.get("SNAPBACK_PUBLIC_SNAPSHOT") == "1"
                                 and not _have_history()),
    reason="public snapshot: private history not available")


@pytest.fixture
def env(tmp_path):
    if not _have_history():
        pytest.fail("pinned history missing (set SNAPBACK_PUBLIC_SNAPSHOT=1 only in the public snapshot)")
    repo = tmp_path / "repo"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(ROOT), str(repo)], check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "droplet", START], cwd=repo, check=True)
    subprocess.run(["git", "update-ref", "refs/remotes/origin/target", UNTRACK], cwd=repo, check=True)
    (repo / ".env").write_text("FAKE=1\n")
    (repo / ".env.donchian").write_text("FAKE=2\n")
    sd = tmp_path / "sd"
    sd.mkdir()
    units = tmp_path / "units"
    units.mkdir()
    shim = tmp_path / "bin"
    shim.mkdir()
    for name, body in (("systemctl", FAKE_SYSTEMCTL), ("crontab", FAKE_CRONTAB),
                       ("faketar", FAKE_TAR)):
        p = shim / name
        p.write_text(body)
        p.chmod(0o755)
    # Installed units: v1 identical to the repo's pre-merge copy (gets replaced),
    # sol carries a local edit (gets a drop-in). cnh/donchian not installed.
    for u, edit in (("snapback-btc.service", ""), ("snapback-sol-supertrend.service", "# local\n")):
        f = units / u
        f.write_text(subprocess.run(["git", "show", f"{START}:deploy/{u}"], cwd=repo,
                                    capture_output=True, text=True, check=True).stdout + edit)
        (sd / f"frag_{u}").write_text(str(f))
    (sd / "timers").write_text("snapback-sol-be.timer\nunrelated.timer\n")
    (sd / "active_snapback-sol-be.timer").write_text("active\n")
    (sd / "active_unrelated.timer").write_text("active\n")
    (sd / "crontab").write_text("*/2 * * * * watchdog\n")
    e = dict(os.environ, REPO=str(repo), TARGET="origin/target", BRANCH="droplet", FETCH="0",
             ASIDE_ROOT=str(tmp_path / "aside"), PYTHON=str(PYTHON),
             SYSTEMCTL=str(shim / "systemctl"), UNIT_DIR=str(units), TAR=str(shim / "faketar"),
             FAKE_SD=str(sd), FAKE_UNIT_DIR=str(units), PATH=f"{shim}:{os.environ['PATH']}")
    (tmp_path / "aside").mkdir()
    return {"repo": repo, "sd": sd, "units": units, "env": e}


def _params_bytes(repo: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted((repo / "config").glob("params*.yaml"))
            if ".example." not in p.name} | {".env": (repo / ".env").read_bytes()}


def _run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env["env"],
                          capture_output=True, text=True, timeout=600)


def _head(repo):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True).stdout.strip()


def test_dry_run_changes_nothing(env):
    before = _params_bytes(env["repo"])
    r = _run(env)
    assert r.returncode == 0, r.stderr
    assert _head(env["repo"]) == START
    assert _params_bytes(env["repo"]) == before
    assert "stop" not in (env["sd"] / "log").read_text().split()


def test_apply_success(env):
    repo, sd = env["repo"], env["sd"]
    (sd / "tar_mode").write_text("changed")           # GNU tar exit 1 on live files
    before = _params_bytes(repo)
    r = _run(env, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _head(repo) == UNTRACK
    assert _params_bytes(repo) == before              # byte-identical restore
    assert subprocess.run(["git", "status", "--porcelain", "config/"], cwd=repo,
                          capture_output=True, text=True).stdout == ""
    log = (sd / "log").read_text()
    assert "stop snapback-sol-be.timer" in log and "start snapback-sol-be.timer" in log
    assert "stop unrelated.timer" not in log and "start unrelated.timer" not in log
    assert "daemon-reload" in log and "restart" not in log
    assert (sd / "crontab").read_text() == "*/2 * * * * watchdog\n"
    v1 = (env["units"] / "snapback-btc.service").read_text()
    assert "RestartPreventExitStatus=78" in v1                    # replaced unit
    sol = env["units"] / "snapback-sol-supertrend.service"
    assert sol.read_text().endswith("# local\n")                   # local edit kept
    assert "RestartPreventExitStatus=78" in (
        env["units"] / "snapback-sol-supertrend.service.d" / "50-restart-prevent-config.conf").read_text()


def test_tar_real_error_aborts_before_merge_and_restores_cron_and_timers(env):
    repo, sd = env["repo"], env["sd"]
    (sd / "tar_mode").write_text("broken")
    r = _run(env, "--apply")
    assert r.returncode != 0 and "tar failed" in r.stderr
    assert _head(repo) == START
    assert (sd / "crontab").read_text() == "*/2 * * * * watchdog\n"
    assert (sd / "active_snapback-sol-be.timer").read_text().strip() == "active"


def test_failure_after_merge_rolls_back_params_verified(env, tmp_path):
    repo, sd = env["repo"], env["sd"]
    before = _params_bytes(repo)
    bad = tmp_path / "badpy"
    bad.write_text(f'#!/usr/bin/env bash\nfor a in "$@"; do [ "$a" = --check-config ] && exit 78; done\nexec {PYTHON} "$@"\n')
    bad.chmod(0o755)
    env["env"]["PYTHON"] = str(bad)
    r = _run(env, "--apply")
    assert r.returncode == 78
    assert _params_bytes(repo) == before
    assert (sd / "crontab").exists()


def test_crontab_restore_failure_is_loud_and_nonzero(env):
    sd = env["sd"]
    (sd / "crontab_fail").write_text("")
    r = _run(env, "--apply")
    assert r.returncode == 5
    assert "CRONTAB NOT RESTORED" in r.stderr and "crontab.bak" in r.stderr


def test_transient_timer_running_in_repo_aborts_preflight(env):
    """systemd-run names its timers run-*.timer (no TIMER_RE match) — still caught."""
    sd, repo = env["sd"], env["repo"]
    (sd / "timers").write_text("run-r1234.timer\nsnapback-sol-be.timer\n")
    (sd / "active_run-r1234.timer").write_text("active\n")
    (sd / "transient_run-r1234.timer").write_text("yes\n")
    (sd / "unit_run-r1234.timer").write_text("run-r1234.service\n")
    (sd / "exec_run-r1234.service").write_text(
        f"{{ path={repo}/.venv/bin/python ; argv[]=python -m tools.leg_safe_restart sol }}\n{repo}\n")
    r = _run(env, "--apply")
    assert r.returncode == 1 and "transient timer run-r1234.timer" in r.stderr
    assert _head(repo) == START
    assert "stop" not in (sd / "log").read_text().split()


def test_unrelated_transient_timer_is_logged_not_fatal(env):
    sd, repo = env["sd"], env["repo"]
    (sd / "timers").write_text("run-r9.timer\nsnapback-sol-be.timer\n")
    (sd / "active_run-r9.timer").write_text("active\n")
    (sd / "transient_run-r9.timer").write_text("yes\n")
    (sd / "unit_run-r9.timer").write_text("run-r9.service\n")
    (sd / "exec_run-r9.service").write_text("{ path=/usr/bin/apt ; }\n/\n")
    r = _run(env, "--apply")
    assert r.returncode == 0, r.stderr
    assert "does not reference" in r.stdout
    assert "stop run-r9.timer" not in (sd / "log").read_text()


def test_timer_listing_failure_fails_closed(env):
    (env["sd"] / "list_fail").write_text("")
    r = _run(env, "--apply")
    assert r.returncode == 1 and "could not list systemd timers" in r.stderr
    assert _head(env["repo"]) == START


def test_tar_exit_1_with_empty_stderr_aborts(env):
    (env["sd"] / "tar_mode").write_text("empty")
    r = _run(env, "--apply")
    assert r.returncode != 0 and "tar failed" in r.stderr
    assert _head(env["repo"]) == START


def test_dry_run_says_step_9_previews_pre_merge_units(env):
    r = _run(env)
    assert r.returncode == 0
    assert "previews the PRE-merge deploy/*.service" in r.stdout


def test_failed_merge_triggers_verified_trap_restore(env):
    """MERGE_ATTEMPTED is set before the merge, so a merge that dies (here: an
    untracked file it would overwrite) still runs the sha256-verified restore."""
    repo = env["repo"]
    before = _params_bytes(repo)
    (repo / "config_check.py").write_text("# blocks the ff-merge\n")
    r = _run(env, "--apply")
    assert r.returncode != 0
    assert "restored files verified byte-identical" in r.stderr
    assert _head(repo) == START
    assert _params_bytes(repo) == before
    assert (env["sd"] / "crontab").exists()
