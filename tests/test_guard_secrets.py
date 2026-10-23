"""tools/guard_secrets.sh fails closed and scans added lines faithfully."""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "guard_secrets.sh"
FAKE_IP = "10.99.88.77"
IP_RE = r"10\.99\.88\.77"


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    return tmp_path


def _run(repo, content, patterns=None, env=None):
    (repo / "f.txt").write_text(content)
    _git(repo, "add", "f.txt")
    if patterns is not None:
        (repo / ".guard_patterns").write_bytes(patterns.encode())
    e = {k: v for k, v in os.environ.items() if k != "GUARD_SECRETS_ALLOW_MISSING"}
    e.update(env or {})
    return subprocess.run(["bash", str(SCRIPT)], cwd=repo, env=e, capture_output=True, text=True)


def test_missing_patterns_blocks(repo):
    r = _run(repo, "hello\n")
    assert r.returncode == 1
    assert ".guard_patterns.example" in r.stderr


def test_missing_patterns_escape_hatch(repo):
    r = _run(repo, "hello\n", env={"GUARD_SECRETS_ALLOW_MISSING": "1"})
    assert r.returncode == 0


def test_invalid_regex_blocks(repo):
    r = _run(repo, "hello\n", patterns="([\n")
    assert r.returncode == 1


def test_staged_secret_blocks_without_echo(repo):
    r = _run(repo, f"host {FAKE_IP}\n", patterns=IP_RE + "\n")
    assert r.returncode == 1
    assert FAKE_IP not in r.stderr


def test_clean_passes(repo):
    r = _run(repo, "hello\n", patterns=f"# c\n\n   \n{IP_RE}\n")
    assert r.returncode == 0


def test_plus_plus_line_is_scanned(repo):
    r = _run(repo, f"++ {FAKE_IP}\n", patterns=IP_RE + "\n")
    assert r.returncode == 1


def test_crlf_patterns_still_match(repo):
    r = _run(repo, f"host {FAKE_IP}\n", patterns=IP_RE + "\r\n")
    assert r.returncode == 1


def test_whitespace_only_line_does_not_block_everything(repo):
    r = _run(repo, "hello\n", patterns="   \n")
    assert r.returncode == 0


def test_unfilled_placeholder_warns_and_is_ignored(repo):
    r = _run(repo, "hello\n", patterns="<EMAIL_LOCALPART>\n")
    assert r.returncode == 0
    assert "placeholder" in r.stderr



def test_patterns_in_git_common_dir_serve_every_worktree(repo, tmp_path):
    # One copy inside .git covers linked worktrees that have no local file.
    _git(repo, "commit", "-q", "--allow-empty", "-m", "init")
    (repo / ".git" / "guard_patterns").write_text(IP_RE + "\n")
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(wt))
    (wt / "f.txt").write_text(f"host {FAKE_IP}\n")
    _git(wt, "add", "f.txt")
    e = {k: v for k, v in os.environ.items() if k != "GUARD_SECRETS_ALLOW_MISSING"}
    r = subprocess.run(["bash", str(SCRIPT)], cwd=wt, env=e, capture_output=True, text=True)
    assert r.returncode == 1
    assert "BLOCKED" in r.stderr
