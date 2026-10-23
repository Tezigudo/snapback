"""Insert the `risk:` block into a live params YAML (one-time, per leg).

RISK_REVIEW 2026-10-22 moved the hard ceilings out of risk.py into each leg's
untracked params YAML. Code at or after that change REFUSES TO BOOT (exit 78)
until the leg's file has a complete `risk:` block — this tool writes it, with
exactly the values the pre-change risk.py enforced.

The values are NOT in this file (the repo is headed public). They are read at
run time from one of:
  --from-git SHA     (default) `git show SHA:risk.py`, parsed with ast (never
                     executed); SHA defaults to PRE_CHANGE_SHA, the last droplet
                     commit whose risk.py held the numbers.
  --values-file F    a local untracked YAML: either a `risk:` mapping or a flat
                     mapping of the params keys (max_notional_usd, ...).

Safe by default:
  * dry run unless --apply; always prints a unified diff per file;
  * inserts TEXT (appends a block) — never a yaml dump round-trip, so the live
    files keep their comments and ordering;
  * verifies the result: every non-risk key unchanged, config_check passes,
    the built ceilings equal the source values with the same python types;
  * an existing `risk:` block equal to the source → "already migrated", no-op;
    different → refused (never overwrites a ceiling);
  * --apply writes <file>.<UTC stamp>.bak first (config/*.bak is gitignored),
    then replaces the file atomically (fsynced temp .<file>.migrate.tmp, also
    gitignored, removed on any failure), preserving its mode;
  * --from-git takes a full 40-hex sha only (no branch names / short shas).

Usage (droplet, repo root, BEFORE restarting any leg onto the new code):
  .venv/bin/python tools/migrate_risk_params.py                 # dry run, all legs present
  .venv/bin/python tools/migrate_risk_params.py --apply
  for i in v1 donchian sol_supertrend cnh_short cnh_short_sol; do
      .venv/bin/python -m bot --instance $i --check-config; done
Exit: 0 ok / already migrated, 1 refused or verification failed, 2 bad usage.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_check import config_problems  # noqa: E402
from risk import RISK_FIELDS, ceilings_from_params  # noqa: E402

# Last droplet commit whose risk.py carried the live ceilings as constants.
PRE_CHANGE_SHA = "80a1f6fc16aaf02c2dc4be72b9611180ae6d7303"
FULL_SHA = re.compile(r"[0-9a-f]{40}")

LEG_FILES = (
    "params.yaml",
    "params_donchian.yaml",
    "params_sol_supertrend.yaml",
    "params_cnh_hybrid_short.yaml",
    "params_cnh_hybrid_short_sol.yaml",
)


class MigrateError(Exception):
    pass


def values_from_risk_source(src: str) -> dict[str, Any]:
    """{params key: value} from a pre-change risk.py source, via ast only."""
    tree = ast.parse(src)
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "RiskCeilings"), None)
    if cls is None:
        raise MigrateError("no class RiskCeilings in the source risk.py")
    found: dict[str, Any] = {}
    for node in cls.body:
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id in RISK_FIELDS and node.value is not None):
            try:
                found[node.target.id] = ast.literal_eval(node.value)
            except ValueError as e:
                raise MigrateError(f"{node.target.id}: not a literal ({e})") from e
    out: dict[str, Any] = {}
    for field, (key, typ) in RISK_FIELDS.items():
        v = found.get(field)
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            raise MigrateError(f"{field}: no numeric value in the source risk.py "
                               "(is that the pre-change commit?)")
        if type(v) is not typ:
            raise MigrateError(f"{field}: source type {type(v).__name__}, "
                               f"expected {typ.__name__}")
        out[key] = v
    return out


def values_from_git(sha: str, cwd: Path = ROOT) -> dict[str, Any]:
    if not FULL_SHA.fullmatch(sha):
        # A branch name or short sha could silently resolve to different values.
        raise MigrateError(f"--from-git needs a full 40-hex commit sha, got {sha!r}")
    r = subprocess.run(["git", "show", f"{sha}:risk.py"], cwd=cwd,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise MigrateError(f"git show {sha}:risk.py failed: {r.stderr.strip()}")
    return values_from_risk_source(r.stdout)


def values_from_file(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text())
    if isinstance(doc, dict) and isinstance(doc.get("risk"), dict):
        doc = doc["risk"]
    if not isinstance(doc, dict):
        raise MigrateError(f"{path}: expected a mapping")
    out: dict[str, Any] = {}
    for _field, (key, typ) in RISK_FIELDS.items():
        v = doc.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise MigrateError(f"{path}: {key} missing or non-numeric ({v!r})")
        if typ is int and not float(v).is_integer():
            raise MigrateError(f"{path}: {key} must be an integer ({v!r})")
        out[key] = typ(v)
    return out


def render_block(values: dict[str, Any], source: str) -> str:
    lines = [f"risk:  # hard ceilings, moved out of risk.py (RISK_REVIEW 2026-10-22); "
             f"source: {source}"]
    for _field, (key, typ) in RISK_FIELDS.items():
        lines.append(f"  {key}: {typ(values[key])!r}")
    return "\n".join(lines) + "\n"


def migrate_text(text: str, values: dict[str, Any], source: str) -> str | None:
    """New file text, or None if already migrated with identical values.
    Raises MigrateError on anything it will not do."""
    old_doc = yaml.safe_load(text)
    if not isinstance(old_doc, dict):
        raise MigrateError("not a YAML mapping")
    want = ceilings_from_params({"risk": values})
    if "risk" in old_doc:
        try:
            have = ceilings_from_params(old_doc)
        except ValueError as e:
            raise MigrateError(f"existing `risk:` block is incomplete ({e}); "
                               "fix it by hand — refusing to merge") from e
        if have == want and all(type(getattr(have, f)) is type(getattr(want, f))
                                for f in RISK_FIELDS):
            return None
        raise MigrateError("existing `risk:` block differs from the source values "
                           "— refusing to overwrite a ceiling")
    new = text if text.endswith("\n") or not text else text + "\n"
    new += "\n" + render_block(values, source)
    new_doc = yaml.safe_load(new)
    risk_block = new_doc.pop("risk", None)
    if new_doc != old_doc:
        raise MigrateError("verification: a non-risk key changed")
    got = ceilings_from_params({"risk": risk_block})
    for f in RISK_FIELDS:
        if getattr(got, f) != getattr(want, f) or type(getattr(got, f)) is not type(getattr(want, f)):
            raise MigrateError(f"verification: {f} round-trips as {getattr(got, f)!r}")
    new_doc["risk"] = risk_block
    problems = config_problems(new_doc)
    if problems:
        raise MigrateError("verification: config_check still refuses:\n  "
                           + "\n  ".join(problems))
    return new


def _apply(path: Path, new: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bak = path.with_name(f"{path.name}.{stamp}.bak")
    mode = path.stat().st_mode
    bak.write_bytes(path.read_bytes())
    os.chmod(bak, mode)
    tmp = path.with_name(f".{path.name}.migrate.tmp")   # gitignored (config/.*.migrate.tmp)
    try:
        with open(tmp, "w") as f:
            f.write(new)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return bak


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", type=Path,
                    help="params YAMLs (default: every live leg file in config/)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--from-git", metavar="SHA", default=None,
                     help=f"read values from SHA:risk.py; full 40-hex sha only (default {PRE_CHANGE_SHA})")
    src.add_argument("--values-file", type=Path, default=None,
                     help="read values from a local untracked YAML instead")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args(argv)

    try:
        if args.values_file is not None:
            values = values_from_file(args.values_file)
            source = f"values-file {args.values_file.name}"
        else:
            sha = args.from_git or PRE_CHANGE_SHA
            values = values_from_git(sha)
            source = f"git {sha}:risk.py"
    except (MigrateError, OSError, yaml.YAMLError) as e:
        print(f"REFUSED: cannot read source values: {e}", file=sys.stderr)
        return 1

    files = args.files or [ROOT / "config" / n for n in LEG_FILES
                           if (ROOT / "config" / n).is_file()]
    if not files:
        print("no params files found", file=sys.stderr)
        return 2

    rc = 0
    for path in files:
        try:
            text = path.read_text()
            new = migrate_text(text, values, source)
        except (MigrateError, OSError, yaml.YAMLError) as e:
            print(f"REFUSED {path}: {e}", file=sys.stderr)
            rc = 1
            continue
        if new is None:
            print(f"OK {path}: already migrated (identical values)")
            continue
        sys.stdout.writelines(difflib.unified_diff(
            text.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=str(path), tofile=f"{path} (migrated)"))
        if args.apply:
            bak = _apply(path, new)
            print(f"APPLIED {path} (backup {bak.name})")
        else:
            print(f"DRY RUN {path}: re-run with --apply to write")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
