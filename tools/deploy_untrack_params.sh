#!/usr/bin/env bash
# deploy_untrack_params.sh — fast-forward the droplet across the commit that
# UNTRACKS config/params*.yaml, without losing the live params files.
#
# WHY: that commit contains `git rm --cached config/params*.yaml`. A
# `git merge --ff-only` over it DELETES those files from the working tree, and
# the running legs re-read nothing until restart — but the NEXT restart (or any
# cron/timer tool reading them) would find no config and refuse to boot. So:
# pause cron + snapback timers, copy aside (sha256-verified), merge, restore,
# verify byte-identical, validate, make the units stop-not-loop on a refused
# boot (RestartPreventExitStatus=78 + daemon-reload), then resume cron/timers.
# NO service is restarted here; restart each leg later, flat, via
# tools/leg_safe_restart.py.
#
# DRY-RUN BY DEFAULT: prints every step and runs only the read-only checks.
#   tools/deploy_untrack_params.sh                 # dry run
#   tools/deploy_untrack_params.sh --apply         # do it
#
# Environment overrides (defaults suit the droplet):
#   REPO=/root/snapback-btc      repo working tree
#   TARGET=origin/droplet        ref to fast-forward to
#   BRANCH=droplet               branch that must be checked out
#   ASIDE_ROOT=/root             where the aside dir is created (outside REPO)
#   PYTHON=$REPO/.venv/bin/python
#   FETCH=1                      0 = skip `git fetch origin`
#   PAUSE_CRON=1                 0 = do not touch crontab (e.g. testing on a Mac)
#   MANAGE_SYSTEMD=1             0 = skip timer pause + unit install (testing)
#   SYSTEMCTL=systemctl          (tests point this at a fake)
#   UNIT_DIR=/etc/systemd/system where installed units / drop-ins live
#   TIMER_RE='snapback|sol[_-]'  timers paused for the window
#   TAR=tar   SHA256="sha256sum" (or "shasum -a 256", auto-detected)
set -euo pipefail

APPLY=0
for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

REPO="${REPO:-/root/snapback-btc}"
TARGET="${TARGET:-origin/droplet}"
BRANCH="${BRANCH:-droplet}"
ASIDE_ROOT="${ASIDE_ROOT:-/root}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
FETCH="${FETCH:-1}"
PAUSE_CRON="${PAUSE_CRON:-1}"
MANAGE_SYSTEMD="${MANAGE_SYSTEMD:-1}"
SYSTEMCTL="${SYSTEMCTL:-systemctl}"
UNIT_DIR="${UNIT_DIR:-/etc/systemd/system}"
TIMER_RE="${TIMER_RE:-snapback|sol[_-]}"
TAR="${TAR:-tar}"
if [[ -z "${SHA256:-}" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then SHA256="sha256sum"; else SHA256="shasum -a 256"; fi
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
ASIDE="$ASIDE_ROOT/snapback-params-aside-$TS"
CRON_PAUSED=0
TIMERS_PAUSED=()
MERGE_ATTEMPTED=0
RESTORED=0
UNITS_TOUCHED=0
PRE_HEAD=""

step() { printf '\n== %s\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die()  { printf '\nABORT: %s\n' "$*" >&2; exit 1; }
loud() { printf '\n!!!!!!!! %s\n' "$*" >&2; }
# run: mutating command — executed only with --apply.
run() {
  if [[ $APPLY -eq 1 ]]; then printf '   $ %s\n' "$*"; "$@";
  else printf '   [dry-run] $ %s\n' "$*"; fi
}

restore_files() {
  # Put every set-aside file back where it was (idempotent, overwrites).
  [[ -d "$ASIDE/files" ]] || return 0
  (cd "$ASIDE/files" && find . -type f -print0) | while IFS= read -r -d '' rel; do
    rel="${rel#./}"
    mkdir -p "$REPO/$(dirname "$rel")"
    cp -p "$ASIDE/files/$rel" "$REPO/$rel"
    printf '   restored %s\n' "$rel"
  done
}

verify_restored() { (cd "$REPO" && $SHA256 -c "$ASIDE/SHA256SUMS"); }

on_exit() {
  rc=$?
  [[ $APPLY -eq 1 ]] || exit $rc
  if [[ $rc -ne 0 ]]; then
    loud "FAILED (exit $rc) — rolling back what this script changed"
    # A merge that was started (even if interrupted) may have deleted params.
    if [[ $MERGE_ATTEMPTED -eq 1 && $RESTORED -eq 0 ]]; then
      printf '!! restoring params/.env from %s\n' "$ASIDE" >&2
      if restore_files && verify_restored; then
        printf '!! restored files verified byte-identical (sha256)\n' >&2
      else
        loud "RESTORE NOT VERIFIED — copy by hand from $ASIDE/files and check $ASIDE/SHA256SUMS"
        rc=3
      fi
    fi
    if [[ $UNITS_TOUCHED -eq 1 ]]; then
      loud "unit files/drop-ins were changed; originals are in $ASIDE/units — review before restarting any leg"
    fi
    if [[ -n "$PRE_HEAD" && "$(git -C "$REPO" rev-parse HEAD)" != "$PRE_HEAD" ]]; then
      printf '!! git HEAD moved %s -> %s. Investigate before restarting any leg.\n' \
        "${PRE_HEAD:0:7}" "$(git -C "$REPO" rev-parse --short HEAD)" >&2
    fi
  fi
  for t in "${TIMERS_PAUSED[@]+"${TIMERS_PAUSED[@]}"}"; do
    if $SYSTEMCTL start "$t"; then printf '   timer resumed: %s\n' "$t"
    else loud "TIMER $t NOT RESUMED — run: $SYSTEMCTL start $t"; rc=4; fi
  done
  if [[ $CRON_PAUSED -eq 1 ]]; then
    printf '\n== restore crontab\n'
    if crontab "$ASIDE/crontab.bak"; then
      printf '   crontab restored (%s lines)\n' "$(wc -l < "$ASIDE/crontab.bak")"
    else
      loud "CRONTAB NOT RESTORED — cron jobs (watchdog, dashboard, monitor) are OFF. Run: crontab $ASIDE/crontab.bak"
      rc=5
    fi
  fi
  [[ -d "$ASIDE" ]] && printf '\nAside copy kept at %s (delete once all legs restarted cleanly).\n' "$ASIDE"
  exit $rc
}
trap on_exit EXIT

printf 'deploy_untrack_params  mode=%s  repo=%s  target=%s  aside=%s\n' \
  "$([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)" "$REPO" "$TARGET" "$ASIDE"

# ---------------------------------------------------------------------------
step "0. preflight (read-only)"
cd "$REPO" || die "no repo at $REPO"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BRANCH" ]] || die "not on branch $BRANCH"
case "$ASIDE" in "$REPO"/*) die "ASIDE_ROOT must be outside REPO" ;; esac
if [[ "$FETCH" == "1" ]]; then info "git fetch origin"; git fetch -q origin; fi
git rev-parse -q --verify "$TARGET^{commit}" >/dev/null || die "unknown target $TARGET"
git merge-base --is-ancestor HEAD "$TARGET" || die "HEAD is not an ancestor of $TARGET — not a fast-forward"
PRE_HEAD="$(git rev-parse HEAD)"
dirty="$(git status --porcelain --untracked-files=no config/)"
[[ -z "$dirty" ]] || die "config/ has local modifications to tracked files — a merge would fail or clobber them. Resolve first:
$dirty"
# The restore puts back the params as they are at THIS checkout. If HEAD is
# behind the untrack commit's parent AND a commit in between edited a params
# file, the merge would delete the newer version and the restore would silently
# put the older one back (--check-config can't tell). Refuse that case.
untrack="$(git log -1 --format=%H --diff-filter=D "$TARGET" -- config/params.yaml)"
if [[ -n "$untrack" ]] && ! git merge-base --is-ancestor "$untrack" HEAD; then
  if ! git diff --quiet HEAD "$untrack^" -- 'config/params*.yaml'; then
    die "tracked params differ between HEAD and ${untrack:0:7}^ (the last commit that tracks them):
$(git diff --stat HEAD "$untrack^" -- 'config/params*.yaml')
Deploy normally (plain ff-merge) to ${untrack:0:7}^ first, restart as needed, then re-run this."
  fi
  info "untrack commit ${untrack:0:7}; tracked params at HEAD == ${untrack:0:7}^ (no in-between edits lost)"
fi
shopt -s nullglob
params=()
for f in config/params*.yaml; do [[ "$f" == *.example.yaml ]] || params+=("$f"); done
envs=(.env .env.*)
envs_real=()
for f in "${envs[@]}"; do [[ -f "$f" && "$f" != *.example ]] && envs_real+=("$f"); done
shopt -u nullglob
[[ ${#params[@]} -gt 0 ]] || die "no config/params*.yaml in $REPO — nothing to protect?"
info "HEAD   ${PRE_HEAD:0:7}  →  $TARGET $(git rev-parse --short "$TARGET")"
info "params to protect: ${params[*]}"
info "env files to protect: ${envs_real[*]:-(none)}"
if git cat-file -e "$TARGET:config/params.yaml" 2>/dev/null; then
  info "NOTE: $TARGET still TRACKS config/params.yaml — this merge will not untrack anything"
fi
[[ -x "$PYTHON" ]] || die "python not found at $PYTHON"

timers=()
if [[ "$MANAGE_SYSTEMD" == "1" ]]; then
  # Fail closed: a failed listing must not read as "no timers to pause".
  timer_list="$($SYSTEMCTL list-units --type=timer --all --no-legend --plain)" \
    || die "could not list systemd timers ($SYSTEMCTL list-units failed) — refusing to continue blind"
  while read -r t _; do
    [[ -n "$t" ]] || continue
    state="$($SYSTEMCTL is-active "$t" || true)"
    [[ "$state" == "active" ]] || continue
    # EVERY pending transient timer (systemd-run --on-active names them
    # run-*.timer, so TIMER_RE would miss them) is inspected: stop would DESTROY
    # it, so it can't be paused+resumed. If it would run anything in this repo
    # (a deferred leg restart) it could fire mid-merge against missing params.
    if [[ "$($SYSTEMCTL show -p Transient --value "$t")" == "yes" ]]; then
      target="$($SYSTEMCTL show -p Unit --value "$t")"
      what="$($SYSTEMCTL show -p ExecStart -p WorkingDirectory --value "$target" 2>/dev/null || true)"
      if [[ "$what" == *"$REPO"* ]]; then
        die "transient timer $t → $target runs in $REPO (e.g. a deferred leg restart). It would fire mid-deploy against missing params and cannot be paused+resumed. Let it fire or stop it deliberately, then re-run."
      fi
      info "transient timer $t → ${target:-?} does not reference $REPO — left alone"
      continue
    fi
    if [[ "$t" =~ $TIMER_RE ]]; then timers+=("$t"); fi
  done <<< "$timer_list"
  info "active snapback timers to pause: ${timers[*]:-(none)}"
fi

# ---------------------------------------------------------------------------
step "1. create aside dir (mode 700, outside the repo)"
run mkdir -p "$ASIDE/files" "$ASIDE/units"
run chmod 700 "$ASIDE"

# ---------------------------------------------------------------------------
step "2. pause crontab + snapback timers (both restored by the EXIT trap)"
if [[ "$PAUSE_CRON" == "1" ]]; then
  if [[ $APPLY -eq 1 ]]; then
    # `crontab -l` exits non-zero when there is no crontab — that is fine.
    crontab -l > "$ASIDE/crontab.bak" 2>/dev/null || : > "$ASIDE/crontab.bak"
    info "saved crontab ($(wc -l < "$ASIDE/crontab.bak") lines) → $ASIDE/crontab.bak"
    if [[ -s "$ASIDE/crontab.bak" ]]; then
      CRON_PAUSED=1
      run crontab -r
    else
      info "no crontab installed; nothing to pause"
    fi
  else
    info "[dry-run] \$ crontab -l > $ASIDE/crontab.bak ; crontab -r"
  fi
else
  info "PAUSE_CRON=0 — crontab untouched"
fi
for t in "${timers[@]+"${timers[@]}"}"; do
  run $SYSTEMCTL stop "$t"
  if [[ $APPLY -eq 1 ]]; then TIMERS_PAUSED+=("$t"); fi
done

# ---------------------------------------------------------------------------
step "3. tar the tree as a last-resort fallback (minus .venv, logs, sqlite wal/shm, heartbeats)"
if [[ $APPLY -eq 1 ]]; then
  set +e
  $TAR -czf "$ASIDE/tree.tgz" --exclude='./.venv' --exclude='./logs' \
    --exclude='./data/*-wal' --exclude='./data/*-shm' --exclude='./data/heartbeat*' \
    -C "$REPO" . 2> "$ASIDE/tar.stderr"
  trc=$?
  set -e
  # GNU tar exits 1 (not 2) when a file changed while being read — a running
  # bot writes data/*.db constantly. Accept exactly that, nothing else.
  # Exit 1 is accepted only with a NON-empty stderr made up solely of these
  # anchored GNU warnings; empty stderr or any other line aborts.
  if [[ $trc -eq 1 && -s "$ASIDE/tar.stderr" ]] && ! grep -vqE \
      '^tar: .*: (file changed as we read it|[Ff]ile removed before we read it|socket ignored)$|^tar: Exiting with failure status due to previous errors$' \
      "$ASIDE/tar.stderr"; then
    info "tar exit 1 with only live-file-changed warnings — accepted:"
    sed 's/^/     /' "$ASIDE/tar.stderr"
  elif [[ $trc -ne 0 ]]; then
    cat "$ASIDE/tar.stderr" >&2
    die "tar failed (exit $trc)"
  fi
  info "tree.tgz $(du -h "$ASIDE/tree.tgz" | cut -f1)"
else
  info "[dry-run] \$ $TAR -czf $ASIDE/tree.tgz --exclude=./.venv --exclude=./logs --exclude='./data/*-wal' --exclude='./data/*-shm' --exclude='./data/heartbeat*' -C $REPO ."
fi

# ---------------------------------------------------------------------------
step "4. copy params + .env aside; verify the COPIES by sha256 before merging"
for f in "${params[@]}" "${envs_real[@]+"${envs_real[@]}"}"; do
  run mkdir -p "$ASIDE/files/$(dirname "$f")"
  run cp -p "$f" "$ASIDE/files/$f"
done
if [[ $APPLY -eq 1 ]]; then
  (cd "$REPO" && $SHA256 "${params[@]}" "${envs_real[@]+"${envs_real[@]}"}") > "$ASIDE/SHA256SUMS"
  sed 's/^/   /' "$ASIDE/SHA256SUMS"
  (cd "$ASIDE/files" && $SHA256 -c "$ASIDE/SHA256SUMS") || die "aside copies do not match the originals"
else
  info "[dry-run] \$ (cd $REPO && $SHA256 <params> <env>) > $ASIDE/SHA256SUMS ; (cd $ASIDE/files && $SHA256 -c ../SHA256SUMS)"
fi

# ---------------------------------------------------------------------------
step "5. fast-forward merge (this DELETES the params files from the tree)"
if [[ $APPLY -eq 1 ]]; then MERGE_ATTEMPTED=1; fi   # BEFORE the merge: an interrupted merge still restores
run git merge --ff-only "$TARGET"
if [[ $APPLY -eq 1 ]]; then
  for f in "${params[@]}"; do
    if [[ -e "$f" ]]; then info "still present: $f"; else info "deleted by merge: $f"; fi
  done
fi

# ---------------------------------------------------------------------------
step "6. restore params + .env from aside and verify byte-identical"
if [[ $APPLY -eq 1 ]]; then
  restore_files
  printf '   $ (cd %s && %s -c %s/SHA256SUMS)\n' "$REPO" "$SHA256" "$ASIDE"
  verify_restored
  RESTORED=1
else
  info "[dry-run] copy every file under $ASIDE/files back into $REPO ; $SHA256 -c"
fi

# ---------------------------------------------------------------------------
step "7. REQUIRED_KEYS check on every leg's params (no exchange/env access)"
for f in "${params[@]}"; do
  inst="$("$PYTHON" - "$f" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
import bot
name = Path(sys.argv[1]).name
hits = [i for i, p in bot.INSTANCE_PROFILES.items() if p["config"].name == name]
print(hits[0] if hits else "")
PY
)"
  if [[ -z "$inst" ]]; then info "skip $f (no instance profile uses it)"; continue; fi
  run "$PYTHON" -m bot --instance "$inst" --check-config
done

# ---------------------------------------------------------------------------
step "8. config/ must be clean (params now ignored, templates tracked)"
if [[ $APPLY -eq 1 ]]; then
  porcelain="$(git status --porcelain config/)"
  [[ -z "$porcelain" ]] || die "git status --porcelain config/ not empty:
$porcelain"
  for f in "${params[@]}"; do git check-ignore -q "$f" || die "$f is not gitignored"; done
  info "clean; all params files ignored"
else
  info "[dry-run] \$ git status --porcelain config/   # must print nothing"
fi

# ---------------------------------------------------------------------------
step "9. units: stop-not-loop on a refused boot (RestartPreventExitStatus=78), daemon-reload, NO restart"
# Only units ALREADY installed are touched (a never-installed unit, e.g. cnh, stays
# uninstalled). If the installed file is byte-identical to the repo's PRE-merge
# copy it is replaced by the new repo unit; otherwise it carries droplet-local
# edits, so a drop-in adds just the one line instead of clobbering them.
if [[ $APPLY -ne 1 ]]; then
  info "NOTE (dry-run): no merge has happened, so this previews the PRE-merge deploy/*.service"
  info "units and compares them with the PRE-merge repo copies. The real run uses the merged"
  info "($TARGET) units: changed/added/dropped units in the target are NOT reflected here."
fi
if [[ "$MANAGE_SYSTEMD" == "1" ]]; then
  installed=()
  for src in deploy/*.service; do
    u="$(basename "$src")"
    if [[ "$($SYSTEMCTL show -p LoadState --value "$u" 2>/dev/null)" != "loaded" ]]; then
      info "$u not installed — skipped"; continue
    fi
    frag="$($SYSTEMCTL show -p FragmentPath --value "$u")"
    installed+=("$u")
    if [[ -n "$frag" && -f "$frag" ]] && git show "$PRE_HEAD:deploy/$u" 2>/dev/null | cmp -s - "$frag"; then
      run cp -p "$frag" "$ASIDE/units/$u.orig"
      run cp "$src" "$frag"
      info "$u: installed new unit file ($frag)"
    else
      dropin="$UNIT_DIR/$u.d/50-restart-prevent-config.conf"
      if [[ -f "$dropin" ]]; then run cp -p "$dropin" "$ASIDE/units/$u.dropin.orig"; fi
      run mkdir -p "$UNIT_DIR/$u.d"
      if [[ $APPLY -eq 1 ]]; then
        printf '[Service]\n# 78 = EX_CONFIG: params refused (config_check.py). Stop, do not crash-loop.\nRestartPreventExitStatus=78\n' > "$dropin"
        info "$u: installed unit differs from repo (local edits kept) → drop-in $dropin"
      else
        info "[dry-run] write drop-in $dropin (RestartPreventExitStatus=78)"
      fi
    fi
    if [[ $APPLY -eq 1 ]]; then UNITS_TOUCHED=1; fi
  done
  if [[ ${#installed[@]} -gt 0 ]]; then
    run $SYSTEMCTL daemon-reload
    if [[ $APPLY -eq 1 ]]; then
      for u in "${installed[@]}"; do
        v="$($SYSTEMCTL show -p RestartPreventExitStatus --value "$u")"
        [[ " $v " == *" 78 "* ]] || die "$u: RestartPreventExitStatus is '$v' after daemon-reload, expected 78"
        info "$u: RestartPreventExitStatus=$v (service NOT restarted)"
      done
    fi
  fi
else
  info "MANAGE_SYSTEMD=0 — units untouched. Install RestartPreventExitStatus=78 + daemon-reload BEFORE any leg restart."
fi

# crontab + timers are restored by the EXIT trap (also on failure).
step "10. done — NO service was restarted"
info "running legs keep their in-memory params; restart each leg later, while"
info "FLAT, via tools/leg_safe_restart.py <instance>."
