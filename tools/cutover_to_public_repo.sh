#!/usr/bin/env bash
# cutover_to_public_repo.sh — move the LIVE droplet checkout from the private
# repo (Tezigudo/snapback-btc, branch `droplet`) to the public repo
# (Tezigudo/snapback, branch `master`) WITHOUT touching running services,
# crontab, or any ignored/untracked live file.
#
# The public commit must be the private HEAD's tree minus excluded paths
# (deletions only). The script proves that before it changes anything, pins
# local `master` to the EXPECTED public sha (not whatever the tip is), and
# rolls back automatically on any failure after the first mutation.
#
# Usage:
#   cutover_to_public_repo.sh --private-sha <40hex> --public-sha <40hex> \
#                             --excludes <file> [--apply]
#   cutover_to_public_repo.sh --rollback <aside-dir>
#
# Default is a DRY RUN: read-only except that `git fetch` stores objects in
# .git (no refs/remotes/config are changed). The dry run still fetches with the
# new deploy key, proves the equivalence, and runs --check-config + the cron
# imports against an exported copy of the new tree.
#
# Environment (defaults suit the droplet):
#   REPO=/root/snapback-btc
#   PUBLIC_URL=git@github-snapback:Tezigudo/snapback.git   (remote URL to store)
#   FETCH_URL=git@github.com:Tezigudo/snapback.git          (dry-run fetch URL)
#   DEPLOY_KEY=$HOME/.ssh/gh_deploy_snapback
#   SSH_ALIAS=github-snapback
#   PYTHON=$REPO/.venv/bin/python
#   ASIDE_ROOT=$HOME                  aside dir = $ASIDE_ROOT/cutover-aside-<ts>
#   SIM=1                             local simulation: skip ssh key/config,
#                                     systemctl and crontab checks
#   SKIP_PY_CHECKS=1                  skip --check-config/cron imports (reported)
#   CUTOVER_FAIL_AFTER=<phase>        test hook: fail after ssh|remote|rename|switch|verify
set -euo pipefail
umask 077

# ---------------------------------------------------------------- constants --
INSTANCES=(v1 donchian sol_supertrend)
UNITS=(snapback-btc snapback-btc-donchian snapback-sol-supertrend)
CRON_MODULES=(tools.build_dashboard tools.watchdog tools.capitulation_watch
              monitor daily_digest tools.consolidate_futures_push)
RUNTIME_DIRS=(data logs reports)   # units' ReadWritePaths — must exist

REPO="${REPO:-/root/snapback-btc}"
PUBLIC_URL="${PUBLIC_URL:-git@github-snapback:Tezigudo/snapback.git}"
FETCH_URL="${FETCH_URL:-git@github.com:Tezigudo/snapback.git}"
DEPLOY_KEY="${DEPLOY_KEY:-$HOME/.ssh/gh_deploy_snapback}"
SSH_ALIAS="${SSH_ALIAS:-github-snapback}"
SSH_CONFIG="$HOME/.ssh/config"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
ASIDE_ROOT="$(cd "${ASIDE_ROOT:-$HOME}" && pwd)"
SIM="${SIM:-0}"
SKIP_PY_CHECKS="${SKIP_PY_CHECKS:-0}"
FAIL_AFTER="${CUTOVER_FAIL_AFTER:-}"

APPLY=0; ROLLBACK_DIR=""; PRIV_SHA=""; PUB_SHA=""; EXCLUDES=""
ASIDE=""; APPLY_STARTED=0; DONE=0; FAILS=0

# ------------------------------------------------------------------ helpers --
info() { printf '[cutover] %s\n' "$*"; }
step() { printf '\n[cutover] === %s ===\n' "$*"; }
die()  { printf '[cutover] FATAL: %s\n' "$*" >&2; exit 1; }
fail() { printf '[cutover] CHECK FAILED: %s\n' "$*" >&2; FAILS=$((FAILS+1)); }
run()  { printf '[cutover] + %s\n' "$*"; "$@"; }
sha()  { sha256sum "$@"; }
inode() { stat -c %i "$1" 2>/dev/null || stat -f %i "$1"; }
maybe_fail() { [[ "$FAIL_AFTER" == "$1" ]] && die "injected failure after phase '$1'"; return 0; }
g() { git -C "$REPO" "$@"; }
state_set() { printf '%s=%s\n' "$1" "$2" >> "$ASIDE/state"; }
state_get() { sed -n "s/^$1=//p" "$2/state" | tail -1; }
is_full_sha() { [[ "$1" =~ ^[0-9a-f]{40}$ ]]; }
ssh_cmd() { printf 'ssh -F /dev/null -i %q -o IdentitiesOnly=yes -o BatchMode=yes' "$DEPLOY_KEY"; }

usage() { sed -n '2,33p' "$0"; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --private-sha) PRIV_SHA="$2"; shift 2 ;;
    --public-sha)  PUB_SHA="$2";  shift 2 ;;
    --excludes)    EXCLUDES="$2"; shift 2 ;;
    --apply)       APPLY=1; shift ;;
    --rollback)    ROLLBACK_DIR="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *) die "unknown argument: $1" ;;
  esac
done

# --------------------------------------------------------------- manifests --
# Ignored live files. Secrets/params are compared by sha256 (they must be
# byte-identical). data/ logs/ reports/ are written by the RUNNING bots every
# tick, so their hashes legitimately move: for those we assert existence +
# same inode (git never rewrites ignored files; the real risk is deletion).
build_manifest() {   # $1 = output prefix
  local out="$1"
  ( cd "$REPO"
    : > "$out.strict"; : > "$out.inode"
    for f in .env* config/params*.yaml* confirm_mainnet.lock; do
      [[ -f "$f" ]] && git check-ignore -q "$f" && sha "$f" >> "$out.strict" || true
    done
    while IFS= read -r -d '' f; do
      [[ -f "$f" ]] || continue
      case "$f" in .env*|config/params*|confirm_mainnet.lock) continue ;; esac
      printf '%s\t%s\n' "$(inode "$f")" "$f" >> "$out.inode"
    done < <(git ls-files -z -o -i --exclude-standard -- data logs reports archive)
    # every ignored entry (dirs collapsed: .venv/, __pycache__/, archive/ ...)
    git ls-files -o -i --exclude-standard --directory | LC_ALL=C sort > "$out.ignored"
    git ls-files -o --exclude-standard | LC_ALL=C sort > "$out.untracked"
  )
}

compare_manifests() {   # $1 = before prefix, $2 = after prefix
  local b="$1" a="$2" missing x
  for x in strict inode ignored untracked; do
    [[ -f "$b.$x" && -f "$a.$x" ]] || { fail "manifest file missing: $x"; return 0; }
  done
  if cmp -s "$b.strict" "$a.strict"; then
    info "OK  secrets/params sha256 unchanged ($(wc -l < "$b.strict" | tr -d ' ') files)"
  else
    fail "secrets/params sha256 manifest changed (paths only):"
    diff "$b.strict" "$a.strict" | grep '^[<>]' | cut -c1-2,69- >&2 || true
  fi
  missing="$(LC_ALL=C comm -23 <(LC_ALL=C sort "$b.inode") <(LC_ALL=C sort "$a.inode") || true)"
  if [[ -z "$missing" ]]; then
    info "OK  runtime files (data/logs/reports) all present, same inode ($(wc -l < "$b.inode" | tr -d ' ') files)"
  else
    fail "runtime files missing or replaced:"; printf '%s\n' "$missing" >&2
  fi
  missing="$(LC_ALL=C comm -23 "$b.ignored" "$a.ignored" || true)"
  [[ -z "$missing" ]] && info "OK  all $(wc -l < "$b.ignored" | tr -d ' ') ignored entries still present" \
    || { fail "ignored entries vanished:"; printf '%s\n' "$missing" >&2; }
  missing="$(LC_ALL=C comm -23 "$b.untracked" "$a.untracked" || true)"
  if [[ -z "$missing" ]]; then
    info "OK  untracked files: before $(wc -l < "$b.untracked" | tr -d ' '), after $(wc -l < "$a.untracked" | tr -d ' '), none lost"
  else
    fail "untracked files lost:"; printf '%s\n' "$missing" >&2
  fi
}

# ------------------------------------------------------------ equivalence --
# HEAD -> PUB must be deletions only, every deletion covered by the exclude
# list (entries ending in '/' are directory prefixes). Writes the deleted
# paths NUL-separated to $1.
prove_equivalence() {
  local out="$1" st p e covered n=0 bad=0
  local -a ex=()
  while IFS= read -r e || [[ -n "$e" ]]; do
    e="${e%$'\r'}"; e="${e#"${e%%[![:space:]]*}"}"; e="${e%"${e##*[![:space:]]}"}"
    [[ -z "$e" || "$e" == \#* ]] && continue
    [[ "$e" == *[\*\?\[]* ]] && die "exclude list has a glob entry ($e) — not supported"
    ex+=("$e")
  done < "$EXCLUDES"
  [[ ${#ex[@]} -gt 0 ]] || die "exclude list is empty"
  : > "$out"
  while IFS= read -r -d '' st && IFS= read -r -d '' p; do
    if [[ "$st" != D ]]; then
      fail "public tree differs by '$st' (not a deletion): $p"; bad=$((bad+1)); continue
    fi
    covered=0
    for e in "${ex[@]}"; do
      if [[ "$e" == */ ]]; then [[ "$p" == "$e"* ]] && { covered=1; break; }
      else [[ "$p" == "$e" ]] && { covered=1; break; }; fi
    done
    if [[ $covered -eq 0 ]]; then fail "deletion not covered by exclude list: $p"; bad=$((bad+1)); continue; fi
    printf '%s\0' "$p" >> "$out"; n=$((n+1))
  done < <(g diff --no-renames -z --name-status "$PRIV_SHA" "$PUB_SHA")
  [[ $bad -eq 0 ]] || die "equivalence FAILED ($bad offending paths) — public commit is not private HEAD minus excludes"
  [[ $n -gt 0 ]] || die "no deletions at all — wrong sha pair?"
  local dl; dl="$(tr '\0' '\n' < "$out")"
  for e in "${ex[@]}"; do   # informational: exclude entries with no tracked match
    if [[ "$e" == */ ]]; then awk -v e="$e" 'index($0,e)==1{f=1} END{exit !f}' <<<"$dl" && continue
    else grep -qxF -- "$e" <<<"$dl" && continue; fi
    info "note: exclude entry not tracked at private HEAD (fine): $e"
  done
  info "OK  public = private HEAD minus $n deleted paths, all covered by ${#ex[@]} exclude entries; 0 M/A/R/T"
}

# Live wiring must not reference a path that is about to disappear.
check_live_references() {   # $1 = NUL list of deleted paths
  if [[ "$SIM" == 1 ]]; then info "SKIP crontab/unit reference check (SIM)"; return; fi
  local blob p u hits=0
  blob="$( { crontab -l 2>/dev/null || true
             for u in "${UNITS[@]}"; do systemctl cat "$u" 2>/dev/null || true; done
             for u in /etc/systemd/system/snapback*.service /etc/systemd/system/*/snapback*; do
               [[ -e "$u" || -L "$u" ]] && readlink -f "$u"; done; } )"
  while IFS= read -r -d '' p; do
    if grep -qF -- "$p" <<<"$blob"; then fail "live crontab/unit references deleted path: $p"; hits=$((hits+1)); fi
  done < "$1"
  [[ $hits -eq 0 ]] && info "OK  crontab + units + unit symlinks reference none of the deleted paths" \
    || die "live wiring references deleted paths"
  for u in /etc/systemd/system/snapback*.service; do
    [[ -L "$u" && ! -e "$u" ]] && info "WARN dangling unit symlink (pre-existing): $u"
  done
  return 0
}

# --check-config for every live leg + import of every cron module, run in $1.
py_checks() {   # $1 = tree to run in
  if [[ "$SKIP_PY_CHECKS" == 1 ]]; then info "SKIPPED (not passed): python checks (SKIP_PY_CHECKS=1)"; return; fi
  [[ -x "$PYTHON" ]] || die "python not found: $PYTHON"
  local i ok=0
  for i in "${INSTANCES[@]}"; do
    if (cd "$1" && "$PYTHON" -m bot --instance "$i" --check-config >/dev/null 2>"$ASIDE_TMP/cc.err"); then
      info "OK  --check-config $i"; ok=$((ok+1))
    else fail "--check-config $i: $(tail -1 "$ASIDE_TMP/cc.err")"; fi
  done
  if (cd "$1" && "$PYTHON" -c 'import importlib,sys; [importlib.import_module(m) for m in sys.argv[1:]]' \
        "${CRON_MODULES[@]}" >/dev/null 2>"$ASIDE_TMP/imp.err"); then
    info "OK  cron modules import: ${CRON_MODULES[*]}"
  else fail "cron module import: $(tail -1 "$ASIDE_TMP/imp.err")"; fi
}

services_snapshot() {   # $1 = output file ("unit active pid")
  : > "$1"
  [[ "$SIM" == 1 ]] && return 0
  local u
  for u in "${UNITS[@]}"; do
    printf '%s %s %s\n' "$u" "$(systemctl is-active "$u" 2>/dev/null || true)" \
      "$(systemctl show -p MainPID --value "$u" 2>/dev/null || echo ?)" >> "$1"
  done
}

check_runtime_dirs() {
  local d u p
  for d in "${RUNTIME_DIRS[@]}"; do [[ -d "$REPO/$d" ]] || fail "runtime dir missing: $REPO/$d"; done
  [[ "$SIM" == 1 ]] && { info "OK  runtime dirs present: ${RUNTIME_DIRS[*]} (unit ReadWritePaths scan skipped, SIM)"; return; }
  for u in "${UNITS[@]}"; do
    for p in $(systemctl show -p ReadWritePaths --value "$u" 2>/dev/null); do
      p="${p#-}"; [[ -d "$p" ]] || fail "$u ReadWritePaths dir missing: $p (unit would fail on next restart)"
    done
  done
  info "OK  runtime dirs + all unit ReadWritePaths exist"
}

# ---------------------------------------------------------------- rollback --
rollback() {   # $1 = aside dir; best effort, every step reported
  local a="$1" rc=0
  set +e
  step "ROLLBACK from $a"
  REPO="$(state_get REPO "$a")"; PRIV_SHA="$(state_get PRIV_SHA "$a")"
  local pub_added; pub_added="$(state_get PUBLIC_ADDED "$a")"
  [[ -n "$REPO" && -d "$REPO/.git" ]] || { echo "[cutover] rollback: bad state file" >&2; return 1; }

  if [[ "$(g symbolic-ref -q --short HEAD)" != droplet ]]; then
    run g checkout -q droplet || { echo "[cutover] rollback: checkout droplet FAILED — fix by hand" >&2; rc=1; }
  fi
  [[ "$(g rev-parse HEAD)" == "$PRIV_SHA" ]] || { echo "[cutover] rollback: droplet HEAD != $PRIV_SHA" >&2; rc=1; }
  if g show-ref -q --verify refs/heads/master && [[ "$(state_get MASTER_CREATED "$a")" == 1 ]]; then
    run g branch -q -D master
  fi
  # restore any deleted tracked file that is still missing (checkout normally does it)
  local missing; missing="$(g ls-files -d)"
  if [[ -n "$missing" && -f "$a/deleted-tracked.tgz" ]]; then
    info "restoring $(wc -l <<<"$missing" | tr -d ' ') missing tracked files from tarball"
    (cd "$REPO" && tar -xzf "$a/deleted-tracked.tgz") || rc=1
  fi

  # Idempotent: act only while the 'private' remote exists (i.e. renames not yet undone).
  local pub_url; pub_url="$(state_get PUBLIC_URL "$a")"
  if g remote get-url private >/dev/null 2>&1; then
    if [[ "$(g remote get-url origin 2>/dev/null)" == "$pub_url" ]] && ! g remote get-url public >/dev/null 2>&1; then
      run g remote rename origin public
    fi
    g remote get-url origin >/dev/null 2>&1 || run g remote rename private origin
  else
    info "remotes already restored (no 'private' remote)"
  fi
  if [[ "$pub_added" == 1 && "$(g remote get-url public 2>/dev/null)" == "$pub_url" ]]; then
    run g remote remove public
  fi
  [[ "$(g config branch.droplet.remote)" == origin ]] || { echo "[cutover] rollback: droplet upstream remote is not origin" >&2; rc=1; }

  if [[ "$(state_get SSH_MODIFIED "$a")" == 1 ]]; then
    if [[ -f "$a/ssh_config.bak" ]]; then run cp -p "$a/ssh_config.bak" "$SSH_CONFIG"
    else run rm -f "$SSH_CONFIG"; fi   # there was no config before
  fi

  ASIDE="$a"; ASIDE_TMP="$a/tmp"; mkdir -p "$ASIDE_TMP"
  build_manifest "$a/rollback"
  FAILS=0; compare_manifests "$a/before" "$a/rollback"
  [[ -z "$(g status --porcelain --untracked-files=no)" ]] && info "OK  no tracked changes" \
    || { fail "tracked changes after rollback"; rc=1; }
  [[ $FAILS -eq 0 ]] || rc=1
  set -e
  [[ $rc -eq 0 ]] && info "ROLLBACK COMPLETE: branch droplet @ ${PRIV_SHA:0:7}, origin = private repo" \
                  || echo "[cutover] ROLLBACK FINISHED WITH ERRORS — inspect $REPO and $a" >&2
  return $rc
}

on_exit() {
  local rc=$?
  if [[ $APPLY_STARTED -eq 1 && $DONE -ne 1 ]]; then
    echo "[cutover] apply did not complete (rc=$rc) — rolling back automatically" >&2
    APPLY_STARTED=0
    rollback "$ASIDE" || true
    exit $(( rc == 0 ? 1 : rc ))
  fi
  [[ -n "${EXPORT_DIR:-}" && -d "${EXPORT_DIR:-}" ]] && rm -rf "$EXPORT_DIR"
  exit $rc
}
trap on_exit EXIT
trap 'exit 130' INT TERM HUP

if [[ -n "$ROLLBACK_DIR" ]]; then
  [[ -f "$ROLLBACK_DIR/state" ]] || die "no state file in $ROLLBACK_DIR"
  ROLLBACK_DIR="$(cd "$ROLLBACK_DIR" && pwd)"
  rollback "$ROLLBACK_DIR"; exit $?
fi

# ------------------------------------------------------------------ preflight
is_full_sha "$PRIV_SHA" || die "--private-sha must be a full 40-hex sha"
is_full_sha "$PUB_SHA"  || die "--public-sha must be a full 40-hex sha"
[[ -f "$EXCLUDES" ]] || die "--excludes file not found"
EXCLUDES="$(cd "$(dirname "$EXCLUDES")" && pwd)/$(basename "$EXCLUDES")"

step "$([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)  repo=$REPO  private=${PRIV_SHA:0:7}  public=${PUB_SHA:0:7}  sim=$SIM"

step "1. preflight"
[[ -d "$REPO/.git" ]] || die "$REPO is not a git checkout"
[[ "$(g symbolic-ref -q --short HEAD)" == droplet ]] || die "branch is not 'droplet'"
[[ "$(g rev-parse HEAD)" == "$PRIV_SHA" ]] || die "HEAD $(g rev-parse --short HEAD) != expected ${PRIV_SHA:0:7}"
g diff --quiet && g diff --cached --quiet || die "tracked modifications present — refusing"
info "OK  branch droplet @ ${PRIV_SHA:0:7}, no tracked modifications"
g show-ref -q --verify refs/heads/master && die "a local 'master' branch already exists — refusing"
g remote get-url private >/dev/null 2>&1 && die "a remote named 'private' already exists — refusing"
g remote get-url origin >/dev/null 2>&1 || die "no 'origin' remote"
PUBLIC_PREEXISTS=0
if g remote get-url public >/dev/null 2>&1; then
  [[ "$(g remote get-url public)" == "$PUBLIC_URL" ]] || die "remote 'public' exists with a different URL"
  PUBLIC_PREEXISTS=1; info "remote 'public' already present with the expected URL (idempotent)"
fi
[[ "$(g config branch.droplet.remote || true)" == origin ]] || info "WARN droplet has no upstream on origin"
info "OK  no local master, no 'private' remote"

if [[ "$SIM" == 1 ]]; then
  info "SKIP deploy key + ssh alias checks (SIM)"; GIT_SSH_FOR_FETCH=""
  FETCH_URL="$PUBLIC_URL"
else
  [[ -f "$DEPLOY_KEY" && -f "$DEPLOY_KEY.pub" ]] || die "deploy key $DEPLOY_KEY(.pub) missing — generate it first"
  info "OK  deploy key present: $DEPLOY_KEY"
  GIT_SSH_FOR_FETCH="$(ssh_cmd)"
fi

ASIDE_TMP="$(mktemp -d)"; EXPORT_DIR="$ASIDE_TMP"
ASIDE="$ASIDE_TMP"   # dry-run scratch; real aside created on --apply

step "2. fetch public master (no refs/config changed) and pin ${PUB_SHA:0:7}"
if [[ -n "$GIT_SSH_FOR_FETCH" ]]; then
  run env GIT_SSH_COMMAND="$GIT_SSH_FOR_FETCH" git -C "$REPO" fetch -q --no-tags "$FETCH_URL" master
else
  run g fetch -q --no-tags "$FETCH_URL" master
fi
TIP="$(g rev-parse FETCH_HEAD)"
g cat-file -e "$PUB_SHA^{commit}" 2>/dev/null || die "expected public sha not present after fetch"
g merge-base --is-ancestor "$PUB_SHA" "$TIP" || die "expected public sha is not on public/master"
info "OK  public/master tip ${TIP:0:7}; pinned ${PUB_SHA:0:7} is on it$([[ $TIP != "$PUB_SHA" ]] && echo ' (tip has newer commits — they are NOT applied here)')"

step "3. prove equivalence"
prove_equivalence "$ASIDE_TMP/deleted.z"
check_live_references "$ASIDE_TMP/deleted.z"

step "3b. python checks against an export of the NEW tree (live params/env symlinked in)"
mkdir -p "$ASIDE_TMP/tree"
g archive "$PUB_SHA" | tar -x -C "$ASIDE_TMP/tree"
for f in "$REPO"/.env* "$REPO"/config/params*.yaml; do
  t="$ASIDE_TMP/tree/${f#"$REPO"/}"
  [[ -e "$f" && ! -e "$t" ]] && ln -s "$f" "$t" || true
done
mkdir -p "$ASIDE_TMP/tree/data" "$ASIDE_TMP/tree/logs"
py_checks "$ASIDE_TMP/tree"
[[ $FAILS -eq 0 ]] || die "$FAILS check(s) failed against the new tree — not cutting over"

services_snapshot "$ASIDE_TMP/services.before"
[[ -s "$ASIDE_TMP/services.before" ]] && while read -r u a p; do info "service $u: $a pid=$p"; done < "$ASIDE_TMP/services.before"
if [[ "$SIM" != 1 ]] && grep -qv ' active ' "$ASIDE_TMP/services.before"; then
  die "a live service is not active before cutover — investigate first"
fi

if [[ $APPLY -ne 1 ]]; then
  step "DRY-RUN complete — all checks passed. Would now:"
  info "  aside dir $ASIDE_ROOT/cutover-aside-<ts> (0700): manifest + tarball of $(tr -cd '\0' < "$ASIDE_TMP/deleted.z" | wc -c | tr -d ' ') tracked files"
  [[ "$SIM" != 1 ]] && info "  back up $SSH_CONFIG, add 'Host $SSH_ALIAS' (HostName github.com, IdentityFile $DEPLOY_KEY, IdentitiesOnly yes)"
  info "  git remote add public $PUBLIC_URL; git fetch public master"
  info "  git remote rename origin private; git remote rename public origin"
  info "  git checkout -B master ${PUB_SHA:0:7}; git branch --set-upstream-to=origin/master master"
  info "  mkdir -p ${RUNTIME_DIRS[*]}; re-verify manifests, python checks, services/PIDs"
  info "re-run with --apply to perform it"
  exit 0
fi

# -------------------------------------------------------------------- apply --
step "4. aside dir + snapshots"
ASIDE="$ASIDE_ROOT/cutover-aside-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -m 700 "$ASIDE"; mkdir -m 700 "$ASIDE/tmp"
mv "$ASIDE_TMP/deleted.z" "$ASIDE/deleted.z"; mv "$ASIDE_TMP/services.before" "$ASIDE/services.before"
rm -rf "$ASIDE_TMP"; EXPORT_DIR=""; ASIDE_TMP="$ASIDE/tmp"
state_set REPO "$REPO"; state_set PRIV_SHA "$PRIV_SHA"; state_set PUB_SHA "$PUB_SHA"; state_set PUBLIC_URL "$PUBLIC_URL"
build_manifest "$ASIDE/before"
info "manifest: $(wc -l < "$ASIDE/before.strict" | tr -d ' ') secrets/params hashed, $(wc -l < "$ASIDE/before.inode" | tr -d ' ') runtime files, $(wc -l < "$ASIDE/before.untracked" | tr -d ' ') untracked"
(cd "$REPO" && tar --null -T "$ASIDE/deleted.z" -czf "$ASIDE/deleted-tracked.tgz")
info "tarred $(tr -cd '\0' < "$ASIDE/deleted.z" | wc -c | tr -d ' ') to-be-deleted tracked files -> $ASIDE/deleted-tracked.tgz"
APPLY_STARTED=1   # from here on any exit without DONE=1 rolls back

step "5a. ssh alias"
if [[ "$SIM" == 1 ]]; then info "SKIP (SIM)"
else
  if [[ -f "$SSH_CONFIG" ]]; then cp -p "$SSH_CONFIG" "$ASIDE/ssh_config.bak"; fi
  if grep -qE "^[[:space:]]*Host[[:space:]]+$SSH_ALIAS([[:space:]]|$)" "$SSH_CONFIG" 2>/dev/null; then
    info "Host $SSH_ALIAS already present (idempotent)"
  else
    state_set SSH_MODIFIED 1
    printf '\nHost %s\n    HostName github.com\n    User git\n    IdentityFile %s\n    IdentitiesOnly yes\n' \
      "$SSH_ALIAS" "$DEPLOY_KEY" >> "$SSH_CONFIG"
    chmod 600 "$SSH_CONFIG"; info "appended Host $SSH_ALIAS to $SSH_CONFIG"
  fi
  ids="$(ssh -G "$SSH_ALIAS" | awk '$1=="identityfile"{print $2}' | sort -u)"
  [[ "$ids" == "$DEPLOY_KEY" ]] \
    || die "ssh -G $SSH_ALIAS resolves identityfile(s) other than the new key: $(tr '\n' ' ' <<<"$ids")"
  info "OK  ssh -G $SSH_ALIAS -> single identity $(basename "$ids")"
fi
maybe_fail ssh

step "5b. add + fetch remote 'public'"
if [[ $PUBLIC_PREEXISTS -eq 0 ]]; then run g remote add public "$PUBLIC_URL"; state_set PUBLIC_ADDED 1; fi
run env GIT_SSH_COMMAND="ssh -o BatchMode=yes" git -C "$REPO" fetch -q --no-tags public master
g merge-base --is-ancestor "$PUB_SHA" refs/remotes/public/master || die "pinned sha not on public/master via the alias remote"
info "OK  fetched via $PUBLIC_URL"
maybe_fail remote

step "5c. rename remotes"
state_set RENAMED 1
run g remote rename origin private
run g remote rename public origin
info "OK  origin -> $(g remote get-url origin | sed 's#^.*[:/]\([^/]*/[^/]*\)$#\1#'), private -> $(g remote get-url private | sed 's#^.*[:/]\([^/]*/[^/]*\)$#\1#')"
maybe_fail rename

step "5d. switch working tree"
state_set MASTER_CREATED 1
run g checkout -q -B master "$PUB_SHA"
run g branch -q --set-upstream-to=origin/master master
for d in "${RUNTIME_DIRS[@]}"; do [[ -d "$REPO/$d" ]] || { run mkdir -p "$REPO/$d"; }; done
maybe_fail switch

step "6. verify"
FAILS=0
[[ "$(g rev-parse HEAD)" == "$PUB_SHA" && "$(g symbolic-ref --short HEAD)" == master ]] \
  && info "OK  master @ ${PUB_SHA:0:7}, upstream $(g rev-parse --abbrev-ref master@{u})" || fail "HEAD/branch wrong"
g show-ref -q --verify refs/heads/droplet && info "OK  local 'droplet' kept for rollback" || fail "droplet branch gone"
[[ -z "$(g status --porcelain --untracked-files=no)" ]] && info "OK  no tracked changes" || fail "tracked changes after switch"
left=0; while IFS= read -r -d '' p; do [[ -e "$REPO/$p" ]] && left=$((left+1)); done < "$ASIDE/deleted.z"
[[ $left -eq 0 ]] && info "OK  all excluded tracked files removed from working tree" || info "note: $left excluded paths still on disk (dirs with ignored files)"
build_manifest "$ASIDE/after"
compare_manifests "$ASIDE/before" "$ASIDE/after"
check_runtime_dirs
py_checks "$REPO"
services_snapshot "$ASIDE/services.after"
if cmp -s "$ASIDE/services.before" "$ASIDE/services.after"; then
  [[ "$SIM" == 1 ]] && info "SKIP services/PIDs (SIM)" || info "OK  services active, MainPIDs unchanged"
else fail "service state/PID changed:"; diff "$ASIDE/services.before" "$ASIDE/services.after" >&2 || true; fi
maybe_fail verify
[[ $FAILS -eq 0 ]] || die "$FAILS verification check(s) failed"
DONE=1

step "CUTOVER COMPLETE"
info "aside (keep until confident): $ASIDE   rollback: $0 --rollback $ASIDE"
info "No service restart needed: running bots keep their loaded code; the next restart loads identical files."
info "Future deploys:  cd $REPO && git fetch origin && git merge --ff-only origin/master"
info "Then restart legs ONLY via tools/leg_safe_restart.py <instance>."
info "tools/deploy_untrack_params.sh defaults to origin/droplet + branch droplet on ${PUB_SHA:0:7};"
info "  pass TARGET=origin/master BRANCH=master until the fix commit is merged."
info "Old remote kept as 'private' (old deploy key). Remove later: git remote remove private"
