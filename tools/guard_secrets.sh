#!/usr/bin/env bash
# Reject staged additions that contain the droplet IP, the owner's email, or
# common key material. IP/email patterns live in an UNTRACKED file
# (.guard_patterns, one ERE per line, '#' comments) so they are never committed.
# Setup: cp .guard_patterns.example "$(git rev-parse --git-common-dir)/guard_patterns"
#        (then fill in; ONE copy inside .git serves every worktree of the clone).
#        A worktree-local .guard_patterns, if present, takes precedence.
#        git config core.hooksPath .githooks
# Fails CLOSED: a missing patterns file, an invalid regex or a failed git diff
# blocks the commit. Escape hatch for the missing file only:
#        GUARD_SECRETS_ALLOW_MISSING=1 git commit ...   (built-in key checks still run)
# Binary files are not scanned (git diff shows no content for them).
set -uo pipefail
root=$(git rev-parse --show-toplevel)
patterns_file="$root/.guard_patterns"
if [[ ! -f "$patterns_file" ]]; then
  common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || git rev-parse --git-common-dir)
  patterns_file="$common/guard_patterns"
fi

builtin=(
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'
  '(API_?KEY|API_?SECRET|SECRET)[A-Za-z_]*[=:"'"'"' ]+[0-9a-fA-F]{64}'
  '[0-9a-fA-F]{64}.{0,20}(API_?KEY|SECRET)'
)
local_pats=()
if [[ -f "$patterns_file" ]]; then
  while IFS= read -r l || [[ -n "$l" ]]; do
    l=${l%$'\r'}
    l=${l#"${l%%[![:space:]]*}"}
    l=${l%"${l##*[![:space:]]}"}
    [[ -z "$l" || "$l" == \#* ]] && continue
    if [[ "$l" == \<*\> ]]; then
      echo "guard_secrets: WARNING unfilled placeholder in .guard_patterns (ignored); fill it in or delete the line" >&2
      continue
    fi
    local_pats+=("$l")
  done < "$patterns_file"
elif [[ "${GUARD_SECRETS_ALLOW_MISSING:-0}" == "1" ]]; then
  echo "guard_secrets: WARNING no .guard_patterns (IP/email checks skipped via GUARD_SECRETS_ALLOW_MISSING=1)" >&2
else
  echo "guard_secrets: BLOCKED - $patterns_file is missing, so IP/email checks cannot run." >&2
  echo "  Create it once per clone: cp .guard_patterns.example \"\$(git rev-parse --git-common-dir)/guard_patterns\"  (then fill in the real values)." >&2
  echo "  Skip once (built-in key checks still run): GUARD_SECRETS_ALLOW_MISSING=1 git commit ..." >&2
  exit 1
fi

diff_out=$(git diff --cached --no-color -U0 --diff-filter=ACMR) || {
  echo "guard_secrets: BLOCKED - git diff failed, cannot scan staged changes" >&2
  exit 1
}
# Keep only hunk body additions (strip the leading '+'); file headers precede the first '@@'.
added=$(printf '%s\n' "$diff_out" | awk '
  /^diff --git / { inh = 0; next }
  /^@@ /         { inh = 1; next }
  inh && substr($0, 1, 1) == "+" { print substr($0, 2) }
')
[[ -z "$added" ]] && exit 0

fail=0
for p in "${builtin[@]}" ${local_pats[@]+"${local_pats[@]}"}; do
  n=$(printf '%s\n' "$added" | grep -c -E -e "$p" 2>/dev/null)
  rc=$?
  if [[ $rc -ge 2 ]]; then
    echo "guard_secrets: BLOCKED - a pattern is not a valid regex (pattern not shown); fix .guard_patterns" >&2
    fail=1
  elif [[ $rc -eq 0 && "$n" -gt 0 ]]; then
    # never echo the matched text, only the count
    echo "guard_secrets: BLOCKED - $n staged line(s) match a guarded pattern (not shown)" >&2
    fail=1
  fi
done
[[ $fail -eq 1 ]] && { echo "  Remove the sensitive value (use <droplet>/<alert-email> placeholders or .env)." >&2; exit 1; }
exit 0
