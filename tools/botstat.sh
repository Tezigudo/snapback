#!/usr/bin/env bash
# Snapback quick-status — no Claude tokens, no API calls.
# Usage:
#   ./tools/botstat.sh
#   alias bots='/path/to/snapback-btc/tools/botstat.sh'   # then just type: bots
#
# Or drop a copy in your SwiftBar / xbar plugin dir to get a menu-bar widget
# (the first line becomes the label, rest becomes the dropdown).
#
# To add a new leg: append name + heartbeat + state.db + log to the LEGS_*
# arrays below. Everything else loops — TTY render, SwiftBar render, and the
# remote-side heartbeat / outbox / gate scans all iterate the same lists.

set -u
DROPLET="${SNAPBACK_DROPLET:-snapback-droplet}"

# ─── Legs definition — single source of truth ───
# Parallel arrays: LEGS_NAMES[i] pairs with LEGS_HB[i] / LEGS_DB[i] / LEGS_LOG[i].
#
# Roster must match the LIVE systemd units. cnh_short was retired 2026-07-12
# (archive/cnh_short_retired_20260712): no service, no .env.cnh_short, and its
# heartbeat is frozen — it rendered a permanent 🔴 that trained the eye to
# ignore red. sol_supertrend has been real money since 2026-07-27 and was
# missing entirely, so the one tool used to eyeball the book at a glance was
# blind to a funded leg. monitor.py and daily_digest.py were corrected on
# 2026-08-05; this file was missed in that pass.
LEGS_NAMES=(v1 donchian sol_supertrend)
LEGS_HB=(
  /root/snapback-btc/data/heartbeat
  /root/snapback-btc/data/heartbeat_donchian
  /root/snapback-btc/data/heartbeat_sol_supertrend
)
LEGS_DB=(
  /root/snapback-btc/data/state.db
  /root/snapback-btc/data/state_donchian.db
  /root/snapback-btc/data/state_sol_supertrend.db
)
LEGS_LOG=(
  /root/snapback-btc/logs/bot.jsonl
  /root/snapback-btc/logs/donchian.jsonl
  /root/snapback-btc/logs/sol_supertrend.jsonl
)

# Build space-separated "name=path" specs that the remote shell can split.
# We pass them as env vars on the ssh command line rather than escaping $-vars
# through nested heredocs.
hb_spec=""; db_spec=""; log_spec=""
for i in "${!LEGS_NAMES[@]}"; do
  hb_spec+="${LEGS_NAMES[i]}=${LEGS_HB[i]} "
  db_spec+="${LEGS_NAMES[i]}=${LEGS_DB[i]} "
  log_spec+="${LEGS_NAMES[i]}=${LEGS_LOG[i]} "
done

# ─── Remote snapshot (one SSH call) ───
# Env vars set on the remote command line; heredoc is single-quoted so local
# $-interpolation is off and the remote script sees its own $vars.
REMOTE_OUT=$(ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no "$DROPLET" \
  "HB_SPEC=\"$hb_spec\" DB_SPEC=\"$db_spec\" LOG_SPEC=\"$log_spec\" bash -s" <<'REMOTE' 2>/dev/null
NOW=$(date +%s)

for kv in $HB_SPEC; do
  name="${kv%%=*}"; f="${kv##*=}"
  if [ -e "$f" ]; then
    age=$(( NOW - $(stat -c %Y "$f") ))
    echo "HB ${name} ${age}"
  else
    echo "HB ${name} -1"
  fi
done

# Mode per leg: scan running bot procs. A live leg has --instance NAME and
# NO --dry-run flag; dry has --dry-run; absent process = down.
for kv in $HB_SPEC; do
  name="${kv%%=*}"
  pline=$(ps -eo args 2>/dev/null | grep -- "--instance ${name}" | grep -v grep | head -1)
  if   [ -z "$pline" ];                                then echo "MODE ${name} down"
  elif printf '%s' "$pline" | grep -q -- '--dry-run';  then echo "MODE ${name} dry"
  else                                                      echo "MODE ${name} live"
  fi
done

python3 - <<PYEOF 2>/dev/null
import os, sqlite3
for entry in os.environ.get("DB_SPEC", "").split():
    name, db = entry.split("=", 1)
    if not os.path.exists(db):
        print(f"OB {name} -1"); print(f"FILLS {name} -1"); continue
    try:
        c = sqlite3.connect(db)
        n = c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        print(f"OB {name} {n}")
        try:
            fz = c.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
        except Exception:
            fz = -1
        print(f"FILLS {name} {fz}")
        c.close()
    except Exception:
        print(f"OB {name} -1"); print(f"FILLS {name} -1")
PYEOF

for kv in $LOG_SPEC; do
  name="${kv%%=*}"; f="${kv##*=}"
  line=$(grep '"msg": "gates:' "$f" 2>/dev/null | tail -1)
  if [ -n "$line" ]; then
    ts=$(printf "%s" "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['ts'][:19])")
    msg=$(printf "%s" "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['msg'])")
    printf "GATE %s|%s|%s\n" "$name" "$ts" "$msg"
  fi
done
REMOTE
)

if [ -z "$REMOTE_OUT" ]; then
  echo "🔴 droplet | color=red"
  echo "---"
  echo "SSH to $DROPLET failed (is the ssh alias set up? see DEPLOY.md)"
  exit 1
fi

# ─── Local: live BTC price (no auth) ───
BTC=$(curl -s --max-time 3 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["price"])' 2>/dev/null || echo "?")

# ─── Parse remote output into HB[] / OB[] ───
declare -A HB OB FILLS MODE
while IFS= read -r line; do
  case "$line" in
    "HB "*)    read -r _ n v <<<"$line"; HB[$n]=$v ;;
    "OB "*)    read -r _ n v <<<"$line"; OB[$n]=$v ;;
    "FILLS "*) read -r _ n v <<<"$line"; FILLS[$n]=$v ;;
    "MODE "*)  read -r _ n v <<<"$line"; MODE[$n]=$v ;;
  esac
done <<<"$REMOTE_OUT"

# ─── Render helpers ───
icon_for_age() {
  local a=$1
  if   [ "$a" -lt 0 ];   then printf "⚫"
  elif [ "$a" -le 30 ];  then printf "🟢"
  elif [ "$a" -le 120 ]; then printf "🟡"
  else                        printf "🔴"
  fi
}

swiftbar_color_for_icon() {
  case "$1" in
    🟢) printf "#10b981" ;;
    🟡) printf "#f59e0b" ;;
    ⚫) printf "#6b7280" ;;
    *)  printf "#ef4444" ;;
  esac
}

# TTY vs SwiftBar render mode
if [ -t 1 ]; then
  # ─── Terminal — ANSI colors, no SwiftBar `| color=...` syntax ───
  R=$'\033[0m'; B=$'\033[1m'
  GREEN=$'\033[38;5;42m'; RED=$'\033[38;5;203m'; YEL=$'\033[38;5;221m'
  CYAN=$'\033[38;5;81m'; GRAY=$'\033[38;5;245m'; BLUE=$'\033[38;5;111m'

  # Header line: icon + name + age per leg, then BTC
  printf "%s" "$B"
  for n in "${LEGS_NAMES[@]}"; do
    icon=$(icon_for_age "${HB[$n]:-9999}")
    printf "%s %s %ss  " "$icon" "$n" "${HB[$n]:-?}"
  done
  printf "%sBTC \$%s%s\n" "$CYAN" "$BTC" "$R"

  # Per-leg detail lines (mode + heartbeat + fills + outbox)
  for n in "${LEGS_NAMES[@]}"; do
    case "${MODE[$n]:-?}" in
      live) mt="${GREEN}LIVE${R}" ;;
      dry)  mt="${GRAY}dry ${R}" ;;
      down) mt="${RED}DOWN${R}" ;;
      *)    mt="${GRAY}?   ${R}" ;;
    esac
    printf "%s%-14s%s %s · hb %ss · fills %s · outbox %s\n" \
      "$BLUE" "$n" "$R" "$mt" "${HB[$n]:-?}" "${FILLS[$n]:-?}" "${OB[$n]:-?}"
  done
  echo
  printf "%sStrategy gates — what each leg is waiting on%s\n" "$B" "$R"
  while IFS= read -r line; do
    [[ "$line" != GATE* ]] && continue
    rest=${line#GATE }
    name=${rest%%|*}; rest=${rest#*|}
    ts=${rest%%|*}; msg=${rest#*|}

    # Parse "long waiting on X, Y | short waiting on Z"
    long_part="${msg#*long waiting on }"; long_part="${long_part%% \| short*}"
    short_part="${msg#*short waiting on }"

    printf "  %s[%-14s]%s last change %s%s%s\n" "$BLUE" "$name" "$R" "$GRAY" "$ts" "$R"
    # LONG = green label, waiting items = yellow
    printf "    %sLONG%s  waiting on: " "$GREEN" "$R"
    IFS=, read -ra items <<<"$long_part"
    for ((i=0; i<${#items[@]}; i++)); do
      [ $i -gt 0 ] && printf ", "
      printf "%s%s%s" "$YEL" "$(echo "${items[i]}" | sed 's/^ *//')" "$R"
    done
    echo
    # SHORT = red label, waiting items = yellow
    printf "    %sSHORT%s waiting on: " "$RED" "$R"
    IFS=, read -ra items <<<"$short_part"
    for ((i=0; i<${#items[@]}; i++)); do
      [ $i -gt 0 ] && printf ", "
      printf "%s%s%s" "$YEL" "$(echo "${items[i]}" | sed 's/^ *//')" "$R"
    done
    echo
  done <<<"$REMOTE_OUT"
  exit 0
fi

# ─── SwiftBar mode (existing syntax with | color=...) ───

# Color the menu-bar label by the worst leg so red/yellow jumps out at a glance.
worst_age=0
for n in "${LEGS_NAMES[@]}"; do
  age=${HB[$n]:-9999}
  [ "$age" -gt "$worst_age" ] && worst_age=$age
done
if   [ "$worst_age" -le 30 ];  then label_color="#10b981"   # green
elif [ "$worst_age" -le 120 ]; then label_color="#f59e0b"   # amber
else                                label_color="#ef4444"   # red
fi

# Compact BTC: $77.2k instead of $77250.09
btc_short=$(printf "%.1fk" "$(echo "$BTC / 1000" | bc -l)")

# Menu-bar label: icon + age per leg, then short BTC
label=""
for n in "${LEGS_NAMES[@]}"; do
  icon=$(icon_for_age "${HB[$n]:-9999}")
  label+="${icon} ${HB[$n]:-?}s "
done
printf "%s\$%s | color=%s font=Menlo size=12\n" "$label" "$btc_short" "$label_color"
echo "---"

# Dropdown — colored per leg, with heartbeat age + outbox
for n in "${LEGS_NAMES[@]}"; do
  icon=$(icon_for_age "${HB[$n]:-9999}")
  c=$(swiftbar_color_for_icon "$icon")
  printf "%-14s [%s] hb %ss · fills %s · outbox %s pending | color=%s font=Menlo\n" \
    "$n" "${MODE[$n]:-?}" "${HB[$n]:-?}" "${FILLS[$n]:-?}" "${OB[$n]:-?}" "$c"
done
printf "BTC \$%s | color=#38bdf8 font=Menlo\n" "$BTC"

echo "---"
echo "Last gate change | color=#94a3b8 size=11"
while IFS= read -r line; do
  case "$line" in
    "GATE "*)
      rest=${line#GATE }
      name=${rest%%|*}; rest=${rest#*|}
      ts=${rest%%|*}; msg=${rest#*|}
      printf "%-14s %s | color=#cfd6e8 font=Menlo size=11\n" "$name" "$ts"
      # `|` is SwiftBar's param separator — replace with `/` in the msg.
      printf "  %s | color=#94a3b8 font=Menlo size=10 length=80\n" "${msg//|//}"
      ;;
  esac
done <<<"$REMOTE_OUT"

echo "---"
echo "Refresh | refresh=true"
