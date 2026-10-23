"""Generate a static dashboard.html for a live bot LEG.

Reads a leg's state.db + heartbeat + jsonl, writes reports/dashboard[_<instance>].html.
Designed to run from a cron every ~60s so the HTML stays fresh.

Per-leg aware: `--instance <name>` picks that leg's files AND its HALT flag.
A leg that self-halts writes data/HALT_<instance> (not the shared data/HALT), so
the dashboard reports THAT leg as HALTED via exchange.env.halt_source. Default
instance is v1 (unchanged output path reports/dashboard.html).

Third-party touches are exchange.env (local module) for the HALT semantics and
PyYAML, used only to read the leg's configured kill-switch fraction rather than
hardcode a risk constant here. Everything else is stdlib. Auto-refreshes in
browser every 30s.
"""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from exchange.env import halt_source, is_halted  # per-leg HALT semantics

LOCAL_TZ = ZoneInfo("Asia/Bangkok")
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_FLAG = REPO_ROOT / "confirm_mainnet.lock"

# Per-leg file names, mirroring bot.INSTANCE_PROFILES. Kept as a local map so the
# dashboard stays lightweight (no bot/ccxt import). Default v1 preserves the
# legacy output path exactly.
INSTANCE_FILES: dict[str, dict[str, str]] = {
    "v1":            {"state": "state.db",              "hb": "heartbeat",              "log": "bot.jsonl",           "out": "dashboard.html",               "cfg": "params.yaml"},
    "donchian":      {"state": "state_donchian.db",     "hb": "heartbeat_donchian",     "log": "donchian.jsonl",      "out": "dashboard_donchian.html",      "cfg": "params_donchian.yaml"},
    "cnh_short":     {"state": "state_cnh_short.db",     "hb": "heartbeat_cnh_short",     "log": "cnh_short.jsonl",     "out": "dashboard_cnh_short.html",     "cfg": "params_cnh_hybrid_short.yaml"},
    "cnh_short_sol": {"state": "state_cnh_short_sol.db", "hb": "heartbeat_cnh_short_sol", "log": "cnh_short_sol.jsonl", "out": "dashboard_cnh_short_sol.html", "cfg": "params_cnh_hybrid_short_sol.yaml"},
}

# Module-level path handles; reassigned per --instance in main().
STATE_DB = REPO_ROOT / "data" / "state.db"
HEARTBEAT = REPO_ROOT / "data" / "heartbeat"
JSONL = REPO_ROOT / "logs" / "bot.jsonl"
OUT = REPO_ROOT / "reports" / "dashboard.html"


def _apply_instance(instance: str) -> None:
    """Point the module path handles at `instance`'s leg files."""
    global STATE_DB, HEARTBEAT, JSONL, OUT
    f = INSTANCE_FILES[instance]
    STATE_DB = REPO_ROOT / "data" / f["state"]
    HEARTBEAT = REPO_ROOT / "data" / f["hb"]
    JSONL = REPO_ROOT / "logs" / f["log"]
    OUT = REPO_ROOT / "reports" / f["out"]


# Per-leg params YAMLs (untracked; see config/README.md). Module-level so tests
# can point it at tests/fixtures/.
CONFIG_DIR = REPO_ROOT / "config"


def read_kill_fraction(instance: str) -> float | None:
    """The leg's CONFIGURED kill-switch fraction, or None if it can't be read.

    Read, never hardcoded. This dashboard used to compute its kill line as
    `deploy_start_equity * 0.82`, which was the deploy-era rule; the live switch
    moved to `principal * kill_switch_equity_fraction` and the page kept printing the old one for
    months. A dashboard that carries its own copy of a risk constant will
    eventually disagree with the bot that enforces it, and it disagrees
    silently.

    None (rather than a default) on any failure, so the page renders an honest
    dash instead of a made-up floor -- an invented kill level is worse than no
    kill level, because you would plan around it.
    """
    cfg = CONFIG_DIR / INSTANCE_FILES[instance]["cfg"]
    try:
        with cfg.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        frac = float(data["deploy"]["kill_switch_equity_fraction"])
    except Exception:
        return None
    return frac if 0.0 < frac < 1.0 else None


def to_local(iso_ts: str) -> str:
    """Convert UTC ISO (aware or naive) to Bangkok display string."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_ts


def epoch_to_local(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def _live_principal(c: sqlite3.Connection, meta: dict) -> float | None:
    """P = principal_base + Sigma(USDT ledger), derived on the SAME connection.

    Deliberately NOT meta['principal_anchor']: principal.py:52 documents that
    key as a *cache* of this sum, refreshed hourly by the bot's reconcile. A
    transfer lands in the ledger first, so between the ledger write and the next
    cache refresh the cached anchor is stale -- and a dashboard rendering from
    it would report against a different quantity than the kill switch enforces.
    That is exactly the defect PR #25's review round found in monitor.py, whose
    fix was to derive from the same connection and the same ledger read. Same
    remedy here.

    None when the leg is not initialised, mirroring principal.is_initialized()
    (META_SOURCE unset) and principal.get_principal(). principal_base is
    legitimately 0.0 on the live legs, so it must NOT be treated as missing.
    """
    if meta.get("principal_source") is None:
        return None
    try:
        base = float(meta.get("principal_base", 0) or 0)
        row = c.execute(
            "SELECT COALESCE(SUM(income_usd), 0) FROM principal_ledger WHERE asset=?",
            ("USDT",),
        ).fetchone()
    except sqlite3.Error:
        # Pre-Part-C leg DBs have no principal_ledger table at all.
        return None
    return base + (float(row[0]) if row else 0.0)


def read_db() -> dict:
    if not STATE_DB.exists():
        return {"meta": {}, "fills": [], "events": [], "principal": None}
    with sqlite3.connect(STATE_DB, timeout=5.0) as c:
        meta_rows = c.execute("SELECT key, value FROM meta").fetchall()
        meta = {k: v for k, v in meta_rows}
        fills = c.execute(
            "SELECT ts, side, qty, price, pnl_usd, reason, equity_after "
            "FROM fills ORDER BY id DESC LIMIT 50"
        ).fetchall()
        events = c.execute(
            "SELECT ts, level, kind, msg FROM events ORDER BY id DESC LIMIT 50"
        ).fetchall()
        principal = _live_principal(c, meta)
    return {"meta": meta, "fills": fills, "events": events,
            "principal": principal}


def read_log_tail(n: int = 30) -> list[dict]:
    if not JSONL.exists():
        return []
    with open(JSONL, "rb") as f:
        try:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, 200_000)
            f.seek(size - chunk)
            data = f.read().decode("utf-8", errors="replace")
        except Exception:
            return []
    lines = [ln for ln in data.splitlines() if ln.strip()]
    out: list[dict] = []
    for ln in lines[-n:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def render_html(db: dict, logs: list[dict], instance: str = "v1") -> str:
    now_utc = time.time()
    hb_age = (now_utc - HEARTBEAT.stat().st_mtime) if HEARTBEAT.exists() else None
    alive = hb_age is not None and hb_age < 60
    # Per-leg HALT: this leg is halted when the GLOBAL data/HALT exists OR its
    # own data/HALT_<instance> exists. halt_source names which one fired.
    halt_by = halt_source(instance)
    halted = is_halted(instance)
    if halt_by == "global":
        halt_note = "GLOBAL data/HALT (manual stop-all)"
    elif halt_by is not None:
        halt_note = f"self data/HALT_{instance}"
    else:
        halt_note = ""
    mainnet_locked = LOCK_FLAG.exists()

    meta = db["meta"]
    deploy_start_eq = float(meta.get("deploy_start_equity", 0) or 0)
    deploy_start_ts = meta.get("deploy_start_ts", "")
    # P = net deposited principal, derived live from the ledger by read_db()
    # (NOT the cached meta.principal_anchor -- see _live_principal).
    # This is the denominator the LIVE kill switch uses (bot._check_kill_switch).
    principal_anchor = db.get("principal")

    last_eq_after = None
    last_fill_side = None
    if db["fills"]:
        for row in db["fills"]:
            if row[6] is not None:
                last_eq_after = row[6]
                last_fill_side = row[1]
                break

    cur_equity = last_eq_after if last_eq_after is not None else deploy_start_eq

    # Measure P&L against PRINCIPAL, not against deploy-start equity.
    # deploy_start_equity is a RAW snapshot frozen at first boot, so every later
    # deposit moves the numerator and leaves the basis behind. On 2026-09-09 this
    # page showed v1 UP while quoting the bot's own log line saying DOWN further
    # down the SAME page (a later top-up counted as profit). donchian was worse:
    # a small first-boot snapshot vs its full principal rendered a gain several
    # times the leg's real one.
    # Same defect as PR #25 (monitor.py) and PR #30 (bot.py boot line); this was
    # the fourth reader of the pair.
    have_principal = principal_anchor is not None and principal_anchor > 0
    if have_principal:
        pnl_basis, pnl_basis_label = principal_anchor, "principal"
    else:
        # Pre-Part-C legs have no ledger. Fall back, but SAY SO -- printing an
        # unadjusted number bare is exactly what made this bug invisible.
        pnl_basis, pnl_basis_label = deploy_start_eq, "deploy start (NOT transfer-adjusted)"
    pnl_pct = ((cur_equity / pnl_basis) - 1) * 100 if pnl_basis else 0.0
    pnl_usd = cur_equity - pnl_basis if pnl_basis else 0.0

    # Kill line ONLY when a real principal exists. principal.breached() is
    # fail-safe -- `if principal is None or principal <= 0: return False` -- so
    # with no principal the bot enforces NO floor at all. Printing one computed
    # off deploy-start equity would repeat this whole bug's mistake: naming a
    # level the bot does not enforce. Otherwise the fraction comes from the
    # leg's config, against the same basis, so card and bot cannot disagree.
    kill_fraction = read_kill_fraction(instance)
    if kill_fraction is not None and have_principal:
        kill_threshold = pnl_basis * kill_fraction
        kill_note = f"{(kill_fraction - 1) * 100:.1f}% from {pnl_basis_label}"
    else:
        kill_threshold = None
        kill_note = ("kill switch DISABLED: no principal anchor" if not have_principal
                     else "config unreadable")

    if alive:
        status_label, status_class = "ALIVE", "ok"
    else:
        status_label, status_class = "DEAD", "bad"
    if halted:
        status_label, status_class = "HALTED", "warn"

    last_msg = logs[-1]["msg"] if logs else "(no log yet)"
    last_msg_ts = to_local(logs[-1]["ts"]) if logs else ""

    interesting = [e for e in logs if "heartbeat" not in (e.get("msg", "").lower())]
    recent_events = list(reversed(interesting[-20:]))

    errors = [e for e in logs if e.get("level") in ("ERROR", "WARNING")]
    recent_errors = list(reversed(errors[-10:]))

    fills_html_rows = []
    for ts, side, qty, price, pnl, reason, eq in db["fills"][:20]:
        pnl_cell = "—" if pnl is None else f"${pnl:+.2f}"
        eq_cell = "—" if eq is None else f"${eq:.2f}"
        fills_html_rows.append(
            f"<tr><td>{to_local(ts)}</td><td>{html.escape(side)}</td>"
            f"<td>{qty:.4f}</td><td>${price:,.2f}</td>"
            f"<td>{pnl_cell}</td><td>{html.escape(reason or '')}</td>"
            f"<td>{eq_cell}</td></tr>"
        )
    fills_table = "\n".join(fills_html_rows) or (
        '<tr><td colspan="7" style="text-align:center;opacity:.5">No fills yet</td></tr>'
    )

    events_html_rows = []
    for ev in recent_events:
        lvl = ev.get("level", "INFO")
        cls = {"ERROR": "row-err", "WARNING": "row-warn"}.get(lvl, "")
        events_html_rows.append(
            f"<tr class='{cls}'><td>{to_local(ev.get('ts',''))}</td>"
            f"<td>{html.escape(lvl)}</td>"
            f"<td>{html.escape(ev.get('logger',''))}</td>"
            f"<td>{html.escape(ev.get('msg',''))}</td></tr>"
        )
    events_table = "\n".join(events_html_rows) or (
        '<tr><td colspan="4" style="text-align:center;opacity:.5">No events yet</td></tr>'
    )

    err_html_rows = []
    for ev in recent_errors:
        lvl = ev.get("level", "INFO")
        cls = {"ERROR": "row-err", "WARNING": "row-warn"}.get(lvl, "")
        err_html_rows.append(
            f"<tr class='{cls}'><td>{to_local(ev.get('ts',''))}</td>"
            f"<td>{html.escape(lvl)}</td>"
            f"<td>{html.escape(ev.get('msg',''))}</td></tr>"
        )
    err_table = "\n".join(err_html_rows) or (
        '<tr><td colspan="3" style="text-align:center;opacity:.5">No errors or warnings</td></tr>'
    )

    pnl_class = "ok" if pnl_pct >= 0 else "bad"
    pnl_sign = "+" if pnl_pct >= 0 else ""
    # An em dash beats a fabricated number on both of these: a made-up kill
    # level is something you would plan around.
    # "-$26.04", not "$-26.04" -- the old template put the sign outside the
    # dollar sign, which only ever showed up once a leg was actually down.
    pnl_usd_text = f"{'+' if pnl_usd >= 0 else '-'}${abs(pnl_usd):,.2f}"
    principal_text = f"${principal_anchor:,.2f}" if have_principal else "—"
    kill_threshold_text = (f"${kill_threshold:,.2f}" if kill_threshold is not None
                           else "—")
    generated_at = epoch_to_local(now_utc)
    hb_text = fmt_age(hb_age) if hb_age is not None else "never"
    status_sub = f"HALT: {halt_note}" if halted else f"Heartbeat {hb_text}"
    deploy_text = to_local(deploy_start_ts) if deploy_start_ts else "—"
    mode_text = "MAINNET" + (" 🔒" if mainnet_locked else "")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<meta http-equiv="refresh" content="30" />
<title>snapback-btc dashboard</title>
<style>
  :root {{
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e;
    --ok: #3fb950; --warn: #d29922; --bad: #f85149;
  }}
  * {{ box-sizing: border-box }}
  body {{
    margin: 0; padding: 16px; background: var(--bg); color: var(--text);
    font: 14px/1.4 -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
  }}
  h1 {{ font-size: 18px; margin: 0 0 4px 0 }}
  .sub {{ color: var(--muted); font-size: 12px; margin-bottom: 16px }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 12px }}
  .card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
  }}
  .card .lbl {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .03em }}
  .card .val {{ font-size: 22px; font-weight: 600; margin-top: 4px }}
  .badge {{
    display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: 600;
    font-size: 13px;
  }}
  .ok   {{ color: var(--ok) }}
  .warn {{ color: var(--warn) }}
  .bad  {{ color: var(--bad) }}
  .badge.ok   {{ background: rgba(63,185,80,.15); color: var(--ok) }}
  .badge.warn {{ background: rgba(210,153,34,.15); color: var(--warn) }}
  .badge.bad  {{ background: rgba(248,81,73,.15); color: var(--bad) }}
  section {{ margin-top: 24px }}
  h2 {{ font-size: 14px; color: var(--muted); text-transform: uppercase;
        letter-spacing: .05em; margin: 0 0 8px 0 }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel);
           border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
           font-size: 12.5px }}
  th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--border); text-align: left;
            vertical-align: top }}
  th {{ background: #0a0f17; color: var(--muted); font-weight: 500; font-size: 11px;
        text-transform: uppercase; letter-spacing: .04em }}
  tr:last-child td {{ border-bottom: none }}
  .row-err  td {{ color: var(--bad) }}
  .row-warn td {{ color: var(--warn) }}
  .footer {{ color: var(--muted); font-size: 11px; margin-top: 24px; text-align: center }}
  code {{ background: rgba(255,255,255,.05); padding: 1px 5px; border-radius: 4px }}
</style>
</head>
<body>

<h1>snapback-btc — live dashboard <span style="opacity:.6;font-size:14px">[{html.escape(instance)}]</span></h1>
<div class="sub">All timestamps shown in Bangkok time (GMT+7). Auto-refresh every 30s.</div>

<div class="grid">
  <div class="card">
    <div class="lbl">Status</div>
    <div class="val"><span class="badge {status_class}">{status_label}</span></div>
    <div class="sub" style="margin:6px 0 0">{html.escape(status_sub)}</div>
  </div>
  <div class="card">
    <div class="lbl">Mode</div>
    <div class="val">{mode_text}</div>
    <div class="sub" style="margin:6px 0 0">Mainnet lock: {"✓" if mainnet_locked else "—"}</div>
  </div>
  <div class="card">
    <div class="lbl">Equity</div>
    <div class="val">${cur_equity:,.2f}</div>
    <div class="sub" style="margin:6px 0 0">Start: ${deploy_start_eq:,.2f} · Principal: {principal_text}</div>
  </div>
  <div class="card">
    <div class="lbl">P&amp;L</div>
    <div class="val {pnl_class}">{pnl_sign}{pnl_pct:.2f}%</div>
    <div class="sub" style="margin:6px 0 0">{pnl_usd_text} USDT vs {html.escape(pnl_basis_label)}</div>
  </div>
  <div class="card">
    <div class="lbl">Kill switch at</div>
    <div class="val">{kill_threshold_text}</div>
    <div class="sub" style="margin:6px 0 0">{html.escape(kill_note)}</div>
  </div>
  <div class="card">
    <div class="lbl">Deploy started</div>
    <div class="val" style="font-size:14px">{deploy_text}</div>
    <div class="sub" style="margin:6px 0 0">{"Last fill: " + html.escape(last_fill_side or "—") if last_fill_side else "No fills yet"}</div>
  </div>
</div>

<section>
  <h2>Last event</h2>
  <div class="card">
    <div class="sub" style="margin-bottom:4px">{last_msg_ts}</div>
    <div style="font-family:'SF Mono', Menlo, monospace; font-size: 12.5px">{html.escape(last_msg)}</div>
  </div>
</section>

<section>
  <h2>Recent activity (last 20, excluding heartbeats)</h2>
  <table>
    <thead><tr><th>Time (GMT+7)</th><th>Level</th><th>Logger</th><th>Message</th></tr></thead>
    <tbody>{events_table}</tbody>
  </table>
</section>

<section>
  <h2>Fills</h2>
  <table>
    <thead><tr><th>Time (GMT+7)</th><th>Side</th><th>Qty BTC</th><th>Price</th>
      <th>P&amp;L</th><th>Reason</th><th>Equity after</th></tr></thead>
    <tbody>{fills_table}</tbody>
  </table>
</section>

<section>
  <h2>Errors &amp; warnings</h2>
  <table>
    <thead><tr><th>Time (GMT+7)</th><th>Level</th><th>Message</th></tr></thead>
    <tbody>{err_table}</tbody>
  </table>
</section>

<div class="footer">
  Generated {generated_at} Bangkok · auto-refresh 30s ·
  source: <code>tools/build_dashboard.py</code>
</div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a live-bot leg dashboard.")
    ap.add_argument("--instance", default="v1", choices=list(INSTANCE_FILES),
                    help="Which leg to render (picks its state.db/heartbeat/jsonl "
                         "and its per-leg HALT flag). Default v1.")
    args = ap.parse_args()
    _apply_instance(args.instance)
    db = read_db()
    logs = read_log_tail(150)
    html_out = render_html(db, logs, args.instance)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_out, encoding="utf-8")
    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
