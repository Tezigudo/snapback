"""Push the bot's REAL futures account to investing-consolidate's dashboard.

Architecture (chosen 2026-05-31): the consolidate dashboard's Futures page is
fed from THIS droplet rather than from Fly. Why: the droplet already has a
static IP and the bot's futures-enabled key (it trades futures), so it can read
the bot's actual account with no new key, no Fly static-egress-IP cost, and the
account it reads IS the bot's — so the dashboard's wallet-vs-bot-equity
reconciliation is real instead of comparing a different account.

READ-ONLY BY CONSTRUCTION. This module only calls GET endpoints:
  - ex.fetch_balance({"type": "future"})   → /fapi/v3/account (balances)
  - ex.fetch_positions()                   → /fapi/v2/positionRisk (open pos)
  - ex.fetch_open_orders()                 → /fapi/v1/openOrders (resting SL/TP)
  - ex.fapiPrivateGetOpenAlgoOrders()      → /fapi/v1/openAlgoOrders (conditional SL/TP)
  - ex.fapiPrivateGetIncome(...)           → /fapi/v1/income (realized/funding/fees)
  - ex.fapiPrivateGetSymbolConfig()        → /fapi/v1/symbolConfig (leverage setting)
It NEVER places, cancels, or modifies orders. It does not import or touch the
trading loop. Run by cron, hourly. Fire-and-forget: any failure is logged and
the bot is entirely unaffected (this is a separate process from bot.py).

Posts to the consolidate API (bearer-authed, same token the bot uses for
bot-events):
  POST {CONSOLIDATE_API_URL}/futures/account-snapshot
  POST {CONSOLIDATE_API_URL}/futures/positions
  POST {CONSOLIDATE_API_URL}/futures/income

Config via .env (reused from consolidate_push.py):
  CONSOLIDATE_API_URL    e.g. https://<your-consolidate-host>
  CONSOLIDATE_API_TOKEN  the same Bearer token the web app uses
Plus the bot's existing BINANCE_API_KEY/SECRET + BINANCE_ENV (read by from_env).

Usage (local dev has uv; the DROPLET has no uv — use its venv directly):
  uv run python -m tools.consolidate_futures_push                      # local, hourly
  .venv/bin/python -m tools.consolidate_futures_push                   # droplet, hourly
  .venv/bin/python -m tools.consolidate_futures_push --income-days 365 # one-off backfill
  .venv/bin/python -m tools.consolidate_futures_push --dry-run         # print, don't POST
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 5.0
INCOME_PAGE_LIMIT = 1000      # Binance /fapi/v1/income max; also the server's max
INCOME_MAX_PAGES = 50         # backstop against a runaway pagination loop


def _config() -> tuple[str | None, str | None]:
    url = (os.environ.get("CONSOLIDATE_API_URL") or "").strip().rstrip("/")
    token = (os.environ.get("CONSOLIDATE_API_TOKEN") or "").strip()
    return (url or None), (token or None)


# ── pure payload mappers (unit-tested in tests/test_consolidate_futures_push.py) ──

def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build_account_payload(balance_info: dict[str, Any]) -> dict[str, float]:
    """Map ccxt fetch_balance()['info'] (raw fapi /v3/account) → ingest shape."""
    # A missing/garbled wallet balance must NOT be pushed as $0: raising makes
    # collect_all_accounts skip (and log) this account instead.
    try:
        float(balance_info["totalWalletBalance"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"totalWalletBalance unreadable: {e!r}") from e
    return {
        "walletBalanceUsd": _f(balance_info.get("totalWalletBalance")),
        "marginBalanceUsd": _f(balance_info.get("totalMarginBalance")),
        "unrealizedPnlUsd": _f(balance_info.get("totalUnrealizedProfit")),
        "availableBalanceUsd": _f(balance_info.get("availableBalance")),
    }


def build_bracket_map(
    orders: list[dict[str, Any]],
    reduce_sides: dict[str, str] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Map open trigger orders → {symbol: {"slPrice", "tpPrice"}}.

    Accepts BOTH shapes: ccxt fetch_open_orders() rows (type/stopPrice) and raw
    /fapi/v1/openAlgoOrders rows wrapped as {"info": row} (orderType/
    triggerPrice) — Binance migrated conditional orders to the algo system, so
    resting SL/TP no longer appear in /fapi/v1/openOrders at all (the same
    blind spot that broke the bot's reprotect loop on 2026-07-22).

    Keeps only reduce-only (or closePosition) STOP* / TAKE_PROFIT* orders.
    `reduce_sides` ({symbol: "BUY"|"SELL"}, from the live position's sign)
    keeps only orders whose side can actually reduce the CURRENT position —
    load-bearing on the algo endpoint, where orphaned brackets from closed
    trades keep resting (their side matches the OLD position, so this filter
    drops them instead of letting them overwrite the live bracket's prices).
    If a symbol has multiple surviving SL (or TP) orders, the last one wins.
    """
    reduce_sides = reduce_sides or {}
    out: dict[str, dict[str, float | None]] = {}
    for o in orders:
        info = o.get("info") or {}
        symbol = info.get("symbol") or o.get("symbol") or ""
        if not symbol:
            continue
        reduce_only = (
            str(info.get("reduceOnly")).lower() == "true"
            or str(info.get("closePosition")).lower() == "true"
        )
        if not reduce_only:
            continue
        want_side = reduce_sides.get(symbol)
        side = str(info.get("side") or o.get("side") or "").upper()
        if want_side and side and side != want_side:
            continue
        otype = str(info.get("type") or info.get("origType")
                    or info.get("orderType") or o.get("type") or "").upper()
        stop = _f(info.get("stopPrice") or info.get("triggerPrice")
                  or o.get("stopPrice") or o.get("triggerPrice"))
        if stop <= 0:
            continue
        entry = out.setdefault(symbol, {"slPrice": None, "tpPrice": None})
        if "TAKE_PROFIT" in otype:
            entry["tpPrice"] = stop
        elif "STOP" in otype:
            entry["slPrice"] = stop
    return out


def build_position_payloads(
    positions: list[dict[str, Any]],
    brackets: dict[str, dict[str, float | None]] | None = None,
    leverages: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Map ccxt fetch_positions() → ingest shape. Skips flat positions.

    Reads raw `info` for signed positionAmt + liq price (ccxt's unified
    `contracts` is unsigned), falling back to unified fields where present.
    `brackets` (from build_bracket_map) merges each symbol's resting SL/TP;
    absent when the open-orders fetch was skipped or failed → slPrice/tpPrice
    are None, which the server stores as NULL.

    `leverages` (from fetch_symbol_leverages) supplies the configured leverage
    per raw symbol: Binance's /fapi/v3 position payload dropped the `leverage`
    field entirely, so without it every position rendered as "0×". When the map
    is absent too, leverage is DERIVED from |notional| / positionInitialMargin
    (exact in cross mode: initial margin = notional / leverage by definition).
    """
    brackets = brackets or {}
    leverages = leverages or {}
    out: list[dict[str, Any]] = []
    for p in positions:
        info = p.get("info") or {}
        amt = _f(info.get("positionAmt"))
        if abs(amt) <= 0:
            continue
        liq = _f(info.get("liquidationPrice") or p.get("liquidationPrice"))
        symbol = info.get("symbol") or p.get("symbol") or ""
        b = brackets.get(symbol) or {}
        # Margin actually tied up by this position: isolated wallet when the
        # position is isolated, else the cross-mode initial margin.
        iso = _f(info.get("isolatedWallet"))
        pos_im = _f(info.get("positionInitialMargin") or info.get("initialMargin"))
        margin = iso if iso > 0 else pos_im
        lev = _f(info.get("leverage") or p.get("leverage") or 0)
        if lev <= 0:
            lev = _f(leverages.get(symbol))
        if lev <= 0 and margin > 0:
            notional = abs(_f(info.get("notional")))
            if notional <= 0:
                notional = abs(amt) * _f(info.get("markPrice") or p.get("markPrice"))
            lev = round(notional / margin, 1)
        out.append({
            "symbol": symbol,
            "positionSide": info.get("positionSide") or "BOTH",
            "positionAmt": amt,
            "entryPrice": _f(info.get("entryPrice") or p.get("entryPrice")),
            "markPrice": _f(info.get("markPrice") or p.get("markPrice")),
            "unrealizedPnlUsd": _f(info.get("unRealizedProfit") or p.get("unrealizedPnl")),
            "liquidationPrice": liq if liq > 0 else None,
            "leverage": lev,
            "marginUsd": margin if margin > 0 else None,
            "slPrice": b.get("slPrice"),
            "tpPrice": b.get("tpPrice"),
        })
    return out


def fetch_symbol_leverages(ex: Any) -> dict[str, float]:
    """{raw symbol → configured leverage} via GET /fapi/v1/symbolConfig.

    Read-only, one call per account. Needed because Binance removed `leverage`
    from the position payload (/fapi/v3), so the setting now lives only in the
    per-symbol account config. Best-effort: {} on any failure — the caller then
    falls back to deriving leverage from margin, and 0 renders as unknown.
    """
    try:
        cfg = ex.fapiPrivateGetSymbolConfig()
        return {str(r.get("symbol")): _f(r.get("leverage"))
                for r in cfg if r.get("symbol")}
    except Exception as e:  # noqa: BLE001 — cosmetic field, never block the push
        log.warning("symbolConfig fetch failed (%s) — leverage will be derived", e)
        return {}


def build_income_payloads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map raw /fapi/v1/income rows → ingest shape. `time` is Binance's stable
    event time, so re-pushes of the same row dedupe server-side."""
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            out.append({
                "tranId": int(r.get("tranId") or 0),
                "symbol": r.get("symbol") or "",
                "incomeType": str(r.get("incomeType") or ""),
                "incomeUsd": _f(r.get("income")),
                "asset": r.get("asset") or "USDT",
                "ts": int(r.get("time") or 0),
            })
        except (TypeError, ValueError) as e:
            log.warning("skipping malformed income row %s: %s", r, e)
    return out


# ── HTTP ────────────────────────────────────────────────────────────────────

def _post(url: str, token: str, path: str, body: dict[str, Any],
          timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{url}{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "snapback-btc-futures/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return {"status": int(resp.status), "body": json.loads(resp.read().decode("utf-8"))}
    except urllib.error.HTTPError as e:
        msg = f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"
        log.warning("futures push %s failed: %s", path, msg)
        return {"error": msg}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.info("futures push %s deferred (net): %s", path, e)
        return {"error": f"net: {e}"}


def _fetch_income(ex: Any, start_ms: int) -> list[dict[str, Any]]:
    """Paginate /fapi/v1/income forward from start_ms (ascending by time)."""
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    for _ in range(INCOME_MAX_PAGES):
        page = ex.fapiPrivateGetIncome({"startTime": cursor, "limit": INCOME_PAGE_LIMIT})
        if not page:
            break
        rows.extend(page)
        if len(page) < INCOME_PAGE_LIMIT:
            break
        cursor = int(page[-1].get("time") or cursor) + 1
    else:
        # Loop exhausted the page cap without a short final page → there may be
        # more income we didn't fetch. Loud so a truncated backfill is visible.
        log.warning(
            "income pagination hit the %d-page cap (%d rows) — older rows may be "
            "truncated; re-run with a smaller --income-days or raise INCOME_MAX_PAGES",
            INCOME_MAX_PAGES, len(rows),
        )
    return rows


# ── multi-account support (2026-07-25) ───────────────────────────────────────
# Until now this relay read ONE account (whatever base .env pointed at = v1's),
# so the donchian and sol_supertrend sub-accounts were invisible to the
# dashboard — their wallet balance never counted toward the futures total and
# their resting SL/TP brackets never appeared.
#
# Each leg keeps its own sub-account key in `.env.<instance>`, so we iterate the
# instances, build one client per account, and combine:
#
#   balances  → SUMMED. "Total futures equity" is the sum across sub-accounts.
#   income    → CONCATENATED. Safe: the server dedupes on Binance `tranId`,
#               which is unique per ledger row across accounts.
#   positions → CONCATENATED, each tagged with `account` = the leg instance.
#
# v1 and donchian BOTH trade BTCUSDT in separate sub-accounts, so two rows can
# share a symbol. The server keys futures_positions by (account, symbol)
# (consolidate migration 20), so both are stored and shown. Before that it was
# keyed by symbol alone and this relay had to drop the second BTC row.
ACCOUNT_INSTANCES: tuple[str, ...] = ("v1", "donchian", "sol_supertrend")


def _key_fingerprint() -> str:
    """Short hash of the ACTIVE BINANCE_API_KEY, to identify the account.

    Load-bearing: `.env.<instance>` overlays os.environ, so if an instance's file
    is missing or omits BINANCE_API_KEY, the previous instance's key silently
    stays active and we would read the SAME account twice — double-counting it in
    a SUMMED balance. Fingerprinting lets us skip duplicates instead.
    """
    import hashlib
    k = (os.environ.get("BINANCE_API_KEY") or "").strip()
    return hashlib.sha256(k.encode()).hexdigest()[:12] if k else ""


def _collect_one_account(income_days: int) -> dict[str, Any]:
    """Read one futures account (whatever env is currently active). Read-only."""
    from exchange.binance_client import BinanceClient

    client = BinanceClient.from_env()
    ex = client.ex

    balance = ex.fetch_balance({"type": "future"})
    account = build_account_payload(balance.get("info") or {})

    raw_positions = ex.fetch_positions()
    has_open = any(
        abs(_f((p.get("info") or {}).get("positionAmt"))) > 0 for p in raw_positions
    )
    # Side that would REDUCE each open position: short → BUY, long → SELL.
    # Feeds the bracket map's orphan filter (see build_bracket_map).
    reduce_sides: dict[str, str] = {}
    for p in raw_positions:
        info = p.get("info") or {}
        amt = _f(info.get("positionAmt"))
        if abs(amt) > 0:
            sym = info.get("symbol") or p.get("symbol") or ""
            reduce_sides[sym] = "BUY" if amt < 0 else "SELL"

    # Resting bracket orders (SL/TP). Two sources, both needed: plain open
    # orders (legacy shape) AND the algo/conditional endpoint, where Binance
    # now parks STOP_MARKET / TAKE_PROFIT_MARKET triggers. Best-effort: if the
    # reads fail we must NOT let this push overwrite the last-known SL/TP with
    # null (the orders are still resting on the exchange). See `bracketsKnown`.
    try:
        # ccxt (binanceusdm) refuses fetch_open_orders() with no symbol by
        # default — it raises to flag the stricter (40x) rate-limit weight.
        # Acknowledge it: hourly cron, few symbols, negligible weight.
        ex.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
        trigger_orders = list(ex.fetch_open_orders())
        trigger_orders += [{"info": r} for r in ex.fapiPrivateGetOpenAlgoOrders()]
        brackets = build_bracket_map(trigger_orders, reduce_sides)
        brackets_known = True
    except Exception as e:  # noqa: BLE001 — any ccxt/network error, keep pushing
        log.warning("open-orders read failed (%s) — preserving prior SL/TP this push", e)
        brackets, brackets_known = {}, False
    # Only bother with the symbolConfig call when there's a position to label.
    leverages = fetch_symbol_leverages(ex) if has_open else {}
    positions = build_position_payloads(raw_positions, brackets, leverages)

    start_ms = int(time.time() * 1000) - income_days * 86_400_000
    income = build_income_payloads(_fetch_income(ex, start_ms))
    return {"account": account, "positions": positions,
            "income": income, "brackets_known": brackets_known}


def collect_all_accounts(income_days: int) -> dict[str, Any]:
    """Iterate ACCOUNT_INSTANCES and combine their futures state.

    Returns the same shape run() used to build from a single account, plus
    `accounts` (per-instance detail) for logging and `accounts_read` (the
    instances read successfully this run; the server scopes its closed-position
    delete to these).
    """
    from exchange.env import load_env_for_instance

    totals = {"walletBalanceUsd": 0.0, "marginBalanceUsd": 0.0,
              "unrealizedPnlUsd": 0.0, "availableBalanceUsd": 0.0}
    positions: list[dict[str, Any]] = []
    income: list[dict[str, Any]] = []
    brackets_known = True
    seen_keys: dict[str, str] = {}          # fingerprint -> instance that claimed it
    detail: list[dict[str, Any]] = []
    # Instances actually read this run. The server only deletes closed rows
    # within these, so a leg whose read failed keeps its last-known position.
    accounts_read: list[str] = []
    # False if ANY account's read raised: the dashboard stores ONE summed
    # snapshot, so a partial sum would silently understate the wallet.
    account_complete = True

    for inst in ACCOUNT_INSTANCES:
        try:
            if inst != "v1":
                # v1's account IS the base .env; the others need their overlay.
                if load_env_for_instance(inst) is None:
                    log.info("relay: no .env.%s — skipping (leg not keyed)", inst)
                    detail.append({"instance": inst, "skipped": "no_env_file"})
                    continue
            fp = _key_fingerprint()
            if not fp:
                log.warning("relay: %s has no BINANCE_API_KEY active — skipping", inst)
                detail.append({"instance": inst, "skipped": "no_api_key"})
                continue
            if fp in seen_keys:
                log.warning("relay: %s resolves to the SAME account as %s "
                            "(key fp=%s) — skipping to avoid double-counting",
                            inst, seen_keys[fp], fp)
                detail.append({"instance": inst, "skipped": f"duplicate_of_{seen_keys[fp]}"})
                continue
            seen_keys[fp] = inst

            got = _collect_one_account(income_days)
        except Exception as e:  # noqa: BLE001 — one bad account must not lose the rest
            log.error("relay: reading %s failed (%s) — continuing with other accounts",
                      inst, e)
            detail.append({"instance": inst, "error": str(e)})
            brackets_known = False
            account_complete = False
            continue

        for k in totals:
            totals[k] += float(got["account"].get(k) or 0.0)
        positions.extend({**p, "account": inst} for p in got["positions"])
        accounts_read.append(inst)
        income.extend(got["income"])
        brackets_known = brackets_known and bool(got["brackets_known"])
        detail.append({
            "instance": inst, "key_fp": fp,
            "wallet": got["account"].get("walletBalanceUsd"),
            "positions": len(got["positions"]), "income": len(got["income"]),
        })

    return {"account": totals, "positions": positions, "income": income,
            "brackets_known": brackets_known, "accounts": detail,
            "accounts_read": accounts_read, "account_complete": account_complete}


def run(income_days: int = 2, dry_run: bool = False) -> dict[str, Any]:
    # Trigger the bot's .env auto-load BEFORE reading config. exchange.env calls
    # load_dotenv(REPO_ROOT/.env) at import time (override=False), so this makes
    # CONSOLIDATE_API_URL/TOKEN + BINANCE_* visible to os.environ even when run
    # from cron with no env sourced in the crontab line — same mechanism the
    # other tools.* crons rely on. Must happen before _config() reads os.environ.
    from exchange import env as _env  # noqa: F401

    url, token = _config()
    if not url or not token:
        log.info("consolidate not configured (CONSOLIDATE_API_URL/TOKEN unset) — skipping")
        return {"skipped": "not_configured"}

    # Read EVERY leg's sub-account (see ACCOUNT_INSTANCES) and combine. Balances
    # are summed, income concatenated (tranId-deduped server-side), positions
    # concatenated and tagged with their account.
    combined = collect_all_accounts(income_days)
    account = combined["account"]
    positions = combined["positions"]
    income = combined["income"]
    brackets_known = combined["brackets_known"]

    log.info("relay: %d account(s) read — %s", len(combined["accounts"]),
             "; ".join(
                 f"{a['instance']}="
                 + (a.get("skipped") or a.get("error")
                    or f"${a.get('wallet', 0):.2f}/{a.get('positions', 0)}pos")
                 for a in combined["accounts"]))

    if dry_run:
        print(json.dumps({"account": account, "positions": positions,
                          "bracketsKnown": brackets_known,
                          "income_count": len(income),
                          "accounts": combined["accounts"]}, indent=2))
        return {"dry_run": True, "positions": len(positions), "income": len(income),
                "bracketsKnown": brackets_known, "accounts": combined["accounts"]}

    results: dict[str, Any] = {}
    errors: list[str] = []
    if combined.get("account_complete", True):
        results["account"] = _post(url, token, "/futures/account-snapshot", account)
    else:
        log.error("relay: an account read failed — NOT pushing the summed "
                  "account snapshot (would understate wallet); positions/income "
                  "still pushed")
        results["account"] = {"skipped": "incomplete_accounts"}
    results["positions"] = _post(url, token, "/futures/positions",
                                 {"positions": positions, "bracketsKnown": brackets_known,
                                  "accountsRead": combined["accounts_read"]})
    for key in ("account", "positions"):
        if "error" in results[key]:
            errors.append(f"{key}: {results[key]['error']}")
    # Chunk income to the server's 1000-row cap.
    inserted = 0
    for i in range(0, len(income), INCOME_PAGE_LIMIT):
        chunk = income[i:i + INCOME_PAGE_LIMIT]
        r = _post(url, token, "/futures/income", {"income": chunk})
        if "error" in r:
            errors.append(f"income[{i}]: {r['error']}")
        inserted += int((r.get("body") or {}).get("inserted") or 0)
    results["income"] = {"sent": len(income), "inserted": inserted}
    results["errors"] = errors
    if errors:
        log.error("futures push had %d error(s): %s", len(errors), "; ".join(errors))
    else:
        log.info("futures push: %d positions, %d income rows (+%d new)",
                 len(positions), len(income), inserted)
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Push bot futures account to consolidate dashboard")
    ap.add_argument("--income-days", type=int, default=2,
                    help="lookback window for income history (use 365 for first backfill)")
    ap.add_argument("--dry-run", action="store_true", help="print payloads, don't POST")
    args = ap.parse_args()
    res = run(income_days=args.income_days, dry_run=args.dry_run)
    # Non-zero exit on push failure so cron (and the operator reading the log)
    # can see a bad run instead of it silently "succeeding".
    return 1 if res.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
