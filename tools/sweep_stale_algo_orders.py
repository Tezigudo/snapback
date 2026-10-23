"""One-off sweep of STALE algo/conditional orders on the active account.

Context (2026-07-26): Binance migrated conditional orders (STOP_MARKET /
TAKE_PROFIT_MARKET triggers) to the algo system — /fapi/v1/openAlgoOrders —
where the bot's plain-endpoint sweeps couldn't see them. v1's account
accumulated 100 stale orders: 2 orphaned brackets from closed longs plus 98
copies of one TP spammed by the (since-disabled) reprotect loop on 07-22.

Classification rule:
  KEEP    clientAlgoId starts with "snap-" AND the order's side can REDUCE the
          current live position (short → BUY, long → SELL). That is exactly the
          live bracket pair; everything else cannot belong to the open trade.
  CANCEL  everything else — wrong-side snap-* orders are orphans of closed
          trades by definition; non-snap orders on a bot sub-account are the
          reprotect spam (ccxt default COID prefix "x-cvBPrNm9").

If the account is FLAT, every resting algo order is stale → all cancel.

Dry-run by default; pass --execute to actually cancel. Run on the droplet with
the target leg's env active (base .env = v1):

  cd /root/snapback-btc && set -a && . ./.env && set +a && \
      .venv/bin/python tools/sweep_stale_algo_orders.py --execute
"""

from __future__ import annotations

import argparse
import os
import sys

import ccxt


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep stale algo orders (dry-run by default)")
    ap.add_argument("--symbol", default="BTCUSDT", help="raw exchange symbol id")
    ap.add_argument("--execute", action="store_true", help="actually cancel (default: dry-run)")
    args = ap.parse_args()

    key = os.environ.get("BINANCE_API_KEY", "")
    sec = os.environ.get("BINANCE_API_SECRET", "")
    if not key or not sec:
        print("FATAL: BINANCE_API_KEY/SECRET not in env — source the leg's .env first",
              file=sys.stderr)
        return 2
    ex = ccxt.binanceusdm({"apiKey": key, "secret": sec})

    pos_amt = 0.0
    for p in ex.fetch_positions():
        info = p.get("info") or {}
        if info.get("symbol") == args.symbol:
            pos_amt = float(info.get("positionAmt") or 0)
    reduce_side = "BUY" if pos_amt < 0 else ("SELL" if pos_amt > 0 else None)
    print(f"{args.symbol} positionAmt={pos_amt} -> live-bracket side="
          f"{reduce_side or 'NONE (flat: everything is stale)'}")

    rows = ex.fapiPrivateGetOpenAlgoOrders({"symbol": args.symbol})
    keep, cancel = [], []
    for r in rows:
        coid = str(r.get("clientAlgoId") or "")
        if coid.startswith("snap-") and reduce_side and r.get("side") == reduce_side:
            keep.append(r)
        else:
            cancel.append(r)

    print(f"open algo orders: {len(rows)} | keep {len(keep)} | cancel {len(cancel)}")
    for r in keep:
        print(f"  KEEP   {r['algoId']} {r['clientAlgoId']} "
              f"{r['orderType']} {r['side']} trig={r['triggerPrice']} qty={r['quantity']}")
    # The spam is 98 near-identical rows — summarise, print the distinct ones.
    seen: set[tuple] = set()
    for r in cancel:
        sig = (str(r.get("clientAlgoId") or "")[:8], r["orderType"], r["side"],
               r["triggerPrice"], r["quantity"])
        if sig in seen:
            continue
        seen.add(sig)
        n_same = sum(1 for x in cancel
                     if (str(x.get("clientAlgoId") or "")[:8], x["orderType"], x["side"],
                         x["triggerPrice"], x["quantity"]) == sig)
        print(f"  CANCEL {r['orderType']} {r['side']} trig={r['triggerPrice']} "
              f"qty={r['quantity']} coid~{sig[0]}… ×{n_same}")

    if not args.execute:
        print("\nDRY-RUN — nothing cancelled. Re-run with --execute to cancel.")
        return 0
    ok = fail = 0
    for r in cancel:
        try:
            ex.fapiPrivateDeleteAlgoOrder({"symbol": args.symbol, "algoId": r["algoId"]})
            ok += 1
        except Exception as e:  # noqa: BLE001 — keep sweeping, report at the end
            fail += 1
            print(f"  FAILED {r['algoId']}: {str(e)[:120]}", file=sys.stderr)
    remaining = len(ex.fapiPrivateGetOpenAlgoOrders({"symbol": args.symbol}))
    print(f"cancelled {ok}, failed {fail}; remaining open algo orders: {remaining} "
          f"(expected {len(keep)})")
    return 0 if fail == 0 and remaining == len(keep) else 1


if __name__ == "__main__":
    raise SystemExit(main())
