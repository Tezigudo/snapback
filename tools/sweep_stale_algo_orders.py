"""One-off sweep of STALE algo/conditional orders on the active account.

Context (2026-07-26): Binance migrated conditional orders (STOP_MARKET /
TAKE_PROFIT_MARKET triggers) to the algo system — /fapi/v1/openAlgoOrders —
where the bot's plain-endpoint sweeps couldn't see them. v1's account
accumulated 100 stale orders: 2 orphaned brackets from closed longs plus 98
copies of one TP spammed by the (since-disabled) reprotect loop on 07-22.

Classification rule (2026-11 round 3, G6 — was: raw fetch_positions, a
missing row read as positionAmt=0, and --execute then cancelled EVERY algo
order on the symbol, live SLs and other legs' orders included):
  - The position is read with BinanceClient.fetch_position_checked (an empty
    positionRisk is confirmed through the account endpoint). UNKNOWN aborts
    the whole run: nothing is cancelled.
  - Only orders whose clientAlgoId starts with --prefix (this leg's coid
    prefix, required) are candidates. Anything else is reported and LEFT.
  - Position OPEN: an SL leg (-s / -sb) is NEVER cancelled; orders on the side
    that reduces the position are kept; only our wrong-side non-SL orders
    (orphans of closed trades) are cancelled.
  - Position confirmed FLAT: every one of OUR resting algo orders is stale.

Dry-run by default; pass --execute to actually cancel. Run on the droplet with
the target leg's env active (base .env = v1):

  cd /root/snapback-btc && set -a && . ./.env && set +a && \
      .venv/bin/python tools/sweep_stale_algo_orders.py --prefix snap-v1- --execute
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ccxt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot_internals import algo_bracket_leg
from exchange.binance_client import BinanceClient, PositionUnknown


def _ccxt_symbol(raw: str) -> str:
    """"BTCUSDT" -> "BTC/USDT:USDT" (USDT-margined perps only)."""
    if not raw.endswith("USDT"):
        raise SystemExit(f"FATAL: only USDT perps supported, got {raw}")
    return f"{raw[:-4]}/USDT:USDT"


def classify(rows: list[dict], prefix: str, side: str) -> tuple[list, list, list]:
    """(keep, cancel, foreign). `side` is the CONFIRMED position side:
    "long" / "short" / "flat". Pure; unit-tested."""
    reduce_side = {"long": "SELL", "short": "BUY"}.get(side)
    keep, cancel, foreign = [], [], []
    for r in rows:
        coid = str(r.get("clientAlgoId") or "")
        if not prefix or not coid.startswith(prefix):
            foreign.append(r)
            continue
        if side == "flat":
            cancel.append(r)
            continue
        if algo_bracket_leg(r, prefix) == "sl" or r.get("side") == reduce_side:
            keep.append(r)          # never cancel an SL while a position is open
        else:
            cancel.append(r)
    return keep, cancel, foreign


def main(argv: list[str] | None = None, ex: ccxt.Exchange | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sweep stale algo orders (dry-run by default)")
    ap.add_argument("--symbol", default="BTCUSDT", help="raw exchange symbol id")
    ap.add_argument("--prefix", required=True,
                    help="this leg's clientOrderId prefix, e.g. snap-d3- (only these are touched)")
    ap.add_argument("--execute", action="store_true", help="actually cancel (default: dry-run)")
    args = ap.parse_args(argv)

    if ex is None:
        key = os.environ.get("BINANCE_API_KEY", "")
        sec = os.environ.get("BINANCE_API_SECRET", "")
        if not key or not sec:
            print("FATAL: BINANCE_API_KEY/SECRET not in env — source the leg's .env first",
                  file=sys.stderr)
            return 2
        ex = ccxt.binanceusdm({"apiKey": key, "secret": sec})
    client = BinanceClient(ex=ex, env="tool", coid_prefix=args.prefix)
    try:
        pos = client.fetch_position_checked(_ccxt_symbol(args.symbol))
    except PositionUnknown as e:
        print(f"ABORT: position for {args.symbol} is UNKNOWN ({e}) — nothing cancelled.",
              file=sys.stderr)
        return 3
    print(f"{args.symbol} position: {pos.side} {pos.qty} (confirmed)")

    rows = ex.fapiPrivateGetOpenAlgoOrders({"symbol": args.symbol})
    keep, cancel, foreign = classify(rows, args.prefix, pos.side)
    print(f"open algo orders: {len(rows)} | keep {len(keep)} | cancel {len(cancel)} "
          f"| not ours (left) {len(foreign)}")
    for r in keep:
        print(f"  KEEP   {r.get('algoId')} {r.get('clientAlgoId')} "
              f"{r.get('orderType')} {r.get('side')} trig={r.get('triggerPrice')}")
    for r in cancel:
        print(f"  CANCEL {r.get('algoId')} {r.get('clientAlgoId')} "
              f"{r.get('orderType')} {r.get('side')} trig={r.get('triggerPrice')}")
    for r in foreign:
        print(f"  LEAVE  {r.get('algoId')} {r.get('clientAlgoId')} (not prefix {args.prefix})")

    if not args.execute:
        print("\nDRY-RUN — nothing cancelled. Re-run with --execute to cancel.")
        return 0
    ok = fail = 0
    for r in cancel:
        try:
            ex.fapiPrivateDeleteAlgoOrder({"symbol": args.symbol, "algoId": r["algoId"]})
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAILED {r.get('algoId')}: {str(e)[:120]}", file=sys.stderr)
    print(f"cancelled {ok}, failed {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
