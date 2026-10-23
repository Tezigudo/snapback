"""Repair `fills` close rows that were recorded with price 0.0 / pnl None.

`Bot._resolve_fill_price` deliberately gives up after one failed fetch_order
(bookkeeping must not stall the loop), and the close row is then written with
price=0.0. This module finds those rows later and fills them in from the
exchange's own trade history. Bookkeeping only: it never places or cancels an
order and never decides whether to close.

Matching: Binance userTrades carry only orderId, not clientOrderId, so the
close ORDER is found first (fetch_orders, matched on
`<prefix><root>-<close leg>`), then its trades by orderId. A trade whose
info.clientOrderId is present is matched directly as well.

pnl_usd is the sum of the trades' realizedPnl: GROSS of fees, the same
convention as bot._exit_pnl_usd and the bracket-exit path.
"""

from __future__ import annotations

import logging
from datetime import datetime
from itertools import combinations

log = logging.getLogger(__name__)

# Every leg tag a close can carry (see exchange/binance_client.py). Entry ("e")
# is deliberately absent so an entry sharing the root is never mistaken for it.
# "c" is close_position()'s default leg.
CLOSE_LEGS = ("x", "ce", "be", "bf", "h", "k", "ns", "s", "t", "c")

WINDOW_BEFORE_MS = 30 * 60 * 1000   # close order fills BEFORE its row is written
WINDOW_AFTER_MS = 5 * 60 * 1000
FETCH_LIMIT = 200
MAX_SPLIT_ORDERS = 4                # a split close spans at most this many orders
QTY_REL_TOL = 0.005                 # matched qty must equal the row qty within 0.5%


def _is_close_coid(coid: str, prefix: str, root: str, reason: str | None) -> bool:
    if any(coid == f"{prefix}{root}-{leg}" for leg in CLOSE_LEGS):
        return True
    # A no-stop emergency close is sent under a FRESH root (so a retry never
    # hits the duplicate-id path) while its row carries the ENTRY root. The qty
    # and time-window checks below keep this from matching anything else.
    if reason == "no_stop_close" and coid.startswith(prefix) and coid.endswith("-ns"):
        return coid[len(prefix):-3].isdigit()
    return False


def _pick_groups(groups: dict[str, list], qty: float, ts_ms: int) -> list[str]:
    """Keys of the order group(s) whose summed filled qty equals `qty`.
    A single exact order is preferred (nearest the row in time); failing that
    the smallest set of orders (a split close) that sums to it."""
    tol = max(1e-9, QTY_REL_TOL * qty)
    info = []
    for k, trs in groups.items():
        gq = sum(float(t.get("amount") or 0.0) for t in trs)
        dist = min(abs(float(t.get("timestamp") or 0.0) - ts_ms) for t in trs)
        if gq > 0:
            info.append((k, gq, dist))
    info.sort(key=lambda x: x[2])
    for n in range(1, min(len(info), MAX_SPLIT_ORDERS) + 1):
        for combo in combinations(info, n):
            if abs(sum(x[1] for x in combo) - qty) <= tol:
                return [x[0] for x in combo]
    return []


def price_close_row(ex, symbol: str, prefix: str, root: str, row_ts: str,
                    qty: float, reason: str | None = None,
                    claimed: frozenset[str] | set[str] = frozenset(),
                    ) -> tuple[float, float | None, list[str]] | None:
    """(qty-weighted avg price, summed realizedPnl, order keys used) for the
    close of `root` recorded at `row_ts` for `qty`, or None if the exchange has
    no close order (or split-close set of orders) that matches. Trades are
    grouped per close ORDER; orders in `claimed` (already used by another row)
    are skipped. Two REST calls, bounded by a tight since/until window and a
    limit. Exchange errors propagate; a malformed `row_ts` raises ValueError
    BEFORE any exchange call."""
    ts_ms = int(datetime.fromisoformat(row_ts).timestamp() * 1000)
    since = ts_ms - WINDOW_BEFORE_MS
    until = ts_ms + WINDOW_AFTER_MS
    params = {"endTime": until}

    order_ids: set[str] = set()
    for o in ex.fetch_orders(symbol, since=since, limit=FETCH_LIMIT, params=params) or []:
        coid = str(o.get("clientOrderId") or (o.get("info") or {}).get("clientOrderId") or "")
        if _is_close_coid(coid, prefix, root, reason) and o.get("id") is not None:
            order_ids.add(str(o["id"]))

    groups: dict[str, list] = {}
    for tr in ex.fetch_my_trades(symbol, since=since, limit=FETCH_LIMIT, params=params) or []:
        stamp = float(tr.get("timestamp") or 0.0)
        if stamp < since or stamp > until:
            continue
        info = tr.get("info") or {}
        coid = str(info.get("clientOrderId") or "")
        oid = str(tr.get("order") or info.get("orderId") or "")
        key = oid or coid
        if key in claimed:
            continue
        if (coid and _is_close_coid(coid, prefix, root, reason)) or (oid and oid in order_ids):
            groups.setdefault(key, []).append(tr)

    keys = _pick_groups(groups, qty, ts_ms)
    if not keys:
        return None
    picked = [t for k in keys for t in groups[k]]
    gq = sum(float(t.get("amount") or 0.0) for t in picked)
    price = sum(float(t.get("amount") or 0.0) * float(t.get("price") or 0.0)
                for t in picked) / gq
    if price <= 0:
        return None
    pnls = [(t.get("info") or {}).get("realizedPnl") for t in picked]
    pnl = (round(sum(float(p) for p in pnls), 4)
           if all(p is not None for p in pnls) else None)
    return price, pnl, keys
