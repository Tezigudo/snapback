"""Live exchange constraints, per traded symbol.

Hard-coded fallbacks so a config glitch can't loosen them. The bot reads the
LIVE values from ccxt at boot and applies whichever is TIGHTER — that
safe-by-default merge is unchanged.

Was BTC-only until 2026-07-25. The fallbacks are per-symbol now because the
sol_supertrend leg trades SOL, and a single set of BTC numbers is not a
conservative default for another instrument — it is just the wrong one:

  * BTC's $50 min-notional applied to SOL (real minimum $5) forced a $200
    account minimum on a leg that only needs ~$25, and the skipped signals
    clustered in high-volatility periods — exactly the trades a trend follower
    wants. Over-constraining is safe for capital but not free.
  * BTC's $0.10 price tick applied to SOL (real tick $0.01) rounds a limit
    entry at ~$74 by up to 13 bps, which is 2-3x the maker/taker spread the
    limit order exists to capture.

BTC's numbers are byte-identical to the pre-split values, so the BTC legs are
unaffected.

Sources: ccxt market("BTC/USDT:USDT") fetched 2026-05-17;
Binance fapi exchangeInfo SOLUSDT fetched 2026-07-25
(LOT_SIZE step/min 0.01, MIN_NOTIONAL 5, PRICE_FILTER tick 0.01).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExchangeConstraints:
    # Minimum order quantity in the BASE asset (BTC for BTCUSDT, SOL for SOLUSDT).
    min_qty_base: float = 0.001
    # Minimum notional cost per order in quote asset (USDT)
    min_notional_usdt: float = 50.0
    # Price tick size
    price_step: float = 0.1
    # Qty step size
    qty_step: float = 0.001
    # Standard taker fee (private; treat as upper bound)
    taker_fee: float = 0.0005


# Per-symbol hard-coded fallbacks. Keyed by the ccxt unified symbol the bot
# trades. Only symbols in risk.py ALLOWED_SYMBOLS can ever reach here.
_FALLBACKS: dict[str, ExchangeConstraints] = {
    "BTC/USDT:USDT": ExchangeConstraints(
        min_qty_base=0.001, min_notional_usdt=50.0,
        price_step=0.1, qty_step=0.001, taker_fee=0.0005,
    ),
    "SOL/USDT:USDT": ExchangeConstraints(
        min_qty_base=0.01, min_notional_usdt=5.0,
        price_step=0.01, qty_step=0.01, taker_fee=0.0005,
    ),
}

# Module-level constant kept for backwards compatibility: BTC's values, which
# is what DEFAULT_CONSTRAINTS has always meant.
DEFAULT_CONSTRAINTS = _FALLBACKS["BTC/USDT:USDT"]


def fallbacks_for_symbol(symbol: str) -> ExchangeConstraints:
    """Hard-coded fallback constraints for `symbol`.

    Unknown symbols get BTC's numbers. That is deliberately conservative on the
    two limits that gate an order (min qty, min notional) — a too-high minimum
    skips trades rather than placing undersized ones. It is NOT correct for
    `price_step`, so any new symbol must be registered in `_FALLBACKS` before it
    is traded; risk.py's ALLOWED_SYMBOLS is the gate that makes that enforceable.
    """
    return _FALLBACKS.get(symbol, _FALLBACKS["BTC/USDT:USDT"])


def merge_with_live(c: ExchangeConstraints, live_market: dict) -> ExchangeConstraints:
    """Return new constraints using the TIGHTER value between hard-coded
    fallback and the live ccxt market spec. Safe-by-default."""
    limits = live_market.get("limits", {}) or {}
    precision = live_market.get("precision", {}) or {}

    live_min_qty = (limits.get("amount", {}) or {}).get("min")
    live_min_cost = (limits.get("cost", {}) or {}).get("min")
    live_price_step = precision.get("price")
    live_qty_step = precision.get("amount")

    def tighter_min(a: float, b: float | None) -> float:
        return max(a, float(b)) if b is not None else a

    def tighter_step(a: float, b: float | None) -> float:
        return max(a, float(b)) if b is not None else a

    return ExchangeConstraints(
        min_qty_base=tighter_min(c.min_qty_base, live_min_qty),
        min_notional_usdt=tighter_min(c.min_notional_usdt, live_min_cost),
        price_step=tighter_step(c.price_step, live_price_step),
        qty_step=tighter_step(c.qty_step, live_qty_step),
        taker_fee=c.taker_fee,
    )


def round_qty_down(qty: float, step: float) -> float:
    """Floor-round to step. Never rounds UP — that could violate balance."""
    if step <= 0:
        return qty
    return (int(qty / step)) * step


def passes_minimums(qty: float, price: float, c: ExchangeConstraints) -> tuple[bool, str]:
    """Return (ok, reason) — reason describes the FIRST failing check."""
    notional = qty * price
    if qty < c.min_qty_base:
        return False, f"qty {qty:.6f} below min {c.min_qty_base}"
    if notional < c.min_notional_usdt:
        return False, f"notional ${notional:.2f} below min ${c.min_notional_usdt}"
    return True, "ok"
