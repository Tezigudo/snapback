"""Entry-recording helpers that bundle the (record_fill, record_event,
enqueue_bot_event, send_alert) quartet for the bot's main entry paths.

Why these exist: each entry flow in bot.py needs to write to multiple sinks
per event — local sqlite for audit (record_fill + record_event), the outbox
for push to consolidate (enqueue_bot_event), and the user's inbox (send_alert).
Bundling them keeps `_maybe_enter` focused on the trading decision rather
than bookkeeping.

Three helpers, one per entry path:
  - record_dry_run_entry — DRY-RUN mode, no order placed.
  - record_market_entry  — live MARKET entry with bracket SL/TP.
  - record_limit_entry   — live LIMIT entry (may have fallen back to market;
                           filled_as captures which).

Close paths are NOT extracted here — their post-close bookkeeping varies
too much (boot-flatten has no fill record, kill-switch enqueues its own
event in _check_kill_switch, HALT uses kind="halt" not "exit"). Keeping
them inline in bot.py is clearer than a parameter-bag helper.
"""

from __future__ import annotations

from typing import Any

from alerts import send_alert
from exchange import state


def record_dry_run_entry(
    *,
    side: str,
    qty: float,
    price: float,
    sl_price: float,
    tp_price: float,
    notional: float,
    equity: float,
    signal_id: str,
    strategy_name: str,
    order_type: str,
    dbg: dict,
) -> None:
    """Records a would-have-entered signal in dry-run mode. No order placed."""
    state.record_event(
        "INFO", "dry_run_signal",
        {**dbg, "side": side, "qty": qty,
         "sl": sl_price, "tp": tp_price,
         "notional": notional, "order_type": order_type,
         "signal_id": signal_id},
        signal_id=signal_id,
    )
    state.enqueue_bot_event(
        "dry_run_signal",
        signal_id=signal_id, strategy=strategy_name,
        side=side, qty=float(qty), price_usd=float(price),
        notional_usd=float(notional), equity_usd=float(equity),
        payload={"sl_price": float(sl_price), "tp_price": float(tp_price),
                 "order_type": order_type},
    )
    send_alert(
        f"DRY-RUN: would {side.upper()} [{order_type}]",
        f"DRY-RUN: would have entered {side.upper()} "
        f"{qty:.4f} BTC @ {price:.2f} ({order_type})\n"
        f"SL: {sl_price:.2f}  TP: {tp_price:.2f}\n"
        f"Notional: ${notional:.2f}\n"
        f"Equity: {equity:.2f} USDT\n"
        f"signal_id: {signal_id}\n"
        f"(No real order placed.)",
    )


def record_market_entry(
    *,
    side: str,
    qty: float,
    price: float,
    sl_price: float,
    tp_price: float,
    equity: float,
    signal_id: str,
    strategy_name: str,
    orders: dict[str, Any],
    dbg: dict,
) -> None:
    """Records a market-order entry that just placed (with bracket SL/TP)."""
    state.record_fill(
        side=side, qty=qty, price=price,
        reason="entry", equity_after=equity,
        client_order_id_root=signal_id,
    )
    state.record_event(
        "INFO", "entry",
        {**dbg, "side": side, "qty": qty,
         "sl": sl_price, "tp": tp_price,
         "signal_id": signal_id,
         "order_ids": {k: v.get("id") for k, v in orders.items()
                       if isinstance(v, dict)}},
        signal_id=signal_id,
    )
    state.enqueue_bot_event(
        "entry",
        signal_id=signal_id, strategy=strategy_name,
        side=side, qty=float(qty), price_usd=float(price),
        notional_usd=float(qty * price), equity_usd=float(equity),
        payload={"filled_as": "market",
                 "sl_price": float(sl_price), "tp_price": float(tp_price)},
    )
    send_alert(
        f"Bot {side.upper()} entry",
        f"{side.upper()} {qty:.4f} BTC @ {price:.2f}\n"
        f"SL: {sl_price:.2f}  TP: {tp_price:.2f}\n"
        f"signal_id: {signal_id}\n"
        f"Equity: {equity:.2f} USDT",
    )


def record_limit_entry(
    *,
    side: str,
    filled_qty: float,
    fill_price: float,
    sl_distance: float,
    tp_distance: float,
    limit_price: float,
    signal_price: float,
    limit_offset_bps: float,
    equity: float,
    signal_id: str,
    strategy_name: str,
    filled_as: str,
    orders: dict[str, Any],
    dbg: dict,
) -> None:
    """Records a limit-order entry. `filled_as` is one of:
    "limit" (fully filled at limit), "limit_partial" (some filled, no
    market top-up), or "market_fallback" (limit timed out, market placed).
    """
    state.record_fill(
        side=side, qty=filled_qty, price=fill_price,
        reason="entry", equity_after=equity,
        client_order_id_root=signal_id,
    )
    state.record_event(
        "INFO", "entry",
        {**dbg, "side": side, "qty": filled_qty,
         "fill_price": fill_price, "limit_price": limit_price,
         "filled_as": filled_as,
         "sl_distance": sl_distance, "tp_distance": tp_distance,
         "signal_id": signal_id,
         "order_ids": {k: v.get("id") for k, v in orders.items()
                       if isinstance(v, dict)}},
        signal_id=signal_id,
    )
    state.enqueue_bot_event(
        "entry",
        signal_id=signal_id, strategy=strategy_name,
        side=side, qty=float(filled_qty), price_usd=float(fill_price),
        notional_usd=float(filled_qty * fill_price), equity_usd=float(equity),
        payload={"filled_as": filled_as, "limit_price": float(limit_price),
                 "sl_distance": float(sl_distance),
                 "tp_distance": float(tp_distance)},
    )
    send_alert(
        f"Bot {side.upper()} entry [{filled_as}]",
        f"{side.upper()} {filled_qty:.4f} BTC fill @ {fill_price:.2f}\n"
        f"limit was {limit_price:.2f} (close {signal_price:.2f}, "
        f"offset {limit_offset_bps:.1f}bp); filled_as={filled_as}\n"
        f"SL dist {sl_distance:.2f}  TP dist {tp_distance:.2f}\n"
        f"signal_id: {signal_id}\n"
        f"Equity: {equity:.2f} USDT",
    )
