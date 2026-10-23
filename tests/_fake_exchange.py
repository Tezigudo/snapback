"""A scripted, in-memory stand-in for ccxt.binanceusdm — for the no-stop tests.

MagicMock cannot model the thing these tests are about: an order book whose
contents change as orders are placed, cancelled and closed, and an exchange
that fails a SPECIFIC order type a SPECIFIC number of times. This fake keeps
both books (plain + algo) and the one position, records every call in order,
and lets a test script failures per order type.

Not a general-purpose simulator. It implements exactly the surface
BinanceClient touches on the entry / bracket / close / reprotect paths.
"""

from __future__ import annotations

from typing import Any

import ccxt

SYMBOL = "BTC/USDT:USDT"
MARKET_ID = "BTCUSDT"

# Realistic Binance -2021 as ccxt 4.5.x surfaces it: there is NO exact mapping
# for -2021 in this ccxt build (checked 2026-10-13), so it arrives as a generic
# error whose text carries the raw JSON body.
WOULD_TRIGGER = ccxt.ExchangeError(
    'binanceusdm {"code":-2021,"msg":"Order would immediately trigger."}')


def timeout() -> ccxt.RequestTimeout:
    return ccxt.RequestTimeout("binanceusdm POST https://fapi.binance.com/fapi/v1/order "
                               "request timeout (15000 ms)")


def rejected(code: int = -4045, msg: str = "Reach max stop order limit.") -> ccxt.ExchangeError:
    return ccxt.OperationRejected(f'binanceusdm {{"code":{code},"msg":"{msg}"}}')


class FakeEx:
    def __init__(self, fill_price: float = 65000.0, close_price: float = 64900.0) -> None:
        self.calls: list[tuple] = []
        self.algo: list[dict[str, Any]] = []
        self.plain: list[dict[str, Any]] = []
        self.pos: tuple[str, float, float] | None = None  # (side, qty, entry)
        self.fill_price = fill_price
        self.close_price = close_price
        # Scripted failures, consumed one per create of that kind. A None entry
        # means "this attempt succeeds".
        self.fail: dict[str, list[BaseException | None]] = {
            "STOP_MARKET": [], "TAKE_PROFIT_MARKET": [], "close": [], "entry": [],
        }
        # A timeout that the exchange nevertheless ACCEPTED — the order rests.
        self.timeout_lands = False
        # Limit-entry scripting: fetch_order returns these, in order.
        self.limit_polls: list[dict[str, Any]] = []
        self._n = 0
        self.algo_readable = True
        # A limit-entry create that times out but RESTS on the book.
        self.entry_timeout_lands = False
        self.mark_price = 64500.0
        # Filled qty reported for the entry order when read by client id
        # (None = the read fails / order unknown).
        self.entry_filled: float | None = None
        # Number of upcoming fetch_positions reads that report FLAT regardless.
        self.position_lag_reads = 0

    # --- ccxt surface ------------------------------------------------------
    def market(self, symbol: str) -> dict[str, Any]:
        return {"id": MARKET_ID, "precision": {"amount": 3}}

    def parse_timeframe(self, tf: str) -> int:
        return 900

    def _next_id(self) -> str:
        self._n += 1
        return str(1000 + self._n)

    def _rest_algo(self, coid: str | None, side: str, qty: float, stop: float) -> dict:
        row = {"algoId": self._next_id(), "clientAlgoId": coid, "side": side.upper(),
               "positionSide": "BOTH", "triggerPrice": str(stop), "origQty": None,
               "reduceOnly": True, "algoStatus": "NEW", "strategyType": None}
        self.algo.append(row)
        return row

    def create_order(self, symbol, type_, side, qty, price=None, params=None):
        params = dict(params or {})
        self.calls.append(("create_order", symbol, type_, side, qty, price, params))
        coid = params.get("newClientOrderId")
        if type_ in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
            if coid and any(r["clientAlgoId"] == coid for r in self.algo):
                raise ccxt.InvalidOrder(
                    'binanceusdm {"code":-4116,"msg":"ClientOrderId is duplicated."}')
            script = self.fail[type_]
            err = script.pop(0) if script else None
            if err is not None:
                if self.timeout_lands and isinstance(err, ccxt.RequestTimeout):
                    self._rest_algo(coid, side, qty, params["stopPrice"])
                raise err
            row = self._rest_algo(coid, side, qty, params["stopPrice"])
            return {"id": row["algoId"], "clientOrderId": coid, "info": row}
        if type_ == "market" and params.get("reduceOnly"):
            script = self.fail["close"]
            err = script.pop(0) if script else None
            if err is not None:
                raise err
            if self.pos is None:
                raise ccxt.InvalidOrder(
                    'binanceusdm {"code":-2022,"msg":"ReduceOnly Order is rejected."}')
            self.pos = None
            return {"id": self._next_id(), "clientOrderId": coid,
                    "average": self.close_price}
        if type_ in ("market", "limit"):
            script = self.fail["entry"]
            err = script.pop(0) if script else None
            if type_ == "market":
                # A market entry that errors client-side may still have FILLED.
                self.pos = ("long" if side == "buy" else "short", qty, self.fill_price)
                if err is not None:
                    raise err
                return {"id": self._next_id(), "clientOrderId": coid,
                        "average": self.fill_price}
            if err is not None:
                if self.entry_timeout_lands and isinstance(err, ccxt.RequestTimeout):
                    # The create timed out on our side but the GTC limit RESTS.
                    self.plain.append({"id": "lim-1", "clientOrderId": coid,
                                       "side": side, "type": "limit",
                                       "reduceOnly": False,
                                       "info": {"clientOrderId": coid, "type": "LIMIT",
                                                "reduceOnly": "false"}})
                raise err
            return {"id": "lim-1", "clientOrderId": coid}
        raise AssertionError(f"unexpected order type {type_}")

    def fetch_order(self, oid, symbol=None, params=None):
        self.calls.append(("fetch_order", oid, symbol, dict(params or {})))
        if oid is None and (params or {}).get("origClientOrderId"):
            if self.entry_filled is None:
                raise ccxt.OrderNotFound(
                    'binanceusdm {"code":-2013,"msg":"Order does not exist."}')
            return {"id": "lim-1", "clientOrderId": params["origClientOrderId"],
                    "filled": self.entry_filled, "status": "canceled"}
        if oid == "lim-1" and self.limit_polls:
            o = self.limit_polls.pop(0) if len(self.limit_polls) > 1 else self.limit_polls[0]
            if float(o.get("filled") or 0) > 0 and self.pos is None:
                self.pos = (o["_side"], float(o["filled"]), float(o["average"]))
            return {k: v for k, v in o.items() if not k.startswith("_")}
        return {"id": oid, "average": self.fill_price, "status": "closed"}

    def cancel_order(self, oid, symbol=None, params=None):
        if params:
            self.calls.append(("cancel_order", oid, symbol, dict(params)))
            coid = params.get("origClientOrderId")
            hit = [o for o in self.plain if o.get("clientOrderId") == coid]
            if not hit:
                raise ccxt.OrderNotFound(
                    'binanceusdm {"code":-2011,"msg":"Unknown order sent."}')
            self.plain = [o for o in self.plain if o.get("clientOrderId") != coid]
            return {"id": hit[0].get("id")}
        self.calls.append(("cancel_order", oid, symbol))
        self.plain = [o for o in self.plain if o.get("id") != oid]
        return {"id": oid}

    def fapiPublicGetPremiumIndex(self, params):  # ccxt name
        self.calls.append(("fapiPublicGetPremiumIndex", dict(params)))
        return {"markPrice": str(self.mark_price)}

    def load_time_difference(self):
        self.calls.append(("load_time_difference",))
        return 0

    def fetch_positions(self, symbols=None):
        self.calls.append(("fetch_positions", tuple(symbols or ())))
        if self.position_lag_reads > 0:
            # positionRisk lagging a fill: reads flat although a position exists
            self.position_lag_reads -= 1
            return []
        if self.pos is None:
            return []
        side, qty, entry = self.pos
        return [{"symbol": SYMBOL, "contracts": qty, "side": side,
                 "entryPrice": entry, "unrealizedPnl": 0.0, "initialMargin": 10.0}]

    def fetch_open_orders(self, symbol=None):
        self.calls.append(("fetch_open_orders", symbol))
        return [dict(o) for o in self.plain]

    def fapiPrivateGetOpenAlgoOrders(self, params):
        self.calls.append(("fapiPrivateGetOpenAlgoOrders", dict(params)))
        if not self.algo_readable:
            raise ccxt.NetworkError("algo read failed")
        return [dict(r) for r in self.algo]

    def fapiPrivateDeleteAlgoOrder(self, params):
        self.calls.append(("fapiPrivateDeleteAlgoOrder", dict(params)))
        self.algo = [r for r in self.algo if r["algoId"] != params["algoId"]]
        return {"code": 200}

    # --- helpers for assertions ---------------------------------------------
    def creates(self, type_: str | None = None) -> list[tuple]:
        return [c for c in self.calls
                if c[0] == "create_order" and (type_ is None or c[2] == type_)]

    def closes(self) -> list[tuple]:
        return [c for c in self.creates("market") if c[6].get("reduceOnly")]
