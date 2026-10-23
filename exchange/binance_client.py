"""Binance USDM Futures client wrapper (ccxt-based, testnet-aware).

Wraps ccxt.binanceusdm with sandbox_mode auto-set when BINANCE_ENV=testnet.
Provides the small surface area the bot loop needs:
  - fetch_ohlcv(symbol, tf, limit)
  - fetch_funding_rate(symbol)
  - fetch_equity_usdt()
  - fetch_position(symbol)
  - cancel_open_orders(symbol)   # plain + algo/conditional (openAlgoOrders)
  - market_order_with_bracket(symbol, side, qty, sl, tp)
  - close_position(symbol)

NEVER call from Claude Code sessions — this is the trading runtime.
Bot uses this; Claude Code is read-only per CLAUDE.md.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any

import ccxt
import pandas as pd

from .env import get_api_credentials, get_env

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 15_000

# clientOrderId scheme that lets investing-consolidate attribute trades back
# to this bot. Format: snap-<version>-<signal_id>-<leg>
#   version  : strategy version, e.g. "v1"
#   signal_id: millisecond epoch when the bot decided on the signal
#              (passed in by bot.py — anchors all 3 legs of a trade)
#   leg      : "e" entry, "s" stop-loss, "t" take-profit,
#              "x" time-stop close, "ce" channel-exit close,
#              "bf" boot-flatten, "h" HALT close, "k" kill-switch close,
#              "ns" no-stop emergency close (stop could not be placed)
# Binance allows alnum + ._-, max 36 chars. snap-v1-<13 digit ms>-tx = 24 chars.
# Donchian leg uses prefix "snap-d3-" to distinguish in fills/logs.
_COID_PREFIX_DEFAULT = "snap-v1-"
_COID_VALID = re.compile(r"^[A-Za-z0-9._\-]{1,36}$")


def _coid(root: str | None, leg: str, prefix: str = _COID_PREFIX_DEFAULT) -> str | None:
    """Build a Binance-safe clientOrderId. Returns None if root is None.

    Returns None (not a raised error) on invalid inputs so callers can fall
    back to an untagged order placement gracefully.
    """
    if root is None:
        return None
    coid = f"{prefix}{root}-{leg}"
    return coid if _COID_VALID.fullmatch(coid) else None


class EquityUnavailable(RuntimeError):
    """Balance payload unreadable. Deliberately NOT a ccxt error, so the loop's
    NetworkError clause cannot swallow it; callers treat it as "unknown"."""


def _order_avg(order: dict[str, Any] | None) -> float | None:
    """Positive average fill price from a ccxt order, else None."""
    o = order or {}
    for v in (o.get("average"), (o.get("info") or {}).get("avgPrice"), o.get("price")):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            return f
    return None


def _with_tp_error(out: dict[str, Any], tp_error: str | None) -> dict[str, Any]:
    """Add `tp_error` ONLY when the TP failed, so a healthy result is unchanged."""
    if tp_error is not None:
        out["tp_error"] = tp_error
    return out


def is_would_trigger(e: BaseException) -> bool:
    """Binance -2021 "Order would immediately trigger" — for a STOP this means
    price is already through the stop level.

    Matched on the message text because ccxt 4.5.x has NO exact mapping for
    -2021 (checked 2026-10-13): it arrives as a generic ExchangeError whose
    string carries the raw JSON body.
    """
    s = str(e)
    return "-2021" in s or "immediately trigger" in s.lower()


def is_nonce_error(e: BaseException) -> bool:
    """-1021: request timestamp outside recvWindow (clock drift). Retrying
    without re-syncing the clock fails the same way."""
    return isinstance(e, ccxt.InvalidNonce) or "-1021" in str(e)


def is_transient_error(e: BaseException | None) -> bool:
    """Timeout / network / clock-drift: says nothing about whether a stop CAN be
    placed. Everything else (-2021, -4045, -1111, ...) does."""
    if e is None:
        return False
    return isinstance(e, ccxt.NetworkError) or is_nonce_error(e)


def _is_duplicate_id(e: BaseException) -> bool:
    s = str(e)
    return "-4116" in s or "duplicat" in s.lower()


class UnprotectedFillError(RuntimeError):
    """An entry FILLED but no stop-loss could be placed for it.

    Raised only AFTER the client has already tried to close the position at
    market (God's rule, 2026-10-13: retry the stop 2–3×, else close + alert).
    It carries everything the bot needs to RECORD what happened — the entry
    fill and, if it went through, the emergency close — because a bare
    exception here used to reach `_maybe_enter`'s generic handler, which said
    "entry failed" about a position that was open and naked, and never wrote
    the fill at all.

    `closed` False means every close attempt failed too: the position is OPEN
    and UNPROTECTED and the bot must keep trying.
    """

    def __init__(self, *, side: str, filled_qty: float, fill_price: float,
                 filled_as: str, entry: dict[str, Any] | None,
                 sl_error: BaseException, closed: bool,
                 close_order: dict[str, Any] | None,
                 close_error: BaseException | None,
                 sl_price: float | None = None) -> None:
        self.side = side
        self.sl_price = sl_price
        self.filled_qty = filled_qty
        self.fill_price = fill_price
        self.filled_as = filled_as
        self.entry = entry
        self.sl_error = sl_error
        self.closed = closed
        self.close_order = close_order
        self.close_error = close_error
        super().__init__(
            f"{side} {filled_qty} filled @ {fill_price} but NO STOP could be placed "
            f"({sl_error}); emergency close "
            + ("succeeded" if closed else f"FAILED ({close_error})"))


@dataclass
class Position:
    symbol: str
    side: str        # "long" | "short" | "flat"
    qty: float       # absolute BTC qty (>= 0)
    entry_price: float
    unrealized_pnl: float
    margin_used: float


class BinanceClient:
    """Thin facade over ccxt.binanceusdm with bot-friendly methods.

    Hedge mode + position_side support: when running two bots on one account,
    each instance MUST pass its `position_side` ("LONG" or "SHORT") to the
    exchange so Binance tracks both legs independently. The bot reads
    `hedge.enabled` from params.yaml and constructs the client accordingly.
    """

    def __init__(self, ex: ccxt.Exchange, env: str,
                 hedge_mode: bool = False,
                 coid_prefix: str = _COID_PREFIX_DEFAULT) -> None:
        self.ex = ex
        self.env = env
        self.hedge_mode = hedge_mode
        self.coid_prefix = coid_prefix

    @classmethod
    def from_env(cls, hedge_mode: bool = False,
                 coid_prefix: str = _COID_PREFIX_DEFAULT) -> BinanceClient:
        env = get_env()
        api_key, api_secret = get_api_credentials()
        ex = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "timeout": DEFAULT_TIMEOUT_MS,
            "options": {"defaultType": "future", "adjustForTimeDifference": True},
        })
        if env == "testnet":
            ex.set_sandbox_mode(True)
            log.info("BinanceClient: SANDBOX/TESTNET mode")
        else:
            log.info("BinanceClient: MAINNET mode (real money)")
        ex.load_markets()
        if hedge_mode:
            log.info("BinanceClient: HEDGE MODE on (positionSide will be sent on every order)")
        return cls(ex=ex, env=env, hedge_mode=hedge_mode, coid_prefix=coid_prefix)

    def _position_side(self, side: str) -> str | None:
        """Return positionSide for Binance Futures hedge mode.

        side: "long" | "short" — direction of the trade we're opening.
        Returns "LONG" / "SHORT" when hedge mode is on; None otherwise (one-way
        mode — Binance rejects positionSide when account is in one-way mode).
        """
        if not self.hedge_mode:
            return None
        return "LONG" if side == "long" else "SHORT"

    # --- market data --------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        raw = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.set_index("ts")
        return df

    def fetch_funding_rate(self, symbol: str) -> float:
        """Most-recent funding rate (per 8h, decimal). 0.0 on failure."""
        try:
            r = self.ex.fetch_funding_rate(symbol)
            rate = r.get("fundingRate")
            return float(rate) if rate is not None else 0.0
        except Exception as e:
            log.warning("fetch_funding_rate failed: %s", e)
            return 0.0

    # --- account ------------------------------------------------------------
    def fetch_equity_usdt(self) -> float:
        """Total USDT-denominated equity (wallet + unrealized PnL).

        Raises EquityUnavailable when either key is missing/None/non-numeric/
        non-finite. NEVER substitutes 0: a zero read looks like a total loss to
        the principal kill switch and would flatten + HALT a healthy leg.

        Side effect: caches the two components of THIS read on
        `last_wallet_usdt` / `last_unrealized_usdt` (None when unavailable) so
        the bot can tell a price move (unrealized) from a wallet move
        (realized/transfer) without a second exchange call."""
        self.last_wallet_usdt: float | None = None
        self.last_unrealized_usdt: float | None = None
        bal = self.ex.fetch_balance({"type": "future"})
        info = bal.get("info", {}) or {}
        try:
            wallet = float(info["totalWalletBalance"])
            unreal = float(info["totalUnrealizedProfit"])
        except (KeyError, TypeError, ValueError) as e:
            raise EquityUnavailable(
                f"equity unreadable from balance payload: {type(e).__name__}: {e}"
            ) from e
        total = wallet + unreal
        if not math.isfinite(total):
            raise EquityUnavailable(f"equity non-finite: wallet={wallet} unreal={unreal}")
        self.last_wallet_usdt, self.last_unrealized_usdt = wallet, unreal
        return total

    def fetch_income(
        self, start_ms: int, end_ms: int | None = None,
        income_type: str | None = None,
        page_limit: int = 1000, max_pages: int = 50,
        caller_log: Any = None,
    ) -> list[dict[str, Any]]:
        """Read-only paginated GET /fapi/v1/income from start_ms (ascending).

        Returns raw income rows (each has incomeType, income, asset, tranId,
        time). Dedup/idempotency is the CALLER's job (by tranId) — this method
        only reads. NEVER places, cancels, or modifies anything.

        Used by the principal anchor (exchange/principal.py) to build P from
        TRANSFER/DEPOSIT/WITHDRAW events. The time-cursor pagination is for
        completeness of the fetch; same-ms boundary rows are re-covered on the
        next overlapping call and de-duplicated by tranId in the ledger.

        `caller_log`: when the page cap is hit (older rows TRUNCATED), the
        warning is surfaced to this logger too — not just the module logger — so
        a leg's own log/alerts pipeline sees a silent-transfer-drop risk. Pass
        the bot's `log` here.
        """
        rows: list[dict[str, Any]] = []
        cursor = int(start_ms)
        for _ in range(max_pages):
            params: dict[str, Any] = {"startTime": cursor, "limit": page_limit}
            if end_ms is not None:
                params["endTime"] = int(end_ms)
            if income_type:
                params["incomeType"] = income_type
            page = self.ex.fapiPrivateGetIncome(params)
            if not page:
                break
            rows.extend(page)
            if len(page) < page_limit:
                break
            cursor = int(page[-1].get("time") or cursor) + 1
        else:
            (caller_log or log).warning(
                "fetch_income hit the %d-page cap (%d rows) for incomeType=%s — "
                "older principal rows may be TRUNCATED; a dropped TRANSFER would "
                "understate deposited principal P. Raise max_pages or narrow the "
                "window.",
                max_pages, len(rows), income_type or "ALL",
            )
        return rows

    def fetch_position(self, symbol: str) -> Position:
        positions = self.ex.fetch_positions([symbol])
        for p in positions:
            if p.get("symbol") != symbol:
                continue
            qty = float(p.get("contracts") or 0.0)
            if qty == 0:
                continue
            side = "long" if (p.get("side") or "").lower() == "long" else "short"
            return Position(
                symbol=symbol, side=side, qty=abs(qty),
                entry_price=float(p.get("entryPrice") or 0.0),
                unrealized_pnl=float(p.get("unrealizedPnl") or 0.0),
                margin_used=float(p.get("initialMargin") or 0.0),
            )
        return Position(symbol=symbol, side="flat", qty=0.0,
                        entry_price=0.0, unrealized_pnl=0.0, margin_used=0.0)

    # --- order management ---------------------------------------------------
    def cancel_open_orders(self, symbol: str,
                           coid_prefix: str | None = None,
                           report_unknown: bool = False) -> int | None:
        """Cancel open orders for `symbol` — plain AND algo/conditional.

        When `coid_prefix` is given, only orders whose clientOrderId (or
        clientAlgoId) starts with that prefix are cancelled — manual orders and
        orders from other bot legs are left untouched.  When `coid_prefix` is
        None, all open orders for the symbol are cancelled.

        Two sweeps, independently fault-isolated: Binance migrated conditional
        orders (STOP_MARKET / TAKE_PROFIT_MARKET triggers — i.e. our brackets)
        to the algo system, where they rest under /fapi/v1/openAlgoOrders and
        are INVISIBLE to /fapi/v1/openOrders. A plain-only sweep therefore sees
        nothing to cancel and silently leaves the surviving bracket sibling
        orphaned — the blind spot that broke the reprotect loop on 2026-07-22
        and re-orphaned brackets on the live v1 account (found 2026-07-26).

        Returns the count of orders actually cancelled (both kinds). With
        `report_unknown=True`, returns None instead when either book could not
        be read (so "swept 0" is not confused with "could not look"); default
        callers keep the best-effort int.
        """
        plain = self._cancel_plain_orders(symbol, coid_prefix)
        algo = self._cancel_algo_orders(symbol, coid_prefix)
        if report_unknown and (plain is None or algo is None):
            return None
        return (plain or 0) + (algo or 0)

    def _cancel_plain_orders(self, symbol: str, coid_prefix: str | None) -> int | None:
        try:
            orders = self.ex.fetch_open_orders(symbol=symbol)
        except Exception as e:
            log.warning("cancel_open_orders fetch failed: %s", e)
            return None
        n = 0
        for o in orders:
            if coid_prefix is not None:
                # ccxt may surface clientOrderId in the top-level field or nested
                # inside the raw `info` dict — check both. External adapters can
                # also return a missing or non-string id; treat those as
                # non-matching rather than letting `.startswith` raise and abort
                # the whole sweep.
                coid = o.get("clientOrderId") or (o.get("info") or {}).get("clientOrderId")
                if not isinstance(coid, str) or not coid.startswith(coid_prefix):
                    continue
            try:
                self.ex.cancel_order(o["id"], symbol)
                n += 1
            except Exception as e:
                log.warning("cancel_order %s failed: %s", o.get("id"), e)
        return n

    def _market_id(self, symbol: str) -> str:
        """Exchange id for the raw fapi endpoints, which don't speak ccxt symbols."""
        try:
            return self.ex.market(symbol)["id"]
        except Exception:
            # "BTC/USDT:USDT" -> "BTCUSDT" (markets not loaded yet, e.g. boot)
            return symbol.split(":")[0].replace("/", "")

    def fetch_algo_orders(self, symbol: str) -> tuple[list[dict[str, Any]], bool]:
        """Resting algo/conditional trigger orders — READ ONLY.

        Returns ``(rows, ok)``. **`ok` is the whole point of the signature.**

        `_place_brackets` lands SL/TP on Binance's algo endpoint, which
        `fetch_open_orders()` (/fapi/v1/openOrders) does NOT return. Anything
        asking "is the bracket still there?" must read this too, or it sees a
        healthy bracket as missing — the false negative behind the 2026-07-22
        -4045 re-place spam.

        This never raises, so a bare list return would make "there are no algo
        orders" and "I could not ask" the same value — and a caller that acts on
        the ambiguity re-places a live bracket on a 5-second network blip, which
        is that same July bug wearing a different hat. Checking whether the ccxt
        method *exists* is not sufficient: the endpoint can be present and the
        CALL still fail. `ok=False` therefore covers all three — missing
        endpoint, failed call, unexpected payload shape — and callers that would
        place orders must abstain unless `ok` is True.
        """
        fetch = getattr(self.ex, "fapiPrivateGetOpenAlgoOrders", None)
        if fetch is None:
            log.warning("ccxt build lacks openAlgoOrders — algo brackets invisible")
            return [], False
        try:
            rows = fetch({"symbol": self._market_id(symbol)})
        except Exception as e:
            log.warning("openAlgoOrders fetch failed: %s", e)
            return [], False
        if not isinstance(rows, list):
            log.warning("openAlgoOrders returned %s, not a list", type(rows).__name__)
            return [], False
        return rows, True

    def _cancel_algo_orders(self, symbol: str, coid_prefix: str | None) -> int | None:
        """Sweep resting algo/conditional trigger orders (raw fapi endpoints).

        The raw endpoint speaks exchange ids ("BTCUSDT"), not ccxt symbols, and
        the client order id lives in `clientAlgoId`. Missing endpoint (old ccxt)
        degrades to a no-op with a warning rather than breaking the sweep.
        Returns None when the book could not be read / swept at all.
        """
        cancel = getattr(self.ex, "fapiPrivateDeleteAlgoOrder", None)
        if cancel is None:
            log.warning("ccxt build lacks algo-order endpoints — "
                        "conditional brackets cannot be swept")
            return None
        market_id = self._market_id(symbol)
        # Sweeping is best-effort: an unreadable book means nothing to cancel
        # here, and the boot/pre-entry sweeps will retry.
        rows, ok = self.fetch_algo_orders(symbol)
        if not ok:
            return None
        n = 0
        for r in rows:
            if coid_prefix is not None:
                coid = r.get("clientAlgoId")
                if not isinstance(coid, str) or not coid.startswith(coid_prefix):
                    continue
            try:
                cancel({"symbol": market_id, "algoId": r["algoId"]})
                n += 1
            except Exception as e:
                log.warning("cancel algo order %s failed: %s", r.get("algoId"), e)
        return n

    def fetch_mark_price(self, symbol: str) -> float | None:
        """Current MARK price (what our STOP_MARKETs trigger on). None on failure."""
        try:
            r = self.ex.fapiPublicGetPremiumIndex({"symbol": self._market_id(symbol)})
            v = float((r or {}).get("markPrice") or 0.0)
            return v if v > 0 else None
        except Exception as e:
            log.warning("fetch_mark_price failed: %s", e)
            return None

    def place_tagged_stop(
        self, symbol: str, side: str, qty: float, stop_price: float,
        client_order_id_root: str, leg: str,
    ) -> dict[str, Any]:
        """Place ONE reduce-only STOP_MARKET for an open `side` position, tagged
        `{prefix}{root}-{leg}` — and never untagged.

        Unlike `_create_order_with_coid_retry` there is NO untagged fallback,
        on purpose. That wrapper retries without a clientOrderId when Binance
        rejects the id, which is harmless for an entry but dangerous for a stop
        placed mid-trade: a placement that timed out on our side yet was
        accepted, then retried, gets "duplicate id" — and the fallback would
        rest a SECOND, untagged stop that no prefix-scoped sweep can ever find.
        It would outlive the trade and could close a later position — the
        orphan-bracket bug class. Callers must check the algo book for an
        existing `-{leg}` order before calling this.
        """
        coid = _coid(client_order_id_root, leg, self.coid_prefix)
        if coid is None:
            raise ValueError(f"cannot build a valid clientOrderId from root={client_order_id_root!r}")
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            raise ValueError(f"qty rounded to <= 0 for {symbol}")
        params: dict[str, Any] = {"stopPrice": float(stop_price), "reduceOnly": True,
                                  "workingType": "MARK_PRICE", "newClientOrderId": coid}
        if (pos_side := self._position_side(side)) is not None:
            params["positionSide"] = pos_side
        bracket_side = "sell" if side == "long" else "buy"
        return self.ex.create_order(symbol, "STOP_MARKET", bracket_side, qty, None, params=params)

    def cancel_algo_by_coid(self, symbol: str, coid: str) -> bool:
        """Cancel the ONE resting algo order whose clientAlgoId == `coid`.

        Returns True only if it was found and the cancel call succeeded. An
        unreadable book, a missing order, or a failed cancel all return False —
        the caller decides whether that is alarming.
        """
        cancel = getattr(self.ex, "fapiPrivateDeleteAlgoOrder", None)
        if cancel is None:
            return False
        rows, ok = self.fetch_algo_orders(symbol)
        if not ok:
            return False
        for r in rows:
            if r.get("clientAlgoId") == coid:
                try:
                    cancel({"symbol": self._market_id(symbol), "algoId": r["algoId"]})
                    return True
                except Exception as e:
                    log.warning("cancel algo order %s (%s) failed: %s", r.get("algoId"), coid, e)
                    return False
        return False

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.ex.set_leverage(int(leverage), symbol)
        except Exception as e:
            # Binance returns 400 if leverage is already set; safe to swallow.
            log.info("set_leverage(%dx, %s): %s", leverage, symbol, e)

    def _round_qty(self, symbol: str, qty: float) -> float:
        m = self.ex.market(symbol)
        step = m.get("precision", {}).get("amount", 3)
        if isinstance(step, int):
            return round(qty, step)
        try:
            return math.floor(qty / float(step)) * float(step)
        except Exception:
            return round(qty, 3)

    def _create_order_with_coid_retry(
        self, symbol: str, order_type: str, side: str, qty: float,
        price: float | None, params: dict[str, Any],
    ) -> dict[str, Any]:
        """create_order wrapper that retries without newClientOrderId if the
        exchange rejects the COID specifically.

        Causes of COID rejection in practice: duplicate (effectively impossible
        with ms-precision signal_id), invalid chars, length > 36, or Binance
        silently tightening their regex. We never want a tagging failure to
        prevent a real trade from being placed.
        """
        coid = params.get("newClientOrderId")
        try:
            return self.ex.create_order(symbol, order_type, side, qty, price, params=params)
        except (ccxt.InvalidOrder, ccxt.BadRequest, ccxt.OperationFailed) as e:
            msg = str(e).lower()
            if coid and ("client" in msg or "newclientorderid" in msg or "duplicate" in msg):
                log.warning("clientOrderId %r rejected (%s) — retrying untagged", coid, e)
                params_clean = {k: v for k, v in params.items() if k != "newClientOrderId"}
                return self.ex.create_order(symbol, order_type, side, qty, price, params=params_clean)
            raise

    def market_order_with_bracket(
        self, symbol: str, side: str, qty: float,
        sl_price: float, tp_price: float,
        client_order_id_root: str | None = None,
        place_tp: bool = True,
    ) -> dict[str, Any]:
        """Place market entry + stop-market SL (+ optional take-profit-market TP).

        Returns {"entry": order, "sl": order, "tp": order | None} — `tp` is
        None when place_tp=False (channel-exit strategies manage the profit
        side themselves).
        Brackets are reduce-only and trigger off mark price.

        place_tp=False places entry + SL ONLY and returns "tp": None — for
        strategies that manage their own exit (donchian-v3 closes on the live
        Donchian channel cross, so it has no TP leg). `tp_price` is ignored when
        place_tp is False. Default True preserves the v1/multifactor behaviour
        byte-for-byte.

        Hedge mode: when self.hedge_mode is True, every leg includes the
        position_side matching the entry (LONG for long entries, SHORT for
        shorts) — even reduce-only legs. Binance uses positionSide to identify
        which position SLOT the order touches, not the order direction.

        If `client_order_id_root` is provided, each leg gets a Binance
        clientOrderId of form {prefix}<root>-{e|s|t} so consolidate-investment
        can attribute the trade. See `_coid` for the scheme.
        """
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            raise ValueError(f"qty rounded to <= 0 for {symbol}")
        ccxt_side = "buy" if side == "long" else "sell"
        pos_side = self._position_side(side)   # LONG/SHORT/None

        entry_params: dict[str, Any] = {"reduceOnly": False}
        if (coid := _coid(client_order_id_root, "e", self.coid_prefix)):
            entry_params["newClientOrderId"] = coid
        if pos_side is not None:
            entry_params["positionSide"] = pos_side
        entry = self._create_order_with_coid_retry(
            symbol, "market", ccxt_side, qty, None, entry_params)

        # From here a position is OPEN. Never let an exception escape without
        # either a resting stop or an attempted close — see _protect_fill.
        sl, tp, tp_error = self._protect_fill(
            symbol, side, qty, sl_price, tp_price, client_order_id_root, place_tp,
            entry=entry, fill_price=_order_avg(entry), filled_as="market")
        out: dict[str, Any] = {"entry": entry, "sl": sl, "tp": tp}
        if tp_error is not None:
            out["tp_error"] = tp_error
        return out

    def _place_brackets(
        self, symbol: str, side: str, qty: float,
        fill_price: float, sl_distance: float, tp_distance: float,
        client_order_id_root: str | None = None,
        place_tp: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Place SL (+ optional TP) brackets after a fill. place_tp=False
        returns (sl, None) and skips the TP leg — see market_order_with_bracket."""
        bracket_side = "sell" if side == "long" else "buy"
        pos_side = self._position_side(side)
        if side == "long":
            sl_price = fill_price - sl_distance
            tp_price = fill_price + tp_distance
        else:
            sl_price = fill_price + sl_distance
            tp_price = fill_price - tp_distance
        sl_params: dict[str, Any] = {"stopPrice": float(sl_price), "reduceOnly": True,
                                      "workingType": "MARK_PRICE"}
        if (coid := _coid(client_order_id_root, "s", self.coid_prefix)):
            sl_params["newClientOrderId"] = coid
        if pos_side is not None:
            sl_params["positionSide"] = pos_side
        sl = self._create_order_with_coid_retry(
            symbol, "STOP_MARKET", bracket_side, qty, None, sl_params)
        tp = None
        if place_tp:
            tp_params: dict[str, Any] = {"stopPrice": float(tp_price), "reduceOnly": True,
                                          "workingType": "MARK_PRICE"}
            if (coid := _coid(client_order_id_root, "t", self.coid_prefix)):
                tp_params["newClientOrderId"] = coid
            if pos_side is not None:
                tp_params["positionSide"] = pos_side
            tp = self._create_order_with_coid_retry(
                symbol, "TAKE_PROFIT_MARKET", bracket_side, qty, None, tp_params)
        return sl, tp

    # --- no-stop protection (God's rule, 2026-10-13) -------------------------
    # "If a stop-loss can't be placed (or re-placed), retry 2–3×; if it still
    # can't, close the position at market and alert." The healthy path is
    # untouched: attempt 1 is the exact call the bracket code always made.
    SL_ATTEMPTS = 3
    SL_RETRY_BACKOFF_S = (0.5, 1.5)
    CLOSE_ATTEMPTS = 3
    CLOSE_RETRY_BACKOFF_S = (1.0, 2.0)

    def _leg_params(self, stop_price: float, root: str | None, leg: str,
                    pos_side: str | None) -> dict[str, Any]:
        params: dict[str, Any] = {"stopPrice": float(stop_price), "reduceOnly": True,
                                  "workingType": "MARK_PRICE"}
        if (coid := _coid(root, leg, self.coid_prefix)):
            params["newClientOrderId"] = coid
        if pos_side is not None:
            params["positionSide"] = pos_side
        return params

    def order_resting(self, symbol: str, coid: str, side: str) -> bool:
        """Is an order with EXACTLY `coid`, on the side that CLOSES a `side`
        position, resting in either book? An unreadable book counts as "not
        seen" — callers use this only to avoid a duplicate, never to conclude
        a position is protected."""
        close_side = "SELL" if side == "long" else "BUY"
        rows, ok = self.fetch_algo_orders(symbol)
        if ok and any(r.get("clientAlgoId") == coid
                      and str(r.get("side") or "").upper() == close_side for r in rows):
            return True
        try:
            plain = self.ex.fetch_open_orders(symbol=symbol)
        except Exception as e:
            log.warning("order_resting: fetch_open_orders failed: %s", e)
            return False
        for o in plain or []:
            info = o.get("info") or {}
            if ((o.get("clientOrderId") or info.get("clientOrderId")) == coid
                    and str(o.get("side") or info.get("side") or "").upper() == close_side):
                return True
        return False

    def _place_sl_with_retry(self, symbol: str, side: str, qty: float,
                             sl_params: dict[str, Any]) -> dict[str, Any]:
        """Place the stop, retrying up to SL_ATTEMPTS in total. Raises the last
        error if no stop could be placed.

        - Attempt 1 is byte-identical to the historical call (coid-retry
          wrapper), so a healthy entry makes exactly the calls it always did.
        - Retries go out TAGGED ONLY, and only after checking the books for our
          exact id: a timed-out attempt may have been ACCEPTED, and a blind
          retry would get "duplicate id" — which the coid-retry wrapper answers
          by resting a second, UNTAGGED stop no sweep can find (the
          orphan-bracket bug class).
        - -2021 (would immediately trigger) is never retried: price is already
          through the stop, so retrying only lets the loss grow.
        - An untagged stop (no root) is not retried: there is no id to check
          the book for, so a retry could duplicate it.
        """
        bracket_side = "sell" if side == "long" else "buy"
        coid = sl_params.get("newClientOrderId")
        last: BaseException | None = None
        for attempt in range(self.SL_ATTEMPTS):
            if attempt:
                if coid is None:
                    break
                time.sleep(self.SL_RETRY_BACKOFF_S[min(attempt - 1,
                                                       len(self.SL_RETRY_BACKOFF_S) - 1)])
                if self.order_resting(symbol, coid, side):
                    log.warning("stop %s found resting after a failed attempt — "
                                "the earlier placement landed", coid)
                    return {"id": None, "clientOrderId": coid, "recovered": True}
            try:
                if attempt == 0:
                    return self._create_order_with_coid_retry(
                        symbol, "STOP_MARKET", bracket_side, qty, None, sl_params)
                return self.ex.create_order(symbol, "STOP_MARKET", bracket_side, qty,
                                            None, params=sl_params)
            except Exception as e:
                last = e
                log.error("stop placement attempt %d/%d failed: %s",
                          attempt + 1, self.SL_ATTEMPTS, e)
                if is_would_trigger(e):
                    break
                if is_nonce_error(e):
                    self.resync_time()
                if attempt and coid is not None and _is_duplicate_id(e):
                    # Binance reserved our id: an earlier, timed-out attempt is
                    # in flight and has not reached the book yet. Give it a
                    # moment rather than declaring "no stop" and closing.
                    for _ in range(3):
                        time.sleep(1.0)
                        if self.order_resting(symbol, coid, side):
                            log.warning("stop %s appeared after a duplicate-id "
                                        "rejection — the earlier attempt landed", coid)
                            return {"id": None, "clientOrderId": coid, "recovered": True}
        assert last is not None
        # The final attempt may itself have timed out yet landed.
        if coid is not None and not is_would_trigger(last) \
                and self.order_resting(symbol, coid, side):
            log.warning("stop %s found resting after the last attempt", coid)
            return {"id": None, "clientOrderId": coid, "recovered": True}
        raise last

    def resync_time(self) -> None:
        """Re-sync ccxt's clock offset (after -1021 InvalidNonce)."""
        try:
            self.ex.load_time_difference()
        except Exception as e:
            log.warning("load_time_difference() failed: %s", e)

    def cancel_root_orders(self, symbol: str, root: str | None) -> int:
        """Cancel resting orders (both books) of ONE entry root only:
        `{prefix}{root}-*`. Never the whole prefix — another root's live
        bracket must survive an emergency on this one."""
        if not root:
            return 0
        scope = f"{self.coid_prefix}{root}-"
        return self._cancel_plain_orders(symbol, scope) + self._cancel_algo_orders(symbol, scope)

    def emergency_close(self, symbol: str, root: str | None,
                        side: str | None = None, qty: float | None = None,
                        coid_root: str | None = None,
                        ) -> tuple[bool, dict[str, Any] | None, BaseException | None]:
        """Reduce-only market close tagged `-ns` (no stop), up to CLOSE_ATTEMPTS.

        Returns (closed, order, last_error). `closed` with order None means the
        exchange was already flat. Never raises — the caller must record and
        alert either way.

        With a KNOWN `side` + `qty` (the entry we just filled, or a position the
        caller has just verified is this root's) the first attempt is sent
        directly, WITHOUT first reading the position: positionRisk can lag a
        fill by a round-trip, and a read-gated close would take that lag for
        "flat" and declare the emergency over with the position still open. A
        reduce-only order cannot open or flip anything. If it is rejected
        (nothing to reduce), later attempts read the position first.

        `coid_root` overrides the root used for the close's own `-ns` id (retries
        pass a fresh one so a reused id cannot hit the duplicate path); sweeps
        still use `root`.

        Only this root's own orders are swept (before AND after the close — a
        timed-out stop can land late), never the whole prefix.
        """
        last: BaseException | None = None
        blind = side in ("long", "short") and float(qty or 0) > 0
        for attempt in range(self.CLOSE_ATTEMPTS):
            if attempt:
                time.sleep(self.CLOSE_RETRY_BACKOFF_S[min(attempt - 1,
                                                          len(self.CLOSE_RETRY_BACKOFF_S) - 1)])
            try:
                if blind and attempt == 0:
                    assert side is not None and qty is not None
                    c_side, c_qty = side, float(qty)
                else:
                    p = self.fetch_position(symbol)
                    if p.side == "flat" or p.qty == 0:
                        self._sweep_root_quietly(symbol, root)
                        return True, None, None
                    c_side, c_qty = p.side, p.qty
                order = self._reduce_only_close(symbol, c_side, c_qty, root, coid_root)
                self._sweep_root_quietly(symbol, root)
                return True, order, None
            except Exception as e:
                last = e
                log.error("EMERGENCY close attempt %d/%d failed: %s",
                          attempt + 1, self.CLOSE_ATTEMPTS, e)
                if is_nonce_error(e):
                    self.resync_time()
        return False, None, last

    def _sweep_root_quietly(self, symbol: str, root: str | None) -> None:
        try:
            self.cancel_root_orders(symbol, root)
        except Exception as e:
            log.warning("emergency close: root sweep failed: %s", e)

    def _reduce_only_close(self, symbol: str, side: str, qty: float,
                           root: str | None, coid_root: str | None = None) -> dict[str, Any]:
        """Sweep THIS root's legs, then a reduce-only market close of a KNOWN
        position."""
        self._sweep_root_quietly(symbol, root)
        params: dict[str, Any] = {"reduceOnly": True}
        if (coid := _coid(coid_root or root, "ns", self.coid_prefix)):
            params["newClientOrderId"] = coid
        if (pos_side := self._position_side(side)) is not None:
            params["positionSide"] = pos_side
        return self._create_order_with_coid_retry(
            symbol, "market", "sell" if side == "long" else "buy",
            self._round_qty(symbol, qty), None, params)

    def cancel_entry_order(self, symbol: str, root: str | None) -> bool | None:
        """Cancel a possibly-resting ENTRY order `{prefix}{root}-e` by
        origClientOrderId, then CONFIRM it is gone with a fresh open-orders
        read. A limit entry whose create call timed out can still be resting
        (GTC) and fill minutes later.

        Returns True = confirmed not resting, False = still resting after the
        cancel, None = could not tell (no id, or the book was unreadable).
        "Unknown order" on the cancel is the normal answer for a market entry
        or an already-filled/cancelled limit, and is not an error.
        """
        coid = _coid(root, "e", self.coid_prefix)
        if coid is None:
            return None
        try:
            self.ex.cancel_order(None, symbol, params={"origClientOrderId": coid})
        except Exception as e:
            log.info("cancel entry %s: %s", coid, e)
        try:
            plain = self.ex.fetch_open_orders(symbol=symbol)
        except Exception as e:
            log.warning("cancel entry %s: open-orders read failed: %s", coid, e)
            return None
        for o in plain or []:
            if (o.get("clientOrderId") or (o.get("info") or {}).get("clientOrderId")) == coid:
                return False
        return True

    def protect_open_position(self, symbol: str, side: str, qty: float,
                              sl_price: float, tp_price: float, root: str | None,
                              place_tp: bool, fill_price: float,
                              close_on_failure: bool = True,
                              ) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
        """The no-stop rule for a position that is ALREADY open (a late fill
        after an entry error): stop with retry, else close + UnprotectedFillError.
        close_on_failure=False never closes (the position is not wholly
        attributable to us): it raises UnprotectedFillError with closed=False
        and no close attempted."""
        return self._protect_fill(symbol, side, self._round_qty(symbol, qty), sl_price,
                                  tp_price, root, place_tp, entry=None,
                                  fill_price=fill_price, filled_as="late_fill",
                                  close_on_failure=close_on_failure)

    def entry_order_filled(self, symbol: str, root: str | None) -> float | None:
        """Filled qty of the entry order `{prefix}{root}-e`, read by
        origClientOrderId — the only way to attribute an open position to THIS
        root (positionRisk carries no order ids). None = could not read it."""
        coid = _coid(root, "e", self.coid_prefix)
        if coid is None:
            return None
        try:
            o = self.ex.fetch_order(None, symbol, params={"origClientOrderId": coid})
        except Exception as e:
            log.warning("entry_order_filled %s: %s", coid, e)
            return None
        try:
            return float((o or {}).get("filled"))
        except (TypeError, ValueError):
            return None

    def _protect_fill(self, symbol: str, side: str, qty: float,
                      sl_price: float, tp_price: float, root: str | None,
                      place_tp: bool, *, entry: dict[str, Any] | None,
                      fill_price: float | None, filled_as: str,
                      close_on_failure: bool = True,
                      ) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
        """Bracket a position that has ALREADY FILLED. Returns (sl, tp, tp_error).

        No stop after retries ⇒ emergency close, then raise UnprotectedFillError
        (with the close outcome) so the bot records both fills and alerts. A
        TP failure with a resting stop is NOT a reason to close: it is returned
        as `tp_error` for the bot to alert on, and reprotect (v1) or the trend
        exit (donchian/SOL) handles the profit side.
        """
        bracket_side = "sell" if side == "long" else "buy"
        pos_side = self._position_side(side)
        sl_params = self._leg_params(sl_price, root, "s", pos_side)
        try:
            sl = self._place_sl_with_retry(symbol, side, qty, sl_params)
        except Exception as e:
            log.critical("NO STOP for %s %s %.6f — closing at market: %s",
                         symbol, side, qty, e)
            if not close_on_failure:
                raise UnprotectedFillError(
                    side=side, filled_qty=float(qty),
                    fill_price=float(fill_price or 0.0), filled_as=filled_as,
                    entry=entry, sl_error=e, closed=False, close_order=None,
                    close_error=None, sl_price=float(sl_price)) from e
            closed, order, cerr = self.emergency_close(symbol, root, side=side, qty=qty)
            raise UnprotectedFillError(
                side=side, filled_qty=float(qty),
                fill_price=float(fill_price or 0.0), filled_as=filled_as,
                entry=entry, sl_error=e, closed=closed, close_order=order,
                close_error=cerr, sl_price=float(sl_price)) from e
        tp = None
        tp_error: str | None = None
        if place_tp:
            tp_params = self._leg_params(tp_price, root, "t", pos_side)
            try:
                tp = self._create_order_with_coid_retry(
                    symbol, "TAKE_PROFIT_MARKET", bracket_side, qty, None, tp_params)
            except Exception as e:
                tp_error = str(e) or type(e).__name__
                log.error("TP placement failed (stop is resting, position kept): %s", e)
        return sl, tp, tp_error

    def _protect_fill_at(self, symbol: str, side: str, qty: float,
                         fill_price: float, sl_distance: float, tp_distance: float,
                         root: str | None, place_tp: bool, *,
                         entry: dict[str, Any] | None, filled_as: str,
                         ) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
        """_protect_fill with prices from distances — the same arithmetic as
        _place_brackets, so the limit path's levels are unchanged."""
        if side == "long":
            sl_price = fill_price - sl_distance
            tp_price = fill_price + tp_distance
        else:
            sl_price = fill_price + sl_distance
            tp_price = fill_price - tp_distance
        return self._protect_fill(symbol, side, qty, sl_price, tp_price, root,
                                  place_tp, entry=entry, fill_price=fill_price,
                                  filled_as=filled_as)

    def place_tagged_leg(self, symbol: str, side: str, qty: float, stop_price: float,
                         client_order_id_root: str, leg: str) -> dict[str, Any]:
        """Like place_tagged_stop, for either bracket leg: "s"/"sb" →
        STOP_MARKET, "t" → TAKE_PROFIT_MARKET. Tagged ONLY — no untagged
        fallback, for the reason given on place_tagged_stop."""
        if leg != "t":
            return self.place_tagged_stop(symbol, side, qty, stop_price,
                                          client_order_id_root, leg)
        coid = _coid(client_order_id_root, leg, self.coid_prefix)
        if coid is None:
            raise ValueError(f"cannot build a valid clientOrderId from root={client_order_id_root!r}")
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            raise ValueError(f"qty rounded to <= 0 for {symbol}")
        params = self._leg_params(stop_price, client_order_id_root, leg,
                                  self._position_side(side))
        bracket_side = "sell" if side == "long" else "buy"
        return self.ex.create_order(symbol, "TAKE_PROFIT_MARKET", bracket_side, qty,
                                    None, params=params)

    def limit_order_with_bracket(
        self, symbol: str, side: str, qty: float,
        limit_price: float, sl_distance: float, tp_distance: float,
        timeout_s: float = 20.0, poll_s: float = 2.0,
        client_order_id_root: str | None = None,
        place_tp: bool = True,
    ) -> dict[str, Any]:
        """Place a maker-style limit entry; fall back to market if not filled.

        place_tp=False places entry + SL ONLY (returned "tp" is None) for
        channel-exit strategies; `tp_distance` is ignored then. Default True
        preserves the v1/multifactor behaviour byte-for-byte.

        sl_distance and tp_distance are PRICE units. Brackets are computed AFTER
        the entry fills (based on actual fill price), so a slow limit fill that
        gets a better price than `limit_price` propagates correctly into the
        SL/TP geometry.

        Returns {"entry", "sl", "tp", "filled_as", "fill_price", "filled_qty"}.
        filled_as is one of: "limit", "limit_partial" (some filled, market
        fallback skipped to avoid doubling position), or "market_fallback".
        """
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            raise ValueError(f"qty rounded to <= 0 for {symbol}")
        ccxt_side = "buy" if side == "long" else "sell"
        pos_side = self._position_side(side)

        entry_params: dict[str, Any] = {"reduceOnly": False, "timeInForce": "GTC"}
        if (coid := _coid(client_order_id_root, "e", self.coid_prefix)):
            entry_params["newClientOrderId"] = coid
        if pos_side is not None:
            entry_params["positionSide"] = pos_side
        entry = self._create_order_with_coid_retry(
            symbol, "limit", ccxt_side, qty, float(limit_price), entry_params)
        order_id = entry["id"]
        log.info("limit %s placed id=%s qty=%.6f @ %.2f (timeout=%.0fs, coid=%s)",
                 ccxt_side, order_id, qty, limit_price, timeout_s,
                 entry_params.get("newClientOrderId", "—"))

        filled_qty = 0.0
        avg_price = float(limit_price)
        elapsed = 0.0
        last_status: str | None = None
        while elapsed < timeout_s:
            time.sleep(poll_s)
            elapsed += poll_s
            try:
                o = self.ex.fetch_order(order_id, symbol)
            except Exception as e:
                log.warning("fetch_order(%s) failed during limit poll: %s", order_id, e)
                continue
            last_status = o.get("status")
            filled_qty = float(o.get("filled") or 0.0)
            avg = o.get("average")
            if avg:
                avg_price = float(avg)
            if last_status in ("closed", "filled") and filled_qty >= qty * 0.999:
                log.info("limit filled fully @ %.2f after %.0fs", avg_price, elapsed)
                sl, tp, tp_error = self._protect_fill_at(
                    symbol, side, filled_qty, avg_price, sl_distance, tp_distance,
                    client_order_id_root, place_tp, entry=o, filled_as="limit")
                return _with_tp_error(
                    {"entry": o, "sl": sl, "tp": tp, "filled_as": "limit",
                     "fill_price": avg_price, "filled_qty": filled_qty}, tp_error)
            if last_status in ("canceled", "rejected", "expired"):
                log.warning("limit ended with status=%s, filled_qty=%.6f",
                            last_status, filled_qty)
                break

        # Timed out OR cancelled exchange-side.
        ended = ("canceled", "rejected", "expired")
        order_dead = last_status in ended
        if not order_dead:
            try:
                self.ex.cancel_order(order_id, symbol)
                order_dead = True
                log.info("cancelled limit after %.0fs (filled=%.6f / %.6f)",
                         elapsed, filled_qty, qty)
            except Exception as e:
                # If the order filled between our last poll and cancel, ccxt
                # raises. The re-fetch below finds out what really happened.
                log.info("cancel race: %s — re-fetching order", e)
            # ALWAYS re-fetch: the poll value is up to poll_s stale and a partial
            # fill can land between the last poll and a SUCCESSFUL cancel.
            try:
                o = self.ex.fetch_order(order_id, symbol)
                filled_qty = max(filled_qty, float(o.get("filled") or 0.0))
                avg = o.get("average")
                if avg:
                    avg_price = float(avg)
                if o.get("status") in ended + ("closed", "filled"):
                    order_dead = True
            except Exception as e2:
                log.warning("post-cancel fetch_order failed: %s", e2)

        # The position is the ground truth for exposure. Read it before any
        # market fallback and bracket THAT qty, not a possibly-stale order read.
        pos_qty: float | None = None
        try:
            p = self.fetch_position(symbol)
            pos_qty = p.qty if p.side == side else 0.0
        except Exception as e:
            log.warning("post-cancel fetch_position failed: %s", e)
        actual_qty = pos_qty if (pos_qty is not None and pos_qty > 0) else filled_qty

        if actual_qty >= qty * 0.999:
            # Filled fully right at the cancel boundary.
            sl, tp, tp_error = self._protect_fill_at(
                symbol, side, self._bracket_qty(symbol, actual_qty), avg_price,
                sl_distance, tp_distance,
                client_order_id_root, place_tp, entry=entry, filled_as="limit")
            return _with_tp_error(
                {"entry": entry, "sl": sl, "tp": tp, "filled_as": "limit",
                 "fill_price": avg_price, "filled_qty": actual_qty}, tp_error)

        if actual_qty > 0:
            # The unfilled remainder may still rest and fill later with no stop.
            remainder_dead = order_dead or self._kill_resting_limit(symbol, order_id)
            # More may have filled between the position read above and the
            # kill. Re-read so the bracket covers everything that is now held.
            try:
                p = self.fetch_position(symbol)
                if p.side == side and p.qty > actual_qty:
                    actual_qty = p.qty
            except Exception as e:
                log.warning("post-kill fetch_position failed: %s", e)

            def _flag_if_resting() -> None:
                # Alert only. Do NOT raise: the position is live (and bracketed
                # when partial_qty > 0), so the caller must still record the
                # entry or exit tracking has no entry row to price against.
                if remainder_dead:
                    return
                msg = (f"limit entry {order_id} partially filled "
                       f"({actual_qty:.6f}) and the remainder MAY STILL REST — "
                       f"check the book")
                log.error(msg)
                try:
                    from alerts import send_alert
                    send_alert("Limit partial: remainder may still rest", msg)
                except Exception:
                    log.exception("limit-entry alert failed")

            # Partial fill — bracket what we got, do NOT fall back to market
            # (would double the position size). Caller can log the size mismatch.
            partial_qty = self._bracket_qty(symbol, actual_qty)
            if partial_qty <= 0:
                log.error("limit partial %.8f rounds to 0 — exposure left "
                          "UNBRACKETED, no market fallback", actual_qty)
                _flag_if_resting()
                return {"entry": entry, "sl": None, "tp": None,
                        "filled_as": "limit_partial",
                        "fill_price": avg_price, "filled_qty": actual_qty,
                        "remainder_may_rest": not remainder_dead}
            log.warning("limit partially filled %.6f / %.6f — bracketing partial, "
                        "skipping market fallback to avoid position doubling",
                        partial_qty, qty)
            sl, tp, tp_error = self._protect_fill_at(
                symbol, side, partial_qty, avg_price, sl_distance, tp_distance,
                client_order_id_root, place_tp, entry=entry, filled_as="limit_partial")
            _flag_if_resting()
            return _with_tp_error(
                {"entry": entry, "sl": sl, "tp": tp, "filled_as": "limit_partial",
                 "fill_price": avg_price, "filled_qty": partial_qty,
                 "remainder_may_rest": not remainder_dead}, tp_error)

        if not order_dead or pos_qty is None:
            # Cannot prove the limit is dead AND the account is flat: a market
            # order now could stack on a late limit fill. Do nothing further.
            # Last resort before giving up: kill the resting limit so it cannot
            # fill later unbracketed. Retry by id, then sweep ONLY this leg's own
            # coid-prefixed orders (never another leg's or a manual order).
            swept = False if not order_dead else True
            if not order_dead:
                swept = self._kill_resting_limit(symbol, order_id)
            may_rest = (not order_dead) and (not swept)
            msg = (f"limit entry {order_id} state unknown after cancel "
                   f"(order_dead={order_dead}, position_read={pos_qty is not None}, "
                   f"cleanup_cancel_ok={swept}) — refusing market fallback; "
                   f"a resting limit order {'MAY STILL REST — check the book' if may_rest else 'was cancelled'}")
            try:
                from alerts import send_alert
                send_alert("Limit entry state unknown", msg)
            except Exception:
                log.exception("limit-entry alert failed")
            raise RuntimeError(msg)

        # Zero fill, order dead, account verified flat — fall back to market for
        # full qty. Reuse the same client_order_id_root with leg "e" — there's
        # only ever ONE entry per signal_id, so this is unambiguous from the
        # importer's view.
        log.info("limit unfilled after %.0fs — falling back to market", elapsed)
        mkt_params: dict[str, Any] = {"reduceOnly": False}
        if (coid := _coid(client_order_id_root, "e", self.coid_prefix)):
            mkt_params["newClientOrderId"] = coid
        if pos_side is not None:
            mkt_params["positionSide"] = pos_side
        market_entry = self._create_order_with_coid_retry(
            symbol, "market", ccxt_side, qty, None, mkt_params)
        try:
            m = self.ex.fetch_order(market_entry["id"], symbol)
            m_avg = m.get("average") or m.get("price") or limit_price
            market_fill = float(m_avg)
        except Exception:
            market_fill = float(limit_price)
        bracket_qty = qty
        try:
            p2 = self.fetch_position(symbol)
            if p2.side == side and p2.qty > 0:
                bracket_qty = p2.qty
        except Exception as e:
            log.warning("post-market fetch_position failed (bracketing intended "
                        "qty): %s", e)
        sl, tp, tp_error = self._protect_fill_at(
            symbol, side, self._bracket_qty(symbol, bracket_qty), market_fill,
            sl_distance, tp_distance,
            client_order_id_root, place_tp, entry=market_entry,
            filled_as="market_fallback")
        return _with_tp_error(
            {"entry": market_entry, "sl": sl, "tp": tp,
             "filled_as": "market_fallback",
             "fill_price": market_fill, "filled_qty": bracket_qty}, tp_error)

    def _kill_resting_limit(self, symbol: str, order_id: Any) -> bool:
        """Last resort: make sure a limit entry cannot fill later unbracketed.
        Retry cancel by id, then sweep ONLY this leg's own coid-prefixed orders
        (never another leg's or a manual order), then VERIFY by re-reading open
        orders (cancel_open_orders swallows read failures and returns 0, so its
        return proves nothing). True only if proven gone."""
        try:
            self.ex.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            log.warning("limit cancel retry failed: %s", e)
        try:
            self.cancel_open_orders(symbol, coid_prefix=self.coid_prefix)
        except Exception as e:
            log.warning("limit prefix sweep failed: %s", e)
        try:
            resting = self.ex.fetch_open_orders(symbol=symbol)
            return not any(str(o.get("id")) == str(order_id) for o in resting)
        except Exception as e:
            log.warning("limit sweep verification failed: %s", e)
            return False

    def _bracket_qty(self, symbol: str, qty: float) -> float:
        """Round an ACTUAL position qty to the amount step without the float
        floor trap (floor(0.043/0.001) == 42), which would leave one step of
        the position with no stop. Sizing keeps using _round_qty unchanged."""
        return self._round_qty(symbol, qty + 1e-9)

    def close_position(
        self, symbol: str,
        client_order_id_root: str | None = None, close_leg: str = "c",
        sweep_first: bool = True,
    ) -> dict[str, Any] | None:
        """Flatten via reduce-only market order. No-op if already flat.

        close_leg names the exit reason for tagging:
          - "x"  time-stop
          - "ce" Donchian channel-exit close
          - "bf" boot-flatten (recovering from a stale position)
          - "h"  HALT-triggered close
          - "k"  kill-switch close
          - "c"  generic (fallback)
          - "be" breakeven close (price already back through breakeven)
        (The no-stop close's "ns" id is built by _reduce_only_close, not here.)
        If client_order_id_root is None, places the close untagged.

        sweep_first=False sends the reduce-only close WITHOUT first cancelling
        the leg's resting SL/TP. The caller must then sweep only after the
        close succeeds. Used by the breakeven close: with the default order, a
        close that raises leaves the position with no stop AND no TP, and a
        later retry could restore only the stop (review of PR #32).
        """
        p = self.fetch_position(symbol)
        if p.side == "flat" or p.qty == 0:
            return None
        ccxt_side = "sell" if p.side == "long" else "buy"
        pos_side = self._position_side(p.side)
        if sweep_first:
            self.cancel_open_orders(symbol, coid_prefix=self.coid_prefix)
        params: dict[str, Any] = {"reduceOnly": True}
        if (coid := _coid(client_order_id_root, close_leg, self.coid_prefix)):
            params["newClientOrderId"] = coid
        if pos_side is not None:
            params["positionSide"] = pos_side
        return self._create_order_with_coid_retry(
            symbol, "market", ccxt_side, p.qty, None, params)

    def __repr__(self) -> str:
        return f"BinanceClient(env={self.env!r}, api_key='***')"
