"""Principal anchor for the kill switch (God's rule: protect DEPOSITED PRINCIPAL).

The kill switch trips at `equity < P * kill_fraction`, where P = NET DEPOSITED
PRINCIPAL — NOT a high-water mark and NOT a balance snapshot. P is built purely
from Binance USDM-futures income-ledger events of type TRANSFER (signed):

    P = principal_base + Σ(USDT principal-moving income)

Trading income (REALIZED_PNL, FUNDING_FEE, COMMISSION) is EXCLUDED, so bot P&L
never moves P. A spot->futures DCA (positive TRANSFER) RAISES P; a withdrawal
(negative TRANSFER) LOWERS P.

Idempotency (the reason a wrong-account over-read can never corrupt the anchor
the way a balance snapshot could): every principal-moving income event is stored
in `state.principal_ledger` keyed by its stable Binance `tranId` (PRIMARY KEY,
INSERT OR IGNORE). P is DERIVED as SUM over the ledger, so re-fetching an
overlapping income window on restart cannot double-count.

Initialisation — REQUIRED CORRECTION: NEVER snapshot current balance (that bakes
trading profit into principal, violating the rule). Two seeds only:
  - manual mode      : deploy.manual_principal_usdt in params → principal_base =
                       that value, no history backfill (watermark starts at now,
                       future transfers still accumulate on top).
  - income_backfill  : (default) principal_base = 0, full-history income backfill
                       into the ledger. P = Σ ledger.

Validated 2026-07-01 (read-only, donchian sub-account): spot->futures transfers
appear as incomeType=TRANSFER, asset USDT, each with a stable tranId; no
REALIZED_PNL/FUNDING/COMMISSION contamination in the transfer set.
"""

from __future__ import annotations

import time
from typing import Any

from . import state

# Income types that move deposited principal. Binance already SIGNS `income`
# (deposit in = +, withdrawal out = −), so a plain signed sum is net principal.
# NOTE: only TRANSFER is a valid Binance USDM-futures income-type filter value
# for principal moves — it is SIGNED (spot->futures in = +, futures->spot out =
# −), so a signed sum is net principal. DEPOSIT/WITHDRAW are NOT valid futures
# incomeType values; sending them to /fapi/v1/income returns error -1130.
PRINCIPAL_INCOME_TYPES: frozenset[str] = frozenset({"TRANSFER"})
# P is USDT-denominated. Non-USDT principal transfers (e.g. BNB moved into
# futures) are stored but NOT summed into P — they need manual USDT valuation.
PRINCIPAL_ASSET = "USDT"

META_SOURCE = "principal_source"                  # "manual" | "income_backfill"
META_BASE = "principal_base"                       # float seed (manual base, else 0)
META_ANCHOR = "principal_anchor"                   # cached float P = base + Σledger
META_WATERMARK = "principal_income_watermark_ms"   # int, newest income ts seen

# Default full-history backfill start (2024-01-01 UTC). Sub-accounts are far
# newer; this is a cheap, safe lower bound that still captures every real
# transfer. Override via deploy.principal_backfill_start_ms.
DEFAULT_BACKFILL_START_MS = 1_704_067_200_000
# Refetch overlap so a restart never gaps rows at the watermark boundary;
# tranId idempotency makes the overlap harmless.
RECONCILE_OVERLAP_MS = 60_000


def is_initialized(timeout: float = 10.0) -> bool:
    """True once initialize() has established the seed. get_principal() returns
    None until then, which keeps the kill switch fail-safe (disabled)."""
    return state.get_meta(META_SOURCE, timeout=timeout) is not None


def get_principal() -> float | None:
    """P = principal_base + Σ(USDT ledger income). None if not initialised."""
    if not is_initialized():
        return None
    return state.get_float(META_BASE, 0.0) + state.principal_ledger_sum(PRINCIPAL_ASSET)


def get_watermark_ms() -> int:
    v = state.get_meta(META_WATERMARK)
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def breached(equity: float, principal: float | None, fraction: float) -> bool:
    """Pure kill-switch predicate. Fail-safe: never trips on an unknown or
    degenerate anchor (principal None/<=0). That is deliberate — the whole point
    of Part C is to stop false-positive trips from a bad/absent anchor."""
    if principal is None or principal <= 0:
        return False
    return equity < principal * fraction


def _normalize(income_rows: list[dict[str, Any]]) -> tuple[list[tuple], int, int]:
    """Filter raw income rows to principal-moving ledger tuples.

    Returns (ledger_rows, non_usdt_count, skipped_no_id).
    ledger_rows: (tran_id, income_type, income_usd, asset, ts_ms).
    """
    ledger_rows: list[tuple] = []
    non_usdt = 0
    skipped_no_id = 0
    for r in income_rows:
        itype = str(r.get("incomeType") or "")
        if itype not in PRINCIPAL_INCOME_TYPES:
            continue
        try:
            tran_id = int(r.get("tranId") or 0)
        except (TypeError, ValueError):
            tran_id = 0
        if tran_id <= 0:
            # No stable id → we cannot dedupe safely. Skip + flag rather than
            # risk double-counting on the next overlapping fetch.
            skipped_no_id += 1
            continue
        asset = str(r.get("asset") or PRINCIPAL_ASSET)
        try:
            income = float(r.get("income") or 0.0)
            ts = int(r.get("time") or 0)
        except (TypeError, ValueError):
            skipped_no_id += 1
            continue
        if asset != PRINCIPAL_ASSET:
            non_usdt += 1
        ledger_rows.append((tran_id, itype, income, asset, ts))
    return ledger_rows, non_usdt, skipped_no_id


def reconcile_from_income(income_rows: list[dict[str, Any]], *, log: Any = None) -> dict:
    """Idempotently fold principal-moving income into the ledger, recompute P,
    persist anchor + watermark. Safe to call repeatedly with overlapping
    windows — the tranId PRIMARY KEY dedupes."""
    ledger_rows, non_usdt, skipped = _normalize(income_rows)
    applied = state.principal_ledger_upsert(ledger_rows)
    # Advance watermark monotonically to the newest income we've seen.
    wm = max(get_watermark_ms(), state.principal_ledger_max_ts())
    if wm > 0:
        state.set_meta(META_WATERMARK, str(wm))
    principal = get_principal()
    if principal is not None:
        state.set_float(META_ANCHOR, principal)
    if non_usdt and log is not None:
        log.warning(
            "principal: %d non-USDT principal-moving transfer(s) in ledger are "
            "NOT counted toward P (P is USDT-denominated). If a non-USDT asset "
            "was transferred into futures, value it in USDT manually.", non_usdt)
    if skipped and log is not None:
        log.warning(
            "principal: skipped %d principal-moving income row(s) with no usable "
            "tranId (cannot dedupe safely).", skipped)
    return {"principal": principal, "applied": applied, "watermark_ms": wm,
            "non_usdt": non_usdt, "skipped_no_id": skipped}


def _fetch_principal_income(
    client: Any, start_ms: int, *, log: Any = None,
) -> list[dict[str, Any]]:
    """Fetch principal-moving income with a SERVER-SIDE incomeType filter, one
    call per type (currently just TRANSFER — the only valid futures
    principal-income type; DEPOSIT/WITHDRAW return Binance -1130).

    Why per-type instead of one unfiltered /fapi/v1/income fetch: on a busy
    account (v1's MAIN account especially) the ledger is dominated by
    REALIZED_PNL / COMMISSION / FUNDING_FEE rows. Under fetch_income's page cap
    those trading rows can push real TRANSFER rows past the truncation boundary,
    so a deposit is SILENTLY DROPPED and P understates deposited principal.
    Filtering server-side spends the cap only on principal rows, so no transfer
    can be lost. Any per-type truncation is surfaced through `log`
    (fetch_income(caller_log=log)). Overlapping/duplicate rows across types are
    harmless — the tranId PRIMARY KEY in principal_ledger dedupes on upsert.
    """
    rows: list[dict[str, Any]] = []
    for itype in sorted(PRINCIPAL_INCOME_TYPES):
        rows.extend(client.fetch_income(start_ms, income_type=itype, caller_log=log))
    return rows


def initialize(client: Any, params: dict, *, log: Any = None) -> float | None:
    """Establish the initial principal anchor. Idempotent: no-op if already
    initialised. NEVER snapshots current balance.

    May raise on a network/API failure in the income backfill — callers should
    treat that as "not initialised yet" and retry; the kill switch stays
    fail-safe (disabled) meanwhile.
    """
    if is_initialized():
        return get_principal()
    deploy = params.get("deploy", {}) or {}
    manual = deploy.get("manual_principal_usdt")
    now_ms = int(time.time() * 1000)

    if manual is not None:
        # Manual seed: base = manual value, no history backfill. Future
        # transfers accumulate on top via reconcile.
        state.set_float(META_BASE, float(manual))
        state.set_meta(META_WATERMARK, str(now_ms))
        state.set_meta(META_SOURCE, "manual")
        principal = get_principal()
        state.set_float(META_ANCHOR, principal if principal is not None else float(manual))
        if log is not None:
            log.info("principal: manual anchor P=%.2f USDT (no history backfill)",
                     float(manual))
        return get_principal()

    # income_backfill (default): base 0, full-history income → ledger.
    state.set_float(META_BASE, 0.0)
    start_ms = int(deploy.get("principal_backfill_start_ms", DEFAULT_BACKFILL_START_MS))
    # fetch + reconcile may raise (network/keys/DB) → we must NOT have marked
    # ourselves initialised, so the loop keeps retrying at the fast (60s) cadence
    # and the kill switch stays fail-safe. reconcile_from_income persists the
    # LEDGER + watermark; while META_SOURCE is still unset get_principal() returns
    # None inside it (that only skips caching META_ANCHOR — the anchor is derived,
    # so it is recomputed below once we mark initialised).
    rows = _fetch_principal_income(client, start_ms, log=log)
    res = reconcile_from_income(rows, log=log)
    # Mark initialised ONLY after the backfill fully succeeds. Setting META_SOURCE
    # before reconcile could leave is_initialized()=True with an empty ledger
    # (P=0) if reconcile raised — which drops the retry to the 3600s reconcile
    # interval (a 1-hour blind window) instead of the 60s init retry. Order fixed.
    state.set_meta(META_SOURCE, "income_backfill")
    P = get_principal()
    if P is not None:
        state.set_float(META_ANCHOR, P)
    if log is not None:
        log.info("principal: backfilled %d new income row(s) since %d → P=%.2f USDT",
                 res["applied"], start_ms, P or 0.0)
    return P


def reconcile_recent(client: Any, *, log: Any = None) -> dict:
    """Fetch income since (watermark − overlap) and fold it in. Used on restart
    and periodically. Requires initialize() to have run. Same server-side
    incomeType filtering as the backfill (no transfer lost to the page cap)."""
    wm = get_watermark_ms()
    start = max(0, wm - RECONCILE_OVERLAP_MS) if wm > 0 else DEFAULT_BACKFILL_START_MS
    rows = _fetch_principal_income(client, start, log=log)
    return reconcile_from_income(rows, log=log)
