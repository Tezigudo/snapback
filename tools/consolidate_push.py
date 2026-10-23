"""Drain state.db outbox → POST /bot-event/batch to investing-consolidate.

Push-based, fire-and-forget (with retry). The bot's own state.db is the
source of truth; this module ships a downstream copy to consolidate so the
dashboard can render bot status.

Failure mode: if consolidate is unreachable (network, Fly.io down, auth
mismatch), events stay in the outbox and replay on the next drain. The
bot keeps trading regardless — this is best-effort, never blocking.

Configuration via .env:
  CONSOLIDATE_API_URL    e.g. https://<your-consolidate-host>
  CONSOLIDATE_API_TOKEN  the same Bearer token the web app uses

If either is unset, drain() is a no-op (returns {"skipped": "not_configured"}).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from exchange import state

log = logging.getLogger(__name__)

# Source identifier in the consolidate /bot-event payload. Each running bot
# instance needs a DISTINCT source so the dashboard can show them as separate
# bot cards. Default is "snapback-btc" (the original v1 leg, unchanged).
# The Donchian leg sets CONSOLIDATE_SOURCE=snapback-btc-donchian in its env.
#
# Read at CALL time, NOT import time. bot.py imports this module at top level,
# but load_env_for_instance() overlays the per-instance .env INSIDE main() —
# AFTER this import. A module-level constant would freeze to the default, so
# every leg (donchian, cnh_short, …) would push under "snapback-btc" and
# collide on one dashboard card. Reading os.environ live fixes that.
def _source() -> str:
    return (os.environ.get("CONSOLIDATE_SOURCE") or "snapback-btc").strip()


DEFAULT_TIMEOUT_S = 3.0
DEFAULT_BATCH_LIMIT = 50

# After this many failed attempts with a 4xx response, a row is moved from the
# outbox to the dead_letter table so it can no longer block subsequent rows.
# Transient 5xx / network errors do NOT count toward this limit — those stay
# in the outbox and retry indefinitely. Configurable via env var for ops tuning.
DEAD_LETTER_AFTER_ATTEMPTS: int = int(
    os.environ.get("CONSOLIDATE_DEAD_LETTER_ATTEMPTS", "5")
)


def _config() -> tuple[str | None, str | None]:
    url = (os.environ.get("CONSOLIDATE_API_URL") or "").strip().rstrip("/")
    token = (os.environ.get("CONSOLIDATE_API_TOKEN") or "").strip()
    return (url or None), (token or None)


def is_configured() -> bool:
    url, token = _config()
    return url is not None and token is not None


def _build_event(row_id: int, kind: str, payload_str: str, source: str) -> dict[str, Any]:
    """Translate an outbox row into the /bot-event JSON shape."""
    body = json.loads(payload_str)
    return {
        "source": source,
        "external_id": f"{source}:{row_id}",
        "bot_ts_ms": int(body.get("bot_ts_ms", 0)),
        "kind": kind,
        "signal_id": body.get("signal_id"),
        "strategy": body.get("strategy"),
        "side": body.get("side"),
        "qty": body.get("qty"),
        "price_usd": body.get("price_usd"),
        "notional_usd": body.get("notional_usd"),
        "equity_usd": body.get("equity_usd"),
        "payload": body.get("payload") or {},
    }


def _post_batch(url: str, token: str, events: list[dict[str, Any]],
                timeout_s: float) -> dict[str, Any]:
    """POST one batch and CLASSIFY the outcome. Never mutates the outbox.

    Returns one of:
      {"class": "ok",        "status": int, "resp": dict}   # 2xx / 207
      {"class": "poison",    "status": int, "err": str}     # 400-class: a bad row
      {"class": "auth",      "status": int, "err": str}     # 401/403: global, not a row
      {"class": "transient",                "err": str}     # 5xx / 429 / net / unexpected
    """
    body = json.dumps({"events": events}).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/bot-event/batch",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "snapback-btc/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return {"class": "ok", "status": int(resp.status),
                    "resp": json.loads(resp.read().decode("utf-8"))}
    except urllib.error.HTTPError as e:
        err_msg = f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"
        if e.code in (401, 403):
            # Auth failure is a GLOBAL condition, not a poison row. Never
            # dead-letter innocent rows over an auth misconfig — keep queued.
            return {"class": "auth", "status": e.code, "err": err_msg}
        if e.code == 429:
            return {"class": "transient", "err": err_msg}  # rate limit
        if 400 <= e.code < 500:
            # Schema/validation rejection → at least one row in THIS batch is
            # the offender. Caller isolates it (bisect) instead of blaming all.
            return {"class": "poison", "status": e.code, "err": err_msg}
        return {"class": "transient", "err": err_msg}  # 5xx
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"class": "transient", "err": f"net: {e}"}
    except Exception as e:  # never let a telemetry push crash the caller
        log.exception("consolidate push exception")
        return {"class": "transient", "err": f"unexpected: {e}"}


def _handle_ok(source: str, items: list[tuple], outcome: dict[str, Any]) -> dict[str, Any]:
    """2xx/207 path: delete successes, keep per-event (207) failures queued."""
    resp_status = int(outcome["status"])
    resp_data = outcome["resp"]
    ids = [it[0] for it in items]
    error_external_ids: set[str] = set()
    for err in (resp_data.get("errors") or []):
        if isinstance(err, dict) and err.get("external_id"):
            error_external_ids.add(str(err["external_id"]))
    success_ids: list[int] = []
    failed_ids: list[int] = []
    for row_id in ids:
        if f"{source}:{row_id}" in error_external_ids:
            failed_ids.append(row_id)
        else:
            success_ids.append(row_id)
    deleted = state.outbox_delete(success_ids) if success_ids else 0
    if failed_ids:
        state.outbox_mark_failed(
            failed_ids, f"server status {resp_status}: per-event failure")
        log.warning("consolidate push: %d/%d events failed server-side",
                    len(failed_ids), len(ids))
    return {
        "sent": len(items),
        "inserted": int(resp_data.get("inserted") or 0),
        "skipped": int(resp_data.get("skipped") or 0),
        "failed": len(failed_ids),
        "deleted_locally": deleted,
        "status": resp_status,
    }


def _dead_letter_single(item: tuple, err: str) -> dict[str, Any]:
    """A size-1 batch got a 4xx → THIS row is the poison. Increment its attempts
    and dead-letter only if it has now exhausted its own retry budget."""
    row_id = item[0]
    state.outbox_mark_failed([row_id], err)
    dead = state.outbox_dead_letter_over_limit([row_id], err, DEAD_LETTER_AFTER_ATTEMPTS)
    for did, dk, payload in dead:
        log.error(
            "OUTBOX DEAD LETTER: row %s (kind=%s reached %d attempts) moved to "
            "dead_letter table — payload: %.500s",
            did, dk, DEAD_LETTER_AFTER_ATTEMPTS, payload,
        )
    return {"error": err, "isolated_poison": 1, "dead_lettered": len(dead),
            "queued": 1 - len(dead)}


_MERGE_KEYS = ("sent", "inserted", "skipped", "failed", "deleted_locally",
               "dead_lettered", "isolated_poison", "queued")


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in _MERGE_KEYS:
        if k in a or k in b:
            out[k] = int(a.get(k) or 0) + int(b.get(k) or 0)
    errs = [x["error"] for x in (a, b) if x.get("error")]
    if errs:
        out["error"] = "; ".join(errs)
    status = a.get("status") or b.get("status")
    if status is not None:
        out["status"] = status
    return out


def _drain_items(url: str, token: str, source: str, items: list[tuple],
                 timeout_s: float) -> dict[str, Any]:
    """Send `items` (list of (row_id, kind, event, attempts)) as one batch.

    On a 4xx-poison the batch is BISECTED so the offending row is isolated to a
    size-1 batch and dead-lettered on its own budget — innocent rows in the same
    original batch re-send in their own sub-batch and (if the backend accepts
    them) are delivered immediately, so they never accrue attempts or get
    dead-lettered. This replaces the old behaviour where one poison row failed
    the whole batch and dead-lettered the ~49 innocent heartbeats with it.
    """
    if not items:
        return {"sent": 0, "pending": state.outbox_size()}
    events = [it[2] for it in items]
    ids = [it[0] for it in items]
    outcome = _post_batch(url, token, events, timeout_s)
    cls = outcome["class"]
    if cls == "ok":
        return _handle_ok(source, items, outcome)
    if cls == "auth":
        log.warning("consolidate push auth %s (kept queued, NOT dead-lettered): %s",
                    outcome["status"], outcome["err"])
        state.outbox_mark_failed(ids, outcome["err"])
        return {"error": outcome["err"], "queued": len(ids), "auth": True}
    if cls == "transient":
        log.info("consolidate push transient (kept queued) — %d event(s): %s",
                 len(ids), outcome["err"])
        state.outbox_mark_failed(ids, outcome["err"])
        return {"error": outcome["err"], "queued": len(ids)}
    # cls == "poison": a bad row lives in this batch.
    if len(items) == 1:
        return _dead_letter_single(items[0], outcome["err"])
    log.warning("consolidate push 4xx on %d-row batch — bisecting to isolate the "
                "offending row: %s", len(items), outcome["err"])
    mid = len(items) // 2
    left = _drain_items(url, token, source, items[:mid], timeout_s)
    right = _drain_items(url, token, source, items[mid:], timeout_s)
    return _merge(left, right)


def drain(limit: int = DEFAULT_BATCH_LIMIT, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Send up to `limit` queued events to consolidate. Returns status dict.

    On success: outbox rows for sent events are DELETED.
    On a batch 4xx: the poison row is ISOLATED (bisect) and dead-lettered on its
    own budget; innocent rows are unaffected.
    On transient failure (5xx/429/net) or auth (401/403): rows stay queued.
    """
    url, token = _config()
    if not url or not token:
        return {"skipped": "not_configured"}

    rows = state.outbox_pending(limit)
    if not rows:
        return {"sent": 0, "pending": 0}

    source = _source()
    items: list[tuple] = []
    for row_id, kind, payload_str, attempts in rows:
        try:
            ev = _build_event(row_id, kind, payload_str, source)
            items.append((row_id, kind, ev, attempts))
        except Exception as e:
            # Malformed payload — should never happen, but tolerate. Mark this
            # one as failed and skip; the others still go.
            log.warning("outbox row %s payload parse failed: %s", row_id, e)
            state.outbox_mark_failed([row_id], f"parse: {e}")

    if not items:
        return {"sent": 0, "pending": state.outbox_size()}

    return _drain_items(url, token, source, items, timeout_s)


if __name__ == "__main__":
    # Manual one-shot drain for ops/debugging.
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(drain(), indent=2))
