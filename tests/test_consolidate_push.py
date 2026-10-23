"""Tests for the outbox + consolidate push module.

The bot queues events in state.db `outbox` and pushes them to consolidate's
/bot-event/batch endpoint. These tests cover:
  - enqueue + drain happy path
  - dedup behavior: external_id = "snapback-btc:<outbox.id>" is stable
  - HTTP failure leaves rows queued, increments attempts, doesn't lose data
  - Missing config (CONSOLIDATE_API_URL/TOKEN unset) → drain is a no-op
  - Dead-letter policy: persistent 4xx rows are moved to dead_letter after K
    attempts so they cannot block subsequent rows (head-of-line blocking fix)
  - 5xx / network errors are NOT dead-lettered (transient, retry indefinitely)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError, URLError


def _with_temp_db_and_env(test_fn):
    """Patch DB_PATH + clear consolidate env vars so tests are isolated."""
    def wrapped():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "state.db"
            with patch("exchange.state.DB_PATH", tmp), \
                 patch.dict("os.environ",
                            {"CONSOLIDATE_API_URL": "",
                             "CONSOLIDATE_API_TOKEN": ""},
                            clear=False):
                test_fn(tmp)
    return wrapped


@_with_temp_db_and_env
def test_outbox_enqueue_inserts_row(tmp: Path) -> None:
    from exchange import state
    state.init_db()
    row_id = state.enqueue_bot_event(
        "heartbeat", equity_usd=101.0, payload={"halt_present": False})
    assert row_id > 0
    pending = state.outbox_pending(10)
    assert len(pending) == 1
    assert pending[0][0] == row_id
    assert pending[0][1] == "heartbeat"
    body = json.loads(pending[0][2])
    assert body["equity_usd"] == 101.0
    assert body["payload"]["halt_present"] is False
    assert "bot_ts_ms" in body and body["bot_ts_ms"] > 0


@_with_temp_db_and_env
def test_outbox_delete_removes_rows(tmp: Path) -> None:
    from exchange import state
    state.init_db()
    a = state.enqueue_bot_event("heartbeat")
    b = state.enqueue_bot_event("heartbeat")
    assert state.outbox_size() == 2
    state.outbox_delete([a, b])
    assert state.outbox_size() == 0


@_with_temp_db_and_env
def test_outbox_mark_failed_increments_attempts(tmp: Path) -> None:
    from exchange import state
    state.init_db()
    row_id = state.enqueue_bot_event("heartbeat")
    state.outbox_mark_failed([row_id], "net: timeout")
    with sqlite3.connect(tmp) as c:
        attempts, err = c.execute(
            "SELECT attempts, last_error FROM outbox WHERE id=?", (row_id,)
        ).fetchone()
    assert attempts == 1
    assert err == "net: timeout"
    state.outbox_mark_failed([row_id], "net: again")
    with sqlite3.connect(tmp) as c:
        attempts, err = c.execute(
            "SELECT attempts, last_error FROM outbox WHERE id=?", (row_id,)
        ).fetchone()
    assert attempts == 2
    assert err == "net: again"


@_with_temp_db_and_env
def test_drain_no_op_when_unconfigured(tmp: Path) -> None:
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    state.enqueue_bot_event("heartbeat")
    result = consolidate_push.drain()
    assert result == {"skipped": "not_configured"}
    # Event stays queued.
    assert state.outbox_size() == 1


@_with_temp_db_and_env
def test_drain_no_op_when_outbox_empty(tmp: Path) -> None:
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok"}):
        result = consolidate_push.drain()
    assert result == {"sent": 0, "pending": 0}


@_with_temp_db_and_env
def test_drain_pushes_and_deletes_on_success(tmp: Path) -> None:
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    state.enqueue_bot_event(
        "entry", signal_id="1716120000000", strategy="multifactor-v1",
        side="long", qty=0.001, price_usd=65000, equity_usd=101)
    state.enqueue_bot_event("heartbeat", equity_usd=101)

    # Fake HTTP response: {ok: true, inserted: 2, skipped: 0}
    fake_response = MagicMock()
    fake_response.read.return_value = b'{"ok":true,"inserted":2,"skipped":0,"errors":[]}'
    fake_response.__enter__ = lambda self: self
    fake_response.__exit__ = lambda self, *_a: None

    captured_request = {}
    def fake_urlopen(req, timeout=None):
        captured_request["url"] = req.full_url
        captured_request["headers"] = dict(req.headers)
        captured_request["body"] = json.loads(req.data.decode("utf-8"))
        return fake_response

    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok123"}), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = consolidate_push.drain()

    assert result["sent"] == 2
    assert result["inserted"] == 2
    assert state.outbox_size() == 0  # both deleted
    # Request structure
    assert captured_request["url"] == "https://example.test/bot-event/batch"
    assert captured_request["headers"]["Authorization"] == "Bearer tok123"
    body = captured_request["body"]
    assert len(body["events"]) == 2
    e0 = body["events"][0]
    assert e0["source"] == "snapback-btc"
    assert e0["external_id"].startswith("snapback-btc:")
    assert e0["kind"] == "entry"
    assert e0["signal_id"] == "1716120000000"
    assert e0["side"] == "long"
    assert e0["price_usd"] == 65000


@_with_temp_db_and_env
def test_drain_keeps_rows_on_network_error(tmp: Path) -> None:
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    state.enqueue_bot_event("heartbeat", equity_usd=101)

    def fake_urlopen(_req, timeout=None):
        raise URLError("connection refused")

    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok"}), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = consolidate_push.drain()

    assert "error" in result
    # Row stays queued.
    assert state.outbox_size() == 1
    # attempts incremented.
    with sqlite3.connect(tmp) as c:
        attempts, last_error = c.execute(
            "SELECT attempts, last_error FROM outbox LIMIT 1"
        ).fetchone()
    assert attempts == 1
    assert "net:" in last_error


@_with_temp_db_and_env
def test_drain_keeps_rows_on_4xx_below_dead_letter_limit(tmp: Path) -> None:
    """A 4xx where attempts < DEAD_LETTER_AFTER_ATTEMPTS keeps the row queued.

    This verifies a row that hasn't yet exhausted its retry budget stays in the
    outbox (old behavior preserved for the non-poison-pill case).
    """
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    state.enqueue_bot_event("heartbeat", equity_usd=101)

    def fake_urlopen(_req, timeout=None):
        raise HTTPError(
            url="https://example.test/bot-event/batch", code=401,
            msg="Unauthorized", hdrs={}, fp=BytesIO(b'{"error":"unauthorized"}'),
        )

    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "bad-token"}), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = consolidate_push.drain()

    assert "error" in result
    assert "HTTP 401" in result["error"]
    # Row stays queued — only 1 attempt, well below the dead-letter limit.
    assert state.outbox_size() == 1
    assert result.get("dead_lettered", 0) == 0


@_with_temp_db_and_env
def test_drain_dead_letters_row_after_k_4xx_attempts(tmp: Path) -> None:
    """A row that has been 4xx-rejected K times must be moved to dead_letter
    so it cannot block subsequent heartbeats head-of-line.

    This is the core regression test for the 2026-06-30 incident where
    kind='daily_loss_breaker' caused 2584+ heartbeats to pile up.
    """
    from exchange import state
    from tools import consolidate_push
    state.init_db()

    # Use a small K so the test is fast without patching the module-level constant.
    dead_letter_k = 3

    poison_id = state.enqueue_bot_event("daily_loss_breaker", equity_usd=100.0)
    good_id = state.enqueue_bot_event("heartbeat", equity_usd=101.0)
    assert state.outbox_size() == 2

    # Pre-load the poison row with (K-1) failed attempts so the next drain tips it over.
    for _ in range(dead_letter_k - 1):
        state.outbox_mark_failed([poison_id], "HTTP 400: ...")

    def fake_urlopen(_req, timeout=None):
        raise HTTPError(
            url="https://example.test/bot-event/batch", code=400,
            msg="Bad Request", hdrs={}, fp=BytesIO(b'{"error":"invalid_batch"}'),
        )

    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok"}), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("tools.consolidate_push.DEAD_LETTER_AFTER_ATTEMPTS", dead_letter_k):
        result = consolidate_push.drain()

    # Poison row dead-lettered; good row still in outbox.
    assert result.get("dead_lettered") == 1, f"expected dead_lettered=1, got {result}"
    assert state.outbox_size() == 1, "poison row should be removed from outbox"
    assert state.dead_letter_size() == 1, "poison row should be in dead_letter"

    # Verify the dead_letter entry has the right data.
    with sqlite3.connect(tmp) as c:
        dl_row = c.execute(
            "SELECT outbox_id, kind, reason FROM dead_letter"
        ).fetchone()
    assert dl_row[0] == poison_id
    assert dl_row[1] == "daily_loss_breaker"
    assert "400" in dl_row[2]

    # Good row still drainable on the next call.
    assert state.outbox_pending(10)[0][0] == good_id


@_with_temp_db_and_env
def test_drain_does_not_dead_letter_on_5xx(tmp: Path) -> None:
    """5xx (transient server error) must NOT trigger dead-lettering, even after
    many attempts. 5xx = server-side problem, not a bad row.
    """
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    row_id = state.enqueue_bot_event("heartbeat", equity_usd=101.0)

    # Pre-load many failures (beyond the dead-letter threshold).
    for _ in range(10):
        state.outbox_mark_failed([row_id], "HTTP 503: ...")

    def fake_urlopen(_req, timeout=None):
        raise HTTPError(
            url="https://example.test/bot-event/batch", code=503,
            msg="Service Unavailable", hdrs={}, fp=BytesIO(b'{}'),
        )

    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok"}), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = consolidate_push.drain()

    # Row stays in outbox — 5xx never dead-letters.
    assert state.outbox_size() == 1
    assert state.dead_letter_size() == 0
    assert result.get("dead_lettered", 0) == 0


def _poison_aware_urlopen(poison_external_id: str):
    """Return a urlopen stub that 400s any batch CONTAINING poison_external_id
    and 200s (delivers) any batch that does not. Models a real schema-poison
    row: a batch WITHOUT it succeeds, so bisection can isolate + deliver the
    innocent rows. Content-based (not call-count-based) so it is robust to the
    variable number of HTTP calls bisection makes."""
    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        ext_ids = {e.get("external_id") for e in body.get("events", [])}
        if poison_external_id in ext_ids:
            raise HTTPError(
                url="https://example.test/bot-event/batch", code=400,
                msg="Bad Request", hdrs={}, fp=BytesIO(b'{"error":"invalid_batch"}'),
            )
        n = len(body.get("events", []))
        fake = MagicMock()
        fake.status = 200
        fake.read.return_value = json.dumps(
            {"ok": True, "inserted": n, "skipped": 0, "errors": []}
        ).encode("utf-8")
        fake.__enter__ = lambda self: self
        fake.__exit__ = lambda self, *_a: None
        return fake
    return fake_urlopen


@_with_temp_db_and_env
def test_dead_letter_good_rows_still_drain_after_poison(tmp: Path) -> None:
    """A poison row bundled with an innocent heartbeat: the innocent row is
    isolated and DELIVERED in the same drain, while the poison row is
    dead-lettered on its own retry budget. Head-of-line unblocking.
    """
    from exchange import state
    from tools import consolidate_push
    state.init_db()

    dead_letter_k = 2
    poison_id = state.enqueue_bot_event("daily_loss_breaker", equity_usd=100.0)
    # Pre-load poison row to the threshold so this drain tips it over.
    for _ in range(dead_letter_k - 1):
        state.outbox_mark_failed([poison_id], "HTTP 400: ...")
    good_id = state.enqueue_bot_event("heartbeat", equity_usd=101.0)

    poison_ext = f"snapback-btc:{poison_id}"
    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok"}), \
         patch("urllib.request.urlopen", side_effect=_poison_aware_urlopen(poison_ext)), \
         patch("tools.consolidate_push.DEAD_LETTER_AFTER_ATTEMPTS", dead_letter_k):
        r1 = consolidate_push.drain()

    # Poison isolated + dead-lettered; innocent heartbeat delivered in the SAME
    # drain (never dead-lettered alongside the poison row).
    assert r1.get("dead_lettered") == 1
    assert r1.get("inserted") == 1, "innocent row delivered in the same drain"
    assert state.outbox_size() == 0
    assert state.dead_letter_size() == 1
    with sqlite3.connect(tmp) as c:
        dl_outbox_id = c.execute("SELECT outbox_id FROM dead_letter").fetchone()[0]
    assert dl_outbox_id == poison_id
    _ = good_id  # delivered, no longer in outbox


@_with_temp_db_and_env
def test_drain_isolates_poison_across_repeated_drains(tmp: Path) -> None:
    """MULTI-CALL drain(): one poison row bundled with many innocent heartbeats
    across repeated failures. The innocent heartbeats must ALWAYS survive
    (delivered, never dead-lettered), while ONLY the poison row eventually
    dead-letters after it alone exhausts its retry budget.

    This is the regression guard for the 2026-06-30 incident where a single
    kind='daily_loss_breaker' 4xx dead-lettered ~49 innocent heartbeats.
    """
    from exchange import state
    from tools import consolidate_push
    state.init_db()

    dead_letter_k = 3
    poison_id = state.enqueue_bot_event("daily_loss_breaker", equity_usd=100.0)
    # A first batch of innocent heartbeats bundled with the poison row.
    innocents_1 = [state.enqueue_bot_event("heartbeat", equity_usd=100.0 + i)
                   for i in range(5)]
    poison_ext = f"snapback-btc:{poison_id}"

    env = {"CONSOLIDATE_API_URL": "https://example.test", "CONSOLIDATE_API_TOKEN": "tok"}

    def _drain_once():
        with patch.dict("os.environ", env), \
             patch("urllib.request.urlopen", side_effect=_poison_aware_urlopen(poison_ext)), \
             patch("tools.consolidate_push.DEAD_LETTER_AFTER_ATTEMPTS", dead_letter_k):
            return consolidate_push.drain()

    # Drain 1: innocents delivered, poison isolated (attempts 1, not yet dead).
    r1 = _drain_once()
    assert r1.get("inserted") == len(innocents_1), "all innocents delivered on drain 1"
    assert r1.get("dead_lettered", 0) == 0, "poison not dead-lettered yet (attempt 1)"
    assert state.dead_letter_size() == 0
    # Only the poison row remains queued.
    remaining = [r[0] for r in state.outbox_pending(50)]
    assert remaining == [poison_id]

    # Between drains, MORE innocent heartbeats arrive and get bundled with the
    # still-stuck poison row. They must keep surviving across repeated failures.
    innocents_2 = [state.enqueue_bot_event("heartbeat", equity_usd=200.0 + i)
                   for i in range(4)]

    # Drain 2: new innocents delivered, poison attempt 2 (still not dead).
    r2 = _drain_once()
    assert r2.get("inserted") == len(innocents_2), "new innocents delivered on drain 2"
    assert r2.get("dead_lettered", 0) == 0
    assert state.dead_letter_size() == 0
    assert [r[0] for r in state.outbox_pending(50)] == [poison_id]

    # Drain 3: poison hits attempt K → dead-lettered. No innocents left to lose.
    r3 = _drain_once()
    assert r3.get("dead_lettered") == 1
    assert state.outbox_size() == 0
    assert state.dead_letter_size() == 1
    with sqlite3.connect(tmp) as c:
        dl = c.execute("SELECT outbox_id, kind FROM dead_letter").fetchone()
    assert dl[0] == poison_id and dl[1] == "daily_loss_breaker"


@_with_temp_db_and_env
def test_drain_handles_partial_failure_207(tmp: Path) -> None:
    """Server returned 207 (some events failed, others succeeded). Bot must
    delete the successful local outbox rows AND keep the failed ones queued."""
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    state.enqueue_bot_event("heartbeat", equity_usd=101)
    id2 = state.enqueue_bot_event("entry", signal_id="bad", side="long",
                                   qty=0.001, price_usd=65000)
    state.enqueue_bot_event("heartbeat", equity_usd=101)
    assert state.outbox_size() == 3

    fake_response = MagicMock()
    fake_response.status = 207
    # id2 (the entry) failed; id1 and id3 succeeded.
    fake_response.read.return_value = json.dumps({
        "ok": False,
        "inserted": 2,
        "skipped": 0,
        "errors": [
            {"external_id": f"snapback-btc:{id2}", "message": "constraint violation"},
        ],
    }).encode("utf-8")
    fake_response.__enter__ = lambda self: self
    fake_response.__exit__ = lambda self, *_a: None

    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok"}), \
         patch("urllib.request.urlopen", return_value=fake_response):
        result = consolidate_push.drain()

    assert result["status"] == 207
    assert result["failed"] == 1
    assert result["deleted_locally"] == 2
    # Only the failed event remains; successful ones deleted.
    assert state.outbox_size() == 1
    with sqlite3.connect(tmp) as c:
        remaining = c.execute("SELECT id, attempts, last_error FROM outbox").fetchone()
    assert remaining[0] == id2
    assert remaining[1] == 1
    assert "207" in remaining[2]


@_with_temp_db_and_env
def test_external_id_is_stable_across_retries(tmp: Path) -> None:
    """If push fails and we retry, the same external_id must be sent so
    the server-side dedup kicks in. Tests this by simulating: enqueue,
    fail to push, retry, capture both requests, check external_ids match.
    """
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    state.enqueue_bot_event("heartbeat")

    captured = []

    def fake_urlopen_capture(req, timeout=None):
        captured.append(json.loads(req.data.decode("utf-8")))
        # First call fails, second succeeds — but the SAME row should be sent.
        if len(captured) == 1:
            raise URLError("transient")
        fake = MagicMock()
        fake.read.return_value = b'{"ok":true,"inserted":1,"skipped":0,"errors":[]}'
        fake.__enter__ = lambda self: self
        fake.__exit__ = lambda self, *_a: None
        return fake

    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok"}), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen_capture):
        consolidate_push.drain()
        consolidate_push.drain()

    assert len(captured) == 2
    assert captured[0]["events"][0]["external_id"] == captured[1]["events"][0]["external_id"]
