"""Shared harness seeding for tests that build Bot via object.__new__ (so
Bot.__init__ never runs). #49's no-stop state and loop hooks live here."""
from __future__ import annotations

NO_STOP_HOOKS = ("_maybe_finish_no_stop_close", "_maybe_guard_unverified_entry")


def seed_no_stop_state(b, bot_cls) -> None:
    b._no_stop_pending = None
    b._no_stop_backoff_s = bot_cls.NO_STOP_RETRY_MIN_S
    b._no_stop_next_try = 0.0
    b._latch_until = 0.0
    b._latch_streak = 0
    b._latch_persisted = False
    b._latch_unreadable_alert_ts = 0.0
    b._latch_released_for = 0.0
    b._pending_meta_stale = False
    b._book_unknown_n = 0
    b._book_unknown_since = None
    b._unverified = None
    b._unverified_meta_stale = False
    b._unproven_alerted_root = None
    b._sl_fail_mem = {}
    b._no_stop_alert_ts = 0.0
    b._unverified_next = 0.0
    b._entries_paused_seen = False
    b._pause_alert_retry_at = 0.0
    b._pause_alert_tries = 0
    b._position_held_at = None
