"""God's rule, 2026-10-13: "If a stop-loss can't be placed (or re-placed),
retry 2–3×; if it still can't, close the position at market and alert — all 3
legs."

This is the worst-case-loss path, so every test here was written to FAIL on
the base commit (e6244e6) first. Each class names the finding it pins.

Finding 1: the entry filled, then the SL placement raised. Base: the exception
  escaped to _maybe_enter, which alerted "entry failed" while a position was
  OPEN with no stop, never recorded the fill, and never retried or closed.
  donchian and SOL have no reprotect block, so they stayed naked until the
  channel/flip exit.
Finding 2: _maybe_reprotect never escalated. A stop that kept failing to
  re-place (-2021) alerted every 15 s forever; and (2b) a missing TP made it
  cancel a HEALTHY stop first, opening a window with no stop at all.
Finding 3: reprotect matched legs by prefix only, so an OLDER trade's leftover
  stop read as "our SL present" and the new position got no stop.
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

from bot_internals import SignalDecision
from exchange import state
from exchange.binance_client import BinanceClient

from ._fake_exchange import SYMBOL, WOULD_TRIGGER, FakeEx, rejected, timeout

PREFIX = "snap-v1-"
ROOT = "1791000000000"
OLD = "1790000000000"


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch("exchange.binance_client.time.sleep"), patch("bot.time.sleep"):
        yield


def _client(ex: FakeEx, prefix: str = PREFIX) -> BinanceClient:
    return BinanceClient(ex=ex, env="testnet", coid_prefix=prefix)


def _market(c: BinanceClient, place_tp: bool = True):
    return c.market_order_with_bracket(SYMBOL, "long", 0.005, 64000.0, 67000.0,
                                       client_order_id_root=ROOT, place_tp=place_tp)


def _limit(c: BinanceClient):
    return c.limit_order_with_bracket(SYMBOL, "long", 0.005, 65000.0, 1000.0, 2000.0,
                                      timeout_s=4.0, poll_s=2.0,
                                      client_order_id_root=ROOT)


def _call_quietly(fn):
    """Run `fn`; swallow any exception. The assertions are about what the
    EXCHANGE saw, which must hold whether or not the client also raises."""
    try:
        return fn()
    except Exception as e:
        return e


# ---------------------------------------------------------------------------
# Finding 1 — client: SL placement after a fill
# ---------------------------------------------------------------------------

class TestEntrySlFailureClient:

    def test_sl_failing_every_attempt_closes_the_position(self) -> None:
        ex = FakeEx()
        ex.fail["STOP_MARKET"] = [rejected(), rejected(), rejected()]
        _call_quietly(lambda: _market(_client(ex)))
        assert len(ex.creates("STOP_MARKET")) == 3, "retry 2–3× before giving up"
        closes = ex.closes()
        assert len(closes) == 1, "no stop after retries ⇒ market close"
        params = closes[0][6]
        assert params["reduceOnly"] is True
        assert params["newClientOrderId"] == f"{PREFIX}{ROOT}-ns"
        assert ex.pos is None

    def test_would_immediately_trigger_closes_without_retrying(self) -> None:
        """-2021 on the stop means price is already THROUGH it. Retrying the
        same stop can only fail again while the loss grows."""
        ex = FakeEx()
        ex.fail["STOP_MARKET"] = [WOULD_TRIGGER, None, None]
        _call_quietly(lambda: _market(_client(ex)))
        assert len(ex.creates("STOP_MARKET")) == 1
        assert len(ex.closes()) == 1 and ex.pos is None

    def test_one_transient_failure_then_success_keeps_the_position(self) -> None:
        ex = FakeEx()
        ex.fail["STOP_MARKET"] = [rejected(-1001, "Internal error"), None]
        out = _market(_client(ex))
        assert len(ex.creates("STOP_MARKET")) == 2
        assert ex.closes() == []
        assert ex.pos is not None and out["sl"] is not None
        assert [r["clientAlgoId"] for r in ex.algo] == [f"{PREFIX}{ROOT}-s", f"{PREFIX}{ROOT}-t"]

    def test_timeout_that_landed_is_not_placed_twice(self) -> None:
        """A RequestTimeout on our side can still be ACCEPTED by Binance. A blind
        retry with the same id gets 'duplicate', and the coid-retry wrapper
        would then rest a SECOND, UNTAGGED stop that no prefix sweep can ever
        find — the orphan-bracket bug class. The retry must look first."""
        ex = FakeEx()
        ex.timeout_lands = True
        ex.fail["STOP_MARKET"] = [timeout()]
        out = _market(_client(ex))
        assert len(ex.creates("STOP_MARKET")) == 1
        assert ex.closes() == []
        stops = [r for r in ex.algo if r["clientAlgoId"].endswith("-s")]
        assert len(stops) == 1 and all(r["clientAlgoId"] for r in ex.algo)
        assert out["sl"] is not None

    def test_tp_failure_keeps_a_stopped_position(self) -> None:
        """The stop protects the position; a missing TP is not a reason to close
        it. Return normally, flag the TP, let reprotect / the exit rules cope."""
        ex = FakeEx()
        ex.fail["TAKE_PROFIT_MARKET"] = [rejected()]
        out = _market(_client(ex))
        assert ex.closes() == []
        assert ex.pos is not None
        assert out["tp"] is None and out.get("tp_error")
        assert [r["clientAlgoId"] for r in ex.algo] == [f"{PREFIX}{ROOT}-s"]

    @pytest.mark.parametrize("polls,filled_as", [
        ([{"status": "closed", "filled": 0.005, "average": 64990.0, "_side": "long"}], "limit"),
        ([{"status": "open", "filled": 0.002, "average": 64990.0, "_side": "long"}], "limit_partial"),
        ([{"status": "open", "filled": 0.0, "average": None, "_side": "long"}], "market_fallback"),
    ])
    def test_limit_paths_close_when_no_stop(self, polls, filled_as) -> None:
        ex = FakeEx()
        ex.limit_polls = polls
        ex.fail["STOP_MARKET"] = [WOULD_TRIGGER]
        _call_quietly(lambda: _limit(_client(ex)))
        assert len(ex.closes()) == 1, filled_as
        assert ex.pos is None


# ---------------------------------------------------------------------------
# Finding 1 — bot: the fill and the emergency close are RECORDED
# ---------------------------------------------------------------------------

V1_PARAMS: dict = {
    "symbol": SYMBOL,
    "strategy_name": "multifactor-v1",
    "hedge": {"enabled": False, "client_order_id_prefix": PREFIX},
    "execution": {"poll_interval_s": "5", "order_type": "market",
                  "limit_offset_bps": "0", "limit_timeout_s": "4"},
    "timeframes": {"entry": "15m"},
    "sizing": {"leverage": 5, "risk_per_trade_pct": 1.0},
    "deploy": {"kill_switch_equity_fraction": 0.5, "min_capital_warn_usdt": 1.0},
    # Synthetic; only the keys gate_status reads (the evaluator is mocked).
    "strategy": {"rsi_long_threshold": 31, "rsi_short_threshold": 69,
                 "volume_multiple": 1.25, "funding_extreme_threshold": 0.0011,
                 "require_funding_not_extreme": True},
    "reprotect": {"enabled": True, "observe_only": False,
                  "max_replaces_per_position": 3},
}
# donchian: no TP leg, no reprotect block — the leg that stayed naked longest.
DONCH_PARAMS: dict = {**{k: v for k, v in V1_PARAMS.items() if k != "reprotect"},
                      "strategy_name": "donchian-v3",
                      "hedge": {"enabled": False, "client_order_id_prefix": "snap-d3-"},
                      "timeframes": {"entry": "4h"},
                      "strategy": {"slope_trend_threshold_pct": 0.07}}

DECISION = SignalDecision(side="long", price=65000.0, sl_distance=1000.0,
                          tp_distance=2000.0, debug={})


def _bot(ex: FakeEx, params: dict = V1_PARAMS):
    from bot import Bot
    prefix = params["hedge"]["client_order_id_prefix"]
    with patch("bot.BinanceClient.from_env", return_value=_client(ex, prefix)):
        return Bot(params=params, dry_run=False)


def _fills() -> list[tuple]:
    with sqlite3.connect(state.DB_PATH) as c:
        return c.execute("SELECT reason, side, qty, price, client_order_id_root "
                         "FROM fills ORDER BY id").fetchall()


def _subjects(alert) -> list[str]:
    return [str(c.args[0]) for c in alert.call_args_list]


class TestEntrySlFailureBot:

    @pytest.mark.parametrize("params", [V1_PARAMS, DONCH_PARAMS], ids=["v1", "donchian"])
    def test_no_stop_entry_is_closed_recorded_and_alerted(self, params) -> None:
        ex = FakeEx()
        ex.fail["STOP_MARKET"] = [WOULD_TRIGGER]
        bot = _bot(ex, params)
        with patch("bot.send_alert") as alert:
            bot._place_live_entry(DECISION, 0.005, ROOT, 150.0)
        assert ex.pos is None
        fills = _fills()
        assert [f[0] for f in fills] == ["entry", "no_stop_close"]
        assert fills[0][1] == "long" and fills[0][4] == ROOT
        assert fills[1][1] == "close" and fills[1][3] == pytest.approx(64900.0)
        assert state.get_meta("active_bracket") == ""
        assert any("NO STOP" in s for s in _subjects(alert))
        assert not any("entry failed" in s.lower() for s in _subjects(alert))

    def test_a_lagging_position_read_does_not_end_the_emergency(self) -> None:
        """positionRisk can lag a fill. A close gated on a position read would
        see "flat", return None, and declare the emergency over with the
        position still open. The known fill is closed directly instead."""
        ex = FakeEx()
        ex.fail["STOP_MARKET"] = [WOULD_TRIGGER]
        ex.position_lag_reads = 5
        bot = _bot(ex, DONCH_PARAMS)
        with patch("bot.send_alert"):
            bot._place_live_entry(DECISION, 0.005, ROOT, 150.0)
        assert ex.pos is None
        assert len(ex.closes()) == 1
        assert [f[0] for f in _fills()] == ["entry", "no_stop_close"]

    def test_close_failure_leaves_a_pending_close_that_the_next_tick_finishes(self) -> None:
        ex = FakeEx()
        ex.fail["STOP_MARKET"] = [WOULD_TRIGGER]
        ex.fail["close"] = [rejected(-1001), rejected(-1001), rejected(-1001)]
        bot = _bot(ex, DONCH_PARAMS)
        with patch("bot.send_alert") as alert:
            bot._place_live_entry(DECISION, 0.005, ROOT, 150.0)
        assert ex.pos is not None, "every close attempt failed"
        assert [f[0] for f in _fills()] == ["entry"]
        assert any("FAILED" in s and "NO STOP" in s for s in _subjects(alert))
        # next tick: the exchange recovered (and the retry backoff has elapsed)
        bot._no_stop_next_try = 0.0
        with patch("bot.send_alert") as alert2:
            bot._maybe_finish_no_stop_close(150.0)
        assert ex.pos is None
        assert [f[0] for f in _fills()] == ["entry", "no_stop_close"]
        assert any("NO STOP" in s for s in _subjects(alert2))
        # and it is done — a further tick does nothing
        n = len(ex.calls)
        bot._maybe_finish_no_stop_close(150.0)
        assert len(ex.calls) == n

    def test_entry_error_after_a_fill_gets_a_stop(self) -> None:
        """A market entry can time out on our side and still FILL. Base alerted
        'entry failed' and left the position with no stop. Now the no-stop
        rule applies to it: place the stop (with retry) — it is kept."""
        ex = FakeEx()
        ex.fail["entry"] = [timeout()]
        bot = _bot(ex, DONCH_PARAMS)
        with patch("bot.send_alert") as alert:
            _enter(bot)
        assert ex.pos is not None and ex.closes() == []
        stops = [r["clientAlgoId"] for r in ex.algo]
        assert len(stops) == 1 and stops[0].startswith("snap-d3-") and stops[0].endswith("-s")
        assert [f[0] for f in _fills()] == ["entry"]
        assert not any("entry failed" in s.lower() for s in _subjects(alert))

    def test_entry_error_after_a_fill_with_no_possible_stop_is_closed(self) -> None:
        ex = FakeEx()
        ex.fail["entry"] = [timeout()]
        ex.fail["STOP_MARKET"] = [WOULD_TRIGGER]
        bot = _bot(ex, DONCH_PARAMS)
        with patch("bot.send_alert") as alert:
            _enter(bot)
        assert ex.pos is None and len(ex.closes()) == 1
        assert [f[0] for f in _fills()] == ["entry", "no_stop_close"]
        assert any("NO STOP" in s for s in _subjects(alert))


def _enter(bot, equity: float = 500.0) -> None:
    """Drive _maybe_enter to a long signal (DECISION)."""
    idx = pd.date_range("2026-01-01", periods=300, freq="4h")
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                       "Volume": 1.0}, index=idx)
    bot._last_signal_ts = None
    with patch.object(bot.client, "fetch_ohlcv", return_value=df), \
         patch.object(bot.client, "fetch_funding_rate", return_value=0.0), \
         patch("bot.evaluate_for_strategy", return_value=DECISION), \
         patch.object(bot, "_daily_loss_blocks_entry", return_value=False):
        bot._maybe_enter(equity)


# ---------------------------------------------------------------------------
# Finding 2 / 2b / 3 — reprotect
# ---------------------------------------------------------------------------

def _open(ex: FakeEx, root: str = ROOT, place_tp: bool = True, **extra) -> None:
    ex.pos = ("long", 0.005, 65000.0)
    state.record_fill(side="long", qty=0.005, price=65000.0, reason="entry",
                      client_order_id_root=root)
    state.set_meta("active_bracket", json.dumps({
        "signal_id": root, "side": "long", "entry_price": 65000.0, "qty": 0.005,
        "sl_distance": 1000.0, "tp_distance": 2000.0, "place_tp": place_tp, **extra}))


def _rest(ex: FakeEx, root: str, leg: str, side: str = "SELL", prefix: str = PREFIX) -> dict:
    return ex._rest_algo(f"{prefix}{root}-{leg}", side, 0.005, 1.0)


def _reprotect(bot, n: int = 1) -> None:
    for _ in range(n):
        bot._last_reprotect_ts = 0.0   # defeat the throttle between ticks
        bot._maybe_reprotect(150.0)


def _cancelled_coids(ex: FakeEx, before: list[dict]) -> set[str]:
    ids = {c[1]["algoId"] for c in ex.calls if c[0] == "fapiPrivateDeleteAlgoOrder"}
    return {r["clientAlgoId"] for r in before if r["algoId"] in ids}


class TestReprotectEscalates:

    def test_would_trigger_on_the_replaced_stop_closes_at_once(self) -> None:
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex)
        _rest(ex, ROOT, "t")
        ex.fail["STOP_MARKET"] = [WOULD_TRIGGER] * 5
        with patch("bot.send_alert") as alert:
            _reprotect(bot, 1)
        assert ex.pos is None and len(ex.closes()) == 1
        assert [f[0] for f in _fills()] == ["entry", "no_stop_close"]
        assert any("NO STOP" in s for s in _subjects(alert))

    def test_repeated_replace_failure_closes_after_three(self) -> None:
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex)
        ex.fail["STOP_MARKET"] = [rejected()] * 10
        with patch("bot.send_alert"):
            _reprotect(bot, 2)
            assert ex.pos is not None and ex.closes() == [], "not yet — 2 failures"
            _reprotect(bot, 1)
        assert ex.pos is None and len(ex.closes()) == 1

    def test_failure_count_survives_a_restart(self) -> None:
        ex = FakeEx()
        ex.fail["STOP_MARKET"] = [rejected()] * 10
        _open(ex)
        with patch("bot.send_alert"):
            _reprotect(_bot(ex), 2)
            _reprotect(_bot(ex), 1)    # a fresh process
        assert ex.pos is None

    def test_cap_reached_with_the_stop_missing_closes(self) -> None:
        """Cap = the stop keeps disappearing. Base alerted once and left the
        position naked for good."""
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex, reprotect_count=3)
        _rest(ex, ROOT, "t")
        with patch("bot.send_alert"):
            _reprotect(bot, 1)
        assert ex.pos is None and len(ex.closes()) == 1

    def test_cap_reached_with_only_the_tp_missing_does_not_close(self) -> None:
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex, tp_reprotect_count=3)
        _rest(ex, ROOT, "s")
        with patch("bot.send_alert") as alert:
            _reprotect(bot, 3)
        assert ex.pos is not None and ex.closes() == []
        assert ex.creates() == []
        assert alert.call_count == 1


class TestReprotectNeverCancelsAHealthyStop:

    def test_missing_tp_places_only_the_tp(self) -> None:
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex)
        sl = _rest(ex, ROOT, "s")
        with patch("bot.send_alert"):
            _reprotect(bot, 1)
        assert not any(c[0] == "fapiPrivateDeleteAlgoOrder" for c in ex.calls), \
            "the healthy SL was cancelled to fix a TP"
        assert sl in ex.algo
        assert [c[2] for c in ex.creates()] == ["TAKE_PROFIT_MARKET"]
        assert ex.creates()[0][6]["newClientOrderId"] == f"{PREFIX}{ROOT}-t"

    def test_missing_sl_places_only_the_sl_and_keeps_the_tp(self) -> None:
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex)
        tp = _rest(ex, ROOT, "t")
        with patch("bot.send_alert"):
            _reprotect(bot, 1)
        assert tp in ex.algo
        assert [c[2] for c in ex.creates()] == ["STOP_MARKET"]
        assert ex.creates()[0][6]["newClientOrderId"] == f"{PREFIX}{ROOT}-s"


class TestReprotectExactRoot:

    def test_an_older_trades_stop_does_not_count_as_ours(self) -> None:
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex)
        _rest(ex, OLD, "s")          # leftover from the previous trade
        _rest(ex, ROOT, "t")
        with patch("bot.send_alert"):
            _reprotect(bot, 1)
        placed = [c[6]["newClientOrderId"] for c in ex.creates("STOP_MARKET")]
        assert placed == [f"{PREFIX}{ROOT}-s"]
        # ...and the old stop is cancelled only AFTER ours rests: until then it
        # is the only stop the position has.
        i_place = next(i for i, c in enumerate(ex.calls)
                       if c[0] == "create_order" and c[2] == "STOP_MARKET")
        i_cancel = next(i for i, c in enumerate(ex.calls)
                        if c[0] == "fapiPrivateDeleteAlgoOrder")
        assert i_place < i_cancel
        assert [r["clientAlgoId"] for r in ex.algo] == [f"{PREFIX}{ROOT}-t", f"{PREFIX}{ROOT}-s"]

    def test_old_stop_is_kept_when_ours_cannot_be_placed(self) -> None:
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex)
        old = _rest(ex, OLD, "s")
        ex.fail["STOP_MARKET"] = [rejected()]
        with patch("bot.send_alert"):
            _reprotect(bot, 1)
        assert old in ex.algo, "the only resting stop was cancelled"
        assert ex.pos is not None

    def test_a_stop_on_the_wrong_side_does_not_count(self) -> None:
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex)
        _rest(ex, ROOT, "s", side="BUY")   # cannot close a long
        _rest(ex, ROOT, "t")
        with patch("bot.send_alert"):
            _reprotect(bot, 1)
        assert len(ex.creates("STOP_MARKET")) == 1

    def test_stale_root_legs_are_swept_and_nothing_else_is(self) -> None:
        ex = FakeEx()
        bot = _bot(ex)
        _open(ex)
        ours = [_rest(ex, ROOT, "s"), _rest(ex, ROOT, "t")]
        _rest(ex, OLD, "s")
        _rest(ex, OLD, "t")
        manual = ex._rest_algo("web_abc123", "SELL", 0.005, 1.0)
        other_leg = _rest(ex, OLD, "s", prefix="snap-d3-")
        before = list(ex.algo)
        with patch("bot.send_alert"):
            _reprotect(bot, 1)
        assert _cancelled_coids(ex, before) == {f"{PREFIX}{OLD}-s", f"{PREFIX}{OLD}-t"}
        for r in [*ours, manual, other_leg]:
            assert r in ex.algo
        assert ex.creates() == []
