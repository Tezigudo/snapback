"""Breakeven stop (bot._maybe_breakeven) — SOL leg, TRAILING_STOP_VERDICT.md.

Two halves:
  1. PARITY — the live `breakeven_due` replayed over SOL 4h history arms on the
     same bar, at the same stop price, as the validated backtest arm
     (tools/trailing_stop_study.py, be_at_r). Same idea as the v1 exit
     parity test: the harness only earns trust if live does what it measured.
  2. The hook against a mocked exchange — every failure path that could leave
     a live position unprotected, double-stopped, or carrying an untagged order.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import ccxt
import pandas as pd
import pytest

from bot_internals import algo_bracket_leg, breakeven_due, breakeven_stop_price
from exchange import state
from exchange.binance_client import BinanceClient, Position

ROOT_DIR = Path(__file__).resolve().parent.parent
SOL_4H = ROOT_DIR / "data" / "historical" / "SOL_USDT_USDT_4h.parquet"

PREFIX = "sol-st-"
ROOT = "1789000000000"
BAR = 4 * 3600


# ---------------------------------------------------------------------------
# 1. pure helpers
# ---------------------------------------------------------------------------

class TestBreakevenDue:

    def test_long_arms_at_exactly_2R_including_fill_bar(self) -> None:
        # entry 100, 1R = 5 → due once any closed High >= 110
        assert breakeven_due("long", 100.0, 5.0, [104, 109.99], [95, 99], 2.0) == (False, pytest.approx(1.998))
        due, mfe = breakeven_due("long", 100.0, 5.0, [104, 110.0], [95, 99], 2.0)
        assert due and mfe == pytest.approx(2.0)

    def test_short_mirrors(self) -> None:
        due, mfe = breakeven_due("short", 100.0, 5.0, [101, 102], [95, 90.0], 2.0)
        assert due and mfe == pytest.approx(2.0)
        assert breakeven_due("short", 100.0, 5.0, [101], [90.5], 2.0)[0] is False

    def test_adverse_moves_never_count(self) -> None:
        assert breakeven_due("long", 100.0, 5.0, [99, 98], [80, 70], 2.0) == (False, 0.0)

    @pytest.mark.parametrize("args", [
        ("long", 100.0, 0.0, [200], [1], 2.0),     # no 1R
        ("long", 0.0, 5.0, [200], [1], 2.0),       # no entry
        ("long", 100.0, 5.0, [200], [1], 0.0),     # disabled
        ("flat", 100.0, 5.0, [200], [1], 2.0),
    ])
    def test_bad_inputs_are_not_due(self, args) -> None:
        assert breakeven_due(*args)[0] is False

    def test_nan_bars_are_ignored(self) -> None:
        assert breakeven_due("long", 100.0, 5.0, [float("nan"), 111.0], [90, 90], 2.0)[0]

    def test_stop_price_buffer(self) -> None:
        assert breakeven_stop_price("long", 100.0, 0.0012) == pytest.approx(100.12)
        assert breakeven_stop_price("short", 100.0, 0.0012) == pytest.approx(99.88)

    def test_sb_is_classified_as_the_stop(self) -> None:
        row = {"algoId": "1", "clientAlgoId": f"{PREFIX}{ROOT}-sb", "reduceOnly": True}
        assert algo_bracket_leg(row, PREFIX) == "sl"
        # and the close leg of a market breakeven exit is NOT a bracket leg
        assert algo_bracket_leg({**row, "clientAlgoId": f"{PREFIX}{ROOT}-be"}, PREFIX) is None


# ---------------------------------------------------------------------------
# 2. PARITY vs the validated backtest arm
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SOL_4H.exists(), reason="SOL 4h cache not present")
class TestParityWithBacktest:

    @pytest.fixture(scope="class")
    def replay(self):
        # The research harness lives on main only; the curated droplet branch
        # does not carry it, so there this class skips rather than errors.
        T = pytest.importorskip("tools.trailing_stop_study")
        sink: list = []
        r = T._run_sol({"be_at_r": 2.0, "_be_sink": sink}, 0.0005, "parity_test")
        bars = pd.read_parquet(SOL_4H)
        bars = bars.rename(columns={c: c.capitalize() for c in bars.columns})
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        return sink, r["trades_df"], bars

    def _live_arming(self, bars, entry_time, entry_price, risk, is_long, exit_time):
        """First closed bar (fill bar included) at which live breakeven_due fires."""
        win = bars[(bars.index >= entry_time) & (bars.index <= exit_time)]
        side = "long" if is_long else "short"
        for k in range(len(win)):
            due, _ = breakeven_due(side, entry_price, risk,
                                   win["High"].iloc[: k + 1].tolist(),
                                   win["Low"].iloc[: k + 1].tolist(), 2.0)
            if due:
                return win.index[k]
        return None

    def test_backtest_actually_moved_stops(self, replay) -> None:
        sink, trades, _ = replay
        assert len(trades) >= 100
        assert len(sink) >= 20, "the arm should fire on a meaningful share of trades"

    def test_every_backtest_move_is_reproduced_on_the_same_bar_and_price(self, replay) -> None:
        sink, _, bars = replay
        for ev in sink:
            t = self._live_arming(bars, ev["entry_time"], ev["entry_price"], ev["risk"],
                                  ev["is_long"], ev["armed_bar_time"])
            assert t == ev["armed_bar_time"], ev
            side = "long" if ev["is_long"] else "short"
            assert breakeven_stop_price(side, ev["entry_price"], 0.001) == pytest.approx(ev["stop"])

    def test_live_never_arms_where_the_backtest_did_not(self, replay) -> None:
        """No MISSED arms either: every trade live would arm (before its exit
        bar) has a backtest move. Uses the backtest's 1R = |entry − initial SL|."""
        sink, trades, bars = replay
        armed = {pd.Timestamp(ev["entry_time"]) for ev in sink}
        for _, tr in trades.iterrows():
            entry_t = pd.Timestamp(tr["EntryTime"]).tz_localize(None) \
                if pd.Timestamp(tr["EntryTime"]).tzinfo else pd.Timestamp(tr["EntryTime"])
            exit_t = pd.Timestamp(tr["ExitTime"]).tz_localize(None) \
                if pd.Timestamp(tr["ExitTime"]).tzinfo else pd.Timestamp(tr["ExitTime"])
            if entry_t in armed or pd.isna(tr["SL"]):
                continue
            # The stop is only applied from the NEXT bar, so a trade that exits
            # on the arming bar itself never shows a move — exclude that bar.
            prev_bar = exit_t - pd.Timedelta(hours=4)
            risk = abs(float(tr["EntryPrice"]) - float(tr["SL"]))
            t = self._live_arming(bars, entry_t, float(tr["EntryPrice"]), risk,
                                  float(tr["Size"]) > 0, prev_bar)
            assert t is None, (entry_t, t)


# ---------------------------------------------------------------------------
# 3. the bot hook against a mocked exchange
# ---------------------------------------------------------------------------

PARAMS: dict = {
    "symbol": "SOL/USDT:USDT",
    "hedge": {"enabled": False, "client_order_id_prefix": PREFIX},
    "execution": {"poll_interval_s": "5", "order_type": "limit",
                  "limit_offset_bps": "0", "limit_timeout_s": "25"},
    "timeframes": {"entry": "4h"},
    "sizing": {"leverage": 3, "risk_per_trade_pct": 2.2},   # synthetic
    "deploy": {"kill_switch_equity_fraction": 0.5, "min_capital_warn_usdt": 1.0},
    "strategy_name": "supertrend",
    "strategy": {},
    "breakeven": {"enabled": True, "at_r": 1.8, "buffer_pct": 0.15},   # synthetic
}
ENTRY = 100.0
R = 5.0           # sl_distance
QTY = 0.66


def _bot(params: dict | None = None, dry_run: bool = False):
    from bot import Bot
    mc = MagicMock()
    mc.ex.parse_timeframe.return_value = BAR
    mc.coid_prefix = PREFIX
    mc.env = "testnet"
    mc.fetch_algo_orders.return_value = ([_algo("s"), _algo("t")], True)
    mc.cancel_algo_by_coid.return_value = True
    mc.fetch_mark_price.return_value = 111.0
    mc.fetch_position.return_value = Position("SOL/USDT:USDT", "long", QTY, ENTRY, 5.0, 20.0)
    with patch("bot.BinanceClient.from_env", return_value=mc):
        bot = Bot(params=params or PARAMS, dry_run=dry_run, instance="sol_supertrend")
    return bot, mc


def _algo(leg: str) -> dict:
    return {"algoId": f"id-{leg}", "clientAlgoId": f"{PREFIX}{ROOT}-{leg}", "reduceOnly": True}


def _seed(bars_ago_filled: int = 3, side: str = "long", signal_id: str = ROOT,
          entry_price: float = ENTRY, **extra) -> None:
    """An open trade: active_bracket + an entry fill `bars_ago_filled` bars back."""
    state.set_meta("active_bracket", json.dumps({
        "signal_id": signal_id, "side": side, "entry_price": entry_price,
        "sl_distance": R, "tp_distance": 50.0, "place_tp": True, **extra}))
    state.record_fill(side=side, qty=QTY, price=entry_price, reason="entry",
                      client_order_id_root=ROOT)
    fill_ts = (int(time.time() // BAR) - bars_ago_filled) * BAR + 5
    with sqlite3.connect(state.DB_PATH) as c:
        c.execute("UPDATE fills SET ts=? WHERE reason='entry'",
                  (pd.Timestamp(fill_ts, unit="s").isoformat(),))


def _ohlcv(mc: MagicMock, highs: list[float], lows: list[float] | None = None) -> None:
    """Closed bars ending at the last closed bar, plus one forming bar."""
    n = len(highs)
    last_closed = (int(time.time() // BAR) - 1) * BAR
    idx = pd.to_datetime([last_closed - (n - 1 - k) * BAR for k in range(n)] + [last_closed + BAR],
                         unit="s")
    lows = lows or [ENTRY - 1] * n
    mc.fetch_ohlcv.return_value = pd.DataFrame(
        {"Open": ENTRY, "High": [*highs, 999.0], "Low": [*lows, 1.0], "Close": ENTRY,
         "Volume": 1.0}, index=idx)


def _ab() -> dict:
    return json.loads(state.get_meta("active_bracket"))


class TestHookGating:

    def test_disabled_leg_makes_zero_api_calls(self) -> None:
        bot, mc = _bot({**PARAMS, "breakeven": {"enabled": False}})
        mc.reset_mock()
        bot._maybe_breakeven(100.0)
        assert mc.method_calls == []

    def test_absent_block_makes_zero_api_calls(self) -> None:
        p = {k: v for k, v in PARAMS.items() if k != "breakeven"}
        bot, mc = _bot(p)
        mc.reset_mock()
        bot._maybe_breakeven(100.0)
        assert mc.method_calls == []

    def test_once_per_closed_bar(self) -> None:
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 105, 106, 107])      # not due
        bot._maybe_breakeven(100.0)
        bot._maybe_breakeven(100.0)
        assert mc.fetch_position.call_count == 1
        mc.place_tagged_stop.assert_not_called()

    def test_flat_is_a_noop(self) -> None:
        bot, mc = _bot()
        mc.fetch_position.return_value = Position("SOL/USDT:USDT", "flat", 0, 0, 0, 0)
        bot._maybe_breakeven(100.0)
        mc.fetch_ohlcv.assert_not_called()
        mc.place_tagged_stop.assert_not_called()


class TestMove:

    def test_moves_stop_keeps_tp_persists(self) -> None:
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 110.5, 106, 107])     # fill bar .. last closed; 2.1R reached
        with patch("bot.send_alert") as alert:
            bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_called_once()
        a = mc.place_tagged_stop.call_args.args
        assert a[0] == "SOL/USDT:USDT" and a[1] == "long" and a[2] == QTY
        assert a[3] == pytest.approx(100.15) and a[4] == ROOT and a[5] == "sb"
        # ONLY the original stop is cancelled — never a prefix-wide sweep that
        # would take the TP with it.
        mc.cancel_algo_by_coid.assert_called_once_with("SOL/USDT:USDT", f"{PREFIX}{ROOT}-s")
        mc.cancel_open_orders.assert_not_called()
        mc.close_position.assert_not_called()
        assert _ab()["be_moved"] is True and _ab()["be_price"] == pytest.approx(100.15)
        alert.assert_called_once()

    def test_fill_bar_high_counts(self) -> None:
        """The fill bar's own High is in the window (backtest fills at its open)."""
        bot, mc = _bot()
        _seed(bars_ago_filled=1)                 # filled at the open of the last closed bar
        _ohlcv(mc, [99, 99, 111.0])              # ...whose own High is the only one in the window
        with patch("bot.send_alert"):
            bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_called_once()

    def test_bars_before_the_fill_do_not_count(self) -> None:
        bot, mc = _bot()
        _seed(bars_ago_filled=1)
        _ohlcv(mc, [130, 130, 104, 104])          # the 130s are before the fill
        bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_not_called()

    def test_short_side(self) -> None:
        bot, mc = _bot()
        mc.fetch_position.return_value = Position("SOL/USDT:USDT", "short", QTY, ENTRY, 5.0, 20.0)
        mc.fetch_mark_price.return_value = 89.0
        _seed(side="short")
        _ohlcv(mc, [101, 101, 101], lows=[95, 89.9, 96])
        with patch("bot.send_alert"):
            bot._maybe_breakeven(100.0)
        a = mc.place_tagged_stop.call_args.args
        assert a[1] == "short" and a[3] == pytest.approx(99.85)

    def test_already_moved_is_never_repeated(self) -> None:
        bot, mc = _bot()
        _seed(be_moved=True)
        _ohlcv(mc, [104, 120, 106])
        bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_not_called()
        mc.fetch_ohlcv.assert_not_called()

    def test_dry_run_places_nothing(self) -> None:
        bot, mc = _bot(dry_run=True)
        _seed()
        _ohlcv(mc, [104, 120, 106])
        bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_not_called()
        mc.close_position.assert_not_called()


class TestSourceryRound:
    """PR #32 Sourcery findings — each pinned to the behaviour it asked for."""

    def test_late_fill_does_not_credit_pre_fill_extremes(self) -> None:
        """A fill 2h into a bar must not count that bar's High (printed maybe
        before the position existed)."""
        bot, mc = _bot()
        _seed(bars_ago_filled=2)
        with sqlite3.connect(state.DB_PATH) as c:
            fill_bar = (int(time.time() // BAR) - 2) * BAR
            c.execute("UPDATE fills SET ts=? WHERE reason='entry'",
                      (pd.Timestamp(fill_bar + 7200, unit="s").isoformat(),))
        _ohlcv(mc, [99, 99, 120.0, 104])     # 120 = the fill bar, 104 = the bar after
        bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_not_called()

    def test_fill_inside_grace_still_counts_its_bar(self) -> None:
        bot, mc = _bot()
        _seed(bars_ago_filled=2)            # _seed fills 5 s after the bar open
        _ohlcv(mc, [99, 99, 120.0, 104])
        with patch("bot.send_alert"):
            bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_called_once()

    def test_running_extreme_survives_a_truncated_window(self) -> None:
        """An excursion no longer inside the fetched window still counts."""
        bot, mc = _bot()
        _seed(be_ext=111.0)                 # +2.2R reached earlier
        _ohlcv(mc, [104, 104, 104])
        with patch("bot.send_alert"):
            bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_called_once()

    def test_running_extreme_is_persisted_when_not_due(self) -> None:
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 107.5, 106, 105])
        bot._maybe_breakeven(100.0)
        assert _ab()["be_ext"] == pytest.approx(107.5)

    def test_ledger_failure_after_close_is_recovered(self) -> None:
        """Sourcery: an exception after close_position() loses the exit.
        It does not — the next tick's flat-edge detector records it."""
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 120, 106])
        mc.fetch_mark_price.return_value = 100.0
        mc.close_position.return_value = {"average": 100.0}
        with patch("bot.send_alert"), \
             patch("bot.state.record_fill", side_effect=sqlite3.OperationalError("disk")):
            bot._maybe_breakeven(100.0)          # swallowed + logged by the hook
        # next tick: the exchange says flat
        bot._last_position_side = "long"
        bot._last_position_entry = ENTRY
        bot._last_position_qty = QTY
        mc.fetch_position.return_value = Position("SOL/USDT:USDT", "flat", 0, 0, 0, 0)
        mc.ex.fetch_my_trades.return_value = [{"side": "sell", "price": 100.0}]
        with patch("bot.send_alert"):
            bot._detect_bracket_exit(100.0)
        with sqlite3.connect(state.DB_PATH) as c:
            rows = list(c.execute("SELECT reason, price FROM fills ORDER BY id"))
        assert rows[-1][0] == "bracket_exit" and rows[-1][1] == pytest.approx(100.0)


class TestReviewRound:
    """/code-review of PR #32."""

    def test_failed_close_leaves_the_original_bracket_untouched(self) -> None:
        """The close raising must not have swept -s/-t first. Otherwise the
        position sits with no stop, and a retry after price recovers restores
        only -sb and loses the TP for good."""
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 120, 106])
        mc.fetch_mark_price.return_value = 100.0
        mc.close_position.side_effect = ccxt.NetworkError("timeout")
        bot._maybe_breakeven(100.0)
        mc.cancel_open_orders.assert_not_called()
        mc.cancel_algo_by_coid.assert_not_called()
        assert mc.close_position.call_args.kwargs["sweep_first"] is False
        assert "be_moved" not in _ab()
        assert bot._be_retry_after > time.time()


class TestStaleRecords:

    @pytest.mark.parametrize("seed", [
        {"signal_id": "someOtherRoot"},        # record belongs to an older trade
        {"side": "short"},                     # wrong side
        {"entry_price": 80.0},                 # >2% entry drift
    ])
    def test_mismatched_bracket_record_is_ignored(self, seed) -> None:
        bot, mc = _bot()
        _seed(**seed)
        _ohlcv(mc, [104, 120, 106])
        bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_not_called()
        mc.close_position.assert_not_called()


class TestFailurePaths:

    def test_unreadable_algo_book_places_nothing_and_retries(self) -> None:
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 120, 106])
        mc.fetch_algo_orders.return_value = ([], False)
        bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_not_called()
        assert bot._be_retry_after > time.time()
        assert "be_moved" not in _ab()
        # once the retry window passes, it tries again (same bar)
        bot._be_retry_after = 0.0
        mc.fetch_algo_orders.return_value = ([_algo("s"), _algo("t")], True)
        with patch("bot.send_alert"):
            bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_called_once()

    def test_existing_sb_is_not_doubled(self) -> None:
        """A placement that timed out on our side but was accepted: the retry
        must find the resting -sb and NOT place another."""
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 120, 106])
        mc.fetch_algo_orders.return_value = ([_algo("s"), _algo("t"), _algo("sb")], True)
        with patch("bot.send_alert"):
            bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_not_called()
        mc.cancel_algo_by_coid.assert_called_once_with("SOL/USDT:USDT", f"{PREFIX}{ROOT}-s")
        assert _ab()["be_moved"] is True

    def test_mark_already_through_closes_at_market(self) -> None:
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 120, 106])
        mc.fetch_mark_price.return_value = 100.1       # below 100.15 → through
        mc.close_position.return_value = {"average": 100.04}
        with patch("bot.send_alert"):
            bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_not_called()
        mc.close_position.assert_called_once_with(
            "SOL/USDT:USDT", client_order_id_root=ROOT, close_leg="be", sweep_first=False)
        # the sweep comes AFTER the close, never before it
        names = [c[0] for c in mc.method_calls]
        assert names.index("close_position") < names.index("cancel_open_orders")
        with sqlite3.connect(state.DB_PATH) as c:
            reasons = [r[0] for r in c.execute("SELECT reason FROM fills ORDER BY id")]
        assert reasons[-1] == "breakeven_exit"
        assert _ab()["be_moved"] is True

    def test_refused_placement_then_mark_through_closes(self) -> None:
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 120, 106])
        mc.fetch_mark_price.side_effect = [111.0, 100.0]  # fine, then crossed
        mc.place_tagged_stop.side_effect = ccxt.InvalidOrder("would immediately trigger")
        mc.close_position.return_value = {"average": 100.0}
        with patch("bot.send_alert"):
            bot._maybe_breakeven(100.0)
        mc.close_position.assert_called_once()
        assert _ab()["be_moved"] is True

    def test_refused_placement_not_through_retries_without_moving(self) -> None:
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 120, 106])
        mc.place_tagged_stop.side_effect = ccxt.NetworkError("timeout")
        bot._maybe_breakeven(100.0)
        mc.close_position.assert_not_called()
        mc.cancel_algo_by_coid.assert_not_called()   # old stop stays until the new one rests
        assert "be_moved" not in _ab()
        assert bot._be_retry_after > time.time()

    def test_repeated_failures_alert_once_then_wait_for_next_bar(self) -> None:
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 120, 106])
        mc.place_tagged_stop.side_effect = ccxt.NetworkError("timeout")
        with patch("bot.send_alert") as alert:
            for _ in range(bot.BE_FAIL_LIMIT):
                bot._be_retry_after = 0.0
                bot._maybe_breakeven(100.0)
        alert.assert_called_once()
        assert bot._be_evaluated_bar == (int(time.time() // BAR) - 1) * BAR

    def test_close_that_finds_position_already_flat_writes_nothing(self) -> None:
        """Review finding: a bogus close row would suppress the real exit."""
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 120, 106])
        mc.fetch_mark_price.return_value = 100.0
        mc.close_position.return_value = None
        with patch("bot.send_alert") as alert:
            bot._maybe_breakeven(100.0)
        with sqlite3.connect(state.DB_PATH) as c:
            reasons = [r[0] for r in c.execute("SELECT reason FROM fills ORDER BY id")]
        assert reasons == ["entry"]
        alert.assert_not_called()

    def test_old_stop_already_gone_is_not_an_alarm(self) -> None:
        """Retry after an earlier attempt cancelled -s but crashed before
        persisting: -s is simply absent — protected by -sb, no scary alert."""
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 120, 106])
        mc.fetch_algo_orders.return_value = ([_algo("t"), _algo("sb")], True)
        mc.cancel_algo_by_coid.return_value = False
        with patch("bot.send_alert") as alert:
            bot._maybe_breakeven(100.0)
        assert "could not be cancelled" not in alert.call_args.args[1]
        assert _ab()["be_moved"] is True

    def test_old_stop_cancel_failure_still_persists_and_warns(self) -> None:
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 120, 106])
        mc.cancel_algo_by_coid.return_value = False
        with patch("bot.send_alert") as alert:
            bot._maybe_breakeven(100.0)
        assert _ab()["be_moved"] is True
        assert "could not be cancelled" in alert.call_args.args[1]

    def test_unpublished_closed_bar_waits(self) -> None:
        bot, mc = _bot()
        _seed()
        _ohlcv(mc, [104, 120, 106])
        df = mc.fetch_ohlcv.return_value
        mc.fetch_ohlcv.return_value = df.iloc[:-1]   # last closed bar missing
        bot._maybe_breakeven(100.0)
        mc.place_tagged_stop.assert_not_called()
        assert bot._be_evaluated_bar is None


class TestClientNoUntaggedFallback:

    def _client(self) -> BinanceClient:
        ex = MagicMock()
        ex.market.return_value = {"precision": {"amount": 0.01}, "id": "SOLUSDT"}
        return BinanceClient(ex=ex, env="testnet", coid_prefix=PREFIX)

    def test_tagged_and_reduce_only_on_mark(self) -> None:
        c = self._client()
        c.place_tagged_stop("SOL/USDT:USDT", "long", 0.66, 100.1, ROOT, "sb")
        args, kw = c.ex.create_order.call_args
        assert args[:5] == ("SOL/USDT:USDT", "STOP_MARKET", "sell", 0.66, None)
        p = kw["params"]
        assert p["newClientOrderId"] == f"{PREFIX}{ROOT}-sb"
        assert p["reduceOnly"] is True and p["workingType"] == "MARK_PRICE"

    def test_duplicate_id_is_raised_never_retried_untagged(self) -> None:
        c = self._client()
        c.ex.create_order.side_effect = ccxt.InvalidOrder("Duplicate clientOrderId")
        with pytest.raises(ccxt.InvalidOrder):
            c.place_tagged_stop("SOL/USDT:USDT", "long", 0.66, 100.1, ROOT, "sb")
        assert c.ex.create_order.call_count == 1

    def test_close_position_sweep_first_false_does_not_cancel(self) -> None:
        c = self._client()
        c.ex.fetch_positions.return_value = [
            {"symbol": "SOL/USDT:USDT", "contracts": 0.66, "side": "long", "entryPrice": 100}]
        c.cancel_open_orders = MagicMock()  # type: ignore[method-assign]
        c.close_position("SOL/USDT:USDT", client_order_id_root=ROOT, close_leg="be",
                         sweep_first=False)
        c.cancel_open_orders.assert_not_called()
        args, kw = c.ex.create_order.call_args
        assert args[:4] == ("SOL/USDT:USDT", "market", "sell", 0.66)
        assert kw["params"]["reduceOnly"] is True
        # default behaviour unchanged for every other caller
        c.close_position("SOL/USDT:USDT", client_order_id_root=ROOT, close_leg="x")
        c.cancel_open_orders.assert_called_once()

    def test_cancel_by_coid_is_exact(self) -> None:
        c = self._client()
        c.ex.fapiPrivateGetOpenAlgoOrders.return_value = [_algo("s"), _algo("t"), _algo("sb")]
        assert c.cancel_algo_by_coid("SOL/USDT:USDT", f"{PREFIX}{ROOT}-s") is True
        c.ex.fapiPrivateDeleteAlgoOrder.assert_called_once_with(
            {"symbol": "SOLUSDT", "algoId": "id-s"})

    def test_cancel_by_coid_unreadable_or_missing_is_false(self) -> None:
        c = self._client()
        c.ex.fapiPrivateGetOpenAlgoOrders.side_effect = RuntimeError("boom")
        assert c.cancel_algo_by_coid("SOL/USDT:USDT", f"{PREFIX}{ROOT}-s") is False
        c.ex.fapiPrivateGetOpenAlgoOrders.side_effect = None
        c.ex.fapiPrivateGetOpenAlgoOrders.return_value = [_algo("t")]
        assert c.cancel_algo_by_coid("SOL/USDT:USDT", f"{PREFIX}{ROOT}-s") is False
        c.ex.fapiPrivateDeleteAlgoOrder.assert_not_called()


def test_shipped_sol_config_enables_it_and_no_other_leg_does() -> None:
    """Deployed-config guard: reads the real (untracked) files, skips where absent."""
    from tests._live_config import live_config
    sol = live_config("params_sol_supertrend.yaml")
    be = sol["breakeven"]
    assert be["enabled"] is True
    assert float(be["at_r"]) > 0 and float(be["buffer_pct"]) >= 0
    for other in ("params.yaml", "params_donchian.yaml"):
        assert not (live_config(other).get("breakeven") or {}).get("enabled")


def test_templates_carry_breakeven_only_on_sol() -> None:
    import yaml
    cfg = ROOT_DIR / "config"
    sol = yaml.safe_load((cfg / "params_sol_supertrend.example.yaml").read_text())
    assert set(sol["breakeven"]) == {"enabled", "at_r", "buffer_pct"}
    for other in ("params.example.yaml", "params_donchian.example.yaml"):
        assert "breakeven" not in yaml.safe_load((cfg / other).read_text())
