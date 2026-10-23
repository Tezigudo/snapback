"""F5: an unreadable equity must mean "unknown this tick", never "zero".

fetch_equity_usdt() used to default a missing totalWalletBalance to 0.0, which
principal.breached() reads as a >99% drawdown -> kill switch -> flatten + HALT.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from exchange.binance_client import BinanceClient
from tests.test_boot_resume import _make_bot, _flat


def _client(info, usdt=None) -> BinanceClient:
    ex = MagicMock()
    bal = {"info": info}
    if usdt is not None:
        bal["USDT"] = usdt
    ex.fetch_balance.return_value = bal
    return BinanceClient(ex, "testnet")


@pytest.mark.parametrize("info", [
    {},
    {"totalUnrealizedProfit": "0"},
    {"totalWalletBalance": "100"},
    {"totalWalletBalance": None, "totalUnrealizedProfit": "0"},
    {"totalWalletBalance": "abc", "totalUnrealizedProfit": "0"},
    {"totalWalletBalance": "nan", "totalUnrealizedProfit": "0"},
    {"totalWalletBalance": "inf", "totalUnrealizedProfit": "0"},
])
def test_bad_equity_read_raises_instead_of_reading_zero(info):
    # Behavioural: base returned 0.0 here (DID NOT RAISE) -> false kill switch.
    with pytest.raises(RuntimeError):
        _client(info, usdt={"total": 0.0}).fetch_equity_usdt()


def test_equity_unavailable_is_not_a_ccxt_error():
    import ccxt
    from exchange.binance_client import EquityUnavailable
    assert not issubclass(EquityUnavailable, ccxt.BaseError)


def test_good_read_unchanged():
    c = _client({"totalWalletBalance": "100.5", "totalUnrealizedProfit": "-0.5"})
    assert c.fetch_equity_usdt() == pytest.approx(100.0)


def _run_one_tick(bot, tmp_path):
    """Drive loop() through exactly one tick, then stop."""
    def _sleep(_s):
        bot._stopped = True
    with patch("bot.halt_source", return_value=None), \
         patch("bot.HEARTBEAT", tmp_path / "hb"), \
         patch("bot.time.sleep", side_effect=_sleep), \
         patch("bot.send_alert") as alert, \
         patch("bot.principal.get_principal", return_value=100.0), \
         patch("bot.principal.is_initialized", return_value=True), \
         patch.object(bot, "_maybe_reconcile_principal"):
        assert bot.loop() == 0
    return alert


def test_missing_wallet_key_does_not_trip_kill_switch_or_flatten(tmp_path):
    bot, mc = _make_bot()
    # Real client on a fake exchange whose balance payload lacks the key.
    real = _client({})
    real.close_position = MagicMock()
    bot.client = real
    _run_one_tick(bot, tmp_path)
    real.close_position.assert_not_called()
    assert not bot.halt_path.exists()


def test_unknown_equity_with_no_history_skips_tick_without_trading_actions(tmp_path):
    from exchange.binance_client import EquityUnavailable
    bot, mc = _make_bot()
    mc.fetch_equity_usdt.side_effect = EquityUnavailable("no key")
    mc.fetch_position.return_value = _flat()
    bot._maybe_enter = MagicMock()
    bot._detect_bracket_exit = MagicMock()
    alert = _run_one_tick(bot, tmp_path)
    mc.close_position.assert_not_called()
    bot._maybe_enter.assert_not_called()
    bot._detect_bracket_exit.assert_not_called()
    assert alert.call_count == 1


def test_unknown_equity_alert_is_rate_limited(tmp_path):
    from exchange.binance_client import EquityUnavailable
    bot, mc = _make_bot()
    mc.fetch_equity_usdt.side_effect = EquityUnavailable("no key")
    ticks = {"n": 0}

    def _sleep(_s):
        ticks["n"] += 1
        if ticks["n"] >= 3:
            bot._stopped = True
    with patch("bot.halt_source", return_value=None), \
         patch("bot.HEARTBEAT", tmp_path / "hb"), \
         patch("bot.time.sleep", side_effect=_sleep), \
         patch("bot.send_alert") as alert, \
         patch("bot.principal.get_principal", return_value=100.0), \
         patch("bot.principal.is_initialized", return_value=True), \
         patch.object(bot, "_maybe_reconcile_principal"):
        bot.loop()
    assert ticks["n"] == 3
    assert alert.call_count == 1


def test_uninitialised_principal_alerts_once_and_still_does_not_trip():
    bot, _ = _make_bot()
    with patch("bot.send_alert") as alert, \
         patch("bot.principal.get_principal", return_value=None):
        assert bot._check_kill_switch(0.0) is False
        assert bot._check_kill_switch(0.0) is False
    assert alert.call_count == 1
    assert not bot.halt_path.exists()


def test_unknown_equity_still_runs_exits_but_not_entries_or_kill_switch(tmp_path):
    from exchange.binance_client import EquityUnavailable
    bot, mc = _make_bot()
    mc.fetch_equity_usdt.side_effect = EquityUnavailable("no key")
    bot._last_known_equity = 1000.0
    for h in ("_detect_bracket_exit", "_maybe_time_stop", "_maybe_channel_exit",
              "_maybe_breakeven", "_maybe_reprotect", "_maybe_enter",
              "_maybe_push_consolidate", "_check_kill_switch"):
        setattr(bot, h, MagicMock())
    _run_one_tick(bot, tmp_path)
    bot._maybe_time_stop.assert_called_once_with(1000.0)
    bot._maybe_reprotect.assert_called_once_with(1000.0)
    bot._maybe_enter.assert_not_called()
    bot._check_kill_switch.assert_not_called()


def test_time_stop_still_fires_while_equity_unreadable(tmp_path):
    from exchange import state
    from exchange.binance_client import EquityUnavailable
    from tests.test_boot_resume import _open
    bot, mc = _make_bot()
    bot.params["strategy"] = {"max_hold_bars": 4}
    bot.bar_seconds = 900
    state.record_fill(side="long", qty=0.01, price=65000.0, reason="entry",
                      client_order_id_root="sig-1")
    with __import__("sqlite3").connect(state.DB_PATH) as c:
        c.execute("UPDATE fills SET ts='2020-01-01T00:00:00+00:00'")
    mc.fetch_equity_usdt.side_effect = EquityUnavailable("no key")
    mc.fetch_position.return_value = _open()
    mc.close_position.return_value = {}
    bot._last_known_equity = 1000.0
    for h in ("_detect_bracket_exit", "_maybe_channel_exit", "_maybe_breakeven",
              "_maybe_reprotect", "_maybe_enter"):
        setattr(bot, h, MagicMock())
    _run_one_tick(bot, tmp_path)
    assert mc.close_position.call_args.kwargs["close_leg"] == "x"
    # The stale stand-in must NOT reach the ledger or the pushed event.
    with __import__("sqlite3").connect(state.DB_PATH) as c:
        row = c.execute("SELECT equity_after FROM fills WHERE reason='time_stop'").fetchone()
    assert row == (None,)
    import json
    pushed = [json.loads(r[2]) for r in state.outbox_pending() if r[1] == "exit"]
    assert pushed and pushed[0]["equity_usd"] is None


def test_exit_with_fresh_equity_still_records_it(tmp_path):
    bot, _ = _make_bot()
    assert bot._ledger_equity(1000.0) == 1000.0
    bot._equity_stale = True
    assert bot._ledger_equity(1000.0) is None


def test_boot_retries_then_aborts_with_alert_before_any_flatten():
    from exchange.binance_client import EquityUnavailable
    bot, mc = _make_bot()
    mc.fetch_equity_usdt.side_effect = EquityUnavailable("no key")
    from tests.test_boot_resume import _boot_patches, _principal_patches
    with _boot_patches(), _principal_patches(), patch("bot.time.sleep"):
        with pytest.raises(EquityUnavailable):
            bot.boot()
        import bot as bot_mod
        assert bot_mod.send_alert.call_count == 1
    assert mc.fetch_equity_usdt.call_count == 4
    mc.close_position.assert_not_called()


def test_consolidate_account_payload_refuses_missing_wallet():
    from tools.consolidate_futures_push import build_account_payload
    with pytest.raises(ValueError):
        build_account_payload({})
    assert build_account_payload({"totalWalletBalance": "5"})["walletBalanceUsd"] == 5.0
