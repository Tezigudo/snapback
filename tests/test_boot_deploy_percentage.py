"""The boot line must not read a deposit as profit.

Regression cover for 2026-09-09. `deploy_start_equity` is a RAW equity snapshot
frozen when the leg first booted, so every later deposit moves the numerator
and leaves the denominator behind. sol_supertrend booted announcing

    Resuming deploy. start=60.00 current=73.71 (+22.85%)

on a leg that was DOWN 7.9%: $60 was deployed on 25 Jul and another $20 funded
on 28 Aug, so $80 went in and $73.71 came back. This is the same two-clocks
defect PR #25 fixed in monitor.py, and it errs in the flattering direction --
the one that talks you out of looking.

The fix reports against P (principal_base + Sigma USDT ledger), which is the
denominator `_check_kill_switch` already names, so the boot line and the
kill-switch line can no longer describe the same leg differently.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exchange.binance_client import Position  # noqa: E402

_MINIMAL_PARAMS = {
    "symbol": "SOL/USDT:USDT",
    "execution": {
        "poll_interval_s": "5",
        "order_type": "market",
        "limit_offset_bps": "0",
        "limit_timeout_s": "25",
    },
    "timeframes": {"entry": "4h"},
    "sizing": {"leverage": 3, "risk_per_trade_pct": 2.2},   # synthetic
    "deploy": {"kill_switch_equity_fraction": 0.5, "min_capital_warn_usdt": 1.0},
    "strategy": {},
}


def _make_bot():
    from bot import Bot
    mc = MagicMock()
    mc.ex.parse_timeframe.return_value = 14400  # 4h
    mc.coid_prefix = "snap-sol-"
    mc.env = "testnet"
    mc.cancel_open_orders.return_value = 0
    with patch("bot.BinanceClient.from_env", return_value=mc):
        bot = Bot(params=_MINIMAL_PARAMS, dry_run=False)
    return bot, mc


def _flat() -> Position:
    return Position(symbol="SOL/USDT:USDT", side="flat", qty=0.0,
                    entry_price=0.0, unrealized_pnl=0.0, margin_used=0.0)


def _boot_patches():
    return patch.multiple(
        "bot",
        check_symbol=MagicMock(),
        check_leverage=MagicMock(),
        send_alert=MagicMock(),
    )


def _state_patches(start_equity: float):
    """get_float serves deploy_start_equity; a positive value takes the
    resume branch, which is the one under test."""
    return patch.multiple(
        "bot.state",
        init_db=MagicMock(),
        get_float=MagicMock(return_value=start_equity),
        set_float=MagicMock(),
        set_meta=MagicMock(),
        enqueue_bot_event=MagicMock(),
        record_event=MagicMock(),
        latest_entry_coid_root=MagicMock(return_value=None),
    )


def _boot_and_capture(caplog, *, start_equity: float, equity: float,
                      principal: float | None) -> str:
    bot, mc = _make_bot()
    mc.fetch_equity_usdt.return_value = equity
    mc.fetch_position.return_value = _flat()

    with caplog.at_level(logging.INFO), \
         _boot_patches(), _state_patches(start_equity), \
         patch("bot.principal.initialize", MagicMock(return_value=principal)), \
         patch("bot.principal.get_principal", MagicMock(return_value=principal)):
        bot.boot()

    return "\n".join(r.getMessage() for r in caplog.records
                     if "Resuming deploy" in r.getMessage())


class TestResumingDeployPercentage:

    def test_deposit_is_not_reported_as_profit(self, caplog):
        """The live 2026-09-09 sol_supertrend numbers, exactly."""
        line = _boot_and_capture(caplog, start_equity=60.00, equity=73.71,
                                 principal=80.00)
        assert "-7.86% vs principal" in line
        # The specific wrong number this bug printed. Guarding the string, not
        # just the sign, because a half-fix that still divides by start_eq
        # would keep producing it.
        assert "22.85" not in line

    def test_withdrawal_is_not_reported_as_a_loss(self, caplog):
        """The mirror case. $60 in, $20 taken back out, $45 left = up 12.5% on
        the $40 still at risk -- but start_eq would call it -25%."""
        line = _boot_and_capture(caplog, start_equity=60.00, equity=45.00,
                                 principal=40.00)
        assert "+12.50% vs principal" in line
        assert "-25.00" not in line

    def test_untouched_leg_reads_exactly_as_before(self, caplog):
        """With no transfers since deploy, P == start_eq and the number is
        unchanged. Proves the fix re-denominates rather than moving every
        leg's reported figure."""
        line = _boot_and_capture(caplog, start_equity=100.00, equity=110.00,
                                 principal=100.00)
        assert "+10.00% vs principal" in line

    def test_principal_pending_says_the_number_is_unadjusted(self, caplog):
        """get_principal() is None until the income backfill lands. Falling
        back to start_eq is correct -- printing it WITHOUT the caveat is what
        created this bug, so the caveat is the assertion."""
        line = _boot_and_capture(caplog, start_equity=60.00, equity=73.71,
                                 principal=None)
        assert "+22.85%" in line
        assert "principal pending" in line
        assert "NOT transfer-adjusted" in line

    def test_non_positive_principal_is_not_called_pending(self, caplog):
        """breached() treats P <= 0 as unknown rather than as a real anchor, so
        this branch must agree with it -- but a fully withdrawn leg has a
        ledger that was read fine and nets to nothing. Calling that "pending"
        sends an operator looking for a backfill that already finished."""
        line = _boot_and_capture(caplog, start_equity=60.00, equity=10.00,
                                 principal=0.0)
        assert "principal non-positive (0.00)" in line
        assert "pending" not in line
        assert "NOT transfer-adjusted" in line
