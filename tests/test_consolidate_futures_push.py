"""Tests for the pure payload mappers in tools/consolidate_futures_push.

These cover the row→ingest-shape transforms that turn ccxt responses into the
consolidate /futures/* payloads. The network + ccxt calls are NOT exercised
here (no live key); the mappers are the part that can silently corrupt the
dashboard's numbers, so they're tested directly with synthetic ccxt-shaped
dicts — the same verification strategy used on the TypeScript side.
"""

from __future__ import annotations

import pytest

from tools.consolidate_futures_push import (
    build_account_payload,
    build_bracket_map,
    build_position_payloads,
    build_income_payloads,
)


def test_account_payload_maps_raw_fapi_fields():
    info = {
        "totalWalletBalance": "120.50",
        "totalMarginBalance": "122.10",
        "totalUnrealizedProfit": "1.60",
        "availableBalance": "100.00",
        "ignored": "x",
    }
    assert build_account_payload(info) == {
        "walletBalanceUsd": 120.50,
        "marginBalanceUsd": 122.10,
        "unrealizedPnlUsd": 1.60,
        "availableBalanceUsd": 100.00,
    }


def test_account_payload_refuses_missing_wallet_but_tolerates_other_gaps():
    # A missing wallet balance must not be pushed to the dashboard as $0.
    with pytest.raises(ValueError):
        build_account_payload({})
    assert build_account_payload({"totalWalletBalance": "5"}) == {
        "walletBalanceUsd": 5.0,
        "marginBalanceUsd": 0.0,
        "unrealizedPnlUsd": 0.0,
        "availableBalanceUsd": 0.0,
    }


def test_position_payloads_skip_flat_and_keep_sign():
    positions = [
        {"info": {"symbol": "BTCUSDT", "positionAmt": "0", "positionSide": "BOTH"}},  # flat → skipped
        {"info": {
            "symbol": "BTCUSDT", "positionAmt": "-0.010", "positionSide": "BOTH",
            "entryPrice": "68000", "markPrice": "67800", "unRealizedProfit": "2.0",
            "liquidationPrice": "90000", "leverage": "12",
        }},
    ]
    out = build_position_payloads(positions)
    assert len(out) == 1
    p = out[0]
    assert p["symbol"] == "BTCUSDT"
    assert p["positionAmt"] == -0.010          # sign preserved (short)
    assert p["entryPrice"] == 68000.0
    assert p["liquidationPrice"] == 90000.0
    assert p["leverage"] == 12.0


def test_position_liq_zero_becomes_null():
    positions = [{"info": {
        "symbol": "ETHUSDT", "positionAmt": "1.0", "liquidationPrice": "0",
        "entryPrice": "3000", "markPrice": "3010", "unRealizedProfit": "10", "leverage": "10",
    }}]
    out = build_position_payloads(positions)
    assert out[0]["liquidationPrice"] is None


def test_position_falls_back_to_unified_fields():
    # No raw info beyond positionAmt — unified fields fill the rest.
    positions = [{
        "info": {"positionAmt": "0.5"},
        "symbol": "SOLUSDT", "entryPrice": 150.0, "markPrice": 152.0,
        "unrealizedPnl": 1.0, "liquidationPrice": 90.0, "leverage": 5,
    }]
    out = build_position_payloads(positions)
    assert out[0]["symbol"] == "SOLUSDT"
    assert out[0]["entryPrice"] == 150.0
    assert out[0]["unrealizedPnlUsd"] == 1.0


def test_position_leverage_from_symbol_config_map():
    # Binance /fapi/v3 dropped `leverage` from the position payload — the
    # symbolConfig map is now the normal source.
    positions = [{"info": {
        "symbol": "BTCUSDT", "positionAmt": "-0.005", "entryPrice": "64220.10",
        "markPrice": "64499.50", "unRealizedProfit": "-1.40",
        "positionInitialMargin": "64.52", "isolatedWallet": "0",
        "notional": "-322.62",
    }}]
    out = build_position_payloads(positions, leverages={"BTCUSDT": 5.0})
    assert out[0]["leverage"] == 5.0
    assert out[0]["marginUsd"] == 64.52


def test_position_leverage_derived_from_margin_when_config_absent():
    # symbolConfig unavailable → |notional| / positionInitialMargin (exact in
    # cross mode) rather than a lying 0×.
    positions = [{"info": {
        "symbol": "BTCUSDT", "positionAmt": "-0.005", "entryPrice": "64220.10",
        "markPrice": "64499.50", "positionInitialMargin": "64.52380129",
        "isolatedWallet": "0", "notional": "-322.61900644",
    }}]
    out = build_position_payloads(positions)
    assert out[0]["leverage"] == 5.0
    assert round(out[0]["marginUsd"], 2) == 64.52


def test_position_isolated_wallet_wins_as_margin():
    positions = [{"info": {
        "symbol": "SOLUSDT", "positionAmt": "1.0", "entryPrice": "74.0",
        "markPrice": "74.5", "positionInitialMargin": "14.90",
        "isolatedWallet": "20.00", "notional": "74.5",
    }}]
    out = build_position_payloads(positions)
    assert out[0]["marginUsd"] == 20.00


def test_position_margin_null_and_leverage_zero_when_unknowable():
    # No margin fields at all (defensive: never invent numbers).
    positions = [{"info": {"symbol": "ETHUSDT", "positionAmt": "1.0",
                           "entryPrice": "3000", "markPrice": "3010"}}]
    out = build_position_payloads(positions)
    assert out[0]["marginUsd"] is None
    assert out[0]["leverage"] == 0.0


def test_income_payloads_map_and_preserve_time():
    rows = [
        {"tranId": 9001, "symbol": "BTCUSDT", "incomeType": "REALIZED_PNL",
         "income": "5.0", "asset": "USDT", "time": 1717000000000},
        {"tranId": 9001, "symbol": "BTCUSDT", "incomeType": "COMMISSION",
         "income": "-0.20", "asset": "USDT", "time": 1717000000000},
        {"tranId": 9002, "symbol": "BTCUSDT", "incomeType": "FUNDING_FEE",
         "income": "-0.30", "asset": "USDT", "time": 1717003600000},
    ]
    out = build_income_payloads(rows)
    assert len(out) == 3
    assert out[0] == {"tranId": 9001, "symbol": "BTCUSDT", "incomeType": "REALIZED_PNL",
                      "incomeUsd": 5.0, "asset": "USDT", "ts": 1717000000000}
    # funding paid keeps its negative sign for the server to split
    assert out[2]["incomeUsd"] == -0.30


def test_income_payloads_skip_malformed():
    rows = [
        {"tranId": "not-an-int", "incomeType": "REALIZED_PNL", "income": "1", "time": "x"},
        {"tranId": 1, "incomeType": "REALIZED_PNL", "income": "2.0", "time": 1717000000000},
    ]
    out = build_income_payloads(rows)
    assert len(out) == 1
    assert out[0]["incomeUsd"] == 2.0


def test_bracket_map_extracts_reduce_only_sl_and_tp():
    orders = [
        {"info": {"symbol": "BTCUSDT", "type": "STOP_MARKET",
                  "reduceOnly": "true", "stopPrice": "106400"}},
        {"info": {"symbol": "BTCUSDT", "type": "TAKE_PROFIT_MARKET",
                  "reduceOnly": "true", "stopPrice": "112000"}},
        # an unfilled LIMIT entry (not reduce-only) must be ignored
        {"info": {"symbol": "BTCUSDT", "type": "LIMIT",
                  "reduceOnly": "false", "price": "108000"}},
    ]
    assert build_bracket_map(orders) == {"BTCUSDT": {"slPrice": 106400.0, "tpPrice": 112000.0}}


def test_bracket_map_close_position_flag_and_sl_only():
    # donchian legs place SL only via closePosition=true → tp stays None
    orders = [{"info": {"symbol": "BTCUSDT", "type": "STOP_MARKET",
                        "closePosition": "true", "stopPrice": "105000"}}]
    assert build_bracket_map(orders) == {"BTCUSDT": {"slPrice": 105000.0, "tpPrice": None}}


def test_bracket_map_reads_algo_order_shape():
    # Binance parks conditional orders under /fapi/v1/openAlgoOrders now:
    # orderType + triggerPrice + real booleans, wrapped as {"info": row}.
    orders = [
        {"info": {"symbol": "BTCUSDT", "orderType": "STOP_MARKET", "side": "BUY",
                  "reduceOnly": True, "closePosition": False, "triggerPrice": "65183.4"}},
        {"info": {"symbol": "BTCUSDT", "orderType": "TAKE_PROFIT_MARKET", "side": "BUY",
                  "reduceOnly": True, "closePosition": False, "triggerPrice": "62293.5"}},
    ]
    assert build_bracket_map(orders) == {
        "BTCUSDT": {"slPrice": 65183.4, "tpPrice": 62293.5}}


def test_bracket_map_side_filter_drops_orphans_from_closed_trades():
    # Live position is SHORT (reduce side BUY). Orphaned SELL brackets from an
    # old LONG keep resting on the algo endpoint — they must not overwrite the
    # live bracket's prices.
    orders = [
        {"info": {"symbol": "BTCUSDT", "orderType": "STOP_MARKET", "side": "BUY",
                  "reduceOnly": True, "triggerPrice": "65183.4"}},
        {"info": {"symbol": "BTCUSDT", "orderType": "TAKE_PROFIT_MARKET", "side": "BUY",
                  "reduceOnly": True, "triggerPrice": "62293.5"}},
        # orphan TP from a closed long (plus its 98 reprotect spam copies)
        {"info": {"symbol": "BTCUSDT", "orderType": "TAKE_PROFIT_MARKET", "side": "SELL",
                  "reduceOnly": True, "triggerPrice": "67945.6"}},
        # orphan SL from an older closed long
        {"info": {"symbol": "BTCUSDT", "orderType": "STOP_MARKET", "side": "SELL",
                  "reduceOnly": True, "triggerPrice": "63527.2"}},
    ]
    assert build_bracket_map(orders, reduce_sides={"BTCUSDT": "BUY"}) == {
        "BTCUSDT": {"slPrice": 65183.4, "tpPrice": 62293.5}}


def test_bracket_map_no_side_filter_when_side_unknown():
    # No reduce_sides entry for the symbol → behave as before (keep all).
    orders = [{"info": {"symbol": "ETHUSDT", "orderType": "STOP_MARKET", "side": "SELL",
                        "reduceOnly": True, "triggerPrice": "3000"}}]
    assert build_bracket_map(orders, reduce_sides={"BTCUSDT": "BUY"}) == {
        "ETHUSDT": {"slPrice": 3000.0, "tpPrice": None}}


def test_bracket_map_skips_non_reduce_only_and_zero_stop():
    orders = [
        {"info": {"symbol": "BTCUSDT", "type": "STOP_MARKET",
                  "reduceOnly": "false", "stopPrice": "105000"}},
        {"info": {"symbol": "ETHUSDT", "type": "STOP_MARKET",
                  "reduceOnly": "true", "stopPrice": "0"}},
    ]
    assert build_bracket_map(orders) == {}


def test_position_payloads_merge_brackets():
    positions = [{"info": {
        "symbol": "BTCUSDT", "positionAmt": "0.01", "entryPrice": "108000",
        "markPrice": "108500", "unRealizedProfit": "5", "leverage": "10",
    }}]
    brackets = {"BTCUSDT": {"slPrice": 106400.0, "tpPrice": None}}
    out = build_position_payloads(positions, brackets)
    assert out[0]["slPrice"] == 106400.0
    assert out[0]["tpPrice"] is None


def test_position_payloads_default_brackets_none():
    positions = [{"info": {
        "symbol": "BTCUSDT", "positionAmt": "0.01", "entryPrice": "108000",
        "markPrice": "108500", "unRealizedProfit": "5", "leverage": "10",
    }}]
    out = build_position_payloads(positions)
    assert out[0]["slPrice"] is None
    assert out[0]["tpPrice"] is None


# ── multi-account aggregation (2026-07-25) ───────────────────────────────────
# The relay reads every leg's sub-account and combines them. Two properties are
# load-bearing and both fail silently if broken, so they get explicit tests:
#   1. accounts are deduped by API-key fingerprint — a leg whose .env.<instance>
#      omits BINANCE_API_KEY would otherwise re-read the PREVIOUS leg's account
#      and DOUBLE-COUNT it into a summed balance;
#   2. a symbol open on two accounts keeps the first and drops the second,
#      because the server keys futures_positions by symbol alone.

import tools.consolidate_futures_push as cfp  # noqa: E402


def _mk(wallet, positions=(), income=(), brackets_known=True):
    return {
        "account": {"walletBalanceUsd": wallet, "marginBalanceUsd": wallet,
                    "unrealizedPnlUsd": 0.0, "availableBalanceUsd": wallet},
        "positions": [dict(symbol=s, positionAmt=1.0) for s in positions],
        "income": [{"tranId": i} for i in income],
        "brackets_known": brackets_known,
    }


def _patch(monkeypatch, per_instance, keys):
    """per_instance: {inst: payload}; keys: {inst: api-key fingerprint source}."""
    monkeypatch.setattr(cfp, "ACCOUNT_INSTANCES", tuple(per_instance))
    state = {"inst": None}

    def fake_load(inst):
        state["inst"] = inst
        return None if keys.get(inst) is None else object()

    def fake_fp():
        k = keys.get(state["inst"] or "v1")
        return f"fp-{k}" if k else ""

    def fake_collect(_days):
        return per_instance[state["inst"] or "v1"]

    import exchange.env as envmod
    monkeypatch.setattr(envmod, "load_env_for_instance", fake_load)
    monkeypatch.setattr(cfp, "_key_fingerprint", fake_fp)
    monkeypatch.setattr(cfp, "_collect_one_account", fake_collect)
    # v1 never calls load_env_for_instance, so seed the state for it
    state["inst"] = next(iter(per_instance))


def test_balances_are_summed_across_accounts(monkeypatch):
    _patch(monkeypatch,
           {"v1": _mk(100.0), "donchian": _mk(50.0), "sol_supertrend": _mk(60.0)},
           {"v1": "A", "donchian": "B", "sol_supertrend": "C"})
    out = cfp.collect_all_accounts(income_days=2)
    assert out["account"]["walletBalanceUsd"] == 210.0


def test_duplicate_api_key_is_skipped_not_double_counted(monkeypatch):
    # donchian resolves to the SAME key as v1 (its env file forgot the key).
    _patch(monkeypatch,
           {"v1": _mk(100.0), "donchian": _mk(100.0)},
           {"v1": "A", "donchian": "A"})
    out = cfp.collect_all_accounts(income_days=2)
    assert out["account"]["walletBalanceUsd"] == 100.0, "double-counted one account"
    assert any(a.get("skipped", "").startswith("duplicate_of") for a in out["accounts"])


def test_leg_without_env_file_is_skipped(monkeypatch):
    _patch(monkeypatch,
           {"v1": _mk(100.0), "sol_supertrend": _mk(60.0)},
           {"v1": "A", "sol_supertrend": None})
    out = cfp.collect_all_accounts(income_days=2)
    assert out["account"]["walletBalanceUsd"] == 100.0
    assert any(a.get("skipped") == "no_env_file" for a in out["accounts"])


def test_same_symbol_on_two_accounts_keeps_both_tagged(monkeypatch):
    # v1 and donchian both hold BTCUSDT; the server keys positions by
    # (account, symbol), so both rows are sent, each tagged with its leg.
    _patch(monkeypatch,
           {"v1": _mk(100.0, positions=["BTCUSDT"]),
            "donchian": _mk(50.0, positions=["BTCUSDT"]),
            "sol_supertrend": _mk(60.0, positions=["SOLUSDT"])},
           {"v1": "A", "donchian": "B", "sol_supertrend": "C"})
    out = cfp.collect_all_accounts(income_days=2)
    rows = [(p["account"], p["symbol"]) for p in out["positions"]]
    assert rows == [("v1", "BTCUSDT"), ("donchian", "BTCUSDT"),
                    ("sol_supertrend", "SOLUSDT")]
    assert out["account"]["walletBalanceUsd"] == 210.0


def test_accounts_read_excludes_failed_and_skipped_legs(monkeypatch):
    # The server scopes its delete to accounts_read, so a leg whose read failed
    # (or has no env file) must NOT be listed — its last-known row survives.
    _patch(monkeypatch,
           {"v1": _mk(100.0, positions=["BTCUSDT"]),
            "donchian": _mk(50.0, positions=["BTCUSDT"])},
           {"v1": "A", "donchian": "B", "sol_supertrend": None})
    real = cfp._collect_one_account

    def flaky(days):
        if cfp._key_fingerprint() == "fp-B":
            raise RuntimeError("binance 503")
        return real(days)
    monkeypatch.setattr(cfp, "_collect_one_account", flaky)
    out = cfp.collect_all_accounts(income_days=2)
    assert out["accounts_read"] == ["v1"]
    assert [p["account"] for p in out["positions"]] == ["v1"]


def test_income_concatenated_across_accounts(monkeypatch):
    _patch(monkeypatch,
           {"v1": _mk(100.0, income=[1, 2]), "sol_supertrend": _mk(60.0, income=[3])},
           {"v1": "A", "sol_supertrend": "C"})
    out = cfp.collect_all_accounts(income_days=2)
    assert [r["tranId"] for r in out["income"]] == [1, 2, 3]


def test_failed_account_read_marks_snapshot_incomplete(monkeypatch):
    _patch(monkeypatch,
           {"v1": _mk(100.0), "donchian": _mk(50.0)}, {"v1": "A", "donchian": "B"})
    real = cfp._collect_one_account

    def flaky(days):
        if cfp._key_fingerprint() == "fp-B":
            raise RuntimeError("binance 503")
        return real(days)
    monkeypatch.setattr(cfp, "_collect_one_account", flaky)
    assert cfp.collect_all_accounts(income_days=2)["account_complete"] is False


def test_run_skips_account_snapshot_but_pushes_positions_and_income(monkeypatch):
    monkeypatch.setattr(cfp, "_config", lambda: ("http://x", "t"))
    monkeypatch.setattr(cfp, "collect_all_accounts", lambda d: {
        "account": {"walletBalanceUsd": 100.0}, "positions": [], "income": [{"tranId": 1}],
        "brackets_known": False, "accounts": [], "accounts_read": ["v1"],
        "account_complete": False})
    posted = []
    monkeypatch.setattr(cfp, "_post", lambda u, t, path, body: posted.append(path) or {"body": {}})
    cfp.run()
    assert "/futures/account-snapshot" not in posted
    assert "/futures/positions" in posted and "/futures/income" in posted
