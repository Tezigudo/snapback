"""
Backtest harness with realistic friction modeling.

Phase 1 added a buy-and-hold benchmark with fees + slippage + funding.
Phase 2 adds the real snapback-v1 strategy (RSI + EMA + volume + funding
confluence, multi-timeframe).

CLI:
    # benchmark
    python backtest.py --strategy buy-and-hold --tf 1h --days 30

    # strategy v1
    python backtest.py --strategy snapback-v1 --tf 15m --start 2024-01-01 --end 2025-01-01

Honest reporting: naive B&H / after fees+slip / after funding shown
separately, so friction is never hidden in a headline number.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import FractionalBacktest

from exchange.data import load_funding, load_klines
from strategy.signals import (
    FRACTIONAL_UNIT,
    SnapbackBTC,
    StrategyParams,
    prepare_strategy_data,
    require_params,
)

# 2026-05-17: cleaned up to only the deployable strategy + benchmark.
# Old strategies (carry, donchian, fmom, snapback-v2, multifactor v2/mtf)
# removed from this repo after PATH2_RESULTS.html locked multifactor-v1
# as the production choice. Recover from git history if needed.
from strategy.signals_multifactor import DayTradeMultiFactorBTC
# Research-only variants (v2 / v3 ablations / v1 tuned). Optional: a checkout
# without them (the public snapshot) still imports backtest.py; selecting one of
# their names raises a clear error naming the missing module (run_backtest).
# Where the module exists it is imported and registered exactly as before.
# A module whose own import fails ONLY because another optional research module
# is absent (v3 imports v2) is marked unavailable too, naming that module; any
# other import error inside a present module still raises.
_OPTIONAL_RESEARCH_STRATEGIES: dict[str, dict[str, str]] = {
    "strategy.signals_multifactor_v2": {
        "multifactor-v2-loose": "DayTradeMultiFactorBTCv2Loose",
        "multifactor-v2-strict": "DayTradeMultiFactorBTCv2Strict",
    },
    "strategy.signals_multifactor_v3": {
        "multifactor-v3": "DayTradeMultiFactorBTCv3",
        "v3-dist-ema-only": "V3DistEmaOnly",
        "v3-vol-regime-only": "V3VolRegimeOnly",
        "v3-atr-stops-only": "V3AtrStopsOnly",
        "v3-all": "V3All",
        "v3-all-k-2.5": "V3AllK_2_5",
        "v3-all-wider-2": "V3AllWider2",
        "v3-all-k-3.5": "V3AllK_3_5",
        "v3-all-wider-3": "V3AllWider3",
        "v3-all-wider-4": "V3AllWider4",
        "v3-all-k-5.0": "V3AllK_5_0",
        "v3-all-k-6.0": "V3AllK_6_0",
    },
    "strategy.signals_multifactor_tuned": {
        # multifactor-v1 trend-exit ablation variants
        "v1-debounce-2": "V1Debounce2",
        "v1-debounce-3": "V1Debounce3",
        "v1-debounce-4": "V1Debounce4",
        "v1-floor-0.5": "V1Floor005",
        "v1-floor-1.0": "V1Floor010",
        "v1-deluxe": "V1Deluxe",
    },
}
# strategy name -> class for the optional modules present; name -> module for
# the ones absent from this checkout.
_OPTIONAL_LOADED: dict[str, type[Strategy]] = {}
UNAVAILABLE_STRATEGIES: dict[str, str] = {}
for _mod_name, _entries in _OPTIONAL_RESEARCH_STRATEGIES.items():
    if importlib.util.find_spec(_mod_name) is None:
        UNAVAILABLE_STRATEGIES.update(dict.fromkeys(_entries, _mod_name))
        continue
    try:
        _mod = importlib.import_module(_mod_name)
    except ModuleNotFoundError as _e:
        if _e.name not in _OPTIONAL_RESEARCH_STRATEGIES:
            raise
        UNAVAILABLE_STRATEGIES.update(dict.fromkeys(_entries, _e.name))
        continue
    for _name, _cls_name in _entries.items():
        _OPTIONAL_LOADED[_name] = getattr(_mod, _cls_name)

# Resurrected 2026-05-23 for v1+Donchian parallel-deploy backtest.
from strategy.signals_donchian import (  # noqa: E402  (after the optional-module loader, as before)
    DonchianBreakoutBTCv3, DonchianRiderV1,
    attach_donchian, attach_rider,
)

# multifactor-v1/v2/v3 use a single entry TF (15m); no second-TF prep required.
# Donchian-v3 and rider-v1 are single-TF (entry_tf == channel TF, typically 4h).
_TF_AGNOSTIC_STRATEGIES = {
    "multifactor-v1",
    "multifactor-v2-loose", "multifactor-v2-strict",
    "multifactor-v3", "v3-dist-ema-only", "v3-vol-regime-only",
    "v3-atr-stops-only", "v3-all",
    "donchian-v3",
    "rider-v1",
}

# For snapback, use plain Backtest with large notional cash so 1 BTC fits as
# an integer unit. Returns are scale-invariant so headline metrics are
# unchanged vs running with $10k.
SNAPBACK_DEFAULT_CASH = 1_000_000.0

log = logging.getLogger(__name__)

# --- Friction model ----------------------------------------------------------
# Binance Futures taker fee = 0.04% per side. We bundle a 1bp slippage proxy
# into the same `commission` parameter since backtesting.py has no native
# slippage knob. Total per-side: 0.05%.
TAKER_FEE = 0.0004
SLIPPAGE_PROXY = 0.0001
COMMISSION_PER_SIDE = TAKER_FEE + SLIPPAGE_PROXY


# --- Strategies --------------------------------------------------------------
class BuyAndHold(Strategy):
    """Sanity benchmark — buy on first bar, hold forever."""

    def init(self) -> None:
        self._opened = False

    def next(self) -> None:
        if not self._opened:
            # size=1.0 fails the broker margin check after commission; 0.999
            # leaves a 0.1% cash buffer that's effectively zero capital drag.
            self.buy(size=0.999)
            self._opened = True


STRATEGIES: dict[str, type[Strategy]] = {
    "buy-and-hold": BuyAndHold,
    "snapback-v1": SnapbackBTC,
    "multifactor-v1": DayTradeMultiFactorBTC,
    # v2 / v3 variants / v1 tuned, in their historical order; empty for any
    # module this checkout omits (see _OPTIONAL_RESEARCH_STRATEGIES).
    **_OPTIONAL_LOADED,
    # Resurrected 2026-05-23.
    "donchian-v3": DonchianBreakoutBTCv3,
    # 2026-05-30: 4h native trend-rider (long-only, fixed TP bracket).
    "rider-v1": DonchianRiderV1,
}

# No strategy in the current codebase needs regime columns.
_REGIME_STRATEGIES: set[str] = set()
# Donchian-v3 needs DonchianUpper/Lower/ExitUpper/ExitLower + ATR_1h columns
# attached to the entry-TF frame by attach_donchian().
_DONCHIAN_STRATEGIES: set[str] = {"donchian-v3"}
# rider-v1 needs native-4h RiderDonHi/RiderEma/RiderAtr attached by attach_rider().
_RIDER_STRATEGIES: set[str] = {"rider-v1"}


# --- Funding accounting ------------------------------------------------------
def funding_cost_for_long_btc(
    data: pd.DataFrame,
    funding: pd.DataFrame,
    initial_cash: float,
    commission: float,
) -> tuple[float, int]:
    """Sum funding payments for a long buy-and-hold position over `data`'s span."""
    if funding.empty or data.empty:
        return 0.0, 0

    first_open = float(data["Open"].iloc[0])
    btc_position = initial_cash * (1.0 - commission) / first_open

    span = funding.loc[
        (funding.index >= data.index[0]) & (funding.index <= data.index[-1])
    ]
    if span.empty:
        return 0.0, 0

    prices = data["Close"].reindex(span.index, method="ffill")
    notional = btc_position * prices
    paid = (notional * span["funding_rate"]).sum()
    return float(paid), len(span)


def funding_cost_for_trades(
    trades: pd.DataFrame, data: pd.DataFrame, funding: pd.DataFrame
) -> tuple[float, int]:
    """Sum funding payments across per-trade open intervals.

    backtesting.py 0.6 Size is signed: + for long, - for short. Notional with
    sign × funding rate gives the correct sign for cost (long pays positive
    funding, short receives positive funding).
    """
    if trades is None or trades.empty or funding is None or funding.empty:
        return 0.0, 0

    total = 0.0
    events = 0
    closes = data["Close"]

    for _, t in trades.iterrows():
        entry_time = t["EntryTime"] if "EntryTime" in t else t.get("EntryTime")
        exit_time = t["ExitTime"] if "ExitTime" in t else t.get("ExitTime")
        size = float(t["Size"])  # signed
        if entry_time is None or exit_time is None or pd.isna(entry_time) or pd.isna(exit_time):
            continue
        span = funding.loc[(funding.index >= entry_time) & (funding.index <= exit_time)]
        if span.empty:
            continue
        prices = closes.reindex(span.index, method="ffill")
        notional = size * prices  # carries sign
        total += float((notional * span["funding_rate"]).sum())
        events += len(span)

    return total, events


# --- Runner ------------------------------------------------------------------
def _prepare_buy_and_hold_data(
    symbol: str, timeframe: str, start: datetime, end: datetime
) -> pd.DataFrame:
    days_back = max((end - start).days + 2, 2)
    raw = load_klines(symbol=symbol, timeframe=timeframe, days_back=days_back, end=end)
    df = raw.loc[start:end].copy()
    if df.empty:
        raise RuntimeError(
            f"No klines for {symbol} {timeframe} in {start.date()} → {end.date()}"
        )
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def _prepare_snapback_data(
    symbol: str,
    start: datetime,
    end: datetime,
    params: StrategyParams,
    with_regimes: bool = False,
    with_donchian: bool = False,
    donchian_entry: int | None = None,
    donchian_exit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull 15m + 1h + funding; build indicator-augmented 15m DataFrame.

    Returns (prepared_15m_df, funding_in_span_df). Adds a warm-up buffer so
    indicators are valid by the start of the visible window.
    """
    require_params(params, ("ema_period",), "backtest")
    # Warm-up needed: ema_period 1h bars to start the EMA = ema_period hours.
    # Add 50% slack.
    warmup = timedelta(hours=int(params.ema_period * 1.5))
    pull_start = start - warmup
    days_back = max((end - pull_start).days + 2, 2)

    k15 = load_klines(symbol=symbol, timeframe="15m", days_back=days_back, end=end)
    k1h = load_klines(symbol=symbol, timeframe="1h", days_back=days_back, end=end)
    fund = load_funding(symbol=symbol, days_back=days_back, end=end)

    if k15.empty or k1h.empty:
        raise RuntimeError("Missing 15m or 1h klines for the requested window.")

    prepared = prepare_strategy_data(k15, k1h, fund, params)
    # NOTE: old regime/Donchian attach steps removed with the archived strategies.
    _ = with_regimes  # parameter kept for backward-compat callers
    _ = with_donchian
    _ = donchian_entry
    _ = donchian_exit
    # prepare_strategy_data strips tz; slice bounds + funding must match.
    naive_start = start.replace(tzinfo=None) if start.tzinfo else start
    naive_end = end.replace(tzinfo=None) if end.tzinfo else end
    visible = prepared.loc[naive_start:naive_end]
    if visible.empty:
        raise RuntimeError(f"No bars after warm-up in {start} → {end}")

    if fund.index.tz is not None:
        fund = fund.copy()
        fund.index = fund.index.tz_convert("UTC").tz_localize(None)
    fund_visible = fund.loc[visible.index[0]:visible.index[-1]]

    return visible, fund_visible


def _prepare_tf_agnostic_data(
    symbol: str,
    entry_tf: str,
    start: datetime,
    end: datetime,
    with_donchian: bool = False,
    donchian_entry: int | None = None,
    donchian_exit: int | None = None,
    atr_period: int | None = None,
    with_rider: bool = False,
    rider_donchian_n: int = 55,
    rider_atr_period: int | None = None,
    rider_ema_period: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull klines at entry_tf + funding; attach Donchian channels (computed
    on the SAME entry_tf — single-TF Donchian, not multi-TF) if requested.
    For rider-v1 (with_rider=True), attaches native-4h RiderDonHi/RiderEma/RiderAtr.

    Returns (prepared_df, funding_in_span). Used for carry + donchian + rider
    backtests when entry_tf != 15m, where snapback's 15m+1h prep doesn't
    apply.
    """
    # Warm-up buffer: enough bars for the longest indicator window.
    # rider-v1 needs its long EMA (binding constraint = ema bars * 3 slack).
    # donchian_entry bars at entry_tf is the standard constraint.
    if with_rider:
        # rider-v1: compute indicators on exactly the visible window (no pre-roll).
        # This matches the standalone validator (rider_port_validate) which slices
        # first then computes EMA/DonHi/ATR on the slice.  Pre-rolling would warm
        # the EMA earlier than the standalone does, adding spurious early-window
        # trades and inflating trade counts / WR.  NaN bars at the head are skipped
        # by the isfinite guard in DonchianRiderV1.next().
        warmup_bars = 0
    else:
        if donchian_entry is None or atr_period is None:
            raise ValueError(
                "backtest: strategy.atr_period (and strategy.donchian_period_entry "
                "for donchian strategies) must be set in the params YAML — no code "
                "default (see config/README.md).")
        warmup_bars = max(donchian_entry, atr_period) * 3
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
                  "1h": 60, "2h": 120, "4h": 240, "1d": 1440}.get(entry_tf, 15)
    warmup = timedelta(minutes=warmup_bars * tf_minutes)
    pull_start = start - warmup
    days_back = max((end - pull_start).days + 2, 2)

    klines = load_klines(symbol=symbol, timeframe=entry_tf, days_back=days_back, end=end)
    fund = load_funding(symbol=symbol, days_back=days_back, end=end)
    if klines.empty:
        raise RuntimeError(f"Missing {entry_tf} klines for the requested window.")

    # 2026-05-23: re-enabled for donchian-v3 resurrection.
    klines.columns = [c.capitalize() for c in klines.columns]
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)
    prepared = klines
    if with_donchian:
        prepared = attach_donchian(
            prepared, klines,
            period_entry=donchian_entry,
            period_exit=donchian_exit,
            atr_period=atr_period,
        )
    if with_rider:
        prepared = attach_rider(
            prepared,
            donchian_n=rider_donchian_n,
            atr_period=rider_atr_period,
            ema_period=rider_ema_period,
        )

    naive_start = start.replace(tzinfo=None) if start.tzinfo else start
    naive_end = end.replace(tzinfo=None) if end.tzinfo else end
    visible = prepared.loc[naive_start:naive_end]
    if visible.empty:
        raise RuntimeError(f"No bars after warm-up in {start} → {end}")

    if fund.index.tz is not None:
        fund = fund.copy()
        fund.index = fund.index.tz_convert("UTC").tz_localize(None)
    fund_visible = fund.loc[visible.index[0]:visible.index[-1]]
    return visible, fund_visible


_INJECT_FIELDS = (
    "rsi_long_threshold", "rsi_short_threshold",
    "volume_multiple", "funding_long_max", "funding_short_min",
    "atr_tp_multiple", "atr_sl_multiple",
    "time_stop_bars", "risk_per_trade_pct",
    "donchian_period_entry", "donchian_period_exit", "atr_trail_multiple",
    "funding_threshold", "funding_exit_threshold", "sl_pct",
    "max_24h_change_pct",
    # carry-v3
    "atr_percentile_threshold", "dd_halt_pct",
    # carry-v4
    "trend_ema_period",
    # fmom-v1
    "momentum_lookback_bars", "momentum_threshold", "tp_pct", "require_trend_align",
    # donchian-v3 regime gate
    "regime_ema_period", "regime_slope_window", "slope_trend_threshold_pct",
    # multifactor-v1
    "mf_trend_ema_period", "macd_fast", "macd_slow", "macd_signal",
    "require_candlestick", "require_macd", "require_trend",
    "require_funding_not_extreme", "funding_extreme_threshold", "max_hold_bars",
    # multifactor-v2 trailing (legacy; v2 reborn as TA confirmation)
    "trail_activate_atr", "trail_atr_multiple",
    # multifactor-v2 TA confirmation (current).
    # `confirmations_required` is CLASS-LEVEL per variant (Loose=1, Strict=3,
    # v3=3). DO NOT include it here — _apply_params_to_class would otherwise
    # clobber each variant with the StrategyParams default (2), making all
    # variants identical. Same rationale as the v3 enable_* flags below.
    "swing_k", "swing_lookback_bars",
    "trendline_max_distance_pct", "sr_max_distance_pct",
    "sr_cluster_tolerance_pct", "fib_max_distance_pct",
    # multifactor-v3 — only THRESHOLDS go through param injection.
    # enable_dist_ema_filter / enable_atr_stops / enable_vol_regime_gate are
    # CLASS-LEVEL switches set by the variant subclass (V3DistEmaOnly etc),
    # so do NOT include them here — _apply_params_to_class would otherwise
    # reset them to StrategyParams defaults and break ablation testing.
    "max_distance_above_ema_pct", "max_distance_below_ema_pct",
    # atr_sl_k / atr_tp_k are CLASS-LEVEL per variant
    # (each V3* variant subclass sets its own pair).
    # Same rationale as confirmations_required and the enable_* flags:
    # if we apply StrategyParams defaults here we'd silently clobber the
    # variant's intentional choice, making ablations indistinguishable.
    "vol_regime_lookback_days", "vol_regime_max_pctile",
)
# v1-family fields that used to live only as class defaults. Not injected into
# v2/v3, which define their own rsi_period / volume_ma_period.
_V1_FIELDS = ("rsi_period", "volume_ma_period", "use_mtf_4h_gate", "mtf_4h_ema_period")
# Rider-v1 reads its indicator periods from the YAML too.
_RIDER_FIELDS = ("rider_atr_period", "rider_ema_period")

# Original class values of every injectable attribute, captured at import. The
# classes are shared across runs in one process, so each apply starts from these
# (a key missing from the next YAML must not inherit the previous run's value).
_PRISTINE: dict[type, dict[str, object]] = {
    c: {f: getattr(c, f) for f in _INJECT_FIELDS + _V1_FIELDS + _RIDER_FIELDS
        if hasattr(c, f)}
    for c in STRATEGIES.values()
}


def _apply_params_to_class(cls: type[Strategy], params: StrategyParams) -> None:
    """Inject swept params as class attributes on a Strategy subclass.

    Only sets attributes that already exist on the class — so snapback params
    don't pollute Donchian, Donchian params don't pollute carry, etc.
    Attributes are first reset to the class's original values, then only the
    values the YAML actually provides (non-None) are injected.
    """
    for f, v in _PRISTINE.get(cls, {}).items():
        setattr(cls, f, v)
    for field in _INJECT_FIELDS:
        # Only inject what the YAML actually provides: a None never overwrites a
        # class constant (v2/v3 variants keep their own unless the YAML sets it).
        if hasattr(cls, field) and getattr(params, field) is not None:
            setattr(cls, field, getattr(params, field))
    if issubclass(cls, DayTradeMultiFactorBTC):
        for field in _V1_FIELDS:
            if getattr(params, field) is not None:
                setattr(cls, field, getattr(params, field))
    if issubclass(cls, DonchianRiderV1):
        for field in _RIDER_FIELDS:
            if getattr(params, field) is not None:
                setattr(cls, field, getattr(params, field))


def run_backtest(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    cash: float = 10_000.0,
    leverage: int | None = None,
    quiet: bool = False,
    params_override: StrategyParams | None = None,
    return_equity: bool = False,
    return_trades: bool = False,
    commission: float | None = None,
) -> dict:
    """Run a backtest.

    `params_override` lets walk-forward / sweep harnesses inject specific
    StrategyParams without touching config/params.yaml. When None, params
    come from the yaml as usual. Ignored for buy-and-hold.
    """
    if strategy_name in UNAVAILABLE_STRATEGIES:
        raise ModuleNotFoundError(
            f"strategy {strategy_name!r} needs {UNAVAILABLE_STRATEGIES[strategy_name]}, "
            "which is not in this checkout", name=UNAVAILABLE_STRATEGIES[strategy_name])
    if strategy_name not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy_name}")

    # Result timeframe label (will be overwritten if strategy uses a single TF).
    _result_tf_label = timeframe

    if strategy_name == "buy-and-hold":
        data = _prepare_buy_and_hold_data(symbol, timeframe, start, end)
        funding_in_span = load_funding(
            symbol=symbol, days_back=max((end - start).days + 2, 2), end=end
        ).loc[start:end]
        if funding_in_span.index.tz is not None:
            funding_in_span = funding_in_span.copy()
            funding_in_span.index = funding_in_span.index.tz_convert("UTC").tz_localize(None)
        params = None
    else:
        params = params_override or StrategyParams.from_yaml()
        cls = STRATEGIES[strategy_name]
        # CRITICAL: read donchian periods from `params` (the source of truth for
        # this run), NOT from the class attr (which is sticky from the prior
        # backtest in the same process). The prior-bug version read class attrs
        # before _apply_params_to_class was called, so the data prep used STALE
        # channel periods while the strategy ran with NEW ones — producing
        # mismatched Donchian bands and bogus "OOS passing" results.
        donchian_entry = params.donchian_period_entry
        donchian_exit = params.donchian_period_exit
        if strategy_name in _DONCHIAN_STRATEGIES:
            require_params(params, ("donchian_period_entry", "donchian_period_exit",
                                    "atr_period"), f"backtest {strategy_name}")

        if timeframe != "15m" and strategy_name in _TF_AGNOSTIC_STRATEGIES:
            # Single-TF prep: carry / donchian / rider on arbitrary entry timeframe.
            data, funding_in_span = _prepare_tf_agnostic_data(
                symbol, timeframe, start, end,
                with_donchian=(strategy_name in _DONCHIAN_STRATEGIES),
                donchian_entry=donchian_entry,
                donchian_exit=donchian_exit,
                atr_period=params.atr_period,
                with_rider=(strategy_name in _RIDER_STRATEGIES),
                rider_donchian_n=getattr(cls, "rider_donchian_n", 55),
                rider_atr_period=params.rider_atr_period,
                rider_ema_period=params.rider_ema_period,
            )
        else:
            if timeframe != "15m":
                log.warning("strategy %s requires 15m entry; ignoring tf=%s",
                            strategy_name, timeframe)
            data, funding_in_span = _prepare_snapback_data(
                symbol, start, end, params,
                with_regimes=(strategy_name in _REGIME_STRATEGIES),
                with_donchian=(strategy_name in _DONCHIAN_STRATEGIES),
                donchian_entry=donchian_entry,
                donchian_exit=donchian_exit,
            )
        _apply_params_to_class(STRATEGIES[strategy_name], params)

    eff_leverage = leverage or (params.leverage if params else 1)
    margin = 1.0 / max(eff_leverage, 1)

    eff_commission = commission if commission is not None else COMMISSION_PER_SIDE
    if strategy_name == "buy-and-hold":
        actual_cash = cash
        bt = FractionalBacktest(
            data,
            STRATEGIES[strategy_name],
            cash=actual_cash,
            commission=eff_commission,
            margin=margin,
            trade_on_close=False,
            exclusive_orders=True,
            fractional_unit=FRACTIONAL_UNIT,
            finalize_trades=True,
        )
    else:
        # Plain Backtest with $1M cash so 1 BTC ≈ 4% of equity and integer
        # unit sizing works without FractionalBacktest's price scaling (which
        # would silently desync custom indicator columns).
        actual_cash = cash if cash != 10_000.0 else SNAPBACK_DEFAULT_CASH
        STRATEGIES[strategy_name].leverage = eff_leverage  # type: ignore[attr-defined]
        bt = Backtest(
            data,
            STRATEGIES[strategy_name],
            cash=actual_cash,
            commission=eff_commission,
            margin=margin,
            trade_on_close=False,
            exclusive_orders=True,
            finalize_trades=True,
        )
    stats = bt.run()

    naive_return_pct = (
        float(data["Close"].iloc[-1]) / float(data["Open"].iloc[0]) - 1.0
    ) * 100.0
    bt_return_pct = float(stats["Return [%]"])

    funding_cost_usdt: float | None = None
    funding_events = 0
    after_funding_pct: float | None = None
    if strategy_name == "buy-and-hold":
        funding_cost_usdt, funding_events = funding_cost_for_long_btc(
            data, funding_in_span, initial_cash=cash, commission=COMMISSION_PER_SIDE
        )
    else:
        trades_df = getattr(stats, "_trades", None)
        funding_cost_usdt, funding_events = funding_cost_for_trades(
            trades_df, data, funding_in_span
        )

    if funding_cost_usdt is not None:
        final_equity = actual_cash * (1.0 + bt_return_pct / 100.0) - funding_cost_usdt
        after_funding_pct = (final_equity / actual_cash - 1.0) * 100.0

    result = {
        "strategy": strategy_name,
        "symbol": symbol,
        "timeframe": (
            timeframe if strategy_name == "buy-and-hold"
            else (timeframe if strategy_name in _TF_AGNOSTIC_STRATEGIES and timeframe != "15m"
                  else "15m+1h")
        ),
        "start": data.index[0],
        "end": data.index[-1],
        "bars": len(data),
        "trades": int(stats["# Trades"]),
        "naive_return_pct": naive_return_pct,
        "backtest_return_pct": bt_return_pct,
        "after_funding_pct": after_funding_pct,
        "funding_cost_usdt": funding_cost_usdt,
        "funding_events": funding_events,
        "sharpe": float(stats.get("Sharpe Ratio") or 0.0),
        "max_drawdown_pct": float(stats.get("Max. Drawdown [%]") or 0.0),
        "profit_factor": _safe_pf(stats),
        "win_rate_pct": float(stats.get("Win Rate [%]") or 0.0),
        "avg_trade_pct": float(stats.get("Avg. Trade [%]") or 0.0),
        "commission_per_side": eff_commission,
        "leverage": eff_leverage,
    }

    if return_equity:
        eq = getattr(stats, "_equity_curve", None)
        if eq is not None and "Equity" in eq.columns:
            # Normalise to a returns series anchored at 1.0 so ensembling on
            # different cash bases is meaningful.
            equity = eq["Equity"].astype(float)
            result["equity_series"] = equity
            result["returns_series"] = equity / float(equity.iloc[0])
            result["actual_cash"] = actual_cash

    if return_trades:
        result["trades_df"] = getattr(stats, "_trades", None)

    if not quiet:
        _print_result(result)
    return result


def _safe_pf(stats) -> float:
    pf = stats.get("Profit Factor")
    if pf is None or pd.isna(pf):
        return 0.0
    return float(pf)


def _print_result(r: dict) -> None:
    print()
    print(f"=== {r['strategy']} | {r['symbol']} {r['timeframe']} | {r['leverage']}x ===")
    print(f"  period          : {r['start']} → {r['end']}  ({r['bars']} bars)")
    print(f"  commission/side : {r['commission_per_side']*100:.4f}%  "
          f"({TAKER_FEE*100:.4f}% fee + {SLIPPAGE_PROXY*100:.4f}% slip)")
    print(f"  trades          : {r['trades']}")
    print()
    print(f"  naive B&H       : {r['naive_return_pct']:+.2f}%   (price change, zero friction)")
    print(f"  after fees+slip : {r['backtest_return_pct']:+.2f}%   (backtesting.py)")
    if r["after_funding_pct"] is not None:
        print(f"  after funding   : {r['after_funding_pct']:+.2f}%   "
              f"(funding cost {r['funding_cost_usdt']:+.2f} USDT over {r['funding_events']} events)")
    print()
    print(f"  Sharpe          : {r['sharpe']:.2f}")
    print(f"  Max DD          : {r['max_drawdown_pct']:.2f}%")
    print(f"  Profit factor   : {r['profit_factor']:.2f}")
    print(f"  Win rate        : {r['win_rate_pct']:.1f}%")
    print(f"  Avg trade       : {r['avg_trade_pct']:+.3f}%")


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Run a backtest on cached Binance Futures data.")
    p.add_argument("--strategy", default="snapback-v1", choices=list(STRATEGIES.keys()) + list(UNAVAILABLE_STRATEGIES))
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--tf", default="15m", help="entry timeframe (ignored for snapback-v1 which is fixed 15m+1h)")
    p.add_argument("--start", help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", help="YYYY-MM-DD (UTC)")
    p.add_argument("--days", type=int, help="lookback days (overrides --start)")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--leverage", type=int, help="override params.yaml leverage")
    args = p.parse_args()

    end = (
        datetime.fromisoformat(args.end).replace(tzinfo=UTC)
        if args.end
        else datetime.now(UTC)
    )
    if args.days:
        start = end - timedelta(days=args.days)
    elif args.start:
        start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    else:
        start = end - timedelta(days=365)

    run_backtest(
        args.strategy, args.symbol, args.tf, start, end,
        cash=args.cash, leverage=args.leverage,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
