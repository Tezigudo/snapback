"""Worker for test_neutral_defaults_parity: run research code from a given tree.

usage: python _neutral_parity_worker.py <tree_root> <yaml_dir> <data_dir>

Imports backtest/strategy from <tree_root> (the pinned old tree or this one),
builds each leg's resolved params from the YAMLs in <yaml_dir>, then runs the
research backtest classes on a fixed synthetic OHLCV frame and prints JSON.
Deterministic; no network, no exchange access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

root, yaml_dir, data_dir = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(root))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from backtesting import Backtest  # noqa: E402

from datetime import datetime  # noqa: E402

import backtest as bt_mod  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402
from strategy.signals_donchian import DonchianBreakoutBTCv3, attach_donchian  # noqa: E402
from strategy.signals_multifactor import DayTradeMultiFactorBTC  # noqa: E402


def ohlcv(n: int, freq: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=n // 80 + 1), 80)[:n]
    rets = rng.normal(0.0, 0.004, n) + regime * 0.0009
    close = 30000.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.002, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.lognormal(3.0, 0.5, n) * np.where(rng.random(n) < 0.15, 5.0, 1.0)
    idx = pd.date_range("2022-01-01", periods=n, freq=freq)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


def trades_of(stats) -> list:
    t = stats._trades
    return [[int(r.EntryBar), int(r.ExitBar), float(r.Size),
             float(r.EntryPrice), float(r.ExitPrice)] for r in t.itertuples()]


def run(df: pd.DataFrame, cls, params, post: dict | None = None) -> dict:
    bt_mod._apply_params_to_class(cls, params)
    cls.leverage = params.leverage
    for k, v in (post or {}).items():
        setattr(cls, k, v)
    stats = Backtest(df, cls, cash=1_000_000.0, commission=0.0005,
                     margin=1.0 / params.leverage, trade_on_close=False,
                     exclusive_orders=True, finalize_trades=True).run()
    return {"trades": trades_of(stats), "equity_final": float(stats["Equity Final [$]"])}


def resolved(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    p = StrategyParams.from_yaml(str(path))
    keys = list(raw["strategy"]) + list(raw["sizing"])
    return {k: getattr(p, k) for k in keys if hasattr(p, k)}


def cases_via_run_backtest(d15, d4, yaml_dir: Path, pq: Path) -> dict:
    """run_backtest / prepare_strategy_data / _prepare_*_data paths on stubbed loaders."""
    one_h = d15.resample("1h").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"})
    frames = {"15m": d15, "1h": one_h, "4h": d4}
    fund_idx = pd.date_range(d15.index[0], d15.index[-1], freq="8h")
    sign = np.where((np.arange(len(fund_idx)) // 9) % 2 == 0, -1.0, 1.0)
    fund = pd.DataFrame({"funding_rate": sign * 0.0006}, index=fund_idx)

    def load_klines(symbol=None, timeframe="15m", days_back=0, end=None, **_):
        f = frames[timeframe].drop(columns=["Funding"], errors="ignore").copy()
        f.columns = [c.lower() for c in f.columns]
        return f

    def load_funding(*_a, **_k):
        return fund.copy()

    bt_mod.load_klines, bt_mod.load_funding = load_klines, load_funding
    res = {}
    v1p = StrategyParams.from_yaml(str(yaml_dir / "params.yaml"))
    dp = StrategyParams.from_yaml(str(yaml_dir / "params_donchian.yaml"))
    bt_mod.DayTradeMultiFactorBTC.mtf_4h_parquet_path = str(pq)
    s15, e15 = datetime(2022, 3, 1), datetime(2022, 11, 20)
    for name in ("snapback-v1", "multifactor-v1", "multifactor-v2-loose",
                 "multifactor-v3", "v3-all-wider-4"):
        r = bt_mod.run_backtest(name, "BTC/USDT:USDT", "15m", s15, e15,
                                params_override=v1p, quiet=True, return_trades=True)
        t = r["trades_df"]
        res[name] = {"n": len(t), "ret": r["backtest_return_pct"],
                     "trades": [[int(x.EntryBar), int(x.ExitBar), float(x.Size)]
                                for x in t.itertuples()]}
    r = bt_mod.run_backtest("donchian-v3", "BTC/USDT:USDT", "4h", datetime(2022, 6, 1),
                            datetime(2024, 6, 1), params_override=dp, quiet=True,
                            return_trades=True)
    t = r["trades_df"]
    res["donchian-v3"] = {"n": len(t), "ret": r["backtest_return_pct"],
                          "trades": [[int(x.EntryBar), int(x.ExitBar), float(x.Size)]
                                     for x in t.itertuples()]}
    return res


out: dict = {}
for name in ("params.yaml", "params_donchian.yaml", "params_sol_supertrend.yaml"):
    out[f"resolved:{name}"] = resolved(yaml_dir / name)

# --- v1 research class on a 15m frame + a synthetic higher-TF parquet --------
d15 = ohlcv(30000, "15min", seed=7)
d15["Funding"] = 0.0001
h4 = d15.resample("4h").agg({"Open": "first", "High": "max", "Low": "min",
                             "Close": "last", "Volume": "sum"})
h4.columns = [c.lower() for c in h4.columns]
pq = data_dir / "synthetic_4h.parquet"
h4.to_parquet(pq)
v1 = StrategyParams.from_yaml(str(yaml_dir / "params.yaml"))
DayTradeMultiFactorBTC.mtf_4h_parquet_path = str(pq)
out["v1"] = run(d15, DayTradeMultiFactorBTC, v1)
# Same class with the higher-TF gate forced ON after param injection (exercises
# the _ema_4h branch whatever the YAML says); synthetic gate period; own subclass
# so nothing leaks into later runs.
gate_cls = type("GateOn", (DayTradeMultiFactorBTC,), {})
out["v1_gate_on"] = run(d15, gate_cls, v1,
                        post={"use_mtf_4h_gate": True, "mtf_4h_ema_period": 37})

# --- donchian-v3 research class on a 4h frame -------------------------------
d4 = ohlcv(6000, "4h", seed=11)
dp = StrategyParams.from_yaml(str(yaml_dir / "params_donchian.yaml"))
prep = attach_donchian(d4, d4, period_entry=dp.donchian_period_entry,
                       period_exit=dp.donchian_period_exit, atr_period=dp.atr_period)
out["donchian"] = run(prep, DonchianBreakoutBTCv3, dp)

out["run_backtest"] = cases_via_run_backtest(d15, ohlcv(6000, "4h", seed=11), yaml_dir, pq)
print(json.dumps(out))
