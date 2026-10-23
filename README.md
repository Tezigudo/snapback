# snapback-btc

Deterministic Binance Futures BTC/USDT perpetual bot. RSI(2) extreme + EMA(200) trend filter + relative-volume confirmation + funding-rate edge. Testnet-first, with Claude Code as the human-loop monitor cockpit.

> ⚠️ **Research project.** Not financial advice. May lose money. Testnet only until §6 of the [ultraplan](../btc-bot-ultraplan.md) checklist passes.

## Stack
- Python 3.11+
- `ccxt` (Binance Futures REST + WS)
- `pandas`, `pandas-ta` (indicators)
- `backtesting.py` (research), `freqtrade` (live, later)
- `plotly` (HTML reports)
- SQLite (state), SMTP email (alerts — Gmail app password works)

## Quickstart (testnet)
```bash
git clone https://github.com/Tezigudo/snapback
cd snapback
cp .env.example .env             # fill your keys
sed -i.bak 's/^BINANCE_ENV=.*/BINANCE_ENV=testnet/' .env   # the template defaults to mainnet
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -c "from exchange.env import get_env; print(get_env())"   # -> 'testnet'
```

### Per-leg setup (required on a fresh clone)
A clone ships only `config/*.example.yaml` templates (every value `null`); the
bot refuses to boot until each leg you run has a filled file. For each leg
(`v1` → `params.yaml`, `donchian` → `params_donchian.yaml`,
`sol_supertrend` → `params_sol_supertrend.yaml`):
```bash
cp config/<name>.example.yaml config/<name>.yaml   # e.g. params_donchian
$EDITOR config/<name>.yaml                          # replace EVERY null — meanings in config/README.md
# Sub-account legs only: .env.<instance> overrides the base .env, so put ONLY that
# leg's keys in it — never copy .env.example (it sets BINANCE_ENV=mainnet).
cp .env.<instance>.example .env.<instance> 2>/dev/null \
  || printf 'BINANCE_API_KEY=\nBINANCE_API_SECRET=\n' > .env.<instance>
chmod 600 .env.<instance> && $EDITOR .env.<instance>
python -m bot --instance <instance> --check-config  # must print CONFIG OK
```
Do not start or enable any unit until `--check-config` prints `CONFIG OK` for that leg.

## Layout
| Path | Purpose |
|---|---|
| `bot.py` | Daemon entrypoint |
| `strategy/` | Deterministic signal + execution logic |
| `risk.py` | **Hard ceilings — do not edit** |
| `exchange/` | Binance client + env/lockfile gate |
| `monitor.py` | Cron health checker (email alerts via `alerts.py`) |
| `backtest.py` | `backtesting.py` harness |
| `research/` | Walk-forward + OOS + pluggable researcher seam (zero API cost by default) |
| `report.py` | Plotly HTML generator |
| `.claude/commands/` | Slash commands for the Claude Code cockpit |
| `deploy/` | systemd unit for VPS |

## Safety model
1. Defaults to **testnet** via `BINANCE_ENV`
2. Mainnet requires `confirm_mainnet.lock` file to exist
3. Hard ceilings (`risk.RiskCeilings`) come from each leg's untracked params
   `risk:` block — no code defaults; missing/null/out-of-range refuses boot
   (exit 78). Symbol allowlist stays in `risk.py`. Edits to `risk:` are NOT
   guarded by the pre-commit hook — treat them as a risk review.
4. `data/HALT` file = bot closes all and exits clean
5. Git pre-commit hook rejects `risk.py` edits without `RISK_REVIEW=1`

## Status
- [x] P0 — scaffold
- [x] P1 — data + backtester (honest fees/slippage/funding)
- [x] P2 — strategy v1 (RSI(2) + EMA(200) + volume + funding)
- [x] P3 — walk-forward + OOS (deterministic researcher; LLM seam ready)
- [x] P3.2 — strategy bake-off: snapback v1/v2 retired, Donchian + carry promising
- [x] P3.3 — v2 refinements + ensemble: carry-v2 leads (2/3 promotion checks pass)
- [x] P3.4 — leverage + TF sweeps (leverage set to 20x permanent)
- [x] P3.5 — **promotion gate was broken**; rebuilt with tail-risk checks; carry-v4 + trend gate + tail-aware selection lifted CAGR -18.6% → +4.95% (still at boundary, NOT P4-ready)
- [x] P3.6 — **OOS validation on 2025 killed both candidates** (carry-v4: IS +75% → OOS −5%; donchian-v2: IS +213% → OOS −5%); funding-momentum hypothesis also dead (−66% IS). Carry refactor cuts LOC -72%.
- [x] P3.7 — multi-window OOS reveals **Donchian-v2 4h has regime-dependent edge**: same strategy → +49% OOS on trending 2022 H1, −5.5% OOS on choppy 2025 H1. Not "no edge" — needs a regime classifier to deploy.
- [x] **P4 — Donchian-v3 with directional regime gate PASSES OOS on both windows** ✅ (2020-21 IS → 2022 H1 OOS: +30%; 2022-24 IS → 2025 H1 OOS: +5.6%). First strategy to clear OOS validation. Live deploy blocked only on execution plumbing now.
- [ ] P5 — testnet 7-day soak (execution plumbing: ccxt orders, state DB, monitor)
- [ ] P6 — mainnet gate
- [ ] P5 — monitor + viz
- [ ] P6 — mainnet gate

## Strategy zoo (P3.4 — see `STRATEGY_NOTES.md` for full table)

All ran on the same 2022-06 → 2024-12 walk-forward with fees + slippage + funding.

| Strategy | TF | Stability | Sharpe | Drift | Promotion |
|---|---|---:|---:|---:|:---:|
| snapback-v1 / v2 | 15m | 19-21% | −3.58 / −0.22 | huge | ❌ retired |
| donchian-v1 | 15m | 53% | +0.40 | +58% | ❌ |
| donchian-v2 | 15m | 59% | +0.47 | +64% | ❌ |
| donchian-v2 | 4h | 55% | +0.67 | +70% | ❌ drift only |
| carry-v1 | 15m | 50% | −0.13 | +2% | ❌ Sharpe |
| carry-v2 (P3.3) | 15m | 58% | +0.96 | +54% | ❌ drift only |
| **carry-v2 (P3.4 phaseC)** | **15m** | **60%** | **+0.70** | **+40%** | **✅ PASS at 3x** |
| ensemble(d2 + c2) | 15m | 48% | −0.30 | n/a | ❌ falsified |

P3.4 leverage ablation: phase C pass at 20x and at 3x produced *identical*
numbers (Sharpe +0.70, stability 60%, drift +40%). The pass came from
`test_days=30 + min_trades=6`, NOT from leverage.

**P3.5 correction:** the "PASS" was a broken-gate artefact. Replaying
the 28 fold returns serially produces $100 → $62.30 over 2.3 years —
**CAGR −18.6%/year**. The 3-check median-based gate didn't see three
tail folds (−47%, −25%, −21%) wipe out 25 small winners. Rebuilt the
gate with strict mean/CAGR/max-loss checks. Architectural fixes —
trend EMA gate (carry-v4), tail-aware combo selection — lifted IS
CAGR to **+4.95%** on the same fold set, but live deployment was
blocked on out-of-sample validation against 2025 data.

**P3.6 OOS validation killed both candidates:**
- carry-v4 best IS combo: +74.66% IS → **−5.37% OOS** (2025 Jan-May)
- donchian-v2 best IS combo: +213.47% IS → **−5.50% OOS**
- funding-momentum (inverse hypothesis): −66.56% IS (failed before OOS)

The infrastructure (`research/oos_validate.py`) is the right tool;
the conclusion is that no deterministic strategy in the snapback-btc
family generalises beyond its IS window. P4 testnet is blocked. See
`STRATEGY_NOTES.md` for the five concrete next directions
(different asset, higher timeframe, orderbook data, cross-exchange
basis trade, or accept the negative result).

**Per-user-decision:** leverage permanently set to 20x in
`config/params.yaml` and `risk.py: MAX_LEVERAGE`. Ablation showed
leverage doesn't change carry-v2/v4 returns (cap doesn't bind for the
winning combos) but live tail-risk at 20x is real (flash crash past SL
= liquidation).

## Research

P3 walk-forward sweeps a param grid (`config/sweep.yaml`) across rolling
train/test windows and writes JSON/MD/HTML reports under `reports/`.

```bash
python -m research.walk_forward \
    --sweep config/sweep.yaml \
    --start 2025-01-01 --end 2025-12-31
```

The researcher that comments on results is pluggable via
`research/agents/`. Default is `DeterministicResearcher` — pure stats,
**zero API cost, no network**. See `research/agents/AGENT_ROLES.md` for
the role taxonomy borrowed from TradingAgents / AgentQuant and the list
of opt-in LLM-backed researchers that can be added later.
