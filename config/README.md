# config/

## Per-leg params (`params*.yaml`) — untracked

The live strategy parameters are **not in git**. Each leg reads one file:

| instance (`--instance`) | file the bot reads                         | template to copy                               |
|-------------------------|--------------------------------------------|------------------------------------------------|
| `v1`                    | `config/params.yaml`                       | `config/params.example.yaml`                   |
| `donchian`              | `config/params_donchian.yaml`              | `config/params_donchian.example.yaml`          |
| `sol_supertrend`        | `config/params_sol_supertrend.yaml`        | `config/params_sol_supertrend.example.yaml`    |
| `cnh_short`             | `config/params_cnh_hybrid_short.yaml`      | `config/params_cnh_hybrid_short.example.yaml`  |
| `cnh_short_sol`         | `config/params_cnh_hybrid_short_sol.yaml`  | `config/params_cnh_hybrid_short_sol.example.yaml` |

`config/params*.yaml` is gitignored; `config/params*.example.yaml` is re-admitted.

To create a real file:

```bash
cp config/params_donchian.example.yaml config/params_donchian.yaml
$EDITOR config/params_donchian.yaml          # replace EVERY null with a real value
python -m bot --instance donchian --check-config   # exit 0 = OK, 78 = refused
```

The templates keep the full key structure; **every value is `null`** (except
`strategy_name`, which selects the schema), so an unfilled copy is fail-closed.
**The bot refuses to boot** (exit 78, one alert if SMTP is configured) while any
key it reads is missing or null — see `REQUIRED_KEYS` in `config_check.py`.
There are deliberately no code defaults to fall back on. Also refused:

- a non-bool in a boolean key (`"true"`, `1`) — write `true` / `false`;
- out-of-range safety values: `deploy.kill_switch_equity_fraction` in (0, 1),
  `sizing.leverage` integer >= 1, `sizing.risk_per_trade_pct` > 0,
  `execution.poll_interval_s` > 0, `breakeven.at_r` > 0, `breakeven.buffer_pct` >= 0.

Every leg also needs a `risk:` block — the hard ceilings that used to be
constants in `risk.py` (`max_notional_usd`, `max_leverage`, `max_daily_loss_pct`,
`max_open_positions`, `max_orders_per_minute`, `max_consecutive_losses`,
`min_time_between_trades_s`; see `risk.RiskCeilings` for which are enforced).
Range-checked in `config_check.RANGE_CHECKS`. An existing live file gets it from
`tools/migrate_risk_params.py` (values read from the pre-change `risk.py` in git;
dry run by default, `--apply` writes a gitignored `.bak` first). Also refused:
`sizing.leverage` above `risk.max_leverage`, and `risk.max_notional_usd` below
the symbol's **static fallback** minimum order (`exchange/constraints.py`). That
check cannot see the live exchange minimum, which boot enforces separately: a
cap between the two passes CONFIG OK but every signal is then skipped. Edits to
`risk:` are not guarded by the RISK_REVIEW hook; instead each boot compares the
ceilings with the last acknowledged boot's (state DB meta) and alerts listing
the changed keys — stored only once the alert is delivered, so an undelivered
alert repeats on the next boot.

Safety blocks are **required** on the legs that run them: v1 (`multifactor-v1`)
needs `reprotect` (`enabled`, `observe_only`, `max_replaces_per_position`) and
`boot_resume` (`enabled`, `observe_only`); sol (`supertrend`) needs `breakeven`
(`enabled`, `at_r`, `buffer_pct`). Elsewhere they are optional (absent = off),
as is `no_stop` everywhere. If `breakeven.enabled: true`, its `at_r` and
`buffer_pct` are required on any leg.

Back the real files up out of band (they are not recoverable from git once
untracked). On the droplet, a deploy that untracks them must copy them aside
first — see `tools/deploy_untrack_params.sh`.

## Sweep configs (`sweep*.yaml`)

Research sweep grids for `backtest.py`; still tracked.
