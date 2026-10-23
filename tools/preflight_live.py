"""Pre-flight checks before going live with real money.

Runs through every dependency of the bot's first successful trade:
  1. .env loads, BINANCE_ENV is set
  2. SMTP credentials are wired (sends a test email)
  3. ccxt market loads, exchange constraints are readable
  4. Authenticated endpoints work (fetch balance, fetch position)
  5. Sizing math at current price + current equity passes minimums
  6. risk.py ceilings won't reject typical orders

Does NOT place any orders. Pure read-only verification.

USAGE:
  uv run python -m tools.preflight_live
  uv run python -m tools.preflight_live --send-test-email
"""

from __future__ import annotations

import argparse

# Put the repo root FIRST on sys.path. Without it, `python tools/preflight_live.py`
# resolves `bot` / `exchange` / `risk` through the venv's editable-install .pth
# entry, which can point at a DIFFERENT checkout — so the pre-flight would
# validate that checkout's risk ceilings and constraints instead of this one's.
# Every other tool in tools/ already does this; preflight was the odd one out.
import pathlib as _pathlib  # noqa: E402
import sys

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import risk
from alerts import is_configured as alerts_configured
from alerts import send_alert
from bot import INSTANCE_PROFILES, compute_qty, load_params
from exchange.binance_client import BinanceClient
from exchange.constraints import (
    DEFAULT_CONSTRAINTS,
    fallbacks_for_symbol,
    merge_with_live,
    passes_minimums,
    round_qty_down,
)
from exchange.env import (
    get_api_credentials,
    get_env,
    halt_source,
    load_env_for_instance,
)
from risk import (
    RiskBreach,
    check_leverage,
    check_notional,
    check_symbol,
)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--send-test-email", action="store_true",
                    help="Actually send the SMTP test email (default just checks config).")
    ap.add_argument("--instance", default="v1", choices=list(INSTANCE_PROFILES),
                    help="Which leg to validate: loads its .env.<instance> overlay + "
                         "config and checks its per-leg data/HALT_<instance>. Default v1.")
    args = ap.parse_args()
    instance = args.instance

    failures: list[str] = []
    warnings: list[str] = []

    # 1. env + lockfile
    section("Environment")
    # Overlay this leg's sub-account keys BEFORE any authenticated call, so we
    # validate the SAME account the leg will trade (not v1's base .env). This is
    # also fail-loud: a sub-account leg missing its .env.<instance> aborts here.
    try:
        instance_env = load_env_for_instance(instance)
        if instance_env is not None:
            ok(f"loaded per-instance env overlay: {instance_env.name}")
        else:
            ok(f"instance {instance!r} runs on the base .env (no overlay needed)")
    except Exception as e:
        fail(f"per-instance env load failed: {e}")
        failures.append("instance env")
        return _summarize(failures, warnings)

    try:
        env = get_env()
        ok(f"BINANCE_ENV={env}")
    except Exception as e:
        fail(f"env load failed: {e}")
        failures.append("env")
        return _summarize(failures, warnings)

    if env == "mainnet":
        ok("mainnet lockfile present (required by env.py).")
        warn("MAINNET means real money. Continue only if intended.")
    else:
        warn(f"env={env}. Binance Futures testnet/sandbox private endpoints are deprecated "
             f"in ccxt — only public market data works. Live orders against this env will FAIL.")
        warnings.append("env=testnet (private endpoints will fail)")

    try:
        get_api_credentials()
        ok("API key + secret are set in .env.")
    except Exception as e:
        fail(f"credentials missing: {e}")
        failures.append("creds")
        return _summarize(failures, warnings)

    # Per-leg HALT: the leg is halted by the GLOBAL data/HALT (stops every leg)
    # OR its own data/HALT_<instance> (self-halt). Check the instance's view.
    halt_by = halt_source(instance)
    if halt_by == "global":
        warn("GLOBAL data/HALT exists — stops ALL legs. Bot would flatten + "
             "refuse to enter. Remove data/HALT before live.")
        warnings.append("global HALT present")
    elif halt_by is not None:
        warn(f"self-halt data/HALT_{instance} exists — this leg would flatten + "
             f"refuse to enter. Remove data/HALT_{instance} before live.")
        warnings.append(f"HALT_{instance} present")
    else:
        ok(f"No data/HALT or data/HALT_{instance}. Bot will run.")

    # 2. SMTP
    section("Alerts (SMTP)")
    if not alerts_configured():
        warn("SMTP not configured. Bot will run but you will NOT get email notifications.")
        warnings.append("SMTP not configured")
    else:
        ok("SMTP env vars all set.")
        if args.send_test_email:
            sent = send_alert("preflight test", "If you see this, SMTP wiring is working.",
                              tag="snapback-preflight")
            (ok if sent else fail)(f"Test email send: {'OK' if sent else 'FAILED'}")
            if not sent:
                failures.append("smtp send")

    # 3. ccxt connectivity + constraints
    section("Exchange connectivity")
    try:
        client = BinanceClient.from_env()
        ok(f"ccxt client constructed: {client}")
    except Exception as e:
        fail(f"client construction: {e}")
        failures.append("ccxt client")
        return _summarize(failures, warnings)

    params = load_params(INSTANCE_PROFILES[instance]["config"])
    symbol = params["symbol"]
    try:  # hard ceilings come from the params `risk:` block (no code defaults)
        risk.configure(params)
    except Exception as e:
        fail(f"risk ceilings: {e} (run tools/migrate_risk_params.py)")
        failures.append("risk ceilings")
        return _summarize(failures, warnings)
    CEILINGS = risk.ceilings()

    try:
        m = client.ex.market(symbol)
        constraints = merge_with_live(fallbacks_for_symbol(symbol), m)
        ok(f"market loaded for {symbol}")
        ok(f"min_qty={constraints.min_qty_base} BTC, "
           f"min_notional=${constraints.min_notional_usdt}, "
           f"qty_step={constraints.qty_step}, price_step={constraints.price_step}")
    except Exception as e:
        fail(f"market load: {e}")
        failures.append("market load")
        return _summarize(failures, warnings)

    # 4. private endpoints (balance + position)
    section("Authenticated endpoints")
    try:
        equity = client.fetch_equity_usdt()
        ok(f"equity (futures USDT): ${equity:.2f}")
    except Exception as e:
        fail(f"fetch_balance: {e}")
        failures.append("fetch_balance")
        equity = 0.0

    try:
        pos = client.fetch_position(symbol)
        ok(f"position: side={pos.side}, qty={pos.qty}, entry={pos.entry_price}")
        if pos.side != "flat":
            warn(f"There is an open {pos.side} position. Bot would flatten on boot in live mode.")
            warnings.append("open position at boot")
    except Exception as e:
        fail(f"fetch_position: {e}")
        failures.append("fetch_position")

    # 5. sizing simulation
    section("Sizing simulation (current price, current equity)")
    if equity <= 0:
        warn("Skipping sizing sim — equity is 0 (likely auth failure above).")
    else:
        try:
            tk = client.ex.fetch_ticker(symbol)
            price = float(tk.get("last") or tk.get("close") or 0.0)
            if price <= 0:
                fail("could not fetch live price")
                failures.append("price fetch")
            else:
                ok(f"current price: ${price:,.2f}")
                sl_pct = float(params["strategy"]["sl_pct"])
                risk_pct = float(params["sizing"]["risk_per_trade_pct"])
                leverage = int(params["sizing"]["leverage"])
                raw_qty = compute_qty(equity, price, sl_pct, risk_pct, leverage)
                qty = round_qty_down(raw_qty, constraints.qty_step)
                notional = qty * price
                ok(f"raw target qty = {raw_qty:.6f} BTC → rounded = {qty:.4f} BTC, "
                   f"notional ${notional:.2f}")
                ok2, reason = passes_minimums(qty, price, constraints)
                if ok2:
                    ok("sizing passes exchange minimums.")
                else:
                    fail(f"sizing FAILS exchange minimum: {reason}.")
                    fail(f"  -> with equity ${equity:.2f}, no entry would be placed. "
                         f"Add more capital (try ${constraints.min_notional_usdt * 1.3:.0f}+) "
                         f"OR shrink sl_pct.")
                    failures.append("sizing below exchange minimum")
        except Exception as e:
            fail(f"sizing sim error: {e}")
            failures.append("sizing sim")

    # 6. risk.py ceilings
    section("risk ceilings (params risk: block)")
    try:
        check_symbol(symbol)
        ok(f"symbol {symbol} is allowlisted")
    except RiskBreach as e:
        fail(str(e))
        failures.append("symbol allowlist")
    try:
        check_leverage(int(params["sizing"]["leverage"]))
        ok(f"leverage {params['sizing']['leverage']}x ≤ MAX_LEVERAGE={CEILINGS.MAX_LEVERAGE}x")
    except RiskBreach as e:
        fail(str(e))
        failures.append("leverage")
    try:
        check_notional(CEILINGS.MAX_NOTIONAL_USD)
        ok(f"MAX_NOTIONAL_USD={CEILINGS.MAX_NOTIONAL_USD} (will reject orders above)")
    except RiskBreach as e:
        warn(str(e))
    print(f"  MAX_DAILY_LOSS_PCT: {CEILINGS.MAX_DAILY_LOSS_PCT}%")
    print(f"  MAX_OPEN_POSITIONS: {CEILINGS.MAX_OPEN_POSITIONS}")
    print(f"  MAX_CONSECUTIVE_LOSSES: {CEILINGS.MAX_CONSECUTIVE_LOSSES} (UNENFORCED, informational only)")

    return _summarize(failures, warnings)


def _summarize(failures: list[str], warnings: list[str]) -> int:
    print()
    print("=" * 60)
    if failures:
        print(f"{RED}PREFLIGHT FAILED ({len(failures)} issue(s)){RESET}")
        for f in failures:
            print(f"  - {f}")
        print("\nDo NOT start the bot until these are fixed.")
        return 1
    if warnings:
        print(f"{YELLOW}PREFLIGHT PASSED WITH WARNINGS ({len(warnings)}){RESET}")
        for w in warnings:
            print(f"  - {w}")
        print("\nReview warnings before going live.")
        return 0
    print(f"{GREEN}PREFLIGHT PASSED — bot is safe to start.{RESET}")
    print()
    print("Next step:")
    print("  uv run python -m bot --dry-run    # observe 15-60 min, no orders placed")
    print("  uv run python -m bot              # live (real orders)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
