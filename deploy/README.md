# VPS deploy notes (P6)

Minimal Hetzner CCX13 (€4.50/mo) or DigitalOcean droplet ($6/mo). Static IP.

## One-time setup
```bash
# The units in deploy/ run as root from /root/snapback-btc (User=root,
# WorkingDirectory/ExecStart/ReadWritePaths). Clone exactly there, or edit
# every path in the unit files before installing them.
git clone https://github.com/Tezigudo/snapback /root/snapback-btc
cd /root/snapback-btc
python3.11 -m venv .venv
.venv/bin/pip install -e .
mkdir -p data logs reports              # the units' ReadWritePaths

# The bot loads <repo>/.env itself (exchange/env.py); units set no EnvironmentFile.
cp .env.example /root/snapback-btc/.env
chmod 600 /root/snapback-btc/.env
nano /root/snapback-btc/.env             # fill mainnet keys, BINANCE_ENV=mainnet

touch /root/snapback-btc/confirm_mainnet.lock

# Per-leg params: a clone has only config/*.example.yaml (all null). For each leg:
#   cp config/<name>.example.yaml config/<name>.yaml && $EDITOR config/<name>.yaml
#   (fill EVERY null — see config/README.md). Sub-account legs: .env.<instance>
#   holds ONLY that leg's BINANCE_API_KEY/SECRET (from .env.<instance>.example
#   where one exists — never a copy of .env.example; it overrides the base .env). Then:
.venv/bin/python -m bot --instance v1 --check-config   # must print CONFIG OK
# Do not enable/start a unit until its leg prints CONFIG OK.

sudo cp deploy/snapback-btc.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable snapback-btc

# DO NOT start until promote-mainnet checklist passes:
# sudo systemctl start snapback-btc
```

## Binance API key restrictions (MANDATORY)
- Enable Futures only — disable spot, margin, withdrawals
- IP whitelist: the VPS static IP only
- Read + Trade permissions only — never Withdraw

## Cron monitor (sends email alerts)
```cron
*/5 * * * * /root/snapback-btc/.venv/bin/python /root/snapback-btc/monitor.py
```

Email alerts: configure SMTP_* env vars in `/root/snapback-btc/.env` (the repo-root `.env`) (see
`.env.example`).

⚠ DigitalOcean blocks ports 25/465/587 on droplets (since 2025-03-06), so
Gmail SMTP no longer works from DO. Use a transactional provider on port
2525 — MailerSend free tier (3k/mo) is the default in `.env.example`.
Mailgun, SendGrid, Brevo, and Postmark also expose 2525.

Test the wiring before going live: `python alerts.py "deploy test" "smoke"`.
