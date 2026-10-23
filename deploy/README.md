# VPS deploy notes (P6)

Minimal Hetzner CCX13 (€4.50/mo) or DigitalOcean droplet ($6/mo). Static IP.

## One-time setup
```bash
sudo useradd -r -s /usr/sbin/nologin -d /opt/snapback-btc snapback
sudo mkdir -p /opt/snapback-btc /var/log/snapback-btc
sudo chown -R snapback:snapback /opt/snapback-btc /var/log/snapback-btc

git clone https://github.com/Tezigudo/snapback /opt/snapback-btc
cd /opt/snapback-btc
python3.11 -m venv .venv
.venv/bin/pip install -e .

# The bot loads <repo>/.env itself (exchange/env.py); units set no EnvironmentFile.
sudo -u snapback cp .env.example /opt/snapback-btc/.env
sudo chmod 600 /opt/snapback-btc/.env
sudo -u snapback nano /opt/snapback-btc/.env   # fill mainnet keys, BINANCE_ENV=mainnet

sudo touch /opt/snapback-btc/confirm_mainnet.lock
sudo chown snapback:snapback /opt/snapback-btc/confirm_mainnet.lock

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
*/5 * * * * /opt/snapback-btc/.venv/bin/python /opt/snapback-btc/monitor.py
```

Email alerts: configure SMTP_* env vars in `/opt/snapback-btc/.env` (the repo-root `.env`) (see
`.env.example`).

⚠ DigitalOcean blocks ports 25/465/587 on droplets (since 2025-03-06), so
Gmail SMTP no longer works from DO. Use a transactional provider on port
2525 — MailerSend free tier (3k/mo) is the default in `.env.example`.
Mailgun, SendGrid, Brevo, and Postmark also expose 2525.

Test the wiring before going live: `python alerts.py "deploy test" "smoke"`.
