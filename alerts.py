"""
Out-of-band alerting via SMTP email.

Used by monitor.py (cron) and bot.py (fill/halt events) to notify the operator
without paying for any service.

⚠ DigitalOcean droplets block ports 25/465/587 since 2025-03-06. Use a
transactional provider that exposes port 2525 (MailerSend, Mailgun,
SendGrid, Brevo). Gmail SMTP no longer works from DO droplets.

Config (in .env):
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
    ALERT_EMAIL_FROM, ALERT_EMAIL_TO

If any required setting is missing, `send_alert` returns False and logs a
warning instead of raising — alerting failure must NEVER crash the bot.
"""

from __future__ import annotations

import html as _html
import logging
import os
import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from string import Template

# Import order matters: exchange.env calls load_dotenv() at import time, which
# populates os.environ from .env before we read SMTP_* below.
from exchange import env as _env_module  # noqa: F401  (side-effect import)

log = logging.getLogger(__name__)

SMTP_TIMEOUT_S = 15


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def is_configured() -> bool:
    """True if all required SMTP env vars are populated."""
    return all(
        _env(k)
        for k in (
            "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
            "ALERT_EMAIL_FROM", "ALERT_EMAIL_TO",
        )
    )


def _default_tag() -> str:
    """Subject prefix for alerts. Each bot instance overrides via ALERT_TAG env
    so the Donchian leg's emails thread separately from v1's in Gmail."""
    return (os.environ.get("ALERT_TAG") or "snapback-btc").strip()


def _instance_tag() -> str:
    """Per-leg identifier for the subject. bot.main() sets SNAPBACK_INSTANCE
    right after argparse; tools that send alerts outside the bot (watchdog,
    preflight) can set it explicitly to route to the right thread."""
    return (os.environ.get("SNAPBACK_INSTANCE") or "").strip()


# ── HTML email theme: "Japan sky" — SOFT pastel-blue gradient header with dark
# text, airy light body. Emails carry BOTH a plain-text part (the fallback /
# accessibility copy) and this HTML alternative; Gmail renders the HTML.
# color-scheme:light + a tiny <style> ask Gmail/Apple Mail NOT to dark-ify the
# card (dark mode was inverting the white card to heavy navy); the media query
# re-asserts light bg + dark text together for clients that honour it. Severity
# is inferred from the subject to colour the accent + badge.

# Each tuple: (accent, emoji, badge_text, badge_bg, badge_fg) — soft/light tones
_SEV_ERROR = ("#ef9a9a", "⚠️", "ACTION NEEDED", "#fdeceb", "#c0392b")
_SEV_GOOD = ("#9cc8a6", "✅", "RESOLVED", "#eaf6ee", "#2e7d54")
_SEV_INFO = ("#9fd0f0", "\U0001f514", "INFO", "#e8f3fc", "#1f77c2")

# Error words win over good words so "re-place FAILED" reads as an error.
_ERROR_WORDS = ("fail", "error", "halt", "kill", "unprotected", "breach",
                "cannot", "danger", "urgent", "not protected", "stale", "abort")
_GOOD_WORDS = ("re-placed", "restored", "recovered", "resumed", "started",
               "success", "sent", "resolved", "healthy", "back online", "test")

_HTML_TEMPLATE = Template(
    '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    '<meta name="color-scheme" content="light">'
    '<meta name="supported-color-schemes" content="light">'
    '<style>:root{color-scheme:light;supported-color-schemes:light;}'
    '@media (prefers-color-scheme:dark){'
    '.sb-body{background:#eaf4fb!important;}'
    '.sb-card{background:#ffffff!important;}'
    '.sb-panel{background:#f4faff!important;}'
    '.sb-btext{color:#26404f!important;}'
    '.sb-foot,.sb-foot *{color:#7791a6!important;}'
    '}</style></head>'
    '<body class="sb-body" style="margin:0;padding:0;background-color:#eaf4fb;">'
    '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
    'class="sb-body" style="background-color:#eaf4fb;padding:24px 12px;"><tr><td align="center">'
    '<table role="presentation" width="600" cellpadding="0" cellspacing="0" '
    'class="sb-card" style="width:600px;max-width:600px;background-color:#ffffff;border-radius:16px;'
    'overflow:hidden;border:1px solid #dcecf7;'
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;\">"
    # bright sky-blue header + WHITE text — white survives Gmail dark-mode
    # inversion (a light header with dark text goes invisible; see v2). Lighter
    # & brighter than the original navy. Solid bgcolor fallback for no-gradient.
    '<tr><td style="background-color:#3f97d6;'
    'background-image:linear-gradient(135deg,#3a93d4 0%,#57a8de 52%,#86c6ee 100%);'
    'padding:28px 32px 24px 32px;">'
    '<div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;'
    'color:rgba(255,255,255,0.92);font-weight:700;">$kicker</div>'
    '<div style="font-size:22px;line-height:1.35;color:#ffffff;font-weight:700;'
    'margin-top:9px;">$emoji&nbsp; $subject</div></td></tr>'
    # severity badge
    '<tr><td style="padding:22px 32px 0 32px;">'
    '<span style="display:inline-block;background-color:$badge_bg;color:$badge_fg;'
    'font-size:12px;font-weight:700;letter-spacing:.5px;padding:6px 13px;'
    'border-radius:999px;">$badge</span></td></tr>'
    # body panel (monospace, preserves line breaks, accent left-border)
    '<tr><td style="padding:16px 32px 8px 32px;">'
    '<div class="sb-panel" style="background-color:#f4faff;border:1px solid #dcecf7;'
    'border-left:4px solid $accent;border-radius:10px;padding:16px 18px;">'
    '<div class="sb-btext" style="'
    "font-family:'SF Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:14px;"
    'line-height:1.65;color:#26404f;white-space:pre-wrap;word-break:break-word;">'
    '$body</div></div></td></tr>'
    # footer
    '<tr><td style="padding:18px 32px 28px 32px;">'
    '<div class="sb-foot" style="border-top:1px solid #ecf4fa;padding-top:16px;font-size:12px;'
    'line-height:1.7;color:#7791a6;">'
    '<span style="color:#5aa8d4;">⛩</span> Automated alert from '
    '<strong style="color:#4a6a86;">snapback-btc</strong> · do not reply<br>'
    '$stamp</div></td></tr></table>'
    '<div style="font-size:11px;color:#9fb4c6;margin-top:14px;'
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;\">"
    '☁️ snapback-btc trading bot</div>'
    '</td></tr></table></body></html>'
)


def _severity(subject: str) -> tuple[str, str, str, str, str]:
    """(accent, emoji, badge, badge_bg, badge_fg) inferred from the subject."""
    s = subject.lower()
    if any(w in s for w in _ERROR_WORDS):
        return _SEV_ERROR
    if any(w in s for w in _GOOD_WORDS):
        return _SEV_GOOD
    return _SEV_INFO


def _render_html(subject: str, body: str, tag: str, instance: str) -> str:
    """Render the alert as the Japan-sky HTML email. Pure/​side-effect-free."""
    accent, emoji, badge, badge_bg, badge_fg = _severity(subject)
    kicker = f"{tag} · {instance}".upper() if instance else tag.upper()
    now = datetime.now(timezone.utc)
    ict = now.astimezone(timezone(timedelta(hours=7)))
    stamp = f"{now:%Y-%m-%d %H:%M} UTC · {ict:%H:%M} ICT"
    return _HTML_TEMPLATE.safe_substitute(
        kicker=_html.escape(kicker),
        subject=_html.escape(subject),
        emoji=emoji,
        badge=badge, badge_bg=badge_bg, badge_fg=badge_fg,
        accent=accent,
        body=_html.escape(body).replace("\n", "<br>"),
        stamp=_html.escape(stamp),
    )


def send_alert(subject: str, body: str, *, tag: str | None = None) -> bool:
    """
    Send an email alert. Returns True on success, False on any failure.

    Never raises — the bot must keep running even if alerting is broken.
    """
    if not is_configured():
        log.warning("alerts: SMTP not configured; skipping send (subject=%r)", subject)
        return False

    effective_tag = tag if tag is not None else _default_tag()
    instance = _instance_tag()
    # Subject form: [{ALERT_TAG}] [{instance}] {subject}. When SNAPBACK_INSTANCE
    # is unset (legacy callers, ad-hoc CLI usage), the instance bracket is
    # omitted entirely so we don't ship "[ ] ..." with empty brackets.
    prefix = f"[{effective_tag}]" if not instance else f"[{effective_tag}] [{instance}]"
    msg = EmailMessage()
    msg["Subject"] = f"{prefix} {subject}"
    msg["From"] = _env("ALERT_EMAIL_FROM")
    msg["To"] = _env("ALERT_EMAIL_TO")
    # Plain-text part first = the fallback (accessibility / non-HTML clients),
    # then the Japan-sky HTML alternative Gmail will render. Rendering must never
    # break sending — if the template ever fails, fall back to text-only.
    msg.set_content(body)
    try:
        msg.add_alternative(
            _render_html(subject, body, effective_tag, instance), subtype="html")
    except Exception:  # noqa: BLE001 — cosmetic; a broken template must not drop the alert
        log.exception("alerts: HTML render failed; sending plain-text only")

    host = _env("SMTP_HOST")
    # Parse the port defensively: is_configured() only checks SMTP_PORT is
    # non-empty, so a non-numeric value would make int() raise ValueError —
    # which the send try/except below does NOT catch, breaking the "never
    # raises" contract and crashing cron callers (the watchers).
    try:
        port = int(_env("SMTP_PORT", "587"))
    except ValueError:
        log.warning("alerts: SMTP_PORT=%r is not an integer; skipping send", _env("SMTP_PORT"))
        return False
    user = _env("SMTP_USER")
    password = _env("SMTP_PASSWORD")

    try:
        ctx = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=SMTP_TIMEOUT_S, context=ctx) as s:
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT_S) as s:
                s.ehlo()
                s.starttls(context=ctx)
                s.ehlo()
                s.login(user, password)
                s.send_message(msg)
        log.info("alerts: sent %r to %s", subject, msg["To"])
        return True
    except (smtplib.SMTPException, OSError) as e:
        log.warning("alerts: failed to send %r: %s", subject, e)
        return False


def _main() -> int:
    """CLI test: `python alerts.py "subject" "body"` — useful to verify .env wiring."""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    subject = sys.argv[1] if len(sys.argv) > 1 else "test alert from snapback-btc"
    body = sys.argv[2] if len(sys.argv) > 2 else (
        "If you're reading this, SMTP is wired up correctly.\n"
        "Sent from alerts.py CLI."
    )
    ok = send_alert(subject, body, tag="snapback-btc-test")
    print("ok" if ok else "FAILED — check SMTP_* env vars and Gmail app password")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
