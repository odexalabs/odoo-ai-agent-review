"""Egress: an ALLOWLIST, not a blanket block. The agent must reach its own model provider.
Full egress isolation may be impossible in an arbitrary partner environment; never silently
pretend it exists. Every run records egress_isolation: verified | partial | unavailable.

Mail blocking is MANDATORY and directly testable: a rollback cannot un-send an email."""
from __future__ import annotations

from dataclasses import dataclass, field

import psycopg


class MailNotBlocked(RuntimeError):
    """Mail blocking is MANDATORY: a rollback cannot un-send an email. Raised in every mode."""


@dataclass
class EgressStatus:
    isolation: str                       # verified | partial | unavailable
    allowed_endpoints: list[str] = field(default_factory=list)
    mail_servers: int | None = None      # ACTIVE ir_mail_server rows after neutralisation
    mail_servers_disabled: int = 0       # rows this harness archived on the disposable clone
    mail_sent_before: int | None = None
    mail_blocked_verified: bool = False
    outbound_integrations: dict[str, int] = field(default_factory=dict)   # evidence only; not blocked by v1
    notes: list[str] = field(default_factory=list)


def neutralise_mail(execution_dsn: str) -> int:
    """Archive every outgoing mail server on the DISPOSABLE clone, before the baseline snapshot, so
    the run cannot send mail whatever the fixture was configured with. Returns the rows archived."""
    with psycopg.connect(execution_dsn, autocommit=True) as c:
        try:
            return c.execute("update ir_mail_server set active = false where active").rowcount
        except psycopg.errors.UndefinedTable:
            return 0


def verify_mail_blocked(observer_dsn: str, cron_threads_zero: bool, smtp_fallback_disabled: bool) -> EgressStatus:
    """Three conditions, all required: no active ir_mail_server row (Odoo's per-database servers),
    the config-file SMTP fallback disabled by the driver (odoo-bin `--smtp` defaults to localhost),
    and no cron thread (the mail queue never runs). Synchronous sends then raise 'Missing SMTP
    Server' and the row stays in state `exception`."""
    st = EgressStatus(isolation="unavailable")
    try:
        with psycopg.connect(observer_dsn) as c:
            st.mail_servers = c.execute("select count(*) from ir_mail_server where active").fetchone()[0]
            st.mail_sent_before = c.execute("select count(*) from mail_mail where state = 'sent'").fetchone()[0]
            # Informational counts, per version: a probe that cannot read records "could not
            # determine" and must never abort the mandatory mail verification above (found on Odoo 20,
            # where payment_provider has `active` instead of 19's `state`).
            for label, queries in (("iap_accounts", ("select count(*) from iap_account",)),
                                   ("payment_providers_enabled", ("select count(*) from payment_provider where state <> 'disabled'",
                                                                  "select count(*) from payment_provider where active")),
                                   ("fetchmail_servers_active", ("select count(*) from fetchmail_server where active",))):
                for q in queries:
                    try:
                        st.outbound_integrations[label] = c.execute(q).fetchone()[0]
                        break
                    except psycopg.errors.UndefinedTable:
                        c.rollback()
                        break
                    except psycopg.errors.UndefinedColumn:
                        c.rollback()
                        st.outbound_integrations[label] = None
                if st.outbound_integrations.get(label, 0) is None:
                    st.notes.append(f"{label}: could not determine on this schema")
    except psycopg.Error as e:
        st.notes.append(f"could not inspect mail tables: {e}")
        return st
    if st.mail_servers == 0 and cron_threads_zero and smtp_fallback_disabled:
        st.mail_blocked_verified = True
        st.notes.append("0 active ir_mail_server rows, config SMTP fallback disabled and max_cron_threads = 0: "
                        "nothing can send mail, synchronously or from the queue")
    else:
        st.notes.append(f"MAIL NOT BLOCKED: active ir_mail_server rows = {st.mail_servers}, "
                        f"cron_threads_zero = {cron_threads_zero}, smtp_fallback_disabled = {smtp_fallback_disabled}")
    if any(v for v in st.outbound_integrations.values()):
        st.notes.append("outbound integrations present in the fixture and NOT blocked by v1 (no network jail): "
                        + ", ".join(f"{k}={v}" for k, v in st.outbound_integrations.items() if v))
    return st


def mail_sent_after(observer_dsn: str) -> int | None:
    try:
        with psycopg.connect(observer_dsn) as c:
            return c.execute("select count(*) from mail_mail where state = 'sent'").fetchone()[0]
    except psycopg.Error:
        return None
