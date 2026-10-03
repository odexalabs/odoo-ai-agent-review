"""Credentials come from ONE NAMED SOURCE. No ambient fallback — never OPENAI_API_KEY, never
~/.odoorc, never an application config — and the subprocess environment is scrubbed so the
application cannot find one either. An early prototype billed an unnamed key because nothing said
WHICH key a run may use; this module makes that structurally impossible.

Before the first paid call of a session the harness prints the key source, the last 4
characters and the estimated planned spend, then requires an explicit go.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_SOURCE = "AGENT_REVIEW_PROVIDER_KEY"
# Every ambient variable an Odoo process or a client library would otherwise pick up.
AMBIENT_KEYS = (
    "OPENAI_API_KEY", "ODOO_AI_CHATGPT_TOKEN", "ODOO_AI_GEMINI_TOKEN", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
)
# Ambient CONFIGURATION Odoo 19 reads from the environment with precedence over its -c file
# (odoo/tools/config.py: ChainMap runtime > cli > env > file). PGUSER would silently replace the
# execution role — and with it the pg_stat_statements scope — so these never reach the subprocess;
# the run profile's fixture block is the only source of database settings.
AMBIENT_CONFIG = (
    "ODOO_RC", "OPENERP_SERVER", "ODOO_DEV", "PGDATABASE", "PGUSER", "PGPASSWORD", "PGHOST", "PGPORT", "PGSSLMODE",
    "PGAPPNAME", "PGPATH", "PGDATABASE_TEMPLATE", "PGHOST_REPLICA", "PGPORT_REPLICA", "PGSERVICE", "PGPASSFILE",
    "PGOPTIONS", "PGREQUIRESSL", "PGSSLROOTCERT", "PGSSLCERT", "PGSSLKEY",
)
# The list above cannot be complete, so the namespaces are removed whole. Odoo 19 and 20 GENERATE an
# environment name, `ODOO_<OPTION>`, for every config-file option that declares none (Community
# `odoo/tools/config.py`, where the options are declared): ODOO_SMTP_SERVER, ODOO_MAX_CRON_THREADS,
# ODOO_ADDONS_PATH, … each outranks the -c file. The fixed list missed all of them, so an ambient
# ODOO_SMTP_SERVER would have replaced the unroutable mail fallback the driver writes (found
# in a verification pass). `PG*` is libpq's namespace as well as Odoo's.
AMBIENT_CONFIG_PREFIXES = ("ODOO_", "OPENERP_", "PG")


class CredentialError(RuntimeError):
    pass


@dataclass(frozen=True)
class Credential:
    source: str      # "env:AGENT_REVIEW_PROVIDER_KEY" or "file:<path>"
    last4: str
    value: str

    def record(self) -> dict:
        return {"key_source": self.source, "key_last4": self.last4}


def load_credential(key_file: str | None = None) -> Credential:
    """Exactly one source: the run profile's `key_file`, or the one named environment variable."""
    if key_file:
        p = Path(key_file).expanduser()
        if not p.is_file():
            raise CredentialError(f"key_file {p} does not exist. No ambient fallback is attempted.")
        val = p.read_text().strip()
        if not val:
            raise CredentialError(f"key_file {p} is empty")
        return Credential(f"file:{p}", val[-4:], val)
    val = os.environ.get(ENV_SOURCE, "").strip()
    if not val:
        raise CredentialError(
            f"STOP: credential source {ENV_SOURCE} is empty and no key_file is set in the run profile. "
            f"Refusing to fall back to an ambient key ({', '.join(AMBIENT_KEYS[:5])}, ... are ignored by design). "
            f"Set {ENV_SOURCE} to the key this work bills to."
        )
    return Credential(f"env:{ENV_SOURCE}", val[-4:], val)


def scrubbed_environment(base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    for k in AMBIENT_KEYS + AMBIENT_CONFIG:
        env.pop(k, None)
    for k in [k for k in env if k.startswith(AMBIENT_CONFIG_PREFIXES)]:
        env.pop(k)
    env.pop(ENV_SOURCE, None)   # the subprocess gets the key only under the name the driver chooses
    return env


def spend_plan_line(cred: Credential, planned_runs: int, est_per_run: float | None, est_source: str,
                    provider: str, model: str) -> str:
    est = f"${planned_runs * est_per_run:.3f} ({est_source})" if est_per_run is not None else f"unknown ({est_source})"
    return (f"PLAN — key source: {cred.source} (…{cred.last4}) | provider: {provider} | model: {model} | "
            f"planned runs: {planned_runs} | estimated spend: {est}")
