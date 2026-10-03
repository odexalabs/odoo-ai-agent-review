"""Redaction. The raw diff is a local artifact; everything that leaves the run directory goes
through here. The `always` list can be extended, not disabled: `stripe.secret_key:
sk_live_... -> sk_live_...` must never appear verbatim in a normal report.

This is STRUCTURED redaction: it knows the table and the field of every value it judges. Free text —
a reply, tool arguments, an error, a log line — has neither, and is not passed through here to be
"cleaned": the redacted report withholds it instead (`core/report.py`). A pattern that knows `sk_live_`
misses every secret it does not know."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..resources import bundled

REDACTED = "<redacted>"
DEFAULT_CONFIG = bundled("config", "redaction.yaml")

# The always-redacted CORE, in code. Configuration can only ADD to these lists; a redaction file
# that omits them, or an empty one, still applies them. `config/redaction.yaml` restates them for
# readability, not as the source of truth.
CORE_FIELD_PATTERNS = (
    "password", "passwd", "token", "secret", "api_key", "apikey", "access_key", "private_key",
    "client_secret", "refresh_token", "signature", "credential",
)
CORE_TABLES_VALUES_REDACTED = (
    "ir_config_parameter", "payment_provider", "payment_token", "payment_transaction", "ir_mail_server",
    "fetchmail_server", "iap_account", "auth_oauth_provider", "res_users_apikeys",
)


@dataclass
class RedactionRules:
    field_patterns: list[str] = field(default_factory=list)
    tables_values_redacted: list[str] = field(default_factory=list)
    config_parameter_allowlist: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None, extra: dict | None = None) -> RedactionRules:
        with open(path or DEFAULT_CONFIG) as fh:
            raw = yaml.safe_load(fh) or {}
        always = raw.get("always", {}) or {}
        ext = raw.get("extend", {}) or {}
        rules = cls(field_patterns=list(CORE_FIELD_PATTERNS), tables_values_redacted=list(CORE_TABLES_VALUES_REDACTED))
        for src in (always, ext, extra or {}):
            rules.field_patterns += [p.lower() for p in src.get("field_patterns", []) or [] if p.lower() not in rules.field_patterns]
            rules.tables_values_redacted += [t for t in src.get("tables_values_redacted", []) or [] if t not in rules.tables_values_redacted]
        # an allowlist entry cannot un-redact a secret-looking key: the core patterns win
        for key in always.get("config_parameter_allowlist", []) or []:
            if not rules.field_is_sensitive(key):
                rules.config_parameter_allowlist.append(key)
        return rules

    def field_is_sensitive(self, name: str) -> bool:
        n = name.lower()
        return any(p in n for p in self.field_patterns)

    def table_values_redacted(self, table: str) -> bool:
        return any(fnmatch.fnmatch(table, p) for p in self.tables_values_redacted)

    def redact_row(self, table: str, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        if table == "ir_config_parameter":
            key = row.get("key")
            if key in self.config_parameter_allowlist:
                return dict(row)
            return {k: (v if k in ("id", "key", "create_date", "write_date", "create_uid", "write_uid") else REDACTED)
                    for k, v in row.items()}
        if self.table_values_redacted(table):
            return {k: (v if k in ("id", "name", "create_date", "write_date") else REDACTED) for k, v in row.items()}
        return {k: (REDACTED if self.field_is_sensitive(k) and v not in (None, False, "") else v) for k, v in row.items()}

    def redact_value(self, table: str | None, field: str, value: Any) -> Any:
        """One value of `field` on `table`, as the report may show it: the `redact_row` policy applied to a
        value seen outside its row (a grade's observed or expected value). With no row, an
        ir_config_parameter value has no key to allowlist, so it is redacted; with no table at all,
        every non-empty value is (default-deny)."""
        if value is None or value is False or (isinstance(value, str) and value == ""):
            return value
        if table is None:
            return REDACTED
        return self.redact_row(table, {field: value})[field]

    def redact_changed_fields(self, table: str, changed: dict[str, list[Any]], row_after: dict | None) -> dict:
        out = {}
        for f, (b, a) in changed.items():
            if table == "ir_config_parameter":
                key = (row_after or {}).get("key")
                if key not in self.config_parameter_allowlist and f == "value":
                    out[f] = [REDACTED, REDACTED]
                    continue
            if self.table_values_redacted(table) and f not in ("name",) or self.field_is_sensitive(f):
                out[f] = [REDACTED if b not in (None, False, "") else b, REDACTED if a not in (None, False, "") else a]
            else:
                out[f] = [b, a]
        return out
