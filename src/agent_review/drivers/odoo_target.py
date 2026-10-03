"""Which Odoo a run profile points at, checked BEFORE any environment is created.

A native driver speaks one Odoo version's AI protocol. Pointing it at another version's source tree or template
does not degrade gracefully — Odoo 19's `/ai/generate_response` does not exist on 20, and 20's agent loop never
calls a provider from the Odoo process — so a mismatch REFUSES the run, naming what was found. Nothing here
chooses another driver or transport on the user's behalf."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from ..core.profile import ProfileError, RunProfile
from .native_ai.capabilities import odoo_revision


@dataclass
class TemplateFacts:
    base_version: str | None                      # ir_module_module.latest_version of `base`, e.g. 20.0.1.3
    ai_state: str | None                          # state of the `ai` module
    columns: dict[str, bool] = field(default_factory=dict)   # "table.column" -> present
    tables: dict[str, bool] = field(default_factory=dict)    # "table" -> present
    odoo_ai_service: bool | None = None           # an iap_service row with technical_name odoo_ai (Odoo 20)


def check_tree(profile: RunProfile, major: str, driver: str) -> tuple[str, str | None]:
    """(version, build) of target.odoo_root, refusing unless it is Odoo `major`."""
    root = profile.target.get("odoo_root")
    if not root:
        raise ProfileError(f"profile {profile.name}: target.odoo_root is not set (the Odoo source tree to run)")
    if not (Path(root) / "odoo" / "release.py").is_file():
        raise ProfileError(f"profile {profile.name}: target.odoo_root {root!r} is not an Odoo source tree "
                           "(no odoo/release.py). Set it to your own Odoo installation.")
    version, build = odoo_revision(root)
    if not version:
        raise ProfileError(f"profile {profile.name}: cannot read the Odoo version from {root}/odoo/release.py")
    if version.split(".")[0] != major:
        raise ProfileError(f"profile {profile.name}: target.odoo_root is Odoo {version}; the {driver} driver runs "
                           f"Odoo {major} only. Use the driver for that version; none is chosen automatically.")
    py = str(profile.target.get("python", "python3"))
    if os.sep in py and not Path(py).is_file():
        raise ProfileError(f"profile {profile.name}: target.python {py!r} does not exist")
    return version, build


def launcher(odoo_root: str) -> tuple[list[str], bool]:
    """(argv after the interpreter, whether the tree root must be on PYTHONPATH). A git checkout has odoo-bin;
    a packaged tree (a downloaded .tar.gz) has setup/odoo and imports `odoo` as a namespace package."""
    root = Path(odoo_root)
    if (root / "odoo-bin").is_file():
        return ["./odoo-bin"], False
    if (root / "setup" / "odoo").is_file():
        return ["setup/odoo"], True
    raise ProfileError(f"no launcher in {odoo_root}: expected odoo-bin (source checkout) or setup/odoo (packaged tree)")


def template_facts(template_dsn: str, columns: list[str], tables: list[str]) -> TemplateFacts:
    try:
        with psycopg.connect(template_dsn) as c:
            def one(q, *a):
                row = c.execute(q, a).fetchone()
                return row[0] if row else None
            facts = TemplateFacts(one("select latest_version from ir_module_module where name = 'base'"),
                                  one("select state from ir_module_module where name = 'ai'"))
            for tc in columns:
                t, col = tc.split(".")
                facts.columns[tc] = bool(one("select 1 from information_schema.columns where table_name = %s "
                                             "and column_name = %s", t, col))
            for t in tables:
                facts.tables[t] = bool(one("select 1 from information_schema.tables where table_name = %s", t))
            if facts.tables.get("iap_service"):
                facts.odoo_ai_service = bool(one("select 1 from iap_service where technical_name = 'odoo_ai'"))
            return facts
    except psycopg.Error as e:
        raise ProfileError(f"cannot read the fixture template ({type(e).__name__}): check fixture.templates and "
                           "that this role may connect to it") from e


def check_template(facts: TemplateFacts, major: str, driver: str, template: str,
                   required_columns: list[str], absent_columns: list[str] = ()) -> None:
    if not facts.base_version or not facts.base_version.startswith(f"{major}."):
        raise ProfileError(f"template {template} holds an Odoo {facts.base_version or '?'} database (base module); "
                           f"the {driver} driver needs an Odoo {major} template")
    if facts.ai_state != "installed":
        raise ProfileError(f"template {template}: the `ai` module is {facts.ai_state or 'absent'}, not installed")
    missing = [c for c in required_columns if not facts.columns.get(c)]
    present = [c for c in absent_columns if facts.columns.get(c)]
    if missing or present:
        raise ProfileError(f"template {template} does not have the Odoo {major} AI schema the {driver} driver drives"
                           + (f"; missing {missing}" if missing else "") + (f"; unexpected {present}" if present else ""))
