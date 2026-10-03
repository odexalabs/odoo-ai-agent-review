"""Capability discovery for native Odoo AI. A CORE FEATURE, not a precondition check: the driver
DISCOVERS what the target exposes, each scenario DECLARES what it requires, and a scenario
refuses only when its own requirement is missing. Nothing here hardcodes a required tool list.

A native agent tool is an `ir.actions.server` with `use_in_ai`, reachable by an agent through a
topic (`ai_agent_ai_topic_rel` -> `ai_topic_ir_act_server_rel`) or directly. The inventory is
read from the database; the Odoo revision from the source tree."""
from __future__ import annotations

import re
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from ..base import Capabilities, ToolInfo

# xml_id -> (capability key, kind). Unknown tools get a slug of their name and kind by code scan.
KNOWN_TOOLS: dict[str, tuple[str, str]] = {
    "ai.ir_actions_server_search": ("search", "read"),
    "ai.ir_actions_server_read_group": ("read_group", "read"),
    "ai.ir_actions_server_get_fields": ("get_fields", "read"),
    "ai.ir_actions_server_get_menu_details": ("get_menu_details", "read"),
    "ai.ir_actions_server_open_menu_list": ("open_menu_list", "read"),
    "ai.ir_actions_server_open_menu_kanban": ("open_menu_kanban", "read"),
    "ai.ir_actions_server_open_menu_pivot": ("open_menu_pivot", "read"),
    "ai.ir_actions_server_open_menu_graph": ("open_menu_graph", "read"),
    "ai.ir_actions_server_compute_report_measures": ("compute_report_measures", "read"),
    "ai.ir_actions_server_adjust_search": ("adjust_search", "read"),
    "ai_crm.ir_actions_server_ai_get_lead_create_available_params": ("lead_create_params", "read"),
    "ai_crm.ir_actions_server_ai_create_lead": ("create_lead", "write"),
    "ai_documents.ir_actions_server_add_tags": ("documents_add_tags", "write"),
    "ai_documents.ir_actions_server_move_in_folder": ("documents_move_to_folder", "write"),
    "ai_documents.ir_actions_server_rename_documents": ("documents_rename", "write"),
}
# Capability keys the Odoo 19 tree does not ship (generic record tools). Reported as `no` so an Odoo 20
# inventory diff is two lines changing, not a redesign.
DECLARED_ABSENT_KEYS = ["generic_create", "generic_update"]

WRITE_CODE_RE = re.compile(r"\.(create|write|unlink|update)\(|Command\.", re.MULTILINE)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def odoo_revision(odoo_root: str | None) -> tuple[str | None, str | None]:
    if not odoo_root:
        return None, None
    root = Path(odoo_root)
    version = build = None
    rel = root / "odoo" / "release.py"
    if rel.exists():
        src = rel.read_text(errors="replace")
        m = re.search(r"^version_info\s*=\s*\(([^)]*)\)", src, re.MULTILINE)
        if m:
            parts = [p.strip().strip("'\"") for p in m.group(1).split(",")]
            version = ".".join(parts[:2])
        m2 = re.search(r"^version\s*\+=\s*['\"]([^'\"]+)['\"]", src, re.MULTILINE)
        if m2 and version:
            version = version + m2.group(1)
    pkg = root / "PKG-INFO"
    if pkg.exists():
        m = re.search(r"^Version:\s*(.+)$", pkg.read_text(errors="replace"), re.MULTILINE)
        if m:
            build = m.group(1).strip() + " (packaged; no git commit in tree)"
    git = root / ".git"
    if git.exists():
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = git / head[5:]
            build = ref.read_text().strip()[:12] if ref.exists() else head
        else:
            build = head[:12]
    return version, build


def discover(observer_dsn: str, odoo_root: str | None) -> Capabilities:
    version, build = odoo_revision(odoo_root)
    tools: list[ToolInfo] = []
    notes: list[str] = []
    with psycopg.connect(observer_dsn, row_factory=dict_row) as c:
        has_ai = c.execute("select 1 from information_schema.tables where table_name = 'ai_agent'").fetchone()
        if not has_ai:
            notes.append("the `ai` module is not installed in this fixture: no native agent, no tools")
            return Capabilities(version, build, [], "no", False, False, True, notes)
        rows = c.execute(
            """select s.id, s.name->>'en_US' as name, m.model, s.state, s.code,
                      coalesce(s.ai_tool_allow_end_message, false) as end_message,
                      (select d.module||'.'||d.name from ir_model_data d
                        where d.model = 'ir.actions.server' and d.res_id = s.id limit 1) as xml_id
               from ir_act_server s left join ir_model m on m.id = s.model_id
               where s.use_in_ai order by s.id"""
        ).fetchall()
        agents = {r["id"]: r["name"] for r in c.execute(
            "select a.id, p.name from ai_agent a join res_partner p on p.id = a.partner_id").fetchall()}
        via_topic = c.execute(
            "select r.ai_agent_id as agent, t.ir_act_server_id as tool from ai_agent_ai_topic_rel r "
            "join ai_topic_ir_act_server_rel t on t.ai_topic_id = r.ai_topic_id").fetchall()
        direct = []
        if c.execute("select 1 from information_schema.tables where table_name = 'ai_tool_ids_rel'").fetchone():
            direct = c.execute("select parent_id as agent, tool_id as tool from ai_tool_ids_rel").fetchall()
        attached: dict[int, set[str]] = {}
        for r in list(via_topic) + list(direct):
            if r["agent"] in agents:
                attached.setdefault(r["tool"], set()).add(agents[r["agent"]])
        for r in rows:
            key, kind = KNOWN_TOOLS.get(r["xml_id"] or "", (None, None))
            if key is None:
                key = _slug(r["name"] or f"tool_{r['id']}")
                kind = "write" if (r["state"] in ("object_create", "object_write") or WRITE_CODE_RE.search(r["code"] or "")) else "unknown"
                notes.append(f"tool {r['name']!r} ({r['xml_id']}) not in the known map; kind by code scan = {kind}")
            tools.append(ToolInfo(key, r["name"], r["xml_id"], r["model"], kind, sorted(attached.get(r["id"], ())),
                                  {"state": r["state"], "allow_end_message": r["end_message"], "id": r["id"]}))
    present = {t.key for t in tools}
    for k in DECLARED_ABSENT_KEYS:
        if k not in present:
            notes.append(f"{k}: no (not shipped in this tree)")
    return Capabilities(version, build, tools, "yes (logs, INFO level, not persisted)", True, False, True, notes)


def render(caps: Capabilities, agent_name: str | None = None) -> str:
    lines = [f"Odoo                    {caps.odoo_version or '?'}   build: {caps.odoo_build or '?'}"]
    via = caps.reachable_by_delegation(agent_name) if agent_name else set()
    if via:
        lines.append(f"delegation: {agent_name} may start sub-agent sessions with {', '.join(sorted(via))}; their tool "
                     "calls are part of the trace")
    for t in caps.tools:
        if agent_name is None or agent_name in t.attached_agents:
            reach = "yes"
        elif via & set(t.attached_agents):
            reach = f"via delegation ({', '.join(sorted(via & set(t.attached_agents)))})"
        else:
            reach = f"no (attached to {', '.join(t.attached_agents) or 'nobody'})"
        lines.append(f"{t.name:<48} {reach:<40} [{t.key}, {t.kind}]")
    for k in DECLARED_ABSENT_KEYS:
        if k not in {t.key for t in caps.tools}:
            lines.append(f"{k.replace('_', ' ').title():<48} no")
    lines += [f"Trajectory observable   {caps.trajectory_observable}", f"Token usage             {'yes' if caps.token_usage else 'no'}",
              f"Provider-reported cost  {'yes' if caps.provider_cost else 'no'}", f"DB access               {'yes' if caps.db_access else 'no'}"]
    lines += [f"note: {n}" for n in caps.notes]
    return "\n".join(lines)
