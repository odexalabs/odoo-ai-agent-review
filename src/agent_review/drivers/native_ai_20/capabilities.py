"""Capability discovery for Odoo 20's native agent. An agent's tools are the `use_in_ai` server actions of its
skills, which the model loads on demand during a session; the inventory is read from the database, the Odoo
revision from the source tree.

Capability keys: `generic_create` / `generic_update` for the generic record tools, otherwise the tool's
technical name without its `ai_tool_` prefix (e.g. `create_livechat_lead`)."""
from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from ..base import Capabilities, ToolInfo
from ..native_ai.capabilities import odoo_revision

# Tools that change records, by technical name, as shipped in the Odoo 20 tree the driver was built against.
# A write tool missing here is still traced, but attribution cannot compare its arguments with the effect, so a
# new write tool belongs in this set.
WRITE_TOOLS_20 = frozenset({
    "ai_tool_create_records", "ai_tool_update_records", "ai_tool_create_livechat_lead",
    "ai_tool_add_tags", "ai_tool_move_to_folder", "ai_tool_rename_file", "ai_tool_link_document_to_vehicle",
    "ai_tool_assign_emission_factors_to_emission_line",
    "ai_tool_create_update_automation", "ai_tool_run_trigger",
    "ai_tool_delete_campaign_step", "ai_tool_sort_campaign_steps",
    "ai_tool_create_page", "ai_tool_apply_html_to_page", "ai_tool_write_custom_css", "ai_tool_edit_website_menus",
    "ai_tool_set_webform_action", "ai_tool_apply_brand_kit",
    "accounting_audit_create", "accounting_audit_update_checks", "accounting_audit_set_working_file",
})
CAPABILITY_KEYS = {"ai_tool_create_records": "generic_create", "ai_tool_update_records": "generic_update"}


def capability_key(tool_name: str) -> str:
    return CAPABILITY_KEYS.get(tool_name, tool_name.removeprefix("ai_tool_"))


def discover20(observer_dsn: str, odoo_root: str | None) -> Capabilities:
    version, build = odoo_revision(odoo_root)
    tools: dict[int, ToolInfo] = {}
    with psycopg.connect(observer_dsn, row_factory=dict_row) as c:
        agents = {r["id"]: r["name"] for r in c.execute(
            "select a.id, p.name from ai_agent a join res_partner p on p.id = a.partner_id")}
        for r in c.execute("select t.id, t.ai_tool_name, m.model, t.state from ir_act_server t "
                           "left join ir_model m on m.id = t.model_id where t.use_in_ai"):
            name = r["ai_tool_name"] or f"tool_{r['id']}"
            tools[r["id"]] = ToolInfo(capability_key(name), name, None, r["model"],
                                      "write" if name in WRITE_TOOLS_20 else "read_or_other", [], {"state": r["state"]})
        for r in c.execute("select r.ai_agent_id, st.ir_act_server_id, sk.name as skill from ai_agent_ai_skill_rel r "
                           "join ai_skill sk on sk.id = r.ai_skill_id "
                           "join ai_skill_ir_act_server_rel st on st.ai_skill_id = sk.id"):
            ti = tools.get(r["ir_act_server_id"])
            if ti and r["ai_agent_id"] in agents and agents[r["ai_agent_id"]] not in ti.attached_agents:
                ti.attached_agents.append(agents[r["ai_agent_id"]])
                ti.detail.setdefault("skills", []).append(r["skill"])
        delegation: dict[str, list[str]] = {}
        if c.execute("select to_regclass('ai_agent_delegation_rel') as t").fetchone()["t"]:
            for r in c.execute("select parent_id, child_id from ai_agent_delegation_rel order by parent_id, child_id"):
                if r["parent_id"] in agents and r["child_id"] in agents:
                    delegation.setdefault(agents[r["parent_id"]], []).append(agents[r["child_id"]])
    notes = [f"{len(tools)} use_in_ai tools in this fixture; an agent reaches the tools of its skills, loaded on demand",
             ("trajectory: ai.session.event rows of the session and of every sub-agent session it starts "
              "(Odoo 20 does not log tool calls at INFO)")]
    return Capabilities(version, build, list(tools.values()), "yes (ai.session.event rows)", True, False, True, notes,
                        delegation)
