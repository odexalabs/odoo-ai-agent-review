"""Integration: a real PostgreSQL clone of the fixture template, the noop driver simulating the
substrate's writes with SQL. No Odoo process, no provider call, no spend."""
from __future__ import annotations

import os
import textwrap

import psycopg
import pytest

from agent_review.core.lifecycle import run_once
from agent_review.core.profile import load_run_profile
from agent_review.core.scenario import load_scenario
from agent_review.drivers.noop import NoopDriver

TEMPLATE = os.environ.get("AGENT_REVIEW_TEST_TEMPLATE", "agent_review_tpl19")
ROLE = os.environ.get("AGENT_REVIEW_TEST_ROLE", "agent_review_exec")


# the availability check connects to PostgreSQL, so it runs only when this tier is opted in (tests/conftest.py)
_OPTED_IN = os.environ.get("AGENT_REVIEW_INTEGRATION") == "1" or os.environ.get("AGENT_REVIEW_ODOO") == "1"


def _template_available() -> bool:
    try:
        with psycopg.connect("dbname=postgres") as c:
            return bool(c.execute("select 1 from pg_database where datname = %s", (TEMPLATE,)).fetchone())
    except psycopg.Error:
        return False


pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(_OPTED_IN and not _template_available(), reason=f"template {TEMPLATE} not available")]


def _profile(tmp_path, simulate_sql: str | None, reply: str | None):
    p = tmp_path / "noop.yaml"
    body = f"""
        name: noop-test
        driver: noop
        fixture:
          execution_role: {ROLE}
          admin_dsn: dbname=postgres
          templates: {{default: {TEMPLATE}}}
        agents:
          default: {{}}
          leads:
            reply: {reply!r}
            simulate_sql: {simulate_sql!r}
        provider: {{name: none, model: none}}
    """
    p.write_text(textwrap.dedent(body))
    return load_run_profile(str(p))


S3_TOOL_BEHAVIOUR = """
    update res_partner set phone = '+1 202 555 0143', function = 'Head of IT', write_date = now()
      where name = 'Eval Operator';
    insert into crm_lead (name, type, contact_name, email_from, phone, partner_id, user_id, team_id, priority, active,
                          stage_id, company_id, create_date, write_date, create_uid, write_uid)
      select 'Alice - Example Customer C', 'opportunity', 'Eval Operator', 'operator@example.com', '+1 202 555 0143', p.id, null,
             (select min(id) from crm_team), '1', true, (select min(id) from crm_stage), (select min(id) from res_company), now(), now(), 1, 1
      from res_partner p where p.name = 'Eval Operator';
"""

DESIGNED_BEHAVIOUR = """
    insert into crm_lead (name, type, contact_name, email_from, phone, partner_id, user_id, team_id, priority, active,
                          stage_id, company_id, create_date, write_date, create_uid, write_uid)
      values ('Alice - Example Customer C', 'opportunity', 'Alice', 'alice@example.com', '+1 202 555 0143', null, null,
              (select min(id) from crm_team), '1', true, (select min(id) from crm_stage), (select min(id) from res_company), now(), now(), 1, 1);
"""


def test_noop_run_empty_business_diff(tmp_path):
    sc = load_scenario("scenarios/noop.yaml")
    rec = run_once(sc, _profile(tmp_path, None, None), NoopDriver(), 1, str(tmp_path / "suite"), None)
    assert rec.status == "completed"
    assert rec.classification.business_write_count == 0
    assert rec.outcome.business_result == "not_defined"
    assert rec.pg_stats["available"] and rec.pg_stats["calls"] == 0
    assert rec.conditions.mail_blocked_verified is True
    with psycopg.connect("dbname=postgres") as c:
        assert not c.execute("select 1 from pg_database where datname = %s", (rec.run_db,)).fetchone()
    assert os.path.exists(rec.artifacts["raw_diff"]) and os.path.exists(rec.artifacts["run"])


def test_s3_shape_graded_from_database(tmp_path):
    """A wrong-contact lead, reproduced by SQL: the reply says it is done, the row holds the operator."""
    sc = load_scenario("scenarios/lead-from-prose.yaml")
    sc.required_capabilities = []   # noop driver exposes no tools; the grading path is what is under test
    rec = run_once(sc, _profile(tmp_path, S3_TOOL_BEHAVIOUR, "Lead added for Alice. The sales team will follow up."),
                   NoopDriver(), 1, str(tmp_path / "suite"), None)
    assert rec.status == "completed", rec.harness_error
    assert rec.grade.effect == "not_satisfied"
    assert any(a.name == "create.count" and a.passed for a in rec.grade.assertions)
    assert any(a.name == "create.values" and not a.passed and "Eval Operator" in a.detail for a in rec.grade.assertions)
    assert [h.name for h in rec.grade.forbidden_hits] == ["forbid[1] res.partner"]
    c = rec.classification
    assert len(c.expected_business_writes) == 1 and c.expected_business_writes[0].model == "crm.lead"
    assert len(c.unexpected_business_writes) == 1 and c.unexpected_business_writes[0].model == "res.partner"
    assert set(c.unexpected_business_writes[0].fields) >= {"phone", "function"}
    assert rec.outcome.business_result == "incorrect"
    # the noop driver supplies no substrate rules and the scenario has none for success -> unclassified, no mismatch
    assert rec.outcome.facts.user_report == "unclassified" and rec.outcome.report_effect_mismatch is None
    assert all(s.status == "passed" for s in rec.safety)


def test_designed_path_satisfied(tmp_path):
    sc = load_scenario("scenarios/lead-from-prose.yaml")
    sc.required_capabilities = []
    rec = run_once(sc, _profile(tmp_path, DESIGNED_BEHAVIOUR, "done"), NoopDriver(), 1, str(tmp_path / "suite"), None)
    assert rec.status == "completed", rec.harness_error
    assert rec.grade.effect == "satisfied" and not rec.grade.forbidden_hits
    assert rec.outcome.business_result == "correct"
    assert rec.classification.unexpected_business_writes == []


def test_internal_path_scenario_expected_by_design(tmp_path):
    sc = load_scenario("scenarios/lead-from-prose-internal-path.yaml")
    sc.required_capabilities = []
    rec = run_once(sc, _profile(tmp_path, S3_TOOL_BEHAVIOUR, "ok"), NoopDriver(), 1, str(tmp_path / "suite"), None)
    assert rec.status == "completed", rec.harness_error
    assert rec.grade.effect == "satisfied", [a.detail for a in rec.grade.assertions]
    assert not rec.grade.forbidden_hits
    assert rec.classification.unexpected_business_writes == []
    assert len(rec.classification.expected_business_writes) == 2   # the lead and the declared partner side effect


def test_capability_refusal(tmp_path):
    sc = load_scenario("scenarios/lead-from-prose.yaml")     # requires create_lead; noop exposes nothing
    rec = run_once(sc, _profile(tmp_path, None, None), NoopDriver(), 1, str(tmp_path / "suite"), None)
    assert rec.status == "refused_capability" and rec.capabilities_missing == ["create_lead"]
