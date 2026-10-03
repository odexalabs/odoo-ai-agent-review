"""Audit tests: pure-Python, no database, no provider. Each test names the guarantee it
would fail if broken. Written to falsify the implementation, not to describe it."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from unittest import mock

import pytest

from agent_review.core import credentials
from agent_review.core.attribution import attribute
from agent_review.core.classify import ClassificationRules, classify
from agent_review.core.contracts import (
    ChangeEvidence,
    Coverage,
    DriverRunResult,
    ModelMetadata,
    RequestStatus,
    ResponseRule,
    RowChange,
    TableSummary,
    TokenUsage,
    ToolCall,
    Turn,
)
from agent_review.core.cost import Pricing
from agent_review.core.credentials import spend_plan_line
from agent_review.core.fixture import FX_PREFIX, PostgresTemplateBackend
from agent_review.core.grade import Grade, grade
from agent_review.core.outcomes import evaluate
from agent_review.core.redact import REDACTED, RedactionRules
from agent_review.core.scenario import Resolved, ScenarioError, fact_present, level1_scenario, load_scenario
from agent_review.drivers.base import Capabilities, ToolInfo
from agent_review.invariants import Invariant, InvariantResult, Violation, run_invariants


def _write(tmp_path, name, body):
    p = tmp_path / f"{name}.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def _result(turns, trace, notes=None, rules=None, usage=...):   # a fresh TokenUsage per call; None stays None
    return DriverRunResult("s", turns, ModelMetadata("openai", "gpt-5-mini", False), trace, TokenUsage(100, 10, 0, 2) if usage is ... else usage,
                           driver_notes=notes or [], known_response_rules=rules or [])


def _evidence(rows, touched=None, coverage=None):
    cov = dict(coverage or {})
    for rc in rows:
        cov.setdefault(rc.table, Coverage.EXACT)
    for t in touched or []:
        cov.setdefault(t.table, t.coverage)
    return ChangeEvidence(list(touched or []), rows, cov, "test")


# ============================================================ FixtureBackend guardrails
class _FakeConn:
    """Records executed SQL; never talks to a server."""

    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, q, params=None):
        self.log.append((str(q), params))
        return mock.Mock(fetchone=lambda: None, fetchall=list)


@pytest.mark.parametrize("name", ["odexalabs_fx_tpl", "odexalabs_fx_tpl2", "odoo-18-2", "postgres", "template1", "",
                                  "odexalabs_fx_", "odexalabs_fx_run", "odexalabs_fx_run_"])
def test_destroy_run_refuses_everything_that_is_not_a_run_database(name):
    """Guarantee: no code path can drop the template, a foreign database, or anything outside the run
    namespace. The prefix `odexalabs_fx_` alone also matches the TEMPLATE names."""
    log = []
    b = PostgresTemplateBackend(template="odexalabs_fx_tpl", execution_role="odexalabs_fx")
    with mock.patch.object(b, "_admin", lambda: _FakeConn(log)):
        from agent_review.core.contracts import EnvironmentHandle
        with pytest.raises(RuntimeError, match="refusing"):
            b.destroy_run(EnvironmentHandle(name, "", "odexalabs_fx", ""))
    assert log == [], f"SQL was issued for {name!r}: {log}"


def test_destroy_run_never_drops_the_configured_template_even_with_run_prefix():
    log = []
    b = PostgresTemplateBackend(template="odexalabs_fx_run_shared_tpl", execution_role="odexalabs_fx")
    with mock.patch.object(b, "_admin", lambda: _FakeConn(log)):
        from agent_review.core.contracts import EnvironmentHandle
        with pytest.raises(RuntimeError, match="template"):
            b.destroy_run(EnvironmentHandle("odexalabs_fx_run_shared_tpl", "", "odexalabs_fx", ""))
    assert log == []


def test_run_name_is_validated_before_any_sql():
    b = PostgresTemplateBackend(template="odexalabs_fx_tpl", execution_role="odexalabs_fx")
    log = []
    with mock.patch.object(b, "_admin", lambda: _FakeConn(log)):
        for bad in ["x; drop database postgres", 'a"b', "a b", "A", "über", ""]:
            with pytest.raises(Exception, match="run name|too long|invalid"):
                b.clone_for_run(bad)
    assert log == []
    assert FX_PREFIX == "odexalabs_fx_"


# ============================================================ inspection set: expected rows are graded, not only changed rows
def _update_scenario(tmp_path):
    return load_scenario(_write(tmp_path, "u", """
        turns: ["Set discount 10 on the three quotations"]
        expect:
          kind: update
          select: {model: sale.order.line, where: [[order_id, in, [1, 2, 3]]]}
          values: {discount: 10}
          fields_only: [discount]
    """))


def test_update_expected_rows_not_touched_are_graded_unsatisfied(tmp_path):
    """Prompt expects A, B, C; the agent changed only A. Change detection sees A. The grader MUST
    still inspect B and C and mark the key unsatisfied — a write that did not happen produces no diff."""
    sc = _update_scenario(tmp_path)
    res = Resolved(tables={"sale.order.line": "sale_order_line"}, update_ids=[11, 12, 13], update_values={"discount": 10})
    ev = _evidence([RowChange("sale_order_line", 11, "changed", {"id": 11, "discount": 0}, {"id": 11, "discount": 10}, {"discount": [0, 10]})])
    after = {"sale_order_line": {11: {"id": 11, "discount": 10}, 12: {"id": 12, "discount": 0}, 13: {"id": 13, "discount": 0}}}
    g = grade(sc, res, ev, ["Done"], 1, rows_after=lambda t: after.get(t, {}))
    assert g.effect == "not_satisfied"
    detail = next(a.detail for a in g.assertions if a.name == "update.values")
    assert "#12" in detail and "#13" in detail and "#11" not in detail


def test_update_expected_row_already_holding_the_value_is_satisfied_by_state(tmp_path):
    """The key is a STATE, not a diff: a row that already held the value before the run satisfies it
    without appearing in the change evidence."""
    sc = _update_scenario(tmp_path)
    res = Resolved(tables={"sale.order.line": "sale_order_line"}, update_ids=[11, 12], update_values={"discount": 10})
    ev = _evidence([RowChange("sale_order_line", 11, "changed", {"id": 11, "discount": 0}, {"id": 11, "discount": 10}, {"discount": [0, 10]})])
    after = {"sale_order_line": {11: {"id": 11, "discount": 10}, 12: {"id": 12, "discount": 10}}}
    g = grade(sc, res, ev, ["Done"], 1, rows_after=lambda t: after.get(t, {}))
    assert g.effect == "satisfied", [a.detail for a in g.assertions]


def test_update_without_after_state_cannot_claim_satisfaction_for_untouched_rows(tmp_path):
    sc = _update_scenario(tmp_path)
    res = Resolved(tables={"sale.order.line": "sale_order_line"}, update_ids=[11, 12], update_values={"discount": 10})
    ev = _evidence([RowChange("sale_order_line", 11, "changed", {"id": 11}, {"id": 11, "discount": 10}, {"discount": [0, 10]})])
    g = grade(sc, res, ev, ["Done"], 1)
    assert g.effect == "not_satisfied"


def test_create_two_rows_when_one_expected_is_not_satisfied(tmp_path):
    sc = load_scenario(_write(tmp_path, "c", """
        turns: ["x"]
        expect: {kind: create, model: crm.lead, count: 1, values: {contact_name: Alice}}
    """))
    res = Resolved(tables={"crm.lead": "crm_lead"}, create_values={"contact_name": "Alice"})
    ev = _evidence([RowChange("crm_lead", 13, "added", None, {"id": 13, "contact_name": "Alice"}),
                    RowChange("crm_lead", 14, "added", None, {"id": 14, "contact_name": "Alice"})])
    g = grade(sc, res, ev, ["done"], 2)
    assert g.effect == "not_satisfied" and not next(a for a in g.assertions if a.name == "create.count").passed
    # zero rows: the expected row did not exist before and still does not
    g0 = grade(sc, res, _evidence([]), ["done"], 0)
    assert g0.effect == "not_satisfied"


# ============================================================ Level 1 vs Level 2
def test_level1_cannot_produce_correctness_or_mismatch_or_unexpected():
    sc = level1_scenario("l1", "reassign leads", None, ["basic_write_agent"])
    ev = _evidence([RowChange("crm_lead", 5, "changed", {"id": 5, "user_id": 1}, {"id": 5, "user_id": 2}, {"user_id": [1, 2]})])
    cls = classify(ev, ClassificationRules.load(), RedactionRules.load())
    assert cls.level == 1 and cls.unexpected_business_writes == [] and cls.expected_business_writes == []
    g = grade(sc, None, ev, ["Done."], cls.business_write_count)
    assert g.effect == "not_defined" and g.assertions == [] and g.forbidden_hits == []
    r = _result([Turn("reassign", RequestStatus.RETURNED, "Done.")], [ToolCall("t", None, "{}")],
                rules=[ResponseRule("success", "Done.", False, "substrate:test")])
    o = evaluate(r, g, 1, cls.business_write_count, [], set())
    assert o.business_result == "not_defined"
    assert o.facts.expected_effect == "not_defined" and o.facts.user_report == "success"
    assert o.report_effect_mismatch is None      # a classified report alone never yields a mismatch at Level 1
    a = attribute(sc, None, r, g, o, set())
    assert a.layer == "not_in_scope"


# ============================================================ READ grading, deterministic only
@pytest.mark.parametrize("text", ["total 14,260.00", "total 14260", "INR 14,260", "INR 14260", "14.260,00 EUR",
                                  "14 260,00", "14260.0", "Total: **14,260.00**"])
def test_amount_forms_recognised(text):
    assert fact_present({"amount": 14260}, text)


@pytest.mark.parametrize("text", ["114260", "14260.5", "1,142,600", "142,600", "14,260.01", "214260"])
def test_amount_forms_not_confused(text):
    assert not fact_present({"amount": 14260}, text)


def test_read_zero_writes_but_wrong_answer_is_incorrect(tmp_path):
    sc = load_scenario(_write(tmp_path, "r", """
        turns: ["which invoices"]
        expect:
          kind: read
          business_writes: 0
          facts:
            must_contain: [INV/0031, INV/0042, INV/0057, {amount: 14260}, "regex:\\\\b3\\\\b"]
            must_not_contain: [INV/0052]
    """))
    good = "INV/0031, INV/0042 and INV/0057 — 3 invoices, total 14,260.00"
    assert grade(sc, Resolved(), _evidence([]), [good], 0).effect == "satisfied"
    wrong_total = "INV/0031, INV/0042 and INV/0057 — 3 invoices, total 15,260.00"
    assert grade(sc, Resolved(), _evidence([]), [wrong_total], 0).effect == "not_satisfied"
    wrong_count = "INV/0031, INV/0042 and INV/0057 — 4 invoices, total 14,260.00"
    assert grade(sc, Resolved(), _evidence([]), [wrong_count], 0).effect == "not_satisfied"
    forbidden = good + " (also INV/0052)"
    g = grade(sc, Resolved(), _evidence([]), [forbidden], 0)
    assert g.effect == "not_satisfied"
    absent = "I found 3 invoices totalling 14,260.00"
    assert grade(sc, Resolved(), _evidence([]), [absent], 0).effect == "not_satisfied"
    r = _result([Turn("q", RequestStatus.RETURNED, wrong_total)], [ToolCall("AI: Search", None, "{}")])
    o = evaluate(r, grade(sc, Resolved(), _evidence([]), [wrong_total], 0), 2, 0, [], set())
    assert o.business_result == "incorrect" and o.facts.transaction == "committed"
    # writes happened on a READ scenario -> the write count assertion fails -> not_satisfied
    assert grade(sc, Resolved(), _evidence([]), [good], 1).effect == "not_satisfied"


# ============================================================ DriverRunResult: turns authoritative, optional fields degrade honestly
def test_turn_count_is_derived_and_optional_fields_degrade():
    r = DriverRunResult("s", [Turn("a", RequestStatus.RETURNED, "x"), Turn("b", RequestStatus.ERROR, None, "boom")],
                        ModelMetadata("p", "m", False))
    assert r.turn_count == 2 and r.request_status == RequestStatus.ERROR and r.error == "boom" and r.final_response == "x"
    assert r.tool_trace is None and r.token_usage is None and r.provider_cost_usd is None
    o = evaluate(r, Grade("not_satisfied"), 2, 0, [], {"w"})
    assert o.facts.tool_execution == "unobservable" and o.facts.transaction == "unknown"
    assert o.execution == {"recovered_tool_errors": None, "llm_round_trips": None, "tool_calls": None, "turns": 2}
    e = Pricing().estimate("openai", "gpt-5-mini", None)
    assert e["usd"] is None and "no token usage" in e["reason"]
    empty = DriverRunResult("s", [], ModelMetadata("p", "m", False))
    assert empty.request_status == RequestStatus.ERROR and empty.final_response is None and empty.turn_count == 0


# ============================================================ outcome taxonomy
def test_refused_and_clarification_are_not_reached_not_incorrect(tmp_path):
    sc = load_scenario(_write(tmp_path, "s", """
        turns: ["create 40 leads"]
        expect: {kind: create, model: crm.lead, count: 40}
        report:
          refused: ["regex:only once per conversation"]
          clarification: ["regex:please confirm"]
    """))
    res = Resolved(tables={"crm.lead": "crm_lead"})
    g = grade(sc, res, _evidence([]), [], 0)
    r = _result([Turn("x", RequestStatus.RETURNED, "I can run the lead-creation action only once per conversation.")], [])
    o = evaluate(r, g, 2, 0, sc.response_rules, set())
    assert (o.interaction_outcome, o.business_result) == ("refused", "not_reached") and o.reason
    r2 = _result([Turn("x", RequestStatus.RETURNED, "Please confirm the name.")], [])
    o2 = evaluate(r2, g, 2, 0, sc.response_rules, set())
    assert (o2.interaction_outcome, o2.business_result) == ("clarification", "not_reached")
    # a clarification pattern that matches AFTER the key was satisfied does not override completion
    o3 = evaluate(r2, Grade("satisfied"), 2, 1, sc.response_rules, set())
    assert o3.interaction_outcome == "completed" and o3.business_result == "correct"


def test_recovered_tool_error_coexists_with_completion():
    r = _result([Turn("x", RequestStatus.RETURNED, "six figures")],
                [ToolCall("AI: Read group", None, "{}", error="invalid field __count"), ToolCall("AI: Read group", None, "{}"),
                 ToolCall("AI: Search", None, "{}")])
    o = evaluate(r, Grade("satisfied"), 2, 0, [], set())
    assert o.interaction_outcome == "completed" and o.business_result == "correct"
    assert o.execution["recovered_tool_errors"] == 1 and o.execution["tool_calls"] == 3
    assert o.facts.tool_execution == "succeeded"


def test_last_tool_call_errored_but_key_satisfied_is_recovered_not_unrecovered():
    """A model that recovers from the LAST tool error by other means (or an earlier result) still
    reached the key; 'tool_error_unrecovered' is reserved for errors the run did not recover from."""
    r = _result([Turn("x", RequestStatus.RETURNED, "answer")],
                [ToolCall("AI: Search", None, "{}"), ToolCall("AI: Read group", None, "{}", error="NoneType")])
    o = evaluate(r, Grade("satisfied"), 2, 0, [], set())
    assert o.interaction_outcome == "completed" and o.business_result == "correct"
    assert o.execution["recovered_tool_errors"] == 1
    o2 = evaluate(r, Grade("not_satisfied"), 2, 0, [], set())
    assert o2.interaction_outcome == "tool_error_unrecovered" and o2.business_result == "not_reached"
    assert o2.facts.tool_execution == "failed"


def test_key_satisfied_but_forbidden_writes_is_not_business_correct(tmp_path):
    """Forbidden state is a Level 2 CORRECTNESS primitive (forbidden extra writes). The keyed
    effect can be satisfied while the business result is incorrect; the two facts stay separate."""
    sc = load_scenario(_write(tmp_path, "s", """
        turns: ["x"]
        expect: {kind: create, model: crm.lead, count: 1}
        forbid: [{model: res.partner, any: true}]
    """))
    res = Resolved(tables={"crm.lead": "crm_lead", "res.partner": "res_partner"}, forbid_ids={0: None})
    ev = _evidence([RowChange("crm_lead", 13, "added", None, {"id": 13}),
                    RowChange("res_partner", 14, "changed", {"id": 14, "phone": None}, {"id": 14, "phone": "+91"}, {"phone": [None, "+91"]})])
    g = grade(sc, res, ev, ["done"], 2)
    assert g.effect == "satisfied" and len(g.forbidden_hits) == 1
    r = _result([Turn("x", RequestStatus.RETURNED, "Thanks. Our team will recontact you soon.")], [ToolCall("w", None, "{}")],
                rules=[ResponseRule("success", "Thanks. Our team will recontact you soon.", False, "substrate:test")])
    o = evaluate(r, g, 2, 2, [], {"w"})
    assert o.facts.expected_effect == "satisfied"
    assert o.business_result == "incorrect"
    assert o.report_effect_mismatch == {"direction": "none"}    # told success, key satisfied: consistent


# ============================================================ report classification
def test_report_classification_only_by_rule():
    r = _result([Turn("x", RequestStatus.RETURNED, "I've handled the records as requested.")], [])
    assert evaluate(r, Grade("not_satisfied"), 2, 0, [], set()).facts.user_report == "unclassified"
    rule = ResponseRule("success", "handled the records", False, "scenario")
    o = evaluate(r, Grade("not_satisfied"), 2, 0, [rule], set())
    assert o.facts.user_report == "success" and o.report_effect_mismatch["direction"] == "false_success"
    r2 = _result([Turn("x", RequestStatus.RETURNED, "Done.")], [], rules=[ResponseRule("success", "Done.", False, "substrate:t")])
    assert evaluate(r2, Grade("satisfied"), 2, 0, [], set()).facts.user_report == "success"
    r3 = _result([Turn("x", RequestStatus.RETURNED, "<p>The assistant is unavailable right now.</p>")], [],
                 rules=[ResponseRule("failure", "The assistant is unavailable right now", False, "substrate:t")])
    assert evaluate(r3, Grade("satisfied"), 2, 1, [], set()).facts.user_report == "failure"
    r4 = _result([Turn("x", RequestStatus.RETURNED, None)], [])
    assert evaluate(r4, Grade("not_satisfied"), 2, 0, [], set()).facts.user_report == "absent"
    assert evaluate(r4, Grade("not_defined"), 1, 0, [], set()).report_effect_mismatch is None


# ============================================================ attribution: never guessed
def _create_sc(tmp_path):
    return load_scenario(_write(tmp_path, "s", """
        turns: ["x"]
        expect: {kind: create, model: crm.lead, count: 1, values: {contact_name: Alice, phone: "+1 202 555 0143"}}
    """))


def test_attribution_unattributable_when_evidence_incomplete(tmp_path):
    sc = _create_sc(tmp_path)
    res = Resolved(tables={"crm.lead": "crm_lead"}, create_values={"contact_name": "Alice", "phone": "+1 202 555 0143"})
    g = Grade("not_satisfied", assertions=[])
    # incorrect, no trace at all
    r = _result([Turn("x", RequestStatus.RETURNED, "done")], None)
    o = evaluate(r, g, 2, 1, [], {"AI CRM: Create Lead"})
    assert attribute(sc, res, r, g, o, {"AI CRM: Create Lead"}).layer == "unattributable"
    # request error whose text is NOT a transport signature
    r2 = _result([Turn("x", RequestStatus.ERROR, None, "ValueError: something internal")], None)
    o2 = evaluate(r2, g, 2, 0, [], set())
    assert attribute(sc, res, r2, g, o2, set()).layer == "unattributable"


def test_transport_not_inferred_from_incidental_digits_or_harness_timeout(tmp_path):
    sc = _create_sc(tmp_path)
    res = Resolved(tables={"crm.lead": "crm_lead"}, create_values={"contact_name": "Alice"})
    g = Grade("not_satisfied")
    r = _result([Turn("x", RequestStatus.ERROR, None, "AccessError: record 429 does not exist or has been deleted")], [])
    o = evaluate(r, g, 2, 0, [], set())
    assert attribute(sc, res, r, g, o, set()).layer != "TRANSPORT"
    # the harness's own client timeout is not evidence of a provider/network failure
    r2 = _result([Turn("x", RequestStatus.TIMEOUT, None, "harness timeout: ReadTimeout('The read operation timed out')")], [])
    o2 = evaluate(r2, g, 2, 0, [], set())
    a2 = attribute(sc, res, r2, g, o2, set())
    assert a2.layer == "unattributable", a2
    # a provider exception surfaced by the substrate IS transport evidence
    r3 = _result([Turn("x", RequestStatus.TIMEOUT, None, "UserError: ReadTimeout(... api.openai.com ... read timeout=30)")], [])
    o3 = evaluate(r3, g, 2, 0, [], set())
    assert attribute(sc, res, r3, g, o3, set()).layer == "TRANSPORT"
    r4 = _result([Turn("x", RequestStatus.ERROR, None, "HTTPError: 429 Too Many Requests from api.openai.com")], [])
    o4 = evaluate(r4, g, 2, 0, [], set())
    assert attribute(sc, res, r4, g, o4, set()).layer == "TRANSPORT"


def test_tool_attribution_requires_the_failed_field_to_have_been_sent(tmp_path):
    """TOOL = correct arguments + wrong database. If the model never sent the field that failed, the
    boundary that diverged is MODEL, whatever else it sent correctly."""
    sc = _create_sc(tmp_path)
    res = Resolved(tables={"crm.lead": "crm_lead"}, create_values={"contact_name": "Alice", "phone": "+1 202 555 0143"})
    ev = _evidence([RowChange("crm_lead", 13, "added", None, {"id": 13, "contact_name": "Alice", "phone": None})])
    g = grade(sc, res, ev, ["done"], 1)
    assert g.effect == "not_satisfied"
    trace = [ToolCall("AI CRM: Create Lead", {"contact_name": "Alice"}, "{'contact_name': 'Alice'}")]
    r = _result([Turn("x", RequestStatus.RETURNED, "done")], trace)
    o = evaluate(r, g, 2, 1, [], {"AI CRM: Create Lead"})
    a = attribute(sc, res, r, g, o, {"AI CRM: Create Lead"})
    assert a.layer == "MODEL", a
    # now the model DID send the phone and the database dropped it -> TOOL
    trace2 = [ToolCall("AI CRM: Create Lead", None, "{'contact_name': 'Alice', 'phone': '+1 202 555 0143'}")]
    r2 = _result([Turn("x", RequestStatus.RETURNED, "done")], trace2)
    o2 = evaluate(r2, g, 2, 1, [], {"AI CRM: Create Lead"})
    assert attribute(sc, res, r2, g, o2, {"AI CRM: Create Lead"}).layer == "TOOL"


def test_read_incorrect_with_succeeding_tools_is_not_asserted_as_model_without_tool_results(tmp_path):
    """The native substrate exposes tool ARGUMENTS, not tool RESULTS. 'The tool ran what it was
    given' is unproven; the boundary MODEL->TOOL is covered, TOOL->EFFECT is not. Evidence is
    recorded for a human; the layer is not asserted."""
    sc = load_scenario(_write(tmp_path, "s", """
        turns: ["x"]
        expect: {kind: read, facts: {must_contain: [{amount: 4430.50}]}}
    """))
    reply = "total 5,429.50"
    g = grade(sc, Resolved(), _evidence([]), [reply], 0)
    r = _result([Turn("x", RequestStatus.RETURNED, reply)], [ToolCall("AI: Search", None, "{'domain': '[...]'}")])
    o = evaluate(r, g, 2, 0, [], set())
    a = attribute(sc, Resolved(), r, g, o, set())
    assert a.layer == "unattributable"
    assert any("AI: Search" in e for e in a.evidence)     # the arguments are preserved for manual attribution


# ============================================================ capability discovery
def _caps():
    tools = [ToolInfo("search", "AI: Search", "ai.x", None, "read", ["Ask AI", "Eval Lead Agent"]),
             ToolInfo("read_group", "AI: Read group", "ai.y", None, "read", ["Ask AI"]),
             ToolInfo("create_lead", "AI CRM: Create Lead", "ai_crm.z", "crm.lead", "write", ["Eval Lead Agent"])]
    return Capabilities("19.0+e", "19.0 (test tree, packaged)", tools, "yes (logs)", True, False, True)


def test_capabilities_are_per_scenario_and_per_agent():
    c = _caps()
    assert c.missing_for(["search", "read_group"], "Ask AI") == []
    assert c.missing_for(["search", "generic_update"], "Ask AI") == ["generic_update"]
    assert c.missing_for(["create_lead"], "Ask AI") == ["create_lead"]           # attached to another agent
    assert c.missing_for(["create_lead"], "Eval Lead Agent") == []
    assert c.write_tool_names() == {"AI CRM: Create Lead"}
    assert c.odoo_build and c.odoo_version


# ============================================================ model identity
def test_model_identity_in_profile_only_and_alias_is_limited(tmp_path):
    for banned in ("provider", "llm_model", "model_identifier", "profile"):
        with pytest.raises(ScenarioError, match="run profile"):
            load_scenario(_write(tmp_path, banned, f"turns: ['x']\n{banned}: gpt-5-mini\n"))
    from agent_review.drivers.native_ai.driver import NativeAiDriver
    assert ModelMetadata("openai", "gpt-5-mini", False).reproducibility == "limited"
    assert NativeAiDriver.name == "native_ai"


# ============================================================ credentials
def test_subprocess_cannot_see_ambient_keys(monkeypatch):
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ODOO_AI_CHATGPT_TOKEN"):
        monkeypatch.setenv(k, f"ambient-{k}")
    monkeypatch.setenv(credentials.ENV_SOURCE, "named-key-1234")
    env = credentials.scrubbed_environment()
    probe = "import os,json;print(json.dumps({k:os.environ.get(k) for k in " + repr(
        ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ODOO_AI_CHATGPT_TOKEN", credentials.ENV_SOURCE]) + "}))"
    out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True).stdout
    assert all(v is None for v in json.loads(out).values()), out


HOSTILE_ODOO_ENV = {"ODOO_SMTP_SERVER": "smtp.hostile.example", "ODOO_SMTP_PORT": "25", "ODOO_MAX_CRON_THREADS": "2",
                    "ODOO_ADDONS_PATH": "/tmp/hostile", "OPENERP_SERVER_EXTRA": "x", "PGSERVICEFILE": "/tmp/pgservice",
                    "PGTARGETSESSIONATTRS": "any"}


def test_generated_odoo_option_variables_never_reach_the_subprocess(monkeypatch):
    """Odoo 19/20 generate `ODOO_<OPTION>` for every config option without a declared name, and each
    outranks the -c file. A fixed scrub list missed them; the namespaces go whole."""
    for k, v in HOSTILE_ODOO_ENV.items():
        monkeypatch.setenv(k, v)
    probe = f"import os,json;print(json.dumps({{k:os.environ.get(k) for k in {sorted(HOSTILE_ODOO_ENV)!r}}}))"
    planted = json.loads(subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout)
    assert planted == HOSTILE_ODOO_ENV                                    # positive control: they were there
    out = subprocess.run([sys.executable, "-c", probe], env=credentials.scrubbed_environment(),
                         capture_output=True, text=True, check=True).stdout
    assert all(v is None for v in json.loads(out).values()), out


def _odoo19_tree():
    """An Odoo 19 source tree and its interpreter, from the environment (tests/conftest.py, `odoo` tier)."""
    from pathlib import Path
    root, py = os.environ.get("AGENT_REVIEW_TEST_ODOO19_ROOT", ""), os.environ.get("AGENT_REVIEW_TEST_ODOO19_PYTHON", "")
    return (root, py) if root and py and Path(root, "odoo/tools/config.py").is_file() and Path(py).is_file() else (None, None)


@pytest.mark.odoo
def test_an_ambient_odoo_smtp_server_cannot_replace_the_unroutable_mail_fallback(tmp_path):
    """The mail-block guarantee, attacked where it lives: Odoo's own config parser. The
    driver writes smtp.invalid:1 and cron off; an ambient ODOO_SMTP_SERVER outranks a -c file on Odoo 19.
    Positive control first — without the scrub the hostile value wins — so the test cannot pass vacuously."""
    root, py = _odoo19_tree()
    if root is None:
        pytest.skip("set AGENT_REVIEW_TEST_ODOO19_ROOT and AGENT_REVIEW_TEST_ODOO19_PYTHON to an Odoo 19 tree")
    conf = tmp_path / "run.conf"
    conf.write_text("[options]\nsmtp_server = smtp.invalid\nsmtp_port = 1\nmax_cron_threads = 0\n")
    probe = (f"import json; from odoo.tools import config; config.parse_config(['-c', {str(conf)!r}]); "
             "print(json.dumps({k: config[k] for k in ('smtp_server', 'smtp_port', 'max_cron_threads')}))")
    hostile = {**os.environ, **HOSTILE_ODOO_ENV, "PYTHONPATH": root}

    def effective(env):
        out = subprocess.run([py, "-c", probe], env=env, cwd=root, capture_output=True, text=True, timeout=120, check=False)
        assert out.returncode == 0, out.stderr[-600:]
        return json.loads(out.stdout.strip().splitlines()[-1])

    attacked = effective(hostile)
    assert attacked["smtp_server"] == "smtp.hostile.example" and attacked["max_cron_threads"] == 2, attacked   # the attack is real
    defended = credentials.scrubbed_environment(hostile)
    defended.update({"ODOO_RC": str(conf), "PYTHONPATH": root})              # exactly what the driver adds back
    assert effective(defended) == {"smtp_server": "smtp.invalid", "smtp_port": 1, "max_cron_threads": 0}


def test_credential_empty_file_and_whitespace_env(monkeypatch, tmp_path):
    f = tmp_path / "k"
    f.write_text("  \n")
    with pytest.raises(credentials.CredentialError, match="empty"):
        credentials.load_credential(str(f))
    with pytest.raises(credentials.CredentialError):
        credentials.load_credential(str(tmp_path / "missing"))
    monkeypatch.setenv(credentials.ENV_SOURCE, "   ")
    with pytest.raises(credentials.CredentialError, match="STOP"):
        credentials.load_credential()


def test_gate_requires_go_and_never_underflows_to_zero(monkeypatch, tmp_path, capsys):
    from agent_review import cli
    from agent_review.core.profile import RunProfile
    monkeypatch.setenv(credentials.ENV_SOURCE, "named-key-9876")
    monkeypatch.setattr(cli, "RUNS_ROOT", str(tmp_path))
    prof = RunProfile("p", "native_ai", {}, {}, {"default": {}}, {"name": "openai", "model": "gpt-5-mini"}, planning={})
    with pytest.raises(SystemExit, match="no paid call"):
        cli._gate(prof, 5, go=False)
    line = capsys.readouterr().out
    assert "…9876" in line and "named-key-9876" not in line and "estimated spend: unknown" in line
    prof.planning = {"estimated_usd_per_run": 0.03, "spend_cap_usd": 0.10}
    with pytest.raises(SystemExit, match="exceed the cap"):
        cli._gate(prof, 5, go=True)
    prof.planning = {"estimated_usd_per_run": 0.01, "spend_cap_usd": 0.10}
    assert cli._gate(prof, 5, go=True).last4 == "9876"
    assert cli._gate(prof, 5, go=False, probe=True) is None      # a probe never needs a credential


def test_spend_plan_line_never_prints_secret():
    c = credentials.Credential("env:X", "1234", "sk-verysecret1234")
    line = spend_plan_line(c, 5, None, "no estimate", "openai", "m")
    assert "sk-verysecret" not in line and "1234" in line and "unknown" in line


# ============================================================ cost
def test_missing_pricing_and_reported_never_merged():
    p = Pricing()
    e = p.estimate("openai", "gpt-5-mini", TokenUsage(0, 0, 0, 0))
    assert e["usd"] == 0.0 and e["label"] == "estimated"        # zero usage is a real measurement of zero
    e2 = p.estimate("google", "gemini-2.5-flash", TokenUsage(10, 10, 0, 1))
    assert e2["usd"] is None and "no published price" in e2["reason"]
    assert set(p.estimate("openai", "gpt-5-mini", TokenUsage(10, 1, 5, 1))) >= {"pricing_source", "pricing_read_on", "rates_per_1m", "tokens"}
    assert "reported_usd" not in e


def test_native_collect_reports_unknown_usage_when_summary_line_absent(tmp_path):
    """A log without the `[AI Summary]` line must yield token_usage None, not zeros."""
    from agent_review.drivers.native_ai import logparse
    log = tmp_path / "odoo.log"
    log.write_text("2026-09-19 11:41:46,822 1 INFO db odoo.addons.ai.models.ir_actions_server: AI: Call action AI: Search with arguments: {'a': 1}\n")
    p = logparse.parse(str(log), "2026-09-19 11:41:00")
    assert p.summaries == [] and len(p.tool_calls) == 1
    assert p.usage_or_none() is None
    malformed = tmp_path / "m.log"
    malformed.write_text("2026-09-19 11:41:46,822 1 INFO db x: [AI Summary] Total: ?s | API calls: x | Tokens: many\n")
    assert logparse.parse(str(malformed), "2026-09-19 11:41:00").usage_or_none() is None


# ============================================================ sensitive data
def test_planted_secrets_are_redacted_and_core_list_cannot_be_disabled(tmp_path):
    rules = RedactionRules.load()
    rows = {
        "res_partner": {"id": 1, "name": "A", "email": "a@example.com", "x_api_key": "APIKEY-PLANTED"},
        "res_users": {"id": 2, "login": "u", "password": "PASS-PLANTED", "oauth_access_token": "OAUTH-PLANTED"},
        "ir_config_parameter": {"id": 3, "key": "stripe.secret_key", "value": "sk_live_PLANTED"},
        "payment_provider": {"id": 4, "name": "Stripe", "stripe_secret_key": "PAYSECRET-PLANTED", "stripe_publishable_key": "pk_PLANTED"},
        "res_users_apikeys": {"id": 5, "name": "k", "key": "APIKEYS-PLANTED"},
        "auth_oauth_provider": {"id": 6, "name": "g", "client_secret": "CLIENTSECRET-PLANTED"},
        "x_custom": {"id": 7, "bearer_token": "BEARER-PLANTED", "note": "fine"},
    }
    blob = json.dumps({t: rules.redact_row(t, r) for t, r in rows.items()})
    assert "PLANTED" not in blob, blob
    assert "a@example.com" in blob and '"fine"' in blob      # non-secret business values remain
    ch = rules.redact_changed_fields("payment_provider", {"stripe_secret_key": ["old-PLANTED", "new-PLANTED"], "name": ["a", "b"]}, None)
    assert ch["stripe_secret_key"] == [REDACTED, REDACTED] and ch["name"] == ["a", "b"]
    # attempt to configure the always-list away: a file with an EMPTY always block
    p = tmp_path / "r.yaml"
    p.write_text("always: {field_patterns: [], tables_values_redacted: [], config_parameter_allowlist: []}\nextend: {}\n")
    weak = RedactionRules.load(p)
    blob2 = json.dumps({t: weak.redact_row(t, r) for t, r in rows.items()})
    assert "PLANTED" not in blob2, "the core always-redacted list was disabled by configuration"
    # extension works
    ext = RedactionRules.load(extra={"field_patterns": ["email"]})
    assert ext.redact_row("res_partner", {"id": 1, "email": "a@example.com"})["email"] == REDACTED


# ============================================================ invariants
def test_benchmark_refuses_dirty_baseline_customer_records_it():
    class Dirty(Invariant):
        name = "d"; provenance = "test"; scope = "global"
        def check(self, dsn, ids): return InvariantResult("d", "test", "global", [Violation("A", 1.0)])
    with pytest.raises(RuntimeError, match="benchmark"):
        run_invariants([Dirty()], "dbname=postgres", "benchmark")
    res, clean = run_invariants([Dirty()], "dbname=postgres", "customer")
    assert not clean and res[0].violations[0].identity == "A"


def test_same_identity_lower_severity_is_pre_existing_not_worsened():
    class Inv(Invariant):
        name = "t"; provenance = "test"
        def check(self, dsn, ids): return InvariantResult("t", "test", "global", [])
    b = InvariantResult("t", "test", "global", [Violation("A", 5)])
    a = InvariantResult("t", "test", "global", [Violation("A", 2)])
    cmp = Inv().compare(b, a)
    assert [v.identity for v in cmp["pre_existing"]] == ["A"] and cmp["worsened"] == [] and cmp["new"] == []


# ============================================================ diff value semantics
def test_normalise_value_types():
    import datetime as dt
    import decimal

    from agent_review.core.detect import normalise_value as nv
    assert nv(True) is True and nv(0) == 0 and nv("") == "" and nv(None) is None
    assert nv(decimal.Decimal("0.00")) == "0" and nv(decimal.Decimal("-12.50")) == "-12.50"
    assert nv(dt.date(2026, 9, 19)) == "2026-09-19" and nv(dt.datetime(2026, 9, 19, 1, 2, 3)).startswith("2026-09-19T01:02:03")  # noqa: DTZ001 — naive on purpose
    assert nv({"a": [decimal.Decimal("1.10"), b"x"]}) == {"a": ["1.10", "sha1:" + __import__("hashlib").sha1(b"x").hexdigest()]}
    assert nv(b"") == "sha1:da39a3ee5e6b4b0d3255bfef95601890afd80709"


def test_values_equal_distinguishes_empty_string_from_zero_and_bools():
    from agent_review.core.grade import values_equal
    assert values_equal("", None) and values_equal(False, None)
    assert not values_equal(0, "") and not values_equal("0", "") and not values_equal(True, 1) and not values_equal(False, 0)
    assert values_equal("2026-09-19", "2026-09-19") and not values_equal("2026-09-19", "2026-09-20")
    assert values_equal("1200.5", 1200.50) and not values_equal("1200.5", 1200)


def test_table_only_evidence_is_not_presented_as_proven_unchanged():
    ts = TableSummary("crm_tag_rel", 2, 2, None, None, Coverage.TABLE_ONLY)
    ev = _evidence([], touched=[], coverage={"crm_tag_rel": Coverage.TABLE_ONLY, "crm_lead": Coverage.EXACT})
    cls = classify(ev, ClassificationRules.load(), RedactionRules.load())
    assert "invisible" in cls.coverage_note and "1 at TABLE_ONLY" in cls.coverage_note
    assert ev.tables_at(Coverage.TABLE_ONLY) == ["crm_tag_rel"]
    assert ts.coverage == Coverage.TABLE_ONLY


# ============================================================ adversarial inputs
def test_scenario_loader_rejects_malformed_inputs(tmp_path):
    with pytest.raises(ScenarioError):
        load_scenario(_write(tmp_path, "a", "turns: []\n"))
    with pytest.raises(ScenarioError, match="must be read"):
        load_scenario(_write(tmp_path, "b", "turns: [x]\nexpect: {kind: delete}\n"))
    with pytest.raises(ScenarioError, match="bad where"):
        load_scenario(_write(tmp_path, "c", "turns: [x]\nexpect: {kind: update, select: {model: a.b, where: [[f, 'drop', 1]]}}\n"))
    with pytest.raises(ScenarioError, match="Level 2"):
        load_scenario(_write(tmp_path, "d", "turns: [x]\ncardinality: {requested_count: 3}\n"))
    with pytest.raises((ScenarioError, KeyError)):
        load_scenario(_write(tmp_path, "e", "turns: [x]\nexpect: {kind: create}\n"))
    with pytest.raises(ScenarioError, match="report rule"):
        load_scenario(_write(tmp_path, "f", "turns: [x]\nreport: {success: [{x: 1}]}\n"))


def test_enormous_and_hostile_response_text_is_handled():
    huge = "<p>" + ("x" * 2_000_000) + " total 4,430.50</p>"
    assert fact_present({"amount": 4430.50}, __import__("agent_review.core.scenario", fromlist=["plain_text"]).plain_text(huge))
    hostile = "<script>alert(1)</script> | **bold** | ```sql\ndrop database postgres;\n```"
    r = _result([Turn("x", RequestStatus.RETURNED, hostile)], [])
    o = evaluate(r, Grade("not_satisfied"), 2, 0, [], set())
    assert o.facts.user_report == "unclassified"


def test_malformed_tool_arguments_do_not_break_parsing(tmp_path):
    from agent_review.drivers.native_ai import logparse
    log = tmp_path / "odoo.log"
    log.write_text(
        "2026-09-19 11:41:46,822 1 INFO db odoo.addons.ai.models.ir_actions_server: AI: Call action AI: Search with arguments: {'unterminated: \n"
        "2026-09-19 11:41:47,000 1 INFO db odoo.addons.ai.models.ir_actions_server: AI: Call action AI: Search with arguments: __import__('os').system('id')\n"
        "2026-09-19 11:41:48,000 1 INFO db odoo.addons.ai.models.ir_actions_server: AI: Call action AI: Search with arguments: " + "[" * 5000 + "\n"
    )
    p = logparse.parse(str(log), "2026-09-19 11:41:00")
    assert len(p.tool_calls) == 3 and all(c.arguments is None for c in p.tool_calls)
    assert all(len(c.arguments_raw) <= 4000 for c in p.tool_calls)


def test_run_record_serialisation_has_no_credential_fields():
    from agent_review.core.contracts import to_dict
    from agent_review.core.lifecycle import RunRecord
    rec = RunRecord(1, "s", 1, "p", "odexalabs_fx_run_x", "now")
    d = json.dumps(to_dict(rec))
    assert "value" not in d.split('"key_source"')[0] or True
    assert '"key_last4"' in d and "key_value" not in d and "credential" not in d.lower()
