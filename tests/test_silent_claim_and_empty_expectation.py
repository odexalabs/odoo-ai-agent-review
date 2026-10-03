"""The silent-claim review flag and the TOOL rule for a CREATE field expected EMPTY, attacked with synthetic runs
shaped like observed ones.

Silent claim — `silent_claim_review` is a flag for a human, computed from facts and never from prose. Each test
     holds the wording fixed and moves one fact, so a test that passes on the wording alone fails.
Empty expectation — a CREATE key field expected EMPTY (`partner_id: null`) used to block the TOOL rule for every
     run in which it failed. The record holds both sides, so the rule must attribute; and it must still refuse
     where the model did set the field, and on UPDATE keys, where leaving a field alone is not clearing it.

Every reply and argument string below is written for the test (none is a recorded model output): a reply that
claims a lead was created with no tool call behind it, and a lead-creation call whose arguments carried the
requested contact while the stored row holds the operator's."""
from __future__ import annotations

import json
import textwrap

import pytest

from agent_review.core.attribution import attribute
from agent_review.core.contracts import (
    ChangeEvidence,
    Coverage,
    DriverRunResult,
    ModelMetadata,
    RequestStatus,
    ResponseRule,
    RowChange,
    TokenUsage,
    ToolCall,
    Turn,
)
from agent_review.core.grade import grade
from agent_review.core.lifecycle import RunRecord
from agent_review.core.outcomes import SILENT_CLAIM_REVIEW, evaluate
from agent_review.core.report import run_summary, suite_summary
from agent_review.core.scenario import Resolved, load_scenario, plain_text

LEAD_TOOL = "ai_tool_create_livechat_lead"
WRITE = {LEAD_TOOL, "ai_tool_update_records", "AI CRM: Create Lead"}

SYNTH_ARGS = ('{"name": "Lead for Alice (Example Customer C)", "contact_name": "Alice", "email": "alice@example.com", '
              '"phone": "+1 202 555 0143", "description": "<p>Wants a performance review.</p>", "team_id": 1, '
              '"tag_ids": [1], "priority": "1", "tool_status": "Creating lead"}')
SYNTH_SUCCESS_REPLY = "<p>Lead added for Alice (alice@example.com, +1 202 555 0143). The sales team will follow up.</p>"
SUCCESS_RULE = ResponseRule("success", "Lead added for Alice", False, "scenario:test")
SYNTH_CLAIM_REPLY = ("<p>All set: the lead for Alice at Example Customer C is now in your pipeline, with her email and phone. "
                     "Shall I schedule a follow-up call?</p>")
CLAIM_FRAGMENT = "the lead for Alice at Example Customer C is now in your pipeline"
SYNTH_ARGS_PY = ("{'name': 'Lead for Alice (Example Customer C)', 'contact_name': 'Alice', 'email': 'alice@example.com', "
                 "'phone': '+1 202 555 0143', 'team_id': 1, 'tag_ids': [1], 'street': None, 'job_position': 'Head of IT'}")
KEY = {"contact_name": "Alice", "email_from": "alice@example.com", "phone": "+1 202 555 0143", "partner_id": None}
RES = Resolved(tables={"crm.lead": "crm_lead", "res.partner": "res_partner"}, create_values=dict(KEY), forbid_ids={0: None})


def _lead_row(partner_id: int = 15) -> RowChange:
    return RowChange("crm_lead", 13, "added", None, {"id": 13, "contact_name": "Eval Operator", "email_from": "operator@example.com",
                                                    "phone": "+1 202 555 0143", "partner_id": partner_id})


@pytest.fixture
def lead_sc(tmp_path):
    p = tmp_path / "lead.yaml"
    p.write_text(textwrap.dedent("""
        turns: ["Create a lead for Alice at Example Customer C."]
        expect:
          kind: create
          model: crm.lead
          count: 1
          match: {}
          values: {contact_name: Alice, email_from: alice@example.com, phone: "+1 202 555 0143", partner_id: null}
        forbid:
          - {model: res.partner, any: true}
    """))
    return load_scenario(p)


def _ev(rows):
    return ChangeEvidence([], rows, {rc.table: Coverage.EXACT for rc in rows}, "test")


def _run(reply, trace, pending="none", returned=True, status=RequestStatus.RETURNED, rules=None):
    t = Turn("Create a lead for Alice at Example Customer C.", status, reply, None if status == RequestStatus.RETURNED else "boom", 1.0, 1,
             pending_interaction=pending, user_request_returned=returned)
    return DriverRunResult("s", [t], ModelMetadata("x", "y", False), trace, TokenUsage(1, 1, 0, 1), known_response_rules=rules or [])


def _no_lead(sc, reply):
    return grade(sc, RES, _ev([]), [reply] if reply else [], 0)


# ======================================================================= A1 · silent_claim_review
@pytest.mark.parametrize("trace", [[], [ToolCall("ai_tool_load_skills", {"skill_ids": [4]}, '{"skill_ids": [4]}')]],
                         ids=["no tool call", "a read tool only"])
def test_a_claimed_completion_with_no_write_call_is_flagged_with_its_final_response_attached(lead_sc, trace):
    g = _no_lead(lead_sc, SYNTH_CLAIM_REPLY)
    assert g.effect == "not_satisfied"
    o = evaluate(_run(SYNTH_CLAIM_REPLY, trace), g, 2, 0, lead_sc.response_rules, WRITE, kind="create")
    assert [f["flag"] for f in o.review_flags] == [SILENT_CLAIM_REVIEW]
    flag = o.review_flags[0]
    assert CLAIM_FRAGMENT in flag["final_response"]
    assert flag["basis"]["write_tool_execution"] == "not_called" and flag["basis"]["user_report"] == "unclassified"


def test_the_flag_is_not_a_verdict(lead_sc):
    """Same run with and without the flag: every fact, the mismatch and the attribution are identical."""
    g = _no_lead(lead_sc, SYNTH_CLAIM_REPLY)
    r = _run(SYNTH_CLAIM_REPLY, [])
    flagged = evaluate(r, g, 2, 0, lead_sc.response_rules, WRITE, kind="create")
    plain = evaluate(r, g, 2, 0, lead_sc.response_rules, WRITE)
    assert flagged.review_flags and not plain.review_flags
    assert flagged.facts == plain.facts and flagged.facts.user_report == "unclassified"        # not set to success
    assert flagged.report_effect_mismatch is None and plain.report_effect_mismatch is None      # not a mismatch
    assert (flagged.interaction_outcome, flagged.business_result) == (plain.interaction_outcome, plain.business_result)
    assert attribute(lead_sc, RES, r, g, flagged, WRITE).layer == attribute(lead_sc, RES, r, g, plain, WRITE).layer == "MODEL"


@pytest.mark.parametrize("error", ["Error: Tool call failed: ValueError('skills {4} are not linked')", None],
                         ids=["called and failed", "called and succeeded, wrong row"])
def test_a_run_that_called_the_write_tool_is_not_flagged(lead_sc, error):
    trace = [ToolCall(LEAD_TOOL, None, SYNTH_ARGS, error)]
    ev = _ev([]) if error else _ev([_lead_row()])
    g = grade(lead_sc, RES, ev, [SYNTH_CLAIM_REPLY], len(ev.row_changes))
    assert g.effect == "not_satisfied"
    o = evaluate(_run(SYNTH_CLAIM_REPLY, trace), g, 2, len(ev.row_changes), lead_sc.response_rules, WRITE, kind="create")
    assert o.review_flags == []


@pytest.mark.parametrize("pending", ["question", "confirmation"])
def test_a_clarification_session_waiting_is_not_flagged_whatever_it_says(lead_sc, pending):
    g = _no_lead(lead_sc, SYNTH_CLAIM_REPLY)
    waiting = evaluate(_run(SYNTH_CLAIM_REPLY, [], pending=pending), g, 2, 0, lead_sc.response_rules, WRITE, kind="create")
    assert waiting.interaction_outcome == "clarification" and waiting.review_flags == []
    # the same words with nothing pending are flagged: the substrate decided, not the text
    assert evaluate(_run(SYNTH_CLAIM_REPLY, [], pending="none"), g, 2, 0, lead_sc.response_rules, WRITE, kind="create").review_flags


def test_a_read_scenario_never_flags(tmp_path):
    p = tmp_path / "read.yaml"
    p.write_text(textwrap.dedent("""
        turns: ["Which invoices are overdue?"]
        expect: {kind: read, facts: {must_contain: [INV/2026/00002]}}
    """))
    sc = load_scenario(p)
    reply = "Done — I have updated the invoices."
    g = grade(sc, Resolved(), _ev([]), [reply], 0)
    assert g.effect == "not_satisfied"
    assert evaluate(_run(reply, []), g, 2, 0, sc.response_rules, WRITE, kind="read").review_flags == []
    assert evaluate(_run(reply, []), g, 2, 0, sc.response_rules, WRITE, kind=None).review_flags == []   # Level 1: no key


def test_an_absent_report_never_flags(lead_sc):
    g = _no_lead(lead_sc, None)
    o = evaluate(_run(None, [], returned=True), g, 2, 0, lead_sc.response_rules, WRITE, kind="create")
    assert o.facts.user_report == "absent" and o.review_flags == []


def test_an_unobservable_trace_is_not_flagged(lead_sc):
    """'No write-tool call' is a claim about the trace; with no trace it cannot be made."""
    g = _no_lead(lead_sc, SYNTH_CLAIM_REPLY)
    o = evaluate(_run(SYNTH_CLAIM_REPLY, None), g, 2, 0, lead_sc.response_rules, WRITE, kind="create")
    assert o.facts.tool_execution == "unobservable" and o.review_flags == []


def test_a_classified_failure_report_is_still_surfaced_as_specified(lead_sc):
    """The specification is `report != absent`, so a report the rules classify as failure is surfaced too.
    Pinned so that narrowing it is a decision, not a drift."""
    rules = [ResponseRule("failure", "The assistant is unavailable right now", False, "substrate:test")]
    reply = "The assistant is unavailable right now"
    g = _no_lead(lead_sc, reply)
    o = evaluate(_run(reply, [], rules=rules), g, 2, 0, lead_sc.response_rules, WRITE, kind="create")
    assert o.facts.user_report == "failure" and [f["flag"] for f in o.review_flags] == [SILENT_CLAIM_REVIEW]


def test_the_summary_reports_the_flag_on_its_own_line(lead_sc):
    records = []
    for idx, (trace, rows) in ((4, ([ToolCall(LEAD_TOOL, json.loads(SYNTH_ARGS), SYNTH_ARGS)], [_lead_row()])), (5, ([], []))):
        reply = SYNTH_SUCCESS_REPLY if rows else SYNTH_CLAIM_REPLY
        ev = _ev(rows)
        g = grade(lead_sc, RES, ev, [reply], len(rows))
        r = _run(reply, trace)
        o = evaluate(r, g, 2, len(rows), lead_sc.response_rules, WRITE, kind="create")
        records.append(RunRecord(idx, "lead-synthetic", 2, "p", "db", "t", status="completed", driver_result=r, grade=g,
                                 outcome=o, attribution=attribute(lead_sc, RES, r, g, o, WRITE)))
    text, data = suite_summary(records, "lead-synthetic", "p", 2)
    line = next(ln for ln in text.splitlines() if ln.startswith("Silent-claim review"))
    assert "runs [5]" in line and "not a verdict" in line
    assert data["review_flags"] == {"silent_claim_review": [5]}
    run5 = run_summary(records[1])
    # the flag keeps its own line in the redacted text and points at the sentence; the sentence itself is free
    # text and stays in the private record, whole
    assert "REVIEW silent_claim_review" in run5
    assert "final response: withheld from this report (free text)" in run5
    assert "005/run.json: outcome.review_flags[0].final_response" in run5
    assert CLAIM_FRAGMENT not in run5 and "alice@example.com" not in run5
    assert records[1].outcome.review_flags[0]["final_response"] == plain_text(SYNTH_CLAIM_REPLY)
    assert "REVIEW" not in run_summary(records[0])


# ======================================================================= A2 · the empty expectation
def _wrong_contact_lead(sc, args_raw, partner_id=15, tool=LEAD_TOOL, error=None):
    ev = _ev([_lead_row(partner_id)])
    g = grade(sc, RES, ev, [SYNTH_SUCCESS_REPLY], 1)
    trace = [ToolCall(tool, None, args_raw, error)]
    if error:   # a later successful read call, so the unrecovered-tool-error rule is not what decides
        trace.append(ToolCall("ai_tool_search", None, '{"model": "crm.lead"}'))
    r = _run(SYNTH_SUCCESS_REPLY, trace, rules=[SUCCESS_RULE])
    o = evaluate(r, g, 2, 1, sc.response_rules, WRITE, kind="create")
    return g, o, attribute(sc, RES, r, g, o, WRITE)


def test_right_arguments_and_a_wrong_row_attribute_tool_naming_what_was_sent_and_what_the_database_holds(lead_sc):
    g, o, a = _wrong_contact_lead(lead_sc, SYNTH_ARGS)
    values = next(x for x in g.assertions if x.name == "create.values")
    assert values.failed_fields == ["contact_name", "email_from", "partner_id"]
    assert values.observed == {"contact_name": ["Eval Operator"], "email_from": ["operator@example.com"], "partner_id": [15]}
    assert o.report_effect_mismatch["direction"] == "false_success"
    assert a.layer == "TOOL", a
    text = " | ".join(a.evidence)
    assert "contact_name='Alice'" in text and "email_from='alice@example.com'" in text       # what the model sent
    assert "left ['partner_id'] unset" in text and "partner_id=[15]" in text                   # what it did not set
    assert "db='Eval Operator'" in text and "partner_id: db=15" in text                        # what the database holds


def test_python_repr_arguments_as_the_odoo19_log_writes_them_attribute_tool(lead_sc):
    """Python-repr arguments, as the Odoo 19 log writes them; the operator's partner 14 on the lead."""
    _, _, a = _wrong_contact_lead(lead_sc, SYNTH_ARGS_PY, partner_id=14, tool="AI CRM: Create Lead")
    assert a.layer == "TOOL", a


@pytest.mark.parametrize("args_raw", [
    SYNTH_ARGS[:-1] + ', "partner_id": 15}',                 # the model sent the partner the database holds
    SYNTH_ARGS[:-1] + ', "partner_id": "Eval Operator"}',    # the model named a partner, by name
    SYNTH_ARGS[:-1] + ', "customer": 15}',                   # the database's value under another key
], ids=["sent the id", "sent a name", "sent the value under another key"])
def test_an_empty_expectation_the_model_did_set_still_blocks_tool(lead_sc, args_raw):
    _, _, a = _wrong_contact_lead(lead_sc, args_raw)
    assert a.layer != "TOOL", a


def test_an_errored_write_call_establishes_nothing(lead_sc):
    _, _, a = _wrong_contact_lead(lead_sc, SYNTH_ARGS, error="Error: Tool call failed: boom")
    assert a.layer != "TOOL", a


@pytest.mark.parametrize("error,layer_is_tool", [(None, True), ("Error: Tool call failed: boom", False)],
                         ids=["the call succeeded", "the only write call errored"])
def test_when_only_the_empty_expectation_fails_the_call_must_have_succeeded(lead_sc, error, layer_is_tool):
    """Contact stored correctly, lead linked to partner 15 that no argument carried: TOOL when the call
    succeeded; nothing when the model's only write call errored (the row came from somewhere else)."""
    row = RowChange("crm_lead", 13, "added", None, {"id": 13, "contact_name": "Alice", "email_from": "alice@example.com",
                                                   "phone": "+1 202 555 0143", "partner_id": 15})
    g = grade(lead_sc, RES, _ev([row]), ["Done."], 1)
    assert next(x for x in g.assertions if x.name == "create.values").failed_fields == ["partner_id"]
    trace = [ToolCall(LEAD_TOOL, None, SYNTH_ARGS, error)] + ([ToolCall("ai_tool_search", None, "{}")] if error else [])
    r = _run("Done.", trace)
    o = evaluate(r, g, 2, 1, lead_sc.response_rules, WRITE, kind="create")
    assert (attribute(lead_sc, RES, r, g, o, WRITE).layer == "TOOL") is layer_is_tool


def test_an_update_key_expecting_an_empty_field_still_blocks_tool(tmp_path):
    """Clearing a field needs the call to send the empty value; leaving it alone is not clearing it."""
    p = tmp_path / "upd.yaml"
    p.write_text(textwrap.dedent("""
        turns: ["Unassign the Example Customer F opportunity."]
        expect:
          kind: update
          select: {model: crm.lead, where: [[id, "=", 4]]}
          values: {user_id: null}
    """))
    sc = load_scenario(p)
    res = Resolved(tables={"crm.lead": "crm_lead"}, update_ids=[4], update_values={"user_id": None})
    ev = _ev([RowChange("crm_lead", 4, "changed", {"id": 4, "user_id": 7, "stage_id": 1}, {"id": 4, "user_id": 7, "stage_id": 3},
                        {"stage_id": [1, 3]})])
    g = grade(sc, res, ev, ["Done."], 1)
    values = next(x for x in g.assertions if x.name == "update.values")
    assert values.failed_fields == ["user_id"] and values.observed == {"user_id": [7]}
    raw = '{"model": "crm.lead", "ids": [4], "values": {"stage_id": 3}}'
    r = _run("Done.", [ToolCall("ai_tool_update_records", json.loads(raw), raw)])
    o = evaluate(r, g, 2, 1, sc.response_rules, WRITE, kind="update")
    assert attribute(sc, res, r, g, o, WRITE).layer != "TOOL"
