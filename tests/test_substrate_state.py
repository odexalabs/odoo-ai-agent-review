"""`clarification` and `absent` are read from substrate state, not from prose.
Every test holds the wording fixed against the rule, or the rule fixed against the wording."""
from __future__ import annotations

import pytest

from agent_review.core.contracts import (
    DriverRunResult,
    ModelMetadata,
    RequestStatus,
    ResponseRule,
    TokenUsage,
    ToolCall,
    Turn,
)
from agent_review.core.grade import Grade
from agent_review.core.outcomes import evaluate

CONFIRM_RULE = [ResponseRule("clarification", "confirm", True, "scenario")]


def _run(text, pending=None, returned=None, status=RequestStatus.RETURNED, trace=None, known=None):
    t = Turn("do it", status, text, None if status == RequestStatus.RETURNED else "boom", 1.0, 1,
             pending_interaction=pending, user_request_returned=returned)
    return DriverRunResult("s", [t], ModelMetadata("x", "y", False), trace if trace is not None else [], TokenUsage(1, 1, 0, 1),
                           known_response_rules=known or [])


@pytest.mark.parametrize("wording", ["Bananas.", "", "I have updated everything.", "Choose an option below."])
def test_waiting_for_an_answer_is_clarification_whatever_the_wording(wording):
    o = evaluate(_run(wording, pending="question", returned=True), Grade("not_satisfied"), 2, 0, CONFIRM_RULE, set())
    assert o.interaction_outcome == "clarification" and o.business_result == "not_reached"
    assert o.reason == "substrate: session waiting for question"


def test_waiting_for_a_confirmation_is_clarification():
    o = evaluate(_run("Here is the preview.", pending="confirmation", returned=True), Grade("not_satisfied"), 2, 0, [], set())
    assert o.interaction_outcome == "clarification" and o.reason == "substrate: session waiting for confirmation"


def test_text_asking_to_confirm_is_NOT_clarification_when_the_substrate_says_nothing_is_pending():
    o = evaluate(_run("Please confirm you want me to proceed?", pending="none", returned=True),
                 Grade("not_satisfied"), 2, 0, CONFIRM_RULE, set())
    assert o.interaction_outcome == "completed" and o.business_result == "incorrect"


def test_prose_rule_still_applies_only_where_the_substrate_cannot_report():
    # Odoo 19 exposes no pending-interaction state: the scenario's prose rule is the fallback
    o = evaluate(_run("Please confirm the details", pending=None), Grade("not_satisfied"), 2, 0, CONFIRM_RULE, set())
    assert o.interaction_outcome == "clarification" and o.reason.startswith("scenario")


def test_waiting_does_not_override_a_satisfied_key():
    o = evaluate(_run("Done — anything else?", pending="question", returned=True), Grade("satisfied"), 2, 3, [], set())
    assert o.interaction_outcome == "completed" and o.business_result == "correct"


@pytest.mark.parametrize("status", [RequestStatus.ERROR, RequestStatus.TIMEOUT])
def test_request_returned_cleanly_and_nothing_posted_is_ABSENT_even_if_the_loop_failed(status):
    # Odoo 20 late ack / rejected callback: the loop failed after the user's request had returned
    o = evaluate(_run(None, pending="none", returned=True, status=status), Grade("not_satisfied"), 2, 0, [], set())
    assert o.facts.user_report == "absent"
    assert o.facts.user_report_rule.startswith("substrate: the user's request returned")
    assert o.report_effect_mismatch is None          # absent never produces a mismatch


def test_the_users_own_request_raising_is_FAILURE():
    # Odoo 19: the loop ran inside the user's request, so its error reached the user
    o = evaluate(_run(None, pending=None, returned=False, status=RequestStatus.ERROR), Grade("not_satisfied"), 2, 0, [], set())
    assert o.facts.user_report == "failure" and o.facts.user_report_rule.startswith("substrate: the user's own request raised")


def test_unknown_request_state_keeps_the_old_inference_and_says_so():
    o = evaluate(_run(None, pending=None, returned=None, status=RequestStatus.ERROR), Grade("not_satisfied"), 2, 0, [], set())
    assert o.facts.user_report == "failure" and o.facts.user_report_rule.startswith("inferred:")


def test_a_posted_message_is_still_classified_by_its_deterministic_rules():
    known = [ResponseRule("failure", "Oops, it looks like our AI is unreachable", False, "substrate:test")]
    o = evaluate(_run("Oops, it looks like our AI is unreachable", pending="none", returned=True, known=known),
                 Grade("not_satisfied"), 2, 0, [], set())
    assert o.facts.user_report == "failure"


def test_unrecovered_tool_error_still_wins_when_nothing_is_pending():
    trace = [ToolCall("ai_tool_update_records", {}, "{}", "Access denied")]
    o = evaluate(_run("I could not do it.", pending="none", returned=True, trace=trace), Grade("not_satisfied"), 2, 0, [],
                 {"ai_tool_update_records"})
    assert o.interaction_outcome == "tool_error_unrecovered"
