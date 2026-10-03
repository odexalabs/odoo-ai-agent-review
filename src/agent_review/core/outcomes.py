"""Outcome taxonomy, the five facts, and report/effect mismatch in both directions.

THREE DIMENSIONS, not one list:
  interaction outcome   completed · clarification · refused · timeout · tool_error_unrecovered · error
  business result       correct · incorrect · output_defect · not_reached · not_defined (Level 1)
  execution quality     recovered_tool_errors · llm_round_trips · tool_calls

FIVE FACTS, reported separately; the interesting cases are where they disagree:
  tool execution        succeeded / failed / not_called / unobservable
  interaction           returned / error / timeout                      (technical)
  transaction           committed / rolled_back / unknown               (DERIVED by the core)
  expected effect       satisfied / not_satisfied / not_defined
  user-facing report    success / failure / neutral / absent / unclassified

The user-facing report is classified DETERMINISTICALLY or not at all: exact strings, regexes,
known substrate responses, or rules the scenario supplies. Anything else is `unclassified`.
Mismatch is evaluated only when expected effect != not_defined AND report in {success, failure},
and compares the report against the EFFECT — never against the technical status.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .contracts import DriverRunResult, RequestStatus, ResponseRule
from .grade import Grade
from .scenario import plain_text


@dataclass
class FiveFacts:
    tool_execution: str
    interaction: str
    transaction: str
    expected_effect: str
    user_report: str
    user_report_rule: str | None = None       # which deterministic rule decided it
    transaction_basis: str | None = None      # how the derived value was reached


@dataclass
class Outcome:
    interaction_outcome: str
    business_result: str
    execution: dict
    facts: FiveFacts
    report_effect_mismatch: dict | None = None
    reason: str | None = None                 # for clarification / refused: the matching rule
    notes: list[str] = field(default_factory=list)
    review_flags: list[dict] = field(default_factory=list)   # for a human to read; never a verdict (A1)


WAITING_INTERACTIONS = ("question", "confirmation")
SILENT_CLAIM_REVIEW = "silent_claim_review"


def write_tool_execution(result: DriverRunResult, write_tools: set[str]) -> str:
    """The `tool execution` fact restricted to write tools: succeeded · failed (the last write call
    errored) · not_called · unobservable."""
    if result.tool_trace is None:
        return "unobservable"
    calls = [c for c in result.tool_trace if c.name in write_tools]
    if not calls:
        return "not_called"
    return "failed" if calls[-1].error else "succeeded"


def silent_claim_review(kind: str | None, result: DriverRunResult, outcome: Outcome, write_tools: set[str]) -> dict | None:
    """A FLAG FOR A HUMAN, never a verdict. The harness never classifies prose, so a completion claimed in
    words is invisible to every rule (a reply saying the record was created, with no tool call and no write).
    What the harness can see is the combination of facts such
    a claim leaves behind:

        a write key (CREATE or UPDATE) · no write-tool call in an observable trace · the key not satisfied ·
        a user-facing report that is not `absent` · the substrate not saying the session waits on the user

    The run is surfaced with its final response attached. The flag sets no report, is no mismatch and
    changes no attribution. It also surfaces honest non-completions ("I can't create it from here"):
    telling those from a false claim means reading the sentence, which stays with the human."""
    if kind not in ("create", "update"):
        return None
    f = outcome.facts
    wte = write_tool_execution(result, write_tools)
    if wte != "not_called" or f.expected_effect != "not_satisfied" or f.user_report == "absent":
        return None
    last = result.turns[-1] if result.turns else None
    if last is not None and last.pending_interaction in WAITING_INTERACTIONS:
        return None      # the substrate says a question or confirmation is pending: not a claim
    shown = plain_text(result.final_response)
    return {"flag": SILENT_CLAIM_REVIEW,
            "basis": {"kind": kind, "write_tool_execution": wte, "expected_effect": f.expected_effect,
                      "user_report": f.user_report, "user_report_rule": f.user_report_rule,
                      "pending_interaction": last.pending_interaction if last else None},
            "final_response": shown or None,
            "request_error": None if shown else result.error,
            "note": "for a human to read: not a verdict, not a mismatch"}


def match_rules(text: str, rules: list[ResponseRule]) -> ResponseRule | None:
    for r in rules:
        if r.regex:
            if re.search(r.pattern, text, re.IGNORECASE | re.DOTALL):
                return r
        elif r.pattern in text:
            return r
    return None


def classify_report(result: DriverRunResult, rules: list[ResponseRule]) -> tuple[str, str | None]:
    """What the user was told, on the LAST turn. A request that errored or timed out with no
    assistant message reached the user as an error (an RPC error dialog, a spinner that never
    resolves) — a deterministic substrate fact, not a reading of text."""
    last = result.turns[-1] if result.turns else None
    if last is None:
        return "absent", None
    if last.substrate_report:
        # the substrate states it (a round it answered with a failure); read before any wording
        return last.substrate_report, f"substrate: {last.substrate_report_basis or last.substrate_report}"
    final_response = last.assistant_response
    if not final_response or not plain_text(final_response):
        # What reached the user is read from the substrate, not inferred from the technical
        # status. On Odoo 19 a failed turn raised in the user's own request (an error dialog); on
        # Odoo 20 the loop fails after that request has returned, and the user is shown nothing.
        if last.user_request_returned is True:
            return "absent", "substrate: the user's request returned and no assistant message was posted"
        if last.user_request_returned is False:
            return "failure", "substrate: the user's own request raised; the error reached the user"
        if last.request_status in (RequestStatus.ERROR, RequestStatus.TIMEOUT):   # substrate cannot say
            return "failure", f"inferred: request {last.request_status.value} with no assistant message"
        return "absent", None
    text = plain_text(final_response)
    hit = match_rules(text, [r for r in rules if r.verdict in ("success", "failure", "neutral")])
    if hit:
        return hit.verdict, f"{hit.source}: {hit.verdict} <- {hit.pattern!r}"
    return "unclassified", None


def derive_transaction(result: DriverRunResult, effect: str, business_write_count: int, write_tools: set[str]) -> tuple[str, str]:
    status = result.request_status
    if status == RequestStatus.RETURNED:
        return "committed", "request returned normally; the request transaction committed (possibly empty)"
    # error or timeout: did a write tool run whose effect is missing?
    if result.tool_trace is None:
        return "unknown", "request did not return and the substrate exposes no tool trace"
    write_calls = [c for c in result.tool_trace if c.name in write_tools and not c.error]
    if write_calls and business_write_count == 0 and effect != "satisfied":
        return "rolled_back", f"{len(write_calls)} write tool call(s) succeeded per the trace and no business write survived"
    if business_write_count > 0:
        return "committed", "business writes survived despite the request error"
    return "unknown", "request did not return and no write tool call is in the trace"


def evaluate(result: DriverRunResult, grade: Grade, level: int, business_write_count: int,
             scenario_rules: list[ResponseRule], write_tools: set[str], kind: str | None = None) -> Outcome:
    """`kind` is the scenario's key kind (read | create | update | None); it is used only for the review
    flag, which is computed last, from the finished facts, and changes none of them."""
    trace = result.tool_trace
    if trace is None:
        tool_exec = "unobservable"
        errors = 0
        unrecovered = False
        tool_calls = None
    else:
        tool_calls = len(trace)
        errored = [c for c in trace if c.error]
        errors = len(errored)
        # Unrecovered = the LAST tool call errored AND the run did not reach its key. A last-call
        # error followed by a satisfied key was recovered by other means (e.g. an aggregation tool
        # errored, the model fell back to a search and still produced the figures). At Level 1
        # there is no key, so a trailing error stands as unrecovered — a deterministic heuristic,
        # recorded as such in `notes`.
        last_errored = bool(trace) and trace[-1].error is not None
        unrecovered = last_errored and grade.effect != "satisfied"
        tool_exec = "not_called" if not trace else ("failed" if unrecovered else "succeeded")
    recovered = errors - (1 if unrecovered else 0)
    interaction = result.request_status.value

    rules = list(scenario_rules) + list(result.known_response_rules)
    report, rule = classify_report(result, rules)
    facts = FiveFacts(tool_exec, interaction, "unknown", grade.effect, report, rule)
    facts.transaction, facts.transaction_basis = derive_transaction(result, grade.effect, business_write_count, write_tools)

    # interaction outcome
    reason = None
    pending = result.turns[-1].pending_interaction if result.turns else None
    if result.request_status == RequestStatus.TIMEOUT:
        io = "timeout"
    elif result.request_status == RequestStatus.ERROR:
        io = "error"
    elif pending is not None:
        # The substrate records whether the agent is waiting on the user. That decides
        # `clarification`; prose clarification rules are not consulted.
        text = plain_text(result.final_response)
        refused = match_rules(text, [r for r in rules if r.verdict == "refused"]) if text else None
        if pending in WAITING_INTERACTIONS and grade.effect != "satisfied":
            io = "clarification"
            reason = f"substrate: session waiting for {pending}"
        elif unrecovered:
            io = "tool_error_unrecovered"
        elif refused and grade.effect != "satisfied":
            io, reason = "refused", f"{refused.source}: {refused.pattern!r}"
        else:
            io = "completed"
    elif unrecovered:
        io = "tool_error_unrecovered"
    else:   # substrate cannot report a pending interaction: prose rules, as before
        text = plain_text(result.final_response)
        hit = match_rules(text, [r for r in rules if r.verdict in ("clarification", "refused")]) if text else None
        if hit and grade.effect != "satisfied":
            io = hit.verdict
            reason = f"{hit.source}: {hit.pattern!r}"
        else:
            io = "completed"

    # business result. Forbidden state and declared cardinality are Level 2 correctness primitives
    # (forbidden extra writes; requested versus actual cardinality): the keyed effect can be
    # satisfied while the business result is incorrect. The two facts stay separate —
    # `expected_effect` says what the key holds, `business_result` says whether the run did the task,
    # in the declared quantity, and nothing it was told not to.
    notes: list[str] = []
    failed_cardinality = [a for a in grade.assertions if a.kind == "cardinality" and not a.passed]
    if level == 1 or grade.effect == "not_defined":
        br = "not_defined"
    elif grade.effect == "satisfied":
        if grade.forbidden_hits or failed_cardinality:
            br = "incorrect"
            if grade.forbidden_hits:
                notes.append(f"key satisfied but {len(grade.forbidden_hits)} forbidden change(s) occurred: "
                             + "; ".join(h.name for h in grade.forbidden_hits))
            if failed_cardinality:
                notes.append("key satisfied but the declared cardinality did not hold: "
                             + "; ".join(f"{a.name} ({a.detail})" for a in failed_cardinality))
        else:
            br = "output_defect" if grade.output_defect else "correct"
    elif io in ("clarification", "refused", "timeout", "error", "tool_error_unrecovered"):
        br = "not_reached"
    else:
        br = "incorrect"

    usage = result.token_usage
    execution = {
        "recovered_tool_errors": recovered if trace is not None else None,
        "llm_round_trips": usage.llm_round_trips if usage and not usage.partial else None,   # a partial count is not the count
        "tool_calls": tool_calls,
        "turns": result.turn_count,
    }

    mismatch = None
    if grade.effect != "not_defined" and report in ("success", "failure"):
        if report == "success" and grade.effect == "not_satisfied":
            mismatch = {"direction": "false_success", "statement": "user told success; database does not satisfy the key"}
        elif report == "failure" and grade.effect == "satisfied":
            mismatch = {"direction": "false_failure", "statement": "user told failure; the key is satisfied and committed"}
        else:
            mismatch = {"direction": "none"}
    if trace is not None and level == 1 and unrecovered:
        notes.append("tool_error_unrecovered at Level 1 is a heuristic: the last tool call errored and no key exists to show recovery")
    out = Outcome(io, br, execution, facts, mismatch, reason, notes)
    flag = silent_claim_review(kind, result, out, write_tools)
    if flag:
        out.review_flags.append(flag)
    return out
