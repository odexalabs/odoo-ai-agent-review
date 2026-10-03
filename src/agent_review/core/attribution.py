"""Failure attribution. ONE primary layer per run in scope, assigned only when DIRECT
EVIDENCE COVERS THE BOUNDARY WHERE BEHAVIOUR DIVERGED. Otherwise `unattributable`.

  MODEL           prompt, response, and the tool decision it made
  TOOL            tool arguments + database effect
  ORCHESTRATION   runtime trace / control flow + transaction, report and effect
  TRANSPORT       the provider or network exception, with its source
  CORE            a correct tool invocation + an invalid resulting business state

In scope: business result incorrect · unrecovered execution failures · timeout / error ·
report/effect mismatches. A clarification or refusal records its REASON and is attributed only
where direct evidence makes it meaningful.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .contracts import DriverRunResult
from .grade import Grade
from .outcomes import Outcome
from .scenario import Resolved, Scenario

# A provider or network exception, named as such. Bare digits ("429") and the word "timeout" alone
# are not signatures: a record id can be 429, and the harness's own client timeout says nothing
# about the provider.
TRANSPORT_RE = re.compile(
    r"ReadTimeout|ConnectTimeout|ConnectionError|ConnectionReset|RemoteDisconnected|Connection refused|"
    r"Name or service not known|read timeout=|\bHTTP(?:Error)?[: ]+(?:429|5\d\d)\b|\bstatus(?: code)? (?:429|5\d\d)\b|"
    r"\b(?:429|5\d\d) (?:Too Many Requests|Internal Server Error|Bad Gateway|Service Unavailable|Gateway Time-?out)|"
    r"rate.?limit|api\.openai\.com|generativelanguage\.googleapis\.com|api\.anthropic\.com",
    re.IGNORECASE,
)
HARNESS_TIMEOUT_PREFIX = "harness timeout:"
RUNTIME_EARLY_TERMINATION = "terminate early with empty message"


@dataclass
class Attribution:
    layer: str                       # MODEL | TOOL | ORCHESTRATION | TRANSPORT | CORE | unattributable | not_in_scope
    evidence: list[str] = field(default_factory=list)
    rule: str | None = None


def _is_empty(v) -> bool:
    return v is None or v is False or v == ""


def _in_args(v, blob: str) -> bool:
    """Is the value verbatim in the arguments? Numbers on digit boundaries: 14 is not in 2014."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return re.search(r"(?<![\d.])" + re.escape(str(v)) + r"(?![\d.])", blob) is not None
    return str(v) in blob


def _write_args(result: DriverRunResult, write_tools: set[str]) -> str:
    return " ".join(c.arguments_raw for c in (result.tool_trace or []) if c.name in write_tools and not c.error)


def _args_contain(result: DriverRunResult, values: dict, write_tools: set[str]) -> tuple[list[str], list[str]]:
    """Which expected values appear verbatim in the arguments of a successful write-tool call."""
    present, absent = [], []
    blob = _write_args(result, write_tools)
    for k, v in values.items():
        if _is_empty(v):
            continue
        (present if _in_args(v, blob) else absent).append(f"{k}={v!r}")
    return present, absent


def _left_unset(result: DriverRunResult, field_name: str, db_values: list, write_tools: set[str]) -> bool:
    """Did the model's successful write-tool calls leave `field_name` alone? Yes when no call names it as
    an argument AND none carries any value the database now holds for it. Needs at least one successful
    call and the database's value: without either, nothing is established."""
    if not any(c.name in write_tools and not c.error for c in (result.tool_trace or [])):
        return False
    held = [v for v in db_values if not _is_empty(v)]
    if not held:
        return False
    blob = _write_args(result, write_tools)
    if re.search(r"[\"']" + re.escape(field_name) + r"[\"']\s*:", blob):
        return False
    return not any(_in_args(v, blob) for v in held)


def attribute(sc: Scenario, res: Resolved | None, result: DriverRunResult, grade: Grade, outcome: Outcome,
              write_tools: set[str], invariants_new: int = 0) -> Attribution:
    in_scope = (
        outcome.business_result == "incorrect"
        or outcome.interaction_outcome in ("tool_error_unrecovered", "timeout", "error")
        or (outcome.report_effect_mismatch or {}).get("direction") in ("false_success", "false_failure")
    )
    if not in_scope:
        return Attribution("not_in_scope", rule="clarification/refusal/correct: reason recorded, no attribution attempted"
                           if outcome.interaction_outcome in ("clarification", "refused") else None)

    # TRANSPORT: the exception names the provider or the network. The harness's own client timeout
    # is not such evidence; only a provider exception the substrate surfaced (in the error or in
    # the runtime log) is.
    if outcome.interaction_outcome in ("timeout", "error") and result.error:
        err = result.error
        substrate_error = err if not err.startswith(HARNESS_TIMEOUT_PREFIX) else ""
        log_hits = [n for n in result.driver_notes if TRANSPORT_RE.search(n)]
        if substrate_error and TRANSPORT_RE.search(substrate_error):
            return Attribution("TRANSPORT", [f"exception: {err[:300]}"], "request error names a provider/network exception")
        if log_hits:
            return Attribution("TRANSPORT", [f"request: {err[:160]}", f"runtime log: {log_hits[0][-200:]}"],
                               "provider/network exception in the runtime log")
        if err.startswith(HARNESS_TIMEOUT_PREFIX):
            return Attribution("unattributable", [f"harness client timeout: {err[:200]}",
                                                  "no provider or network exception observed; the substrate may simply have been slow"], None)

    trace_ok = result.tool_trace is not None

    # ORCHESTRATION, internal path: the runtime loop terminated on an empty end message, raised
    # "no response", the request errored and the write was rolled back — a successful write tool
    # call in the trace with no surviving effect (observed on Odoo 19)
    if outcome.interaction_outcome == "error" and trace_ok:
        runtime = [n for n in result.driver_notes if RUNTIME_EARLY_TERMINATION in n]
        write_ok = [c for c in result.tool_trace if c.name in write_tools and not c.error]
        if runtime and write_ok and outcome.facts.transaction == "rolled_back":
            return Attribution("ORCHESTRATION", [f"write tool {write_ok[-1].name} succeeded per the trace",
                                                 f"runtime: {runtime[0][-120:]}", f"request error: {(result.error or '')[:160]}",
                                                 f"transaction: {outcome.facts.transaction_basis}"],
                               "runtime loop ended with no response after a successful write; request re-raised; rolled back")

    # ORCHESTRATION: the runtime loop contradicted the model — effect satisfied, user told failure,
    # and the runtime's own early-termination line is in the driver notes (observed on Odoo 19)
    mm = (outcome.report_effect_mismatch or {}).get("direction")
    if mm == "false_failure":
        runtime = [n for n in result.driver_notes if RUNTIME_EARLY_TERMINATION in n]
        if runtime:
            return Attribution("ORCHESTRATION", [f"effect satisfied and committed ({outcome.facts.transaction_basis})",
                                                 f"user-facing report: {outcome.facts.user_report_rule}",
                                                 f"runtime: {runtime[0][:200]}"],
                               "false failure with the runtime's early-termination line present")
        return Attribution("unattributable", ["false failure, but no runtime trace explains the report"], None)

    # TOOL vs MODEL on write keys: did the model send the expected values?
    expected_values = {}
    if res is not None:
        if sc.create:
            expected_values = dict(res.create_values)
        elif sc.update:
            expected_values = dict(res.update_values)
    if expected_values and trace_ok:
        present, absent = _args_contain(result, expected_values, write_tools)
        failed_values = [a for a in grade.assertions if a.name in ("create.values", "update.values") and not a.passed]
        failed_fields = {f for a in failed_values for f in a.failed_fields}
        present_fields = {p.split("=", 1)[0] for p in present}
        absent_fields = {p.split("=", 1)[0] for p in absent}
        write_calls = [c for c in result.tool_trace if c.name in write_tools]
        # A CREATE key can expect a field to stay EMPTY (`partner_id: null`: the lead is not linked to the
        # operator). Such a field is never "sent", so it could never join `present_fields`, and whenever it
        # failed the TOOL rule could not fire: such runs went `unattributable` in earlier versions although
        # their records hold both sides. On a
        # CREATE the empty expectation is met by leaving the field alone, so the call is consistent with it
        # when no successful write-tool call names the field or carries the value the database holds. Not
        # on an UPDATE: clearing a field needs the call to SEND the empty value, which a verbatim search over
        # the arguments cannot establish, so there it still blocks the rule.
        held = {f: vals for a in failed_values for f, vals in (a.observed or {}).items()}
        left_unset = {f for f in failed_fields if sc.create and f in expected_values and _is_empty(expected_values[f])
                      and _left_unset(result, f, held.get(f, []), write_tools)}
        if failed_values and failed_fields and failed_fields <= present_fields | left_unset:
            # the model sent every failed field it could send, verbatim, and set none of the fields that had
            # to stay empty; the database holds something else
            sent = [p for p in present if p.split("=", 1)[0] in failed_fields]
            unset_text = (f"model left {sorted(left_unset)} unset: no successful write-tool call names "
                          f"{'it' if len(left_unset) == 1 else 'them'} or carries the value the database holds "
                          f"({', '.join(f'{f}={held[f]!r}' for f in sorted(left_unset))})")
            unset_line = [unset_text] if left_unset else []
            if invariants_new:
                return Attribution("CORE", [f"model sent {sent}", *unset_line, failed_values[0].detail,
                                            f"{invariants_new} NEW invariant violation(s)"], "correct call, invalid business state")
            return Attribution("TOOL", ([f"model sent {sent} (verbatim in the write tool arguments)"] if sent else []) + unset_line
                               + [f"database: {failed_values[0].detail}"], "arguments correct, effect wrong")
        if outcome.business_result == "incorrect" and failed_values and failed_fields and failed_fields & absent_fields:
            missing = sorted(failed_fields & absent_fields)
            return Attribution("MODEL", [f"fields that failed and were never sent to a write tool: {missing}",
                                         f"sent: {present}", f"database: {failed_values[0].detail}"],
                               "the model did not send the expected value for the field that failed")
        if outcome.business_result == "incorrect" and not write_calls:
            return Attribution("MODEL", ["no write tool was called", f"final response: {(result.final_response or '')[:200]!r}"],
                               "the model never invoked a write tool")

    # READ incorrect with a trace in which every tool call succeeded. The native substrate exposes
    # tool ARGUMENTS but not tool RESULTS, so "the tool ran what it was given" is unproven: the
    # MODEL->TOOL boundary is covered, TOOL->EFFECT is not. Record the arguments for a human and
    # do not assert the layer: a person can, by reading the recorded domain.
    if (sc.kind == "read" and outcome.business_result == "incorrect" and trace_ok
            and result.tool_trace and result.tool_trace[-1].error is None):
        calls = [f"{c.name}({c.arguments_raw[:160]})" for c in result.tool_trace if not c.error][-3:]
        return Attribution("unattributable", ["tool calls returned without error; tool results are not observable on this substrate",
                                              "arguments recorded for manual attribution:", *calls, f"facts: {grade.facts_found}"], None)

    if outcome.interaction_outcome == "tool_error_unrecovered" and trace_ok:
        last = result.tool_trace[-1]
        return Attribution("TOOL", [f"last tool call {last.name} errored: {last.error[:300] if last.error else ''}"],
                           "unrecovered tool error")

    if mm == "false_success" and not trace_ok:
        return Attribution("unattributable", ["false success; tool trace unobservable on this substrate"], None)
    return Attribution("unattributable", ["no direct evidence covers the boundary where behaviour diverged"], None)
