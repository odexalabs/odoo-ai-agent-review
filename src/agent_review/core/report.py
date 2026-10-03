"""Repetition and reporting. Each individual run passes or fails its deterministic
assertions and is reported per run. The aggregate is THREE INDEPENDENT DIMENSIONS, never one
verdict, with N beside every fraction and the distribution of failures:

    Business correctness   Level 2 only — requires expected state
    Safety profile         Level 1 and 2
    Invariant state        Level 1 and 2

At small N the output is a DESCRIPTIVE COMPARISON ONLY. Never an aggregate reliability
percentage, never a binary verdict, never a claim about chance.

THE RECORD IS PRIVATE; THE REPORT IS REDACTED — NOT CLEARED FOR SHARING. Everything this module writes
or returns — summary.txt, summary.json, the per-run text the CLI prints — is built from `redacted_run()`,
an allowlist projection of the private RunRecord: a field reaches the report only where this module names
it (default-deny). It keeps statuses, counts, ids, table / model / field / tool names, constant harness
templates, configuration text (scenario facts, report-rule patterns, rule rationales) and structured
values that pass RedactionRules by table and field (the normalised diff; a grade's observed and expected
values). It WITHHOLDS every string an agent, a fixture or the substrate wrote — the prompt and replies,
tool arguments, errors, driver and log notes, attribution evidence, the silent-claim flag's final
response, SQL text, invariant detail — and a harness string that interpolates one (a values
assertion's detail is rebuilt from redacted parts instead). A withheld value is replaced by a marker
naming where the private run.json holds it. Free text is withheld, never pattern-scrubbed: a scrubber
that knows `sk_live_` misses every secret it does not know. Nothing here mutates the record.

Redaction is by field NAME, so business values stay visible: a customer name, a confidential
description, a secret someone typed into an ordinary field. The report is reviewed by a person
before it leaves the team (README, "Artifacts")."""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

from .classify import NormalisedChange
from .contracts import to_dict
from .lifecycle import RunRecord
from .redact import RedactionRules
from .safety import RuleResult

REDACTED_REPORT = "redacted-report"
PRIVATE_EVIDENCE_NOTE = ("PRIVATE, never share or attach: every NNN/run.json (the complete record: conversation, tool "
                         "arguments, errors, expected values, attribution evidence), NNN/raw_diff.json (the unredacted "
                         "diff) and every other file under NNN/ (driver logs and config, e.g. odoo.log, odoo.conf). "
                         "REDACTED REPORT, review before sharing outside the team (business-field values remain "
                         "visible): summary.txt and summary.json, built by core/report.py.")
INCOMPLETE = ("unavailable", "not_evaluable")

_CONDITION_KEYS = ("odoo_version", "odoo_build", "fixture", "fixture_backend", "benchmark_date", "egress_isolation",
                   "egress_allowed_endpoints", "mail_blocked_verified", "key_source", "key_last4", "host",
                   "driver", "transport", "scope")
_SESSION_KEYS = ("session_id", "kind", "uid", "company_id", "agent", "agent_id", "llm_model", "model_configured",
                 "price_as", "standin_mode", "ai_session_id", "model_selected_by")
_FLAG_BASIS_KEYS = ("kind", "write_tool_execution", "expected_effect", "user_report", "user_report_rule", "pending_interaction")
_COST_KEYS = ("label", "usd", "at_least_usd", "reason", "pricing_source", "pricing_read_on", "rates_per_1m", "tokens",
              "tokens_partial", "reported_usd", "provider_calls", "priced_as", "usage_basis")
_MAIL_KEYS = ("servers_disabled_on_clone", "outbound_integrations", "sent_before", "sent_after", "leaked")
_PG_KEYS = ("available", "reason", "statements", "calls", "exec_ms", "note")
# assertion details built from counts, ids, table and field names and the scenario's own facts only
_STRUCTURAL_DETAIL = ("read.must_contain", "read.must_not_contain", "read.business_writes", "create.count",
                      "update.fields_only", "cardinality.created_count")
# their details quote database and expected values verbatim: rebuilt from redacted parts instead
_VALUE_ASSERTIONS = {"create.values": "create_values", "update.values": "update_values"}
# safety rules whose hits are identities (table#id, model, field or tool names, ids), never field values
_IDENTITY_HITS = ("forbid_delete", "max_records_changed", "forbid_models", "forbid_fields", "posted_entries_immutable",
                  "single_company", "allowed_tool_calls")
_IDENT = re.compile(r"[A-Za-z_][\w.]*")


def _frac(n: int, d: int) -> str:
    return f"{n}/{d}" if d else "0/0"


def _scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _withheld(value: str | None, where: str) -> dict | None:
    """A free-text value, withheld: that there was one, and where the private record holds it."""
    return None if value is None else {"withheld": "free text", "private": where}


def _withheld_list(values: list | None, where: str) -> dict:
    return {"withheld": len(values or []), "private": where}


def safety_state(rec: RunRecord) -> str:
    """no_rules · violated · incomplete · clean. `clean` needs at least one configured rule and EVERY rule
    evaluated and passed: a rule that was unavailable or not evaluable leaves the run incomplete, never clean."""
    statuses = [s.status for s in rec.safety]
    if not statuses:
        return "no_rules"
    if "violated" in statuses:
        return "violated"
    if any(s != "passed" for s in statuses):
        return "incomplete"
    return "clean"


def _key_table(rv: dict) -> str | None:
    model = rv.get("key_model")
    return (rv.get("tables") or {}).get(model) if model else None


def _redact_mapping(red: RedactionRules, table: str | None, values: dict | None) -> dict:
    return {k: red.redact_value(table, k, v) for k, v in (values or {}).items()}


def _redacted_assertion(a, red: RedactionRules, table: str | None, expected: dict, where: str) -> dict:
    observed = {f: [red.redact_value(table, f, v) for v in vals] for f, vals in (a.observed or {}).items()}
    if a.name in _STRUCTURAL_DETAIL or (a.kind == "forbid" and a.name.startswith("forbid[")):
        detail = a.detail
    elif a.name in _VALUE_ASSERTIONS:
        if a.passed:
            detail = "the expected values hold"
        elif a.failed_fields:
            detail = "; ".join(f"{f}: database holds " + (", ".join(repr(v) for v in observed[f]) if observed.get(f)
                                                            else "no observed value (row removed or not inspected)")
                               + f", expected {expected.get(f, '?')!r}" for f in a.failed_fields)
        else:
            detail = "no matching row to check, or selected rows not inspected (no diff and no after-state snapshot)"
    else:
        detail = f"detail withheld (free text): {where}"
    return {"name": a.name, "passed": a.passed, "kind": a.kind, "detail": detail,
            "failed_fields": list(a.failed_fields), "observed": observed}


def _redacted_rule(r: RuleResult, where: str) -> dict:
    return {"profile": r.profile, "rule_id": r.rule_id, "rule": r.rule, "status": r.status, "detail": r.detail,
            "rationale": r.rationale, "observable_via": r.observable_via, "not_installed": list(r.not_installed),
            "hits": list(r.hits) if r.rule in _IDENTITY_HITS else _withheld_list(r.hits, where)}


def redacted_run(rec: RunRecord, redaction: RedactionRules | None = None) -> dict[str, Any]:
    """The redacted view of one run. Built field by field from the record, which it never mutates."""
    red = redaction or RedactionRules.load()
    base = f"{rec.run_index:03d}/run.json"

    def at(path: str) -> str:
        return f"{base}: {path}"

    c = rec.conditions
    conditions = {k: to_dict(getattr(c, k)) for k in _CONDITION_KEYS}
    conditions["notes"] = _withheld_list(c.notes, at("conditions.notes"))

    dr = rec.driver_result
    driver_result = None
    if dr is not None:
        driver_result = {
            "session_id": dr.session_id if _scalar(dr.session_id) else None,
            "request_status": to_dict(dr.request_status), "turn_count": dr.turn_count,
            "turns": [{"request_status": to_dict(t.request_status), "wall_s": t.wall_s, "message_id": t.message_id,
                       "pending_interaction": t.pending_interaction, "user_request_returned": t.user_request_returned,
                       "user_input": _withheld(t.user_input, at(f"driver_result.turns[{i}].user_input")),
                       "assistant_response": _withheld(t.assistant_response, at(f"driver_result.turns[{i}].assistant_response")),
                       "error": _withheld(t.error, at(f"driver_result.turns[{i}].error"))} for i, t in enumerate(dr.turns)],
            "model_metadata": to_dict(dr.model_metadata),
            "tool_trace": None if dr.tool_trace is None else [
                {"name": tc.name, "turn_index": tc.turn_index, "errored": tc.error is not None,
                 "arguments": _withheld(tc.arguments_raw, at(f"driver_result.tool_trace[{i}].arguments_raw")),
                 "error": _withheld(tc.error, at(f"driver_result.tool_trace[{i}].error"))} for i, tc in enumerate(dr.tool_trace)],
            "token_usage": to_dict(dr.token_usage), "provider_cost_usd": dr.provider_cost_usd,
            "provider_calls": dr.provider_calls, "usage_basis": dr.usage_basis,
            "driver_notes": _withheld_list(dr.driver_notes, at("driver_result.driver_notes")),
        }

    rv = rec.resolved
    resolved, key_table = None, None
    if rv is not None:
        key_table = _key_table(rv)
        resolved = {"tables": dict(rv.get("tables") or {}), "key_model": rv.get("key_model"),
                    "update_ids": list(rv.get("update_ids") or []), "forbid_ids": dict(rv.get("forbid_ids") or {}),
                    "side_effect_ids": dict(rv.get("side_effect_ids") or {}),
                    "existing_row_counts": dict(rv.get("existing_row_counts") or {}),
                    **{k: _redact_mapping(red, key_table, rv.get(k)) for k in ("create_match", "create_values", "update_values")},
                    "notes": _withheld_list(rv.get("notes"), at("resolved.notes"))}

    classification = None
    if rec.classification is not None:
        # the normalised diff, already redacted by table and field (with the full row in view) when classified
        classification = to_dict(rec.classification)
        classification["safety_violations"] = [_redacted_rule(r, at(f"safety[{i}].hits")) for i, r in enumerate(rec.safety)
                                               if r.status == "violated"]

    g = rec.grade
    grade = None
    if g is not None:
        def assertion(a, path):
            src = _VALUE_ASSERTIONS.get(a.name)
            exp = _redact_mapping(red, key_table, (rv or {}).get(src)) if src else {}
            return _redacted_assertion(a, red, key_table, exp, at(path))
        grade = {"effect": g.effect, "output_defect": g.output_defect, "cardinality": to_dict(g.cardinality),
                 "facts_found": to_dict(g.facts_found),
                 "assertions": [assertion(a, f"grade.assertions[{i}].detail") for i, a in enumerate(g.assertions)],
                 "forbidden_hits": [assertion(a, f"grade.forbidden_hits[{i}].detail") for i, a in enumerate(g.forbidden_hits)]}

    o = rec.outcome
    outcome = None
    if o is not None:
        outcome = {"interaction_outcome": o.interaction_outcome, "business_result": o.business_result,
                   "execution": dict(o.execution), "facts": to_dict(o.facts),
                   "report_effect_mismatch": to_dict(o.report_effect_mismatch), "reason": o.reason,
                   "notes": list(o.notes),   # built by evaluate() from counts and assertion names only
                   "review_flags": [{"flag": fl.get("flag"), "note": fl.get("note"),
                                     "basis": {k: (fl.get("basis") or {}).get(k) for k in _FLAG_BASIS_KEYS},
                                     "final_response": _withheld(fl.get("final_response"), at(f"outcome.review_flags[{i}].final_response")),
                                     "request_error": _withheld(fl.get("request_error"), at(f"outcome.review_flags[{i}].request_error"))}
                                    for i, fl in enumerate(o.review_flags)]}

    a = rec.attribution
    attribution = None if a is None else {"layer": a.layer, "rule": a.rule,
                                          "evidence": _withheld_list(a.evidence, at("attribution.evidence"))}

    inv = rec.invariants
    invariants = None
    if inv is not None:
        invariants = {"configured": list(inv.configured), "baseline_clean": inv.baseline_clean, "note": inv.note,
                      "new_count": inv.new_count, "worsened_count": inv.worsened_count,
                      "by_invariant": {n: {cat: len(vs) for cat, vs in cats.items()} for n, cats in inv.by_invariant.items()},
                      "skipped": {n: _withheld(why, at(f"invariants.skipped.{n}")) for n, why in inv.skipped.items()}}

    pg = rec.pg_stats
    pg_stats = None if pg is None else {**{k: pg[k] for k in _PG_KEYS if k in pg},
                                        "top": [{k: q.get(k) for k in ("queryid", "calls", "exec_ms", "rows")} for q in pg.get("top") or []]}

    he = rec.harness_error
    harness_error = None
    if he is not None:
        head = he.split(":", 1)[0]
        harness_error = {"type": head if _IDENT.fullmatch(head) else "error", "message": _withheld(he, at("harness_error"))}

    td = rec.teardown
    teardown = None if td is None else {
        "database": td.get("database"), "kept": td.get("kept"), "dropped": td.get("dropped"),
        "errors": [{"step": e.get("step"), "type": e.get("type"), "substeps": list(e.get("substeps") or []),
                    "message": _withheld(e.get("message"), at(f"teardown.errors[{i}].message"))}
                   for i, e in enumerate(td.get("errors") or [])]}

    return {
        "artifact_class": REDACTED_REPORT,
        "private_evidence": {"files": sorted(f"{rec.run_index:03d}/{os.path.basename(p)}" for p in rec.artifacts.values()),
                             "note": PRIVATE_EVIDENCE_NOTE},
        "run_index": rec.run_index, "scenario": rec.scenario, "level": rec.level, "profile": rec.profile, "run_db": rec.run_db,
        "started_utc": rec.started_utc, "finished_utc": rec.finished_utc, "status": rec.status, "probe": rec.probe,
        "conditions": conditions,
        "session": {k: v for k, v in (rec.session or {}).items() if k in _SESSION_KEYS and _scalar(v)},
        "capabilities_missing": list(rec.capabilities_missing),
        "driver_result": driver_result, "resolved": resolved, "classification": classification, "grade": grade,
        "outcome": outcome, "attribution": attribution,
        "safety": [_redacted_rule(r, at(f"safety[{i}].hits")) for i, r in enumerate(rec.safety)],
        "safety_state": safety_state(rec),
        "invariants": invariants, "wall_s": rec.wall_s, "pg_stats": pg_stats,
        "cost": None if rec.cost is None else {k: rec.cost[k] for k in _COST_KEYS if k in rec.cost},
        "mail": None if rec.mail is None else {k: rec.mail[k] for k in _MAIL_KEYS if k in rec.mail},
        "coverage_note": rec.coverage_note, "harness_error": harness_error, "teardown": teardown,
    }


def run_summary(rec: RunRecord, redaction: RedactionRules | None = None) -> str:
    """The text for one run, rendered from `redacted_run` only — never from the record — so no text can
    carry what the redacted JSON withholds."""
    return render_run(redacted_run(rec, redaction))


def _teardown_lines(p: dict) -> list[str]:
    td = p.get("teardown") or {}
    lines = []
    if td.get("kept"):
        lines.append(f"  run database {td.get('database')} KEPT (--keep): drop it when done")
    for e in td.get("errors") or []:
        lines.append(f"  !! TEARDOWN: {e['step']} failed ({e['type']}" + (f": {'; '.join(e['substeps'])}" if e.get("substeps") else "") + ")"
                     + ("" if td.get("dropped") or td.get("kept") else f" — the run database {td.get('database')} may still exist")
                     + (f"; message in the private record, {e['message']['private']}" if e.get("message") else ""))
    return lines


def _safety_lines(p: dict) -> list[str]:
    rules = p["safety"]
    viol = [r for r in rules if r["status"] == "violated"]
    incomplete = [r for r in rules if r["status"] != "passed" and r["status"] != "violated"]
    passed = [r for r in rules if r["status"] == "passed"]
    head = {"clean": "fully evaluated, clean", "violated": "VIOLATED",
            "incomplete": "INCOMPLETE: no violation found, but not every rule could be evaluated",
            "no_rules": "no rules configured: nothing was evaluated"}[p["safety_state"]]
    lines = [f"  safety: {head} · {len(passed)} passed · {len(viol)} violated · {len(incomplete)} unavailable/not evaluable"
             + ("" if not viol else " — " + "; ".join(f"{r['profile']}/{r['rule_id']}: {r['detail']}" for r in viol))]
    for r in incomplete:
        lines.append(f"    {r['status']} {r['profile']}/{r['rule_id']}: {r['detail']}")
    for r in passed:
        if r["not_installed"]:
            lines.append(f"    passed {r['profile']}/{r['rule_id']} with nothing to inspect for {r['not_installed']} (not installed)")
    return lines


def render_run(p: dict[str, Any]) -> str:
    lines = [f"run {p['run_index']:03d}  {p['scenario']} @ {p['profile']}  [{p['status']}{' · PROBE, not a measurement' if p['probe'] else ''}]  db={p['run_db']}"]
    if p["status"] == "refused_capability":
        lines.append(f"  refused: scenario requires {p['capabilities_missing']} — not exposed to this agent on this target")
        return "\n".join(lines + _teardown_lines(p))
    if p["status"] != "completed" or p["outcome"] is None:   # harness_error, interrupted: not a finished measurement
        he = p.get("harness_error")
        if he:
            lines.append(f"  harness error: {he['type']} — message withheld from this report (free text); "
                         f"read it in the private record, {he['message']['private']}")
        return "\n".join(lines + _teardown_lines(p))
    o, g, c = p["outcome"], p["grade"], p["classification"]
    f, ex = o["facts"], o["execution"]
    def n(v):   # an unobserved count is "unknown", never "None"
        return "unknown" if v is None else v
    lines.append(f"  interaction {o['interaction_outcome']:<22} business {o['business_result']:<14} "
                 f"execution: {n(ex['tool_calls'])} tool calls · {n(ex['llm_round_trips'])} round-trips · "
                 f"{n(ex['recovered_tool_errors'])} recovered errors · {ex['turns']} turn(s)")
    lines.append(f"  facts: tool {f['tool_execution']} · request {f['interaction']} · txn {f['transaction']} · "
                 f"effect {f['expected_effect']} · report {f['user_report']}"
                 + (f"  ({f['user_report_rule']})" if f["user_report_rule"] else ""))
    mm = o["report_effect_mismatch"]
    if mm and mm.get("direction") not in (None, "none"):
        lines.append(f"  REPORT/EFFECT MISMATCH: {mm['direction']} — {mm['statement']}")
    for fl in o["review_flags"]:
        lines.append(f"  REVIEW {fl['flag']}: no write-tool call · key not satisfied · report {fl['basis']['user_report']} — "
                     f"a human must read what the user was told (not a verdict, not a mismatch)")
        # the sentence is the flag's whole purpose, and it is free text: it stays in the private record
        shown = fl["final_response"] or fl["request_error"]
        lines.append(f"    final response: withheld from this report (free text) — read it in the private record, {shown['private']}"
                     if shown else "    final response: none")
    if o["reason"]:
        lines.append(f"  reason: {o['reason']}")
    for n in o["notes"]:
        lines.append(f"  note: {n}")
    if c:
        if p["level"] == 2:
            lines.append(f"  database: {len(c['expected_business_writes'])} expected · {len(c['unexpected_business_writes'])} UNEXPECTED business writes · "
                         f"{len(c['ai_session'])} ai/session tables · {len(c['bookkeeping'])} bookkeeping tables")
        else:
            lines.append(f"  database: {len(c['business_writes'])} business writes observed · {len(c['ai_session'])} ai/session tables · "
                         f"{len(c['bookkeeping'])} bookkeeping tables")
        for n in (c["unexpected_business_writes"] if p["level"] == 2 else c["business_writes"])[:12]:
            fields = {k: v for k, v in n["fields"].items() if k not in ("create_date", "write_date", "create_uid", "write_uid")}
            shown = {k: v for k, v in list(fields.items())[:6]} if n["kind"] != "changed" else fields
            lines.append(f"    {n['kind']:<8} {n['model'] or n['table']}#{n['record_id']} {NormalisedChange(**n).display_name() or ''}  "
                         f"{json.dumps(shown, default=str)[:220]}")
        if c["table_only_business"]:
            lines.append(f"  business tables at TABLE_ONLY coverage: {[t['table'] for t in c['table_only_business']]}")
    if p.get("coverage_note"):
        lines.append(f"  coverage: {p['coverage_note']}")      # table counts and a fixed explanation only
    if g and p["level"] == 2:
        for a in g["assertions"]:
            lines.append(f"  {'PASS' if a['passed'] else 'FAIL'} {a['name']}: {a['detail']}")
        for a in g["forbidden_hits"]:
            lines.append(f"  FORBIDDEN {a['name']}: {a['detail']}")
        if g["cardinality"]:
            lines.append(f"  cardinality: {g['cardinality']}")
    lines += _safety_lines(p)
    inv = p["invariants"]
    if inv:
        lines.append(f"  invariants: {len(inv['configured'])} configured · new {inv['new_count']} · worsened {inv['worsened_count']}"
                     + (f" · NOT EVALUABLE {sorted(inv['skipped'])}" if inv["skipped"] else "")
                     + ("" if inv["configured"] else f"  ({inv['note']})"))
    at = p["attribution"]
    if at and at["layer"] not in ("not_in_scope",):
        lines.append(f"  attribution: {at['layer']}" + (f" — {at['rule']}" if at["rule"] else ""))
        if at["evidence"]["withheld"]:
            lines.append(f"    evidence: {at['evidence']['withheld']} line(s) withheld from this report (they quote arguments, "
                         f"database values and replies) — read them in the private record, {at['evidence']['private']}")
    lines.append(_cost_line(p))
    mail = p["mail"] or {}
    if mail.get("leaked"):
        lines.append("  !! MAIL LEAKED: mail_mail rows reached state=sent during the run")
    if mail.get("servers_disabled_on_clone"):
        lines.append(f"  mail: {mail['servers_disabled_on_clone']} outgoing mail server(s) from the fixture archived on the clone before the run")
    lines += _target_lines(p)
    cond = p["conditions"]
    mail_blocked = "unknown" if cond["mail_blocked_verified"] is None else cond["mail_blocked_verified"]
    lines.append(f"  conditions: egress {cond['egress_isolation']} · mail blocked {mail_blocked} · "
                 f"key {cond['key_source']}" + (f" (…{cond['key_last4']})" if cond.get("key_last4") else ""))
    return "\n".join(lines + _teardown_lines(p))


def _cost_line(p: dict) -> str:
    pg = p["pg_stats"] or {}
    cost = p["cost"] or {}
    tokens = cost.get("tokens")
    wall = f"{p['wall_s']}s" if p.get("wall_s") is not None else "unknown"
    line = (f"  cost: wall {wall} · SQL {pg.get('calls', '?')} calls / {pg.get('exec_ms', '?')} ms"
            + (f" · tokens in/out/cached {tokens['input']}/{tokens['output']}/{tokens['cached_input']}"
               + (" (partial: a lower bound)" if cost.get("tokens_partial") else "") if tokens else ""))
    if cost.get("usd") is not None and cost.get("provider_calls") == 0:
        return line + f" · {cost['label']} $0.0000 — {cost.get('reason', 'no provider request was made')}"
    if cost.get("usd") is not None:
        return line + f" · {cost['label']} ${cost['usd']:.4f} (provider list price" + (
            f", priced as {cost['priced_as']})" if cost.get("priced_as") else ")")
    return line + f" · cost {cost.get('reason', 'n/a')}"


def model_line(p: dict) -> str:
    """Configured, requested and served model identifiers, kept apart: a served identifier is shown only when a
    provider response was observed naming it."""
    dr = p.get("driver_result") or {}
    mm = dr.get("model_metadata") or {}
    if not mm:
        return "model: not recorded"
    calls = dr.get("provider_calls")
    configured = mm.get("configured_identifier") or mm.get("identifier")
    requested = ("none (no provider request was made)" if calls == 0 else mm.get("identifier") or "?")
    served = mm.get("served_identifier") or ("n/a" if calls == 0 else "not observed")
    return (f"model: {mm.get('provider') or '?'} · configured {configured or 'n/a'} · requested {requested} · served {served}"
            f" · reproducibility {mm.get('reproducibility') or '?'}"
            + (f" · selected by {mm['selected_by']}" if mm.get("selected_by") else ""))


def _target_lines(p: dict) -> list[str]:
    cond = p["conditions"]
    odoo = cond.get("odoo_version") or ("n/a (no Odoo process)" if cond.get("driver") == "noop" else "?")
    head = (f"  target: Odoo {odoo}" + (f" ({cond['odoo_build']})" if cond.get("odoo_build") else "")
            + f" · driver {cond.get('driver') or '?'} · transport: {cond.get('transport') or '?'}")
    return [head, "  " + model_line(p)]


def suite_summary(records: list[RunRecord], scenario_name: str, profile_name: str, level: int) -> tuple[str, dict[str, Any]]:
    n = len(records)
    done = [r for r in records if r.status == "completed"]
    refused = [r for r in records if r.status == "refused_capability"]
    errors = [r for r in records if r.status == "harness_error"]
    inter = Counter(r.outcome.interaction_outcome for r in done)
    biz = Counter(r.outcome.business_result for r in done)
    # four states, each counted once per run: a rule that could not be evaluated never counts as clean
    safety = Counter(safety_state(r) for r in done)
    inv_clean = [r for r in done if r.invariants and r.invariants.new_count == 0 and r.invariants.worsened_count == 0 and not r.invariants.skipped]
    mismatches = Counter((r.outcome.report_effect_mismatch or {}).get("direction") for r in done)
    mismatches = {k: v for k, v in mismatches.items() if k not in (None, "none")}
    attr = Counter(r.attribution.layer for r in done if r.attribution and r.attribution.layer != "not_in_scope")
    silent = [r.run_index for r in done if any(f["flag"] == "silent_claim_review" for f in r.outcome.review_flags)]
    failed_assertions: Counter = Counter()
    for r in done:
        if r.grade:
            for a in r.grade.assertions + r.grade.forbidden_hits:
                if not a.passed:
                    failed_assertions[a.name] += 1
    violations: Counter = Counter()
    unevaluated: Counter = Counter()
    for r in done:
        for s in r.safety:
            if s.status == "violated":
                violations[f"{s.profile}/{s.rule_id}"] += 1
            elif s.status != "passed":
                unevaluated[f"{s.profile}/{s.rule_id} {s.status}"] += 1
    tool_calls = sum(r.outcome.execution.get("tool_calls") or 0 for r in done)
    rts = sum(r.outcome.execution.get("llm_round_trips") or 0 for r in done)
    rec_err = sum(r.outcome.execution.get("recovered_tool_errors") or 0 for r in done)
    # a run whose trace or usage could not be observed adds nothing to a total: say which, never count it as zero
    calls_unknown = [r.run_index for r in done if r.outcome.execution.get("tool_calls") is None]
    rts_unknown = [r.run_index for r in done if r.outcome.execution.get("llm_round_trips") is None]
    biz_writes = sum(r.classification.business_write_count for r in done if r.classification)
    unexpected = sum(len(r.classification.unexpected_business_writes) for r in done if r.classification)
    # every run that could have spent (not a probe, not a capability refusal) either has a complete cost or is named:
    # a total that silently leaves a run out reads as the whole spend. A run that failed can still have a complete
    # cost (zero on evidence, or everything its driver observed), so costs are read from all of them, not only
    # from completed runs.
    could_spend = [r for r in records if not r.probe and r.status != "refused_capability"]
    est = [r.cost["usd"] for r in could_spend if r.cost and r.cost.get("usd") is not None]
    known = sorted(r.run_index for r in could_spend if r.cost and r.cost.get("usd") is not None)
    at_least = {r.run_index: r.cost["at_least_usd"] for r in could_spend if r.cost and r.cost.get("at_least_usd") is not None}
    unknown_cost = sorted(r.run_index for r in could_spend if r.run_index not in known and r.run_index not in at_least)
    wall = [r.wall_s for r in done if r.wall_s is not None]
    sql_calls = [r.pg_stats["calls"] for r in done if r.pg_stats and r.pg_stats.get("available")]
    zero_evidence = [r.run_index for r in records if r.cost and r.cost.get("provider_calls") == 0 and r.cost.get("usd") == 0]

    probe = any(r.probe for r in records)
    target = _suite_target(records)
    lines = [f"agent-review  {scenario_name} --profile {profile_name}  (Level {level}){'  ** PROBE: no provider call, not a measurement **' if probe else ''}"]
    if target:
        odoo = target["odoo_version"] or ("n/a (no Odoo process)" if target["driver"] == "noop" else "?")
        lines.append(f"Target                 Odoo {odoo} · driver {target['driver'] or '?'} · "
                     f"transport: {target['transport'] or '?'}")
        if target["scope"]:
            lines.append(f"Scope                  {target['scope']}")
    lines += ["", f"Runs: {n}" + (f"  ({len(done)} completed, {len(refused)} refused on capability, {len(errors)} harness errors)"
                                  if refused or errors else "")]
    if done:
        lines.append("Interaction            " + " · ".join(f"{v} {k}" for k, v in inter.most_common()))
        if level == 2:
            lines.append("Business result        " + " · ".join(f"{v} {k}" for k, v in biz.most_common()))
            lines.append(f"  observed business-correct runs: {_frac(biz.get('correct', 0), len(done))}  (N={len(done)})")
        else:
            lines.append("Business result        not defined at Level 1 — no expected state was supplied")
        lines.append(f"Execution              {tool_calls} tool calls" + (f" (unobservable on runs {calls_unknown})" if calls_unknown else "")
                     + f" · {rts} round-trips" + (f" (unknown on runs {rts_unknown})" if rts_unknown else "")
                     + f" · {rec_err} recovered errors")
        lines.append(f"Database               {biz_writes} business writes" + (f" · {unexpected} unexpected" if level == 2 else " observed"))
        lines.append("Report/effect          " + (" · ".join(f"{v} {k}" for k, v in mismatches.items()) if mismatches else "no mismatch"))
        if level == 2:
            lines.append("Silent-claim review    " + (f"runs {silent} — no write-tool call, key not satisfied, the user was shown a "
                                                      f"message: read it in the run's private run.json (a flag for a human, not a verdict, "
                                                      f"not a mismatch)" if silent else "none flagged"))
        if safety["no_rules"] == len(done):
            lines.append(f"Safety-profile clean   {_frac(0, len(done))}   (no safety rules configured: nothing was evaluated)")
        else:
            lines.append(f"Safety-profile clean   {_frac(safety['clean'], len(done))}   (every configured rule evaluated and passed)")
            lines.append(f"  violated             {_frac(safety['violated'], len(done))}"
                         + (f"   ({'; '.join(f'{k} x{v}' for k, v in violations.items())})" if violations else ""))
            lines.append(f"  incomplete           {_frac(safety['incomplete'], len(done))}"
                         + (f"   (no violation found, but not every rule could be evaluated: "
                            f"{'; '.join(f'{k} x{v}' for k, v in unevaluated.items())})" if safety["incomplete"] else ""))
            if safety["no_rules"]:
                lines.append(f"  no rules configured  {_frac(safety['no_rules'], len(done))}")
        lines.append(f"Invariant-clean        {_frac(len(inv_clean), len(done))}" + ("" if any(r.invariants and r.invariants.configured for r in done) else "   (0 invariants configured)"))
        if attr:
            lines.append("Attribution            " + " · ".join(f"{k}: runs {[r.run_index for r in done if r.attribution and r.attribution.layer == k]}" for k in attr))
        if failed_assertions:
            lines.append("Failed assertions      " + " · ".join(f"{k} x{v}" for k, v in failed_assertions.most_common()))
    # Cost coverage is reported whether or not any run completed: a harness error can come after provider calls
    # (previously a suite of harness errors printed no cost line at all while its JSON listed the unknown cost).
    if wall or sql_calls or est or at_least or unknown_cost or zero_evidence:
        cost_line = f"Cost                   wall {min(wall):.1f}–{max(wall):.1f} s" if wall else "Cost                   "
        if sql_calls:
            cost_line += f" · SQL {min(sql_calls)}–{max(sql_calls)} calls/run"
        if est and not (at_least or unknown_cost):
            cost_line += f" · estimated ${sum(est):.4f} total (${min(est):.4f}–${max(est):.4f}/run)"
        elif est:
            cost_line += (f" · estimated ${sum(est):.4f} KNOWN SUBTOTAL for runs {known} ({len(known)} of {len(could_spend)} runs; "
                          f"${min(est):.4f}–${max(est):.4f}/run)")
        if at_least or unknown_cost:
            gaps = ([f"at least ${sum(at_least.values()):.4f} on runs {sorted(at_least)} (token usage incomplete)"] if at_least else []) \
                + ([f"unknown on runs {unknown_cost}"] if unknown_cost else [])
            cost_line += (" · " if cost_line.strip() != "Cost" else "") + f"no complete cost for runs {sorted([*at_least, *unknown_cost])}: " \
                + "; ".join(gaps)
        if zero_evidence and not (est or at_least or unknown_cost):
            cost_line += " · $0.0000: no provider request was made (the driver vouches for it, per run)"
        lines.append(cost_line)
        if target and target["driver"] == "native_ai_20":
            lines.append("                       estimates are provider list prices for the calls the local stand-in made with your "
                         "key; Odoo IAP credits: none used, none estimated (Odoo's hosted AI service is not contacted)")
    lines += ["", (f"DESCRIPTIVE COMPARISON ONLY at N={len(done)}: no causal or regression conclusion is established, "
                   "and no claim is made about whether a difference is within the range chance would produce.")]
    data = {"scenario": scenario_name, "profile": profile_name, "level": level, "target": target, "runs": n, "completed": len(done),
            "refused_capability": len(refused), "harness_errors": len(errors),
            "interaction": dict(inter), "business_result": dict(biz) if level == 2 else None,
            # fully evaluated and passed; kept under its old name with the corrected meaning
            "safety_clean": [safety["clean"], len(done)],
            "safety": {state: [safety[state], len(done)] for state in ("clean", "violated", "incomplete", "no_rules")},
            "safety_unevaluated_rules": dict(unevaluated),
            "invariant_clean": [len(inv_clean), len(done)],
            "report_effect_mismatch": mismatches, "review_flags": {"silent_claim_review": silent},
            "attribution": dict(attr), "failed_assertions": dict(failed_assertions),
            "execution_not_counted": {"tool_calls": calls_unknown, "llm_round_trips": rts_unknown},
            "safety_violations": dict(violations), "business_writes": biz_writes, "unexpected_business_writes": unexpected if level == 2 else None,
            # a TOTAL only when every run that could have spent has a complete cost; otherwise null, and the subtotal
            # below is named with its runs
            "estimated_cost_usd_total": round(sum(est), 6) if est and not (at_least or unknown_cost) else None,
            "estimated_cost_usd_known_subtotal": round(sum(est), 6) if est else None, "estimated_cost_known_runs": known,
            "estimated_cost_usd_at_least_from_incomplete_runs": round(sum(at_least.values()), 6) if at_least else None,
            "runs_without_complete_cost": {"partial": sorted(at_least), "unknown": unknown_cost},
            "runs_with_no_provider_request": zero_evidence,
            "odoo_iap_credits": ("none used, none estimated" if target and target["driver"] == "native_ai_20" else None),
            "wall_s": wall, "sql_calls": sql_calls,
            "note": "descriptive comparison only; N stated beside every fraction"}
    return "\n".join(lines), data


def _suite_target(records: list[RunRecord]) -> dict | None:
    """What the suite ran against, from its first run (every run of a suite shares the profile)."""
    for r in records:
        c = r.conditions
        if c.driver:
            return {"odoo_version": c.odoo_version, "odoo_build": c.odoo_build, "driver": c.driver,
                    "transport": c.transport, "scope": c.scope}
    return None


def write_suite(suite_dir: str, records: list[RunRecord], text: str, data: dict, redaction: RedactionRules | None = None) -> None:
    """The REDACTED report (review before sharing outside the team). Each run's private evidence (run.json, raw_diff.json) is already on disk, written
    by the lifecycle; nothing here reads it back or copies from it."""
    red = redaction or RedactionRules.load()
    with open(f"{suite_dir}/summary.txt", "w") as fh:
        fh.write(text + "\n\n" + "\n\n".join(run_summary(r, red) for r in records) + "\n")
    with open(f"{suite_dir}/summary.json", "w") as fh:
        json.dump({"artifact_class": REDACTED_REPORT, "private_evidence": PRIVATE_EVIDENCE_NOTE, "suite": data,
                   "runs": [redacted_run(r, red) for r in records]}, fh, indent=1, default=str)
