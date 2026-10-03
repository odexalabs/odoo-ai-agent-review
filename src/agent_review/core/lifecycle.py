"""The run lifecycle — the core owns it:

    template copy -> environment lifecycle -> invoke driver -> diff vs template -> classify ->
    re-check invariants -> report -> tear down

Every run uses the identical measurement boundary, and SNAPSHOTS, never resets:

    session created, warmed, idle
    SNAPSHOT scoped pg_stat_statements · START TIMER
    submit prompt · execute the conversation
    terminal request outcome AND transaction outcome reached
    STOP TIMER · SNAPSHOT again · delta = scoped difference

Teardown runs in the same process, including on failure.

What a run leaves on disk is PRIVATE EVIDENCE: `run.json` (the complete record — conversation, tool
arguments, errors, resolved expected values, grading detail, attribution evidence) and `raw_diff.json`
(the unredacted diff), beside whatever the driver writes into the run directory (the native driver's
odoo.log and odoo.conf). None of it is a report. The redacted report (reviewed before it is shared
outside the team) is built from the record by `core/report.py` and never mutates it: the grader reads raw evidence, the report reads a projection."""
from __future__ import annotations

import dataclasses
import json
import os
import platform
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg import sql

from ..drivers.base import Driver
from ..invariants import Invariant, InvariantReport, differential, run_invariants
from . import pgstats
from .attribution import Attribution, attribute
from .classify import Classification, ClassificationRules, classify
from .contracts import ChangeEvidence, DriverRunResult, EvidenceIncomplete, RunConditions, to_dict
from .cost import Pricing
from .credentials import Credential
from .detect import TableDiffDetector
from .egress import MailNotBlocked, mail_sent_after, neutralise_mail, verify_mail_blocked
from .fixture import RUN_PREFIX, PostgresTemplateBackend
from .grade import Grade, expected_matcher, grade
from .outcomes import Outcome, evaluate
from .profile import RunProfile, load_safety_profile
from .redact import RedactionRules
from .safety import RuleResult, evaluate_profile, rule_models, table_for_model
from .scenario import Resolved, Resolver, Scenario, ScenarioError


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:24]


@dataclass
class RunRecord:
    run_index: int
    scenario: str
    level: int
    profile: str
    run_db: str
    started_utc: str
    finished_utc: str | None = None
    status: str = "started"                       # completed | refused_capability | harness_error | interrupted
    probe: bool = False                           # True = plumbing only, no provider call; never a measurement
    conditions: RunConditions = field(default_factory=RunConditions)
    session: dict[str, Any] = field(default_factory=dict)
    capabilities_missing: list[str] = field(default_factory=list)
    driver_result: DriverRunResult | None = None
    resolved: dict[str, Any] | None = None
    classification: Classification | None = None
    grade: Grade | None = None
    outcome: Outcome | None = None
    attribution: Attribution | None = None
    safety: list[RuleResult] = field(default_factory=list)
    invariants: InvariantReport | None = None
    wall_s: float | None = None
    pg_stats: dict | None = None
    cost: dict | None = None
    mail: dict | None = None
    coverage_note: str | None = None
    harness_error: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    # {"database", "kept", "dropped", "errors": [{"step", "type", "message"}]} — what closing the run did
    teardown: dict | None = None
    # run.json is this record, unredacted. It says so inside the file, for whoever finds it without context.
    artifact_class: str = "private-evidence"


class TeardownError(RuntimeError):
    """The run's environment could not be torn down, or its record could not be written. Raised only when
    nothing else is propagating, and only after the record is written, so the caller stops before cloning
    again. The record, with the failure in `teardown`, travels with it.

    The message is printed to the terminal, so it names each failed step and its error TYPE only — an
    exception's own text can quote a DSN, a path or a value, and stays in the private record."""

    def __init__(self, record: RunRecord, failures: list[tuple[str, BaseException]]):
        self.record = record
        self.failures = failures
        dropped = (record.teardown or {}).get("dropped")
        super().__init__(f"run {record.run_index:03d}: " + "; ".join(failure_summary(step, e) for step, e in failures)
                         + ("" if dropped else f"; the run database {record.run_db} may still exist")
                         + f"; details in the private record, {private_pointer(record)}")


def failure_summary(step: str, e: BaseException) -> str:
    """A teardown failure for the terminal: the step, the exception type, and the sub-steps a driver named
    (harness text). Never the exception's own message."""
    inner = getattr(e, "steps", None)
    return f"{step} failed ({type(e).__name__}" + (f": {'; '.join(inner)}" if inner else "") + ")"


def private_pointer(record: RunRecord) -> str:
    return f"{record.artifacts.get('run', 'run.json')} (teardown.errors)"


def _key_probe(sc: Scenario, res: Resolved | None, observer_dsn: str, pre_counts: dict[str, int]):
    """Mid-conversation check for the optional continuation: has the keyed MUTATION occurred? Cheap SQL,
    never a model. `when: always` sends it without asking, and is the only form a Level 1 review has (no
    key). READ keys cannot be probed before collection, so a READ scenario's continuation is sent only
    with `when: always`."""

    def probe() -> bool:   # True = continue (key NOT satisfied yet)
        if sc.continuation is None:
            return False
        # before the key check: a Level 1 review has no resolved key, and in an earlier version its continuation
        # was never sent because this test came second
        if sc.continuation.when == "always":
            return True
        if res is None:
            return False
        if sc.kind == "read":
            return False
        with psycopg.connect(observer_dsn) as c:
            if sc.kind == "create" and sc.create:
                t = res.tables[sc.create.model]
                conds, params = [], []
                for k, v in res.create_match.items():
                    conds.append(sql.SQL("{}::text = %s").format(sql.Identifier(k)))
                    params.append(str(v))
                where = sql.SQL(" AND ").join(conds) if conds else sql.SQL("true")
                n = c.execute(sql.SQL("select count(*) from {} where {}").format(sql.Identifier(t), where), params).fetchone()[0]
                return (n - pre_counts.get(t, 0)) < sc.create.count
            if sc.kind == "update" and sc.update:
                t = res.tables[sc.update.select.model]
                for pk in res.update_ids:
                    row = c.execute(sql.SQL("select * from {} where id = %s").format(sql.Identifier(t)), (pk,), ).fetchone()
                    cols = [d.name for d in c.execute(sql.SQL("select * from {} limit 0").format(sql.Identifier(t))).description]
                    rec = dict(zip(cols, row or []))
                    from .grade import values_equal
                    if any(not values_equal(rec.get(k), v) for k, v in res.update_values.items()):
                        return True
                return False
        return False

    return probe


def _pre_counts(sc: Scenario, res: Resolved | None, observer_dsn: str) -> dict[str, int]:
    if res is None or not sc.create:
        return {}
    t = res.tables[sc.create.model]
    conds, params = [], []
    for k, v in res.create_match.items():
        conds.append(sql.SQL("{}::text = %s").format(sql.Identifier(k)))
        params.append(str(v))
    where = sql.SQL(" AND ").join(conds) if conds else sql.SQL("true")
    with psycopg.connect(observer_dsn) as c:
        return {t: c.execute(sql.SQL("select count(*) from {} where {}").format(sql.Identifier(t), where), params).fetchone()[0]}


def _known_models(observer_dsn: str) -> dict[str, str]:
    try:
        return Resolver(observer_dsn).known_models()
    except psycopg.Error:
        return {}


def await_no_writers(env, deadline_s: float = 10.0) -> None:
    """No connection of the execution role may remain on the run's copy when the final evidence is taken. Once the
    agent's process is gone its backends normally close within moments; one that lingers is ended (an unfinished
    transaction is rolled back, which is what the evidence should reflect). Raises EvidenceIncomplete otherwise."""
    q = "select pid from pg_stat_activity where datname = %s and usename = %s"
    with psycopg.connect(env.observer_dsn, autocommit=True) as c:
        end = time.monotonic() + deadline_s
        pids = [r[0] for r in c.execute(q, (env.name, env.execution_role)).fetchall()]
        while pids and time.monotonic() < end:
            time.sleep(0.2)
            pids = [r[0] for r in c.execute(q, (env.name, env.execution_role)).fetchall()]
        if pids:
            try:
                for pid in pids:
                    c.execute("select pg_terminate_backend(%s)", (pid,))
            except psycopg.Error as e:
                raise EvidenceIncomplete(f"{len(pids)} connection(s) of the execution role still open on the run's copy "
                                         f"and could not be ended ({type(e).__name__})") from e
            time.sleep(1.0)
            pids = [r[0] for r in c.execute(q, (env.name, env.execution_role)).fetchall()]
        if pids:
            raise EvidenceIncomplete(f"{len(pids)} connection(s) of the execution role still open on the run's copy "
                                     "after the agent was stopped")


def run_once(sc: Scenario, profile: RunProfile, driver: Driver, run_index: int, suite_dir: str,
             credential: Credential | None, invariants: list[Invariant] | None = None,
             safety_names: list[str] | None = None, pricing: Pricing | None = None, keep: bool = False) -> RunRecord:
    invariants = invariants or []
    pricing = pricing or Pricing()
    run_dir = os.path.join(suite_dir, f"{run_index:03d}")
    os.makedirs(run_dir, exist_ok=True)
    # Static configuration is read BEFORE any clone exists: a missing or malformed file fails here, with
    # nothing to tear down. (Read after the clone, outside the `try`, a failure left the clone behind.)
    redaction = RedactionRules.load()
    safety_profiles = [load_safety_profile(n) for n in (safety_names if safety_names is not None else sc.safety_profiles)]
    base_rules = ClassificationRules.load()
    fx = profile.fixture
    backend = PostgresTemplateBackend(template=profile.template_for(sc.fixture), execution_role=fx["execution_role"],
                                      admin_dsn=fx.get("admin_dsn", "dbname=postgres"), host=fx.get("host"), port=fx.get("port"),
                                      execution_password=fx.get("execution_password"))
    backend.prepare()
    # Compatibility is established BEFORE the clone: a scenario the driver cannot execute, or a profile whose
    # Odoo version, template or transport does not match the driver, refuses with nothing to tear down.
    driver.check_scenario(sc)
    preflight_notes = driver.preflight(profile, backend.template_dsn())
    name = f"{slug(sc.name)}_{run_index:03d}"
    rec = RunRecord(run_index, sc.name, sc.level, profile.name, f"{RUN_PREFIX}{name}", datetime.now(timezone.utc).isoformat())
    rec.conditions = RunConditions(fixture=backend.template, fixture_backend=backend.describe()["backend"],
                                   benchmark_date=sc.benchmark_date,
                                   egress_allowed_endpoints=list(profile.egress.get("allowed_endpoints", [])),
                                   host={"platform": platform.platform(), "machine": platform.machine()},
                                   key_source=credential.source if credential else None, key_last4=credential.last4 if credential else None)
    rec.probe = bool(getattr(driver, "probe", False))
    rec.conditions.driver = driver.name
    rec.conditions.transport = driver.transport_description()
    rec.conditions.scope = driver.scope_note()
    rec.conditions.notes += preflight_notes
    if rec.probe:
        rec.conditions.notes.append("PROBE run: no provider call was made; outcomes here are not measurements")
    detector = TableDiffDetector()
    res: Resolved | None = None
    evidence: ChangeEvidence | None = None
    in_flight: BaseException | None = None
    env = backend.clone_for_run(name)
    # Nothing runs between the clone and the `try`: from here every exit passes through the teardown.
    try:
        rec.run_db = env.name
        known = _known_models(env.observer_dsn)
        extra_tables = list(sc.extra_business_tables)
        # every model a safety rule reads ROWS of — `models`, a `fields` mapping, a singular `model` — is
        # snapshotted EXACT; the evaluator reads the same function and refuses a pass without it
        for sp in safety_profiles:
            for r in sp.rules:
                for m in rule_models(r):
                    t = table_for_model(m, base_rules, known)
                    if t:
                        extra_tables.append(t)
        if sc.level == 2:
            res = Resolver(env.observer_dsn).resolve(sc)
            extra_tables += list(res.tables.values())
            rec.resolved = {"tables": res.tables, "update_ids": res.update_ids, "create_match": res.create_match,
                            "create_values": res.create_values, "update_values": res.update_values,
                            "forbid_ids": {str(k): v for k, v in res.forbid_ids.items()},
                            "side_effect_ids": {str(k): v for k, v in res.side_effect_ids.items()},
                            "existing_row_counts": {t: len(ids) for t, ids in res.existing_ids.items()}, "notes": res.notes,
                            # which model the key's values belong to, so the redacted report redacts them by that table
                            "key_model": sc.create.model if sc.create else (sc.update.select.model if sc.update else None)}
            rec.conditions.notes += res.notes
        for inv in invariants:
            extra_tables += [t for t in getattr(inv, "tables", []) if t in known.values()]
        rules = dataclasses.replace(base_rules, business_tables=base_rules.business_tables + sorted(set(extra_tables)))
        exact = set(rules.business_tables)

        # Mail blocking is MANDATORY in every mode: the clone is disposable, so any outgoing mail server the
        # fixture carries is archived here and the block is verified BEFORE the driver starts anything. (In an earlier
        # version this came after driver.prepare, which starts Odoo and logs the operator in: a login-time
        # mail, such as a new-device alert, could have used a restored server first.) A run that cannot
        # establish the block starts no Odoo process and does not execute the agent.
        disabled = neutralise_mail(env.dsn)
        egress = verify_mail_blocked(env.observer_dsn, driver.cron_threads_zero(), driver.smtp_fallback_disabled())
        egress.mail_servers_disabled = disabled
        if disabled:
            egress.notes.append(f"{disabled} outgoing mail server row(s) in the fixture archived on the clone before the run")
        egress.isolation = "partial" if egress.mail_blocked_verified else "unavailable"
        rec.conditions.egress_isolation = egress.isolation
        rec.conditions.mail_blocked_verified = egress.mail_blocked_verified
        rec.conditions.notes += egress.notes
        rec.mail = {"servers_disabled_on_clone": disabled, "outbound_integrations": egress.outbound_integrations}
        if not egress.mail_blocked_verified:
            raise MailNotBlocked("mail blocking could not be established on the clone; refusing to run the agent: "
                                 + "; ".join(egress.notes))

        driver.prepare(env, profile, sc.required_agent, credential, run_dir, sc.required_session)
        caps = driver.capabilities()
        rec.conditions.odoo_version, rec.conditions.odoo_build = caps.odoo_version, caps.odoo_build
        rec.session = driver.session_context()
        rec.session["model_configured"] = profile.model
        missing = caps.missing_for(sc.required_capabilities, rec.session.get("agent"))
        if missing:
            rec.capabilities_missing = missing
            rec.status = "refused_capability"
            rec.conditions.notes.append(f"scenario requires {missing}; not exposed to agent {rec.session.get('agent')!r} on this target")
            return rec
        write_tools = caps.write_tool_names()

        inv_before, _clean = run_invariants(invariants, env.observer_dsn, profile.mode)
        detector.before_run(env, exact)
        pre = _pre_counts(sc, res, env.observer_dsn)
        should_continue = _key_probe(sc, res, env.observer_dsn, pre)

        # ---- measurement boundary opens
        pg_before = pgstats.snapshot(env.observer_dsn, env.name, env.execution_role)
        t0 = time.perf_counter()
        cont = sc.continuation
        driver.execute(sc.prompt, (cont if driver.accepts_structured_continuation else cont.text) if cont else None,
                       should_continue)
        rec.wall_s = round(time.perf_counter() - t0, 3)
        pg_after = pgstats.snapshot(env.observer_dsn, env.name, env.execution_role)
        # ---- measurement boundary closes
        # The final evidence is taken only once nothing can write to the copy any more. A turn that timed
        # out can still have work under way (Odoo 20: a stand-in reply on its way; Odoo 19: a request the client
        # stopped waiting for), and a write landing after the snapshot would leave the report disagreeing with the
        # database. The driver stops what it started, then no connection of the execution role may remain on the
        # copy. A run where either cannot be established is not graded (EvidenceIncomplete).
        rec.conditions.notes += driver.quiesce()
        await_no_writers(env)
        detector.after_run(env)
        rec.pg_stats = pgstats.delta(pg_before, pg_after)
        result = driver.collect()
        rec.driver_result = result
        sent_after = mail_sent_after(env.observer_dsn)
        rec.mail.update({"sent_before": egress.mail_sent_before, "sent_after": sent_after,
                         "leaked": (sent_after or 0) > (egress.mail_sent_before or 0)})
        inv_after, _ = run_invariants(invariants, env.observer_dsn, "customer")
        rec.invariants = differential(invariants, inv_before, inv_after)

        evidence = detector.collect_changes()
        matcher = expected_matcher(sc, res) if res is not None else None
        cls = classify(evidence, rules, redaction, set(known), matcher)
        responses = [t.assistant_response for t in result.turns if t.assistant_response]
        g = grade(sc, res, evidence, responses, cls.business_write_count, rows_after=detector.rows_after)
        for sp in safety_profiles:
            rec.safety += evaluate_profile(sp, evidence, result, rules, known, rec.session, detector.rows_before)
        cls.safety_violations = [to_dict(r) for r in rec.safety if r.status == "violated"]
        rec.classification = cls
        rec.coverage_note = cls.coverage_note
        rec.grade = g
        rec.outcome = evaluate(result, g, sc.level, cls.business_write_count, sc.response_rules + list(profile.report_rules),
                               write_tools, kind=sc.kind)
        rec.attribution = (Attribution("not_in_scope", rule="probe run: no provider call, nothing to attribute") if rec.probe
                           else attribute(sc, res, result, g, rec.outcome, write_tools, rec.invariants.new_count if rec.invariants else 0))
        rec.cost = _cost(pricing, profile, rec.session, result.token_usage, result.provider_calls, result.usage_basis)
        if result.provider_cost_usd is not None:
            rec.cost["reported_usd"] = result.provider_cost_usd
        rec.status = "completed"
    except KeyboardInterrupt as e:
        in_flight = e
        rec.status = "interrupted"
        rec.harness_error = "KeyboardInterrupt: run interrupted; artifact incomplete (`teardown` records whether the clone was dropped)"
        raise
    except Exception as e:  # recorded, then teardown
        rec.status = "harness_error"
        rec.harness_error = f"{type(e).__name__}: {e}"
        if isinstance(e, ScenarioError):
            in_flight = e
            raise
    except BaseException as e:   # SystemExit and the like: never swallowed, never replaced by a teardown failure
        in_flight = e
        raise
    finally:
        # A teardown or write failure is RECORDED, never raised over the run's own outcome: raising here
        # replaced an in-flight error and skipped writing the record. It is surfaced afterwards instead —
        # as a note on the error already propagating, or as a TeardownError once the record is on disk.
        failures: list[tuple[str, BaseException]] = []
        if rec.cost is None and rec.status in ("harness_error", "interrupted"):
            rec.cost = _cost_after_failure(pricing, profile, rec.session, driver)
        try:
            failures += _teardown(rec, driver, backend, env, keep)
        finally:
            failures += _write_private_record(rec, evidence, run_dir)
        if failures:
            if in_flight is None:
                raise TeardownError(rec, failures) from failures[0][1]
            for step, e in failures:
                # printed by the CLI: the step and the error type, never the exception's text
                _add_note(in_flight, f"agent-review: teardown {failure_summary(step, e)}; details in the private "
                                     f"record, {private_pointer(rec)}")
    return rec


def _cost(pricing: Pricing, profile: RunProfile, session: dict, usage, provider_calls: int | None,
          basis: str | None) -> dict:
    """The run's ESTIMATED cost: provider list price x the provider's own token counts. Zero only on positive
    evidence that no provider request was made (`provider_calls == 0`); unknown (usd None) when usage was not
    observed; a lower bound (`at_least_usd`) when it was observed only in part."""
    model = session.get("price_as") or session.get("llm_model") or profile.model
    if provider_calls == 0 and usage is not None and not usage.partial:
        out = {"label": "estimated", "usd": 0.0, "provider_calls": 0,
               "reason": f"no provider request was made ({basis or 'the driver vouches for it'})"}
    else:
        out = pricing.estimate(profile.provider_name, model, usage)
        out["provider_calls"] = provider_calls
    if model != (session.get("llm_model") or profile.model):
        out["priced_as"] = model
    if basis:
        out["usage_basis"] = basis
    return out


def _cost_after_failure(pricing: Pricing, profile: RunProfile, session: dict, driver: Driver) -> dict | None:
    """A run that failed after the driver started may still have spent: record what the driver can vouch for
    (a lower bound, or zero on evidence), never a guess. Must not raise over the run's own failure."""
    try:
        usage, calls, basis = driver.usage_after_failure()
    except Exception as e:  # noqa: BLE001 — the cost stays unknown; the teardown and record write still run
        return {"label": "estimated", "usd": None, "reason": f"unknown: the driver could not report usage ({type(e).__name__})"}
    if usage is None:
        return {"label": "estimated", "usd": None, "reason": f"unknown: {basis}"}
    return _cost(pricing, profile, session or {}, usage, calls, basis)


def _add_note(exc: BaseException, text: str) -> None:
    """`BaseException.add_note` on Python 3.11+, the same `__notes__` list on 3.10 (an earlier version skipped the
    note on 3.10, so an interrupted run's leaked database went unmentioned on the terminal)."""
    if hasattr(exc, "add_note"):
        exc.add_note(text)
        return
    try:
        exc.__notes__ = [*getattr(exc, "__notes__", []), text]
    except AttributeError:
        pass   # an exception type that refuses attributes: the private record still carries the failure


def _teardown(rec: RunRecord, driver: Driver, backend, env, keep: bool) -> list[tuple[str, BaseException]]:
    """Close the driver, then drop the clone unless --keep. The drop runs whatever the close did (a failing
    close must not leave the database behind), and each failure is returned, not raised."""
    failures: list[tuple[str, BaseException]] = []
    dropped = False
    try:
        try:
            driver.close()
        except Exception as e:  # noqa: BLE001 — any failure is recorded; the drop below must still run
            failures.append(("driver.close", e))
    finally:
        if not keep:
            try:
                backend.destroy_run(env)
                dropped = True
            except Exception as e:  # noqa: BLE001 — recorded and surfaced by the caller, never raised over the run's outcome
                failures.append(("destroy_run", e))
        rec.teardown = {"database": env.name, "kept": keep, "dropped": dropped,
                        "errors": [{"step": s, "type": type(e).__name__, "substeps": list(getattr(e, "steps", None) or []),
                                    "message": str(e)} for s, e in failures]}
    return failures


def _write_private_record(rec: RunRecord, evidence: ChangeEvidence | None, run_dir: str) -> list[tuple[str, BaseException]]:
    """Write the PRIVATE evidence — raw_diff.json (the unredacted diff) and run.json (this record, whole). A
    failed raw_diff write is recorded in run.json and never stops run.json itself being written."""
    failures: list[tuple[str, BaseException]] = []
    rec.finished_utc = datetime.now(timezone.utc).isoformat()
    if evidence is not None:
        raw_path = os.path.join(run_dir, "raw_diff.json")
        try:
            with open(raw_path, "w") as fh:
                json.dump(to_dict(evidence), fh, indent=1, default=str)
            rec.artifacts["raw_diff"] = raw_path
        except (OSError, TypeError, ValueError) as e:
            failures.append(("write raw_diff.json", e))
            if rec.teardown is not None:
                rec.teardown["errors"].append({"step": "write raw_diff.json", "type": type(e).__name__, "message": str(e)})
    rec_path = os.path.join(run_dir, "run.json")
    rec.artifacts["run"] = rec_path
    try:
        with open(rec_path, "w") as fh:
            json.dump(to_dict(rec), fh, indent=1, default=str)
    except (OSError, TypeError, ValueError) as e:
        failures.append(("write run.json", e))
    return failures
