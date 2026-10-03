"""Audit tests against a real PostgreSQL clone of the fixture template. No Odoo process,
no provider call, no spend. Every clone is created under the run namespace and dropped in the test's
own `finally`; the template is never written."""
from __future__ import annotations

import json
import os
import textwrap
import time
from pathlib import Path

import psycopg
import pytest

from agent_review.core import pgstats
from agent_review.core.contracts import Coverage
from agent_review.core.detect import TableDiffDetector
from agent_review.core.egress import neutralise_mail, verify_mail_blocked
from agent_review.core.fixture import RUN_PREFIX, PostgresTemplateBackend, PreflightRefused
from agent_review.core.lifecycle import TeardownError, run_once
from agent_review.core.profile import load_run_profile
from agent_review.core.scenario import Resolver, ScenarioError, Selector, load_scenario
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


def _db_exists(name: str) -> bool:
    with psycopg.connect("dbname=postgres") as c:
        return bool(c.execute("select 1 from pg_database where datname = %s", (name,)).fetchone())


def _template_fingerprint() -> tuple:
    with psycopg.connect("dbname=postgres") as c:
        size = c.execute("select pg_database_size(%s)", (TEMPLATE,)).fetchone()[0]
    with psycopg.connect(f"dbname={TEMPLATE}") as c:
        n = c.execute("select (select count(*) from crm_lead), (select count(*) from res_partner), "
                      "(select count(*) from ir_mail_server), (select count(*) from ir_config_parameter)").fetchone()
    return size, n


@pytest.fixture(scope="module")
def template_before():
    return _template_fingerprint()


@pytest.fixture(autouse=True)
def _template_untouched(template_before):
    yield
    assert _template_fingerprint()[1] == template_before[1], "the template changed during a test"


def _profile(tmp_path, agents: dict, mode: str = "customer"):
    p = tmp_path / "prof.yaml"
    p.write_text(textwrap.dedent(f"""
        name: audit-noop
        driver: noop
        mode: {mode}
        fixture: {{execution_role: {ROLE}, admin_dsn: 'dbname=postgres', templates: {{default: {TEMPLATE}}}}}
        agents: {json.dumps(agents)}
        provider: {{name: none, model: none}}
    """))
    return load_run_profile(str(p))


def _scenario(tmp_path, body: str):
    p = tmp_path / "sc.yaml"
    p.write_text(textwrap.dedent(body))
    return load_scenario(p)


class _Clone:
    """A clone for direct detector/resolver tests, always dropped."""

    def __init__(self, name: str):
        self.backend = PostgresTemplateBackend(TEMPLATE, ROLE)
        self.name = name

    def __enter__(self):
        self.env = self.backend.clone_for_run(self.name)
        return self.env

    def __exit__(self, *a):
        self.backend.destroy_run(self.env)
        assert not _db_exists(self.env.name)


# ============================================================ ChangeDetector blind spot, stated not hidden
def test_relation_table_delete_plus_insert_is_reported_as_table_only_not_as_unchanged():
    with _Clone("audit_blindspot") as env:
        with psycopg.connect(env.dsn, autocommit=True) as c:
            c.execute("create table audit_rel (a int, b int, primary key (a, b))")
            c.execute("insert into audit_rel values (1, 10), (2, 20)")
        det = TableDiffDetector()
        det.before_run(env, {"crm_lead"})
        with psycopg.connect(env.dsn, autocommit=True) as c:
            c.execute("delete from audit_rel where a = 1")
            c.execute("insert into audit_rel values (3, 30)")
        det.after_run(env)
        ev = det.collect_changes()
        assert ev.coverage_by_table["audit_rel"] == Coverage.TABLE_ONLY
        assert "audit_rel" not in [t.table for t in ev.tables_touched]        # invisible — and never claimed otherwise
        assert not [rc for rc in ev.row_changes if rc.table == "audit_rel"]
        # the same change on an EXACT table is seen row by row, with the composite key preserved
        det2 = TableDiffDetector()
        with psycopg.connect(env.dsn, autocommit=True) as c:
            c.execute("delete from audit_rel; insert into audit_rel values (1, 10), (2, 20)")
        det2.before_run(env, {"audit_rel"})
        with psycopg.connect(env.dsn, autocommit=True) as c:
            c.execute("delete from audit_rel where a = 1")
            c.execute("insert into audit_rel values (3, 30)")
        det2.after_run(env)
        ev2 = det2.collect_changes()
        kinds = sorted((rc.kind, rc.pk) for rc in ev2.row_changes if rc.table == "audit_rel")
        assert kinds == [("added", "3|30"), ("removed", "1|10")]
        assert ev2.coverage_by_table["audit_rel"] == Coverage.EXACT


# ============================================================ value-type diff correctness
def test_diff_value_types_and_noop_write():
    with _Clone("audit_types") as env:
        with psycopg.connect(env.dsn, autocommit=True) as c:
            c.execute("""create table audit_types (id serial primary key, s text, b boolean, n numeric(16,2), f float8,
                         d date, ts timestamp, j jsonb, blob bytea, m numeric(16,2), write_date timestamp)""")
            c.execute("""insert into audit_types (s, b, n, f, d, ts, j, blob, m, write_date) values
                         (null, false, 0, 1.5, '2026-09-19', '2026-09-19 10:00:00', '{"k": 1}', '\\x00', 1200.00, now()),
                         ('', true, 12.50, 2.0, null, null, null, null, 0.10, now()),
                         ('keep', true, 1, 1, '2026-01-01', '2026-01-01 00:00:00', '[]', '\\x01', 5, now()),
                         ('noop', true, 1, 1, '2026-01-01', '2026-01-01 00:00:00', '[]', '\\x01', 5, '2026-01-01 00:00:00'),
                         ('gone', false, 0, 0, null, null, null, null, 0, now())""")
        det = TableDiffDetector()
        det.before_run(env, {"audit_types"})
        with psycopg.connect(env.dsn, autocommit=True) as c:
            c.execute("""update audit_types set s = 'now', b = true, n = 0.01, f = 1.5, d = '2026-09-20', ts = ts + interval '1 hour',
                         j = '{"k": 2}', blob = '\\x01', m = 1200.01 where id = 1""")                       # NULL->value, false->true, 0->0.01, jsonb, bytea, monetary cent
            c.execute("update audit_types set s = null, b = false, n = null, d = null where id = 2")           # ''->NULL, true->false, value->NULL
            c.execute("update audit_types set write_date = now() where id = 4")                                 # write_date only: a no-op write
            c.execute("delete from audit_types where id = 5")
            c.execute("insert into audit_types (s, m) values ('new', 0.00)")
        det.after_run(env)
        ev = det.collect_changes()
        by = {(rc.pk, rc.kind): rc for rc in ev.row_changes if rc.table == "audit_types"}
        r1 = by[(1, "changed")].changed_fields
        assert r1["s"] == [None, "now"] and r1["b"] == [False, True] and r1["n"] == ["0", "0.01"] and "f" not in r1
        assert r1["d"] == ["2026-09-19", "2026-09-20"] and r1["ts"] == ["2026-09-19T10:00:00", "2026-09-19T11:00:00"]
        assert r1["j"] == [{"k": 1}, {"k": 2}] and r1["blob"][0].startswith("sha1:") and r1["blob"][0] != r1["blob"][1]
        assert r1["m"] == ["1200", "1200.01"]
        r2 = by[(2, "changed")].changed_fields
        assert r2["s"] == ["", None] and r2["b"] == [True, False] and r2["n"] == ["12.50", None] and "d" not in r2
        assert list(by[(4, "changed")].changed_fields) == ["write_date"]        # reported, not hidden, as exactly that
        assert by[(5, "removed")].before["s"] == "gone" and by[(6, "added")].after["s"] == "new"
        assert (3, "changed") not in by
        raw = __import__("agent_review.core.contracts", fromlist=["to_dict"]).to_dict(ev)
        json.dumps(raw)     # the raw artifact is serialisable as-is


# ============================================================ selectors: resolved before the run, explicit failures
def test_selector_resolution_semantics():
    with _Clone("audit_sel") as env:
        r = Resolver(env.observer_dsn)
        one = r.resolve_ids(Selector("res.partner", [["name", "=", "Eval Operator"]]))
        assert len(one) == 1
        many = r.resolve_ids(Selector("crm.lead", [["active", "=", True]]))
        assert len(many) >= 10, "boolean selector values must compare as PostgreSQL text"
        assert r.resolve_ids(Selector("crm.lead", [["active", "=", False]])) == r.resolve_ids(Selector("crm.lead", [["active", "!=", True]]))
        assert r.resolve_ids(Selector("res.partner", [["name", "=", "Nobody Here"]])) == []
        nulls = r.resolve_ids(Selector("crm.lead", [["user_id", "=", None]]))
        assert nulls == r.resolve_ids(Selector("crm.lead", [["user_id", "is null"]]))
        with pytest.raises(ScenarioError, match="exactly one"):
            r._value({"ref": {"model": "res.partner", "where": [["name", "ilike", "%a%"]]}})
        with pytest.raises(ScenarioError, match="not installed"):
            r.resolve_ids(Selector("mrp.production", []))
        with pytest.raises(ScenarioError, match="null cannot"):
            r.resolve_ids(Selector("crm.lead", [["user_id", "in", [None, 1]]]))
        with pytest.raises(psycopg.errors.UndefinedColumn):
            r.resolve_ids(Selector("crm.lead", [["no_such_column; drop table crm_lead", "=", 1]]))
        # ids are compared as text, so a numeric id and its string spelling resolve identically
        assert r.resolve_ids(Selector("res.partner", [["id", "in", [one[0]]]])) == r.resolve_ids(Selector("res.partner", [["id", "in", [str(one[0])]]]))


def test_update_selector_zero_rows_is_an_explicit_error_and_resolution_is_persisted(tmp_path):
    sc = _scenario(tmp_path, """
        turns: ["x"]
        expect: {kind: update, select: {model: crm.lead, where: [[name, "=", "no such lead"]]}, values: {priority: "2"}}
    """)
    with pytest.raises(ScenarioError, match="matched no rows"):
        run_once(sc, _profile(tmp_path, {"default": {}}), NoopDriver(), 1, str(tmp_path / "s"), None)
    assert not _db_exists(f"{RUN_PREFIX}sc_001")
    sc2 = _scenario(tmp_path, """
        turns: ["x"]
        expect: {kind: update, select: {model: crm.lead, where: [[active, "=", true]]}, values: {priority: "2"}}
        forbid: [{model: res.partner, where: [[name, "=", "Eval Operator"]], fields: [email]}]
    """)
    rec = run_once(sc2, _profile(tmp_path, {"default": {}}), NoopDriver(), 2, str(tmp_path / "s"), None)
    assert rec.status == "completed"
    assert len(rec.resolved["update_ids"]) >= 10 and rec.resolved["forbid_ids"]["0"] and rec.resolved["existing_row_counts"]["crm_lead"] >= 10
    saved = json.loads(Path(rec.artifacts["run"]).read_text())
    assert saved["resolved"]["update_ids"] == rec.resolved["update_ids"]


# ============================================================ inspection set on a real clone (UPDATE, partial application)
def test_update_partially_applied_is_not_satisfied_and_state_grading_holds(tmp_path):
    sc = _scenario(tmp_path, """
        turns: ["set priority 3 on Ana's leads"]
        expect:
          kind: update
          select: {model: crm.lead, where: [[active, "=", true]]}
          values: {priority: "3"}
          fields_only: [priority]
    """)
    one = "update crm_lead set priority = '3', write_date = now() where id = (select min(id) from crm_lead where active)"
    rec = run_once(sc, _profile(tmp_path, {"default": {"simulate_sql": one}}), NoopDriver(), 1, str(tmp_path / "s"), None)
    assert rec.status == "completed" and rec.grade.effect == "not_satisfied"
    d = next(a.detail for a in rec.grade.assertions if a.name == "update.values")
    assert "untouched by the run" in d and d.count("#") >= 2
    assert rec.outcome.business_result == "incorrect"
    all_rows = "update crm_lead set priority = '3', write_date = now() where active"
    rec2 = run_once(sc, _profile(tmp_path, {"default": {"simulate_sql": all_rows}}), NoopDriver(), 2, str(tmp_path / "s"), None)
    assert rec2.grade.effect == "satisfied" and rec2.outcome.business_result == "correct", [a.detail for a in rec2.grade.assertions]
    assert rec2.classification.unexpected_business_writes == []
    # a run that changes an extra field on a selected row fails fields_only and is an unexpected write
    extra = "update crm_lead set priority = '3', name = name || '!', write_date = now() where active"
    rec3 = run_once(sc, _profile(tmp_path, {"default": {"simulate_sql": extra}}), NoopDriver(), 3, str(tmp_path / "s"), None)
    assert not next(a for a in rec3.grade.assertions if a.name == "update.fields_only").passed
    assert rec3.grade.effect == "not_satisfied" and len(rec3.classification.unexpected_business_writes) >= 10


# ============================================================ multi-turn: continuation decided by SQL probe, never a model
def test_level2_continuation_sent_only_when_key_not_satisfied(tmp_path):
    body = """
        turns: ["Create a lead for Alice"]
        continuation: {text: "Yes, please create it.", when: key_not_satisfied}
        expect: {kind: create, model: crm.lead, count: 1, values: {contact_name: Alice}}
    """
    sc = _scenario(tmp_path, body)
    rec = run_once(sc, _profile(tmp_path, {"default": {"reply": "Shall I?"}}), NoopDriver(), 1, str(tmp_path / "s"), None)
    assert rec.driver_result.turn_count == 2 and rec.outcome.execution["turns"] == 2
    assert [t.user_input for t in rec.driver_result.turns] == ["Create a lead for Alice", "Yes, please create it."]
    ins = ("insert into crm_lead (name, type, contact_name, active, stage_id, company_id, team_id, priority, create_date, write_date, create_uid, write_uid) "
           "values ('L', 'opportunity', 'Alice', true, (select min(id) from crm_stage), (select min(id) from res_company), (select min(id) from crm_team), '1', now(), now(), 1, 1)")
    rec2 = run_once(sc, _profile(tmp_path, {"default": {"simulate_sql": ins, "reply": "done"}}), NoopDriver(), 2, str(tmp_path / "s"), None)
    assert rec2.driver_result.turn_count == 1 and rec2.grade.effect == "satisfied"
    sc_no = _scenario(tmp_path, "\n".join(l for l in body.splitlines() if "continuation" not in l))
    rec3 = run_once(sc_no, _profile(tmp_path, {"default": {"reply": "Shall I?"}}), NoopDriver(), 3, str(tmp_path / "s"), None)
    assert rec3.driver_result.turn_count == 1


# ============================================================ repetition isolation
def test_repeats_start_from_a_clean_clone_with_unique_names(tmp_path):
    guard = """
        do $$ begin
          if exists (select 1 from crm_lead where name = 'AUDIT-ISOLATION') then raise exception 'carry-over from a previous run'; end if;
          insert into crm_lead (name, type, active, stage_id, company_id, team_id, priority, create_date, write_date, create_uid, write_uid)
            values ('AUDIT-ISOLATION', 'opportunity', true, (select min(id) from crm_stage), (select min(id) from res_company), (select min(id) from crm_team), '1', now(), now(), 1, 1);
        end $$;
    """
    prof = _profile(tmp_path, {"default": {"simulate_sql": guard, "reply": "ok"}})
    sc = _scenario(tmp_path, "turns: ['x']\n")
    recs = [run_once(sc, prof, NoopDriver(), i, str(tmp_path / "s"), None) for i in (1, 2, 3)]
    assert [r.status for r in recs] == ["completed"] * 3, [r.harness_error for r in recs]
    assert len({r.run_db for r in recs}) == 3 and all(not _db_exists(r.run_db) for r in recs)
    assert len({r.artifacts["run"] for r in recs}) == 3 and all(os.path.exists(r.artifacts["run"]) for r in recs)
    assert all(r.classification.business_write_count == 1 for r in recs)
    assert all(r.driver_result.session_id.endswith(r.run_db) for r in recs)


# ============================================================ concurrency: the same run name cannot be cloned twice
def test_second_clone_of_same_run_name_is_refused_and_first_is_untouched():
    b = PostgresTemplateBackend(TEMPLATE, ROLE)
    env = b.clone_for_run("audit_concurrent")
    try:
        with pytest.raises(PreflightRefused, match="already exists"):
            b.clone_for_run("audit_concurrent")
        assert _db_exists(env.name)
    finally:
        b.destroy_run(env)
        b.destroy_run(env)      # idempotent
    assert not _db_exists(env.name)


# ============================================================ failure paths: teardown, artifacts, template
class _PrepareFails(NoopDriver):
    def prepare(self, *a, **k):
        raise RuntimeError("driver preparation exploded")


class _ExecuteInterrupts(NoopDriver):
    def execute(self, *a, **k):
        raise KeyboardInterrupt


class _CloseFails(NoopDriver):
    def close(self):
        raise OSError("close failed")


def _lead_count(dsn):
    with psycopg.connect(dsn) as c:
        return c.execute("select count(*) from crm_lead").fetchone()[0]


def test_driver_prepare_failure_is_recorded_and_torn_down(tmp_path):
    sc = _scenario(tmp_path, "turns: ['x']\n")
    rec = run_once(sc, _profile(tmp_path, {"default": {}}), _PrepareFails(), 1, str(tmp_path / "s"), None)
    assert rec.status == "harness_error" and "exploded" in rec.harness_error
    assert not _db_exists(rec.run_db)
    saved = json.loads(Path(rec.artifacts["run"]).read_text())
    assert saved["status"] == "harness_error" and saved["outcome"] is None and saved["attribution"] is None
    assert "raw_diff" not in rec.artifacts


def test_keyboard_interrupt_tears_down_and_marks_incomplete(tmp_path):
    sc = _scenario(tmp_path, "turns: ['x']\n")
    with pytest.raises(KeyboardInterrupt):
        run_once(sc, _profile(tmp_path, {"default": {}}), _ExecuteInterrupts(), 1, str(tmp_path / "s"), None)
    p = tmp_path / "s" / "001" / "run.json"
    saved = json.loads(Path(p).read_text())
    assert saved["status"] == "interrupted" and not _db_exists(saved["run_db"])


def test_close_failure_still_drops_the_database(tmp_path):
    sc = _scenario(tmp_path, "turns: ['x']\n")
    # a teardown failure is recorded and surfaced as TeardownError, carrying the record
    with pytest.raises(TeardownError) as ei:
        run_once(sc, _profile(tmp_path, {"default": {}}), _CloseFails(), 1, str(tmp_path / "s"), None)
    assert isinstance(ei.value.__cause__, OSError) and ei.value.record.teardown["dropped"] is True
    assert not _db_exists(f"{RUN_PREFIX}sc_001")


def test_grading_failure_is_recorded_and_torn_down(tmp_path, monkeypatch):
    import agent_review.core.lifecycle as lc
    monkeypatch.setattr(lc, "grade", lambda *a, **k: (_ for _ in ()).throw(ValueError("grader broke")))
    sc = _scenario(tmp_path, "turns: ['x']\n")
    rec = run_once(sc, _profile(tmp_path, {"default": {}}), NoopDriver(), 1, str(tmp_path / "s"), None)
    assert rec.status == "harness_error" and "grader broke" in rec.harness_error and not _db_exists(rec.run_db)
    assert os.path.exists(rec.artifacts["raw_diff"])      # evidence collected before the failure is kept


def test_disk_guard_refuses_before_creating_anything(tmp_path):
    b = PostgresTemplateBackend(TEMPLATE, ROLE, disk_headroom_bytes=10**15)
    with pytest.raises(PreflightRefused, match="disk"):
        b.clone_for_run("audit_disk")
    assert not _db_exists(f"{RUN_PREFIX}audit_disk")


def test_missing_template_and_held_template_refuse(tmp_path):
    with pytest.raises(PreflightRefused, match="does not exist"):
        PostgresTemplateBackend("odexalabs_fx_no_such_tpl", ROLE).prepare()
    b = PostgresTemplateBackend(TEMPLATE, ROLE)
    with psycopg.connect(f"dbname={TEMPLATE}"), pytest.raises(PreflightRefused, match="held open"):   # hold it open
        b.clone_for_run("audit_held")
    assert not _db_exists(f"{RUN_PREFIX}audit_held")


def test_subsequent_run_possible_after_failure(tmp_path):
    sc = _scenario(tmp_path, "turns: ['x']\n")
    prof = _profile(tmp_path, {"default": {}})
    run_once(sc, prof, _PrepareFails(), 1, str(tmp_path / "s"), None)
    rec = run_once(sc, prof, NoopDriver(), 1, str(tmp_path / "s"), None)     # same index, same db name
    assert rec.status == "completed"


# ============================================================ mail: neutralised on the clone, mandatory in every mode
def test_fixture_with_live_mail_server_is_neutralised_on_the_clone_only(tmp_path, template_before):
    """Simulate a customer fixture that carries an SMTP server: the row is archived on the clone
    before the run, the block is verified, and the template keeps its row count."""
    with _Clone("audit_mail") as env:
        with psycopg.connect(env.dsn, autocommit=True) as c:
            c.execute("insert into ir_mail_server (name, smtp_host, smtp_port, smtp_encryption, smtp_authentication, active, sequence, create_date, write_date) "
                      "values ('live smtp', 'smtp.example.com', 587, 'starttls', 'login', true, 10, now(), now())")
        st = verify_mail_blocked(env.observer_dsn, True, True)
        assert not st.mail_blocked_verified and st.mail_servers == 1 and "MAIL NOT BLOCKED" in st.notes[0]
        assert neutralise_mail(env.dsn) == 1
        st2 = verify_mail_blocked(env.observer_dsn, True, True)
        assert st2.mail_blocked_verified and st2.mail_servers == 0
        assert not verify_mail_blocked(env.observer_dsn, False, True).mail_blocked_verified
        assert not verify_mail_blocked(env.observer_dsn, True, False).mail_blocked_verified
        assert set(st2.outbound_integrations) == {"iap_accounts", "payment_providers_enabled", "fetchmail_servers_active"}


def test_run_refuses_when_mail_cannot_be_blocked_in_customer_mode(tmp_path):
    class NoCronGuarantee(NoopDriver):
        def cron_threads_zero(self):
            return False
    sc = _scenario(tmp_path, "turns: ['x']\n")
    rec = run_once(sc, _profile(tmp_path, {"default": {}}, mode="customer"), NoCronGuarantee(), 1, str(tmp_path / "s"), None)
    assert rec.status == "harness_error" and "MailNotBlocked" in rec.harness_error
    assert rec.conditions.egress_isolation == "unavailable" and rec.conditions.mail_blocked_verified is False
    assert rec.driver_result is None and not _db_exists(rec.run_db)


# ============================================================ stored provider key removed on the clone
def test_stored_provider_key_in_fixture_is_removed_on_the_clone():
    from agent_review.drivers.native_ai.driver import STORED_PROVIDER_KEYS, NativeAiDriver
    with _Clone("audit_storedkey") as env:
        with psycopg.connect(env.dsn, autocommit=True) as c:
            c.execute("insert into ir_config_parameter (key, value, create_date, write_date) values ('ai.openai_key', 'sk-STORED-IN-FIXTURE', now(), now())")
        d = NativeAiDriver()
        d.env = env
        d._remove_stored_provider_keys()
        with psycopg.connect(env.observer_dsn) as c:
            assert c.execute("select count(*) from ir_config_parameter where key = any(%s)", (list(STORED_PROVIDER_KEYS),)).fetchone()[0] == 0
        assert any("removed on the clone" in n for n in d.notes) and not any("sk-STORED" in n for n in d.notes)


# ============================================================ pg_stat_statements scope
def test_observer_must_differ_from_execution_role():
    with _Clone("audit_pgstat") as env:
        with pytest.raises(RuntimeError, match="must differ"):
            pgstats.snapshot(f"dbname={env.name} user={ROLE}", env.name, ROLE)
        before = pgstats.snapshot(env.observer_dsn, env.name, ROLE)
        with psycopg.connect(env.observer_dsn) as c:              # observer SQL must not appear in the delta
            c.execute("select count(*) from res_partner").fetchone()
        after = pgstats.snapshot(env.observer_dsn, env.name, ROLE)
        assert pgstats.delta(before, after)["calls"] == 0
        with psycopg.connect(env.dsn) as c:                       # execution-role SQL must
            c.execute("select count(*) from crm_lead").fetchone()
        after2 = pgstats.snapshot(env.observer_dsn, env.name, ROLE)
        d = pgstats.delta(before, after2)
        assert d["calls"] >= 1 and any("crm_lead" in t["query"] for t in d["top"])


# ============================================================ sensitive data end to end
def test_planted_secrets_never_reach_run_json_or_summary(tmp_path):
    plant = """
        update ir_config_parameter set value = 'sk_live_PLANTED_CFG', write_date = now() where key = 'web.base.url';
        insert into ir_config_parameter (key, value, create_date, write_date) values ('stripe.secret_key', 'sk_live_PLANTED_NEW', now(), now());
        update res_users set password = 'PLANTED_PASSWORD', write_date = now() where login = 'evalop';
        update res_partner set email = 'customer@example.com', write_date = now() where name = 'Eval Operator';
    """
    sc = _scenario(tmp_path, "turns: ['x']\n")
    rec = run_once(sc, _profile(tmp_path, {"default": {"simulate_sql": plant}}), NoopDriver(), 1, str(tmp_path / "s"), None)
    assert rec.status == "completed", rec.harness_error
    from agent_review.core.report import run_summary, suite_summary, write_suite
    text, data = suite_summary([rec], "sc", "p", 1)
    write_suite(str(tmp_path / "s"), [rec], text, data)
    # run.json is PRIVATE evidence; it is included here only because its normalised diff is
    # redacted too. The redacted-report guarantee over free text is attacked in test_release_guarantees.py.
    public = (Path(rec.artifacts["run"]).read_text() + (tmp_path / "s" / "summary.json").read_text()
              + (tmp_path / "s" / "summary.txt").read_text() + run_summary(rec))
    assert "PLANTED" not in public
    assert "customer@example.com" in public         # business data is shown unless a customer extends redaction
    raw = Path(rec.artifacts["raw_diff"]).read_text()
    assert "PLANTED_CFG" in raw and "PLANTED_PASSWORD" in raw     # the raw diff is complete, and stays local
    # web.base.url is allowlisted by KEY but the planted value looks like a secret: still shown? The
    # allowlist is by key; a customer who stores a secret under an allowlisted key gets it shown.
    # That is the documented contract; the change is at least visible as a change:
    assert any(n.table == "ir_config_parameter" for n in rec.classification.business_writes)


# ============================================================ performance on the v1 fixture
def test_timings_on_fixture(capsys):
    b = PostgresTemplateBackend(TEMPLATE, ROLE)
    t0 = time.perf_counter()
    env = b.clone_for_run("audit_perf")
    t_clone = time.perf_counter() - t0
    try:
        det = TableDiffDetector()
        from agent_review.core.classify import ClassificationRules
        exact = set(ClassificationRules.load().business_tables)
        t0 = time.perf_counter(); det.before_run(env, exact); t_before = time.perf_counter() - t0
        with psycopg.connect(env.dsn, autocommit=True) as c:
            c.execute("update res_partner set write_date = now() where id = (select min(id) from res_partner)")
        t0 = time.perf_counter(); det.after_run(env); t_after = time.perf_counter() - t0
        t0 = time.perf_counter(); ev = det.collect_changes(); t_diff = time.perf_counter() - t0
        n_tables = len(ev.coverage_by_table)
        n_rows = sum(len(det.rows_before(t)) for t in exact)
    finally:
        t0 = time.perf_counter(); b.destroy_run(env); t_drop = time.perf_counter() - t0
    line = (f"TIMINGS template={TEMPLATE}: clone {t_clone:.2f}s · baseline snapshot {t_before:.2f}s ({n_tables} tables, "
            f"{len(exact & set(ev.coverage_by_table))} exact, {n_rows} rows held) · after snapshot {t_after:.2f}s · diff {t_diff:.3f}s · drop {t_drop:.2f}s")
    print(line)
    Path(os.environ.get("AGENT_REVIEW_TIMINGS_OUT", "/dev/null")).write_text(line + "\n") if os.environ.get("AGENT_REVIEW_TIMINGS_OUT") else None
    assert t_clone < 60 and t_before < 60
