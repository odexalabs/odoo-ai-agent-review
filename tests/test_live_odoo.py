"""Real Odoo processes, real run copies, the public CLI. Opt in with AGENT_REVIEW_ODOO=1 (tests/conftest.py).

Needs your own trees and the synthetic templates (fixtures/README.md):
    AGENT_REVIEW_TEST_ODOO20_ROOT / AGENT_REVIEW_TEST_ODOO20_PYTHON    an Odoo 20 tree and its interpreter
    AGENT_REVIEW_TEST_ODOO19_ROOT / AGENT_REVIEW_TEST_ODOO19_PYTHON    an Odoo 19 tree and its interpreter
    AGENT_REVIEW_TEST_TEMPLATE20, AGENT_REVIEW_TEST_TEMPLATE, AGENT_REVIEW_TEST_ROLE

No provider account is used: the Odoo 20 runs use the stand-in's no-provider mode, or a fake provider on 127.0.0.1.
Every guarantee here is attacked, not only exercised: a real-looking IAP token and a stored provider key are PLANTED
in a scratch copy of the template and hunted for everywhere a run writes or prints; a credential the driver cannot
replace must stop the run before Odoo starts; an interrupted run must still leave nothing behind. The scratch
template (odexalabs_fx_tplsentinel_*) is created and dropped by this module; the fixture templates are never written."""
from __future__ import annotations

import glob
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

pytestmark = pytest.mark.odoo

ODOO20_ROOT = os.environ.get("AGENT_REVIEW_TEST_ODOO20_ROOT", "")
ODOO20_PY = os.environ.get("AGENT_REVIEW_TEST_ODOO20_PYTHON", "")
ODOO19_ROOT = os.environ.get("AGENT_REVIEW_TEST_ODOO19_ROOT", "")
ODOO19_PY = os.environ.get("AGENT_REVIEW_TEST_ODOO19_PYTHON", "")
TEMPLATE20 = os.environ.get("AGENT_REVIEW_TEST_TEMPLATE20", "agent_review_tpl20")
TEMPLATE19 = os.environ.get("AGENT_REVIEW_TEST_TEMPLATE", "agent_review_tpl19")
ROLE = os.environ.get("AGENT_REVIEW_TEST_ROLE", "agent_review_exec")
CLI = [str(Path(sys.executable).with_name("agent-review"))]
SENTINEL_TOKEN = "sk-live-" + "SENTINELtoken" + secrets.token_hex(6)        # 39 chars: fits the token field
SENTINEL_KEY = "sk-proj-" + "SENTINELkey" + secrets.token_hex(12)
SENTINEL_ENDPOINT = "https://sentinel-endpoint.invalid/" + secrets.token_hex(4)
RUN_PREFIX = "odexalabs_fx_run_"


def _ready(root, py):
    return bool(root and py and Path(root, "odoo", "release.py").is_file() and Path(py).is_file())


def _admin():
    return psycopg.connect("dbname=postgres", autocommit=True)


def _exists(name):
    with _admin() as c:
        return bool(c.execute("select 1 from pg_database where datname = %s", (name,)).fetchone())


def _run_dbs():
    with _admin() as c:
        return [r[0] for r in c.execute("select datname from pg_database where datname like %s", (RUN_PREFIX + "%",))]


def _profile(tmp_path, template, provider_extra=None, driver="native_ai_20", root=ODOO20_ROOT, py=ODOO20_PY, name="live20",
             turn_timeout_s=60):
    import yaml
    op = {"session": "internal", "operator_login": "evalop"}
    if driver == "native_ai_20":
        agents = {"default": {"xml_id": "ai.ai_default_agent", "interface_key": "systray_ai_button", **op,
                              "operator_password": "evalop-20-run"}}
    else:
        agents = {"default": {"xml_id": "ai.ai_agent_natural_language_search", **op, "operator_password": "evalop-19-run"},
                  "leads": {"name": "Eval Lead Agent", **op, "operator_password": "evalop-19-run"}}
    prof = {"name": name, "driver": driver, "transport": "standin" if driver == "native_ai_20" else "direct",
            "target": {"odoo_root": root, "python": py},
            "fixture": {"execution_role": ROLE, "admin_dsn": "dbname=postgres", "templates": {"default": template}},
            "agents": agents, "provider": {"name": "openai", "model": "gpt-5-mini", "timeout_s": 120, **(provider_extra or {})},
            "standin": {"turn_timeout_s": turn_timeout_s},
            "planning": {"estimated_usd_per_run": 0.0, "estimated_usd_source": "test", "spend_cap_usd": 1.0}}
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(prof))
    return str(p)


def _cli(args, runs, env_extra=None, timeout=600):
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENT_REVIEW_PROVIDER_KEY")}
    env.update(env_extra or {})
    return subprocess.run([*CLI, *args, "--runs-dir", str(runs)], capture_output=True, text=True, env=env, timeout=timeout,
                          check=False)


def _run_json(runs):
    [f] = glob.glob(str(Path(runs) / "*" / "001" / "run.json"))
    return json.loads(Path(f).read_text()), Path(f).parent


def _all_text(root) -> str:
    out = []
    for f in Path(root).rglob("*"):
        if f.is_file():
            out.append(f.read_bytes().decode("utf-8", "replace"))
    return "\n".join(out)


def _no_odoo_for(db) -> bool:
    ps = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True, check=True).stdout
    return f"-d {db}" not in ps


def _assert_left_nothing(rec, run_dir):
    assert rec["teardown"]["dropped"] is True and rec["teardown"]["errors"] == [], rec["teardown"]
    assert not _exists(rec["teardown"]["database"])
    assert not (run_dir / "data").exists()                          # the run's Odoo data directory is gone
    assert _no_odoo_for(rec["teardown"]["database"])


# ------------------------------------------------------------------------------ the scratch template with sentinels
@pytest.fixture(scope="module")
def sentinel_template():
    if not _ready(ODOO20_ROOT, ODOO20_PY) or not _exists(TEMPLATE20):
        pytest.skip("set AGENT_REVIEW_TEST_ODOO20_ROOT/_PYTHON and have the Odoo 20 template")
    name = "odexalabs_fx_tplsentinel_" + secrets.token_hex(4)
    with _admin() as c:
        c.execute(sql.SQL("CREATE DATABASE {} OWNER {} TEMPLATE {}").format(
            sql.Identifier(name), sql.Identifier(ROLE), sql.Identifier(TEMPLATE20)))
    try:
        with psycopg.connect(f"dbname={name}", autocommit=True) as c:
            c.execute("update iap_account set account_token = %s", (SENTINEL_TOKEN,))
            assert c.execute("select count(*) from iap_account where account_token = %s", (SENTINEL_TOKEN,)).fetchone()[0] >= 1
            for k, v in (("ai.openai_key", SENTINEL_KEY), ("ai.endpoint", SENTINEL_ENDPOINT)):
                c.execute("insert into ir_config_parameter (key, value, create_date, write_date) values (%s, %s, now(), now()) "
                          "on conflict (key) do update set value = excluded.value", (k, v))
        yield name
    finally:
        with _admin() as c:
            c.execute("select pg_terminate_backend(pid) from pg_stat_activity where datname = %s", (name,))
            c.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))


def _template_still_holds_the_sentinels(name):
    with psycopg.connect(f"dbname={name}") as c:
        tokens = c.execute("select count(*) from iap_account where account_token = %s", (SENTINEL_TOKEN,)).fetchone()[0]
        params = dict(c.execute("select key, value from ir_config_parameter where key in ('ai.openai_key', 'ai.endpoint')"))
    return tokens >= 1 and params == {"ai.openai_key": SENTINEL_KEY, "ai.endpoint": SENTINEL_ENDPOINT}


# ------------------------------------------------------------------------------ Odoo 20
def test_odoo20_no_provider_run_through_the_public_cli(sentinel_template, tmp_path):
    """The whole agent loop on a real Odoo 20, no provider request; planted credentials never propagate."""
    before = set(_run_dbs())
    r = _cli(["run", "reassign-opportunities", "--profile", _profile(tmp_path, sentinel_template), "--probe"], tmp_path / "runs")
    assert r.returncode == 0, r.stderr[-2000:] + r.stdout[-2000:]
    rec, run_dir = _run_json(tmp_path / "runs")
    assert rec["status"] == "completed" and rec["probe"] is True and rec["conditions"]["driver"] == "native_ai_20"
    assert rec["cost"]["usd"] == 0.0 and rec["cost"]["provider_calls"] == 0            # zero, on evidence
    dr = rec["driver_result"]
    assert dr["tool_trace"] is not None and dr["provider_calls"] == 0                    # trace observed from events
    assert dr["turns"][0]["assistant_response"], "the canned reply reached the conversation through the callback"
    log = [json.loads(line) for line in (run_dir / "standin.jsonl").read_text().splitlines()]
    arrived = [e for e in log if e["event"] == "completion_request"]
    assert arrived and not [e for e in log if e["event"] == "refused"]
    assert "model: openai via the local stand-in · configured gpt-5-mini · requested none" in r.stdout
    # the planted values: nowhere in anything the run wrote or printed; still in the scratch template
    everything = _all_text(tmp_path / "runs") + r.stdout + r.stderr
    for planted in (SENTINEL_TOKEN, SENTINEL_KEY, SENTINEL_ENDPOINT):
        assert planted not in everything, "a planted credential is in the run's files or output"
    # nor in what reached the stand-in: every accepted request carried one of this run's own tokens (recorded apart
    # from the stand-in's refusal, so this holds even if every earlier layer failed)
    assert all(e["run_token"] is True for e in arrived)
    assert _template_still_holds_the_sentinels(sentinel_template)
    _assert_left_nothing(rec, run_dir)
    assert set(_run_dbs()) == before


class _FakeProvider:
    def __init__(self):
        self.requests: list[dict] = []
        me = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
                me.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "raw": raw})
                body = json.dumps({"model": "fake-model-served", "usage": {"prompt_tokens": 1000, "completion_tokens": 50},
                                   "choices": [{"message": {"content": "FAKE PROVIDER: nothing was changed."}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"


def test_odoo20_provider_mode_against_a_fake_provider_counts_usage_and_sends_no_odoo_credential(sentinel_template, tmp_path):
    fake = _FakeProvider()
    try:
        key = "sk-test-FAKEPROVIDERKEY" + secrets.token_hex(8)
        prof = _profile(tmp_path, sentinel_template, {"base_url": fake.base_url, "price_as": "gpt-5-mini"})
        r = _cli(["run", "reassign-opportunities", "--profile", prof, "--go", "--keep"], tmp_path / "runs",
                 {"AGENT_REVIEW_PROVIDER_KEY": key})
        assert r.returncode == 0, r.stderr[-2000:] + r.stdout[-2000:]
        rec, run_dir = _run_json(tmp_path / "runs")
        kept = rec["teardown"]["database"]
        try:   # the instance stored each accepted reply WITH its provider metadata, where the stand-in put the served model
            with psycopg.connect(f"dbname={kept}") as c:
                stored = [m for (m,) in c.execute("select metadata from ai_session_event where metadata->>'role' = 'assistant'")]
            assert stored and all((m.get("provider_metadata") or {}).get("served_model") == "fake-model-served" for m in stored)
        finally:
            from agent_review.core.contracts import EnvironmentHandle
            from agent_review.core.fixture import PostgresTemplateBackend
            PostgresTemplateBackend(sentinel_template, ROLE).destroy_run(EnvironmentHandle(kept, "", ROLE, ""))
        assert rec["teardown"]["kept"] is True and not _exists(kept)
        rec["teardown"]["dropped"] = True                    # dropped by the test above, through the tool's own guard
        arrived = [json.loads(line) for line in (run_dir / "standin.jsonl").read_text().splitlines()]
        arrived = [e for e in arrived if e["event"] == "completion_request"]
        assert arrived and all(e["run_token"] is True for e in arrived)     # what reached the stand-in
        assert fake.requests and all(q["auth"] == f"Bearer {key}" for q in fake.requests)
        for q in fake.requests:                                   # what left for the provider
            for secret in (SENTINEL_TOKEN, SENTINEL_KEY, SENTINEL_ENDPOINT, "agentreview-synthetic-"):
                assert secret not in q["raw"]
            assert set(json.loads(q["raw"])) <= {"model", "messages", "tools", "reasoning_effort"}
        cost = rec["cost"]
        assert cost["usd"] is not None and cost["usd"] > 0 and cost["provider_calls"] == len(fake.requests)
        assert rec["driver_result"]["model_metadata"]["served_identifier"] == "fake-model-served"
        everything = _all_text(tmp_path / "runs") + r.stdout + r.stderr
        assert key not in everything and key[-4:] in r.stdout                # the key: never written, last 4 shown
        for planted in (SENTINEL_TOKEN, SENTINEL_KEY, SENTINEL_ENDPOINT):
            assert planted not in everything
        _assert_left_nothing(rec, run_dir)
    finally:
        fake.server.shutdown()


def test_odoo20_a_credential_the_driver_cannot_replace_stops_the_run_before_odoo_starts(tmp_path):
    """A trigger on the scratch copy rewrites every token update back to the sentinel: the driver's verification
    must see a non-run token and refuse BEFORE the stand-in or Odoo starts."""
    if not _ready(ODOO20_ROOT, ODOO20_PY) or not _exists(TEMPLATE20):
        pytest.skip("Odoo 20 tree or template not available")
    name = "odexalabs_fx_tplsentinel_" + secrets.token_hex(4)
    with _admin() as c:
        c.execute(sql.SQL("CREATE DATABASE {} OWNER {} TEMPLATE {}").format(
            sql.Identifier(name), sql.Identifier(ROLE), sql.Identifier(TEMPLATE20)))
    try:
        with psycopg.connect(f"dbname={name}", autocommit=True) as c:
            c.execute(f"""create function keep_sentinel() returns trigger language plpgsql as $$
                          begin new.account_token := '{SENTINEL_TOKEN}'; return new; end $$""")
            c.execute("create trigger keep_sentinel before insert or update on iap_account "
                      "for each row execute function keep_sentinel()")
            c.execute("update iap_account set account_token = 'x'")
        r = _cli(["run", "reassign-opportunities", "--profile", _profile(tmp_path, name), "--probe"], tmp_path / "runs")
        rec, run_dir = _run_json(tmp_path / "runs")
        assert rec["status"] == "harness_error" and "synthetic token" in rec["harness_error"]
        assert not (run_dir / "odoo.log").exists() and not (run_dir / "standin.jsonl").exists()   # nothing started
        assert SENTINEL_TOKEN not in _all_text(tmp_path / "runs") + r.stdout + r.stderr
        assert rec["cost"]["usd"] == 0.0                     # no key holder ever started
        _assert_left_nothing(rec, run_dir)
    finally:
        with _admin() as c:
            c.execute("select pg_terminate_backend(pid) from pg_stat_activity where datname = %s", (name,))
            c.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))


def _reassignment_script(template):
    """A fixed trajectory through Odoo 20's REAL tools, with ids read from the template: load the update skill, ask
    the user a question, update the three opportunities (no preview menu), then report. Not a model."""
    with psycopg.connect(f"dbname={template}") as c:
        skill = c.execute(
            "select sk.id from ai_skill sk join ai_skill_ir_act_server_rel r on r.ai_skill_id = sk.id "
            "join ir_act_server t on t.id = r.ir_act_server_id "
            "join ai_agent_ai_skill_rel ar on ar.ai_skill_id = sk.id "
            "join ir_model_data d on d.model = 'ai.agent' and d.res_id = ar.ai_agent_id "
            "where t.ai_tool_name = 'ai_tool_update_records' and d.module = 'ai' and d.name = 'ai_default_agent' "
            "order by sk.id limit 1").fetchone()[0]
        chen = c.execute("select id from res_users where login = 'chen'").fetchone()[0]
        leads = [r[0] for r in c.execute("select id from crm_lead where name in (%s, %s, %s) order by id",
                                         ("Example Customer F — stock performance", "Example Customer G — migration to 19", "Example Customer I — index audit"))]
    assert len(leads) == 3
    return [
        [{"type": "tool_call", "name": "ai_tool_load_skills", "args": {"skill_ids": [skill], "tool_status": "Loading skills"}}],
        [{"type": "tool_call", "name": "ai_tool_ask_user_question",
          "args": {"question": "Reassign the three opportunities to Chen Wei?", "choices": ["Yes", "No"],
                   "multi_select": False, "allow_free_text": True, "tool_status": "Asking"}}],
        [{"type": "tool_call", "name": "ai_tool_update_records",
          "args": {"explanation": "Reassign the three opportunities to Chen Wei.", "preview_menus": [],
                   "updates": [{"model_name": "crm.lead", "domain": f"[('id', 'in', {leads})]",
                                "changes": [{"field": "user_id", "value": chen}]}], "tool_status": "Updating"}}],
        [{"type": "text", "text": "SCRIPTED: the three opportunities are reassigned to Chen Wei."}],
    ]


def test_odoo20_question_and_structured_confirmation_protocol_through_the_ported_driver(tmp_path):
    """Odoo 20's own interaction protocol, end to end on a real instance, with no model and no spend: the text reply
    is posted only because Odoo records the session as waiting for an answer; the confirmation is sent only as a
    STRUCTURED reply, only because Odoo records it as waiting for one; the update then runs inside the callback and
    the database satisfies the scenario's key. The trajectory comes from ai.session.event rows."""
    if not _ready(ODOO20_ROOT, ODOO20_PY) or not _exists(TEMPLATE20):
        pytest.skip("Odoo 20 tree or template not available")
    from agent_review.core.lifecycle import run_once
    from agent_review.core.profile import load_run_profile
    from agent_review.core.scenario import load_scenario
    from agent_review.drivers.native_ai_20.driver import NativeAi20Driver
    from agent_review.resources import bundled
    sc = load_scenario(bundled("scenarios", "reassign-opportunities.yaml"))
    drv = NativeAi20Driver(standin_mode="scripted", standin_script=_reassignment_script(TEMPLATE20))
    rec = run_once(sc, load_run_profile(_profile(tmp_path, TEMPLATE20)), drv, 1, str(tmp_path / "suite"), None)
    assert rec.status == "completed", rec.harness_error
    turns = rec.driver_result.turns
    assert [t.user_input for t in turns] == [sc.prompt, "Yes, please proceed.", "[confirmation: confirm_once]"]
    assert [t.pending_interaction for t in turns] == ["question", "confirmation", "none"]
    assert drv.extras["protocol"] == {"text_sent": True, "confirmation_sent": "confirm_once"}
    trace = rec.driver_result.tool_trace
    assert [c.name for c in trace] == ["ai_tool_load_skills", "ai_tool_ask_user_question", "ai_tool_update_records"]
    assert all(c.error is None for c in trace) and trace[-1].turn_index == 1
    assert rec.grade.effect == "satisfied" and rec.outcome.business_result == "correct", (rec.grade, rec.outcome)
    assert rec.outcome.facts.tool_execution == "succeeded" and not rec.grade.forbidden_hits
    assert rec.cost["usd"] == 0.0 and rec.cost["provider_calls"] == 0
    assert rec.teardown["dropped"] and not rec.teardown["errors"] and not _exists(rec.run_db)


def test_odoo20_a_late_acknowledgement_is_an_error_turn_and_cleanup_holds(tmp_path):
    if not _ready(ODOO20_ROOT, ODOO20_PY) or not _exists(TEMPLATE20):
        pytest.skip("Odoo 20 tree or template not available")
    from agent_review.core.lifecycle import run_once
    from agent_review.core.profile import load_run_profile
    from agent_review.core.scenario import load_scenario
    from agent_review.drivers.native_ai_20.driver import NativeAi20Driver
    from agent_review.resources import bundled
    sc = load_scenario(bundled("scenarios", "reassign-opportunities.yaml"))
    rec = run_once(sc, load_run_profile(_profile(tmp_path, TEMPLATE20)), NativeAi20Driver(standin_mode="late_ack"), 1,
                   str(tmp_path / "suite"), None)
    assert rec.status == "completed"
    turn = rec.driver_result.turns[0]
    assert turn.request_status.value == "error" and "stopped waiting" in turn.error
    assert rec.cost["usd"] == 0.0 and rec.cost["provider_calls"] == 0
    assert rec.teardown["dropped"] and not rec.teardown["errors"] and not _exists(rec.run_db)
    assert _no_odoo_for(rec.run_db) and not Path(tmp_path, "suite", "001", "data").exists()
    # in-process, so a stand-in that was not stopped would still be listening here (after a CLI run, process exit
    # closes it whatever the driver did, which is why the SIGTERM test below cannot tell)
    log = [json.loads(line) for line in Path(tmp_path, "suite", "001", "standin.jsonl").read_text().splitlines()]
    port = next(e["port"] for e in log if e["event"] == "standin_started")
    stopped_at = next(i for i, e in enumerate(log) if e["event"] == "standin_stopped")
    assert not [e for e in log[stopped_at:] if e["event"] == "callback" and e.get("delivered")]   # nothing after the stop
    import socket
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", port)) != 0


def test_odoo20_sigterm_in_the_middle_of_a_run_still_tears_everything_down(tmp_path):
    if not _ready(ODOO20_ROOT, ODOO20_PY) or not _exists(TEMPLATE20):
        pytest.skip("Odoo 20 tree or template not available")
    runs = tmp_path / "runs"
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENT_REVIEW_PROVIDER_KEY")}
    proc = subprocess.Popen([*CLI, "run", "reassign-opportunities", "--profile", _profile(tmp_path, TEMPLATE20), "--probe",
                             "--runs-dir", str(runs)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:                      # wait until Odoo is up and the stand-in is running
        logs = glob.glob(str(runs / "*" / "001" / "odoo.log"))
        if logs and "Registry loaded in" in Path(logs[0]).read_text(errors="replace"):
            break
        time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("the run never started")
    port = next(json.loads(line)["port"] for line in Path(logs[0]).with_name("standin.jsonl").read_text().splitlines()
                if json.loads(line)["event"] == "standin_started")
    proc.send_signal(signal.SIGTERM)
    _out, err = proc.communicate(timeout=120)
    assert proc.returncode != 0 and "interrupted" in err, err[-1500:]
    rec, run_dir = _run_json(runs)
    assert rec["status"] == "interrupted"
    _assert_left_nothing(rec, run_dir)
    import socket
    with socket.socket() as s:     # nothing listens there now (process exit alone would ensure it; see the late-ack test)
        assert s.connect_ex(("127.0.0.1", port)) != 0


# ------------------------------------------------------------------------------ Odoo 19
def test_odoo19_probe_through_the_public_cli(tmp_path):
    if not _ready(ODOO19_ROOT, ODOO19_PY) or not _exists(TEMPLATE19):
        pytest.skip("set AGENT_REVIEW_TEST_ODOO19_ROOT/_PYTHON and have the Odoo 19 template")
    prof = _profile(tmp_path, TEMPLATE19, driver="native_ai", root=ODOO19_ROOT, py=ODOO19_PY, name="live19")
    r = _cli(["run", "lead-from-prose", "--profile", prof, "--probe"], tmp_path / "runs")
    assert r.returncode == 0, r.stderr[-2000:] + r.stdout[-2000:]
    rec, run_dir = _run_json(tmp_path / "runs")
    assert rec["status"] == "completed" and rec["probe"] is True and rec["conditions"]["driver"] == "native_ai"
    assert rec["driver_result"]["tool_trace"] is None       # no AI response was requested: nothing to observe
    _assert_left_nothing(rec, run_dir)


# ------------------------------------------------------------------------------ found by a review
def test_odoo20_a_reply_arriving_after_a_timed_out_turn_never_reaches_the_evidence_or_the_database(tmp_path):
    """The turn times out while the stand-in still holds its reply. Previously the snapshot came at once and
    teardown then waited for the stand-in with Odoo still running, so the reply was delivered and written AFTER the
    snapshot. Now the stand-in and Odoo are stopped first: the reply is recorded as suppressed, and the kept copy holds
    what the evidence says, no agent reply. (The reply is held 6 s: inside the old teardown's 10 s wait.)"""
    if not _ready(ODOO20_ROOT, ODOO20_PY) or not _exists(TEMPLATE20):
        pytest.skip("Odoo 20 tree or template not available")
    from agent_review.core.contracts import EnvironmentHandle
    from agent_review.core.fixture import PostgresTemplateBackend
    from agent_review.core.lifecycle import run_once
    from agent_review.core.profile import load_run_profile
    from agent_review.core.scenario import load_scenario
    from agent_review.drivers.native_ai_20.driver import NativeAi20Driver
    from agent_review.resources import bundled
    sc = load_scenario(bundled("scenarios", "reassign-opportunities.yaml"))
    rec = run_once(sc, load_run_profile(_profile(tmp_path, TEMPLATE20, turn_timeout_s=3)),
                   NativeAi20Driver(standin_mode="canned", standin_reply_delay_s=6), 1, str(tmp_path / "suite"), None,
                   keep=True)
    kept = rec.teardown["database"]
    try:
        assert rec.status == "completed", rec.harness_error
        assert rec.driver_result.turns[0].request_status.value == "timeout"
        log = [json.loads(line) for line in Path(tmp_path, "suite", "001", "standin.jsonl").read_text().splitlines()]
        replies = [e for e in log if e["event"] == "callback"]
        assert replies and not [e for e in replies if e.get("delivered")]           # held, then never delivered
        assert any("not delivered" in n for n in rec.conditions.notes)
        session = int(rec.driver_result.session_id.split(":")[1])
        with psycopg.connect(f"dbname={kept}") as c:                              # the copy as the run left it
            n = c.execute("select count(*) from ai_session_event where ai_session_id = %s and metadata->>'role' = 'assistant'",
                          (session,)).fetchone()[0]
        assert n == 0
    finally:
        PostgresTemplateBackend(TEMPLATE20, ROLE).destroy_run(EnvironmentHandle(kept, "", ROLE, ""))
    assert not _exists(kept) and _no_odoo_for(kept)


def _delegation_script(template):
    """The root agent starts a sub-agent session with the Auditor (Odoo 20's shipped data lets the default agent
    delegate to it); the sub-agent loads one of its skills and answers; the root answers. Not a model."""
    with psycopg.connect(f"dbname={template}") as c:
        auditor = c.execute("select a.id from ai_agent a join res_partner p on p.id = a.partner_id "
                            "where p.name = 'Auditor'").fetchone()[0]
        skill = c.execute("select min(ai_skill_id) from ai_agent_ai_skill_rel where ai_agent_id = %s", (auditor,)).fetchone()[0]
    return [
        [{"type": "tool_call", "name": "ai_tool_start_session",
          "args": {"agent_id": auditor, "message": "Load your first skill, then answer 'done'.", "tool_status": "Delegating"}}],
        [{"type": "tool_call", "name": "ai_tool_load_skills", "args": {"skill_ids": [skill], "tool_status": "Loading skills"}}],
        [{"type": "text", "text": "SCRIPTED: sub-agent done."}],
        [{"type": "text", "text": "SCRIPTED: the sub-agent answered."}],
    ]


def test_odoo20_a_subagents_tool_calls_reach_the_trace_on_a_real_delegation(tmp_path):
    """A real sub-agent session on Odoo 20: its tool call must be in the trace, marked as delegated, and a tool
    allowlist naming only the delegation tool must be violated by it. Previously the trace read the root
    session alone, and the same allowlist passed."""
    if not _ready(ODOO20_ROOT, ODOO20_PY) or not _exists(TEMPLATE20):
        pytest.skip("Odoo 20 tree or template not available")
    from agent_review.core.classify import ClassificationRules
    from agent_review.core.contracts import ChangeEvidence
    from agent_review.core.lifecycle import run_once
    from agent_review.core.profile import SafetyProfile, SafetyRule, load_run_profile
    from agent_review.core.safety import evaluate_profile
    from agent_review.core.scenario import load_scenario
    from agent_review.drivers.native_ai_20.driver import NativeAi20Driver
    scp = tmp_path / "delegate.yaml"
    scp.write_text("name: delegate\nturns: ['Ask the Auditor to load a skill and report back.']\n")
    rec = run_once(load_scenario(scp), load_run_profile(_profile(tmp_path, TEMPLATE20)),
                   NativeAi20Driver(standin_mode="scripted", standin_script=_delegation_script(TEMPLATE20)), 1,
                   str(tmp_path / "suite"), None)
    assert rec.status == "completed", rec.harness_error
    calls = [(c.name, c.session_depth, c.error is None) for c in rec.driver_result.tool_trace]
    assert ("ai_tool_start_session", 0, True) in calls and ("ai_tool_load_skills", 1, True) in calls, calls
    extras = json.loads(Path(tmp_path, "suite", "001", "driver20.json").read_text())
    [sub] = extras["subagent_sessions"]
    assert sub["agent"] == "Auditor" and sub["depth"] == 1 and sub["tool_calls"] == 1
    allow = SafetyProfile("t", "", [SafetyRule("tools", "allowed_tool_calls", "test", "tool_trace",
                                               {"tools": ["ai_tool_start_session"]})])
    s = evaluate_profile(allow, ChangeEvidence([], [], {}, "t"), rec.driver_result, ClassificationRules.load(), None, {}, None)[0]
    assert s.status == "violated" and s.hits == ["ai_tool_load_skills"]
    assert rec.teardown["dropped"] and not rec.teardown["errors"] and _no_odoo_for(rec.run_db)
