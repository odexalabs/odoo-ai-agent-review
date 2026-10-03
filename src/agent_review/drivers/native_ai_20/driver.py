"""Native Odoo 20 AI driver, through a per-run local stand-in for Odoo's AI endpoint.

What differs from Odoo 19, and what this driver does about it:

  - The Odoo 20 process never calls a model provider. Every agent round goes to the endpoint named by the
    `ai.endpoint` system parameter (by default Odoo's hosted AI service, paid in IAP credits). On the run's
    DISPOSABLE copy only, this driver points `ai.endpoint` at a stand-in it starts for the run, and the stand-in
    calls the operator's own provider with the one named key. The Odoo process receives no provider key.
  - The request carries the database's IAP account token. Before Odoo or the stand-in starts, every IAP token on
    the copy is replaced with a synthetic run token; the stand-in answers only those exact tokens and refuses
    anything else, so a restored customer credential can neither reach the stand-in nor be forwarded.
  - A turn is POST /ai/start_session_advance, which returns at once. Completion is detected by polling the
    session's `loop_state` until it settles and the stand-in has nothing in flight.
  - A confirmation is a STRUCTURED reply (POST /ai/resume_pending_interaction {kind: confirmation, value}),
    sent only when Odoo records the session as waiting for one; a question is answered with a text reply only
    when Odoo records the session as waiting for an answer.
  - The trajectory is read from `ai.session.event` rows (tool calls are not logged at INFO on Odoo 20); token
    usage is the provider's own, as the stand-in observed it.

Scope, stated in every run's conditions: this evaluates Odoo 20's local agent implementation through our
model connection. It does not evaluate Odoo's hosted AI service — its model selection, billing, prompt
processing or retries."""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx
import psycopg
from psycopg.rows import dict_row

from ...core.contracts import DriverRunResult, ModelMetadata, RequestStatus, TokenUsage, ToolCall, Turn
from ...core.credentials import scrubbed_environment
from ...core.fixture import RUN_PREFIX
from ...core.profile import ProfileError, RunProfile
from ...core.scenario import Continuation
from ..base import Capabilities
from ..native_ai.driver import Client, DriverCloseError, NativeAiDriver, RpcError, _free_port, _port_open
from ..odoo_target import check_template, check_tree, launcher, template_facts
from .capabilities import discover20
from .protocol import SYNTHETIC_TOKEN_PREFIX, parts_text, synthetic_token
from .standin import (
    DEFAULT_PROVIDER_BASE_URL,
    NO_PROVIDER_MODES,
    StandIn,
    StandInWorkersAlive,
    check_provider_base_url,
)

SETTLED = {"ready", "waiting_confirmation", "waiting_answer", "waiting_client_result", "waiting_external_result"}
PENDING_BY_LOOP_STATE = {"ready": "none", "waiting_answer": "question", "waiting_confirmation": "confirmation",
                         "waiting_client_result": "client_result", "waiting_external_result": "external_result",
                         "waiting_model": "none", "waiting_child": "none"}
WAITING_ON_USER = ("waiting_confirmation", "waiting_answer")
SCOPE = ("Evaluates Odoo 20's local agent implementation (agent loop, tools, confirmation protocol, database "
         "effects) through the tool's own model connection, a local stand-in for Odoo's AI endpoint. Does NOT "
         "evaluate Odoo's hosted AI service: its model selection, billing (IAP credits), prompt processing or retries.")
SUPPORTED_PROVIDERS = ("openai",)       # the stand-in translates to OpenAI Chat Completions


class NativeAi20Driver(NativeAiDriver):
    name = "native_ai_20"
    accepts_structured_continuation = True

    def __init__(self, standin_mode: str | None = None, standin_script: list[list[dict]] | None = None,
                 standin_reply_delay_s: float = 0.0, standin_stop_deadline_s: float = 10.0) -> None:
        super().__init__()
        self._mode_override = standin_mode     # fault-injection and scripted modes, for tests; the CLI never sets it
        self._script = standin_script           # the scripted mode's rounds (tests only)
        self._reply_delay_s = standin_reply_delay_s        # tests only: every reply held this long
        self._stop_deadline_s = standin_stop_deadline_s
        self.standin: StandIn | None = None
        self.session_id: int | None = None
        self.turn_event_marks: list[int] = []
        self.pending_at_turn: list[set[str]] = []
        self.typed_turn: list[bool] = []
        self.iap_tokens: set[str] = set()
        self.extras: dict[str, Any] = {"turns": []}
        self.turn_timeout_s = 300.0

    # ------------------------------------------------------------ selection
    @property
    def mode(self) -> str:
        if self._mode_override:
            return self._mode_override
        return "canned" if self.probe else "provider"

    def is_paid(self) -> bool:
        return self.mode == "provider"

    def transport_description(self) -> str:
        if self.mode in NO_PROVIDER_MODES:
            return f"local stand-in for Odoo's AI endpoint, mode {self.mode}: no provider request is made"
        prov = self.profile.provider_name if self.profile else "the provider"
        return f"local stand-in for Odoo's AI endpoint, mode provider: the stand-in calls {prov} with the run's named key"

    def scope_note(self) -> str | None:
        return SCOPE

    def check_scenario(self, sc) -> None:
        self.refuse_other_versions(sc, "20")

    def preflight(self, profile: RunProfile, template_dsn: str) -> list[str]:
        if profile.transport != "standin":
            raise ProfileError(f"profile {profile.name}: the native_ai_20 driver needs `transport: standin`; "
                               "Odoo's hosted AI service (IAP) is not supported")
        version, build = check_tree(profile, "20", self.name)
        launcher(profile.target["odoo_root"])
        prov = profile.provider
        if self.mode == "provider":
            if profile.provider_name not in SUPPORTED_PROVIDERS:
                raise ProfileError(f"profile {profile.name}: provider {profile.provider_name!r} is not supported by the "
                                   f"stand-in (supported: {SUPPORTED_PROVIDERS})")
            if not profile.model:
                raise ProfileError(f"profile {profile.name}: provider.model is not set")
            try:
                check_provider_base_url(prov.get("base_url") or DEFAULT_PROVIDER_BASE_URL)
            except ValueError as e:
                raise ProfileError(f"profile {profile.name}: {e}") from e
        template = template_dsn.split("dbname=", 1)[-1].split()[0]
        facts = template_facts(template_dsn, ["ai_session.loop_state", "ai_agent.llm_model"], ["iap_service", "ir_access"])
        check_template(facts, "20", self.name, template, ["ai_session.loop_state"], ["ai_agent.llm_model"])
        if not facts.odoo_ai_service:
            raise ProfileError(f"template {template}: no `odoo_ai` IAP service row; the Odoo 20 `ai` module is not "
                               "set up in this database")
        return [f"target Odoo {version} ({build or 'build unknown'}); template base {facts.base_version}"]

    # ------------------------------------------------------------ lifecycle
    def prepare(self, env, profile, agent_role, credential, run_dir, session_kind) -> None:
        self.env, self.profile, self.credential, self.run_dir = env, profile, credential, run_dir
        self.agent_cfg = profile.agent(agent_role)
        self.session_kind = session_kind or self.agent_cfg.get("session", "internal")
        if self.session_kind != "internal":
            raise ProfileError("the native_ai_20 driver drives internal (logged-in user) sessions only")
        if self.mode == "provider" and credential is None:
            raise ProfileError("the stand-in's provider mode needs the named credential (it is the only key holder)")
        self.turn_timeout_s = float((profile.standin or {}).get("turn_timeout_s", profile.provider.get("timeout_s", 300)))
        os.makedirs(run_dir, exist_ok=True)
        self.caps = discover20(env.observer_dsn, profile.target.get("odoo_root"))
        self._bind_agent_20()
        self._bind_composer()
        self._remove_stored_provider_keys()
        self._neutralise_iap_credentials()
        self.port = int(self.agent_cfg.get("port") or _free_port())
        base = f"http://127.0.0.1:{self.port}"
        prov = profile.provider
        self.standin = StandIn(instance_base_url=base, instance_db=env.name, expected_tokens=self.iap_tokens,
                               log_path=os.path.join(run_dir, "standin.jsonl"), mode=self.mode,
                               api_key=credential.value if (self.mode == "provider" and credential) else None,
                               model=profile.model or None,
                               provider_base_url=prov.get("base_url") or DEFAULT_PROVIDER_BASE_URL,
                               reasoning_effort=prov.get("reasoning_effort", "medium"),
                               provider_timeout_s=float(prov.get("timeout_s", 300)), script=self._script,
                               reply_delay_s=self._reply_delay_s, stop_deadline_s=self._stop_deadline_s)
        self.standin.start()
        self._point_clone(base, self.standin.url)
        conf = self._write_conf()
        self._start(conf)
        self.client = Client(self.port, env.name, float(prov.get("timeout_s", 300)))
        self._open_session_20()

    def _require_run_clone(self, what: str) -> None:
        name = (self.env.name if self.env else "") or ""
        template = (self.env.extras or {}).get("template") if self.env else None
        if not name.startswith(RUN_PREFIX) or name == template:
            raise RuntimeError(f"refusing to {what} on {name!r}: only a disposable {RUN_PREFIX}* run database is changed")

    def _bind_agent_20(self) -> None:
        xml_id = self.agent_cfg.get("xml_id")
        with psycopg.connect(self.env.observer_dsn, row_factory=dict_row) as c:
            if xml_id:
                mod, _, nm = xml_id.partition(".")
                row = c.execute("select res_id from ir_model_data where model = 'ai.agent' and module = %s and name = %s",
                                (mod, nm)).fetchone()
                if not row:
                    raise ProfileError(f"agent xml_id {xml_id!r} not found in the fixture")
                agent = c.execute("select a.id, a.partner_id, p.name from ai_agent a join res_partner p on p.id = a.partner_id "
                                  "where a.id = %s", (row["res_id"],)).fetchone()
            else:
                agent = c.execute("select a.id, a.partner_id, p.name from ai_agent a join res_partner p on p.id = a.partner_id "
                                  "where p.name = %s", (self.agent_cfg.get("name"),)).fetchone()
        if not agent:
            raise ProfileError(f"agent {xml_id or self.agent_cfg.get('name')!r} not found in the fixture")
        self.agent = dict(agent)
        # Odoo 20's ai.agent has no model field and its chat requests set none: the model is the one the run
        # profile names, requested by the stand-in.
        self.agent["llm_model"] = self.profile.model or None

    def _bind_composer(self) -> None:
        """Optional configuration under test: point a composer (e.g. the systray "Ask AI") at the scenario's
        agent, on the DISPOSABLE copy only. A composer's agent is an ordinary editable field."""
        xml_id = self.agent_cfg.get("bind_composer")
        if not xml_id:
            return
        self._require_run_clone("rebind a composer")
        mod, _, nm = xml_id.partition(".")
        with psycopg.connect(self.env.dsn, autocommit=True) as c:
            row = c.execute("select res_id from ir_model_data where model = 'ai.composer' and module = %s and name = %s",
                            (mod, nm)).fetchone()
            if not row:
                raise ProfileError(f"composer {xml_id!r} not found")
            before = c.execute("select ai_agent_id from ai_composer where id = %s", (row[0],)).fetchone()[0]
            c.execute("update ai_composer set ai_agent_id = %s where id = %s", (self.agent["id"], row[0]))
        self.extras["composer_bound"] = {"composer": xml_id, "agent_before": before, "agent_after": self.agent["id"]}
        self.notes.append(f"CONFIGURATION UNDER TEST: composer {xml_id} bound to agent {self.agent['name']!r} "
                          f"(id {self.agent['id']}, was {before}) on the run's copy only")

    def _neutralise_iap_credentials(self) -> None:
        """Replace EVERY IAP account token on the disposable copy with a synthetic run token, before any process
        that could send one starts. A restored customer database carries live tokens: Odoo 20 sends the
        database's `odoo_ai` token with every agent round to whatever `ai.endpoint` names, and other IAP
        services send theirs to Odoo's servers. Values are never read into this process — only counted — and
        never written to the run's record; the template is not touched."""
        self._require_run_clone("replace IAP credentials")
        tokens: set[str] = set()
        with psycopg.connect(self.env.dsn, autocommit=True) as c, c.transaction():
            total, real = c.execute("select count(*), count(*) filter (where account_token is not null and "
                                    "account_token not like %s) from iap_account",
                                    (SYNTHETIC_TOKEN_PREFIX + "%",)).fetchone()
            for (aid,) in c.execute("select id from iap_account order by id").fetchall():
                t = synthetic_token()
                c.execute("update iap_account set account_token = %s where id = %s", (t, aid))
                tokens.add(t)
            service = c.execute("select id from iap_service where technical_name = 'odoo_ai'").fetchone()
            if not service:
                raise ProfileError("no `odoo_ai` IAP service in this database: the Odoo 20 `ai` module is not set up")
            created = False
            if not c.execute("select 1 from iap_account where service_id = %s", (service[0],)).fetchone():
                # Without one, Odoo mints an account with a random token at the first request, which the
                # stand-in would refuse. A company-less account serves every company.
                t = synthetic_token()
                c.execute("insert into iap_account (service_id, account_token, create_date, write_date) "
                          "values (%s, %s, now(), now())", (service[0], t))
                tokens.add(t)
                created = True
        with psycopg.connect(self.env.observer_dsn) as c:
            left = c.execute("select count(*) from iap_account where account_token is null "
                             "or not (account_token = any(%s))", (sorted(tokens),)).fetchone()[0]
        if left:
            raise RuntimeError(f"{left} IAP account row(s) on the copy do not carry this run's synthetic token; "
                               "refusing to start Odoo or the stand-in")
        self.iap_tokens = tokens
        self.extras["iap"] = {"accounts": total, "replaced_non_synthetic": real, "created_odoo_ai_account": created}
        self.notes.append(f"credentials: {total} IAP account token(s) on the run's copy replaced with synthetic run tokens "
                          f"({real} did not already look synthetic)" + ("; an odoo_ai account was created" if created else "")
                          + ". Values are not recorded.")

    def _point_clone(self, instance_base: str, standin_url: str) -> None:
        """`ai.endpoint` -> the stand-in, `web.base.url` -> this instance (frozen), on the disposable copy only,
        verified by reading them back before Odoo starts."""
        self._require_run_clone("set ai.endpoint")
        if not standin_url.startswith("http://127.0.0.1:"):
            raise RuntimeError("ai.endpoint may only point at the local stand-in")
        want = {"ai.endpoint": standin_url, "web.base.url": instance_base, "web.base.url.freeze": "True"}
        with psycopg.connect(self.env.dsn, autocommit=True) as c:
            had = c.execute("select 1 from ir_config_parameter where key = 'ai.endpoint'").fetchone()
            for k, v in want.items():
                c.execute("insert into ir_config_parameter (key, value, create_date, write_date, create_uid, write_uid) "
                          "values (%s, %s, now(), now(), 1, 1) on conflict (key) do update set value = excluded.value", (k, v))
        with psycopg.connect(self.env.observer_dsn) as c:
            got = dict(c.execute("select key, value from ir_config_parameter where key = any(%s)", (list(want),)).fetchall())
        if got != want:
            raise RuntimeError("ai.endpoint / web.base.url did not take effect on the run's copy; refusing to start Odoo")
        if had:
            self.notes.append("the fixture carried an ai.endpoint; replaced on the run's copy (its value is not recorded)")
        self.extras.update({"ai_endpoint": standin_url, "web_base_url": instance_base})

    def _start(self, conf: str) -> None:
        root = self.profile.target["odoo_root"]
        py = self.profile.target.get("python", "python3")
        argv, needs_path = launcher(root)
        env = scrubbed_environment()
        env["ODOO_RC"] = conf
        if needs_path:
            env["PYTHONPATH"] = root
        # deliberately NO provider key: on Odoo 20 the instance never calls a provider; the stand-in holds the key
        self.proc = subprocess.Popen([py, *argv, "-c", conf, "-d", self.env.name], cwd=root, env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(360):
            if _port_open(self.port) and os.path.exists(self.log):
                with open(self.log, errors="replace") as fh:
                    if "Registry loaded in" in fh.read():
                        return
            if self.proc.poll() is not None:
                raise RuntimeError(f"odoo exited during start (code {self.proc.returncode}); see the run's odoo.log")
            time.sleep(0.5)
        raise RuntimeError("odoo did not become ready in 180 s")

    def _open_session_20(self) -> None:
        login, pw = self.agent_cfg.get("operator_login"), self.agent_cfg.get("operator_password")
        if not login:
            raise ProfileError("the native_ai_20 driver needs agent.operator_login / operator_password in the run profile")
        info = self.client.rpc("/web/session/authenticate", {"db": self.env.name, "login": login, "password": pw})
        self.uid = info["uid"]
        with psycopg.connect(self.env.observer_dsn) as c:
            self.company_id = c.execute("select company_id from res_users where id = %s", (self.uid,)).fetchone()[0]
        interface = self.agent_cfg.get("interface_key", "systray_ai_button")
        res = self.client.call("ai.agent", "action_launch_ai_chat", [interface])
        self.channel_id = res["ai_channel_id"]
        with psycopg.connect(self.env.observer_dsn) as c:
            row = c.execute("select id, agent_id from ai_session where channel_id = %s and parent_session_id is null "
                            "order by id desc limit 1", (self.channel_id,)).fetchone()
        if not row:
            raise RuntimeError("no ai.session was created for the chat channel")
        self.session_id = row[0]
        if row[1] != self.agent["id"]:
            raise ProfileError(f"interface {interface!r} opened agent id {row[1]}; the profile's agent is id {self.agent['id']}")
        self.client.call("res.users", "read", [[self.uid], ["name"]])   # warm-up, before the measurement boundary

    def capabilities(self) -> Capabilities:
        return self.caps

    def session_context(self) -> dict[str, Any]:
        ctx = super().session_context()
        ctx.update({"ai_session_id": self.session_id, "llm_model": self.profile.model or None,
                    "price_as": self.profile.provider.get("price_as") or self.profile.model or None,
                    "standin_mode": self.mode,
                    "model_selected_by": "run profile, requested by the local stand-in (not Odoo's hosted service)"})
        return ctx

    # ------------------------------------------------------------ execution
    # The conversation's session and every sub-agent session under it: Odoo 20 delegates by creating a child session
    # (`parent_session_id`), up to three levels deep, whose tool calls are recorded as the CHILD's events. Reading the
    # root alone missed them — and a forbidden call made by a sub-agent passed `allowed_tool_calls`.
    _TREE = ("with recursive tree(id, depth) as (select id, 0 from ai_session where id = %(root)s "
             "union all select s.id, t.depth + 1 from ai_session s join tree t on s.parent_session_id = t.id) ")

    def _session_state(self) -> dict:
        with psycopg.connect(self.env.observer_dsn, row_factory=dict_row) as c:
            root = c.execute("select loop_state, resume_token, pending_tool_call from ai_session where id = %s",
                             (self.session_id,)).fetchone()
            waiting = c.execute("select count(*) as n from ai_session where channel_id = %s and loop_state = 'waiting_model'",
                                (self.channel_id,)).fetchone()["n"]
            ev = c.execute(self._TREE + "select coalesce(max(e.id), 0) as m from ai_session_event e join tree t "
                           "on t.id = e.ai_session_id", {"root": self.session_id}).fetchone()["m"]
        return {**root, "any_waiting_model": waiting, "max_event_id": ev}

    def _open_call_ids(self) -> set[str]:
        """Tool calls the root session has made that have no result yet (the user answers the root session only)."""
        calls, done = set(), set()
        for ev in self._events():
            md = ev["metadata"] or {}
            for p in md.get("content") or []:
                if md.get("role") == "assistant" and p.get("type") == "tool_call":
                    calls.add(str(p.get("call_id")))
                elif p.get("type") == "tool_result":
                    done.add(str(p.get("tool_call_id")))
        return calls - done

    def _events(self) -> list[dict]:
        """The root session's events: what the user's conversation itself did (the interaction protocol reads these)."""
        with psycopg.connect(self.env.observer_dsn, row_factory=dict_row) as c:
            return c.execute("select e.id, e.metadata from ai_session_event e join ai_session s on s.id = e.ai_session_id "
                             "where s.channel_id = %s and s.parent_session_id is null order by e.id",
                             (self.channel_id,)).fetchall()

    def _tree_events(self) -> list[dict]:
        """Every event of the root session and of every sub-agent session under it, with the session and its depth."""
        with psycopg.connect(self.env.observer_dsn, row_factory=dict_row) as c:
            return c.execute(self._TREE + "select e.id, e.metadata, e.ai_session_id as session_id, t.depth "
                             "from ai_session_event e join tree t on t.id = e.ai_session_id order by e.id",
                             {"root": self.session_id}).fetchall()

    def _subsessions(self) -> list[dict]:
        with psycopg.connect(self.env.observer_dsn, row_factory=dict_row) as c:
            return c.execute(self._TREE + "select s.id, s.parent_session_id, t.depth, s.loop_state, "
                             "(select p.name from ai_agent a join res_partner p on p.id = a.partner_id where a.id = s.agent_id) "
                             "as agent from ai_session s join tree t on t.id = s.id where t.depth > 0 order by s.id",
                             {"root": self.session_id}).fetchall()

    def _settle(self) -> tuple[str | None, str | None]:
        """Poll until the root session settles and the stand-in is idle. Returns (loop_state, timeout_error)."""
        end = time.monotonic() + self.turn_timeout_s
        st = self._session_state()
        while time.monotonic() < end:
            st = self._session_state()
            if st["loop_state"] in SETTLED and not st["any_waiting_model"] and self.standin.inflight == 0:
                time.sleep(0.5)   # let a just-committed callback's follow-up round land, then confirm
                st2 = self._session_state()
                if st2["loop_state"] in SETTLED and not st2["any_waiting_model"] and self.standin.inflight == 0:
                    return st2["loop_state"], None
            time.sleep(0.4)
        return st["loop_state"], f"harness timeout: loop_state still {st['loop_state']} after {self.turn_timeout_s:.0f} s"

    def _agent_message_ids_since(self, mid: int) -> list[int]:
        with psycopg.connect(self.env.observer_dsn) as c:
            return [r[0] for r in c.execute("select id from mail_message where model = 'discuss.channel' and res_id = %s "
                                            "and author_id = %s and id > %s", (self.channel_id, self.agent["partner_id"], mid))]

    def _run_turn(self, label: str, call: Callable[[], Any], mid: int, typed: bool) -> Turn:
        before = self._session_state()
        self.turn_markers.append(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        self.turn_event_marks.append(before["max_event_id"])
        # calls pending on the user when a TYPED message arrives are aborted by Odoo, not failed (see collect)
        self.pending_at_turn.append(self._open_call_ids() if typed and before["loop_state"] in WAITING_ON_USER else set())
        self.typed_turn.append(typed)
        requests_before = self.standin.completion_requests
        refusals_before = len(self.standin.violations)
        records_before = len(self.standin.records)
        t0 = time.perf_counter()
        status, err = RequestStatus.RETURNED, None
        user_returned: bool | None = True
        try:
            call()
        except httpx.TimeoutException as e:
            status, err, user_returned = RequestStatus.TIMEOUT, f"harness timeout: {type(e).__name__}", None
        except (RpcError, httpx.HTTPError) as e:
            status, err, user_returned = RequestStatus.ERROR, str(e)[:3000], False
        loop_state = None
        if status == RequestStatus.RETURNED:
            loop_state, timeout_err = self._settle()
            if timeout_err:
                status, err = RequestStatus.TIMEOUT, timeout_err
            elif loop_state == "ready" and not self._agent_message_ids_since(mid):
                # the session went back to ready and posted nothing: say why, from what the stand-in saw
                seen = [r for r in self.standin.records_since(records_before) if r.get("event") == "completion_request"]
                if len(self.standin.violations) > refusals_before:
                    status, err = RequestStatus.ERROR, f"the stand-in refused the round: {self.standin.violations[-1]}"
                elif self.standin.completion_requests == requests_before:
                    status, err = RequestStatus.ERROR, ("the instance's completion request never reached the stand-in; "
                                                        "see the run's odoo.log")
                elif any(r.get("ack_late") for r in seen):
                    status, err = RequestStatus.ERROR, ("the stand-in acknowledged a round after the instance had stopped "
                                                        "waiting for it, so the instance gave the round up")
        st = self._session_state()
        final_state = loop_state or st["loop_state"]
        pending = PENDING_BY_LOOP_STATE.get(final_state, "none")
        # What the user was shown, from the substrate: when the stand-in answered this turn's LAST round with a
        # failure and the instance then posted a message, that message is the instance's failure notice.
        callbacks = [r for r in self.standin.records_since(records_before) if r.get("event") == "callback"]
        substrate_report = basis = None
        if (status == RequestStatus.RETURNED and callbacks and callbacks[-1].get("llm_result_is_false")
                and self._agent_message_ids_since(mid)):
            substrate_report = "failure"
            basis = ("the stand-in answered the turn's last round with a failure (llm_result false), and the instance "
                     "then posted its failure notice")
        self.extras["turns"].append({"label": label[:80], "loop_state_after": final_state,
                                     "pending_tool_call": bool(st["pending_tool_call"]), "status": status.value,
                                     "user_request_returned": user_returned, "pending_interaction": pending,
                                     "wall_s": round(time.perf_counter() - t0, 3)})
        return Turn(label, status, None, err, round(time.perf_counter() - t0, 3), mid,
                    pending_interaction=pending, user_request_returned=user_returned,
                    substrate_report=substrate_report, substrate_report_basis=basis)

    def _post_message(self, text: str) -> int:
        self.client.rpc("/mail/message/post", {"thread_model": "discuss.channel", "thread_id": self.channel_id,
                                               "post_data": {"body": text, "message_type": "comment",
                                                             "subtype_xmlid": "mail.mt_comment"}})
        with psycopg.connect(self.env.observer_dsn) as c:
            return c.execute("select max(id) from mail_message where model = 'discuss.channel' and res_id = %s",
                             (self.channel_id,)).fetchone()[0]

    def _typed_turn(self, text: str) -> None:
        mid = self._post_message(text)
        self.turns.append(self._run_turn(text, lambda: self.client.rpc(
            "/ai/start_session_advance", {"channel_id": self.channel_id, "mail_message_id": mid}), mid, typed=True))

    def _confirm(self, value: str) -> None:
        st = self._session_state()
        with psycopg.connect(self.env.observer_dsn) as c:
            last_mid = c.execute("select max(id) from mail_message where model = 'discuss.channel' and res_id = %s",
                                 (self.channel_id,)).fetchone()[0]
        self.turns.append(self._run_turn(f"[confirmation: {value}]", lambda: self.client.rpc(
            "/ai/resume_pending_interaction", {"channel_id": self.channel_id, "session_id": self.session_id,
                                               "resume_token": st["resume_token"],
                                               "response": {"kind": "confirmation", "value": value}}),
            last_mid, typed=False))

    def execute(self, first_turn: str, continuation: Continuation | str | None, should_continue: Callable[[], bool]) -> None:
        self.start_marker = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self._typed_turn(first_turn)
        if not continuation or self.turns[-1].request_status != RequestStatus.RETURNED or not should_continue():
            return
        if isinstance(continuation, str):   # a plain follow-up message
            self._typed_turn(continuation)
            return
        if not continuation.structured:
            self._typed_turn(continuation.text)
            return
        self._protocol(continuation.on_question, continuation.on_confirmation)

    def _protocol(self, question_reply: str | None, confirmation: str | None) -> None:
        """Odoo 20's own interaction protocol: a text reply ONLY when the session is recorded as waiting for an
        answer, then a structured confirmation ONLY when Odoo requests one. Whatever was not sent is recorded."""
        self.extras["protocol"] = {"text_sent": False, "confirmation_sent": False}
        state = self._session_state()["loop_state"]
        if question_reply and state == "waiting_answer":
            self._typed_turn(question_reply)
            self.extras["protocol"]["text_sent"] = True
            if self.turns[-1].request_status != RequestStatus.RETURNED:
                return
            state = self._session_state()["loop_state"]
        if confirmation and state == "waiting_confirmation":
            self._confirm(confirmation)
            self.extras["protocol"]["confirmation_sent"] = confirmation
            return
        self.notes.append(f"interaction protocol stopped at loop_state={state}: a text reply is sent only while Odoo waits "
                          "for an answer, a confirmation only while it waits for one")

    # ------------------------------------------------------------ collection
    def collect(self) -> DriverRunResult:
        with psycopg.connect(self.env.observer_dsn, row_factory=dict_row) as c:
            msgs = c.execute("select id, author_id, body from mail_message where model = 'discuss.channel' and res_id = %s "
                             "order by id", (self.channel_id,)).fetchall()
            sess = c.execute("select id, loop_state, auto_confirm from ai_session where id = %s", (self.session_id,)).fetchone()
        events = self._tree_events()
        subs = self._subsessions()
        bounds = [t.message_id for t in self.turns]
        for i, t in enumerate(self.turns):
            lo, hi = bounds[i] or 0, (bounds[i + 1] if i + 1 < len(bounds) else None)
            parts = [m["body"] for m in msgs if m["author_id"] == self.agent["partner_id"] and m["id"] > lo
                     and (hi is None or m["id"] < hi)]
            t.assistant_response = "\n".join(parts) if parts else None
        trace = self._trace(events)
        if subs:
            delegated = [c for c in trace if c.session_depth > 0]
            self.extras["subagent_sessions"] = [
                {"session_id": s["id"], "parent_session_id": s["parent_session_id"], "depth": s["depth"],
                 "agent": s["agent"], "loop_state": s["loop_state"],
                 "tool_calls": sum(1 for e in events if e.get("session_id") == s["id"]
                                   for p in (e["metadata"] or {}).get("content") or []
                                   if (e["metadata"] or {}).get("role") == "assistant" and p.get("type") == "tool_call")}
                for s in subs]
            self.notes.append(f"delegation: {len(subs)} sub-agent session(s) "
                              f"({', '.join(sorted({str(s['agent']) for s in subs}))}); {len(delegated)} of the "
                              f"{len(trace)} tool call(s) were made in them and are part of the trace")
            unfinished = [s for s in subs if s["loop_state"] not in SETTLED]
            if unfinished:
                self.notes.append(f"{len(unfinished)} sub-agent session(s) had not finished when the run was stopped "
                                  "(their calls up to that point are in the trace)")
        usage, provider_calls, basis = self._usage()
        u = self.standin.usage_summary()
        served = u["served_models"]
        model = self.profile.model or None
        no_provider = self.mode in NO_PROVIDER_MODES
        self.extras.update({"session_final": dict(sess) if sess else None, "standin": {**u, "violations": list(self.standin.violations)}})
        self._write_extras()
        self.notes.append(f"stand-in (mode {self.mode}): {u['completion_requests']} completion request(s), "
                          f"{u['provider_attempts']} provider request(s), served model(s) {served or 'none observed'}")
        if self.standin.violations:
            self.notes.append(f"!! the stand-in REFUSED {len(self.standin.violations)} request(s): "
                              + "; ".join(sorted(set(self.standin.violations))))
        self.notes += [f"turn {i + 1}: {t['label'][:40]!r} -> loop_state {t['loop_state_after']}"
                       for i, t in enumerate(self.extras["turns"])]
        for a in self.extras.get("aborted_interactions", []):
            self.notes.append(f"ORCHESTRATION: pending {a['tool']} aborted by a typed reply (Odoo records it as declined); "
                              "not counted as a tool error")
        self.notes += self._log_warnings()[:5]
        return DriverRunResult(
            session_id=f"ai.session:{self.session_id}", turns=list(self.turns),
            model_metadata=ModelMetadata(
                f"{self.profile.provider_name} via the local stand-in", model or "none", True,
                served[0] if len(served) == 1 else (", ".join(served) or None),
                "n/a (no model: no provider request)" if no_provider else ("pinned" if model and served == [model] else "limited"),
                configured_identifier=model,
                selected_by="run profile, requested by the local stand-in (not Odoo's hosted service)"),
            tool_trace=trace, token_usage=usage, provider_cost_usd=None, driver_notes=list(self.notes),
            known_response_rules=[], provider_calls=provider_calls, usage_basis=basis)

    def _trace(self, events: list[dict]) -> list[ToolCall]:
        """Tool calls from assistant events of the root session and of every sub-agent session under it, in the
        order Odoo recorded them. A call is identified by (session, call id): call ids are only unique within a
        session. The turn is the user turn during which the event was recorded, whichever session recorded it. A
        failed result marks its call as errored, except a root call that was pending on the user when a typed message
        arrived — Odoo aborts that call and records it as declined."""
        calls: dict[tuple, ToolCall] = {}
        order: list[tuple] = []
        marks = self.turn_event_marks
        for ev in events:
            md = ev["metadata"] or {}
            sid, depth = ev.get("session_id", self.session_id), ev.get("depth", 0)
            turn = max([i for i, m in enumerate(marks) if ev["id"] > m], default=None)
            for p in md.get("content") or []:
                if md.get("role") == "assistant" and p.get("type") == "tool_call":
                    key = (sid, str(p.get("call_id")))
                    args = p.get("args") or {}
                    calls[key] = ToolCall(p.get("name"), args if isinstance(args, dict) else None,
                                          json.dumps(args, ensure_ascii=False, default=str)[:4000], None, turn,
                                          session_depth=depth)
                    order.append(key)
                elif p.get("type") == "tool_result" and (sid, str(p.get("tool_call_id"))) in calls \
                        and p.get("success") is False:
                    key = (sid, str(p.get("tool_call_id")))
                    if (depth == 0 and turn is not None and turn < len(self.pending_at_turn)
                            and key[1] in self.pending_at_turn[turn]):
                        self.extras.setdefault("aborted_interactions", []).append({"tool": calls[key].name, "turn": turn})
                        continue
                    calls[key].error = parts_text(p.get("result"))[:1000] or "tool failed"
        return [calls[k] for k in order]

    def _usage(self) -> tuple[TokenUsage | None, int | None, str]:
        u = self.standin.usage_summary()
        if self.mode in NO_PROVIDER_MODES:
            if u["provider_attempts"]:
                return None, None, "stand-in accounting is inconsistent: provider attempts in a no-provider mode"
            return TokenUsage(), 0, f"stand-in mode {self.mode} holds no key and counted no provider request"
        t = u["tokens"]
        usage = TokenUsage(t["input"], t["output"], t["cached_input"], u["agent_calls_with_usage"])
        if not u["complete"]:
            why = []
            if u["provider_failures"]:
                why.append(f"{u['provider_failures']} provider request(s) ended without a usage-bearing response")
            if u["inflight"]:
                why.append(f"{u['inflight']} request(s) still in flight when the run was collected")
            if self.standin.log_errors:
                why.append("the stand-in's log could not be written")
            usage.partial = "; ".join(why) or "stand-in accounting incomplete"
        basis = (f"provider-reported usage on {u['agent_calls_with_usage'] + u['other_calls_with_usage']} response(s) "
                 f"observed by the stand-in ({u['other_calls_with_usage']} outside the agent loop, e.g. conversation naming)")
        return usage, u["provider_attempts"], basis

    def usage_after_failure(self) -> tuple[TokenUsage | None, int | None, str]:
        if self.standin is None:
            if self.mode == "provider":
                return TokenUsage(), 0, "the run failed before the stand-in (the only key holder) started"
            return TokenUsage(), 0, f"stand-in mode {self.mode} makes no provider request; the run failed before it started"
        usage, calls, basis = self._usage()
        if usage is not None and calls:
            usage.partial = usage.partial or "the run did not complete: usage recorded by the stand-in before the failure"
        return usage, calls, basis + " (run did not complete)"

    def _log_warnings(self) -> list[str]:
        try:
            with open(self.log, errors="replace") as fh:
                return [ln.strip()[:300] for ln in fh if ln[:19] >= self.start_marker
                        and (" WARNING " in ln or " ERROR " in ln) and "odoo.tools.config" not in ln]
        except OSError:
            return []

    def _write_extras(self) -> None:
        try:
            with open(os.path.join(self.run_dir, "driver20.json"), "w") as fh:
                json.dump(self.extras, fh, indent=1, default=str)
        except OSError as e:
            self.notes.append(f"driver20.json could not be written ({type(e).__name__})")

    def quiesce(self) -> list[str]:
        """Before the final evidence: the stand-in first — from then on no round is answered and no reply reaches the
        instance; replies already being delivered are waited for — then the Odoo process, then the stand-in's workers
        again (a delivery still under way fails once the instance is gone). A reply not delivered is recorded as
        suppressed in the stand-in's log. Previously a timed-out turn went straight to the snapshot while a
        reply could still arrive and write."""
        notes: list[str] = []
        if self.standin is not None:
            self.standin.stop()
        self._stop_odoo()
        if self.standin is not None:
            alive = self.standin.stop()
            if self.standin.callbacks_suppressed:
                notes.append(f"{self.standin.callbacks_suppressed} stand-in reply(ies) not delivered: the run was stopped "
                             "first, and the final evidence is the database as it stood then")
            if alive:
                notes.append(f"{alive} stand-in completion worker(s) still running: their replies are never delivered, "
                             "but a provider request already sent may still be billed")
        return notes

    def close(self) -> None:
        """Stop the stand-in (no reply reaches the instance from here), close the Odoo side (HTTP client, the Odoo
        process, the run's data directory), then wait for the stand-in's workers again. Every step is attempted
        whatever the others did; all failures are raised together afterwards. A worker still alive at the deadline is a
        failure, not a note: its reply is never delivered, but work is still running, and a provider
        request it already sent may still be billed."""
        failures: list[tuple[str, BaseException]] = []
        if self.standin is not None:
            try:
                self.standin.stop()
            except Exception as e:  # noqa: BLE001 — recorded; the steps below must still run
                failures.append(("stop the stand-in", e))
        try:
            super().close()
        except DriverCloseError as e:
            failures += e.failures
        except Exception as e:  # noqa: BLE001
            failures.append(("close the Odoo side", e))
        if self.standin is not None:
            try:
                alive = self.standin.stop()
                if alive:
                    raise StandInWorkersAlive(
                        f"{alive} stand-in completion worker(s) still running {self.standin.stop_deadline_s:.0f} s after "
                        "the stop; their replies are never delivered, but a provider request already sent may be billed")
            except Exception as e:  # noqa: BLE001
                failures.append(("stop the stand-in's workers", e))
        if failures:
            raise DriverCloseError(failures) from failures[0][1]
