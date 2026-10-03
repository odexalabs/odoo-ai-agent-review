"""Native Odoo AI driver (Odoo 19 tree). One Odoo process per run on the disposable database,
one fresh Discuss conversation per run, the same two HTTP calls the web client makes:
`/mail/message/post` then `/ai/generate_response`. Trajectory comes from the run's own log
(INFO), tokens from the `[AI Summary]` line; neither is persisted by Odoo.

What earlier runs proved and this keeps: `workers = 0`, `max_cron_threads = 0`, `dbfilter` pinned to
the run database, `list_db = False`, own `data_dir`, the provider key handed to this process
only under the variable Odoo reads, every ambient key scrubbed."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx
import psycopg
from psycopg.rows import dict_row

from ...core.contracts import (
    DriverRunResult,
    EnvironmentHandle,
    ModelMetadata,
    RequestStatus,
    ResponseRule,
    Turn,
)
from ...core.credentials import Credential, scrubbed_environment
from ...core.profile import ProfileError, RunProfile
from ..base import Capabilities, Driver
from . import capabilities as caps_mod
from . import logparse

PROVIDER_ENV = {"openai": "ODOO_AI_CHATGPT_TOKEN", "google": "ODOO_AI_GEMINI_TOKEN"}
# Odoo 19 reads a stored key from ir.config_parameter BEFORE the environment variable (the `ai`
# module's provider service, `ai/utils/llm_api_service.py`, looks the parameter up first). A fixture that
# carries one would bill to an unnamed key, not the one the run names. Removed on the
# disposable clone so the named credential is the only source.
STORED_PROVIDER_KEYS = ("ai.openai_key", "ai.google_key")
# `smtp_server = False` in a config file is SKIPPED by Odoo 19 ("isn't a boolean option, skip",
# tools/config.py) and the default stays `localhost`. An unroutable name with an unroutable port is
# what actually disables the fallback: smtplib fails before any byte leaves the host.
SMTP_FALLBACK_HOST, SMTP_FALLBACK_PORT = "smtp.invalid", 1
TIMEOUT_MARKERS = ("ReadTimeout", "timed out", "timeout", "ConnectTimeout")

# Messages the substrate shows its users are NOT shipped here: Odoo's own wording is version- and language-specific
# and is not reproduced in this tool. The operator states the exact messages their deployment shows, per verdict, in
# the run profile's `report_rules` (a scenario's `report:` block works the same way). A failed request that posted
# nothing is still classified from the substrate (outcomes.classify_report), and a failed Odoo 20 round from the
# stand-in's own record.
KNOWN_RESPONSES: list[ResponseRule] = []


def _free_port(start: int = 8190, end: int = 8290) -> int:
    for p in range(start, end):
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port in range")


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


class RpcError(RuntimeError):
    pass


class DriverCloseError(RuntimeError):
    """One or more close steps failed; all were attempted. `steps` names them (harness text, safe to print);
    the message carries each failure's own text, which belongs in the private record only."""

    def __init__(self, failures: list[tuple[str, BaseException]]):
        self.failures = failures
        self.steps = [step for step, _ in failures]
        super().__init__("; ".join(f"{step}: {type(e).__name__}: {e}" for step, e in failures))


class Client:
    def __init__(self, port: int, db: str, timeout: float):
        self.base = f"http://127.0.0.1:{port}"
        self.db = db
        self.http = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))

    def rpc(self, path: str, params: dict) -> Any:
        r = self.http.post(self.base + path, json={"jsonrpc": "2.0", "method": "call", "params": params, "id": 1})
        r.raise_for_status()
        j = r.json()
        if "error" in j:
            raise RpcError(json.dumps(j["error"].get("data", j["error"]))[:3000])
        return j.get("result")

    def call(self, model: str, method: str, args: list, kwargs: dict | None = None) -> Any:
        return self.rpc("/web/dataset/call_kw", {"model": model, "method": method, "args": args, "kwargs": kwargs or {}})

    def close(self) -> None:
        self.http.close()


class NativeAiDriver(Driver):
    name = "native_ai"

    def __init__(self) -> None:
        self.env: EnvironmentHandle | None = None
        self.profile: RunProfile | None = None
        self.agent_cfg: dict[str, Any] = {}
        self.credential: Credential | None = None
        self.run_dir = ""
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self.log = ""
        self.client: Client | None = None
        self.agent: dict | None = None
        self.channel_id: int | None = None
        self.session_kind = "internal"
        self.uid: int | None = None
        self.company_id: int | None = None
        self.turns: list[Turn] = []
        self.turn_markers: list[str] = []
        self.start_marker = ""
        self.caps: Capabilities | None = None
        self.notes: list[str] = []
        self.model_before: str | None = None
        self.data_dir: str | None = None     # set only once this driver has created it: close() removes nothing else

    # ---- lifecycle
    def prepare(self, env: EnvironmentHandle, profile: RunProfile, agent_role: str, credential: Credential | None,
                run_dir: str, session_kind: str | None) -> None:
        self.env, self.profile, self.credential, self.run_dir = env, profile, credential, run_dir
        self.agent_cfg = profile.agent(agent_role)
        self.session_kind = session_kind or self.agent_cfg.get("session", "internal")
        os.makedirs(run_dir, exist_ok=True)
        self.caps = caps_mod.discover(env.observer_dsn, profile.target.get("odoo_root"))
        self._bind_agent_and_model()
        self._remove_stored_provider_keys()
        self.port = int(self.agent_cfg.get("port") or _free_port())
        conf = self._write_conf()
        self._start(conf)
        self.client = Client(self.port, env.name, float(profile.provider.get("timeout_s", 900)))
        self._open_session()

    def _bind_agent_and_model(self) -> None:
        assert self.env and self.profile
        with psycopg.connect(self.env.observer_dsn, row_factory=dict_row) as c:
            name = self.agent_cfg.get("name")
            xml_id = self.agent_cfg.get("xml_id")
            if xml_id:
                mod, _, nm = xml_id.partition(".")
                row = c.execute("select res_id from ir_model_data where model = 'ai.agent' and module = %s and name = %s", (mod, nm)).fetchone()
                if not row:
                    raise ProfileError(f"agent xml_id {xml_id!r} not found in the fixture")
                agent = c.execute("select a.id, a.partner_id, p.name, a.llm_model, a.response_style from ai_agent a "
                                  "join res_partner p on p.id = a.partner_id where a.id = %s", (row["res_id"],)).fetchone()
            else:
                agent = c.execute("select a.id, a.partner_id, p.name, a.llm_model, a.response_style from ai_agent a "
                                  "join res_partner p on p.id = a.partner_id where p.name = %s", (name,)).fetchone()
            if not agent:
                raise ProfileError(f"agent {name or xml_id!r} not found in the fixture")
            self.agent = dict(agent)
        # model identity lives in the run profile: pin the clone's agent to it (the clone is disposable)
        model = self.profile.model
        if model:
            self.model_before = self.agent["llm_model"]
            with psycopg.connect(self.env.dsn, autocommit=True) as c:
                c.execute("update ai_agent set llm_model = %s where id = %s", (model, self.agent["id"]))
            if self.model_before != model:
                self.notes.append(f"agent llm_model set from {self.model_before!r} (fixture) to {model!r} (profile) on the clone")
            self.agent["llm_model"] = model

    def _remove_stored_provider_keys(self) -> None:
        assert self.env
        with psycopg.connect(self.env.dsn, autocommit=True) as c:
            rows = c.execute("select key from ir_config_parameter where key = any(%s)", (list(STORED_PROVIDER_KEYS),)).fetchall()
            if rows:
                c.execute("delete from ir_config_parameter where key = any(%s)", (list(STORED_PROVIDER_KEYS),))
                self.notes.append(f"stored provider key(s) {[r[0] for r in rows]} found in the fixture and removed on the clone; "
                                  "the named credential is the only source")

    def smtp_fallback_disabled(self) -> bool:
        return True   # _write_conf points the config fallback at an unroutable host:port

    def _write_conf(self) -> str:
        assert self.env and self.profile
        root = self.profile.target["odoo_root"]
        addons = self.profile.target.get("addons_path") or f"{root}/odoo/addons"
        data_dir = os.path.join(self.run_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.log = os.path.join(self.run_dir, "odoo.log")
        path = os.path.join(self.run_dir, "odoo.conf")
        fx = self.profile.fixture
        db_lines = "".join(f"{k} = {fx[src]}\n" for k, src in (("db_host", "host"), ("db_port", "port"), ("db_password", "execution_password"))
                           if fx.get(src) not in (None, ""))
        with open(path, "w") as fh:
            fh.write(f"""[options]
addons_path = {addons}
data_dir = {data_dir}
db_user = {self.env.execution_role}
{db_lines}db_name = {self.env.name}
dbfilter = ^{self.env.name}$
list_db = False
max_cron_threads = 0
workers = 0
http_interface = 127.0.0.1
http_port = {self.port}
smtp_server = {SMTP_FALLBACK_HOST}
smtp_port = {SMTP_FALLBACK_PORT}
logfile = {self.log}
log_level = info
log_handler = odoo.addons.ai:DEBUG
""")
        return path

    def _start(self, conf: str) -> None:
        assert self.env and self.profile
        root = self.profile.target["odoo_root"]
        py = self.profile.target.get("python", "python3")
        env = scrubbed_environment()
        # Odoo 19 parses its config once at import, with no arguments, reading ODOO_RC or else ~/.odoorc
        # (the config object's constructor, odoo/tools/config.py:187), before -c is applied. Naming the
        # run's own file as ODOO_RC makes both passes read it, so no ambient rc file is ever opened.
        env["ODOO_RC"] = conf
        if self.credential:
            var = PROVIDER_ENV.get(self.profile.provider_name)
            if not var:
                raise ProfileError(f"unknown provider {self.profile.provider_name!r}; known: {sorted(PROVIDER_ENV)}")
            env[var] = self.credential.value
        from ..odoo_target import launcher
        argv, needs_path = launcher(root)       # odoo-bin in a source checkout; setup/odoo in a packaged tree
        if needs_path:
            env["PYTHONPATH"] = root
        self.proc = subprocess.Popen([py, *argv, "server", "-c", conf, "-d", self.env.name], cwd=root, env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(360):
            # the HTTP port opens ~1 s BEFORE the registry is loaded; wait for the log line, not the socket
            if _port_open(self.port) and os.path.exists(self.log):
                with open(self.log, errors="replace") as fh:
                    if "Registry loaded in" in fh.read():
                        return
            if self.proc.poll() is not None:
                raise RuntimeError(f"odoo exited during start (code {self.proc.returncode}); see {self.log}")
            time.sleep(0.5)
        raise RuntimeError("odoo did not become ready in 180 s")

    def _open_session(self) -> None:
        assert self.client and self.agent and self.env
        if self.session_kind == "public":
            with psycopg.connect(self.env.observer_dsn, row_factory=dict_row) as c:
                lc = self.agent_cfg.get("livechat_channel_id") or c.execute("select id from im_livechat_channel order by id limit 1").fetchone()["id"]
            sess = self.client.rpc("/im_livechat/get_session", {"channel_id": lc, "ai_agent_id": self.agent["id"], "persisted": True})
            data = sess.get("store_data", sess) if isinstance(sess, dict) else {}
            chans = data.get("discuss.channel") or []
            if not chans:
                raise RuntimeError(f"no discuss.channel in get_session result: {str(sess)[:400]}")
            self.channel_id = chans[0]["id"]
            self.notes.append(f"public livechat session on channel {lc}; guest cookie set: {'dgid' in self.client.http.cookies}")
        else:
            login, pw = self.agent_cfg.get("operator_login"), self.agent_cfg.get("operator_password")
            if not login:
                raise ProfileError("internal session needs agent.operator_login / operator_password in the run profile")
            info = self.client.rpc("/web/session/authenticate", {"db": self.env.name, "login": login, "password": pw})
            self.uid = info["uid"]
            with psycopg.connect(self.env.observer_dsn) as c:
                self.company_id = c.execute("select company_id from res_users where id = %s", (self.uid,)).fetchone()[0]
            act = self.client.call("ai.agent", "open_agent_chat", [[self.agent["id"]]])
            self.channel_id = act["params"]["channelId"]
            self.client.call("res.users", "read", [[self.uid], ["name"]])   # warm-up read, before the boundary

    def capabilities(self) -> Capabilities:
        assert self.caps
        return self.caps

    def session_context(self) -> dict[str, Any]:
        return {"session_id": f"discuss.channel:{self.channel_id}", "kind": self.session_kind, "uid": self.uid,
                "company_id": self.company_id, "agent": self.agent["name"] if self.agent else None,
                "agent_id": self.agent["id"] if self.agent else None, "llm_model": self.agent["llm_model"] if self.agent else None,
                "port": self.port}

    # ---- execution
    def _turn(self, text: str) -> Turn:
        assert self.client and self.env
        self.turn_markers.append(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        self.client.rpc("/mail/message/post", {"thread_model": "discuss.channel", "thread_id": self.channel_id,
                                               "post_data": {"body": text, "message_type": "comment", "subtype_xmlid": "mail.mt_comment"}})
        with psycopg.connect(self.env.observer_dsn) as c:
            mid = c.execute("select max(id) from mail_message where model = 'discuss.channel' and res_id = %s", (self.channel_id,)).fetchone()[0]
        t0 = time.perf_counter()
        status, err = RequestStatus.RETURNED, None
        if self.probe:
            self.notes.append("PROBE: message posted; /ai/generate_response NOT called; no provider request was made")
            return Turn(text, status, None, None, round(time.perf_counter() - t0, 3), mid)
        try:
            self.client.rpc("/ai/generate_response", {"mail_message_id": mid, "channel_id": self.channel_id})
        except httpx.TimeoutException as e:
            status, err = RequestStatus.TIMEOUT, f"harness timeout: {e!r}"
        except (RpcError, httpx.HTTPError) as e:
            err = str(e)[:3000]
            status = RequestStatus.TIMEOUT if any(m in err for m in TIMEOUT_MARKERS) else RequestStatus.ERROR
        return Turn(text, status, None, err, round(time.perf_counter() - t0, 3), mid,
                    pending_interaction=None, user_request_returned=err is None)

    def execute(self, first_turn: str, continuation: str | None, should_continue: Callable[[], bool]) -> None:
        self.start_marker = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self.turns.append(self._turn(first_turn))
        if continuation and self.turns[-1].request_status == RequestStatus.RETURNED and should_continue():
            self.turns.append(self._turn(continuation))

    # ---- collection
    def collect(self) -> DriverRunResult:
        assert self.env and self.agent and self.profile
        with psycopg.connect(self.env.observer_dsn, row_factory=dict_row) as c:
            msgs = c.execute("select id, author_id, author_guest_id, body from mail_message where model = 'discuss.channel' "
                             "and res_id = %s order by id", (self.channel_id,)).fetchall()
        # assistant responses: agent-authored messages after each user message, up to the next user message
        bounds = [t.message_id for t in self.turns]
        for i, t in enumerate(self.turns):
            lo = bounds[i] or 0
            hi = bounds[i + 1] if i + 1 < len(bounds) else None
            parts = [m["body"] for m in msgs if m["author_id"] == self.agent["partner_id"] and m["id"] > lo and (hi is None or m["id"] < hi)]
            t.assistant_response = "\n".join(parts) if parts else None
        # every turn requests one AI response, except in probe mode, where /ai/generate_response is never called
        requested = 0 if self.probe else len(self.turns)
        parsed = (logparse.parse(self.log, self.start_marker, self.turn_markers, responses=requested) if self.log
                  else logparse.ParsedLog())
        self.notes += parsed.notes
        if parsed.error_lines:
            self.notes.append(f"{len(parsed.error_lines)} ERROR line(s) in the run log; first: {parsed.error_lines[0][:200]}")
        provider = self.profile.provider_name
        model = self.agent["llm_model"]
        from ...core.contracts import TokenUsage
        usage = parsed.usage_or_none()
        if self.probe:
            # positive evidence: the process held no key and the generation endpoint was never called
            calls, basis = 0, "probe: the Odoo process held no key and /ai/generate_response was never called"
            usage = usage or TokenUsage()
        elif usage is not None and not usage.partial:
            calls, basis = usage.llm_round_trips, "Odoo's own [AI Summary] log lines, one per AI response"
        else:
            calls, basis = None, None
        return DriverRunResult(
            session_id=f"discuss.channel:{self.channel_id}", turns=list(self.turns),
            model_metadata=ModelMetadata(provider, model, False, None, "limited", configured_identifier=self.profile.model or model,
                                         selected_by="run profile, pinned into the agent on the run's copy"),
            # None, not [], when the log could not show the calls: "unobservable", never "no tool was called"
            tool_trace=parsed.tool_calls_or_none(), token_usage=usage, provider_cost_usd=None,
            driver_notes=list(self.notes), known_response_rules=list(KNOWN_RESPONSES),
            provider_calls=calls, usage_basis=basis,
        )

    def close(self) -> None:
        """Close the HTTP client, stop the Odoo process, remove the run's data directory — each attempted
        whatever the others did: a client that fails to close must not leave Odoo running. Every
        failure is kept and raised together, after all three were tried."""
        failures: list[tuple[str, BaseException]] = []
        if self.client:
            try:
                self.client.close()
            except Exception as e:  # noqa: BLE001 — kept and raised below, after the process is stopped
                failures.append(("close the HTTP client", e))
            self.client = None
        if self.proc:
            try:
                self._stop_process()
                self.proc = None
            except Exception as e:  # noqa: BLE001 — the process may still run: kept, named with its pid
                failures.append((f"stop the Odoo process (pid {self.proc.pid})", e))
        # Only the directory this driver created. (Previously the path was rebuilt from `run_dir`, which is
        # "" until prepare() runs — a relative `data` in the working directory — and every error was ignored.)
        if self.data_dir:
            try:
                shutil.rmtree(self.data_dir)
            except FileNotFoundError:
                self.data_dir = None       # already gone: nothing left to remove
            except Exception as e:  # noqa: BLE001 — it holds the run's filestore and sessions: reported, never ignored
                failures.append(("remove the run's data directory", e))
            else:
                self.data_dir = None
        if failures:
            raise DriverCloseError(failures) from failures[0][1]

    def _stop_process(self) -> None:
        proc = self.proc
        try:
            proc.terminate()
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(20)       # bounded: a process that survives SIGKILL is reported, not waited on forever

    # ---- selection, compatibility and failure-path accounting
    def transport_description(self) -> str:
        if self.probe:
            return "direct (probe): the Odoo 19 process holds no key and /ai/generate_response is never called"
        prov = self.profile.provider_name if self.profile else "the provider"
        return f"direct: the Odoo 19 process calls {prov} itself, with the run's named key"

    def check_scenario(self, sc) -> None:
        if self.name == NativeAiDriver.name:      # a subclass driving another version decides for itself
            self.refuse_other_versions(sc, "19")
            self.refuse_structured_continuation(sc)

    def preflight(self, profile: RunProfile, template_dsn: str) -> list[str]:
        """Odoo 19 source tree and an Odoo 19 template with the `ai` module; refuses anything else. A subclass that
        drives another Odoo version supplies its own check (the native_ai_20 driver does)."""
        from ..odoo_target import check_template, check_tree, launcher, template_facts
        if self.name != NativeAiDriver.name:
            return [f"{self.name}: no version preflight (subclass of the Odoo 19 driver without its own check)"]
        if profile.transport not in (None, "direct"):
            raise ProfileError(f"profile {profile.name}: the native_ai (Odoo 19) driver's only transport is `direct`")
        version, build = check_tree(profile, "19", self.name)
        launcher(profile.target["odoo_root"])   # odoo-bin or setup/odoo; refuses a tree with neither
        template = template_dsn.split("dbname=", 1)[-1].split()[0]
        facts = template_facts(template_dsn, ["ai_agent.llm_model", "ai_session.loop_state"], [])
        check_template(facts, "19", self.name, template, ["ai_agent.llm_model"], ["ai_session.loop_state"])
        return [f"target Odoo {version} ({build or 'build unknown'}); template base {facts.base_version}"]

    def usage_after_failure(self):
        """On Odoo 19 the Odoo process is the only holder of the key. Before it starts nothing can be spent; after,
        the run's own log is the only witness: its `[AI Summary]` lines give a lower bound, never the total."""
        from ...core.contracts import TokenUsage
        if self.proc is None:
            return TokenUsage(), 0, "the run failed before the Odoo process (the only key holder) was started"
        if self.probe or self.credential is None:
            return TokenUsage(), 0, "the Odoo process held no key (probe): no provider request could be billed"
        if not self.log or not self.start_marker:
            return None, None, "the run failed before its first turn was recorded"
        parsed = logparse.parse(self.log, self.start_marker, self.turn_markers, responses=len(self.turns) or None)
        usage = parsed.usage_or_none()
        if usage is None:
            return None, None, "the run did not complete and its log shows no usage that can be vouched for"
        usage.partial = usage.partial or "the run did not complete: summed from the [AI Summary] lines its log holds"
        return usage, None, "Odoo's own [AI Summary] log lines (run did not complete)"

    # ---- the final evidence is taken with nothing left that could write
    def quiesce(self) -> list[str]:
        """Stop the Odoo process before the final evidence is taken. A request the client stopped waiting for (a
        timeout) is still being processed by the server and could write after the snapshot; once the process is gone,
        an unfinished transaction is rolled back and nothing else writes. It also completes the log before it is read."""
        self._stop_odoo()
        return []

    def _stop_odoo(self) -> None:
        from ...core.contracts import EvidenceIncomplete
        if self.proc is None:
            return
        try:
            self._stop_process()
        except Exception as e:     # the process may still be running: the evidence cannot be trusted
            raise EvidenceIncomplete(f"the Odoo process (pid {self.proc.pid}) could not be stopped before the final "
                                     f"snapshot ({type(e).__name__})") from e
        self.proc = None
