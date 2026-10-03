"""A per-run local stand-in for Odoo 20's AI endpoint.

One stand-in serves ONE run: it binds 127.0.0.1 on a free port, knows the one instance URL and database it
may call back, and the exact synthetic IAP tokens the driver wrote into that run's disposable database. A
completion request naming another URL, another database, or any other token — a real one included — is
REFUSED: no provider call, no callback, and the refusal recorded by its reason, never by the value.

Modes
  provider        translate the round to OpenAI Chat Completions, call the operator's provider with the one
                  named key, translate the answer back, call the instance back
  canned          answer every round with a fixed text; NO provider request is ever made (a plumbing run)
  bad_signature   canned, with a corrupted signature: the instance must reject the callback   (fault tests)
  late_ack        canned, acknowledging after the instance has given up                        (fault tests)
  provider_error  answer every round with `llm_result: false`, as a failed provider call would (fault tests)
  scripted        answer each AGENT round with the next step of a fixed script of tool calls and text (tests): it
                  drives Odoo's real tools and interaction protocol with no provider and no model

Accounting is kept for the report: every provider attempt is counted BEFORE the HTTP request leaves, so
`provider_attempts == 0` is positive evidence that nothing was sent; token counts are the provider's own
per-response `usage`; an attempt that ends without a usage-bearing response makes the totals a lower bound.

Never logged, never forwarded to the provider: the provider key, the instance's `account_token` and `dbuuid`,
the per-request `webhook_secret`, and the callback signature. The log (`standin.jsonl` in the run directory)
holds request summaries, the translated replies and the provider's usage — conversation content, so it is
PRIVATE evidence like the rest of the run directory."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import httpx

from .protocol import (
    CALLBACK_PATH,
    COMPLETIONS_ROUTE,
    KNOWN_PARAMS,
    from_chat_response,
    sign,
    to_chat_messages,
    to_chat_tools,
)

MODES = ("provider", "canned", "bad_signature", "late_ack", "provider_error", "scripted")
NO_PROVIDER_MODES = ("canned", "bad_signature", "late_ack", "provider_error", "scripted")
CANNED_TEXT = "AGENT-REVIEW STAND-IN: canned reply. No provider request was made."
DEFAULT_PROVIDER_BASE_URL = "https://api.openai.com/v1"
INSTANCE_ACK_TIMEOUT_S = 5.0     # how long the instance waits for the acknowledgement
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def check_provider_base_url(url: str) -> str:
    """https for any real host; plain http only on the loopback interface (a local test provider). The key is
    sent to this URL, so it is the operator's explicit choice and is recorded in the run conditions."""
    u = urlparse(url or "")
    if u.scheme == "https" and u.hostname:
        return url.rstrip("/")
    if u.scheme == "http" and u.hostname in LOOPBACK_HOSTS:
        return url.rstrip("/")
    raise ValueError(f"provider base_url must be https://… (or http:// on the loopback interface), got {url!r}")


class ProviderResponseError(RuntimeError):
    """The provider answered, but not with a usable completion (a non-200 status, or no usage)."""


class TranslationError(ValueError):
    """The round could not be translated for the provider: nothing was sent, so nothing was spent."""


class StandInWorkersAlive(RuntimeError):
    """Completion workers were still running at the stop deadline. Their replies are never delivered."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StandIn:
    def __init__(self, *, instance_base_url: str, instance_db: str, expected_tokens: set[str], log_path: str,
                 mode: str = "canned", api_key: str | None = None, model: str | None = None,
                 provider_base_url: str = DEFAULT_PROVIDER_BASE_URL, reasoning_effort: str | None = "medium",
                 provider_timeout_s: float = 120.0, late_ack_s: float = INSTANCE_ACK_TIMEOUT_S + 1.5,
                 script: list[list[dict]] | None = None, reply_delay_s: float = 0.0, stop_deadline_s: float = 10.0):
        if mode not in MODES:
            raise ValueError(f"stand-in mode must be one of {MODES}")
        if mode == "scripted" and not script:
            raise ValueError("scripted mode needs a script: a list of rounds, each a list of content parts")
        if mode == "provider" and (not api_key or not model):
            raise ValueError("provider mode needs the named key and a model")
        if mode in NO_PROVIDER_MODES and api_key:
            raise ValueError(f"{mode} mode makes no provider request and must not hold a key")
        if not expected_tokens:
            raise ValueError("the stand-in needs the run's synthetic IAP token(s)")
        self.base = instance_base_url.rstrip("/")
        self.db = instance_db
        self.mode = mode
        self._key = api_key
        self._expected_tokens = frozenset(expected_tokens)
        self.model = model
        self.provider_url = check_provider_base_url(provider_base_url) + "/chat/completions"
        self.reasoning_effort = reasoning_effort
        self.provider_timeout_s = provider_timeout_s
        self.late_ack_s = late_ack_s
        self.log_path = log_path
        self.records: list[dict] = []
        self.violations: list[str] = []
        self.log_errors: list[str] = []
        self._lock = threading.Lock()
        self.inflight = 0
        self.completion_requests = 0
        self.provider_attempts = 0
        self.provider_failures = 0
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.script = [list(step) for step in (script or [])]
        self._script_step = 0
        self.stopped = False
        # Shutdown. Every completion worker is tracked; once stopping begins no new round is accepted and
        # no reply is delivered to the instance, so a round still being answered cannot change the run's database
        # after the final evidence is taken. `stop()` joins the workers within a deadline and reports any still alive.
        self.reply_delay_s = reply_delay_s          # tests only: hold every reply this long (interrupted by stop())
        self.stop_deadline_s = stop_deadline_s
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._delivering = 0
        self.callbacks_suppressed = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> int:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_class())
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, name="agent-review-standin", daemon=True)
        self.thread.start()
        self._log({"event": "standin_started", "port": self.port, "mode": self.mode,
                   "model": self.model if self.mode == "provider" else None, "bound_instance": self.base,
                   "bound_db": self.db, "expected_tokens": len(self._expected_tokens)})
        return self.port

    @property
    def port(self) -> int:
        if not self.server:
            raise RuntimeError("the stand-in is not running")
        return self.server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self, deadline_s: float | None = None) -> int:
        """Stop for good and return how many completion workers are still alive at the deadline. From the first
        call on, no new round is accepted and no reply is delivered to the instance (a worker that finishes later
        records its callback as suppressed); deliveries already under way are waited for. Idempotent: a later call
        waits again for workers still alive. A worker can only be waited for, not killed: one still inside a provider
        request may yet be billed for it."""
        with self._lock:
            self._stop_event.set()
            workers = list(self._workers)
        if self.server:
            server, self.server = self.server, None
            server.shutdown()
            server.server_close()
        end = time.monotonic() + (self.stop_deadline_s if deadline_s is None else deadline_s)
        for t in workers:
            if t.ident is not None:            # registered but not started yet: it will see the stop and do nothing
                t.join(max(0.0, end - time.monotonic()))
        alive = self.workers_alive()
        if not self.stopped:
            self.stopped = True
            with self._lock:
                delivering = self._delivering
            self._log({"event": "standin_stopped", "completion_requests": self.completion_requests,
                       "provider_attempts": self.provider_attempts, "inflight": self.inflight,
                       "workers_alive": alive, "callbacks_being_delivered": delivering,
                       "callbacks_suppressed": self.callbacks_suppressed})
        return alive

    def workers_alive(self) -> int:
        with self._lock:
            return sum(1 for t in self._workers if t.is_alive())

    def wait_idle(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.inflight == 0:
                return True
            time.sleep(0.2)
        return self.inflight == 0

    def records_since(self, index: int) -> list[dict]:
        with self._lock:
            return list(self.records[index:])

    # ------------------------------------------------------------------ accounting, for the report
    def usage_summary(self) -> dict[str, Any]:
        """Provider usage as observed here. `complete` is False when any attempt ended without a usage-bearing
        response, or a request is still in flight: the token totals are then a lower bound."""
        with self._lock:
            calls = [r for r in self.records if r.get("event") == "callback" and r.get("provider_called")]
            agent = [r for r in calls if str(r.get("usage", "")).startswith("agent:")]
            tokens = {k: sum((r.get("tokens") or {}).get(k, 0) for r in calls) for k in ("input", "output", "cached_input")}
            return {"mode": self.mode, "completion_requests": self.completion_requests,
                    "provider_attempts": self.provider_attempts, "provider_failures": self.provider_failures,
                    "inflight": self.inflight, "agent_calls_with_usage": sum(1 for r in agent if r.get("tokens")),
                    "other_calls_with_usage": sum(1 for r in calls if r.get("tokens") and r not in agent),
                    "tokens": tokens, "served_models": sorted({r["served_model"] for r in calls if r.get("served_model")}),
                    "complete": self.provider_failures == 0 and self.inflight == 0 and not self.log_errors}

    # ------------------------------------------------------------------ logging (callers never pass secrets)
    def _log(self, rec: dict) -> None:
        rec = {"ts": _now(), **rec}
        with self._lock:
            self.records.append(rec)
            try:
                with open(self.log_path, "a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            except OSError as e:          # the in-memory record still counts; the report says the file is short
                self.log_errors.append(type(e).__name__)

    # ------------------------------------------------------------------ requests
    def _handler_class(self):
        standin = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):        # nothing to stderr; everything worth keeping goes to the log
                pass

            def do_GET(self):
                self._send(404, {"error": "the stand-in serves one JSON-RPC route"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    body = {}
                rpc_id = body.get("id") if isinstance(body, dict) else None
                if self.path != COMPLETIONS_ROUTE:
                    standin._log({"event": "unsupported_route", "path": self.path[:200]})
                    return self._rpc_error(rpc_id, "odoo.exceptions.UserError", "the stand-in serves one route only")
                params = body.get("params") if isinstance(body, dict) else None
                params = params if isinstance(params, dict) else {}
                refusal = standin._check_binding(params)
                if refusal:
                    with standin._lock:
                        standin.violations.append(refusal)
                    standin._log({"event": "refused", "reason": refusal})
                    return self._rpc_error(rpc_id, "odoo.exceptions.AccessError", refusal)
                received = time.monotonic()
                summary = standin._summarise(params)
                with standin._lock:            # registered under the lock that stop() takes: none is missed
                    stopping = standin._stop_event.is_set()
                    if not stopping:
                        standin.inflight += 1
                        standin.completion_requests += 1
                        worker = threading.Thread(target=standin._complete, args=(params, summary),
                                                  name="agent-review-standin-worker", daemon=True)
                        standin._workers.append(worker)
                if stopping:                   # not a binding refusal: the run is over, nothing more is answered
                    standin._log({"event": "rejected_while_stopping", "usage": params.get("usage")})
                    return self._rpc_error(rpc_id, "odoo.exceptions.UserError", "the stand-in is stopping")
                # what arrived, recorded apart from the refusal above: whether the token is one of this run's own
                # (a boolean, never the value), so a test can see a planted credential reach the stand-in even
                # if every layer before this one failed
                summary["run_token"] = params.get("account_token") in standin._expected_tokens
                if standin.mode == "late_ack":
                    time.sleep(standin.late_ack_s)
                ack_s = round(time.monotonic() - received, 3)
                # the instance waits about INSTANCE_ACK_TIMEOUT_S for this acknowledgement and then gives the round
                # up; an ack slower than that is recorded as late, whether or not the socket noticed
                summary["ack_s"], summary["ack_late"] = ack_s, ack_s >= INSTANCE_ACK_TIMEOUT_S
                standin._log({"event": "completion_request", **summary})
                self._rpc_result(rpc_id, {"accepted": True, "request_uuid": params.get("request_uuid")})
                worker.start()

            def _rpc_result(self, rpc_id, result):
                self._send(200, {"jsonrpc": "2.0", "id": rpc_id, "result": result})

            def _rpc_error(self, rpc_id, name, message):
                self._send(200, {"jsonrpc": "2.0", "id": rpc_id,
                                 "error": {"code": 200, "message": message, "data": {"name": name, "message": message}}})

            def _send(self, status, obj):
                data = json.dumps(obj).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    standin._log({"event": "ack_undeliverable", "note": "the instance closed the connection first"})

        return Handler

    def _check_binding(self, params: dict) -> str | None:
        """Why a request is refused, as a reason class. Never the offending value: a token that is not this run's
        is exactly the thing that must not be written anywhere."""
        want = self.base + CALLBACK_PATH
        if params.get("webhook_url") != want:
            return "webhook_url is not this run's instance"
        if params.get("webhook_dbname") != self.db:
            return "webhook_dbname is not this run's database"
        if params.get("account_token") not in self._expected_tokens:
            return "account_token is not one of this run's synthetic tokens (a real or unexpected IAP credential)"
        if not params.get("request_uuid") or not params.get("webhook_secret"):
            return "request_uuid or webhook_secret missing"
        return None

    @staticmethod
    def _summarise(params: dict) -> dict:
        msgs = params.get("messages") or []
        return {"request_uuid": params.get("request_uuid"), "usage": params.get("usage"),
                "boost_reasoning": params.get("boost_reasoning"), "timeout": params.get("timeout"),
                "n_messages": len(msgs) if isinstance(msgs, list) else None,
                "tools": [t.get("name") for t in params.get("tools") or [] if isinstance(t, dict)],
                "unknown_params": sorted(set(params) - KNOWN_PARAMS)}

    def _scripted_reply(self, params: dict) -> dict:
        """The next step of the script for an AGENT round (a short text for any other round, e.g. naming the
        conversation). Tool calls get fresh call ids. Not a model: a fixed trajectory, for plumbing tests."""
        if not str(params.get("usage", "")).startswith("agent:"):
            content = [{"type": "text", "text": "Scripted run"}]
        else:
            with self._lock:
                step, self._script_step = self._script_step, self._script_step + 1
            parts = self.script[min(step, len(self.script) - 1)]
            content = [dict(p, call_id=f"scripted_{step}_{k}") if p.get("type") == "tool_call" else dict(p)
                       for k, p in enumerate(parts)]
        return {"role": "assistant", "content": content,
                "provider_metadata": {"agent_review_standin": True, "served_model": None, "scripted": True}}

    def _complete(self, params: dict, summary: dict) -> None:
        uuid_ = params.get("request_uuid")
        t0 = time.perf_counter()
        llm_error: Any = None
        provider: dict[str, Any] = {}
        if self._stop_event.wait(self.reply_delay_s) if self.reply_delay_s else self._stop_event.is_set():
            # stopped before the round was answered: no provider request, no reply to the instance
            with self._lock:
                self.callbacks_suppressed += 1
                self.inflight -= 1
            self._log({"event": "callback", "request_uuid": uuid_, "usage": summary.get("usage"), "delivered": False,
                       "suppressed": "the stand-in was stopped before this round was answered",
                       "provider_called": False, "llm_result_is_false": None, "llm_error": None})
            return
        try:
            if self.mode == "provider":
                llm_result, provider = self._call_provider(params)
            elif self.mode == "provider_error":
                llm_result, llm_error = False, "stand-in: simulated provider failure"
            elif self.mode == "scripted":
                llm_result = {"result": self._scripted_reply(params)}
            else:
                llm_result = {"result": {"role": "assistant", "content": [{"type": "text", "text": CANNED_TEXT}],
                                         "provider_metadata": {"agent_review_standin": True, "served_model": None}}}
        except TranslationError as e:   # nothing left this process: a failed round, no spend
            llm_result, llm_error = False, f"stand-in could not translate the round: {type(e).__name__}"
            provider = {"provider_called": False, "error_type": "TranslationError"}
        except Exception as e:  # noqa: BLE001 — any provider failure becomes the instance's own failure path
            llm_result, llm_error = False, f"stand-in provider call failed: {type(e).__name__}"
            provider = {"provider_called": True, "error_type": type(e).__name__}
            with self._lock:
                self.provider_failures += 1
        body = json.loads(json.dumps({"request_uuid": uuid_, "llm_result": llm_result, "llm_error": llm_error}))
        signature = sign(params["webhook_secret"], body["request_uuid"], body["llm_result"], body["llm_error"])
        if self.mode == "bad_signature":
            signature = ("0" if signature[0] != "0" else "1") + signature[1:]
        body["signature"] = signature
        cb: dict[str, Any] = {"event": "callback", "request_uuid": uuid_, "usage": summary.get("usage"),
                              "provider_s": round(time.perf_counter() - t0, 3), "llm_result_is_false": llm_result is False,
                              "llm_error": llm_error, "signature_valid": self.mode != "bad_signature",
                              "provider_called": False, **provider}
        if llm_result is not False:
            cb["assistant_message"] = llm_result["result"]
        with self._lock:                       # the gate: once stopping has begun, nothing reaches the instance
            deliver = not self._stop_event.is_set()
            if deliver:
                self._delivering += 1
            else:
                self.callbacks_suppressed += 1
        cb["delivered"] = deliver
        if not deliver:
            cb["suppressed"] = "the stand-in was stopped before this reply was delivered"
            self._log(cb)                      # usage stays counted: a provider request may already have been made
            with self._lock:
                self.inflight -= 1
            return
        try:
            r = httpx.post(self.base + CALLBACK_PATH, json=body, timeout=120.0, headers={"X-Odoo-Database": self.db})
            cb["callback_http_status"] = r.status_code
        except Exception as e:  # noqa: BLE001 — recorded; the instance may already be stopping
            cb["callback_error"] = type(e).__name__
        finally:
            self._log(cb)
            with self._lock:
                self._delivering -= 1
                self.inflight -= 1

    def _call_provider(self, params: dict) -> tuple[dict, dict]:
        try:
            messages, notes = to_chat_messages(params.get("instructions") or "", params.get("messages") or [])
            tools = to_chat_tools(params.get("tools"))
        except (AttributeError, TypeError, ValueError) as e:
            raise TranslationError(str(e)) from e
        req: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.reasoning_effort:
            req["reasoning_effort"] = "high" if params.get("boost_reasoning") else self.reasoning_effort
        if tools:
            req["tools"] = tools
        with self._lock:
            self.provider_attempts += 1        # counted BEFORE anything leaves: zero attempts is proof of none
        r = httpx.post(self.provider_url, json=req, timeout=self.provider_timeout_s,
                       headers={"Authorization": f"Bearer {self._key}"})
        if r.status_code != 200:
            raise ProviderResponseError(f"provider HTTP {r.status_code}")
        resp = r.json()
        u = resp.get("usage") if isinstance(resp, dict) else None
        if not u or not hasattr(u, "get"):
            raise ProviderResponseError("the provider response carries no usage")
        provider = {"provider_called": True, "served_model": resp.get("model"), "requested_model": self.model,
                    "translation_notes": notes, "reasoning_effort": req.get("reasoning_effort"),
                    "tokens": {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0),
                               "cached_input": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                               "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)}}
        return {"result": from_chat_response(resp)}, provider
