"""The Odoo 20 stand-in's guarantees, attacked without Odoo, a database or a provider account. A fake instance (the
callback receiver) and a fake provider run as local HTTP servers in the test.

Guarantees under test, each with the attack that would break it:
  - a completion request carrying any token other than the run's synthetic ones is REFUSED: no provider request,
    no callback, and the offending value is written nowhere (a sentinel real-looking token is planted);
  - the per-run binding holds: another instance URL or database is refused;
  - nothing secret is logged or forwarded: the provider key, the instance's account token and dbuuid, the
    per-request webhook secret, the callback signature;
  - the provider request carries only the conversation, tools and model (no Odoo credential);
  - a no-provider mode makes no provider request, holds no key, and says so with a counter;
  - provider failures and missing usage make the usage a lower bound, never a complete figure;
  - the stand-in listens on 127.0.0.1 only;
  - the callback signature is the scheme the instance verifies;
  - once stopped, nothing more reaches the instance: a reply finished later is recorded as suppressed, and a
    worker still running at the deadline is reported, not waited on forever."""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from agent_review.drivers.native_ai_20.protocol import (
    CALLBACK_PATH,
    COMPLETIONS_ROUTE,
    SYNTHETIC_TOKEN_PREFIX,
    from_chat_response,
    sign,
    synthetic_token,
    to_chat_messages,
)
from agent_review.drivers.native_ai_20.standin import StandIn, check_provider_base_url

SENTINEL = "sk-live-REALLOOKING-" + "9f3a" * 6          # a planted, real-looking credential
DBUUID = "dbuuid-" + "7c1e" * 8
KEY = "sk-test-KEYVALUE-" + "c0ffee" * 4


class _Recorder:
    """A local HTTP server that records every request (headers and body) and answers with `reply(body)`."""

    def __init__(self, reply=lambda body: (200, {"ok": True})):
        self.requests: list[dict] = []
        self.reply = reply
        recorder = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                recorder.requests.append({"path": self.path, "headers": dict(self.headers), "raw": raw.decode()})
                status, obj = recorder.reply(json.loads(raw or b"{}"))
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def instance():
    r = _Recorder()
    yield r
    r.close()


def _params(instance_url, db="run_db", token=None, uuid="req-1", secret="whsecret-" + "ab" * 8, usage="agent:ai.default"):
    return {"messages": [{"role": "user", "content": [{"type": "text", "text": "Reassign the three opportunities"}]}],
            "instructions": "You are Odoo's agent.", "tools": [{"name": "ai_tool_update_records", "instructions": "update",
                                                                  "schema": {"type": "object", "properties": {}}}],
            "usage": usage, "boost_reasoning": False, "request_uuid": uuid, "webhook_url": instance_url + CALLBACK_PATH,
            "webhook_secret": secret, "webhook_dbname": db, "llm_retry": False, "account_token": token, "dbuuid": DBUUID}


def _post(standin, params, route=COMPLETIONS_ROUTE):
    return httpx.post(standin.url + route, json={"jsonrpc": "2.0", "method": "call", "params": params, "id": 7}, timeout=10).json()


def _standin(tmp_path, instance, **kw):
    token = synthetic_token()
    kw.setdefault("mode", "canned")
    s = StandIn(instance_base_url=instance.url, instance_db="run_db", expected_tokens={token},
                log_path=str(tmp_path / "standin.jsonl"), **kw)
    s.start()
    return s, token


def _everything_written(tmp_path, standin) -> str:
    log = (tmp_path / "standin.jsonl").read_text() if (tmp_path / "standin.jsonl").exists() else ""
    return log + json.dumps(standin.records, default=str) + json.dumps(standin.violations)


def test_signature_is_the_scheme_the_instance_verifies():
    secret, uuid_, result, error = "s3cr3t", "u-1", {"result": {"role": "assistant", "content": []}}, None
    message = repr(("odoo_ai-webhook", (uuid_, result, error)))
    assert sign(secret, uuid_, result, error) == hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def test_a_correctly_bound_round_is_acknowledged_and_called_back_with_a_valid_signature(tmp_path, instance):
    s, token = _standin(tmp_path, instance)
    try:
        p = _params(instance.url, token=token)
        assert _post(s, p)["result"]["accepted"] is True
        assert s.wait_idle(10)
        [cb] = [r for r in instance.requests if r["path"] == CALLBACK_PATH]
        body = json.loads(cb["raw"])
        assert cb["headers"].get("X-Odoo-Database") == "run_db"
        assert body["signature"] == sign(p["webhook_secret"], body["request_uuid"], body["llm_result"], body["llm_error"])
        assert body["llm_result"]["result"]["provider_metadata"]["served_model"] is None   # canned: no model
        assert s.usage_summary()["provider_attempts"] == 0
    finally:
        s.stop()


@pytest.mark.parametrize("attack", ["sentinel_token", "missing_token", "other_instance", "other_database", "no_secret"])
def test_a_request_outside_the_runs_binding_is_refused_and_never_called_back(tmp_path, instance, attack):
    provider = _Recorder()
    s, token = _standin(tmp_path, instance, mode="provider", api_key=KEY, model="m-1", provider_base_url=provider.url + "/v1")
    try:
        p = _params(instance.url, token=token)
        if attack == "sentinel_token":
            p["account_token"] = SENTINEL
        elif attack == "missing_token":
            p["account_token"] = None
        elif attack == "other_instance":
            p["webhook_url"] = "http://127.0.0.1:1" + CALLBACK_PATH
        elif attack == "other_database":
            p["webhook_dbname"] = "production"
        else:
            p["webhook_secret"] = ""
        reply = _post(s, p)
        assert "error" in reply and "result" not in reply
        s.wait_idle(5)
        assert instance.requests == [] and provider.requests == []          # nothing called back, nothing forwarded
        assert len(s.violations) == 1 and s.usage_summary()["provider_attempts"] == 0
        written = _everything_written(tmp_path, s) + json.dumps(reply)
        assert SENTINEL not in written and DBUUID not in written and KEY not in written
    finally:
        s.stop()
        provider.close()


def test_provider_request_carries_no_odoo_credential_and_nothing_secret_is_logged(tmp_path, instance):
    def reply(body):
        return 200, {"model": "m-1-2026-01-01", "usage": {"prompt_tokens": 120, "completion_tokens": 30,
                                                          "prompt_tokens_details": {"cached_tokens": 20}},
                     "choices": [{"message": {"content": None, "tool_calls": [
                         {"id": "call_1", "function": {"name": "ai_tool_update_records", "arguments": "{\"x\": 1}"}}]}}]}
    provider = _Recorder(reply)
    s, token = _standin(tmp_path, instance, mode="provider", api_key=KEY, model="m-1", provider_base_url=provider.url + "/v1")
    try:
        p = _params(instance.url, token=token)
        _post(s, p)
        assert s.wait_idle(10)
        [req] = provider.requests
        assert req["path"] == "/v1/chat/completions" and req["headers"]["Authorization"] == f"Bearer {KEY}"
        for secret in (token, DBUUID, p["webhook_secret"]):
            assert secret not in req["raw"]                                   # no Odoo credential leaves for the provider
        sent = json.loads(req["raw"])
        assert set(sent) == {"model", "messages", "tools", "reasoning_effort"} and sent["model"] == "m-1"
        [cb] = instance.requests
        body = json.loads(cb["raw"])
        written = _everything_written(tmp_path, s)
        for secret in (KEY, token, DBUUID, p["webhook_secret"], body["signature"]):
            assert secret not in written                                      # never logged
        assert [c["type"] for c in body["llm_result"]["result"]["content"]] == ["tool_call"]
        u = s.usage_summary()
        assert u["complete"] and u["provider_attempts"] == 1 and u["tokens"] == {"input": 120, "output": 30, "cached_input": 20}
        assert u["served_models"] == ["m-1-2026-01-01"] and u["agent_calls_with_usage"] == 1
    finally:
        s.stop()
        provider.close()


@pytest.mark.parametrize("failure", ["http_500", "no_usage"])
def test_a_failed_provider_call_is_a_failed_round_and_makes_usage_a_lower_bound(tmp_path, instance, failure):
    def reply(body):
        if failure == "http_500":
            return 500, {"error": "boom"}
        return 200, {"model": "m-1", "choices": [{"message": {"content": "hi"}}]}   # no usage at all
    provider = _Recorder(reply)
    s, token = _standin(tmp_path, instance, mode="provider", api_key=KEY, model="m-1", provider_base_url=provider.url + "/v1")
    try:
        _post(s, _params(instance.url, token=token))
        assert s.wait_idle(10)
        body = json.loads(instance.requests[0]["raw"])
        assert body["llm_result"] is False                                   # the instance's own failure path
        u = s.usage_summary()
        assert not u["complete"] and u["provider_failures"] == 1 and u["provider_attempts"] == 1
    finally:
        s.stop()
        provider.close()


def test_no_provider_modes_hold_no_key_and_count_no_attempt(tmp_path, instance):
    with pytest.raises(ValueError, match="must not hold a key"):
        StandIn(instance_base_url=instance.url, instance_db="d", expected_tokens={"t"}, log_path=str(tmp_path / "l"),
                mode="canned", api_key=KEY)
    with pytest.raises(ValueError, match="needs the named key"):
        StandIn(instance_base_url=instance.url, instance_db="d", expected_tokens={"t"}, log_path=str(tmp_path / "l"),
                mode="provider", model="m")
    s, token = _standin(tmp_path, instance, mode="provider_error")
    try:
        _post(s, _params(instance.url, token=token))
        assert s.wait_idle(10)
        assert json.loads(instance.requests[0]["raw"])["llm_result"] is False
        assert s.usage_summary()["provider_attempts"] == 0
    finally:
        s.stop()


def test_the_standin_listens_on_loopback_only_and_the_provider_url_must_be_https(tmp_path, instance):
    s, _ = _standin(tmp_path, instance)
    try:
        assert s.server.server_address[0] == "127.0.0.1" and s.url.startswith("http://127.0.0.1:")
    finally:
        s.stop()
    assert check_provider_base_url("https://api.example.com/v1") == "https://api.example.com/v1"
    assert check_provider_base_url("http://127.0.0.1:9/v1") == "http://127.0.0.1:9/v1"
    for bad in ("http://api.example.com/v1", "ftp://x", "", "https://"):
        with pytest.raises(ValueError):
            check_provider_base_url(bad)


def test_a_late_acknowledgement_is_recorded_as_late(tmp_path, instance):
    s, token = _standin(tmp_path, instance, mode="late_ack", late_ack_s=0.3)
    s_fast, token_fast = _standin(tmp_path, instance)
    try:
        import agent_review.drivers.native_ai_20.standin as mod
        orig = mod.INSTANCE_ACK_TIMEOUT_S
        mod.INSTANCE_ACK_TIMEOUT_S = 0.2        # the instance's real wait is ~5 s; shortened for the test
        try:
            _post(s, _params(instance.url, token=token))
            _post(s_fast, _params(instance.url, token=token_fast))
        finally:
            mod.INSTANCE_ACK_TIMEOUT_S = orig
        late = [r for r in s.records if r.get("event") == "completion_request"]
        fast = [r for r in s_fast.records if r.get("event") == "completion_request"]
        assert late[0]["ack_late"] is True and fast[0]["ack_late"] is False     # positive and negative control
    finally:
        s.wait_idle(5)
        s_fast.wait_idle(5)
        s.stop()
        s_fast.stop()


def test_translation_closes_unanswered_calls_and_keeps_unparseable_arguments():
    msgs, notes = to_chat_messages("sys", [
        {"role": "assistant", "content": [{"type": "tool_call", "name": "t", "args": {"a": 1}, "call_id": "c1"}]},
        {"role": "user", "content": [{"type": "text", "text": "next"}]}])
    assert [m["role"] for m in msgs] == ["system", "assistant", "tool", "user"] and notes
    out = from_chat_response({"model": "m", "choices": [{"message": {"tool_calls": [
        {"id": "x", "function": {"name": "t", "arguments": "{not json"}}]}}]})
    assert out["content"][0]["args"] == {"_unparseable_arguments": "{not json"}
    assert synthetic_token().startswith(SYNTHETIC_TOKEN_PREFIX) and len(synthetic_token()) <= 43


def test_scripted_mode_answers_agent_rounds_in_order_and_calls_no_provider(tmp_path, instance):
    script = [[{"type": "tool_call", "name": "ai_tool_load_skills", "args": {"skill_ids": [3]}}],
              [{"type": "text", "text": "done"}]]
    with pytest.raises(ValueError, match="needs a script"):
        StandIn(instance_base_url=instance.url, instance_db="d", expected_tokens={"t"}, log_path=str(tmp_path / "l"),
                mode="scripted")
    s, token = _standin(tmp_path, instance, mode="scripted", script=script)
    try:
        for i, usage in enumerate(["agent:ai.default", "channel_name", "agent:ai.default", "agent:ai.default"]):
            _post(s, _params(instance.url, token=token, uuid=f"r{i}", usage=usage))
            assert s.wait_idle(10)
        replies = [json.loads(r["raw"])["llm_result"]["result"]["content"] for r in instance.requests]
        assert replies[0][0]["name"] == "ai_tool_load_skills" and replies[0][0]["call_id"] == "scripted_0_0"
        assert replies[1] == [{"type": "text", "text": "Scripted run"}]          # a non-agent round
        assert replies[2] == [{"type": "text", "text": "done"}] and replies[3] == replies[2]   # the last step repeats
        assert s.usage_summary()["provider_attempts"] == 0
    finally:
        s.stop()


# ------------------------------------------------------------------ shutdown (found by a review)
def _until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def test_a_worker_still_in_a_provider_request_is_reported_at_stop_and_its_reply_never_reaches_the_instance(tmp_path, instance):
    """The review's reproduction: stop() returned with a worker alive, and releasing it produced a callback after
    `standin_stopped`. Now the stop reports the live worker, and its late reply is recorded as suppressed: it was
    billed (the provider answered) but it is never delivered."""
    release = threading.Event()

    def reply(body):
        release.wait(20)
        return 200, {"model": "m-served", "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                     "choices": [{"message": {"content": "a reply that arrives too late"}}]}
    provider = _Recorder(reply)
    try:
        s, token = _standin(tmp_path, instance, mode="provider", api_key=KEY, model="m", provider_base_url=provider.url + "/v1",
                            stop_deadline_s=0.3)
        assert _post(s, _params(instance.url, token=token))["result"]["accepted"]
        assert _until(lambda: provider.requests)              # the worker is inside the provider request
        assert s.stop() == 1                                   # reported, not waited on forever
        [stopped] = [e for e in s.records if e["event"] == "standin_stopped"]
        assert stopped["workers_alive"] == 1
        release.set()
        assert _until(lambda: s.workers_alive() == 0)
        [cb] = [e for e in s.records if e["event"] == "callback"]
        assert cb["delivered"] is False and cb["provider_called"] is True and cb["suppressed"]
        assert instance.requests == []                         # nothing reached the instance, then or later
        assert s.callbacks_suppressed == 1 and s.stop() == 0
    finally:
        release.set()
        provider.close()


def test_a_reply_held_when_the_standin_stops_is_dropped_at_once_and_never_delivered(tmp_path, instance):
    s, token = _standin(tmp_path, instance, reply_delay_s=30, stop_deadline_s=5)
    assert _post(s, _params(instance.url, token=token))["result"]["accepted"]
    t0 = time.monotonic()
    assert s.stop() == 0 and time.monotonic() - t0 < 5         # the stop wakes the held reply: no 30 s wait
    [cb] = [e for e in s.records if e["event"] == "callback"]
    assert cb["delivered"] is False and cb["provider_called"] is False
    assert instance.requests == [] and s.provider_attempts == 0
