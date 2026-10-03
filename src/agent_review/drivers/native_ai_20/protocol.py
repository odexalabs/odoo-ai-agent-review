"""The completion protocol between an Odoo 20 instance and its AI endpoint, as the local stand-in speaks it.

On Odoo 20 the instance does not call a model provider. Each agent round is a JSON-RPC request to the
endpoint named by the `ai.endpoint` system parameter; the endpoint acknowledges at once and later POSTs the
result back to the instance, signed:

    instance -> POST {ai.endpoint}/api/odoo_ai/1/get_completions      JSON-RPC; the reply body is ignored
                params: messages, instructions, tools, usage, boost_reasoning, [timeout],
                        request_uuid, webhook_url, webhook_secret, webhook_dbname, llm_retry,
                        account_token, dbuuid
                The instance waits about 5 seconds for the acknowledgement, then gives up on the round.
    endpoint -> POST {webhook_url}      JSON body {request_uuid, llm_result, llm_error, signature}
                signature = HMAC-SHA256(webhook_secret,
                            repr(("odoo_ai-webhook", (request_uuid, llm_result, llm_error))))
                llm_result = false (the round failed) or {"result": AssistantMessage}

The signature is Odoo's generic HMAC helper (Community `odoo.tools.misc.hmac`) over the Python repr of the
PARSED objects, so the stand-in signs exactly the objects it serialises. An AssistantMessage is
{role: "assistant", content: [text | inline_data | tool_call parts], provider_metadata: {...}}; the instance
stores each one it accepts, provider_metadata included, and sends them back as history on later rounds.

This module translates between that shape and an OpenAI Chat Completions request/response. It knows nothing
about which model Odoo's own hosted service would choose, how that service prompts, or how it retries: the
stand-in replaces it, so none of that is evaluated."""
from __future__ import annotations

import hashlib
import hmac as hmac_lib
import json
import secrets
from typing import Any

COMPLETIONS_ROUTE = "/api/odoo_ai/1/get_completions"
CALLBACK_PATH = "/ai/completion_result_ready"
KNOWN_PARAMS = {"messages", "instructions", "tools", "usage", "boost_reasoning", "timeout", "request_uuid",
                "webhook_url", "webhook_secret", "webhook_dbname", "llm_retry", "account_token", "dbuuid", "schema"}
# The disposable copy's IAP account tokens are replaced with values carrying this prefix before Odoo starts.
# 22 characters + 20 hex = 42, inside the field's 43-character size.
SYNTHETIC_TOKEN_PREFIX = "agentreview-synthetic-"


def synthetic_token() -> str:
    return SYNTHETIC_TOKEN_PREFIX + secrets.token_hex(10)


def sign(secret: str, request_uuid: Any, llm_result: Any, llm_error: Any) -> str:
    """HMAC-SHA256 over repr(("odoo_ai-webhook", (request_uuid, llm_result, llm_error))). Callers pass objects that
    have already been through a JSON round trip, which is what the instance compares against."""
    message = repr(("odoo_ai-webhook", (request_uuid, llm_result, llm_error)))
    return hmac_lib.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def parts_text(parts: list[dict] | None) -> str:
    out = []
    for p in parts or []:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            out.append(p.get("text") or "")
        elif p.get("type") == "inline_data":
            out.append(f"[inline data omitted by the stand-in: {p.get('mimetype')}]")
    return "\n".join(x for x in out if x)


def to_chat_messages(instructions: str, messages: list[dict]) -> tuple[list[dict], list[str]]:
    """Odoo's round history -> Chat Completions messages. Returns (messages, notes). A tool call the history
    leaves without a result gets a placeholder result, because Chat Completions rejects an unanswered call."""
    notes: list[str] = []
    out: list[dict] = [{"role": "system", "content": instructions or ""}]
    open_calls: list[str] = []

    def close_open(why: str) -> None:
        nonlocal open_calls
        for cid in open_calls:
            out.append({"role": "tool", "tool_call_id": cid, "content": "No result recorded by the instance."})
        if open_calls:
            notes.append(f"{len(open_calls)} tool call(s) had no result {why}; placeholder results inserted")
        open_calls = []

    for m in messages or []:
        content = m.get("content") or []
        if m.get("role") == "assistant":
            close_open("before the next assistant message")
            text = parts_text([p for p in content if p.get("type") in ("text", "inline_data")])
            calls = [p for p in content if p.get("type") == "tool_call"]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = [{"id": str(c.get("call_id")), "type": "function",
                                      "function": {"name": c.get("name"),
                                                   "arguments": json.dumps(c.get("args") or {}, ensure_ascii=False)}}
                                     for c in calls]
                open_calls = [str(c.get("call_id")) for c in calls]
            out.append(msg)
            continue
        results = [p for p in content if p.get("type") == "tool_result"]
        for r in results:
            cid = str(r.get("tool_call_id"))
            text = parts_text(r.get("result"))
            if r.get("success") is False and not text.startswith("Error"):
                text = "Error: " + text
            out.append({"role": "tool", "tool_call_id": cid, "content": text or "success"})
            if cid in open_calls:
                open_calls.remove(cid)
        if open_calls and not results:
            close_open("before a user message")
        text = parts_text([p for p in content if p.get("type") in ("text", "inline_data")])
        if text:
            out.append({"role": "user", "content": text})
    close_open("at the end of the history")
    return out, notes


def to_chat_tools(tools: list[dict] | None) -> list[dict]:
    return [{"type": "function", "function": {"name": t.get("name"), "description": t.get("instructions") or "",
                                              "parameters": t.get("schema") or {"type": "object", "properties": {}}}}
            for t in tools or []]


def from_chat_response(resp: dict) -> dict:
    """Chat Completions response -> an AssistantMessage. The served model goes into provider_metadata, which the
    instance stores with the reply."""
    choice = (resp.get("choices") or [{}])[0].get("message") or {}
    content: list[dict] = []
    if choice.get("content"):
        content.append({"type": "text", "text": choice["content"]})
    for tc in choice.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_unparseable_arguments": fn.get("arguments")}
        content.append({"type": "tool_call", "name": fn.get("name"), "args": args, "call_id": tc.get("id")})
    if not content:
        content.append({"type": "text", "text": ""})
    return {"role": "assistant", "content": content,
            "provider_metadata": {"agent_review_standin": True, "served_model": resp.get("model")}}
