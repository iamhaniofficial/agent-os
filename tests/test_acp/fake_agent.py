#!/usr/bin/env python3
"""A scripted fake ACP *agent* over stdio (stdlib only).

Start it as a harness subprocess: ``[sys.executable, path/to/fake_agent.py]``
(e.g. ``AcpConfig(agents={"fake": AcpAgentOverride(command=sys.executable,
args=[FAKE_AGENT])})``). It speaks newline-delimited JSON-RPC 2.0 like a real
ACP adapter and behaves according to the script chosen by env
``FAKE_ACP_SCRIPT`` (default ``echo``). A prompt whose text starts with
``script:<name>`` runs that script for that one turn instead, so one process
can exercise several behaviours.

Every session gets: modes ``default`` / ``plan`` / ``acceptEdits`` (current
``default``), a ``model`` select config option (``fake-small`` /
``fake-large``), and, right after ``session/new``, a large
``available_commands_update`` (like Claude's adapter) that must never show up
as turn text. ``session/set_mode`` answers and then sends a
``current_mode_update``. ``session/close`` is supported. The process exits on
stdin EOF.

Scripts (prompt behaviour unless noted):

``echo``
    One ``agent_message_chunk`` ``"echo: <prompt text>"``, a
    ``usage_update`` (used 100 / size 1000, cumulative cost +0.01 USD per
    turn), stop ``end_turn`` with ``usage`` 10 in / 5 out.
``tool_then_text``
    ``tool_call`` t1 (read, pending, rawInput ``{"path": "README.md"}``),
    ``tool_call_update`` in_progress, ``tool_call_update`` completed with
    rawOutput ``{"content": "hello"}`` (no title: the client must merge it),
    then text ``"done"``.
``permission_write``
    ``tool_call`` t1 (edit ``Write hello.txt``), then
    ``session/request_permission`` with options ``a1`` allow_once,
    ``a2`` allow_always, ``r1`` reject_once. Replies with text
    ``"permission: allowed (<id>)"`` / ``"permission: rejected (<id>)"``;
    a ``cancelled`` outcome ends the turn with stop ``cancelled``.
``elicit_form``
    ``elicitation/create`` (form: ``{"color": string}``) and replies
    ``"elicit: <action> <json content>"``.
``slow``
    Text ``"working"``, then waits up to ``FAKE_ACP_SLOW_S`` (default 30) s;
    honours ``session/cancel`` by returning stop ``cancelled``.
``stubborn``
    Like ``slow`` but ignores ``session/cancel`` and SIGTERM (tests the
    SIGKILL step of the cancel ladder).
``crash_mid_turn``
    Text ``"about to crash"``, then exits with code 3 mid-turn.
``plan_usage``
    ``plan`` (two entries), ``agent_thought_chunk``, ``session_info_update``
    (title), ``usage_update`` with a 0.25 USD cost, text ``"planned"``; the
    response carries tokens in ``_meta.quota`` instead of ``usage``.
``env``
    Text is JSON ``{"keys": [...sorted env names], "lease": ..., "codex_home":
    ..., "tokio": ..., "extra": $FAKE_ACP_EXTRA}``.
``load_session_ok`` (session behaviour)
    Advertises ``loadSession``; ``session/load`` replays a user and an agent
    chunk then succeeds. Prompts echo ``"resumed:<sessionId>: <text>"``
    once loaded, otherwise behave like ``echo``.
``load_session_fail`` (session behaviour)
    Advertises ``loadSession`` but ``session/load`` returns an error.
``hang_init`` (start-up behaviour)
    Never answers ``initialize``.
``fail_init`` (start-up behaviour)
    Writes a line to stderr and exits 2 before answering anything.
``fail_new_session`` (start-up behaviour)
    ``session/new`` returns a JSON-RPC error.

``FAKE_ACP_STDERR_SECRET=1`` writes a line with a fake API key to stderr at
start (redaction tests). ``FAKE_ACP_PID_FILE=<path>`` writes the pid there.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import sys
import threading
import time
import uuid
from typing import Any

SCRIPT = os.environ.get("FAKE_ACP_SCRIPT", "echo")
SLOW_S = float(os.environ.get("FAKE_ACP_SLOW_S", "30"))

_write_lock = threading.Lock()
_pending: dict[int, dict[str, Any]] = {}
_pending_events: dict[int, threading.Event] = {}
_work: queue.Queue[dict[str, Any] | None] = queue.Queue()
_cancel = threading.Event()
_next_id = 1000
_cost = 0.0
_loaded: set[str] = set()
_modes = {"current": "default"}
_model = {"value": "fake-small"}


def send(msg: dict[str, Any]) -> None:
    msg = {"jsonrpc": "2.0", **msg}
    line = json.dumps(msg, separators=(",", ":"))
    with _write_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def notify(method: str, params: dict[str, Any]) -> None:
    send({"method": method, "params": params})


def update(session_id: str, upd: dict[str, Any]) -> None:
    notify("session/update", {"sessionId": session_id, "update": upd})


def request(method: str, params: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    global _next_id
    _next_id += 1
    rid = _next_id
    ev = threading.Event()
    _pending_events[rid] = ev
    send({"id": rid, "method": method, "params": params})
    ev.wait(timeout)
    _pending_events.pop(rid, None)
    return _pending.pop(rid, {"error": {"code": -1, "message": "timeout"}})


def reader() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if "method" not in msg and "id" in msg:
            rid = msg["id"]
            _pending[rid] = msg
            ev = _pending_events.get(rid)
            if ev is not None:
                ev.set()
            continue
        if msg.get("method") == "session/cancel":
            _cancel.set()
            continue
        _work.put(msg)
    _work.put(None)


def text_chunk(session_id: str, text: str, kind: str = "agent_message_chunk") -> None:
    update(session_id, {"sessionUpdate": kind, "content": {"type": "text", "text": text}})


def modes_state() -> dict[str, Any]:
    return {
        "currentModeId": _modes["current"],
        "availableModes": [
            {"id": "default", "name": "Default"},
            {"id": "plan", "name": "Plan"},
            {"id": "acceptEdits", "name": "Accept edits"},
        ],
    }


def config_options() -> list[dict[str, Any]]:
    return [
        {
            "id": "model",
            "name": "Model",
            "type": "select",
            "category": "model",
            "currentValue": _model["value"],
            "options": [
                {"value": "fake-small", "name": "Small"},
                {"value": "fake-large", "name": "Large"},
            ],
        }
    ]


def send_commands(session_id: str) -> None:
    cmds = [{"name": f"cmd{i}", "description": "x" * 200, "input": None} for i in range(40)]
    update(session_id, {"sessionUpdate": "available_commands_update", "availableCommands": cmds})


def prompt_text(params: dict[str, Any]) -> str:
    parts = []
    for block in params.get("prompt") or []:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def wait_cancel(seconds: float, honour: bool = True) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if honour and _cancel.is_set():
            return True
        time.sleep(0.01)
    return False


def run_prompt(params: dict[str, Any]) -> dict[str, Any]:
    global _cost
    sid = params["sessionId"]
    text = prompt_text(params)
    script = SCRIPT
    if text.startswith("script:"):
        head, _, rest = text.partition(" ")
        script = head[len("script:") :]
        text = rest
    _cancel.clear()

    if script == "tool_then_text":
        update(
            sid,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "t1",
                "title": "Read README.md",
                "kind": "read",
                "status": "pending",
                "rawInput": {"path": "README.md"},
            },
        )
        update(
            sid, {"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "in_progress"}
        )
        update(
            sid,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "t1",
                "status": "completed",
                "rawOutput": {"content": "hello"},
                "content": [{"type": "content", "content": {"type": "text", "text": "hello"}}],
            },
        )
        text_chunk(sid, "done")
        return {"stopReason": "end_turn"}

    if script == "permission_write":
        update(
            sid,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "t1",
                "title": "Write hello.txt",
                "kind": "edit",
                "status": "pending",
                "rawInput": {"path": "hello.txt", "content": "hi"},
            },
        )
        resp = request(
            "session/request_permission",
            {
                "sessionId": sid,
                "toolCall": {"toolCallId": "t1", "title": "Write hello.txt", "kind": "edit"},
                "options": [
                    {"optionId": "a1", "name": "Allow", "kind": "allow_once"},
                    {"optionId": "a2", "name": "Always allow", "kind": "allow_always"},
                    {"optionId": "r1", "name": "Reject", "kind": "reject_once"},
                ],
            },
        )
        outcome = (resp.get("result") or {}).get("outcome") or {}
        if outcome.get("outcome") == "selected":
            oid = outcome.get("optionId")
            allowed = oid in ("a1", "a2")
            update(
                sid,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "t1",
                    "status": "completed" if allowed else "failed",
                },
            )
            text_chunk(sid, f"permission: {'allowed' if allowed else 'rejected'} ({oid})")
            return {"stopReason": "end_turn"}
        text_chunk(sid, "permission: cancelled")
        return {"stopReason": "cancelled"}

    if script == "elicit_form":
        resp = request(
            "elicitation/create",
            {
                "sessionId": sid,
                "mode": "form",
                "message": "Pick a color",
                "requestedSchema": {
                    "type": "object",
                    "properties": {"color": {"type": "string", "title": "Color"}},
                    "required": ["color"],
                },
            },
        )
        result = resp.get("result") or {}
        action = result.get("action", "error")
        content = result.get("content")
        text_chunk(sid, f"elicit: {action} {json.dumps(content, sort_keys=True)}")
        return {"stopReason": "end_turn"}

    if script in ("slow", "stubborn"):
        text_chunk(sid, "working")
        honour = script == "slow"
        if script == "stubborn":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if wait_cancel(SLOW_S, honour=honour):
            return {"stopReason": "cancelled"}
        return {"stopReason": "end_turn"}

    if script == "crash_mid_turn":
        text_chunk(sid, "about to crash")
        sys.stdout.flush()
        os._exit(3)

    if script == "plan_usage":
        update(
            sid,
            {
                "sessionUpdate": "plan",
                "entries": [
                    {"content": "Read code", "priority": "high", "status": "completed"},
                    {"content": "Write fix", "priority": "medium", "status": "pending"},
                ],
            },
        )
        text_chunk(sid, "thinking...", kind="agent_thought_chunk")
        update(sid, {"sessionUpdate": "session_info_update", "title": "Fix the bug"})
        _cost += 0.25
        update(
            sid,
            {
                "sessionUpdate": "usage_update",
                "used": 2000,
                "size": 200000,
                "cost": {"amount": _cost, "currency": "USD"},
            },
        )
        text_chunk(sid, "planned")
        return {
            "stopReason": "end_turn",
            "_meta": {"quota": {"input_tokens": 120, "output_tokens": 30}},
        }

    if script == "env":
        payload = {
            "keys": sorted(os.environ),
            "lease": os.environ.get("AGENTOS_ACP_LEASE"),
            "codex_home": os.environ.get("CODEX_HOME"),
            "tokio": os.environ.get("TOKIO_WORKER_THREADS"),
            "extra": os.environ.get("FAKE_ACP_EXTRA"),
        }
        text_chunk(sid, json.dumps(payload, sort_keys=True))
        return {"stopReason": "end_turn"}

    # echo (default) and load_session_* prompts
    if sid in _loaded:
        text_chunk(sid, f"resumed:{sid}: {text}")
    else:
        text_chunk(sid, f"echo: {text}")
    _cost += 0.01
    update(
        sid,
        {
            "sessionUpdate": "usage_update",
            "used": 100,
            "size": 1000,
            "cost": {"amount": round(_cost, 6), "currency": "USD"},
        },
    )
    return {
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }


def handle(msg: dict[str, Any]) -> None:
    method = msg.get("method")
    rid = msg.get("id")
    params = msg.get("params") or {}

    def reply(result: Any) -> None:
        if rid is not None:
            send({"id": rid, "result": result})

    def error(code: int, message: str) -> None:
        if rid is not None:
            send({"id": rid, "error": {"code": code, "message": message}})

    if method == "initialize":
        if SCRIPT == "hang_init":
            return
        reply(
            {
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": SCRIPT in ("load_session_ok", "load_session_fail"),
                    "sessionCapabilities": {"close": {}},
                },
                "agentInfo": {"name": "fake-acp", "version": "9.9.9"},
                "authMethods": [],
            }
        )
    elif method == "session/new":
        if SCRIPT == "fail_new_session":
            error(-32603, "cannot create session")
            return
        sid = "fake-" + uuid.uuid4().hex[:8]
        reply({"sessionId": sid, "modes": modes_state(), "configOptions": config_options()})
        send_commands(sid)
    elif method == "session/load":
        sid = params.get("sessionId", "")
        if SCRIPT == "load_session_fail":
            error(-32603, "session not found")
            return
        text_chunk(sid, "old question", kind="user_message_chunk")
        text_chunk(sid, "old answer")
        _loaded.add(sid)
        reply({"modes": modes_state(), "configOptions": config_options()})
    elif method == "session/prompt":
        reply(run_prompt(params))
    elif method == "session/set_mode":
        _modes["current"] = params.get("modeId", "default")
        reply({})
        update(
            params.get("sessionId", ""),
            {"sessionUpdate": "current_mode_update", "currentModeId": _modes["current"]},
        )
    elif method == "session/set_config_option":
        if params.get("configId") != "model":
            error(-32602, "unknown config option")
            return
        _model["value"] = str(params.get("value"))
        reply({"configOptions": config_options()})
    elif method == "session/close":
        reply({})
    else:
        error(-32601, f"method not found: {method}")


def main() -> None:
    if os.environ.get("FAKE_ACP_PID_FILE"):
        with open(os.environ["FAKE_ACP_PID_FILE"], "w") as fh:
            fh.write(str(os.getpid()))
    if os.environ.get("FAKE_ACP_STDERR_SECRET") == "1":
        sys.stderr.write(
            "fake-acp starting with OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789\n"
        )
        sys.stderr.flush()
    if SCRIPT == "fail_init":
        sys.stderr.write("fake-acp: fatal: cannot start\n")
        sys.stderr.flush()
        sys.exit(2)
    threading.Thread(target=reader, daemon=True).start()
    while True:
        msg = _work.get()
        if msg is None:
            break
        handle(msg)


if __name__ == "__main__":
    main()
