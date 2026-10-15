"""ACP client side: session/update normalization, permission and elicitation
mapping, refused fs/terminal methods (no subprocess)."""

from __future__ import annotations

from typing import Any

import acp
import pytest
from acp import RequestError
from acp.schema import (
    AvailableCommand,
    AvailableCommandsUpdate,
    ConfigOptionUpdate,
    Cost,
    CurrentModeUpdate,
    ElicitationFormSessionMode,
    ElicitationSchema,
    ElicitationUrlSessionMode,
    PermissionOption,
    SessionInfoUpdate,
    ToolCallUpdate,
    UsageUpdate,
)

from agentos.acp.client import AgentOSAcpClient, normalize_update, pick_permission_option
from agentos.acp.types import (
    AcpEvent,
    ElicitationRequest,
    ElicitationResponse,
    PermissionPromptUnavailableError,
    PermissionRequest,
)

OPTIONS = [
    PermissionOption(option_id="a1", name="Allow", kind="allow_once"),
    PermissionOption(option_id="a2", name="Always", kind="allow_always"),
    PermissionOption(option_id="r1", name="Reject", kind="reject_once"),
]


def test_message_and_thought_chunks() -> None:
    ev = normalize_update(acp.update_agent_message_text("hi"))
    assert ev is not None and (ev.kind, ev.tag, ev.text) == (
        "text_delta",
        "agent_message_chunk",
        "hi",
    )
    ev = normalize_update(acp.update_agent_thought_text("hmm"))
    assert ev is not None and (ev.kind, ev.text) == ("thought_delta", "hmm")
    assert normalize_update(acp.update_user_message_text("replayed")) is None


def test_tool_call_update_merges_earlier_fields() -> None:
    tools: dict[str, dict[str, Any]] = {}
    start = acp.start_tool_call(
        "t1", "Read file", kind="read", status="pending", raw_input={"path": "a"}
    )
    ev = normalize_update(start, tools)
    assert ev is not None and ev.kind == "tool_call" and ev.status == "pending"
    assert ev.tool_call_id == "t1" and ev.tool_kind == "read" and ev.raw_input == {"path": "a"}
    done = acp.update_tool_call(
        "t1",
        status="completed",
        raw_output={"ok": True},
        content=[acp.tool_content(acp.text_block("file body"))],
    )
    ev = normalize_update(done, tools)
    assert ev is not None and ev.kind == "tool_call_update"
    assert ev.title == "Read file" and ev.tool_kind == "read"
    assert ev.status == "completed" and ev.raw_output == {"ok": True}
    assert ev.raw_input == {"path": "a"}
    assert ev.text == "file body"


def test_plan_usage_mode_status_variants() -> None:
    plan = acp.update_plan(
        [
            acp.plan_entry("first", status="completed"),
            acp.plan_entry("second", status="in_progress"),
            acp.plan_entry("third"),
        ]
    )
    ev = normalize_update(plan)
    assert ev is not None and ev.kind == "plan"
    assert ev.text.splitlines() == ["[x] first", "[~] second", "[ ] third"]

    usd = UsageUpdate(
        session_update="usage_update", used=5, size=10, cost=Cost(amount=1.5, currency="USD")
    )
    ev = normalize_update(usd)
    assert ev is not None and ev.kind == "usage" and ev.cost_usd == 1.5
    assert ev.usage is not None and ev.usage["used"] == 5 and ev.usage["size"] == 10
    eur = UsageUpdate(
        session_update="usage_update", used=5, size=10, cost=Cost(amount=2.0, currency="EUR")
    )
    ev = normalize_update(eur)
    assert ev is not None and ev.cost_usd is None and ev.usage["cost_currency"] == "EUR"
    ev = normalize_update(UsageUpdate(session_update="usage_update", used=1, size=2))
    assert ev is not None and ev.cost_usd is None

    ev = normalize_update(
        CurrentModeUpdate(session_update="current_mode_update", current_mode_id="plan")
    )
    assert ev is not None and (ev.kind, ev.text) == ("mode", "plan")
    ev = normalize_update(
        ConfigOptionUpdate(session_update="config_option_update", config_options=[])
    )
    assert ev is not None and ev.kind == "mode" and ev.tag == "config_option_update"
    ev = normalize_update(SessionInfoUpdate(session_update="session_info_update", title="T"))
    assert ev is not None and (ev.kind, ev.title) == ("status", "T")


def test_available_commands_are_summarized_not_forwarded_as_text() -> None:
    cmds = [AvailableCommand(name=f"c{i}", description="x" * 500) for i in range(60)]
    ev = normalize_update(
        AvailableCommandsUpdate(session_update="available_commands_update", available_commands=cmds)
    )
    assert ev is not None and ev.kind == "status" and ev.text == ""
    assert ev.raw["count"] == 60 and len(ev.raw["names"]) == 40
    assert "x" * 500 not in repr(ev.raw)


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ("allow_once", "a1"),
        ("allow_always", "a2"),
        ("reject_once", "r1"),
        ("reject_always", "r1"),  # not offered -> first reject
        ("cancel", None),
    ],
)
def test_pick_permission_option(decision: Any, expected: str | None) -> None:
    resp = pick_permission_option(decision, OPTIONS)
    if expected is None:
        assert resp.outcome.outcome == "cancelled"
    else:
        assert resp.outcome.outcome == "selected"
        assert resp.outcome.option_id == expected


def test_pick_permission_option_degrades_never_upgrades() -> None:
    only_once = [OPTIONS[0], OPTIONS[2]]
    assert pick_permission_option("allow_always", only_once).outcome.option_id == "a1"
    allow_only = [OPTIONS[0]]
    assert pick_permission_option("reject_once", allow_only).outcome.outcome == "cancelled"


class _Bridge:
    def __init__(self, decision: Any = "allow_once", exc: Exception | None = None) -> None:
        self.decision = decision
        self.exc = exc
        self.requests: list[PermissionRequest] = []

    async def decide(self, request: PermissionRequest) -> Any:
        self.requests.append(request)
        if self.exc:
            raise self.exc
        return self.decision


def _client(**kw: Any) -> AgentOSAcpClient:
    return AgentOSAcpClient(agent_id="claude", cwd="/w", session_key="agent:main:acp:1", **kw)


async def test_request_permission_builds_request_and_emits_status() -> None:
    bridge = _Bridge("allow_once")
    client = _client(permission_bridge=bridge)
    events: list[AcpEvent] = []
    client.attach(events.append)
    await client.session_update(
        "s1", acp.start_tool_call("t1", "Write x", kind="edit", raw_input={"path": "x"})
    )
    resp = await client.request_permission("s1", ToolCallUpdate(tool_call_id="t1"), OPTIONS)
    assert resp.outcome.option_id == "a1"
    req = bridge.requests[0]
    assert req.session_key == "agent:main:acp:1" and req.acp_session_id == "s1"
    assert req.title == "Write x" and req.tool_kind == "edit"
    assert req.raw_input == {"path": "x"} and req.cwd == "/w" and req.agent_id == "claude"
    assert [o.kind for o in req.options] == ["allow_once", "allow_always", "reject_once"]
    assert events[-1].tag == "permission_request" and events[-1].raw["decision"] == "allow_once"


async def test_request_permission_without_bridge_rejects() -> None:
    resp = await _client().request_permission(
        "s1", ToolCallUpdate(tool_call_id="t1", title="x"), OPTIONS
    )
    assert resp.outcome.option_id == "r1"


async def test_request_permission_bridge_error_rejects() -> None:
    client = _client(permission_bridge=_Bridge(exc=RuntimeError("boom")))
    resp = await client.request_permission("s1", ToolCallUpdate(tool_call_id="t1"), OPTIONS)
    assert resp.outcome.option_id == "r1"
    assert client.turn_failure is None


async def test_request_permission_unavailable_fails_the_turn() -> None:
    client = _client(permission_bridge=_Bridge(exc=PermissionPromptUnavailableError("no ui")))
    failures: list[BaseException] = []
    client.on_turn_failure = failures.append
    resp = await client.request_permission("s1", ToolCallUpdate(tool_call_id="t1"), OPTIONS)
    assert resp.outcome.outcome == "cancelled"
    assert isinstance(client.turn_failure, PermissionPromptUnavailableError)
    assert failures and failures[0] is client.turn_failure


class _Elicit:
    def __init__(self, response: ElicitationResponse) -> None:
        self.response = response
        self.requests: list[ElicitationRequest] = []

    async def ask(self, request: ElicitationRequest) -> ElicitationResponse:
        self.requests.append(request)
        return self.response


async def test_elicitation_form_and_url() -> None:
    bridge = _Elicit(ElicitationResponse(action="accept", content={"color": "red"}))
    client = _client(elicitation_bridge=bridge)
    form = ElicitationFormSessionMode(
        session_id="s1",
        requested_schema=ElicitationSchema(properties={"color": {"type": "string"}}),
    )
    resp = await client.create_elicitation("Pick", form)
    assert resp.action == "accept" and resp.content == {"color": "red"}
    req = bridge.requests[0]
    assert req.mode == "form" and req.schema and "color" in req.schema["properties"]
    assert req.session_key == "agent:main:acp:1" and req.acp_session_id == "s1"

    bridge.response = ElicitationResponse(action="decline")
    url = ElicitationUrlSessionMode(
        session_id="s1", elicitation_id="e1", url="https://example.com/login"
    )
    resp = await client.create_elicitation("Login", url)
    assert resp.action == "decline"
    assert (
        bridge.requests[1].mode == "url" and bridge.requests[1].url == "https://example.com/login"
    )

    bridge.response = ElicitationResponse(action="cancel")
    assert (await client.create_elicitation("x", form)).action == "cancel"


async def test_elicitation_without_bridge_declines() -> None:
    form = ElicitationFormSessionMode(session_id="s1", requested_schema=ElicitationSchema())
    assert (await _client().create_elicitation("x", form)).action == "decline"


async def test_fs_and_terminal_methods_are_refused() -> None:
    client = _client()
    calls = [
        client.read_text_file("s", "/etc/passwd"),
        client.write_text_file("s", "/tmp/x", "y"),
        client.create_terminal("s", "ls"),
        client.terminal_output("s", "t"),
        client.release_terminal("s", "t"),
        client.wait_for_terminal_exit("s", "t"),
        client.kill_terminal("s", "t"),
    ]
    for coro in calls:
        with pytest.raises(RequestError) as ei:
            await coro
        assert ei.value.code == -32601


async def test_updates_outside_a_turn_only_touch_state() -> None:
    client = _client()
    await client.session_update("s1", acp.update_agent_message_text("replay"))
    await client.session_update(
        "s1", CurrentModeUpdate(session_update="current_mode_update", current_mode_id="plan")
    )
    await client.session_update(
        "s1",
        UsageUpdate(
            session_update="usage_update", used=3, size=9, cost=Cost(amount=0.5, currency="USD")
        ),
    )
    assert client.current_mode_id == "plan"
    assert client.last_cost_usd == 0.5 and client.last_usage == {"used": 3, "size": 9}
    events: list[AcpEvent] = []
    client.attach(events.append)
    await client.session_update("s1", acp.update_agent_message_text("live"))
    client.detach()
    await client.session_update("s1", acp.update_agent_message_text("after"))
    assert [e.text for e in events] == ["live"]
