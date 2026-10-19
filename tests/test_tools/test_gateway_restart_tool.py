"""The ``gateway`` tool's ``restart`` action: gated by the chat toggle, then
hands the request to the process-wide restarter with the turn's chat as origin.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agentos.gateway.config import GatewayConfig
from agentos.gateway.restart import (
    RestartOrigin,
    RestartPendingError,
    RestartRequest,
    RestartUnsupportedError,
    set_restarter,
)
from agentos.tools.builtin import control
from agentos.tools.builtin.control import gateway, set_gateway_config
from agentos.tools.types import SafeToolError, ToolError, current_tool_context


class _FakeRestarter:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.pending: RestartRequest | None = None
        self.error = error

    def request(self, *, reason: str, mode: str, origin: RestartOrigin) -> RestartRequest:
        self.calls.append({"reason": reason, "mode": mode, "origin": origin})
        if self.error is not None:
            raise self.error
        req = RestartRequest(restart_id="tool-1", reason=reason, mode=mode, origin=origin)
        self.pending = req
        return req


@pytest.fixture
def config_on() -> Any:
    previous = control._gateway_config
    set_gateway_config(GatewayConfig(gateway_control={"chat_restart": True}))
    yield
    set_gateway_config(previous)


@pytest.fixture
def config_off() -> Any:
    previous = control._gateway_config
    set_gateway_config(GatewayConfig())
    yield
    set_gateway_config(previous)


@pytest.fixture
def restarter() -> Any:
    fake = _FakeRestarter()
    set_restarter(fake)  # type: ignore[arg-type]
    yield fake
    set_restarter(None)


@pytest.fixture
def channel_turn() -> Any:
    token = current_tool_context.set(
        SimpleNamespace(
            channel_kind="telegram-main",
            channel_id="-100555",
            session_key="agent:main:telegram:-100555",
            source_kind="channel",
            sender_id="u7",
        )  # type: ignore[arg-type]
    )
    yield
    current_tool_context.reset(token)


@pytest.mark.asyncio
async def test_toggle_off_refuses_with_an_actionable_message(
    config_off: Any, restarter: Any
) -> None:
    with pytest.raises(SafeToolError) as exc:
        await gateway(action="restart")

    assert "[gateway_control] chat_restart = true" in exc.value.user_message
    assert restarter.calls == []


@pytest.mark.asyncio
async def test_toggle_on_schedules_a_safe_restart_from_this_chat(
    config_on: Any, restarter: Any, channel_turn: Any
) -> None:
    payload = json.loads(await gateway(action="restart"))

    assert payload["action"] == "restart"
    assert payload["accepted"] is True
    assert payload["restartId"] == "tool-1"
    assert payload["mode"] == "safe"
    assert payload["drainTimeoutSeconds"] == 30.0
    assert "confirmation is posted to this chat" in payload["note"]
    (call,) = restarter.calls
    assert call["reason"] == "agent_tool"
    assert call["origin"].kind == "channel"
    assert call["origin"].channel_name == "telegram-main"
    assert call["origin"].channel_id == "-100555"


@pytest.mark.asyncio
async def test_force_mode_is_passed_through(config_on: Any, restarter: Any) -> None:
    payload = json.loads(await gateway(action="restart", mode="force"))
    assert payload["mode"] == "force"
    assert restarter.calls[0]["mode"] == "force"
    assert "cancelled" in payload["note"]


@pytest.mark.asyncio
async def test_bad_mode_is_a_tool_error(config_on: Any, restarter: Any) -> None:
    with pytest.raises(ToolError, match="mode must be one of"):
        await gateway(action="restart", mode="hard")
    assert restarter.calls == []


@pytest.mark.asyncio
async def test_pending_restart_is_reported(config_on: Any) -> None:
    fake = _FakeRestarter(error=RestartPendingError("busy"))
    fake.pending = RestartRequest(restart_id="first", reason="cli", mode="safe")
    set_restarter(fake)  # type: ignore[arg-type]
    try:
        payload = json.loads(await gateway(action="restart"))
    finally:
        set_restarter(None)

    assert payload == {
        "action": "restart",
        "accepted": False,
        "alreadyPending": True,
        "restartId": "first",
    }


@pytest.mark.asyncio
async def test_unsupported_platform_is_a_safe_error(config_on: Any) -> None:
    set_restarter(_FakeRestarter(error=RestartUnsupportedError("not on Windows")))  # type: ignore[arg-type]
    try:
        with pytest.raises(SafeToolError, match="Windows"):
            await gateway(action="restart")
    finally:
        set_restarter(None)


@pytest.mark.asyncio
async def test_no_restarter_is_a_safe_error(config_on: Any) -> None:
    set_restarter(None)
    with pytest.raises(SafeToolError, match="unavailable"):
        await gateway(action="restart")
