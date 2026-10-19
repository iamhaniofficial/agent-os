"""``gateway.restart``: audiences, the chat toggle, origins, outcomes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agentos.gateway.access import CHANNEL_RPC_METHODS, CONTROL_AND_CHANNEL, ConnectionSurface
from agentos.gateway.auth import AccessContext
from agentos.gateway.config import GatewayConfig
from agentos.gateway.restart import (
    RestartOrigin,
    RestartPendingError,
    RestartRequest,
    RestartUnsupportedError,
    set_restarter,
)
from agentos.gateway.routing import RouteEnvelope, SourceKind
from agentos.gateway.rpc import RpcContext, get_dispatcher


class _FakeRestarter:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.pending: RestartRequest | None = None
        self.error = error

    def request(self, *, reason: str, mode: str, origin: RestartOrigin) -> RestartRequest:
        self.calls.append({"reason": reason, "mode": mode, "origin": origin})
        if self.error is not None:
            raise self.error
        req = RestartRequest(restart_id="r-1", reason=reason, mode=mode, origin=origin)
        self.pending = req
        return req


@pytest.fixture
def restarter() -> Any:
    fake = _FakeRestarter()
    set_restarter(fake)  # type: ignore[arg-type]
    yield fake
    set_restarter(None)


def _envelope() -> RouteEnvelope:
    return RouteEnvelope(
        source_kind=SourceKind.CHANNEL,
        source_name="telegram-main",
        agent_id="main",
        session_key="agent:main:telegram:1",
        channel_name="telegram-main",
        channel_type="telegram",
        channel_id="1",
        thread_id="9",
        sender_id="u1",
    )


def _ctx(
    surface: ConnectionSurface,
    *,
    chat_restart: bool = False,
    agent: Any = None,
    envelope: RouteEnvelope | None = None,
) -> RpcContext:
    return RpcContext(
        conn_id="conn-1",
        access=AccessContext(surface=surface, admitted=True, credential_verified=True, agent=agent),
        config=GatewayConfig(gateway_control={"chat_restart": chat_restart}),
        originating_envelope=envelope,
    )


async def _call(ctx: RpcContext, params: dict[str, Any] | None = None) -> Any:
    return await get_dispatcher().dispatch("req-1", "gateway.restart", params, ctx)


def test_registered_for_control_and_channel() -> None:
    entry = get_dispatcher().get_entry("gateway.restart")
    assert entry is not None
    assert entry.audiences == CONTROL_AND_CHANNEL
    assert "gateway.restart" in CHANNEL_RPC_METHODS


@pytest.mark.asyncio
async def test_channel_caller_is_refused_while_the_toggle_is_off(restarter: Any) -> None:
    res = await _call(_ctx(ConnectionSurface.CHANNEL, envelope=_envelope()), {"key": "k"})

    assert res.ok is False
    assert res.error is not None
    assert res.error.code == "UNAUTHORIZED"
    assert "chat_restart" in res.error.message
    assert restarter.calls == []


@pytest.mark.asyncio
async def test_channel_caller_is_accepted_with_the_toggle_on(restarter: Any) -> None:
    ctx = _ctx(ConnectionSurface.CHANNEL, chat_restart=True, envelope=_envelope())

    res = await _call(ctx, {"key": "agent:main:telegram:1", "mode": "safe", "reason": "slash"})

    assert res.ok is True
    assert res.payload["accepted"] is True
    assert res.payload["restartId"] == "r-1"
    assert res.payload["mode"] == "safe"
    assert res.payload["drainTimeoutSeconds"] == 30.0
    (call,) = restarter.calls
    assert call["reason"] == "slash"
    origin = call["origin"]
    assert origin.kind == "channel"
    assert origin.channel_name == "telegram-main"
    assert origin.channel_id == "1"
    assert origin.thread_id == "9"


@pytest.mark.asyncio
async def test_control_caller_needs_no_toggle(restarter: Any) -> None:
    res = await _call(_ctx(ConnectionSurface.CONTROL), {"key": "agent:main:webchat:x"})

    assert res.ok is True
    (call,) = restarter.calls
    assert call["reason"] == "rpc"
    assert call["origin"] == RestartOrigin(
        kind="web", session_key="agent:main:webchat:x", conn_id="conn-1"
    )


@pytest.mark.asyncio
async def test_control_caller_without_a_key_has_no_origin(restarter: Any) -> None:
    res = await _call(_ctx(ConnectionSurface.CONTROL), {})
    assert res.ok is True
    assert restarter.calls[0]["origin"].kind == "none"


@pytest.mark.asyncio
async def test_agent_bound_control_connection_is_a_chat_caller(restarter: Any) -> None:
    binding = SimpleNamespace(session_key="agent:main:telegram:1", via="token")

    denied = await _call(_ctx(ConnectionSurface.CONTROL, agent=binding), {})
    assert denied.ok is False and denied.error.code == "UNAUTHORIZED"

    allowed = await _call(_ctx(ConnectionSurface.CONTROL, agent=binding, chat_restart=True), {})
    assert allowed.ok is True


@pytest.mark.asyncio
async def test_force_mode_is_passed_through(restarter: Any) -> None:
    res = await _call(_ctx(ConnectionSurface.CONTROL), {"mode": "force"})
    assert res.payload["mode"] == "force"
    assert res.payload["drainTimeoutSeconds"] == 2.0


@pytest.mark.asyncio
async def test_bad_mode_is_an_invalid_request(restarter: Any) -> None:
    res = await _call(_ctx(ConnectionSurface.CONTROL), {"mode": "hard"})
    assert res.ok is False and res.error.code == "INVALID_REQUEST"
    assert restarter.calls == []


@pytest.mark.asyncio
async def test_bad_key_is_an_invalid_request(restarter: Any) -> None:
    res = await _call(_ctx(ConnectionSurface.CONTROL), {"key": 42})
    assert res.ok is False and res.error.code == "INVALID_REQUEST"


@pytest.mark.asyncio
async def test_a_pending_restart_is_reported_not_duplicated() -> None:
    fake = _FakeRestarter(error=RestartPendingError("busy"))
    fake.pending = RestartRequest(restart_id="first", reason="cli", mode="force")
    set_restarter(fake)  # type: ignore[arg-type]
    try:
        res = await _call(_ctx(ConnectionSurface.CONTROL), {})
    finally:
        set_restarter(None)

    assert res.ok is True
    assert res.payload == {
        "accepted": False,
        "alreadyPending": True,
        "restartId": "first",
        "mode": "force",
    }


@pytest.mark.asyncio
async def test_unsupported_platform_is_a_named_error() -> None:
    set_restarter(_FakeRestarter(error=RestartUnsupportedError("not on Windows")))  # type: ignore[arg-type]
    try:
        res = await _call(_ctx(ConnectionSurface.CONTROL), {})
    finally:
        set_restarter(None)

    assert res.ok is False
    assert res.error.code == "RESTART_UNSUPPORTED"
    assert "Windows" in res.error.message


@pytest.mark.asyncio
async def test_no_restarter_wired_is_unavailable() -> None:
    set_restarter(None)
    res = await _call(_ctx(ConnectionSurface.CONTROL), {})
    assert res.ok is False
    assert res.error.code == "UNAVAILABLE"
