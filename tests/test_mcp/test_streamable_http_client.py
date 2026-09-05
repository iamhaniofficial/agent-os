from __future__ import annotations

import asyncio
import os
import stat
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest

from agentos.mcp.discovery import create_client
from agentos.mcp.streamable_http import FileOAuthStorage, MCPStreamableHTTPClient
from agentos.mcp.types import MCPServerConfig


def test_factory_builds_streamable_http_client() -> None:
    config = MCPServerConfig(
        name="remote",
        transport="streamable_http",
        url="https://example.test/mcp",
    )

    assert isinstance(create_client(config), MCPStreamableHTTPClient)


@pytest.mark.asyncio
async def test_oauth_storage_round_trips_privately(tmp_path) -> None:
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    storage = FileOAuthStorage(
        "Robinhood Trading",
        "https://agent.robinhood.com/mcp/trading",
        str(tmp_path),
    )
    tokens = OAuthToken(access_token="access-secret", refresh_token="refresh-secret")
    client_info = OAuthClientInformationFull.model_validate(
        {
            "redirect_uris": ["http://127.0.0.1/control/mcp/oauth/callback"],
            "client_id": "client-id",
            "token_endpoint_auth_method": "none",
        }
    )

    await storage.set_tokens(tokens)
    await storage.set_client_info(client_info)

    restored = FileOAuthStorage(
        "Robinhood Trading",
        "https://agent.robinhood.com/mcp/trading",
        str(tmp_path),
    )
    assert (await restored.get_tokens()).access_token == "access-secret"
    assert (await restored.get_client_info()).client_id == "client-id"
    # Windows does not expose POSIX owner-only mode bits through stat/chmod;
    # credential files inherit the current user's state-directory ACL instead.
    if os.name != "nt":
        assert stat.S_IMODE(restored.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(restored.path.parent.stat().st_mode) == 0o700
    assert await restored.is_authenticated() is True

    restored.clear()
    assert not restored.path.exists()


@pytest.mark.asyncio
async def test_cancelled_connect_closes_all_partial_transport_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.client import session as session_module
    from mcp.client import streamable_http as transport_module

    closed: list[str] = []

    @asynccontextmanager
    async def fake_http_client(**_kwargs: Any):
        try:
            yield object()
        finally:
            closed.append("http")

    @asynccontextmanager
    async def fake_transport(_url: str, **_kwargs: Any):
        try:
            yield object(), object(), None
        finally:
            closed.append("transport")

    class FakeSession:
        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            closed.append("session")

        async def initialize(self) -> None:
            raise asyncio.CancelledError

    monkeypatch.setattr("agentos.mcp.streamable_http.httpx.AsyncClient", fake_http_client)
    monkeypatch.setattr(transport_module, "streamable_http_client", fake_transport)
    monkeypatch.setattr(session_module, "ClientSession", lambda *_args, **_kwargs: FakeSession())
    client = MCPStreamableHTTPClient(
        MCPServerConfig(
            name="remote",
            transport="streamable_http",
            url="https://example.test/mcp",
        )
    )

    with pytest.raises(asyncio.CancelledError):
        await client.connect()

    assert closed == ["session", "transport", "http"]


@pytest.mark.asyncio
async def test_calls_before_connect_name_this_transport() -> None:
    """Both SDK-backed transports share one base; the error still says which one."""
    client = MCPStreamableHTTPClient(
        MCPServerConfig(
            name="remote",
            transport="streamable_http",
            url="https://example.test/mcp",
        )
    )
    with pytest.raises(RuntimeError, match="MCP Streamable HTTP client is not connected"):
        await client.list_tools()
    with pytest.raises(RuntimeError, match="MCP Streamable HTTP client is not connected"):
        await client.call_tool("anything", {})


@pytest.mark.asyncio
async def test_close_works_from_a_different_task_than_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same task-affinity requirement as the SSE transport; same shared base."""
    from mcp.client import session as session_module
    from mcp.client import streamable_http as transport_module

    closed: list[str] = []

    @asynccontextmanager
    async def fake_transport(_url: str, **_kwargs: Any):
        async with anyio.create_task_group():
            try:
                yield object(), object(), None
            finally:
                closed.append("transport")

    class FakeSession:
        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            closed.append("session")

        async def initialize(self) -> None:
            return None

    monkeypatch.setattr(transport_module, "streamable_http_client", fake_transport)
    monkeypatch.setattr(session_module, "ClientSession", lambda *_a, **_k: FakeSession())

    client = MCPStreamableHTTPClient(
        MCPServerConfig(
            name="remote",
            transport="streamable_http",
            url="https://example.test/mcp",
        )
    )
    await asyncio.create_task(client.connect())
    await asyncio.create_task(client.close())

    assert closed == ["session", "transport"]
