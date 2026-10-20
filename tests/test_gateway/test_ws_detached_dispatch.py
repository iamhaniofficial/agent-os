"""A slow RPC must not hold back the next request on the same WebSocket.

At every turn end the renderer asks ``chat.suggestNext`` (an auxiliary model
call, ~5 s); ``chat.history``, ``sessions.list`` and ``acp.status`` sent on
the same socket right after it were answered only when it returned.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.websockets import WebSocketDisconnect, WebSocketState

from agentos.gateway.access import ConnectionSurface
from agentos.gateway.auth import AccessContext
from agentos.gateway.config import GatewayConfig
from agentos.gateway.rpc.registry import RpcRegistry
from agentos.gateway.websocket import DETACHED_METHODS, WsConnection, _message_loop


class _FakeWs:
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[str | None] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.got_sent = asyncio.Event()

    async def receive_text(self) -> str:
        raw = await self.inbox.get()
        if raw is None:
            raise WebSocketDisconnect()
        return raw

    async def send_text(self, raw: str) -> None:
        self.sent.append(json.loads(raw))
        self.got_sent.set()

    async def close(self, code: int = 1000) -> None:
        return None


def _req(req_id: str, method: str) -> str:
    return json.dumps({"type": "req", "id": req_id, "method": method, "params": {}})


async def _answered(ws: _FakeWs, req_id: str) -> None:
    while not any(frame.get("id") == req_id for frame in ws.sent):
        ws.got_sent.clear()
        await ws.got_sent.wait()


async def test_slow_suggest_next_does_not_delay_a_fast_request() -> None:
    assert "chat.suggestNext" in DETACHED_METHODS
    registry = RpcRegistry()
    release = asyncio.Event()
    order: list[str] = []

    @registry.method("chat.suggestNext")
    async def _slow(params: dict | None, ctx: Any) -> dict:
        await release.wait()
        order.append("suggest")
        return {"text": "next"}

    @registry.method("acp.status")
    async def _fast(params: dict | None, ctx: Any) -> dict:
        order.append("status")
        return {"ok": True}

    ws = _FakeWs()
    conn = WsConnection(
        conn_id="c1",
        ws=ws,  # type: ignore[arg-type]
        access=AccessContext(
            surface=ConnectionSurface.CONTROL, admitted=True, credential_verified=True
        ),
    )
    loop = asyncio.create_task(_message_loop(conn, GatewayConfig(), registry, None))
    await ws.inbox.put(_req("1", "chat.suggestNext"))
    await ws.inbox.put(_req("2", "acp.status"))

    await asyncio.wait_for(_answered(ws, "2"), timeout=2.0)
    assert order == ["status"]  # answered while the suggestion is still pending
    release.set()
    await asyncio.wait_for(_answered(ws, "1"), timeout=2.0)
    assert [frame["id"] for frame in ws.sent] == ["2", "1"]
    assert ws.sent[1]["payload"] == {"text": "next"}

    await ws.inbox.put(None)
    await asyncio.wait_for(loop, timeout=2.0)


async def test_other_requests_keep_their_order() -> None:
    registry = RpcRegistry()
    release = asyncio.Event()

    @registry.method("chat.send")
    async def _send(params: dict | None, ctx: Any) -> dict:
        await release.wait()
        return {"ok": "send"}

    @registry.method("sessions.abort")
    async def _abort(params: dict | None, ctx: Any) -> dict:
        return {"ok": "abort"}

    ws = _FakeWs()
    conn = WsConnection(
        conn_id="c2",
        ws=ws,  # type: ignore[arg-type]
        access=AccessContext(
            surface=ConnectionSurface.CONTROL, admitted=True, credential_verified=True
        ),
    )
    loop = asyncio.create_task(_message_loop(conn, GatewayConfig(), registry, None))
    await ws.inbox.put(_req("1", "chat.send"))
    await ws.inbox.put(_req("2", "sessions.abort"))
    await asyncio.sleep(0.05)
    assert ws.sent == []  # abort waits for the send, as clients expect
    release.set()
    await asyncio.wait_for(_answered(ws, "2"), timeout=2.0)
    assert [frame["id"] for frame in ws.sent] == ["1", "2"]

    await ws.inbox.put(None)
    await asyncio.wait_for(loop, timeout=2.0)


async def test_pending_detached_requests_are_cancelled_on_disconnect() -> None:
    registry = RpcRegistry()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @registry.method("chat.suggestNext")
    async def _slow(params: dict | None, ctx: Any) -> dict:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {}

    ws = _FakeWs()
    conn = WsConnection(
        conn_id="c3",
        ws=ws,  # type: ignore[arg-type]
        access=AccessContext(
            surface=ConnectionSurface.CONTROL, admitted=True, credential_verified=True
        ),
    )
    loop = asyncio.create_task(_message_loop(conn, GatewayConfig(), registry, None))
    await ws.inbox.put(_req("1", "chat.suggestNext"))
    await asyncio.wait_for(started.wait(), timeout=2.0)
    await ws.inbox.put(None)
    await asyncio.wait_for(loop, timeout=2.0)
    await asyncio.wait_for(cancelled.wait(), timeout=2.0)
