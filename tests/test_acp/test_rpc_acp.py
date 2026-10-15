"""``acp.*`` gateway RPC: registry views, manager delegation, ACP_UNAVAILABLE."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from agentos.acp.types import AcpSessionHandle, AcpTurnError
from agentos.gateway import rpc_acp
from agentos.gateway.config import AcpAgentOverride, AcpConfig, GatewayConfig
from agentos.gateway.rpc import get_dispatcher
from agentos.gateway.rpc.registry import RpcContext

ON_PATH = {"npx", "claude", "codex"}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr("shutil.which", lambda c: f"/bin/{c}" if c in ON_PATH else None)
    rpc_acp.set_acp_manager(None)
    yield
    rpc_acp.set_acp_manager(None)


def _ctx(**kwargs: Any) -> RpcContext:
    config = GatewayConfig(
        acp=AcpConfig(
            default_agent="codex",
            allowed_agents=["codex"],
            agents={"local": AcpAgentOverride(command="local-acp", args=["serve"])},
        )
    )
    return RpcContext(conn_id="t", config=config, **kwargs)


@dataclass
class FakeManager:
    cancelled: list[tuple[str, str]]
    closed: list[tuple[str, bool]]

    async def list_sessions(self) -> list[Any]:
        return [
            AcpSessionHandle(
                record_id="r1", agent_id="claude", cwd="/w", argv=["x"], argv_hash="h"
            ),
            {"session_key": "agent:main:acp:deadbeef", "agent_id": "codex"},
        ]

    async def cancel(self, session_key: str, *, reason: str = "") -> None:
        if session_key == "agent:main:acp:boom":
            raise AcpTurnError("harness gone", code="ACP_TURN_FAILED")
        self.cancelled.append((session_key, reason))

    async def close(self, session_key: str) -> bool:
        self.closed.append((session_key, False))
        return True

    async def doctor(self, agent_id: str) -> dict[str, Any]:
        return {"probe": "ok", "agent_id": agent_id}


def _manager() -> FakeManager:
    return FakeManager(cancelled=[], closed=[])


def test_methods_are_registered_control_only() -> None:
    from agentos.gateway.access import ConnectionSurface

    d = get_dispatcher()
    for name in ("acp.agents", "acp.doctor", "acp.sessions", "acp.cancel", "acp.close"):
        entry = d.get_entry(name)
        assert entry is not None, name
        assert entry.audiences == frozenset({ConnectionSurface.CONTROL})


async def test_agents_lists_registry_with_config() -> None:
    payload = await rpc_acp._handle_acp_agents(None, _ctx())

    rows = {r["agent_id"]: r for r in payload["agents"]}
    assert payload["default_agent"] == "codex"
    assert payload["manager"] is False
    assert rows["codex"]["default"] is True
    assert rows["claude"]["allowed"] is False
    assert rows["local"]["argv"] == ["local-acp", "serve"]
    assert rows["local"]["source"] == "override"


async def test_doctor_without_probe_needs_no_manager() -> None:
    payload = await rpc_acp._handle_acp_doctor({"agent": "gemini"}, _ctx())

    row = payload["agents"][0]
    assert row["ok"] is False
    assert row["missing_commands"] == ["gemini"]


async def test_doctor_unknown_agent_is_not_found() -> None:
    with pytest.raises(KeyError):
        await rpc_acp._handle_acp_doctor({"agent": "nope"}, _ctx())


async def test_probe_without_manager_is_unavailable() -> None:
    with pytest.raises(rpc_acp.RpcHandlerError) as err:
        await rpc_acp._handle_acp_doctor({"agent": "codex", "probe": True}, _ctx())

    assert err.value.code == "ACP_UNAVAILABLE"


async def test_probe_delegates_to_manager() -> None:
    rpc_acp.set_acp_manager(_manager())

    payload = await rpc_acp._handle_acp_doctor({"agent": "codex", "probe": True}, _ctx())

    assert payload["agents"][0]["probe"] == {"probe": "ok", "agent_id": "codex"}


@pytest.mark.parametrize("method", ["acp.sessions", "acp.cancel", "acp.close"])
async def test_manager_methods_report_unavailable_via_dispatch(method: str) -> None:
    res = await get_dispatcher().dispatch("1", method, {"session_key": "agent:main:acp:1"}, _ctx())

    assert res.ok is False
    assert res.error is not None
    assert res.error.code == "ACP_UNAVAILABLE"


async def test_sessions_uses_manager_and_plains_handles() -> None:
    rpc_acp.set_acp_manager(_manager())

    payload = await rpc_acp._handle_acp_sessions({}, _ctx())

    assert payload["count"] == 2
    assert payload["sessions"][0]["record_id"] == "r1"
    assert payload["sessions"][1]["session_key"] == "agent:main:acp:deadbeef"


async def test_manager_found_on_context_services() -> None:
    class Runtime:
        acp_manager = _manager()

    payload = await rpc_acp._handle_acp_sessions({}, _ctx(task_runtime=Runtime()))

    assert payload["count"] == 2


async def test_cancel_and_close_delegate() -> None:
    mgr = _manager()
    rpc_acp.set_acp_manager(mgr)

    cancelled = await rpc_acp._handle_acp_cancel({"session_key": "agent:main:acp:1"}, _ctx())
    closed = await rpc_acp._handle_acp_close(
        {"session_key": "agent:main:acp:1", "discard": True}, _ctx()
    )

    assert cancelled == {"session_key": "agent:main:acp:1", "cancelled": True}
    assert closed == {"session_key": "agent:main:acp:1", "closed": True}
    assert mgr.cancelled == [("agent:main:acp:1", "operator")]
    # close() here takes no ``discard`` keyword; it must not be passed.
    assert mgr.closed == [("agent:main:acp:1", False)]


async def test_cancel_requires_session_key() -> None:
    rpc_acp.set_acp_manager(_manager())

    res = await get_dispatcher().dispatch("1", "acp.cancel", {}, _ctx())

    assert res.error is not None
    assert res.error.code == "INVALID_REQUEST"


async def test_acp_errors_keep_their_code() -> None:
    rpc_acp.set_acp_manager(_manager())

    res = await get_dispatcher().dispatch(
        "1", "acp.cancel", {"session_key": "agent:main:acp:boom"}, _ctx()
    )

    assert res.error is not None
    assert res.error.code == "ACP_TURN_FAILED"
