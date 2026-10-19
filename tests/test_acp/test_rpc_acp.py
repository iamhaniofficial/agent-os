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

    async def doctor(self, agent_id: str, *, probe: bool = False) -> dict[str, Any]:
        return {"probe": "ok", "agent_id": agent_id}


def _manager() -> FakeManager:
    return FakeManager(cancelled=[], closed=[])


def test_methods_are_registered_control_only() -> None:
    from agentos.gateway.access import ConnectionSurface

    d = get_dispatcher()
    for name in (
        "acp.agents",
        "acp.doctor",
        "acp.sessions",
        "acp.status",
        "acp.cancel",
        "acp.close",
    ):
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


@pytest.mark.parametrize("method", ["acp.sessions", "acp.status", "acp.cancel", "acp.close"])
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
    assert closed == {"session_key": "agent:main:acp:1", "closed": True, "was_live": True}
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


# --------------------------------------------------------------------------
# acp.status / acp.sessions against a real manager and session store
# --------------------------------------------------------------------------


@pytest.fixture
async def real_sessions():
    from agentos.session.manager import SessionManager
    from agentos.session.storage import SessionStorage

    store = SessionStorage(":memory:")
    await store.connect()
    yield SessionManager(store, inject_time_prefix=False)
    await store.close()


def _real_manager(sessions, runtime=None):
    from acp_s_fakes import FakeRuntime

    from agentos.acp.manager import AcpSessionManager

    return AcpSessionManager(runtime or FakeRuntime(), config=AcpConfig(), session_manager=sessions)


async def test_status_answers_for_an_idle_oneshot_session(real_sessions) -> None:
    key = "agent:main:acp:0a0a0a0a"
    await real_sessions.create(
        key,
        agent_id="main",
        origin={
            "kind": "acp",
            "agent_id": "claude",
            "cwd": "/tmp",
            "mode": "oneshot",
            "harness_mode": "plan",
        },
    )
    manager = _real_manager(real_sessions)
    rpc_acp.set_acp_manager(manager)
    await manager.run_turn(key, "hello")

    res = await get_dispatcher().dispatch("1", "acp.status", {"session_key": key}, _ctx())

    assert res.ok, res.error
    status = res.payload
    assert status["session_key"] == key
    assert status["agent_id"] == "claude" and status["agent_name"] == "Claude Code"
    assert status["idle"] is True and status["closed"] is False and status["live"] is False
    assert status["mode"] == "oneshot" and status["harness_mode"] == "plan"
    assert status["cwd"] == "/tmp"
    assert status["acp_session_id"] == "sess-1"
    assert status["activity"] == "Writing the answer" and status["activity_kind"] == "text"
    assert status["turn_tool_calls"] == 0
    assert "last_turn" in status


async def test_status_reports_closed_sessions(real_sessions) -> None:
    key = "agent:main:acp:0b0b0b0b"
    await real_sessions.create(
        key,
        agent_id="main",
        origin={"kind": "acp", "agent_id": "codex", "cwd": "/tmp", "mode": "persistent"},
    )
    manager = _real_manager(real_sessions)
    rpc_acp.set_acp_manager(manager)
    await manager.run_turn(key, "hello")
    sessions = await rpc_acp._handle_acp_sessions({}, _ctx())
    row = sessions["sessions"][0]
    assert row["agent_name"] == "Codex" and row["turn_tool_calls"] == 0
    assert {"activity", "activity_kind"} <= set(row)

    closed = await rpc_acp._handle_acp_close({"session_key": key}, _ctx())
    assert closed == {"session_key": key, "closed": True, "was_live": True}
    status = await rpc_acp._handle_acp_status({"session_key": key}, _ctx())
    assert status["closed"] is True and status["idle"] is False and status["live"] is False
    assert (await rpc_acp._handle_acp_sessions({}, _ctx()))["count"] == 0


async def test_status_rejects_non_acp_sessions(real_sessions) -> None:
    await real_sessions.create("agent:main:main", agent_id="main")
    rpc_acp.set_acp_manager(_real_manager(real_sessions))

    res = await get_dispatcher().dispatch(
        "1", "acp.status", {"session_key": "agent:main:main"}, _ctx()
    )

    assert res.error is not None and res.error.code == "ACP_SESSION_INVALID"


async def test_status_is_unavailable_when_acp_is_off(real_sessions) -> None:
    rpc_acp.set_acp_manager(_real_manager(real_sessions))
    ctx = RpcContext(conn_id="t", config=GatewayConfig(acp=AcpConfig(enabled=False)))

    res = await get_dispatcher().dispatch(
        "1", "acp.status", {"session_key": "agent:main:acp:1"}, ctx
    )

    assert res.error is not None and res.error.code == "ACP_UNAVAILABLE"


async def test_cancel_and_close_also_cancel_queued_turns() -> None:
    mgr = _manager()
    rpc_acp.set_acp_manager(mgr)
    calls: list[tuple[str, str]] = []

    class Runtime:
        async def cancel(self, *, session_key: str, source: str, reason: str) -> int:
            calls.append((session_key, source))
            return 1

    await rpc_acp._handle_acp_cancel(
        {"session_key": "agent:main:acp:1"}, _ctx(task_runtime=Runtime())
    )
    await rpc_acp._handle_acp_close(
        {"session_key": "agent:main:acp:1"}, _ctx(task_runtime=Runtime())
    )

    assert calls == [
        ("agent:main:acp:1", "acp_rpc_cancel"),
        ("agent:main:acp:1", "acp_rpc_close"),
    ]


async def test_status_always_carries_session_key_and_agent_id(real_sessions) -> None:
    key = "agent:main:acp:0c0c0c0c"
    await real_sessions.create(
        key, agent_id="main", origin={"kind": "acp", "agent_id": "gemini", "cwd": "/tmp"}
    )
    rpc_acp.set_acp_manager(_real_manager(real_sessions))

    status = await rpc_acp._handle_acp_status({"session_key": key}, _ctx())

    assert status["session_key"] == key and status["agent_id"] == "gemini"
    assert status["agent_name"] == "Gemini CLI"
    for field in (
        "activity",
        "activity_kind",
        "turn_tool_calls",
        "idle",
        "closed",
        "harness_mode",
        "cwd",
        "mode",
        "last_turn",
    ):
        assert field in status, field
