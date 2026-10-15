"""AcpSessionManager against an in-memory runtime fake and a real SessionManager."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
from acp_s_fakes import FakeRuntime, Script, text_script

from agentos.acp.manager import (
    AcpOptionUnsupportedError,
    AcpSessionClosedError,
    AcpSessionManager,
    AcpUnavailableError,
    spec_from_origin,
)
from agentos.acp.types import (
    AcpError,
    AcpEvent,
    AcpTurnError,
    AcpTurnResult,
    AcpTurnTimeoutError,
)
from agentos.gateway.config import AcpConfig
from agentos.session.manager import SessionManager
from agentos.session.storage import SessionStorage


@pytest_asyncio.fixture
async def sessions():
    store = SessionStorage(":memory:")
    await store.connect()
    mgr = SessionManager(store, inject_time_prefix=False)
    yield mgr
    await store.close()


async def _acp_row(sessions, key: str, *, mode: str = "persistent", **origin) -> None:
    await sessions.create(
        key,
        agent_id="main",
        origin={"kind": "acp", "agent_id": "claude", "cwd": "/tmp", "mode": mode, **origin},
    )


def _manager(runtime, sessions, **cfg) -> AcpSessionManager:
    return AcpSessionManager(runtime, config=AcpConfig(**cfg), session_manager=sessions)


async def test_run_turn_streams_events_and_persists_identity(sessions) -> None:
    key = "agent:main:acp:aaaa0001"
    await _acp_row(sessions, key)
    runtime = FakeRuntime([text_script("hello", input_tokens=3)])
    manager = _manager(runtime, sessions)
    seen: list[AcpEvent] = []

    result = await manager.run_turn(key, "do it", on_event=seen.append)

    assert result.stop_reason == "end_turn"
    assert [e.text for e in seen] == ["hello"]
    row = await sessions.get_session(key)
    assert row.origin["record_id"] == "rec-1"
    assert row.origin["acp_session_id"] == "sess-1"
    assert row.origin["last_stop_reason"] == "end_turn"
    assert manager.list_sessions()[0]["session_key"] == key


async def test_persistent_session_reuses_handle(sessions) -> None:
    key = "agent:main:acp:aaaa0002"
    await _acp_row(sessions, key)
    runtime = FakeRuntime()
    manager = _manager(runtime, sessions)
    await manager.run_turn(key, "one")
    await manager.run_turn(key, "two")
    assert [c[0] for c in runtime.calls].count("ensure_session") == 1
    assert runtime.closed == []


async def test_oneshot_closes_after_first_turn(sessions) -> None:
    key = "agent:main:acp:aaaa0003"
    await _acp_row(sessions, key, mode="oneshot")
    runtime = FakeRuntime()
    manager = _manager(runtime, sessions)
    await manager.run_turn(key, "one")
    assert runtime.closed == [("rec-1", False)]
    assert (await sessions.get_session(key)).origin["closed"] is True
    with pytest.raises(AcpSessionClosedError):
        await manager.run_turn(key, "two")
    assert manager.live_count == 0


async def test_resume_uses_recorded_session_id(sessions) -> None:
    key = "agent:main:acp:aaaa0004"
    await _acp_row(sessions, key, record_id="rec-old", acp_session_id="sess-old")
    runtime = FakeRuntime()
    manager = _manager(runtime, sessions)
    await manager.run_turn(key, "continue")
    spec = runtime.calls[0][1]
    assert spec.resume_session_id == "sess-old"
    assert spec.extra["record_id"] == "rec-old"


async def test_turn_timeout_cancels_and_raises(sessions) -> None:
    key = "agent:main:acp:aaaa0005"
    await _acp_row(sessions, key, mode="oneshot", turn_timeout_s=0.05)
    runtime = FakeRuntime([Script(events=[AcpEvent(kind="text_delta", text="..")], block=True)])
    manager = _manager(runtime, sessions)
    with pytest.raises(AcpTurnTimeoutError):
        await manager.run_turn(key, "slow")
    assert runtime.turn_cancels == 1
    # oneshot: closed on timeout too
    assert runtime.closed == [("rec-1", False)]


async def test_config_turn_timeout_applies_when_spec_has_none(sessions) -> None:
    key = "agent:main:acp:aaaa0006"
    await _acp_row(sessions, key)
    runtime = FakeRuntime([Script(block=True)])
    manager = AcpSessionManager(
        runtime, config=AcpConfig(turn_timeout_s=1), session_manager=sessions
    )
    spec = spec_from_origin((await sessions.get_session(key)).origin, session_key=key)
    spec.turn_timeout_s = 0.05
    with pytest.raises(AcpTurnTimeoutError):
        await manager.run_turn(key, "slow", spec=spec)


async def test_task_cancellation_cancels_harness_turn(sessions) -> None:
    key = "agent:main:acp:aaaa0007"
    await _acp_row(sessions, key)
    runtime = FakeRuntime([Script(block=True)])
    manager = _manager(runtime, sessions)
    task = asyncio.create_task(manager.run_turn(key, "long"))
    await runtime.turn_started.wait()
    await asyncio.sleep(0)
    assert manager.is_turn_active(key)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.turn_cancels == 1
    assert not manager.is_turn_active(key)


async def test_operator_cancel_returns_cancelled_result(sessions) -> None:
    key = "agent:main:acp:aaaa0008"
    await _acp_row(sessions, key)
    runtime = FakeRuntime([Script(block=True)])
    manager = _manager(runtime, sessions)
    task = asyncio.create_task(manager.run_turn(key, "long"))
    await runtime.turn_started.wait()
    await asyncio.sleep(0)
    assert await manager.cancel(key) is True
    result = await task
    assert result.stop_reason == "cancelled"
    assert await manager.cancel(key) is False


async def test_harness_error_result_raises_turn_error(sessions) -> None:
    key = "agent:main:acp:aaaa0009"
    await _acp_row(sessions, key)
    runtime = FakeRuntime([Script(result=AcpTurnResult(stop_reason="error", error="boom"))])
    manager = _manager(runtime, sessions)
    with pytest.raises(AcpTurnError, match="boom"):
        await manager.run_turn(key, "x")
    status = await manager.status(key)
    assert status["last_error"] == "boom"


async def test_runtime_exception_is_wrapped(sessions) -> None:
    key = "agent:main:acp:aaaa0010"
    await _acp_row(sessions, key)
    runtime = FakeRuntime([Script(raise_exc=RuntimeError("pipe closed"))])
    manager = _manager(runtime, sessions)
    with pytest.raises(AcpTurnError, match="pipe closed"):
        await manager.run_turn(key, "x")


async def test_max_concurrent_evicts_idle_lru(sessions) -> None:
    a, b = "agent:main:acp:bbbb0001", "agent:main:acp:bbbb0002"
    await _acp_row(sessions, a)
    await _acp_row(sessions, b)
    runtime = FakeRuntime()
    manager = _manager(runtime, sessions, max_concurrent=1)
    await manager.run_turn(a, "one")
    await manager.run_turn(b, "two")
    assert runtime.closed == [("rec-1", False)]
    assert manager.live_count == 1
    # a resumes on its next turn (and evicts b)
    await manager.run_turn(a, "three")
    assert runtime.closed[-1] == ("rec-2", False)
    assert runtime.calls[-2][1].resume_session_id == "sess-1"


async def test_max_concurrent_waits_while_all_busy(sessions) -> None:
    a, b = "agent:main:acp:bbbb0003", "agent:main:acp:bbbb0004"
    await _acp_row(sessions, a)
    await _acp_row(sessions, b)
    runtime = FakeRuntime([Script(block=True), text_script("b done")])
    manager = _manager(runtime, sessions, max_concurrent=1)
    first = asyncio.create_task(manager.run_turn(a, "busy"))
    await runtime.turn_started.wait()
    second = asyncio.create_task(manager.run_turn(b, "waits"))
    await asyncio.sleep(0.02)
    assert not second.done()
    runtime.release.set()
    await first
    result = await asyncio.wait_for(second, timeout=1)
    assert result.text == "b done"


async def test_set_option_resolves_aliases(sessions) -> None:
    key = "agent:main:acp:cccc0001"
    await _acp_row(sessions, key)
    runtime = FakeRuntime()
    runtime.config_options = [
        {"id": "model"},
        {"id": "reasoning_effort"},
        {"id": "sandbox", "category": "permissions"},
    ]
    manager = _manager(runtime, sessions)
    await manager.set_option(key, "thinking", "high")
    await manager.set_option(key, "permissions", "read-only")
    await manager.set_model(key, "opus")
    sets = [c[1] for c in runtime.calls if c[0] == "set_config_option"]
    assert sets == [("reasoning_effort", "high"), ("sandbox", "read-only"), ("model", "opus")]
    origin = (await sessions.get_session(key)).origin
    assert origin["model"] == "opus"
    assert origin["config_options"]["reasoning_effort"] == "high"
    with pytest.raises(AcpOptionUnsupportedError):
        await manager.set_option(key, "temperature", "1")


async def test_set_mode_persists_harness_mode(sessions) -> None:
    key = "agent:main:acp:cccc0002"
    await _acp_row(sessions, key)
    runtime = FakeRuntime()
    manager = _manager(runtime, sessions)
    await manager.set_mode(key, "plan")
    assert ("set_mode", "plan") in runtime.calls
    assert (await sessions.get_session(key)).origin["harness_mode"] == "plan"
    # The next turn re-applies it through the spec.
    await manager.close(key)
    assert spec_from_origin((await sessions.get_session(key)).origin).harness_mode == "plan"


async def test_close_marks_closed_and_shutdown_closes_all(sessions) -> None:
    a, b = "agent:main:acp:dddd0001", "agent:main:acp:dddd0002"
    await _acp_row(sessions, a)
    await _acp_row(sessions, b)
    runtime = FakeRuntime()
    manager = _manager(runtime, sessions)
    await manager.run_turn(a, "x")
    await manager.run_turn(b, "y")
    assert await manager.close(a) is True
    assert (await sessions.get_session(a)).origin["closed"] is True
    with pytest.raises(AcpSessionClosedError):
        await manager.run_turn(a, "again")
    await manager.shutdown()
    assert {rid for rid, _ in runtime.closed} == {"rec-1", "rec-2"}
    assert not manager.available
    with pytest.raises(AcpUnavailableError):
        await manager.run_turn(b, "after shutdown")


async def test_unavailable_manager_explains_why(sessions) -> None:
    manager = AcpSessionManager(None, unavailable_reason="ACP runtime not installed (x)")
    assert not manager.available
    with pytest.raises(AcpUnavailableError, match="not installed"):
        await manager.run_turn("agent:main:acp:eeee0001", "x")


async def test_non_acp_row_is_refused(sessions) -> None:
    await sessions.create(
        "agent:main:subagent:ffff0001", agent_id="main", origin={"kind": "subagent"}
    )
    manager = _manager(FakeRuntime(), sessions)
    with pytest.raises(AcpError) as info:
        await manager.run_turn("agent:main:subagent:ffff0001", "x")
    assert info.value.code == "ACP_SESSION_INVALID"


def test_spec_from_origin_defaults() -> None:
    spec = spec_from_origin({"agent_id": "Claude-Code", "cwd": "/w"}, AcpConfig(), session_key="k")
    assert spec.agent_id == "claude" and spec.mode == "persistent"
    with pytest.raises(AcpError):
        spec_from_origin({"agent_id": "claude"}, session_key="k")
