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


async def test_oneshot_closes_process_but_stays_resumable(sessions) -> None:
    key = "agent:main:acp:aaaa0003"
    await _acp_row(sessions, key, mode="oneshot")
    runtime = FakeRuntime()
    manager = _manager(runtime, sessions)
    await manager.run_turn(key, "one")
    assert runtime.closed == [("rec-1", False)]
    origin = (await sessions.get_session(key)).origin
    assert origin["idle"] is True and not origin.get("closed")
    assert manager.live_count == 0
    status = await manager.status(key)
    assert status["idle"] is True and status["closed"] is False and status["live"] is False

    # A follow-up resumes the recorded harness session in a new process.
    await manager.run_turn(key, "two")
    ensures = [c[1] for c in runtime.calls if c[0] == "ensure_session"]
    assert [s.resume_session_id for s in ensures] == [None, "sess-1"]
    assert ensures[1].extra["record_id"] == "rec-1"
    assert len(runtime.closed) == 2 and manager.live_count == 0
    assert (await sessions.get_session(key)).origin["idle"] is True


async def test_turn_clears_idle_while_it_runs(sessions) -> None:
    key = "agent:main:acp:aaaa0013"
    await _acp_row(sessions, key, mode="oneshot", idle=True, acp_session_id="sess-old")
    runtime = FakeRuntime([Script(block=True)])
    manager = _manager(runtime, sessions)
    task = asyncio.create_task(manager.run_turn(key, "go"))
    await asyncio.wait_for(runtime.turn_started.wait(), timeout=2)
    assert (await sessions.get_session(key)).origin["idle"] is False
    status = await manager.status(key)
    assert status["idle"] is False and status["active_turn"] is True
    runtime.release.set()
    await task
    assert (await sessions.get_session(key)).origin["idle"] is True


async def test_operator_close_of_an_idle_oneshot_refuses_turns(sessions) -> None:
    key = "agent:main:acp:aaaa0014"
    await _acp_row(sessions, key, mode="oneshot")
    runtime = FakeRuntime()
    manager = _manager(runtime, sessions)
    await manager.run_turn(key, "one")
    assert await manager.close(key) is False  # no process to close
    origin = (await sessions.get_session(key)).origin
    assert origin["closed"] is True and origin["closed_reason"] == "operator_close"
    assert origin["idle"] is False
    with pytest.raises(AcpSessionClosedError):
        await manager.run_turn(key, "two")
    status = await manager.status(key)
    assert status["closed"] is True and status["idle"] is False


@pytest.mark.parametrize(
    ("origin", "status", "closed"),
    [
        ({}, None, False),
        ({"closed": True, "closed_reason": "operator_close", "mode": "oneshot"}, None, True),
        ({"closed": True, "mode": "persistent"}, "done", True),
        # Rows written before idle existed: a finished oneshot carried closed.
        ({"closed": True, "mode": "oneshot"}, "done", False),
        ({"closed": True, "mode": "oneshot"}, "killed", True),
        ({"idle": True, "mode": "oneshot"}, "done", False),
    ],
)
def test_origin_is_closed(origin, status, closed) -> None:
    from agentos.acp.manager import origin_is_closed

    assert origin_is_closed({"kind": "acp", **origin}, session_status=status) is closed


async def test_legacy_closed_oneshot_row_resumes(sessions) -> None:
    key = "agent:main:acp:aaaa0015"
    await _acp_row(
        sessions,
        key,
        mode="oneshot",
        closed=True,
        record_id="rec-old",
        acp_session_id="sess-old",
    )
    runtime = FakeRuntime()
    manager = _manager(runtime, sessions)
    await manager.run_turn(key, "continue")
    spec = runtime.calls[0][1]
    assert spec.resume_session_id == "sess-old"
    origin = (await sessions.get_session(key)).origin
    assert origin["closed"] is False and origin["idle"] is True


async def test_resume_lost_emits_a_notice_first(sessions) -> None:
    key = "agent:main:acp:aaaa0016"
    await _acp_row(sessions, key, mode="oneshot", record_id="rec-old", acp_session_id="sess-old")
    runtime = FakeRuntime([text_script("fresh answer")])
    runtime.resume_supported = False
    manager = _manager(runtime, sessions)
    seen: list[AcpEvent] = []
    await manager.run_turn(key, "continue", on_event=seen.append)
    assert [e.kind for e in seen] == ["text_delta", "text_delta"]
    assert seen[0].text == (
        "Started a new Claude Code session; the previous context could not be resumed.\n\n"
    )
    assert seen[1].text == "fresh answer"
    assert (await sessions.get_session(key)).origin["acp_session_id"] == "sess-1"


async def test_activity_tracks_the_current_harness_step(sessions, monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/home/u")
    key = "agent:main:acp:aaaa0017"
    await _acp_row(sessions, key)
    runtime = FakeRuntime(
        [
            Script(
                events=[
                    AcpEvent(kind="thought_delta", text="hmm"),
                    AcpEvent(
                        kind="tool_call",
                        tool_call_id="t1",
                        tool_kind="read",
                        title="Read /home/u/app/src/app.py",
                    ),
                    AcpEvent(kind="tool_call_update", tool_call_id="t1", status="completed"),
                    AcpEvent(
                        kind="tool_call",
                        tool_call_id="t2",
                        tool_kind="execute",
                        title="`pytest -q`",
                    ),
                ],
                block=True,
            )
        ]
    )
    manager = _manager(runtime, sessions)
    task = asyncio.create_task(manager.run_turn(key, "go"))
    await asyncio.wait_for(runtime.turn_started.wait(), timeout=2)
    for _ in range(50):
        rows = manager.list_sessions()
        if rows and rows[0]["turn_tool_calls"] == 2:
            break
        await asyncio.sleep(0.01)
    row = manager.list_sessions()[0]
    assert row["activity"] == "Run pytest -q" and row["activity_kind"] == "execute"
    assert row["turn_tool_calls"] == 2
    assert row["agent_name"] == "Claude Code"
    assert "turn_elapsed_s" in row
    runtime.release.set()
    await task
    status = await manager.status(key)
    assert status["activity"] == "Run pytest -q"  # the last step stays visible
    assert status["turn_tool_calls"] == 2


def test_activity_label_is_short_single_line_and_home_relative(monkeypatch) -> None:
    from agentos.acp.labels import tool_label
    from agentos.acp.manager import ACTIVITY_MAX_CHARS

    monkeypatch.setenv("HOME", "/Users/x")
    monkeypatch.setenv("USERPROFILE", "/Users/x")  # what expanduser reads on Windows
    assert tool_label("Read /Users/x/app/a.py", "read") == "Read ~/app/a.py"
    assert tool_label("`ls\n  -la`", "execute") == "Run ls"  # first line only
    assert tool_label(None, "search") == "Searching"
    assert tool_label("", None) == "Working"
    long = tool_label("x", "edit", {"file_path": "/Users/x/" + "a" * 200})
    assert len(long) == ACTIVITY_MAX_CHARS and long.endswith("…") and "/Users/x" not in long


async def test_status_requires_an_acp_row_when_asked(sessions) -> None:
    await sessions.create("agent:main:main", agent_id="main")
    manager = _manager(FakeRuntime(), sessions)
    with pytest.raises(AcpError) as info:
        await manager.status("agent:main:main", require_acp=True)
    assert info.value.code == "ACP_SESSION_INVALID"
    with pytest.raises(AcpError):
        await manager.status("agent:main:acp:00000000", require_acp=True)


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


async def test_starting_harness_is_listed_and_not_idle(sessions) -> None:
    key = "agent:main:acp:aaaa0018"
    await _acp_row(sessions, key, mode="oneshot", idle=True, acp_session_id="sess-old")
    runtime = FakeRuntime()
    gate = asyncio.Event()
    entered = asyncio.Event()
    original = runtime.ensure_session

    async def _slow(spec, *, session_key):
        entered.set()
        await gate.wait()
        return await original(spec, session_key=session_key)

    runtime.ensure_session = _slow  # type: ignore[method-assign]
    manager = _manager(runtime, sessions)
    task = asyncio.create_task(manager.run_turn(key, "go"))
    await asyncio.wait_for(entered.wait(), timeout=2)
    rows = manager.list_sessions()
    assert [r["session_key"] for r in rows] == [key]
    assert rows[0]["activity"] == "Starting Claude Code…" and rows[0]["activity_kind"] is None
    assert "turn_elapsed_s" in rows[0]
    status = await manager.status(key)
    assert status["idle"] is False and status["live"] is False
    gate.set()
    await task
    assert manager.list_sessions() == []
    assert (await manager.status(key))["idle"] is True


async def test_prepare_turn_drops_the_previous_step_at_once(sessions) -> None:
    key = "agent:main:acp:aaaa0019"
    await _acp_row(sessions, key, mode="oneshot")
    runtime = FakeRuntime(
        [
            Script(
                events=[AcpEvent(kind="tool_call", tool_call_id="t", tool_kind="read", title="R")],
                result=AcpTurnResult(stop_reason="end_turn"),
            )
        ]
    )
    manager = _manager(runtime, sessions)
    await manager.run_turn(key, "one")
    assert (await manager.status(key))["activity"] == "R"
    manager.prepare_turn(key, "claude")
    status = await manager.status(key)
    assert status["activity"] == "Starting Claude Code…" and status["turn_tool_calls"] == 0
    assert status["idle"] is False


async def test_prepare_turn_leaves_a_running_turn_alone(sessions) -> None:
    key = "agent:main:acp:aaaa0020"
    await _acp_row(sessions, key)
    runtime = FakeRuntime(
        [
            Script(
                events=[AcpEvent(kind="tool_call", tool_call_id="t", tool_kind="read", title="R")],
                block=True,
            )
        ]
    )
    manager = _manager(runtime, sessions)
    task = asyncio.create_task(manager.run_turn(key, "go"))
    await asyncio.wait_for(runtime.turn_started.wait(), timeout=2)
    await asyncio.sleep(0.01)
    manager.prepare_turn(key, "claude")
    assert manager.list_sessions()[0]["activity"] == "R"
    runtime.release.set()
    await task


async def test_status_does_not_wait_for_a_busy_row_read(sessions) -> None:
    """acp.status took 2.9 s right after a turn ended while the end-of-turn
    writes held the database; it answers from the last row it saw instead."""
    key = "agent:main:acp:aaaa0021"
    await _acp_row(sessions, key, mode="oneshot")
    manager = _manager(FakeRuntime(), sessions)
    await manager.run_turn(key, "one")

    gate = asyncio.Event()
    original = sessions.get_session

    async def _slow(session_key):
        await gate.wait()
        return await original(session_key)

    sessions.get_session = _slow  # type: ignore[method-assign]
    try:
        status = await asyncio.wait_for(manager.status(key), timeout=1.0)
    finally:
        gate.set()
        sessions.get_session = original  # type: ignore[method-assign]
    assert status["idle"] is True and status["active_turn"] is False
    assert status["agent_id"] == "claude" and status["last_turn"] is None


async def test_status_reflects_a_close_without_a_new_read(sessions) -> None:
    key = "agent:main:acp:aaaa0022"
    await _acp_row(sessions, key, mode="oneshot")
    manager = _manager(FakeRuntime(), sessions)
    await manager.run_turn(key, "one")
    await manager.status(key)  # warms the row cache
    await manager.close(key)
    assert (await manager.status(key))["closed"] is True


async def test_activity_says_waiting_once_the_prompt_is_sent(sessions) -> None:
    """agent:main:acp:0d499dc5: the resumed harness was up and the prompt sent,
    but the model took over a minute; the bar still said "Starting"."""
    key = "agent:main:acp:aaaa0023"
    await _acp_row(sessions, key, mode="oneshot")
    runtime = FakeRuntime([Script(block=True)])
    manager = _manager(runtime, sessions)
    task = asyncio.create_task(manager.run_turn(key, "go"))
    await asyncio.wait_for(runtime.turn_started.wait(), timeout=2)
    await asyncio.sleep(0.01)
    assert manager.list_sessions()[0]["activity"] == "Waiting for Claude Code…"
    runtime.release.set()
    await task


def test_activity_says_thinking_once_no_tool_is_running() -> None:
    """agent:main:acp:fd260ca2: "Run python3 -I -c ..." stayed up for ~8
    minutes after the command finished while Claude composed its answer."""
    from agentos.acp.manager import _SessionState

    state = _SessionState(session_key="agent:main:acp:aaaa0024")
    track = AcpSessionManager._track_activity

    track(
        state,
        AcpEvent(
            kind="tool_call",
            tool_call_id="a",
            tool_kind="execute",
            title="python3 -I -c 'x'",
            raw_input={"command": "python3 -I -c 'x'"},
        ),
    )
    assert state.activity == "Run python3 -I -c 'x'" and state.activity_kind == "execute"
    # The client merges title/kind into the completion update; the step is done.
    track(
        state,
        AcpEvent(
            kind="tool_call_update",
            tool_call_id="a",
            tool_kind="execute",
            title="python3 -I -c 'x'",
            status="completed",
        ),
    )
    assert state.activity == "Thinking…" and state.activity_kind == "think"

    # Two calls in flight: when one ends, the other is what runs now.
    track(state, AcpEvent(kind="tool_call", tool_call_id="b", tool_kind="read", title="Read b"))
    track(
        state,
        AcpEvent(kind="tool_call", tool_call_id="c", tool_kind="search", title="grep x"),
    )
    track(
        state,
        AcpEvent(
            kind="tool_call_update",
            tool_call_id="c",
            tool_kind="search",
            title="grep x",
            status="completed",
        ),
    )
    assert state.activity == "Read b" and state.activity_kind == "read"
    track(state, AcpEvent(kind="tool_call_update", tool_call_id="b", status="failed"))
    assert state.activity == "Thinking…"

    track(state, AcpEvent(kind="text_delta", text="Here is the report"))
    assert state.activity == "Writing the answer" and state.activity_kind == "text"
    assert len(state.turn_tools) == 3
