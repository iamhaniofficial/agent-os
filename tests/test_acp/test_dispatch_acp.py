"""End to end: sessions_spawn(runtime="acp") -> TaskRuntime -> dispatch ->
AcpSessionManager -> projector -> events, transcript, usage, announce."""

from __future__ import annotations

import asyncio
import json

from acp_runtime_helpers import wait_until
from acp_s_fakes import FakeRuntime, Script
from acp_s_harness import PARENT_KEY, acp_rig

from agentos.acp.types import AcpEvent, AcpTurnResult, AcpTurnTimeoutError, AcpUsage
from agentos.tools.builtin import sessions as sessions_tool


def _coding_script() -> Script:
    return Script(
        events=[
            AcpEvent(kind="thought_delta", text="planning"),
            AcpEvent(kind="tool_call", tool_call_id="t1", tool_kind="edit", title="Edit a.py"),
            AcpEvent(
                kind="tool_call_update", tool_call_id="t1", status="completed", raw_output="ok"
            ),
            AcpEvent(kind="usage", usage={"used": 10, "size": 100}, cost_usd=0.03),
            AcpEvent(kind="text_delta", text="Fixed the bug."),
        ],
        result=AcpTurnResult(
            stop_reason="end_turn",
            usage=AcpUsage(input_tokens=1200, output_tokens=300, cost_usd=0.03),
            tool_calls=1,
            duration_ms=10,
        ),
    )


async def _completions(rig, n: int = 1) -> list[dict]:
    """Parent-side completion rows; the announce runs just after the task settles."""
    for _ in range(200):
        rows = await rig.sessions.read_recent_transcript(PARENT_KEY, 20)
        found = [json.loads(r["content"]) for r in rows if r["role"] == "system"]
        if len(found) >= n:
            return found
        await asyncio.sleep(0.005)
    return found


async def _spawn(**kwargs) -> dict:
    return json.loads(await sessions_tool.sessions_spawn(runtime="acp", **kwargs))


async def test_spawned_acp_turn_streams_persists_and_announces(tmp_path) -> None:
    async with acp_rig(tmp_path, FakeRuntime([_coding_script()])) as rig:
        out = await _spawn(task="fix the bug", acp_agent="claude")
        child = out["session_key"]
        record = await rig.task_runtime.wait(out["task_id"], timeout=5)
        assert str(record.status.value if hasattr(record.status, "value") else record.status) == (
            "succeeded"
        )

        names = rig.event_names(child)
        assert "session.event.tool_use_start" in names
        assert "session.event.tool_result" in names
        assert names.index("session.event.text_delta") < names.index("session.event.done")
        assert "session.event.error" not in names
        done = [p for k, n, p in rig.events if k == child and n == "session.event.done"][0]
        assert done["input_tokens"] == 1200 and abs(done["cost_usd"] - 0.03) < 1e-9
        assert rig.turn_runner.calls == 0

        transcript = await rig.sessions.read_recent_transcript(child, 10)
        roles = [row["role"] for row in transcript]
        assert roles == ["user", "assistant"]
        assistant = transcript[-1]
        assert assistant["content"] == "Fixed the bug."
        assert [s["type"] for s in assistant["tool_calls"]] == ["tool_use", "tool_result", "text"]

        row = await rig.sessions.get_session(child)
        assert row.input_tokens == 1200 and row.output_tokens == 300
        assert abs(row.total_cost_usd - 0.03) < 1e-9 and row.cost_source == "harness"
        assert row.origin["record_id"] == "rec-1"
        # oneshot by default: the process is gone but the session is idle, not closed
        assert row.origin["idle"] is True and not row.origin.get("closed")
        assert row.origin["last_turn"]["input_tokens"] == 1200

        completions = await _completions(rig)
        assert completions and completions[-1]["child_session_key"] == child
        stats = completions[-1]["stats"]
        assert stats.startswith("runtime ") and "tokens 1500 (in 1200 / out 300)" in stats
        assert "cost $0.0300" in stats
        live = [p for _, n, p in rig.events if n == "session.event.subagent_completion"]
        assert live
        # Contract B: the persisted parent message and the live event carry the
        # harness identity and structured usage next to the Stats string.
        for payload in (completions[-1], live[-1]):
            assert payload["runtime"] == "acp"
            assert payload["acp_agent"] == "claude"
            assert payload["acp_agent_name"] == "Claude Code"
            assert payload["cwd"] == str(rig.workspace.resolve())
            assert payload["mode"] == "oneshot"
            assert payload["harness_mode"] is None
            usage = payload["usage"]
            assert set(usage) == {
                "duration_ms",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cost_usd",
                "tool_calls",
            }
            assert usage["input_tokens"] == 1200 and usage["output_tokens"] == 300
            assert usage["total_tokens"] == 1500 and usage["tool_calls"] == 1
            assert abs(usage["cost_usd"] - 0.03) < 1e-9
            assert isinstance(usage["duration_ms"], int)
            assert payload["stats"] == stats


async def test_parent_wake_message_carries_stats_line(tmp_path) -> None:
    async with acp_rig(tmp_path, FakeRuntime([_coding_script()])) as rig:
        sent: list[str] = []
        original_send = rig.task_runtime.send

        async def _capture(session_key, message, provenance=None, stream_event_sink=None):
            if session_key == PARENT_KEY:
                sent.append(message)
                from types import SimpleNamespace

                return SimpleNamespace(task_id="wake")
            return await original_send(session_key, message, provenance=provenance)

        rig.task_runtime.send = _capture  # type: ignore[method-assign]
        out = await _spawn(task="fix", acp_agent="claude")
        # Close the spawn group like sessions_yield does, then let the child finish.
        await sessions_tool.sessions_yield()
        await rig.task_runtime.wait(out["task_id"], timeout=5)
        assert await wait_until(lambda: bool(sent)), "parent was not woken"
        assert "Stats: runtime" in sent[-1]
        assert "Fixed the bug." in sent[-1]


async def test_timeout_fails_task_with_timeout_status(tmp_path) -> None:
    script = Script(raise_exc=AcpTurnTimeoutError("ACP turn timed out after 5s"))
    async with acp_rig(tmp_path, FakeRuntime([script])) as rig:
        out = await _spawn(task="slow", acp_agent="claude")
        record = await rig.task_runtime.wait(out["task_id"], timeout=5)
        assert str(getattr(record.status, "value", record.status)) == "timeout"
        child = out["session_key"]
        errors = [p for k, n, p in rig.events if k == child and n == "session.event.error"]
        assert errors and errors[0]["code"] == "ACP_TURN_TIMEOUT"
        row = await rig.sessions.get_session(child)
        assert row.origin["last_turn"]["error_code"] == "ACP_TURN_TIMEOUT"


async def test_unavailable_manager_fails_turn_cleanly(tmp_path) -> None:
    from agentos.acp.manager import AcpSessionManager, set_acp_manager

    async with acp_rig(tmp_path) as rig:
        out = await _spawn(task="x", acp_agent="claude")
        # Runtime disappears between spawn and dispatch.
        set_acp_manager(AcpSessionManager(None, unavailable_reason="runtime missing"))
        record = await rig.task_runtime.wait(out["task_id"], timeout=5)
        assert str(getattr(record.status, "value", record.status)) == "failed"
        errors = [p for k, n, p in rig.events if n == "session.event.error"]
        assert errors and errors[0]["code"] == "ACP_UNAVAILABLE"
        assert rig.turn_runner.calls == 0


async def test_task_runtime_cancel_cancels_harness_turn(tmp_path) -> None:
    runtime = FakeRuntime([Script(block=True)])
    async with acp_rig(tmp_path, runtime) as rig:
        out = await _spawn(task="long", acp_agent="claude", mode="persistent")
        await asyncio.wait_for(runtime.turn_started.wait(), timeout=5)
        await asyncio.sleep(0)
        await rig.task_runtime.cancel(session_key=out["session_key"], source="test", reason="t")
        record = await rig.task_runtime.wait(out["task_id"], timeout=5)
        assert str(getattr(record.status, "value", record.status)) == "cancelled"
        assert runtime.turn_cancels == 1


async def test_persistent_followup_and_steer(tmp_path) -> None:
    runtime = FakeRuntime(
        [
            Script(
                events=[AcpEvent(kind="text_delta", text="first")],
                result=AcpTurnResult(stop_reason="end_turn"),
            ),
            Script(block=True),
            Script(
                events=[AcpEvent(kind="text_delta", text="steered")],
                result=AcpTurnResult(stop_reason="end_turn"),
            ),
        ]
    )
    async with acp_rig(tmp_path, runtime) as rig:
        out = await _spawn(task="start", acp_agent="claude", mode="persistent")
        child = out["session_key"]
        await rig.task_runtime.wait(out["task_id"], timeout=5)
        await _completions(rig, 1)
        assert (await rig.sessions.get_session(child)).status == "done"

        follow = json.loads(await sessions_tool.sessions_send(child, "keep going"))
        assert follow["runtime"] == "acp" and follow["mode"] == "followup"
        row = await rig.sessions.get_session(child)
        assert row.status == "running"
        await asyncio.wait_for(_wait_for_calls(runtime, 2), timeout=5)

        steer = json.loads(await sessions_tool.sessions_send(child, "do X instead", mode="steer"))
        assert steer["mode"] == "steer"
        await rig.task_runtime.wait(steer["task_id"], timeout=5)
        await rig.task_runtime.wait(follow["task_id"], timeout=5)

        completions = await _completions(rig, 2)
        row = await rig.sessions.get_session(child)
        assert follow["task_id"] in row.origin["superseded_task_ids"]
        assert row.status == "done"
        # Only one harness process for the whole conversation.
        assert [c[0] for c in runtime.calls].count("ensure_session") == 1
        # spawn turn + steered turn announce; the superseded one stays silent.
        assert [c["status"] for c in completions] == ["succeeded", "succeeded"]
        assert completions[-1]["result"]["text"] == "steered"
        transcript = await rig.sessions.read_recent_transcript(child, 20)
        assert [r["content"] for r in transcript if r["role"] == "user"] == [
            "start",
            "keep going",
            "do X instead",
        ]


async def _wait_for_calls(runtime: FakeRuntime, n: int) -> None:
    while [c[0] for c in runtime.calls].count("start_turn") < n:
        await asyncio.sleep(0.005)


async def test_oneshot_followup_resumes_the_harness_session(tmp_path) -> None:
    """2026-10-19 repro: a follow-up into a finished oneshot child got
    ACP_SESSION_CLOSED. It now resumes the recorded harness session in a new
    process, which is closed again after the turn."""
    runtime = FakeRuntime(
        [
            Script(
                events=[AcpEvent(kind="text_delta", text="first answer")],
                result=AcpTurnResult(stop_reason="end_turn"),
            ),
            Script(
                events=[AcpEvent(kind="text_delta", text="second answer")],
                result=AcpTurnResult(stop_reason="end_turn"),
            ),
        ]
    )
    async with acp_rig(tmp_path, runtime) as rig:
        out = await _spawn(task="once", acp_agent="claude")
        child = out["session_key"]
        await rig.task_runtime.wait(out["task_id"], timeout=5)
        await _completions(rig, 1)
        row = await rig.sessions.get_session(child)
        assert row.origin["idle"] is True and row.origin["acp_session_id"] == "sess-1"
        assert runtime.closed == [("rec-1", False)]

        follow = json.loads(await sessions_tool.sessions_send(child, "more"))
        record = await rig.task_runtime.wait(follow["task_id"], timeout=5)
        assert str(getattr(record.status, "value", record.status)) == "succeeded"

        ensures = [c[1] for c in runtime.calls if c[0] == "ensure_session"]
        assert len(ensures) == 2
        assert ensures[1].resume_session_id == "sess-1"
        assert ensures[1].extra["record_id"] == "rec-1"
        assert ensures[1].mode == "oneshot"
        assert len(runtime.closed) == 2  # closed again after the follow-up
        row = await rig.sessions.get_session(child)
        assert row.origin["idle"] is True and not row.origin.get("closed")
        transcript = await rig.sessions.read_recent_transcript(child, 10)
        assert [(r["role"], r["content"]) for r in transcript] == [
            ("user", "once"),
            ("assistant", "first answer"),
            ("user", "more"),
            ("assistant", "second answer"),
        ]
        # The parent sent this follow-up, so it is announced like the spawn.
        completions = await _completions(rig, 2)
        assert completions[-1]["result"]["text"] == "second answer"


async def test_resume_failure_starts_fresh_with_a_notice(tmp_path) -> None:
    runtime = FakeRuntime()
    runtime.resume_supported = False
    async with acp_rig(tmp_path, runtime) as rig:
        out = await _spawn(task="once", acp_agent="claude")
        child = out["session_key"]
        await rig.task_runtime.wait(out["task_id"], timeout=5)
        await _completions(rig, 1)

        follow = json.loads(await sessions_tool.sessions_send(child, "again"))
        record = await rig.task_runtime.wait(follow["task_id"], timeout=5)
        assert str(getattr(record.status, "value", record.status)) == "succeeded"

        notice = "Started a new Claude Code session; the previous context could not be resumed."
        deltas = [
            p["text"] for k, n, p in rig.events if k == child and n == "session.event.text_delta"
        ]
        assert any(notice in d for d in deltas)
        transcript = await rig.sessions.read_recent_transcript(child, 10)
        assert transcript[-1]["role"] == "assistant"
        assert transcript[-1]["content"].startswith(notice + "\n\n")
        assert transcript[-1]["content"].endswith("echo: again")
        # The new harness session is the one a later follow-up resumes.
        row = await rig.sessions.get_session(child)
        assert row.origin["acp_session_id"] == "sess-2"


async def test_turn_into_closed_session_is_logged_persisted_and_emitted(tmp_path) -> None:
    async with acp_rig(tmp_path) as rig:
        out = await _spawn(task="once", acp_agent="claude", mode="persistent")
        child = out["session_key"]
        await rig.task_runtime.wait(out["task_id"], timeout=5)
        await rig.manager.close(child, reason="operator_close")
        row = await rig.sessions.get_session(child)
        assert row.origin["closed"] is True and row.origin["closed_reason"] == "operator_close"

        # The user types into the closed child (desktop sessions.send path).
        entry = await rig.sessions.append_message(child, role="user", content="hello?")
        from agentos.gateway.routing import build_web_route_envelope

        envelope = build_web_route_envelope(
            session_key=child, agent_id="main", conn_id="c1", channel_id="web:c1"
        )
        handle = await rig.task_runtime.enqueue(
            envelope,
            "hello?",
            run_kind="session_turn",
            persisted_user_message_id=entry.message_id,
        )
        record = await rig.task_runtime.wait(handle.task_id, timeout=5)
        assert str(getattr(record.status, "value", record.status)) == "failed"

        errors = [p for k, n, p in rig.events if k == child and n == "session.event.error"]
        assert errors and errors[-1]["code"] == "ACP_SESSION_CLOSED"
        assert "was closed" in errors[-1]["error_message"]
        assert errors[-1]["terminal_message"]
        transcript = await rig.sessions.read_recent_transcript(child, 10)
        assert transcript[-2]["content"] == "hello?"
        assert transcript[-1]["role"] == "system"
        assert transcript[-1]["content"].startswith("Error: This Claude Code session")
        assert "resume_session_id=sess-1" in transcript[-1]["content"]
        # Shown by chat.history (the UI resyncs from it right after the error).
        from agentos.chat.history import transcript_entries_to_chat_messages

        history = transcript_entries_to_chat_messages(await rig.sessions.get_transcript(child))
        assert history[-1]["role"] == "system" and history[-1]["text"].startswith("Error:")
        row = await rig.sessions.get_session(child)
        assert row.origin["last_turn"]["error_code"] == "ACP_SESSION_CLOSED"


async def test_harness_start_failure_leaves_an_error_row(tmp_path) -> None:
    from agentos.acp.types import AcpSessionInitError

    runtime = FakeRuntime()

    async def _broken(spec, *, session_key):
        raise AcpSessionInitError(
            "ACP agent 'claude' needs claude on PATH", code="ACP_HARNESS_MISSING"
        )

    runtime.ensure_session = _broken  # type: ignore[method-assign]
    async with acp_rig(tmp_path, runtime) as rig:
        out = await _spawn(task="x", acp_agent="claude")
        child = out["session_key"]
        record = await rig.task_runtime.wait(out["task_id"], timeout=5)
        assert str(getattr(record.status, "value", record.status)) == "failed"
        transcript = await rig.sessions.read_recent_transcript(child, 10)
        assert [r["role"] for r in transcript] == ["user", "system"]
        assert transcript[-1]["content"] == "Error: ACP agent 'claude' needs claude on PATH"


async def test_user_typed_child_turn_does_not_wake_the_parent(tmp_path) -> None:
    """A turn the user types into the child (desktop sessions.send on the
    child key) runs the harness but never announces to / wakes the parent."""
    async with acp_rig(tmp_path) as rig:
        sent: list[str] = []
        original_send = rig.task_runtime.send

        async def _capture(session_key, message, provenance=None, stream_event_sink=None):
            if session_key == PARENT_KEY:
                sent.append(message)
                from types import SimpleNamespace

                return SimpleNamespace(task_id="wake")
            return await original_send(session_key, message, provenance=provenance)

        rig.task_runtime.send = _capture  # type: ignore[method-assign]
        out = await _spawn(task="once", acp_agent="claude")
        child = out["session_key"]
        await rig.task_runtime.wait(out["task_id"], timeout=5)
        assert len(await _completions(rig, 1)) == 1
        woken = len(sent)

        from agentos.gateway.rpc import get_dispatcher
        from agentos.gateway.rpc.registry import RpcContext

        ctx = RpcContext(
            conn_id="desk",
            config=rig.config,
            session_manager=rig.sessions,
            task_runtime=rig.task_runtime,
        )
        res = await get_dispatcher().dispatch(
            "1", "sessions.send", {"key": child, "message": "and one more thing"}, ctx
        )
        assert res.ok, res.error
        record = await rig.task_runtime.wait(res.payload["task_id"], timeout=5)
        assert str(getattr(record.status, "value", record.status)) == "succeeded"
        assert record.run_kind == "session_turn"
        await asyncio.sleep(0.05)

        assert len(await _completions(rig, 1)) == 1  # no second completion row
        assert len(sent) == woken
        completions = [p for k, n, p in rig.events if n == "session.event.subagent_completion"]
        assert len(completions) == 1
        transcript = await rig.sessions.read_recent_transcript(child, 10)
        assert transcript[-1]["content"] == "echo: and one more thing"


async def test_operator_cancel_ends_task_as_cancelled_not_failed(tmp_path) -> None:
    """``acp cancel`` (manager.cancel, not task_runtime.cancel) must not read as a
    failure to the parent: live testing showed the parent retrying the task."""
    runtime = FakeRuntime([Script(block=True)])
    async with acp_rig(tmp_path, runtime) as rig:
        out = await _spawn(task="long", acp_agent="claude", mode="persistent")
        await asyncio.wait_for(runtime.turn_started.wait(), timeout=5)
        await asyncio.sleep(0)
        assert await rig.manager.cancel(out["session_key"], reason="operator_cancel")
        record = await rig.task_runtime.wait(out["task_id"], timeout=5)
        assert str(getattr(record.status, "value", record.status)) == "cancelled"
        assert record.terminal_reason == "cancelled"
        errors = [p for k, n, p in rig.events if n == "session.event.error"]
        assert errors and errors[0]["code"] == "ACP_TURN_CANCELLED"
        assert errors[0]["terminal_reason"] == "cancelled"
