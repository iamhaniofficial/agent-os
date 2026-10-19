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
        assert row.origin["closed"] is True  # oneshot by default
        assert row.origin["last_turn"]["input_tokens"] == 1200

        completions = await _completions(rig)
        assert completions and completions[-1]["child_session_key"] == child
        stats = completions[-1]["stats"]
        assert stats.startswith("runtime ") and "tokens 1500 (in 1200 / out 300)" in stats
        assert "cost $0.0300" in stats
        assert any(n == "session.event.subagent_completion" for _, n, _ in rig.events)


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


async def test_oneshot_followup_is_refused(tmp_path) -> None:
    from agentos.tools.types import SafeToolError

    async with acp_rig(tmp_path) as rig:
        out = await _spawn(task="once", acp_agent="claude")
        await rig.task_runtime.wait(out["task_id"], timeout=5)
        try:
            await sessions_tool.sessions_send(out["session_key"], "more")
        except SafeToolError as exc:
            assert "closed" in str(exc) or "oneshot" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected refusal")


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
