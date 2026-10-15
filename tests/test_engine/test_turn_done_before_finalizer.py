"""Clients see a turn finish before the finalizer's slow side effects run.

Every client treats the ``done`` event as "turn finished": the gateway's
``sessions.send`` task forwards it as ``session.event.done`` the moment it
comes off ``TurnRunner.run`` (``rpc_sessions._run``), the Web UI and desktop
end the stream on any ``*.done`` frame, and the CLI's gateway client stops on
``session.event.done``. The stream consumer stage yields that ``DoneEvent``
while the agent stream is still open, and ``TurnFinalizerStage`` (transcript
append, memory capture, cost rollup) only runs after the consumer has pulled
past it. So the finalizer is already off the "done" path; these tests pin it.

They drive the real ``TurnRunner`` with a real in-memory ``SessionManager`` and
record, in one sequence, the moment the consumer receives ``done`` and the
moment each finalizer port is entered.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentos.engine.runtime import TurnRunner
from agentos.engine.types import DoneEvent
from agentos.gateway.config import AgentOSRouterConfig, AttachmentsConfig, GatewayConfig
from agentos.provider import DoneEvent as ProviderDone
from agentos.provider import TextDeltaEvent as ProviderText
from agentos.session.manager import SessionManager
from agentos.session.storage import SessionStorage
from agentos.tools.types import CallerKind, ToolContext

KEY = "agent:main:webchat:done-before-finalizer"


class _ReplyProvider:
    provider_name = "test"

    def chat(self, messages: Any, tools: Any = None, config: Any = None) -> AsyncIterator[Any]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        yield ProviderText(text="the answer")
        yield ProviderDone(stop_reason="end_turn", input_tokens=5, output_tokens=2)

    async def list_models(self) -> list[Any]:
        return []


class _Selector:
    current_config = SimpleNamespace(model="test/model")

    def __init__(self, provider: _ReplyProvider) -> None:
        self.provider = provider

    def clone(self) -> _Selector:
        return self

    def resolve(self) -> _ReplyProvider:
        return self.provider

    def override_model(self, model: str) -> None:
        self.current_config = SimpleNamespace(model=model)


@pytest.fixture(autouse=True)
def _state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTOS_STATE_DIR", str(tmp_path))


@pytest.fixture
async def manager(tmp_path: Path) -> AsyncIterator[SessionManager]:
    storage = SessionStorage(":memory:")
    await storage.connect()
    yield SessionManager(storage, inject_time_prefix=False, checkpoint_workspace_dir=tmp_path)
    await storage.close()


def _instrument(runner: TurnRunner, order: list[str], *, fail: bool = False) -> None:
    stage = runner._turn_finalizer_stage
    append_port = stage._transcript_append
    capture_port = stage._turn_memory_capture
    totals_port = stage._session_totals
    real_append = append_port.append_message
    real_rollup = totals_port.rollup

    async def append_message(*args: Any, **kwargs: Any) -> bool:
        order.append("transcript_append")
        return await real_append(*args, **kwargs)

    async def capture_turn(**_kwargs: Any) -> None:
        order.append("capture_turn")
        if fail:
            raise RuntimeError("memory store is down")

    async def rollup(**kwargs: Any) -> Any:
        order.append("rollup")
        if fail:
            raise RuntimeError("totals write failed")
        return await real_rollup(**kwargs)

    append_port.append_message = append_message  # type: ignore[method-assign]
    capture_port.capture_turn = capture_turn  # type: ignore[method-assign]
    totals_port.rollup = rollup  # type: ignore[method-assign]


async def _run(runner: TurnRunner, manager: SessionManager, order: list[str]) -> list[Any]:
    await manager.create(KEY)
    await manager.append_message(KEY, "user", "what is the answer?")
    events: list[Any] = []
    async for event in runner.run(
        "what is the answer?",
        KEY,
        tool_context=ToolContext(caller_kind=CallerKind.WEB),
        history_has_persisted_user=True,
    ):
        if isinstance(event, DoneEvent):
            # What the gateway does here: emit session.event.done to clients.
            order.append("client_sees_done")
        events.append(event)
    return events


def _runner(manager: SessionManager, tmp_path: Path) -> TurnRunner:
    return TurnRunner(
        provider_selector=_Selector(_ReplyProvider()),
        session_manager=manager,
        config=GatewayConfig(
            attachments=AttachmentsConfig(media_root=str(tmp_path / "media")),
            agentos_router=AgentOSRouterConfig(enabled=False),
        ),
    )


@pytest.mark.asyncio
async def test_done_reaches_the_client_before_the_finalizer_runs(
    manager: SessionManager, tmp_path: Path
) -> None:
    runner = _runner(manager, tmp_path)
    order: list[str] = []
    _instrument(runner, order)

    await _run(runner, manager, order)

    assert order == ["client_sees_done", "transcript_append", "capture_turn", "rollup"]
    transcript = await manager.get_transcript(KEY)
    assert [entry.role for entry in transcript] == ["user", "assistant"]
    assert "the answer" in str(transcript[-1].content)


@pytest.mark.asyncio
async def test_memory_capture_and_rollup_failures_never_fail_the_turn(
    manager: SessionManager, tmp_path: Path
) -> None:
    runner = _runner(manager, tmp_path)
    order: list[str] = []
    _instrument(runner, order, fail=True)

    events = await _run(runner, manager, order)

    assert order == ["client_sees_done", "transcript_append", "capture_turn", "rollup"]
    assert not any(getattr(event, "kind", None) == "error" for event in events)
    transcript = await manager.get_transcript(KEY)
    assert [entry.role for entry in transcript] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_gateway_emits_session_event_done_before_the_turn_stream_closes() -> None:
    """The gateway side: ``session.event.done`` goes out as soon as ``done`` arrives,
    not when ``TurnRunner.run`` returns (which is after the finalizer)."""

    from agentos.gateway.boot import dispatch_task_runtime_turn
    from agentos.gateway.routing import RouteEnvelope, SourceKind

    order: list[str] = []

    class _Runner:
        async def run(self, message: str, session_key: str, **_kwargs: Any):  # noqa: ARG002
            yield DoneEvent(text="hi")
            order.append("finalizer")  # what TurnRunner does after yielding done

    async def _emit(_session_key: str, event_name: str, _payload: dict[str, Any]) -> None:
        order.append(event_name)

    envelope = RouteEnvelope(
        source_kind=SourceKind.WEB,
        source_name="test",
        agent_id="main",
        session_key=KEY,
        input_provenance={"kind": "web_message"},
        metadata={},
    )
    run = SimpleNamespace(
        agent_id="main",
        task_id="task-1",
        session_key=KEY,
        message="hi",
        envelope=envelope,
        attachments=[],
        input_provenance=envelope.input_provenance,
        run_kind="default",
        no_memory_capture=False,
        ingress_pipeline_steps=[],
        semantic_message=None,
        persisted_user_message_id=None,
        fresh_user_session=False,
        stream_event_sink=None,
    )
    await dispatch_task_runtime_turn(
        run,
        config=GatewayConfig(agent_stream_heartbeat_interval_seconds=0.0),
        session_manager=None,
        turn_runner=_Runner(),
        event_emitter=_emit,
    )

    assert order == ["session.event.done", "finalizer"]
