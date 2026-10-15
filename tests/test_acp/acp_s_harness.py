"""A gateway-shaped test rig for ACP spawn / dispatch tests.

Real ``SessionManager`` + ``SessionStorage(":memory:")`` + real
``TaskRuntime`` whose turn handler is ``gateway.boot.dispatch_task_runtime_turn``
and whose terminal listener is the real subagent announce, with the
in-memory ``FakeRuntime`` behind a real ``AcpSessionManager``. Only the
harness itself is fake.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from acp_s_fakes import FakeRuntime

from agentos.acp.manager import AcpSessionManager, set_acp_manager
from agentos.gateway.config import AcpConfig, GatewayConfig
from agentos.session.manager import SessionManager
from agentos.session.storage import SessionStorage
from agentos.tools.builtin import sessions as sessions_tool
from agentos.tools.types import CallerKind, ToolContext, current_tool_context

PARENT_KEY = "agent:main:main"


class _NoTurnRunner:
    calls = 0

    def run(self, *args: Any, **kwargs: Any) -> Any:
        type(self).calls += 1
        raise AssertionError("ACP sessions must not reach turn_runner.run")


class Rig:
    def __init__(self, tmp: Path, runtime: FakeRuntime, **acp: Any) -> None:
        self.tmp = tmp
        self.workspace = tmp / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime
        self.config = GatewayConfig(
            workspace_dir=str(self.workspace),
            state_dir=str(tmp / "state"),
            acp=AcpConfig(**acp),
        )
        self.events: list[tuple[str, str, dict[str, Any]]] = []
        self.sessions: SessionManager
        self.task_runtime: Any
        self.manager: AcpSessionManager
        self.turn_runner = _NoTurnRunner()

    async def emit(self, session_key: str, event: str, payload: dict[str, Any]) -> None:
        self.events.append((session_key, event, payload))

    def event_names(self, session_key: str) -> list[str]:
        return [name for key, name, _ in self.events if key == session_key]

    def ctx(self, *, task_id: str = "task-parent", agent_id: str = "main") -> ToolContext:
        return ToolContext(
            caller_kind=CallerKind.AGENT,
            subagent_depth=0,
            agent_id=agent_id,
            session_key=PARENT_KEY,
            task_id=task_id,
            workspace_dir=str(self.workspace),
        )


@asynccontextmanager
async def acp_rig(tmp: Path, runtime: FakeRuntime | None = None, **acp: Any):
    from agentos.gateway.boot import (
        _make_task_session_lifecycle_listener,
        dispatch_task_runtime_turn,
    )
    from agentos.gateway.subagent_announce import announce_subagent_completion
    from agentos.gateway.task_runtime import TaskRuntime

    rig = Rig(tmp, runtime or FakeRuntime(), **acp)
    store = SessionStorage(":memory:")
    await store.connect()
    rig.sessions = SessionManager(store, inject_time_prefix=False)
    rig.manager = AcpSessionManager(
        rig.runtime, config=rig.config.acp, session_manager=rig.sessions
    )

    async def _handler(run: Any) -> None:
        await dispatch_task_runtime_turn(
            run,
            config=rig.config,
            session_manager=rig.sessions,
            turn_runner=rig.turn_runner,
            event_emitter=rig.emit,
        )

    async def _terminal(event: Any) -> None:
        await announce_subagent_completion(
            event,
            session_manager=rig.sessions,
            event_emitter=rig.emit,
            task_runtime=rig.task_runtime,
        )

    rig.task_runtime = TaskRuntime(
        storage=store,
        turn_handler=_handler,
        event_emitter=rig.emit,
        terminal_listener=_terminal,
        lifecycle_listener=_make_task_session_lifecycle_listener(
            session_manager=rig.sessions, event_emitter=rig.emit
        ),
        max_concurrency=4,
    )
    await rig.sessions.create(PARENT_KEY, agent_id="main")
    sessions_tool.set_session_manager(rig.sessions)
    sessions_tool.set_task_runtime(rig.task_runtime)
    sessions_tool.set_gateway_config(rig.config)
    set_acp_manager(rig.manager)
    token = current_tool_context.set(rig.ctx())
    try:
        yield rig
    finally:
        current_tool_context.reset(token)
        try:
            await rig.task_runtime.shutdown(timeout=1.0)
        except Exception:
            pass
        await rig.manager.shutdown()
        set_acp_manager(None)
        sessions_tool.set_session_manager(None)
        sessions_tool.set_task_runtime(None)
        sessions_tool.set_gateway_config(None)
        await store.close()
