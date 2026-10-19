"""``stop_admission`` closes the door on new turns while a restart drains."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentos.gateway.routing import RouteEnvelope, SourceKind
from agentos.gateway.task_runtime import TaskRuntime, TaskRuntimeNotAdmittingError


def _envelope(session_key: str) -> RouteEnvelope:
    return RouteEnvelope(
        source_kind=SourceKind.WEB,
        source_name="test",
        agent_id="a",
        session_key=session_key,
        input_provenance={"kind": "test"},
    )


def _runtime() -> TaskRuntime:
    storage: Any = MagicMock()
    storage.create_agent_task = AsyncMock()
    storage.update_agent_task = AsyncMock()
    storage.get_agent_task = AsyncMock(return_value=None)
    storage.list_agent_tasks = AsyncMock(return_value=[])

    async def turn_handler(_run: Any) -> None:
        await asyncio.sleep(0.001)

    return TaskRuntime(storage=storage, turn_handler=turn_handler, max_concurrency=1)


@pytest.mark.asyncio
async def test_enqueue_is_refused_after_stop_admission() -> None:
    runtime = _runtime()
    assert runtime.admitting is True

    runtime.stop_admission()

    assert runtime.admitting is False
    with pytest.raises(TaskRuntimeNotAdmittingError):
        await runtime.enqueue(_envelope("agent:a:web:s1"), "hello")


@pytest.mark.asyncio
async def test_resume_admission_accepts_work_again() -> None:
    runtime = _runtime()
    runtime.stop_admission()
    runtime.resume_admission()

    handle = await runtime.enqueue(_envelope("agent:a:web:s2"), "hello")

    assert handle.task_id
    await runtime.shutdown(cancel=True, timeout=1.0)
