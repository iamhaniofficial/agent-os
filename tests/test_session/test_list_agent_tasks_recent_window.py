"""``list_agent_tasks`` returns the *newest* window of a session's ledger (#1805).

The query used to be ``ORDER BY created_at ASC ... LIMIT 100``: once a session
had more than 100 lifetime tasks, the page held the 100 *oldest* rows and the
in-flight task was never in it. ``sessions_yield`` picks ``rows[-1]`` as "the
latest task" and so resolved immediately on a long-finished task, and the
session reset/delete sweep never found the running task to cancel or drain.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from agentos.gateway.task_runtime import TaskRuntime
from agentos.session.models import AgentTaskRecord, AgentTaskStatus
from agentos.session.storage import SessionStorage

KEY = "agent:main:webchat:long-lived"


@pytest_asyncio.fixture
async def storage():
    store = SessionStorage(":memory:")
    await store.connect()
    yield store
    await store.close()


async def _seed(storage: SessionStorage, count: int, *, running_last: bool = True) -> None:
    for i in range(count):
        is_last = i == count - 1
        await storage.create_agent_task(
            AgentTaskRecord(
                task_id=f"task-{i:03d}",
                session_key=KEY,
                status=(
                    AgentTaskStatus.RUNNING
                    if is_last and running_last
                    else AgentTaskStatus.SUCCEEDED
                ),
                created_at=1000 + i,
                updated_at=1000 + i,
            )
        )


@pytest.mark.asyncio
async def test_default_window_holds_the_newest_tasks_in_chronological_order(storage) -> None:
    await _seed(storage, 105)

    rows = await storage.list_agent_tasks(session_key=KEY)

    assert len(rows) == 100
    assert rows[0].task_id == "task-005"
    assert rows[-1].task_id == "task-104"
    assert rows[-1].status == AgentTaskStatus.RUNNING
    assert [r.created_at for r in rows] == sorted(r.created_at for r in rows)


@pytest.mark.asyncio
async def test_small_ledger_is_returned_whole_and_ascending(storage) -> None:
    await _seed(storage, 3)

    rows = await storage.list_agent_tasks(session_key=KEY)

    assert [r.task_id for r in rows] == ["task-000", "task-001", "task-002"]


@pytest.mark.asyncio
async def test_offset_pages_backwards_from_the_newest(storage) -> None:
    await _seed(storage, 7)

    newest = await storage.list_agent_tasks(session_key=KEY, limit=3)
    older = await storage.list_agent_tasks(session_key=KEY, limit=3, offset=3)
    oldest = await storage.list_agent_tasks(session_key=KEY, limit=3, offset=6)

    assert [r.task_id for r in newest] == ["task-004", "task-005", "task-006"]
    assert [r.task_id for r in older] == ["task-001", "task-002", "task-003"]
    assert [r.task_id for r in oldest] == ["task-000"]


@pytest.mark.asyncio
async def test_status_filter_still_applies_inside_the_window(storage) -> None:
    await _seed(storage, 105)

    running = await storage.list_agent_tasks(session_key=KEY, status=AgentTaskStatus.RUNNING)

    assert [r.task_id for r in running] == ["task-104"]


@pytest.mark.asyncio
async def test_same_timestamp_ties_keep_insertion_order(storage) -> None:
    for i in range(3):
        await storage.create_agent_task(
            AgentTaskRecord(task_id=f"tie-{i}", session_key=KEY, created_at=500, updated_at=500)
        )

    rows = await storage.list_agent_tasks(session_key=KEY)

    assert [r.task_id for r in rows] == ["tie-0", "tie-1", "tie-2"]


@pytest.mark.asyncio
async def test_task_runtime_list_sees_the_running_task_past_100_turns(storage) -> None:
    await _seed(storage, 105)

    async def _noop(task: AgentTaskRecord) -> None:  # pragma: no cover - never run
        return None

    runtime = TaskRuntime(storage=storage, turn_handler=_noop)
    rows = await runtime.list(session_key=KEY)

    assert rows[-1].task_id == "task-104"
    assert any(r.status == AgentTaskStatus.RUNNING for r in rows)
