"""``subagents(action="list")`` must delegate the ``spawned_by`` filter to
storage (#1799).

It used to call ``mgr.list_sessions()`` with the default ``limit=100`` and
filter ``spawned_by`` in Python, so once a gateway held more than 100 sessions
any subagent older than the 100 most recently updated rows silently vanished
from its parent's list.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agentos.tools.builtin import agents as agents_tool
from agentos.tools.builtin import sessions as sessions_tool
from agentos.tools.types import CallerKind, ToolContext, current_tool_context

PARENT = "agent:main:parent"


class _SqliteLikeSessionManager:
    """Applies ``spawned_by`` / ``limit`` / ``offset`` the way the SQLite store
    does: filter first, then order by recency and page."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.calls: list[dict[str, Any]] = []

    async def get_current_session(self) -> None:
        return None

    async def list_sessions(
        self,
        agent_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        spawned_by: str | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append({"limit": limit, "offset": offset, "spawned_by": spawned_by})
        rows = self._rows
        if spawned_by is not None:
            rows = [r for r in rows if r.get("spawned_by") == spawned_by]
        rows = sorted(rows, key=lambda r: r["updated_at"], reverse=True)
        return rows[offset : offset + limit]


def _row(key: str, *, updated_at: int, spawned_by: str | None = None) -> dict[str, Any]:
    return {
        "session_key": key,
        "spawned_by": spawned_by,
        "status": "running",
        "updated_at": updated_at,
    }


async def _list_as(parent: str) -> list[str]:
    token = current_tool_context.set(
        ToolContext(caller_kind=CallerKind.AGENT, session_key=parent, agent_id="main")
    )
    try:
        payload = json.loads(await agents_tool.subagents("list"))
    finally:
        current_tool_context.reset(token)
    return [s["session_key"] for s in payload["subagents"]]


@pytest.fixture
def _install(request: pytest.FixtureRequest):
    def install(mgr: _SqliteLikeSessionManager) -> _SqliteLikeSessionManager:
        sessions_tool.set_session_manager(mgr)
        sessions_tool.set_task_runtime(None)
        request.addfinalizer(lambda: sessions_tool.set_session_manager(None))
        request.addfinalizer(lambda: sessions_tool.set_task_runtime(None))
        return mgr

    return install


@pytest.mark.asyncio
async def test_list_finds_a_subagent_buried_under_100_newer_sessions(_install) -> None:
    child = _row("subagent:child", updated_at=0, spawned_by=PARENT)
    noise = [_row(f"agent:main:other-{i}", updated_at=i + 1) for i in range(150)]
    mgr = _install(_SqliteLikeSessionManager([child, *noise]))

    assert await _list_as(PARENT) == ["subagent:child"]
    assert all(call["spawned_by"] == PARENT for call in mgr.calls)


@pytest.mark.asyncio
async def test_list_returns_only_the_callers_children(_install) -> None:
    rows = [
        _row("subagent:mine-1", updated_at=3, spawned_by=PARENT),
        _row("subagent:theirs", updated_at=2, spawned_by="agent:main:someone-else"),
        _row("subagent:mine-2", updated_at=1, spawned_by=PARENT),
        _row("agent:main:legacy", updated_at=0),
    ]
    _install(_SqliteLikeSessionManager(rows))

    assert await _list_as(PARENT) == ["subagent:mine-1", "subagent:mine-2"]


@pytest.mark.asyncio
async def test_list_pages_past_a_single_storage_page(_install) -> None:
    children = [_row(f"subagent:c{i:03d}", updated_at=i, spawned_by=PARENT) for i in range(250)]
    mgr = _install(_SqliteLikeSessionManager(children))

    listed = await _list_as(PARENT)

    assert len(listed) == 250
    assert len(set(listed)) == 250
    assert [call["offset"] for call in mgr.calls] == [0, 100, 200]


@pytest.mark.asyncio
async def test_list_does_not_repeat_a_child_that_straddles_a_page_boundary(_install) -> None:
    """Children bump ``updated_at`` while the listing pages, so the row at a
    page boundary can be served on two consecutive pages; it must appear once.
    (The bumped child itself jumps onto the already-read page and is missed
    this time round -- an inherent limit of offset paging, not a duplicate.)
    """
    children = [_row(f"subagent:c{i:03d}", updated_at=i, spawned_by=PARENT) for i in range(150)]
    mgr = _install(_SqliteLikeSessionManager(children))
    real_list = mgr.list_sessions

    async def list_with_shifting_rows(**kwargs: Any) -> list[dict[str, Any]]:
        rows = await real_list(**kwargs)
        if kwargs.get("offset") == 0:
            # After the first page is read, the oldest child becomes the newest.
            children[0]["updated_at"] = 10_000
        return rows

    mgr.list_sessions = list_with_shifting_rows  # type: ignore[method-assign]

    listed = await _list_as(PARENT)

    # c050 was the last row of page one and, after the shift, the first of
    # page two.
    assert listed.count("subagent:c050") == 1
    assert len(listed) == len(set(listed))


def _rich_row(index: int) -> dict[str, Any]:
    """A real session row: origin carries the child's whole grounded task."""
    return {
        "session_key": f"agent:research:subagent:{index:08x}",
        "spawned_by": PARENT,
        "status": "done" if index else "running",
        "agent_id": "research",
        "label": None,
        "display_name": f"Research the market for widget {index} in detail",
        "started_at": 1_760_000_000_000 + index,
        "ended_at": None,
        "updated_at": index,
        "model": "some/model",
        "origin": {
            "kind": "subagent",
            "parent_session_key": PARENT,
            "task": f"Find every supplier of widget {index}. " * 20,
            "execution_task": "You are a subagent. " * 300,
        },
    }


@pytest.mark.asyncio
async def test_list_rows_are_compact_and_survive_the_transcript_cap(_install) -> None:
    """``subagents(list)`` came back as ``"subagents": []`` with
    ``result_truncated``: each raw row was ~2 kB and the 2,000-char transcript
    cap dropped them all."""
    from agentos.engine.runtime import _persisted_tool_result_segment
    from agentos.engine.types import ToolResultEvent

    _install(_SqliteLikeSessionManager([_rich_row(i) for i in range(5)]))
    token = current_tool_context.set(
        ToolContext(caller_kind=CallerKind.AGENT, session_key=PARENT, agent_id="main")
    )
    try:
        raw = await agents_tool.subagents("list")
    finally:
        current_tool_context.reset(token)
    payload = json.loads(raw)
    assert payload["total"] == 5
    first = payload["subagents"][0]
    assert first["status"] == "running"  # active children first
    assert set(first) <= {
        "session_key",
        "runtime",
        "status",
        "agent_id",
        "label",
        "started_at",
        "ended_at",
        "task",
    }
    assert len(first["task"]) <= 80 and "execution_task" not in raw
    assert len(raw) < 2000  # a normal list fits whole

    # A long list is cut from the end, never to nothing.
    _install(_SqliteLikeSessionManager([_rich_row(i) for i in range(40)]))
    token = current_tool_context.set(
        ToolContext(caller_kind=CallerKind.AGENT, session_key=PARENT, agent_id="main")
    )
    try:
        long_raw = await agents_tool.subagents("list")
    finally:
        current_tool_context.reset(token)
    assert len(long_raw) > 2000
    segment = _persisted_tool_result_segment(
        ToolResultEvent(tool_use_id="t", tool_name="subagents", result=long_raw),
        max_chars=2000,
    )
    kept = json.loads(segment["result"])
    assert segment["result_truncated"] is True
    assert kept["subagents"], "the transcript cap must not empty the list"
    assert kept["total"] == 40
