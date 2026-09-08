"""The freshness probe on the send path reads at most one transcript row.

``_persist_user_message`` only needs to know whether a session has any
history yet. Reading the whole transcript to answer that made every user
message cost O(conversation length) — see issue #1368.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentos.compat.inspect_utils import accepts_keyword_arg
from agentos.gateway.rpc_sessions import _probe_fresh_user_session
from agentos.session.manager import SessionManager


class _BoundedManager:
    """Mirrors the real ``SessionManager.get_transcript`` signature."""

    def __init__(self, entries: list[str]) -> None:
        self._entries = entries
        self.calls: list[int | None] = []

    async def get_transcript(self, session_key: str, limit: int | None = None) -> list[str]:
        self.calls.append(limit)
        rows = self._entries
        return list(rows if limit is None else rows[:limit])


class _KwargsManager:
    """A manager that swallows keywords it does not name explicitly."""

    def __init__(self, entries: list[str]) -> None:
        self._entries = entries
        self.calls: list[dict[str, Any]] = []

    async def get_transcript(self, session_key: str, **kwargs: Any) -> list[str]:
        self.calls.append(dict(kwargs))
        return list(self._entries)


class _KeyOnlyManager:
    """A manager that takes the key alone — ``limit=`` would be a TypeError."""

    def __init__(self, entries: list[str]) -> None:
        self._entries = entries
        self.calls = 0

    async def get_transcript(self, session_key: str) -> list[str]:
        self.calls += 1
        return list(self._entries)


@pytest.mark.asyncio
async def test_bounded_manager_is_asked_for_one_row_only() -> None:
    manager = _BoundedManager(["a", "b", "c"])

    fresh = await _probe_fresh_user_session(manager.get_transcript, "agent:main:webchat:x")

    assert fresh is False
    assert manager.calls == [1]


@pytest.mark.asyncio
async def test_bounded_manager_reports_empty_session_as_fresh() -> None:
    manager = _BoundedManager([])

    fresh = await _probe_fresh_user_session(manager.get_transcript, "agent:main:webchat:x")

    assert fresh is True
    assert manager.calls == [1]


@pytest.mark.asyncio
async def test_var_keyword_manager_still_receives_the_bound() -> None:
    manager = _KwargsManager(["a"])

    fresh = await _probe_fresh_user_session(manager.get_transcript, "agent:main:webchat:x")

    assert fresh is False
    assert manager.calls == [{"limit": 1}]


@pytest.mark.asyncio
@pytest.mark.parametrize(("entries", "expected"), [([], True), (["a"], False)])
async def test_key_only_manager_is_called_without_a_bound(
    entries: list[str], expected: bool
) -> None:
    manager = _KeyOnlyManager(entries)

    fresh = await _probe_fresh_user_session(manager.get_transcript, "agent:main:webchat:x")

    assert fresh is expected
    assert manager.calls == 1


def test_real_session_manager_takes_the_bounded_path() -> None:
    """Pins the production manager on the one-row branch, not the fallback."""

    assert accepts_keyword_arg(SessionManager.get_transcript, "limit") is True
