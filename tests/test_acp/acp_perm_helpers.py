"""Shared helpers for the ACP permission / elicitation tests (stream P)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agentos.acp.types import PermissionOptionInfo, PermissionRequest
from agentos.application.approval_queue import ApprovalQueue

STANDARD_OPTIONS = [
    PermissionOptionInfo("allow-once", "Allow once", "allow_once"),
    PermissionOptionInfo("allow-always", "Always allow", "allow_always"),
    PermissionOptionInfo("reject-once", "Reject", "reject_once"),
]


def make_queue(tmp_path: Path, **kwargs: Any) -> ApprovalQueue:
    kwargs.setdefault("poll_interval", 0.01)
    return ApprovalQueue(db_path=str(tmp_path / "approvals.sqlite"), **kwargs)


def perm_request(
    *,
    session_key: str = "agent:main:acp:0123abcd",
    tool_kind: str | None = "execute",
    title: str = "git status",
    raw_input: dict[str, Any] | None = None,
    options: list[PermissionOptionInfo] | None = None,
    agent_id: str = "codex",
) -> PermissionRequest:
    return PermissionRequest(
        session_key=session_key,
        acp_session_id="sess-1",
        agent_id=agent_id,
        tool_call_id="call-1",
        title=title,
        tool_kind=tool_kind,
        raw_input=raw_input if raw_input is not None else {"command": title},
        options=list(STANDARD_OPTIONS if options is None else options),
        cwd="/work/repo",
    )


async def next_pending(
    queue: ApprovalQueue, namespace: str, *, seen: set[str] | None = None
) -> dict[str, Any]:
    """Wait (bounded) for a new pending row in ``namespace``."""
    seen = seen if seen is not None else set()
    for _ in range(400):
        for row in queue.list_pending(namespace):
            if row["id"] not in seen:
                seen.add(row["id"])
                return row
        await asyncio.sleep(0.005)
    raise AssertionError(f"no pending {namespace!r} approval appeared")
