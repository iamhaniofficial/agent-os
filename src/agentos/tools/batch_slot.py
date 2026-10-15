"""Let a tool give back its batch slot while it waits for an operator.

``ToolBatchExecutor`` (``agentos.engine.tool_batch``) runs each tool call
under an admission: an ``exclusive`` gate slot, or a key lock plus a
concurrency permit. A tool that waits inline for an approval would hold that
admission for as long as the operator takes, so the next call of the batch
could not even ask for its own approval. :func:`wait_outside_tool_slot` frees
the admission for the duration of such a wait and queues for it again before
returning.

This module has no engine imports so tools can use it without a cycle; the
executor installs the current call's admission in :data:`current_admission`.
"""

from __future__ import annotations

from collections.abc import Awaitable
from contextvars import ContextVar
from typing import Protocol


class BatchAdmission(Protocol):
    """What the executor exposes to a running call."""

    @property
    def yielded(self) -> bool: ...

    async def wait_outside[T](self, awaitable: Awaitable[T]) -> T: ...


current_admission: ContextVar[BatchAdmission | None] = ContextVar(
    "tool_batch_admission", default=None
)


async def wait_outside_tool_slot[T](awaitable: Awaitable[T]) -> T:
    """Await *awaitable* without holding this tool call's batch admission.

    Meant for a wait on an operator approval: while the operator decides, the
    call's ``exclusive`` slot (or its key lock and permit) is free for the rest
    of the batch, so the other calls can request their approvals at once. The
    call is admitted again under its declared concurrency before this returns,
    so whatever the tool runs next still respects it. Outside a batch (direct
    tool calls, tests) this is a plain ``await``.

    Do not hold any other lock across this call that a later call of the same
    batch could need.
    """

    admission = current_admission.get()
    if admission is None or admission.yielded:
        return await awaitable
    return await admission.wait_outside(awaitable)
