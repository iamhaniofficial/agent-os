"""Issue #1598: ``cron.remove`` must not report success for a missing job.

``scheduler.remove_job`` already returns ``False`` when nothing was deleted;
the handler used to discard that bool and answer with a plain RPC success.
It now raises ``KeyError`` like its ``cron.status`` / ``cron.update``
siblings (which the dispatcher maps to ``NOT_FOUND``) and returns the real
outcome on the happy path.
"""

from __future__ import annotations

import asyncio

import pytest

from agentos.gateway.rpc import RpcContext
from agentos.gateway.rpc_cron import _d, _handle_cron_remove


class _FakeScheduler:
    def __init__(self, *, present: set[str]) -> None:
        self.present = set(present)
        self.removed: list[str] = []

    async def remove_job(self, job_id: str) -> bool:
        self.removed.append(job_id)
        if job_id in self.present:
            self.present.discard(job_id)
            return True
        return False


def test_remove_missing_job_raises_not_found() -> None:
    scheduler = _FakeScheduler(present={"real-job"})

    with pytest.raises(KeyError, match="Cron job not found: missing-job-xyz"):
        asyncio.run(
            _handle_cron_remove(
                {"id": "missing-job-xyz"},
                RpcContext(conn_id="test", cron_scheduler=scheduler),
            )
        )

    assert scheduler.removed == ["missing-job-xyz"]
    assert scheduler.present == {"real-job"}


def test_remove_existing_job_reports_the_real_outcome() -> None:
    scheduler = _FakeScheduler(present={"real-job"})

    result = asyncio.run(
        _handle_cron_remove(
            {"id": "real-job"},
            RpcContext(conn_id="test", cron_scheduler=scheduler),
        )
    )

    assert result == {"id": "real-job", "removed": True}
    assert scheduler.present == set()


def test_remove_missing_job_surfaces_as_not_found_over_the_wire() -> None:
    """Through the dispatcher, so the CLI and web UI see the same shape as
    ``cron.status`` / ``cron.update`` already produce for a missing id."""
    scheduler = _FakeScheduler(present=set())
    ctx = RpcContext(conn_id="test", cron_scheduler=scheduler)

    frame = asyncio.run(_d.dispatch("req-1", "cron.remove", {"id": "missing-job-xyz"}, ctx))

    assert frame.ok is False
    assert frame.error is not None
    assert frame.error.code == "NOT_FOUND"
    assert "Cron job not found: missing-job-xyz" in frame.error.message
