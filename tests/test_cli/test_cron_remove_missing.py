"""Issue #1598: ``agentos cron remove`` must not print a success payload for a
job the gateway says does not exist, and must not invent ``removed: true``.
"""

from __future__ import annotations

import json
from typing import Any

from typer.testing import CliRunner

from agentos.cli.main import app

runner = CliRunner()


class _Gateway:
    calls: list[tuple[str, Any]] = []
    present: set[str] = set()

    async def connect(self, url: str, *, token=None) -> None:
        pass

    async def close(self) -> None:
        pass

    async def call(self, method: str, params: dict | None = None) -> Any:
        from agentos.cli.gateway_client import GatewayRPCError

        type(self).calls.append((method, params or {}))
        assert method == "cron.remove"
        job_id = (params or {})["id"]
        if job_id not in type(self).present:
            raise GatewayRPCError(
                method, code="NOT_FOUND", message=f"'Cron job not found: {job_id}'"
            )
        type(self).present.discard(job_id)
        return {"id": job_id, "removed": True}


def _install(monkeypatch, *, present: set[str]) -> type[_Gateway]:
    from agentos.cli import gateway_client, gateway_rpc

    _Gateway.calls = []
    _Gateway.present = set(present)
    monkeypatch.setattr(gateway_client, "GatewayClient", _Gateway)
    monkeypatch.setattr(gateway_rpc, "_apply_version_skew_policy", lambda *a, **kw: None)
    monkeypatch.setattr(gateway_rpc, "_maybe_emit_update_notice", lambda *a, **kw: None)
    monkeypatch.setattr(gateway_rpc, "_target_gateway_url", lambda **kw: "ws://test")
    monkeypatch.setattr(gateway_rpc, "default_gateway_token", lambda *a, **kw: None)
    return _Gateway


def test_remove_missing_job_fails_like_status_does(monkeypatch) -> None:
    _install(monkeypatch, present={"real-job"})

    result = runner.invoke(app, ["cron", "remove", "missing-job-xyz", "--yes", "--json"])

    assert result.exit_code == 2, result.output
    assert result.stdout.strip() == "", result.stdout
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "NOT_FOUND"
    assert "Cron job not found: missing-job-xyz" in error["message"]


def test_remove_missing_job_human_mode_prints_no_success_table(monkeypatch) -> None:
    _install(monkeypatch, present=set())

    result = runner.invoke(app, ["cron", "remove", "missing-job-xyz", "--yes"])

    assert result.exit_code == 2, result.output
    assert "Cron job removed" not in result.output
    assert "removed" not in result.stdout.lower()
    assert "Cron job not found: missing-job-xyz" in result.output


def test_remove_existing_job_still_succeeds(monkeypatch) -> None:
    gw = _install(monkeypatch, present={"real-job"})

    result = runner.invoke(app, ["cron", "remove", "real-job", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"id": "real-job", "removed": True}
    assert gw.calls == [("cron.remove", {"id": "real-job"})]
    assert gw.present == set()
