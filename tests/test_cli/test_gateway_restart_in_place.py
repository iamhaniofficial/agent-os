"""``agentos gateway restart`` on a healthy gateway the CLI did not start.

Such a gateway (desktop app, launchd, systemd, a terminal) is ``unmanaged``
to the pidfile path, which used to refuse with exit 3. It can restart itself
in place over ``gateway.restart``; the CLI confirms it by a changed
``start_ts`` in ``gateway.pid`` plus a readiness probe.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from agentos.cli import gateway_cmd, gateway_lifecycle
from agentos.cli.main import app
from agentos.paths import default_agentos_home

runner = CliRunner()
Manager = gateway_lifecycle.GatewayLifecycleManager


def _write_pid_record(start_ts: str, pid: int = 4242) -> None:
    path = default_agentos_home() / "state" / "gateway.pid"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": pid, "start_ts": start_ts}), encoding="utf-8")


def _payload(result: Any) -> dict[str, Any]:
    return json.loads(result.stdout)


@pytest.fixture
def unmanaged_gateway(tmp_path, monkeypatch) -> list[dict[str, Any]]:
    """A healthy gateway with no lifecycle record: ``status`` says unmanaged."""
    monkeypatch.setenv("AGENTOS_STATE_DIR", str(tmp_path / "home"))
    _write_pid_record("2026-10-19T00:00:00+00:00")
    monkeypatch.setattr(Manager, "_probe_health", lambda self: True)
    monkeypatch.setattr(Manager, "_probe_ready", lambda self: True)

    def fail_pidfile_restart(self):
        raise AssertionError("the pidfile path must not run for an unmanaged gateway")

    monkeypatch.setattr(Manager, "restart", fail_pidfile_restart)
    calls: list[dict[str, Any]] = []
    return calls


def _accepting_rpc(calls: list[dict[str, Any]], *, comes_back: bool = True):
    def fake_call(*, gateway_url: str, config_path: str | None, mode: str) -> dict[str, Any]:
        calls.append({"gateway_url": gateway_url, "config_path": config_path, "mode": mode})
        if comes_back:
            # The re-executed image wrote a fresh record under the same pid.
            _write_pid_record("2026-10-19T00:01:00+00:00")
        return {"accepted": True, "restartId": "r1", "mode": mode, "pid": 4242}

    return fake_call


def test_unmanaged_gateway_restarts_in_place(unmanaged_gateway, monkeypatch) -> None:
    monkeypatch.setattr(gateway_cmd, "_call_gateway_restart", _accepting_rpc(unmanaged_gateway))

    result = runner.invoke(app, ["gateway", "restart", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = _payload(result)
    assert payload["state"] == "running"
    assert payload["managed"] is False
    assert payload["pid"] == 4242
    assert payload["message"] == "Gateway restarted in place."
    assert payload["details"]["restartId"] == "r1"
    assert payload["details"]["inPlace"] is True
    assert unmanaged_gateway == [
        {"gateway_url": "ws://127.0.0.1:18791/ws", "config_path": None, "mode": "safe"}
    ]


def test_the_rpc_targets_the_requested_port_not_the_default(unmanaged_gateway, monkeypatch) -> None:
    """``--port 18999`` must restart the gateway on 18999, never the CLI's
    default endpoint (which may be a different, live gateway)."""
    monkeypatch.setattr(gateway_cmd, "_call_gateway_restart", _accepting_rpc(unmanaged_gateway))

    result = runner.invoke(app, ["gateway", "restart", "--port", "18999", "--json"])

    assert result.exit_code == 0, result.stdout
    assert unmanaged_gateway[0]["gateway_url"] == "ws://127.0.0.1:18999/ws"


def test_force_flag_cancels_active_turns(unmanaged_gateway, monkeypatch) -> None:
    monkeypatch.setattr(gateway_cmd, "_call_gateway_restart", _accepting_rpc(unmanaged_gateway))

    result = runner.invoke(app, ["gateway", "restart", "--force", "--json"])

    assert result.exit_code == 0, result.stdout
    assert unmanaged_gateway[0]["mode"] == "force"


def test_a_refused_rpc_keeps_the_unmanaged_outcome(unmanaged_gateway, monkeypatch) -> None:
    def refuse(*, gateway_url: str, config_path: str | None, mode: str) -> dict[str, Any]:
        raise gateway_cmd._RestartRpcRefusedError(
            "RESTART_UNSUPPORTED",
            "In-place restart is not supported on Windows; run 'agentos gateway restart'.",
        )

    monkeypatch.setattr(gateway_cmd, "_call_gateway_restart", refuse)

    result = runner.invoke(app, ["gateway", "restart", "--json"])

    assert result.exit_code == 3
    payload = _payload(result)
    assert payload["state"] == "unmanaged"
    assert payload["code"] == "UNMANAGED_GATEWAY_RUNNING"
    assert payload["details"]["rpc"]["code"] == "RESTART_UNSUPPORTED"
    assert "Windows" in payload["message"]


def test_an_unreachable_gateway_is_reported_not_crashed(unmanaged_gateway, monkeypatch) -> None:
    def unreachable(*, gateway_url: str, config_path: str | None, mode: str) -> dict[str, Any]:
        raise gateway_cmd._RestartRpcRefusedError("GATEWAY_UNAVAILABLE", "connection refused")

    monkeypatch.setattr(gateway_cmd, "_call_gateway_restart", unreachable)

    result = runner.invoke(app, ["gateway", "restart", "--json"])

    assert result.exit_code == 3
    assert _payload(result)["details"]["rpc"]["code"] == "GATEWAY_UNAVAILABLE"


def test_accepted_but_not_back_in_time_exits_one(unmanaged_gateway, monkeypatch) -> None:
    monkeypatch.setattr(
        gateway_cmd, "_call_gateway_restart", _accepting_rpc(unmanaged_gateway, comes_back=False)
    )

    result = runner.invoke(
        app, ["gateway", "restart", "--timeout", "0", "--shutdown-timeout", "0", "--json"]
    )

    assert result.exit_code == 1
    payload = _payload(result)
    assert payload["state"] == "restart_timeout"
    assert payload["code"] == "RESTART_TIMEOUT"
    assert payload["pid"] == 4242


def test_a_managed_gateway_keeps_the_pidfile_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTOS_STATE_DIR", str(tmp_path / "home"))
    record_path = gateway_lifecycle.gateway_pidfile_path()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps({"pid": 777, "host": "127.0.0.1", "port": 18791}), encoding="utf-8"
    )
    monkeypatch.setattr(Manager, "_pid_running", lambda self, pid: True)
    monkeypatch.setattr(Manager, "_probe_health", lambda self: True)
    restarts: list[str] = []

    def fake_restart(self):
        restarts.append("pidfile")
        return self._result("restart", "running", pid=777, managed=True)

    monkeypatch.setattr(Manager, "restart", fake_restart)

    def fail_rpc(**kwargs):
        raise AssertionError("a managed gateway must not be restarted over RPC")

    monkeypatch.setattr(gateway_cmd, "_call_gateway_restart", fail_rpc)

    result = runner.invoke(app, ["gateway", "restart", "--json"])

    assert result.exit_code == 0, result.stdout
    assert restarts == ["pidfile"]
    assert _payload(result)["managed"] is True
