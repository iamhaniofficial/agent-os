"""Gateway wiring: lazy runtime / bridge construction, degrade-to-unavailable,
lease reaping at boot, shutdown, and the dispatch discriminator."""

from __future__ import annotations

import types
from typing import Any

import pytest
from acp_s_fakes import FakeRuntime

from agentos.acp.manager import get_acp_manager, set_acp_manager
from agentos.gateway import acp_wiring as wiring
from agentos.gateway.config import AcpConfig, GatewayConfig


def _config(tmp_path, **acp: Any) -> GatewayConfig:
    return GatewayConfig(
        workspace_dir=str(tmp_path / "ws"), state_dir=str(tmp_path / "state"), acp=AcpConfig(**acp)
    )


def _fake_modules(monkeypatch: pytest.MonkeyPatch, modules: dict[str, Any]) -> None:
    real = wiring.importlib.import_module

    def _import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in modules:
            value = modules[name]
            if isinstance(value, BaseException):
                raise value
            return value
        return real(name, *args, **kwargs)

    monkeypatch.setattr(wiring.importlib, "import_module", _import)


def test_missing_runtime_degrades_to_unavailable(tmp_path, monkeypatch) -> None:
    _fake_modules(
        monkeypatch, {"agentos.acp.runtime": ModuleNotFoundError("No module named runtime")}
    )
    manager = wiring.build_acp_manager(_config(tmp_path), session_manager=None)
    assert not manager.available
    assert "runtime not installed" in (manager.unavailable_reason or "")


def test_disabled_config_is_unavailable(tmp_path) -> None:
    manager = wiring.build_acp_manager(_config(tmp_path, enabled=False), session_manager=None)
    assert not manager.available and "disabled" in (manager.unavailable_reason or "")


def test_runtime_gets_only_declared_kwargs(tmp_path, monkeypatch) -> None:
    seen: dict[str, Any] = {}

    class Runtime(FakeRuntime):
        def __init__(self, *, config, state_dir, permission_bridge=None, elicitation_bridge=None):
            super().__init__()
            seen.update(
                config=config,
                state_dir=state_dir,
                permission_bridge=permission_bridge,
                elicitation_bridge=elicitation_bridge,
            )

    class PermBridge:
        def __init__(self, queue, config):
            self.queue = queue
            self.config = config

    _fake_modules(
        monkeypatch,
        {
            "agentos.acp.runtime": types.SimpleNamespace(AcpRuntimeImpl=Runtime),
            "agentos.acp.permissions": types.SimpleNamespace(
                ApprovalQueuePermissionBridge=PermBridge
            ),
            "agentos.acp.elicitation": ModuleNotFoundError("not yet"),
        },
    )
    cfg = _config(tmp_path, state_dir=str(tmp_path / "custom-acp"))
    manager = wiring.build_acp_manager(cfg, session_manager=None)
    assert manager.available
    assert seen["config"] is cfg.acp
    assert str(seen["state_dir"]) == str(tmp_path / "custom-acp")
    assert isinstance(seen["permission_bridge"], PermBridge)
    assert seen["permission_bridge"].queue is not None
    assert seen["elicitation_bridge"] is None


def test_runtime_constructor_failure_degrades(tmp_path, monkeypatch) -> None:
    class Broken:
        def __init__(self, *, config):
            raise RuntimeError("bad argv")

    _fake_modules(
        monkeypatch, {"agentos.acp.runtime": types.SimpleNamespace(AcpRuntimeImpl=Broken)}
    )
    manager = wiring.build_acp_manager(_config(tmp_path), session_manager=None)
    assert not manager.available
    assert "bad argv" in (manager.unavailable_reason or "")


async def test_start_reaps_publishes_and_stop_shuts_down(tmp_path, monkeypatch) -> None:
    reaped: list[Any] = []

    async def reap_stale_leases(state_dir):
        reaped.append(state_dir)
        return 2

    runtime = FakeRuntime()
    _fake_modules(
        monkeypatch,
        {
            "agentos.acp.process": types.SimpleNamespace(reap_stale_leases=reap_stale_leases),
            "agentos.acp.runtime": types.SimpleNamespace(AcpRuntimeImpl=lambda: runtime),
        },
    )
    try:
        manager = await wiring.start_acp(_config(tmp_path), session_manager=None)
        assert manager is not None and manager.available
        assert get_acp_manager() is manager
        assert str(reaped[0]) == str(tmp_path / "state" / "acp")
        await wiring.stop_acp(manager)
        assert get_acp_manager() is None
        assert not manager.available
    finally:
        set_acp_manager(None)


async def test_reap_failure_never_blocks_boot(tmp_path, monkeypatch) -> None:
    def reap_stale_leases(state_dir):
        raise OSError("ps not found")

    _fake_modules(
        monkeypatch,
        {
            "agentos.acp.process": types.SimpleNamespace(reap_stale_leases=reap_stale_leases),
            "agentos.acp.runtime": ModuleNotFoundError("x"),
        },
    )
    try:
        assert await wiring.reap_stale_leases(_config(tmp_path)) == 0
        manager = await wiring.start_acp(_config(tmp_path), session_manager=None)
        assert manager is not None and not manager.available
    finally:
        set_acp_manager(None)


async def test_dispatch_routes_only_acp_origin(tmp_path) -> None:
    """Non-ACP rows still go to turn_runner.run; ACP rows never do."""
    from agentos.gateway.boot import _is_acp_session_row

    assert _is_acp_session_row(types.SimpleNamespace(origin={"kind": "acp"}))
    assert not _is_acp_session_row(types.SimpleNamespace(origin={"kind": "subagent"}))
    assert not _is_acp_session_row(types.SimpleNamespace(origin=None))
    assert not _is_acp_session_row(None)


async def test_service_container_close_stops_acp(tmp_path) -> None:
    from agentos.acp.manager import AcpSessionManager
    from agentos.gateway.boot import ServiceContainer

    manager = AcpSessionManager(FakeRuntime(), config=AcpConfig())
    set_acp_manager(manager)
    svc = ServiceContainer(config=_config(tmp_path))
    svc.acp_manager = manager
    await svc.close()
    assert not manager.available
    assert get_acp_manager() is None
    assert svc.acp_manager is None
