"""Gateway boot / shutdown wiring for the ACP consumer.

The concrete runtime (``agentos.acp.runtime.AcpRuntimeImpl``) and the
permission / elicitation bridges (``agentos.acp.permissions`` /
``agentos.acp.elicitation``) are imported lazily, and their constructors are
called with only the keyword arguments they declare. Anything missing or
failing degrades to "ACP unavailable" with a clear reason; the gateway
always boots.
"""

from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
from typing import Any

import structlog

from agentos.acp.manager import AcpSessionManager, get_acp_manager, set_acp_manager

log = structlog.get_logger("agentos.gateway.acp_wiring")


def acp_config_of(gateway_config: Any) -> Any:
    cfg = getattr(gateway_config, "acp", None) if gateway_config is not None else None
    if cfg is None:
        from agentos.gateway.config import AcpConfig

        cfg = AcpConfig()
    return cfg


def acp_state_dir(gateway_config: Any) -> Path:
    """``[acp].state_dir`` or ``<gateway state>/acp``."""
    acp_cfg = acp_config_of(gateway_config)
    override = getattr(acp_cfg, "state_dir", None)
    if override:
        return Path(os.path.expanduser(str(override)))
    state = getattr(gateway_config, "state_dir", None) if gateway_config is not None else None
    if state:
        return Path(os.path.expanduser(str(state))) / "acp"
    from agentos.paths import state_dir

    return state_dir("acp")


def _construct(factory: Any, candidates: dict[str, Any]) -> Any:
    """Call ``factory`` with the subset of ``candidates`` it accepts by name."""
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return factory()
    named = (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    kwargs = {
        name: value
        for name, value in candidates.items()
        if name in params and params[name].kind in named
    }
    return factory(**kwargs)


def _approval_queue() -> Any:
    try:
        from agentos.application.approval_queue import get_approval_queue

        return get_approval_queue()
    except Exception:  # noqa: BLE001
        return None


def _build_bridge(module_name: str, class_name: str, candidates: dict[str, Any]) -> Any:
    try:
        cls = getattr(importlib.import_module(f"agentos.acp.{module_name}"), class_name)
    except (ImportError, AttributeError):
        return None
    try:
        return _construct(cls, candidates)
    except Exception:  # noqa: BLE001
        log.warning("acp.wiring.bridge_failed", bridge=class_name, exc_info=True)
        return None


def build_acp_manager(config: Any, *, session_manager: Any) -> AcpSessionManager:
    """A manager over the real runtime, or an "unavailable" one explaining why."""
    acp_cfg = acp_config_of(config)
    if not getattr(acp_cfg, "enabled", True):
        return AcpSessionManager(
            None,
            config=acp_cfg,
            session_manager=session_manager,
            unavailable_reason="disabled ([acp] enabled = false)",
        )
    state_dir = acp_state_dir(config)
    queue = _approval_queue()
    common: dict[str, Any] = {
        "config": acp_cfg,
        "acp_config": acp_cfg,
        "gateway_config": config,
        "state_dir": state_dir,
        "queue": queue,
        "approval_queue": queue,
        "session_manager": session_manager,
    }
    permission_bridge = _build_bridge("permissions", "ApprovalQueuePermissionBridge", common)
    elicitation_candidates = dict(common)
    is_unattended = getattr(permission_bridge, "is_unattended", None)
    if callable(is_unattended):
        elicitation_candidates["is_unattended"] = is_unattended
    elicitation_bridge = _build_bridge(
        "elicitation", "ApprovalQueueElicitationBridge", elicitation_candidates
    )
    try:
        runtime_cls = importlib.import_module("agentos.acp.runtime").AcpRuntimeImpl
    except (ImportError, AttributeError) as exc:
        return AcpSessionManager(
            None,
            config=acp_cfg,
            session_manager=session_manager,
            unavailable_reason=f"ACP runtime not installed ({exc})",
        )
    candidates = dict(common)
    candidates.update(
        {
            "permission_bridge": permission_bridge,
            "permissions": permission_bridge,
            "elicitation_bridge": elicitation_bridge,
            "elicitation": elicitation_bridge,
        }
    )
    try:
        runtime = _construct(runtime_cls, candidates)
    except Exception as exc:  # noqa: BLE001 - never crash boot over ACP
        log.warning("acp.wiring.runtime_failed", exc_info=True)
        return AcpSessionManager(
            None,
            config=acp_cfg,
            session_manager=session_manager,
            unavailable_reason=f"ACP runtime failed to start ({type(exc).__name__}: {exc})",
        )
    if permission_bridge is None:
        log.warning("acp.wiring.no_permission_bridge")
    manager = AcpSessionManager(runtime, config=acp_cfg, session_manager=session_manager)
    manager.permission_bridge = permission_bridge
    manager.elicitation_bridge = elicitation_bridge
    return manager


async def reap_stale_leases(config: Any) -> int:
    """Kill harness processes a previous gateway left behind (best effort)."""
    try:
        reaper = importlib.import_module("agentos.acp.process").reap_stale_leases
    except (ImportError, AttributeError):
        return 0
    state_dir = acp_state_dir(config)
    try:
        result = _construct(
            reaper,
            {
                "state_dir": state_dir,
                "acp_state_dir": state_dir,
                "config": acp_config_of(config),
            },
        )
        if inspect.isawaitable(result):
            result = await result
    except Exception:  # noqa: BLE001
        log.warning("acp.wiring.reap_failed", exc_info=True)
        return 0
    if isinstance(result, int):
        return result
    try:
        return len(result)  # type: ignore[arg-type]
    except TypeError:
        return 0


async def start_acp(config: Any, *, session_manager: Any) -> AcpSessionManager | None:
    """Gateway boot: reap leftovers, build the manager, publish it."""
    try:
        reaped = await reap_stale_leases(config)
        if reaped:
            log.info("acp.wiring.reaped", count=reaped)
        manager = build_acp_manager(config, session_manager=session_manager)
    except Exception:  # noqa: BLE001 - ACP must never block gateway boot
        log.warning("acp.wiring.start_failed", exc_info=True)
        return None
    set_acp_manager(manager)
    log.info(
        "acp.wiring.ready",
        available=manager.available,
        reason=manager.unavailable_reason,
    )
    return manager


async def stop_acp(manager: AcpSessionManager | None = None) -> None:
    """Gateway shutdown: close every live harness."""
    manager = manager or get_acp_manager()
    if manager is None:
        return
    try:
        await manager.shutdown()
    except Exception:  # noqa: BLE001
        log.warning("acp.wiring.shutdown_failed", exc_info=True)
    finally:
        if get_acp_manager() is manager:
            set_acp_manager(None)
