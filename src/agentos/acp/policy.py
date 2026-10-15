"""Spawn policy for ACP harness sessions.

Who may start a harness, with which permission mode, and how many at once.
Semantics follow OpenClaw's ``acp-spawn-policy.ts`` / ``spawn-plan.ts`` (MIT,
see ``THIRD_PARTY_NOTICES.md``): a harness runs on the host with its own file
and shell tools, so a requester that is confined (workspace lockdown) must
not be able to escape through it, and an unattended requester cannot start a
session whose every permission prompt would fail.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Literal, cast

import structlog

from agentos.acp.registry import get_harness, normalize_agent_id
from agentos.acp.types import AcpAgentNotAllowedError, AcpError

if TYPE_CHECKING:
    from agentos.gateway.config import AcpConfig

log = structlog.get_logger(__name__)

PermissionMode = Literal["ask", "approve-all", "approve-reads", "deny-all"]

PERMISSION_MODES: tuple[PermissionMode, ...] = ("approve-all", "approve-reads", "ask", "deny-all")

#: Higher = stricter. A per-spawn override may only move up this scale unless
#: the caller is explicitly allowed to loosen it (an operator, not the agent).
_STRICTNESS: dict[str, int] = {"approve-all": 0, "approve-reads": 1, "ask": 2, "deny-all": 3}


class AcpConcurrencyLimitError(AcpError):
    """``[acp].max_concurrent`` live harness sessions already exist."""

    code = "ACP_MAX_CONCURRENT"


class AcpPermissionModeNotAllowedError(AcpAgentNotAllowedError):
    """A per-spawn permission mode would loosen the configured one."""

    code = "ACP_PERMISSION_MODE_NOT_ALLOWED"


def _ctx_attr(ctx: Any, name: str, default: Any = None) -> Any:
    if ctx is None:
        return default
    if isinstance(ctx, dict):
        return ctx.get(name, default)
    return getattr(ctx, name, default)


def _is_unattended(ctx: Any) -> bool:
    mode = _ctx_attr(ctx, "interaction_mode")
    return str(getattr(mode, "value", mode) or "").lower() == "unattended"


def agent_allowed(config: AcpConfig, agent_id: str) -> bool:
    """Whether ``agent_id`` passes ``[acp].allowed_agents`` (``*`` = any)."""
    aid = normalize_agent_id(agent_id)
    allowed = {normalize_agent_id(a) for a in config.allowed_agents if str(a).strip()}
    return "*" in allowed or aid in allowed


def check_spawn_allowed(
    config: AcpConfig,
    agent_id: str,
    *,
    requester_ctx: Any = None,
    permission_mode: str | None = None,
) -> str:
    """Raise :class:`AcpAgentNotAllowedError` unless this spawn may proceed.

    Returns the normalized harness id. ``requester_ctx`` is the spawning
    turn's ``ToolContext`` (or a dict with the same field names, or
    ``None`` for an operator-initiated spawn). ``permission_mode`` is the
    mode the session will run under (see :func:`effective_permission_mode`);
    defaults to the configured one.
    """
    aid = normalize_agent_id(agent_id)
    if not config.enabled:
        raise AcpAgentNotAllowedError(
            "ACP harness sessions are disabled ([acp].enabled = false).",
            agent_id=aid,
            reason="disabled",
        )
    if not aid:
        raise AcpAgentNotAllowedError("No ACP agent id given.", agent_id=aid, reason="empty")
    if not agent_allowed(config, aid):
        raise AcpAgentNotAllowedError(
            f"ACP agent {aid!r} is not in [acp].allowed_agents.",
            agent_id=aid,
            reason="not_allowed",
            allowed_agents=list(config.allowed_agents),
        )
    if get_harness(aid) is None and aid not in config.agents:
        raise AcpAgentNotAllowedError(
            f"Unknown ACP agent {aid!r}: not a built-in harness and no [acp.agents.{aid}] entry.",
            agent_id=aid,
            reason="unknown",
        )
    if _ctx_attr(requester_ctx, "workspace_lockdown", False):
        raise AcpAgentNotAllowedError(
            "ACP harness sessions run on the host with their own file and shell tools, so a "
            "workspace-locked run cannot start one.",
            agent_id=aid,
            reason="sandboxed_requester",
        )
    mode = permission_mode or config.permission_mode
    if (
        _is_unattended(requester_ctx)
        and mode == "ask"
        and config.non_interactive_permissions == "fail"
    ):
        raise AcpAgentNotAllowedError(
            "This run is unattended and [acp].permission_mode is 'ask' with "
            "non_interactive_permissions = 'fail': every harness permission prompt would fail. "
            "Set permission_mode (approve-reads / approve-all / deny-all) or "
            "non_interactive_permissions = 'deny' for unattended use.",
            agent_id=aid,
            reason="unattended_requester",
        )
    return aid


def effective_permission_mode(
    config: AcpConfig,
    override: str | None = None,
    *,
    allow_loosen: bool = False,
) -> PermissionMode:
    """The permission mode a new session runs under.

    ``override`` (per-spawn) wins when it is at least as strict as the
    configured mode. Loosening (for example ``ask`` -> ``approve-all``) is
    only honoured with ``allow_loosen=True`` -- an operator choice, never one
    the agent makes for itself.
    """
    base = config.permission_mode
    if override is None or not str(override).strip():
        return base
    value = str(override).strip().lower()
    if value not in _STRICTNESS:
        raise ValueError(
            f"permission_mode must be one of: {', '.join(PERMISSION_MODES)} (got {override!r})"
        )
    if _STRICTNESS[value] < _STRICTNESS[base] and not allow_loosen:
        raise AcpPermissionModeNotAllowedError(
            f"permission_mode {value!r} is looser than the configured {base!r}; only the "
            "operator can loosen it ([acp].permission_mode).",
            requested=value,
            configured=base,
        )
    return cast(PermissionMode, value)


class AcpAdmission:
    """Counts live harness sessions against ``[acp].max_concurrent``.

    ``acquire(key)`` admits a session (idempotent per key) or raises
    :class:`AcpConcurrencyLimitError`; ``release(key)`` frees the slot
    (idempotent). Thread-safe; the limit is re-read from the config on every
    call so a config reload applies to the next spawn.
    """

    def __init__(self, config: AcpConfig | None = None, *, limit: int | None = None) -> None:
        self._config = config
        self._limit = limit
        self._active: set[str] = set()
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        if self._limit is not None:
            return self._limit
        if self._config is not None:
            return int(self._config.max_concurrent)
        return 8

    def acquire(self, key: str) -> None:
        with self._lock:
            if key in self._active:
                return
            if len(self._active) >= self.limit:
                raise AcpConcurrencyLimitError(
                    f"{len(self._active)} ACP harness sessions are already running "
                    f"([acp].max_concurrent = {self.limit}). Close one first.",
                    active=len(self._active),
                    limit=self.limit,
                )
            self._active.add(key)
        log.debug("acp.policy.admitted", key=key, active=len(self._active))

    def release(self, key: str) -> None:
        with self._lock:
            self._active.discard(key)

    def active(self) -> int:
        with self._lock:
            return len(self._active)

    def has_capacity(self) -> bool:
        with self._lock:
            return len(self._active) < self.limit


__all__ = [
    "PERMISSION_MODES",
    "AcpAdmission",
    "AcpConcurrencyLimitError",
    "AcpPermissionModeNotAllowedError",
    "PermissionMode",
    "agent_allowed",
    "check_spawn_allowed",
    "effective_permission_mode",
]
