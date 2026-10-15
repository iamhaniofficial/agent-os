"""Shared ACP contract types.

Everything that crosses a module boundary inside ``agentos.acp`` is defined
here so the runtime, permission bridge, session manager, projector, tools and
CLI can be built independently against one stable shape. Keep this module
dependency-free (stdlib + typing only).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

# --------------------------------------------------------------------------
# Events (normalized ACP ``session/update`` stream)
# --------------------------------------------------------------------------

AcpEventKind = Literal[
    "text_delta",  # agent_message_chunk
    "thought_delta",  # agent_thought_chunk
    "tool_call",  # tool_call (start)
    "tool_call_update",  # tool_call_update (progress / completion)
    "plan",  # plan / plan content updates
    "usage",  # usage_update (tokens, optional cost)
    "mode",  # current_mode_update / config_option_update
    "status",  # session_info_update, available_commands_update, other
    "done",  # synthesized when ``session/prompt`` returns
    "error",  # synthesized on transport / agent failure
]

ToolCallStatus = Literal["pending", "in_progress", "completed", "failed"]


@dataclass(slots=True)
class AcpEvent:
    """One normalized event from a harness turn.

    ``tag`` keeps the raw ACP ``sessionUpdate`` discriminator (for example
    ``agent_message_chunk``) so projectors can apply per-tag visibility.
    ``raw`` keeps the SDK model dump for anything not lifted into fields.
    """

    kind: AcpEventKind
    tag: str = ""
    text: str = ""
    tool_call_id: str | None = None
    title: str | None = None
    tool_kind: str | None = None  # read | edit | execute | search | fetch | think | other
    status: ToolCallStatus | None = None
    raw_input: dict[str, Any] | None = None
    raw_output: Any = None
    usage: dict[str, Any] | None = None  # {"used": int, "size": int, ...}
    cost_usd: float | None = None
    stop_reason: str | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# Session spec / handle / turn result
# --------------------------------------------------------------------------

SessionMode = Literal["persistent", "oneshot"]


@dataclass(slots=True)
class AcpSessionSpec:
    """What the caller wants: which harness, where, how."""

    agent_id: str
    cwd: str
    mode: SessionMode = "persistent"
    resume_session_id: str | None = None
    model: str | None = None
    harness_mode: str | None = None  # ACP session mode id (e.g. "acceptEdits", "read-only")
    env: dict[str, str] = field(default_factory=dict)
    label: str = ""
    turn_timeout_s: float = 0.0  # 0 = use config default
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AcpSessionHandle:
    """Identity of a live or resumable harness session.

    Persisted by ``agentos.acp.store``; reused only when ``agent_id``,
    ``cwd`` and ``argv_hash`` match exactly (OpenClaw rule).
    """

    record_id: str
    agent_id: str
    cwd: str
    argv: list[str]
    argv_hash: str
    acp_session_id: str | None = None
    pid: int | None = None
    lease_id: str | None = None
    mode: SessionMode = "persistent"
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    current_mode_id: str | None = None
    available_modes: list[dict[str, Any]] = field(default_factory=list)
    config_options: list[dict[str, Any]] = field(default_factory=list)
    agent_capabilities: dict[str, Any] = field(default_factory=dict)
    status: Literal["starting", "idle", "busy", "closed", "unhealthy"] = "starting"


@dataclass(slots=True)
class AcpUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_read_tokens: int = 0
    cached_write_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0
    context_used: int | None = None
    context_size: int | None = None
    cost_usd: float | None = None  # only when the harness reports it (Claude does)
    model: str | None = None


@dataclass(slots=True)
class AcpTurnResult:
    stop_reason: str  # end_turn | max_tokens | max_turn_requests | refusal | cancelled | error
    text: str = ""
    usage: AcpUsage = field(default_factory=AcpUsage)
    error: str | None = None
    duration_ms: int = 0
    tool_calls: int = 0


@dataclass(slots=True)
class AcpTurn:
    """A running turn: consume ``events`` until ``done``/``error``, then
    await ``result``. ``cancel`` is idempotent."""

    events: AsyncIterator[AcpEvent]
    result: Awaitable[AcpTurnResult]
    cancel: Callable[[], Awaitable[None]]
    close_stream: Callable[[], Awaitable[None]]


# --------------------------------------------------------------------------
# Permissions / elicitation bridge
# --------------------------------------------------------------------------

PermissionDecision = Literal["allow_once", "allow_always", "reject_once", "reject_always", "cancel"]


@dataclass(slots=True)
class PermissionOptionInfo:
    option_id: str
    name: str
    kind: str  # allow_once | allow_always | reject_once | reject_always


@dataclass(slots=True)
class PermissionRequest:
    session_key: str  # AgentOS session key that owns the harness
    acp_session_id: str
    agent_id: str
    tool_call_id: str | None
    title: str
    tool_kind: str | None  # read | edit | execute | ...
    raw_input: dict[str, Any] | None
    options: list[PermissionOptionInfo]
    cwd: str


class PermissionBridge(Protocol):
    async def decide(self, request: PermissionRequest) -> PermissionDecision: ...


@dataclass(slots=True)
class ElicitationRequest:
    session_key: str
    acp_session_id: str
    agent_id: str
    message: str
    mode: str  # form | url | other
    schema: dict[str, Any] | None
    url: str | None


@dataclass(slots=True)
class ElicitationResponse:
    action: Literal["accept", "decline", "cancel"]
    content: dict[str, Any] | None = None


class ElicitationBridge(Protocol):
    async def ask(self, request: ElicitationRequest) -> ElicitationResponse: ...


# --------------------------------------------------------------------------
# Runtime contract
# --------------------------------------------------------------------------


class AcpRuntime(Protocol):
    """One implementation drives every harness; the registry supplies argv."""

    async def ensure_session(
        self, spec: AcpSessionSpec, *, session_key: str
    ) -> AcpSessionHandle: ...

    async def start_turn(
        self,
        handle: AcpSessionHandle,
        *,
        text: str,
        attachments: list[dict[str, Any]] | None = None,
        session_key: str,
    ) -> AcpTurn: ...

    async def cancel(self, handle: AcpSessionHandle, *, reason: str = "") -> None: ...

    async def close(self, handle: AcpSessionHandle, *, discard: bool = False) -> None: ...

    async def status(self, handle: AcpSessionHandle) -> AcpSessionHandle: ...

    async def set_mode(self, handle: AcpSessionHandle, mode_id: str) -> None: ...

    async def set_config_option(self, handle: AcpSessionHandle, key: str, value: Any) -> None: ...

    async def doctor(self, agent_id: str, *, probe: bool = False) -> dict[str, Any]: ...


class AcpError(RuntimeError):
    """Base class; ``code`` is a stable machine-readable string."""

    code = "ACP_ERROR"

    def __init__(self, message: str, *, code: str | None = None, **details: Any) -> None:
        super().__init__(message)
        if code:
            self.code = code
        self.details = details


class AcpSessionInitError(AcpError):
    code = "ACP_SESSION_INIT_FAILED"


class AcpTurnError(AcpError):
    code = "ACP_TURN_FAILED"


class AcpTurnTimeoutError(AcpTurnError):
    code = "ACP_TURN_TIMEOUT"


class PermissionPromptUnavailableError(AcpError):
    code = "ACP_PERMISSION_PROMPT_UNAVAILABLE"


class AcpAgentNotAllowedError(AcpError):
    code = "ACP_AGENT_NOT_ALLOWED"
