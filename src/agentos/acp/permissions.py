"""Permission bridge: harness ``session/request_permission`` -> AgentOS policy.

The runtime calls :meth:`ApprovalQueuePermissionBridge.decide` for every
permission prompt a harness raises and answers it with the returned
:data:`~agentos.acp.types.PermissionDecision`. Static modes answer without a
human; ``ask`` puts the prompt on the operator approval queue (namespace
``"acp"``) and waits for the Web UI / desktop / RPC to resolve it.

Semantics follow OpenClaw's ``extensions/acpx/src/harness-attempt.ts``
(MIT, see ``THIRD_PARTY_NOTICES.md``): never widen the operator's grant --
``allow_always`` is only ever returned under ``approve-all`` and only when the
harness offers no ``allow_once`` -- and fail closed on timeout or error.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

from agentos.acp.registry import get_harness, normalize_agent_id
from agentos.acp.types import (
    PermissionDecision,
    PermissionOptionInfo,
    PermissionPromptUnavailableError,
    PermissionRequest,
)
from agentos.util.bounded_registry import BoundedRegistry

if TYPE_CHECKING:
    from agentos.application.approval_queue import ApprovalQueue
    from agentos.gateway.config import AcpConfig

log = structlog.get_logger(__name__)

ACP_APPROVAL_NAMESPACE = "acp"

#: Tool kinds ``approve-reads`` answers without asking (ACP ``ToolKind``).
READ_TOOL_KINDS = frozenset({"read", "search", "fetch", "think"})

#: Cap on the serialized ``raw_input`` shown to the operator.
RAW_INPUT_MAX_BYTES = 4096

_VERBS: dict[str, str] = {
    "read": "read",
    "edit": "edit",
    "delete": "delete",
    "move": "move",
    "search": "search",
    "execute": "run",
    "think": "think",
    "fetch": "fetch",
    "switch_mode": "switch mode",
}

_SECRET_KEY = re.compile(
    r"(pass(word|wd)?|secret|token|api[_-]?key|authorization|auth|cookie|credential|private[_-]?key)",
    re.IGNORECASE,
)

#: ``interaction_mode`` resolver: given a session key, return
#: ``"interactive"``, ``"unattended"`` or ``None`` (unknown; fall through to
#: the next source).
SurfaceResolver = Callable[[str], str | None]


def agent_display_name(agent_id: str) -> str:
    harness = get_harness(agent_id)
    return harness.name if harness is not None else (agent_id or "Harness")


def acp_approval_title(agent_id: str, tool_kind: str | None, title: str) -> str:
    """``"<agent> wants to <kind>: <title>"`` -- the line the operator sees."""
    verb = _VERBS.get(str(tool_kind or "").lower(), "use a tool")
    detail = " ".join(str(title or "").split())
    if len(detail) > 200:
        detail = detail[:199] + "…"
    head = f"{agent_display_name(agent_id)} wants to {verb}"
    return f"{head}: {detail}" if detail else head


def _redact_value(value: Any, key: str | None = None) -> Any:
    from agentos.redact import redact_sensitive_text

    if key is not None and _SECRET_KEY.search(key) and isinstance(value, (str, int, float)):
        return "«redacted»"
    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if isinstance(value, dict):
        return {str(k): _redact_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(v) for v in value]
    return value


def redact_raw_input(raw_input: Any, *, max_bytes: int = RAW_INPUT_MAX_BYTES) -> Any:
    """Mask credentials in a harness tool input and cap its serialized size.

    Over-limit input becomes ``{"_truncated": True, "preview": <first bytes>}``
    so the queue row (and the Web UI) never carries an unbounded blob.
    """
    if raw_input is None:
        return None
    redacted = _redact_value(raw_input)
    try:
        text = json.dumps(redacted, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(redacted)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return redacted
    preview = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return {"_truncated": True, "bytes": len(encoded), "preview": preview}


def command_of(raw_input: Any) -> str:
    """The shell command in a harness ``execute`` input, as one string."""
    if not isinstance(raw_input, dict):
        return ""
    for key in ("command", "cmd", "argv", "commandLine"):
        value = raw_input.get(key)
        if isinstance(value, str) and value.strip():
            extra = raw_input.get("args")
            if isinstance(extra, list) and extra:
                return " ".join([value.strip(), *(str(a) for a in extra)])
            return value.strip()
        if isinstance(value, list) and value:
            return " ".join(str(part) for part in value)
    return ""


def command_prefix(command: str) -> str:
    """Program plus subcommand (``git push``), the unit an "always" covers.

    The second token is kept only when it looks like a subcommand (no flag,
    no path, no shell metacharacter), so ``rm -rf /`` remembers ``rm`` and
    ``git push --force`` remembers ``git push``.
    """
    text = command.strip()
    if not text:
        return ""
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        tokens = text.split()
    if not tokens:
        return ""
    program = tokens[0].rsplit("/", 1)[-1]
    if len(tokens) > 1 and re.fullmatch(r"[A-Za-z][\w.:-]*", tokens[1]):
        return f"{program} {tokens[1]}"
    return program


def remember_key(request: PermissionRequest) -> tuple[str, str]:
    """``(tool_kind, prefix)`` for the per-session "always" memory.

    ``execute`` prompts are remembered per command prefix; any other kind is
    remembered for the whole kind (the harness's own "allow edits this
    session" granularity).
    """
    kind = str(request.tool_kind or "other").lower()
    if kind == "execute":
        return kind, command_prefix(command_of(request.raw_input) or request.title or "")
    return kind, ""


def _pick(options: list[PermissionOptionInfo], *kinds: str) -> PermissionDecision | None:
    offered = {o.kind for o in options}
    if not options:
        # The harness listed no options; let the runtime map the bare kind.
        return kinds[0]  # type: ignore[return-value]
    for kind in kinds:
        if kind in offered:
            return kind  # type: ignore[return-value]
    return None


def allow_decision(
    options: list[PermissionOptionInfo], *, allow_always_fallback: bool = False
) -> PermissionDecision:
    """``allow_once``; ``allow_always`` only as an explicit fallback; else reject.

    OpenClaw rule: never widen a once-grant into an always-grant. Without a
    usable allow option the prompt is rejected rather than over-granted.
    """
    kinds = ("allow_once", "allow_always") if allow_always_fallback else ("allow_once",)
    picked = _pick(options, *kinds)
    return picked if picked is not None else reject_decision(options)


def reject_decision(options: list[PermissionOptionInfo]) -> PermissionDecision:
    picked = _pick(options, "reject_once", "reject_always")
    return picked if picked is not None else "cancel"


def _default_surface(_session_key: str) -> str | None:
    """Interaction mode of the tool call that is driving the harness, if any.

    Mirrors ``tools/builtin/shell.py``: an ``UNATTENDED`` tool context means
    there is no live operator to ask.
    """
    from agentos.tools.types import current_tool_context

    ctx = current_tool_context.get()
    if ctx is None:
        return None
    mode = getattr(ctx, "interaction_mode", None)
    return str(getattr(mode, "value", mode) or "") or None


class ApprovalQueuePermissionBridge:
    """:class:`~agentos.acp.types.PermissionBridge` over ``ApprovalQueue``.

    ``permission_mode`` (default: ``config.permission_mode``) is the mode for
    every session this bridge serves; :meth:`bind_session` overrides it per
    session (the spawn's effective mode) and records whether an operator can
    answer prompts for that session. The runtime holds one bridge and calls
    :meth:`decide` from its ACP client callback.
    """

    def __init__(
        self,
        config: AcpConfig,
        *,
        queue: ApprovalQueue | Callable[[], ApprovalQueue] | None = None,
        permission_mode: str | None = None,
        surface: SurfaceResolver | None = None,
    ) -> None:
        self._config = config
        self._queue_source = queue
        self._mode = permission_mode
        self._surface = surface
        self._session_modes: BoundedRegistry[str, str] = BoundedRegistry(
            name="ApprovalQueuePermissionBridge._session_modes",
            session_of=lambda key, _value: key,
        )
        self._session_surfaces: BoundedRegistry[str, str] = BoundedRegistry(
            name="ApprovalQueuePermissionBridge._session_surfaces",
            session_of=lambda key, _value: key,
        )
        self._remembered: BoundedRegistry[str, set[tuple[str, str]]] = BoundedRegistry(
            name="ApprovalQueuePermissionBridge._remembered",
            session_of=lambda key, _value: key,
        )

    # ------------------------------------------------------------------
    # session bookkeeping
    # ------------------------------------------------------------------

    def bind_session(
        self,
        session_key: str,
        *,
        permission_mode: str | None = None,
        interactive: bool | None = None,
    ) -> None:
        """Record a session's effective mode and whether a human can answer."""
        if permission_mode:
            self._session_modes[session_key] = permission_mode
        if interactive is not None:
            self._session_surfaces[session_key] = "interactive" if interactive else "unattended"

    def forget_session(self, session_key: str) -> None:
        """Drop everything held for a closed session (mode, surface, grants)."""
        self._session_modes.pop(session_key, None)
        self._session_surfaces.pop(session_key, None)
        self._remembered.pop(session_key, None)

    def remembered(self, session_key: str) -> set[tuple[str, str]]:
        return set(self._remembered.get(session_key) or ())

    def remember(self, request: PermissionRequest) -> None:
        grants = set(self._remembered.get(request.session_key) or ())
        grants.add(remember_key(request))
        self._remembered[request.session_key] = grants

    def mode_for(self, session_key: str) -> str:
        return self._session_modes.get(session_key) or self._mode or self._config.permission_mode

    def _queue(self) -> ApprovalQueue:
        source = self._queue_source
        if source is None:
            from agentos.application.approval_queue import get_approval_queue

            return get_approval_queue()
        if callable(source) and not hasattr(source, "request"):
            return source()
        return source  # type: ignore[return-value]

    def is_unattended(self, session_key: str) -> bool:
        """Whether no operator can answer prompts for ``session_key``.

        Order: the surface recorded by :meth:`bind_session`, the injected
        resolver, then the current tool context (as ``shell.py`` does).
        Unknown means interactive: the gateway always has the Web UI.
        """
        bound = self._session_surfaces.get(session_key)
        if bound is not None:
            return bound == "unattended"
        for resolver in (self._surface, _default_surface):
            if resolver is None:
                continue
            value = resolver(session_key)
            if value:
                return str(value).lower() == "unattended"
        # The gateway always has the Web UI / desktop approval surface.
        return False

    # ------------------------------------------------------------------
    # PermissionBridge
    # ------------------------------------------------------------------

    async def decide(self, request: PermissionRequest) -> PermissionDecision:
        mode = self.mode_for(request.session_key)
        kind = str(request.tool_kind or "other").lower()
        options = list(request.options or [])
        if mode == "approve-all":
            return self._log(request, mode, allow_decision(options, allow_always_fallback=True))
        if mode == "deny-all":
            return self._log(request, mode, reject_decision(options))
        if mode == "approve-reads" and kind in READ_TOOL_KINDS:
            return self._log(request, mode, allow_decision(options))
        # ask (and approve-reads for anything that is not a read)
        if remember_key(request) in self.remembered(request.session_key):
            return self._log(request, mode, allow_decision(options), source="remembered")
        if self.is_unattended(request.session_key):
            return self._no_surface(request, mode, options, reason="unattended")
        return await self._ask_operator(request, mode, options)

    def _no_surface(
        self,
        request: PermissionRequest,
        mode: str,
        options: list[PermissionOptionInfo],
        *,
        reason: str,
    ) -> PermissionDecision:
        if self._config.non_interactive_permissions == "deny":
            return self._log(request, mode, reject_decision(options), source=f"no_surface:{reason}")
        log.warning(
            "acp.permission.unavailable",
            session_key=request.session_key,
            agent_id=request.agent_id,
            tool_kind=request.tool_kind,
            reason=reason,
        )
        raise PermissionPromptUnavailableError(
            f"{agent_display_name(request.agent_id)} asked for permission "
            f"({request.tool_kind or 'tool'}: {request.title}) but no operator can answer "
            f"({reason}). Set [acp].permission_mode or non_interactive_permissions = 'deny' "
            "for unattended runs.",
            session_key=request.session_key,
            agent_id=request.agent_id,
            tool_call_id=request.tool_call_id,
            reason=reason,
        )

    def build_params(self, request: PermissionRequest) -> dict[str, Any]:
        """Queue-row params: the brief's contract keys plus the display keys
        the existing Web UI renders (``toolName``, ``command``, ``warning``)."""
        raw_input = redact_raw_input(request.raw_input)
        command = command_of(raw_input) if str(request.tool_kind) == "execute" else ""
        title = acp_approval_title(request.agent_id, request.tool_kind, request.title)
        return {
            "session_key": request.session_key,
            "agent_id": normalize_agent_id(request.agent_id),
            "acp_session_id": request.acp_session_id,
            "tool_call_id": request.tool_call_id,
            "title": request.title,
            "tool_kind": request.tool_kind,
            "raw_input": raw_input,
            "cwd": request.cwd,
            "options": [
                {"option_id": o.option_id, "name": o.name, "kind": o.kind} for o in request.options
            ],
            # display aliases (Web UI / desktop approval modal)
            "displayTitle": title,
            "toolName": title,
            "sessionKey": request.session_key,
            "agent": agent_display_name(request.agent_id),
            "command": command,
            "warning": f"{title}\ncwd: {request.cwd}" if request.cwd else title,
        }

    async def _ask_operator(
        self,
        request: PermissionRequest,
        mode: str,
        options: list[PermissionOptionInfo],
    ) -> PermissionDecision:
        from agentos.application.approval_queue import ApprovalQueueOwnedByGatewayError

        queue = self._queue()
        timeout = float(self._config.approval_timeout_s)
        try:
            approval_id = queue.request(
                namespace=ACP_APPROVAL_NAMESPACE,
                params=self.build_params(request),
                ttl=timeout,
            )
        except ApprovalQueueOwnedByGatewayError:
            return self._no_surface(request, mode, options, reason="queue_owned_by_gateway")
        log.info(
            "acp.permission.requested",
            approval_id=approval_id,
            session_key=request.session_key,
            agent_id=request.agent_id,
            tool_kind=request.tool_kind,
        )
        try:
            approved = await queue.wait(approval_id, timeout=timeout)
        except asyncio.CancelledError:
            _deny_quietly(queue, approval_id)
            raise
        except Exception as exc:  # fail closed
            log.warning("acp.permission.wait_failed", approval_id=approval_id, error=str(exc))
            _deny_quietly(queue, approval_id)
            return self._log(request, mode, reject_decision(options), source="error")
        try:
            entry = queue.get(approval_id)
        except KeyError:
            entry = None
        if entry is not None and not entry.resolved:
            _deny_quietly(queue, approval_id)
            approved = False
        if not approved:
            source = "timeout" if entry is not None and not entry.resolved else "operator"
            return self._log(request, mode, reject_decision(options), source=source)
        if entry is not None and entry.params.get("rememberDecision"):
            self.remember(request)
        return self._log(request, mode, allow_decision(options), source="operator")

    @staticmethod
    def _log(
        request: PermissionRequest,
        mode: str,
        decision: PermissionDecision,
        *,
        source: str = "policy",
    ) -> PermissionDecision:
        log.info(
            "acp.permission.decided",
            session_key=request.session_key,
            agent_id=request.agent_id,
            tool_kind=request.tool_kind,
            tool_call_id=request.tool_call_id,
            mode=mode,
            decision=decision,
            source=source,
        )
        return decision


def _deny_quietly(queue: ApprovalQueue, approval_id: str) -> None:
    """Resolve a still-pending row as denied so no stale prompt lingers."""
    try:
        if not queue.get(approval_id).resolved:
            queue.resolve(approval_id, False)
    except (KeyError, ValueError):
        pass
    except Exception as exc:  # pragma: no cover - best effort
        log.debug("acp.permission.deny_failed", approval_id=approval_id, error=str(exc))


__all__ = [
    "ACP_APPROVAL_NAMESPACE",
    "READ_TOOL_KINDS",
    "ApprovalQueuePermissionBridge",
    "acp_approval_title",
    "agent_display_name",
    "allow_decision",
    "command_of",
    "command_prefix",
    "redact_raw_input",
    "reject_decision",
    "remember_key",
]
