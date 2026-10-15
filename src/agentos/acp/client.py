"""The ACP *client* side AgentOS presents to a harness.

:class:`AgentOSAcpClient` implements the SDK ``acp.Client`` protocol for one
harness process (one ACP session). It

- normalizes every ``session/update`` variant into an :class:`AcpEvent` and
  hands it to the active turn's sink (events outside a turn, such as the
  history replay of ``session/load`` or Claude's start-up
  ``available_commands_update`` blobs, only update session state);
- answers ``session/request_permission`` through the injected
  :class:`PermissionBridge`;
- answers ``elicitation/create`` through the injected
  :class:`ElicitationBridge` (decline when none is wired);
- refuses ``fs/*`` and ``terminal/*``: we advertise neither capability, the
  harness uses its own tools.

The event mapping mirrors OpenClaw's ``src/acp`` event projection (MIT, see
``THIRD_PARTY_NOTICES.md``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import structlog
from acp import RequestError
from acp.schema import (
    AcceptElicitationResponse,
    AllowedOutcome,
    CancelElicitationResponse,
    DeclineElicitationResponse,
    DeniedOutcome,
    PermissionOption,
    RequestPermissionResponse,
    ToolCallUpdate,
)

from agentos.acp.types import (
    AcpEvent,
    ElicitationBridge,
    ElicitationRequest,
    ElicitationResponse,
    PermissionBridge,
    PermissionDecision,
    PermissionOptionInfo,
    PermissionPromptUnavailableError,
    PermissionRequest,
    ToolCallStatus,
)

_log = structlog.get_logger("agentos.acp.client")

_TOOL_STATUSES = frozenset({"pending", "in_progress", "completed", "failed"})
_STATUS_RAW_LIMIT = 40


def _dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        try:
            data = model.model_dump(mode="json", by_alias=True, exclude_none=True)
        except Exception:
            data = model.model_dump(exclude_none=True)
        return data if isinstance(data, dict) else {"value": data}
    if isinstance(model, dict):
        return dict(model)
    return {"value": str(model)}


def content_text(block: Any) -> str:
    """Plain text of an ACP content block (non-text blocks get a short tag)."""
    if block is None:
        return ""
    btype = getattr(block, "type", None)
    if btype == "text":
        return str(getattr(block, "text", "") or "")
    if btype == "resource_link":
        return f"[resource: {getattr(block, 'uri', '')}]"
    if btype == "resource":
        res = getattr(block, "resource", None)
        text = getattr(res, "text", None)
        if isinstance(text, str):
            return text
        return f"[resource: {getattr(res, 'uri', '')}]"
    if btype in ("image", "audio"):
        return f"[{btype}]"
    if isinstance(block, dict) and block.get("type") == "text":
        return str(block.get("text") or "")
    return ""


def _as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return {"value": value}


def _plan_text(entries: Any) -> str:
    lines: list[str] = []
    for e in entries or []:
        status = getattr(e, "status", None) or ""
        content = getattr(e, "content", None) or ""
        mark = {"completed": "[x]", "in_progress": "[~]"}.get(status, "[ ]")
        lines.append(f"{mark} {content}".rstrip())
    return "\n".join(lines)


def normalize_update(
    update: Any, tools: dict[str, dict[str, Any]] | None = None
) -> AcpEvent | None:
    """Map one SDK ``session/update`` payload to an :class:`AcpEvent`.

    ``tools`` is the per-session tool-call state; ``tool_call_update``
    events get the earlier title / kind / raw_input merged in so a consumer
    never sees a nameless completion. Returns ``None`` for updates that
    carry nothing for a turn (``user_message_chunk`` replays).
    """
    tag = str(getattr(update, "session_update", "") or "")
    if tools is None:
        tools = {}
    if tag == "agent_message_chunk":
        return AcpEvent(kind="text_delta", tag=tag, text=content_text(update.content))
    if tag == "agent_thought_chunk":
        return AcpEvent(kind="thought_delta", tag=tag, text=content_text(update.content))
    if tag == "user_message_chunk":
        return None
    if tag in ("tool_call", "tool_call_update"):
        tcid = str(getattr(update, "tool_call_id", "") or "")
        state = tools.setdefault(tcid, {})
        for attr in ("title", "kind", "status", "raw_input", "raw_output"):
            val = getattr(update, attr, None)
            if val is not None:
                state[attr] = val
        status = state.get("status")
        if status not in _TOOL_STATUSES:
            status = "pending" if tag == "tool_call" else None
        content = getattr(update, "content", None)
        text = ""
        if content:
            parts = []
            for item in content:
                if getattr(item, "type", None) == "content":
                    parts.append(content_text(getattr(item, "content", None)))
                elif getattr(item, "type", None) == "diff":
                    parts.append(f"[diff: {getattr(item, 'path', '')}]")
                elif getattr(item, "type", None) == "terminal":
                    parts.append(f"[terminal: {getattr(item, 'terminal_id', '')}]")
            text = "\n".join(p for p in parts if p)
        return AcpEvent(
            kind="tool_call" if tag == "tool_call" else "tool_call_update",
            tag=tag,
            text=text,
            tool_call_id=tcid or None,
            title=state.get("title"),
            tool_kind=state.get("kind"),
            status=cast(ToolCallStatus | None, status),
            raw_input=_as_dict(state.get("raw_input")),
            raw_output=state.get("raw_output"),
            raw=_dump(update),
        )
    if tag == "plan":
        return AcpEvent(kind="plan", tag=tag, text=_plan_text(update.entries), raw=_dump(update))
    if tag in ("plan_update", "plan_removed"):
        return AcpEvent(kind="plan", tag=tag, raw=_dump(update))
    if tag == "usage_update":
        usage: dict[str, Any] = {"used": int(update.used), "size": int(update.size)}
        cost_usd: float | None = None
        cost = getattr(update, "cost", None)
        if cost is not None and getattr(cost, "amount", None) is not None:
            currency = str(getattr(cost, "currency", "") or "").upper()
            usage["cost_amount"] = float(cost.amount)
            usage["cost_currency"] = currency
            if currency in ("", "USD"):
                cost_usd = float(cost.amount)
        return AcpEvent(kind="usage", tag=tag, usage=usage, cost_usd=cost_usd, raw=_dump(update))
    if tag == "current_mode_update":
        return AcpEvent(kind="mode", tag=tag, text=str(update.current_mode_id), raw=_dump(update))
    if tag == "config_option_update":
        return AcpEvent(kind="mode", tag=tag, raw=_dump(update))
    if tag == "session_info_update":
        return AcpEvent(
            kind="status", tag=tag, title=getattr(update, "title", None), raw=_dump(update)
        )
    if tag == "available_commands_update":
        # Claude sends two multi-KB blobs of these at start-up; keep a summary.
        cmds = getattr(update, "available_commands", None) or []
        names = [str(getattr(c, "name", "")) for c in cmds]
        return AcpEvent(
            kind="status",
            tag=tag,
            raw={"count": len(names), "names": names[:_STATUS_RAW_LIMIT]},
        )
    return AcpEvent(kind="status", tag=tag or type(update).__name__, raw=_dump(update))


def _option_infos(options: list[PermissionOption]) -> list[PermissionOptionInfo]:
    return [
        PermissionOptionInfo(option_id=o.option_id, name=o.name, kind=str(o.kind)) for o in options
    ]


def pick_permission_option(
    decision: PermissionDecision, options: list[PermissionOption]
) -> RequestPermissionResponse:
    """Map a bridge decision onto one of the harness's options.

    Exact ``kind`` match first; otherwise the first reject option; with no
    reject option (or ``cancel``) the request is answered ``cancelled``.
    """
    if decision != "cancel":
        for o in options:
            if o.kind == decision:
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(outcome="selected", option_id=o.option_id)
                )
        # allow_always requested but only allow_once offered: degrade, never upgrade.
        if decision == "allow_always":
            for o in options:
                if o.kind == "allow_once":
                    return RequestPermissionResponse(
                        outcome=AllowedOutcome(outcome="selected", option_id=o.option_id)
                    )
        for o in options:
            if o.kind in ("reject_once", "reject_always"):
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(outcome="selected", option_id=o.option_id)
                )
    return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


class AgentOSAcpClient:
    """SDK ``Client`` implementation for one harness process."""

    def __init__(
        self,
        *,
        agent_id: str,
        cwd: str,
        session_key: str,
        permission_bridge: PermissionBridge | None = None,
        elicitation_bridge: ElicitationBridge | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.cwd = cwd
        self.session_key = session_key
        self.permission_bridge = permission_bridge
        self.elicitation_bridge = elicitation_bridge
        self.acp_session_id: str | None = None
        self.tools: dict[str, dict[str, Any]] = {}
        self.current_mode_id: str | None = None
        self.config_options: list[dict[str, Any]] | None = None
        self.last_cost_usd: float | None = None
        self.last_usage: dict[str, Any] | None = None
        self.turn_failure: BaseException | None = None
        self.on_turn_failure: Callable[[BaseException], None] | None = None
        self.permission_requests = 0
        self.available_commands: list[str] = []
        self._sink: Callable[[AcpEvent], None] | None = None

    # -- turn wiring ---------------------------------------------------

    def attach(self, sink: Callable[[AcpEvent], None]) -> None:
        self._sink = sink
        self.turn_failure = None
        self.tools = {}

    def detach(self) -> None:
        self._sink = None

    def _emit(self, event: AcpEvent) -> None:
        sink = self._sink
        if sink is not None:
            try:
                sink(event)
            except Exception:
                _log.exception("acp.client.sink_failed")

    def _fail_turn(self, exc: BaseException) -> None:
        if self.turn_failure is None:
            self.turn_failure = exc
        cb = self.on_turn_failure
        if cb is not None:
            try:
                cb(exc)
            except Exception:
                _log.exception("acp.client.fail_callback_failed")

    # -- session/update ------------------------------------------------

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        tag = str(getattr(update, "session_update", "") or "")
        if tag == "current_mode_update":
            self.current_mode_id = str(update.current_mode_id)
        elif tag == "config_option_update":
            self.config_options = [_dump(o) for o in (update.config_options or [])]
        elif tag == "available_commands_update":
            # Session metadata (Claude sends several KB of it at start-up and
            # it races the first prompt): keep the names, never stream it.
            cmds = getattr(update, "available_commands", None) or []
            self.available_commands = [str(getattr(c, "name", "")) for c in cmds]
            return
        elif tag == "usage_update":
            cost = getattr(update, "cost", None)
            if cost is not None and str(getattr(cost, "currency", "") or "").upper() in (
                "",
                "USD",
            ):
                self.last_cost_usd = float(cost.amount)
            self.last_usage = {"used": int(update.used), "size": int(update.size)}
        if self._sink is None:
            return
        try:
            event = normalize_update(update, self.tools)
        except Exception as exc:  # never let a malformed update kill the turn
            _log.warning("acp.client.normalize_failed", tag=tag, error=str(exc))
            event = AcpEvent(kind="status", tag=tag or "unknown", raw={"error": str(exc)})
        if event is not None:
            self._emit(event)

    # -- permissions ---------------------------------------------------

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        self.permission_requests += 1
        tcid = getattr(tool_call, "tool_call_id", None)
        known = self.tools.get(str(tcid), {}) if tcid else {}
        raw_input = getattr(tool_call, "raw_input", None)
        if raw_input is None:
            raw_input = known.get("raw_input")
        request = PermissionRequest(
            session_key=self.session_key,
            acp_session_id=session_id or (self.acp_session_id or ""),
            agent_id=self.agent_id,
            tool_call_id=str(tcid) if tcid else None,
            title=str(getattr(tool_call, "title", None) or known.get("title") or "tool call"),
            tool_kind=(getattr(tool_call, "kind", None) or known.get("kind")),
            raw_input=_as_dict(raw_input),
            options=_option_infos(options),
            cwd=self.cwd,
        )
        decision: PermissionDecision
        if self.permission_bridge is None:
            decision = "reject_once"
        else:
            try:
                decision = await self.permission_bridge.decide(request)
            except PermissionPromptUnavailableError as exc:
                _log.warning("acp.permission.unavailable", agent_id=self.agent_id)
                self._fail_turn(exc)
                return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
            except Exception as exc:
                _log.warning("acp.permission.bridge_failed", error=str(exc))
                decision = "reject_once"
        response = pick_permission_option(decision, options)
        outcome = response.outcome
        selected = getattr(outcome, "option_id", None)
        self._emit(
            AcpEvent(
                kind="status",
                tag="permission_request",
                tool_call_id=request.tool_call_id,
                title=request.title,
                tool_kind=request.tool_kind,
                raw={"decision": decision, "option_id": selected},
            )
        )
        _log.info(
            "acp.permission.decided",
            agent_id=self.agent_id,
            tool_kind=request.tool_kind,
            decision=decision,
        )
        return response

    # -- elicitation ---------------------------------------------------

    async def create_elicitation(self, message: str, mode: Any, **kwargs: Any) -> Any:
        mode_name = "other"
        schema: dict[str, Any] | None = None
        url: str | None = None
        if getattr(mode, "requested_schema", None) is not None:
            mode_name = "form"
            schema = _dump(mode.requested_schema)
        elif getattr(mode, "url", None) is not None:
            mode_name = "url"
            url = str(mode.url)
        session_id = getattr(mode, "session_id", None) or self.acp_session_id or ""
        request = ElicitationRequest(
            session_key=self.session_key,
            acp_session_id=str(session_id),
            agent_id=self.agent_id,
            message=message,
            mode=mode_name,
            schema=schema,
            url=url,
        )
        if self.elicitation_bridge is None:
            return DeclineElicitationResponse(action="decline")
        try:
            answer: ElicitationResponse = await self.elicitation_bridge.ask(request)
        except Exception as exc:
            _log.warning("acp.elicitation.bridge_failed", error=str(exc))
            return DeclineElicitationResponse(action="decline")
        if answer.action == "accept":
            return AcceptElicitationResponse(action="accept", content=answer.content)
        if answer.action == "cancel":
            return CancelElicitationResponse(action="cancel")
        return DeclineElicitationResponse(action="decline")

    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        return None

    # -- refused capabilities -----------------------------------------

    async def write_text_file(self, session_id: str, path: str, content: str, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(
        self,
        session_id: str,
        path: str,
        line: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Any:
        raise RequestError.method_not_found("fs/read_text_file")

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: list[str] | None = None,
        env: list[Any] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> Any:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/kill")

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(f"_{method}")

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None

    def on_connect(self, conn: Any) -> None:
        return None
