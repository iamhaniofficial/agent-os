"""Elicitation bridge: harness ``elicitation/create`` -> operator questions.

A harness may ask the user for structured input (a small form) or to finish a
step in the browser (a URL). Both go to the operator approval queue under the
namespace ``"acp_elicitation"`` and come back through the same resolve path as
permission prompts; form answers ride on the resolve ``payload``
(``{"content": {<field>: <value>}}``).

Semantics follow OpenClaw's ``src/auto-reply/reply/acp-elicitation-handler.ts``
and ``src/agents/harness/structured-input*.ts`` (MIT, see
``THIRD_PARTY_NOTICES.md``): at most 12 fields, asked in batches of 3, secret
fields asked alone, at most 4 choices per choice field, http(s) URLs only and
never with embedded credentials, 15 minute default timeout, anything
unsupported or unanswered is declined. Secret answers are never logged and
are cleared from the queue row as soon as they are read.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import structlog

from agentos.acp.permissions import agent_display_name
from agentos.acp.registry import normalize_agent_id
from agentos.acp.types import ElicitationRequest, ElicitationResponse

if TYPE_CHECKING:
    from agentos.application.approval_queue import ApprovalQueue
    from agentos.gateway.config import AcpConfig

log = structlog.get_logger(__name__)

ACP_ELICITATION_NAMESPACE = "acp_elicitation"
DEFAULT_ELICITATION_TIMEOUT_S = 15 * 60.0

MAX_FORM_FIELDS = 12
QUESTION_BATCH_SIZE = 3
MAX_CHOICE_COUNT = 4
MAX_CHOICE_LABEL = 64
MAX_MESSAGE_TEXT = 1024
MAX_URL_TEXT = 2048
MAX_FIELD_NAME = 256

URL_CONTINUE_LABEL = "Continue"
URL_DECLINE_LABEL = "Decline"

_UNSAFE_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


@dataclass(slots=True)
class FormField:
    """One question compiled from a JSON-schema property."""

    field_id: str
    type: str  # text | number | integer | boolean | choice | multi_choice
    title: str
    description: str = ""
    required: bool = False
    secret: bool = False
    choices: list[dict[str, str]] = field(default_factory=list)  # [{"value","label"}]
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    min_items: int | None = None
    max_items: int | None = None

    def to_question(self) -> dict[str, Any]:
        q: dict[str, Any] = {
            "id": self.field_id,
            "type": self.type,
            "title": self.title,
            "required": self.required,
            "secret": self.secret,
        }
        if self.description:
            q["description"] = self.description
        if self.choices:
            q["choices"] = list(self.choices)
        if self.default is not None and not self.secret:
            q["default"] = self.default
        return q


class UnsupportedElicitationError(ValueError):
    """The request cannot be shown; the harness gets ``decline``."""


def _clean_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > limit:
        return None
    return text


def _is_secret(schema: dict[str, Any]) -> bool:
    meta = schema.get("_meta")
    if isinstance(meta, dict):
        codex = meta.get("codex")
        if isinstance(codex, dict) and codex.get("isSecret") is True:
            return True
        if meta.get("secret") is True or meta.get("isSecret") is True:
            return True
    return str(schema.get("format") or "").lower() == "password"


def _choices_from(schema: dict[str, Any]) -> list[dict[str, str]] | None:
    """``enum`` / ``oneOf`` / ``anyOf`` -> ``[{value, label}]`` or ``None``."""
    options: list[dict[str, str]] = []
    if isinstance(schema.get("enum"), list):
        names = schema.get("enumNames")
        for i, value in enumerate(schema["enum"]):
            label = names[i] if isinstance(names, list) and i < len(names) else value
            options.append({"value": str(value), "label": str(label)})
        return options
    for key in ("oneOf", "anyOf"):
        if isinstance(schema.get(key), list):
            for opt in schema[key]:
                if not isinstance(opt, dict) or "const" not in opt:
                    raise UnsupportedElicitationError(f"has an invalid {key} option")
                options.append(
                    {"value": str(opt["const"]), "label": str(opt.get("title") or opt["const"])}
                )
            return options
    return None


def _check_choices(options: list[dict[str, str]]) -> list[dict[str, str]]:
    if not options or len(options) > MAX_CHOICE_COUNT:
        raise UnsupportedElicitationError(f"must offer 1 to {MAX_CHOICE_COUNT} choices")
    for opt in options:
        if len(opt["label"]) > MAX_CHOICE_LABEL or _UNSAFE_CHARS.search(opt["label"]):
            raise UnsupportedElicitationError("has an over-long or unsafe choice label")
    return options


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def compile_field(field_id: str, schema: Any, *, required: bool) -> FormField:
    if not isinstance(schema, dict):
        raise UnsupportedElicitationError("has an invalid schema")
    kind = schema.get("type")
    title = _clean_text(schema.get("title"), MAX_FIELD_NAME) or field_id
    description = _clean_text(schema.get("description"), MAX_MESSAGE_TEXT) or ""
    secret = _is_secret(schema)
    base: dict[str, Any] = {
        "field_id": field_id,
        "title": title,
        "description": description,
        "required": required,
        "secret": secret,
        "default": schema.get("default"),
    }
    if kind == "boolean":
        return FormField(
            type="boolean",
            choices=[{"value": "true", "label": "True"}, {"value": "false", "label": "False"}],
            **base,
        )
    if kind in ("number", "integer"):
        return FormField(
            type=str(kind),
            minimum=_num(schema.get("minimum")),
            maximum=_num(schema.get("maximum")),
            **base,
        )
    if kind == "string":
        options = _choices_from(schema)
        if options is not None:
            return FormField(type="choice", choices=_check_choices(options), **base)
        return FormField(
            type="text",
            min_length=schema.get("minLength")
            if isinstance(schema.get("minLength"), int)
            else None,
            max_length=schema.get("maxLength")
            if isinstance(schema.get("maxLength"), int)
            else None,
            pattern=schema.get("pattern") if isinstance(schema.get("pattern"), str) else None,
            **base,
        )
    if kind == "array":
        items = schema.get("items")
        options = _choices_from(items) if isinstance(items, dict) else None
        if options is None:
            raise UnsupportedElicitationError("is a multi-select without fixed choices")
        return FormField(
            type="multi_choice",
            choices=_check_choices(options),
            min_items=schema.get("minItems") if isinstance(schema.get("minItems"), int) else None,
            max_items=schema.get("maxItems") if isinstance(schema.get("maxItems"), int) else None,
            **base,
        )
    raise UnsupportedElicitationError(f"has unsupported type {kind!r}")


def compile_form(schema: Any) -> list[FormField]:
    """Validate a requested schema and turn it into ordered questions."""
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        raise UnsupportedElicitationError("the form schema is not an object")
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        raise UnsupportedElicitationError("the form schema has no properties object")
    if len(properties) > MAX_FORM_FIELDS:
        raise UnsupportedElicitationError(f"the form has more than {MAX_FORM_FIELDS} fields")
    required_raw = schema.get("required") or []
    if not isinstance(required_raw, list) or any(
        not isinstance(r, str) or r not in properties for r in required_raw
    ):
        raise UnsupportedElicitationError("the form has an invalid required list")
    required = set(required_raw)
    fields: list[FormField] = []
    for field_id, prop in properties.items():
        if (
            not isinstance(field_id, str)
            or not field_id
            or len(field_id) > MAX_FIELD_NAME
            or _UNSAFE_CHARS.search(field_id)
        ):
            raise UnsupportedElicitationError("the form has an invalid field name")
        try:
            fields.append(compile_field(field_id, prop, required=field_id in required))
        except UnsupportedElicitationError as exc:
            raise UnsupportedElicitationError(f"field {field_id!r} {exc}") from None
    return fields


def batch_fields(fields: list[FormField]) -> list[list[FormField]]:
    """Batches of up to 3 questions; a secret field is always asked alone."""
    batches: list[list[FormField]] = []
    current: list[FormField] = []
    for f in fields:
        if f.secret:
            if current:
                batches.append(current)
                current = []
            batches.append([f])
            continue
        current.append(f)
        if len(current) >= QUESTION_BATCH_SIZE:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def validate_url(url: Any) -> str:
    if not isinstance(url, str) or not url or len(url) > MAX_URL_TEXT:
        raise UnsupportedElicitationError("the URL is missing or too long")
    if url.strip() != url or _UNSAFE_CHARS.search(url) or " " in url:
        raise UnsupportedElicitationError("the URL contains unsafe characters")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise UnsupportedElicitationError("only http(s) URLs are shown")
    if parts.username or parts.password:
        raise UnsupportedElicitationError("URLs with embedded credentials are not shown")
    return url


def decode_answer(f: FormField, raw: Any) -> tuple[bool, Any]:
    """``(present, value)`` for one field; raises ``ValueError`` when invalid.

    The message never includes the submitted value (it may be a secret).
    """
    if raw is None or (isinstance(raw, str) and raw == "") or raw == []:
        if f.required:
            raise ValueError(f"{f.field_id!r} is required")
        return False, None
    if f.type == "boolean":
        if isinstance(raw, bool):
            return True, raw
        text = str(raw).strip().lower()
        if text in ("true", "yes", "y", "1"):
            return True, True
        if text in ("false", "no", "n", "0"):
            return True, False
        raise ValueError(f"{f.field_id!r} must be true or false")
    if f.type in ("number", "integer"):
        try:
            number = float(raw) if not isinstance(raw, bool) else math.nan
        except (TypeError, ValueError):
            number = math.nan
        if math.isnan(number) or math.isinf(number):
            raise ValueError(f"{f.field_id!r} must be a number")
        if f.type == "integer":
            if number != int(number):
                raise ValueError(f"{f.field_id!r} must be a whole number")
            value: Any = int(number)
        else:
            value = number
        if f.minimum is not None and number < f.minimum:
            raise ValueError(f"{f.field_id!r} is below its minimum")
        if f.maximum is not None and number > f.maximum:
            raise ValueError(f"{f.field_id!r} is above its maximum")
        return True, value
    allowed = {c["value"] for c in f.choices}
    labels = {c["label"].lower(): c["value"] for c in f.choices}

    def pick(item: Any) -> str:
        text = str(item)
        if text in allowed:
            return text
        if text.lower() in labels:
            return labels[text.lower()]
        raise ValueError(f"{f.field_id!r} has a value outside its choices")

    if f.type == "choice":
        return True, pick(raw)
    if f.type == "multi_choice":
        items = raw if isinstance(raw, list) else [raw]
        values = [pick(i) for i in items]
        if f.min_items is not None and len(values) < f.min_items:
            raise ValueError(f"{f.field_id!r} needs more selections")
        if f.max_items is not None and len(values) > f.max_items:
            raise ValueError(f"{f.field_id!r} has too many selections")
        return True, values
    # text
    if not isinstance(raw, str):
        raw = str(raw)
    if f.min_length is not None and len(raw) < f.min_length:
        raise ValueError(f"{f.field_id!r} is too short")
    if f.max_length is not None and len(raw) > f.max_length:
        raise ValueError(f"{f.field_id!r} is too long")
    if f.pattern:
        try:
            ok = re.search(f.pattern, raw) is not None
        except re.error:
            ok = True
        if not ok:
            raise ValueError(f"{f.field_id!r} does not match its pattern")
    return True, raw


def _answers_from(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for key in ("content", "answers"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


class ApprovalQueueElicitationBridge:
    """:class:`~agentos.acp.types.ElicitationBridge` over ``ApprovalQueue``.

    One queue row per question batch. Approving a form row without answers
    for its required fields declines the whole request; denying any row, or
    letting the overall timeout lapse, declines it as well.
    """

    def __init__(
        self,
        config: AcpConfig | None = None,
        *,
        queue: ApprovalQueue | Callable[[], ApprovalQueue] | None = None,
        timeout_s: float = DEFAULT_ELICITATION_TIMEOUT_S,
        is_unattended: Callable[[str], bool] | None = None,
    ) -> None:
        """``is_unattended(session_key)`` -- pass the permission bridge's
        :meth:`~agentos.acp.permissions.ApprovalQueuePermissionBridge.is_unattended`
        so both bridges agree on whether an operator is present. With no
        operator, every elicitation is declined (never fails the turn)."""
        self._config = config
        self._queue_source = queue
        self._timeout = float(timeout_s)
        self._is_unattended = is_unattended

    def _queue(self) -> ApprovalQueue:
        source = self._queue_source
        if source is None:
            from agentos.application.approval_queue import get_approval_queue

            return get_approval_queue()
        if callable(source) and not hasattr(source, "request"):
            return source()
        return source  # type: ignore[return-value]

    async def ask(self, request: ElicitationRequest) -> ElicitationResponse:
        if self._is_unattended is not None and self._is_unattended(request.session_key):
            return self._decline(request, "unattended")
        deadline = time.monotonic() + self._timeout
        try:
            if request.mode == "url":
                return await self._ask_url(request, deadline)
            if request.mode == "form":
                return await self._ask_form(request, deadline)
            raise UnsupportedElicitationError(f"mode {request.mode!r} is not supported")
        except UnsupportedElicitationError as exc:
            return self._decline(request, f"unsupported: {exc}")

    # -- url ---------------------------------------------------------------

    async def _ask_url(self, request: ElicitationRequest, deadline: float) -> ElicitationResponse:
        url = validate_url(request.url)
        message = (
            _clean_text(request.message, MAX_MESSAGE_TEXT) or "The harness needs a browser step."
        )
        agent = agent_display_name(request.agent_id)
        title = f"{agent} asks you to open a link: {url}"
        params = self._base_params(request, title)
        params.update(
            {
                "mode": "url",
                "url": url,
                "message": message,
                "options": [URL_CONTINUE_LABEL, URL_DECLINE_LABEL],
                "warning": (
                    f"{message}\n\n{url}\n\nOpen the link, finish the step in your browser, "
                    f"then choose {URL_CONTINUE_LABEL} (approve) or {URL_DECLINE_LABEL} (deny)."
                ),
            }
        )
        outcome = await self._round(params, deadline)
        if outcome is None:
            return self._decline(request, "timeout")
        approved, _payload = outcome
        if not approved:
            return self._decline(request, "operator")
        log.info("acp.elicitation.accepted", session_key=request.session_key, mode="url")
        return ElicitationResponse(action="accept")

    # -- form --------------------------------------------------------------

    async def _ask_form(self, request: ElicitationRequest, deadline: float) -> ElicitationResponse:
        fields = compile_form(request.schema or {"type": "object", "properties": {}})
        message = _clean_text(request.message, MAX_MESSAGE_TEXT) or "The harness needs input."
        agent = agent_display_name(request.agent_id)
        batches = batch_fields(fields) or [[]]
        content: dict[str, Any] = {}
        for index, batch in enumerate(batches):
            title = f"{agent} asks: {message}"
            if len(batches) > 1:
                title = f"{title} ({index + 1}/{len(batches)})"
            params = self._base_params(request, title)
            params.update(
                {
                    "mode": "form",
                    "message": message,
                    "questions": [f.to_question() for f in batch],
                    "batch": index + 1,
                    "batches": len(batches),
                    "warning": _form_warning(message, batch),
                }
            )
            outcome = await self._round(params, deadline)
            if outcome is None:
                return self._decline(request, "timeout")
            approved, payload = outcome
            if not approved:
                return self._decline(request, "operator")
            answers = _answers_from(payload)
            for f in batch:
                try:
                    present, value = decode_answer(f, answers.get(f.field_id))
                except ValueError as exc:
                    return self._decline(request, f"invalid answer: {exc}")
                if present:
                    content[f.field_id] = value
        log.info(
            "acp.elicitation.accepted",
            session_key=request.session_key,
            mode="form",
            fields=len(content),
        )
        return ElicitationResponse(action="accept", content=content)

    # -- plumbing ----------------------------------------------------------

    def _base_params(self, request: ElicitationRequest, title: str) -> dict[str, Any]:
        return {
            "session_key": request.session_key,
            "agent_id": normalize_agent_id(request.agent_id),
            "acp_session_id": request.acp_session_id,
            "displayTitle": title,
            "toolName": title,
            "sessionKey": request.session_key,
            "agent": agent_display_name(request.agent_id),
            "command": "",
        }

    async def _round(
        self, params: dict[str, Any], deadline: float
    ) -> tuple[bool, dict[str, Any] | None] | None:
        """Queue one row and wait. ``None`` = timed out (row is denied)."""
        from agentos.application.approval_queue import ApprovalQueueOwnedByGatewayError

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        queue = self._queue()
        try:
            approval_id = queue.request(
                namespace=ACP_ELICITATION_NAMESPACE, params=params, ttl=remaining
            )
        except ApprovalQueueOwnedByGatewayError:
            raise UnsupportedElicitationError("no operator surface in this process") from None
        log.info(
            "acp.elicitation.requested",
            approval_id=approval_id,
            session_key=params.get("session_key"),
            mode=params.get("mode"),
        )
        try:
            approved = await queue.wait(approval_id, timeout=remaining)
        except asyncio.CancelledError:
            _deny_quietly(queue, approval_id)
            raise
        except Exception as exc:
            log.warning("acp.elicitation.wait_failed", approval_id=approval_id, error=str(exc))
            _deny_quietly(queue, approval_id)
            return (False, None)
        try:
            entry = queue.get(approval_id)
        except KeyError:
            return (False, None)
        if not entry.resolved:
            _deny_quietly(queue, approval_id)
            return None
        payload = queue.take_payload(approval_id)
        if not approved:
            return (False, None)
        return (True, payload)

    def _decline(self, request: ElicitationRequest, reason: str) -> ElicitationResponse:
        log.info(
            "acp.elicitation.declined",
            session_key=request.session_key,
            agent_id=request.agent_id,
            mode=request.mode,
            reason=reason,
        )
        return ElicitationResponse(action="decline")


def _form_warning(message: str, batch: list[FormField]) -> str:
    lines = [message]
    for f in batch:
        line = f"- {f.title}" + (" (required)" if f.required else "")
        if f.secret:
            line += " [secret]"
        if f.choices:
            line += ": " + " / ".join(c["label"] for c in f.choices)
        lines.append(line)
    lines.append('Answer with payload {"content": {"<field>": <value>}} when approving.')
    return "\n".join(lines)


def _deny_quietly(queue: ApprovalQueue, approval_id: str) -> None:
    try:
        if not queue.get(approval_id).resolved:
            queue.resolve(approval_id, False)
    except (KeyError, ValueError):
        pass
    except Exception as exc:  # pragma: no cover - best effort
        log.debug("acp.elicitation.deny_failed", approval_id=approval_id, error=str(exc))


__all__ = [
    "ACP_ELICITATION_NAMESPACE",
    "DEFAULT_ELICITATION_TIMEOUT_S",
    "ApprovalQueueElicitationBridge",
    "FormField",
    "UnsupportedElicitationError",
    "batch_fields",
    "compile_form",
    "decode_answer",
    "validate_url",
]
