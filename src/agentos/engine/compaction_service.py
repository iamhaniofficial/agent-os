"""One compaction service for every context-overflow path.

Before this module the engine had five compaction paths with different rules
(in-turn preflight, provider-overflow retry, T3-upgrade, before-turn preflight,
emergency request-scoped override). They are now one service with:

* **one trigger** -- the live request estimate against ``window * ratio``
  (``AgentConfig.context_overflow_threshold``, 0.85 by default). A provider
  that still rejects a request as too long invokes the same service with
  ``force=True`` and the window it reported;
* **one shape** -- a single summary exchange followed by the current turn,
  nothing else replayed::

      user:      "[Context summary]\\n<summary>"
      assistant: "Understood. Continuing from summary."
      <current turn: its input message, then its tool rounds, verbatim>

  A user/assistant pair keeps strict role alternation, which every provider
  accepts (a lone system-style note would need a mid-conversation ``system``
  role that several providers reject). The current turn is kept byte for
  byte -- images, per-block thinking with signatures, tool pairs. Only when
  the current turn alone does not fit are its *oldest complete tool rounds*
  folded into the summary as well; the input message and the latest round
  are always kept;
* **one summarizer** -- it goes through :class:`~agentos.provider.LLMProvider`
  (``provider.chat``), never raw HTTP, and falls back to a deterministic
  digest when the provider fails;
* **never mid tool-round** -- a transcript that ends on an assistant
  ``tool_use`` (results not appended yet) is never compacted, and a cut never
  separates a ``tool_use`` from its ``tool_result``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import structlog

from agentos.engine.cache_break_monitor import notify_compaction
from agentos.provider.types import (
    ChatConfig,
    ContentBlockCompaction,
    ContentBlockDocument,
    ContentBlockImage,
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    Message,
)
from agentos.session.compaction_lifecycle import (
    COMPACTION_CHUNK_SUMMARIZED_EVENT,
    COMPACTION_SUMMARY_VERIFIED_EVENT,
    COMPACTION_TRIGGERED_EVENT,
    compaction_effect_payload,
    compaction_lifecycle_payload,
    new_compaction_id,
)
from agentos.session.keys import is_subagent_key

log = structlog.get_logger(__name__)

DEFAULT_COMPACTION_RATIO = 0.85
CONTEXT_SUMMARY_HEADER = "[Context summary]"
SUMMARY_ACK = "Understood. Continuing from summary."
SUMMARY_MAX_TOKENS = 2048
_SUMMARY_TIMEOUT_SECONDS = 90.0
_PREVIEW_CHARS = 240
_FALLBACK_LINE_CHARS = 200
_FALLBACK_MAX_CHARS = 12_000

_SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation compactor. Summarize the conversation concisely, "
    "preserving key facts, decisions, open questions, active tasks and their "
    "status, the last user request, and commitments or follow-ups. Write in the "
    "same language as the conversation. Focus on recent context over older "
    "history.\n\n"
    "IMPORTANT: Preserve all opaque identifiers exactly as written -- UUIDs, "
    "hashes, IDs, tokens, hostnames, IPs, ports, URLs, file names. Do NOT "
    "shorten, reconstruct, or paraphrase any identifier."
)

CompactionStatus = Literal["compacted", "skipped", "failed"]


class _ChatProvider(Protocol):
    def chat(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        config: ChatConfig | None = None,
    ) -> Any: ...


Summarizer = Callable[[list[Message], str | None], Awaitable[tuple[str, str]]]


@dataclass
class CompactionResult:
    """Outcome of one :meth:`CompactionService.compact` call."""

    status: CompactionStatus
    messages: list[Message]
    turn_input_index: int
    reason: str | None = None
    summary: str = ""
    summary_source: str = "skipped"
    compaction_id: str | None = None
    removed_count: int = 0
    kept_count: int = 0
    folded_rounds: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    kept_entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def compacted(self) -> bool:
        return self.status == "compacted"


# ---------------------------------------------------------------------------
# Pure policy helpers
# ---------------------------------------------------------------------------


def should_compact(estimated_tokens: int, window_tokens: int, ratio: float) -> bool:
    """The single trigger: the live request estimate is over ``window * ratio``."""
    if window_tokens <= 0:
        return False
    return estimated_tokens > window_tokens * ratio


def is_ephemeral_session(session_key: str | None) -> bool:
    """Subagent and cron sessions: compacted in memory, never checkpointed.

    Uses :func:`agentos.session.keys.is_subagent_key` so both the legacy
    ``subagent:`` keys and the canonical ``agent:<id>:subagent:<run>`` keys
    match (a ``startswith("subagent:")`` test missed the latter).
    """
    key = (session_key or "").strip().lower()
    if not key:
        return False
    return is_subagent_key(key) or key.startswith("cron:")


def ends_mid_tool_round(messages: Sequence[Message]) -> bool:
    """True while the last assistant ``tool_use`` still waits for its results."""
    if not messages:
        return False
    last = messages[-1]
    return (
        last.role == "assistant"
        and isinstance(last.content, list)
        and any(isinstance(block, ContentBlockToolUse) for block in last.content)
    )


def is_summary_pair(messages: Sequence[Message], index: int = 0) -> bool:
    if len(messages) < index + 2:
        return False
    first, second = messages[index], messages[index + 1]
    return (
        first.role == "user"
        and isinstance(first.content, str)
        and first.content.startswith(CONTEXT_SUMMARY_HEADER)
        and second.role == "assistant"
        and second.content == SUMMARY_ACK
    )


def summary_messages(summary: str) -> list[Message]:
    return [
        Message(role="user", content=f"{CONTEXT_SUMMARY_HEADER}\n{summary}"),
        Message(role="assistant", content=SUMMARY_ACK),
    ]


def _current_turn_rounds(current: Sequence[Message]) -> tuple[int, list[tuple[int, int]]]:
    """Split the current turn into its input group and complete tool rounds.

    Returns ``(input_end, rounds)``: ``current[:input_end]`` are the leading
    user input messages; every round ``(start, end)`` starts at an assistant
    message and runs until the next assistant message, so a ``tool_use`` and
    its ``tool_result`` messages always share a round.
    """
    input_end = 0
    while input_end < len(current) and current[input_end].role == "user":
        input_end += 1
    rounds: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(input_end, len(current)):
        if current[index].role == "assistant":
            if start is not None:
                rounds.append((start, index))
            start = index
    if start is not None:
        rounds.append((start, len(current)))
    return input_end, rounds


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


def _preview(text: str, max_chars: int = _PREVIEW_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n[...omitted {len(text) - max_chars} chars...]\n{text[-tail:]}"


def _json_preview(value: Any) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = str(value)
    if len(rendered) <= _PREVIEW_CHARS:
        return rendered
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
    return f"<json chars={len(rendered)} sha256={digest} preview={_preview(rendered)!r}>"


def render_messages_for_summary(messages: Sequence[Message]) -> str:
    """Readable text for the summarizer: tool payloads as previews, no bytes."""
    lines: list[str] = []
    for message in messages:
        if isinstance(message.content, str):
            lines.append(f"[{message.role}]: {message.content}")
            continue
        parts: list[str] = []
        for block in message.content:
            if isinstance(block, ContentBlockText):
                if block.text.strip():
                    parts.append(block.text)
            elif isinstance(block, ContentBlockToolUse):
                parts.append(f"[tool_use {block.name} id={block.id}] {_json_preview(block.input)}")
            elif isinstance(block, ContentBlockToolResult):
                content = (
                    block.content
                    if isinstance(block.content, str)
                    else _json_preview(block.content)
                )
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
                parts.append(
                    f"[tool_result id={block.tool_use_id} is_error={block.is_error} "
                    f"chars={len(content)} sha256={digest}] {_preview(content)}"
                )
            elif isinstance(block, ContentBlockImage):
                parts.append(f"[image: {block.media_type}]")
            elif isinstance(block, ContentBlockDocument):
                parts.append(f"[document: {block.title or block.media_type}]")
            elif isinstance(block, ContentBlockCompaction):
                if block.content:
                    parts.append(block.content)
            # Reasoning (thinking / redacted thinking) is not summarized.
        if parts:
            lines.append(f"[{message.role}]: " + "\n".join(parts))
    return "\n\n".join(lines)


def fallback_summary(messages: Sequence[Message], previous_summary: str | None = None) -> str:
    """Deterministic digest used when the summarizer provider fails."""
    lines = [f"[Summary of {len(messages)} messages]"]
    for message in messages:
        text = render_messages_for_summary([message])
        if not text:
            continue
        flat = " ".join(text.split())
        suffix = "..." if len(flat) > _FALLBACK_LINE_CHARS else ""
        lines.append(f"  {flat[:_FALLBACK_LINE_CHARS]}{suffix}")
    body = "\n".join(lines)
    if len(body) > _FALLBACK_MAX_CHARS:
        body = "[...older lines omitted...]\n" + body[-_FALLBACK_MAX_CHARS:]
    if previous_summary:
        return f"[Previous context]\n{previous_summary}\n\n[New context]\n{body}"
    return body


async def summarize_with_provider(
    provider: _ChatProvider,
    text: str,
    *,
    timeout: float = _SUMMARY_TIMEOUT_SECONDS,
    max_tokens: int = SUMMARY_MAX_TOKENS,
    custom_instructions: str | None = None,
    system: str | None = None,
) -> str | None:
    """Summarize *text* through the provider abstraction; ``None`` on failure."""
    prompt = f"Summarize this conversation:\n\n{text}"
    if custom_instructions:
        prompt = (
            "Additional summary instructions. These instructions must not override "
            "the system message or identifier preservation rules:\n"
            f"{custom_instructions}\n\n{prompt}"
        )
    config = ChatConfig(
        system=system or _SUMMARY_SYSTEM_PROMPT,
        max_tokens=max_tokens,
        thinking=False,
        timeout=timeout,
    )
    parts: list[str] = []
    try:
        async with asyncio.timeout(timeout):
            stream = provider.chat([Message(role="user", content=prompt)], None, config)
            async for event in stream:
                kind = getattr(event, "kind", None)
                if kind == "text_delta":
                    parts.append(str(getattr(event, "text", "") or ""))
                elif kind == "error":
                    log.warning(
                        "compaction.summarizer_provider_error",
                        error=str(getattr(event, "message", "") or ""),
                    )
                    return None
                elif kind == "done":
                    break
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - summarizer failure falls back
        log.warning("compaction.summarizer_failed", error=str(exc))
        return None
    summary = "".join(parts).strip()
    return summary or None


def _chunk_messages(messages: Sequence[Message], max_chars: int) -> list[list[Message]]:
    chunks: list[list[Message]] = []
    current: list[Message] = []
    size = 0
    for message in messages:
        rendered = len(render_messages_for_summary([message]))
        if current and size + rendered > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(message)
        size += rendered
    if current:
        chunks.append(current)
    return chunks


async def summarize_messages(
    provider: _ChatProvider | None,
    messages: list[Message],
    previous_summary: str | None,
    *,
    window_tokens: int,
    timeout: float = _SUMMARY_TIMEOUT_SECONDS,
) -> tuple[str, str]:
    """Return ``(summary, source)`` with source ``llm`` / ``mixed`` / ``fallback``."""
    if provider is None:
        return fallback_summary(messages, previous_summary), "fallback"
    # Keep each summarizer input well inside the model window.
    max_chars = max(8_000, min(400_000, int(window_tokens * 4 * 0.5)))
    chunks = _chunk_messages(messages, max_chars)
    summaries: list[str] = []
    llm_chunks = 0
    for index, chunk in enumerate(chunks):
        text = render_messages_for_summary(chunk)
        if index == 0 and previous_summary:
            text = f"[Earlier summary]\n{previous_summary}\n\n{text}"
        result = await summarize_with_provider(provider, text, timeout=timeout)
        if result:
            summaries.append(result)
            llm_chunks += 1
        else:
            summaries.append(fallback_summary(chunk, previous_summary if index == 0 else None))
    if len(summaries) == 1:
        merged = summaries[0]
    else:
        merged = "[Merged context summary]" + "".join(
            f"\n\n--- Part {index + 1} ---\n{summary}" for index, summary in enumerate(summaries)
        )
    if llm_chunks == len(chunks):
        source = "llm"
    elif llm_chunks:
        source = "mixed"
    else:
        source = "fallback"
    return merged, source


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CompactionService:
    """Compact a live transcript to one summary exchange plus the current turn."""

    def __init__(
        self,
        *,
        provider: _ChatProvider | None,
        ratio: float = DEFAULT_COMPACTION_RATIO,
        session_key: str | None = None,
        summarizer: Summarizer | None = None,
        timeout: float = _SUMMARY_TIMEOUT_SECONDS,
        phase: str = "agent_inline_overflow",
    ) -> None:
        self._provider = provider
        self._ratio = ratio if 0.0 < ratio <= 1.0 else DEFAULT_COMPACTION_RATIO
        self._session_key = session_key
        self._summarizer = summarizer
        self._timeout = timeout
        self._phase = phase

    @property
    def ratio(self) -> float:
        return self._ratio

    def should_compact(self, estimated_tokens: int, window_tokens: int) -> bool:
        return should_compact(estimated_tokens, window_tokens, self._ratio)

    async def _summarize(
        self,
        messages: list[Message],
        previous_summary: str | None,
        window_tokens: int,
    ) -> tuple[str, str]:
        if self._summarizer is not None:
            return await self._summarizer(messages, previous_summary)
        return await summarize_messages(
            self._provider,
            messages,
            previous_summary,
            window_tokens=window_tokens,
            timeout=self._timeout,
        )

    def _notify(self, status: str, compaction_id: str, **payload: Any) -> None:
        if not self._session_key:
            return
        reason = payload.get("reason")
        notify_compaction(
            self._session_key,
            source="automatic",
            phase=self._phase,
            status=status,
            **payload,
            **compaction_effect_payload(
                status=status,
                reason=reason if isinstance(reason, str) else None,
                user_visible=status != "skipped",
            ),
            **compaction_lifecycle_payload(compaction_id, COMPACTION_TRIGGERED_EVENT),
        )

    async def compact(
        self,
        messages: list[Message],
        *,
        turn_input_index: int,
        estimated_tokens: int,
        window_tokens: int,
        message_tokens: Callable[[Sequence[Message]], int],
        force: bool = False,
        input_entry: dict[str, Any] | None = None,
    ) -> CompactionResult:
        """Compact *messages* when over the trigger (or when *force* is set).

        *message_tokens* estimates the provider size of a message list; the
        rest of the request (system prompt, tools, config) is taken as
        ``estimated_tokens - message_tokens(messages)``. *input_entry* is the
        persisted user row of this turn (``{"role", "content"}``) that a
        durable rewrite keeps; the agent passes it through ``kept_entries``.
        """
        unchanged = CompactionResult(
            status="skipped",
            messages=messages,
            turn_input_index=turn_input_index,
            tokens_before=estimated_tokens,
            tokens_after=estimated_tokens,
        )
        if not force and not self.should_compact(estimated_tokens, window_tokens):
            unchanged.reason = "under_threshold"
            return unchanged
        compaction_id = new_compaction_id()
        if ends_mid_tool_round(messages):
            # Results are not appended yet; compacting now could separate a
            # tool_use from its results. The next request checks again.
            unchanged.reason = "mid_tool_round"
            unchanged.compaction_id = compaction_id
            self._notify("skipped", compaction_id, reason="mid_tool_round")
            return unchanged

        split = max(0, min(turn_input_index, len(messages)))
        prefix = list(messages[:split])
        current = list(messages[split:])
        previous_summary: str | None = None
        if is_summary_pair(prefix):
            previous_summary = str(prefix[0].content)[len(CONTEXT_SUMMARY_HEADER) :].lstrip("\n")
            prefix = prefix[2:]

        overhead = max(0, estimated_tokens - message_tokens(messages))
        budget = max(1, int(window_tokens * self._ratio) - overhead - SUMMARY_MAX_TOKENS)
        input_end, rounds = _current_turn_rounds(current)
        keep_from = 0
        folded_rounds = 0
        if message_tokens(current) > budget * 0.75 and len(rounds) > 1:
            # The current turn alone does not leave room: fold its oldest
            # complete tool rounds, always keeping the input and latest round.
            target = budget * 0.5
            kept_rounds = [rounds[-1]]
            for round_span in reversed(rounds[:-1]):
                candidate = [round_span, *kept_rounds]
                kept = current[:input_end] + [
                    message for start, end in candidate for message in current[start:end]
                ]
                if message_tokens(kept) > target:
                    break
                kept_rounds = candidate
            keep_from = kept_rounds[0][0]
            folded_rounds = sum(1 for span in rounds if span[0] < keep_from)
        folded_current = current[input_end:keep_from] if keep_from else []
        kept_current = current[:input_end] + current[keep_from:] if keep_from else current
        to_fold = prefix + folded_current
        if not to_fold:
            unchanged.reason = "nothing_to_compact"
            unchanged.compaction_id = compaction_id
            self._notify("skipped", compaction_id, reason="nothing_to_compact")
            return unchanged

        self._notify(
            "started",
            compaction_id,
            tokens_before=estimated_tokens,
            context_window_tokens=window_tokens,
        )
        try:
            summary, source = await self._summarize(to_fold, previous_summary, window_tokens)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed compaction
            log.warning("compaction.failed", error=str(exc), session_key=self._session_key)
            self._notify(
                "failed",
                compaction_id,
                reason="compaction_failed",
                message=str(exc),
                tokens_before=estimated_tokens,
                context_window_tokens=window_tokens,
            )
            return CompactionResult(
                status="failed",
                messages=messages,
                turn_input_index=turn_input_index,
                reason="compaction_failed",
                compaction_id=compaction_id,
                tokens_before=estimated_tokens,
            )
        if not summary.strip():
            self._notify(
                "failed",
                compaction_id,
                reason="empty_summary_rejected",
                tokens_before=estimated_tokens,
                context_window_tokens=window_tokens,
            )
            return CompactionResult(
                status="failed",
                messages=messages,
                turn_input_index=turn_input_index,
                reason="empty_summary_rejected",
                compaction_id=compaction_id,
                tokens_before=estimated_tokens,
            )

        compacted = [*summary_messages(summary), *kept_current]
        tokens_after = overhead + message_tokens(compacted)
        removed_count = len(to_fold) + (2 if previous_summary is not None else 0)
        result = CompactionResult(
            status="compacted",
            messages=compacted,
            turn_input_index=2,
            summary=summary,
            summary_source=source,
            compaction_id=compaction_id,
            removed_count=removed_count,
            kept_count=len(kept_current),
            folded_rounds=folded_rounds,
            tokens_before=estimated_tokens,
            tokens_after=tokens_after,
            kept_entries=[dict(input_entry)] if input_entry else [],
        )
        for event in (COMPACTION_CHUNK_SUMMARIZED_EVENT, COMPACTION_SUMMARY_VERIFIED_EVENT):
            if not self._session_key:
                break
            notify_compaction(
                self._session_key,
                source="automatic",
                phase=self._phase,
                status="observed",
                context_window_tokens=window_tokens,
                tokens_before=estimated_tokens,
                tokens_after=tokens_after,
                removed_count=removed_count,
                kept_count=len(kept_current),
                summary_len=len(summary),
                summary_source=source,
                **compaction_effect_payload(status="observed"),
                **compaction_lifecycle_payload(compaction_id, event),
            )
        log.info(
            "compaction.compacted",
            session_key=self._session_key,
            removed=removed_count,
            kept=len(kept_current),
            folded_rounds=folded_rounds,
            summary_source=source,
            tokens_before=estimated_tokens,
            tokens_after=tokens_after,
        )
        return result
